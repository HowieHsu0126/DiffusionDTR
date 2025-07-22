"""
Conservative Q-Learning (CQL) agent implementation for multi-discrete action spaces.

This module provides a CQL agent that implements conservative Q-learning for
offline reinforcement learning in medical decision making scenarios. The agent
incorporates a conservative loss term to prevent overestimation of Q-values
for out-of-distribution actions.
"""

import inspect
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Libs.model.models.agent._compat import ForwardCompatMixin
from Libs.model.models.agent.base_agent import BaseRLAgent
from Libs.utils.model_utils import (ReplayBuffer, apply_gradient_clipping,
                                    as_tensor, safe_float, safe_item)


class CQLAgent(ForwardCompatMixin, BaseRLAgent):
    """Conservative Q-Learning agent implementation for safe offline RL.
    
    This agent implements CQL with conservative penalties to prevent
    overestimation of Q-values for out-of-distribution actions, making
    it particularly suitable for medical decision-making scenarios.
    """

    # 🔧 CRITICAL FIX: Specify that this agent expects tuple batch format
    expects_tuple_batch = True
    _expects_full_transition = True

    def __init__(
        self,
        model: nn.Module,
        action_dims: List[int],
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        cql_alpha: float = 0.1,
        device: str = 'cpu',
        reward_centering: bool = False,
        target_update_freq: int = 100,
        max_grad_norm: float = 1.0,
        alpha_lr: float = 1e-4,
        cql_target_gap: float = 5.0
    ) -> None:
        """
        Initialize the CQL agent.

        Args:
            model: Q-network architecture for action-value estimation.
            action_dims: List of action space sizes for each action head.
            lr: Learning rate for the optimizer.
            gamma: Discount factor for future rewards (0 < gamma <= 1).
            buffer_size: Maximum size of the replay buffer.
            batch_size: Batch size for training updates.
            cql_alpha: Conservative penalty coefficient (>= 0).
            device: Computing device ('cpu' or 'cuda').
            reward_centering: Whether to apply reward centering for stability.
            target_update_freq: Frequency of target network updates.
            max_grad_norm: Maximum gradient norm for gradient clipping.
            alpha_lr: Learning rate for the alpha optimizer.
            cql_target_gap: Target action gap for the CQL algorithm.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError(
                "action_dims must be non-empty with positive values")

        # Safely convert parameters to appropriate types
        lr = safe_float(lr, 1e-3)
        gamma = safe_float(gamma, 0.99)
        cql_alpha = safe_float(cql_alpha, 0.1)
        cql_target_gap = safe_float(cql_target_gap, 5.0)
        alpha_lr = safe_float(alpha_lr, 1e-4)

        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if cql_alpha < 0:
            raise ValueError("cql_alpha must be non-negative")

        # Initialize logger
        self.logger = logging.getLogger(f'{self.__class__.__name__}')

        super().__init__(device=device, gamma=gamma,
                         target_update_freq=target_update_freq, reward_centering=reward_centering)

        self.q_net = model.to(device)

        # Create target network with same architecture
        self.target_q_net = type(model)(*model.init_args).to(device)
        self.target_q_net.load_state_dict(model.state_dict())

        self.action_dims = action_dims
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.cql_alpha = cql_alpha
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training tracking
        self.update_steps = 0
        self.expects_tuple_batch = True
        self._expects_full_transition = True

        # ====== Begin Lagrange multiplier setup for dynamic α ======
        # If cql_alpha is provided as a positive value we treat it as the initial value.
        # We parameterize log_alpha to keep alpha positive during optimization.
        self.log_alpha = torch.tensor(np.log(max(
            cql_alpha, 1e-6)), dtype=torch.float32, requires_grad=True, device=self.device)
        # Dedicated optimizer for α so that its update dynamics are decoupled from Q-network.
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # Target action-gap (see CQL paper, Section 5.2). By default we follow authors' suggestion of 5.0
        self.cql_target_action_gap = cql_target_gap
        # ====== End Lagrange multiplier setup ======

        # Gradient clipping value
        self.max_grad_norm = float(max_grad_norm)

    # BEGIN: unify ForwardCompat helpers
    _is_pog_model = ForwardCompatMixin._is_pog_model  # type: ignore[misc]
    _forward_model = ForwardCompatMixin._forward_model  # type: ignore[misc]
    # END: unify ForwardCompat helpers

    def select_action(
        self,
        obs: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eval_mode: bool = False,
        epsilon: float = 0.1,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Select actions using epsilon-greedy policy for each action head.

        Args:
            obs: Observation tensor of shape (batch_size, seq_len, obs_dim).
            lengths: Sequence lengths of shape (batch_size,).
            edge_index: Graph edge indices of shape (2, num_edges).
            mask: Optional attention mask of shape (batch_size, seq_len).
            eval_mode: If True, use greedy policy without exploration.
            epsilon: Exploration probability for epsilon-greedy policy.
            **kwargs: Additional arguments for compatibility.

        Returns:
            List of action tensors, one for each action head.

        Raises:
            ValueError: If input dimensions are inconsistent.
        """
        if obs.dim() != 3:
            raise ValueError(
                f"Expected 3D observation tensor, got {obs.dim()}D")
        if lengths.dim() != 1 or lengths.size(0) != obs.size(0):
            raise ValueError("lengths must be 1D with same batch size as obs")

        batch_size, seq_len, _ = obs.shape

        # Exploration: random actions
        if not eval_mode and np.random.rand() < epsilon:
            actions = []
            for action_dim in self.action_dims:
                random_actions = torch.randint(
                    action_dim, (batch_size, seq_len), device=self.device
                )
                actions.append(random_actions)
            return actions

        # Exploitation: greedy actions based on Q-values
        with torch.no_grad():
            model_output = self._forward_model(
                self.q_net, obs, lengths, edge_index, mask, mode='q'
            )

            # Handle different output formats
            if isinstance(model_output, tuple):
                q_values = model_output[0]
            else:
                q_values = model_output

            # Select greedy actions for each head
            actions = []
            for q_head in q_values:
                greedy_actions = q_head.argmax(dim=-1)
                actions.append(greedy_actions)

        return actions

    def update(self, batch: Tuple[np.ndarray, ...], *, grad_scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
        """
        Update the Q-network using CQL loss with conservative penalty.

        Performs a single training step using the CQL algorithm, which combines
        the standard Q-learning loss with a conservative penalty that regularizes
        Q-values for out-of-distribution actions.

        Args:
            batch: Tuple containing batch of transitions from replay buffer.
                  Expected format: (obs, actions, rewards, next_obs, dones, mask, lengths, next_lengths, edge_index)
                  Where obs has shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)

        Returns:
            Total loss for the training step.
        """
        # Ensure model is in training mode for backward pass
        self.q_net.train()
        self.target_q_net.eval()  # Target model should be in eval mode

        # 🔧 CRITICAL FIX: Enhanced batch unpacking with shape validation
        try:
            obs, actions, rewards, next_obs, dones, mask, lengths, next_lengths, edge_index = batch
        except ValueError as e:
            self.logger.error(f"❌ Batch unpacking failed: {e}")
            self.logger.error(f"🔍 Expected 9 elements, got {len(batch)}")
            return 0.0

        batch_size = obs.shape[0]
        self.logger.debug(
            f"🔍 CQL batch info: obs.shape={obs.shape}, batch_size={batch_size}")

        # 🔧 CRITICAL FIX: Enhanced tensor conversion with shape validation
        try:
            obs = as_tensor(obs, dtype=torch.float32, device=self.device)
            next_obs = as_tensor(
                next_obs, dtype=torch.float32, device=self.device)
            mask = as_tensor(mask, dtype=torch.float32, device=self.device)
            rewards = as_tensor(
                rewards, dtype=torch.float32, device=self.device)
            dones = as_tensor(dones, dtype=torch.float32, device=self.device)
            edge_index = as_tensor(
                edge_index, dtype=torch.long, device=self.device)

            # 🔧 SHAPE CONSISTENCY VALIDATION
            if obs.dim() not in [2, 3]:
                raise ValueError(f"Invalid obs dimensions: {obs.shape}")
            if next_obs.dim() not in [2, 3]:
                raise ValueError(
                    f"Invalid next_obs dimensions: {next_obs.shape}")

            # Ensure all batch dimensions match
            if obs.size(0) != batch_size or next_obs.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: obs={obs.size(0)}, next_obs={next_obs.size(0)}, expected={batch_size}")

        except Exception as e:
            self.logger.error(f"❌ Tensor conversion failed: {e}")
            return 0.0

        # 🔧 CRITICAL FIX: Proper handling of sequence vs single-step data
        is_sequential = obs.dim() == 3
        seq_len = obs.size(1) if is_sequential else 1

        if is_sequential:
            self.logger.debug(
                f"🔍 Processing sequential data: seq_len={seq_len}")

            # Ensure mask has proper dimensions for sequential data
            if mask.dim() == 1:
                # Expand (B,) -> (B, T)
                mask = mask.unsqueeze(1).expand(batch_size, seq_len)
            elif mask.dim() == 2:
                # Ensure temporal dimension matches
                if mask.size(1) != seq_len:
                    self.logger.warning(
                        f"⚠️ Mask temporal dimension mismatch: {mask.size(1)} vs {seq_len}")
                    mask = mask[:, :seq_len] if mask.size(
                        1) > seq_len else F.pad(mask, (0, seq_len - mask.size(1)))

            # Handle rewards and dones for sequential data
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)  # (B,) -> (B, 1)
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)    # (B,) -> (B, 1)

            # For sequential data, align temporal dimensions
            if rewards.dim() == 2 and rewards.size(1) != seq_len:
                if rewards.size(1) == 1:
                    # Expand single reward to all timesteps
                    rewards = rewards.expand(-1, seq_len)
                else:
                    # Truncate or pad to match sequence length
                    rewards = rewards[:, :seq_len] if rewards.size(
                        1) > seq_len else F.pad(rewards, (0, seq_len - rewards.size(1)))

            if dones.dim() == 2 and dones.size(1) != seq_len:
                if dones.size(1) == 1:
                    # For terminal signals, typically only the last timestep matters
                    dones_expanded = torch.zeros(
                        batch_size, seq_len, device=self.device, dtype=dones.dtype)
                    dones_expanded[:, -1] = dones.squeeze(1)
                    dones = dones_expanded
                else:
                    dones = dones[:, :seq_len] if dones.size(
                        1) > seq_len else F.pad(dones, (0, seq_len - dones.size(1)))
        else:
            # Single-step data
            self.logger.debug("🔍 Processing single-step data")
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)  # (B,) -> (B, 1)
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)

        # 🔧 CRITICAL FIX: Enhanced action processing with proper shape handling
        actions_validated = []
        for i, a in enumerate(actions):
            try:
                # Convert action to tensor
                if isinstance(a, list):
                    # Handle list of actions (potentially from different samples)
                    action_tensor = torch.tensor(
                        a, dtype=torch.long, device=self.device)
                else:
                    action_tensor = as_tensor(
                        a, dtype=torch.long, device=self.device)

                # CQL expects actions to match sequence structure when is_sequential
                if is_sequential:
                    # Ensure actions have temporal dimension
                    if action_tensor.dim() == 1:
                        # (B,) -> (B, T) by expanding to all timesteps
                        action_tensor = action_tensor.unsqueeze(1).expand(-1, seq_len)
                    elif action_tensor.dim() == 2:
                        # Validate temporal dimension
                        if action_tensor.size(1) != seq_len:
                            if action_tensor.size(1) == 1:
                                # Expand single action to all timesteps
                                action_tensor = action_tensor.expand(-1, seq_len)
                            else:
                                # Align temporal dimensions
                                action_tensor = action_tensor[:, :seq_len] if action_tensor.size(
                                    1) > seq_len else F.pad(action_tensor, (0, seq_len - action_tensor.size(1)))
                else:
                    # Single-step: ensure actions are (B, 1)
                    if action_tensor.dim() == 1:
                        action_tensor = action_tensor.unsqueeze(1)
                    elif action_tensor.dim() == 2 and action_tensor.size(1) > 1:
                        # Take only the last action for single-step
                        action_tensor = action_tensor[:, -1:] 
                        
                # Validate batch size consistency
                if action_tensor.size(0) != batch_size:
                    self.logger.error(
                        f"❌ Action batch size mismatch for head {i}: {action_tensor.size(0)} vs {batch_size}")
                    return 0.0

                # Clamp actions to valid range [0, action_dims[i]-1] to prevent gather errors
                action_tensor = torch.clamp(
                    action_tensor, min=0, max=self.action_dims[i] - 1)
                actions_validated.append(action_tensor)

                # Log warning if any invalid actions were corrected
                original_tensor = as_tensor(
                    a, dtype=torch.long, device=self.device)
                if original_tensor.dim() == 1 and action_tensor.dim() == 2:
                    original_tensor = original_tensor.unsqueeze(1)

                invalid_mask = (original_tensor < 0) | (
                    original_tensor >= self.action_dims[i])
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum().item()
                    self.logger.warning(f"⚠️ [CQL] Fixed {invalid_count} invalid actions in head {i}: "
                                        f"range [0, {self.action_dims[i]-1}], max found: {original_tensor.max().item()}")

            except Exception as e:
                self.logger.error(
                    f"❌ Action processing failed for head {i}: {e}")
                return 0.0

        actions = actions_validated

        # 🔧 CRITICAL FIX: Proper lengths handling
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len,
                                 dtype=torch.long, device=self.device)
        else:
            lengths = as_tensor(lengths, dtype=torch.long, device=self.device)
            if lengths.size(0) != batch_size:
                self.logger.warning(
                    f"⚠️ Lengths batch size mismatch: {lengths.size(0)} vs {batch_size}")
                lengths = torch.full((batch_size,), seq_len,
                                     dtype=torch.long, device=self.device)

        if next_lengths is None:
            next_lengths = torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=self.device)
        else:
            next_lengths = as_tensor(
                next_lengths, dtype=torch.long, device=self.device)
            if next_lengths.size(0) != batch_size:
                next_lengths = torch.full(
                    (batch_size,), seq_len, dtype=torch.long, device=self.device)

        # Validate tensor consistency
        self._assert_batch_obs_dims(obs)
        rewards = self._center_rewards(rewards, dim=0)

        # 🔧 ENHANCED MODEL FORWARDING with proper kwargs handling
        try:
            kwargs = {}
            if self._is_pog_model(self.q_net):
                kwargs.update({
                    'rewards': rewards,
                    'reward_centering': self.reward_centering
                })

            model_output = self._forward_model(
                self.q_net, obs, lengths, edge_index, mask, mode='q', **kwargs
            )

            # Extract Q-values; handle different output formats
            if isinstance(model_output, tuple):
                q_values = model_output[0]
            else:
                q_values = model_output

            if not isinstance(q_values, list):
                self.logger.error(
                    f"❌ Expected list of Q-values, got {type(q_values)}")
                return 0.0

            # Forward pass through target network
            with torch.no_grad():
                target_kwargs = {}
                if self._is_pog_model(self.target_q_net):
                    target_kwargs.update({
                        'rewards': rewards,
                        'reward_centering': self.reward_centering
                    })

                target_output = self._forward_model(
                    self.target_q_net, next_obs, next_lengths, edge_index, mask, mode='q', **target_kwargs
                )

                if isinstance(target_output, tuple):
                    next_q_values = target_output[0]
                else:
                    next_q_values = target_output

                if not isinstance(next_q_values, list):
                    self.logger.error(
                        f"❌ Expected list of next Q-values, got {type(next_q_values)}")
                    return 0.0

        except Exception as e:
            self.logger.error(f"❌ Model forward pass failed: {e}")
            self.logger.error(
                f"🔍 obs.shape={obs.shape}, lengths.shape={lengths.shape}, edge_index.shape={edge_index.shape}")
            return 0.0

        # Compute CQL loss for each action head with bounds checking
        total_loss = 0.0
        total_conservative_penalty = 0.0
        alpha_value = torch.exp(self.log_alpha)

        # 🔧 CRITICAL FIX: Enhanced loss computation with proper shape handling
        try:
            for i, action_dim in enumerate(self.action_dims):
                # Get action indices and Q-values for this head
                action_indices = actions[i]  # Shape: (B, T) or (B, 1)
                q_head = q_values[i]         # Shape: (B, T, A) or (B, A)
                next_q_head = next_q_values[i]  # Shape: (B, T, A) or (B, A)

                self.logger.debug(
                    f"🔍 Head {i}: action_indices.shape={action_indices.shape}, q_head.shape={q_head.shape}")

                # Verify action indices are within bounds
                if action_indices.max() >= action_dim:
                    action_indices = torch.clamp(
                        action_indices, max=action_dim - 1)

                # Additional safety check for Q-values dimension
                if q_head.shape[-1] != action_dim:
                    action_indices = torch.clamp(
                        action_indices, max=q_head.shape[-1] - 1)

                # 🔧 CRITICAL FIX: Enhanced tensor rank alignment for sequence vs single-step
                if is_sequential:
                    # Sequential data: use last timestep for Q-value extraction
                    # Always define batch_indices for sequential data
                    batch_indices = torch.arange(
                        batch_size, device=self.device)
                    last_timesteps = lengths.clamp(min=1, max=seq_len) - 1
                    
                    if q_head.dim() == 3:
                        # Q-values: (B, T, A) -> use last valid timestep for each sequence
                        q_head_step = q_head[batch_indices,
                                             last_timesteps, :]  # (B, A)

                        # Actions: use corresponding timestep
                        if action_indices.dim() == 2:
                            # (B,)
                            action_indices_step = action_indices[batch_indices,
                                                                 last_timesteps]
                        else:
                            action_indices_step = action_indices.squeeze()  # (B,)
                    else:
                        # Q-values already 2D: (B, A)
                        q_head_step = q_head
                        # For 2D Q-values with sequential actions, extract last timestep
                        if action_indices.dim() == 2:
                            action_indices_step = action_indices[batch_indices,
                                                                 last_timesteps]
                        else:
                            action_indices_step = action_indices.squeeze(
                            ) if action_indices.dim() > 1 else action_indices

                    # Next Q-values: similar processing
                    if next_q_head.dim() == 3:
                        next_last_timesteps = next_lengths.clamp(
                            min=1, max=seq_len) - 1
                        # (B, A)
                        next_q_head_step = next_q_head[batch_indices,
                                                       next_last_timesteps, :]
                    else:
                        next_q_head_step = next_q_head

                    # Mask processing for sequential data
                    if mask.dim() == 2:
                        mask_step = mask[batch_indices, last_timesteps]  # (B,)
                    else:
                        mask_step = mask.squeeze()

                    # Rewards and dones: use last timestep values
                    if rewards.dim() == 2:
                        # (B,)
                        rewards_step = rewards[batch_indices, last_timesteps]
                    else:
                        rewards_step = rewards.squeeze()

                    if dones.dim() == 2:
                        dones_step = dones[batch_indices,
                                           last_timesteps]  # (B,)
                    else:
                        dones_step = dones.squeeze()

                else:
                    # Single-step data: direct processing
                    if q_head.dim() == 3:
                        q_head_step = q_head.squeeze(1)  # (B, 1, A) -> (B, A)
                    else:
                        q_head_step = q_head  # Already (B, A)

                    if next_q_head.dim() == 3:
                        next_q_head_step = next_q_head.squeeze(
                            1)  # (B, 1, A) -> (B, A)
                    else:
                        next_q_head_step = next_q_head

                    action_indices_step = action_indices.squeeze(
                    ) if action_indices.dim() > 1 else action_indices
                    mask_step = mask.squeeze()
                    rewards_step = rewards.squeeze()
                    dones_step = dones.squeeze()

                # Ensure all tensors have compatible shapes
                if action_indices_step.dim() != 1:
                    action_indices_step = action_indices_step.view(-1)
                if mask_step.dim() != 1:
                    mask_step = mask_step.view(-1)
                if rewards_step.dim() != 1:
                    rewards_step = rewards_step.view(-1)
                if dones_step.dim() != 1:
                    dones_step = dones_step.view(-1)

                # Validate tensor shapes before gather operation
                if q_head_step.size(0) != action_indices_step.size(0):
                    self.logger.error(
                        f"❌ Shape mismatch for head {i}: q_head_step.size(0)={q_head_step.size(0)}, action_indices_step.size(0)={action_indices_step.size(0)}")
                    return 0.0

                # Gather current Q-values for taken actions
                current_q = q_head_step.gather(
                    1, action_indices_step.unsqueeze(1)).squeeze(1)  # (B,)

                # Compute target Q-values
                next_q_max = next_q_head_step.max(dim=-1)[0]  # (B,)
                scale = 100.0
                target_q = (rewards_step + self.gamma *
                            next_q_max * (1 - dones_step)) / scale
                current_q_scaled = current_q / scale

                # Standard Bellman loss (Huber) with proper masking
                bellman_elem = F.smooth_l1_loss(
                    current_q_scaled, target_q, reduction="none")

                # Apply mask for valid timesteps
                mask_sum = mask_step.sum()
                if mask_sum > 0:
                    bellman_loss = (bellman_elem * mask_step).sum() / mask_sum
                else:
                    bellman_loss = bellman_elem.mean()

                # Conservative penalty: logsumexp over all actions minus taken action
                q_all_actions = q_head_step  # (B, A)

                # Add numerical stability: clamp Q-values to prevent overflow in logsumexp
                q_all_actions = torch.clamp(q_all_actions, min=-10.0, max=10.0)
                logsumexp_q = torch.logsumexp(q_all_actions, dim=-1)  # (B,)

                q_taken_action = q_all_actions.gather(
                    1, action_indices_step.unsqueeze(1)).squeeze(1)  # (B,)
                conservative_penalty = (logsumexp_q - q_taken_action).mean()

                # Add numerical stability check for conservative penalty
                if not torch.isfinite(conservative_penalty):
                    self.logger.warning(
                        f"⚠️ [CQL] Non-finite conservative penalty detected in head {i}, clamping to 0.0")
                    conservative_penalty = torch.tensor(
                        0.0, device=self.device)

                total_conservative_penalty += conservative_penalty.detach()

                # Combined CQL loss (α is positive via exp(log_alpha))
                head_loss = bellman_loss + alpha_value * conservative_penalty
                total_loss += head_loss

        except Exception as e:
            self.logger.error(f"❌ CQL loss computation failed: {e}")
            self.logger.error(f"🔍 Debug info:")
            self.logger.error(f"  • is_sequential: {is_sequential}")
            self.logger.error(f"  • seq_len: {seq_len}")
            self.logger.error(f"  • batch_size: {batch_size}")
            if len(q_values) > 0:
                self.logger.error(
                    f"  • q_values[0].shape: {q_values[0].shape}")
            if len(actions) > 0:
                self.logger.error(f"  • actions[0].shape: {actions[0].shape}")
            self.logger.error(f"  • rewards.shape: {rewards.shape}")
            self.logger.error(f"  • mask.shape: {mask.shape}")
            return 0.0

        n_heads = len(self.action_dims)
        # Keep TD + conservative loss summed over heads (no further averaging)
        avg_conservative_penalty = total_conservative_penalty / n_heads

        # ====== Lagrange multiplier update for α ======
        # J(α) = α * (target_gap - conservative_penalty.detach()) . We want penalty ~ target
        alpha_loss = -(self.log_alpha * (avg_conservative_penalty -
                       self.cql_target_action_gap).detach())
        # Note: detach conservative penalty to avoid second-order gradients.

        # Add loss lower bound protection to prevent gradient vanishing
        min_loss_threshold = 1e-4
        loss_value = total_loss.item()
        if loss_value < min_loss_threshold:
            self.logger.debug(f"📊 CQL: Loss {loss_value:.6f} below threshold {min_loss_threshold}, adding regularization")
            regularization = min_loss_threshold - loss_value
            total_loss = total_loss + regularization

        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()

        apply_gradient_clipping(self.q_net, max_norm=self.max_grad_norm)

        self.optimizer.step()

        # Update α
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        # Clip ∇α to avoid exploding updates (Appendix C, CQL-R)
        torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
        self.alpha_optimizer.step()

        # Numerical guard: keep log_alpha within reasonable bounds
        with torch.no_grad():
            # Stricter upper bound to prevent alpha explosion
            self.log_alpha.clamp_(min=-10.0, max=5.0)

        # Expose updated α for logging with numerical safety
        alpha_value_item = safe_item(torch.exp(self.log_alpha))
        if not math.isfinite(alpha_value_item):
            self.logger.warning(
                "⚠️  [CQL] Non-finite alpha detected, resetting to 0.1")
            self.log_alpha.data.fill_(np.log(0.1))
            alpha_value_item = 0.1
        self.cql_alpha = alpha_value_item

        # Update target network periodically
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return safe_item(total_loss)

    def act(
        self,
        state: torch.Tensor,
        greedy: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Selects actions for given states following BaseRLAgent interface.

        This method implements the CQL agent's policy for action selection compatible
        with the BaseRLAgent abstract interface. It uses the conservative Q-learning
        approach to select actions based on learned Q-values.

        Args:
            state: Input state tensor. Expected shapes:
                  - If 2D: (seq_len, obs_dim) - single sequence
                  - If 3D: (batch_size, seq_len, obs_dim) - batch of sequences
            greedy: Whether to use greedy (deterministic) action selection.
                   If False, uses epsilon-greedy exploration.
            **kwargs: Additional arguments including:
                     - lengths: Sequence lengths tensor
                     - edge_index: Graph edge indices  
                     - mask: Attention mask
                     - epsilon: Exploration probability (default: 0.1)

        Returns:
            Selected actions tensor of shape:
            - (seq_len, num_action_heads) for single sequence
            - (batch_size, seq_len, num_action_heads) for batch

        Raises:
            ValueError: If state tensor has invalid dimensions.
        """
        # Handle input dimensions
        if state.dim() == 2:
            # Single sequence: (seq_len, obs_dim) -> (1, seq_len, obs_dim)
            state = state.unsqueeze(0)
            single_sequence = True
        elif state.dim() == 3:
            # Batch of sequences: (batch_size, seq_len, obs_dim)
            single_sequence = False
        else:
            raise ValueError(
                f"Expected 2D or 3D state tensor, got {state.dim()}D")

        batch_size, seq_len, obs_dim = state.shape

        # Extract or create default parameters
        lengths = kwargs.get('lengths', torch.full((batch_size,), seq_len,
                                                   dtype=torch.long, device=self.device))
        edge_index = kwargs.get('edge_index', torch.empty((2, 0),
                                                          dtype=torch.long, device=self.device))
        mask = kwargs.get('mask', torch.ones((batch_size, seq_len),
                                             dtype=torch.float32, device=self.device))
        epsilon = kwargs.get('epsilon', 0.1)

        # Use select_action method for actual action selection
        action_list = self.select_action(
            obs=state,
            lengths=lengths,
            edge_index=edge_index,
            mask=mask,
            eval_mode=greedy,
            epsilon=epsilon
        )

        # Convert list of action tensors to stacked tensor
        # action_list: List[(batch_size, seq_len)] -> (batch_size, seq_len, num_heads)
        actions = torch.stack(action_list, dim=-1)

        # Return appropriate shape
        if single_sequence:
            return actions.squeeze(0)  # (seq_len, num_action_heads)
        else:
            return actions  # (batch_size, seq_len, num_action_heads)

    def save(self, path: str) -> None:
        """
        Save the agent's state to file.

        Args:
            path: File path to save the agent state.
        """
        torch.save({
            'model': self.q_net.state_dict(),
            'target_model': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_steps': self.update_steps,
            'cql_alpha': self.cql_alpha
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's state from file.

        Args:
            path: File path to load the agent state from.
        """
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state['model'])
        self.target_q_net.load_state_dict(state['target_model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.update_steps = state.get('update_steps', 0)
        self.cql_alpha = state.get('cql_alpha', self.cql_alpha)

    def save_checkpoint(self, filepath: str) -> None:
        """Saves agent state to checkpoint file.

        Args:
            filepath: Path to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.q_net.state_dict(),
            'target_model_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'update_steps': self.update_steps,
            'config': {
                'action_dims': self.action_dims,
                'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 1e-3,
                'gamma': self.gamma,
                'target_update_freq': self.target_update_freq,
                'cql_alpha': self.cql_alpha,
                'max_grad_norm': self.max_grad_norm,
                'cql_target_action_gap': self.cql_target_action_gap,
                'reward_centering': self.reward_centering,
            }
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Loads agent state from checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_q_net.load_state_dict(
            checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(
            checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self._training_step = checkpoint['training_step']
        self._episode_count = checkpoint['episode_count']
        self.update_steps = checkpoint.get('update_steps', 0)

    # ------------------------------------------------------------------
    #  Helper – PoG-specific conservative Q-loss (keeps old implementation)
    # ------------------------------------------------------------------

    def _compute_pog_cql_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: List[torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        lengths: torch.Tensor,
        next_lengths: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CQL loss for PoG backbone (old per-branch logic)."""
        kwargs = {
            'rewards': rewards,
            'reward_centering': self.reward_centering
        }

        q_values = self._forward_model(
            self.q_net, obs, lengths, edge_index, mask, mode='q', **kwargs
        )[0]

        with torch.no_grad():
            next_q_values = self._forward_model(
                self.target_q_net, next_obs, next_lengths, edge_index, mask, mode='q', **kwargs
            )[0]

        total_loss = 0.0
        alpha_value = torch.exp(self.log_alpha)
        for i in range(len(self.action_dims)):
            q_head = q_values[i]
            if q_head.dim() == 3:
                q_head = q_head[:, -1, :]
            action_indices = actions[i].squeeze(1)
            current_q = q_head.gather(-1, action_indices).squeeze(-1)
            next_q_max = next_q_values[i].max(dim=-1)[0]
            target_q = rewards + self.gamma * next_q_max * (1 - dones)
            bellman = ((current_q - target_q) ** 2 * mask).sum() / mask.sum()
            conservative = (torch.logsumexp(q_head, dim=-1) - current_q).mean()
            total_loss = total_loss + bellman + alpha_value * conservative

        return total_loss

    def _validate_and_fix_tensor_shapes(self, batch_data: Dict[str, torch.Tensor], batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Validate and automatically fix tensor shape mismatches in batch data.

        This method is specifically designed to handle the common shape mismatches
        that occur in CQL training, particularly when sequence data is processed.

        Args:
            batch_data: Dictionary of tensors to validate and fix
            batch_size: Expected batch size
            seq_len: Expected sequence length

        Returns:
            Dictionary with corrected tensor shapes

        Raises:
            ValueError: If tensors cannot be automatically corrected
        """
        fixed_data = {}

        for key, tensor in batch_data.items():
            if not torch.is_tensor(tensor):
                fixed_data[key] = tensor
                continue

            original_shape = tensor.shape

            try:
                if key in ['obs', 'state', 'next_obs', 'next_state']:
                    # State tensors should be (batch_size, seq_len, features) or (batch_size, features)
                    if tensor.dim() == 2:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        # Add sequence dimension: (B, F) -> (B, 1, F)
                        tensor = tensor.unsqueeze(1)
                        self.logger.debug(
                            f"🔧 Added sequence dim to {key}: {original_shape} -> {tensor.shape}")
                    elif tensor.dim() == 3:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        if tensor.size(1) != seq_len:
                            if tensor.size(1) > seq_len:
                                tensor = tensor[:, :seq_len, :]
                                self.logger.debug(
                                    f"🔧 Truncated {key} sequence: {original_shape} -> {tensor.shape}")
                            else:
                                # Pad sequence dimension
                                pad_len = seq_len - tensor.size(1)
                                padding = tensor[:, -1:,
                                                 :].expand(-1, pad_len, -1)
                                tensor = torch.cat([tensor, padding], dim=1)
                                self.logger.debug(
                                    f"🔧 Padded {key} sequence: {original_shape} -> {tensor.shape}")
                    else:
                        raise ValueError(
                            f"Invalid dimensions for {key}: {tensor.shape}")

                elif key in ['action', 'actions']:
                    # Action tensors should be (batch_size, seq_len, n_heads) or (batch_size, seq_len)
                    if tensor.dim() == 1:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        tensor = tensor.unsqueeze(1)  # (B,) -> (B, 1)
                        self.logger.debug(
                            f"🔧 Added temporal dim to {key}: {original_shape} -> {tensor.shape}")
                    elif tensor.dim() == 2:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        if tensor.size(1) != seq_len and tensor.size(1) != len(self.action_dims):
                            # Could be temporal or action dimension mismatch
                            if tensor.size(1) == 1:
                                # Expand temporal dimension
                                tensor = tensor.expand(-1, seq_len)
                                self.logger.debug(
                                    f"🔧 Expanded {key} temporal: {original_shape} -> {tensor.shape}")
                            elif tensor.size(1) < seq_len:
                                # Pad temporal dimension
                                pad_len = seq_len - tensor.size(1)
                                padding = tensor[:, -1:].expand(-1, pad_len)
                                tensor = torch.cat([tensor, padding], dim=1)
                                self.logger.debug(
                                    f"🔧 Padded {key} temporal: {original_shape} -> {tensor.shape}")
                            else:
                                # Truncate temporal dimension
                                tensor = tensor[:, :seq_len]
                                self.logger.debug(
                                    f"🔧 Truncated {key} temporal: {original_shape} -> {tensor.shape}")

                elif key in ['reward', 'rewards']:
                    # Reward tensors should be (batch_size, seq_len) or (batch_size,)
                    if tensor.dim() == 1:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        tensor = tensor.unsqueeze(1)  # (B,) -> (B, 1)
                        self.logger.debug(
                            f"🔧 Added temporal dim to {key}: {original_shape} -> {tensor.shape}")
                    elif tensor.dim() == 2:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        if tensor.size(1) != seq_len:
                            if tensor.size(1) == 1:
                                tensor = tensor.expand(-1, seq_len)
                                self.logger.debug(
                                    f"🔧 Expanded {key} temporal: {original_shape} -> {tensor.shape}")
                            elif tensor.size(1) > seq_len:
                                tensor = tensor[:, :seq_len]
                                self.logger.debug(
                                    f"🔧 Truncated {key} temporal: {original_shape} -> {tensor.shape}")
                            else:
                                pad_len = seq_len - tensor.size(1)
                                padding = torch.zeros(
                                    batch_size, pad_len, dtype=tensor.dtype, device=tensor.device)
                                tensor = torch.cat([tensor, padding], dim=1)
                                self.logger.debug(
                                    f"🔧 Padded {key} temporal: {original_shape} -> {tensor.shape}")

                elif key in ['done', 'dones', 'terminal']:
                    # Done tensors should be (batch_size, seq_len) or (batch_size,)
                    if tensor.dim() == 1:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        # For done flags, typically only last timestep matters
                        done_expanded = torch.zeros(
                            batch_size, seq_len, dtype=tensor.dtype, device=tensor.device)
                        done_expanded[:, -1] = tensor
                        tensor = done_expanded
                        self.logger.debug(
                            f"🔧 Expanded {key} with terminal placement: {original_shape} -> {tensor.shape}")
                    elif tensor.dim() == 2:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        if tensor.size(1) != seq_len:
                            if tensor.size(1) == 1:
                                # Place terminal flag at the end
                                done_expanded = torch.zeros(
                                    batch_size, seq_len, dtype=tensor.dtype, device=tensor.device)
                                done_expanded[:, -1] = tensor.squeeze(1)
                                tensor = done_expanded
                                self.logger.debug(
                                    f"🔧 Repositioned {key} terminal flag: {original_shape} -> {tensor.shape}")
                            elif tensor.size(1) > seq_len:
                                tensor = tensor[:, :seq_len]
                                self.logger.debug(
                                    f"🔧 Truncated {key} temporal: {original_shape} -> {tensor.shape}")
                            else:
                                pad_len = seq_len - tensor.size(1)
                                padding = torch.zeros(
                                    batch_size, pad_len, dtype=tensor.dtype, device=tensor.device)
                                tensor = torch.cat([tensor, padding], dim=1)
                                self.logger.debug(
                                    f"🔧 Padded {key} temporal: {original_shape} -> {tensor.shape}")

                elif key in ['mask']:
                    # Mask tensors should be (batch_size, seq_len) or (batch_size,)
                    if tensor.dim() == 1:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        tensor = tensor.unsqueeze(1).expand(-1, seq_len)
                        self.logger.debug(
                            f"🔧 Expanded {key}: {original_shape} -> {tensor.shape}")
                    elif tensor.dim() == 2:
                        if tensor.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                        if tensor.size(1) != seq_len:
                            if tensor.size(1) > seq_len:
                                tensor = tensor[:, :seq_len]
                                self.logger.debug(
                                    f"🔧 Truncated {key}: {original_shape} -> {tensor.shape}")
                            else:
                                pad_len = seq_len - tensor.size(1)
                                padding = torch.ones(
                                    batch_size, pad_len, dtype=tensor.dtype, device=tensor.device)
                                tensor = torch.cat([tensor, padding], dim=1)
                                self.logger.debug(
                                    f"🔧 Padded {key}: {original_shape} -> {tensor.shape}")

                elif key in ['lengths', 'next_lengths']:
                    # Length tensors should be (batch_size,)
                    if tensor.dim() == 1:
                        if tensor.size(0) != batch_size:
                            if tensor.size(0) > batch_size:
                                tensor = tensor[:batch_size]
                                self.logger.debug(
                                    f"🔧 Truncated {key}: {original_shape} -> {tensor.shape}")
                            else:
                                pad_len = batch_size - tensor.size(0)
                                padding = torch.full(
                                    (pad_len,), seq_len, dtype=tensor.dtype, device=tensor.device)
                                tensor = torch.cat([tensor, padding], dim=0)
                                self.logger.debug(
                                    f"🔧 Padded {key}: {original_shape} -> {tensor.shape}")
                        # Clamp lengths to valid range
                        tensor = torch.clamp(tensor, min=1, max=seq_len)
                    else:
                        raise ValueError(
                            f"Invalid dimensions for {key}: {tensor.shape}")

                elif key in ['edge_index']:
                    # Edge index should be (2, num_edges) - usually empty for non-graph models
                    if tensor.dim() != 2 or tensor.size(0) != 2:
                        self.logger.warning(
                            f"⚠️ Invalid edge_index shape: {tensor.shape}, creating empty edge_index")
                        tensor = torch.empty(
                            (2, 0), dtype=torch.long, device=tensor.device)

                # Additional validation for all tensors
                if torch.isnan(tensor).any():
                    nan_count = torch.isnan(tensor).sum().item()
                    self.logger.warning(
                        f"⚠️ Found {nan_count} NaN values in {key}, replacing with 0")
                    tensor = torch.nan_to_num(tensor, nan=0.0)

                if torch.isinf(tensor).any():
                    inf_count = torch.isinf(tensor).sum().item()
                    self.logger.warning(
                        f"⚠️ Found {inf_count} infinite values in {key}, clamping")
                    tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)

                fixed_data[key] = tensor

            except Exception as e:
                self.logger.error(
                    f"❌ Failed to fix tensor shape for {key}: {e}")
                self.logger.error(
                    f"🔍 Original shape: {original_shape}, expected batch_size: {batch_size}, seq_len: {seq_len}")
                raise ValueError(
                    f"Cannot fix tensor shape for {key}: {e}") from e

        return fixed_data
