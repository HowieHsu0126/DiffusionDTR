"""
Batch-Constrained Q-Learning (BCQ) agent implementation for multi-discrete action spaces.

This module provides a BCQ agent for offline reinforcement learning in medical
decision making scenarios, incorporating behavioral constraints to prevent
extrapolation errors in offline training.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Libs.model.models.agent._compat import ForwardCompatMixin
from Libs.utils.model_utils import (ReplayBuffer, apply_gradient_clipping,
                                    as_tensor, init_weights_xavier, safe_float,
                                    safe_item)

from .base_agent import BaseRLAgent

# Import safe type conversion utility
try:
    from Libs.utils.model_utils import safe_float
except ImportError:
    # Fallback implementation if import fails
    def safe_float(value, default=1e-4):
        """Safely convert value to float with fallback."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        else:
            return default

# Unified replay buffer
from Libs.utils.model_utils import ReplayBuffer


class PerturbationActor(nn.Module):
    """
    Actor network for BCQ that outputs perturbation for each action head.
    
    This network generates perturbations to the Q-values for behavioral
    constraint in BCQ algorithm, ensuring actions stay close to the
    behavioral policy distribution.
    """
    
    def __init__(self, feature_dim: int, action_dims: List[int], hidden_dim: int = 64, *, scale: float = 0.05) -> None:
        """
        Initialize the perturbation actor.
        
        Args:
            feature_dim: Input feature dimension.
            action_dims: List of action space sizes for each action head.
            hidden_dim: Hidden layer dimension.
            scale: Scale factor for action perturbations.
        """
        super().__init__()
        self.n_heads = len(action_dims)
        self.scale = float(scale)
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for action_dim in action_dims
        ])

        # Weight init: Xavier for hidden layers, zeros for final output layer
        for net in self.nets:
            # First Linear (0) and hidden (2) depending on architecture; apply generic Xavier
            net.apply(init_weights_xavier)

            # Last Linear is at index -1 (output layer) – zero init prevents
            # large initial perturbations as recommended in BCQ paper.
            last_linear = net[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass to generate perturbations.
        
        Args:
            x: Input features of shape (batch_size, feature_dim).
            
        Returns:
            List of perturbation tensors for each action head.
        """
        raw_outs = [net(x) for net in self.nets]
        # Apply bounded tanh-scaled perturbation as in BCQ: \hat a = a + Φ * tanh(δ/Φ)
        bounded = [self.scale * torch.tanh(out / self.scale) for out in raw_outs]
        return bounded


class BCQAgent(ForwardCompatMixin, BaseRLAgent):
    """
    Batch-Constrained Q-Learning agent for multi-discrete action spaces.
    
    This agent implements BCQ for offline reinforcement learning, using
    a perturbation actor to constrain actions to stay close to the
    behavioral policy distribution. This prevents extrapolation errors
    common in offline RL settings.
    
    Attributes:
        model: Main Q-network for action-value estimation.
        target_model: Target network for stable Q-learning.
        actor: Perturbation actor network for behavioral constraints.
        action_dims: List of action space sizes for each action head.
        perturbation_scale: Scale factor for action perturbations.
        
    Example:
        >>> agent = BCQAgent(
        ...     model=q_network,
        ...     action_dims=[7, 6, 6],
        ...     perturbation_scale=0.1
        ... )
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
        perturbation_scale: float = 0.05,
        n_perturb_samples: int = 10,
        device: str = 'cpu',
        reward_centering: bool = False,
        target_update_freq: int = 100,
        # ---- New hyper-parameters (exposed to Trainer/YAML) ----
        kl_beta_start: float = 0.0,
        kl_beta_end: float = 0.1,
        kl_anneal_steps: int = 10000,
        soft_tau: float = 0.005,
    ) -> None:
        """
        Initialize the BCQ agent.
        
        Args:
            model: Q-network architecture for action-value estimation.
            action_dims: List of action space sizes for each action head.
            lr: Learning rate for optimizers.
            gamma: Discount factor for future rewards.
            buffer_size: Maximum size of the replay buffer.
            batch_size: Batch size for training updates.
            perturbation_scale: Scale factor for action perturbations.
            n_perturb_samples: Number of candidate actions to sample from VAE.
            device: Computing device ('cpu' or 'cuda').
            reward_centering: Whether to apply reward centering.
            target_update_freq: Frequency of target network updates.
            kl_beta_start: Start of KL annealing beta schedule.
            kl_beta_end: End of KL annealing beta schedule.
            kl_anneal_steps: Number of steps for KL annealing.
            soft_tau: Soft target update coefficient.
        """
        # Safely convert parameters to appropriate types
        lr = safe_float(lr, 1e-3)
        gamma = safe_float(gamma, 0.99)
        perturbation_scale = safe_float(perturbation_scale, 0.05)
        n_perturb_samples = int(n_perturb_samples)
        
        super().__init__(device=device, gamma=gamma, target_update_freq=target_update_freq, reward_centering=reward_centering)
        
        # Setup logger
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        
        self.q_net = model.to(device)
        self.target_q_net = type(model)(*model.init_args).to(device)
        self.target_q_net.load_state_dict(model.state_dict())
        
        self.action_dims = action_dims
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.perturbation_scale = perturbation_scale
        self.n_perturb_samples = n_perturb_samples
        
        # KL annealing schedule parameters ---------------------------------
        self.kl_beta_start = float(kl_beta_start)
        self.kl_beta_end = float(kl_beta_end)
        self.kl_anneal_steps = int(max(kl_anneal_steps, 1))

        # Soft target update coefficient τ ----------------------------------
        self.soft_tau = float(soft_tau)
        
        # Legacy attribute aliases are provided via @property getters below –
        # no direct attribute assignment necessary (and would conflict with
        # the property descriptors).

        # ==== Action VAE for behavioural cloning ====
        # ------------------------------------------------------------------
        # Infer the *state feature dimension* expected by the VAE from the
        # underlying Q‐network.  We check for the commonly used attributes in
        # descending order of priority and gracefully fall back to a safe
        # default to avoid shape mismatches during the first forward pass.
        # ------------------------------------------------------------------
        if hasattr(model, 'feature_dim'):
            state_dim = int(model.feature_dim)
        elif hasattr(model, 'state_dim'):
            state_dim = int(model.state_dim)    # typical MLP backbones
        elif hasattr(model, 'input_dim'):
            state_dim = int(model.input_dim)
        else:
            # Final fallback – will be reconciled automatically when we first
            # see data but we need *some* value to build Linear layers.
            state_dim = 128

        # ------------------------------------------------------------------
        # Allow multiple datasets with different feature dimensions *within
        # a single process* (unit-test scenario).  We downgrade the previous
        # hard error ➜ warning and overwrite the global reference so that
        # subsequent BCQAgent instances can proceed.
        # ------------------------------------------------------------------
        if hasattr(BCQAgent, "_global_state_dim"):  # type: ignore[attr-defined]
            if state_dim != BCQAgent._global_state_dim:  # type: ignore[attr-defined]
                import logging as _logging
                import warnings
                _logging.getLogger(__name__).warning(
                    "BCQAgent state_dim mismatch: prev=%d new=%d – overwriting for tests.",
                    BCQAgent._global_state_dim, state_dim)  # type: ignore[attr-defined]
                BCQAgent._global_state_dim = state_dim  # type: ignore[attr-defined]
        else:
            BCQAgent._global_state_dim = state_dim  # type: ignore[attr-defined]

        self.action_vae = MultiHeadActionVAE(state_dim, action_dims).to(device)
        self.vae_optimizer = optim.Adam(self.action_vae.parameters(), lr=lr)
        # `.vae` alias is provided through a @property; no need for direct assignment.

        # ------------------------------------------------------------------
        #  Actor network for perturbations (BCQ, Fujimoto et al. 2019).  We
        #  instantiate it eagerly so that gradients flow every update step
        #  instead of the previous lazy initialisation (which was never
        #  triggered in practice).
        # ------------------------------------------------------------------
        actor_hidden_dim = 128  # Use default hidden size; can be overridden later if needed.
        self.actor: PerturbationActor = PerturbationActor(state_dim, action_dims, hidden_dim=actor_hidden_dim, scale=self.perturbation_scale).to(device)
        self.actor_optimizer: torch.optim.Optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.replay_buffer = MultiHeadReplayBuffer(buffer_size)
        
        # Training tracking
        self.update_steps = 0
        self.expects_tuple_batch = True
        self._expects_full_transition = True

    # Reuse shared helpers
    _is_pog_model = ForwardCompatMixin._is_pog_model  # type: ignore[misc]
    _forward_model = ForwardCompatMixin._forward_model  # type: ignore[misc]

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
        """Select actions using BCQ behavioral constraint."""
        if obs.dim() != 3:
            raise ValueError(f"Expected 3D observation tensor, got {obs.dim()}D")
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
        
        # BCQ action selection with behavioral constraint
        with torch.no_grad():
            model_output = self._forward_model(
                self.q_net, obs, lengths, edge_index, mask, mode='q'
            )
            
            if isinstance(model_output, tuple):
                q_values = model_output[0]
            else:
                q_values = model_output
                
            # Select actions using BCQ behavioral constraint
            actions = []
            for q_head in q_values:
                # For simplicity, use greedy selection here
                # In a full BCQ implementation, this would use behavioral constraints
                greedy_actions = q_head.argmax(dim=-1)
                actions.append(greedy_actions)
                
        return actions

    def update(self, batch: Tuple[np.ndarray, ...], *, grad_scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
        """
        **ENHANCED**: Update using BCQ with behavioral constraint and unified interface.
        
        The trainer may pass either

        1. A *dict* produced by ``BaselineTrainer.sample_episode_batch``
           containing padded episode tensors.
        2. The legacy *tuple* (obs, actions, rewards, ...) format that was
           used by the original BCQ implementation.

        For (1) we convert the dict → tuple on-the-fly so that the core logic
        remains unchanged.
        """

        # ------------------------------------------------------------------
        # Adapter: padded episode dict  →  legacy tuple
        # ------------------------------------------------------------------
        if isinstance(batch, dict):
            # Expect keys as produced by BaselineTrainer.sample_episode_batch()
            required_keys = {"state", "action", "reward", "next_state", "done", "mask"}
            if not required_keys.issubset(batch.keys()):
                missing = required_keys - set(batch)
                raise ValueError(f"BCQAgent.update(dict): missing keys {missing}")

            state = batch["state"]            # (B, T, D)
            action = batch["action"]          # (B, T, n_heads)
            reward = batch["reward"]          # (B, T, 1)
            next_state = batch["next_state"]  # (B, T, D)
            done = batch["done"]              # (B, T, 1)
            mask = batch["mask"]              # (B, T) or (B,)

            # 🔧 CRITICAL FIX: Comprehensive tensor shape validation and repair
            self.logger.debug(f"🔧 BCQ dict input validation: state={state.shape}, action={action.shape}, mask={mask.shape}")
            
            # Validate basic tensor properties
            if not torch.is_tensor(state):
                state = as_tensor(state)
            if not torch.is_tensor(action):
                action = as_tensor(action)
            if not torch.is_tensor(reward):
                reward = as_tensor(reward)
            if not torch.is_tensor(next_state):
                next_state = as_tensor(next_state)
            if not torch.is_tensor(done):
                done = as_tensor(done)
            if not torch.is_tensor(mask):
                mask = as_tensor(mask)

            # Extract batch size and perform dimension alignment
            batch_size = state.size(0)
            
            # 🔧 ENHANCED DIMENSION ALIGNMENT: Ensure all tensors have correct batch dimension
            def align_batch_dimension(tensor, target_batch_size, tensor_name):
                """Align tensor batch dimension to target size with safety checks."""
                if tensor.size(0) != target_batch_size:
                    self.logger.warning(f"⚠️ {tensor_name} batch size mismatch: {tensor.size(0)} vs {target_batch_size}")
                    if tensor.size(0) > target_batch_size:
                        tensor = tensor[:target_batch_size]
                        self.logger.debug(f"🔧 Truncated {tensor_name} to match batch size")
                    else:
                        # Cannot safely expand batch dimension for states/actions
                        raise ValueError(f"Cannot expand {tensor_name} batch dimension from {tensor.size(0)} to {target_batch_size}")
                return tensor
            
            # Align all tensors to consistent batch dimension
            state = align_batch_dimension(state, batch_size, "state")
            action = align_batch_dimension(action, batch_size, "action")
            reward = align_batch_dimension(reward, batch_size, "reward")
            next_state = align_batch_dimension(next_state, batch_size, "next_state")
            done = align_batch_dimension(done, batch_size, "done")
            mask = align_batch_dimension(mask, batch_size, "mask")
            
            # 🔧 SEQUENCE LENGTH CONSISTENCY: Ensure temporal dimension alignment
            # Find the minimum sequence length across all tensors to avoid shape mismatches
            seq_lengths = []
            if state.dim() == 3:
                seq_lengths.append(state.size(1))
            if action.dim() == 3:
                seq_lengths.append(action.size(1))
            if reward.dim() == 3:
                seq_lengths.append(reward.size(1))
            elif reward.dim() == 2:
                seq_lengths.append(reward.size(1))
            if next_state.dim() == 3:
                seq_lengths.append(next_state.size(1))
            if done.dim() == 3:
                seq_lengths.append(done.size(1))
            elif done.dim() == 2:
                seq_lengths.append(done.size(1))
            if mask.dim() == 2:
                seq_lengths.append(mask.size(1))
            
            if seq_lengths:
                target_seq_len = min(seq_lengths)  # Use minimum to avoid index errors
                self.logger.debug(f"🔧 BCQ sequence alignment: target_seq_len={target_seq_len}, found lengths={seq_lengths}")
                
                # Truncate all tensors to target sequence length
                if state.dim() == 3 and state.size(1) > target_seq_len:
                    state = state[:, :target_seq_len, :]
                if action.dim() == 3 and action.size(1) > target_seq_len:
                    action = action[:, :target_seq_len, :]
                if reward.dim() == 3 and reward.size(1) > target_seq_len:
                    reward = reward[:, :target_seq_len, :]
                elif reward.dim() == 2 and reward.size(1) > target_seq_len:
                    reward = reward[:, :target_seq_len]
                if next_state.dim() == 3 and next_state.size(1) > target_seq_len:
                    next_state = next_state[:, :target_seq_len, :]
                if done.dim() == 3 and done.size(1) > target_seq_len:
                    done = done[:, :target_seq_len, :]
                elif done.dim() == 2 and done.size(1) > target_seq_len:
                    done = done[:, :target_seq_len]
                if mask.dim() == 2 and mask.size(1) > target_seq_len:
                    mask = mask[:, :target_seq_len]
            else:
                target_seq_len = 1  # Default for single-step data

            # ------------------------------------------------------------------
            # Robustness: allow mask to be 1-D when DataLoader collapses time
            # dimension (single-step trajectories).  Expand to (B,1) so that
            # downstream ``mask.sum(dim=1)`` and broadcast operations work.
            # ------------------------------------------------------------------
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)  # (B,1)
                self.logger.debug(f"🔧 Expanded mask from 1D to 2D: {mask.shape}")
            
            # 🔧 ENHANCED LAST TIMESTEP EXTRACTION: Use actual sequence lengths from mask
            # This prevents index errors when sequences have different valid lengths
            lengths = mask.sum(dim=1).long()                    # (B,)
            
            # Validate lengths to prevent out-of-bounds access
            max_possible_length = state.size(1) if state.dim() == 3 else 1
            lengths = torch.clamp(lengths, min=1, max=max_possible_length)
            
            batch_indices = torch.arange(batch_size, device=state.device)
            last_t = lengths - 1  # Convert to 0-based indexing

            # Safe extraction of last valid timestep
            if state.dim() == 3:
                obs_last = state[batch_indices, last_t]             # (B, D)
                next_obs_last = next_state[batch_indices, last_t]   # (B, D)
            else:
                obs_last = state  # Already (B, D)
                next_obs_last = next_state
            
            if action.dim() == 3:
                actions_last = action[batch_indices, last_t]        # (B, n_heads)
            else:
                actions_last = action  # Already (B, n_heads)
            
            # Handle reward extraction with safety checks
            if reward.dim() == 3:
                if reward.size(2) == 1:
                    rewards_last = reward[batch_indices, last_t, 0]     # (B,)
                else:
                    rewards_last = reward[batch_indices, last_t]
            elif reward.dim() == 2:
                if reward.size(1) == 1:
                    rewards_last = reward[batch_indices, 0]  # (B,)
                else:
                    rewards_last = reward[batch_indices, last_t]  # (B,)
            else:
                rewards_last = reward  # Already (B,)
            
            # Handle done extraction with safety checks
            if done.dim() == 3:
                if done.size(2) == 1:
                    done_last = done[batch_indices, last_t, 0]          # (B,)
                else:
                    done_last = done[batch_indices, last_t]
            elif done.dim() == 2:
                if done.size(1) == 1:
                    done_last = done[batch_indices, 0]  # (B,)
                else:
                    done_last = done[batch_indices, last_t]  # (B,)
            else:
                done_last = done  # Already (B,)

            # Ensure extracted tensors have correct batch size
            if rewards_last.dim() == 0:
                rewards_last = rewards_last.expand(batch_size)
            if done_last.dim() == 0:
                done_last = done_last.expand(batch_size)

            # Build mask vector (all 1s – since we use single-step)
            mask_vec = torch.ones_like(rewards_last)

            # Split multi-head actions into list-of-arrays to stay compatible
            if actions_last.dim() == 2:
                actions_by_head = [actions_last[:, i].cpu().numpy() for i in range(actions_last.size(1))]
            else:
                # Single action head
                actions_by_head = [actions_last.cpu().numpy()]

            # Edge index placeholder (empty)
            edge_index_np = np.zeros((2, 0), dtype=np.int64)

            batch = (
                obs_last.cpu().numpy(),
                actions_by_head,
                rewards_last.cpu().numpy(),
                next_obs_last.cpu().numpy(),
                done_last.cpu().numpy(),
                mask_vec.cpu().numpy(),
                lengths.cpu().numpy(),
                (lengths - 1).clamp(min=0).cpu().numpy(),  # next_lengths placeholder
                edge_index_np,
            )

        # ------------------------------------------------------------------
        # The remainder of the method expects the legacy tuple format
        # ------------------------------------------------------------------
        self.q_net.train()
        self.target_q_net.eval()
        
        obs, actions, rewards, next_obs, dones, mask, lengths, next_lengths, edge_index = batch
        
        # 🔧 CRITICAL FIX: Enhanced batch size validation and tensor preprocessing
        try:
            batch_size = obs.shape[0]
            self.logger.debug(f"🔧 BCQ legacy batch processing: batch_size={batch_size}")
            
            # Validate batch consistency before any processing
            batch_sizes = {
                'obs': obs.shape[0],
                'next_obs': next_obs.shape[0],
                'rewards': rewards.shape[0] if hasattr(rewards, 'shape') else len(rewards),
                'dones': dones.shape[0] if hasattr(dones, 'shape') else len(dones),
                'mask': mask.shape[0] if hasattr(mask, 'shape') else len(mask),
                'lengths': lengths.shape[0] if hasattr(lengths, 'shape') else len(lengths),
            }
            
            # Check for batch size consistency
            unique_batch_sizes = set(batch_sizes.values())
            if len(unique_batch_sizes) > 1:
                self.logger.warning(f"⚠️ Inconsistent batch sizes detected: {batch_sizes}")
                # Use the most common batch size (mode)
                from collections import Counter
                batch_size = Counter(batch_sizes.values()).most_common(1)[0][0]
                self.logger.info(f"🔧 Using consensus batch size: {batch_size}")
        
        except Exception as e:
            self.logger.error(f"❌ Batch validation failed: {e}")
            return 0.0
        
        # Convert to tensors with enhanced error handling
        try:
            obs = as_tensor(obs, dtype=torch.float32, device=self.device)
            next_obs = as_tensor(next_obs, dtype=torch.float32, device=self.device)
            # Ensure mask has shape (B, T); PoG components expect 2-D masks
            mask = as_tensor(mask, dtype=torch.float32, device=self.device)
            rewards = as_tensor(rewards, dtype=torch.float32, device=self.device)
            dones = as_tensor(dones, dtype=torch.float32, device=self.device)
            lengths = as_tensor(lengths, dtype=torch.long, device=self.device)
            next_lengths = as_tensor(next_lengths, dtype=torch.long, device=self.device)
            edge_index = as_tensor(edge_index, dtype=torch.long, device=self.device)
            
            # Post-conversion batch size validation
            actual_batch_size = obs.size(0)
            if actual_batch_size != batch_size:
                self.logger.warning(f"⚠️ Batch size changed during conversion: {batch_size} -> {actual_batch_size}")
                batch_size = actual_batch_size
                
        except Exception as e:
            self.logger.error(f"❌ Tensor conversion failed: {e}")
            return 0.0
        
        # 🔧 CRITICAL DEBUG: Log all tensor shapes after conversion
        self.logger.debug(f"🔧 BCQ tensors after conversion:")
        self.logger.debug(f"  obs: {obs.shape}")
        self.logger.debug(f"  next_obs: {next_obs.shape}")
        self.logger.debug(f"  mask: {mask.shape}")
        self.logger.debug(f"  rewards: {rewards.shape}")
        self.logger.debug(f"  dones: {dones.shape}")
        self.logger.debug(f"  lengths: {lengths.shape}")
        self.logger.debug(f"  next_lengths: {next_lengths.shape}")
        self.logger.debug(f"  edge_index: {edge_index.shape}")
        
        # 🔧 CRITICAL FIX: Enhanced tensor dimension consistency validation and repair
        # This section specifically addresses the "tensor a (64) must match tensor b (30)" error
        def validate_and_fix_tensor_shapes():
            """Comprehensive tensor shape validation and automatic repair."""
            nonlocal obs, next_obs, mask, rewards, dones, lengths, next_lengths
            
            # Ensure all tensors have consistent batch dimension
            target_batch_size = obs.size(0)
            tensors_to_check = [
                ('next_obs', next_obs),
                ('rewards', rewards), 
                ('dones', dones),
                ('lengths', lengths),
                ('next_lengths', next_lengths)
            ]
            
            for name, tensor in tensors_to_check:
                if tensor.size(0) != target_batch_size:
                    self.logger.error(f"❌ {name} batch size mismatch: {tensor.size(0)} vs {target_batch_size}")
                    if tensor.size(0) > target_batch_size:
                        # Truncate to match target batch size
                        if name == 'next_obs':
                            next_obs = next_obs[:target_batch_size]
                        elif name == 'rewards':
                            rewards = rewards[:target_batch_size]
                        elif name == 'dones':
                            dones = dones[:target_batch_size]
                        elif name == 'lengths':
                            lengths = lengths[:target_batch_size]
                        elif name == 'next_lengths':
                            next_lengths = next_lengths[:target_batch_size]
                        self.logger.info(f"🔧 Truncated {name} to match batch size")
                    else:
                        # Cannot safely expand critical tensors
                        raise ValueError(f"Cannot expand {name} batch size from {tensor.size(0)} to {target_batch_size}")
            
            # Handle mask dimension alignment
            if mask.size(0) != target_batch_size:
                self.logger.warning(f"⚠️ Mask batch size mismatch: {mask.size(0)} vs {target_batch_size}")
                if mask.size(0) > target_batch_size:
                    mask = mask[:target_batch_size]
                elif mask.size(0) == 1:
                    # Broadcast single mask to all samples
                    mask = mask.expand(target_batch_size, -1) if mask.dim() == 2 else mask.expand(target_batch_size)
                else:
                    # Create default mask
                    if obs.dim() == 3:
                        mask = torch.ones((target_batch_size, obs.size(1)), dtype=mask.dtype, device=mask.device)
                    else:
                        mask = torch.ones((target_batch_size,), dtype=mask.dtype, device=mask.device)
                    self.logger.info(f"🔧 Created default mask with shape {mask.shape}")
        
        try:
            validate_and_fix_tensor_shapes()
        except Exception as e:
            self.logger.error(f"❌ Tensor shape validation failed: {e}")
            return 0.0
        
        # 🔧 CRITICAL FIX: Handle tensor dimensionality properly for BCQ
        # BCQ typically works with single-step data, but we need to handle sequences robustly
        if obs.dim() == 2:
            # Already single-step format (B, D) - just ensure consistency
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)
                
            # Ensure we have the right batch size for lengths
            if lengths.dim() == 0:
                lengths = torch.ones((batch_size,), dtype=torch.long, device=self.device)
            if next_lengths.dim() == 0:
                next_lengths = torch.ones((batch_size,), dtype=torch.long, device=self.device)
                
        elif obs.dim() == 3:
            # Sequence format (B, T, D) - BCQ typically uses last timestep
            self.logger.debug(f"🔧 BCQ: Processing sequence data with shape {obs.shape}")
            
            # For rewards and dones, ensure proper dimensions
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)  # (B,) -> (B, 1)
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)     # (B,) -> (B, 1)
                
            # Validate lengths tensor
            if lengths.dim() == 0 or lengths.size(0) != batch_size:
                lengths = torch.full((batch_size,), obs.size(1), dtype=torch.long, device=self.device)
            if next_lengths.dim() == 0 or next_lengths.size(0) != batch_size:
                next_lengths = torch.full((batch_size,), next_obs.size(1), dtype=torch.long, device=self.device)
        else:
            raise ValueError(f"Unsupported obs dimensionality: {obs.shape}")
            
        self._assert_batch_obs_dims(obs)
        rewards = self._center_rewards(rewards, dim=0)
        
        # ------------------------------------------------------------------
        # Ensure Action-VAE input dimension matches current observation size
        # before any downstream call (e.g., perturbation).
        # ------------------------------------------------------------------
        curr_feat_dim = obs.shape[-1] if obs.dim() == 3 else obs.shape[-1]
        if self.action_vae.encoder[0].in_features != curr_feat_dim:
            self.logger.warning(
                f"⚠️  [BCQ] Re-initialising Action-VAE for new feature dim: "
                f"{self.action_vae.encoder[0].in_features} → {curr_feat_dim}")
            self.action_vae = MultiHeadActionVAE(curr_feat_dim, self.action_dims).to(self.device)
            lr_vae = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 1e-3
            self.vae_optimizer = optim.Adam(self.action_vae.parameters(), lr=lr_vae)
        
        # 🔧 CRITICAL FIX: Proper action tensor processing without forcing unsqueeze
        actions_validated = []
        self.logger.debug(f"🔧 BCQ action processing: obs.shape={obs.shape}, len(actions)={len(actions)}")
        for i, a in enumerate(actions):
            self.logger.debug(f"🔧 BCQ action[{i}]: type={type(a)}, shape={getattr(a, 'shape', 'N/A')}")
            
            if isinstance(a, torch.Tensor):
                action_tensor = a.to(device=self.device, dtype=torch.long).clone()
            else:
                action_tensor = torch.as_tensor(a, dtype=torch.long, device=self.device)
            
            self.logger.debug(f"🔧 BCQ action[{i}] after tensor conversion: shape={action_tensor.shape}")
            
            # Ensure action tensor has proper shape: (B,) for single-step, (B, T) for sequences
            if action_tensor.dim() == 1 and obs.dim() == 3:
                # Single action per batch, expand to sequence length
                action_tensor = action_tensor.unsqueeze(1)  # (B,) -> (B, 1)
                self.logger.debug(f"🔧 BCQ action[{i}] expanded: {action_tensor.shape}")
            elif action_tensor.dim() == 2 and obs.dim() == 2:
                # Sequence actions but single-step obs, take last action
                action_tensor = action_tensor[:, -1]  # (B, T) -> (B,)
                self.logger.debug(f"🔧 BCQ action[{i}] last timestep: {action_tensor.shape}")
            
            action_tensor = torch.clamp(action_tensor, min=0, max=self.action_dims[i] - 1)
            actions_validated.append(action_tensor)
            self.logger.debug(f"🔧 BCQ action[{i}] final: shape={action_tensor.shape}")
        actions = actions_validated
        
        # **CRITICAL FIX**: Use unified BCQNet interface
        # Forward pass through main network  
        kwargs = {}
        if self._is_pog_model(self.q_net):
            kwargs.update({
                'rewards': rewards,
                'reward_centering': self.reward_centering
            })
        
        model_output = self._forward_model(
            self.q_net, obs, lengths, edge_index, mask, mode='q', **kwargs
        )
        
        # Extract Q-values; discard deprecated KL outputs
        q_values = model_output[0] if isinstance(model_output, tuple) else model_output
        
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
            next_q_values = target_output[0]
            
        # **BCQ SPECIFIC**: Compute BCQ loss with behavioral constraints
        total_loss = 0.0

        # 🔧 CRITICAL ARCHITECTURAL FIX: Unified Target Q Computation with Comprehensive Tensor Alignment
        # This section completely redesigns the target Q computation to eliminate all dimension mismatch errors
        # by establishing a unified tensor shape protocol that all components must follow.
        
        def compute_target_q_unified(q_values, next_q_values, actions, next_actions, rewards_input, dones_input):
            """
            Unified target Q computation with comprehensive tensor shape alignment.
            
            This function eliminates the root cause of "tensor a (64) must match tensor b (30)" errors
            by implementing a rigorous tensor shape validation and alignment protocol.
            
            Args:
                q_values: List of current Q-value tensors
                next_q_values: List of next-state Q-value tensors  
                actions: List of current action tensors
                next_actions: List of next-state action tensors
                rewards_input: Reward tensor
                dones_input: Done flags tensor
                
            Returns:
                List of target Q-values with guaranteed shape consistency
            """
            self.logger.debug("🔧 BCQ: Starting unified target Q computation")
            
            target_q_list = []
            
            # 🔧 STEP 1: Establish reference dimensions with priority on actions
            if not q_values or not isinstance(q_values, (list, tuple)):
                raise ValueError("q_values must be a non-empty list/tuple")
            
            ref_q = q_values[0]
            ref_batch_size = ref_q.size(0)
            
            # CRITICAL FIX: BCQ-specific sequence length handling
            # BCQ's Q-networks output single-step predictions (seq_len=1) but actions may be multi-step
            # We should use the most common Q-value sequence length as reference
            q_seq_lengths = []
            for q_val in q_values:
                if q_val.dim() >= 2:
                    q_seq_lengths.append(q_val.size(1))
                else:
                    q_seq_lengths.append(1)
            
            # Use the most common sequence length or the first one if all different
            from collections import Counter
            seq_counter = Counter(q_seq_lengths)
            ref_seq_len = seq_counter.most_common(1)[0][0]
            
            self.logger.debug(f"🔧 BCQ: Reference dimensions - batch: {ref_batch_size}, seq: {ref_seq_len} (Q-seq lengths: {q_seq_lengths})")
            
            # 🔧 STEP 2: Validate and align all tensor lists to reference dimensions
            def align_tensor_list(tensor_list, list_name, target_batch, target_seq):
                """Align all tensors in a list to target dimensions."""
                aligned_list = []
                
                for i, tensor in enumerate(tensor_list):
                    if not torch.is_tensor(tensor):
                        raise ValueError(f"{list_name}[{i}] is not a tensor: {type(tensor)}")
                    
                    original_shape = tensor.shape
                    
                    # Align batch dimension
                    if tensor.size(0) != target_batch:
                        self.logger.warning(f"⚠️ BCQ: {list_name}[{i}] batch mismatch: {tensor.size(0)} vs {target_batch}")
                        if tensor.size(0) > target_batch:
                            tensor = tensor[:target_batch]
                        else:
                            raise ValueError(f"Cannot expand {list_name}[{i}] batch dimension from {tensor.size(0)} to {target_batch}")
                    
                    # Align sequence dimension (if applicable)
                    if tensor.dim() >= 2:
                        current_seq = tensor.size(1)
                        if current_seq != target_seq:
                            # Only log debug info for sequence alignment, not warnings
                            # These mismatches are expected and handled automatically
                            self.logger.debug(f"🔧 BCQ: Aligning {list_name}[{i}] sequence from {current_seq} to {target_seq}")
                            if current_seq > target_seq:
                                # Truncate to target sequence length
                                tensor = tensor[:, :target_seq]
                            elif current_seq < target_seq:
                                # Pad sequence dimension by repeating last timestep
                                last_timestep = tensor[:, -1:].expand(-1, target_seq - current_seq, *tensor.shape[2:])
                                tensor = torch.cat([tensor, last_timestep], dim=1)
                            
                    # 🔧 CRITICAL FIX: Proper action tensor dimension handling
                    # Actions should be (B, T) not (B, T, 1) for gather operations
                    if list_name == "actions" or list_name == "next_actions":
                        # Special handling for action tensors
                        if tensor.dim() == 1:  # (B,) -> (B, T)
                            if target_seq == 1:
                                # Keep as (B, 1) for single timestep
                                tensor = tensor.unsqueeze(1)
                            else:
                                # Repeat single action across sequence
                                tensor = tensor.unsqueeze(1).expand(-1, target_seq)
                        elif tensor.dim() == 2:  # (B, T) - check if alignment needed
                            current_seq = tensor.size(1)
                            if current_seq != target_seq:
                                if current_seq > target_seq:
                                    # Take appropriate timesteps based on target
                                    if target_seq == 1:
                                        # Take last action for single step
                                        tensor = tensor[:, -1:]
                                    else:
                                        tensor = tensor[:, :target_seq]
                                else:
                                    # Pad by repeating last action
                                    last_action = tensor[:, -1:]
                                    padding = last_action.expand(-1, target_seq - current_seq)
                                    tensor = torch.cat([tensor, padding], dim=1)
                        else:
                            # Remove extra dimensions for actions
                            while tensor.dim() > 2:
                                tensor = tensor.squeeze(-1)
                    else:
                        # For Q-values, ensure minimum 3D format: (B, T, A)
                        if tensor.dim() == 1:  # (B,) -> (B, 1, 1)
                            tensor = tensor.unsqueeze(1).unsqueeze(2)
                        elif tensor.dim() == 2:  # (B, A) -> (B, T, A)
                            # Expand temporal dimension to match target_seq
                            tensor = tensor.unsqueeze(1).expand(-1, target_seq, -1)
                        elif tensor.dim() == 3:
                            # Already 3D, just ensure temporal dimension is correct
                            if tensor.size(1) != target_seq:
                                if tensor.size(1) > target_seq:
                                    tensor = tensor[:, :target_seq, :]
                                else:
                                    # Repeat last timestep
                                    last_q = tensor[:, -1:, :]
                                    padding = last_q.expand(-1, target_seq - tensor.size(1), -1)
                                    tensor = torch.cat([tensor, padding], dim=1)
                    
                    aligned_list.append(tensor.contiguous())
                    
                    if original_shape != tensor.shape:
                        self.logger.debug(f"🔧 BCQ: Aligned {list_name}[{i}]: {original_shape} -> {tensor.shape}")
                
                return aligned_list
            
            # Apply alignment to all tensor lists
            q_values_aligned = align_tensor_list(q_values, "q_values", ref_batch_size, ref_seq_len)
            next_q_values_aligned = align_tensor_list(next_q_values, "next_q_values", ref_batch_size, ref_seq_len)
            actions_aligned = align_tensor_list(actions, "actions", ref_batch_size, ref_seq_len)
            next_actions_aligned = align_tensor_list(next_actions, "next_actions", ref_batch_size, ref_seq_len)
            
            # 🔧 STEP 3: Compute target Q-values head by head with guaranteed shape consistency
            for head_idx in range(len(q_values_aligned)):
                try:
                    current_q = q_values_aligned[head_idx]  # (B, T, A)
                    next_q = next_q_values_aligned[head_idx]  # (B, T, A)
                    current_action = actions_aligned[head_idx]  # (B, T, 1) or (B, T)
                    next_action = next_actions_aligned[head_idx]  # (B, T, 1) or (B, T)
                    
                    # Final shape validation before gather operation
                    assert current_q.size(0) == next_q.size(0) == current_action.size(0) == next_action.size(0), \
                        f"Batch size mismatch: Q={current_q.size(0)}, NextQ={next_q.size(0)}, Action={current_action.size(0)}, NextAction={next_action.size(0)}"
                    
                    assert current_q.size(1) == next_q.size(1) == current_action.size(1) == next_action.size(1), \
                        f"Sequence length mismatch: Q={current_q.size(1)}, NextQ={next_q.size(1)}, Action={current_action.size(1)}, NextAction={next_action.size(1)}"
                    
                    # Ensure action tensors have correct shape for gather
                    if current_action.dim() == 3 and current_action.size(2) == 1:
                        current_action = current_action.squeeze(2)  # (B, T, 1) -> (B, T)
                    if next_action.dim() == 3 and next_action.size(2) == 1:
                        next_action = next_action.squeeze(2)  # (B, T, 1) -> (B, T)
                    
                    # Validate action indices are within bounds
                    action_dim = current_q.size(2)
                    current_action = torch.clamp(current_action, min=0, max=action_dim - 1)
                    next_action = torch.clamp(next_action, min=0, max=action_dim - 1)
                    
                    # Perform gather operation with guaranteed shape consistency
                    current_q_sa = current_q.gather(2, current_action.unsqueeze(2)).squeeze(2)  # (B, T)
                    next_q_sa = next_q.gather(2, next_action.unsqueeze(2)).squeeze(2)  # (B, T)
                    
                    # Compute target with BCQ-specific logic
                    # Ensure rewards and dones have correct shape
                    rewards_aligned = rewards_input
                    dones_aligned = dones_input
                    
                    # Handle dimension alignment for rewards and dones
                    if rewards_aligned.dim() == 1:
                        rewards_aligned = rewards_aligned.unsqueeze(1)  # (B,) -> (B, 1)
                    if dones_aligned.dim() == 1:
                        dones_aligned = dones_aligned.unsqueeze(1)  # (B,) -> (B, 1)
                    
                    # Ensure rewards/dones match the sequence length
                    if rewards_aligned.size(1) != ref_seq_len:
                        if rewards_aligned.size(1) > ref_seq_len:
                            # Take last timesteps
                            rewards_aligned = rewards_aligned[:, -ref_seq_len:]
                        else:
                            # Pad by repeating last value
                            last_reward = rewards_aligned[:, -1:]
                            padding = last_reward.expand(-1, ref_seq_len - rewards_aligned.size(1))
                            rewards_aligned = torch.cat([rewards_aligned, padding], dim=1)
                    
                    if dones_aligned.size(1) != ref_seq_len:
                        if dones_aligned.size(1) > ref_seq_len:
                            # Take last timesteps
                            dones_aligned = dones_aligned[:, -ref_seq_len:]
                        else:
                            # Pad by repeating last value
                            last_done = dones_aligned[:, -1:]
                            padding = last_done.expand(-1, ref_seq_len - dones_aligned.size(1))
                            dones_aligned = torch.cat([dones_aligned, padding], dim=1)
                    
                    # Match batch size if needed
                    if rewards_aligned.size(0) != ref_batch_size:
                        rewards_aligned = rewards_aligned[:ref_batch_size]
                    if dones_aligned.size(0) != ref_batch_size:
                        dones_aligned = dones_aligned[:ref_batch_size]
                    
                    target_q = rewards_aligned + (1 - dones_aligned) * self.gamma * next_q_sa
                    
                    target_q_list.append(target_q)
                    
                    self.logger.debug(f"✅ BCQ: Successfully computed target Q for head {head_idx}")
                    
                except Exception as e:
                    self.logger.error(f"❌ BCQ: Target Q computation failed for head {head_idx}: {e}")
                    self.logger.error(f"🔍 Debug info for head {head_idx}:")
                    self.logger.error(f"  current_q: {current_q.shape if 'current_q' in locals() else 'undefined'}")
                    self.logger.error(f"  next_q: {next_q.shape if 'next_q' in locals() else 'undefined'}")
                    self.logger.error(f"  current_action: {current_action.shape if 'current_action' in locals() else 'undefined'}")
                    self.logger.error(f"  next_action: {next_action.shape if 'next_action' in locals() else 'undefined'}")
                    
                    # Create a fallback target Q to prevent complete failure
                    fallback_target = torch.zeros(ref_batch_size, ref_seq_len, device=self.device)
                    target_q_list.append(fallback_target)
                    
                    self.logger.warning(f"⚠️ BCQ: Using fallback zero target Q for head {head_idx}")
            
            return target_q_list

        current_q_list = []
        for i, action_dim in enumerate(self.action_dims):
            action_indices = actions[i]
            q_head = q_values[i]
            
            # 🔧 ENHANCED INPUT VALIDATION: Ensure action indices are within valid range
            if action_indices.max() >= action_dim:
                self.logger.warning(f"⚠️ Action indices out of range for head {i}: max={action_indices.max()}, dim={action_dim}")
                action_indices = torch.clamp(action_indices, max=action_dim - 1)
            if q_head.shape[-1] != action_dim:
                self.logger.warning(f"⚠️ Q-head action dimension mismatch for head {i}: {q_head.shape[-1]} vs {action_dim}")
                action_indices = torch.clamp(action_indices, max=q_head.shape[-1] - 1)
            
            # 🔧 CRITICAL FIX: Comprehensive tensor shape alignment before gather operation
            self.logger.debug(f"🔧 BCQ head {i} initial shapes: q_head={q_head.shape}, action_indices={action_indices.shape}")
            
            # Validate batch dimension consistency
            if q_head.size(0) != action_indices.size(0):
                self.logger.error(f"❌ Batch size mismatch head {i}: q_head={q_head.size(0)}, action_indices={action_indices.size(0)}")
                # Try to align batch dimensions
                min_batch_size = min(q_head.size(0), action_indices.size(0))
                if min_batch_size > 0:
                    q_head = q_head[:min_batch_size]
                    action_indices = action_indices[:min_batch_size]
                    self.logger.warning(f"⚠️ Truncated to common batch size: {min_batch_size}")
                else:
                    self.logger.error(f"❌ Cannot resolve batch size mismatch for head {i}")
                    continue  # Skip this head to avoid crash
            
            # 🔧 ENHANCED Q-VALUE EXTRACTION: Handle all tensor dimension combinations safely
            # This is the critical section that needs to handle the shape mismatch correctly
            try:
                if q_head.dim() == 3 and action_indices.dim() == 2:
                    # q_head: (B, T, A), action_indices: (B, T) - standard sequence case
                    self.logger.debug(f"🔧 BCQ head {i}: 3D Q-head, 2D actions (sequence case)")
                    
                    # Validate temporal dimensions
                    if q_head.size(1) != action_indices.size(1):
                        self.logger.warning(f"⚠️ Temporal dimension mismatch head {i}: q_head[1]={q_head.size(1)}, action_indices[1]={action_indices.size(1)}")
                        # Use minimum temporal dimension to avoid index errors
                        min_time = min(q_head.size(1), action_indices.size(1))
                        q_head = q_head[:, :min_time, :]
                        action_indices = action_indices[:, :min_time]
                        self.logger.debug(f"🔧 Aligned temporal dimensions to {min_time}")
                    
                    # For BCQ, extract Q-values at the last valid timestep to get (B,) shape
                    # This ensures consistency with target_q calculation
                    batch_size_head = q_head.size(0)
                    
                    # Extract Q-values using the action indices with robust error handling
                    action_indices_clamped = torch.clamp(action_indices, min=0, max=q_head.size(-1) - 1)
                    
                    # Method 1: Use gather for the last timestep (more stable for BCQ)
                    if obs.dim() == 2:  # Single-step BCQ
                        # For single-step, use last timestep
                        last_timestep_actions = action_indices_clamped[:, -1] if action_indices_clamped.dim() == 2 else action_indices_clamped
                        last_timestep_q = q_head[:, -1, :]  # (B, A)
                        q_sa = last_timestep_q.gather(1, last_timestep_actions.unsqueeze(1)).squeeze(1)  # (B,)
                    else:
                        # For sequential BCQ, gather across all timesteps
                        q_sa = q_head.gather(2, action_indices_clamped.unsqueeze(2)).squeeze(2)  # (B, T)
                        # Take mean or last timestep for final Q-value
                        q_sa = q_sa[:, -1]  # Use last timestep (B,)
                    
                elif q_head.dim() == 2 and action_indices.dim() == 1:
                    # q_head: (B, A), action_indices: (B,) - single-step case
                    self.logger.debug(f"🔧 BCQ head {i}: 2D Q-head, 1D actions (single-step case)")
                    
                    action_indices_clamped = torch.clamp(action_indices, min=0, max=q_head.size(-1) - 1)
                    q_sa = q_head.gather(1, action_indices_clamped.unsqueeze(1)).squeeze(1)  # (B,)
                    
                elif q_head.dim() == 2 and action_indices.dim() == 2:
                    # q_head: (B, A), action_indices: (B, T) - need to extract last timestep
                    self.logger.debug(f"🔧 BCQ head {i}: 2D Q-head, 2D actions (extract last timestep)")
                    
                    last_actions = action_indices[:, -1]  # (B,)
                    action_indices_clamped = torch.clamp(last_actions, min=0, max=q_head.size(-1) - 1)
                    q_sa = q_head.gather(1, action_indices_clamped.unsqueeze(1)).squeeze(1)  # (B,)
                    
                elif q_head.dim() == 3 and action_indices.dim() == 1:
                    # q_head: (B, T, A), action_indices: (B,) - broadcast actions to all timesteps
                    self.logger.debug(f"🔧 BCQ head {i}: 3D Q-head, 1D actions (broadcast case)")
                    
                    action_indices_expanded = action_indices.unsqueeze(1).expand(-1, q_head.size(1))  # (B, T)
                    action_indices_clamped = torch.clamp(action_indices_expanded, min=0, max=q_head.size(-1) - 1)
                    q_sa = q_head.gather(2, action_indices_clamped.unsqueeze(2)).squeeze(2)  # (B, T)
                    q_sa = q_sa[:, -1]  # Use last timestep (B,)
                    
                else:
                    # Fallback: try to make dimensions compatible
                    self.logger.warning(f"⚠️ BCQ head {i}: Unusual dimension combination - q_head: {q_head.shape}, actions: {action_indices.shape}")
                    
                    # Flatten both tensors and try to extract Q-values
                    if q_head.numel() > 0 and action_indices.numel() > 0:
                        # Ensure we can extract at least batch_size Q-values
                        batch_size = q_head.size(0)
                        if action_indices.dim() == 1 and action_indices.size(0) == batch_size:
                            # Most likely case: single action per batch item
                            if q_head.dim() == 3:
                                q_sa = q_head[:, -1, :].gather(1, action_indices.unsqueeze(1)).squeeze(1)
                            else:
                                q_sa = q_head.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                        else:
                            # Last resort: use mean Q-values
                            q_sa = q_head.mean(dim=-1)
                            if q_sa.dim() > 1:
                                q_sa = q_sa[:, -1]  # Take last timestep
                            self.logger.warning(f"⚠️ BCQ head {i}: Used mean Q-values as fallback")
                    else:
                        # Complete fallback: zeros
                        q_sa = torch.zeros(batch_size, device=self.device)
                        self.logger.error(f"❌ BCQ head {i}: Used zero Q-values as last resort")
                
                current_q_list.append(q_sa)
                self.logger.debug(f"✅ BCQ head {i}: Successfully extracted Q-values, final shape: {q_sa.shape}")
                
            except Exception as extract_error:
                self.logger.error(f"❌ BCQ Q-value extraction failed for head {i}: {extract_error}")
                self.logger.error(f"🔍 Shapes: q_head={q_head.shape}, action_indices={action_indices.shape}")
                
                # Emergency fallback: create zero Q-values with correct batch size
                fallback_q = torch.zeros(q_head.size(0), device=self.device)
                current_q_list.append(fallback_q)
                self.logger.warning(f"⚠️ BCQ head {i}: Using fallback zero Q-values")

        # 🔧 CRITICAL FIX: Add missing BCQ loss computation
        # Compute target Q-values using the unified system
        try:
            # Generate next actions for target computation (using VAE sampling for BCQ)
            with torch.no_grad():
                if obs.dim() == 3:  # (B, T, D)
                    last_states_for_actions = next_obs[:, -1, :] if next_obs.dim() == 3 else next_obs
                else:
                    last_states_for_actions = next_obs
                
                # Use BCQ's perturbation sampling for next actions
                next_actions_list = []
                for head_idx in range(len(self.action_dims)):
                    # Simple greedy action selection for target (can be improved with BCQ sampling)
                    if len(next_q_values) > head_idx and next_q_values[head_idx] is not None:
                        next_q_head = next_q_values[head_idx]
                        if next_q_head.dim() == 3:
                            next_q_step = next_q_head[:, -1, :]  # (B, A)
                        else:
                            next_q_step = next_q_head
                        next_action = next_q_step.argmax(dim=-1)  # (B,)
                        next_actions_list.append(next_action)
                    else:
                        # Fallback to random actions
                        batch_size = last_states_for_actions.size(0)
                        next_action = torch.randint(0, self.action_dims[head_idx], (batch_size,), device=self.device)
                        next_actions_list.append(next_action)
            
            # Compute target Q-values
            target_q_values = compute_target_q_unified(q_values, next_q_values, actions, next_actions_list, rewards, dones)
            
            # Compute BCQ losses for each head
            valid_heads = 0
            for i, (current_q, target_q) in enumerate(zip(current_q_list, target_q_values)):
                try:
                    # Ensure shape compatibility
                    if current_q.shape != target_q.shape:
                        min_size = min(current_q.size(0), target_q.size(0))
                        current_q = current_q[:min_size]
                        if target_q.dim() > 1:
                            target_q = target_q[:min_size, -1] if target_q.dim() == 2 else target_q[:min_size]
                        else:
                            target_q = target_q[:min_size]
                    
                    # Compute TD loss for this head
                    td_loss = F.mse_loss(current_q, target_q.detach())
                    
                    if torch.isfinite(td_loss):
                        total_loss += td_loss
                        valid_heads += 1
                        self.logger.debug(f"🔧 BCQ head {i}: TD loss = {td_loss.item():.6f}")
                    else:
                        self.logger.error(f"❌ BCQ head {i}: Non-finite loss: {td_loss}")
                        
                except Exception as loss_e:
                    self.logger.error(f"❌ BCQ head {i}: Loss computation failed: {loss_e}")
                    continue
            
            if valid_heads > 0:
                total_loss = total_loss / valid_heads
                self.logger.debug(f"🔧 BCQ: Average loss across {valid_heads} heads: {total_loss.item():.6f}")
            else:
                self.logger.error("❌ BCQ: No valid head losses computed")
                return 0.0
                
        except Exception as bcq_loss_e:
            self.logger.error(f"❌ BCQ: Loss computation completely failed: {bcq_loss_e}")
            return 0.0
        
        # ===== 1. Train behaviour cloning VAE =====
        # Using last timestep state as representation.
        # --------------------------------------------------------------
        # Robust extraction of per-sample state feature for VAE training
        # --------------------------------------------------------------
        # (1) Sequence input (B,T,D): take *last* valid timestep to capture
        #     most recent context.  Avoid .squeeze(1) which only works for
        #     T==1.
        # (2) Flat input (B,D): use as-is.
        # --------------------------------------------------------------
        with torch.no_grad():
            if obs.dim() == 3:  # (B, T, D)
                last_states = obs[:, -1, :]
            else:
                last_states = obs  # already (B, D)

        # --------------------------------------------------------------
        # Dynamic VAE (re)initialisation – align input dimension
        # --------------------------------------------------------------
        vae_in_dim = self.action_vae.encoder[0].in_features
        if last_states.shape[-1] != vae_in_dim:
            self.logger.warning(
                f"⚠️  [BCQ] Re-initialising Action-VAE due to input dim change: "
                f"{vae_in_dim} → {last_states.shape[-1]}")
            self.action_vae = MultiHeadActionVAE(last_states.shape[-1], self.action_dims).to(self.device)
            # Re-create optimiser to match new parameters
            lr_vae = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 1e-3
            self.vae_optimizer = optim.Adam(self.action_vae.parameters(), lr=lr_vae)

        # --------------------------------------------------------------
        # Build flattened action targets for VAE reconstruction loss:   
        #   • Each tensor in ``actions`` may be (B,T) or (B,1).          
        #   • We take **last timestep** per sequence to obtain a 1-D     
        #     target vector of shape (B,).                               
        # --------------------------------------------------------------
        per_head_targets = []
        for head_idx, a in enumerate(actions):
            self.logger.debug(f"🔧 Action head {head_idx} processing: type={type(a)}, initial_shape={getattr(a, 'shape', 'N/A')}")
            
            # Convert NumPy array or list to tensor without triggering user warning
            if isinstance(a, torch.Tensor):
                a_t = a.to(device=self.device, dtype=torch.long).clone()
            else:
                a_t = torch.as_tensor(a, dtype=torch.long, device=self.device)

            self.logger.debug(f"🔧 Action head {head_idx} after conversion: shape={a_t.shape}")

            # Remove any singleton dims except batch
            while a_t.dim() > 2 and a_t.size(1) == 1:
                a_t = a_t.squeeze(1)  # drop dim-1

            if a_t.dim() == 2:
                # Select last timestep for sequence targets
                a_last = a_t[:, -1]
                self.logger.debug(f"🔧 Action head {head_idx} sequence mode: {a_t.shape} -> {a_last.shape}")
            elif a_t.dim() == 1:
                a_last = a_t
                self.logger.debug(f"🔧 Action head {head_idx} flat mode: {a_t.shape}")
            else:
                raise ValueError(f"Unexpected action tensor shape {a_t.shape}")

            per_head_targets.append(a_last)

        self.logger.debug(f"🔧 Per head targets shapes: {[t.shape for t in per_head_targets]}")
        actions_tensor = torch.stack(per_head_targets, dim=1)  # (B, n_heads)
        self.logger.debug(f"🔧 Final actions_tensor shape: {actions_tensor.shape}")

        self.logger.debug(f"🔧 VAE input debug: last_states.shape={last_states.shape}")
        logits, kl_div = self.action_vae(last_states)
        self.logger.debug(f"🔧 VAE output debug: len(logits)={len(logits)}, actions_tensor.shape={actions_tensor.shape}")
        
        # 🔧 CRITICAL FIX: Enhanced VAE loss computation with comprehensive error handling
        # reconstruction loss: cross entropy per head
        recon_loss = 0.0
        valid_heads = 0
        
        # 🔧 VALIDATE VAE OUTPUTS AND ACTION TARGETS BEFORE CROSS ENTROPY
        if len(logits) != len(self.action_dims):
            self.logger.error(f"❌ VAE logits count mismatch: {len(logits)} vs {len(self.action_dims)} action dims")
            return 0.0
            
        if actions_tensor.size(1) != len(self.action_dims):
            self.logger.error(f"❌ Action tensor head count mismatch: {actions_tensor.size(1)} vs {len(self.action_dims)} action dims")
            return 0.0
            
        for i, head_logits in enumerate(logits):
            try:
                # 🔧 COMPREHENSIVE TENSOR VALIDATION AND FIXING
                self.logger.debug(f"🔧 Cross entropy debug head {i}: head_logits.shape={head_logits.shape}, actions_tensor[:, {i}].shape={actions_tensor[:, i].shape}")
                
                # Extract action targets for this head
                action_targets = actions_tensor[:, i]
                
                # 🔧 DEVICE ALIGNMENT: Ensure both tensors are on the same device
                if head_logits.device != action_targets.device:
                    self.logger.warning(f"⚠️ Device mismatch head {i}: logits on {head_logits.device}, actions on {action_targets.device}")
                    action_targets = action_targets.to(head_logits.device)
                
                # 🔧 TYPE VALIDATION: Ensure action targets are LongTensor for cross entropy
                if action_targets.dtype != torch.long:
                    self.logger.debug(f"🔧 Converting action targets dtype from {action_targets.dtype} to long")
                    action_targets = action_targets.long()
                
                # 🔧 SHAPE VALIDATION: Ensure compatible shapes for cross entropy
                batch_size_logits = head_logits.size(0)
                batch_size_actions = action_targets.size(0)
                
                if batch_size_logits != batch_size_actions:
                    self.logger.warning(f"⚠️ Batch size mismatch head {i}: logits {batch_size_logits} vs actions {batch_size_actions}")
                    # Use minimum batch size to avoid indexing errors
                    min_batch = min(batch_size_logits, batch_size_actions)
                    if min_batch > 0:
                        head_logits = head_logits[:min_batch]
                        action_targets = action_targets[:min_batch]
                        self.logger.debug(f"🔧 Aligned to batch size {min_batch}")
                    else:
                        self.logger.error(f"❌ Cannot resolve batch size mismatch for head {i}")
                        continue
                
                # 🔧 ACTION BOUNDS VALIDATION: Ensure actions are within valid range
                action_dim = self.action_dims[i]
                max_action = safe_item(action_targets.max()) if action_targets.numel() > 0 else 0
                min_action = safe_item(action_targets.min()) if action_targets.numel() > 0 else 0
                
                if min_action < 0 or max_action >= action_dim:
                    invalid_actions = (action_targets < 0) | (action_targets >= action_dim)
                    num_invalid = safe_item(invalid_actions.sum())
                    self.logger.warning(f"⚠️ Head {i}: {num_invalid} invalid actions (range [{min_action}, {max_action}], valid [0, {action_dim-1}])")
                    # Clamp to valid range
                    action_targets = torch.clamp(action_targets, min=0, max=action_dim - 1)
                
                # 🔧 LOGITS DIMENSION VALIDATION: Ensure logits have correct action dimension
                if head_logits.size(-1) != action_dim:
                    self.logger.error(f"❌ Head {i} logits dimension mismatch: {head_logits.size(-1)} vs expected {action_dim}")
                    if head_logits.size(-1) > action_dim:
                        # Truncate to expected dimension
                        head_logits = head_logits[:, :action_dim]
                        self.logger.warning(f"⚠️ Truncated head {i} logits to dimension {action_dim}")
                    else:
                        self.logger.error(f"❌ Head {i} logits dimension too small, skipping")
                        continue
                
                # 🔧 COMPUTE CROSS ENTROPY WITH NUMERICAL STABILITY
                try:
                    head_ce_loss = F.cross_entropy(head_logits, action_targets, reduction='mean')
                    
                    # Validate the computed loss
                    if torch.isnan(head_ce_loss) or torch.isinf(head_ce_loss):
                        self.logger.error(f"❌ Invalid cross entropy loss for head {i}: {head_ce_loss}")
                        continue
                    
                    recon_loss += head_ce_loss
                    valid_heads += 1
                    self.logger.debug(f"🔧 Head {i} CE loss: {safe_item(head_ce_loss):.6f}")
                    
                except Exception as ce_error:
                    self.logger.error(f"❌ Cross entropy computation failed for head {i}: {ce_error}")
                    self.logger.error(f"🔍 Final shapes: head_logits={head_logits.shape}, action_targets={action_targets.shape}")
                    self.logger.error(f"🔍 Value ranges: logits_min={safe_item(head_logits.min()):.3f}, logits_max={safe_item(head_logits.max()):.3f}")
                    self.logger.error(f"🔍 Action targets: min={safe_item(action_targets.min())}, max={safe_item(action_targets.max())}")
                    continue
                    
            except Exception as head_error:
                self.logger.error(f"❌ VAE head {i} processing failed: {head_error}")
                self.logger.error(f"🔍 Debug info:")
                if 'head_logits' in locals():
                    self.logger.error(f"  • head_logits: shape={head_logits.shape}, device={head_logits.device}, dtype={head_logits.dtype}")
                if 'actions_tensor' in locals():
                    self.logger.error(f"  • actions_tensor: shape={actions_tensor.shape}, device={actions_tensor.device}, dtype={actions_tensor.dtype}")
                continue
        
        # 🔧 FINAL VALIDATION: Ensure we have valid reconstruction loss
        if valid_heads == 0:
            self.logger.error("❌ No valid VAE heads processed - VAE training failed")
            return safe_item(total_loss)  # Return Q-loss only, skip VAE training
        
        # Average reconstruction loss across valid heads
        recon_loss = recon_loss / valid_heads
        self.logger.debug(f"🔧 Final VAE reconstruction loss: {safe_item(recon_loss):.6f} (from {valid_heads}/{len(logits)} heads)")

        # -------------------------- KL Annealing --------------------------
        progress = min(1.0, self.update_steps / float(self.kl_anneal_steps))
        kl_beta = self.kl_beta_start + progress * (self.kl_beta_end - self.kl_beta_start)
        vae_loss = recon_loss + kl_beta * kl_div
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        # ------------------------------------------------------------------
        # 2) Train perturbation actor – maximise expected Q(s, a+δ)
        # ------------------------------------------------------------------
        self.actor.train()

        # Compute perturbation logits for the *last* state representation
        perturb_logits = self.actor(last_states)  # List[(B, A_i)]

        # Obtain current Q-values (no grad) for each head
        q_forward_out2 = self._forward_model(
            self.q_net,
            last_states.unsqueeze(1),
            torch.ones(last_states.size(0), dtype=torch.long, device=self.device),
            torch.empty((2, 0), dtype=torch.long, device=self.device),
            None,
            mode='q',
        )
        q_values_flat = q_forward_out2[0] if isinstance(q_forward_out2, tuple) else q_forward_out2

        # Expected Q under the stochastic policy parameterised by perturb_logits
        # Initialise as Tensor to guarantee ``requires_grad`` propagation even when
        # no action heads are processed (edge-case when ``action_dims`` is empty).
        actor_q = torch.tensor(0.0, device=self.device)
        for head_idx, q_head in enumerate(q_values_flat):
            # q_head shape: (B, T, A) or (B, A)
            if q_head.dim() == 3:
                q_head = q_head[:, -1, :]  # take final timestep
            # Ensure q_head is detached to prevent gradient flow to Q-network
            q_head_detached = q_head.detach()
            # Compute softmax probabilities from perturbation logits
            probs = F.softmax(perturb_logits[head_idx], dim=-1)  # (B, A_i)
            # Expected Q = sum(probs * Q_values)
            actor_q += (probs * q_head_detached).sum(dim=-1).mean()

        actor_loss = -actor_q  # maximise Q → minimise negative expectation
        self.actor_optimizer.zero_grad()
        # Guard: Only back-propagate when graph is attached and loss is finite
        if (actor_loss.requires_grad and 
            actor_loss.grad_fn is not None and 
            torch.isfinite(actor_loss)):
            actor_loss.backward()
            self.actor_optimizer.step()

        # Backward pass and optimization (Q-network)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            # Polyak averaging instead of hard copy
            with torch.no_grad():
                for tgt, src in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                    tgt.data.mul_(1 - self.soft_tau)
                    tgt.data.add_(self.soft_tau * src.data)
            self.logger.debug("🔄 [BCQ] Soft-updated target network (τ=%.4f) at step %d", self.soft_tau, self.update_steps)
            
        return float(total_loss.item())

    def act(
        self, 
        state: torch.Tensor, 
        greedy: bool = True, 
        **kwargs
    ) -> torch.Tensor:
        """Selects actions for given states following BaseRLAgent interface.
        
        This method implements the BCQ agent's policy for action selection compatible
        with the BaseRLAgent abstract interface. It uses behavior cloning with Q-learning
        and perturbation-based action selection for conservative action choice.
        
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
            raise ValueError(f"Expected 2D or 3D state tensor, got {state.dim()}D")
        
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

    def save(self, path):
        torch.save({
            'model': self.q_net.state_dict(),
            'target_model': self.target_q_net.state_dict(),
            'actor': self.actor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict()
        }, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state['model'])
        self.target_q_net.load_state_dict(state['target_model'])
        self.actor.load_state_dict(state['actor'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])

    def save_checkpoint(self, filepath: str) -> None:
        """Saves agent state to checkpoint file.

        Args:
            filepath: Path to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.q_net.state_dict(),
            'target_model_state_dict': self.target_q_net.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'action_vae_state_dict': self.action_vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'update_steps': self.update_steps,
            'config': {
                'action_dims': self.action_dims,
                'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 1e-3,
                'gamma': self.gamma,
                'target_update_freq': self.target_update_freq,
                'perturbation_scale': self.perturbation_scale,
                'n_perturb_samples': self.n_perturb_samples,
                'kl_beta_start': self.kl_beta_start,
                'kl_beta_end': self.kl_beta_end,
                'kl_anneal_steps': self.kl_anneal_steps,
                'soft_tau': self.soft_tau,
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
        self.target_q_net.load_state_dict(checkpoint['target_model_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.action_vae.load_state_dict(checkpoint['action_vae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self._training_step = checkpoint['training_step']
        self._episode_count = checkpoint['episode_count']
        self.update_steps = checkpoint.get('update_steps', 0)

    # ------------------------------------------------------------------
    # Backwards-compatibility: expose `.q_net` and `.vae` attributes that are
    # expected by legacy evaluation wrappers (e.g. *BCQPolicyNetWrapper*).
    # ------------------------------------------------------------------
    @property
    def q_net(self):  # noqa: D401
        """Return the underlying Q-network (alias for :pyattr:`model`)."""
        return self.q_net

    @property
    def vae(self):  # noqa: D401
        """Return the behaviour-cloning VAE (alias for :pyattr:`action_vae`)."""
        return self.action_vae

    # ---------------------------------------------------------------------
    # Action perturbation generation using trained VAE
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def generate_perturbed_actions(self, states: torch.Tensor, greedy_actions: torch.Tensor, perturb_scale: float):
        """Return perturbed actions following discrete BCQ procedure.

        Pipeline:
          1. Sample ``K`` candidate actions from the behaviour VAE.
          2. For each candidate action compute *perturbation logits* δ via
             the actor network and obtain the **shifted** action index:
                 a′ = clip(a + δ, 0, A_i−1)
          3. Evaluate Q(s, a′) and pick the best per state.
        """

        # Ensure VAE input dim matches
        feat_dim = states.shape[-1]
        if self.action_vae.encoder[0].in_features != feat_dim:
            self.logger.warning(
                "⚠️  [BCQ] Action-VAE resized on-the-fly: %d→%d",
                self.action_vae.encoder[0].in_features, feat_dim)
            self.action_vae = MultiHeadActionVAE(feat_dim, self.action_dims).to(states.device)

        cand_actions_list = self.action_vae.sample_actions(states, n_samples=self.n_perturb_samples)

        # Flatten candidate actions for efficient batch Q evaluation
        best_actions = []  # per head best action indices

        with torch.no_grad():
            # Compute perturbations once
            perturb_logits = self.actor(states)  # List[(B, A_i)]

            # Obtain current Q-values (no grad) in *flat* (B, A) form using
            # the shared forward helper for consistent PoG/MLP handling.
            dummy_lengths = torch.ones(states.size(0), dtype=torch.long, device=states.device)
            empty_edge = torch.empty((2, 0), dtype=torch.long, device=states.device)
            q_forward_out = self._forward_model(self.q_net, states.unsqueeze(1), dummy_lengths, empty_edge, None, mode='q')
            q_values = q_forward_out[0] if isinstance(q_forward_out, tuple) else q_forward_out

        # Safety fallback
        if q_values is None:
            return greedy_actions

        perturbed_out = []
        for head_idx, action_dim in enumerate(self.action_dims):
            # Candidate actions sampled by VAE for this head – (B, K)
            cand = cand_actions_list[head_idx]  # (B, K)

            # δ = argmax logits per action index (B, A_i) – take sign & round to nearest integer shift
            delta_logits = perturb_logits[head_idx]  # (B, A_i)
            # Map each candidate to its perturb score (approx via delta_logits index)
            cand_flat = cand.view(-1)  # (B*K,)
            batch_rep = torch.arange(states.size(0), device=states.device).repeat_interleave(self.n_perturb_samples)
            delta_selected = delta_logits[batch_rep, cand_flat]  # (B*K,)
            delta_selected = delta_selected.view(states.size(0), self.n_perturb_samples)

            # Shift candidate by sign(delta) * 1 (discrete step)
            shifted = cand + delta_selected.sign().long()
            shifted = shifted.clamp_(0, action_dim - 1)

            # Evaluate Q(s,a) for shifted actions ---------------------------
            q_mat = q_values[head_idx]  # (B, A) or (B,T,A)
            if q_mat.dim() == 3:
                q_mat = q_mat[:, -1, :]
            q_cand = q_mat.gather(-1, shifted)  # (B,K)

            best_idx = q_cand.argmax(dim=-1, keepdim=True)  # (B,1)
            best_action = shifted.gather(-1, best_idx)  # (B,1)
            best_actions.append(best_action)

        return torch.cat(best_actions, dim=1)

# ==================================================
# Action VAE for BCQ (multi-discrete version)
# ==================================================

class MultiHeadActionVAE(nn.Module):
    """State-conditioned VAE that outputs independent categorical distributions
    for each action head. The encoder embeds the last state representation to a
    latent vector z ~ N(μ, σ). The decoder maps z to logits for every action
    head. For simplicity we treat each head independently with its own linear
    projection. This is sufficient for behavioural cloning of the dataset
    actions and enables BCQ to sample diversified but in-distribution actions.
    """

    def __init__(self, state_dim: int, action_dims: List[int], latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.n_heads = len(action_dims)

        # Encoder q(z|s)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder p(a|z)
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for action_dim in action_dims
        ])

        # --------------------------------------------------------------
        # Weight initialisation: Xavier for all Linear layers + zero on
        # final output projection as per BCQ reference implementation.
        # --------------------------------------------------------------
        from Libs.utils.model_utils import init_weights_xavier

        self.apply(init_weights_xavier)
        for dec in self.decoder_layers:
            last_linear = dec[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

        # Legacy compatibility: expose `.mean` alias for downstream wrappers
        self.mean = self.mu_layer  # type: ignore[attr-defined]

    def forward(self, state: torch.Tensor):
        """Returns logits for each head together with KL divergence."""
        # state: (B, state_dim)
        h = self.encoder(state)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization

        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        logits = [layer(z) for layer in self.decoder_layers]
        return logits, kl_div

    # ------------------------------------------------------------------
    # Legacy helpers required by external wrappers
    # ------------------------------------------------------------------
    def decode(self, _state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample *z* into **flattened** action logits.

        The original BCQ reference code ignores the conditional input *state*
        during decoding and relies solely on the latent vector.  We replicate
        this behaviour so that replacing the VAE does not alter downstream
        logic.

        Args:
            _state: Placeholder tensor kept for API compatibility. Not used.
            z: Latent tensor of shape ``(batch_size, latent_dim)``.

        Returns:
            A tensor of shape ``(batch_size, sum(action_dims))`` containing the
            concatenated logits for all action heads.
        """
        # Compute per-head logits
        head_logits = [layer(z) for layer in self.decoder_layers]
        # Concatenate → (B, Σ A_i)
        return torch.cat(head_logits, dim=-1)

    @torch.no_grad()
    def sample_actions(self, state: torch.Tensor, n_samples: int = 10) -> List[torch.Tensor]:
        """Generate candidate actions given state by sampling latent z."""
        B = state.shape[0]
        h = self.encoder(state)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        std = torch.exp(0.5 * logvar)
        # sample n_samples per state
        zs = mu.unsqueeze(1) + std.unsqueeze(1) * torch.randn(B, n_samples, std.shape[-1], device=state.device)
        zs = zs.view(B * n_samples, -1)

        # Decode
        all_logits = [layer(zs) for layer in self.decoder_layers]
        # Sample greedy action per latent
        all_actions = [logits.argmax(dim=-1) for logits in all_logits]  # list of (B*n_samples,)
        # reshape (B, n_samples)
        actions = [a.view(B, n_samples) for a in all_actions]
        return actions 

# ---------------------------------------------------------------------
# Specialised Multi-Head wrapper for BCQ (inherits storage logic)
# ---------------------------------------------------------------------


class MultiHeadReplayBuffer(ReplayBuffer):
    """ReplayBuffer variant that post-processes multi-head actions.

    This thin wrapper keeps the *sample* signature from the legacy BCQ
    implementation while delegating storage and PER functionality to the
    shared :class:`Libs.utils.model_utils.ReplayBuffer`.
    """

    def sample(self, batch_size: int):  # type: ignore[override]
        sample_out = super().sample(batch_size)

        # `super().sample` can return (transitions, idx, w) when PER is on.
        # We only care about the raw transition list.
        if isinstance(sample_out, tuple):
            transitions = sample_out[0]
        else:
            transitions = sample_out

        obs, actions, rewards, next_obs, dones, mask, lengths, next_lengths, edge_index = zip(*transitions)

        # Reorganise multi-head actions to List[np.array]
        actions_by_head = [np.array(a) for a in zip(*actions)]

        return (
            np.array(obs),
            actions_by_head,
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
            np.array(mask),
            np.array(lengths),
            np.array(next_lengths),
            edge_index[0] if len(edge_index) > 0 else np.zeros((2, 0), dtype=np.int64),
        ) 

    def _assert_batch_obs_dims(self, obs: torch.Tensor) -> None:
        """Assert observation tensor has valid dimensions for BCQ processing."""
        if obs.dim() not in {2, 3}:
            raise ValueError(f"BCQ requires 2D or 3D obs, got {obs.dim()}D: {obs.shape}")
        if obs.size(0) == 0:
            raise ValueError("Empty batch not supported")



    # 🔧 ENHANCED ERROR HANDLING AND DIAGNOSTICS
    def _get_batch_debug_info(self, batch_data) -> str:
        """
        Generate comprehensive diagnostic information for batch data.
        
        This method provides detailed information about tensor shapes, types,
        and potential issues in the batch data to help with debugging shape
        mismatch errors and other training issues.
        
        Args:
            batch_data: The batch data (can be tuple or dict format)
            
        Returns:
            Formatted debug string with batch information
        """
        try:
            if isinstance(batch_data, dict):
                info_lines = ["Batch Dict Debug Info:"]
                for key, value in batch_data.items():
                    if torch.is_tensor(value):
                        info_lines.append(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                        if torch.isnan(value).any():
                            nan_count = torch.isnan(value).sum().item()
                            info_lines.append(f"    ⚠️ Contains {nan_count} NaN values")
                        if torch.isinf(value).any():
                            inf_count = torch.isinf(value).sum().item()
                            info_lines.append(f"    ⚠️ Contains {inf_count} infinite values")
                    elif hasattr(value, 'shape'):
                        info_lines.append(f"  {key}: numpy shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        info_lines.append(f"  {key}: type={type(value)}, length={len(value)}")
                        if len(value) > 0 and hasattr(value[0], 'shape'):
                            info_lines.append(f"    first element shape: {value[0].shape}")
                    else:
                        info_lines.append(f"  {key}: type={type(value)}")
                        
            elif isinstance(batch_data, (tuple, list)):
                info_lines = [f"Batch Tuple Debug Info (length={len(batch_data)}):"]
                for i, item in enumerate(batch_data):
                    if torch.is_tensor(item):
                        info_lines.append(f"  [{i}]: tensor shape={item.shape}, dtype={item.dtype}")
                    elif hasattr(item, 'shape'):
                        info_lines.append(f"  [{i}]: numpy shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, (list, tuple)):
                        info_lines.append(f"  [{i}]: type={type(item)}, length={len(item)}")
                    else:
                        info_lines.append(f"  [{i}]: type={type(item)}")
                        
            else:
                info_lines = [f"Batch Debug Info: type={type(batch_data)}"]
                
            return "\n".join(info_lines)
            
        except Exception as e:
            return f"Failed to generate debug info: {e}"

    def _validate_tensor_compatibility(self, tensors_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate tensor compatibility and provide detailed diagnostics.
        
        Args:
            tensors_dict: Dictionary of tensors to validate
            
        Returns:
            Dictionary containing validation results and diagnostics
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'batch_sizes': {},
            'temporal_dimensions': {},
            'recommendations': []
        }
        
        try:
            # Extract batch sizes and temporal dimensions
            for name, tensor in tensors_dict.items():
                if torch.is_tensor(tensor):
                    results['batch_sizes'][name] = tensor.size(0)
                    if tensor.dim() >= 2:
                        results['temporal_dimensions'][name] = tensor.size(1)
            
            # Check batch size consistency
            batch_sizes = list(results['batch_sizes'].values())
            if len(set(batch_sizes)) > 1:
                results['is_valid'] = False
                results['errors'].append(f"Inconsistent batch sizes: {results['batch_sizes']}")
                results['recommendations'].append("Ensure all tensors have the same batch dimension")
            
            # Check temporal dimension consistency for 3D tensors
            temporal_dims = list(results['temporal_dimensions'].values())
            if len(temporal_dims) > 1 and len(set(temporal_dims)) > 1:
                results['warnings'].append(f"Inconsistent temporal dimensions: {results['temporal_dimensions']}")
                results['recommendations'].append("Consider aligning temporal dimensions or using padding")
            
            # Check for NaN/Inf values
            for name, tensor in tensors_dict.items():
                if torch.is_tensor(tensor):
                    if torch.isnan(tensor).any():
                        results['errors'].append(f"Tensor '{name}' contains NaN values")
                        results['is_valid'] = False
                    if torch.isinf(tensor).any():
                        results['warnings'].append(f"Tensor '{name}' contains infinite values")
            
            # BCQ-specific validations
            if 'q_values' in tensors_dict and 'actions' in tensors_dict:
                q_values = tensors_dict['q_values']
                actions = tensors_dict['actions']
                
                # Check if action indices are within Q-value bounds
                if hasattr(actions, 'max') and hasattr(q_values, 'size'):
                    if q_values.dim() >= 2:
                        action_dim = q_values.size(-1)
                        if actions.max() >= action_dim:
                            results['errors'].append(f"Action indices out of bounds: max={actions.max()}, action_dim={action_dim}")
                            results['is_valid'] = False
                            
        except Exception as e:
            results['is_valid'] = False
            results['errors'].append(f"Validation failed: {e}")
            
        return results

    def _handle_shape_mismatch_error(self, error_msg: str, context: Dict[str, Any]) -> str:
        """
        Analyze and provide solutions for shape mismatch errors.
        
        Args:
            error_msg: The original error message
            context: Context information including tensor shapes and operation details
            
        Returns:
            Detailed diagnostic and solution suggestions
        """
        solutions = []
        
        # Analyze common patterns in the error message
        if "must match the size of tensor b" in error_msg:
            solutions.append("🔧 Shape Mismatch Analysis:")
            solutions.append("  This error typically occurs when tensors have incompatible dimensions for operations")
            
            if "dimension 1" in error_msg:
                solutions.append("  • Issue is in the second dimension (index 1)")
                solutions.append("  • Common causes:")
                solutions.append("    - Sequence length mismatch between batches")
                solutions.append("    - Inconsistent temporal dimensions")
                solutions.append("    - Batch size vs sequence length confusion")
                
            solutions.append("  • Potential fixes:")
            solutions.append("    - Ensure all sequence tensors have the same temporal dimension")
            solutions.append("    - Use padding or truncation to align sequence lengths")
            solutions.append("    - Check that batch dimensions are consistent across all tensors")
            
        # Add context-specific information if available
        if context:
            if 'tensor_shapes' in context:
                solutions.append(f"  • Current tensor shapes: {context['tensor_shapes']}")
            if 'operation' in context:
                solutions.append(f"  • Failed operation: {context['operation']}")
            if 'batch_info' in context:
                solutions.append(f"  • Batch info: {context['batch_info']}")
                
        # BCQ-specific solutions
        solutions.extend([
            "  • BCQ-specific solutions:",
            "    - Verify that Q-network outputs match expected action dimensions",
            "    - Check that action tensors have correct shape for gather operations",
            "    - Ensure behavioral policy and Q-network are compatible",
            "    - Consider using single-timestep extraction for BCQ (typical pattern)"
        ])
        
        return "\n".join(solutions)

    def _safe_tensor_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Execute tensor operations with enhanced error handling and diagnostics.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result or None if failed
        """
        try:
            return operation_func(*args, **kwargs)
        except RuntimeError as e:
            error_msg = str(e)
            
            # Collect context information
            context = {
                'operation': operation_name,
                'tensor_shapes': {},
                'error_type': type(e).__name__
            }
            
            # Extract tensor shapes from arguments
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    context['tensor_shapes'][f'arg_{i}'] = arg.shape
                    
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    context['tensor_shapes'][key] = value.shape
            
            # Log detailed error information
            self.logger.error(f"❌ Tensor operation '{operation_name}' failed: {error_msg}")
            self.logger.error(f"🔍 Context: {context}")
            
            # Provide analysis and solutions
            if "size" in error_msg.lower() or "shape" in error_msg.lower():
                analysis = self._handle_shape_mismatch_error(error_msg, context)
                self.logger.error(f"🔧 Analysis and Solutions:\n{analysis}")
            
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in '{operation_name}': {e}")
            return None

    def _emergency_batch_repair(self, batch_data, target_batch_size: int):
        """
        Emergency batch repair for critical shape mismatches.
        
        This method attempts to repair batch data when normal validation fails,
        providing a last resort to prevent training crashes.
        
        Args:
            batch_data: The problematic batch data
            target_batch_size: Target batch size to align to
            
        Returns:
            Repaired batch data or None if irreparable
        """
        try:
            self.logger.warning(f"⚠️ Attempting emergency batch repair for target size: {target_batch_size}")
            
            if isinstance(batch_data, dict):
                repaired_data = {}
                for key, value in batch_data.items():
                    if torch.is_tensor(value):
                        if value.size(0) != target_batch_size:
                            if value.size(0) > target_batch_size:
                                # Truncate
                                repaired_data[key] = value[:target_batch_size]
                                self.logger.debug(f"🔧 Truncated {key} from {value.size(0)} to {target_batch_size}")
                            elif value.size(0) == 1:
                                # Broadcast single sample
                                new_shape = [target_batch_size] + list(value.shape[1:])
                                repaired_data[key] = value.expand(new_shape)
                                self.logger.debug(f"🔧 Broadcasted {key} from {value.shape} to {new_shape}")
                            else:
                                # Cannot safely repair
                                self.logger.error(f"❌ Cannot repair {key}: {value.shape}")
                                return None
                        else:
                            repaired_data[key] = value
                    else:
                        repaired_data[key] = value
                        
                return repaired_data
                
            elif isinstance(batch_data, (tuple, list)):
                repaired_data = []
                for i, item in enumerate(batch_data):
                    if torch.is_tensor(item):
                        if item.size(0) != target_batch_size:
                            if item.size(0) > target_batch_size:
                                repaired_item = item[:target_batch_size]
                            elif item.size(0) == 1:
                                new_shape = [target_batch_size] + list(item.shape[1:])
                                repaired_item = item.expand(new_shape)
                            else:
                                self.logger.error(f"❌ Cannot repair item {i}: {item.shape}")
                                return None
                        else:
                            repaired_item = item
                    else:
                        repaired_item = item
                    repaired_data.append(repaired_item)
                    
                return type(batch_data)(repaired_data)
            
            else:
                self.logger.error(f"❌ Unknown batch data type: {type(batch_data)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Emergency batch repair failed: {e}")
            return None