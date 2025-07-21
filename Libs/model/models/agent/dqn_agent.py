"""
Deep Q-Network (DQN) agent implementation for multi-discrete action spaces.

This module provides a DQN agent that supports multi-discrete action spaces
commonly found in medical decision making scenarios. The agent is compatible
with both traditional neural networks and PoG-enhanced models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Optional, Any, Dict
from .base_agent import BaseRLAgent
from Libs.utils.model_utils import safe_float, apply_gradient_clipping, safe_item
from Libs.model.models.agent._compat import ForwardCompatMixin

# Unified replay buffer to avoid duplication
from Libs.utils.model_utils import ReplayBuffer as _SharedReplayBuffer

# Alias old name to shared implementation for backward compatibility
ReplayBuffer = _SharedReplayBuffer  # noqa: N816 – keep CamelCase for API stability

# 新增统一张量转换工具
from Libs.utils.model_utils import as_tensor


class DQNAgent(ForwardCompatMixin, BaseRLAgent):
    """Deep Q-Network agent implementation with enhanced multi-discrete action support.
    """
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
        device: str = 'cpu',
        reward_centering: bool = False,
        target_update_freq: int = 100,
        cql_alpha: float = 0.0,
        max_grad_norm: float = 1.0,
        offline_dataset: Optional[Tuple[np.ndarray, ...]] = None,
        polyak_tau: float = 0.005,
        double_q: bool = True,
        dueling: bool = True,
        noisy: bool = False,
        optimizer: str = 'adam',
    ) -> None:
        """
        Initialize the DQN agent.
        
        Args:
            model: Q-network architecture for action-value estimation.
            action_dims: List of action space sizes for each action head.
            lr: Learning rate for the optimizer.
            gamma: Discount factor for future rewards (0 < gamma <= 1).
            buffer_size: Maximum size of the replay buffer.
            batch_size: Batch size for training updates.
            device: Computing device ('cpu' or 'cuda').
            reward_centering: Whether to apply reward centering for stability.
            target_update_freq: Frequency of target network updates.
            cql_alpha: Conservative Q-learning regularisation coefficient (0.0 ⟹ disabled)
            max_grad_norm: Gradient clipping to improve numerical stability
            polyak_tau: Polyak averaging coefficient for target network updates
            
        Raises:
            ValueError: If action_dims is empty or contains non-positive values.
        """
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError("action_dims must be non-empty with positive values")
        
        # Safely convert parameters to appropriate types
        lr = safe_float(lr, 1e-3)
        gamma = safe_float(gamma, 0.99)
        polyak_tau = safe_float(polyak_tau, 0.005)
        
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
            
        super().__init__(
            device=device,
            gamma=gamma,
            target_update_freq=target_update_freq,
            reward_centering=reward_centering,
            polyak_tau=float(polyak_tau),
        )
        
        # Initialize logger
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
            
        self.model = model.to(device)
        
        # Create target network with same architecture
        self.target_model = type(model)(*model.init_args).to(device)
        self.target_model.load_state_dict(model.state_dict())
        
        self.action_dims = action_dims
        # ------------------------------------------------------------------
        #  Optimizer selection – support both Adam (default) and RMSProp used
        #  by the original Nature-DQN implementation (Mnih et al., 2015).
        # ------------------------------------------------------------------
        optimizer = (optimizer or 'adam').lower()
        if optimizer == 'rmsprop':
            # Hyper-parameters follow the original paper: α=0.95, ε=0.01.
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.95, eps=0.01)
        else:
            # Fallback to Adam for modern variants.
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        
        # ------------------------------------------------------------------
        # Experience storage – online cyclic buffer vs fixed offline dataset
        # ------------------------------------------------------------------
        if offline_dataset is not None:
            from Libs.utils.model_utils import OfflineReplayBuffer

            self.replay_buffer = OfflineReplayBuffer(offline_dataset)  # type: ignore[arg-type]
            self.logger.debug(
                f"Loaded offline dataset with {len(self.replay_buffer)} transitions — disable .add() operations."
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.update_steps = 0

        # Conservative Q-learning regularisation coefficient (0.0 ⟹ disabled)
        self.cql_alpha = float(cql_alpha)

        # Gradient clipping threshold
        self.max_grad_norm = float(max(max_grad_norm, 2.0))

        self.polyak_tau = float(polyak_tau)
        self.dueling = bool(dueling)
        self.noisy = bool(noisy)
        
        self.logger.debug(f"DQN hyperparameters - lr: {lr}, gamma: {gamma}, "
                        f"target_update_freq: {target_update_freq}, "
                        f"max_grad_norm: {self.max_grad_norm}, "
                        f"reward_centering: {reward_centering}")

        # Try to enable dueling/noisy on model if supported
        if self.dueling and hasattr(model, "enable_dueling"):
            try:
                model.enable_dueling(True)
                self.logger.debug("🆕 Dueling architecture enabled for DQN model")
            except Exception:
                self.logger.warning("Dueling enable failed – model does not support it")

        if self.noisy:
            from Libs.model.layers.noisy_linear import NoisyLinear
            replaced = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    noisy_layer = NoisyLinear(module.in_features, module.out_features).to(self.device)
                    setattr(model, name, noisy_layer)
                    replaced += 1
            self.logger.debug("🆕 Replaced %d Linear layers with NoisyLinear", replaced)

        # Double-DQN flag
        self.double_q = bool(double_q)

        # ------------------------------------------------------------------
        # Re-use shared helpers from *ForwardCompatMixin* to remove duplication.
        # ------------------------------------------------------------------
        # NOTE: We no longer assign the unbound mix-in helpers as *instance* attributes
        # here because doing so prevents Python from automatically binding the
        # ``self`` parameter.  The methods are already available through normal
        # method resolution order (DQNAgent → ForwardCompatMixin), so no extra
        # aliasing is necessary.

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
        
        # Exploitation: greedy actions based on Q-values
        with torch.no_grad():
            model_output = self._forward_model(
                self.model, obs, lengths, edge_index, mask, mode='q'
            )
            
            # Extract Q-values; ignore any additional (deprecated) KL-related outputs
            q_values = model_output[0] if isinstance(model_output, tuple) else model_output
            
            # Select greedy actions for each head
            actions = []
            for q_head in q_values:
                greedy_actions = q_head.argmax(dim=-1)
                actions.append(greedy_actions)
                
        return actions

    def update(self, batch: Tuple[np.ndarray, ...], *, grad_scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
        """
        Update the Q-network using a batch of transitions.
        
        Performs a single training step using the DQN algorithm with target
        networks and experience replay.
        
        Args:
            batch: Tuple containing batch of transitions from replay buffer.
            
        Returns:
            Training loss value
        """
        # Ensure model is in training mode for backward pass
        self.model.train()
        self.target_model.eval()  # Target model should be in eval mode
        
        obs, actions, rewards, next_obs, dones, mask, lengths, next_lengths, edge_index = batch
        batch_size = obs.shape[0]
        
        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        
        # 改进mask处理 - 确保mask的形状和值正确
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        
        # 检查mask的有效性
        mask_sum_total = mask.sum().item()
        if mask_sum_total == 0:
            self.logger.warning("⚠️ [DQN] All masks are zero in batch, creating default mask")
            mask = torch.ones_like(mask)
        elif mask_sum_total < batch_size * 0.1:  # 如果有效mask太少
            self.logger.warning(f"⚠️ [DQN] Very few valid masks: {mask_sum_total}/{batch_size * mask.size(1)}")
        
        # 确保mask值在[0,1]范围内
        mask = torch.clamp(mask, min=0.0, max=1.0)
        
        if self.update_steps % 100 == 0:
            self.logger.debug(f"Mask stats - shape: {mask.shape}, sum: {mask.sum().item():.1f}, "
                            f"mean: {mask.mean().item():.3f}")
        
        # CRITICAL FIX: Handle both sequential and single-step formats for rewards and dones
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Check if we have sequential data (B, T) and need to extract single-step for DQN
        if rewards.dim() == 2 and obs.dim() == 3:
            # Sequential data: extract rewards/dones from last valid timestep for each trajectory
            self.logger.debug(f"🔧 [DQN] Converting sequential rewards/dones from shape {rewards.shape} to single-step")
            
            # Use mask to find last valid timestep for each trajectory
            if mask.dim() == 2:
                # Find last valid timestep (where mask is 1)
                last_valid = mask.sum(dim=1).long() - 1  # (B,)
                last_valid = torch.clamp(last_valid, min=0)  # Ensure non-negative
                
                # Extract rewards and dones at last valid timestep
                batch_indices = torch.arange(batch_size, device=self.device)
                rewards = rewards[batch_indices, last_valid]  # (B,)
                dones = dones[batch_indices, last_valid]      # (B,)
            else:
                # If no valid mask info, just take the last timestep
                rewards = rewards[:, -1]  # (B,)
                dones = dones[:, -1]      # (B,)
        
        # Now ensure rewards and dones have shape (B, 1) for compatibility
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # CRITICAL FIX: Ensure rewards and dones have correct batch size
        if rewards.size(0) != batch_size:
            self.logger.warning(f"⚠️ [DQN] Rewards batch size mismatch: {rewards.size(0)} vs {batch_size}")
            if rewards.size(0) < batch_size:
                # Pad with zeros
                pad_size = batch_size - rewards.size(0)
                rewards = torch.cat([rewards, torch.zeros(pad_size, 1, device=self.device)], dim=0)
            else:
                # Truncate
                rewards = rewards[:batch_size]
        
        if dones.size(0) != batch_size:
            self.logger.warning(f"⚠️ [DQN] Dones batch size mismatch: {dones.size(0)} vs {batch_size}")
            if dones.size(0) < batch_size:
                # Pad with False (not done)
                pad_size = batch_size - dones.size(0)
                dones = torch.cat([dones, torch.zeros(pad_size, 1, device=self.device)], dim=0)
            else:
                # Truncate
                dones = dones[:batch_size]
        
        lengths = torch.full((batch_size,), obs.shape[1], dtype=torch.long, device=self.device)
        next_lengths = torch.full((batch_size,), next_obs.shape[1], dtype=torch.long, device=self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        
        # ---------------- Sanity checks & preprocessing ----------------
        self._assert_batch_obs_dims(obs)
        
        # 临时禁用奖励中心化以修复训练问题
        # reward_centering可能导致奖励变得过小，从而使损失接近0
        # rewards = self._center_rewards(rewards, dim=0)
        
        # 添加奖励统计日志用于调试
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        if self.update_steps % 100 == 0:
            self.logger.debug(f"DQN Rewards - mean: {reward_mean:.6f}, std: {reward_std:.6f}")
        
        # Validate action tensors and ensure they're in the correct format
        actions_validated = []
        for i, a in enumerate(actions):
            # 🔧 CRITICAL FIX: Enhanced action tensor conversion and validation
            action_tensor = torch.tensor(a, dtype=torch.long, device=self.device)
            
            # 🔧 COMPREHENSIVE SHAPE HANDLING: Handle different input shapes properly
            if action_tensor.dim() == 0:
                # Scalar action - expand to batch dimension
                action_tensor = action_tensor.unsqueeze(0)  # () -> (1,)
                self.logger.debug(f"🔧 DQN action[{i}] expanded scalar to batch: {action_tensor.shape}")
            elif action_tensor.dim() == 1:
                # (B,) -> already correct for single-step, but may need temporal expansion
                if obs.dim() == 3 and obs.size(1) > 1:
                    # Expand to sequence length for multi-step scenarios
                    action_tensor = action_tensor.unsqueeze(1).expand(-1, obs.size(1))  # (B,) -> (B, T)
                    self.logger.debug(f"🔧 DQN action[{i}] expanded to sequence: {action_tensor.shape}")
            elif action_tensor.dim() == 2:
                # (B, T) - already correct shape for sequences
                pass
            else:
                self.logger.error(f"❌ Unexpected action tensor dimension for head {i}: {action_tensor.shape}")
                continue
            
            # Clamp actions to valid range [0, action_dims[i]-1] to prevent gather errors
            original_max = safe_item(action_tensor.max()) if action_tensor.numel() > 0 else 0
            action_tensor = torch.clamp(action_tensor, min=0, max=self.action_dims[i] - 1)
            
            # Log warning if invalid actions were found and corrected
            if original_max >= self.action_dims[i]:
                invalid_count = safe_item((torch.tensor(a, dtype=torch.long, device=self.device) >= self.action_dims[i]).sum())
                self.logger.warning(f"⚠️  [DQN] Fixed {invalid_count} invalid actions in head {i}: "
                      f"range [0, {self.action_dims[i]-1}], max found: {original_max}")
            
            actions_validated.append(action_tensor)
        
        actions = actions_validated
        
        # Forward pass through main network with model type detection
        kwargs = {}
        if self._is_pog_model(self.model):
            kwargs.update({
                'rewards': rewards,
                'reward_centering': self.reward_centering
            })
        
        try:
            model_output = self._forward_model(
                self.model, obs, lengths, edge_index, mask, mode='q', **kwargs
            )
            q_values = model_output[0]  # Extract Q-values from model output
        except Exception as e:
            self.logger.error(f"❌ Model forward pass failed: {e}")
            self.logger.error(f"🔍 Input shapes: obs={obs.shape}, lengths={lengths.shape}, edge_index={edge_index.shape}")
            return 0.0
        
        # Target network forward pass
        with torch.no_grad():
            try:
                target_output = self._forward_model(
                    self.target_model, next_obs, next_lengths, edge_index, mask, mode='q', **kwargs
                )
                next_q_values = target_output[0]
            except Exception as e:
                self.logger.error(f"❌ Target model forward pass failed: {e}")
                self.logger.error(f"🔍 Next obs shapes: next_obs={next_obs.shape}, next_lengths={next_lengths.shape}")
                return 0.0

        # 🔧 CRITICAL ARCHITECTURAL FIX: Unified DQN Target Q Computation System
        # Implement the same tensor alignment protocol as BCQ to eliminate all dimension mismatches
        
        def compute_unified_dqn_target_q(current_q_values, next_q_values, actions_list, rewards, dones, gamma):
            """
            Unified DQN target Q computation with comprehensive tensor shape alignment.
            
            This function eliminates the root cause of tensor dimension mismatch errors
            by implementing the same rigorous tensor validation and alignment as BCQ.
            """
            self.logger.debug("🔧 DQN: Starting unified target Q computation")
            
            # Establish reference dimensions
            if not current_q_values or not isinstance(current_q_values, (list, tuple)):
                raise ValueError("current_q_values must be a non-empty list/tuple")
                
            ref_q = current_q_values[0]
            ref_batch_size = ref_q.size(0)
            ref_seq_len = ref_q.size(1) if ref_q.dim() >= 2 else 1
            
            self.logger.debug(f"🔧 DQN: Reference dimensions - batch: {ref_batch_size}, seq: {ref_seq_len}")
            
            # Align rewards and dones to reference dimensions
            # Handle rewards tensor shape
            if rewards.dim() > 1:
                rewards = rewards.squeeze()
            if rewards.dim() == 0:
                # Handle scalar rewards
                rewards = rewards.unsqueeze(0)
            if rewards.size(0) != ref_batch_size:
                if rewards.size(0) > ref_batch_size:
                    rewards = rewards[:ref_batch_size]
                else:
                    # Pad with zeros
                    pad_size = ref_batch_size - rewards.size(0)
                    padding = torch.zeros(pad_size, device=rewards.device, dtype=rewards.dtype)
                    rewards = torch.cat([rewards, padding])
            
            # Handle dones tensor shape
            if dones.dim() > 1:
                dones = dones.squeeze()
            if dones.dim() == 0:
                # Handle scalar dones
                dones = dones.unsqueeze(0)
            if dones.size(0) != ref_batch_size:
                if dones.size(0) > ref_batch_size:
                    dones = dones[:ref_batch_size]
                else:
                    # Pad with False (not done)
                    pad_size = ref_batch_size - dones.size(0)
                    padding = torch.zeros(pad_size, device=dones.device, dtype=dones.dtype)
                    dones = torch.cat([dones, padding])
            
            target_q_list = []
            
            # Compute target Q for each head with guaranteed shape consistency
            for head_idx in range(len(current_q_values)):
                try:
                    current_q = current_q_values[head_idx]
                    next_q = next_q_values[head_idx]
                    actions = actions_list[head_idx]
                    
                    # Align all tensors to reference dimensions
                    if current_q.size(0) != ref_batch_size:
                        current_q = current_q[:ref_batch_size]
                    if next_q.size(0) != ref_batch_size:
                        next_q = next_q[:ref_batch_size]
                    if actions.size(0) != ref_batch_size:
                        actions = actions[:ref_batch_size]
                    
                    # 🔧 ENHANCED SEQUENCE DIMENSION HANDLING
                    self.logger.debug(f"🔧 DQN head {head_idx}: current_q={current_q.shape}, next_q={next_q.shape}, actions={actions.shape}")
                    
                    if current_q.dim() == 3:
                        # Extract Q-values for last timestep (DQN typically uses single-step)
                        current_q_step = current_q[:, -1, :]  # (B, A)
                        next_q_step = next_q[:, -1, :]        # (B, A)
                        
                        # Handle action selection with enhanced validation
                        if actions.dim() == 2:
                            actions_step = actions[:, -1]    # (B,)
                        elif actions.dim() == 1:
                            actions_step = actions           # (B,)
                        else:
                            # Handle unusual action dimensions
                            actions_step = actions.view(actions.size(0), -1)[:, -1]  # Take last element
                    elif current_q.dim() == 2:
                        current_q_step = current_q           # (B, A)
                        next_q_step = next_q                 # (B, A)
                        
                        # Handle action dimension compatibility
                        if actions.dim() == 2:
                            actions_step = actions[:, -1]    # (B,)
                        elif actions.dim() == 1:
                            actions_step = actions           # (B,)
                        else:
                            actions_step = actions.view(actions.size(0))  # Flatten to (B,)
                    else:
                        # Fallback for unusual Q dimensions
                        current_q_step = current_q.view(current_q.size(0), -1)
                        next_q_step = next_q.view(next_q.size(0), -1)
                        actions_step = actions.view(actions.size(0)) if actions.dim() > 1 else actions
                    
                    # 🔧 CRITICAL VALIDATION: Ensure all tensors have consistent batch dimension
                    target_batch = min(current_q_step.size(0), next_q_step.size(0), actions_step.size(0))
                    if target_batch <= 0:
                        self.logger.error(f"❌ DQN head {head_idx}: Invalid batch size: {target_batch}")
                        raise ValueError(f"Invalid batch size for head {head_idx}")
                    
                    # Align all tensors to target batch size INCLUDING rewards and dones
                    current_q_step = current_q_step[:target_batch]
                    next_q_step = next_q_step[:target_batch]
                    actions_step = actions_step[:target_batch]
                    
                    # 🔧 CRITICAL FIX: Align rewards and dones to the same target_batch size
                    # This prevents tensor dimension mismatch in target_q computation
                    if rewards.size(0) >= target_batch:
                        rewards_step = rewards[:target_batch]
                    else:
                        # Pad with zeros if rewards is smaller
                        pad_size = target_batch - rewards.size(0)
                        padding = torch.zeros(pad_size, device=rewards.device, dtype=rewards.dtype)
                        rewards_step = torch.cat([rewards, padding])
                    
                    if dones.size(0) >= target_batch:
                        dones_step = dones[:target_batch]
                    else:
                        # Pad with False (not done) if dones is smaller
                        pad_size = target_batch - dones.size(0)
                        padding = torch.zeros(pad_size, device=dones.device, dtype=dones.dtype)
                        dones_step = torch.cat([dones, padding])
                    
                    # Validate action indices and Q dimensions
                    action_dim = current_q_step.size(-1)
                    if next_q_step.size(-1) != action_dim:
                        self.logger.error(f"❌ DQN head {head_idx}: Q dimension mismatch: current={action_dim}, next={next_q_step.size(-1)}")
                        # Align to smaller dimension
                        min_action_dim = min(action_dim, next_q_step.size(-1))
                        current_q_step = current_q_step[:, :min_action_dim]
                        next_q_step = next_q_step[:, :min_action_dim]
                        action_dim = min_action_dim
                    
                    # Clamp actions to valid range with enhanced logging
                    original_actions = actions_step.clone()
                    actions_step = torch.clamp(actions_step, min=0, max=action_dim - 1)
                    
                    invalid_actions = (original_actions != actions_step).sum().item()
                    if invalid_actions > 0:
                        self.logger.warning(f"⚠️ DQN head {head_idx}: Clamped {invalid_actions} invalid actions to range [0, {action_dim-1}]")
                    
                    self.logger.debug(f"✅ DQN head {head_idx}: Aligned shapes - current_q={current_q_step.shape}, next_q={next_q_step.shape}, actions={actions_step.shape}, rewards={rewards_step.shape}, dones={dones_step.shape}")
                
                    
                    # DQN target computation with properly aligned tensors
                    with torch.no_grad():
                        # Add detailed debugging
                        self.logger.debug(f"🔍 DQN head {head_idx} before target Q computation:")
                        self.logger.debug(f"   next_q_step shape: {next_q_step.shape}")
                        self.logger.debug(f"   rewards_step shape before squeeze: {rewards_step.shape}")
                        self.logger.debug(f"   dones_step shape before squeeze: {dones_step.shape}")
                        
                        next_q_max = next_q_step.max(dim=1)[0]  # (target_batch,)
                        
                        # Ensure rewards_step and dones_step are 1D
                        if rewards_step.dim() > 1:
                            rewards_step = rewards_step.squeeze(-1)
                        if dones_step.dim() > 1:
                            dones_step = dones_step.squeeze(-1)
                        
                        self.logger.debug(f"   next_q_max shape: {next_q_max.shape}")
                        self.logger.debug(f"   rewards_step shape after squeeze: {rewards_step.shape}")
                        self.logger.debug(f"   dones_step shape after squeeze: {dones_step.shape}")
                        
                        # Verify all tensors have the same batch size before computation
                        if not (next_q_max.size(0) == rewards_step.size(0) == dones_step.size(0)):
                            self.logger.error(f"❌ DQN head {head_idx}: Batch size mismatch before target computation!")
                            self.logger.error(f"   next_q_max: {next_q_max.size(0)}, rewards: {rewards_step.size(0)}, dones: {dones_step.size(0)}")
                            raise ValueError(f"Batch size mismatch in head {head_idx}")
                        
                        target_q = rewards_step + gamma * next_q_max * (1 - dones_step)  # All tensors now have shape (target_batch,)
                    
                    target_q_list.append(target_q)
                    self.logger.debug(f"✅ DQN: Successfully computed target Q for head {head_idx}")
                    
                except Exception as e:
                    self.logger.error(f"❌ DQN: Target Q computation failed for head {head_idx}: {e}")
                    # Fallback to zero target
                    fallback_target = torch.zeros(ref_batch_size, device=self.device)
                    target_q_list.append(fallback_target)
            
            return target_q_list

        # 🔧 ENHANCED DQN Q-VALUE EXTRACTION: Comprehensive shape handling
        current_q_list = []
        for i, action_dim in enumerate(self.action_dims):
            try:
                action_tensor = actions[i]
                q_head = q_values[i]
                
                # Validate input shapes
                self.logger.debug(f"🔧 DQN head {i}: q_head={q_head.shape}, actions={action_tensor.shape}")
                
                # Ensure batch dimension consistency
                batch_size = q_head.size(0)
                if action_tensor.size(0) != batch_size:
                    self.logger.warning(f"⚠️ DQN head {i}: Batch size mismatch: {action_tensor.size(0)} vs {batch_size}")
                    min_batch = min(action_tensor.size(0), batch_size)
                    action_tensor = action_tensor[:min_batch]
                    q_head = q_head[:min_batch]
                    batch_size = min_batch
                
                # Clamp actions to valid range [0, action_dims[i]-1] to prevent gather errors
                original_max = safe_item(action_tensor.max()) if action_tensor.numel() > 0 else 0
                action_tensor = torch.clamp(action_tensor, min=0, max=self.action_dims[i] - 1)
                
                # Log warning if invalid actions were found and corrected
                if original_max >= self.action_dims[i]:
                    invalid_count = safe_item((torch.tensor(a, dtype=torch.long, device=self.device) >= self.action_dims[i]).sum())
                    self.logger.warning(f"⚠️  [DQN] Fixed {invalid_count} invalid actions in head {i}: "
                              f"range [0, {self.action_dims[i]-1}], max found: {original_max}")
                
                # 🔧 ENHANCED Q-VALUE EXTRACTION: Comprehensive tensor validation and alignment
                self.logger.debug(f"🔧 DQN Q-extraction head {i}: q_head={q_head.shape}, action_tensor={action_tensor.shape}")
                
                # Pre-extraction validation to prevent gather errors
                if q_head.numel() == 0 or action_tensor.numel() == 0:
                    self.logger.error(f"❌ DQN head {i}: Empty tensors detected")
                    q_sa = torch.zeros(batch_size, device=self.device)
                    current_q_list.append(q_sa)
                    continue
                
                # Handle all dimension combinations with robust error checking
                try:
                    if q_head.dim() == 3 and action_tensor.dim() == 2:
                        # Sequence case: (B, T, A) and (B, T)
                        # Extract last timestep for DQN
                        q_last = q_head[:, -1, :]           # (B, A)
                        action_last = action_tensor[:, -1]  # (B,)
                        
                        # Validate gather dimensions
                        if q_last.size(0) != action_last.size(0):
                            min_batch = min(q_last.size(0), action_last.size(0))
                            q_last = q_last[:min_batch]
                            action_last = action_last[:min_batch]
                        
                        action_clamped = torch.clamp(action_last, 0, q_last.size(1) - 1)
                        q_sa = q_last.gather(1, action_clamped.unsqueeze(1)).squeeze(1)  # (B,)
                        
                    elif q_head.dim() == 2 and action_tensor.dim() == 1:
                        # Standard case: (B, A) and (B,)
                        if q_head.size(0) != action_tensor.size(0):
                            min_batch = min(q_head.size(0), action_tensor.size(0))
                            q_head = q_head[:min_batch]
                            action_tensor = action_tensor[:min_batch]
                        
                        action_clamped = torch.clamp(action_tensor, 0, q_head.size(1) - 1)
                        q_sa = q_head.gather(1, action_clamped.unsqueeze(1)).squeeze(1)  # (B,)
                        
                    elif q_head.dim() == 3 and action_tensor.dim() == 1:
                        # Broadcast case: (B, T, A) and (B,)
                        q_last = q_head[:, -1, :]  # (B, A)
                        
                        if q_last.size(0) != action_tensor.size(0):
                            min_batch = min(q_last.size(0), action_tensor.size(0))
                            q_last = q_last[:min_batch]
                            action_tensor = action_tensor[:min_batch]
                        
                        action_clamped = torch.clamp(action_tensor, 0, q_last.size(1) - 1)
                        q_sa = q_last.gather(1, action_clamped.unsqueeze(1)).squeeze(1)  # (B,)
                        
                    elif q_head.dim() == 2 and action_tensor.dim() == 2:
                        # Mixed case: (B, A) and (B, T)
                        action_last = action_tensor[:, -1]  # (B,)
                        
                        if q_head.size(0) != action_last.size(0):
                            min_batch = min(q_head.size(0), action_last.size(0))
                            q_head = q_head[:min_batch]
                            action_last = action_last[:min_batch]
                        
                        action_clamped = torch.clamp(action_last, 0, q_head.size(1) - 1)
                        q_sa = q_head.gather(1, action_clamped.unsqueeze(1)).squeeze(1)  # (B,)
                        
                    else:
                        # Fallback for unusual cases with enhanced safety
                        self.logger.warning(f"⚠️ DQN head {i}: Unusual dimensions - using enhanced fallback")
                        
                        # Ensure we can create a meaningful fallback
                        target_batch = min(q_head.size(0), action_tensor.size(0)) if action_tensor.numel() > 0 else q_head.size(0)
                        
                        if q_head.dim() >= 2 and target_batch > 0:
                            q_flat = q_head.view(target_batch, -1)
                            if action_tensor.numel() > 0:
                                action_flat = action_tensor.view(target_batch, -1)[:, 0]  # Take first action
                                action_clamped = torch.clamp(action_flat, 0, q_flat.size(1) - 1)
                                q_sa = q_flat.gather(1, action_clamped.unsqueeze(1)).squeeze(1)
                            else:
                                q_sa = q_flat[:, 0]  # Take first Q-value
                        else:
                            q_sa = torch.zeros(target_batch, device=self.device)
                
                    # Final validation
                    if q_sa.dim() != 1:
                        self.logger.warning(f"⚠️ DQN head {i}: Q-values not 1D, reshaping from {q_sa.shape}")
                        q_sa = q_sa.view(-1)
                    
                    # Ensure batch size consistency
                    if q_sa.size(0) != batch_size:
                        if q_sa.size(0) > batch_size:
                            q_sa = q_sa[:batch_size]
                        else:
                            # Pad with zeros
                            padding = torch.zeros(batch_size - q_sa.size(0), device=self.device)
                            q_sa = torch.cat([q_sa, padding])
                    
                except Exception as gather_error:
                    self.logger.error(f"❌ DQN head {i}: Gather operation failed: {gather_error}")
                    self.logger.error(f"🔍 Shapes: q_head={q_head.shape}, action_tensor={action_tensor.shape}")
                    q_sa = torch.zeros(batch_size, device=self.device)
                
                current_q_list.append(q_sa)
                self.logger.debug(f"✅ DQN head {i}: Q-value extraction successful, shape: {q_sa.shape}")
                
            except Exception as e:
                self.logger.error(f"❌ DQN head {i}: Q-value extraction failed: {e}")
                # Emergency fallback
                fallback_q = torch.zeros(q_head.size(0), device=self.device)
                current_q_list.append(fallback_q)
                self.logger.warning(f"⚠️ DQN head {i}: Using zero fallback Q-values")

        # 🔧 TARGET Q COMPUTATION: Use unified system
        try:
            target_q_values = compute_unified_dqn_target_q(
                current_q_values=q_values,
                next_q_values=next_q_values,
                actions_list=actions,
                rewards=rewards,
                dones=dones,
                gamma=self.gamma
            )
            
            # Compute losses with guaranteed shape consistency
            total_loss = 0.0
            valid_heads = 0
            
            for i, (current_q, target_q) in enumerate(zip(current_q_list, target_q_values)):
                try:
                    # Final shape validation
                    if current_q.shape != target_q.shape:
                        self.logger.warning(f"⚠️ DQN head {i}: Shape mismatch - current: {current_q.shape}, target: {target_q.shape}")
                        min_size = min(current_q.size(0), target_q.size(0))
                        current_q = current_q[:min_size]
                        target_q = target_q[:min_size]
                    
                    # Compute TD loss
                    td_loss = F.mse_loss(current_q, target_q.detach())
                    
                    if not torch.isfinite(td_loss):
                        self.logger.error(f"❌ DQN head {i}: Non-finite loss: {td_loss}")
                        continue
                    
                    total_loss += td_loss
                    valid_heads += 1
                    
                    self.logger.debug(f"🔧 DQN head {i}: TD loss = {td_loss.item():.6f}")
                    
                except Exception as loss_e:
                    self.logger.error(f"❌ DQN head {i}: Loss computation failed: {loss_e}")
                    continue
                    
            if valid_heads == 0:
                self.logger.error("❌ DQN: No valid head losses computed")
                return 0.0
                
            average_loss = total_loss / valid_heads
            self.logger.debug(f"🔧 DQN: Average loss across {valid_heads} heads: {average_loss.item():.6f}")
            
        except Exception as target_e:
            self.logger.error(f"❌ DQN: Target Q computation completely failed: {target_e}")
            return 0.0

        # Backward pass and optimization
        self.optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        # Update target network
        self.update_target()

        return safe_item(total_loss)

    def act(
        self, 
        state: torch.Tensor, 
        greedy: bool = True, 
        **kwargs
    ) -> torch.Tensor:
        """Selects actions for given states following BaseRLAgent interface.
        
        This method implements the agent's policy for action selection compatible
        with the BaseRLAgent abstract interface. It converts the structured input
        format to the required tensor format for DQN action selection.
        
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

    def save(self, path: str) -> None:
        """
        Save the agent's state to file.
        
        Args:
            path: File path to save the agent state.
        """
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_steps': self.update_steps
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's state from file.
        
        Args:
            path: File path to load the agent state from.
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.target_model.load_state_dict(state['target_model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.update_steps = state.get('update_steps', 0)

    def save_checkpoint(self, filepath: str) -> None:
        """Saves agent state to checkpoint file.

        Args:
            filepath: Path to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
                'polyak_tau': getattr(self, 'polyak_tau', 0.005),
                'double_q': self.double_q,
                'dueling': self.dueling,
                'noisy': self.noisy,
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._training_step = checkpoint['training_step']
        self._episode_count = checkpoint['episode_count']
        self.update_steps = checkpoint.get('update_steps', 0)

    # ------------------------------------------------------------------
    #  Target network helpers
    # ------------------------------------------------------------------
    def _soft_update_target(self) -> None:
        """Polyak averaging update of target network parameters."""
        tau = float(getattr(self, "polyak_tau", 1.0))
        with torch.no_grad():
            for tgt, src in zip(self.target_model.parameters(), self.model.parameters()):
                tgt.data.mul_(1 - tau)
                tgt.data.add_(tau * src.data)
