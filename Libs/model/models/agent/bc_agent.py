"""
Behavioral Cloning (BC) agent implementation for multi-discrete action spaces.

This module provides a BC agent for supervised imitation learning in medical
decision making scenarios, learning directly from demonstration data using
cross-entropy loss rather than temporal difference learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Any
from .base_agent import BaseRLAgent
import logging
from Libs.model.models.agent._compat import ForwardCompatMixin
from Libs.utils.model_utils import safe_float
from Libs.utils.exp_utils import seed_everything
from Libs.utils.model_utils import as_tensor


class BCReplayBuffer:
    """
    **BC SPECIFIC**: Supervised learning buffer for behavioral cloning.

    Unlike Q-learning replay buffers, BC only needs (state, action) pairs
    since it's purely supervised learning from demonstrations.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize the BC buffer with given capacity."""
        self.capacity = capacity
        self.buffer: List[Tuple[Any, Any]] = []
        self.position = 0

    def add(self, state: np.ndarray, action: np.ndarray) -> None:
        """Add a (state, action) pair to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch of (state, action) pairs from the buffer.

        Args:
            batch_size: Number of pairs to sample.

        Returns:
            Tuple of (states, actions) arrays.
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        effective_batch_size = min(batch_size, len(self.buffer))
        idxs = np.random.choice(
            len(self.buffer), effective_batch_size, replace=True)
        batch = [self.buffer[i] for i in idxs]

        states, actions = zip(*batch)
        return np.array(states), np.array(actions)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


# ---------------------------------------------------------------------
#  Main Behavioural-Cloning Agent
# ---------------------------------------------------------------------


class BCAgent(ForwardCompatMixin, BaseRLAgent):
    """
    **BEHAVIORAL CLONING AGENT**: Pure imitation learning for medical decision making.

    This agent implements behavioral cloning, which is fundamentally different
    from Q-learning approaches. BC learns to directly imitate the demonstration
    policy using supervised learning rather than reinforcement learning.

    Key Differences from Q-learning agents:
    - **Learning**: Supervised (state->action) vs RL (state->value->action)
    - **Loss**: Cross-entropy vs MSE/Huber
    - **Data**: Only needs demonstrations vs needs rewards/transitions
    - **Training**: Batch supervised learning vs temporal difference learning

    This ensures BC produces genuinely different results from DQN/CQL/BCQ.

    Attributes:
        model: BC network for direct policy learning.
        action_dims: List of action space sizes for each action head.
        temperature: Temperature for stochastic action selection.

    Example:
        >>> agent = BCAgent(
        ...     model=bc_network,
        ...     action_dims=[7, 6, 6],
        ...     temperature=1.0,
        ...     lr=1e-3
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        action_dims: List[int],
        lr: float = 1e-3,
        gamma: float = 0.99,  # Not used in BC, kept for interface compatibility
        buffer_size: int = 10000,
        batch_size: int = 64,
        temperature: float = 1.0,
        device: str = 'cpu',
        reward_centering: bool = False,  # Not used in BC
        *,
        seed: int | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Initialize the BC agent.

        Args:
            model: BC network architecture for policy learning.
            action_dims: List of action space sizes for each action head.
            lr: Learning rate for optimizer.
            gamma: Discount factor (not used in BC, kept for compatibility).
            buffer_size: Maximum size of the demonstration buffer.
            batch_size: Batch size for supervised learning updates.
            temperature: Temperature for stochastic action selection.
            device: Computing device ('cpu' or 'cuda').
            reward_centering: Not used in BC (kept for compatibility).
            seed: Random seed for reproducibility.
            label_smoothing: Label smoothing factor for cross-entropy loss.
        """
        # Convert parameters safely
        lr = safe_float(lr, 1e-3)
        gamma = safe_float(gamma, 0.99)
        temperature = safe_float(temperature, 1.0)

        super().__init__(device=device, gamma=gamma)

        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        self.model = model.to(device)
        self.action_dims = action_dims
        # Use a **stronger weight decay (1e-3)** to regularise the BC policy
        # network, helping curb over-fitting observed after ~50 epochs in the
        # latest benchmark runs.
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.batch_size = batch_size
        self.temperature = temperature
        self.label_smoothing = float(label_smoothing)

        # **BC SPECIFIC**: Only need demonstration buffer (no Q-learning replay buffer)
        self.demo_buffer = BCReplayBuffer(buffer_size)

        # Training tracking
        self.update_steps = 0
        self.total_accuracy = 0.0
        self.total_loss = 0.0
        # Legacy flag used by Trainer to supply (obs, actions, mask, lengths, ...)
        # instead of Dict batches.
        self.expects_tuple_batch = True

        # RNG reproducibility ------------------------------------------------
        if seed is not None:
            seed_everything(int(seed))

    def _forward_model(self, model: nn.Module, obs: torch.Tensor, lengths: torch.Tensor,
                       edge_index: torch.Tensor, mask: Optional[torch.Tensor] = None,
                       mode: str = 'policy', **kwargs):
        """
        **BC SPECIFIC**: Forward pass through BC model for policy learning.

        BC models output action probabilities, not Q-values.
        """
        if self._is_pog_model(model):
            try:
                # 🔧 CRITICAL FIX: Ensure all tensors are contiguous for cuDNN
                if not obs.is_contiguous():
                    obs = obs.contiguous()
                if not lengths.is_contiguous():
                    lengths = lengths.contiguous()
                    
                # Ensure edge_index is a torch.LongTensor on the correct device
                edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=obs.device)
                if not edge_index.is_contiguous():
                    edge_index = edge_index.contiguous()
                    
                edge_index = as_tensor(edge_index, dtype=torch.long, device=obs.device)
                
                if mask is not None and not mask.is_contiguous():
                    mask = mask.contiguous()

                return model(obs, lengths, edge_index, mask, mode='policy', **kwargs)
            except Exception as e:
                self.logger.error(f"❌ PoG model forward failed: {e}")
                raise RuntimeError(f"PoG model forward failed: {e}") from e
        else:
            try:
                batch_size, seq_len, feature_dim = obs.shape
                last_indices = lengths - 1
                batch_indices = torch.arange(batch_size, device=obs.device)
                final_obs = obs[batch_indices, last_indices]

                # **BC SPECIFIC**: Call BC model forward for action probabilities
                model_out = model(final_obs)
                # BCPolicyNet returns (logits_list, probs_list); we need logits_list
                if isinstance(model_out, tuple):
                    action_logits_list, _ = model_out
                else:
                    # assume already List[Tensor]
                    action_logits_list = model_out

                # Expand to sequence length for compatibility
                policy_outputs = []
                for logit in action_logits_list:
                    # Convert log-probs to expanded format: (B, T, action_dim)
                    expanded = logit.unsqueeze(1).expand(-1, seq_len, -1)
                    policy_outputs.append(expanded)

                # **BC FORMAT**: Return (action_log_probs, None, 0.0, 0.0)
                return (policy_outputs, None, 0.0, 0.0)

            except Exception as e:
                self.logger.error(f"❌ BC model forward failed: {e}")
                self.logger.error(
                    f"📊 Debug info: obs shape={obs.shape}, model type={type(model)}")
                raise RuntimeError(f"BC model forward failed: {e}") from e

    def add_demonstration(self, obs: np.ndarray, actions: List[np.ndarray]) -> None:
        """
        **BC SPECIFIC**: Add demonstration data to the buffer.

        Args:
            obs: Observation data.
            actions: List of actions for each action head.
        """
        # Combine multi-head actions into single array
        combined_actions = np.column_stack(actions)
        self.demo_buffer.add(obs, combined_actions)

    def update(self, batch: tuple, *, grad_scaler: Optional[torch.cuda.amp.GradScaler] = None) -> tuple[float, float, float]:  # noqa: D401
        """执行一次行为克隆监督更新（向量化实现）。

        支持两种批格式：

        1. *(obs, actions, mask, lengths)* – 经典 BC。  
        2. *(obs, actions, mask, lengths, edge_index, _)* – PoG-BC。

        Returns:
            Tuple[float, float, float]: *(loss, accuracy, 0.0)* – 保持 Trainer 接口一致。
        """

        self.model.train()

        # -------- Batch 解包 & 张量化 --------
        if len(batch) not in {4, 6}:
            raise ValueError("BC.update expects 4 or 6-tuple batch")

        obs, actions, mask, lengths = batch[:4]
        edge_index = batch[4] if len(batch) == 6 else None

        obs = as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = as_tensor(
            actions, dtype=torch.long, device=self.device)
        mask = as_tensor(mask, dtype=torch.float32, device=self.device)
        lengths = as_tensor(
            lengths, dtype=torch.long, device=self.device)

        # --------------------------------------------------------------
        #  🚨 OOM SAFEGUARD  ------------------------------------------------
        #  Plan-on-Graph (PoG) backbones incur **quadratic** memory usage
        #  w.r.t. sequence length due to the self-attention stage.  Extremely
        #  long patient trajectories (>1k timesteps) may therefore exhaust
        #  GPU memory even with modest batch sizes.  We truncate each sample
        #  to the *most recent* ``MAX_SEQ_LEN`` timesteps (default 512),
        #  which preserves the clinically relevant recency information while
        #  capping the memory footprint to a predictable upper bound.
        # --------------------------------------------------------------
        MAX_SEQ_LEN = 512  # ⭐ configurable via subclass attribute if needed
        if obs.dim() == 3 and obs.size(1) > MAX_SEQ_LEN:
            # Keep last MAX_SEQ_LEN steps — align *mask* / *actions*.
            obs = obs[:, -MAX_SEQ_LEN:, :]
            mask = mask[:, -MAX_SEQ_LEN:]
            actions = actions[:, -MAX_SEQ_LEN:, :]
            # Update lengths to reflect the truncated sequence
            lengths = torch.clamp(lengths, max=MAX_SEQ_LEN)
            lengths = mask.sum(dim=1).long()

        # 单步样本兼容 (B,D) → (B,1,D)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            mask = mask.unsqueeze(1)
            actions = actions.unsqueeze(1)
            lengths = torch.ones(
                obs.size(0), dtype=torch.long, device=self.device)

        batch_size, seq_len, _ = obs.shape

        # -------- 前向获得 logits --------
        if edge_index is None:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device)

        logits_out = self._forward_model(
            self.model, obs, lengths, edge_index, mask, mode="policy"
        )[0]  # List[(B,T,A_i)]

        total_loss, total_correct, total_valid = 0.0, 0.0, 0.0

        for head_idx, (logits, a_dim) in enumerate(zip(logits_out, self.action_dims)):
            # Flatten time & batch dims for efficient CE
            logits_flat = logits.reshape(-1, a_dim)
            targets_flat = actions[..., head_idx].reshape(-1)
            mask_flat = mask.reshape(-1)

            ce = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction="none",
                label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
            )

            # 仅在有效时间步累积损失
            ce = (ce * mask_flat).sum()
            valid = mask_flat.sum().clamp_min(1)

            total_loss += ce / valid

            # Accuracy
            preds = logits_flat.argmax(dim=-1)
            total_correct += ((preds == targets_flat) * mask_flat).sum()
            total_valid += valid

        total_loss = total_loss / len(self.action_dims)

        # -------- 反向与优化 --------
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Accuracy 可用于日志，但为保持 Trainer 接口不变，此处返回占位 0.0
        self.update_steps += 1

        return total_loss.item(), 0.0, 0.0

    def select_action(
        self,
        obs: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eval_mode: bool = False,
        epsilon: float = 0.0,  # Not used in BC
        **kwargs
    ) -> List[torch.Tensor]:
        """**BC SPECIFIC**: Select actions from learned policy distribution."""
        if obs.dim() != 3:
            raise ValueError(
                f"Expected 3D observation tensor, got {obs.dim()}D")
        if lengths.dim() != 1 or lengths.size(0) != obs.size(0):
            raise ValueError("lengths must be 1D with same batch size as obs")

        batch_size, seq_len, _ = obs.shape

        # **BC SPECIFIC**: Always use policy distribution (no epsilon-greedy)
        with torch.no_grad():
            model_output = self._forward_model(
                self.model, obs, lengths, edge_index, mask, mode='policy'
            )

            if isinstance(model_output, tuple):
                action_log_probs = model_output[0]
            else:
                action_log_probs = model_output

            # **BC ADVANTAGE**: Can use stochastic or deterministic action selection
            actions = []
            for log_probs in action_log_probs:
                if eval_mode or self.temperature <= 0.1:
                    # Deterministic: use most likely action
                    greedy_actions = log_probs.argmax(dim=-1)
                    actions.append(greedy_actions)
                else:
                    # Stochastic: sample from policy distribution
                    # Apply temperature scaling
                    scaled_log_probs = log_probs / self.temperature
                    probs = torch.exp(scaled_log_probs)

                    # Sample actions
                    sampled_actions = torch.multinomial(
                        probs.view(-1, probs.size(-1)),
                        num_samples=1
                    ).view(batch_size, seq_len)
                    actions.append(sampled_actions)

        return actions

    def act(
        self,
        state: torch.Tensor,
        greedy: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        **BC SPECIFIC**: Select actions using behavioral cloning policy.

        Args:
            state: Input state tensor.
            greedy: Whether to use deterministic (True) or stochastic (False) selection.
            **kwargs: Additional arguments.

        Returns:
            Selected actions tensor.
        """
        # Handle input dimensions
        if state.dim() == 2:
            state = state.unsqueeze(0)
            single_sequence = True
        elif state.dim() == 3:
            single_sequence = False
        else:
            raise ValueError(
                f"Expected 2D or 3D state tensor, got {state.dim()}D")

        batch_size, seq_len, obs_dim = state.shape

        # Create default parameters
        lengths = kwargs.get('lengths', torch.full((batch_size,), seq_len,
                                                   dtype=torch.long, device=self.device))
        edge_index = kwargs.get('edge_index', torch.empty((2, 0),
                                                          dtype=torch.long, device=self.device))
        mask = kwargs.get('mask', torch.ones((batch_size, seq_len),
                                             dtype=torch.float32, device=self.device))

        # Use select_action method
        action_list = self.select_action(
            obs=state,
            lengths=lengths,
            edge_index=edge_index,
            mask=mask,
            eval_mode=greedy
        )

        # Convert to stacked tensor
        actions = torch.stack(action_list, dim=-1)

        # Return appropriate shape
        if single_sequence:
            return actions.squeeze(0)
        else:
            return actions

    def save(self, path):
        """Save BC agent state."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_steps': self.update_steps,
            'temperature': self.temperature
        }, path)

    def load(self, path):
        """Load BC agent state."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.update_steps = state.get('update_steps', 0)
        self.temperature = state.get('temperature', 1.0)

    def save_checkpoint(self, filepath: str) -> None:
        """Saves agent state to checkpoint file.

        Args:
            filepath: Path to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'update_steps': self.update_steps,
            'config': {
                'action_dims': self.action_dims,
                'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 1e-3,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'temperature': self.temperature,
                'label_smoothing': self.label_smoothing,
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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._training_step = checkpoint['training_step']
        self._episode_count = checkpoint['episode_count']
        self.update_steps = checkpoint.get('update_steps', 0)
