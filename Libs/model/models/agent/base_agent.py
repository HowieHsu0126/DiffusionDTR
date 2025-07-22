"""Base reinforcement learning agent interface.

This module defines the abstract base class for all reinforcement learning
agents, providing a consistent interface for training, evaluation, and
action selection across different algorithm implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn


class BaseRLAgent(ABC):
    """Abstract base class for reinforcement learning agents.
    
    This class defines the standard interface that all RL agents should
    implement, ensuring consistency across different algorithm implementations
    and facilitating easy integration with training and evaluation frameworks.
    
    The base class handles common functionality such as device management,
    target network updates, and provides template methods for core RL
    operations that subclasses must implement.
    
    Attributes:
        device: The computation device (CPU or CUDA).
        gamma: Discount factor for future rewards.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = 'cpu',
        gamma: float = 0.99,
        target_update_freq: int = 100,
        reward_centering: bool = False,
        polyak_tau: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initializes the base RL agent.
        
        Args:
            device: Computation device ('cpu' or 'cuda' or torch.device).
            gamma: Discount factor for future rewards. Must be in [0, 1].
            
        Raises:
            ValueError: If gamma is not in the valid range [0, 1].
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0.0 and 1.0")
            
        self.device = torch.device(device) if isinstance(device, str) else device
        self.gamma = gamma
        
        # Common hyper-parameters
        self.target_update_freq = int(target_update_freq)
        self.reward_centering = bool(reward_centering)
        
        # Polyak averaging factor; 1.0 == hard copy every N steps
        self.polyak_tau = float(polyak_tau)
        
        # Training statistics
        self._training_step = 0
        self._episode_count = 0
        
        # ------------------------------------------------------------------
        #  Reward scaling configuration
        # ------------------------------------------------------------------
        # Researchers can override these attributes *after* construction
        # (e.g. via Hydra config) to enable custom shaping without touching
        # agent internals.
        self.reward_scale_method: str = 'identity'  # {'identity','sigmoid','tanh'}
        # Empirically, SOFA ∆score ranges roughly in [-24, 24].  Using
        # temperature=1.0 would push most values into the saturated regime
        # of sigmoid/tanh, yielding gradients ≈ 0.  A larger temperature
        # widens the linear region.  We choose 8.0 as robust default while
        # still bounding scaled rewards to (-1,1).
        self.reward_scale_temperature: float = 8.0  # controls steepness
        self.reward_time_decay: float = 0.0         # ∈ [0,1]; 0 == no decay
        
        # ------------------------------------------------------------------
        #  Reward centering (Naik et al., 2024, "Reward Centering")
        # ------------------------------------------------------------------
        # Running estimate µ_t of expected reward updated every call to
        # :meth:`_center_rewards` using an exponential moving average.
        # A small α ensures a slow drift so that subtracting µ_t ≈ 𝔼[r]
        # does not introduce bias yet accelerates learning for γ → 1.
        self._reward_center: torch.Tensor | None = None
        self.reward_center_alpha: float = 1e-3  # can be tuned via cfg
        
        # ------------------------------------------------------------------
        #  RNG reproducibility – centralised seed initialisation so that every
        #  agent benefits from deterministic behaviour without duplicating
        #  calls to ``seed_everything``.
        # ------------------------------------------------------------------
        if seed is not None:
            try:
                from Libs.utils.exp_utils import seed_everything
                seed_everything(int(seed))
            except Exception as exc:
                # Fall back gracefully when utility unavailable (unit tests)
                import random

                import numpy as np

                # 注意：不要重新导入torch，因为它已经在模块级别导入了
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
        
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        """Updates the agent's policy/value function using a batch of experiences.
        
        This method implements the core learning algorithm (e.g., Q-learning,
        policy gradient, etc.) and updates the model parameters based on the
        provided batch of experience tuples.
        
        Args:
            batch: Dictionary containing experience tensors with keys like:
                  'state', 'action', 'reward', 'next_state', 'done', etc.
                  
        Returns:
            The training loss value for logging and monitoring.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement update method")
    
    @abstractmethod
    def act(
        self, 
        state: torch.Tensor, 
        greedy: bool = True, 
        **kwargs
    ) -> torch.Tensor:
        """Selects actions for given states.
        
        This method implements the agent's policy for action selection,
        supporting both greedy (deterministic) and exploratory (stochastic)
        action selection modes.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim).
            greedy: Whether to use greedy (deterministic) action selection.
            **kwargs: Additional arguments specific to the agent implementation.
            
        Returns:
            Selected actions tensor of shape (batch_size, action_dim).
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement act method")
    
    def update_target(self) -> None:
        """Updates target networks with Polyak averaging (soft update).

        The method assumes subclasses define two attributes:
            * ``policy_net``  – the main online network (\*source\*)
            * ``target_net``  – the target network (\*destination\*)

        If either attribute is missing or not an ``nn.Module``, the call is a
        no-op.
        """
        if not hasattr(self, "target_net") or not hasattr(self, "policy_net"):
            # Sub-class does not use target networks ➜ silently ignore.
            return

        target_net = getattr(self, "target_net")
        policy_net = getattr(self, "policy_net")

        if not isinstance(target_net, nn.Module) or not isinstance(policy_net, nn.Module):
            return

        # Hard update (copy) when tau == 1.0; otherwise Polyak averaging.
        tau = float(getattr(self, "polyak_tau", 1.0))
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Saves agent state to a checkpoint file.
        
        Saves the current state of the agent including model parameters,
        optimizer state, and training statistics for later restoration.
        
        Args:
            filepath: Path where to save the checkpoint.
            
        Raises:
            NotImplementedError: Default implementation raises error.
                                Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement save_checkpoint")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Loads agent state from a checkpoint file.
        
        Restores the agent state including model parameters, optimizer state,
        and training statistics from a previously saved checkpoint.
        
        Args:
            filepath: Path to the checkpoint file to load.
            
        Raises:
            NotImplementedError: Default implementation raises error.
                                Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement load_checkpoint")
    
    def set_training_mode(self, training: bool = True) -> None:
        """Switch **all** contained ``nn.Module`` to train/eval mode.

        Sub-classes that need finer control can override this method but should
        call ``super().set_training_mode(...)`` so that shared components (e.g.
        behaviour policy networks) are toggled consistently.
        """
        for attr_name in dir(self):
            # Heuristic: skip dunder & private attributes quickly
            if attr_name.startswith("__"):
                continue
            attr = getattr(self, attr_name, None)
            if isinstance(attr, nn.Module):
                attr.train(training)
    
    def reset(self) -> None:
        """Resets the agent's internal state.
        
        Clears any internal state that might persist between episodes,
        such as recurrent hidden states, exploration schedules, etc.
        """
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Returns current training statistics.
        
        Provides access to training metrics and statistics for monitoring
        and logging purposes.
        
        Returns:
            Dictionary containing training statistics like step count,
            episode count, loss values, etc.
        """
        return {
            'training_step': self._training_step,
            'episode_count': self._episode_count,
            'device': str(self.device),
            'gamma': self.gamma
        }
    
    def increment_training_step(self) -> None:
        """Increments the training step counter."""
        self._training_step += 1
    
    def increment_episode_count(self) -> None:
        """Increments the episode counter."""
        self._episode_count += 1
    
    @property
    def training_step(self) -> int:
        """Returns the current training step count."""
        return self._training_step
    
    @property 
    def episode_count(self) -> int:
        """Returns the current episode count."""
        return self._episode_count

    def _center_rewards(self, rewards: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Conditionally center rewards around zero with numerical stability.

        This implementation follows Naik et al., 2024 "Reward Centering" to improve
        discounted reinforcement learning performance by subtracting the empirical
        average reward. The method is particularly effective when discount factors
        approach 1 and is robust to constant reward shifts.

        Args:
            rewards: Reward tensor of any shape.
            dim: Dimension along which to compute the mean for centering.

        Returns
        -------
        torch.Tensor
            Centered reward tensor if ``self.reward_centering`` is ``True`` otherwise
            the original tensor (unchanged).
        """
        if not self.reward_centering:
            return rewards

        # Compute batch mean with gradient detached
        batch_mean = rewards.mean(dim=dim, keepdim=True).detach()

        # Initialise running mean on first call — ensures correct dtype/device.
        if self._reward_center is None:
            self._reward_center = batch_mean
        else:
            # Exponential moving average update µ ← (1-α)µ + α r̄_t
            self._reward_center = (1 - self.reward_center_alpha) * self._reward_center + self.reward_center_alpha * batch_mean

        # Center rewards with numerical stability
        centered_rewards = rewards - self._reward_center
        
        # Add numerical stability: prevent rewards from becoming too small
        # This addresses the issue mentioned in DQN where centered rewards can lead to near-zero loss
        reward_scale = rewards.abs().mean().detach()
        if reward_scale > 0:
            # Only apply scaling if original rewards have meaningful magnitude
            min_scale = 1e-3  # Minimum scale to maintain numerical stability
            if reward_scale < min_scale:
                # Scale up centered rewards to maintain gradient flow
                scale_factor = min_scale / reward_scale
                centered_rewards = centered_rewards * scale_factor
                
        return centered_rewards

    @staticmethod
    def _assert_batch_obs_dims(obs: torch.Tensor) -> None:
        """Assert that an observation tensor follows the expected (B,T,D) shape.

        Raises
        ------
        ValueError
            If ``obs`` does not have exactly 3 dimensions.
        """
        # Accept both (B,T,F) and flattened (B,F) observations for flexibility.
        # Many offline algorithms operate on per-step states without an
        # explicit time dimension during supervised/TD updates.
        if obs.dim() not in {2, 3}:
            raise ValueError(
                "Expected 2-D or 3-D observation tensor; got " f"{obs.shape}"
            )

    # ------------------------------------------------------------------
    #  New: reward shaping helpers
    # ------------------------------------------------------------------
    def _scale_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Apply non-linear scaling to raw rewards.

        The scaling method and temperature can be modified via the public
        attributes ``reward_scale_method`` and ``reward_scale_temperature`` at
        runtime (e.g. loaded from YAML config).  This keeps the base class
        generic while permitting task-specific shaping such as mapping SOFA
        ∆score to \[-1, 1\].

        Supported methods
        -----------------
        identity: no scaling (default)
        sigmoid:  r → 2·sigmoid(r / T) - 1
        tanh:     r → tanh(r / T)
        """

        if self.reward_scale_method == 'identity':
            scaled = rewards
        elif self.reward_scale_method == 'sigmoid':
            scaled = 2 * torch.sigmoid(rewards / self.reward_scale_temperature) - 1
        elif self.reward_scale_method == 'tanh':
            scaled = torch.tanh(rewards / self.reward_scale_temperature)
        else:
            raise ValueError(f"Unknown reward_scale_method: {self.reward_scale_method}")

        # Optional exponential time-decay: assumes *rewards* indexed by time
        if self.reward_time_decay > 0:
            # Generate decay factors along the *last* dimension (time)
            # Shape broadcasting handles (B,T) or (T,) inputs.
            t = torch.arange(scaled.shape[-1], device=scaled.device, dtype=scaled.dtype)
            decay = torch.exp(-self.reward_time_decay * t)
            scaled = scaled * decay

        return scaled

    def set_reward_center(self, value: float | torch.Tensor) -> None:
        """(Backward-compat) Manually sets running reward mean µ.

        Prefer letting the agent update µ online, but exposing this hook keeps
        YAML/CLI compatibility.  Subsequent EMA updates will continue from this
        initial value.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        self._reward_center = value

    def get_reward_centering_stats(self) -> Dict[str, float]:
        """Get statistics about reward centering for monitoring and debugging.
        
        Returns:
            Dictionary containing:
            - 'enabled': Whether reward centering is enabled
            - 'center_value': Current reward center value (None if not initialized)
            - 'alpha': EMA update rate
        """
        stats = {
            'enabled': self.reward_centering,
            'alpha': self.reward_center_alpha,
            'center_value': None
        }
        
        if self._reward_center is not None:
            stats['center_value'] = self._reward_center.item() if self._reward_center.numel() == 1 else self._reward_center.mean().item()
            
        return stats
