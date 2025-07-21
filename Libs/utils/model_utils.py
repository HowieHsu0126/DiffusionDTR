"""Utility helpers for model training.

Contains commonly used helper functions shared across agents and training
scripts, keeping the codebase DRY.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Optional, Deque, Tuple, List

import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "apply_gradient_clipping",
    "safe_float",
    "get_autocast_context",
    "init_weights_xavier",
    "ReplayBuffer",
    "OfflineReplayBuffer",
    "cql_loss_vectorised",
    "as_tensor",
]


def apply_gradient_clipping(
    model: torch.nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """Clips gradients of a model in-place to avoid exploding gradients.

    Args:
        model: ``torch.nn.Module`` whose parameters are to be clipped.
        max_norm: Maximum allowed norm of the gradients. If non-positive, the
            function is a no-op.
        norm_type: Type of the used p-norm. See
            ``torch.nn.utils.clip_grad_norm_`` for details.
    
    Returns:
        The total norm of the gradients (before clipping).
    """
    if max_norm is None or max_norm <= 0:
        # Calculate gradient norm even when not clipping
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0
        
        # Calculate the total norm manually
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)

    parameters: Iterable[torch.nn.Parameter] = (
        p for p in model.parameters() if p.grad is not None
    )
    # torch.nn.utils.clip_grad_norm_ returns the total norm of the gradients
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm, norm_type=norm_type)
    return float(total_norm)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Converts *value* to ``float``; falls back to *default* on failure.

    Useful when parsing configuration hyper-parameters that might come as
    ``str``/``None``.  Ensures the training code always receives a valid
    ``float``.

    Args:
        value: Value to convert.
        default: Fallback value if conversion fails.

    Returns:
        A valid float.
    """
    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_item(value) -> float:
    """Safely extract scalar value from tensor or float.
    
    This function handles the common case where code calls .item() on values
    that might be either tensors or regular Python floats, preventing the
    AttributeError: 'float' object has no attribute 'item' error.
    
    Args:
        value: A torch.Tensor, numpy array, or Python numeric value
        
    Returns:
        Python float scalar value
        
    Examples:
        >>> safe_item(torch.tensor(3.14))
        3.14
        >>> safe_item(3.14)
        3.14
        >>> safe_item(np.array(3.14))
        3.14
    """
    if torch.is_tensor(value):
        return float(value.item())
    elif hasattr(value, 'item'):  # numpy arrays
        return float(value.item())
    elif isinstance(value, (int, float, complex)):
        return float(value)
    else:
        # Try to convert to float as fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TypeError(f"Cannot convert {type(value)} to float: {value}")


# ---------------------------------------------------------------------------
#  Mixed-precision helpers
# ---------------------------------------------------------------------------
from contextlib import nullcontext


def get_autocast_context(enabled: bool = True, dtype: torch.dtype | None = None):
    """Returns an autocast/no-op context manager for cleaner training code."""
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


# ---------------------------------------------------------------------------
#  Weight initialisation helpers
# ---------------------------------------------------------------------------


def init_weights_xavier(module: nn.Module) -> None:  # noqa: D401
    """Applies Xavier/Glorot *uniform* initialisation to ``nn.Linear`` layers.

    Bias terms are zero-initialised which is the de-facto standard in most RL
    papers (e.g. DQN, CQL, BCQ).  Modules other than ``nn.Linear`` are left
    unchanged so this function can be passed directly to :pymeth:`nn.Module.apply`.
    """

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
#  Experience Replay Buffer (shared across algorithms)
# ---------------------------------------------------------------------------


class ReplayBuffer:  # noqa: D401
    """Generic experience replay buffer with optional Prioritized ER.

    This implementation unifies the multiple *ReplayBuffer* variants that were
    previously duplicated across *baseline* and *agent* sub-packages.  The
    public API is intentionally kept minimal so that existing code only needs
    to replace the class import without touching call-sites.

    The buffer stores generic *transition tuples* (or any Python objects) and
    returns them in the insertion order unless *prioritized* sampling is
    enabled.

    Args
    ----
    capacity : int
        Maximum number of elements to keep in the buffer.  When the buffer is
        full, the oldest items are overwritten FIFO-style.
    prioritized : bool, default=False
        Enables proportional Prioritized Experience Replay (Schaul et al.
        2016).  When *False* the buffer falls back to uniform random sampling.
    alpha : float, default=0.6
        Exponent that controls how much prioritization is used (α → 0 ≈
        uniform).  Ignored if *prioritized* = False.
    beta : float, default=0.4
        Importance-sampling exponent that corrects the sampling bias.  Ignored
        if *prioritized* = False.
    epsilon : float, default=1e-6
        Small constant added to the TD-error to avoid zero probability.
    """

    def __init__(
        self,
        capacity: int,
        *,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if prioritized and not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1] when prioritized=True")
        if prioritized and not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0,1] when prioritized=True")

        self.capacity: int = int(capacity)
        self.buffer: Deque = deque(maxlen=capacity)
        self.position: int = 0  # compatibility with old code

        # PER related
        self.prioritized: bool = bool(prioritized)
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)
        self.epsilon: float = float(epsilon)
        # Store priorities in a NumPy array for fast vectorised ops
        if self.prioritized:
            self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    #  Public API – insertion & sampling
    # ------------------------------------------------------------------
    def push(self, *transition):  # noqa: D401 – keep legacy name
        """Insert a single *transition* (arbitrary Python objects)."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition if len(transition) > 1 else transition[0]

        if self.prioritized:
            max_prio = self._priorities.max() if self.buffer else 1.0
            self._priorities[self.position] = max_prio

        self.position = (self.position + 1) % self.capacity

    # Legacy alias
    add = push

    def sample(self, batch_size: int, *, replace: bool = False):
        """Return *batch_size* transitions (+ IS weights if PER enabled)."""
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        buffer_len = len(self.buffer)
        if not replace and batch_size > buffer_len:
            raise ValueError("batch_size exceeds current buffer size when replace=False")

        if self.prioritized:
            scaled_prio = self._priorities[:buffer_len] ** self.alpha
            prio_sum = scaled_prio.sum()
            if prio_sum == 0 or np.isnan(prio_sum):
                # Fallback to uniform distribution to avoid NaNs when all
                # priorities are (accidentally) zero.  This situation can
                # happen right after initial filling when TD-errors have
                # not yet been assigned.
                probs = np.full(buffer_len, 1.0 / buffer_len, dtype=np.float32)
            else:
                probs = scaled_prio / prio_sum
            indices = np.random.choice(buffer_len, batch_size, p=probs, replace=replace)
            transitions = [self.buffer[idx] for idx in indices]

            # Importance-sampling weights
            weights = (buffer_len * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # normalise to [0,1]
            return transitions, indices, weights

        # Uniform sampling path
        indices = np.random.choice(buffer_len, batch_size, replace=replace)
        transitions = [self.buffer[idx] for idx in indices]
        return transitions

    # ------------------------------------------------------------------
    #  PER specific helper
    # ------------------------------------------------------------------
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update sample priorities after TD-error computation (PER)."""
        if not self.prioritized:
            raise RuntimeError("Prioritized replay is disabled.")
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")
        for idx, prio in zip(indices, priorities):
            self._priorities[idx] = float(prio + self.epsilon)

    # ------------------------------------------------------------------
    #  Misc helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.buffer)


# ---------------------------------------------------------------------------
#  Offline ReplayBuffer – fixed dataset wrapper (read-only)
# ---------------------------------------------------------------------------


class OfflineReplayBuffer:  # noqa: D401
    """Lightweight wrapper that adapts static offline datasets to RB API.

    The object mimics :class:`ReplayBuffer` but **does not** support ``add``.
    Sampling is performed uniformly at random across the pre-loaded arrays.
    """

    def __init__(self, dataset: Tuple[np.ndarray, ...]):
        if not isinstance(dataset, (list, tuple)) or len(dataset) == 0:
            raise ValueError("dataset must be a non-empty tuple/list of arrays")

        length = len(dataset[0])
        if any(len(arr) != length for arr in dataset):
            raise ValueError("All dataset arrays must share the same first dimension")

        self._dataset = tuple(np.asarray(arr) for arr in dataset)
        self._length = length

    # Read-only; method kept for interface compatibility
    def add(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError("OfflineReplayBuffer is read-only – cannot add data")

    def sample(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        idxs = np.random.choice(self._length, batch_size, replace=True)
        return tuple(arr[idxs] for arr in self._dataset)

    def __len__(self) -> int:  # noqa: D401
        return self._length


# ---------------------------------------------------------------------------
#  Conservative Q-Learning helper – multi-discrete action spaces
# ---------------------------------------------------------------------------


def cql_loss_vectorised(
    q_net: "nn.Module",
    target_net: "nn.Module",
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.99,
    alpha: float = 1.0,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """Vectorised Conservative Q-Learning loss (multi-discrete).

    This function matches the original implementation in
    ``Libs.model.models.baseline.cql_core.cql_loss`` so that both baseline
    scripts **and** high-level *CQLAgent* can share the exact same numerical
    computation, ensuring apples-to-apples comparison during ablations.

    Args
    ----
    q_net / target_net
        Main and target Q-networks (**must** implement ``forward`` that returns
        *List[Tensor]* – one tensor per action branch).
    states / next_states
        Current / next state tensors of shape ``(B, D)``.
    actions
        Integer action indices *per branch* – shape ``(B, n_heads)``.
    rewards / dones
        Reward & terminal flag tensors, shape ``(B, 1)``.
    gamma, alpha, temperature
        Standard CQL hyper-parameters.

    Returns
    -------
    total_loss : torch.Tensor
        Bellman + α·conservative penalty (scalar).
    bellman_loss_value : float
        Logged numeric value of the Bellman component.
    conservative_loss_value : float
        Logged numeric value of the conservative gap component.
    """

    # ---------------- Input validation ----------------
    batch_size = states.size(0)
    if any(t.size(0) != batch_size for t in [actions, rewards, next_states, dones]):
        raise ValueError("All input tensors must share the same batch dimension")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0,1]")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    # ---------------- Forward pass ----------------
    q_list = q_net(states)
    with torch.no_grad():
        target_q_list = target_net(next_states)

    total_bellman = 0.0
    total_conservative = 0.0

    for head_idx, (q_head, target_q_head) in enumerate(zip(q_list, target_q_list)):
        adim = q_head.size(-1)

        # Q(s,a) for dataset actions
        q_sa = q_head.gather(1, actions[:, head_idx : head_idx + 1])

        # Bellman target
        with torch.no_grad():
            q_next_max = target_q_head.max(1, keepdim=True)[0]
            q_target = rewards + gamma * q_next_max * (1 - dones)

        total_bellman = total_bellman + F.mse_loss(q_sa, q_target)

        # Conservative penalty: τ * log Σ_a exp(Q/τ) − Q(s,a_d)
        logsumexp_q = torch.logsumexp(q_head / temperature, dim=1, keepdim=True)
        penalty = (temperature * logsumexp_q - q_sa) / adim
        total_conservative = total_conservative + penalty.mean()

    total_loss = total_bellman + alpha * total_conservative

    return total_loss, float(total_bellman.item()), float(total_conservative.item())


def as_tensor(data: Any, *, dtype: torch.dtype = torch.float32, device: torch.device | str = 'cpu') -> torch.Tensor:
    """Return **torch.Tensor** on requested *device* with desired *dtype*.

    This helper consolidates scattered ``torch.as_tensor`` / ``torch.tensor``
    conversions across the codebase and guarantees that the resulting tensor
    lives on the correct device *and* dtype.

    Examples
    --------
    >>> arr = np.random.rand(3, 4)
    >>> t = as_tensor(arr, dtype=torch.float16, device='cuda:0')
    >>> t.dtype, t.device
    (torch.float16, device(type='cuda', index=0))
    """
    if torch.is_tensor(data):
        # Fast-path：已是 Tensor ➜ 确保 dtype/device 正确
        return data.to(dtype=dtype, device=device)
    return torch.as_tensor(data, dtype=dtype, device=device)


# ------------------------------------------------------------------
#  Sequence helpers
# ------------------------------------------------------------------

def select_last_valid_timestep(obs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Return the feature vector at the last *valid* timestep for each sample.

    This utility centralises the common `(B,T,D) → (B,D)` logic used across
    multiple agents (DQN / CQL / BCQ …) and the ForwardCompatMixin.

    Args:
        obs: Observation tensor of shape ``(B,T,D)`` *or* already flattened
             ``(B,D)``.  For 4-D inputs originating from image pipelines
             ``(B,1,T,D)`` we squeeze the spurious channel dim automatically.
        lengths: Integer tensor ``(B,)`` with the **actual** sequence lengths
             (≥1).  Each element *ℓᵢ* denotes that timesteps ``0‥ℓᵢ−1`` are
             valid / unpadded for the *i*-th sample.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B,D)`` containing the last valid timestep for each
        sequence.  If *obs* is already 2-D it is returned unchanged.
    """
    if obs.dim() == 4 and obs.size(1) == 1:
        # (B,1,T,D) – squeeze dummy channel added by certain image loaders
        obs = obs.squeeze(1)

    if obs.dim() == 2:
        # Already flat – nothing to do
        return obs
    if obs.dim() != 3:
        raise ValueError(f"Expected (B,T,D) or (B,D) tensor, got {obs.shape}")

    if lengths.dim() != 1 or lengths.size(0) != obs.size(0):
        raise ValueError("lengths must be 1-D tensor with same batch size as obs")

    batch_idx = torch.arange(obs.size(0), device=obs.device)
    last_idx = lengths.clamp(min=1) - 1  # guard against zero-length sequences
    return obs[batch_idx, last_idx] 