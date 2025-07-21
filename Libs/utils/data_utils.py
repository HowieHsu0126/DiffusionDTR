"""Data utility helpers for dataset preparation and loading.

This module intentionally keeps a lightweight surface: only minimal helper
functions are provided so as not to duplicate logic already handled by
PyTorch, PyTorch‐Geometric or custom data classes elsewhere in the codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "seed_worker",
    "build_dataloader",
]


def seed_worker(worker_id: int) -> None:
    """Sets numpy / Python random seeds for *DataLoader* worker processes."""
    import random, numpy as np, torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Constructs a PyTorch ``DataLoader`` with sane defaults.

    Args:
        dataset: Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle each epoch.
        num_workers: Number of subprocesses.
        pin_memory: Whether to pin memory when using CUDA.

    Returns:
        Configured ``torch.utils.data.DataLoader``.
    """
    generator = torch.Generator()
    generator.manual_seed(42)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

# -----------------------------------------------------------------------------
#  Reward shaping helpers (SOFA + lactate)
# -----------------------------------------------------------------------------

# These utilities were migrated from *reward_utils.py* so that all lightweight
# helpers live in a single module, reducing import overhead and avoiding
# duplicated "magic numbers" across the codebase.

_C0 = -0.025  # 1[ΔSOFA = 0 & SOFA>0]
_C1 = -0.125  # ΔSOFA (will be normalized by SOFA range)
_C2 = -2.0    # tanh(Δlactate)
_R_ALIVE = 1.0   # Changed from 15.0 to 1.0 to match paper standard
_R_DEAD = -1.0   # Changed from -15.0 to -1.0 to match paper standard

# SOFA score normalization (SOFA ranges from 0-24: 6 organ systems × 4 points each)
_SOFA_MIN = 0
_SOFA_MAX = 24
_SOFA_RANGE = _SOFA_MAX - _SOFA_MIN

__all__.extend([
    "compute_step_reward",
    "compute_sequence_rewards",
    "_SOFA_RANGE",
    "_R_ALIVE", 
    "_R_DEAD",
])

import torch  # noqa: E402 – placed after __all__ extension


def compute_step_reward(
    kappa_t: torch.Tensor,
    kappa_next: torch.Tensor,
    lact_t: torch.Tensor,
    lact_next: torch.Tensor,
    *,
    is_terminal: torch.Tensor | bool = False,
    outcome_alive: torch.Tensor | bool | None = None,
    c0: float = _C0,
    c1: float = _C1,
    c2: float = _C2,
    r_alive: float = _R_ALIVE,
    r_dead: float = _R_DEAD,
    sofa_range: float = _SOFA_RANGE,
) -> torch.Tensor:
    """SOFA + lactate reward for a single transition (vectorised).

    Parameters follow Raghu et al. formulation but with normalized SOFA changes
    to match paper specification. Constants can be overridden for ablation.
    
    Reward components:
    - Stability bonus: c0 * 1[ΔSOFA = 0 & SOFA > 0]
    - Normalized SOFA change: c1 * (ΔSOFA / sofa_range)  # Now normalized like paper
    - Lactate change: c2 * tanh(Δlactate)
    - Terminal outcome: ±1 for survival/death (matching paper)
    """
    kappa_t = kappa_t.float(); kappa_next = kappa_next.float()
    lact_t = lact_t.float(); lact_next = lact_next.float()

    delta_kappa = kappa_next - kappa_t
    delta_lact = lact_next - lact_t
    term_stability = (kappa_next == kappa_t) & (kappa_next > 0)

    # Normalize SOFA change by total range (following paper approach)
    normalized_delta_sofa = delta_kappa / sofa_range
    
    r = c0 * term_stability.float() + c1 * normalized_delta_sofa + c2 * torch.tanh(delta_lact)

    # Terminal outcome bonus/penalty
    if isinstance(is_terminal, bool):
        term_mask = torch.full_like(r, float(is_terminal))
    else:
        term_mask = is_terminal.float()

    if outcome_alive is None:
        outcome_alive = torch.zeros_like(r, dtype=torch.bool)
    elif isinstance(outcome_alive, bool):
        outcome_alive = torch.full_like(r, outcome_alive, dtype=torch.bool)

    r_out = torch.where(outcome_alive, r_alive, r_dead)
    return r + term_mask * r_out


def compute_sequence_rewards(
    sofa: torch.Tensor,
    lactate: torch.Tensor,
    outcome_alive: torch.Tensor,
    c0: float = _C0,
    c1: float = _C1,
    c2: float = _C2,
    r_alive: float = _R_ALIVE,
    r_dead: float = _R_DEAD,
    sofa_range: float = _SOFA_RANGE,
) -> torch.Tensor:
    """Vectorised reward for full trajectory tensor ``(B, T)``.
    
    Uses normalized SOFA changes to match paper specification:
    - Intermediate rewards use normalized ΔSOFA/sofa_range
    - Terminal rewards use ±1 for survival/death outcomes
    """
    B, T = sofa.shape
    r = torch.zeros_like(sofa, dtype=torch.float)
    if T < 2:
        # Degenerate case – only outcome reward
        r[:, 0] = torch.where(outcome_alive, r_alive, r_dead)
        return r

    for t in range(T - 1):
        r[:, t] = compute_step_reward(
            sofa[:, t], sofa[:, t + 1], lactate[:, t], lactate[:, t + 1],
            is_terminal=False, outcome_alive=False,
            c0=c0, c1=c1, c2=c2, r_alive=r_alive, r_dead=r_dead,
            sofa_range=sofa_range,
        )
    r[:, -1] = torch.where(outcome_alive, r_alive, r_dead)
    return r 