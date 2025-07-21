"""Pareto‐smoothed importance sampling (PSIS) helpers and PSIS‐WDR estimator.

The implementation follows Vehtari *et al.*, 2017 and Thomas & Brunskill, 2016.
We reuse SciPy for basic statistical functions but avoid heavy dependencies.
"""
from __future__ import annotations

from typing import Tuple

import torch
import numpy as np

# SciPy is optional – fallback to naive clipping if unavailable.
try:
    from scipy import stats  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    stats = None  # type: ignore

__all__ = [
    "psis_smooth_weights",
    "psis_wdr_estimate",
]

def psis_smooth_weights(raw_w: torch.Tensor, threshold_k: float = 0.2) -> torch.Tensor:
    """Applies PSIS smoothing to raw importance weights.

    Args:
        raw_w: Unnormalised importance weights (any shape).
        threshold_k: Tail shape threshold *k* beyond which smoothing is applied.

    Returns:
        Smoothed and **normalised** importance weights of same shape as *raw_w*.
    """
    w = raw_w.double().flatten().cpu().numpy()

    # ------------------------------------------------------------------
    # 1. Stabilise weights (avoid zeros) & rank
    # ------------------------------------------------------------------
    w += 1e-12
    log_w = np.log(w)
    sort_idx = np.argsort(log_w)
    log_w_sorted = log_w[sort_idx]

    # ------------------------------------------------------------------
    # 2. Estimate tail index *k* via Generalised Pareto on last 20 % weights
    # ------------------------------------------------------------------
    n = len(log_w_sorted)
    tail_start = int(0.8 * n)
    tail = np.exp(log_w_sorted[tail_start:])
    tail = tail / tail.mean()  # normalise tail
    tail = np.sort(tail)

    excess = tail - 1.0
    # Method of moments estimate of shape k
    mean_excess = excess.mean()
    var_excess = excess.var(ddof=1)
    k_hat = 0.5 * (mean_excess ** 2) / var_excess if var_excess > 0 else 0.0

    # ------------------------------------------------------------------
    # 3. If k_hat < threshold we keep raw weights; else smooth tail
    # ------------------------------------------------------------------
    if k_hat < threshold_k or stats is None:
        w_smoothed = w
    else:
        # Fit Generalised Pareto to top 20 % log weights
        tail_w = np.exp(log_w_sorted[tail_start:])
        # Fit scale & shape (k)
        params = stats.genpareto.fit(tail_w - tail_w.min())
        k, loc, scale = params  # pylint: disable=unpacking-non-sequence
        # Smooth tail
        smoothed_tail = stats.genpareto.cdf(tail_w - loc, k, scale=scale)
        w_smoothed = w.copy()
        w_smoothed[sort_idx[tail_start:]] = smoothed_tail * tail_w.mean()

    # Normalise
    w_smoothed = w_smoothed / w_smoothed.mean()
    return torch.as_tensor(w_smoothed, dtype=raw_w.dtype, device=raw_w.device).view_as(raw_w)

def psis_wdr_estimate(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    importance_weights: torch.Tensor,
    q_values: torch.Tensor,
    v_values: torch.Tensor,
    gamma: float = 0.99,
) -> float:
    """Weighted Doubly Robust estimate with PSIS‐smoothed weights."""
    iw_smooth = psis_smooth_weights(importance_weights)
    from Libs.utils.ope.wdr import wdr_estimate  # local import to avoid cycle
    return wdr_estimate(rewards, dones, iw_smooth, q_values, v_values, gamma) 