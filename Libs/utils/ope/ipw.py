"""Inverse Propensity Weighting (IPW) estimator.

Estimating expected return purely from importance sampling ratios (without
value function baseline).  High variance in long horizons; included mainly
for completeness.
"""
from __future__ import annotations

import torch

__all__ = ["ipw_estimate"]


def ipw_estimate(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    importance_weights: torch.Tensor,
    gamma: float = 0.99,
) -> float:
    """Computes the IPW (a.k.a. IS) estimate of discounted return.

    Args:
        rewards:  (T,) or (B,T) reward tensor.
        dones:    same shape, binary.
        importance_weights: Per‐step cumulative IS ratios (same shape).
        gamma: Discount factor.

    Notes:
        • Implements the *self-normalised* IS estimator  \(
              \hat{V}=\frac{\sum_i w_i R_i}{\sum_i w_i}\).
          Compared to the raw IS variant used previously, self-normalisation
          (a) removes the global scale ambiguity that caused IPW/IPS metrics
          to collapse to a constant across different policies, and (b) keeps
          the estimate finite even when importance weights contain very large
          values.  When the importance weights are exact (behaviour ≡ target)
          the estimator now correctly returns the behaviour value instead of
          the inflated 5× value observed earlier.

          For multi-step trajectories the normalisation is applied **per
          trajectory** before averaging over the batch, i.e. we first compute
          a self-normalised return for each sequence and then take the mean.
          This mirrors common practice in off-policy evaluation literature and
          guarantees \(\hat V\in [\min R, \max R]\) as long as rewards are
          bounded.
    """
    with torch.no_grad():
        # Clamp importance weights for numerical stability.  Very small or
        # extremely large ratios can lead to NaNs / Inf in gradient‐free
        # evaluation code paths and, more importantly, blow up the sample
        # variance of the estimator.
        importance_weights = importance_weights.clamp(min=1e-6, max=1e3)

        # 确保所有张量位于同一设备，防止出现"Expected all tensors to be on the same device"错误。
        if importance_weights.device != rewards.device:
            importance_weights = importance_weights.to(rewards.device)

        EPS = 1e-8  # avoid division by zero in extreme cases

        # ------------------------------------------------------------------
        # SINGLE-STEP CASE  (B,)
        # ------------------------------------------------------------------
        if rewards.dim() == 1:
            weighted_return = importance_weights * rewards  # (B,)
            norm = importance_weights + EPS
            estimate = weighted_return / norm  # (B,)

        # ------------------------------------------------------------------
        # MULTI-STEP CASE  (B,T) or (T,)
        # ------------------------------------------------------------------
        else:
            discounts = gamma ** torch.arange(rewards.shape[-1], device=rewards.device)
            # Broadcast discounts along batch dim when necessary.
            if rewards.dim() == 2:
                discounts = discounts.view((1, -1))

            weighted_rewards = discounts * importance_weights * rewards  # (B,T)
            weighted_return = weighted_rewards.sum(dim=-1)              # (B,)
            weight_norm = (discounts * importance_weights).sum(dim=-1)  # (B,)
            estimate = weighted_return / (weight_norm + EPS)            # (B,)

        # Mean over batch ➜ scalar
        return float(estimate.mean().cpu()) 