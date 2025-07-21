"""Weighted Doubly Robust (WDR) estimator implementation.

WDR combines importance sampling with a fitted Q‐function baseline to reduce
variance while maintaining (under certain assumptions) low bias.

We assume a *per‐step* IS ratio (ρ) is provided; if not, we fall back to plain
Doubly Robust (DR).  For high‐dimensional ICU data, authors may compute ρ via
behaviour policy classifiers – not included here.

Notes:
    • Supports both multi‐step trajectories ``(B,T)`` and 1-step inputs
      ``(B,)``.  For the latter we bypass the discounted summation and
      directly compute the DR correction term, which prevents discount
      vectors of incorrect length and resolves the constant-metric bug
      seen in earlier runs.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

__all__ = ["wdr_estimate"]


def wdr_estimate(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    importance_weights: torch.Tensor,
    q_values: torch.Tensor,
    v_values: torch.Tensor,
    gamma: float = 0.99,
) -> float:
    """Computes the Weighted Doubly Robust estimate of return.

    Args:
        rewards: (T,) or (B,T) tensor of rewards.
        dones:   same shape, binary.
        importance_weights: Per‐step cumulative IS ratios (same shape).
        q_values: Q(s_t,a_t) estimated by FQE network (same shape).
        v_values: V(s_t) = E_a Q(s_t,a) (same shape).
        gamma: Discount factor.
    """
    with torch.no_grad():
        # --- 设备对齐 -------------------------------------------------
        if importance_weights.device != rewards.device:
            importance_weights = importance_weights.to(rewards.device)
            # ``q_values`` 与 ``v_values`` 也需跟随转换以保持一致
            q_values = q_values.to(rewards.device)
            v_values = v_values.to(rewards.device)

        # ------------------------------------------------------------------
        # 形状对齐：若 rewards 为 (B,T) 而其余张量为 (B,)，则在时间维度
        # 上 *复制* 至 (B,T)。这避免了在折现向量维度与校正项维度不符
        # 时出现的 "size mismatch" 运行时错误（常见于 DQN/CQL 等单步
        # 算法）。
        # ------------------------------------------------------------------
        if rewards.dim() == 2:
            # 统一 importance_weights
            if importance_weights.dim() == 1:
                importance_weights = importance_weights.unsqueeze(1).expand_as(rewards)
            # 统一 q_values / v_values
            if q_values.dim() == 1:
                q_values = q_values.unsqueeze(1).expand_as(rewards)
            if v_values.dim() == 1:
                v_values = v_values.unsqueeze(1).expand_as(rewards)
            # dones 可能还是 (B,)；扩充即可
            if dones.dim() == 1:
                dones = dones.unsqueeze(1).expand_as(rewards)

        EPS = 1e-8  # numerical guard to prevent division by zero

        # --------------------------------------------------------------
        # SINGLE-STEP  (B,)
        # --------------------------------------------------------------
        if rewards.dim() == 1:
            dr_term = importance_weights * (q_values - v_values)
            weighted = importance_weights * rewards + (1 - dones) * dr_term  # (B,)
            norm = importance_weights + EPS
            estimate = weighted / norm

        # --------------------------------------------------------------
        # MULTI-STEP  (B,T)  –  Apply discounting first, then normalise
        # --------------------------------------------------------------
        else:
            discounts = gamma ** torch.arange(rewards.shape[-1], device=rewards.device)
            if rewards.dim() == 2:
                discounts = discounts.view((1, -1))
            elif rewards.dim() == 1:
                # Single trajectory vector – ensure same length
                # discounts already (T,), leave as is
                pass

            dr_term = importance_weights * (q_values - v_values)
            weighted = discounts * (importance_weights * rewards + (1 - dones) * dr_term)  # (B,T)

            numerator = weighted.sum(dim=-1)                       # (B,)
            denom = (discounts * importance_weights).sum(dim=-1)   # (B,)
            estimate = numerator / (denom + EPS)

        # -------- 数值稳定处理 --------
        estimate = torch.nan_to_num(estimate, nan=0.0, posinf=1e8, neginf=-1e8)

        return float(estimate.mean().cpu()) 