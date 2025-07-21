"""Baseline Branch Value Estimation (BVE) core network.

This module provides a thin wrapper around
:class:`Libs.model.models.agent.bve_qnetwork.BranchValueEstimationQNetwork`
so that the baseline subsystem can instantiate a BVE‐compatible Q‐network
without reaching into the agent sub-package.

Having a dedicated *bve_core* mirrors the structure of other baseline
architectures (``bc_core``, ``dqn_core`` …) and simplifies dynamic imports
inside builder factories or unit tests.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn

from Libs.model.modules.bve_qnetwork import BranchValueEstimationQNetwork

__all__ = ["BVEQNet"]


class BVEQNet(BranchValueEstimationQNetwork):
    """Alias subclass that forwards all arguments.

    The subclassing itself is *functionally identical* to using
    :class:`BranchValueEstimationQNetwork` directly.  The indirection simply
    keeps the public import path under the *baseline* namespace, matching the
    existing convention for other algorithms.
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dims=action_dims,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
        )

    # No extra methods – the parent class already implements everything needed.

    # Keep PyTorch happy with deterministic repr to avoid test flakiness.
    def extra_repr(self) -> str:  # noqa: D401
        return (
            f"state_dim={self.state_dim}, action_dims={self.action_dims}, "
            f"hidden_dim={self.hidden_dim}"
        ) 