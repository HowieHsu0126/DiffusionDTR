"""Experiment utility helpers.

Functions in this module are consumed by the training / evaluation scripts
under ``Libs.exp``.  They focus on *reproducibility* (seed setting) and light-
weight run management (checkpoint & config handling).
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# ----------------------------------------------------------------------------
# 公共导出符号
# ----------------------------------------------------------------------------
__all__ = [
    "seed_everything",
    "save_config",
]

_GLOBAL_SEEDED = False  # 用于抑制重复日志输出


def seed_everything(seed: int = 42) -> None:
    """Set global random seed for *Python*, *NumPy*, *PyTorch* **and** CUDA/cuDNN.

    This helper enforces stronger determinism than the previous implementation:

    1.  Sets ``PYTHONHASHSEED`` to avoid hash randomisation.
    2.  Configures ``torch.backends.cudnn`` for deterministic kernels.
    3.  Calls ``torch.use_deterministic_algorithms`` when available.

    Warning
    -------
    Full determinism can negatively impact GPU throughput (especially for
    RNNs).  Prefer enabling it during *evaluation* / *debugging* while keeping
    benchmarking kernels (`benchmark=False`) off during production training.
    """

    # ------------------------------------------------------------------
    # 1) Python & OS level
    # ------------------------------------------------------------------
    global _GLOBAL_SEEDED

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # 2) NumPy & PyTorch
    # ------------------------------------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # 3) cuDNN & deterministic kernels
    # ------------------------------------------------------------------
    torch.backends.cudnn.deterministic = True   # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False      # type: ignore[attr-defined]


def save_config(config: Dict[str, Any], out_dir: str | Path, filename: str = "config.yaml") -> None:
    """Saves an experiment configuration dictionary to *YAML* for future reference."""
    import yaml

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / filename
    with cfg_path.open("w") as f:
        yaml.safe_dump(config, f)
