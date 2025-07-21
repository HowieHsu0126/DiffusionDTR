"""Minimal logging helper wrapping ``logging``.

Provides colourised console output and handy shortcuts for experiment scripts
without introducing heavy dependencies such as *loguru*.
"""
from __future__ import annotations

import logging
import sys
import os
import warnings
from typing import Optional

__all__ = ["get_logger", "suppress_tensorflow_logging"]


class ColourFormatter(logging.Formatter):
    _RESET = "\x1b[0m"
    _COLORS = {
        logging.DEBUG: "\x1b[38;20m",      # grey
        logging.INFO: "\x1b[32m",         # green
        logging.WARNING: "\x1b[33;20m",   # yellow
        logging.ERROR: "\x1b[31;20m",     # red
        logging.CRITICAL: "\x1b[31;1m",   # bold red
    }
    _FMT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        import sys
        # Disable colour codes when output is redirected to a file (i.e. not a TTY)
        use_colour = sys.stdout.isatty() or hasattr(sys, "ps1") or "ipykernel" in sys.modules

        prefix = self._COLORS.get(record.levelno, "") if use_colour else ""
        suffix = self._RESET if use_colour else ""

        log_fmt = prefix + self._FMT + suffix
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def suppress_tensorflow_logging() -> None:
    """Suppress TensorFlow and CUDA logging for cleaner output.
    
    This function configures environment variables and warning filters to suppress
    common TensorFlow and CUDA warnings that clutter the output during training.
    
    Suppressed warnings include:
    • TensorFlow oneDNN optimization warnings
    • CUDA library registration warnings (cuDNN, cuFFT, cuBLAS)
    • TensorFlow device placement warnings
    • Future deprecation warnings from TensorFlow
    """
    # Environment variables for TensorFlow logging suppression
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logging (ERROR only)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Suppress CUDA warnings
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Use async allocator
    
    # Suppress specific TensorFlow warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
    warnings.filterwarnings("ignore", message=".*oneDNN custom operations.*")
    warnings.filterwarnings("ignore", message=".*Unable to register.*factory.*")
    warnings.filterwarnings("ignore", message=".*TF-TRT Warning.*")
    
    # Suppress CUDA-related warnings
    warnings.filterwarnings("ignore", message=".*CUDA.*")
    warnings.filterwarnings("ignore", message=".*cuDNN.*")
    warnings.filterwarnings("ignore", message=".*cuFFT.*")
    warnings.filterwarnings("ignore", message=".*cuBLAS.*")


def get_logger(name: str, level: int = logging.INFO, stream: Optional[object] = None) -> logging.Logger:
    """Returns a colourised console logger with *name*.

    Multiple calls with the same *name* return the same logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Already initialised
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(ColourFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def log_metric(logger: logging.Logger, step: int | str, **metrics):  # noqa: D401
    """Unified metric logging helper.

    Examples
    --------
    >>> log_metric(logger, 10, loss=0.25, reward=38.2)
    """
    for k, v in metrics.items():
        logger.info("%s | %s = %s", step, k, v)
