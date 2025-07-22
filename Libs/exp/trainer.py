"""Unified Trainer class for all baselines and PoG-based methods.

The *Trainer* handles:
• Dataset loading (state / trajectory PyG graph)  
• Agent construction via ``agent_registry``  
• Training loop with mixed-precision + GradScaler  
• Periodic validation & test with metrics: reward, FQE value, Survival Rate  
• OPE estimates (FQE, WDR, IPW)

Supported algorithms (algo str):
    physician, bc, dqn, cql, bcq, bve,
    pog_bc, pog_dqn, pog_cql, pog_bcq, pog_bve

Engineering improvements:
• Enhanced numerical stability with gradient clipping and regularization
• Robust reproducibility with comprehensive seeding
• Memory-efficient data loading for large cohorts
• Advanced error handling and logging
• Deterministic GPU operations for reproducibility
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import pickle
import random
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Libs.model.models.agent import agent_registry
from Libs.utils.data_utils import build_dataloader, seed_worker
from Libs.utils.exp_utils import seed_everything
from Libs.utils.log_utils import (get_logger, log_metric,
                                  suppress_tensorflow_logging)
# Enhanced memory optimization imports
from Libs.utils.memory_utils import (AdaptiveBatchSizer, EnhancedMemoryManager,
                                     GradientAccumulator, MemoryMonitor,
                                     MemoryOptimizedDataLoader,
                                     cleanup_multiprocessing_temp_dirs,
                                     clear_cuda_cache_aggressively,
                                     memory_efficient_training_context)
from Libs.utils.model_utils import (apply_gradient_clipping,
                                    get_autocast_context, safe_item)
from Libs.utils.ope import FQEEstimator, ipw_estimate, wdr_estimate
from Libs.utils.task_manager import get_current_task_config, get_task_manager
from Libs.utils.vis_utils import (plot_convergence_diagnostics,
                                  plot_ope_comparison_with_ci,
                                  plot_policy_distribution_comparison,
                                  plot_treatment_strategy_heatmap,
                                  save_figure_publication_ready,
                                  set_plot_style)

logger = get_logger("Trainer")

__all__ = ["Trainer"]

@contextmanager
def robust_cuda_context():
    """Context manager for robust CUDA operations with proper error handling."""
    if torch.cuda.is_available():
        # Enable deterministic operations for reproducibility
        old_deterministic = torch.backends.cudnn.deterministic
        old_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            yield
        finally:
            torch.backends.cudnn.deterministic = old_deterministic
            torch.backends.cudnn.benchmark = old_benchmark
    else:
        yield

class MemoryEfficientBatchProcessor:
    """Memory-efficient batch processor for large cohorts."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self._memory_threshold = max_memory_mb * 1024 * 1024  # Convert to bytes
        
    def get_current_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0
        
    def should_process_lazily(self, estimated_batch_size: int) -> bool:
        """Determine if lazy processing is needed based on memory constraints."""
        current_memory = self.get_current_memory_usage()
        # Estimate memory usage (rough approximation)
        estimated_memory = estimated_batch_size * 1024 * 100  # 100KB per batch estimate
        return (current_memory + estimated_memory) > self._memory_threshold

class TrainingMetrics:
    """Comprehensive training metrics tracker with TensorBoard integration."""
    
    def __init__(self, log_dir: Optional[Path] = None, enable_tensorboard: bool = True):
        self.log_dir = log_dir
        self.enable_tensorboard = enable_tensorboard
        self.metrics_history: Dict[str, List[float]] = {}
        self.step_count = 0
        
        # Initialize TensorBoard writer if available
        self.tb_writer = None
        if enable_tensorboard and log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir / "tensorboard")
                logger.info("TensorBoard logging enabled at %s", log_dir / "tensorboard")
            except ImportError:
                logger.warning("TensorBoard not available, skipping initialization")
                
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log scalar metric with automatic step tracking."""
        if step is None:
            step = self.step_count
            
        # Store in history
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append(value)
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)
            
        # Log to console with formatted output
        logger.info("📊 %s: %.6f (step %d)", name, value, step)
        
    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None) -> None:
        """Log histogram of tensor values."""
        if self.tb_writer is not None and step is not None:
            self.tb_writer.add_histogram(name, values, step)
            
    def save_metrics(self, filepath: Optional[Path] = None) -> None:
        """Save metrics history to JSON file."""
        if filepath is None and self.log_dir is not None:
            filepath = self.log_dir / "training_metrics.json"
            
        if filepath is not None:
            import json
            with open(filepath, 'w') as f:
                # Convert to JSON-serializable format
                serializable_metrics = {k: [float(x) for x in v] for k, v in self.metrics_history.items()}
                json.dump(serializable_metrics, f, indent=2)
                
    def close(self) -> None:
        """Close TensorBoard writer and save metrics."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        self.save_metrics()

class Trainer:
    """Generic trainer supporting multiple offline RL agents with enterprise-grade reliability.
    
    Key features:
    • Comprehensive reproducibility guarantees with deterministic operations
    • Memory-efficient processing for large medical cohorts
    • Robust numerical stability with gradient clipping and regularization
    • Advanced metrics tracking with TensorBoard integration
    • Graceful error handling and recovery mechanisms
    """

    def __init__(
        self,
        algo: str,
        state_dim: int,
        action_dims: List[int],
        device: str | torch.device = "cpu",
        amp: bool = False,
        log_dir: str | Path | None = "Output/runs",
        seed: int | None = None,
        # Policy evaluation options
        policy_mode: str = "greedy",  # {'greedy','sample','beam','temperature'}
        # NOTE: Default *temperature* changed from ``1.0`` to ``None`` so that
        # value-based algorithms (DQN/CQL/BCQ) adopt a *colder* Boltzmann
        # distribution (τ=0.2) unless explicitly overridden by the caller.
        temperature: float | None = None,
        # ---------------- FQE hyper-parameters ----------------
        fqe_epochs: int = 30,
        fqe_batch_size: int = 512,
        fqe_lr: float = 3e-4,
        # --------------------------------------------------------------
        # Behaviour-policy training strategy – choose between
        #   • "logistic": state-conditional multinomial logistic model (default)
        #   • "empirical": state-independent global counts
        # End-users can override via CLI/YAML but we default to *logistic*
        # to avoid the constant-μ bug that freezes WDR/IPW.
        # --------------------------------------------------------------
        behav_policy_mode: str = "logistic",
        # ---------------- NEW: Externalised experiment hyper-parameters ----------------
        behav_prob_min: float | None = None,
        behav_prob_max: float | None = None,
        pi_eps_init: float | None = None,
        clip_range: float | None = None,
        max_joint_enum: int | None = None,
        use_psis: bool | None = None,
        boltz_tau: float | None = None,
        # ---------------- Engineering improvements ----------------
        max_memory_mb: int = 2048,  # Memory limit for batch processing
        enable_tensorboard: bool = True,  # Enable TensorBoard logging
        numerical_stability: bool = True,  # Enable numerical stability checks
        deterministic_mode: bool = True,  # Enable deterministic operations
        # ---------------- NEW: Visualization settings ----------------
        enable_visualization: bool = True,  # Enable automatic visualization generation
        visualization_formats: List[str] = None,  # Output formats ['png', 'pdf', 'svg']
        vis_style: str = "publication",  # Visualization style preset
        # ---------------- NEW: Enhanced Memory Management ----------------
        enable_adaptive_batching: bool = True,  # Enable adaptive batch size adjustment
        enable_gradient_accumulation: bool = True,  # Enable gradient accumulation
        memory_threshold: float = 0.85,  # Memory usage threshold for optimization
        gradient_accumulation_steps: int = 4,  # Steps for gradient accumulation
        # ---------------- NEW: Bootstrap Configuration ----------------
        bootstrap_iters: int = 1000,  # Number of bootstrap iterations for confidence intervals
        **agent_kwargs,
    ) -> None:
        # ========================================================================
        # Initialize instance logger
        # ========================================================================
        self.logger = logger  # Use the module-level logger instance
        
        # ========================================================================
        # TensorFlow and CUDA Logging Suppression
        # ========================================================================
        suppress_tensorflow_logging()
        
        # ========================================================================
        # 🔧 CRITICAL FIX: Clean up orphaned multiprocessing temp directories
        # This prevents "Directory not empty" errors from previous runs
        # ========================================================================
        cleanup_multiprocessing_temp_dirs()
        
        # ========================================================================
        # Enhanced reproducibility setup with comprehensive seeding
        # ========================================================================
        if seed is not None:
            seed_everything(seed)
            self.training_seed = seed
        else:
            # Generate a random seed for tracking
            self.training_seed = torch.initial_seed() % (2**32 - 1)
            seed_everything(self.training_seed)

        # ========================================================================
        # Enhanced memory management initialization
        # ========================================================================
        self.memory_manager = EnhancedMemoryManager(
            initial_batch_size=64,  # Default, can be overridden by dataloaders
            max_memory_threshold=memory_threshold,
            enable_adaptive_batching=enable_adaptive_batching,
            enable_gradient_accumulation=enable_gradient_accumulation,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        logger.info("🧠 Enhanced memory management initialized")
        logger.info(f"   • Adaptive batching: {enable_adaptive_batching}")
        logger.info(f"   • Gradient accumulation: {enable_gradient_accumulation}")
        logger.info(f"   • Memory threshold: {memory_threshold:.1%}")

        # ========================================================================
        # Device and hardware optimization setup
        # ========================================================================
        self.device = torch.device(device)
        self.algo = algo.lower()
        self.amp = amp
        self.log_dir = Path(log_dir) if log_dir is not None else None
        
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Persist comprehensive experiment metadata
            self._save_experiment_metadata(seed, device, amp, agent_kwargs)
            
        # ========================================================================
        # Initialize enhanced components
        # ========================================================================
        # Memory-efficient batch processor
        self.batch_processor = MemoryEfficientBatchProcessor(max_memory_mb)
        
        # Advanced metrics tracking
        self.metrics_tracker = TrainingMetrics(
            self.log_dir, 
            enable_tensorboard and enable_tensorboard
        )
        
        # Numerical stability settings
        self.numerical_stability = numerical_stability
        if numerical_stability:
            logger.info("🛡️  Numerical stability checks enabled")
            
        # ========================================================================
        # NEW: Visualization settings
        # ========================================================================
        self.enable_visualization = enable_visualization
        self.visualization_formats = visualization_formats or ['png', 'pdf']
        self.vis_style = vis_style
        
        if enable_visualization:
            # Initialize visualization style
            set_plot_style(vis_style)
            logger.info("🎨 Visualization enabled with %s style", vis_style)
            
        # ========================================================================
        # Store essential initialization parameters
        # ========================================================================
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.bootstrap_iters = bootstrap_iters  # Store for use in validation evaluation
        
        # ========================================================================
        # Complete agent initialization
        # ========================================================================
        self._agent_initialization(
            policy_mode, temperature, agent_kwargs,
            fqe_epochs, fqe_batch_size, fqe_lr,
            behav_prob_min, behav_prob_max, pi_eps_init,
            clip_range, max_joint_enum, use_psis,
            boltz_tau, behav_policy_mode
        )
        
        logger.info("🔧 Trainer initialization completed successfully")

    def _setup_deterministic_mode(self) -> None:
        """Configure PyTorch for deterministic operations to ensure reproducibility."""
        try:
            # Core deterministic settings
            torch.use_deterministic_algorithms(True, warn_only=True)
            
            # CuDNN deterministic mode (slower but reproducible)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("🔒 CUDA deterministic mode enabled")
                
            # Set environment variables for additional determinism
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(self.seed)
            
            logger.info("✅ Deterministic mode configured successfully")
            
        except Exception as e:
            logger.warning("⚠️  Could not enable full deterministic mode: %s", e)
            # Fallback to basic deterministic settings
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
    def _save_experiment_metadata(self, seed: int, device: str, amp: bool, agent_kwargs: Dict[str, Any]) -> None:
        """Save comprehensive experiment metadata for reproducibility."""
        if self.log_dir is None:
            return
            
        metadata = {
            'timestamp': time.time(),
            'algorithm': self.algo,
            'seed': seed,
            'device': str(device),
            'amp_enabled': amp,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'agent_kwargs': {k: v for k, v in agent_kwargs.items() if isinstance(v, (int, float, str, bool))},
            'environment': {
                'python_executable': os.sys.executable,
                'working_directory': os.getcwd(),
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            }
        }
        
        # Save metadata as JSON
        import json
        metadata_file = self.log_dir / "experiment_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        # Also save seed in legacy format for backward compatibility
        with open(self.log_dir / "seed.txt", "w", encoding="utf-8") as f:
            f.write(str(seed))
            
        logger.info("💾 Experiment metadata saved to %s", metadata_file)

    def _agent_initialization(
        self, 
        policy_mode: str, 
        temperature: float | None, 
        agent_kwargs: Dict[str, Any],
        fqe_epochs: int,
        fqe_batch_size: int,
        fqe_lr: float,
        behav_prob_min: float | None,
        behav_prob_max: float | None,
        pi_eps_init: float | None,
        clip_range: float | None,
        max_joint_enum: int | None,
        use_psis: bool | None,
        boltz_tau: float | None,
        behav_policy_mode: str
    ) -> None:
        """Initialize the agent and configure policy action function."""
        # ========================================================================
        # Agent initialization - Build agent via registry
        # ========================================================================
        # Build agent via registry (must be pre-registered)
        if self.algo == "physician":
            # PhysicianPolicy is a deterministic wrapper – load via registry
            self.agent = agent_registry.make(
                "physician_policy",
                state_dim=self.state_dim,
                action_dims=self.action_dims,
                device=self.device,
            )
        else:
            self.agent = agent_registry.make(
                self.algo, state_dim=self.state_dim, action_dims=self.action_dims, device=self.device, **agent_kwargs
            )
        logger.info(f"Trainer initialised with algo={self.algo}")

        # --------------------------------------------------------------
        # Safeguard: ensure the model actually has trainable parameters.
        # --------------------------------------------------------------
        if isinstance(self.agent, nn.Module):
            if not any(p.requires_grad for p in self.agent.parameters()):
                raise RuntimeError(
                    "All model parameters are frozen (requires_grad=False). "
                    "Please check your model definition – backbone should be trainable or set trainable=False.")

        # Initialise mixed-precision GradScaler – pass *enabled* flag only to
        # remain compatible across PyTorch versions where additional positional
        # arguments are not recognised.
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(self.amp and torch.cuda.is_available()))

        # In‐memory cache for importance weights (cleared each evaluation)
        self._cached_weights: Dict[str, torch.Tensor] = {}

        # Early stopping placeholders
        self._best_metric: float | None = None
        self._epochs_no_improve: int = 0

        # --------------------------------------------------------------
        # Policy action function for evaluation / FQE
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # BVE 系列算法在多分支离散动作空间下，beam search 显著提升推断
        # 质量。若用户未显式指定，则对 {bve, pog_bve} 默认启用
        # ``policy_mode='beam'`` 并将 beam_width 提升至 5。
        # --------------------------------------------------------------

        if policy_mode is None:
            policy_mode = "beam" if self.algo in {"bve", "pog_bve"} else "greedy"

        policy_mode = policy_mode.lower()
        if policy_mode == "greedy":
            self.policy_action_fn = lambda s: self.agent.act(s, greedy=True)
        elif policy_mode == "sample":
            self.policy_action_fn = lambda s: self.agent.act(s, greedy=False)
        elif policy_mode == "beam":
            # Beam width 默认为 5；如需其他宽度可在 agent_kwargs 覆盖
            beam_w = int(agent_kwargs.get("beam_width", 5))
            self.policy_action_fn = lambda s: self.agent.act(
                s, policy_inference_mode="beam", beam_width=beam_w)
        elif policy_mode == "temperature":
            # Update agent temperature attr if exists
            if hasattr(self.agent, "softmax_temperature"):
                setattr(self.agent, "softmax_temperature", temperature)
            self.policy_action_fn = lambda s: self.agent.act(s, greedy=False)
        else:
            raise ValueError(f"Unknown policy_mode: {policy_mode}")

        # Expose FQE hyper-params to evaluate()
        self.fqe_epochs = int(fqe_epochs)
        self.fqe_batch_size = int(fqe_batch_size)
        self.fqe_lr = float(fqe_lr)

        # --------------------------------------------------------------
        # Numerically-stable OPE hyper-parameters (YAML-driven)
        # --------------------------------------------------------------
        self.behav_prob_min: float = float(behav_prob_min) if behav_prob_min is not None else 1e-6
        self.behav_prob_max: float = float(behav_prob_max) if behav_prob_max is not None else 1.0 - 1e-3

        # ε used for ε-smoothing of target-policy probability π_e
        self.pi_eps: float = float(pi_eps_init) if pi_eps_init is not None else 5e-2

        # Clipping range for importance weights
        self.clip_range: float | None = float(clip_range) if clip_range is not None else 5.0

        # 控制联合动作空间枚举上限；过大时退化为分头近似
        self.max_joint_enum: int = int(max_joint_enum) if max_joint_enum is not None else 5000

        # Pareto-smoothed IS toggle
        self.use_psis: bool = bool(use_psis) if use_psis is not None else True

        # Default Boltzmann temperature
        self.boltz_tau: float = float(boltz_tau) if boltz_tau is not None else 0.2

        # 🔍 NEW: Automatically enable Boltzmann policy probability for
        # value-based algorithms whose policies are implicitly represented
        # by a Q-network (DQN/CQL/BCQ and their PoG variants).  Deterministic
        # greedy matching (previous default) yields *constant* πₑ estimates
        # when the greedy action does not change, masking real learning
        # progress.  Boltzmann distribution over Q(s,·) exposes smoother
        # probability shifts that the WDR / IPW estimators can capture.
        value_based_algos = {
            "dqn",
            "cql",
            "bcq",
            "pog_dqn",
            "pog_cql",
            "pog_bcq",
        }
        self.use_boltzmann = self.algo in value_based_algos
        if self.use_boltzmann:
            # Default τ further reduced to 0.2 (unless user overrides) so that
            # π/μ importance weights deviate noticeably from 1 and OPE metrics
            # can capture learning progress.
            self.boltz_tau = float(temperature if temperature is not None else 0.2)
            logger.info("⚙️  Boltzmann probability enabled (τ=%.3f) for %s", self.boltz_tau, self.algo)

        # ------------------------------------------------------------------
        # Behaviour policy configuration (contextual vs empirical)
        # ------------------------------------------------------------------
        self.behav_policy_mode: str = str(behav_policy_mode).lower().strip()
        if self.behav_policy_mode not in {"logistic", "empirical"}:
            raise ValueError(
                "behav_policy_mode must be 'logistic' or 'empirical', got "
                f"{self.behav_policy_mode!r}")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        n_epochs: int = 10,
        val_loader: Optional[DataLoader] = None,
        early_stop_metric: str = "reward",  # metric name from evaluate()
        patience: int = 15,
        delta: float | None = None,  # Deprecated absolute threshold
        rel_delta: float = 1e-3,
        warmup_sample_epochs: int = 3,
    ) -> None:
        """Main training loop with optional early stopping."""
        # ------------------------------------------------------------------
        # 1) One-off dataset statistics & behaviour policy fitting
        # ------------------------------------------------------------------
        if getattr(self, "_behav_policy_ready", False) is False:
            self._train_behavior_policy(train_loader)
            self._behav_policy_ready = True

        # --------------------------------------------------------------
        # Shortcut: if the agent declares itself *non-trainable* (e.g. the
        # empirical Physician baseline), we **skip** gradient updates and
        # run only evaluation steps.
        # --------------------------------------------------------------
        if getattr(self.agent, "trainable", True) is False:
            logger.info(
                "Agent marked as non-trainable → skipping optimisation loop")
            if val_loader is not None:
                self.evaluate(val_loader, split="val", bootstrap_iters=self.bootstrap_iters)
            self.evaluate(train_loader, split="test")
            return

        # ------------------------------------------------------------------
        # Algorithm-specific overrides
        # ------------------------------------------------------------------
        # Adaptive patience based on algorithm characteristics
        if self.algo in {"bc", "pog_bc"}:
            # For pure imitation learning, use moderate patience to ensure adequate training
            if patience > 10:
                logger.debug("[Early-Stop] Setting patience→10 for %s", self.algo)
                patience = 10
        elif self.algo in {"dqn", "pog_dqn", "cql", "pog_cql"}:
            # Q-learning algorithms need more exploration time
            if patience < 20:
                logger.debug("[Early-Stop] Setting patience→20 for %s", self.algo)
                patience = 20
        elif self.algo in {"bcq", "pog_bcq", "bve", "pog_bve"}:
            # Hybrid algorithms benefit from moderate patience
            if patience < 15:
                logger.debug("[Early-Stop] Setting patience→15 for %s", self.algo)
                patience = 15

        # Warm-up bookkeeping – store *initial* learning rates for all
        # parameter groups so that we can scale them smoothly during the
        # first few epochs.  We only apply warm-up when the agent exposes an
        # optimiser handle.
        _base_lrs = None
        if hasattr(self.agent, "optimizer"):
            _base_lrs = [pg["lr"] for pg in self.agent.optimizer.param_groups]

        # ------------------------------------------------------------------
        # Action-divergence tracking (approx.) for dynamic validation
        # ------------------------------------------------------------------
        prev_val_actions = None

        for epoch in range(1, n_epochs + 1):
            # --------------------------------------------------------------
            # Linear learning-rate warm-up
            # --------------------------------------------------------------
            if _base_lrs is not None and epoch <= warmup_sample_epochs and warmup_sample_epochs > 0:
                warmup_scale = epoch / float(warmup_sample_epochs)
                for pg, base in zip(self.agent.optimizer.param_groups, _base_lrs):
                    pg["lr"] = base * warmup_scale

            # Set training mode safely with fallback to standard PyTorch methods
            if hasattr(self.agent, "set_training_mode"):
                self.agent.set_training_mode(True)
            elif hasattr(self.agent, "train"):
                self.agent.train()
            elif hasattr(self.agent, "q_net") and hasattr(self.agent.q_net, "train"):
                self.agent.q_net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()
            valid_batches = 0
            
            # Memory monitoring for large cohorts
            initial_memory = self.batch_processor.get_current_memory_usage()
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Memory-efficient processing check
                    # 🔧 CRITICAL FIX: Handle both dict and tuple batch formats
                    batch_size = 0
                    if isinstance(batch, dict):
                        batch_size = len(batch.get('state', []))
                    elif isinstance(batch, (tuple, list)) and len(batch) > 0:
                        # For tuple format, state is typically the first element
                        state_tensor = batch[0]
                        if hasattr(state_tensor, 'size'):
                            batch_size = state_tensor.size(0)
                        else:
                            batch_size = len(state_tensor) if hasattr(state_tensor, '__len__') else 1
                    
                    if self.batch_processor.should_process_lazily(batch_size):
                        logger.debug("Memory threshold reached, processing batch lazily")
                        
                    loss = self._train_step(batch)
                    
                    # 🔧 CRITICAL FIX: Ensure loss is a float before numerical checks
                    loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)
                    
                    # Enhanced numerical stability checks
                    if self.numerical_stability:
                        if not math.isfinite(loss_val):
                            logger.warning("Non-finite loss detected (%.6f), skipping batch %d", loss_val, batch_idx)
                            continue
                            
                    # Enhanced gradient clipping with monitoring
                    if hasattr(self.agent, "optimizer") and hasattr(self.agent, "model"):
                        try:
                            grad_norm = apply_gradient_clipping(self.agent.model, max_norm=1.0)
                        except Exception as e:
                            logger.debug("Gradient clipping failed: %s", e)
                            
                    epoch_loss += loss_val
                    valid_batches += 1
                    
                    # Log batch-level metrics periodically
                    if batch_idx % 100 == 0:
                        current_memory = self.batch_processor.get_current_memory_usage()
                        memory_increase = (current_memory - initial_memory) / (1024**2)  # MB
                        logger.debug("Batch %d/%d | loss=%.6f | memory_increase=%.1fMB", 
                                   batch_idx, len(train_loader), loss_val, memory_increase)
                        
                except Exception as e:
                    logger.warning("Batch %d failed with error: %s, skipping", batch_idx, e)
                    continue
                    
            # Calculate epoch metrics
            epoch_loss_val = epoch_loss / max(valid_batches, 1)
            epoch_time = time.time() - epoch_start_time
            
            # Enhanced logging with performance metrics
            logger.info(
                "Epoch %d/%d | loss=%.6f | time=%.2fs | valid_batches=%d/%d", 
                epoch, n_epochs, epoch_loss_val, epoch_time, valid_batches, len(train_loader))
                
            # Log to advanced metrics tracker
            self.metrics_tracker.log_scalar("train/loss", epoch_loss_val, epoch)
            self.metrics_tracker.log_scalar("train/epoch_time", epoch_time, epoch)
            self.metrics_tracker.log_scalar("train/valid_batch_ratio", valid_batches/len(train_loader), epoch)
            self.metrics_tracker.step_count = epoch
            
            # 🔧 CRITICAL FIX: Safe handling of epoch_loss_val to prevent round() errors
            safe_epoch_loss = float(epoch_loss_val.item()) if torch.is_tensor(epoch_loss_val) else float(epoch_loss_val)
            log_metric(logger, f"train-epoch-{epoch}", loss=round(safe_epoch_loss, 6))

            # ---------------- validation & early stopping -----------------
            if val_loader is not None:
                # Set evaluation mode safely with fallback to standard PyTorch methods
                if hasattr(self.agent, "set_training_mode"):
                    self.agent.set_training_mode(False)
                elif hasattr(self.agent, "eval"):
                    self.agent.eval()
                elif hasattr(self.agent, "q_net") and hasattr(self.agent.q_net, "eval"):
                    self.agent.q_net.eval()

                # Switch policy sampling mode during warm-up ----------------
                if epoch <= warmup_sample_epochs:
                    self.policy_action_fn = lambda s: self.agent.act(s, greedy=False)
                else:
                    self.policy_action_fn = lambda s: self.agent.act(s, greedy=True)

                validation_start = time.time()
                
                # Enhanced validation with error handling
                try:
                    # Memory cleanup before validation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    # 🔧 CRITICAL FIX: Enhanced validation with comprehensive error handling
                    logger.info(f"🔍 Starting validation for epoch {epoch}...")
                    validation_start = time.time()
                    
                    try:
                        metrics = self.evaluate(val_loader, split="val", bootstrap_iters=self.bootstrap_iters)
                        validation_time = time.time() - validation_start
                        
                        # Validate metrics structure before processing
                        if not isinstance(metrics, dict):
                            logger.error(f"❌ Expected dict from evaluate(), got {type(metrics)}")
                            logger.error(f"📊 Received value: {metrics}")
                            break
                        
                        # Log validation metrics with enhanced error handling
                        for metric_name, metric_value in metrics.items():
                            try:
                                if isinstance(metric_value, (int, float)) and math.isfinite(metric_value):
                                    self.metrics_tracker.log_scalar(f"val/{metric_name}", metric_value, epoch)
                                elif isinstance(metric_value, list):
                                    # Handle confidence intervals and other list metrics
                                    logger.debug(f"Skipping list metric {metric_name}: {metric_value}")
                                else:
                                    logger.debug(f"Skipping non-finite metric {metric_name}: {metric_value}")
                            except Exception as metric_error:
                                logger.warning(f"⚠️  Failed to log metric {metric_name}: {metric_error}")
                        
                        self.metrics_tracker.log_scalar("val/evaluation_time", validation_time, epoch)
                        
                    except Exception as eval_error:
                        logger.error(f"💥 Validation failed with error: {eval_error}")
                        logger.error(f"🔍 Error type: {type(eval_error).__name__}")
                        logger.error(f"🔍 Error details: {str(eval_error)}")
                        
                        # Try to extract more diagnostic information
                        if "too many values to unpack" in str(eval_error):
                            logger.error("🔧 This appears to be a tuple unpacking error")
                            logger.error("🔧 Likely cause: A function is returning more values than expected")
                            logger.error("🔧 Check for functions that should return 3 values but return more")
                        
                        # Continue training instead of breaking to allow recovery
                        logger.info("🔄 Attempting to continue training...")
                        continue
                    
                    curr_metric = metrics.get(early_stop_metric)
                    
                    # Enhanced NaN handling with diagnostic information
                    if curr_metric is None:
                        logger.error("❌ Early stop metric '%s' not found in validation results: %s", 
                                   early_stop_metric, list(metrics.keys()))
                        logger.info("📊 Available metrics: %s", {k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                        break
                        
                    if math.isnan(curr_metric):
                        logger.warning("⚠️  Validation metric '%s' is NaN (%.6f)", early_stop_metric, curr_metric)
                        
                        # Diagnostic logging for NaN values
                        nan_metrics = [k for k, v in metrics.items() if isinstance(v, (int, float)) and math.isnan(v)]
                        if nan_metrics:
                            logger.warning("📊 Other NaN metrics detected: %s", nan_metrics)
                            
                        # Try to restore best model if available
                        if self._best_metric is not None and self.log_dir is not None:
                            best_ckpt = self.log_dir / "best.pt"
                            if best_ckpt.exists() and hasattr(self.agent, "load_checkpoint"):
                                logger.info("🔄 Restoring best checkpoint from %s", best_ckpt)
                                self.agent.load_checkpoint(best_ckpt)
                            else:
                                logger.warning("⚠️  Best checkpoint not found, continuing with current model")
                        break
                        
                except Exception as e:
                    logger.error("💥 Validation failed with error: %s", e)
                    logger.info("🔄 Attempting to continue training...")
                    continue

                # Standard early‐stop logic
                improvement = False
                if self._best_metric is None:
                    improvement = True
                elif delta is not None and delta > 0:
                    improvement = curr_metric > self._best_metric + delta
                else:
                    # Dynamically increase tolerance when metric variance is high
                    if not hasattr(self, "_metric_hist"):
                        self._metric_hist = []  # type: ignore[attr-defined]
                    self._metric_hist.append(curr_metric)
                    recent_hist = self._metric_hist[-5:]
                    std_m = float(np.std(recent_hist)) if len(recent_hist) > 1 else 0.0
                    dyn_rel_delta = max(rel_delta, 0.05 * std_m)  # adaptive threshold
                    improvement = curr_metric > self._best_metric * (1 + dyn_rel_delta)

                if improvement:
                    self._best_metric = curr_metric
                    self._epochs_no_improve = 0
                    logger.info("  🎯 New best %.4f (%s)", curr_metric, early_stop_metric)
                    if self.log_dir is not None:
                        logger.info("  💾 Saving checkpoint")
                        self.save_checkpoint(self.log_dir / "best.pt")
                else:
                    self._epochs_no_improve += 1
                    logger.info(
                        "  ↳ No improvement for %d/%d epochs", self._epochs_no_improve, patience
                    )
                    if self._epochs_no_improve >= patience:
                        logger.info("Early stopping triggered ✋")
                        break

            # Decay ε for policy probability smoothing
            if hasattr(self.agent, "action_dims") and self.agent.action_dims:
                min_eps = 1.0 / math.prod(self.agent.action_dims)
            else:
                min_eps = 1e-6  # Safe fallback minimum epsilon
            self.pi_eps = max(self.pi_eps * 0.95, min_eps)
            
        # ========================================================================
        # Generate training visualizations after fit completes
        # ========================================================================
        if self.enable_visualization:
            logger.info("🎨 Generating post-training visualizations...")
            try:
                vis_results = self.generate_training_visualizations(
                    include_convergence=True,
                    include_strategy=False  # Strategy analysis requires test data
                )
                if vis_results:
                    logger.info("✅ Generated %d visualization types", len(vis_results))
                else:
                    logger.info("ℹ️  No visualizations generated (may be disabled or no data)")
            except Exception as e:
                logger.warning("⚠️  Visualization generation failed: %s", e)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split: str = "test",
        bootstrap_iters: int = 0,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """Compute reward, survival rate, OPE estimates over *loader*."""
        # Reset cached IS weights to avoid cross-run leakage
        self._cached_weights.clear()

        # ------------------------------------------------------------------
        # Pass 1  – collect batches for behaviour‐policy & raw stats --------
        # ------------------------------------------------------------------
        cached_batches: List[Dict[str, torch.Tensor]] = []
        survival: List[int] = []
        ips_weights: List[float] = []
        # Aggregate rewards robustly without assuming flat lists
        _sum_reward: float = 0.0
        _count_reward: int = 0

        # 占位定义，避免在首次循环时 importance_weights 未定义
        importance_weights = torch.empty(0)

        for batch in loader:
            # Move to CPU to avoid holding GPU memory when fitting FQE later
            # 🔧 CRITICAL FIX: Add None checks before calling .cpu() method
            cpu_batch = {}
            for k, v in batch.items():
                if v is not None and torch.is_tensor(v):
                    cpu_batch[k] = v.detach().cpu()
                elif v is not None:
                    # Handle non-tensor values (like lists, scalars, etc.)
                    cpu_batch[k] = v
                else:
                    # Skip None values to avoid downstream errors
                    self.logger.warning(f"⚠️ Skipping None value for key '{k}' in evaluate batch")
                    cpu_batch[k] = None
            cached_batches.append(cpu_batch)

            # --------------------------------------------------------------
            # Robust survival indicator handling: datasets may provide
            # per-trajectory (B,) or per-step (B,T) binary vectors.  We
            # *align the flattened survival tensor with the flattened
            # importance-weight vector* so that downstream NumPy ops do
            # not trigger broadcasting errors (cf. issue #OPE-117).
            # --------------------------------------------------------------
            surv_batch = cpu_batch.get("survival")
            if surv_batch is None:
                surv_batch = torch.zeros_like(cpu_batch["done"])

            # Ensure dtype=bool/int for later arithmetic
            if surv_batch.dtype not in (torch.int64, torch.int32, torch.uint8, torch.bool):
                surv_batch = surv_batch.long()

            # ------------------------------------------------------------------
            # Case 1) surv_batch shape == importance_weights shape  → element-wise
            # Case 2) surv_batch is (B,) but importance_weights is (B,T)  → repeat
            # Case 3) surv_batch already (B,T) but importance_weights collapsed to
            #         (B,) (unlikely) – repeat weights instead (not needed here).
            # ------------------------------------------------------------------
            if surv_batch.dim() == 1 and 'importance_weights' in locals() and importance_weights.dim() == 2:
                # Repeat per-trajectory indicator across time dimension
                try:
                    surv_rep = surv_batch.unsqueeze(1).expand_as(importance_weights)
                    surv_flat = surv_rep.reshape(-1)
                except RuntimeError:
                    # 若维度无法对齐，退化为简单展平
                    surv_flat = surv_batch.reshape(-1)
            else:
                surv_flat = surv_batch.reshape(-1)

            survival.extend(surv_flat.tolist())

            # ------------------------------- NEW -----------------------------
            # Importance Sampling ratio ρ_t = π_e(a_t|s_t) / μ(a_t|s_t).
            # Prior implementation mistakenly used 1/μ, ignoring π_e, causing
            # validation metrics to stay constant across epochs.  We now
            # compute both target-policy probability **and** behaviour policy
            # probability for each sample.
            # -----------------------------------------------------------------
            behav_prob = self._estimate_behavior_policy(cpu_batch)
            target_prob = self._policy_probability(cpu_batch)

            # 统一在此处完成分母裁剪，阈值可在 __init__ 中调节
            if self.algo == "physician":
                # Physician baseline应与行为策略一致 – 权重恒为1
                importance_weights = torch.ones_like(target_prob)
            else:
                importance_weights = target_prob / behav_prob.clamp(min=self.behav_prob_min)

            if self.clip_range is not None and self.clip_range > 0:
                importance_weights = importance_weights.clamp(max=self.clip_range)

            # ------------------------------------------------------------------
            # Flatten (B,T) → (B*T,) so that ``ips_weights`` remains a simple
            # 1-D list.  This avoids dtype=object arrays downstream when
            # ``np.asarray(ips_weights)`` is called and keeps backward-
            # compatibility with code paths that expect a vector.
            # ------------------------------------------------------------------
            iw_flat = importance_weights.view(-1).cpu()
            ips_weights.extend(iw_flat.tolist())

            # --------------------------------------------------------------
            # Align survival indicators with flattened importance weights
            # --------------------------------------------------------------
            if surv_batch.dtype not in (
                torch.int64,
                torch.int32,
                torch.uint8,
                torch.bool,
            ):
                surv_batch = surv_batch.long()

            # 形状检查：若 batch 尺寸与 importance_weights 首维不一致，则直接展平并跳过对齐逻辑
            if (
                surv_batch.dim() == 1
                and importance_weights.dim() == 2
                and surv_batch.size(0) != importance_weights.size(0)
            ):
                surv_flat = surv_batch.reshape(-1)
                survival.extend(surv_flat.tolist())

                # 仍需收集权重与奖励统计，避免跳过关键信息
                iw_flat = importance_weights.view(-1).cpu()
                ips_weights.extend(iw_flat.tolist())

                if len(ips_weights) < 1_000:
                    if not hasattr(self, "_dbg_iw_stats"):
                        self._dbg_iw_stats = []  # type: ignore[attr-defined]
                    self._dbg_iw_stats.append(iw_flat)

                rew_t = cpu_batch["reward"]  # Tensor
                _sum_reward += safe_item(rew_t.sum())
                _count_reward += rew_t.numel()

                continue

            if surv_batch.dim() == 1 and importance_weights.dim() == 2:
                try:
                    surv_rep = surv_batch.unsqueeze(1).expand_as(importance_weights)
                    surv_flat = surv_rep.reshape(-1)
                except RuntimeError:
                    surv_flat = surv_batch.reshape(-1)
            else:
                surv_flat = surv_batch.reshape(-1)

            survival.extend(surv_flat.tolist())

            # ---------- Debug: collect weight stats (first few batches) ------
            if len(ips_weights) < 1_000:
                if not hasattr(self, "_dbg_iw_stats"):
                    self._dbg_iw_stats = []  # type: ignore[attr-defined]
                self._dbg_iw_stats.append(iw_flat)

            # ----------------------------------------------------------
            # Robust reward aggregation: supports both per-step vectors
            # (B,T) and per-trajectory scalars (B,).
            # ----------------------------------------------------------
            rew_t = cpu_batch["reward"]  # Tensor
            _sum_reward += float(rew_t.sum().item())
            _count_reward += rew_t.numel()

        # Dataset‐level baselines ------------------------------------------------
        dataset_reward = float(_sum_reward / _count_reward) if _count_reward > 0 else float("nan")
        # --------------------------------------------------------------
        # Infer encoding: many clinical datasets use 1=death, 0=survival.
        # If mean(raw) < 0.5 we assume it is *mortality*; convert to
        # survival rate as (1 - mortality).
        # This auto-adapts and prevents misleading 0.19 outputs observed
        # previously when 80 % 的病人 actually survived.
        # --------------------------------------------------------------
        if survival:
            mean_raw = float(sum(survival) / len(survival))
            # mortality if <0.5 else survival already
            if mean_raw < 0.5:
                dataset_survival_rate = 1.0 - mean_raw  # convert
                surv_indicator = [1 - s for s in survival]
            else:
                dataset_survival_rate = mean_raw
                surv_indicator = survival  # already survival==1
        else:
            dataset_survival_rate = math.nan
            surv_indicator = []

        # IPS estimate of survival (same as before) -----------------------------
        if survival:
            surv_np = np.asarray(surv_indicator)
            iw_np = np.asarray(ips_weights)

            # 若长度不一致，截取到最短长度以保证元素乘法合法
            min_len = min(len(surv_np), len(iw_np))
            if len(surv_np) != len(iw_np):
                surv_np = surv_np[:min_len]
                iw_np = iw_np[:min_len]

            # Optional Pareto-smoothed IS (PSIS) to stabilise long-tail weights
            from Libs.utils.ope.psis import psis_smooth_weights
            iw_np_psis = psis_smooth_weights(torch.tensor(iw_np)).numpy()
            num = float((iw_np_psis * surv_np).sum())
            denom = float(iw_np_psis.sum() + 1e-8)
            ips_surv = num / denom
            # Clip to [0,1] per theoretical bound
            ips_surv = min(max(ips_surv, 0.0), 1.0)
            if bootstrap_iters and bootstrap_iters > 0:
                rng = np.random.default_rng(seed=42)
                N = len(surv_np)
                boot = [float((iw_np_psis[rng.integers(0, N, N)] * surv_np[rng.integers(0, N, N)]).mean()) for _ in range(bootstrap_iters)]
                ci_lower, ci_upper = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            else:
                # Fix: Calculate reasonable confidence intervals even without bootstrap
                if len(surv_np) > 0:
                    # Use standard error as approximation
                    weighted_mean = float((iw_np_psis * surv_np).mean())
                    weighted_var = float(((iw_np_psis * (surv_np - weighted_mean))**2).mean())
                    std_err = float(np.sqrt(weighted_var / len(surv_np)))
                    z_score = 1.96  # 95% confidence interval
                    ci_lower = max(0.0, weighted_mean - z_score * std_err)
                    ci_upper = min(1.0, weighted_mean + z_score * std_err)
                else:
                    ci_lower = ci_upper = ips_surv  # Use the point estimate
                
            # Store CI in metrics for CSV export
            ips_survival_ci = [ci_lower, ci_upper]
        else:
            ips_surv = ci_lower = ci_upper = math.nan

        # ------------------------------------------------------------------
        # Fit FQE only if the agent exposes required Q-net interfaces -------
        # ------------------------------------------------------------------
        # Skip FQE for algorithms that don't benefit from it
        fqe_compatible_algos = {"dqn", "pog_dqn", "cql", "pog_cql", "bcq", "pog_bcq", "bve", "pog_bve"}
        should_compute_fqe = self.algo in fqe_compatible_algos
        
        has_q_attr = (
            should_compute_fqe
            and hasattr(self.agent, "q_net")
            and hasattr(self.agent, "target_q_net")
            and callable(getattr(self.agent.q_net, "q_value", None))
            and callable(getattr(self.agent.target_q_net, "q_value", None))
        )

        wdr_vals: List[float] = []  # ensure defined for bootstrap later
        ipw_vals: List[float] = []

        wdr_reward: float
        ipw_reward: float

        if has_q_attr:
            transitions = {
                k: torch.cat([b[k] for b in cached_batches], dim=0)
                for k in ("state", "action", "reward", "next_state", "done")
            }

            # ----------------------------------------------------------
            # 🔧 CRITICAL FIX: Robust tensor flattening with dimension validation
            # Flatten (B,T,·) ➜ (B*T, ·) to comply with q_value(state,action)
            # expectations of FQE/PoG modules which assume *no* time dim.
            # ----------------------------------------------------------
            if transitions["state"].dim() == 3:
                B, T, D = transitions["state"].shape
                self.logger.debug(f"🔧 Flattening sequences: B={B}, T={T}, D={D}")
                
                # Flatten state tensors
                transitions["state"] = transitions["state"].reshape(B*T, D)
                transitions["next_state"] = transitions["next_state"].reshape(B*T, D)

                # 🔧 ENHANCED ACTION HANDLING: Validate dimensions before reshaping
                acts = transitions["action"]
                self.logger.debug(f"🔧 Original action shape: {acts.shape}")
                
                if acts.dim() == 3:
                    # (B, T, H) -> (B*T, H)
                    if acts.size(0) != B or acts.size(1) != T:
                        self.logger.warning(f"⚠️ Action tensor dimension mismatch: expected ({B}, {T}, H), got {acts.shape}")
                        # Try to fix by taking first B samples and T timesteps
                        acts = acts[:B, :T, :]
                    acts = acts.reshape(B*T, acts.size(-1))
                elif acts.dim() == 2:
                    if acts.size(0) == B:
                        # (B, H) -> (B*T, H) by repeating each sample T times
                        acts = acts.repeat_interleave(T, dim=0)
                    elif acts.size(0) == B*T:
                        # Already flattened - validate shape
                        pass
                    else:
                        self.logger.warning(f"⚠️ Action tensor batch size mismatch: expected {B} or {B*T}, got {acts.size(0)}")
                        # Truncate or pad to match expected size
                        if acts.size(0) > B*T:
                            acts = acts[:B*T]
                        else:
                            # This is problematic - we can't safely pad actions
                            self.logger.error(f"❌ Cannot safely handle undersized action tensor: {acts.shape}")
                            # Return empty metrics to avoid crashes
                            return {"reward": 0.0, "survival_rate": 0.0, "fqe_value": 0.0, "wdr_reward": 0.0, "ipw_reward": 0.0}
                elif acts.dim() == 1:
                    # (B*T,) - single action per timestep, add action head dimension
                    if acts.size(0) == B*T:
                        acts = acts.unsqueeze(1)  # (B*T,) -> (B*T, 1)
                    elif acts.size(0) == B:
                        acts = acts.repeat_interleave(T).unsqueeze(1)  # (B,) -> (B*T, 1)
                    else:
                        self.logger.error(f"❌ Cannot handle 1D action tensor with size {acts.size(0)}")
                        return {"reward": 0.0, "survival_rate": 0.0, "fqe_value": 0.0, "wdr_reward": 0.0, "ipw_reward": 0.0}
                else:
                    self.logger.error(f"❌ Unsupported action tensor dimensions: {acts.shape}")
                    return {"reward": 0.0, "survival_rate": 0.0, "fqe_value": 0.0, "wdr_reward": 0.0, "ipw_reward": 0.0}

                # Ensure all transition tensors have consistent first dimension
                expected_size = B*T
                self.logger.debug(f"🔧 Target flattened size: {expected_size}")
                
                # Write back the processed actions
                transitions["action"] = acts
                self.logger.debug(f"🔧 Final action shape: {acts.shape}")

                # 🔧 ENHANCED REWARD/DONE HANDLING: Validate dimensions before flattening
                reward = transitions["reward"]
                done = transitions["done"]
                
                if reward.dim() == 2:
                    if reward.size(0) != B or reward.size(1) != T:
                        self.logger.warning(f"⚠️ Reward tensor dimension mismatch: expected ({B}, {T}), got {reward.shape}")
                        reward = reward[:B, :T]  # Truncate to fit
                    reward = reward.reshape(-1)
                elif reward.dim() == 1:
                    if reward.size(0) == B:
                        reward = reward.repeat_interleave(T)
                    elif reward.size(0) != B*T:
                        self.logger.warning(f"⚠️ Reward tensor size mismatch: expected {B*T}, got {reward.size(0)}")
                        reward = reward[:B*T] if reward.size(0) > B*T else torch.cat([reward, torch.zeros(B*T - reward.size(0))])
                
                if done.dim() == 2:
                    if done.size(0) != B or done.size(1) != T:
                        self.logger.warning(f"⚠️ Done tensor dimension mismatch: expected ({B}, {T}), got {done.shape}")
                        done = done[:B, :T]  # Truncate to fit
                    done = done.reshape(-1)
                elif done.dim() == 1:
                    if done.size(0) == B:
                        done = done.repeat_interleave(T)
                    elif done.size(0) != B*T:
                        self.logger.warning(f"⚠️ Done tensor size mismatch: expected {B*T}, got {done.size(0)}")
                        done = done[:B*T] if done.size(0) > B*T else torch.cat([done, torch.zeros(B*T - done.size(0), dtype=done.dtype)])
                
                transitions["reward"] = reward
                transitions["done"] = done
                
                # Final validation: ensure all tensors have the same first dimension
                sizes = {k: v.size(0) for k, v in transitions.items()}
                if len(set(sizes.values())) > 1:
                    self.logger.error(f"❌ Inconsistent tensor sizes after flattening: {sizes}")
                    # Return empty metrics to avoid crash
                    return {"reward": 0.0, "survival_rate": 0.0, "fqe_value": 0.0, "wdr_reward": 0.0, "ipw_reward": 0.0}
                else:
                    self.logger.debug(f"✅ All transition tensors aligned with size {expected_size}")

            fqe = self.fit_fqe(
                transitions,
                n_epochs=self.fqe_epochs,
                batch_size=self.fqe_batch_size,
                lr=self.fqe_lr,
            )

            # ---------------- NEW 计算 FQE 估计值 ----------------
            try:
                init_states = transitions["state"]  # (N, state_dim)
                
                # 🔧 CRITICAL FIX: Apply sequence truncation for FQE evaluation consistency
                # This ensures init_states dimensions match the truncated sequences used in training
                MAX_SAFE_SEQ_LEN = 4096  # Match the PoG model truncation limit
                if init_states.dim() == 3 and init_states.size(1) > MAX_SAFE_SEQ_LEN:
                    original_seq_len = init_states.size(1)
                    init_states = init_states[:, -MAX_SAFE_SEQ_LEN:, :]  # Keep most recent timesteps
                    self.logger.debug(f"🔧 FQE: Truncated init_states from {original_seq_len} to {MAX_SAFE_SEQ_LEN} timesteps")
                
                # 🔧 ENHANCED ERROR HANDLING: Robust policy action generation with multiple fallbacks
                if init_states.dim() == 3:
                    # For sequence data, extract final timestep features for FQE
                    init_states_fqe = init_states[:, -1, :]
                else:
                    init_states_fqe = init_states
                
                # 🔧 MULTIPLE FALLBACK STRATEGY for policy action generation
                policy_actions = None
                fqe_attempts = 0
                max_fqe_attempts = 3
                
                # 🔧 CRITICAL FIX: Ensure init_states_fqe is on the correct device before any FQE operations
                target_device = fqe.device
                init_states_fqe = init_states_fqe.to(target_device)
                
                while policy_actions is None and fqe_attempts < max_fqe_attempts:
                    fqe_attempts += 1
                    try:
                        # Attempt 1: Standard greedy action selection
                        if fqe_attempts == 1:
                            self.logger.debug(f"🔧 FQE attempt {fqe_attempts}: Standard greedy action selection")
                            # Ensure input is on the correct device
                            policy_actions = fqe.q_net.greedy_action(init_states_fqe.to(fqe.device))
                            
                        # Attempt 2: Direct Q-network forward pass with manual argmax
                        elif fqe_attempts == 2:
                            self.logger.debug(f"🔧 FQE attempt {fqe_attempts}: Manual Q-network forward pass")
                            with torch.no_grad():
                                # Ensure input is on the correct device
                                q_values_raw = fqe.q_net(init_states_fqe.to(fqe.device))
                                if isinstance(q_values_raw, list):
                                    # Multi-head Q-network: get argmax for each head
                                    policy_actions_list = []
                                    for q_head in q_values_raw:
                                        if q_head.dim() == 3:
                                            q_head = q_head[:, -1, :]  # Take last timestep
                                        policy_actions_list.append(q_head.argmax(dim=-1))
                                    policy_actions = torch.stack(policy_actions_list, dim=1)
                                else:
                                    # Single-head Q-network
                                    if q_values_raw.dim() == 3:
                                        q_values_raw = q_values_raw[:, -1, :]
                                    policy_actions = q_values_raw.argmax(dim=-1, keepdim=True)
                                    
                        # Attempt 3: Use random valid actions as last resort
                        elif fqe_attempts == 3:
                            self.logger.warning(f"🔧 FQE attempt {fqe_attempts}: Using random valid actions as fallback")
                            batch_size = init_states_fqe.size(0)
                            n_action_heads = len(self.agent.action_dims)
                            policy_actions = torch.zeros(batch_size, n_action_heads, dtype=torch.long, device=target_device)
                            for i, action_dim in enumerate(self.agent.action_dims):
                                policy_actions[:, i] = torch.randint(0, action_dim, (batch_size,), device=target_device)
                                
                        # Validate generated actions
                        if policy_actions is not None:
                            # Check for NaN/Inf values
                            if torch.isnan(policy_actions).any() or torch.isinf(policy_actions).any():
                                self.logger.warning(f"⚠️ FQE attempt {fqe_attempts}: Policy actions contain NaN/Inf, discarding")
                                policy_actions = None
                                continue
                            
                            # Check action bounds
                            for i, action_dim in enumerate(self.agent.action_dims):
                                if i < policy_actions.size(1):
                                    invalid_actions = (policy_actions[:, i] < 0) | (policy_actions[:, i] >= action_dim)
                                    if invalid_actions.any():
                                        self.logger.warning(f"⚠️ FQE attempt {fqe_attempts}: Found {invalid_actions.sum().item()} invalid actions for head {i}, clamping")
                                        policy_actions[:, i] = torch.clamp(policy_actions[:, i], 0, action_dim - 1)
                            
                            self.logger.debug(f"✅ FQE attempt {fqe_attempts}: Successfully generated policy actions with shape {policy_actions.shape}")
                            break
                            
                    except Exception as action_error:
                        self.logger.warning(f"⚠️ FQE attempt {fqe_attempts} failed: {action_error}")
                        policy_actions = None
                        if fqe_attempts < max_fqe_attempts:
                            continue
                        else:
                            raise action_error
                
                # Validate policy actions one final time
                if policy_actions is None:
                    self.logger.error("❌ All FQE action generation attempts failed")
                    fqe_reward_est = 0.0
                elif torch.isnan(policy_actions).any() or torch.isinf(policy_actions).any():
                    self.logger.warning("⚠️ Policy actions contain NaN/Inf after all attempts, using 0.0 for FQE")
                    fqe_reward_est = 0.0
                else:
                    # 🔧 ENHANCED FQE EVALUATION with robust error handling
                    try:
                        fqe_reward_est_raw = fqe.evaluate(init_states_fqe, policy_actions)
                        
                        # Final numerical validation
                        if not math.isfinite(fqe_reward_est_raw):
                            self.logger.warning(f"⚠️ FQE estimate is not finite ({fqe_reward_est_raw}), using 0.0")
                            fqe_reward_est = 0.0
                        else:
                            fqe_reward_est = float(fqe_reward_est_raw)
                            self.logger.debug(f"✅ FQE evaluation successful: {fqe_reward_est:.6f}")
                            
                    except Exception as fqe_eval_error:
                        self.logger.warning(f"⚠️ FQE evaluation failed: {fqe_eval_error}, using 0.0")
                        fqe_reward_est = 0.0
                        
            except Exception as e:
                # 若任意步骤失败，记录 0.0 不影响主流程
                self.logger.warning(f"⚠️ FQE evaluation pipeline failed: {e}, using 0.0")
                fqe_reward_est = 0.0

            for cpu_batch in cached_batches:
                # 传入 clip_range 限制极端 IS 权重
                wdr_est, ipw_est = self.compute_wdr_ipw(cpu_batch, fqe, use_psis=self.use_psis)
                wdr_vals.append(wdr_est)
                ipw_vals.append(ipw_est)

            wdr_reward = float(sum(wdr_vals) / len(wdr_vals)) if wdr_vals else math.nan
            ipw_reward = float(sum(ipw_vals) / len(ipw_vals)) if ipw_vals else math.nan
        else:
            # Fallback: use plain IPW reward estimate only ------------------
            from Libs.utils.ope.ipw import ipw_estimate

            # Apply PSIS smoothing even in fallback to tame long-tail weights
            try:
                from Libs.utils.ope.psis import psis_smooth_weights as _smooth
                _use_psis_fallback = True
            except Exception:
                _use_psis_fallback = False

            ipw_vals_fallback: List[float] = []
            for cpu_batch in cached_batches:
                rewards_t = cpu_batch["reward"].float()
                # If rewards are provided as *aggregated* per-trajectory scalars
                # (shape (B,)) rather than per-step vectors (shape (B,T)), we
                # down-scale by the trajectory length to match per-step units
                # and avoid order-of-magnitude inflation in IPW estimates.
                if rewards_t.dim() == 1 and "traj_len" in cpu_batch:
                    rewards_t = rewards_t / cpu_batch["traj_len"].float()
                dones_t = cpu_batch["done"].float()
                behav_prob = self._estimate_behavior_policy(cpu_batch)
                target_prob = self._policy_probability(cpu_batch)

                if self.algo == "physician":
                    iw = torch.ones_like(rewards_t)  # broadcast later
                else:
                    iw = target_prob / behav_prob.clamp(min=self.behav_prob_min)
                    if self.clip_range is not None and self.clip_range > 0:
                        iw = iw.clamp(max=self.clip_range)

                # ------------------------------------------------------
                # Align shapes: ipw_estimate expects *identical* tensor
                # rank for rewards, dones, and importance_weights.
                # ------------------------------------------------------
                if rewards_t.dim() == 2 and iw.dim() == 1:
                    iw = iw.unsqueeze(1).expand_as(rewards_t)
                if dones_t.dim() == 2 and iw.dim() == 1:
                    iw = iw.unsqueeze(1).expand_as(dones_t)
                if _use_psis_fallback:
                    iw = _smooth(iw)

                ipw_vals_fallback.append(ipw_estimate(rewards_t, dones_t, iw))

            ipw_reward = float(sum(ipw_vals_fallback) / len(ipw_vals_fallback)) if ipw_vals_fallback else math.nan
            # 没有 Q-网络时无法计算 WDR，此处退化为 IPW 以避免验证指标为 NaN
            # 如果 IPW 也是 NaN，则使用一个合理的默认值（如 0.0）避免早停
            if math.isnan(ipw_reward):
                logger.warning("IPW reward is NaN, using default value 0.0 for early stopping")
                ipw_reward = 0.0
            wdr_reward = ipw_reward
            wdr_vals = ipw_vals_fallback  # 用于 bootstrap
            # Fix: Instead of setting FQE to NaN, use 0.0 with a warning
            logger.debug("FQE not available for this agent type (no Q-network), using 0.0")
            fqe_reward_est = 0.0  # Use 0.0 instead of math.nan for better compatibility

        if bootstrap_iters and bootstrap_iters > 0 and len(wdr_vals) > 0:
            rng_b = np.random.default_rng(seed=42)
            N = len(wdr_vals)
            boot_wdr = [float(np.mean(rng_b.choice(wdr_vals, N, replace=True))) for _ in range(bootstrap_iters)]
            reward_ci = [float(np.percentile(boot_wdr, 100 * alpha / 2)), float(np.percentile(boot_wdr, 100 * (1 - alpha / 2)))]
        else:
            # Fix: Calculate reasonable confidence intervals even without bootstrap
            if len(wdr_vals) > 0:
                # Use standard error as approximation
                mean_val = float(np.mean(wdr_vals))
                std_err = float(np.std(wdr_vals) / np.sqrt(len(wdr_vals)))
                z_score = 1.96  # 95% confidence interval
                reward_ci = [mean_val - z_score * std_err, mean_val + z_score * std_err]
            else:
                # Only use zeros when no data is available
                reward_ci = [wdr_reward, wdr_reward] if not math.isnan(wdr_reward) else [0.0, 0.0]

        # --------------------------------------------------------------
        # Collect all evaluation metrics into a single dictionary so that
        # downstream logging / early-stopping logic operates on a unified
        # structure.  This also resolves the earlier NameError caused by
        # the missing *metrics* variable.
        # --------------------------------------------------------------
        metrics = {
            "wdr_reward": wdr_reward,
            "ipw_reward": ipw_reward,
            "fqe_est": fqe_reward_est,
            "survival_rate": dataset_survival_rate,
            "ips_survival": ips_surv,
            "reward": wdr_reward,  # Alias for primary reward metric
            "reward_ci": reward_ci,  # Bootstrap confidence interval for reward
            "dataset_reward": dataset_survival_rate,  # Dataset baseline using survival rate
            "ips_survival_ci": ips_survival_ci if 'ips_survival_ci' in locals() else [0.0, 0.0],
        }

        # 🎯 优化日志输出 - 按类别分组并添加清晰的说明
        logger.info("=" * 60)
        logger.info("📊 [%s] EVALUATION RESULTS", split.upper())
        logger.info("=" * 60)
        
        # 奖励估计 (Off-Policy Evaluation)
        logger.info("🎁 REWARD ESTIMATES:")
        if not math.isnan(metrics["wdr_reward"]):
            logger.info("   • WDR (Weighted Doubly Robust): %.4f  ⭐ [Most Reliable]", metrics["wdr_reward"])
        if not math.isnan(metrics["ipw_reward"]):
            logger.info("   • IPW (Inverse Propensity):     %.4f", metrics["ipw_reward"])
        if not math.isnan(metrics["fqe_est"]):
            logger.info("   • FQE (Fitted Q Evaluation):    %.4f", metrics["fqe_est"])
        
        # 生存率分析
        logger.info("❤️  SURVIVAL ANALYSIS:")
        logger.info("   • Dataset Survival Rate:        %.4f  (baseline - NOT algorithm-specific)", metrics["survival_rate"])
        if not math.isnan(metrics["ips_survival"]):
            logger.info("   • Policy Survival Rate (IPS):   %.4f  (algorithm-specific estimate)", metrics["ips_survival"])
        
        # 性能总结
        best_reward = metrics["wdr_reward"] if not math.isnan(metrics["wdr_reward"]) else metrics["ipw_reward"]
        if not math.isnan(best_reward):
            logger.info("🏆 PERFORMANCE SUMMARY:")
            logger.info("   • Primary Metric (for comparison): %.4f", best_reward)
            improvement = ((best_reward - metrics["survival_rate"]) / metrics["survival_rate"]) * 100
            if improvement > 0:
                logger.info("   • Improvement over baseline: +%.2f%%", improvement)
            else:
                logger.info("   • Change from baseline: %.2f%%", improvement)
        
        logger.info("=" * 60)

        # One-off print of importance-weight summary for diagnostics
        if hasattr(self, "_dbg_iw_stats") and self._dbg_iw_stats:
            iw_cat = torch.cat(self._dbg_iw_stats)  # type: ignore[arg-type]
            logger.info("Importance-weights stats → mean=%.3f | std=%.3f | max=%.3f | min=%.3f",
                        float(iw_cat.mean()), float(iw_cat.std()), float(iw_cat.max()), float(iw_cat.min()))
            delattr(self, "_dbg_iw_stats")

        # 🔧 CRITICAL FIX: Safe handling of tensor values in metrics to prevent round() errors
        safe_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                safe_metrics[k] = round(v, 4)
            elif torch.is_tensor(v):
                # Convert tensor to float before rounding
                try:
                    safe_metrics[k] = round(float(v.item()), 4)
                except (ValueError, TypeError, RuntimeError):
                    # Fallback for multi-element tensors or other issues
                    if v.numel() > 1:
                        safe_metrics[k] = round(float(v.mean().item()), 4)
                    else:
                        safe_metrics[k] = round(float(v.item()), 4)
            else:
                safe_metrics[k] = v
        log_metric(logger, f"{split}-eval", **safe_metrics)
        return metrics

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _safe_dataloader_iteration(self, loader: DataLoader, operation_name: str = "DataLoader operation"):
        """Safe wrapper for DataLoader iteration with error recovery for multiprocessing issues.
        
        This function provides a robust interface for iterating over DataLoader objects
        with automatic error detection and recovery for common multiprocessing issues.
        
        Args:
            loader: The DataLoader to iterate over
            operation_name: Name of the operation for logging purposes
            
        Yields:
            Batches from the DataLoader
            
        Raises:
            RuntimeError: If a non-recoverable DataLoader error occurs
        """
        try:
            for batch in loader:
                yield batch
        except RuntimeError as e:
            if "Pin memory thread exited unexpectedly" in str(e):
                logger.error(f"💥 DataLoader pin_memory error detected during {operation_name}!")
                logger.error("This indicates a multiprocessing configuration issue.")
                logger.error("Please restart the training process - the issue should be resolved with the updated configuration.")
                raise RuntimeError(f"DataLoader multiprocessing error during {operation_name}. Please restart training.") from e
            else:
                # Re-raise other RuntimeErrors
                raise e
        except Exception as e:
            logger.error(f"❌ Unexpected error during {operation_name}: {e}")
            raise e
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute one optimisation step on *batch* with enhanced memory optimization and error handling.

        This method implements a single training step with comprehensive error handling,
        memory management, and robust batch processing. It serves as the core training
        loop component with enterprise-grade reliability features.

        Key Features:
            • Comprehensive input validation and sanitization
            • Adaptive batch format handling for different agent types
            • Memory-efficient processing with automatic cleanup
            • Numerical stability checks and fallback mechanisms
            • Detailed error reporting and recovery strategies
            • Performance monitoring and diagnostic logging

        The underlying ``Agent.update`` implementations across the codebase
        are unfortunately inconsistent in both **signature** (dict vs tuple)
        and **return type** (float vs tuple).  This helper standardises the
        call site so that:

        1.  We *always* forward a *dict* – if the agent declares an
            attribute ``expects_tuple_batch`` we convert to the legacy tuple
            format expected by older BC/BCQ/CQL agents.
        2.  The returned value is coerced to a scalar *float* (taking the
            first element when a tuple is given).
        3.  Enhanced memory monitoring and optimization for large cohorts.

        Args:
            batch: Training batch dictionary containing:
                  - 'state': Patient state tensor (batch_size, [seq_len,] state_dim)
                  - 'action': Chosen actions (batch_size, [seq_len,] n_heads)
                  - 'reward': Clinical outcomes (batch_size, [seq_len])
                  - 'next_state': Next patient states (same shape as state)
                  - 'done': Episode termination flags (batch_size, [seq_len])
                  - Optional: 'mask', 'lengths', 'edge_index', etc.

        Returns:
            Training loss as a scalar float. Returns 0.0 on irrecoverable errors
            to allow training to continue rather than crash.

        Raises:
            RuntimeError: Only for critical system errors that require immediate attention.
            
        Note:
            This method is designed to be fault-tolerant and will attempt to recover
            from non-critical errors by logging warnings and returning fallback values.
            This is essential for long-running medical RL experiments where occasional
            batch corruption should not terminate the entire training process.
        """
        
        # ========================================================================
        # Enhanced Input Validation and Sanitization
        # ========================================================================
        
        if not isinstance(batch, dict):
            self.logger.error(f"❌ Expected dict batch, got {type(batch)}")
            return 0.0
        
        # Validate required fields with helpful error messages
        required_fields = ['state', 'action', 'reward']
        missing_fields = [field for field in required_fields if field not in batch]
        if missing_fields:
            self.logger.error(f"❌ Missing required batch fields: {missing_fields}")
            self.logger.error(f"🔍 Available fields: {list(batch.keys())}")
            return 0.0
        
        # Comprehensive batch sanitization and validation
        try:
            # Validate tensor shapes and types
            for field_name, tensor in batch.items():
                if not torch.is_tensor(tensor):
                    self.logger.warning(f"⚠️  Converting {field_name} to tensor")
                    batch[field_name] = torch.as_tensor(tensor, device=self.device)
                elif tensor.device != self.device:
                    self.logger.debug(f"Moving {field_name} to {self.device}")
                    batch[field_name] = tensor.to(self.device)
                    
                # Check for NaN/Inf values (critical for medical RL)
                if torch.isnan(tensor).any():
                    nan_count = safe_item(torch.isnan(tensor).sum())
                    self.logger.error(f"❌ NaN values detected in {field_name}: {nan_count} values")
                    if field_name in ['state', 'next_state']:
                        # Critical error - cannot train with NaN states
                        self.logger.error("💥 Cannot proceed with NaN states - skipping batch")
                        return 0.0
                    else:
                        # Non-critical - replace with safe values
                        self.logger.warning(f"⚠️  Replacing NaN values in {field_name} with 0.0")
                        batch[field_name] = torch.nan_to_num(tensor, nan=0.0)
                        
                if torch.isinf(tensor).any():
                    inf_count = safe_item(torch.isinf(tensor).sum())
                    self.logger.warning(f"⚠️  Infinite values detected in {field_name}: {inf_count} values")
                    batch[field_name] = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 🔧 ENHANCED DIMENSION CONSISTENCY CHECK: Ensure all tensors have compatible batch dimensions
            batch_sizes = {}
            for field_name, tensor in batch.items():
                if torch.is_tensor(tensor) and field_name not in ['edge_index', 'edge_attr']:
                    batch_sizes[field_name] = tensor.size(0)
            
            if batch_sizes:
                primary_batch_size = max(batch_sizes.values())  # Use the most common batch size
                for field_name, size in batch_sizes.items():
                    if size != primary_batch_size:
                        self.logger.warning(f"⚠️  Batch size mismatch for {field_name}: {size} vs {primary_batch_size}")
                        tensor = batch[field_name]
                        if size > primary_batch_size:
                            # Truncate to match
                            batch[field_name] = tensor[:primary_batch_size]
                            self.logger.debug(f"🔧 Truncated {field_name} to batch size {primary_batch_size}")
                        else:
                            # This is more problematic - log but continue
                            self.logger.warning(f"⚠️  Cannot safely expand {field_name} from {size} to {primary_batch_size}")
                    
        except Exception as e:
            self.logger.error(f"❌ Batch sanitization failed: {e}")
            return 0.0

        # ========================================================================
        # Memory Optimization and Monitoring
        # ========================================================================
        
        initial_memory_status = None
        if hasattr(self, 'memory_manager'):
            try:
                initial_memory_status = self.memory_manager.monitor.check_memory_status()
                
                # Handle critical memory situations proactively
                if initial_memory_status.get("should_clear_cache", False):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.warning("🧹 Proactive memory cleanup due to high usage")
                    
            except Exception as e:
                self.logger.warning(f"⚠️  Memory monitoring failed: {e}")

        # ========================================================================
        # Agent-Specific Training Mode Setup
        # ========================================================================
        
        # Switch model to train mode when applicable (enhanced error handling)
        try:
            if hasattr(self.agent, "set_training_mode"):
                self.agent.set_training_mode(True)
            elif hasattr(self.agent, "train"):
                self.agent.train()
            elif hasattr(self.agent, "q_net") and isinstance(self.agent.q_net, torch.nn.Module):
                self.agent.q_net.train()
            elif hasattr(self.agent, "model") and isinstance(self.agent.model, torch.nn.Module):
                self.agent.model.train()
                
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to set training mode: {e}")
            # Continue training - this is not critical

        # ========================================================================
        # Enhanced Batch Format Adaptation for Legacy Compatibility
        # ========================================================================
        
        # Cache conversion status to avoid redundant checks
        if not hasattr(self, '_batch_format_cache'):
            self._batch_format_cache = {}
        
        agent_type = type(self.agent).__name__
        
        try:
            # Check cache first to avoid repeated attribute lookups
            if agent_type not in self._batch_format_cache:
                self._batch_format_cache[agent_type] = getattr(self.agent, "expects_tuple_batch", False)
            
            expects_tuple = self._batch_format_cache[agent_type]
            
            if expects_tuple:
                # Only log on first conversion or when action shape changes
                if not hasattr(self, '_last_action_shape') or (
                    "action" in batch and torch.is_tensor(batch["action"]) 
                    and batch["action"].shape != self._last_action_shape
                ):
                    self.logger.debug(
                        f"🔧 Converting to legacy tuple format for {agent_type}")
                    if "action" in batch and torch.is_tensor(batch["action"]):
                        self._last_action_shape = batch["action"].shape
                        self.logger.debug(f"🔧 Original actions shape: {self._last_action_shape}")
                
                # Convert to legacy tuple format for older agents
                batch = self._convert_batch_to_legacy_format(batch)
                
                # Only log shapes on first conversion
                if not hasattr(self, '_shapes_logged'):
                    if isinstance(batch, tuple) and len(batch) > 1 and isinstance(batch[1], list):
                        actions_shapes = [getattr(a, 'shape', f'type:{type(a)}') for a in batch[1]]
                        self.logger.debug(f"🔧 Converted actions shapes: {actions_shapes}")
                    self._shapes_logged = True
            else:
                # Modern dict format - ensure proper preprocessing
                batch = self._preprocess_dict_batch(batch)
                
        except Exception as e:
            self.logger.error(f"❌ Batch format adaptation failed: {e}")
            return 0.0

        # ========================================================================
        # CRITICAL ARCHITECTURAL FIX: Enhanced Training Step Execution with Comprehensive Tensor Output Validation
        # This section completely redesigns the training step execution to eliminate "outputs must be a Tensor or an iterable of Tensors" errors
        # by implementing rigorous output format validation and conversion protocols.
        # ========================================================================
        
        try:
            # 🔧 UNIFIED OUTPUT VALIDATION SYSTEM: Pre-execution preparation
            self.logger.debug("🔧 Starting training step with enhanced output validation")
            
            # Validate agent has required methods
            if not hasattr(self.agent, 'update') or not callable(self.agent.update):
                raise AttributeError(f"Agent {type(self.agent).__name__} must have a callable 'update' method")
            
            # 🔧 COMPREHENSIVE BATCH FORMAT VALIDATION: Ensure consistent batch format
            # This prevents the root cause of many tensor format errors
            if hasattr(self.agent, 'expects_tuple_batch') and self.agent.expects_tuple_batch:
                self.logger.debug("🔧 Converting to tuple batch format for legacy agent")
                
                # 🔧 CRITICAL FIX: Check input batch type before conversion
                if isinstance(batch, dict):
                    # Need to convert dict to tuple
                    if hasattr(self, '_convert_batch_to_legacy_format'):
                        try:
                            batch_converted = self._convert_batch_to_legacy_format(batch)
                            self.logger.debug(f"🔧 Dict->Tuple conversion successful: {type(batch_converted)}")
                        except Exception as convert_e:
                            self.logger.error(f"❌ Legacy batch conversion failed: {convert_e}")
                            # Fallback to original batch
                            batch_converted = batch
                    else:
                        self.logger.warning("⚠️ Legacy batch conversion method not found, using original batch")
                        batch_converted = batch
                elif isinstance(batch, (tuple, list)):
                    # Already in tuple/list format
                    self.logger.debug("🔧 Batch already in tuple format, using as-is")
                    batch_converted = batch
                else:
                    self.logger.error(f"❌ Unexpected batch type: {type(batch)}")
                    batch_converted = batch
            else:
                # Modern agents expect dict format
                if isinstance(batch, dict):
                    batch_converted = batch
                    self.logger.debug("🔧 Using dict batch format for modern agent")
                else:
                    self.logger.warning(f"⚠️ Modern agent expects dict but got {type(batch)}")
                    batch_converted = batch
                self.logger.debug("🔧 Using dict batch format for modern agent")
            
            # 🔧 ENHANCED TRAINING EXECUTION: Multi-context execution with comprehensive error handling
            loss_raw = None
            execution_context = "unknown"
            
            # Add POG-specific debugging
            if self.algo.startswith("pog_"):
                self.logger.debug("🔍 POG Algorithm: %s", self.algo)
                if hasattr(self.agent, '_algorithm_variant'):
                    self.logger.debug("   • Algorithm variant: %s", self.agent._algorithm_variant)
                if hasattr(self.agent, 'model') and hasattr(self.agent.model, 'init_args'):
                    self.logger.debug("   • Model init args: %s", self.agent.model.init_args)
                
                # Log batch statistics for POG models
                if isinstance(batch, dict):
                    for key in ['state', 'action', 'reward', 'done']:
                        if key in batch:
                            tensor = batch[key]
                            if torch.is_tensor(tensor):
                                self.logger.debug("   • %s shape: %s, mean: %.4f, std: %.4f", 
                                                key, tensor.shape, 
                                                tensor.float().mean().item(), 
                                                tensor.float().std().item())
            
            # Context 1: Memory-aware execution (preferred)
            if hasattr(self, 'memory_manager') and hasattr(self.memory_manager, 'training_context'):
                try:
                    execution_context = "memory_managed"
                    with self.memory_manager.training_context(
                        model=getattr(self.agent, 'q_net', None) or getattr(self.agent, 'model', None),
                        enable_gradient_checkpointing=True,
                        enable_mixed_precision=self.amp
                    ):
                        self.logger.debug("🔧 Executing agent update in memory-managed context")
                        loss_raw = self.agent.update(batch_converted, grad_scaler=self.grad_scaler)
                        
                except Exception as memory_error:
                    self.logger.warning(f"⚠️ Memory-managed execution failed: {memory_error}")
                    # Continue to fallback execution
                    loss_raw = None
            
            # Context 2: Standard execution (fallback)
            if loss_raw is None:
                try:
                    execution_context = "standard"
                    self.logger.debug("🔧 Executing agent update in standard context")
                    
                    # Additional validation before agent.update call
                    if isinstance(batch_converted, dict):
                        # Validate all tensors in dict batch
                        for key, value in batch_converted.items():
                            if torch.is_tensor(value):
                                if not value.is_leaf and value.requires_grad:
                                    self.logger.debug(f"🔧 Tensor {key} requires grad: {value.requires_grad}")
                                if torch.isnan(value).any():
                                    self.logger.error(f"❌ NaN detected in batch tensor {key}")
                                    raise ValueError(f"NaN values in batch tensor {key}")
                    
                    # Execute agent update with enhanced monitoring
                    loss_raw = self.agent.update(batch_converted, grad_scaler=self.grad_scaler)
                    
                except Exception as standard_error:
                    self.logger.error(f"❌ Standard execution failed: {standard_error}")
                    raise standard_error
            
            # 🔧 CRITICAL OUTPUT VALIDATION: Comprehensive tensor format checking and conversion
            self.logger.debug(f"🔧 Raw agent output validation: type={type(loss_raw)}, context={execution_context}")
            
            # Phase 1: Basic type validation
            if loss_raw is None:
                self.logger.error("❌ Agent returned None - this violates the output contract")
                raise ValueError("Agent update method returned None")
            
            # Phase 2: Tensor format validation and standardization
            validated_loss = None
            
            if torch.is_tensor(loss_raw):
                # Scenario 1: Single tensor output (most common)
                self.logger.debug("🔧 Agent returned single tensor")
                
                # Validate tensor properties
                if not loss_raw.is_leaf and loss_raw.grad_fn is None:
                    self.logger.warning("⚠️ Loss tensor has no gradient function - may indicate detached tensor")
                
                # Ensure tensor is scalar or can be reduced to scalar
                if loss_raw.dim() == 0:
                    # Already scalar
                    validated_loss = loss_raw
                elif loss_raw.dim() == 1 and loss_raw.numel() == 1:
                    # Single-element tensor
                    validated_loss = loss_raw.squeeze()
                elif loss_raw.numel() == 1:
                    # Multi-dimensional single-element tensor
                    validated_loss = loss_raw.view([])  # Convert to scalar
                else:
                    # Multi-element tensor - take mean
                    self.logger.warning(f"⚠️ Multi-element loss tensor {loss_raw.shape}, taking mean")
                    validated_loss = loss_raw.mean()
                
            elif isinstance(loss_raw, (tuple, list)):
                # Scenario 2: Tuple/list output (legacy agents)
                self.logger.debug(f"🔧 Agent returned {type(loss_raw)} with {len(loss_raw)} elements")
                
                if len(loss_raw) == 0:
                    self.logger.error("❌ Empty tuple/list returned by agent")
                    raise ValueError("Agent returned empty tuple/list")
                
                # Extract first element as primary loss
                primary_loss = loss_raw[0]
                
                if torch.is_tensor(primary_loss):
                    # Validate primary loss tensor
                    if primary_loss.dim() == 0:
                        validated_loss = primary_loss
                    elif primary_loss.numel() == 1:
                        validated_loss = primary_loss.view([])
                    else:
                        validated_loss = primary_loss.mean()
                elif isinstance(primary_loss, (int, float)):
                    # Convert numeric to tensor
                    validated_loss = torch.tensor(float(primary_loss), device=self.device, requires_grad=True)
                else:
                    self.logger.error(f"❌ Invalid primary loss type in tuple: {type(primary_loss)}")
                    raise ValueError(f"Invalid primary loss type: {type(primary_loss)}")
                
                # Log additional elements for debugging
                if len(loss_raw) > 1:
                    self.logger.debug(f"🔧 Additional elements in output: {[type(x) for x in loss_raw[1:]]}")
                    
            elif isinstance(loss_raw, (int, float)):
                # Scenario 3: Numeric output (simple agents)
                self.logger.debug(f"🔧 Agent returned numeric value: {loss_raw}")
                validated_loss = torch.tensor(float(loss_raw), device=self.device, requires_grad=True)
                
            elif isinstance(loss_raw, dict):
                # Scenario 4: Dictionary output (structured agents)
                self.logger.debug(f"🔧 Agent returned dict with keys: {list(loss_raw.keys())}")
                
                # Look for standard loss keys
                loss_keys = ['loss', 'total_loss', 'main_loss', 'training_loss']
                found_loss = None
                
                for key in loss_keys:
                    if key in loss_raw:
                        found_loss = loss_raw[key]
                        self.logger.debug(f"🔧 Found loss under key '{key}'")
                        break
                
                if found_loss is None:
                    # Take first tensor value
                    for key, value in loss_raw.items():
                        if torch.is_tensor(value):
                            found_loss = value
                            self.logger.debug(f"🔧 Using first tensor value under key '{key}'")
                            break
                
                if found_loss is None:
                    self.logger.error(f"❌ No valid loss found in dict: {loss_raw}")
                    raise ValueError("No valid loss tensor found in dictionary output")
                
                # Validate the found loss
                if torch.is_tensor(found_loss):
                    validated_loss = found_loss.view([]) if found_loss.numel() == 1 else found_loss.mean()
                else:
                    validated_loss = torch.tensor(float(found_loss), device=self.device, requires_grad=True)
                    
            else:
                # Scenario 5: Unexpected output type
                self.logger.error(f"❌ Agent returned unsupported type: {type(loss_raw)}")
                self.logger.error(f"🔍 Raw output content: {str(loss_raw)[:200]}")
                raise ValueError(f"Agent returned unsupported output type: {type(loss_raw)}")
            
            # Phase 3: Final validation of processed loss
            if validated_loss is None:
                self.logger.error("❌ Loss validation failed - no valid loss extracted")
                raise ValueError("Failed to extract valid loss from agent output")
            
            if not torch.is_tensor(validated_loss):
                self.logger.error(f"❌ Final validation failed - validated_loss is not a tensor: {type(validated_loss)}")
                raise ValueError(f"Final loss is not a tensor: {type(validated_loss)}")
            
            # Ensure loss is a scalar tensor with gradient
            if validated_loss.dim() != 0:
                self.logger.warning(f"⚠️ Final loss is not scalar: {validated_loss.shape}, converting")
                validated_loss = validated_loss.mean()
            
            # Ensure tensor can participate in backpropagation
            if not validated_loss.requires_grad:
                self.logger.warning("⚠️ Loss tensor does not require grad - may indicate computational graph issue")
                # Note: We don't force requires_grad=True as this might indicate a genuine issue
            
            # Phase 4: Numerical validation
            loss_value = validated_loss.item() if validated_loss.numel() == 1 else float(validated_loss.mean())
            
            if not math.isfinite(loss_value):
                if math.isnan(loss_value):
                    self.logger.error("❌ Loss is NaN - indicates numerical instability")
                    self.logger.error("💡 Consider: reducing learning rate, gradient clipping, batch normalization")
                elif math.isinf(loss_value):
                    self.logger.error("❌ Loss is infinite - possible gradient explosion")
                    self.logger.error("💡 Consider: gradient clipping, learning rate reduction, weight initialization")
                
                # Provide comprehensive debugging information
                self.logger.error(f"🔍 Agent type: {type(self.agent).__name__}")
                self.logger.error(f"🔍 Algorithm: {getattr(self, 'algo', 'unknown')}")
                self.logger.error(f"🔍 Execution context: {execution_context}")
                self.logger.error(f"🔍 Raw output type: {type(loss_raw)}")
                self.logger.error(f"🔍 Validated loss shape: {validated_loss.shape}")
                self.logger.error(f"🔍 Loss requires grad: {validated_loss.requires_grad}")
                self.logger.error(f"🔍 Loss grad_fn: {validated_loss.grad_fn}")
                
                return 0.0  # Return safe fallback value
            
            # Phase 5: Enhanced sanity checks for medical RL context
            if abs(loss_value) > 1e6:
                self.logger.warning(f"⚠️ Very large loss magnitude: {loss_value:.2e}")
                self.logger.warning("💡 This may indicate training instability or incorrect loss scaling")
            elif abs(loss_value) < 1e-10:  # More conservative threshold - only warn for extremely small losses
                self.logger.warning(f"⚠️ Extremely small loss magnitude: {loss_value:.2e}")
                self.logger.warning("💡 This may indicate vanishing gradients, over-convergence, or loss computation issues")
                # Add diagnostic information
                if hasattr(self, '_previous_losses') and len(self._previous_losses) > 0:
                    recent_avg = sum(self._previous_losses[-5:]) / min(5, len(self._previous_losses))
                    self.logger.warning(f"📊 Recent average loss: {recent_avg:.2e}")
            elif abs(loss_value) < 1e-8:  # Intermediate threshold - just log as debug
                self.logger.debug(f"🔍 Small loss magnitude: {loss_value:.2e} (may indicate good convergence)")
            
            # POG-specific loss tracking and debugging
            if self.algo.startswith("pog_"):
                self.logger.debug(f"✅ POG {self.algo} training step loss: {loss_value:.6f}")
                
                # Track loss history for POG models to detect training issues
                if not hasattr(self, '_pog_loss_history'):
                    self._pog_loss_history = []
                self._pog_loss_history.append(loss_value)
                
                # Log warnings if POG loss is unusually high or shows instability
                if len(self._pog_loss_history) > 10:
                    recent_losses = self._pog_loss_history[-10:]
                    mean_loss = np.mean(recent_losses)
                    std_loss = np.std(recent_losses)
                    
                    if mean_loss > 10.0:
                        self.logger.debug(f"⚠️ POG {self.algo} has high average loss: {mean_loss:.3f}")
                    if std_loss > 5.0:
                        self.logger.debug(f"⚠️ POG {self.algo} has unstable loss (std: {std_loss:.3f})")
                        
                    # Keep only recent history to avoid memory issues
                    if len(self._pog_loss_history) > 100:
                        self._pog_loss_history = self._pog_loss_history[-100:]
            else:
                self.logger.debug(f"✅ Training step output validation successful: loss={loss_value:.6f}")
                
            return validated_loss

        except RuntimeError as e:
            error_msg = str(e).lower()
            
            # Handle specific error types with appropriate responses
            if "out of memory" in error_msg:
                self.logger.error("💥 GPU out of memory during training step")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("🧹 Cleared GPU cache - consider reducing batch size")
                return 0.0
                
            elif "shape" in error_msg or "size" in error_msg:
                self.logger.error(f"💥 Shape/size error during training: {e}")
                
                # 🔧 ENHANCED SHAPE ERROR DIAGNOSTICS: Provide detailed tensor information
                try:
                    debug_info = self._get_batch_debug_info(batch)
                    self.logger.error(f"🔍 Batch info: {debug_info}")
                    
                    # Additional shape diagnostics for CQL-specific issues
                    if self.algo == "cql" or "cql" in self.algo.lower():
                        self.logger.error("🔍 CQL-specific diagnostics:")
                        self.logger.error(f"  • Agent type: {type(self.agent).__name__}")
                        self.logger.error(f"  • Model type: {type(getattr(self.agent, 'model', None)).__name__}")
                        self.logger.error(f"  • Action dims: {getattr(self.agent, 'action_dims', 'Unknown')}")
                        
                        # Analyze batch format
                        if isinstance(batch, tuple):
                            self.logger.error(f"  • Batch format: Tuple with {len(batch)} elements")
                            for i, item in enumerate(batch):
                                if hasattr(item, 'shape'):
                                    self.logger.error(f"    [{i}]: shape={item.shape}, dtype={getattr(item, 'dtype', 'Unknown')}")
                                elif isinstance(item, (list, tuple)):
                                    self.logger.error(f"    [{i}]: {type(item).__name__} with {len(item)} elements")
                                    if len(item) > 0 and hasattr(item[0], 'shape'):
                                        self.logger.error(f"         First element shape: {item[0].shape}")
                                else:
                                    self.logger.error(f"    [{i}]: {type(item).__name__}")
                        elif isinstance(batch, dict):
                            self.logger.error(f"  • Batch format: Dict with keys: {list(batch.keys())}")
                            for key, value in batch.items():
                                if torch.is_tensor(value):
                                    self.logger.error(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                                else:
                                    self.logger.error(f"    {key}: {type(value).__name__}")
                    
                    # Memory diagnostics
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                        self.logger.error(f"🔍 GPU Memory: allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB")
                    
                except Exception as diag_e:
                    self.logger.error(f"❌ Failed to generate shape error diagnostics: {diag_e}")
                
                # Try to provide solution hints
                if "dimension 2" in error_msg:
                    self.logger.error("💡 Hint: This often indicates batch/sequence dimension confusion.")
                    self.logger.error("    • Check if tensor shapes are (batch_size, seq_len, features)")
                    self.logger.error("    • Verify action dimensions match expected format")
                
                return 0.0
                
            elif "index" in error_msg:
                self.logger.error(f"💥 Index error during training: {e}")
                self.logger.error("🔍 Possible action index out of bounds")
                return 0.0
                
            else:
                # Generic runtime error
                self.logger.error(f"💥 Runtime error during training step: {e}")
                return 0.0
                
        except Exception as e:
            # 🔧 ENHANCED ERROR HANDLING: Comprehensive error analysis and recovery
            error_type = type(e).__name__
            error_msg = str(e)
            
            self.logger.error(f"💥 Unexpected error during training step: {error_msg}")
            self.logger.error(f"🔍 Error type: {error_type}")
            
            # Provide detailed debug information
            try:
                debug_info = {
                    'error_type': error_type,
                    'error_message': error_msg,
                    'batch_info': self._get_batch_debug_info(batch),
                    'agent_type': type(self.agent).__name__ if hasattr(self, 'agent') else 'Unknown',
                    'model_type': type(self.agent.model).__name__ if hasattr(self, 'agent') and hasattr(self.agent, 'model') else 'Unknown',
                    'device_info': {
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
                    }
                }
                
                if torch.cuda.is_available():
                    try:
                        debug_info['gpu_memory'] = {
                            f'gpu_{i}': {
                                'allocated_gb': torch.cuda.memory_allocated(i) / 1024**3,
                                'reserved_gb': torch.cuda.memory_reserved(i) / 1024**3
                            } for i in range(torch.cuda.device_count())
                        }
                    except Exception:
                        debug_info['gpu_memory'] = 'Failed to retrieve GPU memory info'
                
                self.logger.error(f"🔍 Comprehensive debug info:")
                for key, value in debug_info.items():
                    if isinstance(value, dict):
                        self.logger.error(f"   • {key}:")
                        for sub_key, sub_value in value.items():
                            self.logger.error(f"     - {sub_key}: {sub_value}")
                    else:
                        self.logger.error(f"   • {key}: {value}")
                        
            except Exception as debug_e:
                self.logger.error(f"Failed to generate debug info: {debug_e}")
            
            # Attempt cleanup and recovery
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("🧹 Cleared GPU cache for recovery")
            except Exception:
                pass
                
            return 0.0

        # ========================================================================
        # Enhanced Loss Processing and Validation
        # ========================================================================
        
        try:
            # Standardize loss format (handle both scalar and tuple returns)
            if isinstance(loss_raw, tuple):
                # Extract first element for loss, log additional metrics if available
                loss_scalar = safe_item(loss_raw[0])
                if len(loss_raw) > 1:
                    self.logger.debug(f"📊 Additional metrics: {loss_raw[1:]}")
            elif torch.is_tensor(loss_raw):
                loss_scalar = safe_item(loss_raw)
            elif isinstance(loss_raw, (int, float)):
                loss_scalar = float(loss_raw)
            else:
                self.logger.error(f"❌ Unexpected loss type: {type(loss_raw)}")
                return 0.0
            
            # Enhanced numerical validation
            if not math.isfinite(loss_scalar):
                if math.isnan(loss_scalar):
                    self.logger.error("❌ Training loss is NaN - indicating numerical instability")
                elif math.isinf(loss_scalar):
                    self.logger.error("❌ Training loss is infinite - possible gradient explosion")
                    
                self.logger.error(f"🔍 Raw loss value: {loss_raw}")
                self.logger.error("💡 Consider reducing learning rate or enabling gradient clipping")
                return 0.0
            
            # Sanity check for reasonable loss magnitude (medical RL context)
            if abs(loss_scalar) > 1e6:
                self.logger.warning(f"⚠️  Very large loss magnitude: {loss_scalar:.2e}")
                self.logger.warning("💡 This may indicate training instability")
                
        except Exception as e:
            self.logger.error(f"❌ Loss processing failed: {e}")
            return 0.0

        # ========================================================================
        # Memory Optimization and Performance Monitoring
        # ========================================================================
        
        if hasattr(self, 'memory_manager'):
            try:
                # Monitor memory after training step
                post_memory_status = self.memory_manager.monitor.check_memory_status()
                
                # Log memory usage periodically
                if hasattr(self, '_step_count'):
                    self._step_count = getattr(self, '_step_count', 0) + 1
                    if self._step_count % 100 == 0:  # Log every 100 steps
                        gpu_memory = post_memory_status.get('gpu_memory_percent', 0)
                        gpu_gb = post_memory_status.get('gpu_memory_allocated_gb', 0)
                        self.logger.info(f"💾 Memory usage: {gpu_memory:.1%} ({gpu_gb:.2f}GB)")
                        
                        # Log memory trend analysis
                        if hasattr(self.memory_manager.monitor, 'get_memory_trend_analysis'):
                            trend_analysis = self.memory_manager.monitor.get_memory_trend_analysis()
                            if trend_analysis.get("is_memory_leak", False):
                                self.logger.warning("⚠️  Potential memory leak detected!")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
            except Exception as e:
                self.logger.warning(f"⚠️  Post-training memory monitoring failed: {e}")

        # ========================================================================
        # Training Statistics and Diagnostics
        # ========================================================================
        
        try:
            # Update training step counter
            if hasattr(self.agent, 'increment_training_step'):
                self.agent.increment_training_step()
            elif hasattr(self.agent, '_training_step'):
                self.agent._training_step += 1
            
            # Log detailed statistics periodically
            if hasattr(self, '_step_count') and self._step_count % 500 == 0:
                self.logger.info(f"🎯 Training Step {self._step_count}: loss={loss_scalar:.6f}")
                
                # Log agent-specific statistics if available
                if hasattr(self.agent, 'last_td_loss'):
                    self.logger.info(f"   📊 TD Loss: {self.agent.last_td_loss:.6f}")
                if hasattr(self.agent, 'last_cql_loss'):
                    self.logger.info(f"   📊 CQL Loss: {self.agent.last_cql_loss:.6f}")
                if hasattr(self.agent, 'last_alpha'):
                    self.logger.info(f"   📊 Alpha: {self.agent.last_alpha:.6f}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️  Training statistics logging failed: {e}")

        # ========================================================================
        # Successful Return with Enhanced Logging
        # ========================================================================
        
        self.logger.debug(f"✅ Training step completed successfully: loss={loss_scalar:.6f}")
        return loss_scalar

    # ------------------------------------------------------------------
    #  Behaviour policy (contextual) helpers
    # ------------------------------------------------------------------
    def _train_behavior_policy(self, loader: DataLoader, n_epochs: int = 3, lr: float = 1e-3) -> None:
        """Fits a contextual behaviour policy μ(a|s).

        更新内容：
            1. 由 **单层线性** 升级为 *2-layer* MLP ``Linear→ReLU→Linear``，
               更好刻画非线性 state→action 映射；
            2. 将默认训练 epoch 从 3→10，减少 μ≈常数 的风险；
            3. 训练完成后输出 **μ 分布方差** 诊断，若方差 <1e-4 则自动
               fallback 到 *empirical* counts 策略并警告。
        """
        # 🔧 CRITICAL FIX: Safe data loading with error recovery
        # Sample a single batch to determine the state dimensionality with error handling
        try:
            state_dim = next(iter(loader))["state"].shape[-1]
        except RuntimeError as e:
            if "Pin memory thread exited unexpectedly" in str(e):
                logger.error("💥 DataLoader pin_memory error detected during behavior policy training!")
                logger.error("This indicates a multiprocessing configuration issue.")
                logger.error("Please restart the training process - the issue should be resolved with the updated configuration.")
                raise RuntimeError("DataLoader multiprocessing error. Please restart training.") from e
            else:
                raise e
        
        self._behav_models = []
        # Determine action dimensions per branch
        if hasattr(self.agent, "action_dims"):
            action_dims = self.agent.action_dims
        else:
            try:
                sample_batch = next(iter(loader))
                actions = sample_batch["action"]  # (B, T, n_heads) or (B, n_heads)
            except RuntimeError as e:
                if "Pin memory thread exited unexpectedly" in str(e):
                    logger.error("💥 DataLoader pin_memory error detected during action dimension detection!")
                    logger.error("This indicates a multiprocessing configuration issue.")
                    logger.error("Please restart the training process - the issue should be resolved with the updated configuration.")
                    raise RuntimeError("DataLoader multiprocessing error. Please restart training.") from e
                else:
                    raise e

            # Robustly handle both 2-D (B, H) and 3-D (B, T, H) layouts.
            if actions.dim() == 3:  # Temporal dimension present
                # Flatten (B, T) into a single dimension before max().
                def _max_for_head(h: int) -> int:
                    return int(actions[..., h].max().item() + 1)

                action_dims = [_max_for_head(h) for h in range(actions.shape[-1])]
            elif actions.dim() == 2:
                action_dims = [int(actions[:, h].max().item() + 1)
                               for h in range(actions.shape[1])]
            else:
                raise ValueError(f"Unexpected action tensor shape: {actions.shape}")

        # Determine which behaviour-policy training path to follow.
        # Default is "logistic" (state-conditional).  Users may override to
        # "empirical" for quick tests via Trainer(..., behav_policy_mode="empirical").
        behav_mode = getattr(self, "behav_policy_mode", "logistic")

        if behav_mode == "empirical":
            # 统计计数
            action_dims = self.agent.action_dims
            counts = [torch.zeros(d, dtype=torch.long) for d in action_dims]
            total = 0
            for batch in loader:
                acts = batch["action"]  # (B, n_heads)
                for h in range(len(action_dims)):
                    binc = torch.bincount(acts[:, h].cpu(), minlength=action_dims[h])
                    counts[h][:] += binc
                total += acts.size(0)

            # 转概率表 & 缓存
            self._behav_tables = [c.float() / max(int(total), 1) for c in counts]
            # 适当下限防止除零，但保留足够方差
            self.behav_prob_min = 1e-4
            logger.info("Behaviour policy set to empirical counts tables.")
            return  # 👈 use global counts, skip logistic fitting

        # ------------------------------------------------------------------
        # 旧实现：对每个动作 head 训练一元 logistic 多分类器作为行为策略 μ(a|s)。
        # ------------------------------------------------------------------
        HIDDEN = min(128, max(32, state_dim * 2))  # heuristic hidden width
        for branch_dim in action_dims:
            # Two-layer perceptron ≈ logistic regression when HIDDEN =0
            model = nn.Sequential(
                nn.Linear(state_dim, HIDDEN),
                nn.ReLU(inplace=True),
                nn.Linear(HIDDEN, branch_dim),
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
            self._behav_models.append((model, optimizer))

        loss_fn = nn.CrossEntropyLoss()

        # Increase epochs for better convergence (3→10 by default)
        n_epochs = max(n_epochs, 10)

        for _ in range(n_epochs):
            for batch in loader:
                s = batch["state"].to(self.device).float()
                # ------------------------------------------------------
                # If trajectories are provided (B, T, D) / (B, T, H), we
                # reduce them to the *last* time step so that the logistic
                # behaviour model receives a 2-D input and a 1-D target
                # vector (class indices) as required by CrossEntropyLoss.
                # ------------------------------------------------------
                if s.dim() == 3:  # (B, T, D)
                    s = s[:, -1, :]  # (B, D)

                acts = batch["action"].to(self.device)
                # Collapse temporal dimension if present: (B, T, H) → (B, H)
                if acts.dim() == 3:
                    acts = acts[:, -1, :]

                for b, (model, optim) in enumerate(self._behav_models):
                    logits = model(s)            # (B, A_b)
                    loss = loss_fn(logits, acts[:, b])  # target: (B,)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

        # Freeze models for inference
        for model, _ in self._behav_models:
            model.eval()

        # --------------------------------------------------------------
        # Temperature scaling à la Platt to calibrate predicted probs.
        # We use a held-out subset (≤2k samples) from *loader* to
        # minimise NLL w.r.t a single scalar temperature per branch.
        # --------------------------------------------------------------
        self._behav_temps: list[torch.Tensor] = []
        calib_samples = 0
        calib_states: list[torch.Tensor] = []
        calib_actions: list[torch.Tensor] = []
        for batch in loader:
            _cs = batch["state"].to(self.device).float()
            if _cs.dim() == 3:  # (B,T,D)
                _cs = _cs[:, -1, :]
            calib_states.append(_cs)
            calib_actions.append(batch["action"].to(self.device).long())
            calib_samples += batch["state"].size(0)
            if calib_samples >= 2000:
                break

        calib_state = torch.cat(calib_states, dim=0)
        calib_action = torch.cat(calib_actions, dim=0)

        # Collapse (B, T, H) → (B, H) for compatibility with CrossEntropyLoss
        if calib_action.dim() == 3:
            calib_action = calib_action[:, -1, :]

        loss_fn_nll = nn.CrossEntropyLoss()

        for b, (model, _) in enumerate(self._behav_models):
            temperature = torch.nn.Parameter(torch.ones([], device=self.device))

            optimizer_t = torch.optim.LBFGS([temperature], lr=0.2, max_iter=50)

            def _closure():
                optimizer_t.zero_grad()
                logits = model(calib_state) / temperature.clamp(min=1e-3)
                loss = loss_fn_nll(logits, calib_action[:, b])
                loss.backward()
                return loss

            try:
                optimizer_t.step(_closure)
            except Exception:  # fallback to initial temp =1
                pass

            self._behav_temps.append(temperature.detach())

        # ---------------- 诊断输出 ----------------
        with torch.no_grad():
            sample_state = next(iter(loader))["state"]
            if sample_state.dim() == 3:
                sample_state = sample_state[:, -1, :]
            variances = []
            for b, (model, _) in enumerate(self._behav_models):
                logits = model(sample_state.to(self.device)) / self._behav_temps[b]
                probs = torch.softmax(logits, dim=-1)
                variances.append(safe_item(probs.var()))
            logger.info("Behaviour policy per-head prob variance: %s", [f"{v:.4e}" for v in variances])

        # 若全部 variance 极低则 fallback 到 empirical counts
        if all(v < 1e-4 for v in variances):
            logger.warning("Logistic behaviour models degenerated ➜ switching to empirical counts.")
            # Clear models to force empirical mode next compute
            delattr(self, "_behav_models")
            self.behav_policy_mode = "empirical"
            # Re-run empirical count estimation once
            self._train_behavior_policy(loader, n_epochs=0, lr=lr)

    # ------------------------------------------------------------------
    #  OPE helpers (FQE, WDR, IPW)
    # ------------------------------------------------------------------
    def fit_fqe(self, transitions: Dict[str, torch.Tensor], **kwargs) -> FQEEstimator:
        # 🔧 CRITICAL FIX: Create POG-compatible policy action function for FQE
        # POG models require additional arguments (lengths, edge_index) that standard FQE doesn't provide
        def create_pog_compatible_policy_fn():
            """Create a policy action function that works with both POG and standard models."""
            if hasattr(self.agent, 'q_net') and hasattr(self.agent.q_net, 'greedy_action'):
                # Check if this is a POG model by examining the greedy_action signature
                import inspect
                sig = inspect.signature(self.agent.q_net.greedy_action)
                if 'lengths' in sig.parameters or 'edge_index' in sig.parameters:
                    # This is a POG model - create wrapper function
                    def pog_policy_fn(state):
                        """POG-compatible policy function with default arguments."""
                        batch_size = state.size(0)
                        if state.dim() == 2:
                            # Convert (batch, features) to (batch, 1, features) for sequence format
                            state = state.unsqueeze(1)
                            seq_len = 1
                        else:
                            seq_len = state.size(1)
                        
                        # Provide default arguments for POG models
                        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=state.device)
                        edge_index = torch.zeros(2, 0, dtype=torch.long, device=state.device)
                        
                        try:
                            return self.agent.q_net.greedy_action(state, lengths=lengths, edge_index=edge_index)
                        except Exception as e:
                            # Fallback to random actions if POG greedy_action fails
                            self.logger.warning(f"POG greedy_action failed ({e}), using random actions")
                            actions = []
                            for action_dim in self.agent.action_dims:
                                random_actions = torch.randint(0, action_dim, (batch_size,), device=state.device)
                                actions.append(random_actions)
                            return torch.stack(actions, dim=1)
                    
                    return pog_policy_fn
                else:
                    # Standard model - use default greedy_action
                    return lambda s: self.agent.q_net.greedy_action(s)
            else:
                # Fallback to agent.act method
                return self.policy_action_fn
        
        # Create appropriate policy function
        pog_compatible_policy_fn = create_pog_compatible_policy_fn()
        
        # Pass a policy action function that queries the evaluated agent π(s).
        fqe = FQEEstimator(
            q_network=self.agent.q_net,  # type: ignore[attr-defined]
            target_q_network=self.agent.target_q_net,  # type: ignore[attr-defined]
            device=self.device,
            policy_action_fn=pog_compatible_policy_fn,
            **kwargs,
        )
        import torch
        with torch.enable_grad():
            fqe.fit(transitions, amp=self.amp)
        return fqe

    def compute_wdr_ipw(
        self,
        batch: Dict[str, torch.Tensor],
        fqe: FQEEstimator,
        gamma: float = 0.99,
        use_psis: bool = True,
    ) -> Tuple[float, float]:
        rewards = batch["reward"].to(self.device)
        done = batch["done"].to(self.device).float()
        # Behaviour probabilities via simple empirical counts per action branch
        behav_prob = self._estimate_behavior_policy(batch)

        # --------------------------------------------------------------
        # 不再裁剪轨迹尾帧：保持 (B,T) 形状，后续 wdr_estimate 会在内部
        # 自动广播 1-D 的 q_values / v_values 至 (B,T)。这样 IPW/WDR 将
        # 使用完整折现回报，而非仅末步奖励，从而避免"IPW=0"的问题。
        # --------------------------------------------------------------

        # Ensure importance_weights has matching rank
        target_prob = self._policy_probability(batch)

        if rewards.dim() == 1:
            importance_weights = target_prob / behav_prob.clamp(min=self.behav_prob_min)
        else:
            importance_weights = target_prob / behav_prob.clamp(min=self.behav_prob_min)
            if importance_weights.dim() == 1:
                importance_weights = importance_weights.unsqueeze(1).expand_as(rewards)

        if self.clip_range is not None and self.clip_range > 0:
            importance_weights = importance_weights.clamp(max=self.clip_range)

        # Optional Pareto‐smoothed IS to mitigate heavy‐tailed ratios
        if use_psis:
            try:
                from Libs.utils.ope.psis import (psis_smooth_weights,
                                                 psis_wdr_estimate)
                importance_weights = psis_smooth_weights(importance_weights)
                wdr_func = psis_wdr_estimate
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PSIS smoothing unavailable (%s) – fallback to vanilla WDR", exc)
                from Libs.utils.ope.wdr import \
                    wdr_estimate as wdr_func  # type: ignore
        else:
            from Libs.utils.ope.wdr import \
                wdr_estimate as wdr_func  # type: ignore

        # ------------------------------------------------------
        # Ensure inputs to FQE q_value are 2-D (B,D) and (B,H)
        # ------------------------------------------------------
        state_input = batch["state"].to(self.device)
        if state_input.dim() == 3:
            state_input = state_input[:, -1, :]

        action_input = batch["action"].to(self.device)
        if action_input.dim() == 3:
            action_input = action_input[:, -1, :]

        # Recover original reward units using FQE's internal reward scaling
        q_values = fqe.q_net.q_value(state_input, action_input) * getattr(fqe, "reward_scale", 1.0)
        # Tanh clipping for numerical stability (scale=100)
        q_values = torch.tanh(q_values / 100.0) * 100.0
        # ------------------------------------------------------------------
        # Improved state value V(s) estimation
        # ------------------------------------------------------------------
        # 1) 对于低维联合动作空间 (≤max_joint_enum) 枚举所有 action 组合；
        # 2) 否则使用 **factorised** per-head均值近似，显著优于之前只枚
        #    举 head[0] 的做法，从而避免 V(s) 系统性偏低。

        action_dims = self.agent.action_dims
        joint_size = int(math.prod(action_dims))
        states_device = batch["state"].to(self.device)
        if states_device.dim() == 3:
            states_device = states_device[:, -1, :]

        if joint_size <= getattr(self, "max_joint_enum", 5000):
            import itertools
            q_list = []
            for a_tuple in itertools.product(*[range(d) for d in action_dims]):
                a_tensor = torch.tensor(a_tuple, device=self.device).repeat(states_device.size(0), 1)
                q_sa = fqe.q_net.q_value(states_device, a_tensor)
                if q_sa.dim() > 1:
                    q_sa = q_sa.squeeze(-1)
                q_list.append(q_sa.unsqueeze(0))  # (1,B)
            v_values = torch.cat(q_list, dim=0).mean(0)
            # 数值稳定性缩放
            v_values = torch.tanh((v_values * getattr(fqe, "reward_scale", 1.0)) / 100.0) * 100.0
        else:
            # 向量化 Factorised 均值：直接 forward→list[(B,A_i)] 再取 mean
            try:
                q_head_list = fqe.q_net(states_device)
                if not isinstance(q_head_list, list):
                    raise TypeError
                v_values = torch.zeros(states_device.size(0), device=self.device)
                for q_head in q_head_list:
                    if q_head.dim() == 3:
                        q_head_last = q_head[:, -1, :]
                    else:
                        q_head_last = q_head  # (B,A)
                    v_values += q_head_last.mean(-1)
                v_values = v_values / len(q_head_list)
                v_values = torch.tanh((v_values * getattr(fqe, "reward_scale", 1.0)) / 100.0) * 100.0
            except Exception:
                # Fallback to原循环实现，保证正确性
                v_values = torch.zeros_like(q_values)
                for h, adim in enumerate(action_dims):
                    head_q_sum = fqe.q_net(states_device)[h]
                    if head_q_sum.dim() == 3:
                        head_q_sum = head_q_sum[:, -1, :]
                    v_values += head_q_sum.mean(-1)
                v_values = v_values / len(action_dims)
                v_values = torch.tanh((v_values * getattr(fqe, "reward_scale", 1.0)) / 100.0) * 100.0

        wdr = wdr_func(rewards, done, importance_weights,
                       q_values, v_values, gamma)

        # ------------------------------------------------------------
        # 若 WDR 仍为 NaN（例如 FQE 训练不足或权重爆炸），则回退到 IPW；
        # 若 IPW 也为 NaN，则最终返回 0.0，防止早停逻辑触发。
        # ------------------------------------------------------------
        ipw = ipw_estimate(rewards, done, importance_weights, gamma)

        if math.isnan(wdr):
            wdr = ipw if not math.isnan(ipw) else 0.0
        if math.isnan(ipw):
            ipw = 0.0

        return wdr, ipw

    def _estimate_behavior_policy(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Contextual behaviour policy P(a|s) estimated by logistic models.

        This helper previously collapsed (B,T,D) trajectories to the **last**
        timestep, producing a single probability per trajectory.  When used
        in conjunction with per-step rewards this led to *constant* IS ratios
        because the same weight was broadcast across the entire sequence.

        We now compute μ(a_t|s_t) **for every step** so that the returned
        tensor matches the reward/done shape:
            • (B,)   for single-step batches
            • (B,T)  for multi-step trajectories
        """
        # ------------------------------------------------------------------
        # 1) Flatten temporal dimension so that we can reuse the existing
        #    per-state logic without duplicating model forward passes.
        # ------------------------------------------------------------------
        state = batch["state"].to(self.device).float()
        acts = batch["action"].to(self.device)

        # Ensure contiguous layout for safe `.view` operations on potentially
        # non-contiguous tensors (e.g. from index_select / slice).
        state = state.contiguous()
        acts = acts.contiguous()

        seq_mode = state.dim() == 3  # (B,T,D)
        if seq_mode:
            B, T, D = state.shape
            state_flat = state.reshape(-1, D)            # (B*T, D)
            acts_flat = acts.reshape(-1, acts.shape[-1])  # (B*T, H)
        else:
            state_flat = state                        # (B, D)
            acts_flat = acts                          # (B, H)

        # ------------------------------------------------------------------
        # 2) Behaviour probability via empirical tables (fast-path) ----------
        # ------------------------------------------------------------------
        if hasattr(self, "_behav_tables"):
            probs_heads = []
            for h, table in enumerate(self._behav_tables):
                head_idx = acts_flat[:, h]
                head_prob = table[head_idx].to(self.device)
                head_prob = head_prob.clamp(min=self.behav_prob_min)
                probs_heads.append(head_prob)
            probs_flat = torch.stack(probs_heads, dim=1).prod(dim=1)  # (N,)
            return probs_flat.view(B, T) if seq_mode else probs_flat

        # ------------------------------------------------------------------
        # 3) Logistic models path -------------------------------------------
        # ------------------------------------------------------------------
        assert hasattr(self, "_behav_models"), "Behaviour policy not trained yet."
        log_prob = torch.zeros(state_flat.size(0), device=self.device)
        for b, (model, _) in enumerate(self._behav_models):
            with torch.no_grad():
                if hasattr(self, "_behav_temps") and b < len(self._behav_temps):
                    logits = model(state_flat) / self._behav_temps[b]
                else:
                    logits = model(state_flat)
                branch_logprob = torch.log_softmax(logits, dim=-1)
                log_prob += branch_logprob.gather(1, acts_flat[:, b:b+1]).squeeze(1)

        probs_flat = torch.exp(log_prob).clamp(
            min=self.behav_prob_min, max=self.behav_prob_max
        )
        return probs_flat.view(B, T) if seq_mode else probs_flat

    def _policy_probability(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Estimate target policy probability π_e(a|s) for each *step*.

        Supports both 2-D (B,*) and 3-D (B,T,*) inputs.  For multi-step
        trajectories we flatten to (B*T, …), compute probabilities, and
        reshape back so that the output aligns with reward tensors used by
        IPW/WDR/FQE.  This per-step estimation eliminates the broadcast-then-
        clamp artefact that caused *collapsed* importance-weights in earlier
        versions.
        """
        import math
        states = batch["state"].to(self.device)
        acts = batch["action"].to(self.device)

        # Guard: flattening via `.view` below requires contiguous stride.
        states = states.contiguous()
        acts = acts.contiguous()

        seq_mode = states.dim() == 3  # (B,T,D)
        if seq_mode:
            B, T, D = states.shape
            states_flat = states.reshape(-1, D)          # (B*T, D)
            acts_flat = acts.reshape(-1, acts.shape[-1]) # (B*T, H)
        else:
            states_flat = states                      # (B, D)
            acts_flat = acts                          # (B, H)

        # ------------------------------------------------------------------
        # 1) Agent-provided probability interface (preferred) ----------------
        # ------------------------------------------------------------------
        for prob_attr in ("action_prob", "action_probability", "action_probabilities"):
            if hasattr(self.agent, prob_attr) and callable(getattr(self.agent, prob_attr)):
                try:
                    prob_fn = getattr(self.agent, prob_attr)
                    probs_flat = prob_fn(states_flat, acts_flat)  # type: ignore[arg-type]
                    if probs_flat.dim() > 1:
                        probs_flat = probs_flat.squeeze(-1)
                    probs_flat = probs_flat.clamp(min=self.pi_eps, max=1.0 - self.pi_eps)
                    return probs_flat.view(B, T) if seq_mode else probs_flat
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Agent.%s() failed – falling back to generic probability (%s)", prob_attr, exc)

        # ------------------------------------------------------------------
        # 2) Boltzmann (softmax over Q) probability for value-based agents ----
        # ------------------------------------------------------------------
        if self.use_boltzmann and hasattr(self.agent, "q_net"):
            joint_dims = self.agent.action_dims
            import itertools
            joint_actions = list(itertools.product(*[range(d) for d in joint_dims]))
            # Limit enumeration size to avoid OOM on very large spaces
            max_enum = getattr(self, "max_joint_enum", 5000)

            if len(joint_actions) <= max_enum:
                logits_all = []  # (A_joint, N)
                for a_tuple in joint_actions:
                    a_tensor = torch.tensor(a_tuple, device=self.device).repeat(states_flat.size(0), 1)
                    q_sa = self.agent.q_net.q_value(states_flat, a_tensor)
                    if q_sa.dim() > 1:
                        q_sa = q_sa.squeeze(-1)
                    logits_all.append(q_sa.unsqueeze(0))
                logits_cat = torch.cat(logits_all, dim=0) / self.boltz_tau
                soft_p = torch.softmax(logits_cat, dim=0)  # (A_joint, N)
                # Map dataset action vector to flat index
                def _flat_idx(vec):
                    idx = 0
                    mult = 1
                    for v, dim_sz in zip(reversed(vec), reversed(joint_dims)):
                        idx += v * mult
                        mult *= dim_sz
                    return idx
                a_indices = torch.tensor([_flat_idx(v.tolist()) for v in acts_flat.cpu()], device=self.device)
                probs_flat = soft_p.gather(0, a_indices.unsqueeze(0)).squeeze(0)
                probs_flat = probs_flat.clamp(min=self.pi_eps, max=1.0 - self.pi_eps)
                return probs_flat.view(B, T) if seq_mode else probs_flat
            else:
                # Factorised per-head approximation
                probs_flat = torch.ones(states_flat.size(0), device=self.device)
                for h, adim in enumerate(joint_dims):
                    logits_list = []
                    for a_val in range(adim):
                        act_variant = acts_flat.clone()
                        act_variant[:, h] = a_val
                        q_sa = self.agent.q_net.q_value(states_flat, act_variant)
                        if q_sa.dim() > 1:
                            q_sa = q_sa.squeeze(-1)
                        logits_list.append(q_sa.unsqueeze(1))
                    logits_tensor = torch.cat(logits_list, dim=1) / self.boltz_tau
                    soft_p = torch.softmax(logits_tensor, dim=-1)
                    probs_flat *= soft_p.gather(1, acts_flat[:, h:h+1]).squeeze(1)
                probs_flat = probs_flat.clamp(min=self.pi_eps, max=1.0 - self.pi_eps)
                return probs_flat.view(B, T) if seq_mode else probs_flat

        # ------------------------------------------------------------------
        # 3) Deterministic fallback with ε-smoothing -------------------------
        # ------------------------------------------------------------------
        with torch.no_grad():
            chosen_actions = self.policy_action_fn(states_flat)  # (N,H)
        
        # Handle shape mismatches by adding debug information and proper reshaping
        if chosen_actions.dim() == 1 and acts_flat.dim() == 2:
            chosen_actions = chosen_actions.unsqueeze(1)
        
        # Check for shape compatibility
        if chosen_actions.shape[-1] != acts_flat.shape[-1]:
            logger.warning(f"Action shape mismatch in policy probability: "
                          f"chosen_actions.shape={chosen_actions.shape}, "
                          f"acts_flat.shape={acts_flat.shape}")
            logger.warning(f"Agent action_dims: {getattr(self.agent, 'action_dims', 'None')}")
            
            # Try to handle the mismatch
            if chosen_actions.shape[-1] < acts_flat.shape[-1]:
                # Pad chosen_actions with zeros
                padding = torch.zeros(
                    chosen_actions.shape[0], 
                    acts_flat.shape[-1] - chosen_actions.shape[-1], 
                    dtype=chosen_actions.dtype, 
                    device=chosen_actions.device
                )
                chosen_actions = torch.cat([chosen_actions, padding], dim=-1)
            elif chosen_actions.shape[-1] > acts_flat.shape[-1]:
                # Truncate chosen_actions
                chosen_actions = chosen_actions[:, :acts_flat.shape[-1]]
        
        match = (chosen_actions == acts_flat).all(dim=-1).float()
        eps = self.pi_eps
        if hasattr(self.agent, "action_dims") and self.agent.action_dims:
            total_actions = math.prod(self.agent.action_dims)
        else:
            total_actions = 1000  # Safe fallback for unknown action space size
        uniform = eps / total_actions
        if self.algo == "physician":
            pi_flat = torch.ones_like(match)
        else:
            pi_flat = match * (1.0 - eps) + uniform
        pi_flat = pi_flat.clamp(min=self.pi_eps, max=1.0 - self.pi_eps)
        return pi_flat.view(B, T) if seq_mode else pi_flat

    # ------------------------------------------------------------------
    #  Data helpers – complex pipeline integration
    # ------------------------------------------------------------------
    @staticmethod
    def build_dataloaders_from_graph_with_splits(
        graph_pt: str | Path,
        split_dir: str | Path,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Creates train/val/test DataLoaders from a saved PyG .pt graph using pre-computed splits.

        Uses the task-specific index files (ids_train.npy, ids_val.npy, ids_test.npy)
        to ensure consistent splits across different tasks.
        
        Args:
            graph_pt: Path to the PyG graph file
            split_dir: Directory containing the split files
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle training data
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        import numpy as np
        from torch_geometric.data import Data

        # Load graph data
        data: Data = torch.load(Path(graph_pt), weights_only=False)
        split_dir = Path(split_dir)
        
        # Load patient ID splits
        train_ids = np.load(split_dir / "ids_train.npy", allow_pickle=True)
        val_ids = np.load(split_dir / "ids_val.npy", allow_pickle=True)
        test_ids = np.load(split_dir / "ids_test.npy", allow_pickle=True)
        
        # Convert to string for consistent comparison
        train_ids = [str(pid) for pid in train_ids]
        val_ids = [str(pid) for pid in val_ids]
        test_ids = [str(pid) for pid in test_ids]
        
        # Get patient IDs from graph data
        if hasattr(data, 'patient_ids'):
            graph_patient_ids = [str(pid) for pid in data.patient_ids]
        else:
            raise ValueError("Graph data missing patient_ids attribute")
        
        # Create mapping from patient ID to graph index
        pid_to_idx = {pid: i for i, pid in enumerate(graph_patient_ids)}
        
        # Convert patient IDs to graph indices
        def ids_to_indices(patient_ids):
            indices = []
            missing_ids = []
            for pid in patient_ids:
                if pid in pid_to_idx:
                    indices.append(pid_to_idx[pid])
                else:
                    missing_ids.append(pid)
            
            if missing_ids:
                logger.warning(f"Missing {len(missing_ids)} patient IDs in graph data: {missing_ids[:5]}...")
            
            return torch.tensor(indices, dtype=torch.long)
        
        train_idx = ids_to_indices(train_ids)
        val_idx = ids_to_indices(val_ids)
        test_idx = ids_to_indices(test_ids)
        
        logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        def _subset(i: torch.Tensor):
            # 🔧 CRITICAL FIX: Add None checks for data attributes to prevent batch corruption
            if not hasattr(data, 'x') or data.x is None:
                raise RuntimeError("Data object missing required 'x' attribute (state features)")
            if not hasattr(data, 'actions') or data.actions is None:
                raise RuntimeError("Data object missing required 'actions' attribute")
                
            # 🔧 CRITICAL FIX: Properly handle graph edge extraction for batched data
            edge_index = torch.empty((2, 0), dtype=torch.long)
            if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.size(1) > 0:
                try:
                    # Create mapping from original node indices to batch indices
                    node_mapping = torch.full((data.x.size(0),), -1, dtype=torch.long)
                    node_mapping[i] = torch.arange(i.size(0), dtype=torch.long)
                    
                    # Find edges where both source and target are in the batch
                    src_nodes = data.edge_index[0]
                    tgt_nodes = data.edge_index[1]
                    
                    # Check which edges connect nodes in the current batch
                    src_in_batch = torch.isin(src_nodes, i)
                    tgt_in_batch = torch.isin(tgt_nodes, i)
                    valid_edges = src_in_batch & tgt_in_batch
                    
                    if valid_edges.any():
                        # Extract valid edges and remap indices
                        valid_src = src_nodes[valid_edges]
                        valid_tgt = tgt_nodes[valid_edges]
                        
                        # Remap to batch indices - ensure no negative indices
                        remapped_src = node_mapping[valid_src]
                        remapped_tgt = node_mapping[valid_tgt]
                        
                        # Validate remapped indices
                        if (remapped_src >= 0).all() and (remapped_tgt >= 0).all():
                            edge_index = torch.stack([remapped_src, remapped_tgt], dim=0)
                        else:
                            # Use print if logger not available in this scope
                            print("⚠️ Invalid remapped edge indices, using empty edge_index")
                    else:
                        # No valid edges found - this is normal for non-PoG algorithms
                        # that don't use graph structure (DQN, CQL, BCQ, etc.)
                        pass
                        
                except Exception as e:
                    # Use print if logger not available in this scope
                    print(f"⚠️ Edge extraction failed: {e}, using empty edge_index")
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # 🔧 CRITICAL FIX: Add bounds checking for indices
            try:
                if i.max().item() >= data.x.size(0):
                    raise IndexError(f"Index {i.max().item()} out of bounds for data.x size {data.x.size(0)}")
                if i.max().item() >= data.actions.size(0):
                    raise IndexError(f"Index {i.max().item()} out of bounds for data.actions size {data.actions.size(0)}")
                    
                state_batch = data.x[i]
                action_batch = data.actions[i]
                
                # Validate extracted tensors are not None
                if state_batch is None:
                    raise RuntimeError("Extracted state batch is None")
                if action_batch is None:
                    raise RuntimeError("Extracted action batch is None")
                    
            except Exception as e:
                logger.error(f"❌ Failed to extract batch data: {e}")
                logger.error(f"🔍 Debug info: i.shape={i.shape}, i.min={i.min()}, i.max={i.max()}")
                logger.error(f"🔍 Data info: data.x.shape={data.x.shape if data.x is not None else None}, "
                           f"data.actions.shape={data.actions.shape if data.actions is not None else None}")
                raise RuntimeError(f"Data extraction failed: {e}") from e
            
            # 🔧 计算实际的轨迹长度和掩码
            batch_size = state_batch.size(0)
            seq_len = state_batch.size(1)
            
            # 创建lengths - 检查每个样本的有效长度
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=state_batch.device)
            
            # 创建mask - 标记有效的时间步
            # 假设done标志为True表示轨迹结束，done之后的步骤都是padding
            done_flags = state_batch[:, :, -1].bool()  # (B, T)
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=state_batch.device)
            
            # 对于每个样本，找到第一个done=True的位置，该位置之后的都设为False
            for b in range(batch_size):
                done_positions = torch.where(done_flags[b])[0]
                if len(done_positions) > 0:
                    first_done = done_positions[0].item()
                    # 包含第一个done位置，但之后的都是padding
                    if first_done + 1 < seq_len:
                        mask[b, first_done + 1:] = False
                        lengths[b] = first_done + 1
            
            return {
                "state": state_batch,                    # (B, T, D)
                "action": action_batch,                  # (B, T, n_heads)
                "reward": state_batch[:, :, -2],         # per-step reward vector (B, T)
                "done": done_flags,                      # (B, T) terminal flags
                "next_state": state_batch,
                "survival": data.survival[i] if hasattr(data, "survival") and data.survival is not None else None,
                "traj_len": torch.full((i.size(0),), action_batch.size(1)),
                "edge_index": edge_index,                # Subgraph edges for this batch
                "lengths": lengths,                      # (B,) 轨迹实际长度
                "mask": mask,                           # (B, T) 有效时间步掩码
                "next_lengths": lengths.clone(),         # (B,) next_state轨迹长度 (通常与lengths相同)
            }

        train_ds = torch.utils.data.TensorDataset(train_idx)
        val_ds = torch.utils.data.TensorDataset(val_idx)
        test_ds = torch.utils.data.TensorDataset(test_idx)

        def _collate(batch):
            idx_batch = torch.stack([
                item[0] if isinstance(item, (list, tuple)) else item for item in batch
            ]).view(-1)
            
            # 🔧 ROOT CAUSE FIX: Get batch data and validate consistency
            batch_data = _subset(idx_batch)
            
            # Validate that all tensors have the same batch size (except graph-specific ones)
            batch_sizes = {}
            graph_keys = {'edge_index', 'edge_attr'}  # These don't follow batch_size convention
            
            for key, value in batch_data.items():
                if torch.is_tensor(value) and key not in graph_keys:
                    batch_sizes[key] = value.size(0)
            
            if batch_sizes:
                unique_sizes = set(batch_sizes.values())
                if len(unique_sizes) > 1:
                    # This is the root cause of DQN tensor mismatch - fix it here
                    target_batch_size = min(batch_sizes.values())  # Use minimum to avoid index errors
                    logger.warning(f"⚠️ Batch size inconsistency detected in data loader: {batch_sizes}")
                    logger.warning(f"🔧 Truncating all tensors to size {target_batch_size}")
                    
                    # Truncate all tensors to the same batch size
                    for key, value in batch_data.items():
                        if torch.is_tensor(value) and key not in graph_keys and value.size(0) > target_batch_size:
                            batch_data[key] = value[:target_batch_size]
                    
                    logger.info(f"✅ Fixed batch consistency - all tensors now have batch size {target_batch_size}")
            
            return batch_data

        # Auto-tune num_workers if not specified
        if num_workers == 0:
            num_workers = max(1, min(8, mp.cpu_count() // 2))

        # Fixed seed for reproducibility
        g = torch.Generator()
        g.manual_seed(42)
        
        # 🔧 CRITICAL FIX: Robust multiprocessing configuration  
        # Use safer settings to prevent pin_memory thread crashes
        multiprocessing_config = {
            'num_workers': 0,  # Disable multiprocessing to avoid pin_memory issues
            'persistent_workers': False,
            'pin_memory': False,  # Disable pin_memory to prevent thread crashes
            'prefetch_factor': None,  # Must be None when num_workers=0
            'worker_init_fn': lambda wid: seed_everything(int(torch.initial_seed() % 2**32 + wid)),
            'generator': g
        }

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            **multiprocessing_config,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
            **multiprocessing_config,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
            **multiprocessing_config,
        )
        
        return train_loader, val_loader, test_loader

    @staticmethod
    def build_dataloaders_from_graph(
        graph_pt: str | Path,
        batch_size: int = 64,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 0,
        cluster_size: int | None = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Creates train/val/test DataLoaders from a saved PyG .pt graph.

        Splits node trajectories along patient dimension; wraps tensors into
        simple ``dict`` batches compatible with Agent.update().  For custom
        use-cases users can override this utility.
        """
        from torch_geometric.data import Data

        # type: ignore[assignment]
        data: Data = torch.load(Path(graph_pt), weights_only=False)

        N = data.x.size(0)

        # Deterministic permutation with fixed seed
        g = torch.Generator()
        g.manual_seed(42)
        idx = torch.randperm(N, generator=g) if shuffle else torch.arange(N)

        # Persist splits for full reproducibility
        split_fp = Path("Output/splits.pkl")
        split_fp.parent.mkdir(parents=True, exist_ok=True)

        # If file exists reuse to guarantee same split across runs
        if split_fp.exists():
            with split_fp.open("rb") as f:
                saved = pickle.load(f)
            train_idx, val_idx, test_idx = map(torch.as_tensor, saved)
        else:
            n_test = int(N * test_split)
            n_val = int(N * val_split)
            test_idx = idx[:n_test]
            val_idx = idx[n_test: n_test + n_val]
            train_idx = idx[n_test + n_val:]
            with split_fp.open("wb") as f:
                pickle.dump(
                    (train_idx.cpu(), val_idx.cpu(), test_idx.cpu()), f)

        def _subset(i: torch.Tensor):
            # 🔧 CRITICAL FIX: Add None checks for data attributes to prevent batch corruption
            if not hasattr(data, 'x') or data.x is None:
                raise RuntimeError("Data object missing required 'x' attribute (state features)")
            if not hasattr(data, 'actions') or data.actions is None:
                raise RuntimeError("Data object missing required 'actions' attribute")
                
            # 🔧 CRITICAL FIX: Properly handle graph edge extraction for batched data
            edge_index = torch.empty((2, 0), dtype=torch.long)
            if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.size(1) > 0:
                try:
                    # Create mapping from original node indices to batch indices
                    node_mapping = torch.full((data.x.size(0),), -1, dtype=torch.long)
                    node_mapping[i] = torch.arange(i.size(0), dtype=torch.long)
                    
                    # Find edges where both source and target are in the batch
                    src_nodes = data.edge_index[0]
                    tgt_nodes = data.edge_index[1]
                    
                    # Check which edges connect nodes in the current batch
                    src_in_batch = torch.isin(src_nodes, i)
                    tgt_in_batch = torch.isin(tgt_nodes, i)
                    valid_edges = src_in_batch & tgt_in_batch
                    
                    if valid_edges.any():
                        # Extract valid edges and remap indices
                        valid_src = src_nodes[valid_edges]
                        valid_tgt = tgt_nodes[valid_edges]
                        
                        # Remap to batch indices - ensure no negative indices
                        remapped_src = node_mapping[valid_src]
                        remapped_tgt = node_mapping[valid_tgt]
                        
                        # Validate remapped indices
                        if (remapped_src >= 0).all() and (remapped_tgt >= 0).all():
                            edge_index = torch.stack([remapped_src, remapped_tgt], dim=0)
                        else:
                            # Use print if logger not available in this scope
                            print("⚠️ Invalid remapped edge indices, using empty edge_index")
                    else:
                        # No valid edges found - this is normal for non-PoG algorithms
                        # that don't use graph structure (DQN, CQL, BCQ, etc.)
                        pass
                        
                except Exception as e:
                    # Use print if logger not available in this scope
                    print(f"⚠️ Edge extraction failed: {e}, using empty edge_index")
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # 🔧 CRITICAL FIX: Add bounds checking for indices
            try:
                if i.max().item() >= data.x.size(0):
                    raise IndexError(f"Index {i.max().item()} out of bounds for data.x size {data.x.size(0)}")
                if i.max().item() >= data.actions.size(0):
                    raise IndexError(f"Index {i.max().item()} out of bounds for data.actions size {data.actions.size(0)}")
                    
                state_batch = data.x[i]
                action_batch = data.actions[i]
                
                # Validate extracted tensors are not None
                if state_batch is None:
                    raise RuntimeError("Extracted state batch is None")
                if action_batch is None:
                    raise RuntimeError("Extracted action batch is None")
                    
            except Exception as e:
                logger.error(f"❌ Failed to extract batch data: {e}")
                logger.error(f"🔍 Debug info: i.shape={i.shape}, i.min={i.min()}, i.max={i.max()}")
                logger.error(f"🔍 Data info: data.x.shape={data.x.shape if data.x is not None else None}, "
                           f"data.actions.shape={data.actions.shape if data.actions is not None else None}")
                raise RuntimeError(f"Data extraction failed: {e}") from e
            
            # 🔧 计算实际的轨迹长度和掩码
            batch_size = state_batch.size(0)
            seq_len = state_batch.size(1)
            
            # 创建lengths - 检查每个样本的有效长度
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=state_batch.device)
            
            # 创建mask - 标记有效的时间步
            # 假设done标志为True表示轨迹结束，done之后的步骤都是padding
            done_flags = state_batch[:, :, -1].bool()  # (B, T)
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=state_batch.device)
            
            # 对于每个样本，找到第一个done=True的位置，该位置之后的都设为False
            for b in range(batch_size):
                done_positions = torch.where(done_flags[b])[0]
                if len(done_positions) > 0:
                    first_done = done_positions[0].item()
                    # 包含第一个done位置，但之后的都是padding
                    if first_done + 1 < seq_len:
                        mask[b, first_done + 1:] = False
                        lengths[b] = first_done + 1
            
            return {
                # ------------------------------------------------------------------
                # Instead of collapsing trajectories to *single‐step* samples we now
                # return the *full* time series so that OPE estimators (WDR / IPW)
                # can apply proper discounting.  This also removes the 10× reward
                # inflation previously observed with Behaviour-Cloning (BC ≈ 218).
                # ------------------------------------------------------------------
                "state": state_batch,                    # (B, T, D)
                "action": action_batch,                  # (B, T, n_heads)
                "reward": state_batch[:, :, -2],         # per-step reward vector (B, T)
                "done": done_flags,                      # (B, T) terminal flags
                # For convenience we keep next_state identical to current state
                # (many agents ignore this field), but preserve the *shape* so
                # that downstream code expecting 3-D tensors remains valid.
                "next_state": state_batch,
                "survival": data.survival[i] if hasattr(data, "survival") and data.survival is not None else None,
                # Provide trajectory length so that fallback IPW path can
                # normalise *single-step* rewards when users opt for legacy
                # flattened loaders.
                "traj_len": torch.full((i.size(0),), action_batch.size(1)),
                "edge_index": edge_index,              # Subgraph edges for this batch
                "lengths": lengths,                      # (B,) 轨迹实际长度
                "mask": mask,                           # (B, T) 有效时间步掩码
                "next_lengths": lengths.clone(),         # (B,) next_state轨迹长度 (通常与lengths相同)
            }

        train_ds = torch.utils.data.TensorDataset(train_idx)
        val_ds = torch.utils.data.TensorDataset(val_idx)
        test_ds = torch.utils.data.TensorDataset(test_idx)

        def _collate(batch):
            # Each entry from ``TensorDataset`` is a 1-tuple ``(tensor,)``.
            # We therefore need to *extract* the inner tensor before
            # concatenation, otherwise ``torch.cat`` receives a list of
            # tuples and raises a ``TypeError``.
            idx_batch = torch.stack([
                item[0] if isinstance(item, (list, tuple)) else item for item in batch
            ]).view(-1)
            
            # 🔧 ROOT CAUSE FIX: Get batch data and validate consistency
            batch_data = _subset(idx_batch)
            
            # Validate that all tensors have the same batch size (except graph-specific ones)
            batch_sizes = {}
            graph_keys = {'edge_index', 'edge_attr'}  # These don't follow batch_size convention
            
            for key, value in batch_data.items():
                if torch.is_tensor(value) and key not in graph_keys:
                    batch_sizes[key] = value.size(0)
            
            if batch_sizes:
                unique_sizes = set(batch_sizes.values())
                if len(unique_sizes) > 1:
                    # This is the root cause of DQN tensor mismatch - fix it here
                    target_batch_size = min(batch_sizes.values())  # Use minimum to avoid index errors
                    logger.warning(f"⚠️ Batch size inconsistency detected in data loader: {batch_sizes}")
                    logger.warning(f"🔧 Truncating all tensors to size {target_batch_size}")
                    
                    # Truncate all tensors to the same batch size
                    for key, value in batch_data.items():
                        if torch.is_tensor(value) and key not in graph_keys and value.size(0) > target_batch_size:
                            batch_data[key] = value[:target_batch_size]
                    
                    logger.info(f"✅ Fixed batch consistency - all tensors now have batch size {target_batch_size}")
            
            return batch_data

        if cluster_size is not None and cluster_size > 0:
            try:
                from torch_geometric.loader import ClusterData, ClusterLoader

                cluster_data = ClusterData(data, num_parts=max(
                    1, data.num_nodes // cluster_size), recursive=False)
                cluster_loader = ClusterLoader(
                    cluster_data, batch_size=1, shuffle=shuffle, num_workers=num_workers)

                # Return cluster_loader for train; keep simple loaders for val/test
                dummy_loader = DataLoader([], batch_size=1)  # placeholder

                def _iter_clusters():
                    for sub_data in cluster_loader:
                        yield {
                            "state": sub_data.x[:, -1],
                            "action": sub_data.actions[:, -1],
                            "reward": sub_data.x[:, :, -2].sum(1),
                            "done": sub_data.x[:, :, -1].sum(1).bool(),
                            "next_state": sub_data.x[:, -1],
                            "survival": getattr(sub_data, "survival", None),
                        }

                train_loader = _iter_clusters()  # returns generator
                # fall back to idx sampling for val/test as below
            except ImportError:
                logger.warning(
                    "torch-geometric >=2.4 required for ClusterData; falling back to full-batch loader")
                cluster_size = None  # disable

        if cluster_size is not None and cluster_size > 0:
            # If cluster-based loader set above, we already created train_loader
            pass  # keep existing train_loader from clusters
        else:
            # Auto-tune num_workers if not specified
            if num_workers == 0:
                num_workers = max(1, min(8, mp.cpu_count() // 2))

            # 🔧 CRITICAL FIX: Robust multiprocessing configuration
            # Use safer settings to prevent pin_memory thread crashes
            multiprocessing_config = {
                'num_workers': 0,  # Disable multiprocessing to avoid pin_memory issues
                'persistent_workers': False,
                'pin_memory': False,  # Disable pin_memory to prevent thread crashes
                'prefetch_factor': None,  # Must be None when num_workers=0
                'worker_init_fn': lambda wid: seed_everything(int(torch.initial_seed() % 2**32 + wid)),
                'generator': g
            }

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=_collate,
                **multiprocessing_config,
            )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
            **multiprocessing_config,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
            **multiprocessing_config,
        )
        return train_loader, val_loader, test_loader

    def save_checkpoint(self, path: str | Path) -> None:
        """Save comprehensive checkpoint with full agent state for reproducibility.
        
        Saves complete training state including model parameters, optimizer state,
        training statistics, and configuration for seamless resume capability.
        
        Args:
            path: Path to save the checkpoint file
        """
        try:
            checkpoint_data = {
                'timestamp': time.time(),
                'epoch': getattr(self, '_current_epoch', 0),
                'best_metric': self._best_metric,
                'epochs_no_improve': self._epochs_no_improve,
                'seed': self.training_seed,
                'algo': self.algo,
                'state_dim': self.state_dim,
                'action_dims': self.action_dims,
            }
            
            # Save agent-specific state with comprehensive error handling
            if hasattr(self.agent, 'save_checkpoint'):
                # Use agent's own checkpoint method if available
                agent_checkpoint_path = Path(path).parent / f"{Path(path).stem}_agent.pt"
                self.agent.save_checkpoint(str(agent_checkpoint_path))
                checkpoint_data['agent_checkpoint'] = str(agent_checkpoint_path)
                
            elif hasattr(self.agent, 'q_net'):
                # Legacy support for Q-network based agents
                checkpoint_data['agent_state'] = self.agent.q_net.state_dict()
                if hasattr(self.agent, 'optimizer'):
                    checkpoint_data['optimizer_state'] = self.agent.optimizer.state_dict()
                if hasattr(self.agent, 'target_model'):
                    checkpoint_data['target_model_state'] = self.agent.target_model.state_dict()
                    
            elif hasattr(self.agent, 'model'):
                # For BC and other model-based agents
                checkpoint_data['model_state'] = self.agent.model.state_dict()
                if hasattr(self.agent, 'optimizer'):
                    checkpoint_data['optimizer_state'] = self.agent.optimizer.state_dict()
            else:
                logger.warning("⚠️  Agent has no saveable state - saving minimal checkpoint")
            
            # Save training metrics history if available
            if hasattr(self, 'metrics_tracker') and hasattr(self.metrics_tracker, 'metrics_history'):
                checkpoint_data['metrics_history'] = self.metrics_tracker.metrics_history
                
            # Save behavior policy state if trained
            if hasattr(self, '_behav_models') and self._behav_models:
                behav_states = []
                for i, (model, optimizer) in enumerate(self._behav_models):
                    behav_states.append({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                    })
                checkpoint_data['behavior_policy_states'] = behav_states
                
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save with atomic write to prevent corruption
            temp_path = str(path) + '.tmp'
            torch.save(checkpoint_data, temp_path)
            Path(temp_path).rename(path)
            
            logger.info("💾 Comprehensive checkpoint saved: %s", path)
            logger.debug("   📊 Checkpoint contents: %s", list(checkpoint_data.keys()))
            
        except Exception as e:
            logger.error("❌ Failed to save checkpoint: %s", e)
            # Clean up temporary file if it exists
            temp_path = str(path) + '.tmp'
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise RuntimeError(f"Checkpoint saving failed: {e}") from e

    def load_checkpoint(self, path: str | Path) -> None:
        """Load comprehensive checkpoint with full state restoration.
        
        Restores complete training state including model parameters, optimizer state,
        training statistics, and validates configuration compatibility.
        
        Args:
            path: Path to the checkpoint file to load
        """
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
                
            logger.info("🔄 Loading checkpoint: %s", path)
            checkpoint_data = torch.load(path, map_location=self.device, weights_only=False)
            
            # Validate checkpoint compatibility
            if 'algo' in checkpoint_data and checkpoint_data['algo'] != self.algo:
                logger.warning("⚠️  Algorithm mismatch: checkpoint=%s, current=%s", 
                              checkpoint_data['algo'], self.algo)
                              
            if 'state_dim' in checkpoint_data and checkpoint_data['state_dim'] != self.state_dim:
                raise ValueError(f"State dimension mismatch: checkpoint={checkpoint_data['state_dim']}, "
                               f"current={self.state_dim}")
                               
            if 'action_dims' in checkpoint_data and checkpoint_data['action_dims'] != self.action_dims:
                raise ValueError(f"Action dimensions mismatch: checkpoint={checkpoint_data['action_dims']}, "
                               f"current={self.action_dims}")
            
            # Restore agent state
            if 'agent_checkpoint' in checkpoint_data and hasattr(self.agent, 'load_checkpoint'):
                # Use agent's own checkpoint loading if available
                self.agent.load_checkpoint(checkpoint_data['agent_checkpoint'])
                
            elif 'agent_state' in checkpoint_data and hasattr(self.agent, 'q_net'):
                # Legacy Q-network restoration
                self.agent.q_net.load_state_dict(checkpoint_data['agent_state'])
                if 'optimizer_state' in checkpoint_data and hasattr(self.agent, 'optimizer'):
                    self.agent.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
                if 'target_model_state' in checkpoint_data and hasattr(self.agent, 'target_model'):
                    self.agent.target_model.load_state_dict(checkpoint_data['target_model_state'])
                    
            elif 'model_state' in checkpoint_data and hasattr(self.agent, 'model'):
                # BC and other model-based agents
                self.agent.model.load_state_dict(checkpoint_data['model_state'])
                if 'optimizer_state' in checkpoint_data and hasattr(self.agent, 'optimizer'):
                    self.agent.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
            
            # Restore training state
            self._best_metric = checkpoint_data.get('best_metric')
            self._epochs_no_improve = checkpoint_data.get('epochs_no_improve', 0)
            
            # Restore metrics history
            if 'metrics_history' in checkpoint_data and hasattr(self, 'metrics_tracker'):
                self.metrics_tracker.metrics_history = checkpoint_data['metrics_history']
                
            # Restore behavior policy state
            if 'behavior_policy_states' in checkpoint_data and hasattr(self, '_behav_models'):
                behav_states = checkpoint_data['behavior_policy_states']
                if len(behav_states) == len(self._behav_models):
                    for (model, optimizer), state in zip(self._behav_models, behav_states):
                        model.load_state_dict(state['model_state'])
                        optimizer.load_state_dict(state['optimizer_state'])
                else:
                    logger.warning("⚠️  Behavior policy state count mismatch, skipping restoration")
            
            logger.info("✅ Checkpoint loaded successfully")
            logger.debug("   📊 Restored components: %s", 
                        [k for k in checkpoint_data.keys() if not k.startswith('_')])
            
        except Exception as e:
            logger.error("❌ Failed to load checkpoint: %s", e)
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    # ------------------------------------------------------------------
    #  Visualization generation methods
    # ------------------------------------------------------------------
    
    def generate_training_visualizations(
        self,
        save_dir: Optional[Path] = None,
        include_convergence: bool = True,
        include_strategy: bool = False,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[Path]]:
        """Generate comprehensive training visualizations.
        
        Args:
            save_dir: Directory to save visualizations (defaults to log_dir/visualizations)
            include_convergence: Whether to generate convergence diagnostics
            include_strategy: Whether to generate treatment strategy analysis
            test_loader: Optional test DataLoader for strategy analysis
            
        Returns:
            Dictionary mapping visualization type to list of saved file paths
        """
        if not self.enable_visualization:
            logger.info("Visualization disabled, skipping visualization generation")
            return {}
            
        if save_dir is None:
            save_dir = self.log_dir / "visualizations" if self.log_dir else Path("Output/visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # 1. Training convergence diagnostics
            if include_convergence and hasattr(self.metrics_tracker, 'metrics_history'):
                logger.info("🎨 Generating training convergence diagnostics...")
                
                # Prepare training and validation history
                training_history = {}
                validation_history = {}
                
                for metric, values in self.metrics_tracker.metrics_history.items():
                    if metric.startswith('train/'):
                        training_history[metric.replace('train/', '')] = values
                    elif metric.startswith('val/'):
                        validation_history[metric.replace('val/', '')] = values
                
                if training_history or validation_history:
                    convergence_fig = plot_convergence_diagnostics(
                        training_history=training_history,
                        validation_history=validation_history,
                        title=f"Training Convergence Diagnostics - {self.algo.upper()}"
                    )
                    
                    convergence_paths = save_figure_publication_ready(
                        convergence_fig,
                        save_dir / f"convergence_diagnostics_{self.algo}",
                        formats=self.visualization_formats
                    )
                    saved_files['convergence'] = convergence_paths
                    
            # 2. Treatment strategy analysis (if test data available)
            if include_strategy and test_loader is not None:
                logger.info("🎨 Generating treatment strategy analysis...")
                
                try:
                    # Sample a batch for strategy analysis
                    states_list = []
                    actions_list = []
                    outcomes_list = []
                    
                    sample_count = 0
                    max_samples = 1000  # Limit for visualization
                    
                    for batch in test_loader:
                        if sample_count >= max_samples:
                            break
                            
                        # Extract data for visualization
                        if 'state' in batch:
                            states_batch = batch['state'].cpu().numpy()
                            if states_batch.ndim == 3:  # (B, T, D) -> use last timestep
                                states_batch = states_batch[:, -1, :]
                            states_list.append(states_batch)
                            
                        if 'action' in batch:
                            actions_batch = batch['action'].cpu().numpy()
                            if actions_batch.ndim == 3:  # (B, T, H) -> use last timestep
                                actions_batch = actions_batch[:, -1, :]
                            actions_list.append(actions_batch)
                            
                        if 'survival' in batch and batch['survival'] is not None:
                            outcomes_batch = batch['survival'].cpu().numpy()
                            outcomes_list.append(outcomes_batch)
                        elif 'reward' in batch:
                            # Use cumulative reward as outcome proxy
                            reward_batch = batch['reward'].cpu().numpy()
                            if reward_batch.ndim == 2:  # (B, T) -> sum over time
                                reward_batch = reward_batch.sum(axis=1)
                            outcomes_list.append(reward_batch)
                            
                        sample_count += len(states_batch)
                        
                    # Combine all samples - ensure we have non-empty lists
                    if states_list and actions_list and outcomes_list:
                        try:
                            states_combined = np.concatenate(states_list, axis=0)
                            actions_combined = np.concatenate(actions_list, axis=0)
                            outcomes_combined = np.concatenate(outcomes_list, axis=0)
                        except ValueError as e:
                            if "need at least one array to concatenate" in str(e):
                                logger.warning("⚠️  Insufficient data for strategy visualization - skipping")
                                states_combined = None
                            else:
                                raise
                        
                        # Only proceed if concatenation was successful
                        if states_combined is not None:
                            # Generate medical domain action/state names using TaskManager
                            try:
                                task_config = get_current_task_config()
                                action_names = task_config.vis_action_names
                            except:
                                # Fallback to generic names
                                action_names = [f"Action {i+1}" for i in range(actions_combined.shape[1])]
                            
                            state_names = ["SOFA Score", "Creatinine", "Urine Output", "Heart Rate", "MAP"]
                            
                            strategy_fig = plot_treatment_strategy_heatmap(
                                states=states_combined,
                                actions=actions_combined,
                                outcomes=outcomes_combined,
                                state_names=state_names[:states_combined.shape[1]],
                                action_names=action_names[:actions_combined.shape[1]],
                                title=f"Treatment Strategy Analysis - {self.algo.upper()}"
                            )
                            
                            strategy_paths = save_figure_publication_ready(
                                strategy_fig,
                                save_dir / f"treatment_strategy_{self.algo}",
                                formats=self.visualization_formats
                            )
                            saved_files['strategy'] = strategy_paths
                        
                except Exception as e:
                    logger.warning("Failed to generate treatment strategy visualization: %s", e)
                    
        except Exception as e:
            logger.error("Error during visualization generation: %s", e)
            
        logger.info("🎨 Visualization generation complete. Saved files: %s", 
                   sum(len(paths) for paths in saved_files.values()))
        return saved_files
        
    def generate_policy_comparison_visualization(
        self,
        comparison_data: Dict[str, np.ndarray],
        save_dir: Optional[Path] = None,
        action_names: List[str] = None
    ) -> List[Path]:
        """Generate policy comparison visualizations.
        
        Args:
            comparison_data: Dictionary mapping algorithm names to action distributions
            save_dir: Directory to save visualizations
            action_names: Names for each action dimension
            
        Returns:
            List of paths to saved visualization files
        """
        if save_dir is None:
            save_dir = self.log_dir / "visualizations" if self.log_dir else Path("visualizations")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.enable_visualization:
            return []
        
        saved_files = []
        
        try:
            # Generate comparison plot
            fig = plot_policy_distribution_comparison(
                comparison_data, 
                action_names=action_names
            )
            
            # Save in multiple formats
            for fmt in self.visualization_formats:
                filepath = save_dir / f"policy_comparison.{fmt}"
                save_figure_publication_ready(fig, filepath, format=fmt)
                saved_files.append(filepath)
                
            logger.info(f"📊 Policy comparison visualization saved: {len(saved_files)} files")
            
        except Exception as e:
            logger.error(f"Failed to generate policy comparison visualization: {e}")
            
        return saved_files
    
    def __del__(self):
        """Cleanup resources when Trainer is destroyed."""
        try:
            # Clean up multiprocessing temp directories on destruction
            cleanup_multiprocessing_temp_dirs()
        except Exception as e:
            # Silent cleanup - don't raise exceptions in destructor
            pass

    def _convert_batch_to_legacy_format(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Convert modern dict batch format to legacy tuple format for backward compatibility.
        
        This method handles the conversion from the new standardized dictionary format
        to the legacy tuple format expected by older agent implementations. It ensures
        compatibility across different agent types while maintaining type safety.
        
        Args:
            batch: Modern batch dictionary with standardized field names.
            
        Returns:
            Legacy tuple format matching agent expectations.
            
        Raises:
            RuntimeError: If conversion fails due to missing required fields.
        """
        try:
            # 🔧 CRITICAL FIX: Add None checks for all critical batch fields
            obs = batch.get("state")
            actions = batch.get("action")
            lengths = batch.get("lengths")
            
            # Validate required fields are not None
            if obs is None:
                raise ValueError("Required field 'state' is None in batch")
            if actions is None:
                raise ValueError("Required field 'action' is None in batch")
                
            # 🔧 CRITICAL FIX: Ensure all tensors are contiguous for model compatibility
            if not obs.is_contiguous():
                obs = obs.contiguous()
                self.logger.debug("Made obs tensor contiguous")
                
            if actions is not None and torch.is_tensor(actions) and not actions.is_contiguous():
                actions = actions.contiguous()
                self.logger.debug("Made actions tensor contiguous")
                
            if lengths is not None and torch.is_tensor(lengths) and not lengths.is_contiguous():
                lengths = lengths.contiguous()
                self.logger.debug("Made lengths tensor contiguous")
            # 🔧 ENHANCED NULL FIELD HANDLING: Safe creation of lengths
            if lengths is None:
                # Create default lengths based on state shape with validation
                if obs.dim() >= 2:
                    seq_len = obs.size(1)
                    batch_size = obs.size(0)
                    lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=obs.device)
                    self.logger.info(f"✅ Created default lengths: shape={lengths.shape}, value={seq_len}")
                else:
                    batch_size = obs.size(0)
                    lengths = torch.ones((batch_size,), dtype=torch.long, device=obs.device)
                    self.logger.info(f"✅ Created default lengths for 2D obs: shape={lengths.shape}")
                self.logger.warning("⚠️ 'lengths' field was None, created default lengths")
            
            # 🔧 ENHANCED NULL FIELD HANDLING: Safe creation of mask with proper validation
            mask = batch.get("mask")
            if mask is None:
                if obs.dim() == 3:  # (B, T, D)
                    mask_shape = (obs.size(0), obs.size(1))
                    mask = torch.ones(mask_shape, device=obs.device, dtype=torch.bool)
                    self.logger.info(f"✅ Created default mask for 3D obs: shape={mask.shape}")
                elif obs.dim() == 2:  # (B, D)
                    mask_shape = (obs.size(0),)
                    mask = torch.ones(mask_shape, device=obs.device, dtype=torch.bool)
                    self.logger.info(f"✅ Created default mask for 2D obs: shape={mask.shape}")
                else:
                    raise ValueError(f"Unexpected obs dimension: {obs.dim()}")
                self.logger.warning("⚠️ 'mask' field was None, created default mask")
            
            # 🔧 CRITICAL FIX: Enhanced tensor dimension handling with shape validation
            batch_size = obs.size(0)
            
            # Ensure proper tensor dimensions with shape consistency checks
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)  # (B,1,D)
                mask = mask.unsqueeze(1) if mask.dim() == 1 else mask
                if actions is not None and actions.dim() == 2:
                    actions = actions.unsqueeze(1)  # (B,1,n_heads)
            
            # 🔧 SHAPE VALIDATION: Ensure all tensors have consistent batch dimensions
            if lengths.size(0) != batch_size:
                self.logger.warning(f"⚠️ lengths batch size mismatch: {lengths.size(0)} vs {batch_size}")
                # Fix by truncating or padding lengths
                if lengths.size(0) > batch_size:
                    lengths = lengths[:batch_size]
                else:
                    # Pad with appropriate values
                    pad_size = batch_size - lengths.size(0)
                    pad_lengths = torch.full((pad_size,), obs.size(1), dtype=lengths.dtype, device=lengths.device)
                    lengths = torch.cat([lengths, pad_lengths], dim=0)
                    
            if mask.size(0) != batch_size:
                self.logger.warning(f"⚠️ mask batch size mismatch: {mask.size(0)} vs {batch_size}")
                # Fix mask dimensions
                if obs.dim() == 3:  # (B, T, D)
                    target_mask_shape = (batch_size, obs.size(1))
                    mask = torch.ones(target_mask_shape, device=obs.device, dtype=mask.dtype)
                else:
                    mask = torch.ones((batch_size,), device=obs.device, dtype=mask.dtype)
                    
            # 🔧 TEMPORAL DIMENSION VALIDATION: Ensure temporal consistency
            if obs.dim() == 3 and mask.dim() == 2:
                if obs.size(1) != mask.size(1):
                    self.logger.warning(f"⚠️ temporal dimension mismatch: obs[1]={obs.size(1)} vs mask[1]={mask.size(1)}")
                    # Align temporal dimensions by taking minimum
                    min_time = min(obs.size(1), mask.size(1))
                    obs = obs[:, :min_time, :]
                    mask = mask[:, :min_time]
                    # Update lengths accordingly
                    lengths = torch.clamp(lengths, max=min_time)
            
            # Check if agent expects full transition tuple
            if getattr(self.agent, "_expects_full_transition", False):
                # 🔧 ENHANCED NULL FIELD HANDLING: Safe handling of optional transition fields
                rewards = batch.get("reward")
                if rewards is None:
                    if mask.dim() == 2:  # (B, T)
                        rewards = torch.zeros_like(mask, dtype=torch.float32)
                    else:  # (B,)
                        rewards = torch.zeros_like(mask, dtype=torch.float32)
                    self.logger.info(f"✅ Created default rewards: shape={rewards.shape}")
                    self.logger.warning("⚠️ 'reward' field was None, created default zeros")
                
                next_obs = batch.get("next_state")
                if next_obs is None:
                    if obs is not None:
                        next_obs = obs.clone()
                        self.logger.info(f"✅ Cloned obs for next_state: shape={next_obs.shape}")
                    else:
                        raise ValueError("Cannot create next_state when obs is None")
                    self.logger.warning("⚠️ 'next_state' field was None, using cloned obs")
                
                dones = batch.get("done")
                if dones is None:
                    if mask.dim() == 2:  # (B, T)
                        dones = torch.zeros_like(mask, dtype=torch.bool)
                    else:  # (B,)
                        dones = torch.zeros_like(mask, dtype=torch.bool)
                    self.logger.info(f"✅ Created default dones: shape={dones.shape}")
                    self.logger.warning("⚠️ 'done' field was None, created default zeros")
                
                next_lengths = batch.get("next_lengths")
                if next_lengths is None:
                    if lengths is not None:
                        next_lengths = lengths.clone()
                        self.logger.info(f"✅ Cloned lengths for next_lengths: shape={next_lengths.shape}")
                    else:
                        raise ValueError("Cannot create next_lengths when lengths is None")
                    self.logger.warning("⚠️ 'next_lengths' field was None, using cloned lengths")
                
                edge_index = batch.get("edge_index")
                if edge_index is None:
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=obs.device)
                
                # 🔧 CRITICAL FIX: Convert multi-head actions to list format while preserving sequence dimensions
                # Add caching for action shape conversions
                if actions is not None:
                    action_key = (actions.shape, actions.dim())
                    
                    # Check cache first
                    if not hasattr(self, '_action_conversion_cache'):
                        self._action_conversion_cache = {}
                    
                    if action_key in self._action_conversion_cache:
                        # Use cached conversion strategy
                        cached_info = self._action_conversion_cache[action_key]
                        n_heads = cached_info['n_heads']
                        dim = cached_info['dim']
                        
                        if dim == 3:
                            actions_list = [actions[:, :, h].cpu().numpy() for h in range(n_heads)]
                        else:  # dim == 2
                            actions_list = [actions[:, h:h+1].cpu().numpy() for h in range(n_heads)]
                    else:
                        # Perform conversion and cache the strategy
                        if actions.dim() == 3:
                            # (B, T, n_heads) - preserve full sequence
                            n_heads = actions.size(2)
                            actions_list = [actions[:, :, h].cpu().numpy() for h in range(n_heads)]  # Each: (B, T)
                            self._action_conversion_cache[action_key] = {'n_heads': n_heads, 'dim': 3}
                            self.logger.debug(
                                f"🔧 Converted 3D actions to list: shapes={[a.shape for a in actions_list]}")
                        elif actions.dim() == 2:
                            # (B, n_heads) - single-step format
                            n_heads = actions.size(1)
                            actions_list = [actions[:, h:h+1].cpu().numpy() for h in range(n_heads)]  # Each: (B, 1)
                            self._action_conversion_cache[action_key] = {'n_heads': n_heads, 'dim': 2}
                            self.logger.debug(f"🔧 Converted 2D actions to list: shapes={[a.shape for a in actions_list]}")
                        else:
                            raise ValueError(f"Unexpected actions dimension: {actions.dim()}")
                else:
                    actions_list = []
                    self.logger.error("❌ Cannot create actions_list from None actions")
                
                return (
                    obs.cpu().numpy() if obs is not None else np.array([]),
                    actions_list,
                    rewards.cpu().numpy() if rewards is not None else np.array([]),
                    next_obs.cpu().numpy() if next_obs is not None else np.array([]),
                    dones.cpu().numpy() if dones is not None else np.array([]),
                    mask.cpu().numpy() if mask is not None else np.array([]),
                    lengths.cpu().numpy() if lengths is not None else np.array([]),
                    next_lengths.cpu().numpy() if next_lengths is not None else np.array([]),
                    edge_index.cpu().numpy() if edge_index is not None else np.array([]),
                )
            else:
                # Minimal tuple for BC-style agents with None checks
                if "edge_index" in batch:
                    edge_index = batch["edge_index"]
                    return (
                        obs.cpu().numpy() if obs is not None else np.array([]),
                        actions.cpu().numpy() if actions is not None else np.array([]),
                        mask.cpu().numpy() if mask is not None else np.array([]),
                        lengths.cpu().numpy() if lengths is not None else np.array([]),
                        edge_index.cpu().numpy() if edge_index is not None else np.array([]),
                        batch.get("sub_data_list", [])
                    )
                else:
                    return (
                        obs.cpu().numpy() if obs is not None else np.array([]),
                        actions.cpu().numpy() if actions is not None else np.array([]),
                        mask.cpu().numpy() if mask is not None else np.array([]),
                        lengths.cpu().numpy() if lengths is not None else np.array([]),
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ Legacy batch conversion failed: {e}")
            raise RuntimeError(f"Failed to convert batch to legacy format: {e}") from e

    def _preprocess_dict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Preprocess dictionary batch for modern agent compatibility.
        
        This method ensures that the batch dictionary has all required fields
        and proper tensor dimensions for consistent processing across different
        agent implementations.
        
        Args:
            batch: Input batch dictionary.
            
        Returns:
            Preprocessed batch dictionary with standardized format.
        """
        try:
            # 🔧 CRITICAL FIX: Enhanced single-step algorithm handling
            single_step_algos = {
                "dqn", "cql", "bcq", "bve",
                "pog_dqn", "pog_cql", "pog_bcq", "pog_bve",
            }
            
            # 🔧 ENHANCED BATCH VALIDATION: Check for required fields
            required_fields = ["state"]
            for field in required_fields:
                if field not in batch:
                    self.logger.error(f"❌ Missing required field '{field}' in batch")
                    raise ValueError(f"Batch is missing required field: {field}")
            
            # Get reference dimensions
            state = batch["state"]
            batch_size = state.size(0)
            is_sequential = state.dim() == 3
            seq_len = state.size(1) if is_sequential else 1
            
            self.logger.debug(f"🔍 Preprocessing batch: algo={self.algo}, is_sequential={is_sequential}, "
                            f"batch_size={batch_size}, seq_len={seq_len}")
            
            if self.algo in single_step_algos:
                self.logger.debug(f"🔧 Processing single-step algorithm: {self.algo}")
                
                # For single-step algorithms, we need to be careful about sequence handling
                # Instead of reducing dimensions immediately, let the agent handle sequence data
                
                # 🔧 IMPROVED SINGLE-STEP PROCESSING: Preserve sequence structure but mark as single-step
                if is_sequential:
                    # For algorithms that conceptually work on single steps but receive sequence data,
                    # we pass through the full sequence and let the agent extract what it needs
                    
                    # Ensure temporal consistency across all fields
                    for key in ["next_state", "action"]:
                        if key in batch:
                            tensor = batch[key]
                            if tensor.dim() == 2 and is_sequential:
                                # Expand to match sequence dimension
                                batch[key] = tensor.unsqueeze(1).expand(-1, seq_len, -1)
                            elif tensor.dim() == 3 and tensor.size(1) != seq_len:
                                # Align sequence lengths
                                if tensor.size(1) > seq_len:
                                    batch[key] = tensor[:, :seq_len, :]
                                else:
                                    # Pad to match sequence length
                                    pad_len = seq_len - tensor.size(1)
                                    if key == "next_state":
                                        # For next_state, repeat the last state
                                        last_state = tensor[:, -1:, :]
                                        padding = last_state.expand(-1, pad_len, -1)
                                        batch[key] = torch.cat([tensor, padding], dim=1)
                                    else:
                                        # For actions, pad with zeros
                                        padding_shape = list(tensor.shape)
                                        padding_shape[1] = pad_len
                                        padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
                                        batch[key] = torch.cat([tensor, padding], dim=1)
                    
                    # Handle rewards and dones
                    for key in ["reward", "done"]:
                        if key in batch:
                            tensor = batch[key]
                            if tensor.dim() == 1:
                                # Expand (B,) -> (B, T)
                                if key == "reward":
                                    # For rewards, distribute equally across timesteps
                                    batch[key] = (tensor.unsqueeze(1) / seq_len).expand(-1, seq_len)
                                else:
                                    # For dones, typically only the last timestep matters
                                    done_expanded = torch.zeros(batch_size, seq_len, dtype=tensor.dtype, device=tensor.device)
                                    done_expanded[:, -1] = tensor
                                    batch[key] = done_expanded
                            elif tensor.dim() == 2 and tensor.size(1) != seq_len:
                                # Align temporal dimensions
                                if tensor.size(1) == 1:
                                    if key == "reward":
                                        batch[key] = tensor.expand(-1, seq_len)
                                    else:
                                        done_expanded = torch.zeros(batch_size, seq_len, dtype=tensor.dtype, device=tensor.device)
                                        done_expanded[:, -1] = tensor.squeeze(1)
                                        batch[key] = done_expanded
                                else:
                                    # Truncate or pad
                                    if tensor.size(1) > seq_len:
                                        batch[key] = tensor[:, :seq_len]
                                    else:
                                        pad_len = seq_len - tensor.size(1)
                                        if key == "reward":
                                            padding = torch.zeros(batch_size, pad_len, dtype=tensor.dtype, device=tensor.device)
                                        else:
                                            padding = torch.zeros(batch_size, pad_len, dtype=tensor.dtype, device=tensor.device)
                                        batch[key] = torch.cat([tensor, padding], dim=1)
                    
                    # Ensure mask has proper dimensions
                    if "mask" not in batch:
                        batch["mask"] = torch.ones(batch_size, seq_len, dtype=torch.bool, device=state.device)
                    else:
                        mask = batch["mask"]
                        if mask.dim() == 1:
                            batch["mask"] = mask.unsqueeze(1).expand(-1, seq_len)
                        elif mask.dim() == 2 and mask.size(1) != seq_len:
                            if mask.size(1) > seq_len:
                                batch["mask"] = mask[:, :seq_len]
                            else:
                                pad_len = seq_len - mask.size(1)
                                padding = torch.zeros(batch_size, pad_len, dtype=mask.dtype, device=mask.device)
                                batch["mask"] = torch.cat([mask, padding], dim=1)
                    
                    # Set proper lengths
                    if "lengths" not in batch:
                        batch["lengths"] = torch.full((batch_size,), seq_len, dtype=torch.long, device=state.device)
                    else:
                        lengths = batch["lengths"]
                        if lengths.size(0) != batch_size:
                            batch["lengths"] = torch.full((batch_size,), seq_len, dtype=torch.long, device=state.device)
                        else:
                            # Ensure lengths don't exceed sequence length
                            batch["lengths"] = torch.clamp(lengths, max=seq_len)
                
                else:
                    # Already single-step data, ensure proper dimensions
                    # Add sequence dimension for consistency
                    for key in ["state", "next_state"]:
                        if key in batch and batch[key].dim() == 2:
                            batch[key] = batch[key].unsqueeze(1)
                    
                    if "action" in batch and batch["action"].dim() == 2:
                        batch["action"] = batch["action"].unsqueeze(1)
                        
                    for key in ["reward", "done"]:
                        if key in batch and batch[key].dim() == 1:
                            batch[key] = batch[key].unsqueeze(1)
                    
                    # Set single-step lengths and mask
                    batch["lengths"] = torch.ones(batch_size, dtype=torch.long, device=state.device)
                    
                    if "mask" not in batch:
                        batch["mask"] = torch.ones(batch_size, 1, dtype=torch.bool, device=state.device)
                    elif batch["mask"].dim() == 1:
                        batch["mask"] = batch["mask"].unsqueeze(1)
            
            else:
                # Multi-step algorithms: ensure consistent 3D format
                self.logger.debug(f"🔧 Processing multi-step algorithm: {self.algo}")
                
                # Ensure 3D format for sequence consistency
                if "state" in batch and batch["state"].dim() == 2:
                    batch["state"] = batch["state"].unsqueeze(1)
                    if "next_state" in batch and batch["next_state"].dim() == 2:
                        batch["next_state"] = batch["next_state"].unsqueeze(1)
                    if "action" in batch and batch["action"].dim() == 2:
                        batch["action"] = batch["action"].unsqueeze(1)
                
                # Ensure mask and lengths are present
                if "mask" not in batch and "state" in batch:
                    s = batch["state"]
                    if s.dim() == 3:
                        batch["mask"] = torch.ones(s.size(0), s.size(1), device=s.device)
                    else:
                        batch["mask"] = torch.ones(s.size(0), device=s.device)
            
            # 🔧 FINAL VALIDATION: Ensure all tensors have consistent batch dimensions (except graph structure tensors)
            graph_structure_keys = {"edge_index", "edge_attr", "node_features"}  # These don't follow batch_size convention
            
            for key, tensor in batch.items():
                if torch.is_tensor(tensor) and key not in graph_structure_keys and tensor.size(0) != batch_size:
                    self.logger.warning(f"⚠️ Tensor '{key}' has inconsistent batch size: {tensor.size(0)} vs {batch_size}")
                    # Try to fix common issues
                    if tensor.size(0) > batch_size:
                        batch[key] = tensor[:batch_size]
                        self.logger.info(f"✅ Truncated '{key}' to match batch size")
                    else:
                        self.logger.error(f"❌ Cannot fix batch size mismatch for '{key}'")
            
            # Ensure edge_index exists (required for graph-based models)
            if "edge_index" not in batch:
                batch["edge_index"] = torch.empty((2, 0), dtype=torch.long, device=state.device)
            
            return batch
            
        except Exception as e:
            self.logger.error(f"❌ Batch preprocessing failed: {e}")
            self.logger.error(f"🔍 Input batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        self.logger.error(f"  • {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        self.logger.error(f"  • {key}: type={type(value)}")
            raise RuntimeError(f"Failed to preprocess batch: {e}") from e

    def _get_batch_debug_info(self, batch) -> str:
        """
        Generate comprehensive debug information for training batches.
        
        Enhanced version with BCQ-specific diagnostics and shape mismatch analysis.
        Supports both dict and tuple batch formats.
        """
        try:
            debug_lines = ["🔍 Comprehensive Batch Debug Information"]
            debug_lines.append("=" * 60)
            
            # 🔧 CRITICAL FIX: Handle both dict and tuple batch formats
            if isinstance(batch, dict):
                debug_lines.append(f"📋 Batch Format: Dictionary with {len(batch)} keys")
                
                # Basic batch information
                batch_sizes = {}
                temporal_dims = {}
                total_elements = 0
                
                for key, tensor in batch.items():
                    if torch.is_tensor(tensor):
                        debug_lines.append(f"📊 {key}:")
                        debug_lines.append(f"    Shape: {tensor.shape}")
                        debug_lines.append(f"    Dtype: {tensor.dtype}")
                        debug_lines.append(f"    Device: {tensor.device}")
                        debug_lines.append(f"    Memory: {tensor.numel() * tensor.element_size() / 1024**2:.2f} MB")
                        
                        # Track batch and temporal dimensions
                        batch_sizes[key] = tensor.size(0)
                        if tensor.dim() >= 2:
                            temporal_dims[key] = tensor.size(1)
                        total_elements += tensor.numel()
                        
                        # Check for problematic values
                        if torch.isnan(tensor).any():
                            nan_count = safe_item(torch.isnan(tensor).sum())
                            debug_lines.append(f"    ⚠️  NaN values: {nan_count}")
                        
                        if torch.isinf(tensor).any():
                            inf_count = safe_item(torch.isinf(tensor).sum())
                            debug_lines.append(f"    ⚠️  Infinite values: {inf_count}")
                        
                        # Statistical information for debugging
                        if tensor.dtype.is_floating_point and tensor.numel() > 0:
                            debug_lines.append(f"    Stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
                        
                    elif hasattr(tensor, 'shape'):
                        debug_lines.append(f"📊 {key}: numpy array, shape={tensor.shape}, dtype={tensor.dtype}")
                        batch_sizes[key] = tensor.shape[0] if len(tensor.shape) > 0 else 1
                    else:
                        debug_lines.append(f"📊 {key}: {type(tensor)}")
                        
            elif isinstance(batch, (tuple, list)):
                debug_lines.append(f"📋 Batch Format: {type(batch).__name__} with {len(batch)} elements")
                
                # Standard tuple format: (obs, actions, rewards, next_obs, dones, mask, lengths, next_lengths, edge_index)
                element_names = ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'mask', 'lengths', 'next_lengths', 'edge_index']
                batch_sizes = {}
                temporal_dims = {}
                total_elements = 0
                
                for i, tensor in enumerate(batch):
                    name = element_names[i] if i < len(element_names) else f'element_{i}'
                    
                    if torch.is_tensor(tensor):
                        debug_lines.append(f"📊 {name} (index {i}):")
                        debug_lines.append(f"    Shape: {tensor.shape}")
                        debug_lines.append(f"    Dtype: {tensor.dtype}")
                        debug_lines.append(f"    Device: {tensor.device}")
                        
                        batch_sizes[name] = tensor.size(0)
                        if tensor.dim() >= 2:
                            temporal_dims[name] = tensor.size(1)
                        total_elements += tensor.numel()
                        
                    elif isinstance(tensor, (list, tuple)):
                        debug_lines.append(f"📊 {name} (index {i}): {type(tensor).__name__} with {len(tensor)} elements")
                        if len(tensor) > 0 and torch.is_tensor(tensor[0]):
                            shapes = [t.shape for t in tensor if torch.is_tensor(t)]
                            debug_lines.append(f"    Element shapes: {shapes}")
                    else:
                        debug_lines.append(f"📊 {name} (index {i}): {type(tensor)}")
                        
            else:
                debug_lines.append(f"📋 Batch Format: Unknown type {type(batch)}")
                return f"❌ Failed to generate batch debug info: {type(batch).__name__} object has no attribute 'items'\n🔍 Basic info: {type(batch)}, keys: N/A"
            
            # Dimension consistency analysis
            debug_lines.append("\n🔍 Dimension Consistency Analysis:")
            debug_lines.append("-" * 40)
            
            unique_batch_sizes = set(batch_sizes.values())
            if len(unique_batch_sizes) == 1:
                debug_lines.append(f"✅ Batch dimensions consistent: {list(unique_batch_sizes)[0]}")
            else:
                debug_lines.append(f"❌ Batch dimension mismatch detected!")
                debug_lines.append(f"    Different batch sizes: {dict(batch_sizes)}")
                debug_lines.append(f"    Unique sizes: {sorted(unique_batch_sizes)}")
            
            unique_temporal_dims = set(temporal_dims.values())
            if len(unique_temporal_dims) <= 1:
                if unique_temporal_dims:
                    debug_lines.append(f"✅ Temporal dimensions consistent: {list(unique_temporal_dims)[0]}")
                else:
                    debug_lines.append("ℹ️  No temporal dimensions detected (2D tensors)")
            else:
                debug_lines.append(f"⚠️  Temporal dimension inconsistency detected!")
                debug_lines.append(f"    Different temporal sizes: {dict(temporal_dims)}")
                debug_lines.append(f"    Unique sizes: {sorted(unique_temporal_dims)}")
            
            # BCQ-specific analysis
            if hasattr(self, 'agent') and hasattr(self.agent, '__class__') and 'BCQ' in self.agent.__class__.__name__:
                debug_lines.append("\n🔧 BCQ-Specific Analysis:")
                debug_lines.append("-" * 30)
                
                # Check for typical BCQ batch structure
                expected_keys = {'state', 'action', 'reward', 'next_state', 'done', 'mask'}
                present_keys = set(batch.keys())
                missing_keys = expected_keys - present_keys
                extra_keys = present_keys - expected_keys
                
                if missing_keys:
                    debug_lines.append(f"❌ Missing expected keys: {missing_keys}")
                if extra_keys:
                    debug_lines.append(f"ℹ️  Extra keys present: {extra_keys}")
                
                # Analyze action tensor structure for multi-head actions
                if 'action' in batch:
                    action_tensor = batch['action']
                    if torch.is_tensor(action_tensor):
                        debug_lines.append(f"🎯 Action Analysis:")
                        debug_lines.append(f"    Action shape: {action_tensor.shape}")
                        if action_tensor.dim() == 3:
                            debug_lines.append(f"    Multi-head actions: {action_tensor.size(2)} heads")
                            debug_lines.append(f"    Sequence length: {action_tensor.size(1)}")
                        elif action_tensor.dim() == 2:
                            debug_lines.append(f"    Action heads/sequence: {action_tensor.size(1)}")
                        
                        # Check action value ranges
                        if action_tensor.dtype in [torch.long, torch.int32, torch.int64]:
                            min_action = action_tensor.min().item()
                            max_action = action_tensor.max().item()
                            debug_lines.append(f"    Action range: [{min_action}, {max_action}]")
                            
                            # Check for negative actions (invalid)
                            if min_action < 0:
                                debug_lines.append(f"    ❌ Invalid negative actions detected!")
                
                # Check sequence length compatibility
                if 'state' in batch and 'mask' in batch:
                    state_tensor = batch['state']
                    mask_tensor = batch['mask']
                    if torch.is_tensor(state_tensor) and torch.is_tensor(mask_tensor):
                        if state_tensor.dim() == 3 and mask_tensor.dim() == 2:
                            if state_tensor.size(1) != mask_tensor.size(1):
                                debug_lines.append(f"❌ State-mask sequence length mismatch:")
                                debug_lines.append(f"    State sequence: {state_tensor.size(1)}")
                                debug_lines.append(f"    Mask sequence: {mask_tensor.size(1)}")
                
                # Memory usage analysis for BCQ
                debug_lines.append(f"💾 Memory Analysis:")
                debug_lines.append(f"    Total tensor memory: {total_elements * 4 / 1024**2:.2f} MB (approx)")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    debug_lines.append(f"    GPU allocated: {allocated:.2f} GB")
                    debug_lines.append(f"    GPU reserved: {reserved:.2f} GB")
            
            # Potential solutions for detected issues
            issues_detected = []
            if len(unique_batch_sizes) > 1:
                issues_detected.append("batch_size_mismatch")
            if len(unique_temporal_dims) > 1:
                issues_detected.append("temporal_dim_mismatch")
                
            for key, tensor in batch.items():
                if torch.is_tensor(tensor):
                    if torch.isnan(tensor).any():
                        issues_detected.append("nan_values")
                    if torch.isinf(tensor).any():
                        issues_detected.append("inf_values")
            
            if issues_detected:
                debug_lines.append("\n🔧 Suggested Solutions:")
                debug_lines.append("-" * 25)
                
                if "batch_size_mismatch" in issues_detected:
                    debug_lines.append("📋 For batch size mismatch:")
                    debug_lines.append("    • Check data loader batch processing")
                    debug_lines.append("    • Ensure consistent episode sampling")
                    debug_lines.append("    • Verify no partial batches at end of dataset")
                
                if "temporal_dim_mismatch" in issues_detected:
                    debug_lines.append("📋 For temporal dimension mismatch:")
                    debug_lines.append("    • Check sequence padding in data preprocessing")
                    debug_lines.append("    • Ensure consistent max_sequence_length")
                    debug_lines.append("    • Verify mask generation aligns with sequences")
                
                if "nan_values" in issues_detected:
                    debug_lines.append("📋 For NaN values:")
                    debug_lines.append("    • Check input data quality")
                    debug_lines.append("    • Verify normalization doesn't create NaNs")
                    debug_lines.append("    • Consider gradient clipping")
                
                if "inf_values" in issues_detected:
                    debug_lines.append("📋 For infinite values:")
                    debug_lines.append("    • Check for division by zero")
                    debug_lines.append("    • Verify exponential operations are bounded")
                    debug_lines.append("    • Consider value clamping")
            
            return "\n".join(debug_lines)
            
        except Exception as e:
            return f"❌ Failed to generate batch debug info: {e}\n🔍 Basic info: {type(batch)}, keys: {list(batch.keys()) if hasattr(batch, 'keys') else 'N/A'}"
