"""
End-to-end experiment runner for all baseline algorithms and PoG-BVE.

Changes:
• All parameters now read from yaml configuration files
• Minimal CLI interface with only config file paths
• Removed redundant command-line argument overrides
• Enhanced yaml-based configuration system
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from Libs.exp.trainer import Trainer
from Libs.utils.log_utils import get_logger
from Libs.utils.task_adapter import (TaskAdapter,
                                     validate_model_task_combination)
from Libs.utils.task_manager import (get_current_task_config, get_task_manager,
                                     set_global_task)
# Add visualization imports
from Libs.utils.vis_utils import (plot_ope_comparison_with_ci,
                                  plot_policy_distribution_comparison,
                                  save_figure_publication_ready,
                                  set_plot_style)

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
#  Minimal CLI (only config file paths)
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse minimal command-line arguments for configuration file paths."""
    parser = argparse.ArgumentParser(
        description="Run offline RL experiments based on yaml configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python -m Libs.exp.run_all
    
    # Run with custom config files
    python -m Libs.exp.run_all --config Libs/configs/exp.yaml --dataset-config Libs/configs/dataset.yaml
    
    # Run with custom experiment config only
    python -m Libs.exp.run_all --config custom_exp.yaml
        """
    )

    parser.add_argument(
        "--config", "-c",
        default="Libs/configs/exp.yaml",
        help="Path to experiment configuration file (default: Libs/configs/exp.yaml)"
    )
    parser.add_argument(
        "--dataset-config", "-d",
        default="Libs/configs/dataset.yaml",
        help="Path to dataset configuration file (default: Libs/configs/dataset.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from yaml file with environment variable overrides.

    Args:
        config_path: Path to experiment configuration file.

    Returns:
        Complete experiment configuration dictionary with environment variable overrides.
    """
    import os

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "experiment" not in config:
        raise ValueError(
            f"Configuration file {config_path} must contain 'experiment' section")

    # Apply environment variable overrides
    exp_cfg = config["experiment"]

    # Override task if TASK environment variable is set
    if "TASK" in os.environ:
        exp_cfg["task"] = os.environ["TASK"]
        logger.debug(
            f"Overriding task with environment variable: {os.environ['TASK']}")

    # Override device if DEVICE environment variable is set
    if "DEVICE" in os.environ:
        exp_cfg["device"] = os.environ["DEVICE"]
        logger.debug(
            f"Overriding device with environment variable: {os.environ['DEVICE']}")

    # Override other common parameters if set
    if "SEED" in os.environ:
        exp_cfg["seed"] = int(os.environ["SEED"])
        logger.debug(
            f"Overriding seed with environment variable: {os.environ['SEED']}")

    if "N_EPOCHS" in os.environ:
        exp_cfg["n_epochs"] = int(os.environ["N_EPOCHS"])
        logger.debug(
            f"Overriding n_epochs with environment variable: {os.environ['N_EPOCHS']}")

    if "BATCH_SIZE" in os.environ:
        exp_cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
        logger.debug(
            f"Overriding batch_size with environment variable: {os.environ['BATCH_SIZE']}")

    return config


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """Load dataset configuration from yaml file.

    Args:
        config_path: Path to dataset configuration file.

    Returns:
        Complete dataset configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_task_config(dataset_config: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Get task-specific configuration from dataset config.

    Args:
        dataset_config: Dataset configuration dictionary.
        task: Task name ('vent', 'rrt', 'iv').

    Returns:
        Task-specific configuration dictionary.
    """
    if task not in dataset_config.get("tasks", {}):
        raise ValueError(f"Task '{task}' not found in dataset configuration")

    task_config = dataset_config["tasks"][task].copy()

    # Merge with global config
    merged_config = dataset_config.copy()
    merged_config.update(task_config)

    # Set task-specific paths
    base_data_root = Path(dataset_config["data_root"])
    merged_config["data_root"] = str(
        base_data_root / task_config["output_dir"])
    merged_config["data_filename"] = task_config["processed_files"]["pyg_graph"]

    return merged_config


def infer_dims_from_graph(graph_pt: Path, task_name: str) -> tuple[int, List[int]]:
    """Infer state and action dimensions from graph file.

    Args:
        graph_pt: Path to the graph file.
        task_name: Name of the task for fallback configuration.

    Returns:
        Tuple of (state_dim, action_dims).
    """
    data = torch.load(graph_pt)
    state_dim = data.x.shape[-1]

    if hasattr(data, 'actions'):
        # Compute action dimensions from actual data
        action_dims = []
        for i in range(data.actions.shape[-1]):
            max_action = int(data.actions[..., i].max().item())
            action_dims.append(max_action + 1)
    else:
        # Fallback to task-specific configuration
        task_manager = get_task_manager()
        action_dims = task_manager.get_action_dims(task_name)
        logger.warning(
            f"No action data found, using task config for {task_name}: {action_dims}")

    return state_dim, action_dims


# -----------------------------------------------------------------------------
#  Main execution function
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """Main execution function - all parameters read from yaml configuration."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 1) Load configurations from yaml files ---------------------------
    # ------------------------------------------------------------------
    logger.info("Loading configuration files...")

    # Load experiment configuration
    exp_config = load_experiment_config(args.config)
    exp_cfg: Dict[str, Any] = exp_config.get("experiment", {})

    # Load dataset configuration
    dataset_config = load_dataset_config(args.dataset_config)

    # Get task from experiment config (primary) or dataset config (fallback)
    task = exp_cfg.get("task") or dataset_config.get("task")
    if not task:
        raise ValueError(
            "Task must be specified in experiment configuration (experiment.task)")

    logger.info(f"Selected task: {task}")

    # ------------------------------------------------------------------
    # 2) Load task-specific configuration -------------------------------
    # ------------------------------------------------------------------
    task_config = get_task_config(dataset_config, task)

    # Initialize TaskManager with current task
    task_manager = get_task_manager()
    set_global_task(task)
    current_task_config = get_current_task_config()

    data_root = Path(task_config["data_root"])
    data_filename = task_config["data_filename"]
    graph_pt = data_root / data_filename

    if not graph_pt.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {graph_pt}\n"
            f"Please build the dataset first:\n"
            f"python -m Libs.scripts.build_dataset --task {task}"
        )

    logger.info("Task: %s - %s", task, current_task_config.description)
    logger.info("Dataset: %s", graph_pt)
    logger.info("Action columns: %s", current_task_config.action_cols)

    # ------------------------------------------------------------------
    # 3) Infer dimensions from graph file -------------------------------
    # ------------------------------------------------------------------
    state_dim, action_dims = infer_dims_from_graph(graph_pt, task)
    logger.info("Inferred dims — state: %s | actions: %s",
                state_dim, action_dims)

    # Validate action dimensions match task configuration
    if not task_manager.validate_action_dims(action_dims, task):
        logger.warning("Action dimensions don't match task configuration!")
        logger.warning("Using inferred dimensions from data: %s", action_dims)

    # ------------------------------------------------------------------
    # 4) Extract experiment parameters from yaml ------------------------
    # ------------------------------------------------------------------

    # Algorithm selection
    algos: List[str] = [str(a).lower() for a in exp_cfg.get(
        "algos",
        [
            "physician",
            "bc",
            "dqn",
            "cql",
            "bcq",
            "bve",
            "pog_bc",
            "pog_dqn",
            "pog_cql",
            "pog_bcq",
            "pog_bve",
        ],
    )]

    # Training parameters
    epochs: int = int(exp_cfg.get("n_epochs", 10))
    batch_size: int = int(exp_cfg.get("batch_size",
                                      task_config.get("batch_size", 64)))
    device: str = str(exp_cfg.get("device", "cpu"))

    # Data splits
    val_split: float = float(exp_cfg.get("val_split", 0.1))
    test_split: float = float(exp_cfg.get("test_split", 0.1))

    # Bootstrap and confidence intervals
    bootstrap_iters: int = int(exp_cfg.get("bootstrap_iters", 100))
    ci_alpha: float = float(exp_cfg.get("ci_alpha", 0.05))

    # Policy parameters
    policy_mode: str = str(exp_cfg.get("policy_mode", "greedy"))
    temperature = exp_cfg.get("temperature")  # can be None

    # FQE parameters
    enable_fqe: bool = bool(exp_cfg.get("enable_fqe", True))
    fqe_epochs: int = int(exp_cfg.get("fqe_epochs", 20))
    fqe_lr: float = float(exp_cfg.get("fqe_lr", 3e-4))
    fqe_batch_size: int = int(exp_cfg.get("fqe_batch_size", 512))

    # Output paths
    log_root = Path(exp_cfg.get("log_root", "Output/runs")) / task

    # Create task-specific results file in Output/results directory
    results_base = exp_cfg.get(
        "results_base", "Output/results/experiment_results")
    results_fp = Path(f"{results_base}_{task}.csv")

    # Performance settings
    amp_flag: bool = bool(exp_cfg.get("amp", False))
    seed: int = int(exp_cfg.get("seed", 42))

    # Behaviour Policy & OPE parameters
    behav_policy_mode: str = str(exp_cfg.get("behav_policy_mode", "logistic"))
    behav_prob_min: float | None = exp_cfg.get("behav_prob_min")
    behav_prob_max: float | None = exp_cfg.get("behav_prob_max")
    pi_eps_init: float | None = exp_cfg.get("pi_eps")
    clip_range: float | None = exp_cfg.get("clip_range")
    max_joint_enum: int | None = exp_cfg.get("max_joint_enum")
    use_psis: bool | None = exp_cfg.get("use_psis")
    boltz_tau: float | None = exp_cfg.get("boltz_tau")

    # Visualization parameters
    vis_cfg = exp_cfg.get("visualization", {})
    enable_visualization: bool = bool(vis_cfg.get("enable", True))
    vis_style: str = str(vis_cfg.get("style", "publication"))
    vis_formats: List[str] = vis_cfg.get("formats", ["png", "pdf"])
    include_convergence: bool = bool(vis_cfg.get("include_convergence", True))
    include_strategy: bool = bool(vis_cfg.get("include_strategy", False))
    include_comparison: bool = bool(vis_cfg.get("include_comparison", True))
    save_individual: bool = bool(vis_cfg.get("save_individual", True))
    save_comparison: bool = bool(vis_cfg.get("save_comparison", True))

    # Early stopping parameters
    early_stop_metric: str = str(exp_cfg.get(
        "early_stop_metric", "ips_survival"))
    early_stop_patience: int = int(exp_cfg.get("early_stop_patience", 15))
    early_stop_rel_delta: float = float(
        exp_cfg.get("early_stop_rel_delta", 1e-3))
    early_stop_delta = exp_cfg.get("early_stop_delta")  # may be null
    warmup_sample_epochs: int = int(exp_cfg.get("early_stop_warmup", 3))

    # ------------------------------------------------------------------
    # 5) Build data loaders --------------------------------------------
    # ------------------------------------------------------------------
    split_dir = Path(graph_pt).parent  # Same directory as graph file
    train_loader, val_loader, test_loader = Trainer.build_dataloaders_from_graph_with_splits(
        graph_pt=graph_pt,
        split_dir=split_dir,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # 6) Initialize results storage ------------------------------------
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []  # CSV rows
    from datetime import datetime

    # Add task info to results
    results["task"] = task
    results["task_description"] = current_task_config.description
    results["action_cols"] = current_task_config.action_cols
    results["action_names"] = current_task_config.action_names
    results["n_actions"] = len(action_dims)
    results["action_dims"] = action_dims
    results["medical_context"] = current_task_config.medical_context

    # ------------------------------------------------------------------
    # 7) Pre-compute FQE transitions ----------------------------------
    # ------------------------------------------------------------------
    if enable_fqe:
        data = torch.load(graph_pt)
        N, T, F = data.x.shape  # type: ignore
        # Flatten along time dim excluding last step
        state = data.x[:, :-1].reshape(-1, F)
        next_state = data.x[:, 1:].reshape(-1, F)
        action = data.actions[:, :-1].reshape(-1, data.actions.size(-1))
        # Reward: reuse heuristic as trainer
        reward = data.x[:, :-1, -2].reshape(-1, 1)
        done = (torch.zeros_like(reward)).bool()
        # Mark terminal when t == T-2 (next_state last step)
        done_indices = torch.arange(T-1) == (T-2)
        done = done_indices.repeat(N, 1).reshape(-1, 1)

        transitions_dict = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done.float(),
        }

    # ------------------------------------------------------------------
    # 8) Display configuration summary ---------------------------------
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Experiment Configuration Summary")
    logger.info("=" * 80)
    logger.info("Configuration files:")
    logger.info("  Experiment: %s", args.config)
    logger.info("  Dataset: %s", args.dataset_config)
    logger.info("Task: %s", task)
    logger.info("Algorithms: %s", algos)
    logger.info("Training epochs: %d", epochs)
    logger.info("Batch size: %d", batch_size)
    logger.info("Device: %s", device)
    logger.info("Random seed: %d", seed)
    logger.info("Enable FQE: %s", enable_fqe)
    logger.info("Enable visualization: %s", enable_visualization)
    logger.info("Results file: %s", results_fp)
    logger.info("=" * 80)

    # Initialize task adapter for compatibility checks
    task_adapter = TaskAdapter()

    # ------------------------------------------------------------------
    # 9) Run experiments for each algorithm ----------------------------
    # ------------------------------------------------------------------
    for algo in algos:
        logger.info("\n=== [RUN] %s ===", algo.upper())

        # Check critical compatibility issues that would cause failures
        # Note: Removed outdated PoG-BVE dimension check - it now supports any number of dimensions
        # if algo == "pog_bve" and len(action_dims) != 3:
        #     logger.warning(f"🚫 Skipping {algo.upper()}: requires exactly 3 action dimensions, "
        #                  f"but task '{task}' has {len(action_dims)}. Use 'bve' instead.")
        #     continue

        # Check model-task compatibility
        compat = task_adapter.check_compatibility(algo, task)
        if not compat['compatible']:
            logger.warning(
                f"⚠️  {algo.upper()} may not be compatible with task {task}")
            logger.warning(f"Reason: {compat['reason']}")
            if compat['suggestions']:
                logger.warning(
                    f"Consider using: {', '.join(compat['suggestions'])}")
            logger.warning(
                "Proceeding anyway, but results may be suboptimal...")

        log_dir = log_root / algo
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            trainer = Trainer(
                algo=algo,
                state_dim=state_dim,
                action_dims=action_dims,
                device=device,
                amp=amp_flag,
                log_dir=log_dir,
                policy_mode=policy_mode,
                temperature=temperature,
                fqe_epochs=fqe_epochs,
                fqe_batch_size=fqe_batch_size,
                fqe_lr=fqe_lr,
                # Behaviour policy parameters
                behav_policy_mode=behav_policy_mode,
                behav_prob_min=behav_prob_min,
                behav_prob_max=behav_prob_max,
                pi_eps_init=pi_eps_init,
                clip_range=clip_range,
                max_joint_enum=max_joint_enum,
                use_psis=use_psis,
                boltz_tau=boltz_tau,
                # Visualization parameters
                enable_visualization=enable_visualization,
                visualization_formats=vis_formats,
                vis_style=vis_style,
                # Bootstrap configuration
                bootstrap_iters=bootstrap_iters,
            )

            # Train the model
            trainer.fit(
                train_loader,
                n_epochs=epochs,
                val_loader=val_loader,
                early_stop_metric=early_stop_metric,
                patience=early_stop_patience,
                delta=early_stop_delta,
                rel_delta=early_stop_rel_delta,
                warmup_sample_epochs=warmup_sample_epochs,
            )

            # Evaluate the model
            import time as _time
            _t0 = _time.time()
            metrics = trainer.evaluate(
                test_loader,
                split="test",
                bootstrap_iters=bootstrap_iters,
                alpha=ci_alpha,
            )
            runtime_s = _time.time() - _t0

            # Store results
            metrics["runtime_s"] = round(runtime_s, 2)
            results[algo] = metrics

            # Prepare CSV row
            row_dict = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "baseline": algo,
                "seed": seed,
                "reward": metrics.get("reward"),
                "reward_ci": metrics.get("reward_ci"),
                "ipw_reward": metrics.get("ipw_reward"),
                # 📝 CLARIFICATION: This is dataset-level, not algorithm-specific
                "dataset_survival_rate": metrics.get("survival_rate"),
                "dataset_reward": metrics.get("dataset_reward"),
                # 📝 This is the algorithm-specific survival estimate
                "ips_survival": metrics.get("ips_survival"),
                "ips_survival_ci": metrics.get("ips_survival_ci"),
                "runtime_s": round(runtime_s, 2),
            }
            rows.append(row_dict)

        except Exception as e:
            logger.error(f"❌ Failed to train/evaluate {algo.upper()}: {e}")
            logger.error(f"🔍 Error type: {type(e).__name__}")

            # Store error results in proper dictionary format for visualization compatibility
            error_metrics = {
                "wdr_reward": 0.0,
                "ipw_reward": 0.0,
                "fqe_est": 0.0,
                "survival_rate": 0.0,
                "ips_survival": 0.0,
                "reward": 0.0,
                "reward_ci": [0.0, 0.0],
                "dataset_reward": 0.0,
                "ips_survival_ci": [0.0, 0.0],
                "runtime_s": 0.0,
                "error": str(e),
                "status": "failed"
            }
            results[algo] = error_metrics

            # Prepare CSV row for failed experiment
            row_dict = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "baseline": algo,
                "seed": seed,
                "reward": 0.0,
                "reward_ci": "[0.0, 0.0]",
                "ipw_reward": 0.0,
                "dataset_survival_rate": 0.0,
                "dataset_reward": 0.0,
                "ips_survival": 0.0,
                "ips_survival_ci": "[0.0, 0.0]",
                "runtime_s": 0.0,
            }
            rows.append(row_dict)

            # Continue with next algorithm
            continue

    # ------------------------------------------------------------------
    # 10) Save results to CSV file ------------------------------------
    # ------------------------------------------------------------------
    import csv
    import os
    results_fp.parent.mkdir(parents=True, exist_ok=True)

    # CSV header definition
    header = [
        "timestamp",
        "baseline",
        "seed",
        "reward",
        "reward_ci",
        "ipw_reward",
        # 📝 CLARIFICATION: Dataset-level baseline, not algorithm-specific
        "dataset_survival_rate",
        "dataset_reward",
        "ips_survival",  # 📝 Algorithm-specific survival estimate using Importance Sampling
        "ips_survival_ci",
        "runtime_s",
    ]

    write_header = not results_fp.exists()
    with open(results_fp, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if write_header:
            writer.writeheader()
        for r in rows:
            # Convert lists to string format for CSV compatibility
            for k, v in list(r.items()):
                if isinstance(v, (list, tuple)):
                    # Convert to simple string format [val1, val2]
                    r[k] = f"[{', '.join(str(x) for x in v)}]"
            writer.writerow(r)

    logger.info("Results saved to %s", results_fp)

    # ------------------------------------------------------------------
    # 11) Generate experiment comparison visualizations ----------------
    # ------------------------------------------------------------------
    if enable_visualization and save_comparison:
        try:
            logger.info("🎨 Generating experiment comparison visualizations...")

            # Set publication style for visualizations
            set_plot_style(vis_style)

            # Create visualization directory
            vis_dir = Path("Output/experiment_visualizations") / task
            vis_dir.mkdir(parents=True, exist_ok=True)

            # 1. OPE Methods Comparison
            if results and include_comparison:
                generate_ope_comparison_chart(results, vis_dir, vis_formats)

            # 2. Algorithm Performance Summary
            if results and include_comparison:
                generate_algorithm_summary_chart(results, vis_dir, vis_formats)

            # 3. Policy Distribution Comparison (if data available)
            if include_strategy:
                generate_policy_comparison_charts(
                    results, vis_dir, train_loader)

            logger.info("✅ Experiment visualization generation complete")

        except Exception as e:
            logger.warning(
                "⚠️  Experiment visualization generation failed: %s", e)
    elif enable_visualization:
        logger.info(
            "ℹ️  Experiment comparison visualizations disabled in config")
    else:
        logger.info("ℹ️  Visualization disabled in config")

    # ------------------------------------------------------------------
    # 12) Analyze experiment results for anomalies ---------------------
    # ------------------------------------------------------------------
    try:
        logger.info("🔍 Analyzing experiment results for potential issues...")
        analysis = analyze_experiment_results(results_fp)
        print_analysis_report(analysis)

        # Save analysis report
        analysis_fp = results_fp.parent / f"analysis_{results_fp.stem}.json"
        import json
        with open(analysis_fp, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info("📊 Analysis report saved to %s", analysis_fp)

    except Exception as e:
        logger.warning("⚠️  Experiment analysis failed: %s", e)

    logger.info("\n" + "=" * 80)
    logger.info("🎉 EXPERIMENT BATCH COMPLETE")
    logger.info("=" * 80)
    logger.info("📁 Results saved to: %s", results_fp)
    if enable_visualization:
        logger.info(
            "🎨 Visualizations saved to: Output/experiment_visualizations/%s", task)
    logger.info("📊 Analysis report: %s", results_fp.parent /
                f"analysis_{results_fp.stem}.json")
    logger.info("=" * 80)


def generate_ope_comparison_chart(results: Dict[str, Any], vis_dir: Path, vis_formats: List[str]) -> None:
    """Generate OPE methods comparison chart across all algorithms."""
    try:
        # Enhanced input validation
        if not results:
            logger.warning("⚠️  No results provided for OPE comparison chart")
            return

        if not isinstance(results, dict):
            logger.error(
                "❌ Results must be a dictionary, got: %s", type(results))
            return

        # Extract OPE estimates for comparison
        algorithms = []
        wdr_estimates = []
        ipw_estimates = []
        fqe_estimates = []
        survival_rates = []

        # Prepare confidence intervals (placeholder for now)
        wdr_cis = []
        ipw_cis = []

        # Enhanced logging for debugging
        logger.info(
            "🔍 Processing %d algorithm results for OPE comparison", len(results))

        # 🔧 CRITICAL FIX: Filter out non-algorithm metadata from results
        # The results dict contains both algorithm results and task metadata
        metadata_keys = {"task", "task_description", "action_cols",
                         "action_names", "n_actions", "action_dims", "medical_context"}

        for algo, metrics in results.items():
            # Skip metadata entries that are not algorithm results
            if algo in metadata_keys:
                logger.debug(f"🔍 Skipping metadata entry: {algo}")
                continue

            # 🔧 ENHANCED FIX: Comprehensive data type and structure validation
            if not isinstance(metrics, dict):
                logger.warning(
                    f"⚠️  Skipping {algo}: metrics is not a dictionary (type: {type(metrics)}, value: {str(metrics)[:100]}...)")
                continue

            # Additional structure validation
            if not metrics:
                logger.warning(
                    f"⚠️  Skipping {algo}: metrics dictionary is empty")
                continue

            algorithms.append(algo.upper())

            # Extract metrics with fallbacks and safe conversion
            def safe_get_float(key: str, default: float = np.nan) -> float:
                """Safely extract float value from metrics dict."""
                value = metrics.get(key, default)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return default

            def safe_get_ci(key: str, fallback_val: float) -> List[float]:
                """Safely extract confidence interval from metrics dict."""
                ci = metrics.get(
                    key, [fallback_val * 0.95, fallback_val * 1.05])
                if isinstance(ci, str):
                    try:
                        # Try to parse JSON string
                        import json
                        ci = json.loads(ci)
                    except (json.JSONDecodeError, TypeError):
                        return [fallback_val * 0.95, fallback_val * 1.05]

                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    try:
                        return [float(ci[0]), float(ci[1])]
                    except (ValueError, TypeError):
                        return [fallback_val * 0.95, fallback_val * 1.05]
                else:
                    return [fallback_val * 0.95, fallback_val * 1.05]

            # Extract metrics with fallbacks
            wdr_val = safe_get_float("wdr_reward", safe_get_float("reward"))
            ipw_val = safe_get_float("ipw_reward")
            fqe_val = safe_get_float("fqe_est")
            surv_val = safe_get_float("survival_rate")

            wdr_estimates.append(wdr_val)
            ipw_estimates.append(ipw_val)
            fqe_estimates.append(fqe_val)
            survival_rates.append(surv_val)

            # Extract confidence intervals if available
            reward_ci = safe_get_ci("reward_ci", wdr_val)
            wdr_cis.append((reward_ci[0], reward_ci[1]))

            # Placeholder IPW CI
            ipw_ci = safe_get_ci("ipw_ci", ipw_val)
            ipw_cis.append((ipw_ci[0], ipw_ci[1]))

        # Generate OPE comparison plot for valid algorithms
        valid_indices = [i for i, val in enumerate(
            wdr_estimates) if not np.isnan(val) and np.isfinite(val)]
        if valid_indices:
            valid_algos = [algorithms[i] for i in valid_indices]
            valid_estimates = [wdr_estimates[i] for i in valid_indices]
            valid_cis = [wdr_cis[i] for i in valid_indices]

            baseline_value = np.nanmean(
                [s for s in survival_rates if not np.isnan(s)]) if survival_rates else None

            fig = plot_ope_comparison_with_ci(
                methods=valid_algos,
                estimates=valid_estimates,
                confidence_intervals=valid_cis,
                baseline_value=baseline_value,
                title="Algorithm Performance Comparison (WDR Estimates)"
            )

            save_figure_publication_ready(
                fig,
                vis_dir / "ope_comparison",
                formats=vis_formats
            )

            logger.info(
                "📊 Generated OPE comparison chart successfully with %d algorithms", len(valid_algos))
            logger.info("   • Processed algorithms: %s",
                        ', '.join(valid_algos))
        else:
            logger.warning(
                "⚠️  No valid algorithms found for OPE comparison chart")
            logger.info("   • Total entries processed: %d", len(results))
            logger.info("   • Valid algorithm entries: 0")
            logger.info(
                "   • Recommendation: Check algorithm training logs for failures")

        # Final summary
        total_processed = len([k for k in results.keys() if k not in {
                              "task", "task_description", "action_cols", "action_names", "n_actions", "action_dims", "medical_context"}])
        valid_count = len(algorithms)
        logger.info("🎯 OPE comparison summary: %d/%d algorithms processed successfully",
                    valid_count, total_processed)

    except Exception as e:
        logger.warning("Failed to generate OPE comparison chart: %s", e)
        logger.warning("   • Error type: %s", type(e).__name__)
        logger.warning(
            "   • This may be due to missing dependencies or corrupted data")
        import traceback
        logger.debug("Full traceback: %s", traceback.format_exc())


def generate_algorithm_summary_chart(results: Dict[str, Any], vis_dir: Path, vis_formats: List[str]) -> None:
    """Generate algorithm performance summary chart."""
    try:
        import matplotlib.pyplot as plt

        # Enhanced input validation
        if not results:
            logger.warning(
                "⚠️  No results provided for algorithm summary chart")
            return

        if not isinstance(results, dict):
            logger.error(
                "❌ Results must be a dictionary, got: %s", type(results))
            return

        # Extract performance metrics
        algorithms = []
        metrics_data = {
            'WDR Reward': [],
            'IPW Reward': [],
            'Survival Rate': [],
            'Runtime (s)': []
        }

        # Enhanced logging for debugging
        logger.info(
            "🔍 Processing %d algorithm results for summary chart", len(results))

        # 🔧 CRITICAL FIX: Filter out non-algorithm metadata from results
        # The results dict contains both algorithm results and task metadata
        metadata_keys = {"task", "task_description", "action_cols",
                         "action_names", "n_actions", "action_dims", "medical_context"}

        for algo, metrics in results.items():
            # Skip metadata entries that are not algorithm results
            if algo in metadata_keys:
                logger.debug(f"🔍 Skipping metadata entry: {algo}")
                continue

            # 🔧 ENHANCED FIX: Comprehensive data type and structure validation
            if not isinstance(metrics, dict):
                logger.warning(
                    f"⚠️  Skipping {algo}: metrics is not a dictionary (type: {type(metrics)}, value: {str(metrics)[:100]}...)")
                continue

            # Additional structure validation
            if not metrics:
                logger.warning(
                    f"⚠️  Skipping {algo}: metrics dictionary is empty")
                continue

            algorithms.append(algo.upper())

            # Safe extraction function
            def safe_get_float(key: str, default: float = 0.0) -> float:
                """Safely extract float value from metrics dict."""
                value = metrics.get(key, default)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return default

            # Extract metrics with safe conversion
            wdr_reward = safe_get_float(
                "wdr_reward", safe_get_float("reward", 0.0))
            ipw_reward = safe_get_float("ipw_reward", 0.0)
            dataset_survival_rate = safe_get_float("dataset_survival_rate", safe_get_float(
                "survival_rate", 0.0))  # Backward compatibility
            runtime_s = safe_get_float("runtime_s", 0.0)

            metrics_data['WDR Reward'].append(wdr_reward)
            metrics_data['IPW Reward'].append(ipw_reward)
            metrics_data['Survival Rate'].append(dataset_survival_rate)
            metrics_data['Runtime (s)'].append(runtime_s)

        # Only proceed if we have valid algorithms
        if not algorithms:
            logger.warning("⚠️  No valid algorithms found for summary chart")
            logger.info("   • Total entries processed: %d", len(results))
            logger.info("   • Valid algorithm entries: 0")
            logger.info(
                "   • Recommendation: Check algorithm training logs for failures")
            total_processed = len([k for k in results.keys() if k not in {
                                  "task", "task_description", "action_cols", "action_names", "n_actions", "action_dims", "medical_context"}])
            logger.info(
                "🎯 Algorithm summary chart: 0/%d algorithms processed successfully", total_processed)
            return

        # Create summary heatmap
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # WDR Reward comparison
        ax1.bar(
            algorithms, metrics_data['WDR Reward'], color='steelblue', alpha=0.7)
        ax1.set_title('WDR Reward Estimates')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)

        # IPW Reward comparison
        ax2.bar(algorithms, metrics_data['IPW Reward'],
                color='forestgreen', alpha=0.7)
        ax2.set_title('IPW Reward Estimates')
        ax2.set_ylabel('Reward')
        ax2.tick_params(axis='x', rotation=45)

        # Survival Rate comparison
        ax3.bar(
            algorithms, metrics_data['Survival Rate'], color='coral', alpha=0.7)
        ax3.set_title('Survival Rate')
        ax3.set_ylabel('Rate')
        ax3.tick_params(axis='x', rotation=45)

        # Runtime comparison
        ax4.bar(
            algorithms, metrics_data['Runtime (s)'], color='gold', alpha=0.7)
        ax4.set_title('Training Runtime')
        ax4.set_ylabel('Seconds')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save the figure
        save_figure_publication_ready(
            fig,
            vis_dir / "algorithm_summary",
            formats=vis_formats
        )

        plt.close(fig)
        logger.info(
            "📊 Generated algorithm summary chart successfully with %d algorithms", len(algorithms))
        logger.info("   • Processed algorithms: %s", ', '.join(algorithms))

        # Final summary
        total_processed = len([k for k in results.keys() if k not in {
                              "task", "task_description", "action_cols", "action_names", "n_actions", "action_dims", "medical_context"}])
        valid_count = len(algorithms)
        logger.info("🎯 Algorithm summary chart: %d/%d algorithms processed successfully",
                    valid_count, total_processed)

    except Exception as e:
        logger.warning("Failed to generate algorithm summary chart: %s", e)
        logger.warning("   • Error type: %s", type(e).__name__)
        logger.warning(
            "   • This may be due to missing dependencies or corrupted data")
        import traceback
        logger.debug("Full traceback: %s", traceback.format_exc())


def generate_policy_comparison_charts(results: Dict[str, Any], vis_dir: Path, train_loader) -> None:
    """Generate policy distribution comparison charts."""
    try:
        # This is a placeholder for policy comparison
        # In practice, you would need to sample actions from each trained policy
        logger.info(
            "📊 Policy comparison charts require trained models - skipping for now")

        # TODO: Implement policy action sampling for comparison
        # This would involve:
        # 1. Loading saved model checkpoints
        # 2. Sampling actions from each policy on a common state set
        # 3. Generating policy distribution comparison plots

    except Exception as e:
        logger.warning("Failed to generate policy comparison charts: %s", e)


def analyze_experiment_results(results_csv_path: Path) -> Dict[str, Any]:
    """Analyze experiment results for potential anomalies and issues.

    Args:
        results_csv_path: Path to the experiment results CSV file

    Returns:
        Dictionary containing analysis results and detected anomalies
    """
    analysis_results = {
        "anomalies": [],
        "warnings": [],
        "statistics": {},
        "recommendations": []
    }

    try:
        import numpy as np
        import pandas as pd

        # Read the CSV file
        df = pd.read_csv(results_csv_path)

        if df.empty:
            analysis_results["warnings"].append("Empty results file")
            return analysis_results

        logger.info("🔍 Analyzing experiment results from %s", results_csv_path)
        logger.info("📊 Found %d algorithms: %s",
                    len(df), df['baseline'].tolist())

        # Basic statistics
        analysis_results["statistics"]["n_algorithms"] = len(df)
        analysis_results["statistics"]["algorithms"] = df['baseline'].tolist()

        # 1. Check for significantly negative rewards (more than 1 std below mean)
        # Note: Small negative rewards are acceptable in medical RL due to death penalties
        reward_mean = df['reward'].mean()
        reward_std = df['reward'].std()
        significant_negative_threshold = reward_mean - \
            2 * reward_std  # 2-sigma below mean

        logger.debug("Reward statistics - Mean: %.3f, Std: %.3f, Threshold: %.3f",
                     reward_mean, reward_std, significant_negative_threshold)

        significant_negative_rewards = df[df['reward']
                                          < significant_negative_threshold]
        if not significant_negative_rewards.empty:
            for _, row in significant_negative_rewards.iterrows():
                logger.debug("Algorithm %s has significantly negative reward: %.3f (threshold: %.3f)",
                             row['baseline'], row['reward'], significant_negative_threshold)
                analysis_results["anomalies"].append({
                    "type": "negative_reward",
                    "algorithm": row['baseline'],
                    "value": row['reward'],
                    "severity": "high",
                    "description": f"{row['baseline']} has significantly negative reward ({row['reward']:.3f}, threshold: {significant_negative_threshold:.3f})"
                })

        # 2. Check for extremely high/low values (outliers)
        reward_mean = df['reward'].mean()
        reward_std = df['reward'].std()
        reward_threshold = 3 * reward_std  # 3-sigma rule

        outliers = df[abs(df['reward'] - reward_mean) > reward_threshold]
        if not outliers.empty:
            for _, row in outliers.iterrows():
                z_score = abs(row['reward'] - reward_mean) / reward_std
                analysis_results["anomalies"].append({
                    "type": "reward_outlier",
                    "algorithm": row['baseline'],
                    "value": row['reward'],
                    "z_score": z_score,
                    "severity": "medium" if z_score < 5 else "high",
                    "description": f"{row['baseline']} reward ({row['reward']:.3f}) is {z_score:.1f} standard deviations from mean"
                })

        # 3. Check for missing or NaN values
        missing_data = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_data[col] = missing_count

        if missing_data:
            analysis_results["warnings"].append({
                "type": "missing_data",
                "columns": missing_data,
                "description": f"Missing data detected in columns: {list(missing_data.keys())}"
            })

        # 4. Check for confidence interval issues
        if 'reward_ci' in df.columns:
            for _, row in df.iterrows():
                try:
                    if isinstance(row['reward_ci'], str):
                        import json
                        ci = json.loads(row['reward_ci'])
                    else:
                        ci = row['reward_ci']

                    if isinstance(ci, (list, tuple)) and len(ci) == 2:
                        ci_width = ci[1] - ci[0]
                        reward_val = row['reward']

                        # Check if reward is outside its own CI (suspicious)
                        if not (ci[0] <= reward_val <= ci[1]):
                            analysis_results["anomalies"].append({
                                "type": "reward_outside_ci",
                                "algorithm": row['baseline'],
                                "reward": reward_val,
                                "ci": ci,
                                "severity": "high",
                                "description": f"{row['baseline']} reward ({reward_val:.3f}) is outside its CI [{ci[0]:.3f}, {ci[1]:.3f}]"
                            })

                        # Check for unusually wide CIs (might indicate instability)
                        relative_width = ci_width / \
                            abs(reward_val) if reward_val != 0 else float('inf')
                        if relative_width > 0.5:  # CI width > 50% of point estimate
                            analysis_results["warnings"].append({
                                "type": "wide_confidence_interval",
                                "algorithm": row['baseline'],
                                "ci_width": ci_width,
                                "relative_width": relative_width,
                                "description": f"{row['baseline']} has wide CI (width={ci_width:.3f}, {relative_width*100:.1f}% of estimate)"
                            })

                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    analysis_results["warnings"].append({
                        "type": "malformed_ci",
                        "algorithm": row['baseline'],
                        "description": f"{row['baseline']} has malformed confidence interval"
                    })

        # 5. Performance ranking and recommendations
        df_sorted = df.sort_values('reward', ascending=False)
        analysis_results["statistics"]["best_performer"] = {
            "algorithm": df_sorted.iloc[0]['baseline'],
            "reward": df_sorted.iloc[0]['reward']
        }
        analysis_results["statistics"]["worst_performer"] = {
            "algorithm": df_sorted.iloc[-1]['baseline'],
            "reward": df_sorted.iloc[-1]['reward']
        }

        # 6. Runtime analysis
        if 'runtime_s' in df.columns:
            runtime_stats = {
                "mean_runtime": df['runtime_s'].mean(),
                "min_runtime": df['runtime_s'].min(),
                "max_runtime": df['runtime_s'].max(),
                "slowest_algorithm": df.loc[df['runtime_s'].idxmax(), 'baseline'],
                "fastest_algorithm": df.loc[df['runtime_s'].idxmin(), 'baseline']
            }
            analysis_results["statistics"]["runtime"] = runtime_stats

            # Flag extremely slow algorithms
            mean_runtime = df['runtime_s'].mean()
            slow_algorithms = df[df['runtime_s'] > 2 * mean_runtime]
            for _, row in slow_algorithms.iterrows():
                analysis_results["warnings"].append({
                    "type": "slow_training",
                    "algorithm": row['baseline'],
                    "runtime": row['runtime_s'],
                    "description": f"{row['baseline']} took {row['runtime_s']:.1f}s (>2x average runtime)"
                })

        # Generate recommendations
        if analysis_results["anomalies"]:
            high_severity = [
                a for a in analysis_results["anomalies"] if a.get("severity") == "high"]
            if high_severity:
                analysis_results["recommendations"].append(
                    "🚨 High-severity anomalies detected. Consider re-running affected algorithms."
                )

            negative_reward_algos = [
                a["algorithm"] for a in analysis_results["anomalies"] if a["type"] == "negative_reward"]
            if negative_reward_algos:
                analysis_results["recommendations"].append(
                    f"🔍 Investigate negative rewards in: {', '.join(negative_reward_algos)}. Check reward shaping and environment setup."
                )

        # Check for PoG-specific issues
        pog_algorithms = [alg for alg in df['baseline']
                          if 'pog' in alg.lower()]
        if pog_algorithms:
            pog_df = df[df['baseline'].str.lower().str.contains('pog')]
            pog_reward_variance = pog_df['reward'].var()
            baseline_df = df[~df['baseline'].str.lower().str.contains('pog')]
            baseline_reward_variance = baseline_df['reward'].var()

            if pog_reward_variance > 2 * baseline_reward_variance:
                analysis_results["warnings"].append({
                    "type": "high_pog_variance",
                    "pog_variance": pog_reward_variance,
                    "baseline_variance": baseline_reward_variance,
                    "description": "PoG algorithms show high reward variance compared to baselines"
                })
                analysis_results["recommendations"].append(
                    "📊 PoG algorithms show high variance. Consider adjusting hyperparameters or increasing training epochs."
                )

        logger.info("✅ Experiment analysis complete")
        logger.info(
            f"📊 Found {len(analysis_results['anomalies'])} anomalies and {len(analysis_results['warnings'])} warnings")

    except Exception as e:
        logger.error("❌ Failed to analyze experiment results: %s", e)
        analysis_results["warnings"].append(f"Analysis failed: {e}")

    return analysis_results


def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print a formatted analysis report."""
    print("\n" + "="*80)
    print("🔍 EXPERIMENT RESULTS ANALYSIS REPORT")
    print("="*80)

    # Statistics
    if "statistics" in analysis:
        stats = analysis["statistics"]
        print(f"\n📊 Basic Statistics:")
        print(f"   • Total algorithms: {stats.get('n_algorithms', 'N/A')}")
        if "best_performer" in stats:
            best = stats["best_performer"]
            print(
                f"   • Best performer: {best['algorithm']} (reward: {best['reward']:.3f})")
        if "worst_performer" in stats:
            worst = stats["worst_performer"]
            print(
                f"   • Worst performer: {worst['algorithm']} (reward: {worst['reward']:.3f})")

        if "runtime" in stats:
            runtime = stats["runtime"]
            print(f"   • Runtime - Mean: {runtime['mean_runtime']:.1f}s, "
                  f"Fastest: {runtime['fastest_algorithm']} ({runtime['min_runtime']:.1f}s), "
                  f"Slowest: {runtime['slowest_algorithm']} ({runtime['max_runtime']:.1f}s)")

    # Anomalies
    if analysis["anomalies"]:
        print(f"\n🚨 Detected Anomalies ({len(analysis['anomalies'])}):")
        for i, anomaly in enumerate(analysis["anomalies"], 1):
            severity_emoji = "🔴" if anomaly.get("severity") == "high" else "🟡"
            print(f"   {i}. {severity_emoji} {anomaly['description']}")

    # Warnings
    if analysis["warnings"]:
        print(f"\n⚠️  Warnings ({len(analysis['warnings'])}):")
        for i, warning in enumerate(analysis["warnings"], 1):
            if isinstance(warning, dict):
                print(f"   {i}. {warning['description']}")
            else:
                print(f"   {i}. {warning}")

    # Recommendations
    if analysis["recommendations"]:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"   {i}. {rec}")

    if not analysis["anomalies"] and not analysis["warnings"]:
        print("\n✅ No anomalies or warnings detected. Results look good!")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
