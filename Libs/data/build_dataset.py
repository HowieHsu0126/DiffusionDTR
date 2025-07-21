#!/usr/bin/env python3
"""build_dataset.py

End-to-end dataset construction pipeline for the PoG-BVE framework.

This script orchestrates the complete data preparation workflow for multiple tasks:

1. Read raw CSVs from Input/raw/ directory exported from PostgreSQL
2. Build ICU state space table state.csv using build_state_space
3. Build RL trajectory table using build_trajectory 
4. Apply preprocessing and tensorization using Preprocessor
5. Build patient similarity graph and save as PyG Data object

Supports multiple tasks:
- vent: Mechanical ventilation strategy optimization
- rrt: Renal replacement therapy strategy optimization  
- iv: IV fluids and vasopressor strategy optimization

Examples:
    # Build single task
    python -m Libs.scripts.build_dataset --task vent --k 5 --t-max 48
    
    # Build multiple tasks
    python -m Libs.scripts.build_dataset --tasks vent rrt iv
    
    # Build all tasks with custom parameters
    python -m Libs.scripts.build_dataset --all-tasks --k 3 --t-max 30
    
    # Build with parallel execution (experimental)
    python -m Libs.scripts.build_dataset --tasks vent rrt --parallel
"""

from __future__ import annotations

import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Final, Dict, Any, List

import Libs.data.build_state_space as bss
from Libs.data.build_trajectory import ChunkedTrajectoryBuilder
from Libs.data.preprocessing import Preprocessor
from Libs.data.build_graph import build_trajectory_graph
from Libs.utils.log_utils import get_logger

__all__: Final = [
    "main",
]

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Helper functions – each encapsulates **one** stage so that高级用户可单独调用
# -----------------------------------------------------------------------------

def step_state_space(raw_dir: Path, processed_dir: Path, raw_files: dict, state_filename: str, id_cols: list = None) -> Path:
    """Construct the ``state.csv`` file from raw vitals/lab results/demographics.

    Args:
        raw_dir: Directory containing raw CSVs.
        processed_dir: Directory to write processed artefacts.
        raw_files: Dictionary containing paths to raw CSV files.
        state_filename: Name of the state CSV file.
        id_cols: List of ID columns to preserve during imputation.

    Returns:
        Path to the generated ``state.csv``.
    """
    logger.info("[STEP 1] Building state space …")

    vitals_path = raw_dir / raw_files["vitals"]
    labres_path = raw_dir / raw_files["labres"]
    demo_path = raw_dir / raw_files["demographics"]
    sofa_path = raw_dir / raw_files["sofa"]
    out_state = processed_dir / state_filename

    vitals, labres, demo, sofa = bss.load_data(
        str(vitals_path), str(labres_path), str(demo_path), str(sofa_path)
    )
    merged = bss.merge_data(vitals, labres, demo, sofa)

    # Use ID columns from configuration or default values
    if id_cols is None:
        id_cols = ["subject_id", "hadm_id", "stay_id", "hours_from_onset"]
    
    merged = bss.impute_missing(merged, id_cols)
    merged = bss.select_state_columns(merged)
    bss.save_df(merged, str(out_state))
    logger.info("State space saved to %s", out_state)
    return out_state


def step_trajectory(
    state_csv: Path,
    raw_dir: Path,
    processed_dir: Path,
    action_file: str,
    mortality_file: str,
    trajectory_filename: str,
    chunksize: int,
    task_name: str,
) -> Path:
    """Construct the RL trajectory table for a specific task.

    Args:
        state_csv: Path to ``state.csv``.
        raw_dir: Directory containing raw CSVs.
        processed_dir: Directory to write processed artefacts.
        action_file: Task-specific action file path (relative to raw_dir).
        mortality_file: Mortality file path (relative to raw_dir).
        trajectory_filename: Name of the trajectory CSV file.
        chunksize: Row chunk size for streaming processing.
        task_name: Name of the task for logging.

    Returns:
        Path to the generated trajectory CSV.
    """
    logger.info("[STEP 2] Building trajectory table for task: %s …", task_name)

    action_path = raw_dir / action_file
    mortality_path = raw_dir / mortality_file
    out_traj = processed_dir / trajectory_filename

    builder = ChunkedTrajectoryBuilder(
        state_path=str(state_csv),
        action_path=str(action_path),
        mortality_path=str(mortality_path),
        output_path=str(out_traj),
        chunksize=chunksize,
    )
    builder.run(subject_id_path=None)
    
    # Save terminal-only rewards table (for OPE use)
    terminal_rewards_file = trajectory_filename.replace("trajectory_", "terminal_rewards_")
    builder.save_terminal_rewards_only(str(processed_dir / terminal_rewards_file))

    logger.info("Trajectory table saved to %s", out_traj)
    return out_traj


def step_preprocess(traj_csv: Path, processed_dir: Path, resample_hours: int, reward_params: dict) -> Path:
    """Run trajectory preprocessing and tensorisation.

    Args:
        traj_csv: Path to trajectory CSV.
        processed_dir: Directory for saving tensors (a sub-folder will be used).
        resample_hours: Resample window in hours.
        reward_params: Parameters for reward calculation.

    Returns:
        Path to the tensor file produced by the :class:`Preprocessor`.
    """
    logger.info("[STEP 3] Running preprocessing …")
    preproc_dir = processed_dir / "preproc"
    preproc_dir.mkdir(parents=True, exist_ok=True)

    pre = Preprocessor(out_dir=str(preproc_dir), resample_hours=resample_hours, concat_mask=True, reward_params=reward_params)
    tensors = pre.run(str(traj_csv), save=True)
    tensor_path = preproc_dir / "preproc_tensors.pt"
    logger.info("Preprocessed tensors saved to %s", tensor_path)
    return tensor_path


def step_graph(
    traj_csv: Path,
    raw_dir: Path,
    processed_dir: Path,
    comorbidity_file: str,
    mortality_file: str,
    action_cols: list,
    k: int,
    t_max: int,
    graph_filename: str,
    task_name: str,
) -> Path:
    """Build the PyG patient similarity graph for a specific task.

    Args:
        traj_csv: Path to trajectory CSV.
        raw_dir: Directory containing raw CSVs.
        processed_dir: Directory to write graph pt file.
        comorbidity_file: Comorbidity file path (relative to raw_dir).
        mortality_file: Mortality file path (relative to raw_dir).
        action_cols: List of action columns to include in the graph.
        k: Number of neighbours per node in k-NN graph.
        t_max: Maximum trajectory length (# of time-steps).
        graph_filename: Name of the graph file.
        task_name: Name of the task for logging.

    Returns:
        Path to the generated graph file.
    """
    logger.info("[STEP 4] Constructing patient graph for task: %s …", task_name)

    comorb_csv = raw_dir / comorbidity_file
    mort_csv = raw_dir / mortality_file
    graph_path = processed_dir / graph_filename

    build_trajectory_graph(
        traj_csv=str(traj_csv),
        comorbidity_csv=str(comorb_csv),
        action_cols=action_cols,
        k=k,
        T_max=t_max,
        save_path=str(graph_path),
        mortality_csv=str(mort_csv),
    )
    logger.info("Graph saved to %s", graph_path)
    return graph_path


def step_split_ids(traj_csv: Path, processed_dir: Path, split_cfg: dict, seed: int = 42):
    """Randomly split patient IDs into train/val/test.

    Args:
        traj_csv: Path to trajectory CSV with subject_id column.
        processed_dir: Directory to save split files.
        split_cfg: Dict with keys train/val/test and float ratios summing to 1.
        seed: Random seed for reproducibility.
    """
    import numpy as np
    import pandas as pd

    logger.info("[STEP 5] Splitting patient IDs ...")
    
    # Validate split configuration
    required_keys = ["train", "val", "test"]
    missing_keys = [k for k in required_keys if k not in split_cfg]
    if missing_keys:
        raise ValueError(f"Missing split configuration keys: {missing_keys}")
    
    # Check that ratios sum to 1.0 (within tolerance)
    total_ratio = sum(split_cfg.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    logger.info("Split configuration: %s", split_cfg)
    
    df = pd.read_csv(traj_csv, usecols=["subject_id"])
    ids = df["subject_id"].astype(str).unique()
    
    np.random.seed(seed)
    np.random.shuffle(ids)
    
    n_total = len(ids)
    n_train = int(n_total * split_cfg["train"])
    n_val = int(n_total * split_cfg["val"])
    # Test gets the remainder to ensure all patients are included
    
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    
    # Verify no overlap
    all_splits = [set(train_ids), set(val_ids), set(test_ids)]
    for i, split1 in enumerate(all_splits):
        for j, split2 in enumerate(all_splits):
            if i != j and split1 & split2:
                raise ValueError(f"Overlapping patient IDs found between splits {i} and {j}")
    
    # Save splits - note that string arrays require allow_pickle=True when loading
    splits_info = {
        "train": (train_ids, "ids_train.npy"),
        "val": (val_ids, "ids_val.npy"),
        "test": (test_ids, "ids_test.npy")
    }
    
    for split_name, (ids_array, filename) in splits_info.items():
        # Save as object array since IDs are strings - requires allow_pickle=True when loading
        np.save(processed_dir / filename, ids_array, allow_pickle=True)
        logger.info("Saved %s split: %d patients to %s", split_name, len(ids_array), filename)
    
    # Final validation
    total_split = len(train_ids) + len(val_ids) + len(test_ids)
    if total_split != n_total:
        raise ValueError(f"Split size mismatch: {total_split} != {n_total}")
    
    logger.info("Split summary: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)", 
                len(train_ids), 100*len(train_ids)/n_total,
                len(val_ids), 100*len(val_ids)/n_total,
                len(test_ids), 100*len(test_ids)/n_total)


def validate_dataset_build(task_name: str, task_cfg: Dict[str, Any], processed_dir: Path) -> bool:
    """Validate that the dataset build matches the YAML configuration.
    
    Args:
        task_name: Name of the task.
        task_cfg: Task configuration dictionary.
        processed_dir: Directory where processed files are stored.
        
    Returns:
        True if validation passes, False otherwise.
    """
    logger.info("Validating dataset build for task: %s", task_name)
    
    validation_passed = True
    
    # Check that actually created files exist
    # Note: comorbidity_filtered is not created by the current pipeline, so we skip it
    expected_files = task_cfg.get("processed_files", {})
    files_to_check = {k: v for k, v in expected_files.items() if k != "comorbidity_filtered"}
    
    for file_type, filename in files_to_check.items():
        file_path = processed_dir / filename
        if not file_path.exists():
            logger.error("Missing expected file: %s (%s)", filename, file_type)
            validation_passed = False
        else:
            logger.info("✓ Found %s: %s", file_type, filename)
    
    # Check split files with proper error handling for object arrays
    split_files = ["ids_train.npy", "ids_val.npy", "ids_test.npy"]
    for split_file in split_files:
        file_path = processed_dir / split_file
        if not file_path.exists():
            logger.error("Missing split file: %s", split_file)
            validation_passed = False
        else:
            # Check that split file is not empty with proper error handling
            try:
                import numpy as np
                # Handle object arrays by allowing pickle
                ids = np.load(file_path, allow_pickle=True)
                if len(ids) == 0:
                    logger.error("Empty split file: %s", split_file)
                    validation_passed = False
                else:
                    logger.info("✓ Found %s: %d patients", split_file, len(ids))
            except Exception as e:
                logger.error("Error loading split file %s: %s", split_file, e)
                validation_passed = False
    
    # Check trajectory file structure
    traj_file = expected_files.get("trajectory")
    if traj_file:
        traj_path = processed_dir / traj_file
        if traj_path.exists():
            import pandas as pd
            try:
                df = pd.read_csv(traj_path, nrows=10)  # Just check first few rows
                
                # Check that action columns exist
                action_cols = task_cfg.get("action_cols", [])
                missing_actions = [col for col in action_cols if col not in df.columns]
                if missing_actions:
                    logger.error("Missing action columns in trajectory: %s", missing_actions)
                    validation_passed = False
                else:
                    logger.info("✓ All action columns found: %s", action_cols)
                
                # Check required columns
                required_cols = ["subject_id", "hours_from_onset", "reward"]
                missing_required = [col for col in required_cols if col not in df.columns]
                if missing_required:
                    logger.error("Missing required columns in trajectory: %s", missing_required)
                    validation_passed = False
                else:
                    logger.info("✓ All required columns found")
                    
            except Exception as e:
                logger.error("Error reading trajectory file: %s", e)
                validation_passed = False
    
    # Check graph file structure
    graph_file = expected_files.get("pyg_graph")
    if graph_file:
        graph_path = processed_dir / graph_file
        if graph_path.exists():
            try:
                import torch
                data = torch.load(graph_path)
                
                # Check required attributes
                required_attrs = ["x", "edge_index", "patient_ids", "actions"]
                missing_attrs = [attr for attr in required_attrs if not hasattr(data, attr)]
                if missing_attrs:
                    logger.error("Missing graph attributes: %s", missing_attrs)
                    validation_passed = False
                else:
                    logger.info("✓ Graph structure validated")
                    logger.info("  Nodes: %d, Edges: %d", data.x.size(0), data.edge_index.size(1))
                    
            except Exception as e:
                logger.error("Error reading graph file: %s", e)
                validation_passed = False
    
    if validation_passed:
        logger.info("✓ Dataset validation passed for task: %s", task_name)
    else:
        logger.error("✗ Dataset validation failed for task: %s", task_name)
    
    return validation_passed


def load_task_config(cfg: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """Load task-specific configuration and merge with global config.
    
    Args:
        cfg: Global configuration dictionary.
        task_name: Name of the task (vent, rrt, iv).
        
    Returns:
        Task-specific configuration dictionary.
    """
    if task_name not in cfg.get("tasks", {}):
        raise ValueError(f"Task '{task_name}' not found in configuration. Available tasks: {list(cfg.get('tasks', {}).keys())}")
    
    task_cfg = cfg["tasks"][task_name].copy()
    
    # Merge with global config (task-specific overrides global)
    merged_cfg = cfg.copy()
    merged_cfg.update(task_cfg)
    
    # Set task-specific data_root
    base_data_root = Path(cfg["data_root"])
    merged_cfg["data_root"] = str(base_data_root / task_cfg["output_dir"])
    
    # Set backward compatibility fields
    merged_cfg["data_filename"] = task_cfg["processed_files"]["pyg_graph"]
    merged_cfg["processed_files"] = task_cfg["processed_files"]
    
    logger.info("Loaded configuration for task: %s", task_name)
    logger.info("  Description: %s", task_cfg.get("description", "N/A"))
    logger.info("  Action columns: %s", task_cfg["action_cols"])
    logger.info("  Output directory: %s", merged_cfg["data_root"])
    
    return merged_cfg


def build_single_task(task_name: str, args: argparse.Namespace, cfg: Dict[str, Any]) -> bool:
    """Build dataset for a single task.
    
    Args:
        task_name: Name of the task to build.
        args: Command line arguments.
        cfg: Global configuration dictionary.
        
    Returns:
        True if successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Building dataset for task: %s", task_name)
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load task-specific configuration
        task_cfg = load_task_config(cfg, task_name)
        
        # Extract paths and parameters from YAML configuration
        raw_dir = Path(task_cfg["raw_dir"])
        processed_dir = Path(task_cfg["data_root"])
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_files: dict = task_cfg.get("raw_files", {})
        processed_files: dict = task_cfg.get("processed_files", {})

        # Processing parameters - use YAML values or defaults
        chunk_size: int = task_cfg.get("chunk_size", 50_000)
        resample_hours: int = task_cfg.get("resample_hours", 4)
        action_cols: list = task_cfg.get("action_cols", [])
        graph_k: int = args.k or task_cfg.get("graph_k", 3)
        graph_T_max: int = args.t_max or task_cfg.get("graph_T_max", 30)
        reward_params: dict = task_cfg.get("reward_shaping", {})
        id_cols: list = task_cfg.get("id_cols", ["subject_id", "hadm_id", "stay_id", "hours_from_onset"])
        
        # Validate required parameters
        if not action_cols:
            raise ValueError(f"No action columns specified for task {task_name}")
        
        if not raw_files:
            raise ValueError(f"No raw files configuration found for task {task_name}")

        # Split ratios - use YAML values with validation
        split_cfg = {
            "train": task_cfg.get("train_split", 0.7),
            "val": task_cfg.get("val_split", 0.2),
            "test": task_cfg.get("test_split", 0.1),
        }
        
        # Validate split ratios
        total_split = sum(split_cfg.values())
        if abs(total_split - 1.0) > 1e-6:
            logger.warning("Split ratios don't sum to 1.0 (%.3f), normalizing...", total_split)
            split_cfg = {k: v/total_split for k, v in split_cfg.items()}

        # Log configuration summary
        logger.info("Task configuration summary:")
        logger.info("  Raw directory: %s", raw_dir)
        logger.info("  Output directory: %s", processed_dir)
        logger.info("  Action columns: %s", action_cols)
        logger.info("  Graph parameters: k=%d, T_max=%d", graph_k, graph_T_max)
        logger.info("  Processing: chunk_size=%d, resample_hours=%d", chunk_size, resample_hours)
        logger.info("  Split ratios: %s", split_cfg)
        logger.info("  Reward parameters: %s", reward_params)

        # ------------------------------------------------------------------
        # STEP 1 – State space
        # ------------------------------------------------------------------
        state_filename = processed_files.get("state", "state.csv")
        state_csv = processed_dir / state_filename
        if args.skip_state and state_csv.exists():
            logger.info("[SKIP] Using existing state file %s", state_csv)
        else:
            state_csv = step_state_space(raw_dir, processed_dir, raw_files, state_filename, id_cols=id_cols)

        # ------------------------------------------------------------------
        # STEP 2 – Trajectory
        # ------------------------------------------------------------------
        traj_filename = processed_files.get("trajectory", f"trajectory_{task_name}.csv")
        traj_csv = processed_dir / traj_filename
        if args.skip_trajectory and traj_csv.exists():
            logger.info("[SKIP] Using existing trajectory file %s", traj_csv)
        else:
            traj_csv = step_trajectory(
                state_csv=state_csv,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                action_file=task_cfg["action_file"],
                mortality_file=raw_files["mortality"],
                trajectory_filename=traj_filename,
                chunksize=chunk_size,
                task_name=task_name,
            )

        # ------------------------------------------------------------------
        # STEP 3 – Preprocessing (optional)
        # ------------------------------------------------------------------
        if not args.skip_preprocess:
            step_preprocess(traj_csv, processed_dir, resample_hours, reward_params)

        # ------------------------------------------------------------------
        # STEP 4 – Graph construction
        # ------------------------------------------------------------------
        graph_filename = processed_files.get("pyg_graph", f"patient_traj_graph_{task_name}.pt")
        graph_path = processed_dir / graph_filename
        if args.skip_graph and graph_path.exists():
            logger.info("[SKIP] Using existing graph file %s", graph_path)
        else:
            graph_path = step_graph(
                traj_csv=traj_csv,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                comorbidity_file=raw_files["comorbidity"],
                mortality_file=raw_files["mortality"],
                action_cols=action_cols,
                k=graph_k,
                t_max=graph_T_max,
                graph_filename=graph_filename,
                task_name=task_name,
            )

        # ------------------------------------------------------------------
        # STEP 5 – Split patient IDs
        # ------------------------------------------------------------------
        step_split_ids(traj_csv, processed_dir, split_cfg, args.seed)

        # ------------------------------------------------------------------
        # STEP 6 – Validate dataset build
        # ------------------------------------------------------------------
        logger.info("[STEP 6] Validating dataset build ...")
        validation_passed = validate_dataset_build(task_name, task_cfg, processed_dir)
        
        if not validation_passed:
            logger.error("Dataset validation failed for task: %s", task_name)
            if args.stop_on_error:
                raise ValueError(f"Dataset validation failed for task: {task_name}")
            return False

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Dataset construction completed for task: %s", task_name)
        logger.info("=" * 60)
        logger.info("Output directory: %s", processed_dir)
        logger.info("Files generated:")
        logger.info("  • State space: %s", state_csv.name)
        logger.info("  • Trajectory: %s", traj_csv.name)
        logger.info("  • Graph: %s", graph_path.name)
        logger.info("  • ID splits: ids_train.npy, ids_val.npy, ids_test.npy")
        if not args.skip_preprocess:
            logger.info("  • Preprocessed tensors: preproc/preproc_tensors.pt")
        logger.info("Configuration used:")
        logger.info("  • Action columns: %s", action_cols)
        logger.info("  • Graph parameters: k=%d, T_max=%d", graph_k, graph_T_max)
        logger.info("  • Processing parameters: chunk_size=%d, resample_hours=%d", chunk_size, resample_hours)
        logger.info("  • Split ratios: train=%.1f%%, val=%.1f%%, test=%.1f%%", 
                   100*split_cfg["train"], 100*split_cfg["val"], 100*split_cfg["test"])
        logger.info("  • Reward shaping: %s", reward_params)
        logger.info("Build time: %.1fs", elapsed_time)
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error("✗ Failed to build dataset for task: %s (%.1fs)", task_name, elapsed_time)
        logger.error("Error: %s", e)
        
        if args.stop_on_error:
            raise
        
        return False


def build_multiple_tasks(task_names: List[str], args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Build datasets for multiple tasks.
    
    Args:
        task_names: List of task names to build.
        args: Command line arguments.
        cfg: Global configuration dictionary.
        
    Returns:
        Dictionary mapping task names to success status.
    """
    logger.info("=" * 80)
    logger.info("Multi-Task Dataset Construction Pipeline")
    logger.info("=" * 80)
    logger.info("Tasks to build: %s", task_names)
    logger.info("Configuration: %s", args.config)
    logger.info("Random seed: %s", args.seed)
    
    if args.parallel:
        logger.warning("Parallel execution is experimental and may cause issues")
    
    start_time = time.time()
    results = {}
    
    if args.parallel:
        # Parallel execution (experimental)
        import concurrent.futures
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(build_single_task, task, args, cfg): task 
                for task in task_names
            }
            
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    success = future.result()
                    results[task] = success
                except Exception as e:
                    logger.error("Task %s failed with exception: %s", task, e)
                    results[task] = False
    else:
        # Sequential execution
        for task in task_names:
            success = build_single_task(task, args, cfg)
            results[task] = success
    
    # Summary
    total_time = time.time() - start_time
    successful_tasks = [task for task, success in results.items() if success]
    failed_tasks = [task for task, success in results.items() if not success]
    
    logger.info("=" * 80)
    logger.info("Build Summary")
    logger.info("=" * 80)
    logger.info("Total time: %.1fs", total_time)
    logger.info("Successful tasks: %s", successful_tasks)
    
    if failed_tasks:
        logger.error("Failed tasks: %s", failed_tasks)
        logger.error("Please check the logs above for error details")
        if args.stop_on_error:
            sys.exit(1)
    else:
        logger.info("✓ All datasets built successfully!")
        
        # Print dataset locations
        logger.info("\nDataset locations:")
        for task in successful_tasks:
            task_cfg = load_task_config(cfg, task)
            task_dir = Path(task_cfg["data_root"])
            logger.info("  • %s: %s", task, task_dir)
            
            # List key files for each task
            key_files = task_cfg.get("processed_files", {})
            if key_files:
                logger.info("    Files: %s", ", ".join(key_files.values()))
        
        logger.info("\nNext steps:")
        logger.info("1. Run experiments: python -m Libs.exp.run_all --task <task>")
        logger.info("2. Available tasks: %s", ", ".join(successful_tasks))
    
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build dataset for PoG-BVE framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build single task
    python -m Libs.scripts.build_dataset --task vent --k 5 --t-max 48
    
    # Build multiple tasks
    python -m Libs.scripts.build_dataset --tasks vent rrt iv
    
    # Build all tasks
    python -m Libs.scripts.build_dataset --all-tasks
    
    # Build with custom parameters and skip existing files
    python -m Libs.scripts.build_dataset --tasks vent rrt --skip-state --skip-trajectory
    
    # Build with parallel execution (experimental)
    python -m Libs.scripts.build_dataset --tasks vent rrt --parallel
        """
    )
    
    # Task selection (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task", "-t",
        choices=["vent", "rrt", "iv"],
        help="Single task type to build dataset for"
    )
    task_group.add_argument(
        "--tasks", nargs="+",
        choices=["vent", "rrt", "iv"],
        help="Multiple tasks to build datasets for"
    )
    task_group.add_argument(
        "--all-tasks", action="store_true",
        help="Build datasets for all available tasks"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c", 
        default="Libs/configs/dataset.yaml",
        help="Path to dataset configuration YAML file"
    )
    
    # Build options
    parser.add_argument(
        "--skip-state",
        action="store_true",
        help="Skip state space construction if file exists"
    )
    parser.add_argument(
        "--skip-trajectory", 
        action="store_true",
        help="Skip trajectory construction if file exists"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true", 
        help="Skip preprocessing step"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph construction if file exists"
    )
    
    # Graph parameters
    parser.add_argument(
        "--k", type=int,
        help="Number of neighbors for k-NN graph (overrides config)"
    )
    parser.add_argument(
        "--t-max", type=int,
        help="Maximum trajectory length (overrides config)"
    )
    
    # Execution options
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop execution if any task fails"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Build datasets in parallel (experimental)"
    )
    
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """Main orchestration logic for single or multi-task dataset construction."""

    args = parse_args(argv)

    # ---------------- Load YAML configuration ----------------
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # ---------------- Determine tasks to build ----------------
    if args.all_tasks:
        task_names = list(cfg.get("tasks", {}).keys())
    elif args.tasks:
        task_names = args.tasks
    elif args.task:
        task_names = [args.task]
    else:
        # Default to single task from config or 'vent'
        task_names = [cfg.get("task", "vent")]
    
    if not task_names:
        logger.error("No tasks specified. Use --task, --tasks, or --all-tasks")
        sys.exit(1)
    
    # Validate task names
    available_tasks = set(cfg.get("tasks", {}).keys())
    invalid_tasks = set(task_names) - available_tasks
    if invalid_tasks:
        logger.error("Invalid tasks: %s. Available tasks: %s", 
                    list(invalid_tasks), list(available_tasks))
        sys.exit(1)

    # ---------------- Build datasets ----------------
    if len(task_names) == 1:
        # Single task build
        success = build_single_task(task_names[0], args, cfg)
        if not success:
            sys.exit(1)
    else:
        # Multi-task build
        results = build_multiple_tasks(task_names, args, cfg)
        failed_tasks = [task for task, success in results.items() if not success]
        if failed_tasks:
            sys.exit(1)


if __name__ == "__main__":
    main() 