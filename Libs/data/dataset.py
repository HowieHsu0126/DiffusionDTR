"""
Medical trajectory dataset implementation for PoG-BVE framework.

This module provides PyTorch-Geometric compatible dataset classes for loading
and processing medical trajectory data with graph structures. Supports multiple
tasks including mechanical ventilation, RRT, and IV fluids/vasopressor management.
"""

import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import Optional, Union, Callable, List, Dict, Any
import numpy as np
import copy
import yaml
from pathlib import Path


class PatientTrajectoryDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for medical patient trajectories with graph structure.

    This dataset loads preprocessed patient trajectory data stored as PyTorch Geometric
    Data objects, providing standardized access to clinical features, actions, outcomes,
    and patient similarity graphs for reinforcement learning applications.

    Supports multiple clinical tasks:
    - vent: Mechanical ventilation strategy optimization
    - rrt: Renal replacement therapy strategy optimization  
    - iv: IV fluids and vasopressor strategy optimization

    Args:
        root: Root directory where the dataset is stored.
        task: Task type ('vent', 'rrt', 'iv') - determines subdirectory and filename.
        filename: Name of the processed .pt file (auto-determined if None).
        cohort: Clinical cohort identifier (e.g., 'ICU_AKI', 'Sepsis').
        config_path: Path to dataset configuration file.
        transform: Optional transform to be applied on each data object.
        pre_transform: Optional pre-transform to be applied on raw data.

    Attributes:
        data: The loaded PyTorch Geometric Data object containing all trajectories.
        cohort: Clinical cohort identifier.
        task: Task type identifier.

    Example:
        >>> # Load ventilation task dataset
        >>> dataset = PatientTrajectoryDataset(
        ...     root="Input/processed/",
        ...     task="vent",
        ...     cohort="ICU_AKI"
        ... )
        >>> data = dataset[0]  # Get the full graph data
        >>> stats = dataset.get_clinical_stats()
        >>> print(f"Task: {dataset.task}, Cohort size: {stats['n_patients']}")
        
        >>> # Load RRT task dataset  
        >>> dataset_rrt = PatientTrajectoryDataset(
        ...     root="Input/processed/",
        ...     task="rrt",
        ...     cohort="ICU_AKI"
        ... )
    """

    def __init__(
        self,
        root: str = "Input/processed/",
        task: str = "vent",
        filename: Optional[str] = None,
        cohort: str = "ICU_AKI",
        config_path: str = "Libs/configs/dataset.yaml",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        self.task = task
        self.cohort = cohort
        self.config_path = config_path
        
        # Load configuration to determine task-specific settings
        self.config = self._load_config()
        
        # Determine filename and directory structure
        if filename is None:
            task_config = self.config.get("tasks", {}).get(task, {})
            filename = task_config.get("processed_files", {}).get("pyg_graph", f"patient_traj_graph_{task}.pt")
        
        self.filename = filename
        self.original_filename = filename  # Store for reference
        
        # Set task-specific root directory
        task_root = Path(root) / task
        
        super().__init__(str(task_root), transform, pre_transform)

        # Load the processed data
        data_path = self._get_data_path()
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {data_path}\n"
                f"Please run the dataset construction pipeline first:\n"
                f"python -m Libs.scripts.build_dataset --task {task}"
            )

        self.data = torch.load(data_path)
        self.slices = None  # No slicing needed for single graph structure

        # Validate loaded data
        self._validate_data()

    def _load_config(self) -> Dict[str, Any]:
        """Load dataset configuration from YAML file."""
        config_path = Path(self.config_path)
        if not config_path.exists():
            return {}
        
        with config_path.open("r") as f:
            return yaml.safe_load(f)

    def _get_data_path(self) -> Path:
        """Get the path to the data file, checking multiple possible locations."""
        # Primary location: task-specific directory
        primary_path = Path(self.root) / self.filename
        if primary_path.exists():
            return primary_path
        
        # Fallback: processed_paths for backward compatibility
        if hasattr(self, 'processed_paths') and self.processed_paths:
            fallback_path = Path(self.processed_paths[0])
            if fallback_path.exists():
                return fallback_path
        
        # Last resort: check parent directory
        parent_path = Path(self.root).parent / self.filename
        if parent_path.exists():
            return parent_path
            
        return primary_path  # Return primary even if it doesn't exist for error reporting

    @property
    def raw_file_names(self) -> List[str]:
        """
        Names of raw files in the raw directory.

        Returns:
            Empty list as raw files are processed externally.
        """
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """
        Names of processed files in the processed directory.

        Returns:
            List containing the filename of the processed data.
        """
        return [self.filename]

    def download(self):
        """
        Download raw files if needed.

        Note:
            This dataset assumes data is already processed by external scripts.
            No download functionality is implemented.
        """
        pass

    def process(self):
        """
        Process raw data into PyTorch Geometric format.

        Note:
            Processing is handled by external data construction scripts.
            This method is left empty as data is pre-processed.
        """
        pass

    def len(self) -> int:
        """
        Get the number of data objects in the dataset.

        Returns:
            Always returns 1 as this dataset contains a single graph.
        """
        return 1

    def get(self, idx: int) -> Data:
        """Return a **defensive copy** of the graph at *idx* (always 0).

        External callers might perform in-place ops on the returned
        :class:`torch_geometric.data.Data`.  To avoid accidental mutation of
        the cached copy stored inside the dataset, we return
        :pyfunc:`copy.deepcopy` of the data.
        """
        assert idx == 0, f"Dataset contains only one graph, got index {idx}"
        return copy.deepcopy(self.data)

    def get_patient_ids(self) -> List[str]:
        """
        Get list of patient IDs in the dataset.

        Returns:
            List of patient ID strings.
        """
        return self.data.patient_ids

    def get_action_info(self) -> Dict[str, Any]:
        """
        Get information about the action space for this task.
        
        Returns:
            Dictionary containing action space information.
        """
        action_info = {
            'task': self.task,
            'action_cols': getattr(self.data, 'action_cols', []),
            'action_indices': getattr(self.data, 'action_indices', []),
            'n_actions': len(getattr(self.data, 'action_cols', [])),
        }
        
        # Add task-specific action descriptions
        task_config = self.config.get("tasks", {}).get(self.task, {})
        action_info['description'] = task_config.get('description', f'{self.task} task')
        
        if hasattr(self.data, 'actions'):
            action_info['action_dims'] = [
                int(self.data.actions[..., i].max().item()) + 1 
                for i in range(self.data.actions.shape[-1])
            ]
        
        return action_info

    def get_trajectory(self, patient_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get individual patient trajectory by patient index.

        Args:
            patient_idx: Index of the patient in the dataset.

        Returns:
            Dictionary containing trajectory data for the specified patient.

        Raises:
            IndexError: If patient_idx is out of bounds.
        """
        if patient_idx >= len(self.data.patient_ids):
            raise IndexError(
                f"Patient index {patient_idx} out of bounds for {len(self.data.patient_ids)} patients")

        # Resolve mask attribute name differences (mask vs masks)
        mask_attr = 'mask' if hasattr(self.data, 'mask') else (
            'masks' if hasattr(self.data, 'masks') else None
        )

        return {
            'patient_id': self.data.patient_ids[patient_idx],
            'states': self.data.x[patient_idx],
            'actions': self.data.actions[patient_idx] if hasattr(self.data, 'actions') else None,
            'mask': getattr(self.data, mask_attr)[patient_idx] if mask_attr else None,
            'length': self.data.lengths[patient_idx],
            'survival': self.data.survival[patient_idx] if hasattr(self.data, 'survival') else None,
        }

    def get_clinical_stats(self) -> Dict[str, Any]:
        """
        Get clinical statistics for the dataset.

        Returns:
            Dictionary containing dataset statistics including:
            - n_patients: Number of patients
            - avg_length: Average trajectory length
            - survival_rate: Overall survival rate
            - action_distribution: Distribution of actions
            - feature_stats: Feature mean and standard deviation
        """
        stats = {
            'n_patients': len(self.data.patient_ids),
            'avg_length': float(self.data.lengths.float().mean()),
            'max_length': int(self.data.lengths.max()),
            'min_length': int(self.data.lengths.min()),
            # survival=0 means alive
            'survival_rate': float(1 - self.data.survival.float().mean()),
            'cohort': self.cohort,
            'task': self.task
        }

        # Action distribution statistics
        if hasattr(self.data, 'actions'):
            valid_mask = self.data.mask.bool()
            # Calculate action distributions for each action dimension
            action_dims = self.data.actions.shape[-1]
            action_dist = {}
            for i in range(action_dims):
                actions_i = self.data.actions[..., i][valid_mask]
                unique_actions, counts = torch.unique(
                    actions_i, return_counts=True)
                action_dist[f'action_{i}'] = {
                    'unique_values': unique_actions.tolist(),
                    'counts': counts.tolist(),
                    'probabilities': (counts.float() / counts.sum()).tolist()
                }
            stats['action_distribution'] = action_dist

        # Feature statistics
        if hasattr(self.data, 'x'):
            valid_states = self.data.x[self.data.mask.bool()]
            stats['feature_stats'] = {
                'mean': valid_states.mean(dim=0).tolist(),
                'std': valid_states.std(dim=0).tolist(),
                'n_features': self.data.x.shape[-1]
            }

        return stats

    def subset(self, patient_indices: Union[List[int], np.ndarray, torch.Tensor]) -> 'PatientTrajectoryDataset':
        """
        Create a subset of the dataset with specified patients.

        Args:
            patient_indices: Indices of patients to include in the subset.

        Returns:
            New PatientTrajectoryDataset containing only the specified patients.
        """
        if isinstance(patient_indices, (list, np.ndarray)):
            patient_indices = torch.tensor(patient_indices, dtype=torch.long)

        # Create subset data
        subset_data = Data(
            x=self.data.x[patient_indices],
            actions=self.data.actions[patient_indices] if hasattr(
                self.data, 'actions') else None,
            mask=self.data.mask[patient_indices],
            lengths=self.data.lengths[patient_indices],
            survival=self.data.survival[patient_indices] if hasattr(
                self.data, 'survival') else None,
            patient_ids=[self.data.patient_ids[i] for i in patient_indices],
            action_cols=getattr(self.data, 'action_cols', None),
            action_indices=getattr(self.data, 'action_indices', None)
        )

        # Handle edge_index (and edge_attr) if present —— 使用一次性张量映射提升效率
        if hasattr(self.data, 'edge_index'):
            edge_index = self.data.edge_index

            # 创建映射张量: old_idx -> new_idx, 其余置 -1
            mapping = torch.full(
                (self.data.x.size(0),),
                -1,
                dtype=torch.long,
                device=edge_index.device,
            )
            mapping[patient_indices] = torch.arange(
                patient_indices.size(0), device=edge_index.device
            )

            src_mapped = mapping[edge_index[0]]
            dst_mapped = mapping[edge_index[1]]

            valid_mask = (src_mapped >= 0) & (dst_mapped >= 0)

            subset_edges = torch.stack(
                (src_mapped[valid_mask], dst_mapped[valid_mask]), dim=0
            )
            subset_data.edge_index = subset_edges

            # 同步 edge_attr（若有）
            if hasattr(self.data, 'edge_attr'):
                subset_data.edge_attr = self.data.edge_attr[valid_mask]

        # Create new dataset instance
        subset_dataset = PatientTrajectoryDataset.__new__(
            PatientTrajectoryDataset)
        subset_dataset.data = subset_data
        subset_dataset.cohort = self.cohort
        subset_dataset.task = self.task
        subset_dataset.filename = f"subset_{self.filename}"
        subset_dataset.slices = None

        return subset_dataset

    def get_outcomes(self) -> torch.Tensor:
        """
        Get patient outcomes (survival labels).

        Returns:
            Tensor of survival outcomes (0=alive, 1=dead).
        """
        return self.data.survival if hasattr(self.data, 'survival') else torch.zeros(len(self.data.patient_ids))

    def save_subset(self, patient_indices: Union[List[int], np.ndarray, torch.Tensor], save_path: str):
        """
        Save a subset of the dataset to file.

        Args:
            patient_indices: Indices of patients to include in the subset.
            save_path: Path where to save the subset.
        """
        subset_dataset = self.subset(patient_indices)
        torch.save(subset_dataset.data, save_path)
        print(
            f"Subset with {len(patient_indices)} patients saved to {save_path}")

    def _validate_data(self):
        """
        Validate the loaded data structure.

        Raises:
            ValueError: If required fields are missing or inconsistent.
        """
        required_fields = ['x', 'mask', 'lengths', 'patient_ids']
        for field in required_fields:
            if not hasattr(self.data, field):
                raise ValueError(f"Required field '{field}' missing from data")

        # Check consistency of dimensions
        n_patients = len(self.data.patient_ids)
        if self.data.x.shape[0] != n_patients:
            raise ValueError(
                f"Inconsistent number of patients: {self.data.x.shape[0]} vs {n_patients}")

        if self.data.mask.shape[0] != n_patients:
            raise ValueError(
                f"Inconsistent mask shape: {self.data.mask.shape[0]} vs {n_patients}")

        if len(self.data.lengths) != n_patients:
            raise ValueError(
                f"Inconsistent lengths: {len(self.data.lengths)} vs {n_patients}")

        print(
            f"✓ Dataset validation passed: {n_patients} patients, task: {self.task}, cohort: {self.cohort}")


def load_dataset_by_task(task: str, root: str = "Input/processed/", **kwargs) -> PatientTrajectoryDataset:
    """
    Convenience function to load a dataset for a specific task.
    
    Args:
        task: Task type ('vent', 'rrt', 'iv').
        root: Root directory for datasets.
        **kwargs: Additional arguments passed to PatientTrajectoryDataset.
        
    Returns:
        PatientTrajectoryDataset instance for the specified task.
        
    Example:
        >>> dataset = load_dataset_by_task('vent')
        >>> rrt_dataset = load_dataset_by_task('rrt', root='Input/processed/')
    """
    return PatientTrajectoryDataset(root=root, task=task, **kwargs)
