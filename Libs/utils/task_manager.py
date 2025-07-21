"""Task Configuration Manager for Multi-Task Medical RL.

This module provides a centralized system for managing task-specific configurations
across all components of the medical RL framework. It replaces hardcoded values
with dynamic, task-aware configurations.

Supported tasks:
- iv: IV fluids and vasopressor strategy optimization
- rrt: Renal replacement therapy strategy optimization  
- vent: Mechanical ventilation strategy optimization

Author: Google Engineer-style Task Management System
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "TaskConfig",
    "TaskManager", 
    "get_task_manager",
    "TASK_CONFIGS"
]


@dataclass
class TaskConfig:
    """Configuration for a specific medical task.
    
    This dataclass encapsulates all task-specific parameters including:
    - Action space dimensions and names
    - Visualization labels and colors
    - Model architecture parameters
    - Medical domain-specific constants
    """
    
    # Task identification
    task_name: str
    description: str
    
    # Action space configuration
    action_dims: List[int]
    action_cols: List[str] 
    action_names: List[str]  # Human-readable names for visualization
    action_descriptions: List[str]  # Detailed descriptions
    
    # Visualization configuration
    vis_action_names: List[str]  # Short names for plots
    vis_colors: List[str] = field(default_factory=list)
    
    # Model configuration
    expected_n_actions: int = 3  # Expected number of action dimensions
    
    # Medical domain specifics
    medical_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration consistency."""
        if len(self.action_dims) != len(self.action_cols):
            raise ValueError(f"Mismatch: action_dims ({len(self.action_dims)}) vs action_cols ({len(self.action_cols)})")
        
        if len(self.action_dims) != len(self.action_names):
            raise ValueError(f"Mismatch: action_dims ({len(self.action_dims)}) vs action_names ({len(self.action_names)})")
        
        if len(self.action_dims) != self.expected_n_actions:
            logger.warning(f"Task {self.task_name} has {len(self.action_dims)} actions, expected {self.expected_n_actions}")
        
        # Set default visualization colors if not provided
        if not self.vis_colors:
            default_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D']
            self.vis_colors = default_colors[:len(self.action_dims)]


# Task-specific configurations
TASK_CONFIGS = {
    "vent": TaskConfig(
        task_name="vent",
        description="Mechanical ventilation strategy optimization",
        action_dims=[7, 7, 7],  # Updated to match actual data dimensions
        action_cols=['peep_bin', 'fio2_bin', 'tidal_volume_ibw_bin'],
        action_names=['PEEP', 'FiO₂', 'Tidal Volume'],
        action_descriptions=[
            'Positive End-Expiratory Pressure (cmH2O)',
            'Fraction of Inspired Oxygen (%)',
            'Tidal Volume per Ideal Body Weight (ml/kg)'
        ],
        vis_action_names=['PEEP', 'FiO₂', 'VT'],
        medical_context={
            'intervention_type': 'respiratory_support',
            'units': ['cmH2O', '%', 'ml/kg'],
            'ranges': [(0, 20), (21, 100), (4, 12)],
            'clinical_guidelines': {
                'low_peep': 'PEEP ≤ 5 cmH2O',
                'high_peep': 'PEEP > 10 cmH2O',
                'low_fio2': 'FiO₂ ≤ 40%',
                'high_fio2': 'FiO₂ > 60%'
            }
        }
    ),
    
    "rrt": TaskConfig(
        task_name="rrt",
        description="Renal replacement therapy strategy optimization",
        action_dims=[5, 5, 5, 2],  # Updated to match actual data dimensions
        action_cols=['rrt_type_bin', 'rrt_dose_bin', 'blood_flow_bin', 'anticoagulation_bin'],
        action_names=['RRT Type', 'RRT Dose', 'Blood Flow', 'Anticoagulation'],
        action_descriptions=[
            'Renal Replacement Therapy modality',
            'Dialysis dose (ml/kg/hr)',
            'Blood flow rate (ml/min)',
            'Anticoagulation strategy'
        ],
        vis_action_names=['RRT Type', 'Dose', 'Flow', 'Anticoag'],
        expected_n_actions=4,
        medical_context={
            'intervention_type': 'renal_support',
            'units': ['type', 'ml/kg/hr', 'ml/min', 'strategy'],
            'ranges': [(0, 3), (10, 40), (100, 300), (0, 1)],
            'clinical_guidelines': {
                'low_dose': 'RRT dose ≤ 20 ml/kg/hr',
                'high_dose': 'RRT dose > 30 ml/kg/hr',
                'low_flow': 'Blood flow ≤ 150 ml/min',
                'high_flow': 'Blood flow > 200 ml/min'
            }
        }
    ),
    
    "iv": TaskConfig(
        task_name="iv",
        description="IV fluids and vasopressor strategy optimization",
        action_dims=[5, 5],  # Updated to match actual data dimensions
        action_cols=['iv_fluids_bin', 'vasopressor_bin'],
        action_names=['IV Fluids', 'Vasopressor'],
        action_descriptions=[
            'Intravenous fluid administration rate (ml/hr)',
            'Vasopressor dosage (mcg/kg/min)'
        ],
        vis_action_names=['IV Fluids', 'Vasopressor'],
        expected_n_actions=2,
        medical_context={
            'intervention_type': 'hemodynamic_support',
            'units': ['ml/hr', 'mcg/kg/min'],
            'ranges': [(0, 500), (0, 20)],
            'clinical_guidelines': {
                'low_fluid': 'IV fluids ≤ 100 ml/hr',
                'high_fluid': 'IV fluids > 300 ml/hr',
                'low_vasopressor': 'Vasopressor ≤ 5 mcg/kg/min',
                'high_vasopressor': 'Vasopressor > 15 mcg/kg/min'
            }
        }
    )
}


class TaskManager:
    """Centralized manager for task-specific configurations.
    
    This class provides a unified interface for accessing task-specific
    configurations across the entire medical RL framework. It ensures
    consistency and eliminates hardcoded values.
    
    Features:
    - Automatic task detection from dataset files
    - Configuration validation and error handling
    - Fallback mechanisms for missing configurations
    - Integration with existing YAML configuration system
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the TaskManager.
        
        Args:
            config_path: Optional path to additional task configurations
        """
        self.configs = TASK_CONFIGS.copy()
        self.current_task: Optional[str] = None
        
        if config_path:
            self._load_additional_configs(config_path)
    
    def _load_additional_configs(self, config_path: Union[str, Path]):
        """Load additional task configurations from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Task config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                additional_configs = yaml.safe_load(f)
            
            # Merge with existing configurations
            for task_name, config_data in additional_configs.items():
                if task_name in self.configs:
                    logger.info(f"Overriding existing config for task: {task_name}")
                
                # Convert dict to TaskConfig
                task_config = TaskConfig(**config_data)
                self.configs[task_name] = task_config
                
        except Exception as e:
            logger.error(f"Failed to load additional configs from {config_path}: {e}")
    
    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task.
        
        Args:
            task_name: Name of the task (iv, rrt, vent)
            
        Returns:
            TaskConfig object for the specified task
            
        Raises:
            ValueError: If task is not supported
        """
        if task_name not in self.configs:
            available_tasks = list(self.configs.keys())
            raise ValueError(f"Unsupported task: {task_name}. Available tasks: {available_tasks}")
        
        return self.configs[task_name]
    
    def set_current_task(self, task_name: str):
        """Set the current active task.
        
        Args:
            task_name: Name of the task to set as current
        """
        config = self.get_task_config(task_name)  # Validates task exists
        self.current_task = task_name
        logger.info(f"✅ Current task set to: {task_name} - {config.description}")
    
    def get_current_config(self) -> TaskConfig:
        """Get configuration for the current active task.
        
        Returns:
            TaskConfig object for the current task
            
        Raises:
            RuntimeError: If no task is currently set
        """
        if self.current_task is None:
            raise RuntimeError("No task is currently set. Call set_current_task() first.")
        
        return self.get_task_config(self.current_task)
    
    def get_action_dims(self, task_name: Optional[str] = None) -> List[int]:
        """Get action dimensions for a task.
        
        Args:
            task_name: Task name, or None to use current task
            
        Returns:
            List of action dimensions
        """
        config = self.get_task_config(task_name) if task_name else self.get_current_config()
        return config.action_dims
    
    def get_action_names(self, task_name: Optional[str] = None, for_visualization: bool = False) -> List[str]:
        """Get action names for a task.
        
        Args:
            task_name: Task name, or None to use current task
            for_visualization: Whether to return short names for visualization
            
        Returns:
            List of action names
        """
        config = self.get_task_config(task_name) if task_name else self.get_current_config()
        return config.vis_action_names if for_visualization else config.action_names
    
    def get_action_cols(self, task_name: Optional[str] = None) -> List[str]:
        """Get action column names for a task.
        
        Args:
            task_name: Task name, or None to use current task
            
        Returns:
            List of action column names
        """
        config = self.get_task_config(task_name) if task_name else self.get_current_config()
        return config.action_cols
    
    def get_vis_colors(self, task_name: Optional[str] = None) -> List[str]:
        """Get visualization colors for a task.
        
        Args:
            task_name: Task name, or None to use current task
            
        Returns:
            List of color hex codes
        """
        config = self.get_task_config(task_name) if task_name else self.get_current_config()
        return config.vis_colors
    
    def get_medical_context(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get medical context information for a task.
        
        Args:
            task_name: Task name, or None to use current task
            
        Returns:
            Dictionary containing medical context
        """
        config = self.get_task_config(task_name) if task_name else self.get_current_config()
        return config.medical_context
    
    def validate_action_dims(self, action_dims: List[int], task_name: Optional[str] = None) -> bool:
        """Validate that action dimensions match task configuration.
        
        Args:
            action_dims: Action dimensions to validate
            task_name: Task name, or None to use current task
            
        Returns:
            True if action dimensions match, False otherwise
        """
        expected_dims = self.get_action_dims(task_name)
        if action_dims != expected_dims:
            logger.warning(f"Action dimension mismatch: got {action_dims}, expected {expected_dims}")
            return False
        return True
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of all supported tasks.
        
        Returns:
            List of task names
        """
        return list(self.configs.keys())
    
    def infer_task_from_data(self, data_path: Union[str, Path]) -> Optional[str]:
        """Infer task type from dataset path or filename.
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            Inferred task name, or None if cannot infer
        """
        path_str = str(data_path).lower()
        
        # Check for task-specific patterns in path
        for task_name in self.get_supported_tasks():
            if task_name in path_str:
                return task_name
        
        # Check for task-specific action column patterns
        try:
            import torch
            if str(data_path).endswith('.pt'):
                data = torch.load(data_path, map_location='cpu')
                if hasattr(data, 'actions'):
                    n_actions = data.actions.shape[-1]
                    
                    # Match based on number of action dimensions
                    for task_name, config in self.configs.items():
                        if len(config.action_dims) == n_actions:
                            logger.info(f"Inferred task '{task_name}' from action dimensions: {n_actions}")
                            return task_name
        except Exception as e:
            logger.debug(f"Could not infer task from data: {e}")
        
        return None
    
    def create_model_config(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Create model configuration dictionary for a task.
        
        Args:
            task_name: Task name, or None to use current task
            
        Returns:
            Dictionary containing model configuration
        """
        config = self.get_task_config(task_name) if task_name else self.get_current_config()
        
        return {
            'action_dims': config.action_dims,
            'action_cols': config.action_cols,
            'n_actions': len(config.action_dims),
            'task_name': config.task_name,
            'medical_context': config.medical_context
        }


# Global task manager instance
_global_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global TaskManager instance.
    
    Returns:
        Global TaskManager instance
    """
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = TaskManager()
    return _global_task_manager


def set_global_task(task_name: str):
    """Set the global current task.
    
    Args:
        task_name: Name of the task to set as current
    """
    task_manager = get_task_manager()
    task_manager.set_current_task(task_name)


def get_current_task_config() -> TaskConfig:
    """Get the current task configuration.
    
    Returns:
        TaskConfig for the current task
    """
    task_manager = get_task_manager()
    return task_manager.get_current_config()


# Convenience functions for common operations
def get_action_dims(task_name: Optional[str] = None) -> List[int]:
    """Get action dimensions for current or specified task."""
    return get_task_manager().get_action_dims(task_name)


def get_action_names(task_name: Optional[str] = None, for_visualization: bool = False) -> List[str]:
    """Get action names for current or specified task."""
    return get_task_manager().get_action_names(task_name, for_visualization)


def get_action_cols(task_name: Optional[str] = None) -> List[str]:
    """Get action column names for current or specified task."""
    return get_task_manager().get_action_cols(task_name) 