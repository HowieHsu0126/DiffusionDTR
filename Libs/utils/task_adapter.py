"""Task Adapter for Model Compatibility.

This module provides adapters to handle compatibility between models designed 
for specific action spaces and different medical tasks. It ensures that models
can work across different tasks while maintaining their architectural integrity.

Author: Google Engineer-style Task Adaptation System
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Union, Type
import torch
import torch.nn as nn

from Libs.utils.task_manager import get_task_manager, TaskConfig

logger = logging.getLogger(__name__)

__all__ = [
    "TaskAdapter",
    "ModelTaskCompatibility",
    "check_model_task_compatibility",
    "adapt_model_for_task"
]


class ModelTaskCompatibility:
    """Defines compatibility between models and tasks."""

    # Models that work with any number of action dimensions
    UNIVERSAL_MODELS = {
        'bc', 'dqn', 'cql', 'bcq', 'bve', 'pog_bve', 'pog_bc', 'pog_dqn', 'pog_cql', 'pog_bcq', 'physician'
    }

    # Models with specific action dimension requirements
    SPECIFIC_MODELS = {
        # Currently no models have specific action dimension requirements
        # All implemented models are designed to work with arbitrary action dimensions
    }

    # Task-specific action dimension mappings
    TASK_MAPPINGS = {
        'vent': {'n_actions': 3, 'compatible_models': ['all']},
        'rrt': {'n_actions': 4, 'compatible_models': ['all']},
        'iv': {'n_actions': 2, 'compatible_models': ['all']},
    }


class TaskAdapter:
    """Adapter to handle model-task compatibility issues.
    
    This class provides methods to check compatibility between models and tasks,
    and suggest alternatives when incompatibilities are detected.
    """

    def __init__(self):
        self.task_manager = get_task_manager()
        self.compatibility = ModelTaskCompatibility()

    def check_compatibility(self, model_name: str, task_name: str) -> Dict[str, Any]:
        """Check if a model is compatible with a task.
        
        Args:
            model_name: Name of the model/algorithm
            task_name: Name of the task
            
        Returns:
            Dictionary containing compatibility information
        """
        result = {
            'compatible': False,
            'reason': '',
            'suggestions': [],
            'can_adapt': False
        }

        # Normalize model name
        model_name = model_name.lower().replace('_', '').replace('-', '')

        # Check if model is universal (works with any action dimensions)
        if model_name in self.compatibility.UNIVERSAL_MODELS:
            result['compatible'] = True
            result['reason'] = f"Model {model_name} is universal and works with any task"
            return result

        # Check specific model requirements
        if model_name in self.compatibility.SPECIFIC_MODELS:
            spec = self.compatibility.SPECIFIC_MODELS[model_name]

            # Check if task is in compatible tasks list
            if task_name in spec['compatible_tasks']:
                result['compatible'] = True
                result['reason'] = f"Model {model_name} is specifically designed for task {task_name}"
                return result

            # Check action dimension compatibility
            task_config = self.task_manager.get_task_config(task_name)
            required_actions = spec['required_actions']
            actual_actions = len(task_config.action_dims)

            if actual_actions != required_actions:
                # Use detailed reason if available, otherwise fall back to generic message
                if 'reason' in spec:
                    result['reason'] = f"{spec['reason']}. Task {task_name} has {actual_actions} actions, but {required_actions} are required."
                else:
                    result['reason'] = f"Model {model_name} requires {required_actions} actions, but task {task_name} has {actual_actions}"
                result['suggestions'] = self._get_alternative_models(
                    task_name, exclude=model_name)
                return result

        # Check task-specific compatibility
        if task_name in self.compatibility.TASK_MAPPINGS:
            task_spec = self.compatibility.TASK_MAPPINGS[task_name]
            compatible_models = task_spec['compatible_models']

            if 'all' in compatible_models or model_name in compatible_models:
                result['compatible'] = True
                result['reason'] = f"Task {task_name} supports model {model_name}"
                return result
            else:
                result['reason'] = f"Task {task_name} does not support model {model_name}"
                result['suggestions'] = compatible_models
                return result

        # Default to compatible for unknown combinations
        result['compatible'] = True
        result['reason'] = "No specific compatibility rules found, assuming compatible"
        return result

    def _get_alternative_models(self, task_name: str, exclude: str = None) -> List[str]:
        """Get alternative models for a task.
        
        Args:
            task_name: Name of the task
            exclude: Model name to exclude from suggestions
            
        Returns:
            List of alternative model names
        """
        if task_name not in self.compatibility.TASK_MAPPINGS:
            return list(self.compatibility.UNIVERSAL_MODELS)

        task_spec = self.compatibility.TASK_MAPPINGS[task_name]
        compatible_models = task_spec['compatible_models']

        if 'all' in compatible_models:
            alternatives = list(self.compatibility.UNIVERSAL_MODELS)
        else:
            alternatives = compatible_models.copy()

        if exclude and exclude in alternatives:
            alternatives.remove(exclude)

        return alternatives

    def suggest_task_for_model(self, model_name: str) -> List[str]:
        """Suggest compatible tasks for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of compatible task names
        """
        model_name = model_name.lower().replace('_', '').replace('-', '')

        if model_name in self.compatibility.UNIVERSAL_MODELS:
            return list(self.compatibility.TASK_MAPPINGS.keys())

        if model_name in self.compatibility.SPECIFIC_MODELS:
            return self.compatibility.SPECIFIC_MODELS[model_name]['compatible_tasks']

        # Default to all tasks
        return list(self.compatibility.TASK_MAPPINGS.keys())

    def get_compatibility_report(self, model_name: str, task_name: str) -> str:
        """Generate a human-readable compatibility report.
        
        Args:
            model_name: Name of the model
            task_name: Name of the task
            
        Returns:
            Formatted compatibility report
        """
        compat = self.check_compatibility(model_name, task_name)

        report = f"Model-Task Compatibility Report\n"
        report += f"================================\n"
        report += f"Model: {model_name.upper()}\n"
        report += f"Task: {task_name.upper()}\n"
        report += f"Compatible: {'✅ YES' if compat['compatible'] else '❌ NO'}\n"
        report += f"Reason: {compat['reason']}\n"

        if compat['suggestions']:
            report += f"Suggested alternatives: {', '.join(compat['suggestions'])}\n"

        return report


def check_model_task_compatibility(model_name: str, task_name: str) -> bool:
    """Check if a model is compatible with a task.
    
    Args:
        model_name: Name of the model/algorithm
        task_name: Name of the task
        
    Returns:
        True if compatible, False otherwise
    """
    adapter = TaskAdapter()
    result = adapter.check_compatibility(model_name, task_name)
    return result['compatible']


def adapt_model_for_task(model_name: str, task_name: str, **kwargs) -> Dict[str, Any]:
    """Adapt model parameters for a specific task.
    
    Args:
        model_name: Name of the model
        task_name: Name of the task
        **kwargs: Additional model parameters
        
    Returns:
        Dictionary of adapted model parameters
    """
    adapter = TaskAdapter()
    task_config = adapter.task_manager.get_task_config(task_name)

    # Get base configuration
    adapted_config = {
        'action_dims': task_config.action_dims,
        'task_name': task_name,
        'n_actions': len(task_config.action_dims),
        **kwargs
    }

    # Check compatibility and add warnings if needed
    compat = adapter.check_compatibility(model_name, task_name)
    if not compat['compatible']:
        logger.warning(
            f"Model {model_name} may not be compatible with task {task_name}")
        logger.warning(f"Reason: {compat['reason']}")
        if compat['suggestions']:
            logger.warning(
                f"Consider using: {', '.join(compat['suggestions'])}")

    return adapted_config


def validate_model_task_combination(model_name: str, task_name: str,
                                    action_dims: List[int], raise_error: bool = True) -> bool:
    """Validate that a model-task combination is valid.
    
    Args:
        model_name: Name of the model
        task_name: Name of the task
        action_dims: Action dimensions from data
        raise_error: Whether to raise an error on incompatibility
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If combination is invalid and raise_error=True
    """
    adapter = TaskAdapter()

    # Check model-task compatibility
    compat = adapter.check_compatibility(model_name, task_name)

    # Check action dimensions match task configuration
    task_config = adapter.task_manager.get_task_config(task_name)
    expected_dims = task_config.action_dims

    if action_dims != expected_dims:
        msg = f"Action dimensions mismatch for {task_name}: expected {expected_dims}, got {action_dims}"
        if raise_error:
            raise ValueError(msg)
        logger.warning(msg)
        return False

    if not compat['compatible']:
        msg = f"Model {model_name} is not compatible with task {task_name}: {compat['reason']}"
        if raise_error:
            raise ValueError(msg)
        logger.warning(msg)
        return False

    return True
