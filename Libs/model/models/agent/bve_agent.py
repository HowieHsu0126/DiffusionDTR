"""Branch Value Estimation (BVE) reinforcement learning agent.

This module implements the BVE agent that combines the BVE Q-network with
advanced training techniques including CQL regularization, branch normalization,
flexible sampling strategies, and adaptive alpha tuning.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Libs.model.models.agent.base_agent import BaseRLAgent
from Libs.model.modules.bve_qnetwork import BranchValueEstimationQNetwork
from Libs.utils.log_utils import get_logger
from Libs.utils.model_utils import (apply_gradient_clipping, safe_float,
                                    safe_item)

logger = get_logger(__name__)


class BranchValueEstimationAgent(BaseRLAgent):
    """Advanced Branch Value Estimation agent for multi-discrete action spaces.

    This agent implements the state-of-the-art Branch Value Estimation (BVE) approach
    specifically designed for medical decision making with structured multi-discrete
    action spaces. The agent combines several advanced techniques:

    Key Features:
        • Branch-wise Q-value decomposition for independent action dimensions
        • Conservative Q-Learning (CQL) regularization for offline RL robustness
        • Adaptive alpha tuning with Lagrange multipliers for automatic hyperparameter optimization
        • Branch Q-value normalization for stable multi-head training
        • Flexible sampling strategies (uniform/policy/mixed) for CQL loss computation
        • Beam search inference for high-quality action selection
        • Enhanced numerical stability with gradient clipping and regularization

    Medical Application Context:
        The agent is specifically engineered for ICU-AKI treatment scenarios where
        different action dimensions represent distinct medical interventions:
        - Mechanical ventilation settings (PEEP, FiO₂, Tidal Volume)
        - Medication dosages (vasopressors, sedatives)
        - Fluid management protocols

    Architecture Overview:
        Input: Patient state representation (vital signs, lab values, demographics)
        ↓
        Shared feature extraction (state encoder)
        ↓
        Branch-wise Q-value heads (one per action dimension)
        ↓
        Greedy/Beam search action selection + CQL regularization during training

    Mathematical Foundation:
        For state s and multi-dimensional action a = (a₁, a₂, ..., aₖ):
        Q(s,a) = Σᵢ Qᵢ(s,aᵢ) / k  (branch-wise decomposition)

        CQL Loss: L_CQL = α * (log Σₐ exp(Q(s,a)) - Q(s,a_data))
        TD Loss: L_TD = (Q(s,a) - r - γ * max_{a'} Q(s',a'))²
        Total Loss: L = L_TD + α * L_CQL + λ * behavior_regularization

    Attributes:
        q_net: Primary Q-network implementing branch-wise value estimation.
        target_q_net: Target Q-network for stable training (updated via Polyak averaging).
        optimizer: AdamW optimizer for Q-network parameters with weight decay.
        alpha_optimizer: Separate optimizer for adaptive alpha parameter.
        state_dim: Dimensionality of patient state representation.
        action_dims: List of action space dimensions for each medical intervention type.
        lr: Learning rate for Q-network optimization.
        lambda_reg: Weight for behavior consistency regularization.
        alpha: CQL regularization weight (auto-tuned via Lagrange multipliers).
        cql_n_samples: Number of action samples for CQL loss computation.
        normalize_branch: Whether to apply branch-wise Q-value normalization.
        cql_sample_mode: Sampling strategy for CQL loss:
                            'uniform': Sample actions uniformly from each dimension
                            'policy': Sample from current policy (higher variance)
                            'mixed': Combine uniform + policy sampling (recommended)
            cql_target_gap: Target conservative gap for alpha adaptation.
        alpha_lr: Learning rate for alpha parameter optimization.
        max_grad_norm: Maximum gradient norm for clipping.
        behavior_penalty_mode: Mode for behavior consistency penalty computation.
        softmax_temperature: Temperature scaling for action probability computation.
        n_step: Number of steps for n-step TD target computation.

    Example:
        >>> # Initialize BVE agent for ICU mechanical ventilation control
        >>> agent = BranchValueEstimationAgent(
        ...     state_dim=87,  # Patient vital signs + lab values + demographics
        ...     action_dims=[7, 6, 6],  # PEEP levels, FiO₂ settings, Tidal Volume bins
        ...     lr=1e-3,
        ...     alpha=1.0,  # Will be auto-tuned during training
        ...     cql_n_samples=10,
        ...     normalize_branch=True,
        ...     cql_sample_mode='mixed'
        ... )
        >>> 
        >>> # Training step
        >>> batch = {
        ...     'state': patient_states,      # (batch_size, state_dim)
        ...     'action': chosen_actions,     # (batch_size, n_heads)
        ...     'reward': clinical_outcomes,  # (batch_size,)
        ...     'next_state': next_states,    # (batch_size, state_dim)
        ...     'done': episode_termination   # (batch_size,)
        ... }
        >>> loss = agent.update(batch)
        >>>
        >>> # Inference
        >>> best_actions = agent.greedy_action(patient_states)  # (batch_size, n_heads)
        >>> top_k_actions, q_values = agent.beam_search(patient_states, beam_width=5)
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        lr: float = 1e-3,
        gamma: float = 0.99,
        lambda_reg: float = 0.1,
        alpha: float = 1.0,
        device: Union[str, torch.device] = 'cpu',
        q_net: Optional[BranchValueEstimationQNetwork] = None,
        cql_n_samples: int = 10,
        normalize_branch: bool = True,
        cql_sample_mode: str = 'uniform',
        cql_target_gap: float = 5.0,
        alpha_lr: float = 1e-4,
        target_update_freq: int = 100,
        hidden_dim: int = 128,
        reward_centering: bool = False,
        max_grad_norm: float = 1.0,
        behavior_penalty_mode: str = 'count',
        n_step: int = 1,
        softmax_temperature: float = 1.0,
        polyak_tau: float = 0.005,
        mixed_ratio: float = 0.5,
        action_support_masks: Optional[List[torch.Tensor]] = None,
        safe_max_indices: Optional[List[int]] = None,
        **kwargs
    ) -> None:
        """Initializes the Branch Value Estimation agent with comprehensive validation.

        Args:
            state_dim: Dimension of patient state representation. Should match the
                      feature dimension of preprocessed patient data (typically 64-128
                      for ICU datasets after feature engineering).
            action_dims: List of action space dimensions for each medical intervention.
                        Example: [7, 6, 6] for PEEP×FiO₂×TidalVolume in mechanical
                        ventilation. Each dimension should be ≥2 for meaningful choice.
            lr: Learning rate for Q-network optimization. Default 1e-3 works well
               for most medical RL scenarios. Consider reducing to 1e-4 for fine-tuning.
            gamma: Discount factor for future rewards. Default 0.99 is appropriate
                  for medical scenarios where long-term outcomes matter.
            lambda_reg: Weight for behavior consistency regularization. Higher values
                       (>0.1) encourage actions closer to observed clinical practice.
            alpha: Initial CQL regularization weight. Will be auto-tuned during training
                  via Lagrange multipliers to maintain target conservative gap.
            device: Computation device. Use 'cuda' for GPU acceleration with large
                   datasets (>10K patients). CPU sufficient for smaller studies.
            q_net: Optional pre-initialized Q-network. If None, creates default
                  BranchValueEstimationQNetwork with specified architecture.
            cql_n_samples: Number of action samples for CQL loss computation. More
                          samples (10-20) provide better CQL estimates but increase
                          computational cost quadratically.
            normalize_branch: Whether to normalize branch Q-values. Recommended True
                             for medical applications to prevent one intervention type
                             from dominating the learning signal.
            cql_sample_mode: Sampling strategy for CQL loss:
                            'uniform': Sample actions uniformly from each dimension
                            'policy': Sample from current policy (higher variance)
                            'mixed': Combine uniform + policy sampling (recommended)
            cql_target_gap: Target conservative gap for alpha adaptation. Higher values
                           (5-10) provide more conservative policies, suitable for
                           safety-critical medical applications.
            alpha_lr: Learning rate for alpha parameter adaptation. Should be much
                     smaller than main lr (typically 1e-4) for stable convergence.
            target_update_freq: Frequency of target network updates. Every 100 steps
                               provides good stability-plasticity tradeoff.
            hidden_dim: Hidden dimension for Q-network layers. 128-256 typical for
                       medical applications with moderate state dimensionality.
            reward_centering: Whether to apply reward centering for variance reduction.
                             Recommended False for medical rewards with meaningful scale.
            max_grad_norm: Maximum gradient norm for clipping. 1.0 prevents exploding
                          gradients while preserving learning speed.
            behavior_penalty_mode: Mode for behavior consistency penalty:
                                  'count': Frequency-based penalty (fast)
                                  'log_prob': Log-probability penalty (more principled)
                                  'kl': KL divergence penalty (most sophisticated)
            n_step: Number of steps for n-step TD targets. 1-step sufficient for most
                   medical applications with proper reward engineering.
            softmax_temperature: Temperature for action probability computation. Lower
                                values (0.1-1.0) provide sharper action distributions.
            polyak_tau: Polyak averaging coefficient for target network updates.
                       0.005 provides smooth target updates for stable learning.
            mixed_ratio: Ratio of policy vs uniform samples in mixed CQL mode.
                        0.5 balances exploration with policy refinement.
            action_support_masks: Optional boolean masks indicating valid actions per
                                 branch. Useful for incorporating clinical constraints
                                 (e.g., contraindicated drug combinations).
            **kwargs: Additional arguments passed to parent BaseRLAgent.

        Raises:
            ValueError: If any parameter is outside valid range or incompatible.
            RuntimeError: If device setup fails or network initialization fails.

        Note:
            The agent automatically validates input parameters and provides detailed
            error messages for invalid configurations. For medical applications,
            ensure action_dims match the discretization of your intervention space
            and state_dim matches your feature preprocessing pipeline.
        """
        # ========================================================================
        # Enhanced Parameter Validation with Medical Domain Context
        # ========================================================================

        # Core dimensional validation
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if len(action_dims) < 2:
            raise ValueError(
                f"BVE requires at least 2 action dimensions for branch decomposition, "
                f"got {len(action_dims)}. For single-action problems, use DQN instead."
            )
        if any(dim < 2 for dim in action_dims):
            raise ValueError(
                f"All action dimensions must be ≥2 for meaningful choice, "
                f"got {action_dims}. Consider removing trivial action dimensions."
            )

        # Medical safety validation for typical ICU scenarios
        total_action_space = np.prod(action_dims)
        if total_action_space > 10000:
            logger.warning(
                f"⚠️  Large action space ({total_action_space:,} combinations) "
                f"may require increased cql_n_samples for adequate exploration. "
                f"Consider action space reduction or hierarchical approaches."
            )

        # Hyperparameter range validation
        if not 0.5 <= gamma <= 0.999:
            raise ValueError(
                f"gamma should be in [0.5, 0.999] for medical RL, got {gamma}")
        if not 0.0 <= lambda_reg <= 1.0:
            raise ValueError(
                f"lambda_reg should be in [0.0, 1.0], got {lambda_reg}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if cql_target_gap <= 0:
            raise ValueError(
                f"cql_target_gap must be positive, got {cql_target_gap}")
        if n_step <= 0:
            raise ValueError(f"n_step must be positive, got {n_step}")
        if softmax_temperature <= 0.0:
            raise ValueError(
                f"softmax_temperature must be positive, got {softmax_temperature}")

        # Mode validation with helpful error messages
        valid_sample_modes = ['uniform', 'policy', 'mixed']
        if cql_sample_mode not in valid_sample_modes:
            raise ValueError(
                f"cql_sample_mode must be one of {valid_sample_modes}, got '{cql_sample_mode}'. "
                f"Recommended: 'mixed' for medical applications."
            )

        valid_penalty_modes = ['count', 'log_prob', 'kl']
        if behavior_penalty_mode not in valid_penalty_modes:
            raise ValueError(
                f"behavior_penalty_mode must be one of {valid_penalty_modes}, "
                f"got '{behavior_penalty_mode}'"
            )

        # ========================================================================
        # Enhanced Initialization with Comprehensive Logging
        # ========================================================================

        # Initialize parent class with enhanced error handling
        # Extract base agent parameters from kwargs
        base_agent_kwargs = {
            'device': device,
            'gamma': gamma,
            'target_update_freq': target_update_freq,
            'reward_centering': reward_centering,
            'polyak_tau': polyak_tau,
        }
        
        # Filter out BVE-specific parameters from kwargs before passing to base class
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['lambda_reg', 'alpha', 'cql_n_samples', 'normalize_branch', 
                                       'cql_sample_mode', 'cql_target_gap', 'alpha_lr', 'hidden_dim', 
                                       'max_grad_norm', 'behavior_penalty_mode', 'n_step', 
                                       'softmax_temperature', 'mixed_ratio', 'action_support_masks']}
        
        # Merge filtered kwargs with base agent kwargs
        base_agent_kwargs.update(filtered_kwargs)
        
        try:
            super().__init__(**base_agent_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize parent BaseRLAgent: {e}") from e

        # Store configuration parameters for reproducibility and debugging
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.lr = lr
        
        # Set safe action limits for medical applications based on task type
        self.safe_max_indices = self._configure_medical_safety_limits(
            action_dims, safe_max_indices)
            
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.cql_n_samples = cql_n_samples
        self.normalize_branch = normalize_branch
        self.cql_sample_mode = cql_sample_mode
        self.cql_target_gap = cql_target_gap
        self.alpha_lr = alpha_lr
        self.softmax_temperature = float(softmax_temperature)
        self.n_step = int(n_step)
        self.mixed_ratio = float(mixed_ratio)

        # Behavior consistency penalty configuration
        self.behavior_penalty_mode = behavior_penalty_mode.lower()

        # ========================================================================
        # Enhanced Q-Network Initialization with Architecture Logging
        # ========================================================================

        try:
            if q_net is None:
                self.q_net = BranchValueEstimationQNetwork(
                    state_dim=state_dim,
                    action_dims=action_dims,
                    hidden_dim=hidden_dim
                ).to(self.device)
                logger.debug(
                    f"✅ Created BVE Q-network: {state_dim}→{hidden_dim}→{action_dims}")
            else:
                self.q_net = q_net.to(self.device)
                logger.debug("✅ Using provided pre-initialized Q-network")

            # Create target network with identical architecture
            self.target_q_net = copy.deepcopy(self.q_net)
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            logger.debug("✅ Target Q-network initialized via deep copy")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Q-networks: {e}") from e

        # ========================================================================
        # Enhanced Optimizer Setup with Medical-Specific Configurations
        # ========================================================================

        try:
            # Use AdamW with weight decay for better generalization in medical data
            self.optimizer = torch.optim.AdamW(
                self.q_net.parameters(),
                lr=lr,
                weight_decay=1e-4,  # L2 regularization for medical robustness
                eps=1e-8,           # Numerical stability
                amsgrad=True        # Better convergence for medical RL
            )

            # Lagrange multiplier for adaptive alpha tuning
            self.log_alpha = torch.tensor(
                np.log(alpha),
                requires_grad=True,
                device=self.device,
                dtype=torch.float32
            )

            # Separate optimizer for alpha with smaller learning rate
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=alpha_lr,
                weight_decay=0,  # No regularization on alpha
                eps=1e-8
            )

            logger.debug(
                f"✅ Optimizers initialized: Q-net_lr={lr}, alpha_lr={alpha_lr}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize optimizers: {e}") from e

        # ========================================================================
        # Advanced Training State Management
        # ========================================================================

        # Training progression tracking (training_step is inherited from BaseRLAgent)
        # Store BVE-specific training parameters
        self.max_grad_norm = max_grad_norm

        # Performance monitoring for medical safety
        self.last_td_loss = 0.0
        self.last_cql_loss_final = 0.0
        self.last_cql_loss = 0.0
        self.last_alpha = alpha
        self.convergence_history = []

        # Medical-specific action constraints
        self.action_support_masks = action_support_masks
        if action_support_masks is not None:
            if len(action_support_masks) != len(action_dims):
                raise ValueError(
                    f"action_support_masks length ({len(action_support_masks)}) "
                    f"must match action_dims length ({len(action_dims)})"
                )
            logger.debug(
                "✅ Action support masks loaded for clinical constraint enforcement")
        
        # Configure valid action values to address data sparsity issues
        self.valid_action_values = self._configure_valid_action_values(action_dims)
        logger.debug(f"✅ Valid action values configured: {self.valid_action_values}")

        # ========================================================================
        # Comprehensive Configuration Summary for Reproducibility
        # ========================================================================

        logger.debug("🏥 BVE Agent Configuration Summary:")
        logger.debug(f"   • Medical Application: Multi-discrete action space RL")
        logger.debug(f"   • State Dimension: {state_dim} (patient features)")
        logger.debug(
            f"   • Action Space: {action_dims} → {total_action_space:,} combinations")
        logger.debug(f"   • Architecture: {hidden_dim}-dim hidden layers")
        logger.debug(
            f"   • CQL Configuration: n_samples={cql_n_samples}, mode={cql_sample_mode}")
        logger.debug(f"   • Conservative Gap Target: {cql_target_gap}")
        logger.debug(
            f"   • Behavior Regularization: λ={lambda_reg}, mode={behavior_penalty_mode}")
        logger.debug(
            f"   • Optimization: lr={lr}, alpha_lr={alpha_lr}, grad_clip={max_grad_norm}")
        logger.debug(
            f"   • Target Updates: freq={target_update_freq}, τ={polyak_tau}")
        logger.debug(f"   • Device: {self.device}")
        logger.debug("🚀 BVE Agent ready for medical decision making")

        # Set ready flag for safe operation
        self._initialized = True

    # ========================================================================
    # Medical Safety Configuration Methods
    # ========================================================================
    
    def _configure_medical_safety_limits(
        self, 
        action_dims: List[int], 
        user_safe_max_indices: Optional[List[int]] = None
    ) -> List[int]:
        """Configure medical safety limits based on task type and clinical guidelines.
        
        This method automatically detects the medical task based on action_dims
        and applies appropriate safety constraints following ICU clinical guidelines.
        
        Args:
            action_dims: List of action space dimensions
            user_safe_max_indices: User-provided safety limits (overrides defaults)
            
        Returns:
            List of maximum safe indices for each action dimension
            
        Medical Safety Guidelines:
        -------------------------
        VENT Task [7,7,7] - Mechanical Ventilation:
            • PEEP: Full range (0-6) - Conservative approach allows flexibility
            • FiO₂: Exclude highest (0-5) - Prevent hyperoxia (FiO₂ >80%)
            • Tidal Volume: Exclude highest (0-5) - Lung-protective ventilation (<8ml/kg)
            
        RRT Task [5,5,5,2] - Renal Replacement Therapy:
            • RRT Type: Conservative range (0-3) - Avoid experimental modalities
            • RRT Dose: Conservative range (0-3) - Prevent overdialysis complications
            • Blood Flow: Conservative range (0-3) - Minimize hemodynamic stress
            • Anticoagulation: Full range (0-1) - Binary choice, both safe
            
        IV Task [5,5] - IV Fluids & Vasopressors:
            • IV Fluids: Conservative range (0-3) - Prevent fluid overload
            • Vasopressor: Conservative range (0-3) - Avoid excessive vasoconstriction
        """
        if user_safe_max_indices is not None:
            if len(user_safe_max_indices) != len(action_dims):
                raise ValueError(
                    f"user_safe_max_indices length ({len(user_safe_max_indices)}) "
                    f"must match action_dims length ({len(action_dims)})"
                )
            logger.debug(f"🔒 Using user-provided safety limits: {user_safe_max_indices}")
            return user_safe_max_indices
        
        # Detect task type based on action_dims pattern
        task_type = self._detect_task_type(action_dims)
        
        if task_type == "vent":
            # Mechanical Ventilation Safety Guidelines
            safe_limits = [6, 5, 5]  # PEEP: full, FiO₂: conservative, VT: lung-protective
            logger.debug("🫁 Configured VENT task safety limits: PEEP=full, FiO₂=conservative, VT=lung-protective")
            
        elif task_type == "rrt":
            # Renal Replacement Therapy Safety Guidelines
            safe_limits = [3, 3, 3, 1]  # Conservative for all except anticoagulation
            logger.debug("🩸 Configured RRT task safety limits: Conservative approach for all modalities")
            
        elif task_type == "iv":
            # IV Fluids & Vasopressor Safety Guidelines  
            safe_limits = [3, 3]  # Conservative fluid and vasopressor management
            logger.debug("💧 Configured IV task safety limits: Conservative fluid & vasopressor management")
            
        else:
            # Generic fallback: allow full range except last index for safety
            safe_limits = [dim - 1 for dim in action_dims]
            logger.warning(f"⚠️ Unknown task pattern {action_dims}, using conservative fallback: {safe_limits}")
            
        return safe_limits
        
    def _detect_task_type(self, action_dims: List[int]) -> str:
        """Detect medical task type based on action dimensions pattern.
        
        Args:
            action_dims: List of action space dimensions
            
        Returns:
            Detected task type: 'vent', 'rrt', 'iv', or 'unknown'
        """
        # Exact pattern matching for known tasks
        if action_dims == [7, 7, 7]:
            return "vent"  # Mechanical ventilation
        elif action_dims == [5, 5, 5, 2]:
            return "rrt"   # Renal replacement therapy
        elif action_dims == [5, 5]:
            return "iv"    # IV fluids & vasopressor
        else:
            # Try to detect by pattern characteristics
            n_dims = len(action_dims)
            if n_dims == 3 and all(dim >= 6 for dim in action_dims):
                return "vent"  # Likely ventilation variant
            elif n_dims == 4 and action_dims[-1] == 2:
                return "rrt"   # Likely RRT variant
            elif n_dims == 2 and all(dim >= 4 for dim in action_dims):
                return "iv"    # Likely IV variant
            else:
                return "unknown"

    def _normalize_q_values(self, q_tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes Q-values along the last dimension.

        Args:
            q_tensor: Q-values tensor of shape (..., n_actions).

        Returns:
            Normalized Q-values with zero mean and unit variance.
        """
        mean = q_tensor.mean(dim=-1, keepdim=True)

        # ------------------------------------------------------------------
        #  Fix std() warning: ensure sufficient degrees of freedom
        # ------------------------------------------------------------------
        # For single-action branches (action_dim=1), std() has 0 degrees of freedom
        # and triggers a warning. We handle this explicitly.
        if q_tensor.size(-1) == 1:
            # Single action: no variance to normalize, return zero-centered values
            return q_tensor - mean
        else:
            # Multiple actions: use unbiased estimator with proper degrees of freedom
            # unbiased=False uses N-1 denominator (sample std), unbiased=True uses N (population std)
            # For normalization, we prefer sample std (unbiased=False) as it's more stable
            std = q_tensor.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
            return (q_tensor - mean) / std

    def _sample_actions(
        self,
        state: torch.Tensor,
        n_samples: int,
        mode: str = 'uniform'
    ) -> torch.Tensor:
        """Samples actions for CQL loss computation with medical safety constraints.

        This implementation enforces a 50%/50% blend when ``mode`` is
        ``mixed`` so that the Conservative-Q gap estimation receives *both*
        behaviour‐agnostic uniform samples **and** on‐policy samples.  The
        previous version concatenated the two sets and then drew a second
        round of random indices which biased the effective ratio towards the
        larger set.  When the policy distribution is highly peaked, that
        approach could severely under-sample unlikely actions, leading to an
        underestimated conservative gap.
        
        Medical Safety Enhancement:
        This method now uses medical safety limits instead of raw action_dims
        to prevent sampling from unrealistic action ranges that may exist in
        corrupted data.
        """
        batch_size = state.size(0)

        # Use medical safety limits instead of raw action_dims for sampling
        # This prevents sampling from corrupted data ranges that exceed medical safety
        safe_action_dims = self.safe_max_indices
        if safe_action_dims is None or len(safe_action_dims) != len(self.action_dims):
            # Fallback: detect task and apply appropriate limits
            task_type = self._detect_task_type(self.action_dims)
            if task_type == "rrt":
                safe_action_dims = [4, 4, 4, 1]  # Conservative RRT limits: indices 0-3, 0-3, 0-3, 0-1
            elif task_type == "vent":
                safe_action_dims = [6, 5, 5]    # Conservative VENT limits
            elif task_type == "iv":
                safe_action_dims = [3, 3]       # Conservative IV limits
            else:
                # Use reduced limits for unknown tasks
                safe_action_dims = [min(dim - 1, 4) for dim in self.action_dims]
            
            logger.warning(f"🔒 Using medical safety limits for {task_type} task: {safe_action_dims}")
        
        # Convert safety limits to actual action dimensions (add 1 for inclusive range)
        safe_dims = [limit + 1 for limit in safe_action_dims]

        # Helper to draw uniform random actions, respecting medical safety constraints and data distribution.
        def _uniform_sample() -> torch.Tensor:
            branches = []
            for branch_idx, safe_dim in enumerate(safe_dims):
                # Get valid action values for this branch (addresses data sparsity)
                if branch_idx < len(self.valid_action_values):
                    valid_actions = self.valid_action_values[branch_idx]
                    # Further restrict by safety limits
                    valid_actions = [a for a in valid_actions if a <= safe_dim]
                else:
                    # Fallback to range-based sampling
                    valid_actions = list(range(min(safe_dim + 1, self.action_dims[branch_idx])))
                
                if len(valid_actions) == 0:
                    logger.warning(f"⚠️ No valid actions for head {branch_idx}, using fallback")
                    valid_actions = [0]  # Fallback to prevent errors
                
                # Apply action support masks if available
                if self.action_support_masks is not None and branch_idx < len(self.action_support_masks):
                    mask = self.action_support_masks[branch_idx]
                    valid_actions = [a for a in valid_actions if a < len(mask) and mask[a]]
                    if len(valid_actions) == 0:
                        valid_actions = [0]  # Fallback
                
                # Sample from valid actions only
                valid_tensor = torch.tensor(valid_actions, device=self.device)
                random_indices = torch.randint(0, len(valid_actions), (batch_size, n_samples), device=self.device)
                sampled_actions = valid_tensor[random_indices]
                branches.append(sampled_actions)
                
                logger.debug(f"🎯 Head {branch_idx}: Sampling from valid actions {valid_actions}")
                
            return torch.stack(branches, dim=2)  # (B, n, H)

        if mode == 'uniform':
            return _uniform_sample()

        # ------------------------------------------------------------------
        # Policy-based sampling (greedy) – shared by 'policy' & 'mixed'
        # ------------------------------------------------------------------
        with torch.no_grad():
            # For policy-based sampling, we need to ensure greedy actions are also safe
            policy_actions = self.q_net.greedy_action(state).unsqueeze(1)  # (B,1,H)
            
            # Apply valid action constraints to policy actions (not just safety limits)
            for i, safe_dim in enumerate(safe_dims):
                # Get valid actions for this head, considering both data distribution and safety
                if i < len(self.valid_action_values):
                    valid_actions = self.valid_action_values[i]
                    # Further restrict by safety limits
                    valid_actions = [a for a in valid_actions if a <= safe_dim]
                else:
                    # Fallback to range-based constraints
                    actual_dim = min(safe_dim, self.action_dims[i])
                    valid_actions = list(range(actual_dim))
                
                if len(valid_actions) == 0:
                    logger.warning(f"⚠️ No valid policy actions for head {i}, using fallback")
                    valid_actions = [0]
                
                # Map policy actions to nearest valid actions
                head_actions = policy_actions[:, :, i]
                valid_tensor = torch.tensor(valid_actions, device=self.device)
                
                # For each action, find the nearest valid action
                corrected_actions = head_actions.clone()
                for batch_idx in range(head_actions.shape[0]):
                    for sample_idx in range(head_actions.shape[1]):
                        current_action = int(head_actions[batch_idx, sample_idx].item())
                        if current_action not in valid_actions:
                            # Find nearest valid action
                            nearest_valid = min(valid_actions, key=lambda x: abs(x - current_action))
                            corrected_actions[batch_idx, sample_idx] = nearest_valid
                
                policy_actions[:, :, i] = corrected_actions
                logger.debug(f"🎯 Policy Head {i}: Constrained to valid actions {valid_actions}")
            
            policy_actions = policy_actions.expand(-1, n_samples, -1)  # (B,n,H)

        if mode == 'policy':
            return policy_actions

        if mode == 'mixed':
            # Blend according to self.mixed_ratio ∈ (0,1)
            n_uniform = int(round(n_samples * self.mixed_ratio))
            n_uniform = max(0, min(n_samples, n_uniform))
            n_policy = n_samples - n_uniform

            uniform_part = _uniform_sample()[:, :n_uniform]  # (B, n_u, H)
            policy_part = policy_actions[:, :n_policy]       # (B, n_p, H)

            mixed = torch.cat([uniform_part, policy_part],
                              dim=1)  # (B, n_samples, H)
            # Shuffle along the sample dimension to avoid positional bias.
            perm = torch.randperm(n_samples, device=self.device)
            mixed = mixed[:, perm]
            return mixed

        raise ValueError("Unknown sampling mode: " + str(mode))

    def update(self, batch: Dict[str, torch.Tensor], *, grad_scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
        """Updates the agent using a batch of experiences.

        Implements the BVE training algorithm with TD loss, CQL regularization,
        and adaptive alpha tuning. Now properly handles multi-step trajectory data.

        Args:
            batch: Dictionary containing experience tensors with keys:
                  'state', 'action', 'reward', 'next_state', 'done',
                  'behavior_action', 'mask', 'lengths'.
            grad_scaler: Optional GradScaler for AMP support.

        Returns:
            Total training loss value.
        """
        # Move batch to device and get basic batch info
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device).long()
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        behavior_action = batch.get('behavior_action', action).to(self.device).long()
        
        # Get mask and lengths for trajectory handling
        mask = batch.get('mask', None)
        lengths = batch.get('lengths', None)
        
        original_batch_size = state.size(0)
        
        # 🔧 CRITICAL FIX: Properly handle multi-step trajectory data
        if state.dim() == 3:  # Multi-step trajectory data: (B, T, D)
            seq_len = state.size(1)
            
            # Handle mask for valid timesteps
            if mask is not None:
                mask = mask.to(self.device)
                if mask.dim() == 2:  # (B, T)
                    pass  # Already correct shape
                elif mask.dim() == 1:  # (B,) - expand to (B, T)
                    mask = mask.unsqueeze(1).expand(original_batch_size, seq_len)
                else:
                    logger.warning(f"⚠️ Unexpected mask shape {mask.shape}, creating default mask")
                    mask = torch.ones(original_batch_size, seq_len, dtype=torch.bool, device=self.device)
            else:
                # Create default mask if not provided
                mask = torch.ones(original_batch_size, seq_len, dtype=torch.bool, device=self.device)
            
            # Handle lengths
            if lengths is not None:
                lengths = lengths.to(self.device)
            else:
                # Infer lengths from done flags or use max length
                if done.dim() == 2:  # (B, T)
                    # Find first done=True for each trajectory, or use full length
                    lengths = torch.full((original_batch_size,), seq_len, dtype=torch.long, device=self.device)
                    for i in range(original_batch_size):
                        done_positions = torch.where(done[i])[0]
                        if len(done_positions) > 0:
                            lengths[i] = min(done_positions[0].item() + 1, seq_len)
                else:
                    lengths = torch.full((original_batch_size,), seq_len, dtype=torch.long, device=self.device)
            
            # 🚀 TRAJECTORY FLATTENING: Convert (B, T, D) -> (B*T, D) for valid timesteps
            # This allows every valid timestep to contribute to training
            
            # Flatten trajectories while respecting mask
            flat_indices = []
            for b in range(original_batch_size):
                traj_len = lengths[b].item()
                for t in range(min(traj_len, seq_len)):
                    if mask[b, t]:
                        flat_indices.append((b, t))
            
            if len(flat_indices) == 0:
                logger.warning("⚠️ No valid timesteps found in batch, skipping update")
                return 0.0
            
            # Convert flat_indices to tensors for efficient indexing
            batch_indices = torch.tensor([idx[0] for idx in flat_indices], device=self.device)
            time_indices = torch.tensor([idx[1] for idx in flat_indices], device=self.device)
            
            # Extract valid samples
            state = state[batch_indices, time_indices]  # (N, D) where N = num_valid_timesteps
            action = action[batch_indices, time_indices]  # (N, n_heads)
            behavior_action = behavior_action[batch_indices, time_indices]  # (N, n_heads)
            
            # Handle rewards and done flags
            if reward.dim() == 2:  # (B, T)
                reward = reward[batch_indices, time_indices]  # (N,)
            elif reward.dim() == 1:  # (B,) - assume same reward for all timesteps
                reward = reward[batch_indices]  # (N,)
            
            if done.dim() == 2:  # (B, T)
                done = done[batch_indices, time_indices]  # (N,)
            elif done.dim() == 1:  # (B,)
                done = done[batch_indices]  # (N,)
                
            # Handle next_state - for each timestep, next_state is the next timestep in sequence
            # For last timestep in trajectory, use the same state (terminal state)
            next_state_indices = []
            for i, (b, t) in enumerate(flat_indices):
                if t + 1 < min(lengths[b].item(), seq_len) and mask[b, t + 1]:
                    next_state_indices.append((b, t + 1))
                else:
                    # Terminal state or last valid timestep
                    next_state_indices.append((b, t))
            
            next_batch_indices = torch.tensor([idx[0] for idx in next_state_indices], device=self.device)
            next_time_indices = torch.tensor([idx[1] for idx in next_state_indices], device=self.device)
            next_state = next_state[next_batch_indices, next_time_indices]  # (N, D)
            
            batch_size = len(flat_indices)  # Update batch_size to reflect flattened data
            
            logger.debug(f"🔧 Flattened trajectory data: {original_batch_size} trajs x {seq_len} steps -> {batch_size} samples")
            
        elif state.dim() == 2:  # Single-step data: (B, D) 
            batch_size = original_batch_size
            
            # Ensure all 2D data is consistent
            if action.dim() == 3:
                if action.size(1) == 1:
                    action = action.squeeze(1)  # (B, 1, n_heads) -> (B, n_heads)
                    behavior_action = behavior_action.squeeze(1)
                else:
                    logger.error(f"❌ Cannot handle action shape {action.shape} for 2D state")
                    return 0.0
            
            # Handle reward/done reshaping for consistency
            if reward.dim() > 1:
                reward = reward.squeeze()
            if done.dim() > 1:
                done = done.squeeze()
                
        else:
            raise ValueError(f"❌ Invalid state dimensions: {state.shape}")

        # 🔧 Enhanced action shape validation with trajectory context
        if action.dim() != 2 or action.size(1) != len(self.action_dims):
            error_msg = (
                f"❌ Action shape mismatch in BVE agent update: "
                f"Expected (batch_size, {len(self.action_dims)}), "
                f"got {tuple(action.shape)}. "
                f"Agent action_dims: {self.action_dims}"
            )
            logger.error(error_msg)
            logger.error(f"🔍 Diagnostic info:")
            logger.error(f"  • Original batch size: {original_batch_size}")
            logger.error(f"  • Processed batch size: {batch_size}")
            logger.error(f"  • State shape: {state.shape}")
            logger.error(f"  • Action shape: {action.shape}")
            logger.error(f"  • Expected action dims: {len(self.action_dims)}")
            return 0.0

        # Enhanced reward processing with bounds checking
        reward = self._center_rewards(reward, dim=0)
        reward = self._scale_rewards(reward)

        # Enhanced numerical stability check for rewards
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            nan_count = safe_item(torch.isnan(reward).sum())
            inf_count = safe_item(torch.isinf(reward).sum())
            logger.warning(
                f"⚠️ Invalid rewards detected: {nan_count} NaN, {inf_count} Inf - replacing with 0")
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle n-step TD targets
        if self.n_step > 1 and 'n_step_reward' in batch and 'n_step_state' in batch and 'n_step_done' in batch:
            next_state_n = batch['n_step_state'].to(self.device)
            reward_n = self._scale_rewards(batch['n_step_reward'].to(self.device))
            done_n = batch['n_step_done'].to(self.device)
            discount = self.gamma ** self.n_step
        else:
            next_state_n = next_state
            reward_n = reward
            done_n = done
            discount = self.gamma

        # Ensure all tensors have correct batch dimension
        assert state.size(0) == batch_size, f"State batch size mismatch: {state.size(0)} vs {batch_size}"
        assert action.size(0) == batch_size, f"Action batch size mismatch: {action.size(0)} vs {batch_size}"
        assert reward.size(0) == batch_size, f"Reward batch size mismatch: {reward.size(0)} vs {batch_size}"
        assert next_state.size(0) == batch_size, f"Next state batch size mismatch: {next_state.size(0)} vs {batch_size}"

        # Compute current Q-values
        q_current = self.q_net.q_value(state, action)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.q_net.greedy_action(next_state_n)

            # Double Q-learning: Use min of online and target network estimates
            q_next_target = self.target_q_net.q_value(next_state_n, next_actions)
            q_next_online = self.q_net.q_value(next_state_n, next_actions)
            q_next = torch.min(q_next_online, q_next_target)

            # Behavior consistency penalty
            behavior_penalty = self._compute_behavior_penalty(
                state, next_actions, behavior_action
            )

            # TD target with behavior regularization
            td_target = reward_n + discount * (1 - done_n.float()) * (
                q_next - self.lambda_reg * behavior_penalty
            )

        # TD loss
        td_loss = F.mse_loss(q_current, td_target)

        # CQL regularization
        cql_loss = self._compute_cql_loss(state, batch_size)

        # Adaptive alpha update
        current_alpha = self.log_alpha.detach().exp()

        # Aggregate CQL losses dynamically
        branch_losses = []
        for key, value in cql_loss.items():
            if key != 'final':
                branch_losses.append(value)

        # Combine losses
        loss_components = torch.stack([cql_loss['final']] + branch_losses)
        total_cql_loss = loss_components.mean()

        # Update alpha (Lagrange multiplier)
        alpha_loss = -self.log_alpha * (
            cql_loss['final'].detach() - self.cql_target_gap
        )

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
        self.alpha_optimizer.step()

        # Total loss
        total_loss = td_loss + current_alpha * total_cql_loss

        # Numerical stability safeguard
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error("❌ Total loss is NaN/Inf - skipping gradient update")
            return 0.0

        # Update Q-network
        self.optimizer.zero_grad()

        if grad_scaler is not None:
            grad_scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Clip gradients for numerical stability
        apply_gradient_clipping(self.q_net, max_norm=self.max_grad_norm)

        if grad_scaler is not None:
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        else:
            self.optimizer.step()

        # Update target network periodically
        if (self.training_step + 1) % self.target_update_freq == 0:
            self.update_target()

        # Store training statistics dynamically
        self.last_td_loss = td_loss.item()
        self.last_cql_loss_final = cql_loss['final'].item()

        # Store branch-specific CQL losses dynamically
        self.last_cql_branch_losses = {}
        for key, value in cql_loss.items():
            if key != 'final':
                self.last_cql_branch_losses[key] = value.item()

        # Maintain backward compatibility for existing code
        if 'q1' in cql_loss:
            self.last_cql_loss_q1 = cql_loss['q1'].item()
        if 'q2' in cql_loss:
            self.last_cql_loss_q2 = cql_loss['q2'].item()
        if 'q3' in cql_loss:
            self.last_cql_loss_q3 = cql_loss['q3'].item()

        self.last_cql_loss = safe_item(total_cql_loss)
        self.last_alpha = safe_item(current_alpha)

        self.increment_training_step()

        return safe_item(total_loss)

    def _compute_cql_loss(self, state: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Computes Conservative Q-Learning (CQL) loss with enhanced numerical stability.

        This method implements the core CQL regularization that encourages conservative
        Q-value estimates for out-of-distribution actions, which is crucial for safe
        medical decision making where overoptimistic value estimates could lead to
        harmful treatment recommendations.

        The CQL loss penalizes Q-values for actions not observed in the training data
        while preserving Q-values for actions that were actually taken by clinicians.
        This creates a conservative bias that prioritizes safety over potential gains.

        Mathematical Foundation:
            For each branch i, the CQL loss is computed as:
            L_CQL^i = log(∑_a exp(Q_i(s,a))) - Q_i(s,a_data)

            Total CQL Loss = (1/k) * ∑_i L_CQL^i
            where k is the number of action branches.

        Args:
            state: Current patient states, shape (batch_size, state_dim).
                  Contains normalized vital signs, lab values, and demographics.
            batch_size: Number of patient samples in the current batch.
                       Used for efficient tensor operations and memory management.

        Returns:
            Dictionary containing branch-specific and aggregated CQL losses:
            - 'q1', 'q2', 'q3', ...: Individual branch losses (if applicable)
            - 'final': Averaged CQL loss across all branches

        Raises:
            RuntimeError: If Q-network forward pass fails or produces invalid values.
            ValueError: If state tensor has incompatible dimensions.

        Note:
            The method uses different sampling strategies based on self.cql_sample_mode:
            - 'uniform': Sample actions uniformly from each dimension (baseline)
            - 'policy': Sample from current policy (higher variance, more targeted)
            - 'mixed': Combine uniform and policy sampling (recommended for medical RL)

            For medical applications, 'mixed' mode is recommended as it balances
            exploration of the action space with focus on clinically relevant actions.
        """
        # ========================================================================
        # Enhanced Input Validation and Preprocessing
        # ========================================================================

        if not hasattr(self, '_initialized') or not self._initialized:
            raise RuntimeError(
                "BVE agent not properly initialized. Call __init__ first.")

        if state.dim() != 2:
            raise ValueError(
                f"Expected 2D state tensor (batch_size, state_dim), got shape {state.shape}. "
                f"For sequence data, use only the current timestep."
            )

        if state.size(0) != batch_size:
            raise ValueError(
                f"State batch size ({state.size(0)}) doesn't match expected ({batch_size})"
            )

        if state.size(1) != self.state_dim:
            raise ValueError(
                f"State feature dimension ({state.size(1)}) doesn't match expected ({self.state_dim})"
            )

        # ========================================================================
        # Numerical Stability Checks for Medical Safety
        # ========================================================================

        # Check for NaN/Inf in input states (critical for medical applications)
        if torch.isnan(state).any():
            nan_count = torch.isnan(state).sum().item()
            logger.error(
                f"❌ NaN values detected in {nan_count} state features - this is unsafe for medical RL")
            raise RuntimeError(
                "Cannot compute CQL loss with NaN states - check data preprocessing")

        if torch.isinf(state).any():
            inf_count = torch.isinf(state).sum().item()
            logger.error(
                f"❌ Infinite values detected in {inf_count} state features")
            raise RuntimeError(
                "Cannot compute CQL loss with infinite states - check data normalization")

        # ========================================================================
        # Adaptive Action Sampling Strategy for CQL
        # ========================================================================

        try:
            # Sample actions based on configured strategy with enhanced error handling
            # Use the existing _sample_actions method that handles all sampling modes
            sampled_actions = self._sample_actions(
                state, self.cql_n_samples, mode=self.cql_sample_mode)
            
            logger.debug(
                f"CQL: Using {self.cql_sample_mode} sampling with {self.cql_n_samples} samples per state")

        except Exception as e:
            logger.error(f"❌ Action sampling failed: {e}")
            raise RuntimeError(f"CQL action sampling failed: {e}") from e

        # ========================================================================
        # Q-Value Computation with Enhanced Error Handling
        # ========================================================================

        try:
            # Compute Q-values for all sampled actions
            # Shape: (batch_size, n_samples, n_heads) → List[Tensor(batch_size, n_samples)]
            with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=True):
                q_values_list = []

                # Get medical safety limits and valid action values for validation
                safe_action_dims = self.safe_max_indices
                if safe_action_dims is None or len(safe_action_dims) != len(self.action_dims):
                    task_type = self._detect_task_type(self.action_dims)
                    if task_type == "rrt":
                        safe_action_dims = [4, 4, 4, 1]  # Conservative RRT limits
                    elif task_type == "vent":
                        safe_action_dims = [6, 5, 5]    # Conservative VENT limits
                    elif task_type == "iv":
                        safe_action_dims = [3, 3]       # Conservative IV limits
                    else:
                        safe_action_dims = [min(dim - 1, 4) for dim in self.action_dims]

                for head_idx in range(len(self.action_dims)):
                    # Extract actions for current head: (batch_size, n_samples)
                    head_actions = sampled_actions[:, :, head_idx]

                    # Validate action indices are within valid action values and medical safety bounds
                    max_action = safe_item(head_actions.max())
                    min_action = safe_item(head_actions.min())
                    
                    # Get valid action values for this head
                    if head_idx < len(self.valid_action_values):
                        valid_actions = set(self.valid_action_values[head_idx])
                        valid_min = min(self.valid_action_values[head_idx])
                        valid_max = max(self.valid_action_values[head_idx])
                    else:
                        # Fallback to range-based validation
                        safe_max = safe_action_dims[head_idx] if head_idx < len(safe_action_dims) else self.action_dims[head_idx] - 1
                        expected_max = min(safe_max, self.action_dims[head_idx] - 1)
                        valid_actions = set(range(expected_max + 1))
                        valid_min = 0
                        valid_max = expected_max
                    
                    # Check if all actions are valid
                    invalid_actions = []
                    unique_actions = torch.unique(head_actions).cpu().tolist()
                    for action_val in unique_actions:
                        if int(action_val) not in valid_actions:
                            invalid_actions.append(int(action_val))
                    
                    if invalid_actions or min_action < valid_min or max_action > valid_max:
                        logger.error(
                            f"❌ Invalid actions for head {head_idx}: "
                            f"range [{min_action}, {max_action}], valid actions: {sorted(valid_actions)}"
                        )
                        if invalid_actions:
                            logger.error(f"❌ Unseen action values: {invalid_actions} (not in training data)")
                        
                        # Apply intelligent clamping: map to nearest valid action
                        corrected_actions = head_actions.clone()
                        for invalid_val in invalid_actions:
                            mask = (head_actions == invalid_val)
                            # Find nearest valid action
                            valid_list = sorted(valid_actions)
                            nearest_valid = min(valid_list, key=lambda x: abs(x - invalid_val))
                            corrected_actions[mask] = nearest_valid
                            logger.debug(f"🔧 Mapped invalid action {invalid_val} to nearest valid {nearest_valid}")
                        
                        sampled_actions[:, :, head_idx] = corrected_actions  # Update the original tensor
                        logger.warning(f"⚠️ Corrected invalid actions for head {head_idx} to valid action set")

                    # Compute Q-values for this head
                    q_head = self.q_net.compute_branch_q_values(
                        state, head_actions, head_idx)

                    # Numerical stability check
                    if torch.isnan(q_head).any() or torch.isinf(q_head).any():
                        logger.error(
                            f"❌ Invalid Q-values from head {head_idx}")
                        raise RuntimeError(
                            f"Q-network produced NaN/Inf values for head {head_idx}")

                    q_values_list.append(q_head)

        except Exception as e:
            logger.error(f"❌ Q-value computation failed: {e}")
            raise RuntimeError(
                f"Failed to compute Q-values for CQL: {e}") from e

        # ========================================================================
        # Branch-wise CQL Loss Computation with Medical-Specific Normalization
        # ========================================================================

        cql_losses = {}
        all_branch_losses = []

        try:
            for head_idx, q_head in enumerate(q_values_list):
                # Apply branch normalization if enabled (recommended for medical RL)
                if self.normalize_branch:
                    q_head = self._normalize_q_values(q_head)

                # Compute logsumexp with numerical stability
                # Use temperature scaling for better numerical properties
                q_scaled = q_head / self.softmax_temperature
                logsumexp_q = torch.logsumexp(
                    q_scaled, dim=1) * self.softmax_temperature

                # CQL loss for this branch: encourage conservatism
                branch_cql_loss = logsumexp_q.mean()

                # Store with dynamic naming for flexibility
                cql_losses[f'q{head_idx+1}'] = branch_cql_loss
                all_branch_losses.append(branch_cql_loss)

                # Enhanced logging for medical safety monitoring
                if head_idx == 0:  # Log details for first branch only to avoid spam
                    q_mean = safe_item(q_head.mean())
                    q_std = safe_item(q_head.std())
                    logger.debug(
                        f"CQL Head {head_idx}: Q_mean={q_mean:.3f}, Q_std={q_std:.3f}, "
                        f"logsumexp={safe_item(logsumexp_q.mean()):.3f}"
                    )

        except Exception as e:
            logger.error(f"❌ Branch CQL loss computation failed: {e}")
            raise RuntimeError(
                f"Failed to compute branch-wise CQL losses: {e}") from e

        # ========================================================================
        # Final Aggregation with Enhanced Stability
        # ========================================================================

        try:
            # Compute final aggregated CQL loss with numerical stability
            if all_branch_losses:
                # Use mean reduction to ensure consistent scaling across different numbers of heads
                final_cql_loss = torch.stack(all_branch_losses).mean()

                # Additional numerical stability check
                if torch.isnan(final_cql_loss) or torch.isinf(final_cql_loss):
                    logger.error(
                        "❌ Final CQL loss is NaN/Inf after aggregation")
                    # Fallback to conservative estimate
                    final_cql_loss = torch.tensor(
                        0.0, device=self.device, requires_grad=True)
                    logger.warning(
                        "⚠️  Using fallback CQL loss = 0.0 for numerical stability")

                cql_losses['final'] = final_cql_loss

                # Enhanced monitoring for medical applications
                if self.training_step % 100 == 0:  # Log every 100 steps
                    logger.debug(
                        f"🎯 CQL Summary (step {self.training_step}): "
                        f"final_loss={safe_item(final_cql_loss):.4f}, "
                        f"n_branches={len(all_branch_losses)}, "
                        f"sample_mode={self.cql_sample_mode}, "
                        f"n_samples={self.cql_n_samples}"
                    )
            else:
                # This should never happen, but handle gracefully
                logger.error(
                    "❌ No branch losses computed - this indicates a serious error")
                cql_losses['final'] = torch.tensor(
                    0.0, device=self.device, requires_grad=True)

        except Exception as e:
            logger.error(f"❌ CQL final aggregation failed: {e}")
            # Fallback for numerical stability
            cql_losses['final'] = torch.tensor(
                0.0, device=self.device, requires_grad=True)
            logger.warning("⚠️  Using fallback final CQL loss for stability")

        # ========================================================================
        # Validation and Return
        # ========================================================================

        # Final validation of all returned losses
        for loss_name, loss_value in cql_losses.items():
            if not torch.is_tensor(loss_value):
                raise RuntimeError(f"CQL loss '{loss_name}' is not a tensor")
            if not loss_value.requires_grad:
                logger.warning(
                    f"⚠️  CQL loss '{loss_name}' doesn't require gradients")

        logger.debug(
            f"✅ CQL computation completed: {len(cql_losses)} losses computed")
        return cql_losses

    def act(
        self,
        state: torch.Tensor,
        greedy: bool = True,
        policy_inference_mode: Optional[str] = None,
        beam_width: int = 5,
        **kwargs
    ) -> torch.Tensor:
        """Selects actions for given states.

        Args:
            state: State tensor of shape (batch_size, state_dim).
            greedy: Whether to use greedy action selection (ignored if mode specified).
            policy_inference_mode: Inference mode ('greedy' or 'beam').
            beam_width: Beam width for beam search inference.
            **kwargs: Additional arguments (ignored).

        Returns:
            Selected actions of shape (batch_size, 3) for greedy mode or
            (batch_size, beam_width, 3) for beam mode.
        """
        state = state.to(self.device)

        # Default fallback: greedy unless user explicitly requests stochastic behaviour
        mode = policy_inference_mode or ('greedy' if greedy else 'sample')

        with torch.no_grad():
            if mode == 'beam':
                actions, _ = self.q_net.beam_search(
                    state, beam_width=beam_width)
                return self._clip_actions(actions)
            elif mode == 'sample':
                # Stochastic hierarchical sampling using branch-wise softmax.
                q_values = self.q_net.forward(state)  # logits per branch

                batch_size = state.size(0)
                batch_idx = torch.arange(batch_size, device=self.device)
                actions = []

                # Sample actions hierarchically through all branches
                current_idx = batch_idx
                for i, q in enumerate(q_values):
                    if i == 0:
                        # First branch: directly sample from q1
                        dist = torch.distributions.Categorical(logits=q)
                        action = dist.sample()
                        actions.append(action)
                        current_idx = (current_idx, action)
                    else:
                        # Subsequent branches: index with previously sampled actions
                        q_selected = q[current_idx]
                        dist = torch.distributions.Categorical(
                            logits=q_selected)
                        action = dist.sample()
                        actions.append(action)
                        current_idx = current_idx + (action,)

                return self._clip_actions(torch.stack(actions, dim=1))
            else:
                actions = self.q_net.greedy_action(state)
                return self._clip_actions(actions)

    # ------------------------------------------------------------------
    #  Safety guard – clip actions to valid bins (0 … dim-1)
    # ------------------------------------------------------------------
    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Clamps discrete actions to medically safe ranges with enhanced task detection.

        This method now automatically detects the medical task and applies appropriate
        safety constraints based on clinical guidelines. It prevents actions that could
        be harmful or unrealistic in medical practice.

        Medical Safety Guidelines:
        -------------------------
        RRT Task - Renal Replacement Therapy:
        * RRT Type: Conservative range [0-3] - Avoid experimental modalities
        * RRT Dose: Conservative range [0-3] - Prevent overdialysis complications  
        * Blood Flow: Conservative range [0-3] - Minimize hemodynamic stress
        * Anticoagulation: Safe range [0-1] - Binary choice, both clinically acceptable

        VENT Task - Mechanical Ventilation:
        * PEEP: Full range [0-6] - Conservative approach allows flexibility
        * FiO₂: Exclude highest [0-5] - Prevent hyperoxia (FiO₂ >80%)
        * Tidal Volume: Exclude highest [0-5] - Lung-protective ventilation (<8ml/kg)

        IV Task - IV Fluids & Vasopressors:
        * IV Fluids: Conservative range [0-3] - Prevent fluid overload
        * Vasopressor: Conservative range [0-3] - Avoid excessive vasoconstriction

        Users can override ``self.safe_max_indices`` at construction time.
        """
        # Get medical safety limits with task-specific detection
        safe_max_indices = self.safe_max_indices
        if safe_max_indices is None or len(safe_max_indices) != len(self.action_dims):
            # Auto-detect task and apply appropriate medical safety limits
            task_type = self._detect_task_type(self.action_dims)
            if task_type == "rrt":
                safe_max_indices = [3, 3, 3, 1]  # Conservative RRT limits  
                logger.debug("🔒 Applying RRT medical safety limits: [3, 3, 3, 1]")
            elif task_type == "vent":
                safe_max_indices = [6, 5, 5]    # Conservative VENT limits
                logger.debug("🔒 Applying VENT medical safety limits: [6, 5, 5]")
            elif task_type == "iv":
                safe_max_indices = [3, 3]       # Conservative IV limits
                logger.debug("🔒 Applying IV medical safety limits: [3, 3]")
            else:
                # Generic fallback: conservative limits
                safe_max_indices = [min(dim - 1, 4) for dim in self.action_dims]
                logger.warning(f"⚠️ Unknown task {task_type}, using conservative limits: {safe_max_indices}")

        # Apply safety clipping with enhanced logging
        original_actions = actions.clone()
        
        if actions.dim() == 2:  # (B, H) - Standard batch format
            for i in range(min(len(self.action_dims), len(safe_max_indices))):
                # Clamp to medical safety limits 
                max_safe = min(safe_max_indices[i], self.action_dims[i] - 1)
                actions[:, i] = actions[:, i].clamp(0, max_safe)
                
        elif actions.dim() == 3:  # (B, K, H) - Beam search format  
            for i in range(min(len(self.action_dims), len(safe_max_indices))):
                # Clamp to medical safety limits
                max_safe = min(safe_max_indices[i], self.action_dims[i] - 1)
                actions[..., i] = actions[..., i].clamp(0, max_safe)
        
        # Log if any actions were clipped for medical safety
        if not torch.equal(original_actions, actions):
            clipped_heads = []
            for i in range(min(len(self.action_dims), len(safe_max_indices))):
                if actions.dim() == 2:
                    if not torch.equal(original_actions[:, i], actions[:, i]):
                        clipped_heads.append(i)
                elif actions.dim() == 3:
                    if not torch.equal(original_actions[..., i], actions[..., i]):
                        clipped_heads.append(i)
            
            if clipped_heads:
                logger.debug(f"🔒 Applied medical safety clipping to heads: {clipped_heads}")
        
        return actions

    def update_target(self) -> None:
        """Polyak (τ) averaging update of target network parameters."""
        tau = float(getattr(self, "polyak_tau", 1.0))
        with torch.no_grad():
            for tgt, src in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                tgt.data.mul_(1 - tau)
                tgt.data.add_(tau * src.data)

    def set_training_mode(self, training: bool = True) -> None:
        """Sets the training mode for the agent's networks."""
        self.q_net.train(training)
        self.target_q_net.train(training)

    def save_checkpoint(self, filepath: str) -> None:
        """Saves agent state to checkpoint file.

        Args:
            filepath: Path to save the checkpoint.
        """
        checkpoint = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': {
                'state_dim': self.state_dim,
                'action_dims': self.action_dims,
                'lr': self.lr,
                'gamma': self.gamma,
                'lambda_reg': self.lambda_reg,
                'alpha': self.alpha,
                'cql_n_samples': self.cql_n_samples,
                'normalize_branch': self.normalize_branch,
                'cql_sample_mode': self.cql_sample_mode,
                'cql_target_gap': self.cql_target_gap,
                'alpha_lr': self.alpha_lr,
                'target_update_freq': self.target_update_freq,
                'max_grad_norm': self.max_grad_norm,
                'behavior_penalty_mode': self.behavior_penalty_mode,
                'softmax_temperature': self.softmax_temperature,
                'n_step': self.n_step,
                'polyak_tau': self.polyak_tau,
                'mixed_ratio': self.mixed_ratio,
            }
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Loads agent state from checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(
            checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(
            checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self._training_step = checkpoint['training_step']
        self._episode_count = checkpoint['episode_count']

    def get_training_stats(self) -> Dict[str, float]:
        """Returns current training statistics."""
        stats = super().get_training_stats()
        stats.update({
            'td_loss': self.last_td_loss,
            'cql_loss': self.last_cql_loss,
            'cql_loss_final': self.last_cql_loss_final,
            'alpha': self.last_alpha,
            'lr': self.optimizer.param_groups[0]['lr'],
            'alpha_lr': self.alpha_optimizer.param_groups[0]['lr']
        })

        # Add branch-specific CQL losses dynamically
        if hasattr(self, 'last_cql_branch_losses'):
            for key, value in self.last_cql_branch_losses.items():
                stats[f'cql_loss_{key}'] = value

        # Maintain backward compatibility for existing code
        if hasattr(self, 'last_cql_loss_q1'):
            stats['cql_loss_q1'] = self.last_cql_loss_q1
        if hasattr(self, 'last_cql_loss_q2'):
            stats['cql_loss_q2'] = self.last_cql_loss_q2
        if hasattr(self, 'last_cql_loss_q3'):
            stats['cql_loss_q3'] = self.last_cql_loss_q3

        return stats

    # ------------------------------------------------------------------
    # Penalty helpers
    # ------------------------------------------------------------------

    def _compute_behavior_penalty(
        self,
        state: torch.Tensor,
        next_actions: torch.Tensor,
        behavior_action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute behaviour regularisation term.

        Modes
        -----
        count (default)
            Simple mismatch count across branches (0‥3).
        log_prob / kl
            Negative log-probability of behaviour action under current
            policy; equivalent to one-hot → policy KL divergence.
        """

        if self.behavior_penalty_mode == 'count':
            return (next_actions != behavior_action).float().sum(dim=1)

        # ---- probability-based penalty ----
        # Obtain branch Q-values once to avoid redundant forward passes
        with torch.no_grad():
            q_values = self.q_net.forward(state)

        batch_size = state.size(0)
        batch_idx = torch.arange(batch_size, device=self.device)

        # Compute log probabilities for each branch dynamically
        total_log_prob = 0.0
        current_idx = batch_idx

        for i, q in enumerate(q_values):
            # Extract action for this branch
            action = behavior_action[:, i]

            if i == 0:
                # First branch: direct log-softmax
                logp = F.log_softmax(
                    q / self.softmax_temperature, dim=1)[batch_idx, action]
                current_idx = (batch_idx, action)
            else:
                # Subsequent branches: select with previous actions
                q_selected = q[current_idx]
                logp = F.log_softmax(
                    q_selected / self.softmax_temperature, dim=1)[batch_idx, action]
                current_idx = current_idx + (action,)

            total_log_prob += logp

        neg_log_prob = -total_log_prob  # (B,)
        return neg_log_prob  # KL == cross-entropy for one-hot

    # ------------------------------------------------------------------
    #  Public helper to update masks after agent initialisation
    # ------------------------------------------------------------------
    def set_action_support_masks(self, masks: List[torch.Tensor]) -> None:
        """Register/override behaviour-policy action support masks.

        Args:
            masks: List[torch.Tensor] – Boolean vectors per action branch.
        """
        if len(masks) != len(self.action_dims):
            raise ValueError("Mask length mismatch")
        self.action_support_masks = [m.to(self.device).bool() for m in masks]

    def _configure_valid_action_values(self, action_dims: List[int]) -> List[List[int]]:
        """Configure valid action values based on task type and actual data distribution.
        
        This method addresses the data sparsity issue where some action values
        are missing from the training data, causing Q-network instability when
        sampling unseen actions.
        
        Args:
            action_dims: List of action space dimensions
            
        Returns:
            List of valid action values for each dimension
            
        Medical Data Quality Issues:
        ----------------------------
        RRT Task - Known data sparsity:
        * RRT Type (Head 0): Only values [1, 2, 4] exist, missing [0, 3]
        * Other heads: Full coverage [0, 1, 2, 3, 4] or [0, 1]
        
        This prevents sampling from unseen action combinations that cause
        Q-network instability and extreme value generation.
        """
        task_type = self._detect_task_type(action_dims)
        
        if task_type == "rrt":
            # Based on actual RRT data analysis from trajectory_rrt.csv
            valid_values = [
                [1, 2, 4],        # Head 0 (rrt_type_bin): Missing 0, 3
                [0, 1, 2, 3, 4],  # Head 1 (rrt_dose_bin): Full coverage
                [0, 1, 2, 3, 4],  # Head 2 (blood_flow_bin): Full coverage  
                [0, 1]            # Head 3 (anticoagulation_bin): Full coverage
            ]
            logger.debug("🔍 RRT valid action values configured based on actual data distribution")
            
        elif task_type == "vent":
            # Assume full coverage for VENT task (can be updated if needed)
            valid_values = [
                list(range(action_dims[0])),  # Head 0: 0-6
                list(range(action_dims[1])),  # Head 1: 0-6  
                list(range(action_dims[2]))   # Head 2: 0-6
            ]
            logger.debug("🔍 VENT valid action values: full coverage assumed")
            
        elif task_type == "iv":
            # Assume full coverage for IV task (can be updated if needed)
            valid_values = [
                list(range(action_dims[0])),  # Head 0: 0-4
                list(range(action_dims[1]))   # Head 1: 0-4
            ]
            logger.debug("🔍 IV valid action values: full coverage assumed")
            
        else:
            # Generic fallback: assume full coverage
            valid_values = [list(range(dim)) for dim in action_dims]
            logger.warning(f"⚠️ Unknown task {task_type}, assuming full action coverage")
            
        return valid_values
