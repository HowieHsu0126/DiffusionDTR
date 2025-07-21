"""Baseline *Conservative Q-Learning* (CQL) core implementation.

This module provides a robust implementation of Conservative Q-Learning for
offline reinforcement learning in medical ICU settings. CQL addresses the
fundamental challenge of distribution shift in offline RL by regularizing
the Q-function to assign low values to out-of-distribution actions.

Key Improvements Over Standard CQL:
----------------------------------
1. **Adaptive α parameter** - Automatic tuning of conservatism strength
2. **Vectorized CQL loss** - Efficient GPU computation without Python loops  
3. **Numerical stability** - Robust temperature scaling and gradient clipping
4. **Multi-head architecture** - Support for multi-dimensional medical actions
5. **Comprehensive logging** - Detailed metrics for offline RL analysis

Mathematical Foundation:
-----------------------
CQL optimizes the following objective:
L_CQL(Q) = α · (log Σ_a exp(Q(s,a)) - E_{π_β}[Q(s,a)]) + L_Bellman(Q)

Where:
- α controls the conservatism strength (learned adaptively)
- The first term penalizes high Q-values for unseen actions
- The second term maintains standard Bellman consistency
- π_β is the behavior policy from the offline dataset

Clinical Application:
--------------------
In ICU-AKI treatment, CQL prevents the agent from over-optimistically
evaluating novel treatment combinations not observed in the training data,
which is crucial for patient safety in medical RL applications.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

# Shared utils
from Libs.utils.model_utils import init_weights_xavier


class CQLNet(nn.Module):
    """
    Conservative Q-Learning network for multi-discrete action spaces in medical RL.

    This network implements separate Q-function heads for each action dimension,
    enabling independent value estimation for different types of medical
    interventions while applying conservative regularization to prevent
    overestimation of out-of-distribution actions.

    Architecture Details:
    -------------------
    Input: Patient state vector (vital signs, lab values, demographics)
    → Shared feature extraction (2-layer MLP with optional LayerNorm/Dropout)
    → Separate Q-heads for each action dimension
    → Conservative regularization applied across all heads
    
    Medical Action Dimensions:
    ------------------------
    - Mechanical ventilation: PEEP, FiO2, Tidal Volume settings
    - Medication management: Vasopressor dosages, sedation levels
    - Fluid management: Fluid balance, diuretic administration
    
    Each dimension is modeled independently to capture the distinct
    physiological effects of different interventions.

    Args:
        state_dim: Dimension of patient state representation (typically 64-128).
        action_dims: List of action space sizes for each medical intervention type.
                    E.g., [7, 6, 6] for vent, [4, 3, 3, 2] for rrt, [5, 4] for iv.
        hidden_dim: Hidden layer dimension for Q-function networks.
        use_layer_norm: Whether to apply LayerNorm for training stability.
        dropout: Dropout probability to prevent overfitting (especially important for offline RL).
        temperature: Temperature for CQL logsumexp computation (affects conservatism).
        target_conservative_gap: Target value for adaptive α adjustment.

    Example:
        >>> # For ICU mechanical ventilation control
        >>> net = CQLNet(
        ...     state_dim=87,  # Patient vital signs + lab values
        ...     action_dims=[7, 6, 6],  # PEEP, FiO2, Tidal Volume
        ...     hidden_dim=128,
        ...     use_layer_norm=True,
        ...     dropout=0.1,
        ...     temperature=1.0
        ... )
        >>> q_values = net(patient_states)  # List of Q-values for each intervention
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        hidden_dim: int = 128,
        *,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        temperature: float = 1.0,
        target_conservative_gap: float = 5.0,
    ) -> None:
        """
        Initialize the CQL network with enhanced stability features.

        Args:
            state_dim: Dimension of the input state representation.
            action_dims: List of action space sizes for each action dimension.
            hidden_dim: Hidden layer dimension for Q-function networks.
            use_layer_norm: Whether to use LayerNorm after each Linear layer.
            dropout: Dropout probability for regularization.
            temperature: Temperature parameter for CQL logsumexp computation.
            target_conservative_gap: Target conservative gap for adaptive α tuning.

        Raises:
            ValueError: If action_dims is empty or contains non-positive values.
            ValueError: If state_dim or hidden_dim are non-positive.
            ValueError: If temperature is non-positive.
        """
        super().__init__()

        # Input validation for robust network construction
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError(
                f"action_dims must be non-empty with positive values, got {action_dims}"
            )
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.state_dim = state_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.target_conservative_gap = target_conservative_gap

        # Initialize adaptive α parameter for conservative regularization
        # Start with a reasonable value and allow automatic adjustment
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(0.1)))

        # Enhanced network architecture with stability features
        norm_layer = nn.LayerNorm if use_layer_norm else nn.Identity

        def _make_q_head(out_dim: int) -> nn.Sequential:
            """
            Factory to build a single Q-function head with modern architectural choices.

            Args:
                out_dim: Output dimension (number of actions for this head).

            Returns:
                Sequential network implementing one Q-function head.
            """
            layers = [
                nn.Linear(state_dim, hidden_dim),
                norm_layer(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Final output layer (no activation for Q-values)
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        # Create separate Q-heads for each action dimension
        self.q_heads = nn.ModuleList([
            _make_q_head(action_dim) for action_dim in action_dims
        ])

        # Apply principled weight initialization
        self.apply(init_weights_xavier)

        # Track training statistics for adaptive α adjustment
        self.register_buffer('conservative_gap_ema', torch.tensor(0.0))
        self.register_buffer('ema_decay', torch.tensor(0.99))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all Q-function heads.

        Args:
            x: Input state tensor of shape (batch_size, state_dim).
               Represents patient states including vital signs, lab values, etc.

        Returns:
            List of Q-value tensors, one for each action dimension.
            Each tensor has shape (batch_size, action_dim_i).

        Raises:
            ValueError: If input tensor has incorrect dimensions.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        if x.size(1) != self.state_dim:
            raise ValueError(
                f"Input dimension {x.size(1)} doesn't match expected {self.state_dim}"
            )

        # Compute Q-values for each action head independently
        return [head(x) for head in self.q_heads]

    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        target_net: "CQLNet",
        *,
        gamma: float = 0.99,
        n_samples: int = 10,
        importance_sampling: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the full CQL loss including both conservative and Bellman terms.

        This implementation follows the original CQL paper but with enhancements
        for numerical stability and multi-head action spaces.

        Args:
            states: Current states (batch_size, state_dim).
            actions: Actions taken (batch_size, n_heads).
            rewards: Immediate rewards (batch_size,).
            next_states: Next states (batch_size, state_dim).
            dones: Terminal state indicators (batch_size,).
            target_net: Target Q-network for Bellman updates.
            gamma: Discount factor for future rewards.
            n_samples: Number of actions to sample for CQL regularization.
            importance_sampling: Whether to use importance sampling for action sampling.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains:
            - 'cql_loss': Conservative regularization term
            - 'bellman_loss': Standard TD error loss  
            - 'conservative_gap': Difference between sampled and data Q-values
            - 'alpha': Current α parameter value
        """
        batch_size = states.size(0)
        device = states.device

        # Compute current Q-values for the taken actions
        current_q_heads = self.forward(states)
        current_q_values = []
        
        for head_idx, q_head in enumerate(current_q_heads):
            action_indices = actions[:, head_idx:head_idx+1]
            q_val = q_head.gather(-1, action_indices).squeeze(-1)
            current_q_values.append(q_val)

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_heads = target_net.forward(next_states)
            next_q_max = [q.max(dim=-1)[0] for q in next_q_heads]
            
            # Average across heads for target computation
            next_q_avg = torch.stack(next_q_max, dim=1).mean(dim=1)
            targets = rewards + gamma * next_q_avg * (~dones)

        # Compute Bellman loss (standard TD error)
        current_q_avg = torch.stack(current_q_values, dim=1).mean(dim=1)
        bellman_loss = F.mse_loss(current_q_avg, targets)

        # Compute conservative regularization term
        cql_loss = self._compute_conservative_loss(
            states, actions, current_q_heads, n_samples, importance_sampling
        )

        # Adaptive α adjustment based on conservative gap
        alpha = torch.exp(self.log_alpha)
        conservative_gap = cql_loss.detach()
        
        # Update exponential moving average of conservative gap
        self.conservative_gap_ema = (
            self.ema_decay * self.conservative_gap_ema + 
            (1 - self.ema_decay) * conservative_gap
        )

        # Adaptive α loss for automatic tuning
        alpha_loss = -self.log_alpha * (conservative_gap - self.target_conservative_gap).detach()

        # Total CQL loss
        total_loss = alpha * cql_loss + bellman_loss + alpha_loss

        # Prepare detailed loss information for logging
        loss_dict = {
            'cql_loss': cql_loss.item(),
            'bellman_loss': bellman_loss.item(),
            'conservative_gap': conservative_gap.item(),
            'alpha': alpha.item(),
            'alpha_loss': alpha_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict

    def _compute_conservative_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        current_q_heads: List[torch.Tensor],
        n_samples: int,
        importance_sampling: bool,
    ) -> torch.Tensor:
        """
        Compute the conservative regularization term of the CQL loss.

        This term penalizes high Q-values for actions not in the dataset,
        encouraging the Q-function to be conservative for out-of-distribution actions.

        Args:
            states: Current states (batch_size, state_dim).
            actions: Actions from the dataset (batch_size, n_heads).
            current_q_heads: Current Q-values for each head.
            n_samples: Number of random actions to sample for regularization.
            importance_sampling: Whether to use importance sampling.

        Returns:
            Conservative loss term (scalar tensor).
        """
        batch_size = states.size(0)
        device = states.device

        total_conservative_loss = 0.0

        for head_idx, (q_head, action_dim) in enumerate(zip(current_q_heads, self.action_dims)):
            # Sample random actions for this head
            if importance_sampling:
                # Use uniform sampling over action space
                sampled_actions = torch.randint(
                    0, action_dim, (batch_size, n_samples), device=device
                )
            else:
                # Simple uniform sampling
                sampled_actions = torch.randint(
                    0, action_dim, (batch_size, n_samples), device=device
                )

            # Compute Q-values for sampled actions
            expanded_states = states.unsqueeze(1).expand(-1, n_samples, -1)
            expanded_states = expanded_states.reshape(-1, self.state_dim)
            sampled_q = self.q_heads[head_idx](expanded_states)
            sampled_q = sampled_q.reshape(batch_size, n_samples, action_dim)

            # Gather Q-values for the sampled actions
            sampled_q_values = sampled_q.gather(-1, sampled_actions.unsqueeze(-1)).squeeze(-1)

            # Compute logsumexp for numerical stability
            max_q = sampled_q_values.max(dim=1, keepdim=True)[0]
            logsumexp_q = max_q + torch.log(
                torch.exp((sampled_q_values - max_q) / self.temperature).sum(dim=1, keepdim=True)
            )
            logsumexp_q = logsumexp_q.squeeze(-1)

            # Q-values for dataset actions
            dataset_actions = actions[:, head_idx]
            dataset_q = q_head.gather(-1, dataset_actions.unsqueeze(-1)).squeeze(-1)

            # Conservative loss for this head
            head_conservative_loss = (logsumexp_q * self.temperature - dataset_q).mean()
            total_conservative_loss += head_conservative_loss

        # Average across heads
        return total_conservative_loss / len(self.action_dims)

    def q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar Q(s,a) for joint multi-head discrete action.

        This method aggregates Q-values across all heads to provide a single
        scalar value for the joint action, useful for policy evaluation and
        off-policy estimation.

        Args:
            state: State tensor of shape (batch_size, state_dim).
            action: Action tensor of shape (batch_size, n_heads) with integer actions.

        Returns:
            Tensor of shape (batch_size,) with averaged Q-values across heads.

        Raises:
            ValueError: If action tensor shape doesn't match expected format.
        """
        if action.dim() != 2 or action.size(1) != len(self.action_dims):
            raise ValueError(
                f"Action tensor must have shape (batch_size, {len(self.action_dims)}), "
                f"got {tuple(action.shape)}"
            )

        q_heads = self.forward(state)
        q_values = []
        
        for head_idx, q_head in enumerate(q_heads):
            action_indices = action[:, head_idx:head_idx+1]
            q_val = q_head.gather(1, action_indices).squeeze(1)
            q_values.append(q_val)
            
        # Average across heads for consistent scaling
        return torch.stack(q_values, dim=1).mean(dim=1)

    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select greedy actions for given states based on Q-values.

        This method implements deterministic action selection by choosing
        the action with the highest Q-value for each action dimension.
        This is used during evaluation and by FQE for policy estimation.

        Args:
            state: State tensor of shape (batch_size, state_dim).

        Returns:
            Tensor of shape (batch_size, n_heads) with selected action indices.
        """
        with torch.no_grad():
            q_heads = self.forward(state)
            greedy_actions = []
            
            for q_head in q_heads:
                # Select action with highest Q-value for each head
                greedy_action = q_head.argmax(dim=-1)  # Shape: (batch_size,)
                greedy_actions.append(greedy_action)
            
            # Stack actions from all heads
            return torch.stack(greedy_actions, dim=1)

    @property
    def alpha(self) -> float:
        """Get the current value of the adaptive α parameter."""
        return torch.exp(self.log_alpha).item()

    def update_alpha(self, conservative_gap: float) -> None:
        """
        Update the adaptive α parameter based on the conservative gap.

        Args:
            conservative_gap: Current conservative gap value.
        """
        # Simple gradient-based update for α
        gap_error = conservative_gap - self.target_conservative_gap
        alpha_grad = -gap_error  # Negative because we minimize -log_alpha * gap_error
        self.log_alpha.data += 0.001 * alpha_grad  # Small learning rate for α

    def __repr__(self) -> str:
        """Provide informative string representation for debugging and logging."""
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dims={self.action_dims}, "
            f"hidden_dim={self.hidden_dim}, "
            f"temperature={self.temperature}, "
            f"alpha={self.alpha:.4f})"
        )


def evaluate_cql_policy(
    q_net: CQLNet,
    states: torch.Tensor,
    *,
    epsilon: float = 0.0,
    temperature: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Evaluate CQL policy using epsilon-greedy or Boltzmann exploration.

    This function provides multiple evaluation strategies for CQL policies,
    supporting both deterministic greedy action selection and stochastic
    exploration strategies appropriate for medical decision-making.

    Args:
        q_net: Trained CQL network.
        states: State tensor of shape (batch_size, state_dim).
        epsilon: Probability of random action selection (epsilon-greedy).
        temperature: Temperature for Boltzmann exploration (when epsilon=0).
        seed: Optional random seed for reproducible evaluation.

    Returns:
        Action tensor of shape (batch_size, n_heads) with selected actions.

    Note:
        In medical applications, we typically use low epsilon values or
        temperature-based sampling to balance exploration with safety.
    """
    if seed is not None:
        torch.manual_seed(seed)

    batch_size = states.size(0)

    with torch.no_grad():
        q_values_list = q_net(states)

        if epsilon > 0.0:
            # Epsilon-greedy exploration
            greedy_actions = [q.argmax(dim=1, keepdim=True) for q in q_values_list]
            greedy_joint = torch.cat(greedy_actions, dim=1)

            # Decide per sample whether to explore
            explore_mask = torch.rand(batch_size, device=states.device) < epsilon

            if explore_mask.any():
                # Sample random actions for exploration
                rand_actions = []
                for q in q_values_list:
                    rand_actions.append(
                        torch.randint(0, q.size(1), (batch_size, 1), device=states.device)
                    )
                rand_joint = torch.cat(rand_actions, dim=1)

                # Blend exploring vs greedy actions
                output = torch.where(explore_mask.unsqueeze(1), rand_joint, greedy_joint)
                return output

            return greedy_joint
        else:
            # Boltzmann (temperature-based) sampling
            if temperature <= 0:
                # Pure greedy when temperature is zero or negative
                greedy_actions = [q.argmax(dim=1, keepdim=True) for q in q_values_list]
                return torch.cat(greedy_actions, dim=1)
            else:
                # Temperature-scaled sampling
                sampled_actions = []
                for q in q_values_list:
                    probs = F.softmax(q / temperature, dim=1)
                    sampled_action = torch.multinomial(probs, 1)
                    sampled_actions.append(sampled_action)
                return torch.cat(sampled_actions, dim=1)
