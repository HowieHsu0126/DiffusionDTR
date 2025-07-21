import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Optional, Sequence

# Reuse shared replay buffer implementation
from Libs.utils.model_utils import ReplayBuffer

class DQNNet(nn.Module):
    """
    Multi-head Deep Q-Network for multi-dimensional discrete action spaces in medical RL.
    
    This implementation supports both standard DQN and modern variants including:
    - Double DQN for reduced overestimation bias
    - Dueling DQN for improved value estimation
    - Separate Q-value heads for independent action dimensions
    
    The multi-head design is specifically tailored for ICU-AKI treatment where
    different action dimensions represent distinct medical interventions:
    - Mechanical ventilation settings (PEEP, FiO2, Tidal Volume)
    - Medication dosages
    - Fluid management protocols
    
    Architecture Design:
    -------------------
    Input: Patient state vector (vital signs, lab values, demographics)
    → Shared feature extraction layers (2-layer MLP with ReLU)
    → Separate Q-heads for each action dimension
    → Output: Q-values for each possible action in each dimension
    
    Mathematical Foundation:
    ----------------------
    For state s and multi-dimensional action a = (a₁, a₂, ..., aₖ):
    Q(s,a) = Σᵢ Qᵢ(s,aᵢ) / k  (averaged across heads for joint evaluation)
    
    Args:
        state_dim: Dimension of input state representation (typically 64-128 for ICU data).
        action_dims: List of action space sizes for each dimension.
                    E.g., [7, 6, 6] for vent task, [4, 3, 3, 2] for rrt task, [5, 4] for iv task.
        hidden_dim: Width of hidden layers in the shared MLP backbone.
        enable_double_dqn: Whether to enable Double DQN target calculation.
        enable_dueling: Whether to use dueling architecture (V(s) + A(s,a)).
        device: Computation device for the network.
        dtype: Tensor data type.
        
    Example:
        >>> # For ICU mechanical ventilation control
        >>> net = DQNNet(
        ...     state_dim=87,  # Patient vital signs + lab values
        ...     action_dims=[7, 6, 6],  # PEEP, FiO2, Tidal Volume
        ...     hidden_dim=128,
        ...     enable_double_dqn=True,
        ...     enable_dueling=True
        ... )
        >>> q_values = net(patient_states)  # List of Q-tensors per action head
    """
    def __init__(
        self,
        state_dim: int,
        action_dims: Sequence[int],
        hidden_dim: int = 128,
        *,
        enable_double_dqn: bool = True,
        enable_dueling: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Validate input parameters to ensure robust network construction
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError(f"action_dims must be non-empty with positive values, got {action_dims}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.state_dim: int = state_dim
        self.action_dims: Tuple[int, ...] = tuple(action_dims)
        self.hidden_dim: int = hidden_dim
        self.enable_double_dqn: bool = enable_double_dqn
        self.enable_dueling: bool = enable_dueling
        
        # Gradient checkpointing state
        self._gradient_checkpointing = False
        
        # Shared feature extraction backbone - processes patient state information
        # Uses 2-layer MLP with ReLU activation for non-linear feature learning
        self.shared: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
        )

        if self.enable_dueling:
            # Dueling DQN architecture: V(s) + A(s,a) - mean(A(s,·))
            # This separates state value estimation from action advantage estimation
            self.value_head = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)
            self.advantage_heads: nn.ModuleList = nn.ModuleList([
                nn.Linear(hidden_dim, adim, device=device, dtype=dtype) 
                for adim in self.action_dims
            ])
        else:
            # Standard DQN: Direct Q-value estimation for each action head
            self.heads: nn.ModuleList = nn.ModuleList([
                nn.Linear(hidden_dim, adim, device=device, dtype=dtype) 
                for adim in self.action_dims
            ])

        # Apply principled weight initialization for stable training
        self._initialize_weights()

        # Ensure network is on the correct device and dtype
        if device is not None:
            self.to(device=device, dtype=dtype)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def _checkpointed_forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared layers with optional gradient checkpointing."""
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.shared, x, use_reentrant=False)
        else:
            return self.shared(x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the DQN network.

        Args:
            x: Input state tensor of shape (batch_size, state_dim).
               Represents patient states including vital signs, lab values, etc.

        Returns:
            List of Q-value tensors, one per action dimension.
            Each tensor has shape (batch_size, action_dim_i).
            
        Raises:
            ValueError: If input tensor has incorrect shape or dimensions.
            
        Note:
            For dueling networks, Q(s,a) = V(s) + A(s,a) - mean_a'(A(s,a'))
            This formulation ensures identifiability while maintaining the
            advantage of separate value and advantage streams.
        """
        if x.dim() != 2 or x.size(1) != self.state_dim:
            raise ValueError(
                f"Expected input of shape (batch_size, {self.state_dim}), "
                f"got {tuple(x.shape)}"
            )

        # Extract shared features from patient state with optional gradient checkpointing
        h = self._checkpointed_forward_shared(x)
        
        if self.enable_dueling:
            # Dueling DQN: Decompose Q-values into state value + action advantages
            state_value = self.value_head(h)  # V(s): (batch_size, 1)
            
            q_list: List[torch.Tensor] = []
            for advantage_head in self.advantage_heads:
                advantages = advantage_head(h)  # A(s,a): (batch_size, action_dim)
                
                # Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
                # The subtraction of mean ensures identifiability
                q_values = state_value + advantages - advantages.mean(dim=-1, keepdim=True)
                q_list.append(q_values)
            
            return q_list
        else:
            # Standard DQN: Direct Q-value estimation
            return [head(h) for head in self.heads]

    def create_target_network(self) -> "DQNNet":
        """
        Create a target network for stable Q-learning updates.
        
        The target network is a frozen copy of the current network used to
        compute target Q-values, reducing training instability from the
        moving target problem in temporal difference learning.
        
        Returns:
            DQNNet: Target network with identical architecture and copied weights.
                   All parameters have requires_grad=False for computational efficiency.
                   
        Note:
            Target networks should be periodically synchronized with the main
            network (e.g., every 1000-10000 training steps) to balance
            stability and learning progress.
        """
        target_net = DQNNet(
            self.state_dim,
            self.action_dims,
            hidden_dim=self.hidden_dim,
            enable_double_dqn=self.enable_double_dqn,
            enable_dueling=self.enable_dueling,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        target_net.load_state_dict(self.state_dict())
        
        # Freeze target network parameters to prevent accidental updates
        for param in target_net.parameters():
            param.requires_grad = False
            
        return target_net

    def compute_double_dqn_targets(
        self, 
        target_net: "DQNNet",
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> List[torch.Tensor]:
        """
        Compute target Q-values using Double DQN to reduce overestimation bias.
        
        Double DQN uses the main network to select actions and the target
        network to evaluate them, breaking the correlation that leads to
        overestimation in standard DQN.
        
        Algorithm:
        1. Use main network to select best actions: a* = argmax_a Q_main(s', a)
        2. Use target network to evaluate: Q_target(s', a*)
        
        Args:
            target_net: Target network for value estimation.
            next_states: Next states tensor (batch_size, state_dim).
            rewards: Immediate rewards tensor (batch_size,).
            dones: Terminal state indicators (batch_size,).
            gamma: Discount factor for future rewards.
            
        Returns:
            List of target Q-value tensors for each action head.
        """
        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN: Use main network for action selection
                next_q_main = self.forward(next_states)
                
                # Get greedy actions from main network
                next_actions = [q.argmax(dim=-1) for q in next_q_main]
                
                # Use target network for value estimation
                next_q_target = target_net.forward(next_states)
                
                # Evaluate selected actions with target network
                target_values = []
                for i, (q_target, action) in enumerate(zip(next_q_target, next_actions)):
                    target_value = q_target.gather(-1, action.unsqueeze(-1)).squeeze(-1)
                    target_values.append(target_value)
            else:
                # Standard DQN: Use target network for both selection and evaluation
                next_q_target = target_net.forward(next_states)
                target_values = [q.max(dim=-1)[0] for q in next_q_target]
            
            # Compute TD targets: r + γ * max_a Q_target(s', a) * (1 - done)
            targets = []
            for target_value in target_values:
                target = rewards + gamma * target_value * (~dones)
                targets.append(target)
                
            return targets

    def q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar Q(s,a) for joint multi-head action evaluation.
        
        This method is used for off-policy evaluation and Boltzmann exploration
        where we need a single scalar Q-value for the joint action.
        
        Args:
            state: State tensor of shape (batch_size, state_dim).
            action: Action tensor of shape (batch_size, n_heads) with integer actions.
                   Each column represents the discrete action for one head.

        Returns:
            Tensor of shape (batch_size,) containing averaged Q-values across heads.
            
        Note:
            We average Q-values across heads to maintain consistent scaling
            regardless of the number of action dimensions. This is crucial
            for fair comparison across different medical intervention types.
        """
        if action.dim() != 2 or action.size(1) != len(self.action_dims):
            raise ValueError(
                f"Action tensor must have shape (batch_size, {len(self.action_dims)}), "
                f"got {tuple(action.shape)}"
            )

        q_heads = self.forward(state)
        q_values = []
        
        for head_idx, q_head in enumerate(q_heads):
            # Extract Q-value for the selected action in this head
            action_indices = action[:, head_idx:head_idx+1]
            q_val = q_head.gather(1, action_indices).squeeze(1)
            q_values.append(q_val)
        
        # Average across heads for consistent scaling
        return torch.stack(q_values, dim=1).mean(dim=1)

    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select greedy actions for given states.
        
        This method is required for FQE evaluation and implements greedy policy
        based on current Q-value estimates.
        
        Args:
            state: State tensor of shape (batch_size, state_dim).
                  Represents patient states including vital signs, lab values, etc.

        Returns:
            Greedy actions tensor of shape (batch_size, n_heads) with integer actions.
            Each column represents the greedy action for one action dimension.
            
        Note:
            This method is essential for FQE evaluation where we need to extract
            the greedy policy from the learned Q-function.
        """
        with torch.no_grad():
            q_heads = self.forward(state)
            greedy_actions = []
            
            for q_head in q_heads:
                # Select action with highest Q-value for each head
                greedy_action = q_head.argmax(dim=-1)  # Shape: (batch_size,)
                greedy_actions.append(greedy_action)
            
            # Stack into action tensor: (batch_size, n_heads)
            return torch.stack(greedy_actions, dim=1)

    def _initialize_weights(self) -> None:
        """
        Initialize network weights using principled schemes for stable training.
        
        Uses Kaiming initialization for ReLU networks to maintain proper
        variance propagation through the network. This is particularly
        important for deep Q-networks to ensure stable gradient flow.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming (He) initialization for ReLU networks
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
                if module.bias is not None:
                    # Initialize bias with small uniform values
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)

    def __repr__(self) -> str:
        """Provide informative string representation for debugging and logging."""
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dims={self.action_dims}, "
            f"hidden_dim={self.hidden_dim}, "
            f"double_dqn={self.enable_double_dqn}, "
            f"dueling={self.enable_dueling})"
        )
