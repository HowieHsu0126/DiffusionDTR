"""Baseline Behavior Cloning (BC) implementation for medical RL.

This module provides robust implementations of Behavior Cloning for offline
imitation learning in medical ICU settings. BC learns to directly mimic
physician behavior from historical treatment data, serving as a fundamental
baseline for more sophisticated offline RL algorithms.

Key Features:
------------
1. **Multi-head Architecture** - Independent prediction heads for different interventions
2. **LSTM Support** - Handles sequential patient data with temporal dependencies  
3. **Regularization Mechanisms** - Dropout, weight decay, and label smoothing
4. **Numerical Stability** - Robust training for medical applications
5. **Comprehensive Evaluation** - Detailed metrics for imitation learning analysis

Mathematical Foundation:
-----------------------
BC optimizes the supervised learning objective:
L_BC(π) = E_{(s,a)~D}[-log π(a|s)]

Where:
- π(a|s) is the learned policy (multi-head categorical distributions)
- D is the offline dataset of physician demonstrations
- The loss encourages the policy to maximize likelihood of observed actions

Clinical Application:
--------------------
In ICU-AKI treatment, BC provides a conservative baseline that directly
replicates historical physician decision patterns, offering interpretable
behavior that can be compared against more complex RL approaches.

Architecture Variants:
---------------------
1. **BCPolicyNet** - Standard MLP for static state representations
2. **BCPolicyNetLSTM** - RNN-based model for sequential patient trajectories
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class BCPolicyNet(nn.Module):
    """
    Multi-head Behavior Cloning MLP policy network for medical RL.

    This network learns to predict physician actions from patient states using
    separate categorical distributions for each action dimension. The multi-head
    design enables independent modeling of different medical interventions while
    sharing lower-level feature representations.

    Architecture Design:
    -------------------
    Input: Patient state vector (vital signs, lab values, demographics)
    → Shared feature extraction (2-layer MLP with dropout)
    → Separate prediction heads for each action dimension
    → Softmax outputs for categorical action distributions

    Medical Context:
    ---------------
    Each action head corresponds to a specific medical intervention:
    - Mechanical ventilation: PEEP, FiO2, Tidal Volume settings
    - Medication management: Vasopressor dosages, sedation levels
    - Fluid management: Fluid balance targets, diuretic protocols

    The shared backbone learns common patient assessment patterns while
    specialized heads capture intervention-specific decision logic.

    Args:
        state_dim: Dimension of input patient state vector (typically 64-128).
        action_dims: List of action space sizes for each intervention type.
                    E.g., [7, 6, 6] for vent, [4, 3, 3, 2] for rrt, [5, 4] for iv.
        hidden_dim: Hidden layer dimension for the shared MLP backbone.
        dropout: Dropout probability for regularization (important for medical safety).
        label_smoothing: Label smoothing factor to prevent overconfident predictions.
        device: Computation device for the network.

    Example:
        >>> # For ICU mechanical ventilation control
        >>> policy = BCPolicyNet(
        ...     state_dim=87,  # Patient vital signs + lab values
        ...     action_dims=[7, 6, 6],  # PEEP, FiO2, Tidal Volume
        ...     hidden_dim=128,
        ...     dropout=0.1,
        ...     label_smoothing=0.05  # Slight smoothing for medical safety
        ... )
        >>> logits, probs = policy(patient_states)
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        hidden_dim: int = 128,
        *,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        # Input validation for robust network construction
        if not action_dims or any(d <= 0 for d in action_dims):
            raise ValueError(f"action_dims must be non-empty with positive integers, got {action_dims}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

        self.state_dim = state_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.device = torch.device(device)

        # Gradient checkpointing state
        self._gradient_checkpointing = False

        # Shared feature extraction backbone with enhanced regularization
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch normalization for stable training
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate prediction heads for each action dimension
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, action_dim),
                # No activation here - raw logits for numerical stability
            )
            for action_dim in action_dims
        ])

        # Apply principled weight initialization
        self._initialize_weights()

        # Track training statistics
        self.register_buffer('prediction_entropy_ema', torch.tensor(0.0))
        self.register_buffer('ema_decay', torch.tensor(0.99))

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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the behavior cloning network.

        Args:
            x: Input state tensor of shape (batch_size, state_dim).
               Represents patient states including vital signs, lab values, etc.

        Returns:
            Tuple of (logits_list, probs_list) where:
            - logits_list: List of raw logits for each action head [(batch_size, action_dim_i)]
            - probs_list: List of probability distributions [(batch_size, action_dim_i)]

        Raises:
            ValueError: If input tensor has incorrect shape.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2-D input (batch_size, state_dim), got {x.shape}")
        if x.size(1) != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {x.size(1)}")

        # Ensure input is on the correct device
        param_device = next(self.parameters()).device
        if x.device != param_device:
            x = x.to(param_device)

        # Extract shared features
        h = self._checkpointed_forward_shared(x)
        
        # Compute logits and probabilities for each action head
        logits_list = [head(h) for head in self.heads]
        
        # Apply label smoothing if specified (for training stability)
        if self.training and self.label_smoothing > 0:
            probs_list = [
                self._apply_label_smoothing(F.softmax(logits, dim=-1), self.label_smoothing)
                for logits in logits_list
            ]
        else:
            probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]

        # Track prediction entropy for monitoring
        self._update_entropy_stats(probs_list)

        return logits_list, probs_list

    def compute_loss(
        self, 
        logits_list: List[torch.Tensor], 
        actions: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute behavior cloning loss with optional sample weighting.

        Args:
            logits_list: Predicted action logits for each head.
            actions: Ground truth actions (batch_size, n_heads).
            weights: Optional sample weights (batch_size,) for importance sampling.

        Returns:
            Tuple of (total_loss, loss_dict) containing detailed loss information.
        """
        if actions.dim() != 2 or actions.size(1) != len(self.action_dims):
            raise ValueError(f"Actions must have shape (batch_size, {len(self.action_dims)})")

        total_loss = 0.0
        head_losses = []

        for head_idx, head_logits in enumerate(logits_list):
            target_actions = actions[:, head_idx]
            
            # Compute cross-entropy loss for this head
            if self.label_smoothing > 0 and self.training:
                head_loss = self._label_smoothed_cross_entropy(
                    head_logits, target_actions, self.label_smoothing
                )
            else:
                head_loss = F.cross_entropy(head_logits, target_actions, reduction='none')
            
            # Apply sample weights if provided
            if weights is not None:
                head_loss = head_loss * weights
            
            head_loss = head_loss.mean()
            head_losses.append(head_loss)
            total_loss += head_loss

        # Average across heads
        total_loss = total_loss / len(self.action_dims)

        # Prepare detailed loss information
        loss_dict = {
            'total_loss': total_loss.item(),
            'head_losses': [loss.item() for loss in head_losses],
            'avg_entropy': self.prediction_entropy_ema.item(),
        }

        return total_loss, loss_dict

    def predict_actions(
        self, 
        states: torch.Tensor, 
        deterministic: bool = True,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Predict actions for given states using the learned policy.

        Args:
            states: Input states (batch_size, state_dim).
            deterministic: If True, use argmax; if False, sample from distribution.
            temperature: Temperature for sampling (higher = more exploration).

        Returns:
            Predicted actions (batch_size, n_heads).
        """
        with torch.no_grad():
            logits_list, _ = self.forward(states)
            
            predicted_actions = []
            for head_logits in logits_list:
                if deterministic:
                    # Greedy action selection
                    actions = head_logits.argmax(dim=-1)
                else:
                    # Temperature-scaled sampling
                    if temperature != 1.0:
                        head_logits = head_logits / temperature
                    probs = F.softmax(head_logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                predicted_actions.append(actions)
            
            return torch.stack(predicted_actions, dim=1)

    def _apply_label_smoothing(
        self, 
        probs: torch.Tensor, 
        smoothing: float
    ) -> torch.Tensor:
        """Apply label smoothing to probability distributions."""
        num_classes = probs.size(-1)
        uniform_prob = 1.0 / num_classes
        return (1 - smoothing) * probs + smoothing * uniform_prob

    def _label_smoothed_cross_entropy(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        smoothing: float
    ) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss."""
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        
        # One-hot encoding with smoothing
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

    def _update_entropy_stats(self, probs_list: List[torch.Tensor]) -> None:
        """Update exponential moving average of prediction entropy."""
        if not self.training:
            return
            
        with torch.no_grad():
            total_entropy = 0.0
            for probs in probs_list:
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                total_entropy += entropy.item()
            
            avg_entropy = total_entropy / len(probs_list)
            self.prediction_entropy_ema = (
                self.ema_decay * self.prediction_entropy_ema + 
                (1 - self.ema_decay) * avg_entropy
            )

    def _initialize_weights(self) -> None:
        """
        Initialize network weights using principled schemes.
        
        Uses Xavier initialization for Linear layers and proper initialization
        for BatchNorm layers to ensure stable training dynamics.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def n_heads(self) -> int:
        """Number of action heads (intervention types)."""
        return len(self.action_dims)

    def __repr__(self) -> str:
        """Provide informative string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dims={self.action_dims}, "
            f"hidden_dim={self.hidden_dim}, "
            f"dropout={self.dropout})"
        )


class BCPolicyNetLSTM(nn.Module):
    """
    Multi-head Behavior Cloning LSTM policy network for sequential medical data.

    This network extends the basic BC approach to handle temporal dependencies
    in patient trajectories. It uses LSTM layers to process variable-length
    sequences of patient states, making it suitable for modeling the evolution
    of patient conditions over time.

    Architecture Design:
    -------------------
    Input: Sequential patient states (batch_size, seq_len, state_dim)
    → LSTM layers with optional bidirectional processing
    → Sequence-aware feature extraction
    → Multi-head action prediction from final time step
    → Softmax outputs for categorical action distributions

    Temporal Modeling:
    -----------------
    The LSTM captures important temporal patterns in ICU care such as:
    - Gradual escalation of ventilator support
    - Response to medication titration over time
    - Adaptation to changing patient physiological state
    - Sequential treatment protocols and care bundles

    Args:
        state_dim: Dimension of input patient state at each time step.
        action_dims: List of action space sizes for each intervention type.
        hidden_dim: Hidden layer dimension for LSTM and prediction heads.
        lstm_layers: Number of LSTM layers for temporal modeling.
        bidirectional: Whether to use bidirectional LSTM processing.
        dropout: Dropout probability for regularization.
        device: Computation device for the network.

    Example:
        >>> # For sequential ICU treatment modeling
        >>> lstm_policy = BCPolicyNetLSTM(
        ...     state_dim=87,  # Patient features per time step
        ...     action_dims=[7, 6, 6],  # PEEP, FiO2, Tidal Volume
        ...     hidden_dim=128,
        ...     lstm_layers=2,
        ...     bidirectional=True,
        ...     dropout=0.1
        ... )
        >>> logits, probs = lstm_policy(patient_sequences, sequence_lengths)
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        hidden_dim: int = 128,
        *,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        # Input validation
        if not action_dims or any(d <= 0 for d in action_dims):
            raise ValueError(f"action_dims must be non-empty with positive integers, got {action_dims}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if lstm_layers <= 0:
            raise ValueError(f"lstm_layers must be positive, got {lstm_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.state_dim = state_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = torch.device(device)

        # Gradient checkpointing state
        self._gradient_checkpointing = False

        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,  # LSTM dropout only for multi-layer
        )

        # Calculate LSTM output dimension
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Prediction heads with enhanced architecture for sequential data
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Layer normalization for sequence models
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, action_dim),
            )
            for action_dim in action_dims
        ])

        # Apply principled weight initialization
        self._initialize_weights()

        # Track sequence modeling statistics
        self.register_buffer('avg_sequence_length', torch.tensor(0.0))
        self.register_buffer('temporal_consistency_ema', torch.tensor(0.0))
        self.register_buffer('ema_decay', torch.tensor(0.99))

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def _checkpointed_forward_lstm(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through LSTM with optional gradient checkpointing."""
        if self._gradient_checkpointing and self.training:
            if lengths is not None:
                # For packed sequences, we need a wrapper function
                def lstm_wrapper(x_inner):
                    lengths_cpu = lengths.cpu()
                    packed_input = nn.utils.rnn.pack_padded_sequence(
                        x_inner, lengths_cpu, batch_first=True, enforce_sorted=False
                    )
                    packed_output, _ = self.lstm(packed_input)
                    lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                    return lstm_output
                return torch.utils.checkpoint.checkpoint(lstm_wrapper, x, use_reentrant=False)
            else:
                def lstm_wrapper(x_inner):
                    lstm_output, _ = self.lstm(x_inner)
                    return lstm_output
                return torch.utils.checkpoint.checkpoint(lstm_wrapper, x, use_reentrant=False)
        else:
            if lengths is not None:
                lengths_cpu = lengths.cpu()
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    x, lengths_cpu, batch_first=True, enforce_sorted=False
                )
                packed_output, _ = self.lstm(packed_input)
                lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                return lstm_output
            else:
                lstm_output, _ = self.lstm(x)
                return lstm_output

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the LSTM behavior cloning network.

        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, state_dim).
               Represents sequential patient states over time.
            lengths: Optional tensor of actual sequence lengths (batch_size,).
                    If None, assumes all sequences use the full length.

        Returns:
            Tuple of (logits_list, probs_list) where:
            - logits_list: List of action logits for each head [(batch_size, action_dim_i)]
            - probs_list: List of probability distributions [(batch_size, action_dim_i)]

        Raises:
            ValueError: If input tensor has incorrect dimensions.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3-D input (batch_size, seq_len, state_dim), got {x.shape}")
        if x.size(2) != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {x.size(2)}")

        batch_size, seq_len = x.size(0), x.size(1)

        # Ensure input is on correct device
        param_device = next(self.parameters()).device
        if x.device != param_device:
            x = x.to(param_device)

        # Process sequences through LSTM with optional gradient checkpointing
        if lengths is not None:
            if lengths.dim() != 1 or lengths.size(0) != batch_size:
                raise ValueError(f"lengths must be 1-D with {batch_size} elements")

            # Use gradient checkpointing for LSTM
            lstm_output = self._checkpointed_forward_lstm(x, lengths)

            # Extract final time step output for each sequence
            batch_indices = torch.arange(batch_size, device=lstm_output.device)
            lengths_device = lengths.to(lstm_output.device)
            final_outputs = lstm_output[batch_indices, lengths_device - 1]
            
            # Update sequence length statistics
            self._update_sequence_stats(lengths)
        else:
            # Process full sequences with gradient checkpointing
            lstm_output = self._checkpointed_forward_lstm(x)
            final_outputs = lstm_output[:, -1]  # Use last time step

        # Generate action predictions from final LSTM outputs
        logits_list = [head(final_outputs) for head in self.heads]
        probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]

        return logits_list, probs_list

    def compute_temporal_consistency_loss(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        consistency_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute temporal consistency regularization loss.
        
        This encourages smooth changes in predicted action probabilities
        over time, which is medically reasonable for ICU interventions.
        
        Args:
            x: Input sequence tensor (batch_size, seq_len, state_dim).
            lengths: Optional sequence lengths.
            consistency_weight: Weight for consistency regularization.
            
        Returns:
            Temporal consistency loss (scalar tensor).
        """
        if x.size(1) < 2:  # Need at least 2 time steps
            return torch.tensor(0.0, device=x.device)

        # Get predictions for all time steps
        lstm_output, _ = self.lstm(x)
        
        consistency_losses = []
        for head in self.heads:
            # Compute probabilities for all time steps
            all_logits = head(lstm_output)  # (batch_size, seq_len, action_dim)
            all_probs = F.softmax(all_logits, dim=-1)
            
            # Compute differences between consecutive time steps
            prob_diffs = all_probs[:, 1:] - all_probs[:, :-1]  # (batch_size, seq_len-1, action_dim)
            
            # L2 norm of differences (penalize large changes)
            consistency_loss = torch.mean(torch.sum(prob_diffs ** 2, dim=-1))
            consistency_losses.append(consistency_loss)
        
        total_consistency_loss = sum(consistency_losses) / len(consistency_losses)
        return consistency_weight * total_consistency_loss

    def _update_sequence_stats(self, lengths: torch.Tensor) -> None:
        """Update exponential moving average of sequence statistics."""
        with torch.no_grad():
            avg_length = lengths.float().mean().item()
            self.avg_sequence_length = (
                self.ema_decay * self.avg_sequence_length + 
                (1 - self.ema_decay) * avg_length
            )

    def _initialize_weights(self) -> None:
        """
        Initialize network weights for stable LSTM training.
        
        Uses Xavier initialization for Linear layers and orthogonal
        initialization for LSTM weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
                        # Set forget gate bias to 1 for better gradient flow
                        if 'bias_ih' in name:
                            hidden_size = param.size(0) // 4
                            param.data[hidden_size:2*hidden_size].fill_(1.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def n_heads(self) -> int:
        """Number of action heads (intervention types)."""
        return len(self.action_dims)

    def __repr__(self) -> str:
        """Provide informative string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dims={self.action_dims}, "
            f"hidden_dim={self.hidden_dim}, "
            f"lstm_layers={self.lstm_layers}, "
            f"bidirectional={self.bidirectional})"
        )
