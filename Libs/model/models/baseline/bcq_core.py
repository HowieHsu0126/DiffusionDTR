"""Baseline BCQ (Batch-Constrained deep Q-learning) core components.

This module implements Batch-Constrained deep Q-learning for safe offline
reinforcement learning in medical ICU settings. BCQ addresses the fundamental
challenge of extrapolation error in offline RL by constraining the policy to
stay close to the behavior policy distribution.

Key Features:
------------
1. **Variational Autoencoder (VAE)** - Models behavior policy distribution
2. **Action Constraint Mechanism** - Prevents out-of-distribution actions
3. **Multi-head Support** - Handles multi-dimensional medical interventions
4. **Numerical Stability** - Robust training for medical applications
5. **Comprehensive Logging** - Detailed metrics for offline RL analysis

Mathematical Foundation:
-----------------------
BCQ constrains action selection to:
π(a|s) = argmax_a {Q(s,a) : a ~ G_ω(s,ξ), ||a - a'||_p ≤ Φ}

Where:
- G_ω(s,ξ) is the VAE generative model
- Φ is the perturbation threshold
- a' is the VAE-generated action
- Q(s,a) is the learned Q-function

Clinical Safety:
---------------
In ICU-AKI treatment, BCQ ensures that the learned policy only recommends
treatment combinations that are similar to those historically administered
by physicians, providing an important safety constraint for medical RL.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _init_weights_xavier(module: nn.Module) -> None:
    """
    Apply Xavier-uniform initialization to Linear layers with zero bias.
    
    This initialization scheme is particularly effective for networks with
    sigmoid/tanh activations and helps maintain proper gradient flow in
    the VAE components.
    
    Args:
        module: Neural network module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VAE(nn.Module):
    """
    Variational Autoencoder for modeling behavior policy in BCQ.
    
    This VAE learns to generate actions that are similar to those in the
    offline dataset, serving as a constraint mechanism to prevent the
    policy from selecting out-of-distribution actions that could be harmful.
    
    Architecture:
    ------------
    Encoder: (state, action) → latent distribution parameters (μ, σ)
    Decoder: (state, latent) → action reconstruction
    
    The latent space captures the underlying structure of physician
    decision-making patterns, enabling safe action generation.
    
    Medical Context:
    ---------------
    In ICU settings, the VAE learns patterns like:
    - High PEEP + High FiO2 for severe respiratory failure
    - Conservative fluid management for AKI patients
    - Coordinated ventilator adjustments
    
    Args:
        state_dim: Dimension of patient state vector (vital signs, lab values).
        action_dims: List of action space sizes for each intervention type.
        latent_dim: Dimension of latent space (controls capacity/regularization trade-off).
        hidden_dim: Hidden layer dimension for encoder/decoder networks.
        log_std_min: Minimum log standard deviation for numerical stability.
        log_std_max: Maximum log standard deviation to prevent collapse.
        beta: β-VAE regularization strength (higher = more regularization).
        
    Example:
        >>> # For ICU mechanical ventilation
        >>> vae = VAE(
        ...     state_dim=87,  # Patient features
        ...     action_dims=[7, 6, 6],  # PEEP, FiO2, Tidal Volume
        ...     latent_dim=32,
        ...     hidden_dim=128,
        ...     beta=0.5  # Moderate regularization for medical safety
        ... )
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        latent_dim: int = 32,
        hidden_dim: int = 128,
        *,
        log_std_min: float = -4.0,
        log_std_max: float = 4.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        # Input validation for robust construction
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError(f"action_dims must be non-empty with positive values, got {action_dims}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.state_dim = state_dim
        self.action_dims = action_dims
        self.latent_dim = latent_dim
        self.action_total_dim = sum(action_dims)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.beta = float(beta)

        # Encoder network: q(z|s,a) - learns to compress state-action pairs
        encoder_input_dim = state_dim + self.action_total_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate heads for mean and log-variance to ensure proper gradients
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder network: p(a|s,z) - reconstructs actions from state and latent
        decoder_input_dim = state_dim + latent_dim
        self.decoder_hidden = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_out = nn.Linear(hidden_dim, self.action_total_dim)

        # Apply principled weight initialization
        self.apply(_init_weights_xavier)

        # Track reconstruction quality for monitoring
        self.register_buffer('reconstruction_loss_ema', torch.tensor(0.0))
        self.register_buffer('kl_divergence_ema', torch.tensor(0.0))
        self.register_buffer('ema_decay', torch.tensor(0.99))

    def _concat_one_hot(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert multi-head discrete actions to concatenated one-hot encoding.
        
        This vectorized implementation efficiently handles the conversion from
        discrete action indices to one-hot vectors for all action heads.
        
        Args:
            action: Integer action tensor of shape (batch_size, n_heads).
                   Each column represents the discrete action for one head.

        Returns:
            One-hot encoded tensor of shape (batch_size, total_action_dim).
            
        Raises:
            ValueError: If action tensor has incorrect shape.
            
        Note:
            The one-hot encoding is necessary because the VAE operates on
            continuous representations while the action space is discrete.
        """
        if action.dim() != 2 or action.size(1) != len(self.action_dims):
            raise ValueError(
                f"Action tensor must have shape (batch_size, {len(self.action_dims)}), "
                f"got {tuple(action.shape)}"
            )

        batch_size = action.size(0)
        device = action.device

        # Compute cumulative offsets for flattening multi-head actions
        # E.g., for action_dims=[7,6,6]: offsets=[0,7,13]
        offsets = [0] + list(np.cumsum(self.action_dims[:-1]))
        offsets_tensor = torch.tensor(offsets, device=device)
        
        # Map each head's action to the global flattened index
        flat_indices = action.long() + offsets_tensor.unsqueeze(0)  # (batch_size, n_heads)

        # Create one-hot encoding
        one_hot = torch.zeros(batch_size, self.action_total_dim, device=device, dtype=torch.float32)
        one_hot.scatter_(1, flat_indices, 1.0)
        
        return one_hot

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling: z = μ + σ·ε.
        
        This enables backpropagation through the stochastic latent variable
        by expressing it as a deterministic function of the parameters plus
        independent noise.
        
        Args:
            mu: Mean of latent distribution (batch_size, latent_dim).
            logvar: Log-variance of latent distribution (batch_size, latent_dim).
            
        Returns:
            Sampled latent variables (batch_size, latent_dim).
            
        Note:
            Using log-variance instead of variance improves numerical stability
            and ensures the variance is always positive.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass: encode → reparameterize → decode.
        
        Args:
            state: Patient state tensor of shape (batch_size, state_dim).
            action: Action tensor of shape (batch_size, n_heads).

        Returns:
            Tuple containing:
            - logits_list: List of action logits for each head [(batch_size, action_dim_i)]
            - mu: Latent mean (batch_size, latent_dim)
            - logvar: Latent log-variance (batch_size, latent_dim)  
            - z: Sampled latent variable (batch_size, latent_dim)
            - kl_div: KL divergence loss (scalar)
            
        Raises:
            ValueError: If input tensors have incorrect shapes.
        """
        if state.dim() != 2 or action.dim() != 2:
            raise ValueError("state must be (batch_size, state_dim), action must be (batch_size, n_heads)")
        if state.size(0) != action.size(0):
            raise ValueError(f"Batch size mismatch: state {state.size(0)}, action {action.size(0)}")

        # Encode: (state, action) → latent distribution parameters
        action_one_hot = self._concat_one_hot(action)
        encoder_input = torch.cat([state, action_one_hot], dim=1)
        
        h_enc = self.encoder(encoder_input)
        mu = self.mu_layer(h_enc)
        logvar = self.logvar_layer(h_enc).clamp(self.log_std_min, self.log_std_max)

        # Reparameterize: sample from latent distribution
        z = self._reparameterize(mu, logvar)

        # Decode: (state, latent) → action reconstruction
        logits_flat = self.decode(state, z)

        # Split flat logits into per-head predictions
        logits_list: List[torch.Tensor] = []
        start_idx = 0
        for action_dim in self.action_dims:
            logits_list.append(logits_flat[:, start_idx:start_idx + action_dim])
            start_idx += action_dim

        # Compute KL divergence: D_KL(q(z|s,a) || p(z))
        # where p(z) = N(0,I) is the prior
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Apply β-VAE weighting for controllable regularization
        kl_div = self.beta * kl_div

        # Update exponential moving averages for monitoring
        with torch.no_grad():
            self.kl_divergence_ema = self.ema_decay * self.kl_divergence_ema + (1 - self.ema_decay) * kl_div.item()

        return logits_list, mu, logvar, z, kl_div

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variables to action logits.
        
        This method can be used independently for action generation during
        policy execution, where we sample from the prior p(z) and generate
        actions conditioned on the current state.
        
        Args:
            state: Patient state tensor (batch_size, state_dim).
            latent: Latent variables (batch_size, latent_dim).
            
        Returns:
            Flattened action logits (batch_size, total_action_dim).
        """
        decoder_input = torch.cat([state, latent], dim=1)
        h_dec = self.decoder_hidden(decoder_input)
        return self.decoder_out(h_dec)

    def sample_actions(
        self, 
        state: torch.Tensor, 
        n_samples: int = 10,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample actions from the learned behavior policy distribution.
        
        This is the key method used during BCQ policy execution to generate
        candidate actions that are constrained to the behavior policy support.
        
        Args:
            state: Patient state tensor (batch_size, state_dim).
            n_samples: Number of action samples to generate per state.
            temperature: Temperature for categorical sampling (higher = more diverse).
            
        Returns:
            Sampled actions (batch_size, n_samples, n_heads).
        """
        batch_size = state.size(0)
        device = state.device
        
        with torch.no_grad():
            # Sample from prior latent distribution
            z = torch.randn(batch_size, n_samples, self.latent_dim, device=device)
            z = z.reshape(-1, self.latent_dim)
            
            # Expand states to match latent samples
            expanded_states = state.unsqueeze(1).expand(-1, n_samples, -1)
            expanded_states = expanded_states.reshape(-1, self.state_dim)
            
            # Decode to action logits
            logits_flat = self.decode(expanded_states, z)
            
            # Sample actions for each head
            sampled_actions = []
            start_idx = 0
            for action_dim in self.action_dims:
                head_logits = logits_flat[:, start_idx:start_idx + action_dim]
                
                # Apply temperature scaling for controlled exploration
                if temperature != 1.0:
                    head_logits = head_logits / temperature
                    
                # Sample from categorical distribution
                probs = F.softmax(head_logits, dim=-1)
                samples = torch.multinomial(probs, 1).squeeze(-1)
                sampled_actions.append(samples)
                start_idx += action_dim
            
            # Reshape to (batch_size, n_samples, n_heads)
            sampled_actions = torch.stack(sampled_actions, dim=-1)
            sampled_actions = sampled_actions.reshape(batch_size, n_samples, len(self.action_dims))
            
        return sampled_actions

    def compute_reconstruction_loss(
        self, 
        logits_list: List[torch.Tensor], 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for the VAE decoder.
        
        Args:
            logits_list: Predicted action logits for each head.
            actions: Ground truth actions (batch_size, n_heads).
            
        Returns:
            Mean reconstruction loss across all action heads.
        """
        total_loss = 0.0
        
        for head_idx, head_logits in enumerate(logits_list):
            target_actions = actions[:, head_idx]
            head_loss = F.cross_entropy(head_logits, target_actions)
            total_loss += head_loss
            
        reconstruction_loss = total_loss / len(logits_list)
        
        # Update exponential moving average for monitoring
        with torch.no_grad():
            self.reconstruction_loss_ema = (
                self.ema_decay * self.reconstruction_loss_ema + 
                (1 - self.ema_decay) * reconstruction_loss.item()
            )
            
        return reconstruction_loss

    def __repr__(self) -> str:
        """Provide informative string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dims={self.action_dims}, "
            f"latent_dim={self.latent_dim}, "
            f"beta={self.beta})"
        )


class BCQNet(nn.Module):
    """
    Q-network for BCQ that evaluates state-action pairs.
    
    This network provides Q-value estimates for the BCQ algorithm, working
    in conjunction with the VAE to perform constrained policy optimization.
    Unlike value-based methods, BCQ uses the Q-network to evaluate actions
    generated by the VAE rather than performing unconstrained maximization.
    
    Architecture:
    ------------
    Input: Concatenated (state, one-hot action) vector
    → 2-layer MLP with ReLU activations  
    → Single Q-value output
    
    Medical Context:
    ---------------
    The Q-network learns to evaluate the expected outcomes of different
    treatment combinations, providing value estimates that guide the
    constrained action selection process.
    
    Args:
        state_dim: Dimension of patient state representation.
        action_dims: List of action space sizes for each intervention type.
        hidden_dim: Hidden layer dimension for the Q-network.
        
    Example:
        >>> # For ICU treatment evaluation
        >>> q_net = BCQNet(
        ...     state_dim=87,  # Patient features
        ...     action_dims=[7, 6, 6],  # Treatment dimensions
        ...     hidden_dim=128
        ... )
        >>> q_value = q_net(patient_state, treatment_action)
    """

    def __init__(
        self, 
        state_dim: int, 
        action_dims: List[int], 
        hidden_dim: int = 128
    ) -> None:
        super().__init__()

        # Input validation
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError(f"action_dims must be non-empty with positive values, got {action_dims}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.state_dim = state_dim
        self.action_dims = action_dims
        self.action_total_dim = sum(action_dims)

        # Q-network architecture: (state, action) → Q(s,a)
        input_dim = state_dim + self.action_total_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Single Q-value output
        )

        # Apply principled weight initialization
        self.apply(_init_weights_xavier)

    def _concat_one_hot(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete actions to one-hot encoding (same as VAE implementation).
        
        Args:
            action: Discrete action tensor (batch_size, n_heads).
            
        Returns:
            One-hot encoded actions (batch_size, total_action_dim).
        """
        if action.dim() != 2 or action.size(1) != len(self.action_dims):
            raise ValueError(
                f"Action tensor must have shape (batch_size, {len(self.action_dims)}), "
                f"got {tuple(action.shape)}"
            )

        batch_size = action.size(0)
        device = action.device

        # Compute offsets for multi-head action flattening
        offsets = [0] + list(np.cumsum(self.action_dims[:-1]))
        offsets_tensor = torch.tensor(offsets, device=device)
        
        flat_indices = action.long() + offsets_tensor.unsqueeze(0)

        one_hot = torch.zeros(batch_size, self.action_total_dim, device=device, dtype=torch.float32)
        one_hot.scatter_(1, flat_indices, 1.0)
        
        return one_hot

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s,a) for given state-action pairs.
        
        Args:
            state: Patient state tensor (batch_size, state_dim).
            action: Discrete action tensor (batch_size, n_heads).
            
        Returns:
            Q-values (batch_size, 1).
            
        Raises:
            ValueError: If input tensors have incorrect shapes.
        """
        if state.dim() != 2 or action.dim() != 2:
            raise ValueError("state must be (batch_size, state_dim), action must be (batch_size, n_heads)")
        if state.size(0) != action.size(0):
            raise ValueError(f"Batch size mismatch: state {state.size(0)}, action {action.size(0)}")

        # Convert actions to one-hot and concatenate with state
        action_one_hot = self._concat_one_hot(action)
        q_input = torch.cat([state, action_one_hot], dim=1)
        
        return self.net(q_input)

    def __repr__(self) -> str:
        """Provide informative string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dims={self.action_dims}, "
            f"hidden_dim={self.action_total_dim})"
        )
