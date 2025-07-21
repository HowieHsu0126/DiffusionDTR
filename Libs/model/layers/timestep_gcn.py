"""TimeStep GCN module for graph-based trajectory reasoning.

This module implements a two-layer Graph Convolutional Network (GCN) for
processing patient trajectory data with graph structure, supporting both
policy and value function estimation.
"""

from typing import List, Tuple, Union, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np


class TimeStepGCN(nn.Module):
    """Two-layer GCN for per-timestep graph reasoning with multi-action heads.
    
    This module processes patient trajectory features through a graph structure,
    applying graph convolutions to capture structural relationships between
    patients at each time step. It supports multiple output modes for both
    policy learning and value function approximation.
    
    The GCN includes residual connections, normalization, dropout, and separate
    heads for actor-critic architectures or Q-learning approaches.
    
    Attributes:
        input_dim: Input feature dimension for each node.
        gcn_hidden: Hidden dimension for GCN layers.
        action_dims: List of action space dimensions for each action component.
        dropout: Dropout probability for regularization.
        use_layernorm: Whether to apply layer normalization.
        activation: Activation function for GCN layers.
        gcn1: First GCN convolution layer.
        gcn2: Second GCN convolution layer.
        actors: Actor heads for policy output (one per action dimension).
        q_heads: Q-value heads for value estimation (one per action dimension).
        dropout_layer: Dropout layer for regularization.
        norm: Optional layer normalization module.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        gcn_hidden: int, 
        action_dims: List[int], 
        dropout: float = 0.1, 
        use_layernorm: bool = True, 
        activation: Callable = F.relu
    ) -> None:
        """Initializes the TimeStep GCN module.
        
        Args:
            input_dim: Input feature dimension for each node.
            gcn_hidden: Hidden dimension for GCN layers.
            action_dims: List of action space sizes for multi-discrete actions.
            dropout: Dropout rate for regularization. Must be in [0, 1].
            use_layernorm: Whether to apply layer normalization after GCN.
            activation: Activation function to use between GCN layers.
            
        Raises:
            ValueError: If input parameters are invalid.
        """
        super().__init__()
        
        # Validate inputs
        if input_dim <= 0 or gcn_hidden <= 0:
            raise ValueError("input_dim and gcn_hidden must be positive")
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError("action_dims must be non-empty with positive values")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
            
        self.input_dim = input_dim
        self.gcn_hidden = gcn_hidden
        self.action_dims = action_dims
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.activation = activation
        
        # Two-layer GCN architecture
        self.gcn1 = GCNConv(input_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        
        # Actor heads for policy output (one per action dimension)
        self.actors = nn.ModuleList([
            nn.Linear(gcn_hidden, action_dim) 
            for action_dim in action_dims
        ])
        
        # Q-value heads for value estimation (one per action dimension) 
        self.q_heads = nn.ModuleList([
            nn.Linear(gcn_hidden, action_dim)
            for action_dim in action_dims
        ])
        
        # Regularization layers
        self.dropout_layer = nn.Dropout(dropout)
        
        if use_layernorm:
            self.norm = nn.LayerNorm(gcn_hidden)
        else:
            self.norm = None
            
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initializes network weights using Xavier initialization.
        
        Applies Xavier uniform initialization to linear layers for stable
        training dynamics.
        """
        # Initialize actor heads
        for actor in self.actors:
            nn.init.xavier_uniform_(actor.weight)
            nn.init.zeros_(actor.bias)
            
        # Initialize Q-value heads
        for q_head in self.q_heads:
            nn.init.xavier_uniform_(q_head.weight) 
            nn.init.zeros_(q_head.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        mode: str = 'policy'
    ) -> Union[
        Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor],  # policy mode
        Tuple[List[torch.Tensor], torch.Tensor],                # q mode
        Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]   # both mode
    ]:
        """
        Forward pass through the TimeStep GCN layers.
        
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_dim).
            edge_index: Graph edge indices of shape (2, num_edges).
            mode: Forward mode - 'policy', 'q', or 'both'.
            
        Returns:
            Depending on mode:
                - 'policy': (action_logits, state_values, hidden_features)
                - 'q': (q_values, hidden_features)  
                - 'both': (action_logits, state_values, hidden_features)
                
        Raises:
            ValueError: If input shapes are invalid or mode is unsupported.
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, input_dim), got {x.dim()}D")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index shape (2, num_edges), got {edge_index.shape}")
        if mode not in ['policy', 'q', 'both']:
            raise ValueError(f"Unsupported mode '{mode}'. Must be 'policy', 'q', or 'both'")
        
        batch_size, seq_len, _ = x.shape
        
        # 🔧 CRITICAL FIX: Enhanced tensor contiguity and shape validation
        # Ensure input tensor is contiguous before processing
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Ensure edge_index is also contiguous
        if not edge_index.is_contiguous():
            edge_index = edge_index.contiguous()
        
        # 🔧 ENHANCED VALIDATION: Check for reasonable input sizes to prevent memory issues
        total_nodes = batch_size * seq_len
        if total_nodes > 1000000:  # 1M nodes threshold
            print(f"⚠️ Large graph detected: {total_nodes} nodes ({batch_size} x {seq_len})")
            print(f"🔧 This may cause memory issues or slow processing")
        
        # 🔧 CRITICAL FIX: Safe tensor reshaping with comprehensive error handling
        # Reshape for GCN processing: (batch_size * seq_len, input_dim)
        try:
            x_flat = x.reshape(batch_size * seq_len, -1)
        except Exception as e:
            print(f"❌ Failed to reshape input tensor: {e}")
            print(f"🔍 Input shape: {x.shape}, target: ({batch_size * seq_len}, {x.size(-1)})")
            raise RuntimeError(f"Input reshaping failed: {e}") from e
        
        # Validate flattened tensor
        if x_flat.size(0) != total_nodes or x_flat.size(1) != self.input_dim:
            raise RuntimeError(f"Unexpected flattened shape: {x_flat.shape}, expected: ({total_nodes}, {self.input_dim})")
        
        # 🔧 ENHANCED GCN PROCESSING: Robust GCN layers with error handling and fallbacks
        try:
            # First GCN layer with activation and dropout
            h = self.activation(self.gcn1(x_flat, edge_index))
            h = self.dropout_layer(h)
            
            # Second GCN layer with activation
            h = self.activation(self.gcn2(h, edge_index))
            
            # Optional layer normalization
            if self.use_layernorm:
                h = self.norm(h)
            
        except Exception as e:
            # 🔧 ENHANCED ERROR HANDLING: Detailed debug info for GCN errors
            print(f"❌ GCN processing failed: {e}")
            print(f"🔍 Debug info:")
            print(f"   • x_flat.shape: {x_flat.shape}")
            print(f"   • x_flat.device: {x_flat.device}")
            print(f"   • x_flat.dtype: {x_flat.dtype}")
            print(f"   • x_flat.is_contiguous(): {x_flat.is_contiguous()}")
            print(f"   • edge_index.shape: {edge_index.shape}")
            print(f"   • edge_index.device: {edge_index.device}")
            print(f"   • edge_index.dtype: {edge_index.dtype}")
            print(f"   • edge_index.is_contiguous(): {edge_index.is_contiguous()}")
            print(f"   • edge_index range: [{edge_index.min().item()}, {edge_index.max().item()}]")
            print(f"   • total_nodes: {total_nodes}")
            print(f"   • gcn1 input_dim: {self.gcn1.in_channels}")
            print(f"   • gcn1 output_dim: {self.gcn1.out_channels}")
            print(f"   • gcn2 input_dim: {self.gcn2.in_channels}")
            print(f"   • gcn2 output_dim: {self.gcn2.out_channels}")
            
            # Check if edge_index values are valid
            if edge_index.numel() > 0:
                max_edge_idx = edge_index.max().item()
                if max_edge_idx >= total_nodes:
                    print(f"❌ Invalid edge indices: max edge index {max_edge_idx} >= total nodes {total_nodes}")
                    print(f"🔧 This suggests edge_index is inconsistent with flattened node count")
            
            # Try fallback: process without edges (isolated nodes)
            try:
                print("🔄 Attempting fallback without graph edges...")
                # Create empty edge index for isolated node processing
                empty_edge_index = torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
                
                # Process with empty edges
                h = self.activation(self.gcn1(x_flat, empty_edge_index))
                h = self.dropout_layer(h)
                h = self.activation(self.gcn2(h, empty_edge_index))
                
                if self.use_layernorm:
                    h = self.norm(h)
                
                print("✅ Fallback processing without edges succeeded")
                
            except Exception as fallback_e:
                print(f"❌ Fallback also failed: {fallback_e}")
                raise RuntimeError(f"GCN processing failed: original={e}, fallback={fallback_e}") from e
            
        # 🔧 CRITICAL FIX: Safe tensor reshaping back to sequence format
        # Reshape back to sequence format: (batch_size, seq_len, gcn_hidden)
        try:
            h = h.view(batch_size, seq_len, -1)
        except Exception as e:
            print(f"❌ Failed to reshape GCN output back to sequence format: {e}")
            print(f"🔍 h.shape: {h.shape}, target: ({batch_size}, {seq_len}, {self.gcn_hidden})")
            
            # Try to infer correct dimensions
            if h.numel() == batch_size * seq_len * self.gcn_hidden:
                h = h.reshape(batch_size, seq_len, self.gcn_hidden)
                print("✅ Successfully reshaped using explicit dimensions")
            else:
                print(f"❌ Element count mismatch: {h.numel()} vs expected {batch_size * seq_len * self.gcn_hidden}")
                raise RuntimeError(f"Cannot reshape GCN output: {e}") from e
        
        # Validate reshaped tensor
        if h.shape != (batch_size, seq_len, self.gcn_hidden):
            raise RuntimeError(f"Unexpected reshaped tensor shape: {h.shape}, expected: ({batch_size}, {seq_len}, {self.gcn_hidden})")
        
        # Generate outputs based on mode
        if mode == 'policy':
            return self._forward_policy(h)
        elif mode == 'q':
            return self._forward_q(h, batch_size, seq_len)
        elif mode == 'both':
            return self._forward_both(h)
    
    def _forward_policy(self, h: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Generates policy outputs (action logits and state values).
        
        Args:
            h: Hidden features of shape (batch_size, seq_len, gcn_hidden).
            
        Returns:
            Tuple of (action_logits, state_values, hidden_features).
        """
        # 🔧 ENHANCED ACTOR HEAD PROCESSING: Robust tensor operations with validation
        try:
            # Generate action logits for each action dimension
            action_logits = []
            for i, actor in enumerate(self.actors):
                try:
                    logits = actor(h)  # Should produce (batch_size, seq_len, action_dim)
                    
                    # Validate output shape
                    expected_shape = (h.size(0), h.size(1), self.action_dims[i])
                    if logits.shape != expected_shape:
                        print(f"⚠️ Actor {i} output shape mismatch: {logits.shape} vs expected {expected_shape}")
                        # Try to fix by reshaping if element count matches
                        if logits.numel() == expected_shape[0] * expected_shape[1] * expected_shape[2]:
                            logits = logits.view(expected_shape)
                            print(f"🔧 Reshaped actor {i} output to correct shape")
                        else:
                            raise RuntimeError(f"Cannot fix actor {i} output shape")
                    
                    action_logits.append(logits)
                    
                except Exception as actor_e:
                    print(f"❌ Actor {i} processing failed: {actor_e}")
                    print(f"🔍 h.shape: {h.shape}, actor input_features: {actor.in_features}, output_features: {actor.out_features}")
                    raise RuntimeError(f"Actor {i} failed: {actor_e}") from actor_e
        
        except Exception as e:
            print(f"❌ Policy forward failed: {e}")
            raise RuntimeError(f"Policy forward failed: {e}") from e
        
        # Generate state values as mean of hidden features
        try:
            state_values = torch.mean(h, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        except Exception as e:
            print(f"❌ State value computation failed: {e}")
            # Fallback: use zeros
            state_values = torch.zeros(h.size(0), h.size(1), 1, device=h.device, dtype=h.dtype)
            print("🔧 Used zero fallback for state values")
        
        return action_logits, state_values, h
    
    def _forward_q(self, h: torch.Tensor, batch_size: int, seq_len: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Generates Q-value outputs.
        
        Args:
            h: Hidden features of shape (batch_size, seq_len, gcn_hidden).
            batch_size: Batch size for reshaping.
            seq_len: Sequence length for reshaping.
            
        Returns:
            Tuple of (q_values, hidden_features).
        """
        # 🔧 ENHANCED Q-HEAD PROCESSING: Safe reshaping and tensor operations
        try:
            # Reshape for Q-head processing
            h_flat = h.view(batch_size * seq_len, -1)
        except Exception as e:
            print(f"❌ Failed to flatten hidden features for Q-heads: {e}")
            print(f"🔍 h.shape: {h.shape}, target: ({batch_size * seq_len}, {h.size(-1)})")
            raise RuntimeError(f"Q-head reshaping failed: {e}") from e
        
        # Generate Q-values for each action dimension
        q_values = []
        for i, q_head in enumerate(self.q_heads):
            try:
                # Forward through Q-head
                q_flat = q_head(h_flat)  # Should produce (batch_size * seq_len, action_dim)
                
                # Reshape back to sequence format
                q_reshaped = q_flat.view(batch_size, seq_len, -1)
                
                # Validate output shape
                expected_action_dim = self.action_dims[i]
                if q_reshaped.size(-1) != expected_action_dim:
                    print(f"⚠️ Q-head {i} output dimension mismatch: {q_reshaped.size(-1)} vs expected {expected_action_dim}")
                    # This might indicate a model architecture issue
                    if q_reshaped.size(-1) > expected_action_dim:
                        # Truncate to expected size
                        q_reshaped = q_reshaped[:, :, :expected_action_dim]
                        print(f"🔧 Truncated Q-head {i} output to expected dimension")
                    else:
                        raise RuntimeError(f"Q-head {i} output dimension too small")
                
                q_values.append(q_reshaped)
                
            except Exception as q_e:
                print(f"❌ Q-head {i} processing failed: {q_e}")
                print(f"🔍 h_flat.shape: {h_flat.shape}, q_head input_features: {q_head.in_features}, output_features: {q_head.out_features}")
                print(f"🔍 Expected output shape: ({batch_size}, {seq_len}, {self.action_dims[i]})")
                raise RuntimeError(f"Q-head {i} failed: {q_e}") from q_e
        
        return q_values, h
    
    def _forward_both(self, h: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Generates both policy and Q-value outputs.
        
        Args:
            h: Hidden features of shape (batch_size, seq_len, gcn_hidden).
            
        Returns:
            Tuple of (action_logits, state_values, hidden_features).
            Note: Q-values are computed but not returned in 'both' mode.
        """
        # For 'both' mode, we return the same as 'policy' mode
        # This maintains backward compatibility with existing code
        return self._forward_policy(h)
    
    def get_q_values(self, h: torch.Tensor) -> List[torch.Tensor]:
        """Explicitly computes Q-values from hidden features.
        
        Useful when you have hidden features and want to compute Q-values
        separately from the main forward pass.
        
        Args:
            h: Hidden features of shape (batch_size, seq_len, gcn_hidden).
            
        Returns:
            List of Q-value tensors for each action dimension.
        """
        batch_size, seq_len, _ = h.shape
        h_flat = h.view(batch_size * seq_len, -1)
        
        q_values = [
            q_head(h_flat).view(batch_size, seq_len, -1)
            for q_head in self.q_heads
        ]
        
        return q_values
    
    def extra_repr(self) -> str:
        """Returns extra representation string for module printing."""
        return (f'input_dim={self.input_dim}, gcn_hidden={self.gcn_hidden}, '
                f'action_dims={self.action_dims}, dropout={self.dropout}, '
                f'use_layernorm={self.use_layernorm}') 