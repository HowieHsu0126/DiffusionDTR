"""PoG-BVE Q-network implementation.

This module implements the PoG-BVE Q-network that combines the Plan-on-Graph
model as a feature backbone with Branch Value Estimation heads for structured
multi-discrete action spaces.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpHead(nn.Module):
    """Multi-layer perceptron head for Q-value output.
    
    A flexible MLP implementation that can be used as Q-value heads in the
    BVE architecture. Supports configurable depth, normalization, dropout,
    and residual connections.
    
    Attributes:
        layers: Sequential container of MLP layers.
        use_residual: Whether to apply residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_residual: bool = False,
        activation: str = 'relu'
    ) -> None:
        """Initializes the MLP head.
        
        Args:
            input_dim: Input feature dimension.
            output_dim: Output dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of layers (including output layer).
            dropout: Dropout probability.
            use_layer_norm: Whether to use layer normalization.
            use_residual: Whether to use residual connections.
            activation: Activation function name.
            
        Raises:
            ValueError: If input parameters are invalid.
        """
        super().__init__()
        
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
            
        self.use_residual = use_residual and (input_dim == output_dim)
        
        # Build layers
        layers = []
        
        if num_layers == 1:
            # Single layer case
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Multi-layer case
            # Input layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation)
            ])
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    self._get_activation(activation)
                ])
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Returns activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation '{activation}'")
        
        return activations[activation.lower()]
    
    def _initialize_weights(self) -> None:
        """Initializes weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP head.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        output = self.layers(x)
        
        if self.use_residual:
            output = output + x
            
        return output


class SelfAttentionBlock(nn.Module):
    """Self-attention block for fusing branch hidden states.
    
    This module applies self-attention to fuse information between different
    branch representations in the BVE architecture.
    
    Attributes:
        attn: Multi-head attention module.
        norm: Layer normalization module.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 2, 
        dropout: float = 0.0
    ) -> None:
        """Initializes the self-attention block.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies self-attention with optional branch importance masking.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, embed_dim)``.
            attn_mask: Optional *branch-wise* importance mask broadcastable to
                ``(batch_size, seq_len)``.  Elements ==0 suppress attention for
                corresponding positions.

        Returns:
            Tensor of same shape as *x* with residual connection.
        """
        # PyTorch MultiheadAttention expects mask of shape (N, S) or (S, S)
        if attn_mask is not None:
            # Convert bool/float mask → additive mask where -inf blocks attention
            if attn_mask.dtype != torch.bool:
                attn_mask_bool = attn_mask == 0
            else:
                attn_mask_bool = ~attn_mask  # flip: True where blocked
            # Expand to (batch_size, seq_len)
            attn_mask_bool = attn_mask_bool.to(x.device)
        else:
            attn_mask_bool = None

        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask_bool)
        return self.norm(attn_out + x)


class LowRankLinear(nn.Module):
    """Low-rank approximation of a fully-connected layer.

    Decomposes the weight matrix **W ∈ ℝ^{out×in}** into the product of two
    smaller matrices **B ∈ ℝ^{out×r}** and **A ∈ ℝ^{r×in}** (with *r* « min(in,out)).
    This reduces parameter count from *in·out* to *r·(in+out)* whilst retaining
    a full-rank approximation when *r* ≥ rank(W).

    Args:
        in_features: Input dimension *in*.
        out_features: Output dimension *out*.
        rank: Bottleneck rank *r* (must be < min(in,out) to gain benefit).
        bias: Whether to include bias term.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()

        if rank <= 0 or rank >= min(in_features, out_features):
            raise ValueError("rank must be positive and smaller than both in_features and out_features")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Factors: (rank, in) and (out, rank) so that output =  x @ A^T @ B^T
        self.A = nn.Parameter(torch.Tensor(rank, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, rank))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier initialisation on factors
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in)
        # (B, in) @ (in, rank)^T  -> (B, rank) -> @ (rank, out)^T -> (B, out)
        z = x @ self.A.T  # (B, rank)
        out = z @ self.B.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:  # noqa: D401
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


class PogBveQNetwork(nn.Module):
    """PoG-BVE Q-network combining Plan-on-Graph backbone with BVE heads.
    
    This network uses a pre-trained Plan-on-Graph model as a feature backbone
    and adds Branch Value Estimation heads for structured action space modeling.
    The design enables efficient transfer learning from PoG to BVE while
    maintaining the hierarchical action structure.
    
    Attributes:
        pog_model: Plan-on-Graph backbone model.
        action_dims: List of action space dimensions.
        device: Computation device.
        q_head_hidden_dim: Hidden dimension for Q-heads.
        q_head_layers: Number of layers in Q-heads.
        share_q2: Whether to share Q2 head parameters.
        share_q3: Whether to share Q3 head parameters.
        use_attention: Whether to use attention for branch fusion.
        q1_head: Q1 branch head.
        q2_head: Q2 branch heads (shared or separate).
        q3_head: Q3 branch heads (shared or separate).
        attention_block: Optional attention block for branch fusion.
        q_head_rank: Rank for LowRankLinear heads.
    """
    
    def __init__(
        self,
        pog_model: nn.Module,
        action_dims: List[int],
        device: Union[str, torch.device] = 'cpu',
        q_head_hidden_dim: int = 64,
        q_head_layers: int = 1,
        q_head_layer_norm: bool = False,
        q_head_dropout: float = 0.0,
        q_head_residual: bool = False,
        q_head_rank: Optional[int] = None,
        # Deprecated parameters (kept for compatibility)
        share_q2: bool = True,
        share_q3: bool = True,
        use_attention: bool = False,
        attention_heads: int = 1,
        attention_dropout: float = 0.0,
        use_dynamic_weight: bool = False,
        return_fused_q: bool = False
    ) -> None:
        """Initializes the PoG-BVE Q-network.
        
        This network now supports arbitrary action dimensions using a dynamic
        flat Q-head approach similar to standard BVE, while maintaining
        PoG backbone features.
        
        Args:
            pog_model: Pre-trained Plan-on-Graph model.
            action_dims: List of action space dimensions (any length ≥ 2).
            device: Computation device.
            q_head_hidden_dim: Hidden dimension for Q-heads.
            q_head_layers: Number of layers in Q-heads.
            q_head_layer_norm: Whether to use layer norm in Q-heads.
            q_head_dropout: Dropout probability for Q-heads.
            q_head_residual: Whether to use residual connections in Q-heads.
            q_head_rank: Rank for LowRankLinear heads.
            
            Deprecated parameters (ignored):
                share_q2, share_q3: No longer used in flat design.
                use_attention: Attention not currently implemented.
                attention_heads, attention_dropout: Attention parameters.
                use_dynamic_weight, return_fused_q: Legacy parameters.
            
        Raises:
            ValueError: If input parameters are invalid.
        """
        super().__init__()
        
        # Validate inputs
        if len(action_dims) < 2:
            raise ValueError(
                f"PoG-BVE requires at least 2 action dimensions for branch decomposition, "
                f"got {len(action_dims)}. For single-action problems, use DQN instead."
            )
        if any(dim <= 0 for dim in action_dims):
            raise ValueError("All action dimensions must be positive")
        if q_head_hidden_dim <= 0:
            raise ValueError("q_head_hidden_dim must be positive")
        if q_head_layers < 1:
            raise ValueError("q_head_layers must be at least 1")
            
        # Store configuration
        self.pog_model = pog_model
        self.action_dims = action_dims
        self.num_branches = len(action_dims)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.q_head_hidden_dim = q_head_hidden_dim
        self.q_head_layers = q_head_layers
        self.q_head_rank = q_head_rank  # None ⇒ standard MLP head
        
        # Gradient checkpointing state
        self._gradient_checkpointing = False
        
        # ------------------------------------------------------------------
        # Robustly *infer* the dimensionality of PoG features instead of
        # assuming they match ``pog_model.gcn_hidden``.  Empirically the PoG
        # encoder may concatenate bidirectional LSTM states or apply other
        # feature-level transformations that double the hidden size (e.g.
        # 64 → 128), leading to runtime shape mismatches like:
        #     RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x128 and 64x7)
        # We therefore perform a one-step forward pass with a **dummy** input
        # to obtain the actual feature dimension returned by the PoG model.
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Ensure dummy tensors are on the same device as pog_model parameters
            pog_device = next(pog_model.parameters()).device
            _dummy_state = torch.zeros(1, 1, pog_model.input_dim, device=pog_device)
            _dummy_len = torch.ones(1, dtype=torch.long, device=pog_device)
            _empty_edge = torch.zeros(2, 0, dtype=torch.long, device=pog_device)
            _mask = torch.ones(1, 1, dtype=torch.bool, device=pog_device)
            # PoG forward in *policy* mode – returns (logits, state_values, gcn_z)
            _, _, _gcn_z = pog_model(_dummy_state, _dummy_len, _empty_edge, _mask, mode="policy")
            pog_output_dim = _gcn_z.size(-1)
        
        # Build Q-heads
        self._build_q_heads(
            pog_output_dim,
            q_head_hidden_dim,
            q_head_layers,
            q_head_layer_norm,
            q_head_dropout,
            q_head_residual,
            q_head_rank
        )
        
        # Optional attention for branch fusion
        if use_attention:
            self.attention_block = SelfAttentionBlock(
                embed_dim=pog_output_dim,
                num_heads=attention_heads,
                dropout=attention_dropout
            )
        else:
            self.attention_block = None
            
        # Move to device
        self.to(self.device)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        self._gradient_checkpointing = True
        # Also enable gradient checkpointing for the PoG backbone if it supports it
        if hasattr(self.pog_model, 'gradient_checkpointing_enable') and callable(self.pog_model.gradient_checkpointing_enable):
            self.pog_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        # Also disable gradient checkpointing for the PoG backbone if it supports it
        if hasattr(self.pog_model, 'gradient_checkpointing_disable') and callable(self.pog_model.gradient_checkpointing_disable):
            self.pog_model.gradient_checkpointing_disable()

    def _checkpointed_forward_pog(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through PoG backbone with optional gradient checkpointing."""
        if self._gradient_checkpointing and self.training:
            def pog_wrapper(state_inner):
                # PoG forward in policy mode to get features
                dummy_len = torch.ones(state_inner.size(0), dtype=torch.long, device=state_inner.device)
                empty_edge = torch.zeros(2, 0, dtype=torch.long, device=state_inner.device)
                mask = torch.ones(state_inner.size(0), state_inner.size(1), dtype=torch.bool, device=state_inner.device)
                _, _, gcn_z = self.pog_model(state_inner, dummy_len, empty_edge, mask, mode="policy")
                return gcn_z
            return torch.utils.checkpoint.checkpoint(pog_wrapper, state, use_reentrant=False)
        else:
            # Standard forward pass
            dummy_len = torch.ones(state.size(0), dtype=torch.long, device=state.device)
            empty_edge = torch.zeros(2, 0, dtype=torch.long, device=state.device)
            mask = torch.ones(state.size(0), state.size(1), dtype=torch.bool, device=state.device)
            _, _, gcn_z = self.pog_model(state, dummy_len, empty_edge, mask, mode="policy")
            return gcn_z
    
    def _build_q_heads(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_layer_norm: bool,
        dropout: float,
        use_residual: bool,
        rank: Optional[int] = None
    ) -> None:
        """Builds the Q-value head networks dynamically for any number of action dimensions.
        
        This method creates a flat Q-head for each branch level, similar to the standard
        BVE implementation but with PoG-specific features like LowRankLinear support.
        
        Args:
            input_dim: Input dimension from PoG backbone.
            hidden_dim: Hidden dimension for MLP heads.
            num_layers: Number of layers in each head.
            use_layer_norm: Whether to use layer normalization.
            dropout: Dropout probability.
            use_residual: Whether to use residual connections.
            rank: Rank for LowRankLinear heads.
        """
        # Helper to create an output head depending on *rank*
        def _make_head(out_dim: int):
            if rank is not None:
                return LowRankLinear(input_dim, out_dim, rank)
            else:
                return MlpHead(
                    input_dim=input_dim,
                    output_dim=out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    use_residual=use_residual,
                )

        # Dynamic branch Q-value heads (similar to standard BVE)
        # Create Q heads for each branch level dynamically
        self.q_heads = nn.ModuleList()
        
        for i in range(self.num_branches):
            # Calculate the output dimension for this branch
            # Branch i produces Q-values for all combinations of actions from branches 0 to i
            output_dim = 1
            for j in range(i + 1):
                output_dim *= self.action_dims[j]
            
            # Create head that produces all combinations for this branch level
            head = _make_head(output_dim)
            self.q_heads.append(head)
    
    def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the PoG-BVE network.
        
        Processes input states through the PoG backbone and then through
        BVE heads to produce hierarchical Q-values for any number of action dimensions.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim) or
                  (batch_size, seq_len, state_dim).
                  
        Returns:
            A list of Q-value tensors, where the i-th tensor contains Q-values
            for all action combinations from branches 0 to i.
            For example, with action_dims=[7, 6, 6]:
            - q[0]: Q1 values of shape (batch_size, 7).
            - q[1]: Q2 values of shape (batch_size, 7, 6).  
            - q[2]: Q3 values of shape (batch_size, 7, 6, 6).
        """
        # 🔧 CRITICAL FIX: Ensure input tensor contiguity for PoG model compatibility
        if not state.is_contiguous():
            state = state.contiguous()
        
        # Handle input format conversion
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
            lengths = torch.ones(state.size(0), dtype=torch.long, device=state.device)
        elif state.dim() == 3:
            lengths = torch.full((state.size(0),), state.size(1), dtype=torch.long, device=state.device)
        else:
            raise ValueError("state must be (batch_size, state_dim) or (batch_size, seq_len, state_dim)")
        
        batch_size, seq_len, _ = state.shape
        
        # 🔧 CRITICAL FIX: Ensure all tensors are contiguous for PoG model
        state = state.contiguous()
        lengths = lengths.contiguous() 
        
        # Create dummy edge index and mask for PoG model with contiguity
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=state.device).contiguous()
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=state.device).contiguous()
        
        # PoG forward pass with gradient checkpointing support
        if self._gradient_checkpointing and self.training:
            def pog_wrapper(state_inner):
                # PoG forward in policy mode to get features
                _, _, gcn_z, *_ = self.pog_model(state_inner, lengths, edge_index, mask, mode='policy')
                # Extract features from the last timestep
                last_indices = lengths - 1
                batch_indices = torch.arange(batch_size, device=state_inner.device)
                return gcn_z[batch_indices, last_indices]
            
            h = torch.utils.checkpoint.checkpoint(pog_wrapper, state, use_reentrant=False)
        else:
            # Standard forward pass through PoG model
            _, _, gcn_z, *_ = self.pog_model(state, lengths, edge_index, mask, mode='policy')
            
            # Extract features from the last timestep
            last_indices = lengths - 1
            batch_indices = torch.arange(batch_size, device=state.device)
            h = gcn_z[batch_indices, last_indices]
        
        # Compute Q-values for all branches dynamically
        return self._compute_branch_q_values(h, batch_size)
    
    def _compute_branch_q_values(
        self, 
        h: torch.Tensor, 
        batch_size: int
    ) -> List[torch.Tensor]:
        """Computes Q-values for all branches dynamically.
        
        Args:
            h: Hidden features from PoG backbone.
            batch_size: Batch size.
            
        Returns:
            List of Q-value tensors for each branch level.
        """
        q_values = []
        
        for i in range(self.num_branches):
            # Get flat Q-values for this branch
            q_flat = self.q_heads[i](h)  # (B, A1*A2*...*A_i)
            
            # Reshape to proper dimensions
            shape = [batch_size] + self.action_dims[:i+1]
            q_reshaped = q_flat.view(shape)
            
            q_values.append(q_reshaped)
        
        return q_values
    
    def q_value(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        use_fused_q: bool = False
    ) -> torch.Tensor:
        """Computes Q(s,a) for specific state-action pairs.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or
                  (batch_size, seq_len, state_dim).
            action: Action tensor of shape (batch_size, num_branches).
            use_fused_q: Ignored for compatibility (always uses final branch).
            
        Returns:
            Q-values of shape (batch_size,).
        """
        if action.dim() != 2 or action.size(-1) != self.num_branches:
            raise ValueError(f"Expected action shape (batch_size, {self.num_branches}), got {action.shape}")
        
        # Forward pass to get all Q-values
        q_values = self.forward(state)
        
        # Use the final branch Q-values (contains all action combinations)
        q_final = q_values[-1]  # Shape: (batch_size, action_dims[0], action_dims[1], ..., action_dims[n-1])
        
        # Compute linearised indices to gather Q-values
        batch_size = state.size(0) if state.dim() == 2 else state.size(0)
        flat_idx = torch.zeros(batch_size, dtype=torch.long, device=state.device)
        
        # Calculate flattened index for each action combination
        stride = 1
        for i in range(self.num_branches - 1, -1, -1):
            flat_idx += action[:, i] * stride
            stride *= self.action_dims[i]
        
        # Flatten Q-values and gather using computed indices
        q_flat = q_final.view(batch_size, -1)  # (B, A1*A2*...*An)
        q_result = torch.gather(q_flat, 1, flat_idx.unsqueeze(1)).squeeze(1)  # (B,)

        return q_result
    
    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """Selects greedy actions using dynamic branch-wise maximization.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or
                  (batch_size, seq_len, state_dim).
                  
        Returns:
            Greedy actions of shape (batch_size, num_branches).
        """
        with torch.no_grad():
            q_values = self.forward(state)
            batch_size = state.size(0) if state.dim() == 2 else state.size(0)
            
            # Use the final branch Q-values for greedy selection
            q_final = q_values[-1]  # Shape: (batch_size, action_dims[0], ..., action_dims[n-1])
            
            # Flatten and find best action combination
            q_flat = q_final.view(batch_size, -1)  # (B, A1*A2*...*An)
            best_flat_idx = q_flat.argmax(dim=1)  # (B,)
            
            # Decode flat indices to multi-dimensional actions
            actions = []
            remaining_idx = best_flat_idx
            
            for i in range(self.num_branches - 1, -1, -1):
                stride = 1
                for j in range(i + 1, self.num_branches):
                    stride *= self.action_dims[j]
                
                action_i = remaining_idx // stride
                remaining_idx = remaining_idx % stride
                actions.append(action_i)
            
            # Reverse to get correct order (branch 0, branch 1, ...)
            actions.reverse()
            result = torch.stack(actions, dim=1)  # (B, num_branches)
            
        return result
    
    def beam_search(
        self, 
        state: torch.Tensor, 
        beam_width: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs beam search for top-k action sequences.
        
        Args:
            state: State tensor.
            beam_width: Number of beams to maintain.
            
        Returns:
            Tuple of (actions, q_values) where actions has shape
            (batch_size, beam_width, num_branches) and q_values has shape
            (batch_size, beam_width).
        """
        with torch.no_grad():
            # Forward once to obtain final Q tensor and flatten
            q_values = self.forward(state)
            q_final = q_values[-1]  # (B, A1, A2, ..., An)

            batch_size = q_final.size(0)
            
            # Calculate total action space size
            total_actions = 1
            for dim in self.action_dims:
                total_actions *= dim

            q_flat = q_final.view(batch_size, -1)  # (B, total_actions)

            topk_q, topk_idx = torch.topk(q_flat, k=min(beam_width, total_actions), dim=1)

            # Decode flat indices to multi-dimensional actions
            actions_list = []
            remaining_idx = topk_idx
            
            for i in range(self.num_branches - 1, -1, -1):
                stride = 1
                for j in range(i + 1, self.num_branches):
                    stride *= self.action_dims[j]
                
                action_i = remaining_idx // stride
                remaining_idx = remaining_idx % stride
                actions_list.append(action_i)
            
            # Reverse to get correct order and stack
            actions_list.reverse()
            actions = torch.stack(actions_list, dim=-1)  # (B, k, num_branches)

            return actions.long(), topk_q
    
    def freeze_pog_backbone(self) -> None:
        """Freezes the PoG backbone parameters for fine-tuning only Q-heads."""
        for param in self.pog_model.parameters():
            param.requires_grad = False
    
    def unfreeze_pog_backbone(self) -> None:
        """Unfreezes the PoG backbone parameters for end-to-end training."""
        for param in self.pog_model.parameters():
            param.requires_grad = True
    
    def get_pog_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extracts features from the PoG backbone.
        
        Args:
            state: Input state tensor.
            
        Returns:
            PoG features of shape (batch_size, feature_dim).
        """
        with torch.no_grad():
            # Handle input format
            if state.dim() == 2:
                state = state.unsqueeze(1)
                lengths = torch.ones(state.size(0), dtype=torch.long, device=state.device)
            else:
                lengths = torch.full((state.size(0),), state.size(1), dtype=torch.long, device=state.device)
            
            batch_size, seq_len, _ = state.shape
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=state.device)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=state.device)
            
            _, _, gcn_z, *_ = self.pog_model(state, lengths, edge_index, mask, mode='policy')
            
            last_indices = lengths - 1
            batch_indices = torch.arange(batch_size, device=state.device)
            features = gcn_z[batch_indices, last_indices]
            
        return features
    
    def compute_branch_q_values(
        self, 
        state: torch.Tensor, 
        actions: torch.Tensor, 
        branch_idx: int
    ) -> torch.Tensor:
        """Computes Q-values for specific actions at a given branch.
        
        This method computes Q-values for sampled actions at a specific branch level,
        which is needed for CQL loss computation during training. Compatible with
        the standard BVE interface.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (batch_size, seq_len, state_dim).
            actions: Action indices for the specified branch, shape (batch_size, n_samples).
            branch_idx: Index of the branch to compute Q-values for (0-indexed).
            
        Returns:
            Q-values tensor of shape (batch_size, n_samples).
            
        Raises:
            ValueError: If branch_idx is out of range or input shapes are invalid.
        """
        if not 0 <= branch_idx < self.num_branches:
            raise ValueError(f"branch_idx must be in [0, {self.num_branches-1}], got {branch_idx}")
        if actions.dim() != 2:
            raise ValueError(f"Expected actions shape (batch_size, n_samples), got {actions.shape}")
        
        batch_size, n_samples = actions.shape
        
        # Validate action indices are within bounds
        max_action = actions.max().item()
        min_action = actions.min().item()
        if min_action < 0 or max_action >= self.action_dims[branch_idx]:
            raise ValueError(
                f"Invalid actions for branch {branch_idx}: "
                f"range [{min_action}, {max_action}], valid [0, {self.action_dims[branch_idx]-1}]"
            )
        
        # Forward pass to get all Q-values (similar to standard BVE)
        q_values = self.forward(state)
        
        # Get Q-values for the specified branch
        branch_q_values = q_values[branch_idx]  # Shape varies by branch level
        
        # For branch 0: shape is (batch_size, action_dims[0])
        # We can directly index with actions
        if branch_idx == 0:
            # Expand actions for batch indexing
            batch_indices = torch.arange(batch_size, device=state.device).unsqueeze(1)  # (B, 1)
            batch_indices = batch_indices.expand(-1, n_samples)  # (B, n_samples)
            
            # Select Q-values using advanced indexing
            selected_q = branch_q_values[batch_indices, actions]  # (B, n_samples)
            return selected_q
        
        # For higher branches, we need to handle multi-dimensional indexing
        # Since we only have actions for the current branch, we'll use the greedy
        # actions for previous branches to condition the Q-values
        else:
            # Get greedy actions for previous branches
            with torch.no_grad():
                greedy_actions = self.greedy_action(state)  # (B, num_branches)
                prev_actions = greedy_actions[:, :branch_idx]  # (B, branch_idx)
            
            # Build indexing tuple for advanced indexing
            batch_indices = torch.arange(batch_size, device=state.device)
            
            # Start with batch indices
            index_list = [batch_indices]
            
            # Add previous action indices
            for i in range(branch_idx):
                index_list.append(prev_actions[:, i])
            
            # For each sample, we need to select Q-values
            result_list = []
            for sample_idx in range(n_samples):
                # Add current branch action for this sample
                current_sample_actions = actions[:, sample_idx]  # (B,)
                sample_index_list = index_list + [current_sample_actions]
                
                # Select Q-values for this sample
                selected_q = branch_q_values[tuple(sample_index_list)]  # (B,)
                result_list.append(selected_q)
            
            # Stack results
            selected_q = torch.stack(result_list, dim=1)  # (B, n_samples)
            return selected_q 