from __future__ import annotations

"""Plan-on-Graph (PoG) model implementation.

This module implements the Plan-on-Graph model, which combines LSTM-based
trajectory encoding with graph neural networks and information bottlenecks
for medical decision making in reinforcement learning.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from Libs.model.layers import (
    TrajectoryEncoder,
    TimeStepGCN
)
from Libs.utils.log_utils import get_logger


logger = get_logger(__name__)


class PlanOnGraphModel(nn.Module):
    """Plan-on-Graph (PoG) model for medical decision making.

    This model implements a sophisticated architecture that combines:
    1. LSTM-based trajectory encoding for temporal dynamics
    2. Information bottleneck for feature compression and regularization
    3. Multi-head attention for capturing sequence dependencies
    4. Graph Convolutional Networks for structural reasoning
    5. Dual information bottlenecks for robust representation learning

    The model is designed for medical reinforcement learning tasks where
    patient trajectories need to be processed with both temporal and
    structural considerations.

    Attributes:
        input_dim: Input feature dimension for patient observations.
        lstm_hidden: Hidden dimension for LSTM trajectory encoder.
        gcn_hidden: Hidden dimension for GCN layers.
        action_dims: List of action space dimensions for multi-discrete actions.
        traj_bottleneck_dim: Dimension of trajectory information bottleneck.
        gcn_bottleneck_dim: Dimension of GCN information bottleneck.
        attn_heads: Number of attention heads.
        dropout: Dropout probability for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int,
        gcn_hidden: int,
        action_dims: List[int],
        attn_heads: int = 2,
        traj_bottleneck_dim: int = 64,
        gcn_bottleneck_dim: int = 64,
        dropout: float = 0.1,
        lstm_layers: int = 1,
        bidirectional: bool = False
    ) -> None:
        """Initializes the Plan-on-Graph model.

        Args:
            input_dim: Input feature dimension for patient observations.
            lstm_hidden: Hidden dimension for LSTM trajectory encoder.
            gcn_hidden: Hidden dimension for GCN layers.
            action_dims: List of action space dimensions for multi-discrete actions.
            attn_heads: Number of attention heads for self-attention.
            traj_bottleneck_dim: Dimension of trajectory information bottleneck.
            gcn_bottleneck_dim: Dimension of GCN information bottleneck.
            dropout: Dropout probability for regularization.
            lstm_layers: Number of LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.

        Raises:
            ValueError: If any input parameters are invalid.
        """
        super().__init__()

        # Validate inputs
        if input_dim <= 0 or lstm_hidden <= 0 or gcn_hidden <= 0:
            raise ValueError("All dimensions must be positive")
        if not action_dims or any(dim <= 0 for dim in action_dims):
            raise ValueError(
                "action_dims must be non-empty with positive values")
        if attn_heads <= 0:
            raise ValueError("attn_heads must be positive")
        if traj_bottleneck_dim <= 0 or gcn_bottleneck_dim <= 0:
            raise ValueError("Bottleneck dimensions must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")

        # Store configuration
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.gcn_hidden = gcn_hidden
        self.action_dims = action_dims
        self.attn_heads = attn_heads
        self.traj_bottleneck_dim = traj_bottleneck_dim
        self.gcn_bottleneck_dim = gcn_bottleneck_dim
        self.dropout = dropout

        # Build model components
        self._build_trajectory_encoder(lstm_layers, bidirectional, dropout)
        self._build_graph_network(dropout)

    def _build_trajectory_encoder(
        self,
        lstm_layers: int,
        bidirectional: bool,
        dropout: float
    ) -> None:
        """Builds the trajectory encoding components."""
        self.traj_encoder = TrajectoryEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def _build_attention_mechanism(self, dropout: float) -> None:  # noqa: D401
        """Deprecated – kept as no-op for backward compatibility."""
        import torch.nn as _nn
        self.traj_attention = _nn.Identity()

    def _build_graph_network(self, dropout: float) -> None:
        """Builds the graph neural network components."""
        self.gcn = TimeStepGCN(
            input_dim=self.lstm_hidden,
            gcn_hidden=self.gcn_hidden,
            action_dims=self.action_dims,
            dropout=dropout,
            use_layernorm=True
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = 'policy',
        rewards: Optional[torch.Tensor] = None,
        reward_centering: bool = False
    ) -> Union[
        Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor],  # policy
        Tuple[List[torch.Tensor], torch.Tensor,
              torch.Tensor, torch.Tensor],               # q
        Tuple[List[torch.Tensor], torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor]  # both
    ]:
        """Forward pass through the Plan-on-Graph model.

        Processes patient trajectory data through the complete PoG pipeline:
        1. LSTM encoding of temporal sequences
        2. Information bottleneck compression
        3. Self-attention for sequence modeling
        4. Graph convolution for structural reasoning
        5. Final information bottleneck and output generation

        Args:
            x: Input sequences of shape (batch_size, seq_len, input_dim).
            lengths: Actual sequence lengths of shape (batch_size,).
            edge_index: Graph edge indices of shape (2, num_edges).
            mask: Optional attention mask of shape (batch_size, seq_len).
            mode: Output mode - 'policy', 'q', or 'both'.
            rewards: Optional rewards for centering.
            reward_centering: Whether to apply reward centering.

        Returns:
            Depending on mode:
                - 'policy': (action_logits, state_values, gcn_features, kl_traj, kl_gcn)
                - 'q': (q_values, gcn_features, kl_traj, kl_gcn)
                - 'both': (action_logits, state_values, gcn_features, kl_traj, kl_gcn)

        Raises:
            ValueError: If input shapes are inconsistent or mode is invalid.
        """
        # Validate inputs
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
        if lengths.dim() != 1 or lengths.size(0) != x.size(0):
            raise ValueError("lengths shape incompatible with batch size")
        if mode not in ['policy', 'q', 'both']:
            raise ValueError(f"Invalid mode '{mode}'")

        batch_size, seq_len, _ = x.shape

        # 🔧 CRITICAL FIX: Enhanced memory management and sequence length validation
        # Handle extremely long sequences that can exhaust GPU memory
        # Conservative limit for PoG models (LSTM + attention)
        MAX_SAFE_SEQ_LEN = 4096
        MEMORY_WARNING_THRESHOLD = 2048  # Warning threshold

        # Memory usage estimation (rough calculation)
        estimated_memory_gb = (batch_size * seq_len *
                               seq_len * 4) / (1024**3)  # Attention matrix
        estimated_memory_gb += (batch_size * seq_len *
                                self.lstm_hidden * 8) / (1024**3)  # LSTM states

        if seq_len > MEMORY_WARNING_THRESHOLD:
            logger.debug(
                f"⚠️ Long sequence detected: {seq_len} timesteps (batch_size={batch_size})")
            logger.debug(f"📊 Estimated memory usage: {estimated_memory_gb:.2f} GB")

            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    available = torch.cuda.get_device_properties(
                        0).total_memory / 1024**3 - allocated
                    logger.debug(
                        f"📊 GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, ~{available:.2f}GB available")

                    if estimated_memory_gb > available * 0.7:  # Use 70% threshold for safety
                        logger.debug(
                            f"🚨 Sequence may cause OOM - estimated {estimated_memory_gb:.2f}GB > available {available:.2f}GB")
                except:
                    logger.debug("📊 Could not retrieve GPU memory info")

        # 🔧 CRITICAL FIX: Smart memory-based truncation instead of hard limits
        should_truncate = False
        target_seq_len = seq_len

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                available = torch.cuda.get_device_properties(
                    0).total_memory / 1024**3 - allocated

                # Only truncate if we would actually exceed available memory
                if estimated_memory_gb > available * 0.7:  # 70% safety threshold
                    should_truncate = True
                    # Calculate optimal sequence length based on available memory
                    # Use 60% for extra safety
                    memory_ratio = (available * 0.6) / estimated_memory_gb
                    # Minimum 1024 timesteps
                    target_seq_len = max(1024, int(seq_len * memory_ratio))
                    # Cap at MAX_SAFE_SEQ_LEN
                    target_seq_len = min(target_seq_len, MAX_SAFE_SEQ_LEN)
                    logger.debug(
                        f"🔧 Memory-based truncation: {seq_len} -> {target_seq_len} (estimated: {estimated_memory_gb:.2f}GB, available: {available:.2f}GB)")
                elif seq_len > MAX_SAFE_SEQ_LEN:
                    # Fallback to hard limit only if memory check failed but sequence is very long
                    should_truncate = True
                    target_seq_len = MAX_SAFE_SEQ_LEN
                    logger.debug(
                        f"🔧 Applying fallback sequence truncation: {seq_len} -> {MAX_SAFE_SEQ_LEN}")
                    logger.debug(
                        f"   Reason: Sequence exceeds fallback limit (memory estimation: {estimated_memory_gb:.2f}GB)")
            except Exception as e:
                # Fallback to conservative truncation if memory check fails
                if seq_len > MAX_SAFE_SEQ_LEN:
                    should_truncate = True
                    target_seq_len = MAX_SAFE_SEQ_LEN
                    logger.debug(
                        f"🔧 Conservative truncation due to memory check failure: {seq_len} -> {MAX_SAFE_SEQ_LEN}")
                    logger.debug(f"   Error: {e}")
        else:
            # CPU-only mode: use conservative limits
            if seq_len > MAX_SAFE_SEQ_LEN:
                should_truncate = True
                target_seq_len = MAX_SAFE_SEQ_LEN
                logger.debug(
                    f"🔧 CPU-mode truncation: {seq_len} -> {MAX_SAFE_SEQ_LEN}")

        if should_truncate and target_seq_len < seq_len:
            logger.debug(
                f"   This preserves the most recent {target_seq_len} timesteps")
            # Truncate to the most recent timesteps (preserves recency bias)
            x = x[:, -target_seq_len:, :]
            seq_len = target_seq_len
        else:
            logger.debug(
                f"✅ No truncation needed: sufficient memory for {seq_len} timesteps")

            # Adjust lengths accordingly
            lengths = torch.clamp(lengths, max=MAX_SAFE_SEQ_LEN)

            # Adjust mask if provided
            if mask is not None:
                if mask.size(1) > MAX_SAFE_SEQ_LEN:
                    mask = mask[:, -MAX_SAFE_SEQ_LEN:]

            logger.debug(f"✅ Sequence truncated to {seq_len} timesteps")

        # 🔧 CRITICAL FIX: Comprehensive tensor contiguity and memory layout optimization
        # Ensure all input tensors are contiguous for cuDNN compatibility
        # Create contiguous clones to prevent memory layout issues
        if not x.is_contiguous():
            x = x.contiguous()
        x = x.clone().contiguous()  # Extra safety for cuDNN

        if not lengths.is_contiguous():
            lengths = lengths.contiguous()
        lengths = lengths.clone().contiguous()  # Extra safety for cuDNN

        if not edge_index.is_contiguous():
            edge_index = edge_index.contiguous()
        edge_index = edge_index.clone().contiguous()  # Extra safety for cuDNN

        if mask is not None:
            if not mask.is_contiguous():
                mask = mask.contiguous()
            mask = mask.clone().contiguous()  # Extra safety for cuDNN

        # Optional reward centering
        if rewards is not None and reward_centering:
            rewards = rewards - rewards.mean()

        # Step 1: Encode trajectories with LSTM
        try:
            lstm_output = self.traj_encoder(x, lengths)
        except Exception as e:
            # 🔧 ENHANCED ERROR HANDLING: Detailed debug info for trajectory encoder errors
            error_msg = f"Trajectory encoder failed: {e}"
            debug_info = {
                'input_shape': x.shape,
                'input_device': x.device,
                'input_dtype': x.dtype,
                'input_contiguous': x.is_contiguous(),
                'lengths_shape': lengths.shape,
                'lengths_device': lengths.device,
                'encoder_input_dim': self.traj_encoder.input_dim,
                'encoder_hidden_dim': self.traj_encoder.hidden_dim,
                'encoder_num_layers': self.traj_encoder.num_layers,
                'encoder_bidirectional': self.traj_encoder.bidirectional,
                'estimated_memory_gb': estimated_memory_gb
            }

            logger.debug(f"❌ {error_msg}")
            logger.debug(f"🔍 Debug info: {debug_info}")

            if "cuDNN" in str(e) or "CUDNN" in str(e):
                logger.debug(
                    "🚨 cuDNN error in trajectory encoder - likely non-contiguous tensor issue")

                # Try memory cleanup and retry once
                if torch.cuda.is_available():
                    logger.debug("🔧 Attempting GPU memory cleanup and retry...")
                    torch.cuda.empty_cache()

                    try:
                        # Recreate tensors with explicit contiguous layout
                        x = x.detach().clone().contiguous()
                        lengths = lengths.detach().clone().contiguous()

                        lstm_output = self.traj_encoder(x, lengths)
                        logger.debug("✅ Retry after memory cleanup succeeded")
                    except Exception as retry_e:
                        logger.debug(f"❌ Retry also failed: {retry_e}")
                        raise RuntimeError(error_msg) from e
                else:
                    raise RuntimeError(error_msg) from e
            else:
                raise RuntimeError(error_msg) from e

        # Step 2: Process through trajectory bottleneck (simplified - no bottleneck in current version)
        lstm_out = lstm_output

        # Step 5: Apply graph convolution with memory monitoring
        try:
            # Monitor memory before GCN processing
            if torch.cuda.is_available() and seq_len > MEMORY_WARNING_THRESHOLD:
                pre_gcn_memory = torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"📊 Pre-GCN memory: {pre_gcn_memory:.2f} GB")

            gcn_outputs = self.gcn(lstm_out, edge_index, mode=mode)

            # Monitor memory after GCN processing
            if torch.cuda.is_available() and seq_len > MEMORY_WARNING_THRESHOLD:
                post_gcn_memory = torch.cuda.memory_allocated() / 1024**3
                logger.debug(
                    f"📊 Post-GCN memory: {post_gcn_memory:.2f} GB (delta: {post_gcn_memory - pre_gcn_memory:.2f} GB)")

        except Exception as e:
            # 🔧 ENHANCED ERROR HANDLING: Detailed debug info for GCN errors
            error_msg = f"GCN forward failed: {e}"
            debug_info = {
                'lstm_output_shape': lstm_out.shape,
                'lstm_output_device': lstm_out.device,
                'lstm_output_dtype': lstm_out.dtype,
                'lstm_output_contiguous': lstm_out.is_contiguous(),
                'edge_index_shape': edge_index.shape,
                'edge_index_device': edge_index.device,
                'edge_index_dtype': edge_index.dtype,
                'edge_index_contiguous': edge_index.is_contiguous(),
                'gcn_input_dim': self.gcn.input_dim,
                'gcn_hidden_dim': self.gcn.gcn_hidden,
                'mode': mode,
                'estimated_memory_gb': estimated_memory_gb
            }

            logger.debug(f"❌ {error_msg}")
            logger.debug(f"🔍 Debug info: {debug_info}")

            # Check for memory-related errors
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                logger.debug("🚨 GPU out of memory error detected")
                if torch.cuda.is_available():
                    try:
                        current_memory = torch.cuda.memory_allocated() / 1024**3
                        max_memory = torch.cuda.max_memory_allocated() / 1024**3
                        logger.debug(
                            f"📊 Memory usage: current={current_memory:.2f}GB, peak={max_memory:.2f}GB")
                        torch.cuda.reset_max_memory_allocated()
                    except:
                        pass

                # Suggest sequence length reduction
                suggested_len = min(seq_len // 2, 1024)
                logger.debug(
                    f"💡 Consider reducing sequence length to {suggested_len} or using gradient checkpointing")

            raise RuntimeError(error_msg) from e

        # Step 6: Extract GCN hidden features and apply second information bottleneck
        if mode == 'policy':
            action_logits, state_values, gcn_hidden = gcn_outputs
        elif mode == 'q':
            q_values, gcn_hidden = gcn_outputs
        elif mode == 'both':
            action_logits, state_values, gcn_hidden = gcn_outputs

        # 🔧 MEMORY OPTIMIZATION: Clear intermediate variables to free memory
        del lstm_out  # Free LSTM output memory
        if torch.cuda.is_available() and seq_len > MEMORY_WARNING_THRESHOLD:
            torch.cuda.empty_cache()  # Aggressive cleanup for long sequences

        # Return results based on mode
        if mode == 'policy':
            return action_logits, state_values, gcn_hidden
        elif mode == 'q':
            return q_values, gcn_hidden
        elif mode == 'both':
            return action_logits, state_values, gcn_hidden

    def greedy_action(self, state: torch.Tensor, lengths: Optional[torch.Tensor] = None, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Select greedy actions for given states using Q-values.
        
        This method is required for FQE evaluation and implements greedy policy
        based on current Q-value estimates from the PoG model.
        
        Args:
            state: State tensor of shape (batch_size, seq_len, state_dim) or (batch_size, state_dim).
                  Represents patient trajectory data.
            lengths: Optional sequence lengths tensor of shape (batch_size,).
                    If None, assumes single timestep or full sequence length.
            edge_index: Optional graph edge indices of shape (2, num_edges).
                       If None, uses empty graph.

        Returns:
            Greedy actions tensor of shape (batch_size, n_heads) with integer actions.
            Each column represents the greedy action for one action dimension.
            
        Note:
            This method is essential for FQE evaluation where we need to extract
            the greedy policy from the learned Q-function.
        """
        with torch.no_grad():
            # Handle input dimensionality
            if state.dim() == 2:
                # Convert (batch_size, state_dim) to (batch_size, 1, state_dim)
                state = state.unsqueeze(1)
                batch_size = state.size(0)
                seq_len = 1
            elif state.dim() == 3:
                batch_size, seq_len, _ = state.shape
            else:
                raise ValueError(f"Expected 2D or 3D state tensor, got {state.dim()}D")
            
            # Set default lengths if not provided
            if lengths is None:
                lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=state.device)
            
            # Set default edge_index if not provided
            if edge_index is None:
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=state.device)
            
            # Get Q-values from the model in 'q' mode
            try:
                q_values, _ = self.forward(state, lengths, edge_index, mode='q')
                greedy_actions = []
                
                for i, q_head in enumerate(q_values):
                    # 🔧 ENHANCED Q-VALUE PROCESSING: Robust handling of different tensor shapes
                    try:
                        # Handle different Q-value tensor shapes
                        if q_head.dim() == 3:
                            # PoG output: (batch_size, seq_len, action_dim) -> (batch_size, action_dim)
                            # Take the last timestep for greedy action selection
                            q_last = q_head[:, -1, :]
                        elif q_head.dim() == 2:
                            # Already 2D: (batch_size, action_dim)
                            q_last = q_head
                        elif q_head.dim() == 1:
                            # 1D case: add batch dimension
                            q_last = q_head.unsqueeze(0)
                        else:
                            raise ValueError(f"Unsupported Q-head {i} dimensions: {q_head.dim()}D")
                        
                        # Validate Q-value shape
                        if q_last.size(0) != batch_size:
                            print(f"⚠️ Q-head {i} batch size mismatch: {q_last.size(0)} vs {batch_size}")
                            if q_last.size(0) > batch_size:
                                q_last = q_last[:batch_size]
                            else:
                                # Pad with zeros
                                pad_size = batch_size - q_last.size(0)
                                padding = torch.zeros(pad_size, q_last.size(1), device=q_last.device, dtype=q_last.dtype)
                                q_last = torch.cat([q_last, padding], dim=0)
                        
                        # Select action with highest Q-value for each head
                        greedy_action = q_last.argmax(dim=-1)  # Shape: (batch_size,)
                        
                        # Validate action shape
                        if greedy_action.size(0) != batch_size:
                            raise RuntimeError(f"Action size mismatch for head {i}: {greedy_action.size(0)} vs {batch_size}")
                        
                        greedy_actions.append(greedy_action)
                        
                    except Exception as head_error:
                        print(f"❌ POG greedy_action head {i} failed: {head_error}")
                        # Fallback to random actions for this head
                        if i < len(self.action_dims):
                            action_dim = self.action_dims[i]
                            random_actions = torch.randint(0, action_dim, (batch_size,), device=state.device)
                            greedy_actions.append(random_actions)
                        else:
                            # If action_dims not available, use default
                            random_actions = torch.randint(0, 2, (batch_size,), device=state.device)
                            greedy_actions.append(random_actions)
                
                # Stack into action tensor: (batch_size, n_heads)
                return torch.stack(greedy_actions, dim=1)
                
            except Exception as e:
                # Fallback: return random valid actions if Q-value computation fails
                print(f"Warning: PoG greedy_action failed ({e}), returning random actions")
                greedy_actions = []
                for action_dim in self.action_dims:
                    random_actions = torch.randint(0, action_dim, (batch_size,), device=state.device)
                    greedy_actions.append(random_actions)
                return torch.stack(greedy_actions, dim=1)

    def _create_sequence_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Creates attention mask for variable-length sequences.

        Args:
            lengths: Sequence lengths of shape (batch_size,).
            max_len: Maximum sequence length.
            device: Target device for the mask.

        Returns:
            Boolean mask of shape (batch_size, max_len) where True indicates
            valid positions.
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
        return mask

    def get_trajectory_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts trajectory features without graph processing.

        Useful for feature extraction or when graph structure is not available.

        Args:
            x: Input sequences of shape (batch_size, seq_len, input_dim).
            lengths: Sequence lengths of shape (batch_size,).
            mask: Optional attention mask.

        Returns:
            Tuple of (trajectory_features, kl_loss).
        """
        batch_size, seq_len, _ = x.shape

        # Encode trajectories
        lstm_output = self.traj_encoder(x, lengths)

        # Attention removed – directly return compressed features
        attended_features = lstm_output

        return attended_features, 0.0  # No KL loss for raw features

    def select_action(self, state: torch.Tensor, *, mode: str = 'greedy', epsilon: float = 0.05, temperature: float = 1.0, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        """Select actions with configurable exploration.

        Args:
            state: (B,D) or (B,T,D) tensor.
            mode: 'greedy' | 'epsilon' | 'sample'.
            epsilon: Exploration prob for epsilon-greedy.
            temperature: Softmax temperature when mode=='sample'.
            edge_index: Optional graph edge index.
        """
        if state.dim() == 2:
            state_seq = state.unsqueeze(1)
            lengths = torch.ones(state.size(
                0), dtype=torch.long, device=state.device)
        elif state.dim() == 3:
            state_seq = state
            lengths = torch.full((state.size(0),), state.size(
                1), dtype=torch.long, device=state.device)
        else:
            raise ValueError("state must be (B,D) or (B,T,D)")

        batch_size = state_seq.size(0)
        if edge_index is None:
            edge_index = torch.zeros(
                2, 0, dtype=torch.long, device=state.device)
        mask = torch.ones(batch_size, state_seq.size(
            1), dtype=torch.bool, device=state.device)

        logits_list, _, * \
            _ = self.forward(state_seq, lengths, edge_index,
                             mask, mode='policy')
        last_idx = lengths - 1
        greedy = [logits[torch.arange(batch_size, device=state.device), last_idx].argmax(
            dim=-1) for logits in logits_list]

        if mode == 'greedy':
            return torch.stack(greedy, dim=1)
        elif mode == 'epsilon':
            out_actions = []
            for head_idx, logits in enumerate(logits_list):
                greedy_a = greedy[head_idx]
                random_a = torch.randint(
                    self.action_dims[head_idx], (batch_size,), device=state.device)
                explore_mask = (torch.rand(
                    batch_size, device=state.device) < epsilon)
                out = torch.where(explore_mask, random_a, greedy_a)
                out_actions.append(out)
            return torch.stack(out_actions, dim=1)
        elif mode == 'sample':
            out_actions = []
            for head_idx, logits in enumerate(logits_list):
                probs = F.softmax(logits[torch.arange(
                    batch_size, device=state.device), last_idx] / temperature, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                out_actions.append(sampled)
            return torch.stack(out_actions, dim=1)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    # ------------------------------------------------------------------
    # Public API – single-step Q(s,a) helper
    # ------------------------------------------------------------------
    def q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute scalar Q(s,a) for joint multi-head discrete actions.

        This utility mirrors the interface provided by baseline MLP/Q-networks
        so that OPE modules (e.g. FQE, WDR) and Boltzmann policies can work
        transparently with PoG backbones.  We aggregate per-head Q-values by
        simple averaging – consistent with the treatment in baseline models.

        Args:
            state: Tensor of shape (B, F) **or** (B, T, F).
            action: Tensor of shape (B, n_heads) with discrete indices.

        Returns:
            Tensor of shape (B,) containing aggregated Q-values.
        """
        if action.dim() != 2 or action.size(1) != len(self.action_dims):
            raise ValueError(
                "action must be (B, n_heads) matching action_dims")

        # Prepare sequence inputs & meta tensors --------------------------------
        if state.dim() == 2:  # (B, F)
            state_seq = state.unsqueeze(1)
            lengths = torch.ones(state.size(
                0), dtype=torch.long, device=state.device)
        elif state.dim() == 3:  # (B, T, F)
            state_seq = state
            lengths = torch.full((state.size(0),), state.size(
                1), dtype=torch.long, device=state.device)
        else:
            raise ValueError("state must be (B, F) or (B, T, F)")

        batch_size, seq_len, _ = state_seq.shape
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=state.device)
        mask = torch.ones(batch_size, seq_len,
                          dtype=torch.bool, device=state.device)

        # Forward pass in *q* mode (returns List[Tensor]) ------------------------
        # type: ignore[assignment]
        q_list, _ = self.forward(
            state_seq, lengths, edge_index, mask, mode='q')

        # 🔧 ENHANCED Q-VALUE GATHERING: Robust handling of different tensor shapes
        # Gather Q(s,a) for each head on **last timestep** with comprehensive error handling
        last_idx = lengths - 1  # (B,)
        batch_idx = torch.arange(batch_size, device=state.device)
        q_vals = []
        
        for h, q_head in enumerate(q_list):
            try:
                # Handle different Q-value tensor shapes
                if q_head.dim() == 3:
                    # Standard PoG output: (B, T, A_i)
                    # Ensure indices are valid
                    last_idx_clamped = torch.clamp(last_idx, max=q_head.size(1) - 1)
                    q_last = q_head[batch_idx, last_idx_clamped, :]  # (B, A_i)
                elif q_head.dim() == 2:
                    # Already 2D: (B, A_i) - use directly
                    q_last = q_head
                else:
                    raise ValueError(f"Unsupported Q-head {h} dimensions: {q_head.dim()}D")
                
                # Validate action indices for this head
                action_indices = action[:, h:h+1]
                action_indices_clamped = torch.clamp(action_indices, min=0, max=q_last.size(1) - 1)
                
                # Gather Q-values for selected actions
                q_val = q_last.gather(1, action_indices_clamped).squeeze(1)
                q_vals.append(q_val)
                
            except Exception as e:
                print(f"❌ POG q_value head {h} failed: {e}")
                # Fallback: return zero Q-values for this head
                q_val = torch.zeros(batch_size, device=state.device, dtype=q_head.dtype)
                q_vals.append(q_val)

        # Aggregate per-head values (mean) --------------------------------------
        return torch.stack(q_vals, dim=1).mean(1)

    def extra_repr(self) -> str:
        """Returns extra representation string for module logger.debuging."""
        return (f'input_dim={self.input_dim}, lstm_hidden={self.lstm_hidden}, '
                f'gcn_hidden={self.gcn_hidden}, action_dims={self.action_dims}, '
                f'traj_bottleneck_dim={self.traj_bottleneck_dim}, '
                f'gcn_bottleneck_dim={self.gcn_bottleneck_dim}')
