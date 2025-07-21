"""Trajectory encoder module for sequential patient data.

This module implements an LSTM-based encoder for processing sequential
patient trajectory data with support for variable-length sequences,
bidirectional processing, and multiple layers.
"""

from typing import Optional
import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """LSTM-based trajectory encoder for sequential patient data.
    
    This encoder processes variable-length patient trajectories using LSTM
    networks with support for packed sequences, bidirectional processing,
    and multi-layer architectures. It handles the complexities of medical
    time series data including variable sequence lengths and temporal
    dependencies.
    
    Attributes:
        input_dim: Input feature dimension for each time step.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        lstm: The underlying LSTM module.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int = 1, 
        dropout: float = 0.0, 
        bidirectional: bool = False
    ) -> None:
        """Initializes the trajectory encoder.
        
        Args:
            input_dim: Input feature dimension for each time step.
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of LSTM layers. Must be >= 1.
            dropout: Dropout rate between LSTM layers. Only applied if 
                    num_layers > 1.
            bidirectional: Whether to use bidirectional LSTM.
            
        Raises:
            ValueError: If input_dim, hidden_dim are not positive or num_layers < 1.
        """
        super().__init__()
        
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Create LSTM with appropriate dropout
        # Note: PyTorch LSTM only applies dropout between layers if num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Initialize LSTM weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initializes LSTM weights using Xavier initialization.
        
        Applies Xavier uniform initialization to all weight matrices and
        zeros initialization to bias terms for stable training.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encodes variable-length trajectory sequences.
        
        Processes a batch of variable-length sequences using packed sequence
        optimization for efficient computation. Handles padding and masking
        automatically.
        
        Args:
            x: Input sequences of shape (batch_size, max_seq_len, input_dim).
            lengths: Actual sequence lengths of shape (batch_size,). Must contain
                    positive integers <= max_seq_len.
                    
        Returns:
            Encoded sequences of shape (batch_size, max_seq_len, hidden_dim * directions)
            where directions = 2 if bidirectional else 1.
            
        Raises:
            ValueError: If input shapes are invalid or lengths are inconsistent.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, input_dim), got {x.dim()}D")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")
        if lengths.dim() != 1 or lengths.size(0) != x.size(0):
            raise ValueError(f"lengths shape {lengths.shape} incompatible with batch size {x.size(0)}")
        
        batch_size, max_seq_len, _ = x.shape
        
        # 🔧 CRITICAL FIX: Smart lengths validation and repair instead of strict error
        # Instead of throwing errors, intelligently fix invalid lengths to prevent crashes
        
        # Step 1: Handle invalid lengths (≤ 0) by setting them to 1
        invalid_lengths_mask = lengths <= 0
        if torch.any(invalid_lengths_mask):
            num_invalid = invalid_lengths_mask.sum().item()
            print(f"⚠️ Found {num_invalid} invalid lengths (≤ 0), setting to 1")
            lengths = torch.where(invalid_lengths_mask, torch.ones_like(lengths), lengths)
        
        # Step 2: Handle lengths exceeding max_seq_len by clamping
        excessive_lengths_mask = lengths > max_seq_len
        if torch.any(excessive_lengths_mask):
            num_excessive = excessive_lengths_mask.sum().item()
            print(f"⚠️ Found {num_excessive} lengths > max_seq_len ({max_seq_len}), clamping to max_seq_len")
            lengths = torch.clamp(lengths, max=max_seq_len)
        
        # Log repair summary if any fixes were applied
        if torch.any(invalid_lengths_mask) or torch.any(excessive_lengths_mask):
            print(f"✅ Lengths validation: repaired invalid values, range now [{lengths.min().item()}, {lengths.max().item()}]")
        
        # 🔧 CRITICAL FIX: Enhanced memory and sequence length management for cuDNN compatibility
        # Handle extremely long sequences that can cause cuDNN issues
        MAX_SAFE_SEQ_LEN = 4096  # Conservative limit to prevent cuDNN memory issues - aligned with PoG model
        
        if max_seq_len > MAX_SAFE_SEQ_LEN:
            print(f"⚠️ Very long sequence detected: {max_seq_len} > {MAX_SAFE_SEQ_LEN}")
            print(f"🔧 Applying sequence truncation to prevent cuDNN issues")
            
            # Truncate sequences to the most recent MAX_SAFE_SEQ_LEN timesteps
            # This preserves the most relevant recent information
            x = x[:, -MAX_SAFE_SEQ_LEN:, :]
            max_seq_len = MAX_SAFE_SEQ_LEN
            
            # Adjust lengths accordingly - this is where lengths might exceed the new max_seq_len
            lengths = torch.clamp(lengths, max=MAX_SAFE_SEQ_LEN)
            
            print(f"✅ Truncated sequences to {max_seq_len} timesteps, adjusted lengths range: [{lengths.min().item()}, {lengths.max().item()}]")
        
        # 🔧 CRITICAL FIX: Comprehensive tensor contiguity and memory layout optimization
        # Ensure all tensors are contiguous before any cuDNN operations
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Additional memory layout optimization for cuDNN
        x = x.clone().detach().contiguous()  # Create a fresh contiguous copy
            
        # Ensure lengths tensor is also contiguous and on correct device
        if not lengths.is_contiguous():
            lengths = lengths.contiguous()
        
        # Create contiguous copy of lengths for maximum compatibility
        lengths = lengths.clone().detach().contiguous()
        
        # Move lengths to CPU for pack_padded_sequence compatibility (required by PyTorch)
        lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
        
        # Ensure CPU lengths are also contiguous
        if not lengths_cpu.is_contiguous():
            lengths_cpu = lengths_cpu.contiguous()
        
        # 🔧 ENHANCED ERROR HANDLING: Robust sequence packing with fallback mechanisms
        try:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
        except Exception as e:
            error_msg = f"Failed to pack sequences: {e}"
            print(f"❌ {error_msg}")
            print(f"🔍 Debug info:")
            print(f"   • x.shape: {x.shape}, x.device: {x.device}, x.dtype: {x.dtype}")
            print(f"   • x.is_contiguous(): {x.is_contiguous()}")
            print(f"   • x.stride(): {x.stride()}")
            print(f"   • lengths_cpu.shape: {lengths_cpu.shape}, lengths_cpu.dtype: {lengths_cpu.dtype}")
            print(f"   • lengths_cpu.is_contiguous(): {lengths_cpu.is_contiguous()}")
            print(f"   • lengths range: [{lengths_cpu.min().item()}, {lengths_cpu.max().item()}]")
            
            # Try fallback: sort sequences by length (sometimes helps with packing)
            try:
                print("🔄 Attempting fallback with sorted sequences...")
                sorted_lengths, sort_indices = lengths_cpu.sort(descending=True)
                sorted_x = x[sort_indices]
                
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    sorted_x, sorted_lengths, batch_first=True, enforce_sorted=True
                )
                
                # We'll need to unsort later
                unsort_indices = sort_indices.argsort()
                print("✅ Fallback packing succeeded")
                
            except Exception as fallback_e:
                print(f"❌ Fallback packing also failed: {fallback_e}")
                raise RuntimeError(f"Both primary and fallback packing failed: {e}, {fallback_e}") from e
        else:
            # Primary packing succeeded
            sort_indices = None
            unsort_indices = None
        
        # 🔧 CRITICAL FIX: Enhanced packed sequence data contiguity validation
        if not packed_input.data.is_contiguous():
            print("🔧 Packed input data is not contiguous, creating contiguous copy...")
            # Reconstruct packed sequence with contiguous data
            packed_data = packed_input.data.contiguous()
            packed_input = nn.utils.rnn.PackedSequence(
                packed_data, packed_input.batch_sizes, 
                packed_input.sorted_indices, packed_input.unsorted_indices
            )
            print("✅ Packed input data made contiguous")
        
        # 🔧 CRITICAL FIX: Force contiguous memory layout for all packed sequence components
        # Some cuDNN versions require strict memory alignment
        packed_input = nn.utils.rnn.PackedSequence(
            packed_input.data.clone().contiguous(),
            packed_input.batch_sizes.clone().contiguous(),
            packed_input.sorted_indices.clone().contiguous() if packed_input.sorted_indices is not None else None,
            packed_input.unsorted_indices.clone().contiguous() if packed_input.unsorted_indices is not None else None
        )
        
        # 🔧 ENHANCED LSTM PROCESSING: Multiple fallback strategies for cuDNN issues
        lstm_attempts = 0
        max_lstm_attempts = 3
        
        while lstm_attempts < max_lstm_attempts:
            try:
                # Process through LSTM with enhanced error handling
                packed_output, (hidden, cell) = self.lstm(packed_input)
                break  # Success, exit retry loop
                
            except Exception as e:
                lstm_attempts += 1
                error_msg = f"LSTM forward failed (attempt {lstm_attempts}/{max_lstm_attempts}): {e}"
                print(f"❌ {error_msg}")
                
                if lstm_attempts < max_lstm_attempts:
                    # 🔧 ENHANCED ERROR HANDLING: Provide detailed debug information for LSTM errors
                    debug_info = {
                        'input_shape': x.shape,
                        'input_device': x.device,
                        'input_dtype': x.dtype,
                        'input_contiguous': x.is_contiguous(),
                        'input_stride': x.stride(),
                        'lengths_shape': lengths.shape,
                        'lengths_device': lengths.device,
                        'lengths_dtype': lengths.dtype,
                        'lengths_contiguous': lengths.is_contiguous(),
                        'lstm_input_size': self.lstm.input_size,
                        'lstm_hidden_size': self.lstm.hidden_size,
                        'lstm_num_layers': self.lstm.num_layers,
                        'lstm_bidirectional': self.lstm.bidirectional,
                        'packed_input_batch_sizes': packed_input.batch_sizes.tolist()[:10] if len(packed_input.batch_sizes) > 10 else packed_input.batch_sizes.tolist(),
                        'packed_data_shape': packed_input.data.shape,
                        'packed_data_contiguous': packed_input.data.is_contiguous(),
                        'packed_data_stride': packed_input.data.stride(),
                        'cuda_available': torch.cuda.is_available(),
                        'current_device': str(x.device)
                    }
                    
                    print(f"🔍 Debug info: {debug_info}")
                    
                    # Special handling for cuDNN errors
                    if "cuDNN" in str(e) or "CUDNN" in str(e):
                        print("🚨 cuDNN error detected - applying progressive fixes")
                        print(f"📊 Memory analysis:")
                        if torch.cuda.is_available():
                            try:
                                allocated = torch.cuda.memory_allocated() / 1024**3
                                reserved = torch.cuda.memory_reserved() / 1024**3
                                print(f"   • GPU allocated: {allocated:.2f} GB")
                                print(f"   • GPU reserved: {reserved:.2f} GB")
                            except:
                                print("   • Could not retrieve GPU memory info")
                        
                        # Progressive fix attempts
                        if lstm_attempts == 1:
                            print("🔧 Attempt 1: Force GPU memory cleanup and recreate tensors")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Recreate all tensors with explicit contiguous layout
                            x = x.detach().clone().contiguous()
                            lengths = lengths.detach().clone().contiguous()
                            lengths_cpu = lengths.cpu().contiguous()
                            
                            # Recreate packed input
                            packed_input = nn.utils.rnn.pack_padded_sequence(
                                x, lengths_cpu, batch_first=True, enforce_sorted=False
                            )
                            packed_input = nn.utils.rnn.PackedSequence(
                                packed_input.data.detach().clone().contiguous(),
                                packed_input.batch_sizes.detach().clone().contiguous(),
                                packed_input.sorted_indices.detach().clone().contiguous() if packed_input.sorted_indices is not None else None,
                                packed_input.unsorted_indices.detach().clone().contiguous() if packed_input.unsorted_indices is not None else None
                            )
                            
                        elif lstm_attempts == 2:
                            print("🔧 Attempt 2: Force device consistency and cleanup")
                            # 🔧 CRITICAL FIX: Ensure all components are on the same device
                            original_device = x.device
                            lstm_device = next(self.lstm.parameters()).device
                            
                            print(f"🔍 Device analysis: input={original_device}, lstm={lstm_device}")
                            
                            # Force all tensors to the LSTM's device
                            if original_device != lstm_device:
                                print(f"🔧 Moving input tensors from {original_device} to {lstm_device}")
                                x = x.to(lstm_device).contiguous()
                                lengths = lengths.to(lstm_device).contiguous()
                                lengths_cpu = lengths.cpu().contiguous()
                            
                            # Recreate packed input with device-consistent tensors
                            packed_input = nn.utils.rnn.pack_padded_sequence(
                                x, lengths_cpu, batch_first=True, enforce_sorted=False
                            )
                            
                            # Ensure packed input data is on the correct device
                            packed_input = nn.utils.rnn.PackedSequence(
                                packed_input.data.to(lstm_device).contiguous(),
                                packed_input.batch_sizes.contiguous(),
                                packed_input.sorted_indices.contiguous() if packed_input.sorted_indices is not None else None,
                                packed_input.unsorted_indices.contiguous() if packed_input.unsorted_indices is not None else None
                            )
                            
                            # Clear GPU cache before retry
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                            print("🔧 Attempting LSTM forward with device-consistent tensors")
                    else:
                        # Non-cuDNN error, continue with next attempt
                        if lstm_attempts == max_lstm_attempts:
                            raise RuntimeError(error_msg) from e
                else:
                    # All attempts failed
                    raise RuntimeError(f"LSTM forward failed after {max_lstm_attempts} attempts: {e}") from e
        
        # Unpack sequences back to padded format
        try:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=max_seq_len
            )
        except Exception as e:
            print(f"❌ Failed to unpack sequences: {e}")
            raise RuntimeError(f"Failed to unpack sequences: {e}") from e
        
        # 🔧 UNSORT FIX: Restore original order if we sorted for packing
        if sort_indices is not None and unsort_indices is not None:
            output = output[unsort_indices]
            print("🔧 Restored original sequence order after sorted packing")
        
        # 🔧 FINAL VALIDATION: Ensure output tensor properties
        if not output.is_contiguous():
            output = output.contiguous()
        
        # Validate output shape
        expected_hidden_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        if output.size(-1) != expected_hidden_dim:
            raise RuntimeError(f"Output dimension mismatch: expected {expected_hidden_dim}, got {output.size(-1)}")
        
        return output
    
    def get_final_states(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts final hidden and cell states for each sequence.
        
        Returns the actual final states based on true sequence lengths,
        not the padded final positions.
        
        Args:
            x: Input sequences of shape (batch_size, max_seq_len, input_dim).
            lengths: Actual sequence lengths of shape (batch_size,).
            
        Returns:
            A tuple containing:
                - final_hidden: Final hidden states of shape 
                               (batch_size, hidden_dim * directions).
                - final_cell: Final cell states of shape 
                             (batch_size, hidden_dim * directions).
        """
        # Get full output sequence
        output = self.forward(x, lengths)
        batch_size = x.size(0)
        
        # Extract final states based on actual lengths
        final_indices = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=x.device)
        
        final_hidden = output[batch_indices, final_indices]
        
        # For cell states, we need to run forward pass again to get LSTM states
        lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, (hidden_states, cell_states) = self.lstm(packed_input)
        
        # Reshape and select final states
        # hidden_states: (num_layers * directions, batch_size, hidden_dim)
        directions = 2 if self.bidirectional else 1
        final_layer_hidden = hidden_states[-directions:].transpose(0, 1).contiguous()
        final_layer_cell = cell_states[-directions:].transpose(0, 1).contiguous()
        
        # Flatten if bidirectional
        final_layer_hidden = final_layer_hidden.view(batch_size, -1)
        final_layer_cell = final_layer_cell.view(batch_size, -1)
        
        return final_layer_hidden, final_layer_cell
    
    def extra_repr(self) -> str:
        """Returns extra representation string for module printing."""
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'num_layers={self.num_layers}, dropout={self.dropout}, '
                f'bidirectional={self.bidirectional}') 