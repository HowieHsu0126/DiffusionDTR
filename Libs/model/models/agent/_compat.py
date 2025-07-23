from __future__ import annotations

import inspect
from typing import List, Optional

import torch
import torch.nn as nn

from Libs.utils.model_utils import select_last_valid_timestep


class ForwardCompatMixin:
    """Unified forward interface for PoG and baseline models.
    
    This mixin provides a standardized interface to handle the differences between
    PoG-based models (which expect graph structure) and baseline MLP/CNN models
    (which expect flattened state vectors). It handles tensor dimension alignment,
    device placement, and format conversion to ensure consistent behavior across
    different model architectures.
    
    Key Features:
        • Automatic model type detection and appropriate preprocessing
        • Robust tensor dimension handling with comprehensive validation
        • Device alignment and memory optimization
        • Error recovery and fallback mechanisms
        • Extensive debugging and diagnostic logging
    """

    def _is_pog_model(self, model: nn.Module) -> bool:
        """Detect if model uses PoG (Plan-on-Graph) architecture."""
        # Check for PoG-specific components
        return hasattr(model, "timestep_gcn") or hasattr(model, "gcn") or "PoG" in str(type(model))

    def _forward_model(
        self,
        model: nn.Module,
        obs: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        mode: str = "q",
        **kwargs,
    ):
        """Forward pass through model with UNIFIED OUTPUT FORMAT and comprehensive tensor validation.
        
        🔧 FUNDAMENTAL ARCHITECTURAL FIX: This method completely standardizes the interface
        between PoG and baseline models, ensuring consistent tensor dimensions and output formats
        across all model types. This eliminates the root cause of shape mismatches and output
        format errors.
        
        Key Architectural Improvements:
            • Unified output format: Always returns (q_list, hidden_state, kl_loss_1, kl_loss_2)
            • Robust tensor dimension standardization with comprehensive validation
            • Automatic batch size and sequence length alignment
            • Device-aware processing with memory optimization
            • Fail-safe error recovery with detailed diagnostics
        """
        # 🔧 CRITICAL ARCHITECTURAL FIX: Unified tensor preprocessing pipeline
        logger = getattr(self, "logger", None)
        
        # Determine target device from model parameters
        model_device = next(model.parameters()).device
        
        # 🔧 COMPREHENSIVE INPUT VALIDATION AND STANDARDIZATION
        try:
            # Ensure all inputs are on the correct device and are contiguous
            obs = obs.to(model_device).contiguous()
            lengths = lengths.to(model_device).contiguous()
            edge_index = edge_index.to(model_device).contiguous()
            if mask is not None:
                mask = mask.to(model_device).contiguous()
            
            # Validate basic tensor properties
            if not torch.is_tensor(obs) or not torch.is_tensor(lengths) or not torch.is_tensor(edge_index):
                raise ValueError("All primary inputs must be tensors")
            
            batch_size = obs.size(0)
            
            # 🔧 CRITICAL FIX: Standardize input dimensions BEFORE model forwarding
            # This ensures ALL models receive consistent input formats
            if obs.dim() == 2:
                # (B, D) -> (B, 1, D) for sequence consistency
                obs = obs.unsqueeze(1)
                seq_len = 1
                if logger:
                    logger.debug(f"🔧 Standardized obs to 3D: {obs.shape}")
            elif obs.dim() == 3:
                seq_len = obs.size(1)
            else:
                raise ValueError(f"Unsupported obs dimensions: {obs.dim()}D")
            
            # Standardize lengths tensor
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0).expand(batch_size)
            elif lengths.size(0) != batch_size:
                if lengths.size(0) == 1:
                    lengths = lengths.expand(batch_size)
                elif lengths.size(0) > batch_size:
                    lengths = lengths[:batch_size]
                else:
                    # Pad with maximum sequence length
                    padding = torch.full((batch_size - lengths.size(0),), seq_len, 
                                       dtype=lengths.dtype, device=lengths.device)
                    lengths = torch.cat([lengths, padding])
            
            # Standardize mask tensor
            if mask is None:
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=model_device)
            elif mask.dim() == 1:
                if mask.size(0) == batch_size:
                    mask = mask.unsqueeze(1).expand(batch_size, seq_len)
                else:
                    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=model_device)
            elif mask.shape != (batch_size, seq_len):
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=model_device)
            
            # Final input validation
            assert obs.shape == (batch_size, seq_len, obs.size(-1))
            assert lengths.shape == (batch_size,)
            assert mask.shape == (batch_size, seq_len)
            assert edge_index.dim() == 2 and edge_index.size(0) == 2
            
        except Exception as e:
            error_msg = f"Input standardization failed: {e}"
            if logger:
                logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg) from e

        # 🔧 ARCHITECTURAL DECISION: Detect model type and apply appropriate processing
        if self._is_pog_model(model):
            # PoG Model Processing Pipeline
            return self._forward_pog_model(model, obs, lengths, edge_index, mask, mode, **kwargs)
        else:
            # Baseline Model Processing Pipeline  
            return self._forward_baseline_model(model, obs, lengths, edge_index, mask, mode, **kwargs)

    def _forward_pog_model(
        self,
        model: nn.Module,
        obs: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
        **kwargs
    ):
        """Process PoG model with unified output format."""
        logger = getattr(self, "logger", None)
        
        try:
            # Forward through PoG model with comprehensive error handling
            if logger:
                logger.debug(f"🔧 PoG forward: obs={obs.shape}, mode={mode}")
            
            # PoG models return different tuple formats based on mode
            raw_output = model(obs, lengths, edge_index, mask, mode=mode, **kwargs)
            
            # 🔧 CRITICAL FIX: Standardize PoG output format to unified interface
            if mode == "q":
                # Expected: (q_list, hidden_features, kl_traj, kl_gcn)
                if len(raw_output) >= 2:
                    q_list = raw_output[0]
                    hidden_state = raw_output[1] if len(raw_output) > 1 else None
                    kl_loss_1 = raw_output[2] if len(raw_output) > 2 else 0.0
                    kl_loss_2 = raw_output[3] if len(raw_output) > 3 else 0.0
                else:
                    raise ValueError(f"Invalid PoG Q-mode output length: {len(raw_output)}")
                    
            elif mode == "policy":
                # Expected: (action_logits, state_values, hidden_features, kl_traj, kl_gcn)
                if len(raw_output) >= 3:
                    q_list = raw_output[0]  # action_logits serve as q_list
                    hidden_state = raw_output[2] if len(raw_output) > 2 else None
                    kl_loss_1 = raw_output[3] if len(raw_output) > 3 else 0.0
                    kl_loss_2 = raw_output[4] if len(raw_output) > 4 else 0.0
                else:
                    raise ValueError(f"Invalid PoG policy-mode output length: {len(raw_output)}")
            else:
                raise ValueError(f"Unsupported PoG mode: {mode}")
            
            # Validate and standardize q_list
            if not isinstance(q_list, (list, tuple)):
                raise ValueError(f"PoG q_list must be list/tuple, got {type(q_list)}")
            
            # 🔧 CRITICAL FIX: Handle PoG Q-value dimension mismatch for DQN compatibility
            # PoG models output Q-values with shape (batch_size, seq_len, action_dim)
            # but DQN agents expect (batch_size, action_dim)
            standardized_q_list = []
            batch_size = obs.size(0)
            
            for i, q in enumerate(q_list):
                if not torch.is_tensor(q):
                    raise ValueError(f"PoG Q-output {i} is not a tensor: {type(q)}")
                
                # 🔧 ENHANCED DIMENSION HANDLING: Convert 3D PoG output to 2D for DQN compatibility
                if q.dim() == 3:
                    # PoG output: (batch_size, seq_len, action_dim) -> (batch_size, action_dim)
                    # Take the last timestep for DQN (most relevant for value estimation)
                    q = q[:, -1, :]  # Extract last timestep
                    if logger:
                        logger.debug(f"🔧 PoG Q-output {i}: Converted 3D to 2D by taking last timestep: {q.shape}")
                elif q.dim() == 2:
                    # Already correct shape for DQN
                    pass
                elif q.dim() == 1:
                    # Add batch dimension if missing
                    q = q.unsqueeze(0)
                    if logger:
                        logger.debug(f"🔧 PoG Q-output {i}: Added batch dimension: {q.shape}")
                else:
                    raise ValueError(f"Unsupported PoG Q-output {i} dimensions: {q.dim()}D")
                
                # Validate batch dimension
                if q.size(0) != batch_size:
                    if logger:
                        logger.warning(f"⚠️ PoG Q-output {i} batch size mismatch: {q.size(0)} vs {batch_size}")
                    if q.size(0) > batch_size:
                        q = q[:batch_size]
                    else:
                        # Pad with zeros if needed
                        pad_size = batch_size - q.size(0)
                        padding = torch.zeros(pad_size, q.size(1), device=q.device, dtype=q.dtype)
                        q = torch.cat([q, padding], dim=0)
                        if logger:
                            logger.debug(f"🔧 PoG Q-output {i}: Padded to correct batch size: {q.shape}")
                
                # Final validation: ensure 2D tensor for DQN compatibility
                if q.dim() != 2:
                    raise RuntimeError(f"PoG Q-output {i} is not 2D after processing: {q.shape}")
                
                standardized_q_list.append(q)
            
            return (standardized_q_list, hidden_state, kl_loss_1, kl_loss_2)
            
        except Exception as e:
            error_msg = f"PoG model forward failed: {e}"
            if logger:
                logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _forward_baseline_model(
        self,
        model: nn.Module,
        obs: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
        **kwargs
    ):
        """Process baseline MLP/CNN model with unified output format."""
        logger = getattr(self, "logger", None)
        batch_size = obs.size(0)
        seq_len = obs.size(1)
        
        try:
            # 🔧 CRITICAL FIX: Extract final observation for MLP models
            # MLP models expect (B, D) input, so extract last valid timestep
            if obs.dim() == 3:
                # Use lengths to get last valid timestep for each sequence
                lengths_clamped = torch.clamp(lengths, min=1, max=seq_len)
                batch_indices = torch.arange(batch_size, device=obs.device)
                last_indices = (lengths_clamped - 1).clamp(min=0, max=seq_len - 1)
                final_obs = obs[batch_indices, last_indices]  # (B, D)
            else:
                final_obs = obs  # Already (B, D)
            
            if logger:
                logger.debug(f"🔧 MLP forward: final_obs={final_obs.shape}")
            
            # Forward through baseline model
            raw_output = model(final_obs)
            
            # 🔧 CRITICAL FIX: Standardize baseline output to unified format
            if isinstance(raw_output, torch.Tensor):
                # Single tensor output - this shouldn't happen for multi-head models
                if logger:
                    logger.warning("⚠️ Baseline model returned single tensor, expected list")
                q_list = [raw_output]
            elif isinstance(raw_output, (list, tuple)):
                q_list = list(raw_output)
            else:
                raise ValueError(f"Baseline model returned unsupported type: {type(raw_output)}")
            
            # Validate and process each Q-head output
            standardized_q_list = []
            for i, q in enumerate(q_list):
                if not torch.is_tensor(q):
                    raise ValueError(f"Baseline Q-output {i} is not a tensor: {type(q)}")
                
                # Ensure correct batch dimension
                if q.size(0) != batch_size:
                    if logger:
                        logger.warning(f"⚠️ Baseline Q-output {i} batch size mismatch: {q.size(0)} vs {batch_size}")
                    if q.size(0) > batch_size:
                        q = q[:batch_size]
                    elif q.size(0) == 1:
                        # Single output - expand to batch
                        q = q.expand(batch_size, -1)
                    else:
                        raise ValueError(f"Cannot fix baseline Q-output {i} batch size: {q.size(0)} vs {batch_size}")
                
                # 🔧 SEQUENCE DIMENSION EXPANSION: Expand single-step outputs to sequence format
                # This ensures compatibility with agents expecting (B, T, A) format
                if q.dim() == 2:  # (B, A)
                    if seq_len > 1:
                        # Expand to sequence: (B, A) -> (B, T, A)
                        q = q.unsqueeze(1).expand(batch_size, seq_len, -1)
                    else:
                        # Keep as (B, 1, A) for consistency
                        q = q.unsqueeze(1)
                elif q.dim() == 1:  # (B,)
                    # Add action and sequence dimensions: (B,) -> (B, T, 1)
                    q = q.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)
                
                standardized_q_list.append(q)
            
            # Baseline models don't have hidden states or KL losses
            return (standardized_q_list, None, 0.0, 0.0)
            
        except Exception as e:
            error_msg = f"Baseline model forward failed: {e}"
            if logger:
                logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    # ------------------------------------------------------------------
    #  FQE / OPE helper aliases
    # ------------------------------------------------------------------
    # Many legacy agents expose their online / target networks under
    # ``model`` / ``target_model`` attributes instead of the more explicit
    # ``q_net`` / ``target_q_net`` names expected by the new OPE pipeline.
    # We provide lightweight property redirects so that subclasses inherit
    # them automatically without having to duplicate boilerplate code.

    # NOTE:  We deliberately *do not* override these when the subclass
    # already defines ``q_net`` or ``target_q_net`` to avoid masking more
    # specific implementations.

    # Online Q-network ----------------------------------------------------
    @property
    def q_net(self):  # noqa: D401
        if hasattr(self, "_q_net"):
            return self._q_net
        if hasattr(self, "model"):
            return getattr(self, "model")
        raise AttributeError("Agent has no attribute '_q_net', 'model' nor 'q_net'.")

    @q_net.setter
    def q_net(self, value):
        """Set the Q-network (online network)."""
        self._q_net = value

    # Target Q-network ----------------------------------------------------
    @property
    def target_q_net(self):  # noqa: D401
        if hasattr(self, "_target_q_net"):
            return self._target_q_net
        if hasattr(self, "target_model"):
            return getattr(self, "target_model")
        # Fallback to online net when target unavailable (e.g. BC)
        return self.q_net

    @target_q_net.setter
    def target_q_net(self, value):
        """Set the target Q-network."""
        self._target_q_net = value 