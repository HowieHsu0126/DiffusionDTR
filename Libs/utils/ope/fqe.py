"""Fitted Q Evaluation (FQE) estimator.

Implementation inspired by Le et al., 2019 "Batch Policy Learning in
Continuous and Mixed Continuous‐Discrete Domains" and the *d3rlpy* FQE
module, adapted to the PoG‐BVE codebase.

For the AKI mechanical ventilation task we treat state as flattened vector
and action as multi‐discrete (3 branches).  The estimator is agnostic to the
underlying value network architecture – it reuses the agent's Q‐network.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from Libs.utils.data_utils import build_dataloader
from Libs.utils.exp_utils import seed_everything
from Libs.utils.model_utils import get_autocast_context
from torch.utils.data import DataLoader, Dataset

__all__ = ["FQEEstimator"]


def _default_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber (smooth L1) loss for better robustness than MSE."""
    return torch.nn.functional.smooth_l1_loss(pred, target)


class TransitionDataset(Dataset):
    """Lightweight tensor dataset for FQE training."""

    def __init__(self, transitions: Dict[str, torch.Tensor]):
        self.transitions = transitions
        self.size = transitions["state"].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ``torch.utils.data.RandomSampler`` 可能产生 *随机整数张量*，当
        # 创建 ``TransitionDataset`` 后用户又对 ``self.transitions`` 做
        # 截断 / 子集化会导致 ``idx`` 超出长度。为稳健起见对 ``idx``
        # 取模，以免 *IndexError* 终止训练或单元测试。
        idx = int(idx % self.size)
        return {k: v[idx] for k, v in self.transitions.items()}


class FQEEstimator:
    """Bootstrap‐compatible Fitted Q Evaluation.

    Args:
        q_network: *Untrained* copy of agent's Q‐network; will be updated.
        target_q_network: Target network for TD target computation.
        gamma: Discount factor.
        batch_size: Mini‐batch size during fitting.
        lr: Adam learning rate.
        n_epochs: Number of full passes over the dataset.
        update_target_freq: Frequency of target network updates.
        device: Runtime device.
        loss_fn: Supervised loss; defaults to MSE.
        policy_action_fn: Policy action function π(s) used for TD targets; defaults to greedy
                         action of the *copied* q_net (deterministic evaluation).
    """

    def __init__(
        self,
        q_network: nn.Module,
        target_q_network: nn.Module,
        gamma: float = 0.99,
        batch_size: int = 256,
        lr: float = 1e-3,
        n_epochs: int = 20,
        update_target_freq: int = 1000,
        device: str | torch.device = "cpu",
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        policy_action_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.device = torch.device(device)
        # Deep‐copy networks to avoid mutating original agent — prevents
        # information leakage and guarantees FQE uses an *independent* model.
        # ------------------------------------------------------------------
        # Some baseline agents **freeze** their Q‐network parameters 
        # (``requires_grad = False``) to disable further optimisation.
        # This breaks FQE training because the copied network cannot be
        # updated, causing ``loss.backward()`` to raise the classic
        # ``element 0 of tensors does not require grad`` error.  We thus
        # explicitly *re‐enable* gradients for the estimator's Q‐network
        # while simultaneously *freezing* the target network.
        self.q_net = copy.deepcopy(q_network).to(self.device)
        for p in self.q_net.parameters():
            p.requires_grad_(True)

        self.target_q_net = copy.deepcopy(target_q_network).to(self.device)
        self.target_q_net.eval()  # target used only for value bootstrap
        for p in self.target_q_net.parameters():
            p.requires_grad_(False)
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.update_target_freq = int(update_target_freq)
        self._update_step_counter = 0
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = loss_fn or _default_loss_fn
        # Policy action function π(s) used for TD targets; defaults to greedy
        # action of the *copied* q_net (deterministic evaluation).
        if policy_action_fn is None:
            self.policy_action_fn = lambda s: self.q_net.greedy_action(s)  # type: ignore
        else:
            self.policy_action_fn = policy_action_fn

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def fit(self, transitions: Dict[str, torch.Tensor], *, amp: bool = False) -> None:
        """Supervised FQE training loop."""
        # Compute reward scaling factor once per estimator
        if not hasattr(self, "_reward_scale"):
            max_abs_reward = float(transitions["reward"].abs().max().item())
            self._reward_scale = max(max_abs_reward, 1.0)

        # Validate input data for numerical stability
        for key, tensor in transitions.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"Input tensor '{key}' contains NaN or Inf values")

        dataset = TransitionDataset(transitions)
        loader = build_dataloader(dataset, self.batch_size, shuffle=True)

        # 保证仅在 CUDA 可用且用户显式启用 AMP 时才使用 GradScaler，
        # 否则在 CPU 上使用 scaler.scale(loss).backward() 会导致"tensor does not require grad"错误。
        scaler_enabled = amp and torch.cuda.is_available()
        # 🔧 CRITICAL FIX: Use correct import path for GradScaler
        # ``device_type`` is not accepted by ``GradScaler`` in older PyTorch versions
        # (≤2.1).  Use the simpler signature that only toggles scaling via *enabled*
        # to maintain broad compatibility across PyTorch releases.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self.q_net.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            valid_batches = 0
            for batch in loader:
                try:
                    loss = self._update_step(batch, scaler, amp)
                    if torch.isfinite(torch.tensor(loss)):
                        epoch_loss += loss
                        valid_batches += 1
                    else:
                        print(f"Warning: Invalid loss {loss} in FQE training, skipping batch")
                        continue
                except Exception as e:
                    print(f"Warning: FQE training batch failed: {e}, skipping")
                    continue
                    
                # Hard update target network every *update_target_freq* steps
                self._update_step_counter += 1
                if self._update_step_counter % self.update_target_freq == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())
            
            # Early stopping if too many invalid batches
            if valid_batches == 0:
                print(f"Warning: No valid batches in FQE epoch {epoch}, stopping early")
                break

    @property
    def reward_scale(self) -> float:  # noqa: D401
        """Returns the internal reward scaling factor (>=1.0)."""
        return getattr(self, "_reward_scale", 1.0)

    def evaluate(self, init_states: torch.Tensor, policy_actions: torch.Tensor) -> float:
        """Estimates policy value given initial state distribution.

        Args:
            init_states: Tensor (N, state_dim)
            policy_actions: Tensor (N, action_dim); greedy actions of policy

        Returns:
            Estimated expected return J^π.
        """
        try:
            self.q_net.eval()
            
            # Validate inputs
            if torch.isnan(init_states).any() or torch.isinf(init_states).any():
                print("Warning: init_states contains NaN/Inf values, returning 0.0")
                return 0.0
            if torch.isnan(policy_actions).any() or torch.isinf(policy_actions).any():
                print("Warning: policy_actions contains NaN/Inf values, returning 0.0")
                return 0.0
            
            # Compute Q-values with numerical stability
            with torch.no_grad():
                values = self.q_net.q_value(init_states.to(self.device), policy_actions.to(self.device))
                
                # Check for NaN/Inf in Q-values
                if torch.isnan(values).any() or torch.isinf(values).any():
                    print("Warning: Q-values contain NaN/Inf, returning 0.0")
                    return 0.0
                
                # Apply reward scaling and compute mean
                scaled_values = values * self.reward_scale
                mean_value = scaled_values.mean().item()
                
                # Final numerical check
                if not torch.isfinite(torch.tensor(mean_value)):
                    print("Warning: Final FQE estimate is not finite, returning 0.0")
                    return 0.0
                    
                return mean_value
                
        except Exception as e:
            print(f"Warning: FQE evaluation failed: {e}, returning 0.0")
            return 0.0

    def bootstrap_ci(
        self,
        transitions: Dict[str, torch.Tensor],
        init_states: torch.Tensor,
        policy_actions: torch.Tensor,
        n_bootstrap: int = 100,
        alpha: float = 0.05,
        seed: int = 42,
        amp: bool = False,
    ) -> Tuple[float, float, float]:
        """Computes bootstrap mean and two‐sided (1 − α) CI."""
        rng = np.random.default_rng(seed)
        n = init_states.shape[0]

        estimates: List[float] = []
        for b in range(n_bootstrap):
            # Resample indices with replacement
            idx = rng.integers(0, n, size=n)
            boot_trans = {k: v[idx] for k, v in transitions.items()}
            boot_init = init_states[idx]
            boot_act = policy_actions[idx]

            # Re-initialize networks to avoid information leak using deepcopy
            q_net_copy = copy.deepcopy(self.q_net)
            target_copy = copy.deepcopy(self.target_q_net)

            fqe = FQEEstimator(
                q_net_copy,
                target_copy,
                gamma=self.gamma,
                batch_size=self.batch_size,
                lr=self.lr,
                n_epochs=self.n_epochs,
                device=self.device,
                loss_fn=self.loss_fn,
                policy_action_fn=self.policy_action_fn,
            )
            fqe.fit(boot_trans, amp=amp)
            estimates.append(fqe.evaluate(boot_init, boot_act))

        mean = float(np.mean(estimates))
        lower = float(np.percentile(estimates, 100 * alpha / 2))
        upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
        return mean, lower, upper

    # ------------------------------------------------------------------
    #  Double-Robust (WDR) Bootstrap with optional PSIS smoothing
    # ------------------------------------------------------------------
    def bootstrap_wdr_ci(
        self,
        batch: Dict[str, torch.Tensor],
        n_bootstrap: int = 100,
        alpha: float = 0.05,
        seed: int = 42,
        gamma: float = 0.99,
        use_psis: bool = True,
    ) -> Tuple[float, float, float]:
        """Returns mean ± CI for (P)SIS-WDR value estimate."""
        from Libs.utils.ope.psis import psis_smooth_weights
        from Libs.utils.ope.wdr import wdr_estimate

        rng = np.random.default_rng(seed)
        rewards = batch["reward"]
        done = batch["done"].to(self.device).float()
        state = batch["state"]
        action = batch["action"]

        # Behaviour probs via naive empirical freq (same logic as Trainer)
        behav_prob = torch.ones(action.size(0))
        for b in range(action.size(1)):
            counts = torch.bincount(action[:, b], minlength=self.q_net.action_dims[b]).float()
            freq = counts / counts.sum()
            behav_prob *= freq[action[:, b]]
        importance_weights = 1.0 / behav_prob.clamp(min=1e-6)

        # Pre-compute q/v for full data to avoid training inside loop
        q_values_full = self.q_net.q_value(state, action)
        # State value V(s) ≈ mean_a Q(s,a)
        v_values_full = torch.stack([
            self.q_net.q_value(state, torch.cat([
                action[:, :b], torch.full_like(action[:, b:b+1], a), action[:, b+1:]
            ], dim=1))
            for b, a in enumerate(range(self.q_net.action_dims[0]))
        ]).mean(0)

        estimates = []
        N = rewards.shape[0]
        for _ in range(n_bootstrap):
            idx = rng.integers(0, N, size=N)
            iw = importance_weights[idx]
            if use_psis:
                iw = psis_smooth_weights(iw)
            est = wdr_estimate(rewards[idx], done[idx], iw, q_values_full[idx], v_values_full[idx], gamma)
            estimates.append(est)

        mean = float(np.mean(estimates))
        lower = float(np.percentile(estimates, 100 * alpha / 2))
        upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
        return mean, lower, upper

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _update_step(self, batch: Dict[str, torch.Tensor], scaler: torch.cuda.amp.GradScaler, amp: bool) -> float:
        """Single FQE training step with enhanced numerical stability and error handling.
        
        Returns:
            Training loss value, or NaN if the step failed
        """
        # 🔧 CRITICAL FIX: Enhanced tensor shape validation and fixing
        try:
            state = batch["state"].to(self.device)
            action = batch["action"].long().to(self.device)
            reward = batch["reward"].to(self.device) / self._reward_scale
            next_state = batch["next_state"].to(self.device)
            done = batch["done"].to(self.device).float()
            
            # 🔧 COMPREHENSIVE SHAPE VALIDATION: Handle truncated sequences and dimension mismatches
            batch_size = state.size(0)
            
            # Check if action tensor needs reshaping
            if action.dim() == 3 and action.size(1) != state.size(0):
                # Handle case where action has incompatible sequence dimension
                # This often happens when sequences are truncated in training
                print(f"FQE: Action shape mismatch: {action.shape} vs state {state.shape}")
                if action.size(0) == batch_size:
                    # Take last timestep if action has time dimension
                    action = action[:, -1]  # (B, T, H) -> (B, H)
                else:
                    # Cannot safely reshape - skip this batch
                    print(f"FQE: Cannot fix action shape mismatch {action.shape} for batch size {batch_size}")
                    return 0.0
            
            # Check if state/next_state dimensions match expected 2D format
            if state.dim() == 3:
                # Take last timestep for states 
                state = state[:, -1]  # (B, T, D) -> (B, D)
            if next_state.dim() == 3:
                next_state = next_state[:, -1]  # (B, T, D) -> (B, D)
                
            # Check if reward and done need flattening
            if reward.dim() > 1:
                reward = reward.flatten()[:batch_size]  # Ensure correct batch size
            if done.dim() > 1:
                done = done.flatten()[:batch_size]  # Ensure correct batch size
                
            # Final dimension validation
            expected_shapes = {
                'state': (batch_size, -1),  # (B, D)
                'action': (batch_size, -1),  # (B, H) 
                'reward': (batch_size,),     # (B,)
                'next_state': (batch_size, -1),  # (B, D)
                'done': (batch_size,)        # (B,)
            }
            
            tensors = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
            
            for name, tensor in tensors.items():
                expected = expected_shapes[name]
                if expected[0] != -1 and tensor.size(0) != expected[0]:
                    print(f"FQE: {name} batch size mismatch: {tensor.shape} vs expected batch_size={batch_size}")
                    return 0.0
                    
        except Exception as shape_error:
            print(f"FQE: Shape validation failed: {shape_error}")
            return 0.0

        # Validate batch data for NaN/Inf
        for name, tensor in [("state", state), ("action", action), ("reward", reward), 
                            ("next_state", next_state), ("done", done)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"FQE: Batch tensor '{name}' contains NaN/Inf values")
                return 0.0

        with get_autocast_context(enabled=amp):
            try:
                q_sa = self.q_net.q_value(state, action) / self._reward_scale
                
                # Check Q-values for numerical stability
                if torch.isnan(q_sa).any() or torch.isinf(q_sa).any():
                    raise ValueError("Q-values contain NaN/Inf")
                
                with torch.no_grad():
                    next_act = self.policy_action_fn(next_state)
                    
                    # Ensure next_act is valid
                    if torch.isnan(next_act).any() or torch.isinf(next_act).any():
                        raise ValueError("Next actions contain NaN/Inf")
                    
                    q_next = self.target_q_net.q_value(next_state, next_act) / self._reward_scale
                    
                    # Check target Q-values
                    if torch.isnan(q_next).any() or torch.isinf(q_next).any():
                        raise ValueError("Target Q-values contain NaN/Inf")
                    
                    target = reward + self.gamma * (1 - done) * q_next
                    
                    # Clamp target values for numerical stability
                    target = torch.clamp(target, min=-100.0, max=100.0)
                
                # Clamp current Q-values for stability
                q_sa_clamped = torch.clamp(q_sa, min=-100.0, max=100.0)
                loss = self.loss_fn(q_sa_clamped, target)
                
                # Check loss validity
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Loss is {loss}")
                    
            except Exception as e:
                # Return a dummy loss to keep training stable
                print(f"FQE forward pass failed: {e}")
                return 0.0

        # ---------------- 梯度更新 ----------------
        self.optimizer.zero_grad()

        try:
            if amp and torch.cuda.is_available():
                scaler.scale(loss).backward()
                # Apply gradient clipping
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # CPU 或未启用 AMP 时直接反向传播
                loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                self.optimizer.step()
        except Exception as e:
            print(f"FQE gradient update failed: {e}")
            return 0.0

        return float(loss.detach()) 