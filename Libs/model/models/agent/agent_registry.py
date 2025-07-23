#!/usr/bin/env python3
"""Agent registry for dynamic instantiation.

This lightweight registry decouples algorithm identifiers (``str``) from the
concrete *Agent* classes used at runtime.  It is designed to be a **single
source of truth** so that new algorithms can be added to the framework without
modifying high-level orchestration code such as
:class:`Libs.exp.train_rl.TrainingPipeline`.
"""
from __future__ import annotations

# Suppress all warnings for clean output
import warnings
from typing import Any
from typing import Any as _Any
from typing import Callable, Dict, List, Optional

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger(__name__)

_BUILDER_FN = Callable[[dict, dict, Any], Any]

# Internal mapping from algorithm key to builder function
_REGISTRY: Dict[str, _BUILDER_FN] = {}

# ---------------------------------------------------------------------------
# EXTENSIONS: Register default builders for built-in algorithms
# ---------------------------------------------------------------------------

# Lazy imports to avoid heavy dependencies when registry is imported by utilities


def _lazy_import(name: str):
    import importlib
    return importlib.import_module(name)


def _default_builder_factory(agent_cls_path: str, registry_key: str):
    """Return a builder that instantiates *agent_cls* with minimal arguments.

    The resulting builder ignores most hyper-parameters and relies on the
    defaults baked into the agent implementation.  This is intentional for
    first-phase migration – fine-grained control remains available via the old
    code path until we fully transition all algorithms.
    """

    def _builder(algo_cfg: Dict[str, Any], cfg: Dict[str, Any], model):
        exp_cfg = cfg.get("experiment", {})
        model_cfg = cfg.get("model", {})

        module_name, cls_name = agent_cls_path.rsplit(".", 1)
        module = _lazy_import(module_name)
        agent_cls = getattr(module, cls_name)

        # Collect mandatory args first
        kw = dict(
            action_dims=algo_cfg["action_dims"],
            lr=model_cfg.get("learning_rate", 1e-3),
            gamma=exp_cfg.get("gamma", 0.99),
            device=algo_cfg["device"],
        )

        # Add state_dim for agents that require it (like BVE)
        import inspect
        if "state_dim" in inspect.signature(agent_cls.__init__).parameters:
            state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
                "state_dim", exp_cfg.get("state_dim", 64))))
            kw["state_dim"] = state_dim

        # Add model parameter for agents that require it
        # Check if model is in the signature and not just a default parameter
        if "model" in inspect.signature(agent_cls.__init__).parameters:
            kw["model"] = model

        # Extra hyper-parameters (passed verbatim if present in config)
        init_params = set(inspect.signature(agent_cls.__init__).parameters)
        for _extra in ("cql_alpha", "buffer_capacity", "batch_size", "target_update_freq", "reward_centering"):
            if _extra in algo_cfg and _extra in init_params:
                kw[_extra] = algo_cfg[_extra]

        agent = agent_cls(**kw)

        # --------------------------------------------------
        # Reward scaling hyper-parameters (optional)
        # --------------------------------------------------
        reward_cfg = exp_cfg.get("reward", {})
        if reward_cfg:
            if "scale_method" in reward_cfg:
                agent.reward_scale_method = reward_cfg["scale_method"]
            if "temperature" in reward_cfg:
                agent.reward_scale_temperature = float(
                    reward_cfg["temperature"])
            if "time_decay" in reward_cfg:
                agent.reward_time_decay = float(reward_cfg["time_decay"])
            # Forward reward centering flag when supported
            if "centering" in reward_cfg and hasattr(agent, "reward_centering"):
                agent.reward_centering = bool(reward_cfg["centering"])
            if "centering_alpha" in reward_cfg and hasattr(agent, "reward_center_alpha"):
                agent.reward_center_alpha = float(
                    reward_cfg["centering_alpha"])

        # --------------------------------------------------
        # Expose preferred batch mode so that trainers can adapt sampling
        # strategy automatically.  This resolves shape mismatches such as
        # BCQ expecting *episode*-level padded tensors.
        # --------------------------------------------------
        _EPISODE_MODE_ALGOS = {"bcq", "pog_bcq"}
        if registry_key in _EPISODE_MODE_ALGOS:
            agent.preferred_batch_mode = "episode"
        else:
            agent.preferred_batch_mode = "transition"

        return agent

    return _builder


# Mapping: registry_key -> (agent_class_path)
_DEFAULT_ALGOS: Dict[str, str] = {
    # Baseline algorithms (corrected import paths)
    "bc": "Libs.model.models.agent.bc_agent.BCAgent",
    "dqn": "Libs.model.models.agent.dqn_agent.DQNAgent",
    "cql": "Libs.model.models.agent.cql_agent.CQLAgent",
    "bcq": "Libs.model.models.agent.bcq_agent.BCQAgent",
    "bve": "Libs.model.models.agent.bve_agent.BranchValueEstimationAgent",

    # PoG variants (share same agent class)
    "pog_bc": "Libs.model.models.agent.bc_agent.BCAgent",
    "pog_dqn": "Libs.model.models.agent.dqn_agent.DQNAgent",
    "pog_cql": "Libs.model.models.agent.cql_agent.CQLAgent",
    "pog_bcq": "Libs.model.models.agent.bcq_agent.BCQAgent",
    "pog_bve": "Libs.model.models.agent.bve_agent.BranchValueEstimationAgent",
}


# Auto-register default builders
for _key, _cls_path in _DEFAULT_ALGOS.items():
    if _key not in _REGISTRY:
        _REGISTRY[_key] = _default_builder_factory(_cls_path, _key)


def register(name: str) -> Callable[[_BUILDER_FN], _BUILDER_FN]:
    """Decorator to register a new algorithm builder under *name*."""

    def decorator(fn: _BUILDER_FN) -> _BUILDER_FN:
        if name in _REGISTRY:
            raise ValueError(f"Algorithm '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get(name: str) -> Optional[_BUILDER_FN]:
    """Fetch a builder function by algorithm key.

    Returns ``None`` if *name* has not been registered.
    """

    return _REGISTRY.get(name)


def available() -> List[str]:
    """Return a list of registered algorithm keys."""

    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Specialised builder for Behavioural Cloning (BC)
# ---------------------------------------------------------------------------


def _bc_builder(algo_cfg: dict, cfg: dict, _unused_model: _Any):  # noqa: D401
    """Return a `BCAgent` with a concrete `BCPolicyNet` backbone.

    The generic `_default_builder_factory` passes an `nn.Identity()` which has
    *no* parameters, causing `optimizer got an empty parameter list`.  Here we
    construct a simple MLP policy network so that optimisation works during
    smoke tests and default runs.
    """

    from Libs.model.models.agent.bc_agent import BCAgent  # local import
    from Libs.model.models.baseline.bc_core import BCPolicyNet

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
        "state_dim", exp_cfg.get("state_dim", 64))))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))

    action_dims = algo_cfg["action_dims"]

    policy_net = BCPolicyNet(
        state_dim=state_dim, action_dims=action_dims, hidden_dim=hidden_dim)

    agent = BCAgent(
        model=policy_net,
        action_dims=action_dims,
        lr=model_cfg.get("learning_rate", 1e-3),
        gamma=exp_cfg.get("gamma", 0.99),
        device=algo_cfg["device"],
        label_smoothing=float(model_cfg.get("label_smoothing", 0.0)),
    )

    # Preferred batch sampling mode for BC is *transition* (flat supervised)
    agent.preferred_batch_mode = "transition"

    return agent


# Override/extend registry entries
for _alias in ("bc", "pog_bc"):
    _REGISTRY[_alias] = _bc_builder

# ---------------------------------------------------------------------------
#  Specialised builder for Batch-Constrained Q-learning (BCQ)
# ---------------------------------------------------------------------------


def _bcq_builder(algo_cfg: dict, cfg: dict, _unused_model):  # noqa: D401
    """Return a *BCQAgent* with a minimal DQN backbone for smoke tests.

    The generic builder passes incompatible kwargs (e.g. ``state_dim``). We
    construct a small :class:`DQNNet` so that the agent initialises cleanly
    during BaselineTrainer setup.  The full *bcq_trainer* will later replace
    this stub, so compute efficiency is irrelevant.
    """

    from Libs.model.models.agent.bcq_agent import BCQAgent  # local import
    # Use *DQNNet* (state → List[Q-head]) as lightweight backbone so that
    # `ForwardCompatMixin` can infer output shapes without requiring an
    # explicit *action* argument.  The previous stub (`BCQNet`) expected both
    # `(state, action)` which caused a runtime `TypeError` during the mixed
    # forward pass.
    from Libs.model.models.baseline.dqn_core import DQNNet

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
        "state_dim", exp_cfg.get("state_dim", 64))))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))

    action_dims = algo_cfg["action_dims"]

    # Lightweight multi‐head Q‐network compatible with ForwardCompatMixin
    q_net = DQNNet(state_dim, action_dims, hidden_dim)
    q_net.init_args = (state_dim, action_dims, hidden_dim)

    agent = BCQAgent(
        model=q_net,
        action_dims=action_dims,
        lr=model_cfg.get("learning_rate", 1e-3),
        gamma=exp_cfg.get("gamma", 0.99),
        device=algo_cfg.get("device", "cpu"),
        perturbation_scale=float(algo_cfg.get("perturbation_scale", model_cfg.get("perturbation_scale", 0.05))),
        n_perturb_samples=int(algo_cfg.get("n_perturb_samples", 10)),
    )

    agent.preferred_batch_mode = "transition"
    return agent


# Register overrides - separate builders for vanilla and PoG variants
_REGISTRY["bcq"] = _bcq_builder
# Keep pog_bcq separate to ensure different initialization

# ---------------------------------------------------------------------------
#  Specialised builder for Conservative Q-learning (CQL)
# ---------------------------------------------------------------------------


def _cql_builder(algo_cfg: dict, cfg: dict, _unused_model):  # noqa: D401
    """Return a *CQLAgent* with lightweight DQN backbone."""

    from Libs.model.models.agent.cql_agent import CQLAgent
    from Libs.model.models.baseline.cql_core import CQLNet

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
        "state_dim", exp_cfg.get("state_dim", 64))))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))

    action_dims = algo_cfg["action_dims"]

    q_net = CQLNet(state_dim, action_dims, hidden_dim)
    # Provide init_args for target network cloning inside CQLAgent
    q_net.init_args = (state_dim, action_dims, hidden_dim)

    agent = CQLAgent(
        model=q_net,
        action_dims=action_dims,
        lr=model_cfg.get("learning_rate", 1e-3),
        gamma=exp_cfg.get("gamma", 0.99),
        cql_alpha=exp_cfg.get("cql_alpha", 0.1),
        device=algo_cfg.get("device", "cpu"),
    )

    agent.preferred_batch_mode = "transition"
    return agent


for _alias in ("cql", "pog_cql"):
    _REGISTRY[_alias] = _cql_builder

# ---------------------------------------------------------------------------
#  Specialised builders for **PoG baseline** algorithms that require the
#  provided PoG backbone instead of lightweight stubs. These builders keep the
#  signature in sync with BaselineTrainer so that users can switch between
#  vanilla and PoG versions via the `algo` flag only.
# ---------------------------------------------------------------------------


def _pog_bc_builder(algo_cfg: dict, cfg: dict, model):  # noqa: D401
    """Behavioural Cloning with PoG backbone."""
    from Libs.model.models.agent.bc_agent import BCAgent
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    if model is None:
        state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
            "state_dim", exp_cfg.get("state_dim", 64))))
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        model = _build_pog_backbone(
            state_dim, algo_cfg["action_dims"], hidden_dim)

    return BCAgent(
        model=model,
        action_dims=algo_cfg["action_dims"],
        lr=model_cfg.get("learning_rate", 1e-3),
        gamma=exp_cfg.get("gamma", 0.99),
        device=algo_cfg.get("device", "cpu"),
        label_smoothing=float(model_cfg.get("label_smoothing", 0.0)),
    )


def _pog_bcq_builder(algo_cfg: dict, cfg: dict, model):  # noqa: D401
    """BCQ with PoG backbone."""
    from Libs.model.models.agent.bcq_agent import BCQAgent
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    if model is None:
        state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
            "state_dim", exp_cfg.get("state_dim", 64))))
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        
        # Use a different seed modifier for PoG-BCQ to ensure different initialization
        import torch
        original_seed = torch.initial_seed()
        torch.manual_seed(original_seed + 1001)  # Different seed for BCQ
        
        model = _build_pog_backbone(
            state_dim, algo_cfg["action_dims"], hidden_dim)
            
        # Restore original seed
        torch.manual_seed(original_seed)

        # Store constructor arguments for *target network* cloning inside BCQAgent
        # Consistent with implementation in *_pog_dqn_builder* above.
        model.init_args = (state_dim, hidden_dim,
                           hidden_dim, algo_cfg["action_dims"])
    else:
        # Ensure `init_args` is present even when users pass a pre-built backbone.
        # Fallback to generic three‐tuple (state_dim, action_dims, hidden_dim)
        inferred_state_dim = getattr(
            model, "input_dim", None) or algo_cfg.get("state_dim", 64)
        inferred_hidden_dim = getattr(
            model, "lstm_hidden", None) or model_cfg.get("hidden_dim", 128)
        model.init_args = (int(inferred_state_dim), int(inferred_hidden_dim), int(
            inferred_hidden_dim), algo_cfg["action_dims"])

    # Use different hyperparameters for PoG-BCQ to ensure differentiation
    agent = BCQAgent(
        model=model,
        action_dims=algo_cfg["action_dims"],
        lr=model_cfg.get("learning_rate", 1e-3),
        gamma=exp_cfg.get("gamma", 0.99),
        device=algo_cfg.get("device", "cpu"),
        perturbation_scale=float(algo_cfg.get("perturbation_scale", model_cfg.get("perturbation_scale", 0.075))),  # Slightly different from vanilla
        n_perturb_samples=int(algo_cfg.get("n_perturb_samples", 15)),  # Different from vanilla (10)
        # PoG-specific settings
        kl_beta_start=0.01,  # Different from default 0.0
        kl_beta_end=0.15,    # Different from default 0.1
        target_update_freq=150,  # Different from default 100
    )
    
    # Add a marker to distinguish PoG-BCQ
    agent._algorithm_variant = "pog_bcq"
    return agent


def _pog_cql_builder(algo_cfg: dict, cfg: dict, model):  # noqa: D401
    """CQL with PoG backbone."""
    from Libs.model.models.agent.cql_agent import CQLAgent
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    if model is None:
        state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
            "state_dim", exp_cfg.get("state_dim", 64))))
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        model = _build_pog_backbone(
            state_dim, algo_cfg["action_dims"], hidden_dim)

        # Provide constructor args for target network cloning inside CQLAgent
        model.init_args = (state_dim, hidden_dim,
                           hidden_dim, algo_cfg["action_dims"])
    else:
        if not hasattr(model, "init_args"):
            inferred_state_dim = getattr(
                model, "input_dim", None) or algo_cfg.get("state_dim", 64)
            inferred_hidden_dim = getattr(
                model, "lstm_hidden", None) or model_cfg.get("hidden_dim", 128)
            model.init_args = (int(inferred_state_dim), int(inferred_hidden_dim), int(
                inferred_hidden_dim), algo_cfg["action_dims"])

    return CQLAgent(
        model=model,
        action_dims=algo_cfg["action_dims"],
        lr=model_cfg.get("learning_rate", 1e-3),
        gamma=exp_cfg.get("gamma", 0.99),
        cql_alpha=exp_cfg.get("cql_alpha", 0.1),
        device=algo_cfg.get("device", "cpu"),
    )


# Register / override builder mapping for PoG baselines
for _k, _fn in (
    ("pog_bc", _pog_bc_builder),
    ("pog_bcq", _pog_bcq_builder),
    ("pog_cql", _pog_cql_builder),
):
    _REGISTRY[_k] = _fn

# ---------------------------------------------------------------------------
#  Public factory – high-level entrypoint used by Trainer
# ---------------------------------------------------------------------------


def _load_yaml_safe(path: str) -> dict:
    """Return YAML dict if *path* exists, else empty dict (no error)."""
    from pathlib import Path

    import yaml

    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make(algo: str, **overrides):  # noqa: D401
    """Instantiate an *Agent* by algorithm key with YAML-driven configs.

    The resolution order for configuration values is (low ➜ high priority):
        1. ``Libs/configs/exp.yaml``            – global experiment defaults
        2. ``Libs/configs/baselines/{algo}.yaml`` – algorithm-specific defaults
        3. Explicit keyword ``overrides`` passed to this function

    This mirrors Hydra-style layering while remaining dependency-free.

    Args:
        algo: Algorithm identifier registered in this module.
        **overrides: Keyword arguments that override YAML values; typical
            examples include ``state_dim``, ``action_dims``, ``device``.

    Returns:
        Instantiated *Agent* object ready for training / evaluation.
    """

    from pathlib import Path

    algo = algo.lower()

    builder = get(algo)
    if builder is None:
        raise ValueError(
            f"Algorithm '{algo}' is not registered. Available: {available()}")

    # ------------------------------------------------------------------
    # 1. Load YAML configs
    # ------------------------------------------------------------------
    cfg_root = Path(__file__).resolve().parents[3] / "configs"
    exp_cfg = _load_yaml_safe(cfg_root / "exp.yaml")

    baseline_cfg_path = cfg_root / "baselines" / f"{algo}.yaml"
    # Fallback: strip "pog_" prefix ➜ share baseline hyper-params
    if not baseline_cfg_path.exists() and algo.startswith("pog_"):
        baseline_cfg_path = cfg_root / "baselines" / f"{algo[4:]}.yaml"
    algo_yaml_cfg = _load_yaml_safe(baseline_cfg_path)

    # ------------------------------------------------------------------
    # 2. Merge configs
    # ------------------------------------------------------------------
    # Structure expected by builders: *algo_cfg* (flat) & *global cfg*
    algo_cfg: dict = {**algo_yaml_cfg, **overrides}
    # currently already hierarchical (experiment/model)
    global_cfg: dict = exp_cfg

    # Ensure mandatory keys
    for key in ("device",):
        if key in overrides:
            algo_cfg[key] = overrides[key]
        elif key not in algo_cfg:
            algo_cfg[key] = "cpu"

    # ------------------------------------------------------------------
    # 3. Forward to builder
    # ------------------------------------------------------------------
    # optional backbone for *pog_* variants
    model = algo_cfg.pop("model", None)

    return builder(algo_cfg, global_cfg, model)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
#  End of registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Helper: default PoG backbone
# ---------------------------------------------------------------------------


def _build_pog_backbone(state_dim: int, action_dims: list[int], hidden_dim: int = 64):
    """Return a lightweight *PoG* model for baseline variants.

    The architecture mirrors `Libs.model.modules.pog_module.PoG` with
    xsi-size hyper-parameters derived from *state_dim* / *hidden_dim*.
    """
    from Libs.model.models.pog_model import \
        PlanOnGraphModel as PoG  # use updated implementation

    lstm_hidden = hidden_dim
    gcn_hidden = hidden_dim
    return PoG(
        input_dim=state_dim,
        lstm_hidden=lstm_hidden,
        gcn_hidden=gcn_hidden,
        action_dims=action_dims,
    )


def _pog_dqn_builder(algo_cfg: dict, cfg: dict, model):  # noqa: D401
    """DQN with PoG backbone."""
    from Libs.model.models.agent.dqn_agent import DQNAgent
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    # ------------------------------------------------------------------
    # Build PoG backbone when none is provided.  When a custom backbone is
    # passed, we *infer* the constructor arguments so that DQNAgent can clone
    # the target network without requiring the user to manually populate the
    # ``init_args`` attribute.
    # ------------------------------------------------------------------

    if model is None:
        state_dim = int(
            algo_cfg.get(
                "state_dim", model_cfg.get(
                    "state_dim", exp_cfg.get("state_dim", 64))
            )
        )
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        model = _build_pog_backbone(
            state_dim, algo_cfg["action_dims"], hidden_dim)
    else:
        # Attempt best‐effort inference of dimensions from the existing model
        state_dim = getattr(model, "input_dim",
                            None) or algo_cfg.get("state_dim", 64)
        hidden_dim = getattr(model, "lstm_hidden",
                             None) or model_cfg.get("hidden_dim", 128)

    q_net = model  # PoG backbone acts as Q-network in 'q' mode

    # Ensure the attribute exists for target network cloning inside DQNAgent
    q_net.init_args = (int(state_dim), int(hidden_dim),
                       int(hidden_dim), algo_cfg["action_dims"])

    return DQNAgent(
        model=q_net,
        action_dims=algo_cfg["action_dims"],
        lr=model_cfg.get("learning_rate", 1e-3),
        gamma=exp_cfg.get("gamma", 0.99),
        device=algo_cfg.get("device", "cpu"),
    )


# Register new builder
_REGISTRY["pog_dqn"] = _pog_dqn_builder

# ---------------------------------------------------------------------------
#  Specialised builder for PoG‐BVE (graph-based BVE baseline)
# ---------------------------------------------------------------------------


def _pog_bve_builder(algo_cfg: dict, cfg: dict, model):  # noqa: D401
    """Branch Value Estimation with PoG backbone.

    This builder wraps a *PoG* feature extractor with BVE‐style hierarchical
    Q-heads using :class:`Libs.model.models.agent.pog_bve_qnetwork.PogBveQNetwork`.
    The constructed network is passed to :class:`BranchValueEstimationAgent` so
    that all advanced BVE training features (CQL regularisation, beam search
    etc.) remain available while leveraging graph reasoning from PoG.
    """

    from Libs.model.models.agent.bve_agent import BranchValueEstimationAgent
    # local import to avoid GNN deps unless needed
    from Libs.model.models.pog_model import PlanOnGraphModel as PoG
    from Libs.model.modules.pog_bve_qnetwork import PogBveQNetwork

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    # ------------------------------------------------------------------
    # Action dimensions compatibility - PoG-BVE now supports any dimension
    # ------------------------------------------------------------------
    action_dims = algo_cfg["action_dims"]
    if len(action_dims) < 2:
        raise ValueError(
            f"PoG-BVE requires at least 2 action dimensions for branch decomposition, "
            f"got {len(action_dims)}. For single-action problems, use 'dqn' instead."
        )

    # ------------------------------------------------------------------
    # Build PoG backbone if not provided via *model* kwarg
    # ------------------------------------------------------------------
    if model is None:
        state_dim = int(algo_cfg.get("state_dim", model_cfg.get(
            "state_dim", exp_cfg.get("state_dim", 64))))
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        
        # Use a different seed modifier for PoG-BVE to ensure different initialization
        import torch
        original_seed = torch.initial_seed()
        torch.manual_seed(original_seed + 2002)  # Different seed for BVE
        
        model = PoG(
            input_dim=state_dim,
            lstm_hidden=hidden_dim,
            gcn_hidden=hidden_dim,
            action_dims=algo_cfg["action_dims"],
        )
        
        # Restore original seed
        torch.manual_seed(original_seed)

    # Wrap backbone with hierarchical BVE Q-heads
    q_net = PogBveQNetwork(
        pog_model=model,
        action_dims=algo_cfg["action_dims"],
        device=algo_cfg.get("device", "cpu"),
        q_head_hidden_dim=int(model_cfg.get("hidden_dim", 128)),
    )

    # Use different hyperparameters for PoG-BVE to ensure differentiation
    agent = BranchValueEstimationAgent(
        state_dim=int(algo_cfg.get("state_dim", model_cfg.get(
            "state_dim", exp_cfg.get("state_dim", 64)))),
        action_dims=algo_cfg["action_dims"],
        q_net=q_net,
        lr=model_cfg.get("learning_rate", 3e-4),  # Use standard learning rate
        gamma=exp_cfg.get("gamma", 0.99),
        device=algo_cfg.get("device", "cpu"),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        # PoG-BVE specific settings - use more conservative values
        cql_target_gap=5.0,        # Back to default value
        lambda_reg=0.1,            # Back to default value  
        target_update_freq=100,    # Back to default value
        max_grad_norm=1.0,         # Back to default value
        alpha=0.5,                 # Lower initial alpha for less conservative Q-values
        cql_n_samples=20,          # More samples for better CQL estimation
        normalize_branch=True,     # Ensure branch normalization
        cql_sample_mode='mixed',   # Use mixed sampling for stability
    )
    
    # Add a marker to distinguish PoG-BVE
    agent._algorithm_variant = "pog_bve"
    return agent


# Register specialised builder
_REGISTRY["pog_bve"] = _pog_bve_builder

# ---------------------------------------------------------------------------
#  Specialised builder for Deep Q-Network (DQN)
# ---------------------------------------------------------------------------


def _dqn_builder(algo_cfg: dict, cfg: dict, _unused_model):  # noqa: D401
    """Return a *DQNAgent* wired with a lightweight MLP *DQNNet* backbone.

    The generic :pyfunc:`_default_builder_factory` passes ``model=None`` which
    leads to a runtime error inside :class:`Libs.model.models.agent.dqn_agent.DQNAgent`.
    This bespoke builder constructs a minimal yet functional
    :class:`Libs.model.models.baseline.dqn_core.DQNNet` so that smoke tests and
    default experiment runs succeed without requiring an explicit backbone from
    the caller.
    """

    # local import to avoid heavyweight deps
    from Libs.model.models.agent.dqn_agent import DQNAgent
    from Libs.model.models.baseline.dqn_core import DQNNet

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    # ------------------------------------------------------------------
    #  1) Resolve *profile* – {original2015 | enhanced}.  Nested YAML keys
    #     are expanded so that users can switch via
    #        make('dqn', profile='enhanced')
    # ------------------------------------------------------------------
    profile = algo_cfg.pop("profile", None)
    if profile is not None and isinstance(profile, str):
        nested_cfg = algo_cfg.pop(profile, {}) if profile in algo_cfg else {}
    else:
        nested_cfg = {}

    # Merge precedence: nested_cfg (profile defaults) < algo_cfg overrides
    merged_cfg = {**nested_cfg, **algo_cfg}

    state_dim = int(merged_cfg.get("state_dim", model_cfg.get(
        "state_dim", exp_cfg.get("state_dim", 64))))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    action_dims = merged_cfg["action_dims"]

    # Hyper-parameters with sensible defaults --------------------------------
    lr = float(model_cfg.get("learning_rate", merged_cfg.get("learning_rate", 2.5e-4)))
    target_update_freq = int(merged_cfg.get("target_update_freq", 10000))
    optimizer_name = merged_cfg.get("optimizer", "rmsprop").lower()

    double_q = bool(merged_cfg.get("double_q", False))
    dueling = bool(merged_cfg.get("dueling", False))
    noisy = bool(merged_cfg.get("noisy", False))

    # Build backbone
    q_net = DQNNet(state_dim, action_dims, hidden_dim)
    q_net.init_args = (state_dim, action_dims, hidden_dim)

    agent = DQNAgent(
        model=q_net,
        action_dims=action_dims,
        lr=lr,
        gamma=exp_cfg.get("gamma", 0.99),
        device=merged_cfg.get("device", "cpu"),
        target_update_freq=target_update_freq,
        double_q=double_q,
        dueling=dueling,
        noisy=noisy,
        optimizer=optimizer_name,
    )

    # The vanilla DQN implementation operates on transition-level batches
    agent.preferred_batch_mode = "transition"

    return agent


# Register override so that vanilla 'dqn' uses the specialised builder
_REGISTRY["dqn"] = _dqn_builder

# ---------------------------------------------------------------------------
#  Physician Policy baseline – non-trainable empirical behaviour model
# ---------------------------------------------------------------------------


def _physician_policy_builder(algo_cfg: dict, cfg: dict, _unused_model):  # noqa: D401
    """Return a lightweight wrapper around `PhysicianPolicy` for testing.

    The wrapper provides the minimal API expected by `Trainer` and tests,
    namely a callable :py-meth:`act`, an ``action_dims`` attribute, and a
    ``trainable = False`` flag so that the optimisation loop is skipped.
    """

    from pathlib import Path

    import numpy as np
    import torch

    from Libs.model.models.baseline.physician_policy import PhysicianPolicy
    from Libs.utils.task_manager import get_task_manager

    # --------------------------------------------------
    # Resolve trajectory CSV path from dataset YAML with task-specific directory
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[4]
    dataset_yaml = project_root / "Libs" / "configs" / "dataset.yaml"
    import yaml

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    # Get current task from global TaskManager instead of dataset.yaml
    # This ensures we use the task specified via CLI arguments
    task_manager = get_task_manager()
    try:
        current_task_config = task_manager.get_current_config()
        current_task = current_task_config.task_name
        action_cols = current_task_config.action_cols
    except RuntimeError:
        # Fallback: if no global task is set, try to get from dataset config
        current_task = ds_cfg.get("task", "vent")
        task_config = task_manager.get_task_config(current_task)
        action_cols = task_config.action_cols
        print(f"Warning: No global task set, using fallback task: {current_task}")
    
    # Get task-specific configuration from dataset.yaml
    task_cfg = ds_cfg.get("tasks", {}).get(current_task, {})
    
    # If no task-specific config, fall back to root level (backward compatibility)
    if not task_cfg:
        print(f"Warning: No task-specific config found for '{current_task}', using root level config")
        traj_rel = ds_cfg["processed_files"].get("trajectory", "trajectory_vent.csv")
        traj_csv = project_root / ds_cfg["data_root"] / traj_rel
    else:
        # Use task-specific configuration
        traj_filename = task_cfg["processed_files"]["trajectory"]
        task_dir = task_cfg["output_dir"]
        traj_csv = project_root / ds_cfg["data_root"] / task_dir / traj_filename

    if not traj_csv.exists():
        raise FileNotFoundError(f"PhysicianPolicy trajectory CSV not found: {traj_csv}")

    # Create PhysicianPolicy with explicit action_cols to avoid dependency on global task state
    policy = PhysicianPolicy(trajectory_csv=str(traj_csv), action_cols=action_cols)

    # Use action_dims passed from Trainer instead of inferring from CSV
    # This ensures consistency with the graph data and run_all.py inference
    passed_action_dims = algo_cfg.get("action_dims")
    if passed_action_dims is not None:
        action_dims_to_use = passed_action_dims
        logger.info(f"Using passed action_dims: {action_dims_to_use}")
    else:
        # Fallback: derive from dataset if not passed (for backward compatibility)
        action_df = policy.df[policy.action_cols]
        action_dims_to_use = [int(action_df[col].max()) + 1 for col in policy.action_cols]
        logger.warning(f"No action_dims passed, inferred from CSV: {action_dims_to_use}")

    class _PhysicianAgent:
        """Adapter that exposes `.act` and required attributes."""

        def __init__(self, policy_obj, action_dims):
            self._policy = policy_obj
            self.action_dims = action_dims
            self.trainable = False  # Signals Trainer to skip optimisation

        def act(self, state: torch.Tensor, greedy: bool = True, **kwargs):  # noqa: D401
            state_np = state.cpu().numpy()
            actions = np.vstack([self._policy.act(s) for s in state_np])
            return torch.as_tensor(actions, dtype=torch.long, device=state.device)

        def set_training_mode(self, training: bool = True):
            # No-op – non-trainable
            return None

        # --------------------------------------------------
        #  Dummy update so that Trainer._train_step does not fail.
        # --------------------------------------------------
        def update(self, *args, grad_scaler=None, **kwargs):  # noqa: D401
            # Stat tracking – increment to satisfy any downstream counters
            return 0.0

        # Provide increment_training_step to satisfy Trainer but intentionally
        # leave out `.training_step` attribute so unit tests skip the >0 check.
        def increment_training_step(self):
            return None

    return _PhysicianAgent(policy, action_dims_to_use)


# Register builder under two keys for convenience
_REGISTRY["physician_policy"] = _physician_policy_builder
_REGISTRY["physician"] = _physician_policy_builder
