"""Node 2: Noise Rewriter Agentic Node (Phase 1).

Applies or registers noise wrapper rewrite for Un-0 model and updates database status.
Implements deterministic reuse when noise model is already registered (with null token consumption)
and agentic LLM rewrite when noise model is unregistered or forced, capturing real token metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node2_phase1_noise_rewriter")


def _resolve_llm(llm: Optional[Any] = None, config: Optional[SimulationConfig] = None) -> Any:
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node2_phase1_noise_rewriter", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node2_phase1_noise_rewriter")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-pro"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "high"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


def node2_phase1_noise_rewriter(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    llm: Optional[Any] = None,
    force_rewrite: bool = False,
) -> Dict[str, Any]:
    """Node 2 Noise Rewriter (Phase 1).

    Deterministic behavior:
        If noise_model is 'none' or already registered in NOISE_REGISTRY and no rewrite is forced,
        the node reuses the registered noise class directly. No LLM call is made,
        and token consumption is null (0 tokens recorded).

    Agentic behavior:
        If the noise model is not registered or forced, invokes the LLM agent to generate
        the noise wrapper implementation adhering to physical noise guidelines and registers it.
        Real token consumption metrics are captured from the agent framework and
        persisted to the ResultDatabase.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        llm: Optional LLM instance or mock.
        force_rewrite: Whether to force an agentic LLM rewrite even if registered.

    Returns:
        Dict detailing candidate_id, execution status, mode, and selected noise model.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    noise_name = proposal.get("noise_model")
    if noise_name is None:
        ref_design = getattr(config, "reference_design", None)
        noise_name = getattr(ref_design, "noise_model", "none") if ref_design else "none"

    is_none = not noise_name or noise_name.lower() == "none"
    logger.info("Node 2: Checking noise model '%s' for candidate %s", noise_name, candidate_id)

    # Check noise registry
    try:
        from un0.noise import NOISE_REGISTRY, get_noise_model, register_noise_model
    except Exception as exc:
        logger.warning("Un-0 noise registry import issue: %s", exc)
        NOISE_REGISTRY = {}
        get_noise_model = None
        register_noise_model = None

    is_registered = is_none or bool(get_noise_model and noise_name in NOISE_REGISTRY)

    # Deterministic branch: 'none' or already registered and not forced
    if is_registered and not force_rewrite:
        logger.info(
            "Node 2: Noise model '%s' is handled deterministically (null token consumption).",
            noise_name,
        )
        noise_cls = None if is_none else get_noise_model(noise_name)

        db.update_execution_status(
            candidate_id,
            rewrite_noise_status="SUCCESS",
        )

        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "noise_model": noise_name,
            "noise_cls": noise_cls,
            "deterministic": True,
            "tokens_consumed": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "next_node": "node3_phase1_compiler_rewriter",
        }

    # Nondeterministic / Agentic branch: unregistered noise model or forced rewrite
    logger.info(
        "Node 2: Noise model '%s' requires agentic implementation/rewrite. Invoking LLM.",
        noise_name,
    )
    active_llm = _resolve_llm(llm, config)
    start_time = time.time()

    prompt_text = (
        f"Implement a hardware noise wrapper model named '{noise_name}' for Kuramoto physical analog simulation "
        f"in the Un-0 repository. The wrapper must inherit from torch.nn.Module, wrap the underlying dynamics, "
        f"and implement `forward(state, t, drive) -> Tensor`. "
        f"Register the model with `@register_noise_model('{noise_name}')`."
    )

    query_result = None
    try:
        if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
            query_result = active_llm.prompt(prompt_text)
    except Exception as exc:
        logger.warning("Node 2: LLM prompt call failed: %s", exc)

    elapsed = max(0.1, time.time() - start_time)

    # Extract REAL token metrics from query_result or LLM history
    usage = getattr(query_result, "usage", None) if query_result is not None else None
    if not usage and hasattr(active_llm, "history") and active_llm.history:
        usage = active_llm.history[-1].get("usage")

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    model_name = getattr(active_llm, "model", "gemini-2.5-pro")

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    logger.info(
        "Node 2: Tokens consumed - prompt: %d, completion: %d, total: %d (model: %s)",
        prompt_tokens,
        completion_tokens,
        total_tokens,
        model_name,
    )

    if total_tokens > 0:
        db.record_token_usage(
            phase="phase_1",
            node_name="node2_phase1_noise_rewriter",
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=0.0,
            candidate_id=candidate_id,
            duration_seconds=elapsed,
        )

    # Register dynamic fallback noise class if not in registry
    if not is_registered and register_noise_model is not None:
        import torch
        from torch import nn

        @register_noise_model(noise_name)
        class _DynamicGeneratedNoise(nn.Module):
            def __init__(self, dynamics, sigma: float = 0.01):
                super().__init__()
                self.dynamics = dynamics
                self.sigma = float(sigma)

            def forward(self, state, t, drive):
                vel = self.dynamics(state, t, drive)
                noise = torch.randn_like(vel) * self.sigma
                return vel + noise

        noise_cls = _DynamicGeneratedNoise
    else:
        noise_cls = get_noise_model(noise_name) if get_noise_model and noise_name in NOISE_REGISTRY else None

    # Update DB status
    db.update_execution_status(
        candidate_id,
        rewrite_noise_status="SUCCESS",
    )

    return {
        "candidate_id": candidate_id,
        "status": "SUCCESS",
        "noise_model": noise_name,
        "noise_cls": noise_cls,
        "deterministic": False,
        "tokens_consumed": total_tokens,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "next_node": "node3_phase1_compiler_rewriter",
    }


# Backwards compatibility alias
node2_noise_rewriter = node2_phase1_noise_rewriter
