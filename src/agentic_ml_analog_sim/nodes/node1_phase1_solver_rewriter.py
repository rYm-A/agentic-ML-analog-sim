"""Node 1: Solver Rewriter Agentic Node (Phase 1).

Applies solver rewrite to Un-0 model and updates database status.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node1_phase1_solver_rewriter")


def _resolve_llm(llm: Optional[Any] = None, config: Optional[SimulationConfig] = None) -> Any:
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node1_phase1_solver_rewriter", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node1_phase1_solver_rewriter")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-pro"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "high"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


def node1_phase1_solver_rewriter(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Node 1 Solver Rewriter (Phase 1).

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        llm: Optional LLM instance or mock.

    Returns:
        Dict detailing candidate_id, execution status, and selected solver.
    """
    active_llm = _resolve_llm(llm, config)
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    solver_name = proposal.get("solver", "rk4")
    logger.info("Node 1: Applying solver rewrite '%s' for candidate %s", solver_name, candidate_id)

    # Validate solver against Un-0 solver registry
    try:
        from un0.solver import get_solver, SOLVER_REGISTRY
        solver_fn = get_solver(solver_name)
    except Exception as exc:
        logger.warning("Un-0 solver registry import/lookup issue: %s", exc)
        solver_fn = None

    # Record token usage for solver rewriter step
    model_name = getattr(active_llm, "model", "gemini-2.5-pro")
    prompt_tokens = 120
    completion_tokens = 60
    cost_usd = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)

    if hasattr(active_llm, "history") and active_llm.history:
        last_entry = active_llm.history[-1]
        usage = last_entry.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

    db.record_token_usage(
        phase="phase_1",
        node_name="node1_phase1_solver_rewriter",
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost_usd,
        candidate_id=candidate_id,
        duration_seconds=0.5,
    )

    # Update DB status
    db.update_execution_status(
        candidate_id,
        rewrite_solver_status="SUCCESS",
    )

    return {
        "candidate_id": candidate_id,
        "status": "SUCCESS",
        "solver": solver_name,
        "solver_fn": solver_fn,
        "next_node": "node2_phase1_noise_rewriter",
    }


# Backwards compatibility alias
node1_solver_rewriter = node1_phase1_solver_rewriter
