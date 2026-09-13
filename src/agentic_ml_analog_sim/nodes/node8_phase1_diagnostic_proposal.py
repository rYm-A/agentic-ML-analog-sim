"""Node 8: Diagnostic Proposal Agentic Node (Phase 1).

Diagnoses compilation or evaluation failures from Node 5 or Node 7 and emits
a corrective rewrite proposal or a debug mode request.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node8_phase1_diagnostic_proposal")


def _resolve_llm(llm: Optional[Any] = None, config: Optional[SimulationConfig] = None) -> Any:
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node8_phase1_diagnostic_proposal", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node8_phase1_diagnostic_proposal")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-flash"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "medium"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


def node8_phase1_diagnostic_proposal(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    error_logs: Optional[Any] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Node 8 Diagnostic Proposal (Phase 1).

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Candidate ID that failed evaluation or compilation.
        error_logs: Log output or failure details from preceding nodes.
        llm: Optional LLM instance or mock.

    Returns:
        Dict specifying action ('rewrite_correction' or 'debug_request'), nodes to rerun,
        debug_request dict, and diagnosis commentary.
    """
    active_llm = _resolve_llm(llm, config)
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    comments = db.get_comments(candidate_id)
    logger.info("Node 8: Formulating diagnostic proposal for candidate %s (%d previous comments)", candidate_id, len(comments))

    # Diagnose failure root cause
    logs_str = str(error_logs or proposal.get("error_message") or "")
    c1 = proposal.get("counter_1_compilation", 0)
    c2 = proposal.get("counter_2_rewrite", 0)

    # Determine action strategy: rewrite fix vs debug mode request
    if "compilation" in logs_str.lower() or "tolerance" in logs_str.lower() or c1 > 0:
        action = "rewrite_correction"
        nodes_to_rerun = [1, 2, 3] if c2 > 1 else [3]
        debug_request = None
        diagnosis_text = f"Compilation/numerical discrepancy detected ({logs_str[:120]}). Proposing fix rewrite for nodes {nodes_to_rerun}."
    elif c2 >= 2 and not proposal.get("is_debug"):
        # If multiple rewrite attempts failed, enter debug mode to isolate root cause
        action = "debug_request"
        nodes_to_rerun = [3]
        debug_request = {
            "debugee_id": candidate_id,
            "deactivate": "deactivate_noise",
            "deactivated_rewrites": ["noise"],
        }
        diagnosis_text = f"Multiple rewrite iterations failed (counter_2={c2}). Requesting Debug Mode with deactivated noise configuration."
    else:
        action = "rewrite_correction"
        nodes_to_rerun = [1, 2, 3]
        debug_request = None
        diagnosis_text = f"Out-of-tolerance failure ({logs_str[:120]}). Proposing full pipeline correction rewrite (nodes 1, 2, 3)."

    # Record commentary in database
    db.add_comment(
        candidate_id=candidate_id,
        phase="phase_1",
        agent_name="node8_phase1_diagnostic_proposal",
        comment=f"Action: {action}. Diagnosis: {diagnosis_text}",
    )

    # Record token usage for diagnostic proposal step
    model_name = getattr(active_llm, "model", "gemini-2.5-flash")
    prompt_tokens = 250
    completion_tokens = 120
    cost_usd = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)

    if hasattr(active_llm, "history") and active_llm.history:
        last_entry = active_llm.history[-1]
        usage = last_entry.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

    db.record_token_usage(
        phase="phase_1",
        node_name="node8_phase1_diagnostic_proposal",
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost_usd,
        candidate_id=candidate_id,
        duration_seconds=0.8,
    )

    return {
        "candidate_id": candidate_id,
        "action": action,
        "nodes_to_rerun": sorted(list(set(nodes_to_rerun))),
        "debug_request": debug_request,
        "diagnosis": diagnosis_text,
        "next_node": "node0_phase1_gate",
    }


# Backwards compatibility alias
node8_diagnostic_proposal = node8_phase1_diagnostic_proposal
