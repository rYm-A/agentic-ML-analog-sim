"""Node 7: Evaluation Gate Node (Phase 1, Programmatic).

Evaluates simulation metrics against accuracy and FID tolerances.
Manages Counter ID 2 (counter_2_rewrite) and Counter ID 4 (counter_4_debug_loop).
Enforces Debug Mode invariants and blocks invalid transition to Phase 2.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import node9_phase1_error_handler

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node7_phase1_evaluation_gate")


def node7_phase1_evaluation_gate(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
) -> Dict[str, Any]:
    """Node 7 Evaluation Gate (Phase 1).

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.

    Returns:
        Dict specifying outcome status (PASSED/FAILED), next_node, and counter states.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    is_debug = bool(proposal.get("is_debug"))
    is_reference = bool(proposal.get("is_reference"))

    logger.info("Node 7: Evaluating gate checks for candidate %s (is_debug=%s, is_ref=%s)", candidate_id, is_debug, is_reference)

    # Branch 1: Debug Mode Handling
    if is_debug:
        # Enforce Debug Mode Invariants
        active_debug = db.get_active_debug_candidate()
        parent_id = proposal.get("debug_parent_id")
        quarantined = db.get_quarantined_candidate(parent_id)

        if active_debug is None or quarantined is None:
            logger.warning("Node 7: Debug mode invariant check warning for candidate %s (parent %s)", candidate_id, parent_id)

        # Increment Counter ID 4 (counter_4_debug_loop)
        curr_counter_4 = proposal.get("counter_4_debug_loop", 0) + 1
        db.update_counters(candidate_id, counter_4_debug_loop=curr_counter_4)

        counter_limits = getattr(config, "counter_limits", None)
        max_counter_4 = None
        if counter_limits is not None:
            max_counter_4 = getattr(counter_limits, "max_counter_4_debug_loop", None)
            if max_counter_4 is None and isinstance(counter_limits, dict):
                max_counter_4 = counter_limits.get("max_counter_4_debug_loop")
        if max_counter_4 is None:
            max_counter_4 = 5
        if curr_counter_4 > max_counter_4:
            logger.error("Node 7: Counter 4 limit exceeded (%d > %d) in debug mode for %s", curr_counter_4, max_counter_4, candidate_id)
            return node9_phase1_error_handler(
                db=db,
                config=config,
                candidate_id=candidate_id,
                counter_name="counter_4_debug_loop",
                current_count=curr_counter_4,
                max_count=max_counter_4,
                execution_mode="debug",
                error_stage="node7_phase1_evaluation_gate",
                details="counter_4_debug_loop_exceeded",
            )

        # BLOCKS transition to Phase 2 during Debug Mode! Send control to Node 8 for diagnosis/resolution
        logger.info("Node 7: Debug mode iteration %d complete for %s. Blocking Phase 2 transition; routing to Node 8.", curr_counter_4, candidate_id)
        return {
            "status": "EVALUATED_DEBUG",
            "counter_4": curr_counter_4,
            "candidate_id": candidate_id,
            "next_node": "node8_phase1_diagnostic_proposal",
        }

    # Branch 2: Reference Proposal Handling
    if is_reference:
        db.update_counters(candidate_id, counter_2_rewrite=0)
        db.mark_proposal_status(candidate_id, status="COMPLETED")
        db.compute_pareto_front()

        logger.info("Node 7: Reference proposal %s successfully evaluated and marked COMPLETED. Routing to Phase 2.", candidate_id)
        return {
            "status": "PASSED",
            "is_reference": True,
            "candidate_id": candidate_id,
            "next_node": "phase2_node1",
        }

    # Branch 3: Standard / Rewrite Candidate Evaluation
    solver_acc = getattr(config.tolerances, "solver_accuracy", None)
    rtol = getattr(solver_acc, "rtol", 5e-2)
    atol = getattr(solver_acc, "atol", 5e-2)
    max_fid_deg = getattr(solver_acc, "max_fid_degradation", 2.0)

    rel_error = proposal.get("relative_error")
    accuracy_fid = proposal.get("accuracy_fid")

    # Evaluate metric tolerance bounds
    within_tolerance = True
    reasons = []

    if rel_error is not None and rel_error > rtol:
        within_tolerance = False
        reasons.append(f"relative_error ({rel_error:.4f}) > rtol ({rtol:.4f})")

    if accuracy_fid is not None and accuracy_fid > max_fid_deg:
        within_tolerance = False
        reasons.append(f"accuracy_fid ({accuracy_fid:.4f}) > max_fid_degradation ({max_fid_deg:.4f})")

    if within_tolerance:
        # Success: reset counter 2, mark COMPLETED, compute Pareto front, route to Phase 2
        db.update_counters(candidate_id, counter_2_rewrite=0)
        db.mark_proposal_status(candidate_id, status="COMPLETED")

        # If this proposal was derived after a debug session, reactivate parent if needed
        if proposal.get("debug_parent_id"):
            try:
                db.reactivate_candidate(proposal["debug_parent_id"])
            except Exception:
                pass

        db.compute_pareto_front()

        logger.info("Node 7: Candidate %s PASSED evaluation gate. Routing to Phase 2.", candidate_id)
        return {
            "status": "PASSED",
            "candidate_id": candidate_id,
            "next_node": "phase2_node1",
        }
    else:
        # Failure: manage Counter ID 2 (counter_2_rewrite)
        curr_counter_2 = proposal.get("counter_2_rewrite", 0) + 1
        db.update_counters(candidate_id, counter_2_rewrite=curr_counter_2)

        counter_limits = getattr(config, "counter_limits", None)
        max_retries = None
        if counter_limits is not None:
            max_retries = getattr(counter_limits, "max_counter_2_rewrite", None)
            if max_retries is None and isinstance(counter_limits, dict):
                max_retries = counter_limits.get("max_counter_2_rewrite")
        if max_retries is None:
            limits = getattr(config, "experiment_limits", None) or getattr(config, "budget", None)
            max_retries = getattr(limits, "max_node_retries", 3) if limits else 3

        reason_str = "; ".join(reasons)
        if curr_counter_2 > max_retries:
            logger.error("Node 7: Counter 2 limit exceeded (%d > %d) for candidate %s: %s", curr_counter_2, max_retries, candidate_id, reason_str)
            return node9_phase1_error_handler(
                db=db,
                config=config,
                candidate_id=candidate_id,
                counter_name="counter_2_rewrite",
                current_count=curr_counter_2,
                max_count=max_retries,
                execution_mode="standard",
                error_stage="node7_phase1_evaluation_gate",
                details=f"counter_2_rewrite_exceeded: {reason_str}",
            )
        else:
            logger.warning("Node 7: Candidate %s out-of-tolerance (%s). Counter 2 = %d/%d. Routing to Node 8.", candidate_id, reason_str, curr_counter_2, max_retries)
            return {
                "status": "FAILED",
                "reason": "out_of_tolerance",
                "counter_2": curr_counter_2,
                "error_details": reason_str,
                "candidate_id": candidate_id,
                "next_node": "node8_phase1_diagnostic_proposal",
            }


# Backwards compatibility alias
node7_evaluation_gate = node7_phase1_evaluation_gate
