"""Node 0: Phase 1 Orchestration Gate (Programmatic).

Gates control flow and configures rewrite / debug modes for Phase 1 nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import node9_phase1_error_handler

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node0_phase1_gate")


def node0_gate(
    db: ResultDatabase,
    config: SimulationConfig,
    trigger_source: str,
    previous_node_id: str,
    proposal_id: Optional[str] = None,
    debug_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 1 Gate Node.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        trigger_source: Origin of trigger ('phase_0', 'node8', etc.).
        previous_node_id: Identifier of preceding node.
        proposal_id: Candidate ID under evaluation.
        debug_request: Optional debug request dict from Node 8.

    Returns:
        Dict specifying target execution mode, candidate_id, nodes to run, and next_node.
    """
    # Resolve candidate ID if not explicitly provided
    if proposal_id is None:
        latest = db.get_latest_proposal()
        proposal_id = latest["candidate_id"] if latest else "unknown_candidate"

    proposal = db.get_proposal(proposal_id) if proposal_id else None
    is_reference = bool(proposal and proposal.get("is_reference"))

    is_from_phase_0 = (
        trigger_source in ("phase_0", "phase0_node3")
        or previous_node_id in ("phase0_node3", "node3_validate_proposal", "phase_0", "phase_0_node_3")
    )
    # Every time we enter Phase 1 coming from Phase 0, all four counters must be reset to 0
    if is_from_phase_0 and proposal_id and proposal:
        logger.info("Node 0: Entering Phase 1 from Phase 0 for candidate '%s'. Resetting all 4 counters to 0.", proposal_id)
        db.reset_counters(proposal_id)

    # Mode 1: Reference Profiling Mode (Triggered by Phase 0 Node 3 / reference design)
    if is_reference or is_from_phase_0:
        logger.info("Node 0: Activating Reference Profiling Mode for %s", proposal_id)
        return {
            "status": "SUCCESS",
            "mode": "reference" if is_reference else "standard",
            "candidate_id": proposal_id,
            "next_node": "node3_phase1_compiler_rewriter" if is_reference else "node1_phase1_solver_rewriter",
            "nodes_to_run": [3] if is_reference else [1, 2, 3],
            "bypassed_nodes": [1, 2] if is_reference else [],
        }

    # Mode 2 / 3: Triggered by Node 8 (Diagnostic Proposal)
    if previous_node_id in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal", "node8"):
        if debug_request:
            # Mode 3: Debug Mode
            debugee_id = debug_request.get("debugee_id") or proposal_id
            deactivated = debug_request.get("deactivated_rewrites") or debug_request.get("deactivate") or "deactivate_noise"
            if isinstance(deactivated, str):
                deactivated = [deactivated]

            logger.info("Node 0: Debug Mode requested for debugee %s (deactivated: %s)", debugee_id, deactivated)

            debugee = db.get_proposal(debugee_id)
            if debugee is None:
                raise KeyError(f"Debugee candidate '{debugee_id}' not found in database.")

            # Validate debug request against DB history
            debugee_is_ref = bool(debugee.get("is_reference"))
            is_valid_request = True

            # If non-reference candidate, verify at least one configuration remains active
            deact_set = set(deactivated)
            both_deactivated = (
                "deactivate_both" in deact_set
                or ("noise" in deact_set and "solver" in deact_set)
                or ("deactivate_noise" in deact_set and "deactivate_solver" in deact_set)
            )

            if not debugee_is_ref and both_deactivated:
                is_valid_request = False
                logger.warning("Node 0: Invalid debug request - non-reference candidate cannot deactivate both solver and noise.")

            if not is_valid_request:
                curr_counter_3 = debugee.get("counter_3_debug_gate", 0) + 1
                db.update_counters(debugee_id, counter_3_debug_gate=curr_counter_3)

                counter_limits = getattr(config, "counter_limits", None)
                max_counter_3 = None
                if counter_limits is not None:
                    max_counter_3 = getattr(counter_limits, "max_counter_3_debug_gate", None)
                    if max_counter_3 is None and isinstance(counter_limits, dict):
                        max_counter_3 = counter_limits.get("max_counter_3_debug_gate")
                if max_counter_3 is None:
                    limits = getattr(config, "experiment_limits", None) or getattr(config, "budget", None)
                    max_counter_3 = getattr(limits, "max_node_retries", 3) if limits else 3

                if curr_counter_3 > max_counter_3:
                    logger.error("Node 0: Counter 3 limit exceeded (%d > %d) for debugee %s", curr_counter_3, max_counter_3, debugee_id)
                    return node9_phase1_error_handler(
                        db=db,
                        config=config,
                        candidate_id=debugee_id,
                        counter_name="counter_3_debug_gate",
                        current_count=curr_counter_3,
                        max_count=max_counter_3,
                        execution_mode="debug",
                        error_stage="node0_phase1_gate",
                        details="Invalid debug request limit exceeded (Counter 3)",
                    )
                else:
                    return {
                        "status": "REJECTED",
                        "reason": "invalid_debug_request",
                        "counter_3": curr_counter_3,
                        "candidate_id": debugee_id,
                        "next_node": "node8_phase1_diagnostic_proposal",
                    }

            # Create debug candidate and quarantine debugee
            debug_id = db.create_debug_candidate(debugee_id, deactivated_rewrites=deactivated)
            db.quarantine_candidate(debugee_id)
            # Reset counter 3 on debugee upon successful valid debug request acceptance
            db.update_counters(debugee_id, counter_3_debug_gate=0)

            return {
                "status": "SUCCESS",
                "mode": "debug",
                "candidate_id": debug_id,
                "debugee_id": debugee_id,
                "next_node": "node3_phase1_compiler_rewriter",
                "nodes_to_run": [3],
                "bypassed_nodes": [1, 2],
            }

        else:
            # Mode 2: Rewrite Correction Mode
            nodes_to_run: List[int] = debug_request.get("nodes_to_rerun") if debug_request else None  # type: ignore
            if not nodes_to_run:
                # Default to full rewrite pipeline if unspecified
                nodes_to_run = [1, 2, 3]

            # Node 3 is ALWAYS executed
            if 3 not in nodes_to_run:
                nodes_to_run.append(3)
            nodes_to_run = sorted(list(set(nodes_to_run)))

            bypassed = [n for n in [1, 2] if n not in nodes_to_run]
            first_node = nodes_to_run[0]
            node_map = {1: "node1_phase1_solver_rewriter", 2: "node2_phase1_noise_rewriter", 3: "node3_phase1_compiler_rewriter"}
            next_node = node_map[first_node]

            logger.info("Node 0: Rewrite Correction Mode for %s (nodes: %s)", proposal_id, nodes_to_run)
            return {
                "status": "SUCCESS",
                "mode": "rewrite_correction",
                "candidate_id": proposal_id,
                "next_node": next_node,
                "nodes_to_run": nodes_to_run,
                "bypassed_nodes": bypassed,
            }

    # Standard entry mode (N1 -> N2 -> N3)
    logger.info("Node 0: Standard entry mode for %s", proposal_id)
    return {
        "status": "SUCCESS",
        "mode": "standard",
        "candidate_id": proposal_id,
        "next_node": "node1_phase1_solver_rewriter",
        "nodes_to_run": [1, 2, 3],
        "bypassed_nodes": [],
    }


# Backwards compatibility alias
node0_phase1_gate = node0_gate
