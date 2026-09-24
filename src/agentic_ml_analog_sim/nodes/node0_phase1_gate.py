"""Node 0: Phase 1 Orchestration Gate (Programmatic).

Gates control flow and configures rewrite / debug modes for Phase 1 nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import node9_phase1_error_handler

try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:
    def ChiaFunction(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator

try:
    from chia.base.profiling import get_profiler
except ImportError:
    get_profiler = None

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node0_phase1_gate")


@ChiaFunction(resources={"control_worker": 1})
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
        Dict specifying target execution mode, candidate_id, nodes to run, bypassed nodes,
        explicit graph edges, and next_node.
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

        # Transition active candidate to EVALUATING state
        if hasattr(db, "mark_proposal_status") and callable(db.mark_proposal_status):
            try:
                db.mark_proposal_status(proposal_id, status="EVALUATING", is_active_evaluation=1)
            except Exception as e:
                logger.debug("Node 0: Could not set status EVALUATING for %s: %s", proposal_id, e)

        # Pre-flight hygiene sweep: Clean up any lingering PENDING proposals from prior sessions
        try:
            pending_proposals = db.get_proposals(status="PENDING") if hasattr(db, "get_proposals") else []
            for p in pending_proposals:
                cid = p.get("candidate_id") if isinstance(p, dict) else getattr(p, "candidate_id", None)
                if cid and cid != proposal_id:
                    c1 = p.get("counter_1_compilation", 0) if isinstance(p, dict) else getattr(p, "counter_1_compilation", 0)
                    c2 = p.get("counter_2_rewrite", 0) if isinstance(p, dict) else getattr(p, "counter_2_rewrite", 0)
                    c4 = p.get("counter_4_debug_loop", 0) if isinstance(p, dict) else getattr(p, "counter_4_debug_loop", 0)
                    if c1 > 3 or c2 > 3 or c4 > 5:
                        logger.warning("Node 0: Cleaning lingering zombie candidate '%s' with exceeded counters (c1=%s, c2=%s). Transitioning to FAILED.", cid, c1, c2)
                        node9_phase1_error_handler(
                            db=db,
                            config=config,
                            candidate_id=cid,
                            counter_name="counter_1_compilation" if c1 > 3 else "counter_2_rewrite",
                            current_count=max(c1, c2),
                            max_count=3,
                            execution_mode="cleanup",
                            error_stage=(p.get("error_stage") if isinstance(p, dict) else getattr(p, "error_stage", None)) or "compilation",
                            details="Pre-flight hygiene transition of unfinalized counter overflow.",
                        )
                    else:
                        logger.warning("Node 0: Cleaning lingering candidate '%s' abandoned in previous run. Transitioning to FAILED.", cid)
                        if hasattr(db, "mark_proposal_status"):
                            db.mark_proposal_status(
                                cid,
                                status="FAILED",
                                error_stage="interrupted_run",
                                error_message="Candidate abandoned mid-run during an interrupted prior session.",
                                is_active_evaluation=0,
                            )
        except Exception as sweep_err:
            logger.debug("Node 0: Pre-flight hygiene sweep encountered non-critical error: %s", sweep_err)

    # Mode 1: Reference Profiling Mode (Triggered by Phase 0 Node 3 / reference design)
    if is_reference or is_from_phase_0:
        mode = "reference" if is_reference else "standard"
        next_node = "node3_phase1_compiler_rewriter" if is_reference else "node1_phase1_solver_rewriter"
        nodes_to_run = [3] if is_reference else [1, 2, 3]
        bypassed_nodes = [1, 2] if is_reference else []
        logger.info("Node 0: Activating %s mode for %s", mode, proposal_id)

        # Formalized task graph edge declarations
        task_graph_edges = (
            [("node0_phase1_gate", "node3_phase1_compiler_rewriter")]
            if is_reference
            else [
                ("node0_phase1_gate", "node1_phase1_solver_rewriter"),
                ("node1_phase1_solver_rewriter", "node2_phase1_noise_rewriter"),
                ("node2_phase1_noise_rewriter", "node3_phase1_compiler_rewriter"),
            ]
        )

        if get_profiler is not None:
            try:
                get_profiler().add_info({
                    "node": "node0_phase1_gate",
                    "mode": mode,
                    "candidate_id": proposal_id,
                    "nodes_to_run": nodes_to_run,
                    "bypassed_nodes": bypassed_nodes,
                    "edges": task_graph_edges,
                })
            except Exception:
                pass

        return {
            "status": "SUCCESS",
            "mode": mode,
            "candidate_id": proposal_id,
            "next_node": next_node,
            "nodes_to_run": nodes_to_run,
            "bypassed_nodes": bypassed_nodes,
            "edges": task_graph_edges,
        }

    # Mode 2 / 3: Triggered by Node 8 (Diagnostic Proposal)
    is_node8_trigger = (
        previous_node_id in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal", "node8")
        or trigger_source in ("node_8", "node8", "node8_phase1_diagnostic_proposal")
    )
    if is_node8_trigger:
        # Discern Mode 3 (Debug Mode) vs Mode 2 (Rewrite Correction Mode)
        is_debug_mode = bool(
            debug_request
            and (
                debug_request.get("debugee_id")
                or debug_request.get("deactivate")
                or debug_request.get("deactivated_rewrites")
                or debug_request.get("action") == "debug_request"
                or debug_request.get("type") == "DEBUG_MODE_REQUEST"
            )
        )
        if is_debug_mode:
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

            task_graph_edges = [("node0_phase1_gate", "node3_phase1_compiler_rewriter")]
            if get_profiler is not None:
                try:
                    get_profiler().add_info({
                        "node": "node0_phase1_gate",
                        "mode": "debug",
                        "candidate_id": debug_id,
                        "debugee_id": debugee_id,
                        "nodes_to_run": [3],
                        "bypassed_nodes": [1, 2],
                        "edges": task_graph_edges,
                    })
                except Exception:
                    pass

            return {
                "status": "SUCCESS",
                "mode": "debug",
                "candidate_id": debug_id,
                "debugee_id": debugee_id,
                "next_node": "node3_phase1_compiler_rewriter",
                "nodes_to_run": [3],
                "bypassed_nodes": [1, 2],
                "edges": task_graph_edges,
                "compiler_steering_instructions": debug_request.get("compiler_steering_instructions") if debug_request else None,
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

            # Construct explicit task graph edges for rerun sequence
            task_graph_edges = [("node0_phase1_gate", next_node)]
            for i in range(len(nodes_to_run) - 1):
                task_graph_edges.append((node_map[nodes_to_run[i]], node_map[nodes_to_run[i + 1]]))

            logger.info("Node 0: Rewrite Correction Mode for %s (nodes: %s)", proposal_id, nodes_to_run)

            if get_profiler is not None:
                try:
                    get_profiler().add_info({
                        "node": "node0_phase1_gate",
                        "mode": "rewrite_correction",
                        "candidate_id": proposal_id,
                        "nodes_to_run": nodes_to_run,
                        "bypassed_nodes": bypassed,
                        "edges": task_graph_edges,
                    })
                except Exception:
                    pass

            return {
                "status": "SUCCESS",
                "mode": "rewrite_correction",
                "candidate_id": proposal_id,
                "next_node": next_node,
                "nodes_to_run": nodes_to_run,
                "bypassed_nodes": bypassed,
                "edges": task_graph_edges,
                "compiler_steering_instructions": debug_request.get("compiler_steering_instructions") if debug_request else None,
            }

    # Standard entry mode (N1 -> N2 -> N3)
    logger.info("Node 0: Standard entry mode for %s", proposal_id)
    std_edges = [
        ("node0_phase1_gate", "node1_phase1_solver_rewriter"),
        ("node1_phase1_solver_rewriter", "node2_phase1_noise_rewriter"),
        ("node2_phase1_noise_rewriter", "node3_phase1_compiler_rewriter"),
    ]
    if get_profiler is not None:
        try:
            get_profiler().add_info({
                "node": "node0_phase1_gate",
                "mode": "standard",
                "candidate_id": proposal_id,
                "nodes_to_run": [1, 2, 3],
                "bypassed_nodes": [],
                "edges": std_edges,
            })
        except Exception:
            pass

    return {
        "status": "SUCCESS",
        "mode": "standard",
        "candidate_id": proposal_id,
        "next_node": "node1_phase1_solver_rewriter",
        "nodes_to_run": [1, 2, 3],
        "bypassed_nodes": [],
        "edges": std_edges,
    }


# Backwards compatibility alias
node0_phase1_gate = node0_gate
