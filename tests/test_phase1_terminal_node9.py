"""Unit tests for Phase 1 terminal Node 9 invocation and EVALUATING candidate lifecycle."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from agentic_ml_analog_sim.nodes.node0_phase1_gate import node0_phase1_gate
from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import node9_phase1_error_handler
from agentic_ml_analog_sim.phase_1 import run_phase_1, Phase1Result
from agentic_ml_analog_sim.db.database import ResultDatabase


def test_node0_hygiene_sweep_transitions_stale_pending():
    mock_db = MagicMock(spec=ResultDatabase)
    # Mock returns two stale pending proposals
    mock_db.get_proposals.return_value = [
        {
            "candidate_id": "cand_stale_failed",
            "status": "PENDING",
            "counter_1_compilation": 4,
            "error_stage": "compilation",
            "error_message": "Lowering error",
        },
        {
            "candidate_id": "cand_stale_abandoned",
            "status": "PENDING",
            "counter_1_compilation": 0,
            "counter_2_rewrite": 0,
            "counter_3_debug_gate": 0,
            "counter_4_debug_loop": 0,
            "error_stage": None,
        },
    ]

    mock_config = MagicMock()
    mock_config.counter_limits.max_counter_1_compilation = 3
    mock_config.counter_limits.max_counter_2_rewrite = 3

    # Call node 0 for a new candidate
    res = node0_phase1_gate(
        db=mock_db,
        config=mock_config,
        trigger_source="phase_0",
        previous_node_id="phase0_node3",
        proposal_id="cand_new_01",
    )

    assert res["status"] == "SUCCESS"
    assert res["candidate_id"] == "cand_new_01"

    # Verify cand_new_01 was transitioned to EVALUATING
    mock_db.mark_proposal_status.assert_any_call(
        "cand_new_01",
        status="EVALUATING",
        is_active_evaluation=1,
    )

    # Verify cand_stale_failed was marked FAILED via node9
    found_stale_failed = any(
        call.kwargs.get("candidate_id") == "cand_stale_failed"
        and call.kwargs.get("status") == "FAILED"
        for call in mock_db.mark_proposal_status.call_args_list
    )
    assert found_stale_failed, "cand_stale_failed was not marked as FAILED"

    # Verify cand_stale_abandoned was marked FAILED
    found_stale_abandoned = any(
        (call.args and call.args[0] == "cand_stale_abandoned" or call.kwargs.get("candidate_id") == "cand_stale_abandoned")
        and call.kwargs.get("status") == "FAILED"
        for call in mock_db.mark_proposal_status.call_args_list
    )
    assert found_stale_abandoned, "cand_stale_abandoned was not marked as FAILED"


def test_phase1_driver_invokes_node9_on_node5_terminated():
    mock_db = MagicMock(spec=ResultDatabase)
    mock_db.get_proposal.return_value = {
        "candidate_id": "cand_test_01",
        "solver": "euler",
        "noise_model": "L0_static_mismatch",
        "is_reference": 0,
    }

    mock_config = MagicMock()
    mock_config.budget.max_loop_cycles = 5
    mock_config.counter_limits.max_counter_1_compilation = 3
    mock_config.device = "cpu"

    def fake_dispatch(node_fn, *args, **kwargs):
        name = getattr(node_fn, "__name__", str(node_fn))
        if "node0" in name:
            return {
                "status": "SUCCESS",
                "mode": "standard",
                "candidate_id": "cand_test_01",
                "nodes_to_run": [1, 2, 3],
            }
        elif "node1" in name or "node2" in name or "node3" in name:
            return {"status": "SUCCESS"}
        elif "node4" in name:
            return {"status": "SUCCESS", "compiled_model": MagicMock()}
        elif "node5" in name:
            return {
                "status": "TERMINATED_MAX_CYCLES",
                "counter_1": 4,
                "counter_name": "counter_1_compilation",
                "max_count": 3,
                "error_logs": "LoweringException: failed to lower kernel",
                "error_stage": "compilation",
            }
        return {"status": "SUCCESS"}

    with (
        patch("agentic_ml_analog_sim.phase_1.load_simulation_config", return_value=mock_config),
        patch("agentic_ml_analog_sim.phase_1._dispatch_node", side_effect=fake_dispatch),
        patch("agentic_ml_analog_sim.phase_1.node9_phase1_error_handler") as mock_n9,
    ):
        mock_n9.return_value = {
            "status": "COMPLETED",
            "candidate_id": "cand_test_01",
            "db_snapshot": "/tmp/results_snapshot.db",
        }

        res = run_phase_1(
            config_path="mock/path/config.yaml",
            db_path=mock_db,
            candidate_id="cand_test_01",
        )

        assert res.action == "TERMINATED_MAX_CYCLES"
        # Node 9 should have been dispatched on the driver
        mock_n9.assert_called_once()
        n9_kwargs = mock_n9.call_args.kwargs
        assert n9_kwargs["candidate_id"] == "cand_test_01"
        assert n9_kwargs["counter_name"] == "counter_1_compilation"
        assert n9_kwargs["current_count"] == 4
        assert n9_kwargs["error_stage"] == "compilation"


def test_node3_and_node4_unresolvable_num_classes_route_to_node9():
    """Verify that node3 and node4 invoke node9 directly on unresolvable num_classes."""
    from agentic_ml_analog_sim.nodes.node3_phase1_compiler_rewriter import node3_phase1_compiler_rewriter
    from agentic_ml_analog_sim.nodes.node4_phase1_compilation_runner import node4_phase1_compilation_runner

    mock_db = MagicMock(spec=ResultDatabase)
    mock_db.get_proposal.return_value = {
        "candidate_id": "cand_cfg_err",
        "solver": "euler",
        "noise_model": "none",
    }

    mock_config = MagicMock()
    mock_config.reference_design.n_oscillators = 1024
    mock_config.reference_design.n_conditional_oscillators = 100
    mock_config.reference_design.family = "custom_unknown_family"
    mock_config.reference_design.num_classes = None
    mock_config.num_classes = None

    # Test Node 3
    res3 = node3_phase1_compiler_rewriter(
        db=mock_db,
        config=mock_config,
        candidate_id="cand_cfg_err",
        target_device="cpu",
    )
    assert res3["status"] == "TERMINATED_MAX_CYCLES"
    assert res3["counter_name"] == "configuration_error"
    assert "node3_phase1_compiler_rewriter" in res3["error_stage"]
    assert "Unrecoverable configuration error" in res3["error_message"]

    # Test Node 4
    res4 = node4_phase1_compilation_runner(
        db=mock_db,
        config=mock_config,
        candidate_id="cand_cfg_err",
        target_device="cpu",
    )
    assert res4["status"] == "TERMINATED_MAX_CYCLES"
    assert res4["counter_name"] == "configuration_error"
    assert "node4_phase1_compilation_runner" in res4["error_stage"]
    assert "Unrecoverable configuration error" in res4["error_message"]


def test_phase1_driver_terminates_on_node6_configuration_error():
    """Verify phase 1 driver stops immediately on Node 6 TERMINATED_MAX_CYCLES and does NOT call Node 8."""
    mock_db = MagicMock(spec=ResultDatabase)
    mock_db.get_proposal.return_value = {
        "candidate_id": "cand_n6_term",
        "solver": "rk4",
        "noise_model": "none",
        "is_reference": 0,
    }

    mock_config = MagicMock()
    mock_config.budget.max_loop_cycles = 5
    mock_config.counter_limits.max_counter_1_compilation = 3
    mock_config.device = "cpu"

    invoked_nodes = []

    def fake_dispatch(node_fn, *args, **kwargs):
        name = getattr(node_fn, "__name__", str(node_fn))
        invoked_nodes.append(name)
        if "node0" in name:
            return {
                "status": "SUCCESS",
                "mode": "standard",
                "candidate_id": "cand_n6_term",
                "nodes_to_run": [3],
            }
        elif "node3" in name:
            return {"status": "SUCCESS"}
        elif "node4" in name:
            return {"status": "SUCCESS", "compiled_model": MagicMock()}
        elif "node5" in name:
            return {"status": "PASSED"}
        elif "node6" in name:
            return {
                "status": "TERMINATED_MAX_CYCLES",
                "counter_name": "configuration_error",
                "details": "Node 6 could not resolve a valid positive 'num_classes'",
                "next_node": None,
            }
        return {"status": "SUCCESS"}

    with (
        patch("agentic_ml_analog_sim.phase_1.load_simulation_config", return_value=mock_config),
        patch("agentic_ml_analog_sim.phase_1._dispatch_node", side_effect=fake_dispatch),
    ):
        res = run_phase_1(
            config_path="mock/path/config.yaml",
            db_path=mock_db,
            candidate_id="cand_n6_term",
        )

        assert res.action == "TERMINATED_MAX_CYCLES"
        assert "Node 6 terminated via Node 9" in res.details
        assert not any("node8" in n for n in invoked_nodes), "Node 8 should NOT have been invoked!"

