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
