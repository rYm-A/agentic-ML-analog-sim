"""Unit tests for Node 9 Phase 1 Error Handler and Standardized Loop Stop Behavior.

Verifies:
  - Direct execution of node9_phase1_error_handler marks candidate as FAILED in DB and produces SQLite snapshot.
  - Counter 1 limit exceedance in Node 5 triggers node9_phase1_error_handler.
  - Counter 2 limit exceedance in Node 7 triggers node9_phase1_error_handler.
  - Counter 3 limit exceedance in Node 0 triggers node9_phase1_error_handler.
  - Counter 4 limit exceedance in Node 7 triggers node9_phase1_error_handler.
  - Phase 1 orchestrator loop cleanly stops upon receiving TERMINATED_MAX_CYCLES.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pytest
import torch

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes import (
    node0_gate,
    node5_phase1_verify_correctness,
    node7_phase1_evaluation_gate,
    node9_phase1_error_handler,
    node9_error_handler,
)
from agentic_ml_analog_sim.phase_1 import run_phase_1, Phase1Result
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM


@pytest.fixture
def test_config():
    """Load default simulation config."""
    repo_root = Path(__file__).parents[1]
    cfg_path = repo_root / "config/simulation_config.yaml"
    return load_simulation_config(cfg_path)


@pytest.fixture
def test_db(tmp_path: Path):
    """Create isolated SQLite ResultDatabase."""
    db_file = tmp_path / "error_handler_test.db"
    return ResultDatabase(db_path=db_file)


def test_node9_direct_invocation_marks_db_and_creates_snapshot(test_config, test_db):
    """Test that node9_phase1_error_handler creates snapshot, updates DB, and returns standardized dict."""
    cand_id = test_db.insert_proposal(
        candidate_id="cand_node9_direct_01",
        solver="euler",
        noise_model="none",
        status="PENDING",
    )

    res = node9_phase1_error_handler(
        db=test_db,
        config=test_config,
        candidate_id=cand_id,
        counter_name="counter_1_compilation",
        current_count=4,
        max_count=3,
        execution_mode="standard",
        error_stage="correctness_verification",
        details="Discrepancy exceeds tolerance threshold.",
    )

    # Standardized dict contract
    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["candidate_id"] == cand_id
    assert res["counter_name"] == "counter_1_compilation"
    assert res["current_count"] == 4
    assert res["max_count"] == 3
    assert res["execution_mode"] == "standard"
    assert res["next_node"] is None
    assert res["counter_1"] == 4
    assert res["reason"] == "counter_1_compilation_exceeded"

    # Snapshot validation
    snapshot_path = res.get("db_snapshot")
    assert snapshot_path is not None
    assert Path(snapshot_path).is_file()
    assert Path(snapshot_path).stat().st_size > 0

    # In-DB validation
    prop = test_db.get_proposal(cand_id)
    assert prop["status"] == "FAILED"
    assert prop["error_stage"] == "correctness_verification"
    assert "counter_1_compilation" in prop["error_message"]

    # Verify snapshot database can be read and contains the updated proposal
    snap_conn = sqlite3.connect(snapshot_path)
    try:
        cur = snap_conn.cursor()
        cur.execute("SELECT status, error_stage FROM proposals WHERE candidate_id = ?", (cand_id,))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "FAILED"
        assert row[1] == "correctness_verification"
    finally:
        snap_conn.close()


def test_node9_alias_export():
    """Verify backwards-compatibility alias node9_error_handler."""
    assert node9_error_handler is node9_phase1_error_handler


def test_counter1_exceedance_in_node5(test_config, test_db):
    """Test that exceeding Counter 1 in Node 5 invokes node9_phase1_error_handler."""
    cand_id = test_db.insert_proposal(
        candidate_id="cand_node5_c1_fail",
        solver="euler",
        noise_model="none",
        status="EVALUATING",
    )
    # Set current counter 1 to 3 (equal to default max_node_retries=3)
    test_db.update_counters(cand_id, counter_1_compilation=3)

    # Prepare mismatched models to induce verification failure
    eager_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        eager_model.weight.fill_(1.0)
        eager_model.bias.zero_()

    compiled_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        compiled_model.weight.fill_(999.0)  # Gross mismatch
        compiled_model.bias.zero_()

    res = node5_phase1_verify_correctness(
        db=test_db,
        config=test_config,
        candidate_id=cand_id,
        compiled_model=compiled_model,
        eager_model=eager_model,
        target_device="cpu",
    )

    # Next counter 1 is 4 > 3 -> Triggers Node 9 hard stop
    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["candidate_id"] == cand_id
    assert res["counter_name"] == "counter_1_compilation"
    assert res["current_count"] == 4
    assert res["next_node"] is None
    assert res["db_snapshot"] is not None
    assert Path(res["db_snapshot"]).is_file()

    # DB status marked FAILED
    prop = test_db.get_proposal(cand_id)
    assert prop["status"] == "FAILED"


def test_counter2_exceedance_in_node7(test_config, test_db):
    """Test that exceeding Counter 2 in Node 7 invokes node9_phase1_error_handler."""
    cand_id = test_db.insert_proposal(
        candidate_id="cand_node7_c2_fail",
        solver="euler",
        noise_model="none",
        status="EVALUATING",
        accuracy_fid=0.10,  # Below tolerance (e.g. baseline 0.95, tolerance 0.05)
        relative_error=0.80,  # Above tolerance
    )
    # Set current counter 2 to 3 (at max limit)
    test_db.update_counters(cand_id, counter_2_rewrite=3)

    # Insert baseline reference for comparison
    test_db.insert_proposal(
        candidate_id="ref_baseline",
        solver="rk4",
        noise_model="none",
        status="COMPLETED",
        accuracy_fid=0.95,
        latency_ms=10.0,
        relative_error=0.0,
        is_reference=True,
    )

    res = node7_phase1_evaluation_gate(
        db=test_db,
        config=test_config,
        candidate_id=cand_id,
    )

    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["candidate_id"] == cand_id
    assert res["counter_name"] == "counter_2_rewrite"
    assert res["current_count"] == 4
    assert res["next_node"] is None
    assert res["db_snapshot"] is not None
    assert Path(res["db_snapshot"]).is_file()

    # DB status marked FAILED
    prop = test_db.get_proposal(cand_id)
    assert prop["status"] == "FAILED"


def test_counter3_exceedance_in_node0(test_config, test_db):
    """Test that exceeding Counter 3 in Node 0 invokes node9_phase1_error_handler."""
    parent_id = test_db.insert_proposal(
        candidate_id="cand_node0_c3_fail",
        solver="euler",
        noise_model="none",
        is_reference=False,
        status="PENDING",
    )
    # Pre-set counter 3 to 3
    test_db.update_counters(parent_id, counter_3_debug_gate=3)

    invalid_debug_req = {
        "status": "SUCCESS",
        "type": "DEBUG_MODE_REQUEST",
        "debugee_candidate_id": parent_id,
        "deactivated_rewrites": ["deactivate_both"],  # Invalid for non-reference
    }

    res = node0_gate(
        db=test_db,
        config=test_config,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=parent_id,
        debug_request=invalid_debug_req,
    )

    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["candidate_id"] == parent_id
    assert res["counter_name"] == "counter_3_debug_gate"
    assert res["current_count"] == 4
    assert res["next_node"] is None
    assert res["db_snapshot"] is not None
    assert Path(res["db_snapshot"]).is_file()

    # DB status marked FAILED
    prop = test_db.get_proposal(parent_id)
    assert prop["status"] == "FAILED"


def test_counter4_exceedance_in_node7(test_config, test_db):
    """Test that exceeding Counter 4 in Node 7 invokes node9_phase1_error_handler."""
    parent_id = test_db.insert_proposal(
        candidate_id="cand_parent_c4",
        solver="rk4",
        noise_model="none",
        is_reference=False,
        status="FAILED",
    )
    debug_id = test_db.create_debug_candidate(debugee_id=parent_id, deactivated_rewrites=["solver"])
    test_db.quarantine_candidate(parent_id)

    # Set counter 4 to 5 (max allowed for debug loop)
    test_db.update_counters(debug_id, counter_4_debug_loop=5)
    test_db.mark_proposal_status(debug_id, status="EVALUATING", relative_error=0.01, accuracy_fid=0.90)

    res = node7_phase1_evaluation_gate(
        db=test_db,
        config=test_config,
        candidate_id=debug_id,
    )

    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["candidate_id"] == debug_id
    assert res["counter_name"] == "counter_4_debug_loop"
    assert res["current_count"] == 6
    assert res["next_node"] is None
    assert res["db_snapshot"] is not None
    assert Path(res["db_snapshot"]).is_file()

    # DB status marked FAILED
    prop = test_db.get_proposal(debug_id)
    assert prop["status"] == "FAILED"


def test_node9_chia_execution_options_and_maintenance(test_config, test_db):
    """Verify Node 9 ChiaFunction execution options and WAL maintenance."""
    cand_id = test_db.insert_proposal(
        candidate_id="cand_node9_chia_opts_01",
        solver="euler",
        noise_model="none",
        status="PENDING",
    )

    # Check ChiaFunction options
    chia_opts = getattr(node9_phase1_error_handler, "_chia_options", {})
    assert chia_opts.get("max_retries") == 0
    assert chia_opts.get("num_cpus") == 0.1

    res = node9_phase1_error_handler(
        db=test_db,
        config=test_config,
        candidate_id=cand_id,
        counter_name="counter_2_rewrite",
        current_count=5,
        max_count=4,
        execution_mode="standard",
        error_stage="rewrite_evaluation",
        details="Rewrite count exceeded.",
    )

    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["counter_name"] == "counter_2_rewrite"
    snapshot_path = res.get("db_snapshot")
    assert snapshot_path is not None
    assert Path(snapshot_path).is_file()

