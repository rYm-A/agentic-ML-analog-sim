"""Tests for ResultDatabase SQLite CRUD operations, duplicate detection, and Pareto optimization."""

import tempfile
from pathlib import Path
import pytest

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db import ResultDatabase


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = ResultDatabase(db_path)
        yield db


def test_init_db_and_reference_lifecycle(temp_db):
    config_path = Path("config/simulation_config.yaml")
    if not config_path.is_file():
        config_path = Path(__file__).parent.parent / "config" / "simulation_config.yaml"
    cfg = load_simulation_config(config_path)
    assert not temp_db.has_complete_reference(cfg)

    # Insert pending reference
    ref_id = temp_db.insert_proposal(
        solver=cfg.reference_design.solver,
        noise_model=cfg.reference_design.noise_model,
        solver_params={"num_steps": cfg.reference_design.num_steps},
        noise_params={},
        sparsity_config={"sparsity_ratio": cfg.reference_design.sparsity_ratio},
        is_reference=True,
        status="PENDING",
    )
    assert not temp_db.has_complete_reference(cfg)

    # Mark completed
    temp_db.mark_proposal_status(
        ref_id, "COMPLETED", accuracy_fid=10.0, latency_ms=100.0, wall_clock_s=2.0
    )
    assert temp_db.has_complete_reference(cfg)

    ref = temp_db.get_reference_proposal()
    assert ref is not None
    assert ref["candidate_id"] == ref_id
    assert ref["accuracy_fid"] == 10.0
    assert ref["latency_ms"] == 100.0


def test_crud_and_status_filtering(temp_db):
    """Test CRUD operations: get_proposal, get_latest_proposal, get_proposals with status filter."""
    p1 = temp_db.insert_proposal(
        solver="euler",
        noise_model="none",
        solver_params={"num_steps": 5},
        status="PENDING",
    )
    p2 = temp_db.insert_proposal(
        solver="rk4",
        noise_model="L0_static_mismatch",
        solver_params={"num_steps": 10},
        status="COMPLETED",
        accuracy_fid=12.5,
        latency_ms=80.0,
    )

    # get_proposal
    rec1 = temp_db.get_proposal(p1)
    assert rec1 is not None
    assert rec1["solver"] == "euler"
    assert rec1["status"] == "PENDING"

    # get_latest_proposal
    latest = temp_db.get_latest_proposal()
    assert latest is not None
    assert latest["candidate_id"] == p2

    # get_proposals with status filtering
    pending = temp_db.get_proposals(status="PENDING")
    assert len(pending) == 1
    assert pending[0]["candidate_id"] == p1

    completed = temp_db.get_proposals(status="COMPLETED")
    assert len(completed) == 1
    assert completed[0]["candidate_id"] == p2


def test_duplicate_proposal_detection(temp_db):
    temp_db.insert_proposal(
        solver="rk4",
        noise_model="none",
        solver_params={"steps": 25},
        noise_params={},
        sparsity_config={"sparsity_ratio": 0.0},
    )

    # Duplicate with same params
    assert temp_db.is_duplicate_proposal(
        solver="rk4",
        noise_model="none",
        solver_params={"steps": 25},
        noise_params={},
        sparsity_config={"sparsity_ratio": 0.0},
    )

    # Non-duplicate with different steps
    assert not temp_db.is_duplicate_proposal(
        solver="rk4",
        noise_model="none",
        solver_params={"steps": 15},
        noise_params={},
        sparsity_config={"sparsity_ratio": 0.0},
    )


def test_pareto_front_computation(temp_db):
    # Candidate 1: baseline
    c1 = temp_db.insert_proposal(solver="rk4", noise_model="none", status="COMPLETED")
    temp_db.mark_proposal_status(c1, "COMPLETED", accuracy_fid=12.0, latency_ms=100.0)

    # Candidate 2: faster with slightly worse FID (Pareto frontier)
    c2 = temp_db.insert_proposal(solver="euler", noise_model="none", status="COMPLETED")
    temp_db.mark_proposal_status(c2, "COMPLETED", accuracy_fid=15.0, latency_ms=40.0)

    # Candidate 3: dominated by c2 (higher error and higher latency)
    c3 = temp_db.insert_proposal(solver="euler_backward", noise_model="none", status="COMPLETED")
    temp_db.mark_proposal_status(c3, "COMPLETED", accuracy_fid=18.0, latency_ms=50.0)

    front = temp_db.compute_pareto_front()
    front_ids = {p["candidate_id"] for p in front}

    assert c1 in front_ids
    assert c2 in front_ids
    assert c3 not in front_ids


def test_comments(temp_db):
    cand_id = temp_db.insert_proposal(solver="euler", noise_model="none")
    comment_id = temp_db.add_comment(
        candidate_id=cand_id,
        phase="Phase_1",
        agent_name="Analyst",
        comment="Promising candidate.",
    )
    assert comment_id > 0

    comments = temp_db.get_comments(cand_id)
    assert len(comments) == 1
    assert comments[0]["agent_name"] == "Analyst"
    assert comments[0]["comment"] == "Promising candidate."


def test_create_debug_candidate(temp_db):
    debugee_id = temp_db.insert_proposal(
        solver="rk4",
        noise_model="L1_thermal",
        solver_params={"num_steps": 20},
        status="FAILED",
    )

    debug_id = temp_db.create_debug_candidate(
        debugee_id=debugee_id,
        deactivated_rewrites=["rewrite_solver_rk4"],
        solver="euler",
    )

    debug_cand = temp_db.get_proposal(debug_id)
    assert debug_cand is not None
    assert debug_cand["is_debug"] == 1
    assert debug_cand["debug_parent_id"] == debugee_id
    assert debug_cand["solver"] == "euler"
    assert debug_cand["noise_model"] == "L1_thermal"
    assert debug_cand["status"] == "PENDING"
    assert "rewrite_solver_rk4" in debug_cand["selection_reason"]

    active_debug = temp_db.get_active_debug_candidate()
    assert active_debug is not None
    assert active_debug["candidate_id"] == debug_id


def test_quarantine_and_reactivation(temp_db):
    cand_id = temp_db.insert_proposal(solver="rk4", noise_model="none", status="FAILED")

    quarantined = temp_db.quarantine_candidate(cand_id)
    assert quarantined["status"] == "QUARANTINED"

    fetched = temp_db.get_quarantined_candidate(cand_id)
    assert fetched is not None
    assert fetched["candidate_id"] == cand_id

    latest_q = temp_db.get_quarantined_candidate()
    assert latest_q is not None
    assert latest_q["candidate_id"] == cand_id

    reactivated = temp_db.reactivate_candidate(cand_id, new_status="PENDING")
    assert reactivated["status"] == "PENDING"
    assert temp_db.get_quarantined_candidate(cand_id) is None


def test_update_counters(temp_db):
    cand_id = temp_db.insert_proposal(solver="rk4", noise_model="none")

    updated = temp_db.update_counters(
        candidate_id=cand_id,
        counter_1_compilation=1,
        counter_2_rewrite=2,
        counter_3_debug_gate=3,
        counter_4_debug_loop=4,
    )

    assert updated["counter_1_compilation"] == 1
    assert updated["counter_2_rewrite"] == 2
    assert updated["counter_3_debug_gate"] == 3
    assert updated["counter_4_debug_loop"] == 4


def test_reset_counters(temp_db):
    cand_id = temp_db.insert_proposal(solver="rk4", noise_model="none")

    temp_db.update_counters(
        candidate_id=cand_id,
        counter_1_compilation=5,
        counter_2_rewrite=4,
        counter_3_debug_gate=3,
        counter_4_debug_loop=2,
    )

    reset_cand = temp_db.reset_counters(cand_id)
    assert reset_cand["counter_1_compilation"] == 0
    assert reset_cand["counter_2_rewrite"] == 0
    assert reset_cand["counter_3_debug_gate"] == 0
    assert reset_cand["counter_4_debug_loop"] == 0

    # Also check reactivation resets counter 4
    temp_db.quarantine_candidate(cand_id)
    temp_db.update_counters(cand_id, counter_4_debug_loop=4)
    reactivated = temp_db.reactivate_candidate(cand_id, new_status="PENDING")
    assert reactivated["counter_4_debug_loop"] == 0


