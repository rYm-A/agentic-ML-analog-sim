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
    cfg = load_simulation_config("config/simulation_config.yaml")
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
