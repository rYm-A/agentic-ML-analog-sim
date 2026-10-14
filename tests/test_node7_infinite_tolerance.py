"""Unit tests for Node 7 Evaluation Gate with positive infinite tolerances."""

import pytest
from pathlib import Path
from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node7_phase1_evaluation_gate import node7_phase1_evaluation_gate

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_db_and_config(tmp_path):
    db_file = tmp_path / "test_results.db"
    db = ResultDatabase(db_path=db_file)
    db.init_db()

    config = load_simulation_config(REPO_ROOT / "config" / "simulation_config.yaml")
    return db, config


def test_node7_finite_tolerances_fail_out_of_bounds(tmp_db_and_config):
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_finite_fail",
        solver="euler",
        noise_model="L0_static_mismatch",
        relative_error=0.99,
        absolute_error=0.50,
    )
    db.update_execution_status(cand_id, simulation_status="SUCCESS")

    res = node7_phase1_evaluation_gate(db=db, config=config, candidate_id=cand_id)
    assert res["status"] == "FAILED"
    assert res["reason"] == "out_of_tolerance"
    assert "relative_error (0.9900) > rtol (0.0500)" in res["error_details"]
    assert "absolute_error (0.5000) > atol (0.0500)" in res["error_details"]


def test_node7_infinite_tolerances_pass_any_error(tmp_db_and_config):
    db, config = tmp_db_and_config

    # Set both rtol and atol to infinity
    config.tolerances.solver_accuracy.rtol = float("inf")
    config.tolerances.solver_accuracy.atol = float("inf")

    cand_id = db.insert_proposal(
        candidate_id="cand_infinite_pass",
        solver="euler",
        noise_model="L2_functional_interface",
        relative_error=150.0,
        absolute_error=42.0,
    )
    db.update_execution_status(cand_id, simulation_status="SUCCESS")

    res = node7_phase1_evaluation_gate(db=db, config=config, candidate_id=cand_id)
    assert res["status"] == "PASSED"
    assert res["next_node"] == "phase2_node1"

    prop = db.get_proposal(cand_id)
    assert prop["status"] == "COMPLETED"


def test_node7_infinite_rtol_handles_overflow(tmp_db_and_config):
    db, config = tmp_db_and_config

    # Overflow from sensitivity lookup returns float("inf")
    config.tolerances.solver_accuracy.rtol = float("inf")
    config.tolerances.solver_accuracy.atol = float("inf")

    cand_id = db.insert_proposal(
        candidate_id="cand_overflow_pass",
        solver="euler",
        noise_model="L3_correlated_drift",
        relative_error=float("inf"),
        absolute_error=10.0,
    )
    db.update_execution_status(cand_id, simulation_status="SUCCESS")

    res = node7_phase1_evaluation_gate(db=db, config=config, candidate_id=cand_id)
    assert res["status"] == "PASSED"
    assert res["next_node"] == "phase2_node1"


def test_node7_partial_infinite_tolerances(tmp_db_and_config):
    db, config = tmp_db_and_config

    # Only rtol is infinite; atol is finite
    config.tolerances.solver_accuracy.rtol = float("inf")
    config.tolerances.solver_accuracy.atol = 0.05

    cand_id = db.insert_proposal(
        candidate_id="cand_partial_fail",
        solver="euler",
        noise_model="L1_stochastic_parameter_noise",
        relative_error=50.0,
        absolute_error=0.80,  # exceeds atol 0.05
    )
    db.update_execution_status(cand_id, simulation_status="SUCCESS")

    res = node7_phase1_evaluation_gate(db=db, config=config, candidate_id=cand_id)
    assert res["status"] == "FAILED"
    assert "absolute_error (0.8000) > atol (0.0500)" in res["error_details"]
    assert "relative_error" not in res["error_details"]  # rtol was infinite, so not flagged
