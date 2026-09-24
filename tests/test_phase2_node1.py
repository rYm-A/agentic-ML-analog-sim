"""Unit tests for Phase 2 Node 1: phase2_check_experiment_limit."""

from pathlib import Path
import pytest

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node1_phase2_check_experiment_limit import phase2_check_experiment_limit


@pytest.fixture
def test_db(tmp_path: Path) -> ResultDatabase:
    db_file = tmp_path / "node1_test.db"
    return ResultDatabase(db_path=db_file)


def test_experiment_limit_continue_loop(test_db: ResultDatabase):
    cfg = load_simulation_config("config/simulation_config.yaml")
    # By default, max_experiments = 10, current experiments = 0
    res = phase2_check_experiment_limit(test_db, cfg)

    assert res["limit_reached"] is False
    assert res["action"] == "CONTINUE_LOOP"
    assert res["completed_experiments"] == 0
    assert res["max_experiments"] == 10


def test_experiment_limit_reached(test_db: ResultDatabase):
    cfg = load_simulation_config("config/simulation_config.yaml")

    # Insert 10 completed candidate proposals
    for i in range(10):
        cid = test_db.insert_proposal(
            solver="euler",
            noise_model="none",
            is_reference=False,
            solver_params={"dt": 0.01 * (i + 1)},
        )
        test_db.mark_proposal_status(cid, "COMPLETED", latency_ms=10.0 + i, relative_error=0.01)

    res = phase2_check_experiment_limit(test_db, cfg)
    assert res["limit_reached"] is True
    assert res["action"] == "TERMINATE_AND_REPORT"
    assert res["completed_experiments"] == 10
    assert "Experiment limit reached" in res["reason"]


def test_experiment_limit_with_failures(test_db: ResultDatabase):
    cfg = load_simulation_config("config/simulation_config.yaml")

    # Insert 7 completed and 3 failed proposals = 10 evaluated
    for i in range(7):
        cid = test_db.insert_proposal(solver="euler", noise_model="none", is_reference=False, solver_params={"i": i})
        test_db.mark_proposal_status(cid, "COMPLETED", latency_ms=10.0)

    for i in range(3):
        cid = test_db.insert_proposal(solver="rk4", noise_model="none", is_reference=False, solver_params={"j": i})
        test_db.mark_proposal_status(cid, "FAILED", error_stage="simulation")

    res = phase2_check_experiment_limit(test_db, cfg)
    assert res["limit_reached"] is True
    assert res["action"] == "TERMINATE_AND_REPORT"
    assert res["completed_experiments"] == 10


def test_iteration_limit_trigger(test_db: ResultDatabase):
    cfg = load_simulation_config("config/simulation_config.yaml")

    # 3 completed experiments, but iteration >= max_iterations
    for i in range(3):
        cid = test_db.insert_proposal(solver="euler", noise_model="none", is_reference=False, solver_params={"k": i})
        test_db.mark_proposal_status(cid, "COMPLETED")

    res = phase2_check_experiment_limit(test_db, cfg, current_iteration=15)
    assert res["limit_reached"] is True
    assert res["action"] == "TERMINATE_AND_REPORT"
    assert "Iteration limit reached" in res["reason"]


def test_with_custom_dict_config(test_db: ResultDatabase):
    custom_cfg = {
        "experiment_limits": {
            "max_experiments": 3,
            "max_iterations": 5,
        }
    }
    # 0 completed
    res = phase2_check_experiment_limit(test_db, custom_cfg)
    assert res["limit_reached"] is False
    assert res["action"] == "CONTINUE_LOOP"
    assert res["max_experiments"] == 3

    # Add 3
    for i in range(3):
        cid = test_db.insert_proposal(solver="euler", noise_model="none", is_reference=False, solver_params={"l": i})
        test_db.mark_proposal_status(cid, "COMPLETED")

    res2 = phase2_check_experiment_limit(test_db, custom_cfg)
    assert res2["limit_reached"] is True
    assert res2["action"] == "TERMINATE_AND_REPORT"
