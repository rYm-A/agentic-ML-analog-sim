"""End-to-end integration tests for the Phase 0 pipeline orchestrator."""

import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

pytestmark = [pytest.mark.node1_phase0, pytest.mark.node2_phase0]

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db import ResultDatabase
from agentic_ml_analog_sim.mock import MockAntigravityLLM
from agentic_ml_analog_sim.phase_0 import Phase0Result, run_phase_0

CONFIG_PATH = Path("config/simulation_config.yaml")


@pytest.fixture
def temp_env():
    """Create a temporary directory with a database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "phase0_test.db"
        yield db_path


def test_case_a_fresh_db_yields_proceed_to_phase_1_reference(temp_env):
    """Case A: Fresh DB -> yields PROCEED_TO_PHASE_1_REFERENCE."""
    db_path = temp_env

    result: Phase0Result = run_phase_0(
        config_path=CONFIG_PATH,
        db_path=db_path,
        llm=None,
    )

    assert result.action == "PROCEED_TO_PHASE_1_REFERENCE"
    assert result.candidate_id is not None
    assert result.candidate_id.startswith("ref_")
    assert result.phase_1_instructions is not None
    assert "Phase 1 Baseline Reference Design Evaluation" in result.phase_1_instructions
    assert result.cycle_count == 0
    assert "missing or incomplete" in result.details.lower()

    # Verify that a pending reference proposal was recorded in the database
    db = ResultDatabase(db_path)
    ref_prop = db.get_reference_proposal()
    assert ref_prop is not None
    assert ref_prop["candidate_id"] == result.candidate_id
    assert ref_prop["status"] == "PENDING"
    assert ref_prop["is_reference"] in (1, True)


def test_case_b_populated_reference_yields_proceed_to_phase_1_candidate(temp_env):
    """Case B: Populated reference DB -> yields PROCEED_TO_PHASE_1_CANDIDATE with valid instructions."""
    db_path = temp_env
    db = ResultDatabase(db_path)
    cfg = load_simulation_config(CONFIG_PATH)

    # Populate baseline reference as COMPLETED
    ref_id = db.insert_proposal(
        solver=cfg.reference_design.solver,
        noise_model=cfg.reference_design.noise_model,
        solver_params={"num_steps": cfg.reference_design.num_steps, "integration_time": cfg.reference_design.integration_time},
        noise_params={},
        sparsity_config={"sparsity_ratio": cfg.reference_design.sparsity_ratio},
        is_reference=True,
        status="COMPLETED",
        accuracy_fid=10.0,
        latency_ms=100.0,
    )

    mock_llm = MockAntigravityLLM(mode="valid")

    result: Phase0Result = run_phase_0(
        config_path=CONFIG_PATH,
        db_path=db_path,
        llm=mock_llm,
    )

    assert result.action == "PROCEED_TO_PHASE_1_CANDIDATE"
    assert result.candidate_id is not None
    assert result.candidate_id != ref_id
    assert result.phase_1_instructions is not None
    assert "Phase 1" in result.phase_1_instructions
    assert result.cycle_count == 1
    assert result.details == "Candidate approved for Phase 1"

    # Verify candidate recorded in database
    cand = db.get_proposal(result.candidate_id)
    assert cand is not None
    assert cand["status"] == "PENDING"
    assert cand["solver"] == "euler"

    # Verify token usage recorded
    summary = db.get_token_usage_summary(phase="phase_0")
    assert summary["overall"]["total_calls"] >= 1
    assert summary["overall"]["total_tokens"] > 0
    assert "phase_0.node2_phase0_propose_candidate" in summary["by_node"]


def test_case_c_cyclic_duplicates_yields_terminated_max_cycles(temp_env):
    """Case C: Cyclic duplicates from Mock LLM -> yields TERMINATED_MAX_CYCLES."""
    db_path = temp_env
    db = ResultDatabase(db_path)
    cfg = load_simulation_config(CONFIG_PATH)

    # Populate reference design
    db.insert_proposal(
        solver=cfg.reference_design.solver,
        noise_model=cfg.reference_design.noise_model,
        solver_params={"num_steps": cfg.reference_design.num_steps, "dt": 0.04},
        noise_params={},
        sparsity_config={"sparsity_ratio": cfg.reference_design.sparsity_ratio},
        is_reference=True,
        status="COMPLETED",
        accuracy_fid=10.0,
        latency_ms=100.0,
    )

    # Populate an existing candidate proposal to be duplicated
    dup_prop = {
        "candidate_id": "cand_dup_l0",
        "solver": "rk4",
        "num_steps": 25,
        "solver_params": {"num_steps": 25, "dt": 0.04},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"sparsity_ratio": 0.0},
    }
    db.insert_proposal(
        solver=dup_prop["solver"],
        noise_model=dup_prop["noise_model"],
        solver_params=dup_prop["solver_params"],
        noise_params=dup_prop["noise_params"],
        sparsity_config=dup_prop["sparsity_config"],
        status="COMPLETED",
    )

    # Mock LLM returns exact duplicate of existing candidate on every prompt
    mock_llm = MockAntigravityLLM(mode="duplicate", duplicate_proposal=dup_prop)

    result: Phase0Result = run_phase_0(
        config_path=CONFIG_PATH,
        db_path=db_path,
        llm=mock_llm,
        enable_safety_net=False,
    )

    assert result.action == "TERMINATED_MAX_CYCLES"
    assert result.phase_1_instructions is None
    assert result.cycle_count == cfg.budget.max_node_retries
    assert "exceeded maximum proposal retry cycles" in result.details.lower()


def test_case_d_missing_config_returns_error(temp_env):
    """Case D: Missing or invalid configuration file path returns ERROR result."""
    db_path = temp_env
    result: Phase0Result = run_phase_0(
        config_path="config/non_existent_config.yaml",
        db_path=db_path,
    )
    assert result.action == "ERROR"
    assert result.candidate_id is None
    assert "not found" in result.details.lower()


def test_cli_dry_run_execution(temp_env):
    """Test CLI entrypoint execution via python -m agentic_ml_analog_sim.phase_0."""
    db_path = temp_env
    cmd = [
        sys.executable,
        "-m",
        "agentic_ml_analog_sim.phase_0",
        "--config",
        str(CONFIG_PATH),
        "--db",
        str(db_path),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "PROCEED_TO_PHASE_1_REFERENCE" in proc.stdout
