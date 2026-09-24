"""Unit tests for Phase 0 error handling, fallback synthesis, and programmatic safety net."""

import json
from pathlib import Path
import pytest
import yaml

from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import node2_phase0_propose_candidate
from agentic_ml_analog_sim.nodes.node3_phase0_validate_proposal import node3_phase0_validate_proposal
from agentic_ml_analog_sim.phase_0 import Phase0Result, run_phase_0


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide a fresh SQLite ResultDatabase."""
    db_file = tmp_path / "test_phase0.db"
    return ResultDatabase(db_path=str(db_file))


def test_node2_fallback_synthesizes_from_config_solvers(temp_db):
    """Test 1: Fallback synthesis uses config solvers when LLM produces invalid JSON or empty output."""
    config = {
        "solvers": {
            "rk4": {"steps": [10, 20]},
            "euler_backward": {"steps": [5, 10]},
        },
        "noise_models": ["L0_static_mismatch"],
        "sparsity_configs": {"dense": {"sparsity_ratio": 0.0}},
        "budget": {"max_node_retries": 3},
    }

    # Test with invalid JSON response
    mock_llm_invalid = MockAntigravityLLM(responses=["```json\n{this is not valid json\n```"])
    proposal = node2_phase0_propose_candidate(
        db=temp_db,
        config=config,
        llm=mock_llm_invalid,
    )

    assert proposal is not None
    assert proposal["solver"] in ["rk4", "euler_backward"]
    assert proposal["solver"] != "euler"

    # Test with empty string response
    mock_llm_empty = MockAntigravityLLM(responses=[""])
    proposal_empty = node2_phase0_propose_candidate(
        db=temp_db,
        config=config,
        llm=mock_llm_empty,
    )

    assert proposal_empty is not None
    assert proposal_empty["solver"] in ["rk4", "euler_backward"]
    assert proposal_empty["solver"] != "euler"


def test_node2_prevalidation_intercepts_unconfigured_solver(temp_db):
    """Test 2: Prevalidation intercepts unconfigured solver ('euler') and auto-corrects to valid solver."""
    config = {
        "solvers": {
            "rk4": {"steps": [10, 20]},
            "euler_backward": {"steps": [5, 10]},
        },
        "noise_models": ["L0_static_mismatch"],
        "sparsity_configs": {"dense": {"sparsity_ratio": 0.0}},
        "budget": {"max_node_retries": 3},
        "prevalidate": True,
    }

    # Mock LLM returns proposal with 'euler', which is not in config.solvers
    raw_proposal = {
        "candidate_id": "cand_euler_test_1",
        "solver": "euler",
        "num_steps": 5,
        "solver_params": {"num_steps": 5, "dt": 0.2},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"type": "dense", "sparsity_ratio": 0.0},
    }
    mock_llm = MockAntigravityLLM(responses=[json.dumps(raw_proposal)])

    proposal = node2_phase0_propose_candidate(
        db=temp_db,
        config=config,
        llm=mock_llm,
        prevalidate=True,
    )

    assert proposal is not None
    assert proposal["solver"] in ["rk4", "euler_backward"]
    assert proposal["solver"] != "euler"

    # Verify candidate stored in database has the auto-corrected solver
    cand_id = proposal["candidate_id"]
    db_record = temp_db.get_proposal(cand_id)
    assert db_record is not None
    assert db_record["solver"] in ["rk4", "euler_backward"]
    assert db_record["solver"] != "euler"


def test_node2_prompt_history_capping(temp_db):
    """Test 3: Prompt history table rows are capped to exactly 10 in user prompt passed to LLM."""
    # Insert 25 dummy proposals in DB
    for i in range(25):
        temp_db.insert_proposal(
            candidate_id=f"cand_hist_{i:02d}",
            solver="rk4",
            noise_model="L0_static_mismatch",
            solver_params={"num_steps": 10, "dt": 0.1},
            noise_params={"sigma": 0.01},
            sparsity_config={"type": "dense", "sparsity_ratio": 0.0},
            status="COMPLETED" if i % 2 == 0 else "REJECTED",
        )

    config = {
        "solvers": {"rk4": {"steps": [10, 20]}},
        "noise_models": ["L0_static_mismatch"],
        "sparsity_configs": {"dense": {"sparsity_ratio": 0.0}},
        "budget": {"max_node_retries": 3},
    }

    mock_llm = MockAntigravityLLM(mode="valid")
    node2_phase0_propose_candidate(
        db=temp_db,
        config=config,
        llm=mock_llm,
    )

    # Inspect the prompt text passed to the LLM
    last_prompt = mock_llm.last_prompt
    assert last_prompt is not None

    table_data_rows = [
        line for line in last_prompt.splitlines()
        if line.strip().startswith("|")
        and not line.strip().startswith("| Candidate ID")
        and not line.strip().startswith("|---")
    ]
    assert len(table_data_rows) == 10, f"Expected exactly 10 history rows, found {len(table_data_rows)}"


def test_node3_rich_rejection_message_contents(temp_db):
    """Test 4: Node 3 rich rejection message for unconfigured solver specifies allowed solvers."""
    cand_id = "cand_test_euler_unconfigured"
    temp_db.insert_proposal(
        candidate_id=cand_id,
        solver="euler",
        noise_model="L0_static_mismatch",
        solver_params={"num_steps": 5, "dt": 0.2},
        noise_params={"sigma": 0.01},
        sparsity_config={"type": "dense", "sparsity_ratio": 0.0},
        status="PENDING",
    )

    config = {
        "solvers": ["rk4", "euler_backward"],
        "noise_models": ["L0_static_mismatch"],
        "budget": {"max_node_retries": 3},
    }

    is_valid, reason, cycle_count = node3_phase0_validate_proposal(
        candidate_id=cand_id,
        db=temp_db,
        config=config,
        cycle_count=1,
    )

    assert not is_valid
    assert "INVALID_SOLVER: Solver 'euler' is not allowed" in reason
    assert "['euler_backward', 'rk4']" in reason


def test_phase0_loop_safety_net_prevents_premature_exit(tmp_path):
    """Test 5: Safety net intercepts on final retry, synthesizes valid candidate, and avoids premature exit."""
    db_file = tmp_path / "test_safety_net.db"
    db = ResultDatabase(db_path=str(db_file))

    # Populate baseline reference design as COMPLETED so Node 1 passes
    db.insert_proposal(
        candidate_id="cand_golden_ref",
        solver="rk4",
        noise_model="none",
        solver_params={"num_steps": 20, "dt": 0.05},
        noise_params={},
        sparsity_config={"sparsity_ratio": 0.0},
        status="COMPLETED",
        is_reference=True,
        accuracy_fid=10.0,
        latency_ms=100.0,
    )

    # Create config YAML with max_node_retries = 3 and specific valid solvers
    base_config_path = Path(__file__).resolve().parent.parent / "config" / "simulation_config.yaml"
    with open(base_config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict["solvers"] = {
        "rk4": {"steps": [10, 20]},
        "euler_backward": {"steps": [5, 10]},
    }
    config_dict["budget"]["max_node_retries"] = 3
    config_dict["prevalidate"] = False

    cfg_file = tmp_path / "sim_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)

    # Mock LLM that always proposes an invalid solver (mode="invalid_solver" proposes "unsupported_leapfrog")
    mock_llm = MockAntigravityLLM(mode="invalid_solver")

    result: Phase0Result = run_phase_0(
        config_path=cfg_file,
        db_path=str(db_file),
        llm=mock_llm,
        enable_safety_net=True,
    )

    # Verify that instead of TERMINATED_MAX_CYCLES, Phase 0 proceeded via safety net
    assert result.action == "PROCEED_TO_PHASE_1_CANDIDATE"
    assert result.candidate_id is not None
    assert result.cycle_count == 3
    assert "Candidate generated via Phase 0 programmatic safety net" in result.details

    # Verify the synthesized candidate in DB has valid solver and steps
    candidate_record = db.get_proposal(result.candidate_id)
    assert candidate_record is not None
    assert candidate_record["solver"] in ["rk4", "euler_backward"]
    assert candidate_record["solver"] != "unsupported_leapfrog"
