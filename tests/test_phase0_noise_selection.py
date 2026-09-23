"""Tests for programmatic noise model selection, per-noise quotas, and budgeting in Phase 0 & Phase 2."""

from __future__ import annotations

import pytest
from pathlib import Path
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.phase_0 import run_phase_0, Phase0Result
from agentic_ml_analog_sim.nodes.node3_phase0_validate_proposal import node3_phase0_validate_proposal
from agentic_ml_analog_sim.nodes.node1_phase2_check_experiment_limit import node1_phase2_check_experiment_limit
from agentic_ml_analog_sim.mock import MockAntigravityLLM


@pytest.fixture
def temp_db(tmp_path: Path) -> ResultDatabase:
    """Create an isolated test database with tables initialized."""
    db_file = tmp_path / "test_noise_selection.db"
    return ResultDatabase(db_path=db_file)


def test_db_noise_model_queries(temp_db: ResultDatabase):
    """Test get_completed_experiments_count, get_proposal_counts_by_noise_model, and get_next_unfulfilled_noise_model."""
    # Insert golden reference (is_reference=True)
    temp_db.insert_proposal(
        candidate_id="cand_golden_ref",
        solver="euler",
        solver_params={"num_steps": 20, "dt": 0.05},
        noise_model="none",
        noise_params={},
        sparsity_config={"type": "dense"},
        status="COMPLETED",
        is_reference=True,
    )

    # Insert candidate proposals for L0 and L1
    temp_db.insert_proposal(
        candidate_id="cand_l0_1",
        solver="euler",
        solver_params={"num_steps": 5, "dt": 0.2},
        noise_model="L0_static_mismatch",
        noise_params={"sigma": 0.01},
        sparsity_config={"type": "dense"},
        status="COMPLETED",
        is_reference=False,
    )
    temp_db.insert_proposal(
        candidate_id="cand_l0_2",
        solver="euler",
        solver_params={"num_steps": 10, "dt": 0.1},
        noise_model="L0_static_mismatch",
        noise_params={"sigma": 0.01},
        sparsity_config={"type": "dense"},
        status="FAILED",
        is_reference=False,
    )
    temp_db.insert_proposal(
        candidate_id="cand_l1_1",
        solver="par_ode",
        solver_params={"num_steps": 5, "dt": 0.2},
        noise_model="L1_thermal_drift",
        noise_params={"alpha": 0.005},
        sparsity_config={"type": "dense"},
        status="COMPLETED",
        is_reference=False,
    )
    temp_db.insert_proposal(
        candidate_id="cand_l1_2",
        solver="par_ode",
        solver_params={"num_steps": 10, "dt": 0.1},
        noise_model="L1_thermal_drift",
        noise_params={"alpha": 0.005},
        sparsity_config={"type": "dense"},
        status="PENDING",
        is_reference=False,
    )

    # Test get_completed_experiments_count
    assert temp_db.get_completed_experiments_count() == 3  # cand_l0_1, cand_l0_2, cand_l1_1
    assert temp_db.get_completed_experiments_count("L0_static_mismatch") == 2
    assert temp_db.get_completed_experiments_count("L1_thermal_drift") == 1
    assert temp_db.get_completed_experiments_count("L2_1f_flicker") == 0

    # Test get_proposal_counts_by_noise_model
    counts = temp_db.get_proposal_counts_by_noise_model(exclude_reference=True)
    assert "none" not in counts  # golden reference excluded
    assert counts["L0_static_mismatch"]["completed"] == 1
    assert counts["L0_static_mismatch"]["failed"] == 1
    assert counts["L0_static_mismatch"]["total"] == 2
    assert counts["L1_thermal_drift"]["completed"] == 1
    assert counts["L1_thermal_drift"]["pending"] == 1
    assert counts["L1_thermal_drift"]["total"] == 2

    # Test get_next_unfulfilled_noise_model with max_per_noise=2
    # L0 has 2 attempts (1 completed + 1 failed), so its quota of 2 is full.
    # L1 has 1 attempt (1 completed), so its quota of 2 is NOT full.
    next_nm = temp_db.get_next_unfulfilled_noise_model(
        noise_models=["L0_static_mismatch", "L1_thermal_drift"],
        max_per_noise=2,
    )
    assert next_nm == "L1_thermal_drift"

    # If L1 also gets another completed experiment:
    temp_db.update_proposal_status("cand_l1_2", "COMPLETED")
    next_nm_full = temp_db.get_next_unfulfilled_noise_model(
        noise_models=["L0_static_mismatch", "L1_thermal_drift"],
        max_per_noise=2,
    )
    assert next_nm_full is None


def test_node3_rejection_of_mismatched_target_noise(temp_db: ResultDatabase):
    """Test that Node 3 rejects a proposal if it doesn't match target_noise_model."""
    temp_db.insert_proposal(
        candidate_id="cand_wrong_noise",
        solver="euler",
        solver_params={"num_steps": 5, "dt": 0.2},
        noise_model="L0_static_mismatch",
        noise_params={"sigma": 0.01},
        sparsity_config={"type": "dense"},
        status="PENDING",
        is_reference=False,
    )

    mock_config = {
        "solvers": {"euler": {"steps": [5, 10, 20]}},
        "noise_models": ["L0_static_mismatch", "L1_thermal_drift"],
        "sparsity_configs": {"dense": {"sparsity_ratio": 0.0}},
        "budget": {"max_node_retries": 3},
    }

    # When target_noise_model is L1_thermal_drift, candidate with L0_static_mismatch must be rejected
    is_valid, reason, cycle = node3_phase0_validate_proposal(
        candidate_id="cand_wrong_noise",
        db=temp_db,
        config=mock_config,
        cycle_count=1,
        target_noise_model="L1_thermal_drift",
    )
    assert not is_valid
    assert reason == "REJECTED_WRONG_NOISE_MODEL"

    # When target_noise_model is L0_static_mismatch, proposal is approved
    is_valid_ok, reason_ok, cycle_ok = node3_phase0_validate_proposal(
        candidate_id="cand_wrong_noise",
        db=temp_db,
        config=mock_config,
        cycle_count=1,
        target_noise_model="L0_static_mismatch",
    )
    assert is_valid_ok
    assert reason_ok == "VALID"


def test_phase0_programmatic_noise_selection_and_termination(tmp_path: Path):
    """Test Phase 0 deterministic sequencing across noise models and termination when budget per noise is reached."""
    # Write a test YAML config with 2 noise models and max_experiments = 2
    config_content = """
experiment_name: "test_noise_budget_campaign"

reference_design:
  name: "cifar10/n1024"
  family: "cifar10"
  n_oscillators: 1024
  n_conditional_oscillators: 8
  solver: "rk4"
  num_steps: 25
  integration_time: 1.0
  precision: "fp32"
  noise_model: "none"
  sparsity_ratio: 0.0

solvers:
  euler:
    steps: [1, 2, 5, 10]
  rk4:
    steps: [5, 10, 15, 25]

noise_models:
  - "L0_static_mismatch"
  - "L1_thermal_drift"

sparsity_configs:
  dense:
    sparsity_ratio: 0.0
  threshold_pruned:
    allowed_sparsity_ratios: [0.2, 0.4, 0.6, 0.8]
  degree_bounded:
    max_degree: [64, 128]

tolerances:
  compile_check:
    rtol: 1e-4
    atol: 1e-4
  solver_accuracy:
    rtol: 5e-2
    atol: 5e-2
    max_fid_degradation: 2.0

budget:
  max_experiments: 2
  max_iterations: 15
  max_node_retries: 3
  timeout_seconds_per_eval: 300

counter_limits:
  max_counter_1_compilation: 3
  max_counter_2_rewrite: 3
  max_counter_3_debug_gate: 3
  max_counter_4_debug_loop: 5
"""
    config_path = tmp_path / "test_sim_config.yaml"
    config_path.write_text(config_content)
    db_path = tmp_path / "test_campaign.db"
    db = ResultDatabase(db_path=db_path)

    mock_llm = MockAntigravityLLM(mode="valid")

    # 1. First invocation: Reference is missing -> PROCEED_TO_PHASE_1_REFERENCE
    res_ref = run_phase_0(config_path=config_path, db_path=db_path, iteration=1, llm=mock_llm)
    assert res_ref.action == "PROCEED_TO_PHASE_1_REFERENCE"

    # Complete the reference in DB
    db.insert_proposal(
        candidate_id="cand_golden_ref",
        solver="rk4",
        solver_params={"num_steps": 25, "integration_time": 1.0},
        noise_model="none",
        noise_params={},
        sparsity_config={"sparsity_ratio": 0.0},
        status="COMPLETED",
        is_reference=True,
        accuracy_fid=10.0,
        latency_ms=100.0,
    )

    # 2. Second invocation: Reference exists, neither noise model has attempts.
    # Should propose for the first noise model: L0_static_mismatch.
    res_cand1 = run_phase_0(config_path=config_path, db_path=db_path, iteration=2, llm=mock_llm)
    assert res_cand1.action == "PROCEED_TO_PHASE_1_CANDIDATE"
    p1 = db.get_proposal(res_cand1.candidate_id)
    assert p1["noise_model"] == "L0_static_mismatch"

    # Simulate evaluation complete (1st L0 attempt completed)
    db.update_proposal_status(res_cand1.candidate_id, "COMPLETED")

    # 3. Third invocation: L0 has 1 attempt (< 2). Should propose 2nd candidate for L0.
    res_cand2 = run_phase_0(config_path=config_path, db_path=db_path, iteration=3, llm=mock_llm)
    assert res_cand2.action == "PROCEED_TO_PHASE_1_CANDIDATE"
    p2 = db.get_proposal(res_cand2.candidate_id)
    assert p2["noise_model"] == "L0_static_mismatch"

    # Simulate evaluation failure for this candidate (counts as executed attempt)
    db.update_proposal_status(res_cand2.candidate_id, "FAILED")

    # 4. Fourth invocation: L0 has 2 attempts (1 COMPLETED, 1 FAILED -> quota 2 fulfilled).
    # Next unfulfilled is L1_thermal_drift. Phase 0 must programmatically select L1!
    res_cand3 = run_phase_0(config_path=config_path, db_path=db_path, iteration=4, llm=mock_llm)
    assert res_cand3.action == "PROCEED_TO_PHASE_1_CANDIDATE"
    p3 = db.get_proposal(res_cand3.candidate_id)
    assert p3["noise_model"] == "L1_thermal_drift"

    # Complete 1st L1 candidate
    db.update_proposal_status(res_cand3.candidate_id, "COMPLETED")

    # 5. Fifth invocation: L1 has 1 attempt (< 2). Should propose 2nd candidate for L1.
    res_cand4 = run_phase_0(config_path=config_path, db_path=db_path, iteration=5, llm=mock_llm)
    assert res_cand4.action == "PROCEED_TO_PHASE_1_CANDIDATE"
    p4 = db.get_proposal(res_cand4.candidate_id)
    assert p4["noise_model"] == "L1_thermal_drift"

    # Complete 2nd L1 candidate
    db.update_proposal_status(res_cand4.candidate_id, "COMPLETED")

    # 6. Sixth invocation: BOTH L0 and L1 have completed their quota of 2 experiments.
    # Phase 0 must terminate immediately with TERMINATED_MAX_CYCLES.
    res_term = run_phase_0(config_path=config_path, db_path=db_path, iteration=6, llm=mock_llm)
    assert res_term.action == "TERMINATED_MAX_CYCLES"
    assert "All configured noise models have completed their experiment budget." in res_term.details


def test_node1_phase2_multi_noise_quota_evaluation(temp_db: ResultDatabase):
    """Test node1_phase2_check_experiment_limit per-noise quota evaluation."""
    config = {
        "budget": {"max_experiments": 2, "max_iterations": 20},
        "noise_models": ["L0_static_mismatch", "L1_thermal_drift"],
    }

    # Add 2 completed proposals for L0
    temp_db.insert_proposal(
        candidate_id="cand_l0_a",
        solver="euler",
        solver_params={"num_steps": 5, "dt": 0.2},
        noise_model="L0_static_mismatch",
        noise_params={"sigma": 0.01},
        sparsity_config={"type": "dense"},
        status="COMPLETED",
        is_reference=False,
    )
    temp_db.insert_proposal(
        candidate_id="cand_l0_b",
        solver="euler",
        solver_params={"num_steps": 10, "dt": 0.1},
        noise_model="L0_static_mismatch",
        noise_params={"sigma": 0.01},
        sparsity_config={"type": "dense"},
        status="FAILED",
        is_reference=False,
    )

    # Only L0 has reached quota (2 attempts), L1 has 0 attempts.
    res1 = node1_phase2_check_experiment_limit(
        db=temp_db,
        config=config,
        current_iteration=3,
        candidate_id="cand_l0_b",
    )
    assert not res1["limit_reached"]
    assert res1["action"] == "CONTINUE_LOOP"
    assert res1["per_noise_stats"]["L0_static_mismatch"]["quota_reached"] is True
    assert res1["per_noise_stats"]["L1_thermal_drift"]["quota_reached"] is False

    # Now add 2 attempts for L1
    temp_db.insert_proposal(
        candidate_id="cand_l1_a",
        solver="par_ode",
        solver_params={"num_steps": 5, "dt": 0.2},
        noise_model="L1_thermal_drift",
        noise_params={"alpha": 0.005},
        sparsity_config={"type": "dense"},
        status="COMPLETED",
        is_reference=False,
    )
    temp_db.insert_proposal(
        candidate_id="cand_l1_b",
        solver="par_ode",
        solver_params={"num_steps": 10, "dt": 0.1},
        noise_model="L1_thermal_drift",
        noise_params={"alpha": 0.005},
        sparsity_config={"type": "dense"},
        status="COMPLETED",
        is_reference=False,
    )

    res2 = node1_phase2_check_experiment_limit(
        db=temp_db,
        config=config,
        current_iteration=5,
        candidate_id="cand_l1_b",
    )
    assert res2["limit_reached"] is True
    assert res2["action"] == "TERMINATE_AND_REPORT"
    assert "Experiment limit reached" in res2["reason"]
    assert res2["per_noise_stats"]["L0_static_mismatch"]["quota_reached"] is True
    assert res2["per_noise_stats"]["L1_thermal_drift"]["quota_reached"] is True
