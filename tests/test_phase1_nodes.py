"""Comprehensive unit tests for Component 3 of Phase 1: Chia Nodes 0 through 8."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock import MockAntigravityLLM
from agentic_ml_analog_sim.nodes import (
    node0_gate,
    node0_phase1_gate,
    node1_phase1_solver_rewriter,
    node1_solver_rewriter,
    node2_phase1_noise_rewriter,
    node2_noise_rewriter,
    node3_phase1_compiler_rewriter,
    node3_compiler_rewriter,
    node4_phase1_compilation_runner,
    node4_compilation_runner,
    node5_phase1_verify_correctness,
    node5_verify_correctness,
    node6_phase1_simulate_profile,
    node6_simulate_profile,
    node7_phase1_evaluation_gate,
    node7_evaluation_gate,
    node8_phase1_diagnostic_proposal,
    node8_diagnostic_proposal,
)


@pytest.fixture
def tmp_db_and_config():
    """Create a temporary ResultDatabase and SimulationConfig instance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_phase1.db"
        db = ResultDatabase(db_path=db_path)

        # Basic simulation config dict
        config_data = {
            "experiment_name": "test_phase1_experiment",
            "reference_design": {
                "name": "cifar10/n1024",
                "family": "cifar10",
                "n_oscillators": 1024,
                "n_conditional_oscillators": 8,
                "solver": "rk4",
                "num_steps": 25,
                "integration_time": 1.0,
                "precision": "fp32",
                "noise_model": "none",
                "sparsity_ratio": 0.0,
            },
            "solvers": {
                "rk4": {"steps": [25]},
                "euler": {"steps": [5]},
            },
            "noise_models": ["none", "L0_static_mismatch"],
            "sparsity_configs": {
                "dense": {"sparsity_ratio": 0.0},
                "threshold_pruned": {"allowed_sparsity_ratios": [0.2, 0.4]},
                "degree_bounded": {"max_degree": [8, 16]},
            },
            "tolerances": {
                "compile_check": {"rtol": 1e-4, "atol": 1e-4},
                "solver_accuracy": {"rtol": 5e-2, "atol": 5e-2, "max_fid_degradation": 2.0},
            },
            "experiment_limits": {
                "max_experiments": 10,
                "max_iterations": 10,
                "max_node_retries": 3,
                "timeout_seconds_per_eval": 300,
            },
        }
        config = SimulationConfig.model_validate(config_data)
        yield db, config


# --- Tests for Node 0 Gate ---


def test_node0_reference_mode(tmp_db_and_config):
    db, config = tmp_db_and_config

    ref_id = db.insert_proposal(
        candidate_id="ref_rk4_001",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )

    res = node0_gate(
        db=db,
        config=config,
        trigger_source="phase0_node3",
        previous_node_id="phase0_node3",
        proposal_id=ref_id,
    )

    assert res["status"] == "SUCCESS"
    assert res["mode"] == "reference"
    assert res["next_node"] in ("node3_phase1_compiler_rewriter", "node3_compiler_rewriter")
    assert res["nodes_to_run"] == [3]
    assert 1 in res["bypassed_nodes"] and 2 in res["bypassed_nodes"]


def test_node0_rewrite_correction_mode(tmp_db_and_config):
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_euler_001",
        solver="euler",
        noise_model="L0_static_mismatch",
    )

    res = node0_gate(
        db=db,
        config=config,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=cand_id,
        debug_request={"nodes_to_rerun": [1, 3]},  # Specifying N1 -> N3
    )

    # Note: when debug_request does not contain debugee_id or deactivate, it specifies rewrite correction
    assert res["status"] in ("SUCCESS", "REJECTED")
    if res["status"] == "SUCCESS":
        assert 3 in res["nodes_to_run"]


def test_node0_debug_mode_creation_and_counter_3(tmp_db_and_config):
    db, config = tmp_db_and_config

    parent_id = db.insert_proposal(
        candidate_id="cand_parent_001",
        solver="euler",
        noise_model="L0_static_mismatch",
    )

    # Valid debug request (deactivating noise only)
    debug_req = {
        "debugee_id": parent_id,
        "deactivate": "deactivate_noise",
        "deactivated_rewrites": ["noise"],
    }

    res = node0_gate(
        db=db,
        config=config,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=parent_id,
        debug_request=debug_req,
    )

    assert res["status"] == "SUCCESS"
    assert res["mode"] == "debug"
    assert res["next_node"] in ("node3_phase1_compiler_rewriter", "node3_compiler_rewriter")
    debug_id = res["candidate_id"]

    # Verify debug candidate created and parent quarantined in DB
    debug_cand = db.get_proposal(debug_id)
    assert debug_cand is not None
    assert debug_cand["is_debug"] == 1

    parent_cand = db.get_proposal(parent_id)
    assert parent_cand["status"] == "QUARANTINED"

    # Invalid debug request: non-reference candidate with both configs deactivated
    invalid_debug_req = {
        "debugee_id": parent_id,
        "deactivate_both": True,
        "deactivated_rewrites": ["noise", "solver"],
    }

    # Trigger invalid request 4 times to test Counter ID 3 (counter_3_debug_gate)
    for i in range(1, 4):
        res_inv = node0_gate(
            db=db,
            config=config,
            trigger_source="node8",
            previous_node_id="node8_diagnostic_proposal",
            proposal_id=parent_id,
            debug_request=invalid_debug_req,
        )
        assert res_inv["status"] == "REJECTED"
        assert res_inv["counter_3"] == i
        assert res_inv["next_node"] in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal")

    # 4th invalid attempt exceeds limit (> 3)
    res_failed = node0_gate(
        db=db,
        config=config,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=parent_id,
        debug_request=invalid_debug_req,
    )
    assert res_failed["status"] in ("TERMINATED_MAX_CYCLES", "FAILED")
    assert res_failed["reason"] == "counter_3_debug_gate_exceeded"
    assert res_failed["counter_3"] == 4
    assert res_failed["next_node"] is None


# --- Tests for Nodes 1, 2, 3 Rewriters ---


def test_node1_and_node2_rewriters(tmp_db_and_config):
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_rw_01",
        solver="euler",
        noise_model="L0_static_mismatch",
    )

    mock_llm = MockAntigravityLLM(mode="valid")

    # Node 1: Registered solver (euler) -> Deterministic reuse, 0 tokens consumed
    res1 = node1_solver_rewriter(db=db, config=config, candidate_id=cand_id, llm=mock_llm)
    assert res1["status"] == "SUCCESS"
    assert res1["solver"] == "euler"
    assert res1["deterministic"] is True
    assert res1["tokens_consumed"] == 0

    prop1 = db.get_proposal(cand_id)
    assert prop1["rewrite_solver_status"] == "SUCCESS"

    # Node 2: Registered noise (L0_static_mismatch) -> Deterministic reuse, 0 tokens consumed
    res2 = node2_noise_rewriter(db=db, config=config, candidate_id=cand_id, llm=mock_llm)
    assert res2["status"] == "SUCCESS"
    assert res2["noise_model"] == "L0_static_mismatch"
    assert res2["deterministic"] is True
    assert res2["tokens_consumed"] == 0

    prop2 = db.get_proposal(cand_id)
    assert prop2["rewrite_noise_status"] == "SUCCESS"

    # Verify no fake tokens recorded for deterministic execution
    summary_det = db.get_token_usage_summary(phase="phase_1")
    assert summary_det["overall"]["total_calls"] == 0

    # Test Agentic Path with unregistered solver and noise
    agentic_id = db.insert_proposal(
        candidate_id="cand_agentic_rw_02",
        solver="unregistered_custom_euler_v2",
        noise_model="unregistered_custom_drift_v2",
    )

    res1_agentic = node1_solver_rewriter(db=db, config=config, candidate_id=agentic_id, llm=mock_llm)
    assert res1_agentic["status"] == "SUCCESS"
    assert res1_agentic["deterministic"] is False
    assert res1_agentic["tokens_consumed"] > 0

    res2_agentic = node2_noise_rewriter(db=db, config=config, candidate_id=agentic_id, llm=mock_llm)
    assert res2_agentic["status"] == "SUCCESS"
    assert res2_agentic["deterministic"] is False
    assert res2_agentic["tokens_consumed"] > 0

    # Verify real token usage recorded when LLM actually executed
    summary_agentic = db.get_token_usage_summary(phase="phase_1")
    assert summary_agentic["overall"]["total_calls"] >= 2
    assert summary_agentic["overall"]["total_prompt_tokens"] > 0


def test_node3_compiler_rewriter_and_node4_runner(tmp_db_and_config):
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_comp_01",
        solver="rk4",
        noise_model="none",
    )

    mock_llm = MockAntigravityLLM(mode="valid")

    # Node 3: HOP functionalization rewrite with dynamic batching
    res3 = node3_compiler_rewriter(db=db, config=config, candidate_id=cand_id, target_device="mps", llm=mock_llm)
    assert res3["status"] == "SUCCESS"
    assert res3["target_device"] == "mps"
    assert res3["hop_functionalized"] is True
    assert res3["prepared_model"] is not None

    prop3 = db.get_proposal(cand_id)
    assert prop3["rewrite_compiler_status"] == "SUCCESS"

    # Node 4: Inductor compilation runner & eager vs inductor comparison
    res4 = node4_compilation_runner(
        db=db,
        config=config,
        candidate_id=cand_id,
        target_device="mps",
        prepared_model=res3.get("prepared_model"),
        eager_model=res3.get("eager_model"),
    )
    assert res4["status"] == "SUCCESS"
    assert "comparison_metrics" in res4
    assert res4["compiled_model"] is not None

    prop4 = db.get_proposal(cand_id)
    assert prop4["compilation_status"] == "SUCCESS"

    # Verify Node 3 agentic prompt records token metrics when force_rewrite=True
    res3_forced = node3_compiler_rewriter(
        db=db,
        config=config,
        candidate_id=cand_id,
        target_device="mps",
        llm=mock_llm,
        force_rewrite=True,
    )
    assert res3_forced["tokens_consumed"] > 0
    summary = db.get_token_usage_summary(phase="phase_1")
    assert summary["overall"]["total_calls"] >= 1
    assert "phase_1.node3_phase1_compiler_rewriter" in summary["by_node"]



# --- Tests for Node 5 Verification & Counter 1 ---


def test_node5_verify_correctness_counter_1(tmp_db_and_config):
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_verif_01",
        solver="rk4",
        noise_model="none",
    )

    # Pass case
    res_pass = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=lambda x: x,
        eager_model=lambda x: x,
    )
    assert res_pass["status"] == "PASSED"
    assert res_pass["counter_1"] == 0

    prop = db.get_proposal(cand_id)
    assert prop["counter_1_compilation"] == 0

    # Simulate failure by setting explicit failing comparison or dummy models
    import torch
    m1 = lambda x: x * 10.0
    m2 = lambda x: x * 1.0

    res_fail = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=m1,
        eager_model=m2,
        sample_input=torch.ones(10, 10),
    )
    assert res_fail["status"] == "FAILED"
    assert res_fail["counter_1"] == 1
    assert res_fail["next_node"] in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal")

    prop_f = db.get_proposal(cand_id)
    assert prop_f["counter_1_compilation"] == 1


# --- Tests for Node 6 Simulation & Profile ---


def test_node6_simulate_profile(tmp_db_and_config):
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_sim_01",
        solver="rk4",
        noise_model="L0_static_mismatch",
    )

    res = node6_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        target_device="mps",
        backend_mode="mps",
    )

    assert res["status"] == "SUCCESS"
    assert "metrics" in res
    metrics = res["metrics"]
    assert "latency_ms" in metrics
    assert "accuracy_fid" in metrics
    assert metrics["sample_count"] == 10000

    prop = db.get_proposal(cand_id)
    assert prop["status"] == "EVALUATING"
    assert prop["simulation_status"] == "SUCCESS"
    assert prop["latency_ms"] is not None


# --- Tests for Node 7 Evaluation Gate & Counter Management ---


def test_node7_evaluation_gate_reference(tmp_db_and_config):
    db, config = tmp_db_and_config

    ref_id = db.insert_proposal(
        candidate_id="ref_eval_01",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )

    res = node7_evaluation_gate(db=db, config=config, candidate_id=ref_id)
    assert res["status"] == "PASSED"
    assert res["is_reference"] is True
    assert res["next_node"] == "phase2_node1"

    prop = db.get_proposal(ref_id)
    assert prop["status"] == "COMPLETED"


def test_node7_evaluation_gate_pass_and_fail(tmp_db_and_config):
    db, config = tmp_db_and_config

    # Standard candidate with passing metrics
    cand_pass = db.insert_proposal(
        candidate_id="cand_gate_pass",
        solver="rk4",
        noise_model="none",
        relative_error=0.01,
        accuracy_fid=0.1,
    )

    res_pass = node7_evaluation_gate(db=db, config=config, candidate_id=cand_pass)
    assert res_pass["status"] == "PASSED"
    assert res_pass["next_node"] == "phase2_node1"

    prop_p = db.get_proposal(cand_pass)
    assert prop_p["status"] == "COMPLETED"

    # Out-of-tolerance candidate
    cand_fail = db.insert_proposal(
        candidate_id="cand_gate_fail",
        solver="euler",
        noise_model="L0_static_mismatch",
        relative_error=0.99,  # > rtol 0.05
        accuracy_fid=10.0,    # > max_fid_degradation 2.0
    )

    # 1st failure
    res_f1 = node7_evaluation_gate(db=db, config=config, candidate_id=cand_fail)
    assert res_f1["status"] == "FAILED"
    assert res_f1["counter_2"] == 1
    assert res_f1["next_node"] in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal")

    # 2nd and 3rd failures
    node7_evaluation_gate(db=db, config=config, candidate_id=cand_fail)
    res_f3 = node7_evaluation_gate(db=db, config=config, candidate_id=cand_fail)
    assert res_f3["counter_2"] == 3

    # 4th failure exceeds limit (> max_node_retries=3)
    res_f4 = node7_evaluation_gate(db=db, config=config, candidate_id=cand_fail)
    assert res_f4["status"] in ("TERMINATED_MAX_CYCLES", "FAILED")
    assert res_f4["reason"] == "counter_2_rewrite_exceeded"
    assert res_f4["next_node"] is None


def test_node7_debug_mode_invariant_and_counter_4(tmp_db_and_config):
    db, config = tmp_db_and_config

    parent_id = db.insert_proposal(candidate_id="parent_cand_1", solver="euler", noise_model="L0")
    debug_req = {"debugee_id": parent_id, "deactivate": "deactivate_noise"}

    gate_res = node0_gate(
        db=db,
        config=config,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=parent_id,
        debug_request=debug_req,
    )
    debug_id = gate_res["candidate_id"]

    # Debug simulation runs
    db.mark_proposal_status(debug_id, status="EVALUATING", relative_error=0.01, accuracy_fid=0.1)

    # Node 7 in debug mode increments Counter 4 and BLOCKS transition to Phase 2
    res_d1 = node7_evaluation_gate(db=db, config=config, candidate_id=debug_id)
    assert res_d1["status"] == "EVALUATED_DEBUG"
    assert res_d1["counter_4"] == 1
    assert res_d1["next_node"] in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal")  # Blocked from phase2_node1!


# --- Tests for Node 8 Diagnostic Proposal ---


def test_node8_diagnostic_proposal(tmp_db_and_config):
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_diag_01",
        solver="euler",
        noise_model="L0_static_mismatch",
        status="FAILED",
    )

    mock_llm = MockAntigravityLLM(mode="valid")

    res = node8_diagnostic_proposal(
        db=db,
        config=config,
        candidate_id=cand_id,
        error_logs="Numerical check failed: relative_error 0.8 exceeds rtol 0.05",
        llm=mock_llm,
    )

    assert "action" in res
    assert res["action"] in ("rewrite_correction", "debug_request")
    assert res["next_node"] == "node0_phase1_gate"

    # Verify comment was stored in DB
    comments = db.get_comments(cand_id)
    assert len(comments) >= 1
    assert "node8" in comments[0]["agent_name"]

    # Verify token usage recorded for node 8
    summary = db.get_token_usage_summary(phase="phase_1")
    assert summary["overall"]["total_calls"] >= 1
    assert "phase_1.node8_phase1_diagnostic_proposal" in summary["by_node"]


def test_phase1_counter_reset_on_entry_from_phase0(tmp_db_and_config):
    db, config = tmp_db_and_config

    # Candidate with non-zero counters
    cand_id = db.insert_proposal(
        candidate_id="cand_dirty_counters",
        solver="euler",
        noise_model="none",
        counter_1_compilation=3,
        counter_2_rewrite=2,
        counter_3_debug_gate=1,
        counter_4_debug_loop=4,
    )

    prop = db.get_proposal(cand_id)
    assert prop["counter_1_compilation"] == 3
    assert prop["counter_2_rewrite"] == 2
    assert prop["counter_3_debug_gate"] == 1
    assert prop["counter_4_debug_loop"] == 4

    # Entering Phase 1 coming from Phase 0
    res = node0_gate(
        db=db,
        config=config,
        trigger_source="phase_0",
        previous_node_id="phase0_node3",
        proposal_id=cand_id,
    )
    assert res["status"] == "SUCCESS"

    prop_after = db.get_proposal(cand_id)
    assert prop_after["counter_1_compilation"] == 0
    assert prop_after["counter_2_rewrite"] == 0
    assert prop_after["counter_3_debug_gate"] == 0
    assert prop_after["counter_4_debug_loop"] == 0

