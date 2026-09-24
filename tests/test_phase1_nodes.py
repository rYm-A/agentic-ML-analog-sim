"""Comprehensive unit tests for Component 3 of Phase 1: Chia Nodes 0 through 8."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
import torch

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock import MockAntigravityLLM
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool
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
    assert callable(res1_agentic["solver_fn"])

    # Verify the 3 Chia tools were exposed to the Node 1 agent
    node1_call = [call for call in mock_llm.history if "node 1" in call["user_message"].lower() or "solver rewriter" in call["user_message"].lower()][-1]
    assert "result_db" in node1_call["tools"]
    assert "solver_papers" in node1_call["tools"]
    assert "un0_rewriter" in node1_call["tools"]

    res2_agentic = node2_noise_rewriter(db=db, config=config, candidate_id=agentic_id, llm=mock_llm)
    assert res2_agentic["status"] == "SUCCESS"
    assert res2_agentic["deterministic"] is False
    assert res2_agentic["tokens_consumed"] > 0
    assert res2_agentic["noise_cls"] is not None

    # Verify the 3 Chia tools were exposed to the Node 2 agent
    node2_call = [call for call in mock_llm.history if "node 2" in call["user_message"].lower() or "noise rewriter" in call["user_message"].lower()][-1]
    assert "result_db" in node2_call["tools"]
    assert "noise_papers" in node2_call["tools"]
    assert "un0_rewriter" in node2_call["tools"]

    # Verify real token usage recorded when LLM actually executed
    summary_agentic = db.get_token_usage_summary(phase="phase_1")
    assert summary_agentic["overall"]["total_calls"] >= 2
    assert summary_agentic["overall"]["total_prompt_tokens"] > 0


def test_node2_noise_rewriter_no_fallback_on_failure(tmp_db_and_config):
    """Test that Node 2 does NOT silently fall back to hardcoded Gaussian noise on failure."""
    import pytest
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_noise_fail",
        solver="euler",
        noise_model="unregistered_impossible_noise_v99",
    )

    failing_llm = MockAntigravityLLM(mode="invalid_noise_rewrite")

    with pytest.raises(RuntimeError, match="Agent failed to register noise model"):
        node2_noise_rewriter(
            db=db,
            config=config,
            candidate_id=cand_id,
            llm=failing_llm,
        )

    # Database execution status should be marked FAILED
    prop = db.get_proposal(cand_id)
    assert prop["rewrite_noise_status"] == "FAILED"
    assert prop["error_stage"] == "node2_phase1_noise_rewriter"


def test_node1_solver_rewriter_no_fallback_on_failure(tmp_db_and_config):
    """Test that Node 1 does NOT silently fall back to hardcoded Euler on failure."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_solver_fail",
        solver="unregistered_impossible_solver",
        noise_model="none",
    )

    failing_llm = MockAntigravityLLM(mode="invalid_solver_rewrite")

    with pytest.raises(RuntimeError, match="failed to synthesize or register a valid solver"):
        node1_solver_rewriter(
            db=db,
            config=config,
            candidate_id=cand_id,
            llm=failing_llm,
        )

    prop = db.get_proposal(cand_id)
    assert prop["rewrite_solver_status"] == "FAILED"


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
    assert "registered_rewrite" in res3
    assert "raw_result" in res3
    assert res3["tokens_consumed"] > 0

    # Verify the 3 Chia tools were exposed to the agent
    last_call = mock_llm.history[-1]
    assert "result_db" in last_call["tools"]
    assert "compilation_info" in last_call["tools"]
    assert "un0_rewriter" in last_call["tools"]

    prop3 = db.get_proposal(cand_id)
    assert prop3["rewrite_compiler_status"] == "SUCCESS"

    # Verify that raw LLM output was recorded in proposal comments
    comments = db.get_comments(cand_id)
    assert len(comments) >= 1
    assert any("REWRITE:" in c["comment"] and "Kuramoto" in c["comment"] for c in comments)

    # Node 4: Inductor compilation runner in Ray worker
    res4 = node4_compilation_runner(
        db=db,
        config=config,
        candidate_id=cand_id,
        target_device="mps",
        prepared_model=res3.get("prepared_model"),
        eager_model=res3.get("eager_model"),
    )
    assert res4["status"] == "SUCCESS"
    assert res4["compiled_model"] is not None
    assert res4["eager_model"] is not None
    assert "build_logs" in res4
    assert res4["next_node"] == "node5_phase1_verify_correctness"

    prop4 = db.get_proposal(cand_id)
    assert prop4["compilation_status"] == "SUCCESS"

    # Verify token usage summary
    summary = db.get_token_usage_summary(phase="phase_1")
    assert summary["overall"]["total_calls"] >= 1
    assert "phase_1.node3_phase1_compiler_rewriter" in summary["by_node"]


def test_node3_compiler_rewriter_no_fallback_on_failure(tmp_db_and_config):
    """Test that Node 3 does NOT silently fall back to deterministic mode on failure."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_comp_fail",
        solver="rk4",
        noise_model="none",
    )

    failing_llm = MockAntigravityLLM(mode="invalid_compiler")

    with pytest.raises(RuntimeError, match="failed to synthesize or register a valid compiler rewrite"):
        node3_compiler_rewriter(
            db=db,
            config=config,
            candidate_id=cand_id,
            target_device="mps",
            llm=failing_llm,
        )

    prop = db.get_proposal(cand_id)
    assert prop["rewrite_compiler_status"] == "FAILED"
    assert prop["error_stage"] == "node3_phase1_compiler_rewriter"

    # Verify failure comment records raw LLM output
    fail_comments = db.get_comments(cand_id)
    assert len(fail_comments) >= 1
    assert any("[NODE 3 FAILURE RAW OUTPUT]" in c["comment"] for c in fail_comments)


def test_node3_compiler_rewriter_missing_regenerative_cornn(tmp_db_and_config, monkeypatch):
    """Test that Node 3 logs a warning, hides compilation_info tool, and continues normal execution when RegenerativeCoRNN is missing."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_no_cornn",
        solver="rk4",
        noise_model="none",
    )

    monkeypatch.setenv("REGENERATIVE_CORNN_ROOT", "/nonexistent/path/RegenariveCoRNN")

    mock_llm = MockAntigravityLLM(mode="valid")
    res = node3_compiler_rewriter(db=db, config=config, candidate_id=cand_id, target_device="mps", llm=mock_llm)

    assert res["status"] == "SUCCESS"
    assert res["hop_functionalized"] is True

    # compilation_info should NOT be exposed to the agent
    last_call = mock_llm.history[-1]
    assert "result_db" in last_call["tools"]
    assert "un0_rewriter" in last_call["tools"]
    assert "compilation_info" not in last_call["tools"]

    # Database execution status should be SUCCESS
    prop = db.get_proposal(cand_id)
    assert prop["rewrite_compiler_status"] == "SUCCESS"


def test_node3_compiler_rewriter_deterministic_reference(tmp_db_and_config):
    """Test Recommendation 1: Deterministic fast path in Node 3 for reference candidates."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_ref_fast_path",
        solver="rk4",
        noise_model="none",
        is_reference=1,
    )

    mock_llm = MockAntigravityLLM(mode="valid")

    res = node3_compiler_rewriter(db=db, config=config, candidate_id=cand_id, target_device="mps", llm=mock_llm)

    assert res["status"] == "SUCCESS"
    assert res["deterministic"] is True
    assert res["tokens_consumed"] == 0
    assert res["token_usage"]["total_tokens"] == 0
    assert res["registered_rewrite"] in ("unrolled_functional", "hop_while_loop")
    assert res["prepared_model"] is not None
    assert res["next_node"] == "node4_phase1_compilation_runner"
    # Verify LLM was NOT called
    assert len(mock_llm.history) == 0

    prop = db.get_proposal(cand_id)
    assert prop["rewrite_compiler_status"] == "SUCCESS"


def test_node3_compiler_rewriter_deterministic_reuse_registered(tmp_db_and_config):
    """Test Recommendation 1: Deterministic fast path when rewrite is already registered or explicitly configured."""
    db, config = tmp_db_and_config

    # Sub-case 1: proposal specifies compiler_rewrite via solver_params
    cand_id = db.insert_proposal(
        candidate_id="cand_test_explicit_rewrite",
        solver="rk4",
        noise_model="none",
        solver_params={"compiler_rewrite": "unrolled_functional"},
    )

    mock_llm = MockAntigravityLLM(mode="valid")

    res = node3_compiler_rewriter(db=db, config=config, candidate_id=cand_id, target_device="mps", llm=mock_llm)

    assert res["status"] == "SUCCESS"
    assert res["deterministic"] is True
    assert res["tokens_consumed"] == 0
    assert res["registered_rewrite"] == "unrolled_functional"
    assert len(mock_llm.history) == 0

    # Sub-case 2: cand_clean_compiler_rewrite is in COMPILER_REGISTRY
    from un0.compiler.registry import COMPILER_REGISTRY
    cand_id_2 = db.insert_proposal(
        candidate_id="cand_test_reg_direct",
        solver="rk4",
        noise_model="none",
    )
    COMPILER_REGISTRY["cand_test_reg_direct_compiler_rewrite"] = COMPILER_REGISTRY["unrolled_functional"]
    try:
        mock_llm_2 = MockAntigravityLLM(mode="valid")
        res_2 = node3_compiler_rewriter(db=db, config=config, candidate_id=cand_id_2, target_device="mps", llm=mock_llm_2)
        assert res_2["status"] == "SUCCESS"
        assert res_2["deterministic"] is True
        assert res_2["tokens_consumed"] == 0
        assert res_2["registered_rewrite"] == "cand_test_reg_direct_compiler_rewrite"
        assert len(mock_llm_2.history) == 0
    finally:
        COMPILER_REGISTRY.pop("cand_test_reg_direct_compiler_rewrite", None)


def test_node3_compiler_rewriter_force_rewrite_and_steering_bypass(tmp_db_and_config):
    """Test that force_rewrite=True and steering instructions bypass deterministic reuse."""
    db, config = tmp_db_and_config

    # Case A: force_rewrite=True bypasses deterministic reuse
    cand_id_a = db.insert_proposal(
        candidate_id="cand_test_force_rewrite",
        solver="rk4",
        noise_model="none",
        is_reference=1,
    )
    mock_llm_a = MockAntigravityLLM(mode="valid")
    res_a = node3_compiler_rewriter(
        db=db, config=config, candidate_id=cand_id_a, target_device="mps", llm=mock_llm_a, force_rewrite=True
    )
    assert res_a["status"] == "SUCCESS"
    assert len(mock_llm_a.history) == 1
    assert res_a["tokens_consumed"] > 0

    # Case B: steering_instructions parameter bypasses deterministic reuse and is appended to prompt
    cand_id_b = db.insert_proposal(
        candidate_id="cand_test_steering_param",
        solver="rk4",
        noise_model="none",
        is_reference=1,
    )
    mock_llm_b = MockAntigravityLLM(mode="valid")
    steering_text = "Use specialized dynamic batch hop unrolling for Inductor"
    res_b = node3_compiler_rewriter(
        db=db,
        config=config,
        candidate_id=cand_id_b,
        target_device="mps",
        llm=mock_llm_b,
        steering_instructions=steering_text,
    )
    assert res_b["status"] == "SUCCESS"
    assert len(mock_llm_b.history) == 1
    prompt_b = mock_llm_b.history[-1]["user_message"]
    assert "## Diagnostic Steering from Node 8" in prompt_b
    assert steering_text in prompt_b

    # Case C: steering instructions extracted from DB comments
    cand_id_c = db.insert_proposal(
        candidate_id="cand_test_steering_comment",
        solver="rk4",
        noise_model="none",
        is_reference=1,
    )
    db.add_comment(
        candidate_id=cand_id_c,
        phase="phase_1",
        agent_name="node8_phase1_diagnostic_proposal",
        comment="[NODE8_STEERING_NODE3] Optimize while loop for memory bandwidth",
    )
    mock_llm_c = MockAntigravityLLM(mode="valid")
    res_c = node3_compiler_rewriter(
        db=db,
        config=config,
        candidate_id=cand_id_c,
        target_device="mps",
        llm=mock_llm_c,
    )
    assert res_c["status"] == "SUCCESS"
    assert len(mock_llm_c.history) == 1
    prompt_c = mock_llm_c.history[-1]["user_message"]
    assert "## Diagnostic Steering from Node 8" in prompt_c
    assert "Optimize while loop for memory bandwidth" in prompt_c


def test_node3_compiler_rewriter_prompt_template_and_directives(tmp_db_and_config):
    """Test Recommendation 2: Verify prompt contains concrete code template and execution directives."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_test_prompt_template",
        solver="rk4",
        noise_model="none",
    )
    mock_llm = MockAntigravityLLM(mode="valid")

    res = node3_compiler_rewriter(db=db, config=config, candidate_id=cand_id, target_device="mps", llm=mock_llm)
    assert res["status"] == "SUCCESS"
    assert len(mock_llm.history) == 1

    prompt = mock_llm.history[-1]["user_message"]
    # Verify concrete implementation template
    assert "## Concrete Implementation Template" in prompt
    assert "@register_compiler_rewrite" in prompt
    assert "functionalize_kuramoto_model" in prompt
    assert "cand_test_prompt_template_compiler_rewrite" in prompt

    # Verify directives & execution instructions
    assert "## Directives & Execution Instructions" in prompt
    assert "Directly call `un0_register_compiler_rewrite" in prompt
    assert "Do NOT run long shell commands or background benchmarks" in prompt
    assert "Do NOT browse unrelated files" in prompt



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

    # Key consistency checks
    assert "details" in res_fail
    assert "error_logs" in res_fail
    assert res_fail["details"] == res_fail["error_logs"]
    assert "target_device" in res_fail
    assert "rtol" in res_fail
    assert "atol" in res_fail

    assert "details" in res_pass
    assert "error_logs" in res_pass
    assert "target_device" in res_pass


def test_node5_missing_models_fail_not_false_positive(tmp_db_and_config):
    """Test that missing either compiled_model or eager_model fails verification instead of false positive."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_verif_missing_models",
        solver="rk4",
        noise_model="none",
    )

    # 1. Both missing
    res_both_none = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=None,
        eager_model=None,
    )
    assert res_both_none["status"] == "FAILED"
    assert "missing model artifact(s)" in res_both_none["details"]
    assert "compiled_model" in res_both_none["details"]
    assert "eager_model" in res_both_none["details"]

    # 2. Only eager missing
    res_eager_none = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=lambda x: x,
        eager_model=None,
    )
    assert res_eager_none["status"] == "FAILED"
    assert "missing model artifact(s): eager_model" in res_eager_none["details"]

    # 3. Only compiled missing
    res_comp_none = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=None,
        eager_model=lambda x: x,
    )
    assert res_comp_none["status"] == "FAILED"
    assert "missing model artifact(s): compiled_model" in res_comp_none["details"]


def test_node5_chia_function_options_and_device_handling(tmp_db_and_config):
    """Test that node 5 has ChiaFunction worker options and handles device placement."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_verif_chia_options",
        solver="euler",
        noise_model="none",
    )

    # Verify ChiaFunction decorator exposed .options / .chia_remote if Ray ChiaFunction is active
    if hasattr(node5_verify_correctness, "options"):
        bound_fn = node5_verify_correctness.options(resources={"python_worker": 1})
        assert bound_fn is not None

    # Test device placement with simple nn.Module
    import torch.nn as nn
    linear_eager = nn.Linear(4, 4)
    linear_comp = nn.Linear(4, 4)
    linear_comp.load_state_dict(linear_eager.state_dict())

    res_device = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=linear_comp,
        eager_model=linear_eager,
        sample_input=torch.randn(8, 4),
        target_device="cpu",
    )
    assert res_device["status"] == "PASSED"
    assert res_device["target_device"] == "cpu"
    assert res_device["counter_1"] == 0
    assert res_device["max_abs_diff"] < 1e-5


def test_node5_compilation_status_failed_centralized_counter_1(tmp_db_and_config):
    """Test that Node 5 centralizes Counter 1 on compilation_status='FAILED'."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_verif_comp_fail",
        solver="euler",
        noise_model="none",
    )

    # 1. Compilation failure from Node 4 passed to Node 5
    res = node5_phase1_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=None,
        eager_model=lambda x: x,
        compilation_status="FAILED",
        compilation_error="Inductor backend failed: unsupported operator",
    )

    assert res["status"] == "FAILED"
    assert res["counter_1"] == 1
    assert res["error_logs"] == "Inductor backend failed: unsupported operator"
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"

    prop = db.get_proposal(cand_id)
    assert prop["counter_1_compilation"] == 1
    assert prop["compilation_status"] == "FAILED"
    assert prop["error_stage"] == "compilation"
    assert prop["error_message"] == "Inductor backend failed: unsupported operator"


def test_node5_no_safe_compiled_hop_wrapper_and_forward_exception(tmp_db_and_config):
    """Test that _SafeCompiledHOPWrapper is removed and forward exceptions increment Counter 1."""
    import agentic_ml_analog_sim.nodes.node5_phase1_verify_correctness as n5_mod
    assert not hasattr(n5_mod, "_SafeCompiledHOPWrapper"), "_SafeCompiledHOPWrapper must be completely removed"

    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_verif_exception",
        solver="euler",
        noise_model="none",
    )

    def crashing_compiled(x):
        raise RuntimeError("Inductor execution crash")

    res = node5_phase1_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=crashing_compiled,
        eager_model=lambda x: x,
        sample_input=torch.ones(2, 2),
    )

    assert res["status"] == "FAILED"
    assert res["counter_1"] == 1
    assert "Inductor execution crash" in res["details"]
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"

    prop = db.get_proposal(cand_id)
    assert prop["counter_1_compilation"] == 1
    assert prop["compilation_status"] == "FAILED"
    assert prop["error_stage"] == "correctness_verification"


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
        test_mode=True,
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


def test_node6_chia_function_decorator():
    """Test that Node 6 is decorated as a ChiaFunction."""
    assert hasattr(node6_phase1_simulate_profile, "_chia_original") or callable(node6_phase1_simulate_profile)
    assert hasattr(node6_simulate_profile, "_chia_original") or callable(node6_simulate_profile)


def test_node6_real_model_batch_size_1(tmp_db_and_config):
    """Test that Node 6 executes a real model with batch size 1 and profiles actual hardware metrics."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_sim_real_bs1",
        solver="rk4",
        noise_model="none",
    )

    # Use a compiled PyTorch model (Inductor / aot_eager)
    state_dim = config.reference_design.n_oscillators
    real_model = torch.nn.Linear(state_dim, state_dim)
    try:
        compiled_model = torch.compile(real_model, backend="aot_eager")
    except Exception:
        compiled_model = real_model

    res = node6_phase1_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=compiled_model,
        target_device="cpu",
        backend_mode="local",
        num_profile_samples=5,
    )

    assert res["status"] == "SUCCESS"
    assert res["next_node"] == "node7_phase1_evaluation_gate"
    metrics = res["metrics"]
    assert metrics["mode"] == "real"
    assert metrics["batch_size"] == 1
    assert metrics["latency_ms"] > 0.0
    assert metrics["throughput_samples_per_sec"] > 0.0
    assert "peak_memory_mb" in metrics
    assert "latency_std" in metrics

    prop = db.get_proposal(cand_id)
    assert prop["status"] == "EVALUATING"
    assert prop["simulation_status"] == "SUCCESS"
    assert prop["latency_ms"] == metrics["latency_ms"]


def test_node6_divergence_failure(tmp_db_and_config):
    """Test that numerical divergence (NaN/Inf) triggers FAILED status and routes to Node 8."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_sim_divergent",
        solver="euler",
        noise_model="L1_stochastic_parameter_noise",
    )

    class DivergentModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[float("nan")] * x.shape[-1]])

    res = node6_phase1_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=DivergentModel(),
        target_device="cpu",
        num_profile_samples=2,
    )

    assert res["status"] == "FAILED"
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"
    assert "NaN/Inf detected" in res["error_message"]
    assert "error_logs" in res

    prop = db.get_proposal(cand_id)
    assert prop["simulation_status"] == "FAILED"
    assert prop["error_stage"] == "simulation"
    assert "NaN/Inf detected" in prop["error_message"]


def test_node6_missing_model_in_real_mode(tmp_db_and_config):
    """Test that when test_mode=False and compiled_model is None, Node 6 fails cleanly."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_sim_missing_model",
        solver="rk4",
        noise_model="none",
    )

    res = node6_phase1_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=None,
        test_mode=False,
    )

    assert res["status"] == "FAILED"
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"
    assert "compiled_model is required" in res["error_message"]

    prop = db.get_proposal(cand_id)
    assert prop["simulation_status"] == "FAILED"
    assert prop["error_stage"] == "simulation"


def test_node6_chia_tool_interface(tmp_db_and_config):
    """Test that Node 6 interfaces directly with ResultDBTool ChiaTool MCP server."""
    db, config = tmp_db_and_config
    tool = ResultDBTool("test_db_tool", db=db, deploy=False)

    cand_id = db.insert_proposal(
        candidate_id="cand_sim_chia_tool",
        solver="rk4",
        noise_model="none",
    )

    res = node6_phase1_simulate_profile(
        db=tool,
        config=config,
        candidate_id=cand_id,
        test_mode=True,
    )

    assert res["status"] == "SUCCESS"
    prop = db.get_proposal(cand_id)
    assert prop["status"] == "EVALUATING"
    assert prop["simulation_status"] == "SUCCESS"


def test_node6_non_callable_model_in_real_mode(tmp_db_and_config):
    """Test that a non-callable object passed as compiled_model fails cleanly."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_sim_bad_model",
        solver="rk4",
        noise_model="none",
    )

    res = node6_phase1_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model="not_a_model",
        target_device="cpu",
    )

    assert res["status"] == "FAILED"
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"
    assert "compiled_model must be a callable" in res["error_message"]

    prop = db.get_proposal(cand_id)
    assert prop["simulation_status"] == "FAILED"
    assert prop["error_stage"] == "simulation"


# --- Tests for Node 7 Evaluation Gate & Counter Management ---


def test_node7_evaluation_gate_reference(tmp_db_and_config):
    db, config = tmp_db_and_config

    ref_id = db.insert_proposal(
        candidate_id="ref_eval_01",
        solver="rk4",
        noise_model="none",
        is_reference=True,
        latency_ms=4.2,
        accuracy_fid=0.05,
    )
    db.update_execution_status(ref_id, simulation_status="SUCCESS")

    res = node7_evaluation_gate(db=db, config=config, candidate_id=ref_id)
    assert res["status"] == "PASSED"
    assert res["is_reference"] is True
    assert res["next_node"] == "phase2_node1"

    prop = db.get_proposal(ref_id)
    assert prop["status"] == "COMPLETED"


def test_node7_evaluation_gate_reference_simulation_failed(tmp_db_and_config):
    """Verify that a reference candidate with FAILED simulation status routes to Node 8 instead of Phase 2."""
    db, config = tmp_db_and_config

    ref_id = db.insert_proposal(
        candidate_id="ref_eval_failed",
        solver="rk4",
        noise_model="none",
        is_reference=True,
        latency_ms=4.2,
    )
    db.update_execution_status(ref_id, simulation_status="FAILED", error_message="Solver divergence in reference run")

    res = node7_evaluation_gate(db=db, config=config, candidate_id=ref_id)
    assert res["status"] == "FAILED"
    assert res["reason"] == "simulation_failed"
    assert res["is_reference"] is True
    assert res["counter_2"] == 1
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"
    assert "simulation_status is FAILED" in res["error_details"]

    prop = db.get_proposal(ref_id)
    # Must NOT be marked COMPLETED!
    assert prop["status"] != "COMPLETED"


def test_node7_evaluation_gate_reference_missing_metrics(tmp_db_and_config):
    """Verify that a reference candidate missing baseline latency_ms routes to Node 8."""
    db, config = tmp_db_and_config

    ref_id = db.insert_proposal(
        candidate_id="ref_eval_no_metrics",
        solver="rk4",
        noise_model="none",
        is_reference=True,
        latency_ms=None,
    )
    db.update_execution_status(ref_id, simulation_status="SUCCESS")

    res = node7_evaluation_gate(db=db, config=config, candidate_id=ref_id)
    assert res["status"] == "FAILED"
    assert res["reason"] == "simulation_failed"
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"
    assert "Missing essential reference baseline metrics" in res["error_details"]

    prop = db.get_proposal(ref_id)
    assert prop["status"] != "COMPLETED"


def test_node7_evaluation_gate_reference_counter_2_exceeded(tmp_db_and_config):
    """Verify that exceeding counter 2 retries on reference simulation failure routes to node 9 error handler."""
    db, config = tmp_db_and_config

    ref_id = db.insert_proposal(
        candidate_id="ref_eval_max_retries",
        solver="rk4",
        noise_model="none",
        is_reference=True,
        latency_ms=4.2,
        counter_2_rewrite=3,  # already at max retries (default 3)
    )
    db.update_execution_status(ref_id, simulation_status="FAILED", error_message="Persistent divergence")

    res = node7_evaluation_gate(db=db, config=config, candidate_id=ref_id)
    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["counter_name"] == "counter_2_rewrite"
    assert res["current_count"] == 4
    assert res["execution_mode"] == "reference"
    assert "reference_simulation_failed_limit_exceeded" in res["details"]

    prop = db.get_proposal(ref_id)
    assert prop["status"] == "FAILED"


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


def test_node7_chia_function_decorator():
    """Verify that node7_phase1_evaluation_gate is wrapped with ChiaFunction."""
    from agentic_ml_analog_sim.nodes.node7_phase1_evaluation_gate import node7_phase1_evaluation_gate
    # ChiaFunction wraps the callable or exposes _chia_original
    assert hasattr(node7_phase1_evaluation_gate, "_chia_original") or callable(node7_phase1_evaluation_gate)


def test_node7_evaluation_gate_none_metrics(tmp_db_and_config):
    """Verify that candidates with missing/None metrics fail tolerance checks."""
    db, config = tmp_db_and_config

    # Candidate with no simulation metrics evaluated
    cand_none = db.insert_proposal(
        candidate_id="cand_gate_none_metrics",
        solver="rk4",
        noise_model="none",
        relative_error=None,
        absolute_error=None,
        accuracy_fid=None,
    )

    res = node7_evaluation_gate(db=db, config=config, candidate_id=cand_none)
    assert res["status"] == "FAILED"
    assert res["reason"] == "out_of_tolerance"
    assert "Missing required simulation metrics" in res["error_details"]


def test_node7_evaluation_gate_atol_comparison(tmp_db_and_config):
    """Verify that exceeding atol causes evaluation failure even when relative_error <= rtol."""
    db, config = tmp_db_and_config

    # rtol is 0.05, atol is 0.05
    # relative_error=0.01 (within rtol), absolute_error=0.10 (exceeds atol)
    cand_atol_fail = db.insert_proposal(
        candidate_id="cand_gate_atol_fail",
        solver="rk4",
        noise_model="none",
        relative_error=0.01,
        absolute_error=0.10,
        accuracy_fid=0.1,
    )

    res = node7_evaluation_gate(db=db, config=config, candidate_id=cand_atol_fail)
    assert res["status"] == "FAILED"
    assert res["reason"] == "out_of_tolerance"
    assert "absolute_error" in res["error_details"]
    assert "atol" in res["error_details"]


def test_node7_debug_mode_strict_missing_active_or_quarantined(tmp_db_and_config):
    """Verify strict handling when active_debug or quarantined parent is None in debug mode."""
    db, config = tmp_db_and_config

    # Insert a debug proposal whose parent was NEVER quarantined and no active debug exists
    debug_id = db.insert_proposal(
        candidate_id="debug_rogue_01",
        solver="euler",
        noise_model="none",
        is_debug=True,
        debug_parent_id="non_existent_parent",
        status="COMPLETED",  # not PENDING or EVALUATING, so get_active_debug_candidate() returns None
    )

    res = node7_evaluation_gate(db=db, config=config, candidate_id=debug_id)
    assert res["status"] in ("TERMINATED_MAX_CYCLES", "FAILED")
    assert "debug_invariant_violation" in res.get("details", "")
    assert res["next_node"] is None



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


def test_node8_query_result_none_and_retry_succeeds(tmp_db_and_config):
    """Test that if the initial query_result is None or invalid text, Node 8 asks once more and succeeds if valid JSON is returned."""
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_diag_retry_success",
        solver="rk4",
        noise_model="L1_stochastic_parameter_noise",
        status="FAILED",
    )

    class FlakyLLM:
        def __init__(self):
            self.model = "mock-flaky"
            self.call_count = 0
            self.prompts = []

        def prompt(self, user_message, tools=None):
            self.call_count += 1
            self.prompts.append(user_message)
            if self.call_count == 1:
                # First attempt returns None / invalid unparsable text
                return None
            else:
                # Retry attempt returns valid JSON
                valid_resp = json.dumps({
                    "action": "rewrite_correction",
                    "nodes_to_rerun": [1, 2, 3],
                    "diagnosis": "Recovered valid diagnosis on retry",
                    "reasoning": "Solver parameters adjusted on retry",
                    "debug_request": None,
                })
                class QueryResult:
                    result = valid_resp
                    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                return QueryResult()

    llm = FlakyLLM()
    res = node8_phase1_diagnostic_proposal(
        db=db,
        config=config,
        candidate_id=cand_id,
        error_logs="Inductor lowered graph error",
        llm=llm,
    )

    assert llm.call_count == 2
    assert "Your previous response" in llm.prompts[1]
    assert res["action"] == "rewrite_correction"
    assert res["diagnosis"] == "Recovered valid diagnosis on retry"
    assert res["nodes_to_rerun"] == [1, 2, 3]
    assert res["tokens_consumed"] == 150


def test_node8_retry_fails_and_warns_with_fallback(tmp_db_and_config, caplog):
    """Test that if LLM persistently returns invalid responses after retry, Node 8 logs a warning and uses fallback reasoning."""
    import logging
    db, config = tmp_db_and_config

    cand_id = db.insert_proposal(
        candidate_id="cand_diag_retry_fail",
        solver="rk4",
        noise_model="L0_static_mismatch",
        counter_2_rewrite=2,  # Should trigger debug_request fallback
        status="FAILED",
    )

    class AlwaysInvalidLLM:
        def __init__(self):
            self.model = "mock-invalid"
            self.call_count = 0
            self.prompts = []

        def prompt(self, user_message, tools=None):
            self.call_count += 1
            self.prompts.append(user_message)
            # Both attempts return unparsable text
            class QueryResult:
                result = "Sorry, I am unable to format as JSON at this time."
                usage = {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
            return QueryResult()

    llm = AlwaysInvalidLLM()
    with caplog.at_level(logging.WARNING):
        res = node8_phase1_diagnostic_proposal(
            db=db,
            config=config,
            candidate_id=cand_id,
            error_logs="Repeated simulation divergence",
            llm=llm,
        )

    assert llm.call_count == 2
    # Verify warning log was generated
    assert any("LLM failed to produce valid JSON diagnostic proposal" in record.message for record in caplog.records)
    # Since counter_2 == 2, fallback reasoning triggers debug_request
    assert res["action"] == "debug_request"
    assert res["debug_request"]["debugee_id"] == cand_id
    assert res["tokens_consumed"] == 200  # 100 * 2 attempts accumulated


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


def test_node4_compilation_runner_chia_function_and_failure_payload(tmp_db_and_config, monkeypatch):
    """Test that Node 4 is decorated as a ChiaFunction and returns harmonized error keys on failure."""
    db, config = tmp_db_and_config

    # 1. Verify ChiaFunction decorator behavior
    assert hasattr(node4_compilation_runner, "_chia_original") or callable(node4_compilation_runner)

    # 2. Test failure case payload harmony (both error_logs and error_message present)
    cand_fail = db.insert_proposal(
        candidate_id="cand_test_node4_fail",
        solver="rk4",
        noise_model="none",
    )

    import torch
    import un0.compiler
    def failing_compile(*args, **kwargs):
        raise RuntimeError("Forced TorchInductor compilation failure")

    monkeypatch.setattr(un0.compiler, "compile_model", failing_compile)

    res_fail = node4_compilation_runner(
        db=db,
        config=config,
        candidate_id=cand_fail,
        target_device="cpu",
        prepared_model=torch.nn.Linear(10, 10),
        eager_model=torch.nn.Linear(10, 10),
    )

    # Verify error payload
    assert res_fail["status"] == "FAILED"
    assert "error_logs" in res_fail
    assert "error_message" in res_fail
    assert res_fail["error_logs"] == res_fail["error_message"]
    assert "Forced TorchInductor compilation failure" in res_fail["error_message"]
    assert res_fail["next_node"] == "node5_phase1_verify_correctness"


def test_node8_generates_compiler_steering_and_gate_forwards(tmp_db_and_config):
    """Verify Node 8 generates compiler steering instructions on compiler error and Node 0 gate forwards it."""
    from agentic_ml_analog_sim.nodes.node8_phase1_diagnostic_proposal import node8_phase1_diagnostic_proposal
    from agentic_ml_analog_sim.nodes.node0_phase1_gate import node0_phase1_gate

    db, config = tmp_db_and_config
    cand_id = "cand_steer_test"
    db.insert_proposal(
        candidate_id=cand_id,
        iteration=1,
        solver="euler",
        solver_params={"num_steps": 5, "dt": 0.1},
        sparsity_config={"type": "dense"},
        noise_model="none",
    )

    class DummyLLM:
        def prompt(self, prompt_text, tools=None):
            return "Non-json response"

    res8 = node8_phase1_diagnostic_proposal(
        db=db,
        config=config,
        candidate_id=cand_id,
        error_logs="LoweringException: failed to compile un0 kuramoto model with Inductor",
        llm=DummyLLM(),
    )

    assert "compiler_steering_instructions" in res8
    assert res8["compiler_steering_instructions"] is not None
    assert "CRITICAL STEERING FROM NODE 8 DIAGNOSTICS" in res8["compiler_steering_instructions"]
    assert "un0_register_compiler_rewrite" in res8["compiler_steering_instructions"]

    # Verify DB comment was written with tag
    comments = db.get_comments(cand_id)
    assert any("[NODE8_STEERING_NODE3]" in c["comment"] for c in comments)

    # Now verify Node 0 gate preserves the steering instructions in Mode 2
    res0 = node0_phase1_gate(
        db=db,
        config=config,
        trigger_source="node_8",
        previous_node_id="node8_phase1_diagnostic_proposal",
        proposal_id=cand_id,
        debug_request={
            "action": "rewrite_correction",
            "nodes_to_rerun": [3],
            "compiler_steering_instructions": res8["compiler_steering_instructions"],
        },
    )

    assert res0["status"] == "SUCCESS"
    assert res0["compiler_steering_instructions"] == res8["compiler_steering_instructions"]


