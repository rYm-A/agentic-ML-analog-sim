"""Unit and integration tests for SimulationConfig Counter Limits and Agent Effort Configurations.

Validates:
1. SimulationConfig parsing and defaulting of CounterLimitsConfig and AgentEffortsConfig.
2. Backwards compatibility with preexisting YAML configurations without these blocks.
3. Fallback resolution between budget/experiment_limits and counter_limits.
4. Propagation of counter limits to Node 0, Node 5, and Node 7 (triggering Node 9 when exceeded).
5. Propagation of agent model & effort settings to nodes and MockAntigravityLLM.
"""

from pathlib import Path
import pytest
import torch
from pydantic import ValidationError

from agentic_ml_analog_sim.config import (
    load_simulation_config,
    SimulationConfig,
    CounterLimitsConfig,
    AgentModelConfig,
    AgentEffortsConfig,
)
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.nodes.node0_phase1_gate import node0_phase1_gate
from agentic_ml_analog_sim.nodes.node5_phase1_verify_correctness import node5_phase1_verify_correctness
from agentic_ml_analog_sim.nodes.node7_phase1_evaluation_gate import node7_phase1_evaluation_gate
from agentic_ml_analog_sim.nodes.node1_phase1_solver_rewriter import (
    node1_phase1_solver_rewriter,
    _resolve_llm as resolve_node1_llm,
)
from agentic_ml_analog_sim.nodes.node2_phase1_noise_rewriter import (
    node2_phase1_noise_rewriter,
    _resolve_llm as resolve_node2_p1_llm,
)
from agentic_ml_analog_sim.nodes.node3_phase1_compiler_rewriter import (
    node3_phase1_compiler_rewriter,
    _resolve_llm as resolve_node3_llm,
)
from agentic_ml_analog_sim.nodes.node8_phase1_diagnostic_proposal import (
    node8_phase1_diagnostic_proposal,
    _resolve_llm as resolve_node8_llm,
)
from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
    node2_phase0_propose_candidate,
    _resolve_llm as resolve_node2_p0_llm,
)
from agentic_ml_analog_sim.nodes.node2_phase2_generate_final_report import (
    node2_phase2_generate_final_report,
    _resolve_llm as resolve_node2_p2_llm,
)


@pytest.fixture
def base_config() -> SimulationConfig:
    return load_simulation_config("config/simulation_config.yaml")


@pytest.fixture
def temp_db(tmp_path: Path) -> ResultDatabase:
    return ResultDatabase(db_path=tmp_path / "test_counters_effort.db")


def test_simulation_config_defaults(base_config: SimulationConfig):
    """Verify default counter limits and agent efforts match specification."""
    # Counter limits
    assert base_config.counter_limits.max_counter_1_compilation == 3
    assert base_config.counter_limits.max_counter_2_rewrite == 3
    assert base_config.counter_limits.max_counter_3_debug_gate == 3
    assert base_config.counter_limits.max_counter_4_debug_loop == 5

    # Agent efforts
    efforts = base_config.agent_efforts
    assert efforts.node2_phase0_propose_candidate.model == "gemini-2.5-pro"
    assert efforts.node2_phase0_propose_candidate.effort == "high"

    assert efforts.node1_phase1_solver_rewriter.model == "gemini-2.5-pro"
    assert efforts.node1_phase1_solver_rewriter.effort == "high"

    assert efforts.node2_phase1_noise_rewriter.model == "gemini-2.5-pro"
    assert efforts.node2_phase1_noise_rewriter.effort == "high"

    assert efforts.node3_phase1_compiler_rewriter.model == "gemini-2.5-pro"
    assert efforts.node3_phase1_compiler_rewriter.effort == "high"

    assert efforts.node8_phase1_diagnostic_proposal.model == "gemini-2.5-flash"
    assert efforts.node8_phase1_diagnostic_proposal.effort == "medium"

    assert efforts.node2_phase2_generate_final_report.model == "gemini-2.5-flash"
    assert efforts.node2_phase2_generate_final_report.effort == "medium"


def test_simulation_config_custom_parsing(base_config: SimulationConfig):
    """Verify custom counter limits and agent efforts parse properly from dict & YAML."""
    raw_dict = base_config.model_dump()
    raw_dict["counter_limits"] = {
        "max_counter_1_compilation": 2,
        "max_counter_2_rewrite": 4,
        "max_counter_3_debug_gate": 1,
        "max_counter_4_debug_loop": 8,
    }
    raw_dict["agent_efforts"] = {
        "node2_phase0_propose_candidate": {"model": "gemini-2.5-flash", "effort": "low"},
        "node1_phase1_solver_rewriter": {"model": "gemini-2.5-pro", "effort": "max"},
        "node2_phase1_noise_rewriter": {"model": "gemini-2.5-pro", "effort": "high"},
        "node3_phase1_compiler_rewriter": {"model": "gemini-2.5-pro", "effort": "high"},
        "node8_phase1_diagnostic_proposal": {"model": "gemini-2.5-flash", "effort": "low"},
        "node2_phase2_generate_final_report": {"model": "gemini-2.5-flash", "effort": "max"},
    }

    custom_cfg = SimulationConfig.model_validate(raw_dict)
    assert custom_cfg.counter_limits.max_counter_1_compilation == 2
    assert custom_cfg.counter_limits.max_counter_2_rewrite == 4
    assert custom_cfg.counter_limits.max_counter_3_debug_gate == 1
    assert custom_cfg.counter_limits.max_counter_4_debug_loop == 8

    assert custom_cfg.agent_efforts.node2_phase0_propose_candidate.effort == "low"
    assert custom_cfg.agent_efforts.node1_phase1_solver_rewriter.effort == "max"
    assert custom_cfg.agent_efforts.node2_phase2_generate_final_report.effort == "max"


def test_simulation_config_validation_constraints():
    """Verify validation constraints on counter limits and effort patterns."""
    with pytest.raises(ValidationError):
        CounterLimitsConfig(max_counter_1_compilation=0)  # gt=0

    with pytest.raises(ValidationError):
        CounterLimitsConfig(max_counter_4_debug_loop=-1)

    with pytest.raises(ValidationError):
        AgentModelConfig(model="foo", effort="invalid_effort")  # pattern regex


def test_simulation_config_backwards_compatibility_fallback(base_config: SimulationConfig):
    """Ensure omitting counter_limits & agent_efforts falls back smoothly."""
    data = base_config.model_dump()
    data.pop("counter_limits", None)
    data.pop("agent_efforts", None)
    data.pop("experiment_limits", None)
    data["budget"]["max_node_retries"] = 4

    cfg = SimulationConfig.model_validate(data)
    assert cfg.counter_limits.max_counter_1_compilation == 4
    assert cfg.counter_limits.max_counter_2_rewrite == 4
    assert cfg.counter_limits.max_counter_3_debug_gate == 4
    assert cfg.counter_limits.max_counter_4_debug_loop == 5
    assert cfg.agent_efforts.node1_phase1_solver_rewriter.effort == "high"


def test_mock_llm_effort_storage_and_diagnostics():
    """Verify MockAntigravityLLM stores effort and attaches it to diagnostics."""
    mock_llm = MockAntigravityLLM(effort="high")
    assert mock_llm.effort == "high"

    res = mock_llm.prompt("test prompt")
    assert "effort=high" in res.diagnostics
    assert mock_llm.history[-1]["effort"] == "high"

    # From extra_cli_args
    mock_llm2 = MockAntigravityLLM(extra_cli_args=["--effort", "max"])
    assert mock_llm2.effort == "max"
    res2 = mock_llm2.prompt("test prompt 2")
    assert "effort=max" in res2.diagnostics
    assert mock_llm2.history[-1]["effort"] == "max"


def test_node0_custom_counter_3_debug_gate_limit(temp_db: ResultDatabase, base_config: SimulationConfig):
    """Setting max_counter_3_debug_gate=1 triggers node 9 after 1 failure."""
    raw_dict = base_config.model_dump()
    raw_dict["counter_limits"]["max_counter_3_debug_gate"] = 1
    custom_cfg = SimulationConfig.model_validate(raw_dict)

    # Insert candidate
    parent_id = temp_db.insert_proposal(
        candidate_id="cand_parent_001",
        solver="euler",
        noise_model="L0_static_mismatch",
    )

    invalid_debug_req = {
        "debugee_id": parent_id,
        "deactivate_both": True,
        "deactivated_rewrites": ["noise", "solver"],
    }

    # 1st invalid debug request
    res1 = node0_phase1_gate(
        db=temp_db,
        config=custom_cfg,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=parent_id,
        debug_request=invalid_debug_req,
    )
    assert res1["status"] == "REJECTED"
    assert res1["counter_3"] == 1

    # 2nd invalid debug request exceeds limit 1 -> triggers Node 9
    res2 = node0_phase1_gate(
        db=temp_db,
        config=custom_cfg,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=parent_id,
        debug_request=invalid_debug_req,
    )
    assert res2["status"] in ("TERMINATED_MAX_CYCLES", "FAILED")
    assert res2["reason"] == "counter_3_debug_gate_exceeded"
    assert res2["counter_3"] == 2
    assert res2["next_node"] is None


def test_node5_custom_counter_1_compilation_limit(temp_db: ResultDatabase, base_config: SimulationConfig):
    """Setting max_counter_1_compilation=1 triggers node 9 after 1 failure."""
    raw_dict = base_config.model_dump()
    raw_dict["counter_limits"]["max_counter_1_compilation"] = 1
    custom_cfg = SimulationConfig.model_validate(raw_dict)

    cand_id = temp_db.insert_proposal(
        candidate_id="cand_verif_01",
        solver="rk4",
        noise_model="none",
    )

    m1 = lambda x: x * 10.0
    m2 = lambda x: x * 1.0

    # 1st failure: counter_1 = 1 <= 1 -> routes to Node 8
    res1 = node5_phase1_verify_correctness(
        db=temp_db,
        config=custom_cfg,
        candidate_id=cand_id,
        compiled_model=m1,
        eager_model=m2,
        sample_input=torch.ones(10, 10),
    )
    assert res1["status"] == "FAILED"
    assert res1["counter_1"] == 1
    assert res1["next_node"] in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal")

    # 2nd failure: counter_1 = 2 > 1 -> triggers Node 9
    res2 = node5_phase1_verify_correctness(
        db=temp_db,
        config=custom_cfg,
        candidate_id=cand_id,
        compiled_model=m1,
        eager_model=m2,
        sample_input=torch.ones(10, 10),
    )
    assert res2["status"] in ("TERMINATED_MAX_CYCLES", "FAILED")
    assert res2["reason"] == "counter_1_compilation_exceeded"
    assert res2["counter_1"] == 2
    assert res2["next_node"] is None


def test_node7_custom_counter_2_rewrite_limit(temp_db: ResultDatabase, base_config: SimulationConfig):
    """Setting max_counter_2_rewrite=1 triggers node 9 after 1 failure."""
    raw_dict = base_config.model_dump()
    raw_dict["counter_limits"]["max_counter_2_rewrite"] = 1
    custom_cfg = SimulationConfig.model_validate(raw_dict)

    cand_fail = temp_db.insert_proposal(
        candidate_id="cand_gate_fail",
        solver="euler",
        noise_model="L0_static_mismatch",
        relative_error=0.99,
        accuracy_fid=10.0,
    )

    # 1st failure: counter_2 = 1 <= 1 -> routes to Node 8
    res1 = node7_phase1_evaluation_gate(db=temp_db, config=custom_cfg, candidate_id=cand_fail)
    assert res1["status"] == "FAILED"
    assert res1["counter_2"] == 1
    assert res1["next_node"] in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal")

    # 2nd failure: counter_2 = 2 > 1 -> triggers Node 9
    res2 = node7_phase1_evaluation_gate(db=temp_db, config=custom_cfg, candidate_id=cand_fail)
    assert res2["status"] in ("TERMINATED_MAX_CYCLES", "FAILED")
    assert res2["reason"] == "counter_2_rewrite_exceeded"
    assert res2["counter_2"] == 2
    assert res2["next_node"] is None


def test_node7_custom_counter_4_debug_loop_limit(temp_db: ResultDatabase, base_config: SimulationConfig):
    """Setting max_counter_4_debug_loop=1 triggers node 9 after 1 debug iteration."""
    raw_dict = base_config.model_dump()
    raw_dict["counter_limits"]["max_counter_4_debug_loop"] = 1
    custom_cfg = SimulationConfig.model_validate(raw_dict)

    parent_id = temp_db.insert_proposal(candidate_id="parent_cand_1", solver="euler", noise_model="L0")
    debug_req = {"debugee_id": parent_id, "deactivate": "deactivate_noise"}

    gate_res = node0_phase1_gate(
        db=temp_db,
        config=custom_cfg,
        trigger_source="node8",
        previous_node_id="node8_diagnostic_proposal",
        proposal_id=parent_id,
        debug_request=debug_req,
    )
    debug_id = gate_res["candidate_id"]

    # Mark debug candidate EVALUATING
    temp_db.mark_proposal_status(debug_id, status="EVALUATING", relative_error=0.01, accuracy_fid=0.1)

    # 1st debug iteration: counter_4 = 1 <= 1 -> routes to Node 8
    res1 = node7_phase1_evaluation_gate(db=temp_db, config=custom_cfg, candidate_id=debug_id)
    assert res1["status"] == "EVALUATED_DEBUG"
    assert res1["counter_4"] == 1
    assert res1["next_node"] in ("node8_phase1_diagnostic_proposal", "node8_diagnostic_proposal")

    # 2nd debug iteration: counter_4 = 2 > 1 -> triggers Node 9
    res2 = node7_phase1_evaluation_gate(db=temp_db, config=custom_cfg, candidate_id=debug_id)
    assert res2["status"] in ("TERMINATED_MAX_CYCLES", "FAILED")
    assert res2["reason"] == "counter_4_debug_loop_exceeded"
    assert res2["counter_4"] == 2
    assert res2["next_node"] is None


def test_node_agent_efforts_resolution(base_config: SimulationConfig, monkeypatch):
    """Verify that _resolve_llm in all 6 nodes resolves model and effort from config.agent_efforts."""
    monkeypatch.setattr("chia.models.antigravity.AntigravityLLM", MockAntigravityLLM)

    raw_dict = base_config.model_dump()
    raw_dict["agent_efforts"] = {
        "node2_phase0_propose_candidate": {"model": "gemini-2.5-pro", "effort": "low"},
        "node1_phase1_solver_rewriter": {"model": "gemini-2.5-pro", "effort": "max"},
        "node2_phase1_noise_rewriter": {"model": "gemini-2.5-pro", "effort": "medium"},
        "node3_phase1_compiler_rewriter": {"model": "gemini-2.5-pro", "effort": "high"},
        "node8_phase1_diagnostic_proposal": {"model": "gemini-2.5-flash", "effort": "high"},
        "node2_phase2_generate_final_report": {"model": "gemini-2.5-flash", "effort": "max"},
    }
    custom_cfg = SimulationConfig.model_validate(raw_dict)

    llm_n1 = resolve_node1_llm(None, custom_cfg)
    assert llm_n1.effort == "max"

    llm_n2_p1 = resolve_node2_p1_llm(None, custom_cfg)
    assert llm_n2_p1.effort == "medium"

    llm_n3 = resolve_node3_llm(None, custom_cfg)
    assert llm_n3.effort == "high"

    llm_n8 = resolve_node8_llm(None, custom_cfg)
    assert llm_n8.effort == "high"

    llm_n2_p0 = resolve_node2_p0_llm(None, custom_cfg)
    assert llm_n2_p0.effort == "low"

    llm_n2_p2 = resolve_node2_p2_llm(None, custom_cfg)
    assert llm_n2_p2.effort == "max"


def test_node_agent_efforts_execution_and_token_recording(
    temp_db: ResultDatabase, base_config: SimulationConfig, tmp_path: Path, monkeypatch
):
    """Verify nodes execute and record token usage using configured agent effort/model."""
    monkeypatch.setattr("chia.models.antigravity.AntigravityLLM", MockAntigravityLLM)

    raw_dict = base_config.model_dump()
    raw_dict["agent_efforts"]["node1_phase1_solver_rewriter"] = {"model": "gemini-2.5-pro", "effort": "high"}
    raw_dict["agent_efforts"]["node2_phase1_noise_rewriter"] = {"model": "gemini-2.5-pro", "effort": "high"}
    raw_dict["agent_efforts"]["node3_phase1_compiler_rewriter"] = {"model": "gemini-2.5-pro", "effort": "high"}
    raw_dict["agent_efforts"]["node8_phase1_diagnostic_proposal"] = {"model": "gemini-2.5-flash", "effort": "medium"}
    raw_dict["agent_efforts"]["node2_phase0_propose_candidate"] = {"model": "gemini-2.5-pro", "effort": "high"}
    raw_dict["agent_efforts"]["node2_phase2_generate_final_report"] = {"model": "gemini-2.5-flash", "effort": "medium"}
    custom_cfg = SimulationConfig.model_validate(raw_dict)

    cand_id = temp_db.insert_proposal(solver="euler", noise_model="none")

    # Node 1
    node1_phase1_solver_rewriter(db=temp_db, config=custom_cfg, candidate_id=cand_id, force_rewrite=True)
    summary_n1 = temp_db.get_token_usage_summary(phase="phase_1")
    assert "gemini-2.5-pro" in summary_n1["by_model"]

    # Node 2
    node2_phase1_noise_rewriter(db=temp_db, config=custom_cfg, candidate_id=cand_id, force_rewrite=True)

    # Node 3
    node3_phase1_compiler_rewriter(db=temp_db, config=custom_cfg, candidate_id=cand_id)

    # Node 8
    node8_phase1_diagnostic_proposal(db=temp_db, config=custom_cfg, candidate_id=cand_id)
    summary_n8 = temp_db.get_token_usage_summary(phase="phase_1")
    assert "gemini-2.5-flash" in summary_n8["by_model"]

    # Node 2 Phase 2
    temp_db.mark_proposal_status(cand_id, "COMPLETED", latency_ms=10.0, relative_error=0.01)
    out_dir = tmp_path / "report_out"
    node2_phase2_generate_final_report(db=temp_db, config=custom_cfg, output_dir=out_dir, compile_pdf=False)
    summary_p2 = temp_db.get_token_usage_summary(phase="phase_2")
    assert "gemini-2.5-flash" in summary_p2["by_model"]
