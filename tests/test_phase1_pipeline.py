"""Integration tests for Phase 1 end-to-end pipeline execution (Nodes 0 through 8)."""

from __future__ import annotations

from pathlib import Path
import pytest

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.phase_1 import run_phase_1, Phase1Result


@pytest.fixture
def test_config(tmp_path: Path):
    """Fixture providing a valid SimulationConfig object."""
    repo_root = Path(__file__).parents[1]
    config_file = repo_root / "config/simulation_config.yaml"
    if not config_file.is_file():
        config_file = Path("agentic-ML-analog-sim/config/simulation_config.yaml")
    return load_simulation_config(config_file)


@pytest.fixture
def test_db(tmp_path: Path):
    """Fixture providing a fresh temporary ResultDatabase."""
    db_file = tmp_path / "test_phase1.db"
    return ResultDatabase(db_path=db_file)


def test_phase1_reference_profiling_flow(test_config, test_db):
    """Test Reference Profiling Mode in Phase 1 (bypasses Nodes 1 & 2 directly to Node 3)."""
    # Insert baseline reference proposal
    ref_id = test_db.insert_proposal(
        solver="rk4",
        noise_model="none",
        sparsity_config={"sparsity": 0.0},
        status="PENDING",
        accuracy_fid=0.95,
        relative_error=0.01,
        latency_ms=10.0,
        is_reference=True,
    )

    llm = MockAntigravityLLM(mode="valid")

    result = run_phase_1(
        config_path=test_config,
        db_path=test_db,
        candidate_id=ref_id,
        trigger_source="phase_0",
        target_device="cpu",
        llm=llm,
        dry_run=True,
    )

    assert result.action == "PROCEED_TO_PHASE_2"
    assert result.candidate_id == ref_id
    assert result.is_reference is True
    assert result.execution_mode in ("reference", "REFERENCE_PROFILING")
    assert result.cycle_count == 1
    assert "latency_ms" in result.metrics


def test_phase1_candidate_rewrite_flow(test_config, test_db):
    """Test standard candidate proposal rewrite flow in Phase 1."""
    cand_id = test_db.insert_proposal(
        solver="euler_backward",
        noise_model="L1_stochastic_parameter_noise",
        sparsity_config={"sparsity": 0.1},
        status="PENDING",
        accuracy_fid=0.92,
        relative_error=0.02,
        latency_ms=8.5,
        is_reference=False,
    )

    llm = MockAntigravityLLM(mode="valid")

    result = run_phase_1(
        config_path=test_config,
        db_path=test_db,
        candidate_id=cand_id,
        trigger_source="phase_0",
        target_device="cpu",
        llm=llm,
        dry_run=True,
    )

    assert result.action == "PROCEED_TO_PHASE_2"
    assert result.candidate_id == cand_id
    assert result.is_reference is False
    assert result.execution_mode in ("standard", "REWRITE_CORRECTION")
    assert result.cycle_count >= 1


def test_phase1_compilation_failure_diagnostic_correction_loop(test_config, test_db, monkeypatch):
    """Test compilation failure triggering Node 8 diagnostic agent and Node 0 correction loop."""
    cand_id = test_db.insert_proposal(
        solver="rk4",
        noise_model="L0_static_mismatch",
        sparsity_config={"sparsity": 0.0},
        status="PENDING",
        accuracy_fid=0.90,
        relative_error=0.03,
        latency_ms=12.0,
        is_reference=False,
    )

    llm = MockAntigravityLLM(mode="valid")
    compilation_attempts = [0]

    import agentic_ml_analog_sim.phase_1 as phase1_mod
    original_node4 = getattr(phase1_mod, "node4_phase1_compilation_runner", phase1_mod.node4_compilation_runner)

    def mock_node4(*args, **kwargs):
        compilation_attempts[0] += 1
        if compilation_attempts[0] == 1:
            return {
                "status": "FAILED",
                "compiled_model": None,
                "error_message": "Mock AOT compilation error: unsupported dynamic shape",
            }
        return original_node4(*args, **kwargs)

    monkeypatch.setattr(phase1_mod, "node4_phase1_compilation_runner", mock_node4)
    monkeypatch.setattr(phase1_mod, "node4_compilation_runner", mock_node4)

    result = run_phase_1(
        config_path=test_config,
        db_path=test_db,
        candidate_id=cand_id,
        trigger_source="phase_0",
        target_device="cpu",
        llm=llm,
        dry_run=True,
    )

    assert result.action == "PROCEED_TO_PHASE_2"
    assert result.cycle_count == 2
    assert compilation_attempts[0] == 2


def test_phase1_debug_mode_and_quarantine_workflow(test_config, test_db, monkeypatch):
    """Test evaluation out-of-tolerance triggering Node 8 debug request, debug candidate creation, quarantine, and reactivation."""
    cand_id = test_db.insert_proposal(
        solver="rk4",
        noise_model="L2_functional_interface",
        sparsity_config={"sparsity": 0.2},
        status="PENDING",
        accuracy_fid=0.88,
        relative_error=0.05,
        latency_ms=15.0,
        is_reference=False,
    )

    llm = MockAntigravityLLM(mode="valid")
    eval_attempts = [0]

    import agentic_ml_analog_sim.phase_1 as phase1_mod
    original_node7 = getattr(phase1_mod, "node7_phase1_evaluation_gate", phase1_mod.node7_evaluation_gate)

    def mock_node7(db, config, candidate_id):
        eval_attempts[0] += 1
        if eval_attempts[0] == 1:
            return {
                "action": "TRIGGER_NODE_8",
                "reason": "Accuracy degraded below threshold (0.88 < 0.90)",
            }
        return original_node7(db, config, candidate_id)

    monkeypatch.setattr(phase1_mod, "node7_phase1_evaluation_gate", mock_node7)
    monkeypatch.setattr(phase1_mod, "node7_evaluation_gate", mock_node7)

    result = run_phase_1(
        config_path=test_config,
        db_path=test_db,
        candidate_id=cand_id,
        trigger_source="phase_0",
        target_device="cpu",
        llm=llm,
        dry_run=True,
    )

    assert result.action == "PROCEED_TO_PHASE_2"
    assert result.cycle_count == 2
    parent_prop = test_db.get_proposal(cand_id)
    assert parent_prop["status"] in ("PENDING", "EVALUATING", "COMPLETED")


def test_phase1_pipeline_reset_counters_on_entry_from_phase0(test_db, test_config):
    llm = MockAntigravityLLM(mode="valid")
    cand_id = test_db.insert_proposal(
        candidate_id="cand_pipeline_dirty_counters",
        solver="euler",
        noise_model="none",
        counter_1_compilation=4,
        counter_2_rewrite=2,
        counter_3_debug_gate=3,
        counter_4_debug_loop=1,
    )

    prop_before = test_db.get_proposal(cand_id)
    assert prop_before["counter_1_compilation"] == 4
    assert prop_before["counter_2_rewrite"] == 2
    assert prop_before["counter_3_debug_gate"] == 3
    assert prop_before["counter_4_debug_loop"] == 1

    result = run_phase_1(
        config_path=test_config,
        db_path=test_db,
        candidate_id=cand_id,
        trigger_source="phase_0",
        target_device="cpu",
        llm=llm,
        dry_run=True,
    )

    assert result.action == "PROCEED_TO_PHASE_2"
    prop_after = test_db.get_proposal(cand_id)
    assert prop_after["counter_1_compilation"] == 0
    assert prop_after["counter_2_rewrite"] == 0
    assert prop_after["counter_3_debug_gate"] == 0
    assert prop_after["counter_4_debug_loop"] == 0


def test_dispatch_node_local_fallback():
    from agentic_ml_analog_sim.phase_1 import _dispatch_node

    def dummy_node(x, y):
        return {"sum": x + y}

    # Should execute locally and not pass timeout to dummy_node
    res = _dispatch_node(dummy_node, 3, 5, timeout=10.0)
    assert res == {"sum": 8}


def test_dispatch_node_ray_timeout_and_cancel(monkeypatch):
    import ray
    from agentic_ml_analog_sim.phase_1 import _dispatch_node

    # Mock remote function and ref
    fake_ref = "fake_ref_123"
    cancelled = []

    class MockRemoteNode:
        __name__ = "mock_slow_node"

        def chia_remote(self, *args, **kwargs):
            return fake_ref

    def fake_get(ref, timeout=None):
        assert ref == fake_ref
        assert timeout == 42.0
        raise ray.exceptions.GetTimeoutError("Task timed out")

    def fake_cancel(ref, force=True):
        cancelled.append((ref, force))

    monkeypatch.setattr("agentic_ml_analog_sim.phase_1.ray.is_initialized", lambda: True)
    monkeypatch.setattr("agentic_ml_analog_sim.phase_1.get", fake_get)
    monkeypatch.setattr(ray, "cancel", fake_cancel)

    res = _dispatch_node(MockRemoteNode(), timeout=42.0)
    assert res["status"] == "FAILED"
    assert "mock_slow_node timed out after 42.0s" in res["error"]
    assert cancelled == [(fake_ref, True)]


def test_dispatch_node_effective_timeout_from_config(monkeypatch):
    import ray
    from agentic_ml_analog_sim.phase_1 import _dispatch_node

    observed_timeouts = []

    class MockNode:
        __name__ = "mock_node"
        def chia_remote(self, *args, **kwargs):
            return "ref"

    def fake_get(ref, timeout=None):
        observed_timeouts.append(timeout)
        return {"status": "SUCCESS"}

    monkeypatch.setattr("agentic_ml_analog_sim.phase_1.ray.is_initialized", lambda: True)
    monkeypatch.setattr("agentic_ml_analog_sim.phase_1.get", fake_get)

    class DummyConfig:
        node_timeout = 150.0

    # 1. Timeout from config
    res = _dispatch_node(MockNode(), config=DummyConfig())
    assert res == {"status": "SUCCESS"}
    assert observed_timeouts[-1] == 150.0

    # 2. Explicit timeout overrides config
    _dispatch_node(MockNode(), config=DummyConfig(), timeout=45.0)
    assert observed_timeouts[-1] == 45.0


