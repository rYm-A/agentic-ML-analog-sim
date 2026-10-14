"""Unit tests for Node 6: Simulate & Profile."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest
import torch
import ray

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes import (
    node6_phase1_simulate_profile,
    node6_simulate_profile,
)


@pytest.fixture
def tmp_db_and_config():
    """Create a temporary ResultDatabase and SimulationConfig."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_node6.db"
        db = ResultDatabase(db_path=db_path)
        config = load_simulation_config("config/simulation_config.yaml")
        yield db, config


class DummyAnalogModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dim = 1024

    def forward(self, x, *args, **kwargs):
        return x


def test_node6_simulate_profile_test_mode(tmp_db_and_config):
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n6_testmode",
        solver="rk4",
        noise_model="none",
    )

    res = node6_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        test_mode=True,
        target_device="cpu",
    )
    assert res["status"] == "SUCCESS"
    assert "metrics" in res
    assert res["next_node"] == "node7_phase1_evaluation_gate"


def test_node6_simulate_profile_real_model(tmp_db_and_config):
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n6_real",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )

    model = DummyAnalogModel()
    res = node6_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=model,
        target_device="cpu",
        num_profile_samples=5,
    )
    assert res["status"] == "SUCCESS"
    assert "metrics" in res
    assert res["metrics"]["latency_ms"] >= 0.0
    assert res["next_node"] == "node7_phase1_evaluation_gate"


def test_node6_simulate_profile_remote_worker_returns_metrics_cleanly(tmp_db_and_config, monkeypatch):
    """Verify that on remote worker or read-only DB, failures to write do not crash the node and metrics are returned."""
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n6_remote",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )

    # Monkeypatch to simulate remote worker
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "192.168.1.100")

    model = DummyAnalogModel()
    res = node6_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=model,
        target_device="cpu",
        num_profile_samples=5,
    )
    assert res["status"] == "SUCCESS"
    assert "metrics" in res
    assert "latency_ms" in res["metrics"]
    assert res["next_node"] == "node7_phase1_evaluation_gate"

    # Status should not have been marked in DB directly by worker
    prop = db.get_proposal(cand_id)
    assert prop["status"] != "EVALUATING"


def test_node6_simulate_profile_readonly_db_returns_metrics_cleanly(tmp_db_and_config, monkeypatch):
    """Verify that if SQLite is read-only, node does not crash and returns metrics in result dict."""
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n6_readonly",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )

    orig_access = os.access
    monkeypatch.setattr(os, "access", lambda path, mode: False if mode == os.W_OK else orig_access(path, mode))

    model = DummyAnalogModel()
    res = node6_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=model,
        target_device="cpu",
        num_profile_samples=5,
    )
    assert res["status"] == "SUCCESS"
    assert "metrics" in res
    assert "latency_ms" in res["metrics"]
    assert res["next_node"] == "node7_phase1_evaluation_gate"


def test_node6_simulate_profile_unresolvable_num_classes_routes_to_node9(tmp_db_and_config):
    """Verify that when num_classes cannot be resolved, Node 6 routes cleanly to Node 9."""
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n6_invalid_config",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )

    # Set family to unknown and ensure num_classes is None
    if hasattr(config.reference_design, "num_classes"):
        config.reference_design.num_classes = None
    config.reference_design.family = "unknown_custom_task"
    if hasattr(config, "num_classes"):
        config.num_classes = None

    model = DummyAnalogModel()
    res = node6_simulate_profile(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=model,
        target_device="cpu",
    )

    assert res["status"] == "TERMINATED_MAX_CYCLES"
    assert res["counter_name"] == "configuration_error"
    assert res["next_node"] is None
    assert "Unrecoverable configuration error" in res["error_message"]
    assert "node6_phase1_simulate_profile" in res["error_stage"]

