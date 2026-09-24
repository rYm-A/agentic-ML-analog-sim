"""Unit tests for Node 5: Verify Correctness."""

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
    node5_phase1_verify_correctness,
    node5_verify_correctness,
)


@pytest.fixture
def tmp_db_and_config():
    """Create a temporary ResultDatabase and SimulationConfig."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_node5.db"
        db = ResultDatabase(db_path=db_path)
        config = load_simulation_config("config/simulation_config.yaml")
        yield db, config


class DummyModel(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def test_node5_verify_correctness_success(tmp_db_and_config):
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n5_success",
        solver="rk4",
        noise_model="none",
    )
    # Set counter 1 to 1 to test reset
    db.update_counters(cand_id, counter_1_compilation=1)

    model = DummyModel()
    res = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=model,
        eager_model=model,
        target_device="cpu",
    )
    assert res["status"] == "PASSED"
    assert res["counter_1"] == 0
    assert res["next_node"] == "node6_phase1_simulate_profile"

    # Counter 1 should be reset to 0
    prop = db.get_proposal(cand_id)
    assert prop["counter_1_compilation"] == 0


def test_node5_verify_correctness_remote_worker_suppresses_direct_write(tmp_db_and_config, monkeypatch):
    """Verify that on a remote worker, counter updates and status updates are skipped cleanly."""
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n5_remote",
        solver="rk4",
        noise_model="none",
    )
    db.update_counters(cand_id, counter_1_compilation=1)

    # Simulate remote worker
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "192.168.1.100")

    model = DummyModel()
    res = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=model,
        eager_model=model,
        target_device="cpu",
    )
    assert res["status"] == "PASSED"
    assert res["counter_1"] == 0

    # Counter 1 was NOT directly written by remote worker (driver handles sync)
    prop = db.get_proposal(cand_id)
    assert prop["counter_1_compilation"] == 1


def test_node5_verify_correctness_readonly_db_suppression(tmp_db_and_config, monkeypatch):
    """Verify that with read-only DB, failures to write are handled silently at DEBUG level."""
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n5_readonly",
        solver="rk4",
        noise_model="none",
    )

    orig_access = os.access
    monkeypatch.setattr(os, "access", lambda path, mode: False if mode == os.W_OK else orig_access(path, mode))

    model = DummyModel()
    res = node5_verify_correctness(
        db=db,
        config=config,
        candidate_id=cand_id,
        compiled_model=model,
        eager_model=model,
        target_device="cpu",
    )
    assert res["status"] == "PASSED"
    assert res["counter_1"] == 0
