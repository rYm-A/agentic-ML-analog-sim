"""Unit tests for Node 4: Compilation Runner."""

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
    node4_phase1_compilation_runner,
    node4_compilation_runner,
)


@pytest.fixture
def tmp_db_and_config():
    """Create a temporary ResultDatabase and SimulationConfig."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_node4.db"
        db = ResultDatabase(db_path=db_path)
        config = load_simulation_config("config/simulation_config.yaml")
        yield db, config


class DummyModel(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def test_node4_compilation_runner_success(tmp_db_and_config):
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n4_success",
        solver="rk4",
        noise_model="none",
    )

    model = DummyModel()
    res = node4_compilation_runner(
        db=db,
        config=config,
        candidate_id=cand_id,
        target_device="cpu",
        prepared_model=model,
    )
    assert res["status"] == "SUCCESS"
    assert res["compiled_model"] is not None
    assert res["next_node"] == "node5_phase1_verify_correctness"

    prop = db.get_proposal(cand_id)
    assert prop["compilation_status"] == "SUCCESS"


def test_node4_compilation_runner_remote_worker_suppresses_direct_write(tmp_db_and_config, monkeypatch):
    """Verify that on a remote Ray worker, direct DB writes are skipped and rely on driver sync."""
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n4_remote",
        solver="rk4",
        noise_model="none",
    )

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "192.168.1.100")

    model = DummyModel()
    res = node4_compilation_runner(
        db=db,
        config=config,
        candidate_id=cand_id,
        target_device="cpu",
        prepared_model=model,
    )
    assert res["status"] == "SUCCESS"
    assert res["compiled_model"] is not None

    # Proposal compilation_status in DB should NOT have been updated by worker directly
    prop = db.get_proposal(cand_id)
    assert prop["compilation_status"] != "SUCCESS"


def test_node4_compilation_runner_readonly_db_suppression(tmp_db_and_config, monkeypatch):
    """Verify that with read-only DB, direct writes are skipped cleanly."""
    db, config = tmp_db_and_config
    cand_id = db.insert_proposal(
        candidate_id="cand_test_n4_readonly",
        solver="rk4",
        noise_model="none",
    )

    orig_access = os.access
    monkeypatch.setattr(os, "access", lambda path, mode: False if mode == os.W_OK else orig_access(path, mode))

    res = node4_compilation_runner(
        db=db,
        config=config,
        candidate_id=cand_id,
        target_device="cpu",
        prepared_model=DummyModel(),
    )
    assert res["status"] == "SUCCESS"
