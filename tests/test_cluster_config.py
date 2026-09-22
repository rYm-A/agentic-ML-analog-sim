"""Test suite for cluster configuration and package setup."""

import os
from pathlib import Path
import pytest
from chia.cluster.config import (
    load_config,
    load_raw_config,
    parse_gcp_nodes,
    _expand_node_placeholders,
    apply_cloud_network_mode,
    build_config,
)


def test_cluster_yaml_loads():
    """Verify default bare-metal cluster.yaml loads and validates."""
    cfg = load_config("cluster.yaml")
    assert cfg.cluster_name == "agentic-ml-analog-sim-mac"
    assert cfg.head_ip is not None and len(cfg.head_ip) > 0
    assert "local_worker" in cfg.node_types
    assert cfg.node_types["local_worker"].resources == {
        "control_worker": 1,
        "agent_worker": 1,
        "compile_worker": 1,
        "sim_worker": 1,
        "antigravity_creds": 1,
        "sqlite_db": 1,
    }
    assert any("ray start" in cmd for cmd in cfg.worker_start_ray_commands)
    assert any("conda activate chia_env" in cmd for cmd in cfg.node_types["local_worker"].worker_env_commands)
    assert cfg.node_types["local_worker"].docker is None


def test_cluster_bare_yaml_loads():
    """Verify dedicated bare profile cluster.bare.yaml loads and validates."""
    cfg = load_config("cluster.bare.yaml")
    assert cfg.cluster_name == "agentic-ml-analog-sim-mac"
    assert cfg.head_ip is not None and len(cfg.head_ip) > 0
    assert "local_worker" in cfg.node_types
    assert cfg.node_types["local_worker"].resources == {
        "control_worker": 1,
        "agent_worker": 1,
        "compile_worker": 1,
        "sim_worker": 1,
        "antigravity_creds": 1,
        "sqlite_db": 1,
    }
    assert any("ray start" in cmd for cmd in cfg.worker_start_ray_commands)
    assert any("conda activate chia_env" in cmd for cmd in cfg.node_types["local_worker"].worker_env_commands)
    assert cfg.node_types["local_worker"].docker is None


def test_cluster_cloud_yaml_loads(monkeypatch):
    """Verify loading cluster.cloud.yaml using chia.cluster.config.load_config."""
    monkeypatch.setenv("GCP_PROJECT", "test-project-123")
    monkeypatch.setenv("GCP_ZONE", "us-central1-a")
    monkeypatch.setenv("GCP_SSH_USER", "chia")
    monkeypatch.setenv("GCP_SSH_PRIVATE_KEY", "/tmp/id_ed25519")
    monkeypatch.setenv("TS_AUTHKEY", "tskey-auth-test-key")

    cfg = load_config("cluster.cloud.yaml")
    assert cfg.cluster_name == "agentic-ml-analog-sim-cloud"
    assert cfg.head_ip is not None and len(cfg.head_ip) > 0
    assert "local_orchestrator" in cfg.node_types
    assert "gpu_simulation_worker" in cfg.node_types
    assert cfg.node_types["local_orchestrator"].resources == {
        "control_worker": 1,
        "agent_worker": 1,
        "antigravity_creds": 1,
        "sqlite_db": 1,
    }
    assert cfg.node_types["gpu_simulation_worker"].resources == {
        "sim_worker": 1,
        "compile_worker": 1,
    }
    assert any("conda activate chia_env" in cmd for cmd in cfg.node_types["gpu_simulation_worker"].worker_env_commands)


def test_cluster_cloud_yaml_parses(monkeypatch):
    """Verify Phase 2 cloud burst cluster.cloud.yaml parses via raw loader and gcp parser."""
    monkeypatch.setenv("GCP_PROJECT", "test-project-123")
    monkeypatch.setenv("GCP_ZONE", "us-central1-a")
    monkeypatch.setenv("GCP_SSH_USER", "chia")
    monkeypatch.setenv("GCP_SSH_PRIVATE_KEY", "/tmp/id_ed25519")
    monkeypatch.setenv("TS_AUTHKEY", "tskey-auth-test-key")

    raw = load_raw_config("cluster.cloud.yaml")
    assert raw["cluster_name"] == "agentic-ml-analog-sim-cloud"
    assert raw["tailnet"]["manage_all"] is True

    gcp_result = parse_gcp_nodes(raw)
    assert gcp_result is not None
    node_configs, project, zone, _, _ = gcp_result
    assert project == "test-project-123"
    assert zone == "us-central1-a"
    assert "sim_gpu_pool" in node_configs
    sim_cfg = node_configs["sim_gpu_pool"]
    assert sim_cfg.machine_type == "g2-standard-8"
    assert sim_cfg.count == 1
    assert isinstance(sim_cfg.spot, bool)

    # Simulate cloud IP provisioning and build_config
    simulated_ips = {"sim_gpu_pool": ["34.100.0.1"]}
    raw_expanded = _expand_node_placeholders(raw, simulated_ips)
    apply_cloud_network_mode(raw_expanded, None, {}, gcp_result, simulated_ips, require_auth_key=True)
    cfg = build_config(raw_expanded)

    assert cfg.cluster_name == "agentic-ml-analog-sim-cloud"
    assert "local_orchestrator" in cfg.node_types
    assert "gpu_simulation_worker" in cfg.node_types
