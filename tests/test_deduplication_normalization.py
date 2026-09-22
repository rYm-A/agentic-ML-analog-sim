"""Unit tests for configuration normalization, canonicalization, and deduplication logic."""

import pytest
from unittest.mock import MagicMock

from agentic_ml_analog_sim.nodes.node3_phase0_validate_proposal import (
    _canonicalize_sparsity_config,
    _canonicalize_solver_params,
    _canonicalize_noise_params,
    _check_duplicate,
)
from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
    _synthesize_unexplored_candidate,
)


def test_canonicalize_sparsity_config():
    # Dense representations
    assert _canonicalize_sparsity_config({"sparsity_ratio": 0.0}) == {"type": "dense", "sparsity_ratio": 0.0}
    assert _canonicalize_sparsity_config({"type": "dense", "sparsity_ratio": 0.0}) == {"type": "dense", "sparsity_ratio": 0.0}
    assert _canonicalize_sparsity_config({"type": "DenseSparsityConfig", "sparsity_ratio": 0.0}) == {"type": "dense", "sparsity_ratio": 0.0}
    assert _canonicalize_sparsity_config({}) == {"type": "dense", "sparsity_ratio": 0.0}
    assert _canonicalize_sparsity_config(None) == {"type": "dense", "sparsity_ratio": 0.0}

    # Degree bounded representations
    deg_cfg1 = {"type": "degree_bounded", "max_degree": 128}
    deg_cfg2 = {"type": "DegreeBoundedSparsityConfig", "max_degree": 128, "sparsity_ratio": 0.0}
    assert _canonicalize_sparsity_config(deg_cfg1) == {"type": "degree_bounded", "max_degree": 128, "sparsity_ratio": 0.0}
    assert _canonicalize_sparsity_config(deg_cfg2) == {"type": "degree_bounded", "max_degree": 128, "sparsity_ratio": 0.0}

    # Threshold pruned representations
    thresh_cfg1 = {"type": "threshold_pruned", "sparsity_ratio": 0.5}
    thresh_cfg2 = {"sparsity_ratio": 0.5}
    assert _canonicalize_sparsity_config(thresh_cfg1) == {"type": "threshold_pruned", "sparsity_ratio": 0.5}
    assert _canonicalize_sparsity_config(thresh_cfg2) == {"type": "threshold_pruned", "sparsity_ratio": 0.5}


def test_canonicalize_solver_params():
    # Euler with explicit dt vs derived dt
    p1 = _canonicalize_solver_params("euler", {"num_steps": 10, "dt": 0.1})
    p2 = _canonicalize_solver_params("euler", {"num_steps": 10})
    assert p1 == p2
    assert p1["num_steps"] == 10
    assert p1["dt"] == 0.1

    # Missing num_steps fallback
    p3 = _canonicalize_solver_params("euler", {})
    assert p3["num_steps"] == 10


def test_canonicalize_noise_params():
    assert _canonicalize_noise_params("none", {"sigma": 0.01}) == {}
    assert _canonicalize_noise_params("L0_static_mismatch", {"sigma": 0.01}) == {"sigma": 0.01}
    assert _canonicalize_noise_params("L0_static_mismatch", {}) == {"sigma": 0.01}


def test_check_duplicate_detects_semantic_matches():
    mock_db = MagicMock(spec=["get_all_proposals"])
    # Mock DB returns one existing completed proposal
    mock_db.get_all_proposals.return_value = [
        {
            "candidate_id": "cand_euler_10_L0_dense",
            "status": "COMPLETED",
            "solver": "euler",
            "solver_params": {"num_steps": 10, "dt": 0.1},
            "noise_model": "L0_static_mismatch",
            "noise_params": {"sigma": 0.01},
            "sparsity_config": {"sparsity_ratio": 0.0},  # dense without type
        }
    ]

    # Proposal with type: "dense" and no explicit dt
    new_candidate = {
        "candidate_id": "cand_iter2_proposal",
        "solver": "euler",
        "solver_params": {"num_steps": 10},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"type": "dense", "sparsity_ratio": 0.0},
    }

    is_dup, reason = _check_duplicate("cand_iter2_proposal", new_candidate, mock_db)
    assert is_dup is True
    assert "cand_euler_10_L0_dense" in reason


def test_synthesize_unexplored_candidate():
    mock_db = MagicMock(spec=["get_all_proposals"])
    # Mark euler 5, 10 as already taken for dense L0
    mock_db.get_all_proposals.return_value = [
        {
            "candidate_id": "cand_euler5_dense",
            "status": "COMPLETED",
            "solver": "euler",
            "solver_params": {"num_steps": 5, "dt": 0.2},
            "noise_model": "L0_static_mismatch",
            "noise_params": {"sigma": 0.01},
            "sparsity_config": {"type": "dense", "sparsity_ratio": 0.0},
        },
        {
            "candidate_id": "cand_euler10_dense",
            "status": "COMPLETED",
            "solver": "euler",
            "solver_params": {"num_steps": 10, "dt": 0.1},
            "noise_model": "L0_static_mismatch",
            "noise_params": {"sigma": 0.01},
            "sparsity_config": {"type": "dense", "sparsity_ratio": 0.0},
        },
    ]

    config = {
        "solvers": {
            "euler": {"steps": [5, 10, 20]},
            "par_ode": {"steps": [5, 10]},
        },
        "noise_models": ["L0_static_mismatch"],
        "sparsity_configs": {
            "degree_bounded": {"max_degree": [64, 128]},
            "threshold_pruned": {"allowed_sparsity_ratios": [0.5]},
        },
    }

    synth = _synthesize_unexplored_candidate(config, mock_db, iteration=3)
    assert synth is not None
    assert synth["candidate_id"].startswith("cand_")
    # Euler 5 and 10 with dense are taken, so it should pick deg64, thresh0.5, or euler 20
    assert not (
        synth["solver"] == "euler"
        and synth["solver_params"]["num_steps"] in (5, 10)
        and synth["sparsity_config"]["type"] == "dense"
    )
