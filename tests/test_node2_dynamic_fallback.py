"""Unit tests for Node 2 dynamic fallback, pre-validation, and token capping."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
    _get_allowed_solvers,
    _parse_llm_json_response,
    _synthesize_unexplored_candidate,
    node2_phase0_propose_candidate,
)


@pytest.fixture
def rk4_only_config() -> dict:
    """Config with only rk4 and euler_backward (no explicit euler)."""
    return {
        "solvers": {
            "rk4": {"steps": [15, 25]},
            "euler_backward": {"steps": [10, 20]},
        },
        "noise_models": ["L0_static_mismatch", "L1_thermal_drift"],
        "sparsity_configs": {
            "dense": {"sparsity_ratio": 0.0},
            "degree_bounded": {"max_degree": [64]},
        },
        "tolerances": {"compile_check": {"rtol": 1e-4, "atol": 1e-4}},
        "budget": {"max_experiments": 10, "max_node_retries": 3},
    }


def test_get_allowed_solvers_dict_and_list():
    cfg_dict = {"solvers": {"rk4": {}, "par_ode": {}}}
    assert _get_allowed_solvers(cfg_dict) == ["rk4", "par_ode"]

    cfg_list_dict = {"solvers": [{"name": "euler_backward"}, {"name": "rk4"}]}
    assert _get_allowed_solvers(cfg_list_dict) == ["euler_backward", "rk4"]

    cfg_list_str = {"solvers": ["parareal", "rk4"]}
    assert _get_allowed_solvers(cfg_list_str) == ["parareal", "rk4"]


def test_parse_llm_json_response_synthesizes_from_config(rk4_only_config, tmp_path):
    db_path = tmp_path / "test.db"
    db = ResultDatabase(db_path=db_path)

    # Empty raw text should trigger dynamic synthesis
    fallback = _parse_llm_json_response(
        raw_text="",
        config=rk4_only_config,
        db=db,
        iteration=1,
        target_noise_model="L1_thermal_drift",
    )

    assert fallback["solver"] in ["rk4", "euler_backward"]
    assert fallback["solver"] != "euler"
    assert fallback["noise_model"] == "L1_thermal_drift"
    assert fallback["num_steps"] in [15, 25, 10, 20]
    assert "phase_1_prompt" in fallback


def test_parse_llm_json_response_garbage_text_fallback_without_db(rk4_only_config):
    # Garbage non-JSON output
    fallback = _parse_llm_json_response(
        raw_text="Random conversational text without any JSON code fence",
        config=rk4_only_config,
        db=None,
        iteration=2,
    )
    assert fallback["solver"] in ["rk4", "euler_backward"]
    assert fallback["solver"] != "euler"


def test_node2_in_node_prevalidation_corrects_hallucinated_solver(rk4_only_config, tmp_path):
    db_path = tmp_path / "test.db"
    db = ResultDatabase(db_path=db_path)

    # Mock LLM that returns hallucinated 'euler' solver not in config
    hallucinated_response = json.dumps({
        "candidate_id": "cand_hallucinated",
        "solver": "euler",
        "num_steps": 5,
        "solver_params": {"num_steps": 5, "dt": 0.2},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"sparsity_ratio": 0.0},
    })
    mock_llm = MagicMock()
    mock_llm.prompt.return_value = hallucinated_response

    result = node2_phase0_propose_candidate(
        db=db,
        config=rk4_only_config,
        llm=mock_llm,
        iteration=1,
    )

    # In-node pre-validation should auto-correct solver to allowed solver
    assert result["solver"] in ["rk4", "euler_backward"]
    assert result["solver"] != "euler"


def test_node2_in_node_prevalidation_corrects_target_noise_mismatch(rk4_only_config, tmp_path):
    db_path = tmp_path / "test.db"
    db = ResultDatabase(db_path=db_path)

    # Mock LLM that returns L0_static_mismatch when L1_thermal_drift is required
    response = json.dumps({
        "candidate_id": "cand_mismatch_noise",
        "solver": "rk4",
        "num_steps": 15,
        "solver_params": {"num_steps": 15, "dt": 0.066667},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"sparsity_ratio": 0.0},
    })
    mock_llm = MagicMock()
    mock_llm.prompt.return_value = response

    result = node2_phase0_propose_candidate(
        db=db,
        config=rk4_only_config,
        llm=mock_llm,
        iteration=1,
        target_noise_model="L1_thermal_drift",
    )

    assert result["noise_model"] == "L1_thermal_drift"
    assert "L1_thermal_drift" in result["phase_1_prompt"]


def test_prompt_history_capped_to_10_proposals(rk4_only_config, tmp_path):
    db_path = tmp_path / "test.db"
    db = ResultDatabase(db_path=db_path)

    # Insert 25 proposals into DB
    for i in range(25):
        db.insert_proposal(
            candidate_id=f"cand_hist_{i:02d}",
            solver="rk4",
            solver_params={"num_steps": 15, "dt": 0.066667},
            noise_model="L0_static_mismatch",
            noise_params={"sigma": 0.01},
            sparsity_config={"sparsity_ratio": 0.0},
            status="COMPLETED" if i < 20 else "PENDING",
            is_reference=False,
        )

    mock_llm = MagicMock()
    # Return valid unexplored proposal
    mock_llm.prompt.return_value = json.dumps({
        "candidate_id": "cand_new_unexplored",
        "solver": "euler_backward",
        "num_steps": 20,
        "solver_params": {"num_steps": 20, "dt": 0.05},
        "noise_model": "L1_thermal_drift",
        "noise_params": {"alpha": 0.005},
        "sparsity_config": {"type": "dense", "sparsity_ratio": 0.0},
    })

    node2_phase0_propose_candidate(
        db=db,
        config=rk4_only_config,
        llm=mock_llm,
        iteration=26,
    )

    # Check prompt passed to LLM
    call_args = mock_llm.prompt.call_args
    user_prompt = call_args.kwargs.get("user_message", "")

    # Proposals cand_hist_00 to cand_hist_14 should NOT be in the history table
    assert "cand_hist_00" not in user_prompt
    assert "cand_hist_14" not in user_prompt

    # Proposals cand_hist_15 to cand_hist_24 (the last 10) MUST be in the history table
    for i in range(15, 25):
        assert f"cand_hist_{i:02d}" in user_prompt
