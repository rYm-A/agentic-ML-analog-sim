import tempfile
from pathlib import Path
import torch
import pytest

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node6_phase1_simulate_profile import (
    node6_phase1_simulate_profile,
    _simulate_batch1_worker,
)
from un0.model import ConditionalKuramotoDynamics
from un0.compiler.functional_hop import functionalize_kuramoto_model
from un0.noise.L0_static_mismatch import L0StaticMismatchNoise


@pytest.fixture
def tmp_db_and_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "results.db"
        db = ResultDatabase(db_path=db_path)
        config_data = {
            "experiment_name": "test_traj_exp",
            "reference_design": {
                "name": "cifar10/n64",
                "family": "cifar10",
                "n_oscillators": 64,
                "n_conditional_oscillators": 2,
                "solver": "rk4",
                "num_steps": 10,
                "integration_time": 1.0,
                "precision": "fp32",
                "noise_model": "none",
                "sparsity_ratio": 0.0,
            },
            "solvers": {
                "rk4": {"steps": [10]},
                "euler": {"steps": [5]},
            },
            "noise_models": ["none", "L0_static_mismatch"],
            "sparsity_configs": {
                "dense": {"sparsity_ratio": 0.0},
                "threshold_pruned": {"allowed_sparsity_ratios": [0.0]},
                "degree_bounded": {"max_degree": [8]},
            },
            "tolerances": {
                "compile_check": {"rtol": 1e-4, "atol": 1e-4},
                "solver_accuracy": {"rtol": 5e-2, "atol": 5e-2},
            },
        }
        config = SimulationConfig(**config_data)
        yield db, config


def test_reference_and_candidate_trajectory_errors(tmp_db_and_cfg):
    """Test that reference sets relative_error=0.0 and saves cache, and candidate computes real errors."""
    tmp_db, config = tmp_db_and_cfg

    # Insert reference proposal
    ref_id = tmp_db.insert_proposal(
        candidate_id="ref_test",
        solver="rk4",
        solver_params={"num_steps": 10},
        is_reference=True,
    )

    base_ref = ConditionalKuramotoDynamics(n_oscillators=64, n_conditional_oscillators=2, num_classes=10)
    ref_model = functionalize_kuramoto_model(base_ref, solver_name="rk4", num_steps=10, dt=0.1, use_hop=False)

    ref_res = node6_phase1_simulate_profile(
        db=tmp_db,
        config=config,
        candidate_id=ref_id,
        compiled_model=ref_model,
        target_device="cpu",
        backend_mode="local",
        num_profile_samples=5,
    )

    assert ref_res["status"] == "SUCCESS"
    assert ref_res["metrics"]["relative_error"] == 0.0
    assert ref_res["metrics"]["absolute_error"] == 0.0
    assert ref_res["metrics"]["absolute_error_p90"] == 0.0

    ref_prop = tmp_db.get_proposal(ref_id)
    assert ref_prop["relative_error"] == 0.0
    assert ref_prop["absolute_error"] == 0.0
    assert ref_prop["absolute_error_p90"] == 0.0

    # Insert candidate with L0 noise
    cand_id = tmp_db.insert_proposal(
        candidate_id="cand_test_l0",
        solver="rk4",
        solver_params={"num_steps": 10},
        noise_model="L0_static_mismatch",
        is_reference=False,
    )

    base_cand = ConditionalKuramotoDynamics(n_oscillators=64, n_conditional_oscillators=2, num_classes=10)
    l0_cand = L0StaticMismatchNoise(base_cand, sigma=0.05)
    cand_model = functionalize_kuramoto_model(l0_cand, solver_name="rk4", num_steps=10, dt=0.1, use_hop=False)

    cand_res = node6_phase1_simulate_profile(
        db=tmp_db,
        config=config,
        candidate_id=cand_id,
        compiled_model=cand_model,
        target_device="cpu",
        backend_mode="local",
        num_profile_samples=5,
    )

    assert cand_res["status"] == "SUCCESS"
    rel_err = cand_res["metrics"]["relative_error"]
    abs_err = cand_res["metrics"]["absolute_error"]
    abs_p90 = cand_res["metrics"]["absolute_error_p90"]

    assert rel_err > 0.0, f"Expected non-zero relative error, got {rel_err}"
    assert abs_err > 0.0, f"Expected non-zero absolute error, got {abs_err}"
    assert abs_p90 > 0.0, f"Expected non-zero p90 error, got {abs_p90}"

    cand_prop = tmp_db.get_proposal(cand_id)
    assert cand_prop["relative_error"] == rel_err
    assert cand_prop["absolute_error"] == abs_err
    assert cand_prop["absolute_error_p90"] == abs_p90

    # Insert second candidate with Euler and coarser steps to verify dynamic divergence
    cand2_id = tmp_db.insert_proposal(
        candidate_id="cand_test_euler5",
        solver="euler",
        solver_params={"num_steps": 5},
        noise_model="L0_static_mismatch",
        is_reference=False,
    )

    base_cand2 = ConditionalKuramotoDynamics(n_oscillators=64, n_conditional_oscillators=2, num_classes=10)
    l0_cand2 = L0StaticMismatchNoise(base_cand2, sigma=0.05)
    cand2_model = functionalize_kuramoto_model(l0_cand2, solver_name="euler", num_steps=5, dt=0.2, use_hop=False)

    cand2_res = node6_phase1_simulate_profile(
        db=tmp_db,
        config=config,
        candidate_id=cand2_id,
        compiled_model=cand2_model,
        target_device="cpu",
        backend_mode="local",
        num_profile_samples=5,
    )

    assert cand2_res["status"] == "SUCCESS"
    rel_err_2 = cand2_res["metrics"]["relative_error"]
    abs_err_2 = cand2_res["metrics"]["absolute_error"]
    abs_p90_2 = cand2_res["metrics"]["absolute_error_p90"]

    assert rel_err_2 > 0.0
    assert abs_err_2 > 0.0
    assert abs_p90_2 > 0.0
    # Euler with 5 steps should diverge more than RK4 with 10 steps
    assert rel_err_2 != rel_err, f"Expected distinct relative errors, got {rel_err_2} vs {rel_err}"
    assert abs_err_2 != abs_err, f"Expected distinct absolute errors, got {abs_err_2} vs {abs_err}"


def test_candidate_fails_when_reference_cache_missing(tmp_db_and_cfg, monkeypatch):
    """Test that candidate fails cleanly with status='FAILED' if golden reference cache is missing."""
    import sys
    n6_module = sys.modules["agentic_ml_analog_sim.nodes.node6_phase1_simulate_profile"]
    monkeypatch.setattr(n6_module, "_CACHED_REF_OUTPUTS", None)

    tmp_db, config = tmp_db_and_cfg
    cand_id = tmp_db.insert_proposal(
        candidate_id="cand_test_no_ref",
        solver="rk4",
        solver_params={"num_steps": 10},
        is_reference=False,
    )
    base_cand = ConditionalKuramotoDynamics(n_oscillators=64, n_conditional_oscillators=2, num_classes=10)
    cand_model = functionalize_kuramoto_model(base_cand, solver_name="rk4", num_steps=10, dt=0.1, use_hop=False)

    res = node6_phase1_simulate_profile(
        db=tmp_db,
        config=config,
        candidate_id=cand_id,
        compiled_model=cand_model,
        target_device="cpu",
        backend_mode="local",
        num_profile_samples=5,
    )

    assert res["status"] == "FAILED"
    assert "reference" in res["error_message"].lower()
    assert res["next_node"] == "node8_phase1_diagnostic_proposal"

