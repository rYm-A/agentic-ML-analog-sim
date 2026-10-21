"""Unit and integration tests for Un-0 pretrained weights loading and validation."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from agentic_ml_analog_sim.config import SimulationConfig, ReferenceDesignConfig, load_simulation_config
from agentic_ml_analog_sim.utils.checkpoint_loader import (
    resolve_un0_checkpoint,
    load_un0_dynamics_state_dict,
    validate_checkpoint_for_design,
    apply_dynamics_weights,
    get_checkpoint_parameterization,
)
from un0.model import ConditionalKuramotoDynamics


def test_resolve_and_validate_imagenet64():
    """Verify that imagenet64/n6656 checkpoint is discovered and validated."""
    config = {
        "reference_design": {
            "name": "imagenet64/n6656",
            "family": "imagenet64",
            "n_oscillators": 6656,
            "n_conditional_oscillators": 1,
            "num_classes": 1000,
        }
    }
    ckpt_path = validate_checkpoint_for_design(config)
    assert ckpt_path.is_file(), f"Expected valid checkpoint file, got {ckpt_path}"

    state_dict = load_un0_dynamics_state_dict(ckpt_path)
    assert "K" in state_dict
    assert "omega" in state_dict
    assert "omega_cond" in state_dict
    assert "K_cond" in state_dict
    assert "K_drive" in state_dict
    assert state_dict["K"].shape == (6656, 6656)
    assert state_dict["omega_cond"].shape == (1, 1)


def test_resolve_and_validate_cifar10():
    """Verify that cifar10/n1024 checkpoint is discovered and validated."""
    config = {
        "reference_design": {
            "name": "cifar10/n1024",
            "family": "cifar10",
            "n_oscillators": 1024,
            "n_conditional_oscillators": 8,
            "num_classes": 10,
        }
    }
    ckpt_path = validate_checkpoint_for_design(config)
    assert ckpt_path.is_file(), f"Expected valid checkpoint file, got {ckpt_path}"

    state_dict = load_un0_dynamics_state_dict(ckpt_path)
    assert state_dict["K"].shape == (1024, 1024)
    assert state_dict["omega_cond"].shape == (1, 8)


def test_missing_checkpoint_raises_file_not_found():
    """Verify that missing checkpoint raises FileNotFoundError with helpful message."""
    config = {
        "reference_design": {
            "name": "nonexistent_model/n9999",
            "family": "nonexistent_model",
            "n_oscillators": 9999,
            "n_conditional_oscillators": 1,
            "num_classes": 10,
        }
    }
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_checkpoint_for_design(config)
    assert "could not be found" in str(exc_info.value)


def test_shape_mismatch_raises_value_error():
    """Verify that dimension mismatch (e.g. n_conditional_oscillators=8 for imagenet64) raises ValueError."""
    config = {
        "reference_design": {
            "name": "imagenet64/n6656",
            "family": "imagenet64",
            "n_oscillators": 6656,
            "n_conditional_oscillators": 8,  # Checkpoint has 1!
            "num_classes": 1000,
        }
    }
    with pytest.raises(ValueError) as exc_info:
        validate_checkpoint_for_design(config)
    assert "omega_cond shape mismatch" in str(exc_info.value)
    assert "Did you forget to set n_conditional_oscillators: 1" in str(exc_info.value)


def test_apply_dynamics_weights_identity():
    """Verify that two independently instantiated dynamics models share identical parameters after weight application."""
    config = {
        "reference_design": {
            "name": "cifar10/n1024",
            "family": "cifar10",
            "n_oscillators": 1024,
            "n_conditional_oscillators": 8,
            "num_classes": 10,
        }
    }
    ckpt_path = validate_checkpoint_for_design(config)
    state_dict = load_un0_dynamics_state_dict(ckpt_path)

    model1 = ConditionalKuramotoDynamics(n_oscillators=1024, n_conditional_oscillators=8, num_classes=10)
    model2 = ConditionalKuramotoDynamics(n_oscillators=1024, n_conditional_oscillators=8, num_classes=10)

    # Before loading weights, unseeded initialization means they are different
    assert not torch.equal(model1.K, model2.K)

    # Apply pretrained weights to both
    apply_dynamics_weights(model1, state_dict)
    apply_dynamics_weights(model2, state_dict)

    # After loading weights, they must be bitwise identical
    assert torch.equal(model1.K, model2.K)
    assert torch.equal(model1.omega, model2.omega)
    assert torch.equal(model1.omega_cond, model2.omega_cond)
    assert torch.equal(model1.K_cond, model2.K_cond)
    assert torch.equal(model1.K_drive, model2.K_drive)


def test_node3_compiler_rewriter_loads_pretrained_weights(tmp_path):
    """Verify that node3_compiler_rewriter loads identical pretrained weights for reference and candidate."""
    import tempfile
    from agentic_ml_analog_sim.db.database import ResultDatabase
    from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
    from agentic_ml_analog_sim.nodes.node3_phase1_compiler_rewriter import node3_compiler_rewriter

    db_path = tmp_path / "test_weights.db"
    db = ResultDatabase(db_path=db_path)
    config = load_simulation_config("config/simulation_config.yaml")

    # Insert reference proposal
    ref_id = db.insert_proposal(
        candidate_id="ref_test",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )
    # Insert candidate proposal
    cand_id = db.insert_proposal(
        candidate_id="cand_test",
        solver="rk4",
        noise_model="none",
    )

    mock_llm = MockAntigravityLLM(mode="valid")

    res_ref = node3_compiler_rewriter(db=db, config=config, candidate_id=ref_id, target_device="cpu", llm=mock_llm)
    res_cand = node3_compiler_rewriter(db=db, config=config, candidate_id=cand_id, target_device="cpu", llm=mock_llm)

    assert res_ref["status"] == "SUCCESS"
    assert res_cand["status"] == "SUCCESS"

    eager_ref = res_ref["eager_model"]
    eager_cand = res_cand["eager_model"]

    # Verify identical pretrained weights
    assert torch.equal(eager_ref.K, eager_cand.K), "Reference and Candidate have desynchronized K matrices!"
    assert torch.equal(eager_ref.omega, eager_cand.omega)
    assert torch.equal(eager_ref.omega_cond, eager_cand.omega_cond)
    assert torch.equal(eager_ref.K_cond, eager_cand.K_cond)
    assert torch.equal(eager_ref.K_drive, eager_cand.K_drive)

    # Verify that K matches the official checkpoint
    ckpt_path = validate_checkpoint_for_design(config)
    sd = load_un0_dynamics_state_dict(ckpt_path)
    assert torch.equal(eager_ref.K, sd["K"]), "Eager model K does not match official checkpoint K!"


def test_apply_dynamics_weights_explicit_device():
    """Verify that apply_dynamics_weights works cleanly when device is explicitly passed."""
    config = {
        "reference_design": {
            "name": "cifar10/n1024",
            "family": "cifar10",
            "n_oscillators": 1024,
            "n_conditional_oscillators": 8,
            "num_classes": 10,
        }
    }
    ckpt_path = validate_checkpoint_for_design(config)
    state_dict = load_un0_dynamics_state_dict(ckpt_path)

    model = ConditionalKuramotoDynamics(n_oscillators=1024, n_conditional_oscillators=8, num_classes=10)
    # Passing explicit device must not trigger UnboundLocalError
    apply_dynamics_weights(model, state_dict, device="cpu")
    assert torch.equal(model.K, state_dict["K"])


def test_get_checkpoint_parameterization_inference():
    """Verify that parameterization is properly inferred from checkpoint or fallback heuristics."""
    # 1. ImageNet64 checkpoint uses mup
    p_img = get_checkpoint_parameterization(name="imagenet64/n6656", family="imagenet64", n_oscillators=6656)
    assert p_img == "mup"

    # 2. CIFAR10 checkpoint uses standard
    p_cif = get_checkpoint_parameterization(name="cifar10/n1024", family="cifar10", n_oscillators=1024)
    assert p_cif == "standard"

    # 3. Name heuristic fallback
    p_fallback_img = get_checkpoint_parameterization(name="imagenet64/custom", n_oscillators=128)
    assert p_fallback_img == "mup"

    p_fallback_cif = get_checkpoint_parameterization(name="cifar10/custom", n_oscillators=128)
    assert p_fallback_cif == "standard"


def test_node3_compiler_rewriter_parameterization_mup(tmp_path):
    """Verify that node3_compiler_rewriter instantiates mup dynamics for ImageNet designs."""
    from agentic_ml_analog_sim.db.database import ResultDatabase
    from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
    from agentic_ml_analog_sim.nodes.node3_phase1_compiler_rewriter import node3_compiler_rewriter

    db_path = tmp_path / "test_mup.db"
    db = ResultDatabase(db_path=db_path)
    config = load_simulation_config("config/firsts_run_trial_simulation_config.yaml")

    ref_id = db.insert_proposal(
        candidate_id="ref_imagenet_test",
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )
    mock_llm = MockAntigravityLLM(mode="valid")
    res = node3_compiler_rewriter(db=db, config=config, candidate_id=ref_id, target_device="cpu", llm=mock_llm)

    assert res["status"] == "SUCCESS"
    eager_model = res["eager_model"]
    assert hasattr(eager_model, "parameterization")
    assert eager_model.parameterization == "mup"
    assert hasattr(eager_model, "_K_scale")
    expected_k_scale = 6656 ** -0.5
    assert abs(eager_model._K_scale - expected_k_scale) < 1e-6

