"""Tests for simulation configuration parsing, validation, and field constraints."""

import tempfile
from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError

from agentic_ml_analog_sim.config import (
    SimulationConfig,
    load_simulation_config,
)


def test_load_valid_simulation_config():
    """Verify that default simulation_config.yaml loads and validates all sections."""
    cfg = load_simulation_config("config/simulation_config.yaml")

    # Experiment & Reference Design
    assert cfg.experiment_name == "un0_kuramoto_analog_acceleration"
    assert cfg.reference_design.name == "cifar10/n1024"
    assert cfg.reference_design.family == "cifar10"
    assert cfg.reference_design.n_oscillators == 1024
    assert cfg.reference_design.n_conditional_oscillators == 8
    assert cfg.reference_design.solver == "rk4"
    assert cfg.reference_design.num_steps == 25
    assert cfg.reference_design.integration_time == 1.0
    assert cfg.reference_design.precision == "fp32"
    assert cfg.reference_design.noise_model == "none"
    assert cfg.reference_design.sparsity_ratio == 0.0

    # Solvers
    assert "euler" in cfg.solvers
    assert cfg.solvers["euler"].steps == [1, 2, 5, 10]
    assert "rk4" in cfg.solvers
    assert cfg.solvers["rk4"].steps == [5, 10, 15, 25]
    assert "euler_backward" in cfg.solvers
    assert cfg.solvers["euler_backward"].steps == [2, 5, 10]
    assert "parareal" in cfg.solvers
    assert cfg.solvers["parareal"].steps == [10, 20]
    assert "par_ode" in cfg.solvers
    assert cfg.solvers["par_ode"].steps == [10, 25]

    # Noise Models
    assert "L0_static_mismatch" in cfg.noise_models
    assert "L1_stochastic_parameter_noise" in cfg.noise_models
    assert "L2_functional_interface" in cfg.noise_models
    assert "L3_correlated_drift" in cfg.noise_models

    # Sparsity Configurations
    assert cfg.sparsity_configs.dense.sparsity_ratio == 0.0
    assert cfg.sparsity_configs.threshold_pruned.allowed_sparsity_ratios == [0.2, 0.4, 0.6, 0.8]
    assert cfg.sparsity_configs.degree_bounded.max_degree == [64, 128, 256]

    # Tolerances
    assert cfg.tolerances.compile_check.rtol == 1e-4
    assert cfg.tolerances.compile_check.atol == 1e-4
    assert cfg.tolerances.solver_accuracy.rtol == 0.05
    assert cfg.tolerances.solver_accuracy.atol == 0.05
    assert cfg.tolerances.solver_accuracy.max_fid_degradation == 2.0

    # Budget
    assert cfg.budget.max_iterations == 15
    assert cfg.budget.max_node_retries == 3
    assert cfg.budget.timeout_seconds_per_eval == 300


def test_missing_config_file_raises():
    """Verify FileNotFoundError on non-existent config path."""
    with pytest.raises(FileNotFoundError):
        load_simulation_config("config/non_existent_config.yaml")


def test_invalid_extra_field_forbidden():
    """Verify extra fields are strictly forbidden by ConfigDict(extra='forbid')."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        f.write("experiment_name: test\nextra_unknown_key: true\n")
        f.flush()
        with pytest.raises(ValidationError):
            load_simulation_config(f.name)


def test_reject_missing_required_fields():
    """Verify ValidationError when mandatory fields (like reference_design) are missing."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        f.write("experiment_name: incomplete_experiment\n")
        f.flush()
        with pytest.raises(ValidationError):
            load_simulation_config(f.name)


def test_reject_invalid_field_types():
    """Verify ValidationError when field types violate schema (e.g. string for num_steps)."""
    with open("config/simulation_config.yaml", "r") as f:
        data = yaml.safe_load(f)

    data["reference_design"]["num_steps"] = "invalid_string_steps"

    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        yaml.dump(data, f)
        f.flush()
        with pytest.raises(ValidationError):
            load_simulation_config(f.name)


def test_reject_negative_budget_values():
    """Verify validation constraints on budget fields."""
    with open("config/simulation_config.yaml", "r") as f:
        data = yaml.safe_load(f)

    data["budget"]["max_iterations"] = -5

    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        yaml.dump(data, f)
        f.flush()
        with pytest.raises(ValidationError):
            load_simulation_config(f.name)


def test_solver_accuracy_tolerance_optional_max_fid():
    """Verify max_fid_degradation is optional and defaults to 2.0."""
    from agentic_ml_analog_sim.config import SolverAccuracyTolerance

    # Omitted max_fid_degradation
    tol = SolverAccuracyTolerance(rtol=0.01, atol=0.01)
    assert tol.max_fid_degradation == 2.0

    # Provided max_fid_degradation (backward compatibility with existing YAMLs)
    tol2 = SolverAccuracyTolerance(rtol=0.01, atol=0.01, max_fid_degradation=1.5)
    assert tol2.max_fid_degradation == 1.5

    # Explicit None max_fid_degradation
    tol3 = SolverAccuracyTolerance(rtol=0.01, atol=0.01, max_fid_degradation=None)
    assert tol3.max_fid_degradation is None

