"""Unit tests for Phase 2 configuration extensions and experiment limits."""

from agentic_ml_analog_sim.config import (
    ExperimentLimitsConfig,
    BudgetConfig,
    load_simulation_config,
    SimulationConfig,
)


def test_experiment_limits_defaults():
    limits = ExperimentLimitsConfig()
    assert limits.max_experiments == 10
    assert limits.max_iterations == 10
    assert limits.max_node_retries == 3
    assert limits.timeout_seconds_per_eval == 300


def test_budget_alias_compatibility():
    assert BudgetConfig is ExperimentLimitsConfig
    b = BudgetConfig(max_experiments=20)
    assert b.max_experiments == 20


def test_simulation_config_experiment_limits_access():
    cfg = load_simulation_config("config/simulation_config.yaml")
    assert cfg.experiment_limits is not None
    assert cfg.budget is not None
    assert cfg.experiment_limits.max_experiments == 10
    assert cfg.budget.max_experiments == 10


def test_simulation_config_accepts_experiment_limits_key():
    import yaml
    with open("config/simulation_config.yaml", "r") as f:
        data = yaml.safe_load(f)
    # Replace budget with experiment_limits
    data["experiment_limits"] = {
        "max_experiments": 8,
        "max_iterations": 8,
        "max_node_retries": 2,
        "timeout_seconds_per_eval": 150,
    }
    del data["budget"]
    cfg = SimulationConfig.model_validate(data)
    assert cfg.experiment_limits.max_experiments == 8
    assert cfg.budget.max_experiments == 8
