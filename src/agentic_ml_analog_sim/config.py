"""Configuration models and loader for the analog ML accelerator simulation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceDesignConfig(BaseModel):
    """Configuration for the baseline reference design."""
    model_config = ConfigDict(extra="forbid")

    name: str = "cifar10/n1024"
    family: str = "cifar10"
    n_oscillators: int = 1024
    n_conditional_oscillators: int = 8
    solver: str = "rk4"
    num_steps: int = 25
    integration_time: float = 1.0
    precision: str = "fp32"
    noise_model: str = "none"
    sparsity_ratio: float = 0.0


class SolverConfig(BaseModel):
    """Configuration for an individual ODE / dynamical system numerical solver."""
    model_config = ConfigDict(extra="forbid")

    steps: List[int]


class DenseSparsityConfig(BaseModel):
    """Configuration for dense connectivity."""
    model_config = ConfigDict(extra="forbid")

    sparsity_ratio: float = 0.0


class ThresholdPrunedSparsityConfig(BaseModel):
    """Configuration for threshold-pruned weight matrices."""
    model_config = ConfigDict(extra="forbid")

    allowed_sparsity_ratios: List[float]


class DegreeBoundedSparsityConfig(BaseModel):
    """Configuration for degree-bounded oscillator connection topologies."""
    model_config = ConfigDict(extra="forbid")

    max_degree: List[int]


class SparsityConfigs(BaseModel):
    """Aggregated sparsity configuration options."""
    model_config = ConfigDict(extra="forbid")

    dense: DenseSparsityConfig = Field(default_factory=DenseSparsityConfig)
    threshold_pruned: ThresholdPrunedSparsityConfig
    degree_bounded: DegreeBoundedSparsityConfig


class CompileCheckTolerance(BaseModel):
    """Tolerances for fast compile/sanity checks against reference."""
    model_config = ConfigDict(extra="forbid")

    rtol: float = 1e-4
    atol: float = 1e-4


class SolverAccuracyTolerance(BaseModel):
    """Accuracy tolerances for solver evaluations and fidelity degradation."""
    model_config = ConfigDict(extra="forbid")

    rtol: float = 5e-2
    atol: float = 5e-2
    max_fid_degradation: float = 2.0


class TolerancesConfig(BaseModel):
    """Numerical tolerances for simulation validation and gate checks."""
    model_config = ConfigDict(extra="forbid")

    compile_check: CompileCheckTolerance = Field(default_factory=CompileCheckTolerance)
    solver_accuracy: SolverAccuracyTolerance = Field(default_factory=SolverAccuracyTolerance)


class ExperimentLimitsConfig(BaseModel):
    """Execution and evaluation resource limits."""
    model_config = ConfigDict(extra="forbid")

    max_experiments: int = Field(default=10, gt=0, description="Maximal number of completed candidate experiments")
    max_iterations: int = Field(default=10, gt=0, description="Maximal loop iterations")
    max_node_retries: int = Field(default=3, gt=0, description="Maximal retry attempts for rejected proposals")
    timeout_seconds_per_eval: int = Field(default=300, gt=0)


# Backward-compatibility alias for Phase 0
BudgetConfig = ExperimentLimitsConfig


class SimulationConfig(BaseModel):
    """Top-level simulation configuration model."""
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    reference_design: ReferenceDesignConfig
    solvers: Dict[str, SolverConfig]
    noise_models: List[str]
    sparsity_configs: SparsityConfigs
    tolerances: TolerancesConfig
    experiment_limits: Optional[ExperimentLimitsConfig] = None
    budget: Optional[ExperimentLimitsConfig] = None

    @model_validator(mode="before")
    @classmethod
    def resolve_limits(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Synchronize experiment_limits and budget fields
            if "experiment_limits" in data and "budget" not in data:
                data["budget"] = data["experiment_limits"]
            elif "budget" in data and "experiment_limits" not in data:
                data["experiment_limits"] = data["budget"]
            elif "experiment_limits" not in data and "budget" not in data:
                default_limits = ExperimentLimitsConfig().model_dump()
                data["experiment_limits"] = default_limits
                data["budget"] = default_limits
        return data


def load_simulation_config(path: Union[str, Path]) -> SimulationConfig:
    """Load and validate a simulation configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated SimulationConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If parsing or validation fails.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {config_path}: expected mapping, got {type(data)}")

    return SimulationConfig.model_validate(data)
