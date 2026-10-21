"""Configuration models and loader for the analog ML accelerator simulation loop."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    num_classes: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional number of output classes for classification architectures."
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional path to pretrained weights file. If None, auto-resolved from huggingface cache or model registry."
    )
    require_pretrained_weights: bool = Field(
        default=True,
        description="If True, missing or invalid pretrained weights will raise an error before Phase 1."
    )
    parameterization: Optional[str] = Field(
        default=None,
        description="Optional parameterization ('standard' or 'mup'). If None, inferred from checkpoint configuration."
    )


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


def _parse_tolerance_value(v: Any) -> float:
    """Parse and validate numeric or positive infinite tolerance value."""
    if isinstance(v, str):
        clean = v.strip().lower()
        if clean in (
            ".inf", "+.inf", "inf", "+inf", "infinity", "+infinity",
            "float('inf')", 'float("inf")', "math.inf",
        ):
            return float("inf")
        try:
            v = float(clean)
        except ValueError:
            raise ValueError(f"Invalid numeric tolerance string: {v!r}")
    if isinstance(v, (int, float)):
        if math.isnan(v):
            raise ValueError("Tolerance cannot be NaN")
        if v < 0.0:
            raise ValueError(f"Tolerance must be non-negative or positive infinity, got {v}")
        return float(v)
    raise TypeError(f"Tolerance must be numeric or valid infinity string, got {type(v)}")


class CompileCheckTolerance(BaseModel):
    """Tolerances for fast compile/sanity checks against reference."""
    model_config = ConfigDict(extra="forbid")

    rtol: float = 1e-4
    atol: float = 1e-4

    @field_validator("rtol", "atol", mode="before")
    @classmethod
    def _validate_tolerance(cls, v: Any) -> float:
        return _parse_tolerance_value(v)


class SolverAccuracyTolerance(BaseModel):
    """Accuracy tolerances for solver evaluations and fidelity degradation.

    Attributes:
        rtol: Relative error tolerance vs baseline reference trajectory.
        atol: Absolute error tolerance vs baseline reference trajectory.
        max_fid_degradation: Deprecated: FID is no longer used for evaluation gating.
            Retained as optional for backward compatibility with existing YAML configurations.
    """
    model_config = ConfigDict(extra="forbid")

    rtol: float = 5e-2
    atol: float = 5e-2
    max_fid_degradation: Optional[float] = Field(
        default=2.0,
        description="Deprecated: FID is no longer used for evaluation gating. Retained for backward compatibility.",
    )

    @field_validator("rtol", "atol", mode="before")
    @classmethod
    def _validate_tolerance(cls, v: Any) -> float:
        return _parse_tolerance_value(v)


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


class CounterLimitsConfig(BaseModel):
    """Counter limits for compilation, rewrite, and debugging loops."""
    model_config = ConfigDict(extra="forbid")

    max_counter_1_compilation: int = Field(default=3, gt=0)
    max_counter_2_rewrite: int = Field(default=3, gt=0)
    max_counter_3_debug_gate: int = Field(default=3, gt=0)
    max_counter_4_debug_loop: int = Field(default=5, gt=0)


class AgentModelConfig(BaseModel):
    """Configuration for LLM model and reasoning effort for an agentic node."""
    model_config = ConfigDict(extra="forbid")

    model: str = "gemini-2.5-flash"
    effort: str = Field(default="medium", pattern="^(low|medium|high|max)$")


class AgentEffortsConfig(BaseModel):
    """LLM model and effort configurations for agentic nodes across phases."""
    model_config = ConfigDict(extra="forbid")

    node2_phase0_propose_candidate: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(model="gemini-2.5-pro", effort="high")
    )
    node1_phase1_solver_rewriter: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(model="gemini-2.5-pro", effort="high")
    )
    node2_phase1_noise_rewriter: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(model="gemini-2.5-pro", effort="high")
    )
    node3_phase1_compiler_rewriter: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(model="gemini-2.5-pro", effort="high")
    )
    node8_phase1_diagnostic_proposal: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(model="gemini-2.5-flash", effort="medium")
    )
    node2_phase2_generate_final_report: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(model="gemini-2.5-flash", effort="medium")
    )


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
    counter_limits: CounterLimitsConfig = Field(default_factory=CounterLimitsConfig)
    agent_efforts: AgentEffortsConfig = Field(default_factory=AgentEffortsConfig)
    device: Optional[str] = Field(default="cpu", description="Execution device ('cpu', 'cuda', 'mps')")
    prevalidate: bool = Field(default=True, description="Enable Node 2 candidate pre-validation against config.solvers")

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

            # Extract or fallback counter_limits
            if "counter_limits" not in data:
                for src_key in ("budget", "experiment_limits"):
                    src_val = data.get(src_key)
                    if isinstance(src_val, dict) and "counter_limits" in src_val:
                        data["counter_limits"] = src_val.pop("counter_limits")
                        break

            max_retries = None
            for key in ("budget", "experiment_limits"):
                val = data.get(key)
                if isinstance(val, dict) and "max_node_retries" in val:
                    max_retries = val["max_node_retries"]
                    break
                elif hasattr(val, "max_node_retries"):
                    max_retries = val.max_node_retries
                    break

            if "counter_limits" not in data or data["counter_limits"] is None:
                default_counters = {
                    "max_counter_1_compilation": max_retries if max_retries is not None else 3,
                    "max_counter_2_rewrite": max_retries if max_retries is not None else 3,
                    "max_counter_3_debug_gate": max_retries if max_retries is not None else 3,
                    "max_counter_4_debug_loop": 5,
                }
                data["counter_limits"] = default_counters
            elif isinstance(data["counter_limits"], dict):
                fallback_retry = max_retries if max_retries is not None else 3
                data["counter_limits"].setdefault("max_counter_1_compilation", fallback_retry)
                data["counter_limits"].setdefault("max_counter_2_rewrite", fallback_retry)
                data["counter_limits"].setdefault("max_counter_3_debug_gate", fallback_retry)
                data["counter_limits"].setdefault("max_counter_4_debug_loop", 5)

                # Fallback experiment_limits / budget max_node_retries from counter_limits if not explicitly set
                c2 = data["counter_limits"].get("max_counter_2_rewrite")
                if c2 is not None and max_retries is None:
                    if isinstance(data.get("experiment_limits"), dict):
                        data["experiment_limits"]["max_node_retries"] = c2
                    if isinstance(data.get("budget"), dict):
                        data["budget"]["max_node_retries"] = c2
        return data

    @classmethod
    def load(cls, path: Union[str, Path]) -> SimulationConfig:
        """Load and validate a simulation configuration from a YAML file."""
        return load_simulation_config(path)


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
