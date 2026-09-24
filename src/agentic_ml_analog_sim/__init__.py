"""Agentic ML Analog Accelerator Simulation Loop.

A framework for multi-fidelity hardware noise modeling, ODE solver exploration,
and Pareto optimization of analog silicon neural network accelerators.
"""

from agentic_ml_analog_sim.config import (
    BudgetConfig,
    ExperimentLimitsConfig,
    CompileCheckTolerance,
    DegreeBoundedSparsityConfig,
    DenseSparsityConfig,
    ReferenceDesignConfig,
    SimulationConfig,
    SolverAccuracyTolerance,
    SolverConfig,
    SparsityConfigs,
    ThresholdPrunedSparsityConfig,
    TolerancesConfig,
    load_simulation_config,
)
from agentic_ml_analog_sim.db import ResultDatabase
from agentic_ml_analog_sim.tools import ResultDBReadOnlyTool, ResultDBReadTool, ResultDBTool
from agentic_ml_analog_sim.phase_2 import Phase2Result, run_phase_2

__version__ = "0.1.0"

__all__ = [
    "SimulationConfig",
    "ReferenceDesignConfig",
    "SolverConfig",
    "DenseSparsityConfig",
    "ThresholdPrunedSparsityConfig",
    "DegreeBoundedSparsityConfig",
    "SparsityConfigs",
    "CompileCheckTolerance",
    "SolverAccuracyTolerance",
    "TolerancesConfig",
    "BudgetConfig",
    "ExperimentLimitsConfig",
    "load_simulation_config",
    "ResultDatabase",
    "ResultDBTool",
    "ResultDBReadOnlyTool",
    "ResultDBReadTool",
    "Phase2Result",
    "run_phase_2",
]

