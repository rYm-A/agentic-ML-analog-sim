"""Unit tests for numerical tolerance parsing, validation, and infinite tolerance support."""

import math
import pytest
import yaml

from agentic_ml_analog_sim.config import (
    CompileCheckTolerance,
    SolverAccuracyTolerance,
    TolerancesConfig,
    SimulationConfig,
)


def test_compile_check_defaults():
    tol = CompileCheckTolerance()
    assert tol.rtol == 1e-4
    assert tol.atol == 1e-4


def test_solver_accuracy_defaults():
    tol = SolverAccuracyTolerance()
    assert tol.rtol == 5e-2
    assert tol.atol == 5e-2


@pytest.mark.parametrize(
    "raw_val",
    [
        float("inf"),
        ".inf",
        "+.inf",
        ".Inf",
        ".INF",
        "inf",
        "+inf",
        "INF",
        "infinity",
        "Infinity",
        "+infinity",
        "float('inf')",
        'float("inf")',
        "math.inf",
    ],
)
def test_infinite_tolerance_parsing(raw_val):
    tol_compile = CompileCheckTolerance(rtol=raw_val, atol=raw_val)
    assert math.isinf(tol_compile.rtol) and tol_compile.rtol > 0
    assert math.isinf(tol_compile.atol) and tol_compile.atol > 0

    tol_solver = SolverAccuracyTolerance(rtol=raw_val, atol=raw_val)
    assert math.isinf(tol_solver.rtol) and tol_solver.rtol > 0
    assert math.isinf(tol_solver.atol) and tol_solver.atol > 0


def test_finite_numeric_tolerance_parsing():
    tol = SolverAccuracyTolerance(rtol="0.1", atol=1e-3)
    assert tol.rtol == 0.1
    assert tol.atol == 0.001


def test_invalid_negative_tolerance():
    with pytest.raises(ValueError, match="non-negative or positive infinity"):
        SolverAccuracyTolerance(rtol=-0.01)

    with pytest.raises(ValueError, match="non-negative or positive infinity"):
        CompileCheckTolerance(atol=-1.0)


def test_invalid_nan_tolerance():
    with pytest.raises(ValueError, match="cannot be NaN"):
        SolverAccuracyTolerance(rtol=float("nan"))


def test_invalid_string_tolerance():
    with pytest.raises(ValueError, match="Invalid numeric tolerance string"):
        SolverAccuracyTolerance(rtol="invalid_tolerance")


def test_yaml_safe_load_with_infinite_tolerance(tmp_path):
    yaml_content = """
tolerances:
  compile_check:
    rtol: .inf
    atol: .inf
  solver_accuracy:
    rtol: .inf
    atol: .inf
"""
    raw_data = yaml.safe_load(yaml_content)
    tolerances_cfg = TolerancesConfig.model_validate(raw_data["tolerances"])
    assert math.isinf(tolerances_cfg.compile_check.rtol)
    assert math.isinf(tolerances_cfg.compile_check.atol)
    assert math.isinf(tolerances_cfg.solver_accuracy.rtol)
    assert math.isinf(tolerances_cfg.solver_accuracy.atol)
