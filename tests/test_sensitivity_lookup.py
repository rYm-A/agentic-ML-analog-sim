"""Unit tests for Dynamic Sensitivity Lookup Table and interpolation."""

import tempfile
from pathlib import Path
import pytest
import torch

from agentic_ml_analog_sim.accuracy.sensitivity_lookup import (
    SensitivityLookupTable,
    apply_normal_perturbations,
    generate_perturbation_scales,
)
from agentic_ml_analog_sim.db.database import ResultDatabase


def test_generate_perturbation_scales():
    """Verify log spacing and boundary values."""
    scales = generate_perturbation_scales(min_scale=1e-5, max_scale=1.0, num_points=10, include_zero=True)
    assert len(scales) == 11
    assert scales[0] == 0.0
    assert scales[1] == pytest.approx(1e-5)
    assert scales[-1] == pytest.approx(1.0)
    # Check monotonicity
    for i in range(len(scales) - 1):
        assert scales[i] <= scales[i + 1]


def test_apply_normal_perturbations():
    """Verify deterministic normal noise application."""
    phases = torch.zeros(1000)
    # With zero epsilon, return exact copy
    p0 = apply_normal_perturbations(phases, 0.0)
    assert torch.equal(p0, phases)

    # With non-zero epsilon and seed
    p1 = apply_normal_perturbations(phases, epsilon_d=0.05, seed=42)
    p2 = apply_normal_perturbations(phases, epsilon_d=0.05, seed=42)
    assert torch.equal(p1, p2)
    # Standard deviation should be approximately 0.05
    std = float(p1.std().item())
    assert abs(std - 0.05) < 0.01


def test_sensitivity_lookup_table_interpolation():
    """Test 1D linear interpolation, lower bound clamping, and upper bound extrapolation."""
    # Table mapping epsilon_d -> e_pert
    # e.g.: (0.0, 0.0), (0.01, 0.02), (0.05, 0.10), (0.10, 0.20)
    entries = [
        (0.0, 0.0),
        (0.01, 0.02),
        (0.05, 0.10),
        (0.10, 0.20),
    ]
    table = SensitivityLookupTable(entries=entries, reference_id="ref_0", experiment_name="exp_test")

    # Lower bound clamping
    assert table.interpolate(-0.01) == 0.0
    assert table.interpolate(0.0) == 0.0

    # Exact points
    assert table.interpolate(0.02) == pytest.approx(0.01)
    assert table.interpolate(0.10) == pytest.approx(0.05)

    # In-between points (piecewise linear interpolation)
    # Between (0.01, 0.02) and (0.05, 0.10):
    # e_pert = 0.06 is halfway between 0.02 and 0.10 -> epsilon_d = 0.03
    assert table.interpolate(0.06) == pytest.approx(0.03)

    # Upper bound extrapolation
    # Above 0.20: slope from (0.05, 0.10) to (0.10, 0.20) is (0.10 - 0.05) / (0.20 - 0.10) = 0.05 / 0.10 = 0.5
    # For e_cand = 0.30: eps = 0.10 + 0.5 * (0.30 - 0.20) = 0.15
    assert table.interpolate(0.30) == pytest.approx(0.15)


def test_calibration_from_reference_snapshots():
    """Verify calibration sweep generates monotonic (epsilon_d, e_pert) pairs."""
    torch.manual_seed(123)
    # Create 5 synthetic reference phase snapshots of dimension 64
    ref_snapshots = [torch.randn(64) for _ in range(5)]
    scales = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

    table = SensitivityLookupTable.calibrate_from_reference_snapshots(
        ref_snapshots=ref_snapshots,
        num_classes=2,
        scales=scales,
        reference_id="ref_synth",
        experiment_name="synth_exp",
    )

    assert len(table) == 5
    entries = table.entries
    # Zero epsilon must produce zero error
    assert entries[0] == (0.0, 0.0)
    # Higher noise perturbation should monotonically increase or maintain error
    for i in range(len(entries) - 1):
        assert entries[i][1] <= entries[i + 1][1]


def test_serialization_and_db_persistence():
    """Verify saving to JSON file and SQLite database table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        json_path = Path(tmpdir) / "sensitivity.json"

        db = ResultDatabase(db_path=db_path)
        ref_id = db.insert_proposal(
            candidate_id="ref_persisted",
            solver="rk4",
            is_reference=True,
        )

        entries = [(0.0, 0.0), (0.01, 0.05), (0.1, 0.3)]
        table = SensitivityLookupTable(
            entries=entries,
            reference_id=ref_id,
            experiment_name="persist_test",
        )

        # File save & load
        table.save_to_file(json_path)
        loaded_file = SensitivityLookupTable.load_from_file(json_path)
        assert len(loaded_file) == 3
        assert loaded_file.entries == entries
        assert loaded_file.reference_id == ref_id

        # DB save & load
        count = table.save_to_db(db)
        assert count == 3
        loaded_db = SensitivityLookupTable.load_from_db(db, reference_id=ref_id)
        assert len(loaded_db) == 3
        for (eps_exp, e_exp), (eps_act, e_act) in zip(entries, loaded_db.entries):
            assert eps_act == pytest.approx(eps_exp)
            assert e_act == pytest.approx(e_exp)
