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
    """Verify deterministic normal noise application and seed requirement."""
    phases = torch.zeros(1000)

    # Seed is required
    with pytest.raises(ValueError, match="seed must be provided"):
        apply_normal_perturbations(phases, 0.0, seed=None)

    # With zero epsilon and seed, return exact copy
    p0 = apply_normal_perturbations(phases, 0.0, seed=42)
    assert torch.equal(p0, phases)

    # With non-zero epsilon and seed
    p1 = apply_normal_perturbations(phases, epsilon_d=0.05, seed=42)
    p2 = apply_normal_perturbations(phases, epsilon_d=0.05, seed=42)
    assert torch.equal(p1, p2)
    # Standard deviation should be approximately 0.05
    std = float(p1.std().item())
    assert abs(std - 0.05) < 0.01


def test_sensitivity_lookup_table_interpolation():
    """Test 1D linear interpolation, lower bound underflow, and upper bound overflow."""
    # Table mapping epsilon_d -> e_pert
    # e.g.: (0.0, 0.0), (0.01, 0.02), (0.05, 0.10), (0.10, 0.20)
    entries = [
        (0.0, 0.0),
        (0.01, 0.02),
        (0.05, 0.10),
        (0.10, 0.20),
    ]
    table = SensitivityLookupTable(entries=entries, reference_id="ref_0", experiment_name="exp_test")

    # Lower bound underflow
    assert table.interpolate(-0.01) == "underflow"
    assert table.interpolate(0.0) == 0.0

    # Exact points
    assert table.interpolate(0.02) == pytest.approx(0.01)
    assert table.interpolate(0.10) == pytest.approx(0.05)
    assert table.interpolate(0.20) == pytest.approx(0.10)

    # In-between points (piecewise linear interpolation)
    # Between (0.01, 0.02) and (0.05, 0.10):
    # e_pert = 0.06 is halfway between 0.02 and 0.10 -> epsilon_d = 0.03
    assert table.interpolate(0.06) == pytest.approx(0.03)

    # Upper bound overflow
    assert table.interpolate(0.200001) == "overflow"
    assert table.interpolate(0.30) == "overflow"


def test_calibration_from_reference_snapshots():
    """Verify calibration sweep generates monotonic (epsilon_d, e_pert) pairs and validates inputs."""
    torch.manual_seed(123)
    # Create 5 synthetic reference phase snapshots of dimension 64
    ref_snapshots = [torch.randn(64) for _ in range(5)]
    scales = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

    # Validate num_classes requirement
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        SensitivityLookupTable.calibrate_from_reference_snapshots(
            ref_snapshots=ref_snapshots,
            num_classes=None,
            base_seed=1000,
        )

    # Validate base_seed requirement
    with pytest.raises(ValueError, match="base_seed must be provided"):
        SensitivityLookupTable.calibrate_from_reference_snapshots(
            ref_snapshots=ref_snapshots,
            num_classes=2,
            base_seed=None,
        )

    # Validate empty snapshots requirement
    with pytest.raises(ValueError, match="ref_snapshots is empty"):
        SensitivityLookupTable.calibrate_from_reference_snapshots(
            ref_snapshots=[],
            num_classes=2,
            base_seed=1000,
        )

    table = SensitivityLookupTable.calibrate_from_reference_snapshots(
        ref_snapshots=ref_snapshots,
        num_classes=2,
        scales=scales,
        reference_id="ref_synth",
        experiment_name="synth_exp",
        base_seed=1000,
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


def test_calibration_from_reference_simulation_validation():
    """Verify input validation in calibrate_from_reference_simulation."""
    torch.manual_seed(42)
    ref_inputs = [(torch.randn(1, 32), torch.randn(1, 32, 2)) for _ in range(4)]
    ref_snapshots = [torch.randn(1, 32) for _ in range(4)]
    model = lambda x, d=None: x

    # num_classes validation
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        SensitivityLookupTable.calibrate_from_reference_simulation(
            model=model,
            ref_inputs=ref_inputs,
            ref_snapshots=ref_snapshots,
            num_classes=None,
            base_seed=1000,
        )

    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        SensitivityLookupTable.calibrate_from_reference_simulation(
            model=model,
            ref_inputs=ref_inputs,
            ref_snapshots=ref_snapshots,
            num_classes=0,
            base_seed=1000,
        )

    # base_seed validation
    with pytest.raises(ValueError, match="base_seed must be provided"):
        SensitivityLookupTable.calibrate_from_reference_simulation(
            model=model,
            ref_inputs=ref_inputs,
            ref_snapshots=ref_snapshots,
            num_classes=2,
            base_seed=None,
        )

    # empty snapshots validation
    with pytest.raises(ValueError, match="ref_snapshots is empty"):
        SensitivityLookupTable.calibrate_from_reference_simulation(
            model=model,
            ref_inputs=ref_inputs,
            ref_snapshots=[],
            num_classes=2,
            base_seed=1000,
        )

    # mismatched inputs length validation
    with pytest.raises(ValueError, match="ref_inputs length.*must match ref_snapshots length"):
        SensitivityLookupTable.calibrate_from_reference_simulation(
            model=model,
            ref_inputs=ref_inputs[:2],
            ref_snapshots=ref_snapshots,
            num_classes=2,
            base_seed=1000,
        )


def test_calibration_from_reference_simulation_deterministic_and_monotonic():
    """Verify deterministic reproducibility and monotonic error progression with a mock model."""
    class MockModel(torch.nn.Module):
        def forward(self, x, drive=None):
            if drive is not None:
                return x + 0.01 * torch.sum(drive, dim=-1)
            return x

    torch.manual_seed(42)
    num_samples = 6
    ref_inputs = [(torch.randn(1, 32), torch.randn(1, 32, 2)) for _ in range(num_samples)]
    model = MockModel()
    ref_snapshots = [model(inp[0], inp[1]) for inp in ref_inputs]

    scales = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]

    table1 = SensitivityLookupTable.calibrate_from_reference_simulation(
        model=model,
        ref_inputs=ref_inputs,
        ref_snapshots=ref_snapshots,
        num_classes=2,
        scales=scales,
        reference_id="ref_sim",
        experiment_name="sim_exp",
        base_seed=1000,
    )

    table2 = SensitivityLookupTable.calibrate_from_reference_simulation(
        model=model,
        ref_inputs=ref_inputs,
        ref_snapshots=ref_snapshots,
        num_classes=2,
        scales=scales,
        reference_id="ref_sim",
        experiment_name="sim_exp",
        base_seed=1000,
    )

    # Exact deterministic match
    assert len(table1) == len(scales)
    assert table1.entries == table2.entries

    # Baseline zero perturbation produces zero error
    entries = table1.entries
    assert entries[0] == (0.0, 0.0)

    # Monotonic error curve: error should not decrease as noise magnitude increases
    for i in range(len(entries) - 1):
        assert entries[i][1] <= entries[i + 1][1]
        assert entries[i][0] <= entries[i + 1][0]

    # Substantial error at high noise
    assert entries[-1][1] > 0.1


def test_calibration_from_reference_simulation_3d_trajectory_output():
    """Verify snapshot extraction for 3D trajectory output (batch, steps, n_osc)."""
    class MockTrajectoryModel(torch.nn.Module):
        def forward(self, x, drive=None):
            # Returns 3D trajectory tensor (1, 10, 32)
            return torch.stack([x + i * 0.01 for i in range(10)], dim=1)

    torch.manual_seed(42)
    num_samples = 4
    ref_inputs = [(torch.randn(1, 32), None) for _ in range(num_samples)]
    model = MockTrajectoryModel()
    # Unperturbed snapshot at T=1
    ref_snapshots = [model(inp[0])[:, -1, :].clone() for inp in ref_inputs]

    table = SensitivityLookupTable.calibrate_from_reference_simulation(
        model=model,
        ref_inputs=ref_inputs,
        ref_snapshots=ref_snapshots,
        num_classes=2,
        scales=[0.0, 0.05, 0.2],
        base_seed=1000,
        n_osc=32,
    )

    assert len(table) == 3
    assert table.entries[0] == (0.0, 0.0)
    assert table.entries[1][1] <= table.entries[2][1]
    assert table.entries[2][1] > 0.0
