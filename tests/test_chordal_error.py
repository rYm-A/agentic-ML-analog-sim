"""Unit tests for aligned chordal error and class aggregation."""

import math
import numpy as np
import pytest
import torch

from agentic_ml_analog_sim.accuracy.chordal_error import (
    aggregate_class_errors,
    compute_aligned_chordal_error,
    compute_sample_chordal_error,
)


def test_compute_sample_chordal_error_identical():
    """Identical phase vectors should produce zero chordal error."""
    ref = torch.linspace(0, 2 * math.pi, 64)
    meas = ref.clone()
    err = compute_sample_chordal_error(meas, ref)
    assert abs(err) < 1e-6


def test_compute_sample_chordal_error_global_phase_invariance():
    """A uniform constant phase shift across all oscillators should produce zero aligned error."""
    ref = torch.randn(128)
    # Global phase rotation of pi/3 across all oscillators
    meas = ref + (math.pi / 3.0)
    err = compute_sample_chordal_error(meas, ref)
    # R_s should be 1.0, e_s = sqrt(2(1 - 1)) = 0.0
    assert abs(err) < 1e-5


def test_compute_sample_chordal_error_numpy():
    """Function should handle numpy arrays correctly."""
    ref = np.random.uniform(0, 2 * np.pi, size=(64,))
    meas = ref.copy()
    err = compute_sample_chordal_error(meas, ref)
    assert isinstance(err, float)
    assert abs(err) < 1e-6


def test_compute_sample_chordal_error_shape_mismatch():
    """Shape mismatch between measured and reference raises ValueError."""
    ref = torch.zeros(32)
    meas = torch.zeros(64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_sample_chordal_error(meas, ref)


def test_compute_aligned_chordal_error_batched():
    """Batched aligned chordal error computation across multiple samples."""
    ref = torch.zeros(4, 32)
    meas = torch.zeros(4, 32)
    # Sample 0: identical
    # Sample 1: uniform shift (aligned error = 0)
    meas[1] = meas[1] + 1.5
    # Sample 2: alternating +/- pi/2 phase
    meas[2, 0::2] = math.pi / 2.0
    meas[2, 1::2] = -math.pi / 2.0

    r_s, e_s = compute_aligned_chordal_error(meas, ref)
    assert r_s.shape == (4,)
    assert e_s.shape == (4,)
    assert abs(e_s[0].item()) < 1e-6
    assert abs(e_s[1].item()) < 1e-6
    assert e_s[2].item() > 0.0


def test_aggregate_class_errors():
    """Class aggregation computes pooled mean square and 90th percentile."""
    # 2 classes, 4 samples each
    # Class 0 errors: [0.1, 0.2, 0.3, 0.4] -> MS_0 = mean([0.01, 0.04, 0.09, 0.16]) = 0.075
    # Class 1 errors: [0.2, 0.4, 0.6, 0.8] -> MS_1 = mean([0.04, 0.16, 0.36, 0.64]) = 0.30
    sample_errors = [0.1, 0.2, 0.2, 0.4, 0.3, 0.6, 0.4, 0.8]
    class_labels = [0, 0, 1, 1, 0, 1, 0, 1]

    agg = aggregate_class_errors(sample_errors, class_labels, all_classes=[0, 1])

    expected_ms0 = (0.01 + 0.04 + 0.09 + 0.16) / 4.0
    expected_ms1 = (0.04 + 0.16 + 0.36 + 0.64) / 4.0
    expected_pooled_ms = (expected_ms0 + expected_ms1) / 2.0
    expected_pooled_rms = math.sqrt(expected_pooled_ms)

    assert abs(agg["pooled_mean_square"] - expected_pooled_ms) < 1e-6
    assert abs(agg["pooled_rms"] - expected_pooled_rms) < 1e-6
    assert agg["class_mean_squares"][0] == pytest.approx(expected_ms0)
    assert agg["class_mean_squares"][1] == pytest.approx(expected_ms1)
    assert agg["p90"] == pytest.approx(float(np.percentile(sample_errors, 90)))


def test_aggregate_class_errors_empty():
    """Empty errors list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot aggregate class errors: sample_errors is empty."):
        aggregate_class_errors([], [])
