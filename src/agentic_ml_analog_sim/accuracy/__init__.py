"""Accuracy and sensitivity measurement modules for analog ML simulation."""

from agentic_ml_analog_sim.accuracy.chordal_error import (
    aggregate_class_errors,
    compute_aligned_chordal_error,
    compute_sample_chordal_error,
)
from agentic_ml_analog_sim.accuracy.sensitivity_lookup import (
    SensitivityLookupTable,
    apply_normal_perturbations,
    generate_perturbation_scales,
)

__all__ = [
    "compute_aligned_chordal_error",
    "compute_sample_chordal_error",
    "aggregate_class_errors",
    "SensitivityLookupTable",
    "generate_perturbation_scales",
    "apply_normal_perturbations",
]
