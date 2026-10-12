"""Aligned Chordal Root Mean Square (RMS) error computation.

Implements aligned chordal metric on oscillatory state snapshots at T=1:
  Δ_{s,i} = θ_{s,i} - θ*_{s,i}
  R_s = |(1/N) Σ_i e^{i Δ_{s,i}}|
  e_s = sqrt(2(1 - R_s))

Aggregates individual errors across dataset classes:
  For each class c: MS_c = (1 / |S_c|) Σ_{s in S_c} e_s^2
  Pooled mean square: MS_pooled = (1 / C) Σ_c MS_c
  Percentile: 90th percentile across all samples
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def compute_sample_chordal_error(
    measured: Union[torch.Tensor, np.ndarray],
    reference: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute aligned chordal error e_s for a single sample at T=1.

    Args:
        measured: Measured phase vector of oscillators at T=1, shape (N,) or (1, N).
        reference: Reference phase vector of oscillators at T=1, same shape.

    Returns:
        float: Chordal error e_s = sqrt(2 * (1 - R_s)).
    """
    if isinstance(measured, np.ndarray):
        measured = torch.from_numpy(measured)
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference)

    measured = measured.flatten().float()
    reference = reference.flatten().float()

    if measured.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: measured shape {measured.shape} vs reference shape {reference.shape}"
        )

    delta = measured - reference
    # Compute R_s = |(1/N) sum_i e^{i delta_i}|
    cos_mean = torch.mean(torch.cos(delta))
    sin_mean = torch.mean(torch.sin(delta))
    r_s = torch.sqrt(cos_mean.square() + sin_mean.square())

    # Clamp r_s to [0.0, 1.0] for numerical safety
    r_s_clamped = torch.clamp(r_s, 0.0, 1.0)
    e_s = torch.sqrt(2.0 * (1.0 - r_s_clamped))
    return float(e_s.item())


def compute_aligned_chordal_error(
    measured: Union[torch.Tensor, np.ndarray],
    reference: Union[torch.Tensor, np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute alignment magnitude R_s and chordal error e_s across one or more samples.

    Args:
        measured: Tensor of shape (..., N) representing oscillator phases at T=1.
        reference: Tensor of shape (..., N) representing reference oscillator phases at T=1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (R_s, e_s)
    """
    if isinstance(measured, np.ndarray):
        measured = torch.from_numpy(measured)
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference)

    measured = measured.float()
    reference = reference.float()

    if measured.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: measured {measured.shape} vs reference {reference.shape}"
        )

    delta = measured - reference
    cos_mean = torch.mean(torch.cos(delta), dim=-1)
    sin_mean = torch.mean(torch.sin(delta), dim=-1)
    r_s = torch.sqrt(cos_mean.square() + sin_mean.square())
    r_s_clamped = torch.clamp(r_s, 0.0, 1.0)
    e_s = torch.sqrt(2.0 * (1.0 - r_s_clamped))
    return r_s_clamped, e_s


def aggregate_class_errors(
    sample_errors: List[float],
    class_labels: List[int],
    all_classes: Optional[List[int]] = None,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """Aggregate per-sample errors across classes and pooled across dataset classes.

    Args:
        sample_errors: List of per-sample error values e_s.
        class_labels: Corresponding class label c for each sample.
        all_classes: Optional complete list of class IDs (e.g. range(num_classes)).
                     If None, defaults to unique sorted labels in class_labels.

    Returns:
        Dict containing:
          - "pooled_mean_square": Pooled MS across all classes (1/C sum_c MS_c).
          - "pooled_rms": sqrt(pooled_mean_square).
          - "p90": 90th percentile across all sample errors.
          - "class_mean_squares": Dict mapping class_id -> MS_c.
    """
    if not sample_errors:
        return {
            "pooled_mean_square": 0.0,
            "pooled_rms": 0.0,
            "p90": 0.0,
            "class_mean_squares": {},
        }

    if len(sample_errors) != len(class_labels):
        raise ValueError(
            f"Length mismatch: {len(sample_errors)} sample errors vs {len(class_labels)} class labels"
        )

    # Group sample errors by class
    class_to_errors: Dict[int, List[float]] = {}
    for err, cls_id in zip(sample_errors, class_labels):
        class_to_errors.setdefault(int(cls_id), []).append(float(err))

    if all_classes is not None:
        target_classes = sorted(list(set(all_classes)))
    else:
        target_classes = sorted(list(class_to_errors.keys()))

    class_ms: Dict[int, float] = {}
    total_ms = 0.0
    active_class_count = 0

    for cls_id in target_classes:
        errs = class_to_errors.get(cls_id, [])
        if errs:
            # MS_c = (1 / |S_c|) * sum(e_s^2)
            ms_c = float(np.mean([e * e for e in errs]))
            class_ms[cls_id] = ms_c
            total_ms += ms_c
            active_class_count += 1
        else:
            class_ms[cls_id] = 0.0

    num_eval_classes = active_class_count if active_class_count > 0 else 1
    pooled_ms = total_ms / num_eval_classes
    pooled_rms = math.sqrt(max(0.0, pooled_ms))

    # 90th percentile across all samples
    p90 = float(np.percentile(sample_errors, 90))

    return {
        "pooled_mean_square": pooled_ms,
        "pooled_rms": pooled_rms,
        "p90": p90,
        "class_mean_squares": class_ms,
    }
