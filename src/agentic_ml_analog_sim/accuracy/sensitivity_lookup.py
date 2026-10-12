"""Dynamic Sensitivity Lookup Table for analog ML simulation.

Calculates the equivalent noise / sensitivity perturbation epsilon_d
that corresponds to a candidate's aligned chordal RMS error against
a reference perturbation sweep lookup table: e_pert(epsilon_d).

Reference phases at T=1 are perturbed independently per oscillator phase:
  δ_{s,i} ~ N(0, epsilon_d^2)
and pooled mean square error e_pert is tabulated over logarithmic scales
between 1e-5 and 1e0.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from agentic_ml_analog_sim.accuracy.chordal_error import (
    aggregate_class_errors,
    compute_sample_chordal_error,
)

logger = logging.getLogger(__name__)


def generate_perturbation_scales(
    min_scale: float = 1e-5,
    max_scale: float = 1.0,
    num_points: int = 50,
    include_zero: bool = True,
) -> List[float]:
    """Generate logarithmically spaced perturbation scales between min_scale and max_scale.

    Args:
        min_scale: Lower bound (default 1e-5).
        max_scale: Upper bound (default 1.0).
        num_points: Number of log-spaced points.
        include_zero: If True, prepends 0.0 as baseline.

    Returns:
        List[float]: Sorted list of perturbation scale magnitudes.
    """
    scales = np.geomspace(min_scale, max_scale, num=num_points).tolist()
    if include_zero:
        return [0.0] + [float(s) for s in scales]
    return [float(s) for s in scales]


def apply_normal_perturbations(
    phases: Union[torch.Tensor, np.ndarray],
    epsilon_d: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply independent normally distributed perturbations to each oscillator phase.

    δ_{s,i} ~ N(0, epsilon_d^2)

    Args:
        phases: Phase snapshot tensor of oscillators at T=1, shape (..., N).
        epsilon_d: Standard deviation of perturbation noise.
        seed: Optional deterministic random seed.

    Returns:
        torch.Tensor: Perturbed phases tensor of same shape and device.
    """
    if isinstance(phases, np.ndarray):
        t_phases = torch.from_numpy(phases).float()
    else:
        t_phases = phases.float()

    if abs(epsilon_d) < 1e-15:
        return t_phases.clone()

    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(t_phases.shape, generator=generator, dtype=t_phases.dtype)
        if t_phases.is_cuda:
            noise = noise.to(t_phases.device)
    else:
        noise = torch.randn_like(t_phases)

    return t_phases + (abs(float(epsilon_d)) * noise)


class SensitivityLookupTable:
    """Stores calibrated (epsilon_d, e_pert) points and performs 1D linear interpolation.

    Attributes:
        reference_id: Proposal candidate_id for the reference design.
        experiment_name: Experiment namespace identifier.
        entries: List of (epsilon_d, e_pert) tuples.
    """

    def __init__(
        self,
        entries: Optional[List[Tuple[float, float]]] = None,
        reference_id: str = "",
        experiment_name: str = "default",
    ) -> None:
        self.reference_id = str(reference_id)
        self.experiment_name = str(experiment_name)
        self._entries: List[Tuple[float, float]] = []
        if entries:
            self.set_entries(entries)

    def set_entries(self, entries: List[Tuple[float, float]]) -> None:
        """Set and sort entries by e_pert ascending."""
        clean = [(float(eps), float(e)) for eps, e in entries]
        # Sort by e_pert ASC, secondarily by epsilon_d ASC
        self._entries = sorted(clean, key=lambda pair: (pair[1], pair[0]))

    def add_entry(self, epsilon_d: float, e_pert: float) -> None:
        """Add a single entry and maintain sorted order."""
        self._entries.append((float(epsilon_d), float(e_pert)))
        self._entries.sort(key=lambda pair: (pair[1], pair[0]))

    @property
    def entries(self) -> List[Tuple[float, float]]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def interpolate(self, e_cand: float) -> float:
        """Linearly interpolate epsilon_d from measured candidate error e_cand.

        Inverts the mapping e_pert(epsilon_d) to find epsilon_d(e_cand).

        Args:
            e_cand: Candidate absolute chordal error (e.g. pooled mean square or RMS).

        Returns:
            float: Interpolated dynamic sensitivity error epsilon_d.
        """
        if not self._entries:
            return 0.0

        if len(self._entries) == 1:
            return self._entries[0][0]

        target_e = float(e_cand)
        min_e = self._entries[0][1]
        max_e = self._entries[-1][1]

        # Clamping at lower bound
        if target_e <= min_e:
            return self._entries[0][0]

        # Linear extrapolation or clamping at upper bound
        if target_e >= max_e:
            eps_prev, e_prev = self._entries[-2]
            eps_last, e_last = self._entries[-1]
            denom = e_last - e_prev
            if abs(denom) < 1e-12:
                return eps_last
            slope = (eps_last - eps_prev) / denom
            extrapolated = eps_last + slope * (target_e - e_last)
            return max(0.0, float(extrapolated))

        # Standard piecewise linear interpolation
        for i in range(len(self._entries) - 1):
            eps_i, e_i = self._entries[i]
            eps_next, e_next = self._entries[i + 1]
            if e_i <= target_e <= e_next:
                denom = e_next - e_i
                if abs(denom) < 1e-12:
                    return eps_i
                t = (target_e - e_i) / denom
                return float((1.0 - t) * eps_i + t * eps_next)

        return self._entries[-1][0]

    @classmethod
    def calibrate_from_reference_snapshots(
        cls,
        ref_snapshots: List[torch.Tensor],
        num_classes: int = 10,
        scales: Optional[List[float]] = None,
        reference_id: str = "",
        experiment_name: str = "default",
        base_seed: int = 1000,
    ) -> SensitivityLookupTable:
        """Perform reference perturbation sweep and populate lookup table.

        Args:
            ref_snapshots: List of reference phase vectors at T=1, one per sample s.
            num_classes: Number of dataset classes (used for s % num_classes).
            scales: Optional list of epsilon_d perturbation scales.
            reference_id: Reference proposal candidate ID.
            experiment_name: Experiment namespace identifier.
            base_seed: Base deterministic seed for reproducibility.

        Returns:
            SensitivityLookupTable: Fully populated lookup table.
        """
        if scales is None:
            scales = generate_perturbation_scales(min_scale=1e-5, max_scale=1.0, num_points=50, include_zero=True)

        num_samples = len(ref_snapshots)
        class_labels = [s % num_classes for s in range(num_samples)]
        all_classes = list(range(num_classes))

        entries: List[Tuple[float, float]] = []

        for eps_d in scales:
            if abs(eps_d) < 1e-15:
                entries.append((0.0, 0.0))
                continue

            sample_errors: List[float] = []
            for s_idx, ref_s in enumerate(ref_snapshots):
                sample_seed = base_seed + s_idx
                pert_s = apply_normal_perturbations(ref_s, epsilon_d=eps_d, seed=sample_seed)
                e_s = compute_sample_chordal_error(pert_s, ref_s)
                sample_errors.append(e_s)

            agg = aggregate_class_errors(sample_errors, class_labels, all_classes=all_classes)
            # Use pooled_mean_square as error metric e_pert per specification
            e_pert = float(agg["pooled_mean_square"])
            entries.append((float(eps_d), e_pert))

        table = cls(entries=entries, reference_id=reference_id, experiment_name=experiment_name)
        return table

    def to_dict(self) -> Dict[str, Any]:
        """Serialize table to dictionary."""
        return {
            "reference_id": self.reference_id,
            "experiment_name": self.experiment_name,
            "entries": self._entries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SensitivityLookupTable:
        """Deserialize table from dictionary."""
        entries = [(float(p[0]), float(p[1])) for p in data.get("entries", [])]
        return cls(
            entries=entries,
            reference_id=data.get("reference_id", ""),
            experiment_name=data.get("experiment_name", "default"),
        )

    def save_to_file(self, file_path: Union[str, Path]) -> Path:
        """Save lookup table to JSON file."""
        target_path = Path(file_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target_path

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> SensitivityLookupTable:
        """Load lookup table from JSON file."""
        source_path = Path(file_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Sensitivity lookup table file not found: {source_path}")
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_db(self, db: Any) -> int:
        """Save lookup table to SQLite database."""
        if hasattr(db, "record_sensitivity_curve"):
            return db.record_sensitivity_curve(
                reference_id=self.reference_id,
                curve=self._entries,
                experiment_name=self.experiment_name,
            )
        elif hasattr(db, "save_sensitivity_lookup"):
            return db.save_sensitivity_lookup(
                reference_id=self.reference_id,
                curve=self._entries,
                experiment_name=self.experiment_name,
            )
        else:
            logger.warning("Database does not support saving sensitivity curves directly.")
            return 0

    @classmethod
    def load_from_db(
        cls,
        db: Any,
        reference_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> SensitivityLookupTable:
        """Load lookup table from SQLite database."""
        if hasattr(db, "get_sensitivity_curve"):
            rows = db.get_sensitivity_curve(
                reference_id=reference_id,
                experiment_name=experiment_name,
            )
            return cls(
                entries=rows,
                reference_id=reference_id or "",
                experiment_name=experiment_name or "default",
            )
        raise AttributeError("Database does not implement get_sensitivity_curve.")
