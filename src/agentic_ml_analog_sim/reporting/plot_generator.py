"""Matplotlib publication-grade plotting engine for analog ML accelerator co-design."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/Ray workers
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger(__name__)


def _extract_steps(p: Dict[str, Any]) -> Optional[int]:
    """Extract number of steps from proposal dictionary or candidate_id."""
    sp = p.get("solver_params")
    if isinstance(sp, str):
        try:
            parsed = json.loads(sp)
            if isinstance(parsed, dict):
                if "num_steps" in parsed:
                    return int(parsed["num_steps"])
                if "steps" in parsed:
                    return int(parsed["steps"])
        except Exception:
            pass
    elif isinstance(sp, dict):
        if "num_steps" in sp:
            return int(sp["num_steps"])
        if "steps" in sp:
            return int(sp["steps"])
    cid = str(p.get("candidate_id", ""))
    m = re.search(r"_s(\d+)_", cid)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)steps", cid)
    if m:
        return int(m.group(1))
    return None


def _compute_pareto_front_subset(
    proposals: List[Dict[str, Any]],
    metric: str = "relative_error",
) -> List[Dict[str, Any]]:
    """Compute 2D Pareto front (min latency, min error) among given proposals."""
    candidates_with_objs = []
    for item in proposals:
        if metric == "absolute_error":
            err = item.get("absolute_error")
            if err is None:
                err = item.get("relative_error")
        else:
            err = item.get("relative_error")
            if err is None:
                err = item.get("absolute_error") if item.get("absolute_error") is not None else item.get("accuracy_fid")
        cost = item.get("latency_ms")
        if cost is None:
            cost = item.get("wall_clock_s")
        if err is not None and cost is not None:
            candidates_with_objs.append((item, float(err), float(cost)))

    pareto_front = []
    for item_a, err_a, cost_a in candidates_with_objs:
        is_dominated = False
        for item_b, err_b, cost_b in candidates_with_objs:
            if item_a.get("candidate_id") == item_b.get("candidate_id"):
                continue
            if err_b <= err_a and cost_b <= cost_a and (err_b < err_a or cost_b < cost_a):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(item_a)
    return pareto_front


def generate_pareto_plot(
    db: ResultDatabase,
    output_dir: Union[str, Path],
    filename_prefix: str = "pareto_frontier",
    noise_model: Optional[str] = None,
    metric: str = "relative_error",
) -> Dict[str, str]:
    """Generate publication-grade Pareto frontier plot (Latency vs. Trajectory Relative/Absolute Error).

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        filename_prefix: Base filename without extension.
        noise_model: Optional noise model to filter proposals by (e.g. 'none', 'L0_static_mismatch').
        metric: Error metric to evaluate ('relative_error' or 'absolute_error').

    Returns:
        Dictionary with paths to generated 'pdf' and 'png' files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fetch reference and candidate proposals
    ref_proposals = db.get_proposals(is_reference=True) if hasattr(db, "get_proposals") else []
    all_candidates = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []
    ref_ids = {r.get("candidate_id") for r in ref_proposals}

    def _get_err(item: Dict[str, Any]) -> Optional[float]:
        if metric == "absolute_error":
            e = item.get("absolute_error")
            if e is None:
                e = item.get("relative_error")
        else:
            e = item.get("relative_error")
            if e is None:
                e = item.get("absolute_error")
        return float(e) if e is not None else None

    if noise_model is not None:
        # Filter candidates by noise_model (case-insensitive)
        candidates = [c for c in all_candidates if str(c.get("noise_model", "")).lower() == noise_model.lower()]
        completed_subset = [c for c in candidates if c.get("status") == "COMPLETED"] + [
            r for r in ref_proposals if r.get("status") == "COMPLETED"
        ]
        pareto_front = _compute_pareto_front_subset(completed_subset, metric=metric)
    else:
        candidates = all_candidates
        completed_subset = [c for c in candidates if c.get("status") == "COMPLETED"] + [
            r for r in ref_proposals if r.get("status") == "COMPLETED"
        ]
        pareto_front = _compute_pareto_front_subset(completed_subset, metric=metric)

    pareto_cids = {p["candidate_id"] for p in pareto_front}

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Style configuration
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")

    # 1. Plot all non-Pareto evaluated candidates
    other_x, other_y = [], []
    for c in candidates:
        if c.get("status") == "COMPLETED" and c["candidate_id"] not in pareto_cids:
            lat = c.get("latency_ms")
            err = _get_err(c)
            if lat is not None and err is not None:
                other_x.append(lat)
                other_y.append(max(0.0, err))

    if other_x:
        ax.scatter(
            other_x,
            other_y,
            color="#64748b",
            alpha=0.6,
            s=60,
            edgecolors="#334155",
            label="Dominated Designs",
            zorder=3,
        )

    # 2. Plot Pareto-optimal candidates and connect frontier
    pareto_pts: List[Tuple[float, float, Dict[str, Any]]] = []
    for p in pareto_front:
        lat = p.get("latency_ms")
        err = _get_err(p)
        if lat is not None and err is not None:
            pareto_pts.append((lat, max(0.0, err), p))

    pareto_pts.sort(key=lambda item: item[0])  # Sort by latency ascending

    if pareto_pts:
        px = [pt[0] for pt in pareto_pts]
        py = [pt[1] for pt in pareto_pts]

        # Draw step curve / frontier line
        ax.step(px, py, where="post", color="#10b981", linestyle="-", linewidth=2.0, alpha=0.8, zorder=4)

        # Plot candidate Pareto points (EXCLUDING reference baseline to avoid overlapping markers)
        cand_px = [pt[0] for pt in pareto_pts if not pt[2].get("is_reference") and pt[2].get("candidate_id") not in ref_ids]
        cand_py = [pt[1] for pt in pareto_pts if not pt[2].get("is_reference") and pt[2].get("candidate_id") not in ref_ids]

        if cand_px:
            ax.scatter(
                cand_px,
                cand_py,
                color="#059669",
                s=120,
                marker="o",
                edgecolors="#064e3b",
                linewidths=1.5,
                label="Pareto Optimal Frontier",
                zorder=5,
            )

        # Annotate candidate Pareto configurations with step count and anti-collision offsets
        cand_pts = [pt for pt in pareto_pts if not pt[2].get("is_reference") and pt[2].get("candidate_id") not in ref_ids]
        offset_cycle = [(12, 14), (12, -24), (-95, 14), (-95, -24), (15, 26), (15, -36)]
        annotated_positions: List[Tuple[float, float]] = []

        for idx, (x, y, p) in enumerate(cand_pts[:6]):
            solver = p.get("solver", "solver")
            steps = _extract_steps(p)
            label_text = f"{solver} ({steps} steps)" if steps is not None else f"{solver}"

            # Anti-collision offset selection
            collides = any(
                abs(x - prev_x) < 40.0 and abs(y - prev_y) < 0.04
                for prev_x, prev_y in annotated_positions
            )
            chosen_offset = offset_cycle[(idx + (1 if collides else 0)) % len(offset_cycle)]
            annotated_positions.append((x, y))

            ax.annotate(
                label_text,
                (x, y),
                textcoords="offset points",
                xytext=chosen_offset,
                fontsize=8,
                fontweight="bold",
                color="#064e3b",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#d1fae5", edgecolor="#10b981", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=0.8, alpha=0.7),
                zorder=6,
            )

    # 3. Plot reference design strictly ONCE as a yellow circle (no star, no duplicate green circle)
    for r in ref_proposals:
        if r.get("status") == "COMPLETED":
            rlat = r.get("latency_ms")
            rerr = _get_err(r) if _get_err(r) is not None else 0.0
            if rlat is not None:
                ax.scatter(
                    [rlat],
                    [rerr],
                    color="#facc15",
                    s=160,
                    marker="o",
                    edgecolors="#854d0e",
                    linewidths=1.8,
                    label="Reference Baseline (RK4)",
                    zorder=7,
                )
                break

    err_axis_label = "Absolute Error (vs. Golden)" if metric == "absolute_error" else "Relative Error (vs. Golden)"
    ax.set_xlabel("Simulation Latency (ms)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(err_axis_label, fontsize=11, fontweight="bold", labelpad=8)

    if noise_model:
        metric_suffix = " [Absolute Error]" if metric == "absolute_error" else ""
        ax.set_title(f"Analog ML Co-Design: Pareto Frontier ({noise_model}){metric_suffix}", fontsize=13, fontweight="bold", pad=12)
    else:
        metric_title = "Absolute Error" if metric == "absolute_error" else "Accuracy vs. Latency"
        ax.set_title(f"Analog ML Co-Design: {metric_title} Pareto Frontier", fontsize=13, fontweight="bold", pad=12)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    plt.tight_layout()

    pdf_file = out_path / f"{filename_prefix}.pdf"
    png_file = out_path / f"{filename_prefix}.png"

    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    plt.savefig(png_file, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Pareto plot generated: %s and %s", pdf_file, png_file)
    return {"pdf": str(pdf_file), "png": str(png_file)}


def generate_consolidated_pareto_plot(
    db: ResultDatabase,
    output_dir: Union[str, Path],
    noise_models: List[str],
    filename_prefix: str = "pareto_frontier_all",
    metric: str = "relative_error",
) -> Dict[str, str]:
    """Generate consolidated multi-noise Pareto frontier comparison plot overlaying all noise frontiers.

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        noise_models: List of noise models to overlay.
        filename_prefix: Base filename without extension.
        metric: Error metric to evaluate ('relative_error' or 'absolute_error').

    Returns:
        Dictionary with paths to generated 'pdf' and 'png' files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ref_proposals = db.get_proposals(is_reference=True) if hasattr(db, "get_proposals") else []
    all_candidates = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []
    ref_ids = {r.get("candidate_id") for r in ref_proposals}

    def _get_err(item: Dict[str, Any]) -> Optional[float]:
        if metric == "absolute_error":
            e = item.get("absolute_error")
            if e is None:
                e = item.get("relative_error")
        else:
            e = item.get("relative_error")
            if e is None:
                e = item.get("absolute_error")
        return float(e) if e is not None else None

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")

    color_palette = [
        "#10b981",  # Emerald
        "#3b82f6",  # Blue
        "#8b5cf6",  # Violet
        "#ec4899",  # Pink
        "#f97316",  # Orange
        "#06b6d4",  # Cyan
        "#84cc16",  # Lime
    ]
    marker_palette = ["o", "s", "^", "D", "v", "P", "X"]

    # Faint dominated designs
    other_x, other_y = [], []
    for c in all_candidates:
        if c.get("status") == "COMPLETED":
            lat = c.get("latency_ms")
            err = _get_err(c)
            if lat is not None and err is not None:
                other_x.append(lat)
                other_y.append(max(0.0, err))

    if other_x:
        ax.scatter(
            other_x,
            other_y,
            color="#94a3b8",
            alpha=0.35,
            s=40,
            edgecolors="#64748b",
            label="Evaluated Candidates",
            zorder=2,
        )

    # For each noise model, compute and plot its Pareto frontier
    for idx, nm in enumerate(noise_models):
        nm_candidates = [c for c in all_candidates if str(c.get("noise_model", "")).lower() == nm.lower()]
        completed_subset = [c for c in nm_candidates if c.get("status") == "COMPLETED"] + [
            r for r in ref_proposals if r.get("status") == "COMPLETED"
        ]
        nm_pareto = _compute_pareto_front_subset(completed_subset, metric=metric)

        pareto_pts = []
        for p in nm_pareto:
            lat = p.get("latency_ms")
            err = _get_err(p)
            if lat is not None and err is not None:
                pareto_pts.append((lat, max(0.0, err), p))

        pareto_pts.sort(key=lambda item: item[0])

        color = color_palette[idx % len(color_palette)]
        marker = marker_palette[idx % len(marker_palette)]

        if pareto_pts:
            px = [pt[0] for pt in pareto_pts]
            py = [pt[1] for pt in pareto_pts]
            ax.step(px, py, where="post", color=color, linestyle="--", linewidth=1.8, alpha=0.8, zorder=4)

            # Scatter only candidates, excluding golden baseline to prevent multi-color overlapping markers
            cand_px = [pt[0] for pt in pareto_pts if not pt[2].get("is_reference") and pt[2].get("candidate_id") not in ref_ids]
            cand_py = [pt[1] for pt in pareto_pts if not pt[2].get("is_reference") and pt[2].get("candidate_id") not in ref_ids]

            if cand_px:
                ax.scatter(
                    cand_px,
                    cand_py,
                    color=color,
                    marker=marker,
                    s=100,
                    edgecolors="#0f172a",
                    linewidths=1.2,
                    label=f"Frontier: {nm}",
                    zorder=5,
                )

    # Plot reference design strictly ONCE as a yellow circle
    for r in ref_proposals:
        if r.get("status") == "COMPLETED":
            rlat = r.get("latency_ms")
            rerr = _get_err(r) if _get_err(r) is not None else 0.0
            if rlat is not None:
                ax.scatter(
                    [rlat],
                    [rerr],
                    color="#facc15",
                    s=180,
                    marker="o",
                    edgecolors="#854d0e",
                    linewidths=1.8,
                    label="Reference Baseline (RK4)",
                    zorder=7,
                )
                break

    err_axis_label = "Absolute Error (vs. Golden)" if metric == "absolute_error" else "Relative Error (vs. Golden)"
    ax.set_xlabel("Simulation Latency (ms)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(err_axis_label, fontsize=11, fontweight="bold", labelpad=8)
    title_suffix = " (Absolute Error)" if metric == "absolute_error" else ""
    ax.set_title(f"Analog ML Co-Design: Multi-Noise Pareto Frontiers{title_suffix}", fontsize=13, fontweight="bold", pad=12)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    plt.tight_layout()

    pdf_file = out_path / f"{filename_prefix}.pdf"
    png_file = out_path / f"{filename_prefix}.png"

    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    plt.savefig(png_file, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Consolidated Pareto plot generated: %s and %s", pdf_file, png_file)
    return {"pdf": str(pdf_file), "png": str(png_file)}


def generate_multi_noise_pareto_plots(
    db: ResultDatabase,
    output_dir: Union[str, Path],
    noise_models: List[str],
    prefix: str = "pareto_frontier",
    metric: str = "relative_error",
) -> Dict[str, Any]:
    """Generate per-noise Pareto frontier plots and a consolidated multi-noise comparison plot.

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        noise_models: List of noise models from config (e.g. ['none', 'L0_static_mismatch']).
        prefix: Base filename prefix without extension.
        metric: Error metric to evaluate ('relative_error' or 'absolute_error').

    Returns:
        Dictionary mapping noise model names (and 'all') to their generated file paths dicts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seen = set()
    unique_models: List[str] = []
    for nm in noise_models:
        if nm and nm.lower() not in seen:
            seen.add(nm.lower())
            unique_models.append(nm)

    if not unique_models:
        proposals = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []
        distinct = sorted(list({p.get("noise_model") for p in proposals if p.get("noise_model")}))
        unique_models = distinct if distinct else ["none"]

    results: Dict[str, Any] = {}

    # 1. Generate individual plot for each noise model
    for nm in unique_models:
        nm_clean = nm.lower()
        plot_dict = generate_pareto_plot(
            db=db,
            output_dir=out_path,
            filename_prefix=f"{prefix}_{nm_clean}",
            noise_model=nm,
            metric=metric,
        )
        results[nm_clean] = plot_dict
        if nm != nm_clean:
            results[nm] = plot_dict

    # 2. Generate consolidated multi-noise comparison overlay plot
    overlay_dict = generate_consolidated_pareto_plot(
        db=db,
        output_dir=out_path,
        noise_models=unique_models,
        filename_prefix=f"{prefix}_all",
        metric=metric,
    )
    results["all"] = overlay_dict
    results[f"{prefix}_all"] = overlay_dict

    logger.info("Multi-noise Pareto plots generated for models %s plus consolidated plot.", unique_models)
    return results


def generate_frontier_geometry_plot(
    db: ResultDatabase,
    output_dir: Union[str, Path],
    filename_prefix: str = "pareto_clusters",
) -> Dict[str, str]:
    """Generate publication-grade spatial geometry and clustering plot for Pareto frontier (Section II.B).

    Visualizes Cluster A (coarse Backward Euler, 5 steps), Cluster B (medium ParODE/RK4, 10 steps),
    the reference golden baseline, and the unexplored discretization gap.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ref_proposals = db.get_proposals(is_reference=True) if hasattr(db, "get_proposals") else []
    all_candidates = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")

    # 1. Background shading for the unexplored discretization gap (Regime between 10 steps and 60 steps)
    gap_box = patches.Rectangle(
        (0.04, 340.0), 0.35, 1200.0,
        linewidth=1.2, edgecolor="#d97706", facecolor="#fef3c7",
        linestyle="--", alpha=0.35, zorder=1,
    )
    ax.add_patch(gap_box)
    ax.text(
        0.21, 940.0,
        "Unexplored Discretization Gap\n(Regime: $N_s \\in [15, 50]$ steps, $\\Delta t \\in [0.02, 0.06]$)\nMissing High-Throughput & High-Precision Trade-offs",
        ha="center", va="center", fontsize=9, fontweight="semibold", color="#92400e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9c3", edgecolor="#f59e0b", alpha=0.9),
        zorder=2,
    )

    # 2. Plot non-Pareto or other evaluated completed candidates
    other_x, other_y = [], []
    for c in all_candidates:
        if c.get("status") == "COMPLETED":
            lat = c.get("latency_ms")
            err = c.get("relative_error")
            if lat is not None and err is not None:
                other_x.append(float(err))
                other_y.append(float(lat))
    if other_x:
        ax.scatter(other_x, other_y, color="#94a3b8", alpha=0.4, s=50, edgecolors="#64748b", label="Evaluated Candidates", zorder=3)

    # 3. Identify and plot Cluster B (ParODE / RK4, 10 steps) and Cluster A (Euler Backward, 5 steps)
    cluster_b_pts = []
    cluster_a_pts = []
    for c in all_candidates:
        if c.get("status") == "COMPLETED":
            steps = _extract_steps(c)
            lat = c.get("latency_ms")
            err = c.get("relative_error")
            if lat is not None and err is not None:
                if steps == 10:
                    cluster_b_pts.append((float(err), float(lat), c))
                elif steps == 5:
                    cluster_a_pts.append((float(err), float(lat), c))

    if cluster_b_pts:
        bx = [pt[0] for pt in cluster_b_pts]
        by = [pt[1] for pt in cluster_b_pts]
        ax.scatter(bx, by, color="#3b82f6", s=110, marker="s", edgecolors="#1d4ed8", linewidths=1.5, label="Cluster B: 10 Steps (ParODE / RK4)", zorder=5)
        min_bx, max_bx = min(bx), max(bx)
        min_by, max_by = min(by), max(by)
        cb_patch = patches.FancyBboxPatch(
            (min_bx - 0.015, min_by - 25), max(max_bx - min_bx + 0.03, 0.04), max(max_by - min_by + 50, 40),
            boxstyle="round,pad=0.01", edgecolor="#2563eb", facecolor="#dbeafe", alpha=0.45, linestyle="--", linewidth=1.2, zorder=4,
        )
        ax.add_patch(cb_patch)
        ax.annotate(
            "Cluster B: Medium Discretization\nParODE / RK4 ($N_s = 10, \\Delta t = 0.10$)\nLat: ~281-283 ms, Err: ~0.4255",
            xy=(sum(bx) / len(bx), max(by)),
            xytext=(-40, 40), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color="#1e40af",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor="#3b82f6", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.2),
            zorder=6,
        )

    if cluster_a_pts:
        ax_pts = [pt[0] for pt in cluster_a_pts]
        ay_pts = [pt[1] for pt in cluster_a_pts]
        ax.scatter(ax_pts, ay_pts, color="#ef4444", s=110, marker="^", edgecolors="#b91c1c", linewidths=1.5, label="Cluster A: 5 Steps (Euler Backward)", zorder=5)
        min_ax, max_ax = min(ax_pts), max(ax_pts)
        min_ay, max_ay = min(ay_pts), max(ay_pts)
        ca_patch = patches.FancyBboxPatch(
            (min_ax - 0.015, min_ay - 25), max(max_ax - min_ax + 0.03, 0.04), max(max_ay - min_ay + 50, 40),
            boxstyle="round,pad=0.01", edgecolor="#dc2626", facecolor="#fee2e2", alpha=0.45, linestyle="--", linewidth=1.2, zorder=4,
        )
        ax.add_patch(ca_patch)
        ax.annotate(
            "Cluster A: Fast / Coarse Discretization\nEuler Backward ($N_s = 5, \\Delta t = 0.20$)\nLat: ~208-246 ms, Err: ~0.4871",
            xy=(sum(ax_pts) / len(ax_pts), min(ay_pts)),
            xytext=(-60, -50), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#ef4444", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
            zorder=6,
        )

    # 4. Golden Baseline Reference
    for r in ref_proposals:
        if r.get("status") == "COMPLETED":
            rlat = float(r.get("latency_ms", 1703.0))
            rerr = float(r.get("relative_error", 0.0) or 0.0)
            ax.scatter([rerr], [rlat], color="#facc15", s=180, marker="o", edgecolors="#854d0e", linewidths=1.8, label="Reference Baseline (RK4, 60 steps)", zorder=7)
            ax.annotate(
                f"Golden Baseline\n{r.get('candidate_id', 'ref_rk4_60steps')}\nLat: {rlat:.1f} ms, Err: 0.000",
                xy=(rerr, rlat), xytext=(25, -20), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#854d0e",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fefce8", edgecolor="#eab308", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#ca8a04", lw=1.2),
                zorder=8,
            )
            break

    ax.set_xlabel("Relative Error (L2 vs. Golden)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Simulation Latency (ms)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Analog ML Co-Design: Empirical Pareto Geometry & Spatial Clusters", fontsize=13, fontweight="bold", pad=12)

    ax.set_xlim(-0.03, 0.58)
    ax.set_ylim(-30, 1950)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    plt.tight_layout()

    pdf_file = out_path / f"{filename_prefix}.pdf"
    png_file = out_path / f"{filename_prefix}.png"
    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    plt.savefig(png_file, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Frontier geometry plot generated: %s and %s", pdf_file, png_file)
    return {"pdf": str(pdf_file), "png": str(png_file)}


def generate_breakdown_plot(
    db: ResultDatabase,
    output_dir: Union[str, Path],
    filename_prefix: str = "execution_breakdown",
) -> Dict[str, str]:
    """Generate publication-grade execution reliability breakdown plot.

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        filename_prefix: Base filename without extension.

    Returns:
        Dictionary with paths to generated 'pdf' and 'png' files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stats = db.get_execution_statistics()

    stages = [
        "Solver Rewrite",
        "Noise Rewrite",
        "Compiler Rewrite",
        "Compilation",
        "Simulation",
    ]

    succeeded = [
        stats.get("rewrites_solver_succeeded_count", 0),
        stats.get("rewrites_noise_succeeded_count", 0),
        stats.get("rewrites_compiler_succeeded_count", 0),
        stats.get("compilations_succeeded_count", 0),
        stats.get("simulations_succeeded_count", 0),
    ]

    failed = [
        stats.get("rewrites_solver_failed_count", 0),
        stats.get("rewrites_noise_failed_count", 0),
        stats.get("rewrites_compiler_failed_count", 0),
        stats.get("compilations_failed_count", 0),
        stats.get("simulations_failed_count", 0),
    ]

    total_candidates = max(stats.get("total_candidates", 0), 1)
    pending = [max(0, total_candidates - s - f) for s, f in zip(succeeded, failed)]

    fig, ax = plt.subplots(figsize=(8, 5.0), dpi=300)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")

    y_pos = list(range(len(stages)))
    bar_height = 0.55

    # Stacked horizontal bar chart
    p_succ = ax.barh(y_pos, succeeded, height=bar_height, label="Succeeded", color="#10b981", edgecolor="#064e3b")
    p_fail = ax.barh(y_pos, failed, left=succeeded, height=bar_height, label="Failed", color="#ef4444", edgecolor="#7f1d1d")
    left_pending = [s + f for s, f in zip(succeeded, failed)]
    p_pend = ax.barh(y_pos, pending, left=left_pending, height=bar_height, label="Pending / Skipped", color="#94a3b8", edgecolor="#334155")

    # Add numeric labels on bars
    for i, (s, f) in enumerate(zip(succeeded, failed)):
        if s > 0:
            ax.text(s / 2, i, str(s), ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9)
        if f > 0:
            ax.text(s + f / 2, i, str(f), ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # Top-down reading order
    ax.set_xlabel("Number of Candidates", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Execution Reliability Breakdown by Pipeline Stage", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    plt.tight_layout()

    pdf_file = out_path / f"{filename_prefix}.pdf"
    png_file = out_path / f"{filename_prefix}.png"

    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    plt.savefig(png_file, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"Breakdown plot generated: {pdf_file} and {png_file}")
    return {"pdf": str(pdf_file), "png": str(png_file)}


def generate_token_cost_plot(
    db: ResultDatabase,
    output_dir: Union[str, Path],
    filename_prefix: str = "token_cost_summary",
) -> Dict[str, str]:
    """Generate publication-grade LLM token count and cost distribution plot.

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        filename_prefix: Base filename without extension.

    Returns:
        Dictionary with paths to generated 'pdf' and 'png' files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = db.get_token_usage_summary() if hasattr(db, "get_token_usage_summary") else {}
    by_phase = summary.get("by_phase", {})

    phases = ["phase_0", "phase_1", "phase_2"]
    phase_labels = ["Phase 0\n(Exploration)", "Phase 1\n(Sim / Rewrite)", "Phase 2\n(Reporting)"]

    tokens = [by_phase.get(p, {}).get("total_tokens", 0) for p in phases]
    costs = [by_phase.get(p, {}).get("cost_usd", 0.0) for p in phases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # Plot 1: Total Tokens
    ax1.set_facecolor("#f8fafc")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    bars1 = ax1.bar(phase_labels, tokens, color="#3b82f6", edgecolor="#1e3a8a", width=0.5)
    ax1.set_ylabel("Total Tokens Consumed", fontsize=10, fontweight="bold")
    ax1.set_title("LLM Token Consumption by Phase", fontsize=11, fontweight="bold", pad=10)

    max_t = max(max(tokens, default=0), 1)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + max_t * 0.02, f"{int(yval):,}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Plot 2: Cost in USD
    ax2.set_facecolor("#f8fafc")
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    bars2 = ax2.bar(phase_labels, costs, color="#8b5cf6", edgecolor="#4c1d95", width=0.5)
    ax2.set_ylabel("Estimated Cost (USD $)", fontsize=10, fontweight="bold")
    ax2.set_title("Estimated LLM Cost by Phase", fontsize=11, fontweight="bold", pad=10)

    max_c = max(max(costs, default=0.0), 0.01)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + max_c * 0.02, f"${yval:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()

    pdf_file = out_path / f"{filename_prefix}.pdf"
    png_file = out_path / f"{filename_prefix}.png"

    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    plt.savefig(png_file, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Token cost plot generated: %s and %s", pdf_file, png_file)
    return {"pdf": str(pdf_file), "png": str(png_file)}
