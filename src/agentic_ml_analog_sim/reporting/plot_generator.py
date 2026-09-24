"""Matplotlib publication-grade plotting engine for analog ML accelerator co-design."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/Ray workers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger(__name__)


def _compute_pareto_front_subset(proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute 2D Pareto front (min latency, min error) among given proposals."""
    candidates_with_objs = []
    for item in proposals:
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
) -> Dict[str, str]:
    """Generate publication-grade Pareto frontier plot (Latency vs. Trajectory Relative Error).

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        filename_prefix: Base filename without extension.
        noise_model: Optional noise model to filter proposals by (e.g. 'none', 'L0_static_mismatch').

    Returns:
        Dictionary with paths to generated 'pdf' and 'png' files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fetch reference and candidate proposals
    ref_proposals = db.get_proposals(is_reference=True) if hasattr(db, "get_proposals") else []
    all_candidates = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []

    if noise_model is not None:
        # Filter candidates by noise_model (case-insensitive)
        candidates = [c for c in all_candidates if str(c.get("noise_model", "")).lower() == noise_model.lower()]
        completed_subset = [c for c in candidates if c.get("status") == "COMPLETED"] + [
            r for r in ref_proposals if r.get("status") == "COMPLETED"
        ]
        pareto_front = _compute_pareto_front_subset(completed_subset)
    else:
        candidates = all_candidates
        pareto_front = db.compute_pareto_front() if hasattr(db, "compute_pareto_front") else []
        if not pareto_front:
            completed_subset = [c for c in candidates if c.get("status") == "COMPLETED"] + [
                r for r in ref_proposals if r.get("status") == "COMPLETED"
            ]
            pareto_front = _compute_pareto_front_subset(completed_subset)

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
            err = c.get("relative_error") if c.get("relative_error") is not None else c.get("absolute_error", 0.0)
            if lat is not None and err is not None:
                other_x.append(lat)
                other_y.append(err)

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
        err = p.get("relative_error") if p.get("relative_error") is not None else p.get("absolute_error", 0.0)
        if lat is not None and err is not None:
            pareto_pts.append((lat, err, p))

    pareto_pts.sort(key=lambda item: item[0])  # Sort by latency ascending

    if pareto_pts:
        px = [pt[0] for pt in pareto_pts]
        py = [pt[1] for pt in pareto_pts]

        # Draw step curve / frontier line
        ax.step(px, py, where="post", color="#10b981", linestyle="-", linewidth=2.0, alpha=0.8, zorder=4)
        ax.scatter(
            px,
            py,
            color="#059669",
            s=120,
            marker="o",
            edgecolors="#064e3b",
            linewidths=1.5,
            label="Pareto Optimal Frontier",
            zorder=5,
        )

        # Annotate top Pareto candidates
        for x, y, p in pareto_pts[:4]:
            if p.get("is_reference"):
                label_text = "RK4 (baseline)"
            else:
                solver = p.get("solver", "solver")
                noise = p.get("noise_model", "none").split("_")[0]
                label_text = f"{solver} ({noise})"
            ax.annotate(
                label_text,
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                fontweight="bold",
                color="#064e3b",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#d1fae5", edgecolor="#10b981", alpha=0.9),
                zorder=6,
            )

    # 3. Plot reference design
    for r in ref_proposals:
        if r.get("status") == "COMPLETED":
            rlat = r.get("latency_ms")
            rerr = r.get("relative_error") if r.get("relative_error") is not None else 0.0
            if rlat is not None:
                ax.scatter(
                    [rlat],
                    [rerr],
                    color="#f59e0b",
                    s=200,
                    marker="*",
                    edgecolors="#b45309",
                    linewidths=1.5,
                    label="Reference Baseline (RK4)",
                    zorder=7,
                )
                break

    ax.set_xlabel("Simulation Latency (ms)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Relative Error (vs. Golden)", fontsize=11, fontweight="bold", labelpad=8)
    if noise_model:
        ax.set_title(f"Analog ML Co-Design: Pareto Frontier ({noise_model})", fontsize=13, fontweight="bold", pad=12)
    else:
        ax.set_title("Analog ML Co-Design: Accuracy vs. Latency Pareto Frontier", fontsize=13, fontweight="bold", pad=12)

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
) -> Dict[str, str]:
    """Generate consolidated multi-noise Pareto frontier comparison plot overlaying all noise frontiers.

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        noise_models: List of noise models to overlay.
        filename_prefix: Base filename without extension.

    Returns:
        Dictionary with paths to generated 'pdf' and 'png' files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ref_proposals = db.get_proposals(is_reference=True) if hasattr(db, "get_proposals") else []
    all_candidates = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []

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
            err = c.get("relative_error") if c.get("relative_error") is not None else c.get("absolute_error", 0.0)
            if lat is not None and err is not None:
                other_x.append(lat)
                other_y.append(err)

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
        nm_pareto = _compute_pareto_front_subset(completed_subset)

        pareto_pts = []
        for p in nm_pareto:
            lat = p.get("latency_ms")
            err = p.get("relative_error") if p.get("relative_error") is not None else p.get("absolute_error", 0.0)
            if lat is not None and err is not None:
                pareto_pts.append((lat, err, p))

        pareto_pts.sort(key=lambda item: item[0])

        color = color_palette[idx % len(color_palette)]
        marker = marker_palette[idx % len(marker_palette)]

        if pareto_pts:
            px = [pt[0] for pt in pareto_pts]
            py = [pt[1] for pt in pareto_pts]
            ax.step(px, py, where="post", color=color, linestyle="--", linewidth=1.8, alpha=0.8, zorder=4)
            ax.scatter(
                px,
                py,
                color=color,
                marker=marker,
                s=100,
                edgecolors="#0f172a",
                linewidths=1.2,
                label=f"Frontier: {nm}",
                zorder=5,
            )

    # Plot reference design
    for r in ref_proposals:
        if r.get("status") == "COMPLETED":
            rlat = r.get("latency_ms")
            rerr = r.get("relative_error") if r.get("relative_error") is not None else 0.0
            if rlat is not None:
                ax.scatter(
                    [rlat],
                    [rerr],
                    color="#f59e0b",
                    s=220,
                    marker="*",
                    edgecolors="#b45309",
                    linewidths=1.5,
                    label="Reference Baseline (RK4)",
                    zorder=7,
                )
                break

    ax.set_xlabel("Simulation Latency (ms)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Relative Error (vs. Golden)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Analog ML Co-Design: Multi-Noise Pareto Frontiers", fontsize=13, fontweight="bold", pad=12)

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
) -> Dict[str, Any]:
    """Generate per-noise Pareto frontier plots and a consolidated multi-noise comparison plot.

    Generates pareto_frontier_{noise_model.lower()}.pdf and .png for each noise model,
    and pareto_frontier_all.pdf and .png overlaying all noise frontiers.

    Args:
        db: ResultDatabase instance to query.
        output_dir: Directory where PDF and PNG files are saved.
        noise_models: List of noise models from config (e.g. ['none', 'L0_static_mismatch']).
        prefix: Base filename prefix without extension.

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
    )
    results["all"] = overlay_dict
    results[f"{prefix}_all"] = overlay_dict

    logger.info("Multi-noise Pareto plots generated for models %s plus consolidated plot.", unique_models)
    return results


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
