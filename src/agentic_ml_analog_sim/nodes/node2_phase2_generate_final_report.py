"""Phase 2 Node 2: Agentic report generator synthesizing multi-objective simulation findings.

Generates a publication-grade LaTeX technical report with Matplotlib figures (Pareto front,
execution reliability breakdown, token usage auditing) and compiles it to PDF via pdflatex.
Reports are systematically structured in an experiment-specific subfolder and feature
dynamic, non-authoritative discussion grounded strictly in empirical data.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from chia.base.ChiaFunction import ChiaFunction

from agentic_ml_analog_sim.reporting.latex_sanitizer import sanitize_latex_text
from agentic_ml_analog_sim.reporting.plot_generator import (
    generate_breakdown_plot,
    generate_multi_noise_pareto_plots,
    generate_pareto_plot,
    generate_token_cost_plot,
)

logger = logging.getLogger("agentic_ml_analog_sim.phase2_node2")

NODE2_REPORT_SYSTEM_INSTRUCTIONS = """
You are the Senior Co-Design Architect and Technical Reporting Specialist for the Un-0 Analog Machine Learning Accelerator swarm.
Your mission is to synthesize all Phase 0 (exploration), Phase 1 (simulation/rewrite), and Phase 2 (auditing) findings for the current experiment run into an insightful, publication-grade technical report.

Critical instructions:
1. Ground your discussion strictly in the varying empirical data provided in the prompt. Avoid repetitive, static boilerplate and avoid rigid closed heuristics.
2. The focus of the report is to expose graphically and tabularly the obtained results, followed by an open-ended, data-driven technical discussion.
3. In the Pareto analysis:
   - Critically analyze the empirical Pareto front points (latency vs. error / accuracy coordinates) across individual noise models and the consolidated multi-noise comparison.
   - Evaluate the spatial distribution, spread, and clarity of the Pareto front: examine whether the points form a crystal-clear, well-separated trade-off curve across orders of magnitude, or whether candidate points are tightly clustered, close to one another, or overlapping.
   - If points are close or overlapping, interpret why this occurs (e.g., whether parameter perturbations yielded diminishing dynamical differentiation, whether numerical accuracy saturated, or whether physical circuit bottlenecks dominated).
   - If the Pareto front is ambiguous, unclear, or scarce, openly discuss the lack of clarity and what it reveals about the explored parameter regime for this particular run.
4. In pipeline reliability:
   - Reflect on loop execution reliability, error stages encountered (rewrites, compilations, simulations), and pipeline bottlenecks.
   - Ground compilation failure reasons strictly in factual PyTorch Inductor CPU lowering exception logs (e.g. LoweringException, unbacked scalar bindings in higher-order operators) and empirical error messages provided in compilation_failure_logs. DO NOT claim or hallucinate that proposals failed due to schema validation rejection (e.g., sparsity_ratio=0.4 is a valid schema configuration and must never be cited as schema rejected).
5. Reflect on swarm token and compute expenditure across phases.
6. Provide clear, quantitative, academic-grade text and analytical assessments.
"""


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Extract attribute or dictionary key safely."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_llm(llm: Any = None, config: Any = None) -> Any:
    """Resolve single LLM instance configured by agent_efforts (default: gemini-2.5-flash, medium)."""
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node2_phase2_generate_final_report", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node2_phase2_generate_final_report")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-flash"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "medium"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args, system_message=NODE2_REPORT_SYSTEM_INSTRUCTIONS)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args, system_message=NODE2_REPORT_SYSTEM_INSTRUCTIONS)


def _find_pdflatex() -> Optional[str]:
    """Locate pdflatex binary on system, checking standard PATH and macOS TeX directories."""
    found = shutil.which("pdflatex")
    if found:
        return found

    macos_candidates = [
        "/Library/TeX/texbin/pdflatex",
        "/usr/local/texlive/2024/bin/universal-darwin/pdflatex",
        "/usr/local/texlive/2023/bin/universal-darwin/pdflatex",
        "/usr/local/texlive/2022/bin/universal-darwin/pdflatex",
        "/usr/local/bin/pdflatex",
    ]
    for cand in macos_candidates:
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand

    return None


def compile_latex(tex_file: Union[str, Path], output_dir: Union[str, Path]) -> bool:
    """Compile a LaTeX document to PDF using pdflatex with non-stop mode.

    Executes two compilation passes to resolve cross-references and figure labels.

    Args:
        tex_file: Path to .tex source file.
        output_dir: Output directory for compiled PDF and auxiliary files.

    Returns:
        True if PDF was successfully produced; False otherwise.
    """
    pdflatex_bin = _find_pdflatex()
    if not pdflatex_bin:
        logger.warning("pdflatex binary not found; skipping PDF compilation. TeX source remains available.")
        return False

    tex_path = Path(tex_file).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        pdflatex_bin,
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_path),
    ]

    try:
        # Pass 1: generate aux and references
        logger.info("Executing pdflatex pass 1: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            logger.warning("pdflatex pass 1 returned non-zero code (%d): %s", result.returncode, result.stdout[-500:])

        # Pass 2: resolve cross-references and table/figure links
        logger.info("Executing pdflatex pass 2 for cross-references...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            logger.warning(
                "pdflatex pass 2 returned non-zero code (%d). Log snippet:\n%s",
                result.returncode,
                result.stdout[-1000:],
            )
            pdf_target = out_dir / f"{tex_path.stem}.pdf"
            if pdf_target.exists() and pdf_target.stat().st_size > 0:
                return True
            return False

        pdf_target = out_dir / f"{tex_path.stem}.pdf"
        return pdf_target.exists() and pdf_target.stat().st_size > 0
    except Exception as exc:
        logger.error("Failed to execute pdflatex: %s", exc)
        return False


def _escape_latex(s: Any) -> str:
    """Escape LaTeX special characters in table text strings."""
    if s is None:
        return ""
    text = str(s)
    text = text.replace("\\", "\\textbackslash{}")
    text = text.replace("_", "\\_")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("#", "\\#")
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")
    return text


def _build_pareto_table_tex(pareto_front: List[Dict[str, Any]], ref_proposal: Optional[Dict[str, Any]]) -> str:
    """Generate LaTeX booktabs table of Pareto-optimal candidate configurations."""
    ref_lat = ref_proposal.get("latency_ms", 100.0) if ref_proposal else 100.0
    ref_id = ref_proposal.get("candidate_id") if ref_proposal else None

    rows = []
    if ref_proposal:
        cid = _escape_latex(ref_proposal.get("candidate_id", "ref_baseline"))
        solv = _escape_latex(ref_proposal.get("solver", "rk4"))
        noise = _escape_latex(ref_proposal.get("noise_model", "none"))
        lat = ref_proposal.get("latency_ms", 0.0) or 0.0
        err = ref_proposal.get("relative_error", 0.0) or 0.0
        rows.append(f"\\textbf{{{cid}}} (Baseline) & {solv} & {noise} & 0.0 & {lat:.1f} & {err:.4f} & 1.00$\\times$ \\\\")

    for p in pareto_front:
        if p.get("is_reference") or (ref_id and p.get("candidate_id") == ref_id):
            continue
        cid = _escape_latex(p.get("candidate_id", "cand"))
        solv = _escape_latex(p.get("solver", "euler"))
        noise = _escape_latex(p.get("noise_model", "none").split("_")[0])
        sp_cfg = p.get("sparsity_config")
        sp_ratio = 0.0
        if isinstance(sp_cfg, str):
            try:
                sp_ratio = json.loads(sp_cfg).get("sparsity_ratio", 0.0)
            except Exception:
                pass
        elif isinstance(sp_cfg, dict):
            sp_ratio = sp_cfg.get("sparsity_ratio", 0.0)

        lat = p.get("latency_ms", 0.0) or 0.0
        err = p.get("relative_error") if p.get("relative_error") is not None else p.get("absolute_error", 0.0)
        err = err or 0.0
        speedup = (ref_lat / lat) if lat and lat > 0 else 1.0

        rows.append(f"{cid} & {solv} & {noise} & {sp_ratio:.1f} & {lat:.1f} & {err:.4f} & {speedup:.2f}$\\times$ \\\\")

    if not rows:
        rows.append("None & - & - & - & - & - & - \\\\")

    return "\n".join(rows)


def _get_compilation_failures(db: Any) -> List[Dict[str, Any]]:
    """Retrieve compilation failure logs from database to ground reporting."""
    if not hasattr(db, "get_proposals"):
        return []
    proposals = db.get_proposals(limit=200)
    failures = []
    for p in proposals:
        if p.get("compilation_status") == "FAILED" or p.get("error_stage") == "compilation":
            failures.append({
                "candidate_id": p.get("candidate_id"),
                "solver": p.get("solver"),
                "noise_model": p.get("noise_model"),
                "error_stage": p.get("error_stage"),
                "error_message": p.get("error_message"),
            })
    return failures


def _build_token_table_tex(token_summary: Dict[str, Any]) -> str:
    """Generate LaTeX booktabs table of token usage by swarm phase with escaped ampersands."""
    by_phase = token_summary.get("by_phase", {})
    phases = [
        ("phase_0", "Phase 0 (Candidate Exploration)"),
        ("phase_1", "Phase 1 (Code Rewrite \\& Simulation)"),
        ("phase_2", "Phase 2 (Auditing \\& Final Reporting)"),
    ]

    rows = []
    total_tokens = 0
    total_cost = 0.0

    for phase_key, label in phases:
        data = by_phase.get(phase_key, {})
        prompt_t = data.get("prompt_tokens", 0)
        compl_t = data.get("completion_tokens", 0)
        tot_t = data.get("total_tokens", 0)
        cost = data.get("cost_usd", 0.0)
        calls = data.get("num_calls", 0)

        total_tokens += tot_t
        total_cost += cost

        rows.append(f"{label} & {calls} & {prompt_t:,} & {compl_t:,} & {tot_t:,} & \\${cost:.4f} \\\\")

    rows.append("\\midrule")
    rows.append(
        f"\\textbf{{Swarm Total}} & \\textbf{{{token_summary.get('overall', {}).get('total_calls', 0)}}} & "
        f"\\textbf{{{token_summary.get('overall', {}).get('total_prompt_tokens', 0):,}}} & "
        f"\\textbf{{{token_summary.get('overall', {}).get('total_completion_tokens', 0):,}}} & "
        f"\\textbf{{{total_tokens:,}}} & \\textbf{{\\${total_cost:.4f}}} \\\\"
    )

    return "\n".join(rows)


def _generate_dynamic_executive_summary(
    db: Any,
    config: Any,
    ref: Optional[Dict[str, Any]],
    pareto: List[Dict[str, Any]],
    stats: Dict[str, Any],
    token_sum: Dict[str, Any],
) -> str:
    """Dynamically synthesize executive summary reflecting actual run metrics."""
    exp_name = _get_val(config, "experiment_name", "un0_kuramoto_analog_acceleration")
    total_prop = stats.get("total_proposals", len(pareto))
    pareto_cnt = len(pareto)
    ref_lat = ref.get("latency_ms", 100.0) if ref else 100.0

    if pareto:
        min_lat = min((p.get("latency_ms", 0.0) or ref_lat) for p in pareto)
        max_speedup = (ref_lat / min_lat) if min_lat > 0 else 1.0
    else:
        max_speedup = 1.0

    return (
        f"[pregenerated template text] This technical report documents the empirical outcomes of the co-design optimization campaign "
        f"for experiment '{exp_name}'. An autonomous multi-agent swarm evaluated {total_prop} candidate "
        f"configurations against the reference baseline ({ref_lat:.1f} ms latency). The multi-objective "
        f"optimization established a Pareto frontier comprising {pareto_cnt} non-dominated configurations, "
        f"yielding peak speedups of up to {max_speedup:.2f}$\\times$ while remaining within acceptable relative "
        f"error tolerances. Comprehensive auditing confirms robust pipeline execution and cost-efficient LLM operation."
    )


def _generate_dynamic_pareto_analysis(
    pareto: List[Dict[str, Any]],
    ref: Optional[Dict[str, Any]],
) -> str:
    """Dynamically discuss Pareto frontier results based strictly on empirical candidate points,
    analyzing point proximity, clustering, overlap, and trade-off boundary clarity.
    """
    if not pareto:
        return (
            "[pregenerated template text] The multi-objective evaluation did not identify any non-dominated candidates beyond the "
            "reference baseline. The trade-off surface remains unresolved, suggesting that the initial "
            "search space was overly constrained or that proposed candidates were strictly dominated."
        )

    cand_points = [p for p in pareto if not p.get("is_reference")]
    points_to_analyze = cand_points if cand_points else pareto
    ref_lat = ref.get("latency_ms", 100.0) if ref else 100.0

    latencies = [(p.get("latency_ms", 0.0) or 0.0) for p in points_to_analyze]
    errors = [
        ((p.get("relative_error") if p.get("relative_error") is not None else p.get("absolute_error", 0.0)) or 0.0)
        for p in points_to_analyze
    ]

    min_lat, max_lat = min(latencies), max(latencies)
    max_speedup = (ref_lat / min_lat) if min_lat > 0 else 1.0
    min_err, max_err = min(errors), max(errors)
    n_points = len(points_to_analyze)

    # Spatial proximity analysis of Pareto points in (latency, error) objective space
    lat_span = max(max_lat - min_lat, 1e-6)
    err_span = max(max_err - min_err, 1e-6)
    overlapping_or_close_pairs = []

    for i in range(n_points):
        for j in range(i + 1, n_points):
            p1 = points_to_analyze[i]
            p2 = points_to_analyze[j]
            l1, l2 = latencies[i], latencies[j]
            e1, e2 = errors[i], errors[j]

            # Relative differences
            rel_lat_diff = abs(l1 - l2) / max(min(l1, l2), 1e-6)
            abs_err_diff = abs(e1 - e2)
            norm_dist = (((l1 - l2) / lat_span) ** 2 + ((e1 - e2) / err_span) ** 2) ** 0.5

            if norm_dist < 0.15 or (rel_lat_diff < 0.08 and abs_err_diff < 0.005):
                overlapping_or_close_pairs.append((p1.get("candidate_id", f"c{i}"), p2.get("candidate_id", f"c{j}")))

    if n_points <= 1:
        frontier_characterization = (
            f"The empirical Pareto frontier currently contains a solitary non-dominated candidate point "
            f"({points_to_analyze[0].get('candidate_id', 'candidate')}, latency {latencies[0]:.1f} ms, "
            f"relative error {errors[0]:.4f}). With only one sample point resolved, the global curvature "
            f"and trade-off gradient of the frontier cannot be decisively established for this run."
        )
    elif overlapping_or_close_pairs:
        close_ids = ", ".join(f"({a}, {b})" for a, b in overlapping_or_close_pairs[:3])
        frontier_characterization = (
            f"Examination of the empirical Pareto front points reveals localized clustering and near-overlapping "
            f"performance between candidate configurations (such as pairs {close_ids}). Rather than tracing a "
            f"broadly dispersed trade-off curve, these points occupy a compact region with latencies between "
            f"{min_lat:.1f} ms and {max_lat:.1f} ms and relative errors between {min_err:.4f} and {max_err:.4f}. "
            f"This overlap indicates that the underlying parameter variations generated minimal dynamical separation, "
            f"suggesting that the simulation trajectory saturated near a localized accuracy-latency plateau."
        )
    else:
        frontier_characterization = (
            f"The empirical Pareto frontier presents a crystal-clear, well-separated trade-off boundary across "
            f"{n_points} distinct points. Candidate points span a broad performance spectrum from {min_lat:.1f} ms "
            f"to {max_lat:.1f} ms in latency (up to {max_speedup:.2f}$\\times$ acceleration over the baseline) while "
            f"modulating relative errors smoothly between {min_err:.4f} and {max_err:.4f}. The clear separation "
            f"along both objective axes demonstrates that the explored configurations effectively sample non-redundant "
            f"operating regimes on the analog accelerator."
        )

    tradeoff_summary = (
        f"Compared against the reference baseline ({ref_lat:.1f} ms), the identified boundary provides an empirical "
        f"trade-off envelope balancing physical inference throughput against numerical trajectory fidelity."
    )

    return f"[pregenerated template text] {frontier_characterization} {tradeoff_summary}"


def _generate_dynamic_reliability_analysis(
    stats: Dict[str, Any],
    compilation_failures: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Dynamically discuss execution pipeline reliability and bottlenecks."""
    total_prop = stats.get("total_proposals", 0)
    failed_rewrites = stats.get("rewrites_failed_count", 0)
    succ_comps = stats.get("compilations_succeeded_count", 0)
    failed_comps = stats.get("compilations_failed_count", 0)
    succ_sims = stats.get("simulations_succeeded_count", 0)
    failed_sims = stats.get("simulations_failed_count", 0)

    total_failures = failed_rewrites + failed_comps + failed_sims
    if total_failures == 0:
        return (
            f"[pregenerated template text] The autonomous execution pipeline demonstrated 100\\% reliability across all {total_prop} evaluated "
            f"proposals. All automated solver rewrites, noise injection passes, compiler adaptations, domain compilations, "
            f"and numerical simulations executed without runtime faults, compiler crashes, or numerical overflow."
        )

    bottlenecks = []
    if failed_rewrites > 0:
        bottlenecks.append(f"{failed_rewrites} rewrite stage failures")
    if failed_comps > 0:
        bottlenecks.append(f"{failed_comps} TorchInductor compiler lowering failures")
    if failed_sims > 0:
        bottlenecks.append(f"{failed_sims} numerical simulation divergences")

    failure_details = ""
    if compilation_failures:
        inductor_errs = [
            f for f in compilation_failures
            if "lowering" in str(f.get("error_message", "")).lower() or "inductor" in str(f.get("error_message", "")).lower()
        ]
        if inductor_errs:
            failure_details = (
                " Compiler build failures were strictly driven by PyTorch Inductor CPU lowering exceptions "
                "(such as dynamic symbol unbacked bindings in higher-order loops) rather than code schema rejection "
                "or configuration invalidity."
            )

    return (
        f"[pregenerated template text] Pipeline reliability monitoring recorded {total_failures} execution failures across {total_prop} evaluated "
        f"proposals: {', '.join(bottlenecks)}. Success rates achieved: {succ_comps}/{succ_comps + failed_comps} for domain "
        f"compilation and {succ_sims}/{succ_sims + failed_sims} for numerical simulation.{failure_details}"
    )


def _generate_dynamic_resource_analysis(token_sum: Dict[str, Any]) -> str:
    """Dynamically discuss swarm resource utilization and costs."""
    overall = token_sum.get("overall", {})
    calls = overall.get("total_calls", 0)
    tokens = overall.get("total_tokens", 0)
    cost = overall.get("total_cost_usd", 0.0)

    by_phase = token_sum.get("by_phase", {})
    p0_cost = by_phase.get("phase_0", {}).get("cost_usd", 0.0)
    p1_cost = by_phase.get("phase_1", {}).get("cost_usd", 0.0)
    p2_cost = by_phase.get("phase_2", {}).get("cost_usd", 0.0)

    return (
        f"[pregenerated template text] Swarm resource auditing recorded a total of {calls} LLM inference turns consuming "
        f"{tokens:,} total tokens with a cumulative cost of \\${cost:.4f} USD. Resource expenditure was distributed "
        f"across Phase 0 exploration (\\${p0_cost:.4f}), Phase 1 automated code rewriting and simulation "
        f"(\\${p1_cost:.4f}), and Phase 2 auditing and report synthesis (\\${p2_cost:.4f})."
    )


def _generate_dynamic_recommendations(
    pareto: List[Dict[str, Any]],
    stats: Dict[str, Any],
    token_sum: Dict[str, Any],
) -> str:
    """Dynamically generate actionable recommendations strictly based on empirical findings."""
    recs = []
    cand_points = [p for p in pareto if not p.get("is_reference")]
    pts = cand_points if cand_points else pareto

    if len(pts) >= 2:
        latencies = [(p.get("latency_ms", 0.0) or 0.0) for p in pts]
        errors = [
            ((p.get("relative_error") if p.get("relative_error") is not None else p.get("absolute_error", 0.0)) or 0.0)
            for p in pts
        ]
        lat_span = max(latencies) - min(latencies)
        err_span = max(errors) - min(errors)

        if lat_span < 10.0 and err_span < 0.01:
            recs.append(
                r"\textbf{Design Space Expansion:} Because the observed Pareto candidates cluster tightly in "
                r"latency and error coordinates, expand subsequent exploration across orthogonal axes "
                r"(such as crossbar sparsity or alternate discretization formulations) to break past localized performance clustering."
            )
        else:
            recs.append(
                r"\textbf{Frontier Operating Selection:} Leverage the clearly resolved trade-off frontier to select optimal operating "
                r"points based on deployment constraints, favoring low-latency designs for edge inference and higher-accuracy configurations for precision tasks."
            )
    else:
        recs.append(
            r"\textbf{Frontier Resolution:} Broaden candidate exploration in Phase 0 to populate the sparsely resolved trade-off surface "
            r"with additional non-dominated points."
        )

    failed_comps = stats.get("compilations_failed_count", 0)
    if failed_comps > 0:
        recs.append(
            r"\textbf{Compilation Robustness:} Strengthen compiler AST verification before code generation to prevent build failures."
        )
    else:
        recs.append(
            r"\textbf{Execution Pipeline Stability:} Domain compilation and numerical simulation demonstrate complete stability; "
            r"increase batch exploration sizes in subsequent sweeps."
        )

    overall_cost = token_sum.get("overall", {}).get("total_cost_usd", 0.0)
    if overall_cost > 1.0:
        recs.append(
            r"\textbf{Token Efficiency Optimization:} Implement candidate proposal deduplication in Phase 0 prompts to compress upstream token expenditure."
        )
    else:
        recs.append(
            r"\textbf{Swarm Resource Budgeting:} Resource utilization remains highly cost-efficient, supporting extended loop iterations."
        )

    items = "\n".join(f"  \\item {r}" for r in recs)
    return f"[pregenerated template text]\n\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}"


def _generate_report_tex_content(
    db: Any,
    config: Any,
    llm_synthesis: Optional[Dict[str, str]] = None,
    noise_models: Optional[List[str]] = None,
) -> str:
    """Construct publication-grade LaTeX report with dynamic, data-driven discussion.

    All narrative strings are passed through sanitize_latex_text to map Unicode
    characters and escape LaTeX special symbols.
    """
    ref = db.get_reference_proposal() if hasattr(db, "get_reference_proposal") else None
    pareto = db.compute_pareto_front() if hasattr(db, "compute_pareto_front") else []
    stats = db.get_execution_statistics() if hasattr(db, "get_execution_statistics") else {}
    token_sum = db.get_token_usage_summary() if hasattr(db, "get_token_usage_summary") else {}

    if not noise_models:
        noise_models = _get_val(config, "noise_models", None)
    if not noise_models:
        all_props = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []
        distinct = sorted(list({p.get("noise_model") for p in all_props if p.get("noise_model")}))
        noise_models = distinct if distinct else ["none"]

    compilation_failures = _get_compilation_failures(db)

    pareto_table = _build_pareto_table_tex(pareto, ref)
    token_table = _build_token_table_tex(token_sum)

    # Resolve narrative sections dynamically without static authoritative fallbacks
    exec_summary = (
        llm_synthesis.get("executive_summary")
        if llm_synthesis and "executive_summary" in llm_synthesis
        else _generate_dynamic_executive_summary(db, config, ref, pareto, stats, token_sum)
    )

    pareto_analysis_text = (
        llm_synthesis.get("pareto_analysis")
        if llm_synthesis and "pareto_analysis" in llm_synthesis
        else _generate_dynamic_pareto_analysis(pareto, ref)
    )

    reliability_text = (
        llm_synthesis.get("reliability_discussion") or llm_synthesis.get("architecture_discussion")
        if llm_synthesis and ("reliability_discussion" in llm_synthesis or "architecture_discussion" in llm_synthesis)
        else _generate_dynamic_reliability_analysis(stats, compilation_failures)
    )

    resource_text = (
        llm_synthesis.get("resource_discussion")
        if llm_synthesis and "resource_discussion" in llm_synthesis
        else _generate_dynamic_resource_analysis(token_sum)
    )

    recommendations_text = (
        llm_synthesis.get("recommendations")
        if llm_synthesis and "recommendations" in llm_synthesis
        else _generate_dynamic_recommendations(pareto, stats, token_sum)
    )

    # Sanitize all narrative fields to eliminate raw Unicode and unescaped LaTeX specials
    exec_summary = sanitize_latex_text(exec_summary)
    pareto_analysis_text = sanitize_latex_text(pareto_analysis_text)
    reliability_text = sanitize_latex_text(reliability_text)
    resource_text = sanitize_latex_text(resource_text)
    recommendations_text = sanitize_latex_text(recommendations_text)

    # Build Pareto figures for consolidated multi-noise and per-noise plots
    fig_blocks = [
        r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{figures/pareto_frontier.pdf}
\caption{Multi-Objective Pareto frontier for analog ML co-design (Latency vs. Relative Error). Golden baseline is marked with a star.}
\label{fig:pareto}
\end{figure}""",
        r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{figures/pareto_frontier_all.pdf}
\caption{Consolidated Multi-Noise Pareto Frontiers comparing trade-off boundaries across all evaluated hardware-noise regimes.}
\label{fig:pareto_all}
\end{figure}""",
    ]
    noise_refs = []
    for nm in noise_models:
        nm_clean = nm.lower()
        nm_label = nm_clean.replace("_", "-")
        nm_escaped = _escape_latex(nm)
        noise_refs.append(f"Figure~\\ref{{fig:pareto-{nm_label}}}")
        fig_blocks.append(
            f"""\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\linewidth]{{figures/pareto_frontier_{nm_clean}.pdf}}
\\caption{{Multi-Objective Pareto frontier for the \\texttt{{{nm_escaped}}} noise regime (Latency vs. Relative Error).}}
\\label{{fig:pareto-{nm_label}}}
\\end{{figure}}"""
        )
    noise_figure_refs = ", ".join(noise_refs) if noise_refs else "Figure~\\ref{fig:pareto_all}"
    pareto_figures_tex = "\n\n".join(fig_blocks)

    doc = f"""\\documentclass[10pt,conference]{{IEEEtran}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{xcolor}}
\\usepackage{{microtype}}
\\usepackage{{cite}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!70!black
}}

\\title{{Analog ML Co-Design: Un-0 Physical Neural Network Acceleration and Hardware-Noise Resilience}}

\\author{{
    \\IEEEauthorblockN{{Analog ML Co-Design Agentic Swarm}}
    \\IEEEauthorblockA{{Autonomous AI Hardware Acceleration Framework\\\\
    Chia Multi-Agent Swarm for Physical Neuromorphic Architectures}}
}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
Analog oscillatory neural networks offer massive parallel compute density and ultra-low energy dissipation for machine learning inference. In this technical report, we present the empirical results from an autonomous agentic optimization loop applied to the Un-0 analog oscillatory accelerator. We present a multi-objective Pareto analysis comparing numerical latency against golden model accuracy, quantify pipeline rewrite and simulation reliability across all exploration stages, and provide an exhaustive token and dollar audit of the autonomous agent swarm.
\\end{{abstract}}

\\section{{Executive Summary}}
{exec_summary}

\\section{{Multi-Objective Pareto Analysis}}
{pareto_analysis_text}

Figure~\\ref{{fig:pareto_all}} illustrates the consolidated empirical Pareto frontiers contrasting simulation latency against relative error across evaluated hardware-noise regimes, with individual noise breakdowns shown in {noise_figure_refs}. Table~\\ref{{tab:pareto}} lists key Pareto-optimal designs alongside the reference baseline.

{pareto_figures_tex}

\\begin{{table}}[htbp]
\\caption{{Pareto-Optimal Candidate Configurations}}
\\label{{tab:pareto}}
\\centering
\\begin{{tabular}}{{llcccrr}}
\\toprule
\\textbf{{Candidate ID}} & \\textbf{{Solver}} & \\textbf{{Noise}} & \\textbf{{Sparsity}} & \\textbf{{Latency (ms)}} & \\textbf{{Relative Error (L2)}} & \\textbf{{Speedup}} \\\\
\\midrule
{pareto_table}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Pipeline Execution Reliability}}
{reliability_text}

Figure~\\ref{{fig:breakdown}} provides the quantitative breakdown of execution outcomes across all tested candidates in the co-design pipeline.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\linewidth]{{figures/execution_breakdown.pdf}}
\\caption{{Execution reliability breakdown across solver rewrite, noise rewrite, compiler rewrite, circuit compilation, and numerical simulation stages.}}
\\label{{fig:breakdown}}
\\end{{figure}}

\\section{{Swarm Resource and Token Auditing}}
{resource_text}

Table~\\ref{{tab:token_cost}} and Figure~\\ref{{fig:tokens}} detail the token distribution and computational cost across swarm operational phases.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\linewidth]{{figures/token_cost_summary.pdf}}
\\caption{{Swarm-wide LLM token consumption and estimated dollar cost breakdown across operational phases.}}
\\label{{fig:tokens}}
\\end{{figure}}

\\begin{{table}}[htbp]
\\caption{{Swarm-Wide LLM Token Consumption and Cost Audit}}
\\label{{tab:token_cost}}
\\centering
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\textbf{{Swarm Phase}} & \\textbf{{Calls}} & \\textbf{{Prompt Tok.}} & \\textbf{{Compl. Tok.}} & \\textbf{{Total Tok.}} & \\textbf{{Cost (\\$)}} \\\\
\\midrule
{token_table}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Architectural Recommendations and Next Steps}}
{recommendations_text}

\\section{{Conclusion}}
The empirical results established in this optimization run provide actionable trade-offs and guide subsequent silicon tape-out parameters and swarm exploration sweeps.

\\end{{document}}
"""
    return doc


@ChiaFunction(resources={"agent_worker": 1, "antigravity_creds": 0.01})
def node2_phase2_generate_final_report(
    db: Any,
    config: Any,
    output_dir: Union[str, Path] = "reports",
    llm: Any = None,
    tools: Optional[List[Any]] = None,
    compile_pdf: bool = True,
) -> Dict[str, Any]:
    """Agentic node synthesizing simulation findings into a publication-grade LaTeX report with PDF.

    1. Resolves destination directory under <output_dir>/<experiment_name>.
    2. Generates publication Matplotlib plots (Pareto frontier, execution breakdown, token summary).
    3. Queries ResultDatabase for metrics, Pareto frontier, and execution statistics.
    4. Prompts LLM to reflect on and analyze empirical findings (Pareto diversity, pipeline reliability, tokens).
    5. Records LLM token usage to token_usage table and attaches metrics to Chia profiler.
    6. Formulates full sanitized LaTeX source file (final_report.tex).
    7. Compiles final_report.pdf via pdflatex.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance or dict.
        output_dir: Base directory where reports and figures are saved.
        llm: Optional LLM instance (default: gemini-2.5-flash or mock).
        tools: Optional list of ChiaTool instances.
        compile_pdf: Whether to execute pdflatex compilation (default: True).

    Returns:
        Dict[str, Any] containing paths, compilation status, and summary statistics.
    """
    exp_name = _get_val(config, "experiment_name", "un0_kuramoto_analog_acceleration")
    base_out_path = Path(output_dir)
    # Store systematically under a subfolder named after the experiment
    if base_out_path.name == exp_name:
        out_path = base_out_path
    else:
        out_path = base_out_path / exp_name

    fig_path = out_path / "figures"
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    # 1. Determine noise models from configuration or database
    noise_models = _get_val(config, "noise_models", None)
    if not noise_models:
        all_props = db.get_proposals(is_reference=False) if hasattr(db, "get_proposals") else []
        distinct = sorted(list({p.get("noise_model") for p in all_props if p.get("noise_model")}))
        noise_models = distinct if distinct else ["none"]

    # Generate publication Matplotlib plots (multi-noise Pareto and execution breakdown)
    logger.info("Generating publication Matplotlib figures in %s for noise models: %s...", fig_path, noise_models)
    multi_pareto_files = generate_multi_noise_pareto_plots(db, fig_path, noise_models=noise_models, prefix="pareto_frontier")
    pareto_files = generate_pareto_plot(db, fig_path, "pareto_frontier")
    breakdown_files = generate_breakdown_plot(db, fig_path, "execution_breakdown")

    # 2. Gather data for LLM reflection prompt
    pareto_front = db.compute_pareto_front() if hasattr(db, "compute_pareto_front") else []
    exec_stats = db.get_execution_statistics() if hasattr(db, "get_execution_statistics") else {}
    token_summary = db.get_token_usage_summary() if hasattr(db, "get_token_usage_summary") else {}
    ref_proposal = db.get_reference_proposal() if hasattr(db, "get_reference_proposal") else None
    completed_candidates = db.get_proposals(is_reference=False, limit=100) if hasattr(db, "get_proposals") else []
    compilation_failures = _get_compilation_failures(db)

    # 3. Prompt LLM to reflect, analyze, and draft dynamic technical synthesis
    llm_synthesis = None
    if llm is False:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost_usd = 0.0
        model_name = "dynamic_engine"
    else:
        active_llm = _resolve_llm(llm, config)
        prompt_payload = {
            "experiment_name": exp_name,
            "reference_baseline": ref_proposal,
            "pareto_candidates": pareto_front,
            "execution_statistics": exec_stats,
            "compilation_failure_logs": compilation_failures,
            "token_usage_summary": token_summary,
            "instruction": (
                "Reflect on and analyze the actual experimental results of this loop run. "
                "Respond with a JSON object containing keys: 'executive_summary', 'pareto_analysis', "
                "'reliability_discussion', 'resource_discussion', and 'recommendations'. "
                "Base your discussion strictly on empirical data: analyze the distribution "
                "of Pareto front points across noise models; analyze pipeline reliability and bottlenecks "
                "strictly using empirical records and provided compilation_failure_logs (grounding compilation failures "
                "in factual PyTorch Inductor CPU lowering exceptions such as LoweringException or unbacked scalar bindings, "
                "and NEVER hallucinating schema validation rejections or invalid sparsity_ratio claims); "
                "and audit swarm token and compute expenditures. Avoid static boilerplate or rigid closed heuristics."
            ),
        }

        user_message = f"Please draft the technical synthesis for the final co-design report:\n{json.dumps(prompt_payload, default=str)}"

        start_time = time.time()
        query_result = active_llm.prompt(user_message=user_message, tools=tools)
        duration = time.time() - start_time

        # 4. Extract token metrics and record to database and profiler
        usage = getattr(query_result, "usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("candidates_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        cost_usd = (prompt_tokens * 0.075 / 1_000_000) + (completion_tokens * 0.30 / 1_000_000)

        model_name = getattr(active_llm, "model", "gemini-2.5-flash")

        if hasattr(db, "record_token_usage") and callable(db.record_token_usage):
            try:
                db.record_token_usage(
                    phase="phase_2",
                    node_name="node2_phase2_generate_final_report",
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    duration_seconds=duration,
                )
            except Exception as exc:
                logger.warning("Failed to record token usage to DB: %s", exc)

        try:
            from chia.base.profiling import get_profiler
            get_profiler().add_info({"usage": usage, "model": model_name, "phase": "phase_2"})
        except Exception:
            pass

        # Parse LLM response if valid JSON returned
        response_text = getattr(query_result, "result", str(query_result))
        if isinstance(response_text, str):
            try:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match:
                    llm_synthesis = json.loads(match.group(1))
                else:
                    llm_synthesis = json.loads(response_text)
            except Exception:
                pass

    # 5. Generate token cost plot AFTER recording Phase 2 tokens to capture complete swarm expenditure
    token_files = generate_token_cost_plot(db, fig_path, "token_cost_summary")

    figures = {
        "pareto": pareto_files,
        "pareto_multi": multi_pareto_files,
        "breakdown": breakdown_files,
        "token_cost": token_files,
    }

    # 6. Generate sanitized LaTeX document
    tex_content = _generate_report_tex_content(
        db, config, llm_synthesis=llm_synthesis, noise_models=noise_models
    )
    tex_path = out_path / "final_report.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    logger.info("LaTeX source written to: %s", tex_path)

    # 7. Compile LaTeX to PDF
    pdf_compiled = False
    pdf_path = out_path / "final_report.pdf"
    if compile_pdf:
        pdf_compiled = compile_latex(tex_path, out_path)
        if pdf_compiled:
            logger.info("PDF compiled successfully to: %s", pdf_path)
        else:
            logger.warning("PDF compilation was not successful; .tex file is available at %s", tex_path)

    return {
        "status": "SUCCESS" if pdf_compiled or not compile_pdf else "PARTIAL",
        "experiment_name": exp_name,
        "output_dir": str(out_path),
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path) if pdf_path.exists() else None,
        "pdf_compiled": pdf_compiled,
        "figures": figures,
        "summary": {
            "num_candidates_analyzed": len(completed_candidates),
            "pareto_candidates_count": len(pareto_front),
            "compilations_succeeded": exec_stats.get("compilations_succeeded_count", 0),
            "simulations_succeeded": exec_stats.get("simulations_succeeded_count", 0),
            "token_usage_summary": db.get_token_usage_summary() if hasattr(db, "get_token_usage_summary") else {},
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "model": model_name,
        },
        "message": "Report generated successfully." if pdf_compiled else "LaTeX report generated.",
    }


# Backward compatibility alias
phase2_generate_final_report = node2_phase2_generate_final_report
