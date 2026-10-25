#!/usr/bin/env python3
"""Standalone non-invasive report generation and validation script.

Executes Phase 2 reporting node (node2_phase2_generate_final_report) directly
from an existing results.db without triggering or modifying the main co-design loop.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to python path if running standalone
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node2_phase2_generate_final_report import (
    _find_pdflatex,
    compile_latex,
    node2_phase2_generate_final_report,
)
from agentic_ml_analog_sim.reporting.latex_sanitizer import sanitize_latex_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_report_from_db")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 embellished technical report and Pareto plots from an existing results.db."
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to existing SQLite results.db database.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/simulation_config.yaml",
        help="Path to simulation config YAML (default: config/simulation_config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/embellished_validation",
        help="Target output directory for the report and generated figures (default: reports/embellished_validation).",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        choices=["none", "mock", "antigravity"],
        default="none",
        help="LLM synthesis mode: 'none' (dynamic data-driven fallback), 'mock' (mock synthesis), or 'antigravity'.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip pdflatex compilation to PDF (only generate TeX and figures).",
    )
    parser.add_argument(
        "--sanitize-existing-tex",
        type=str,
        default=None,
        help="Optional path to an existing .tex file to sanitize and compile directly.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        logger.error("Results database not found at %s", db_path)
        sys.exit(1)

    # Optional mode: sanitize and compile an existing .tex file directly
    if args.sanitize_existing_tex:
        tex_path = Path(args.sanitize_existing_tex).resolve()
        if not tex_path.exists():
            logger.error("Existing .tex file not found at %s", tex_path)
            sys.exit(1)
        logger.info("Sanitizing existing TeX file: %s", tex_path)
        content = tex_path.read_text(encoding="utf-8")
        sanitized = sanitize_latex_document(content)
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_tex = out_dir / tex_path.name
        out_tex.write_text(sanitized, encoding="utf-8")
        logger.info("Sanitized TeX written to %s", out_tex)
        if not args.no_pdf:
            compiled = compile_latex(out_tex, out_dir)
            logger.info("PDF compilation %s", "SUCCEEDED" if compiled else "FAILED")
        return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if config_path.exists():
        cfg = load_simulation_config(str(config_path))
    else:
        logger.warning("Config YAML not found at %s; using default config object.", config_path)
        cfg = {"experiment_name": "un0_kuramoto_analog_acceleration"}

    db = ResultDatabase(db_path=db_path)
    logger.info("Connected to database: %s", db_path)
    proposals = db.get_proposals(limit=500)
    pareto = db.compute_pareto_front()
    ref = db.get_reference_proposal()
    logger.info(
        "Database contains %d proposals, %d Pareto-optimal points, reference baseline: %s",
        len(proposals),
        len(pareto),
        ref.get("candidate_id") if ref else "None",
    )

    llm_instance = None
    if args.llm_mode == "none":
        llm_instance = False
    elif args.llm_mode == "mock":
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        llm_instance = MockAntigravityLLM(mode="valid")
    elif args.llm_mode == "antigravity":
        llm_instance = None  # Will resolve default AntigravityLLM

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    # Sync experiment_name if out_dir targets a specific experiment subfolder
    if out_dir.name and out_dir.name not in ("reports", "embellished_validation"):
        target_exp = out_dir.name
        if isinstance(cfg, dict):
            cfg["experiment_name"] = target_exp
        elif hasattr(cfg, "model_copy"):
            cfg = cfg.model_copy(update={"experiment_name": target_exp})

    logger.info("Executing Phase 2 reporting node with output directory: %s", out_dir)
    result = node2_phase2_generate_final_report(
        db=db,
        config=cfg,
        output_dir=out_dir,
        llm=llm_instance,
        compile_pdf=not args.no_pdf,
    )

    logger.info("Report generation result status: %s", result.get("status"))
    logger.info("Generated LaTeX: %s", result.get("tex_path"))
    logger.info("Compiled PDF: %s", result.get("pdf_path"))

    # Print summary of generated figures
    figs = result.get("figures", {})
    logger.info("Generated figures:")
    for key, paths in figs.items():
        if isinstance(paths, dict):
            for subk, p in paths.items():
                logger.info("  - %s (%s): %s", key, subk, p)
        elif isinstance(paths, list):
            for p in paths:
                logger.info("  - %s: %s", key, p)
        else:
            logger.info("  - %s: %s", key, paths)

    # Verification checks
    tex_path = Path(result["tex_path"])
    if tex_path.exists():
        tex_text = tex_path.read_text(encoding="utf-8")
        has_circle = "Golden baseline is marked with a yellow circle" in tex_text
        has_star = "marked with a star" in tex_text
        has_abs_fig = "pareto_frontier_abs" in tex_text
        has_cluster_fig = "pareto_clusters" in tex_text
        has_table_resize = r"\resizebox{\textwidth}{!}" in tex_text
        has_col_resize = r"\resizebox{\columnwidth}{!}" in tex_text

        print("\n================== Verification Checklist ==================")
        print(f"1. Baseline marker is yellow circle: {'PASS' if has_circle and not has_star else 'FAIL'}")
        print(f"2. Absolute error Pareto figures embedded: {'PASS' if has_abs_fig else 'FAIL'}")
        print(f"3. Matplotlib frontier geometry figure embedded: {'PASS' if has_cluster_fig else 'FAIL'}")
        print(f"4. Table tab:pareto wrapped in table* with resizebox: {'PASS' if has_table_resize else 'FAIL'}")
        print(f"5. Table tab:token_cost wrapped with resizebox: {'PASS' if has_col_resize else 'FAIL'}")
        print(f"6. PDF compiled cleanly: {'PASS' if result.get('pdf_compiled') else 'SKIPPED or FAILED'}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
