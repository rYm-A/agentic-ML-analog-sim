#!/usr/bin/env python
"""Unified Execution Loop for the Un-0 Analog ML Co-Design Swarm.

Coordinates and unifies the three phases of the agentic loop:
  - Phase 0: Exploration & Proposal (Nodes 1, 2, 3)
  - Phase 1: Simulation, Rewriting & Verification (Nodes 0 through 8)
  - Phase 2: Auditing, Budget Verification & Technical Reporting (Nodes 1, 2)

Systematically manages experiment-specific output subfolders under reports/
and collocates the SQLite results database for persistent auditability.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure local worktree src/ takes precedence in sys.path
_worktree_src = Path(__file__).resolve().parent / "src"
if _worktree_src.is_dir() and str(_worktree_src) not in sys.path:
    sys.path.insert(0, str(_worktree_src))

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.phase_0 import run_phase_0
from agentic_ml_analog_sim.phase_1 import run_phase_1
from agentic_ml_analog_sim.phase_2 import run_phase_2

logger = logging.getLogger("run_loop")


def parse_args() -> argparse.ArgumentParser:
    """Parse command-line arguments for the unified loop runner."""
    parser = argparse.ArgumentParser(
        description="Unified execution loop for the Un-0 analog ML simulation swarm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/simulation_config.yaml"),
        help="Path to the simulation configuration YAML file.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Base directory where experiment report subfolders are stored.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Custom storage point for the SQLite database. If omitted, defaults to "
        "<reports-dir>/<experiment_name>/results.db.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max loop iterations specified in the configuration budget.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Override max completed experiments specified in the configuration budget.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip compiling LaTeX report into PDF via pdflatex.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    return parser


def setup_experiment_storage(
    config: SimulationConfig,
    reports_base: Path,
    custom_db_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Check/create experiment subfolder in reports/ and resolve DB storage point.

    Args:
        config: Loaded SimulationConfig instance.
        reports_base: Base directory for reports (typically 'reports').
        custom_db_path: Optional user override for database location.

    Returns:
        Tuple of (experiment_dir, db_path).
    """
    exp_name = getattr(config, "experiment_name", "un0_analog_acceleration")
    reports_base = reports_base.resolve()
    reports_base.mkdir(parents=True, exist_ok=True)

    experiment_dir = reports_base / exp_name
    if experiment_dir.exists():
        logger.info("Found existing experiment directory: %s", experiment_dir)
    else:
        logger.info("Creating experiment directory: %s", experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)

    if custom_db_path is not None:
        db_path = custom_db_path.resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        db_path = experiment_dir / "results.db"

    logger.info("Configured database storage point: %s", db_path)
    return experiment_dir, db_path


def execute_loop(
    config_path: Path,
    reports_base: Path = Path("reports"),
    custom_db_path: Optional[Path] = None,
    max_iterations_override: Optional[int] = None,
    max_experiments_override: Optional[int] = None,
    compile_pdf: bool = True,
) -> None:
    """Execute the unified multi-phase loop until termination conditions are met."""
    if not config_path.exists():
        logger.error("Configuration file not found at: %s", config_path.resolve())
        sys.exit(1)

    config = load_simulation_config(config_path)
    experiment_dir, db_path = setup_experiment_storage(
        config=config,
        reports_base=reports_base,
        custom_db_path=custom_db_path,
    )

    db = ResultDatabase(db_path=db_path)

    # Resolve budget constraints
    max_iterations = (
        max_iterations_override
        if max_iterations_override is not None
        else getattr(config.budget, "max_iterations", 15)
    )
    max_experiments = (
        max_experiments_override
        if max_experiments_override is not None
        else getattr(config.budget, "max_experiments", 10)
    )

    logger.info(
        "Starting unified Chia loop: experiment='%s', max_experiments=%d, max_iterations=%d",
        getattr(config, "experiment_name", "unknown"),
        max_experiments,
        max_iterations,
    )

    iteration = 0
    start_time = time.time()

    while iteration < max_iterations:
        iteration += 1
        logger.info(
            "=== Starting Unified Loop Iteration %d/%d ===",
            iteration,
            max_iterations,
        )

        # -----------------------------------------------------------------
        # STEP 1: Phase 0 (Exploration, Reference Check, Candidate Proposal)
        # -----------------------------------------------------------------
        p0_res = run_phase_0(config_path=config_path, db_path=db_path)
        logger.info(
            "Phase 0 action=%s, candidate_id=%s, details=%s",
            p0_res.action,
            p0_res.candidate_id,
            p0_res.details,
        )

        if p0_res.action in ("ERROR", "TERMINATED_MAX_CYCLES"):
            logger.error("Phase 0 halted with action %s: %s", p0_res.action, p0_res.details)
            completed_count = db.get_completed_experiments_count()
            if completed_count > 0:
                logger.info("Generating report with %d completed experiments...", completed_count)
                run_phase_2(
                    config_path=config_path,
                    db_path=db_path,
                    current_iteration=iteration,
                    candidate_id=p0_res.candidate_id,
                    output_dir=reports_base,
                    force_report=True,
                    compile_pdf=compile_pdf,
                )
            break

        candidate_id = p0_res.candidate_id
        if not candidate_id:
            logger.warning("No candidate ID returned from Phase 0; stopping loop.")
            break

        # -----------------------------------------------------------------
        # STEP 2: Phase 1 (Simulation, Rewriting, Verification, Profiling)
        # -----------------------------------------------------------------
        p1_res = run_phase_1(
            config_path=config_path,
            db_path=db_path,
            candidate_id=candidate_id,
            trigger_source="phase_0",
        )
        logger.info(
            "Phase 1 action=%s, candidate_id=%s, mode=%s, details=%s",
            p1_res.action,
            p1_res.candidate_id,
            p1_res.execution_mode,
            p1_res.details,
        )

        # -----------------------------------------------------------------
        # STEP 3: Phase 2 (Budget Checking, Synthesis, Technical Reporting)
        # -----------------------------------------------------------------
        p2_res = run_phase_2(
            config_path=config_path,
            db_path=db_path,
            current_iteration=iteration,
            candidate_id=candidate_id,
            output_dir=reports_base,
            compile_pdf=compile_pdf,
        )
        logger.info(
            "Phase 2 action=%s, completed=%d/%d, details=%s",
            p2_res.action,
            p2_res.completed_experiments,
            p2_res.max_experiments,
            p2_res.details,
        )

        if p2_res.action == "REPORT_GENERATED":
            logger.info("Loop reached goal! Technical report generated at %s", p2_res.tex_path)
            if p2_res.pdf_compiled:
                logger.info("PDF report compiled successfully at %s", p2_res.pdf_path)
            break

    elapsed_s = time.time() - start_time
    completed_total = db.get_completed_experiments_count()
    token_summary = db.get_token_usage_summary()

    logger.info("=== Unified Loop Execution Complete in %.1fs ===", elapsed_s)
    logger.info("Total Completed Designs: %d", completed_total)
    logger.info("Database File: %s", db_path)
    logger.info("Experiment Directory: %s", experiment_dir)
    logger.info("Total Swarm Tokens Consumed: %d", token_summary.get("overall", {}).get("total_tokens", 0))
    logger.info("Total Swarm Cost: $%.4f USD", token_summary.get("overall", {}).get("total_cost_usd", 0.0))


def main() -> None:
    """Entrypoint for the unified loop runner."""
    parser = parse_args()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    execute_loop(
        config_path=args.config,
        reports_base=args.reports_dir,
        custom_db_path=args.db_path,
        max_iterations_override=args.max_iterations,
        max_experiments_override=args.max_experiments,
        compile_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
