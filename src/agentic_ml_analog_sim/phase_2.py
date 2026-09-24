"""Phase 2 Pipeline Orchestrator for the analog ML accelerator simulation loop.

Coordinates:
  - Node 1: Programmatic check for experiment and iteration limits (phase2_check_experiment_limit).
  - Node 2: Agentic report synthesis via single mid-effort LLM (phase2_generate_final_report).
  - Post-report interactive multi-turn consultant session (InteractiveReportSession).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.nodes import (
    node1_phase2_check_experiment_limit,
    node2_phase2_generate_final_report,
    phase2_check_experiment_limit,
    phase2_generate_final_report,
)
from agentic_ml_analog_sim.reporting.interactive_session import InteractiveReportSession

try:
    from chia.base.ChiaFunction import get
except ImportError:
    get = None
import ray

logger = logging.getLogger("agentic_ml_analog_sim.phase_2")


def _dispatch_node(node_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Dispatch a node function via chia_remote if Ray is initialized, else local."""
    if ray.is_initialized() and hasattr(node_fn, "chia_remote") and get is not None:
        ref = node_fn.chia_remote(*args, **kwargs)
        return get(ref)
    return node_fn(*args, **kwargs)


@dataclass
class Phase2Result:
    """Outcome of Phase 2 limit evaluation and report generation."""

    action: Literal["CONTINUE_LOOP", "REPORT_GENERATED", "ERROR"]
    completed_experiments: int
    max_experiments: int
    tex_path: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_compiled: bool = False
    figures: Optional[Dict[str, Dict[str, str]]] = None
    summary: Optional[Dict[str, Any]] = None
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def run_phase_2(
    config_path: Union[str, Path, SimulationConfig],
    db_path: Union[str, Path, ResultDatabase],
    current_iteration: Optional[int] = None,
    candidate_id: Optional[str] = None,
    output_dir: Union[str, Path] = "reports",
    llm: Any = None,
    force_report: bool = False,
    compile_pdf: bool = True,
    interactive: bool = False,
) -> Phase2Result:
    """Execute Phase 2 limit verification and reporting pipeline.

    1. Checks completed candidate experiment count against experiment limits.
    2. If limit not reached and force_report is False:
       - Returns Phase2Result(action="CONTINUE_LOOP").
    3. If limit reached or force_report is True:
       - Executes phase2_generate_final_report (generates Matplotlib figures, LaTeX document, compiles PDF).
       - If interactive is True, launches InteractiveReportSession REPL.
       - Returns Phase2Result(action="REPORT_GENERATED").

    Args:
        config_path: Path to YAML config or SimulationConfig instance.
        db_path: Path to SQLite DB file or ResultDatabase instance.
        current_iteration: Optional current iteration count.
        candidate_id: Optional latest candidate ID from Phase 1 Node 7.
        output_dir: Target directory for report and figures.
        llm: Optional LLM instance (AntigravityLLM or MockAntigravityLLM).
        force_report: If True, bypass limit check and generate report immediately.
        compile_pdf: Whether to compile final_report.pdf via pdflatex.
        interactive: If True, launches REPL post-generation.

    Returns:
        Phase2Result.
    """
    # 1. Resolve Config
    if isinstance(config_path, SimulationConfig):
        config = config_path
    else:
        config = load_simulation_config(config_path)

    # 2. Resolve Database
    if isinstance(db_path, ResultDatabase):
        db = db_path
    else:
        db = ResultDatabase(db_path=db_path)

    # 3. Check Experiment Limits
    limit_check = _dispatch_node(
        node1_phase2_check_experiment_limit,
        db=db,
        config=config,
        current_iteration=current_iteration,
        candidate_id=candidate_id,
    )

    limit_reached = limit_check.get("limit_reached", False)
    completed = limit_check.get("completed_experiments", 0)
    max_exp = limit_check.get("max_experiments", 10)
    reason = limit_check.get("reason", "")

    if not limit_reached and not force_report:
        logger.info("Phase 2: Experiment limit not met (%s). Continuing loop.", reason)
        return Phase2Result(
            action="CONTINUE_LOOP",
            completed_experiments=completed,
            max_experiments=max_exp,
            details=reason,
        )

    # 4. Limit reached (or forced) -> Generate final report
    logger.info("Phase 2: Limit reached or report forced (%s). Generating final report...", reason)
    report_res = _dispatch_node(
        node2_phase2_generate_final_report,
        db=db,
        config=config,
        output_dir=output_dir,
        llm=llm,
        compile_pdf=compile_pdf,
    )

    tex_path = report_res.get("tex_path")
    pdf_path = report_res.get("pdf_path")
    pdf_compiled = report_res.get("pdf_compiled", False)
    figures = report_res.get("figures")
    summary = report_res.get("summary")

    # 5. Launch interactive session if requested
    if interactive:
        logger.info("Starting interactive consultant session...")
        session = InteractiveReportSession(
            db=db,
            report_tex_path=tex_path,
            llm=llm,
        )
        session.run_repl()

    return Phase2Result(
        action="REPORT_GENERATED",
        completed_experiments=completed,
        max_experiments=max_exp,
        tex_path=tex_path,
        pdf_path=pdf_path,
        pdf_compiled=pdf_compiled,
        figures=figures,
        summary=summary,
        details=f"Final report generated ({'PDF compiled' if pdf_compiled else 'LaTeX only'}). {reason}",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for Phase 2."""
    parser = argparse.ArgumentParser(
        description="Phase 2 CLI: Limit verification and final reporting for analog ML accelerator co-design."
    )
    parser.add_argument(
        "--config",
        default="config/simulation_config.yaml",
        help="Path to simulation YAML configuration file (default: config/simulation_config.yaml).",
    )
    parser.add_argument(
        "--db",
        default="results.db",
        help="Path to SQLite results database (default: results.db).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for generated figures and reports (default: reports).",
    )
    parser.add_argument(
        "--force-report",
        action="store_true",
        help="Force report generation immediately, bypassing experiment count check.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable pdflatex PDF compilation and produce only LaTeX source.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal multi-turn consultant session after report generation.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockAntigravityLLM instead of real AntigravityLLM for deterministic offline runs.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Current loop iteration counter.",
    )
    return parser


def main() -> None:
    """Entry point for Phase 2 CLI execution."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    llm = MockAntigravityLLM(mode="valid") if args.mock else None

    result = run_phase_2(
        config_path=args.config,
        db_path=args.db,
        current_iteration=args.iteration,
        output_dir=args.output_dir,
        llm=llm,
        force_report=args.force_report,
        compile_pdf=not args.no_pdf,
        interactive=args.interactive,
    )

    print("\n" + "=" * 60)
    print("PHASE 2 EXECUTION SUMMARY")
    print("=" * 60)
    print(result.to_json())


if __name__ == "__main__":
    main()
