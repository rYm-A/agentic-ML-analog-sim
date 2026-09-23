#!/usr/bin/env python
"""Unified Execution Loop for the Un-0 Analog ML Co-Design Swarm.

Coordinates and unifies the three phases of the agentic loop:
  - Phase 0: Exploration & Proposal (Nodes 1, 2, 3)
  - Phase 1: Simulation, Rewriting & Verification (Nodes 0 through 8)
  - Phase 2: Auditing, Budget Verification & Technical Reporting (Nodes 1, 2)

Systematically manages experiment-specific output subfolders under reports/
and collocates the SQLite results database for persistent auditability.

Cluster Launch Compatibility & Submission via Chia Job Launcher:
---------------------------------------------------------------
This driver script is fully formatted for execution via the Chia cluster
job launcher (`chia job submit`). It connects to the cluster's Ray runtime,
initiates the top-level Chia profiling lifecycle, and orchestrates distributed
logical workers (`control_worker`, `agent_worker`, `compile_worker`, `sim_worker`).

1. Local Apple Silicon Mac Cluster:
   $ chia up cluster.bare.yaml
   $ chia job submit --working-dir . -- python run_loop.py --config config/simulation_config.yaml
   $ chia job status <job_id>
   $ chia job logs <job_id> -f
   $ chia down cluster.bare.yaml

2. Cloud Linux / CUDA Cluster (GCP + Tailnet VPN):
   $ export GCP_PROJECT="project-bd43fccf-d505-4c80-a3c"
   $ export TS_AUTHKEY="tskey-auth-..."
   $ chia up cluster.cloud.yaml
   $ chia job submit \\
       --working-dir . \\
       --runtime-env-json "{\\"env_vars\\": {\\"GCP_PROJECT\\": \\"$GCP_PROJECT\\", \\"PYTHONPATH\\": \\".:src\\"}}" \\
       -- python run_loop.py --config config/simulation_config.yaml
   $ chia job logs <job_id> -f
   $ chia down cluster.cloud.yaml

3. Visualizing Chia Profiles:
   $ chia viz-profile reports/un0_analog_acceleration/profiles/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure local worktree src/ takes precedence in sys.path
_worktree_src = Path(__file__).resolve().parent / "src"
if _worktree_src.is_dir() and str(_worktree_src) not in sys.path:
    sys.path.insert(0, str(_worktree_src))

import ray

try:
    from chia.trace.profiler import start_collector, stop_collector
    _PROFILER_AVAILABLE = True
except ImportError:
    _PROFILER_AVAILABLE = False

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.phase_0 import run_phase_0
from agentic_ml_analog_sim.phase_1 import run_phase_1
from agentic_ml_analog_sim.phase_2 import run_phase_2

logger = logging.getLogger("run_loop")


def _configure_tailnet_proxy_if_present() -> None:
    """If a Chia tailnet CONNECT relay is listening locally, configure Ray gRPC proxying."""
    import os
    import socket
    if "grpc_proxy" in os.environ:
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        if sock.connect_ex(("127.0.0.1", 13129)) == 0:
            os.environ["RAY_grpc_enable_http_proxy"] = "1"
            os.environ["grpc_proxy"] = "http://127.0.0.1:13129"
            try:
                local_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                local_ip = "127.0.0.1"
            current = os.environ.get("no_grpc_proxy", "")
            additions = [local_ip, "127.0.0.1", "localhost"]
            os.environ["no_grpc_proxy"] = ",".join(filter(None, [current] + additions))
            logger.info("Auto-configured Ray gRPC proxy via local Chia relay (127.0.0.1:13129)")
    except Exception:
        pass
    finally:
        sock.close()


def ensure_ray_initialized(
    address: Optional[str] = "auto",
    auto_init_local: bool = True,
) -> bool:
    """Verify and ensure that Ray is initialized for cluster-wide node scheduling.

    Args:
        address: Ray cluster address. Defaults to "auto" to attach to an active Chia cluster.
        auto_init_local: Fallback to initializing a local Ray runtime fallback if cluster is unavailable.

    Returns:
        True if Ray is initialized, False otherwise.
    """
    _configure_tailnet_proxy_if_present()
    if not ray.is_initialized():
        logger.info("Ray is not initialized. Attempting connection to Ray cluster (address=%s)...", address)
        try:
            ray.init(address=address, ignore_reinit_error=True)
            logger.info("Successfully connected to Ray cluster at address: %s", address)
        except Exception as e_cluster:
            if auto_init_local:
                logger.warning(
                    "Could not connect to Ray cluster at '%s' (%s). Initializing local Ray runtime fallback...",
                    address,
                    e_cluster,
                )
                try:
                    ray.init(ignore_reinit_error=True)
                    logger.info("Initialized local Ray runtime fallback.")
                except Exception as e_local:
                    logger.error("Failed to initialize local Ray runtime: %s", e_local)
                    return False
            else:
                logger.error("Could not connect to Ray cluster at '%s': %s", address, e_cluster)
                return False

    if ray.is_initialized():
        ctx = ray.get_runtime_context()
        node_id = ctx.get_node_id()
        resources = ray.cluster_resources()
        logger.info(
            "Ray cluster active: node_id=%s, available_cpus=%s, available_gpus=%s",
            node_id,
            resources.get("CPU", 0),
            resources.get("GPU", 0),
        )
        logical_workers = {
            k: v
            for k, v in resources.items()
            if k in (
                "control_worker",
                "agent_worker",
                "compile_worker",
                "sim_worker",
                "antigravity_creds",
                "sqlite_db",
            )
        }
        if logical_workers:
            logger.info("Detected Chia logical worker resources: %s", logical_workers)
        else:
            logger.warning(
                "No custom Chia logical worker resources detected on cluster; executing with default resources."
            )
        return True
    return False


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
        "--ray-address",
        type=str,
        default="auto",
        help="Ray cluster address to connect to (e.g. 'auto', '127.0.0.1:6379').",
    )
    parser.add_argument(
        "--no-ray-init",
        action="store_true",
        help="Disable automatic Ray initialization/connection verification.",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable top-level Chia profiling collector lifecycle.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Directory to save Chia profiler JSONL traces. Defaults to <experiment_dir>/profiles.",
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
    ray_address: Optional[str] = "auto",
    auto_init_ray: bool = True,
    enable_profiling: bool = True,
    profile_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the unified multi-phase loop until termination conditions are met.

    Args:
        config_path: Path to simulation configuration YAML.
        reports_base: Base directory for reports.
        custom_db_path: Optional custom path for results SQLite database.
        max_iterations_override: Optional loop iteration limit override.
        max_experiments_override: Optional completed experiment quota override.
        compile_pdf: Whether to compile LaTeX into PDF.
        ray_address: Ray cluster address (e.g. 'auto' or host:port).
        auto_init_ray: Whether to auto-initialize/verify Ray connection.
        enable_profiling: Whether to manage top-level Chia profiler collector lifecycle.
        profile_dir: Optional custom directory for profiler traces.

    Returns:
        Summary dictionary containing execution metrics and artifact paths.
    """
    if not config_path.exists():
        logger.error("Configuration file not found at: %s", config_path.resolve())
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Verification Step: Ensure Ray is initialized for Chia cluster dispatch
    # -------------------------------------------------------------------------
    if auto_init_ray:
        ray_ok = ensure_ray_initialized(address=ray_address, auto_init_local=True)
        if not ray_ok:
            logger.warning("Ray could not be initialized; proceeding in standalone synchronous mode.")
    else:
        logger.info("Ray automatic verification bypassed by user request (auto_init_ray=False).")

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
        "Starting unified Chia loop: experiment='%s', max_experiments_per_noise=%d, max_iterations=%d",
        getattr(config, "experiment_name", "unknown"),
        max_experiments,
        max_iterations,
    )

    # -------------------------------------------------------------------------
    # 2. Chia Profiling Lifecycle: Start collector actor on cluster head
    # -------------------------------------------------------------------------
    profile_collector_active = False
    profile_log_dir = (
        Path(profile_dir).resolve()
        if profile_dir is not None
        else (experiment_dir / "profiles")
    )
    if enable_profiling:
        if _PROFILER_AVAILABLE and ray.is_initialized():
            profile_log_dir.mkdir(parents=True, exist_ok=True)
            try:
                start_collector(log_dir=str(profile_log_dir))
                profile_collector_active = True
                logger.info("Chia profiler collector started. Trace directory: %s", profile_log_dir)
            except Exception as exc:
                logger.warning("Failed to start Chia profile collector: %s", exc)
        elif not _PROFILER_AVAILABLE:
            logger.warning("Chia profiler modules not found. Continuing without profiling.")
        elif not ray.is_initialized():
            logger.warning("Ray is not initialized. Skipping Chia profiler collector startup.")

    iteration = 0
    start_time = time.time()
    loop_status = "RUNNING"
    final_report_tex: Optional[str] = None
    final_report_pdf: Optional[str] = None

    try:
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
            p0_res = run_phase_0(config_path=config_path, db_path=db_path, iteration=iteration)
            logger.info(
                "Phase 0 action=%s, candidate_id=%s, details=%s",
                p0_res.action,
                p0_res.candidate_id,
                p0_res.details,
            )

            if p0_res.action in ("ERROR", "TERMINATED_MAX_CYCLES"):
                logger.error("Phase 0 halted with action %s: %s", p0_res.action, p0_res.details)
                loop_status = p0_res.action
                completed_count = db.get_completed_experiments_count()
                if completed_count > 0:
                    logger.info("Generating report with %d completed experiments...", completed_count)
                    p2_res = run_phase_2(
                        config_path=config_path,
                        db_path=db_path,
                        current_iteration=iteration,
                        candidate_id=p0_res.candidate_id,
                        output_dir=reports_base,
                        force_report=True,
                        compile_pdf=compile_pdf,
                    )
                    final_report_tex = str(p2_res.tex_path) if p2_res.tex_path else None
                    final_report_pdf = str(p2_res.pdf_path) if p2_res.pdf_path else None
                break

            candidate_id = p0_res.candidate_id
            if not candidate_id:
                logger.warning("No candidate ID returned from Phase 0; stopping loop.")
                loop_status = "NO_CANDIDATE"
                break

            # -----------------------------------------------------------------
            # STEP 2: Phase 1 (Simulation, Rewriting, Verification, Profiling)
            # -----------------------------------------------------------------
            p1_res = run_phase_1(
                config_path=config_path,
                db_path=db_path,
                candidate_id=candidate_id,
                trigger_source="phase_0",
                target_device=getattr(config, "device", "cpu"),
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
            # Multi-noise quota progress logging
            progress_msg = ""
            if hasattr(db, "get_proposal_counts_by_noise_model"):
                try:
                    noise_counts = db.get_proposal_counts_by_noise_model(exclude_reference=True)
                    configured_noise = getattr(config, "noise_models", ["L0_static_mismatch"])
                    noise_names = [getattr(nm, "name", nm) if not isinstance(nm, str) else nm for nm in configured_noise]
                    items = [
                        f"Noise model {nm}: {noise_counts.get(nm, {}).get('completed', 0) + noise_counts.get(nm, {}).get('failed', 0)}/{max_experiments}"
                        for nm in noise_names
                    ]
                    progress_msg = f" ({', '.join(items)})"
                except Exception:
                    pass

            logger.info(
                "Phase 2 action=%s, completed=%d/%d%s, details=%s",
                p2_res.action,
                p2_res.completed_experiments,
                p2_res.max_experiments,
                progress_msg,
                p2_res.details,
            )

            if p2_res.action == "REPORT_GENERATED":
                loop_status = "SUCCESS_REPORT_GENERATED"
                final_report_tex = str(p2_res.tex_path) if p2_res.tex_path else None
                final_report_pdf = str(p2_res.pdf_path) if p2_res.pdf_path else None
                logger.info("Loop reached goal! Technical report generated at %s", p2_res.tex_path)
                if p2_res.pdf_compiled:
                    logger.info("PDF report compiled successfully at %s", p2_res.pdf_path)
                break

        if loop_status == "RUNNING":
            loop_status = "MAX_ITERATIONS_REACHED"

    finally:
        # Stop profiler collector cleanly regardless of loop outcome
        if profile_collector_active:
            try:
                stop_collector()
                logger.info("Chia profiler collector stopped cleanly.")
                logger.info("Trace logs written under: %s", profile_log_dir)
                logger.info("To visualize the execution graph: chia viz-profile %s", profile_log_dir)
            except Exception as exc:
                logger.warning("Error stopping Chia profile collector: %s", exc)

    elapsed_s = time.time() - start_time
    completed_total = db.get_completed_experiments_count()
    token_summary = db.get_token_usage_summary()

    # Per-noise quota summary
    per_noise_summary = {}
    if hasattr(db, "get_proposal_counts_by_noise_model"):
        try:
            noise_counts = db.get_proposal_counts_by_noise_model(exclude_reference=True)
            configured_noise = getattr(config, "noise_models", ["L0_static_mismatch"])
            noise_names = [getattr(nm, "name", nm) if not isinstance(nm, str) else nm for nm in configured_noise]
            progress_items = [
                f"Noise model {nm}: {noise_counts.get(nm, {}).get('completed', 0) + noise_counts.get(nm, {}).get('failed', 0)}/{max_experiments}"
                for nm in noise_names
            ]
            logger.info("Multi-Noise Quota Summary: %s", ", ".join(progress_items))
            per_noise_summary = noise_counts
        except Exception:
            pass

    logger.info("=== Unified Loop Execution Complete in %.1fs ===", elapsed_s)
    logger.info("Final Loop Status: %s", loop_status)
    logger.info("Total Completed Designs: %d", completed_total)
    logger.info("Database File: %s", db_path)
    logger.info("Experiment Directory: %s", experiment_dir)
    logger.info("Total Swarm Tokens Consumed: %d", token_summary.get("overall", {}).get("total_tokens", 0))
    logger.info("Total Swarm Cost: $%.4f USD", token_summary.get("overall", {}).get("total_cost_usd", 0.0))

    return {
        "status": loop_status,
        "iterations": iteration,
        "completed_experiments": completed_total,
        "noise_model_progress": per_noise_summary,
        "elapsed_seconds": round(elapsed_s, 2),
        "database_path": str(db_path),
        "experiment_dir": str(experiment_dir),
        "report_tex_path": final_report_tex,
        "report_pdf_path": final_report_pdf,
        "profile_dir": str(profile_log_dir) if profile_collector_active else None,
        "token_usage": token_summary,
    }


def main() -> None:
    """Entrypoint for the unified loop runner."""
    parser = parse_args()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    summary = execute_loop(
        config_path=args.config,
        reports_base=args.reports_dir,
        custom_db_path=args.db_path,
        max_iterations_override=args.max_iterations,
        max_experiments_override=args.max_experiments,
        compile_pdf=not args.no_pdf,
        ray_address=args.ray_address,
        auto_init_ray=not args.no_ray_init,
        enable_profiling=not args.no_profile,
        profile_dir=args.profile_dir,
    )

    if summary["status"] in ("SUCCESS_REPORT_GENERATED", "MAX_ITERATIONS_REACHED"):
        sys.exit(0)
    elif summary["status"] in ("ERROR", "TERMINATED_MAX_CYCLES"):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
