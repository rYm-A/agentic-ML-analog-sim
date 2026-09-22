"""Node 9: Error Handler Node (Phase 1, Programmatic).

Standardized loop termination handler for counter exceedances and graceful hard failures.
Persists DB error state, creates forensic snapshot dumps, and signals terminal status.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional, Union

try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:
    def ChiaFunction(*args: Any, **kwargs: Any) -> Any:
        def decorator(f: Any) -> Any:
            return f
        return decorator

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node9_phase1_error_handler")


def _cleanup_node9_resources(*args: Any, **kwargs: Any) -> None:
    """Best-effort cleanup hook executed in worker teardown via ChiaFunction hooks."""
    logger.debug("Node 9: Worker cleanup completed.")


@ChiaFunction(num_cpus=0.1, max_retries=0, resources={"control_worker": 1})
def node9_phase1_error_handler(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    counter_name: str,
    current_count: int,
    max_count: int,
    execution_mode: str = "standard",
    error_stage: str = "unknown",
    details: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform a graceful hard failure and loop termination.

    1. Formats failure message and logs error (reason, problematic candidate, mode).
    2. Marks candidate status as "FAILED" in DB with error_stage and error_message using safe transaction concurrency.
    3. Performs SQLite WAL checkpoint maintenance (folding -wal sidecar) and creates a forensic snapshot dump.
    4. Attaches profiler metadata/events if profiling is active.
    5. Returns a standardized termination dict.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Candidate identifier that triggered failure.
        counter_name: Name of counter that exceeded limit (e.g., 'counter_1_compilation').
        current_count: Current count reached.
        max_count: Maximum allowed threshold for counter.
        execution_mode: Loop execution mode ('standard', 'reference', 'debug', etc.).
        error_stage: Pipeline stage where failure manifested.
        details: Optional additional failure context or traceback info.

    Returns:
        Standardized loop termination dictionary.
    """
    failure_reason = f"Counter '{counter_name}' exceeded maximum threshold ({current_count} > {max_count})"
    error_msg = f"{failure_reason}. Stage: '{error_stage}'. Mode: '{execution_mode}'. Details: {details or 'None'}"

    logger.error(
        "Node 9 Error Handler: Hard failure for candidate '%s' in mode '%s'. Reason: %s. Stage: '%s'. Details: %s",
        candidate_id,
        execution_mode,
        failure_reason,
        error_stage,
        details or "None",
    )

    # 1. Mark candidate status as FAILED in database with error_stage and error_message
    try:
        if hasattr(db, "mark_proposal_status") and callable(db.mark_proposal_status):
            db.mark_proposal_status(
                candidate_id=candidate_id,
                status="FAILED",
                error_stage=error_stage,
                error_message=error_msg,
                is_active_evaluation=0,
            )
        elif hasattr(db, "update_execution_status") and callable(db.update_execution_status):
            db.update_execution_status(
                candidate_id=candidate_id,
                error_stage=error_stage,
                error_message=error_msg,
            )
        elif isinstance(db, dict) and candidate_id in db:
            db[candidate_id]["status"] = "FAILED"
            db[candidate_id]["error_stage"] = error_stage
            db[candidate_id]["error_message"] = error_msg
    except Exception as exc:
        logger.warning("Node 9: Could not update candidate status in DB for %s: %s", candidate_id, exc)

    # 2. Store current state of database by creating snapshot / dump with WAL maintenance
    snapshot_path: Optional[str] = None
    try:
        db_path = getattr(db, "db_path", None)
        if db_path and str(db_path) != ":memory:":
            resolved_db_path = Path(db_path).resolve()
            snapshot_dir = resolved_db_path.parent / "snapshots"

            # WAL checkpoint maintenance: fold -wal sidecar into main DB file for consistent snapshot
            try:
                if resolved_db_path.is_file():
                    with sqlite3.connect(str(resolved_db_path), timeout=30.0) as maint_conn:
                        maint_conn.execute("PRAGMA busy_timeout = 30000;")
                        maint_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception as maint_exc:
                logger.warning("Node 9: PRAGMA wal_checkpoint maintenance encountered warning: %s", maint_exc)
        else:
            snapshot_dir = Path.cwd() / "snapshots"

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        target_snapshot = snapshot_dir / f"{candidate_id}_error_snapshot.db"

        if hasattr(db, "backup") and callable(db.backup):
            snapshot_path = str(db.backup(target_snapshot))
        elif db_path and Path(db_path).is_file():
            # Concurrency-safe backup with busy timeout
            src_conn = sqlite3.connect(str(db_path), timeout=30.0)
            src_conn.execute("PRAGMA busy_timeout = 30000;")
            try:
                dst_conn = sqlite3.connect(str(target_snapshot), timeout=30.0)
                try:
                    src_conn.backup(dst_conn)
                finally:
                    dst_conn.close()
            finally:
                src_conn.close()
            snapshot_path = str(target_snapshot)
        else:
            # Fallback for mock db or purely in-memory structures
            temp_snapshot = snapshot_dir / f"{candidate_id}_error_snapshot.json"
            state_data = {
                "candidate_id": candidate_id,
                "counter_name": counter_name,
                "current_count": current_count,
                "max_count": max_count,
                "execution_mode": execution_mode,
                "error_stage": error_stage,
                "details": details,
            }
            temp_snapshot.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
            snapshot_path = str(temp_snapshot)

        logger.info(
            "Node 9: Successfully created DB snapshot at '%s' for candidate '%s'",
            snapshot_path,
            candidate_id,
        )
    except Exception as exc:
        logger.error("Node 9: Failed to create database snapshot: %s", exc)
        snapshot_path = None

    # Profiler integration if Chia profiling is active
    try:
        from chia.trace.profiler import get_profiler
        profiler = get_profiler()
        if profiler.enabled:
            profiler.add_info({
                "termination_reason": failure_reason,
                "counter_name": counter_name,
                "current_count": current_count,
                "max_count": max_count,
                "error_stage": error_stage,
                "snapshot_path": snapshot_path,
            })
            profiler.log_event(
                "circuit_breaker_termination",
                candidate_id=candidate_id,
                counter_name=counter_name,
                error_stage=error_stage,
            )
    except Exception as prof_exc:
        logger.debug("Node 9: Profiler notification skipped or unavailable: %s", prof_exc)

    # 3. Return standardized termination dictionary
    result: Dict[str, Any] = {
        "status": "TERMINATED_MAX_CYCLES",
        "candidate_id": candidate_id,
        "counter_name": counter_name,
        "current_count": current_count,
        "max_count": max_count,
        "execution_mode": execution_mode,
        "db_snapshot": snapshot_path,
        "next_node": None,
        # Backward-compatibility / helper fields
        "reason": f"{counter_name}_exceeded",
        "error_stage": error_stage,
        "details": details,
    }

    if counter_name == "counter_1_compilation":
        result["counter_1"] = current_count
    elif counter_name == "counter_2_rewrite":
        result["counter_2"] = current_count
    elif counter_name == "counter_3_debug_gate":
        result["counter_3"] = current_count
    elif counter_name == "counter_4_debug_loop":
        result["counter_4"] = current_count

    return result


# Backward compatibility alias
node9_error_handler = node9_phase1_error_handler
