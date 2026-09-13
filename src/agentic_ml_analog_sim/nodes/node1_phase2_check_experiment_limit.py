"""Phase 2 Node 1: Programmatic check for experiment and iteration limits."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from chia.base.ChiaFunction import ChiaFunction

logger = logging.getLogger("agentic_ml_analog_sim.phase2_node1")


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Extract attribute or dictionary key safely."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_experiment_limits(config: Any) -> tuple[int, int]:
    """Extract (max_experiments, max_iterations) from config or config dict."""
    limits = _get_val(config, "experiment_limits") or _get_val(config, "budget")
    max_exp = _get_val(limits, "max_experiments", 10)
    max_iter = _get_val(limits, "max_iterations", 10)
    try:
        max_exp = int(max_exp)
    except (TypeError, ValueError):
        max_exp = 10
    try:
        max_iter = int(max_iter)
    except (TypeError, ValueError):
        max_iter = 10
    return max_exp, max_iter


def _get_completed_count(db: Any) -> int:
    """Retrieve count of completed candidate experiments from DB or mock."""
    if hasattr(db, "get_completed_experiments_count") and callable(db.get_completed_experiments_count):
        try:
            return int(db.get_completed_experiments_count())
        except Exception as exc:
            logger.debug("db.get_completed_experiments_count() raised: %s", exc)

    if hasattr(db, "query_proposals") and callable(db.query_proposals):
        try:
            candidates = db.query_proposals(is_reference=False)
            return sum(
                1 for c in candidates
                if str(_get_val(c, "status", "")).strip().upper() in ("COMPLETED", "FAILED")
            )
        except Exception as exc:
            logger.debug("db.query_proposals(is_reference=False) raised: %s", exc)

    # SQLite direct fallback
    conn = getattr(db, "conn", None) or getattr(db, "connection", None)
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM proposals WHERE is_reference = 0 AND status IN ('COMPLETED', 'FAILED')"
            )
            return cur.fetchone()[0]
        except Exception as exc:
            logger.debug("Direct count query raised: %s", exc)

    # Handle mock dict/list
    if isinstance(db, dict) and "proposals" in db:
        return sum(
            1 for p in db["proposals"]
            if not _get_val(p, "is_reference") and str(_get_val(p, "status", "")).upper() in ("COMPLETED", "FAILED")
        )
    elif isinstance(db, list):
        return sum(
            1 for p in db
            if not _get_val(p, "is_reference") and str(_get_val(p, "status", "")).upper() in ("COMPLETED", "FAILED")
        )

    return 0


@ChiaFunction()
def node1_phase2_check_experiment_limit(
    db: Any,
    config: Any,
    current_iteration: Optional[int] = None,
    candidate_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Programmatically evaluate whether the loop experiment limit has been reached.

    Evaluates candidates from Phase 1 Node 7 against the configured limit
    (default: 10 completed candidate experiments).

    Args:
        db: ResultDatabase instance or mock.
        config: SimulationConfig instance or dict.
        current_iteration: Optional current iteration count.
        candidate_id: Optional latest candidate ID evaluated by Phase 1 Node 7.

    Returns:
        Dict[str, Any] with keys:
            - limit_reached: bool (True if max_experiments or max_iterations reached)
            - action: 'TERMINATE_AND_REPORT' or 'CONTINUE_LOOP'
            - completed_experiments: int
            - max_experiments: int
            - current_iteration: Optional[int]
            - max_iterations: int
            - reason: str explanation
    """
    max_experiments, max_iterations = _get_experiment_limits(config)
    completed_count = _get_completed_count(db)

    # Check candidate status if candidate_id provided
    candidate_info = None
    if candidate_id and hasattr(db, "get_proposal") and callable(db.get_proposal):
        try:
            candidate_info = db.get_proposal(candidate_id)
        except Exception as exc:
            logger.debug("Failed to fetch proposal %s: %s", candidate_id, exc)

    # Evaluate experiment count limit
    exp_limit_hit = completed_count >= max_experiments

    # Evaluate iteration count limit if provided
    iter_limit_hit = False
    if current_iteration is not None and current_iteration >= max_iterations:
        iter_limit_hit = True

    limit_reached = exp_limit_hit or iter_limit_hit

    if exp_limit_hit:
        reason = (
            f"Experiment limit reached: {completed_count}/{max_experiments} "
            "candidate experiments evaluated."
        )
        action = "TERMINATE_AND_REPORT"
    elif iter_limit_hit:
        reason = (
            f"Iteration limit reached: iteration {current_iteration}/{max_iterations} "
            f"completed ({completed_count} experiments evaluated)."
        )
        action = "TERMINATE_AND_REPORT"
    else:
        remaining = max_experiments - completed_count
        reason = (
            f"Experiment limit not yet reached: {completed_count}/{max_experiments} "
            f"completed ({remaining} remaining). Continuing loop."
        )
        action = "CONTINUE_LOOP"

    logger.info("Phase 2 Node 1 result: action=%s, reason=%s", action, reason)

    return {
        "limit_reached": limit_reached,
        "action": action,
        "completed_experiments": completed_count,
        "max_experiments": max_experiments,
        "current_iteration": current_iteration,
        "max_iterations": max_iterations,
        "candidate_id": candidate_id,
        "candidate_info": candidate_info,
        "reason": reason,
    }


# Backward compatibility alias
phase2_check_experiment_limit = node1_phase2_check_experiment_limit

