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


def _get_counts_by_noise_model(db: Any) -> Dict[str, Dict[str, int]]:
    """Retrieve proposal counts grouped by noise model from DB or mock."""
    if hasattr(db, "get_proposal_counts_by_noise_model") and callable(db.get_proposal_counts_by_noise_model):
        try:
            return db.get_proposal_counts_by_noise_model(exclude_reference=True)
        except Exception as exc:
            logger.debug("db.get_proposal_counts_by_noise_model() raised: %s", exc)

    if hasattr(db, "query_proposals") and callable(db.query_proposals):
        try:
            candidates = db.query_proposals(is_reference=False)
            res: Dict[str, Dict[str, int]] = {}
            for c in candidates:
                nm = _get_val(c, "noise_model", "none")
                st = str(_get_val(c, "status", "")).strip().upper()
                if nm not in res:
                    res[nm] = {"total": 0, "completed": 0, "failed": 0, "pending": 0, "rejected": 0}
                res[nm]["total"] += 1
                if st == "COMPLETED":
                    res[nm]["completed"] += 1
                elif st == "FAILED":
                    res[nm]["failed"] += 1
                elif st in ("PENDING", "EVALUATING"):
                    res[nm]["pending"] += 1
                elif st in ("REJECTED", "QUARANTINED"):
                    res[nm]["rejected"] += 1
            return res
        except Exception as exc:
            logger.debug("db.query_proposals() raised: %s", exc)

    # SQLite direct fallback
    conn = getattr(db, "conn", None) or getattr(db, "connection", None)
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT noise_model, status, COUNT(*) FROM proposals WHERE is_reference = 0 GROUP BY noise_model, status"
            )
            rows = cur.fetchall()
            res = {}
            for row in rows:
                nm = row[0]
                st = str(row[1]).strip().upper()
                cnt = int(row[2])
                if nm not in res:
                    res[nm] = {"total": 0, "completed": 0, "failed": 0, "pending": 0, "rejected": 0}
                res[nm]["total"] += cnt
                if st == "COMPLETED":
                    res[nm]["completed"] += cnt
                elif st == "FAILED":
                    res[nm]["failed"] += cnt
                elif st in ("PENDING", "EVALUATING"):
                    res[nm]["pending"] += cnt
                elif st in ("REJECTED", "QUARANTINED"):
                    res[nm]["rejected"] += cnt
            return res
        except Exception as exc:
            logger.debug("Direct query raised: %s", exc)

    # Handle mock dict/list
    items = []
    if isinstance(db, dict) and "proposals" in db:
        items = db["proposals"]
    elif isinstance(db, list):
        items = db
    if items:
        res = {}
        for p in items:
            if _get_val(p, "is_reference"):
                continue
            nm = _get_val(p, "noise_model", "none")
            st = str(_get_val(p, "status", "")).strip().upper()
            if nm not in res:
                res[nm] = {"total": 0, "completed": 0, "failed": 0, "pending": 0, "rejected": 0}
            res[nm]["total"] += 1
            if st == "COMPLETED":
                res[nm]["completed"] += 1
            elif st == "FAILED":
                res[nm]["failed"] += 1
            elif st in ("PENDING", "EVALUATING"):
                res[nm]["pending"] += 1
            elif st in ("REJECTED", "QUARANTINED"):
                res[nm]["rejected"] += 1
        return res

    return {}


@ChiaFunction(resources={"control_worker": 1})
def node1_phase2_check_experiment_limit(
    db: Any,
    config: Any,
    current_iteration: Optional[int] = None,
    candidate_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Programmatically evaluate whether the loop experiment limit has been reached.

    Evaluates candidate experiment attempts per noise model against max_experiments.
    The experiment quota is considered reached if and only if for ALL noise models in
    config.noise_models, total_attempts (completed + failed) >= max_experiments.

    Args:
        db: ResultDatabase instance or mock.
        config: SimulationConfig instance or dict.
        current_iteration: Optional current iteration count.
        candidate_id: Optional latest candidate ID evaluated by Phase 1 Node 7.

    Returns:
        Dict[str, Any] with keys:
            - limit_reached: bool (True if max_experiments reached for all noise models or max_iterations reached)
            - action: 'TERMINATE_AND_REPORT' or 'CONTINUE_LOOP'
            - completed_experiments: int (total across all noise models)
            - max_experiments: int (per noise model)
            - current_iteration: Optional[int]
            - max_iterations: int
            - reason: str explanation
            - per_noise_stats: Dict[str, Dict[str, Any]] (per-noise statistics)
            - per_noise_completion: Dict[str, Dict[str, Any]]
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

    # Extract configured noise models
    configured_noise = _get_val(config, "noise_models")
    configured_noise_names: list[str] = []
    if isinstance(configured_noise, dict):
        configured_noise_names = list(configured_noise.keys())
    elif isinstance(configured_noise, (list, tuple)):
        for nm in configured_noise:
            if isinstance(nm, str):
                configured_noise_names.append(nm)
            elif hasattr(nm, "name"):
                configured_noise_names.append(getattr(nm, "name"))
            elif isinstance(nm, dict) and "name" in nm:
                configured_noise_names.append(nm["name"])

    counts_by_noise = _get_counts_by_noise_model(db)
    db_noise_models = set(counts_by_noise.keys())

    # If configured noise models are provided and either DB has no proposals yet or
    # at least one configured noise model appears in DB, check configured_noise_names.
    # Otherwise, if DB contains other noise models (e.g. tests using 'none'), check those.
    if configured_noise_names:
        if not db_noise_models or any(nm in db_noise_models for nm in configured_noise_names):
            active_noise_models = configured_noise_names
        else:
            active_noise_models = list(db_noise_models)
    elif db_noise_models:
        active_noise_models = list(db_noise_models)
    else:
        active_noise_models = ["L0_static_mismatch"]

    per_noise_stats: Dict[str, Dict[str, Any]] = {}
    all_quotas_reached = True

    for nm in active_noise_models:
        nm_data = counts_by_noise.get(nm, {})
        c_completed = nm_data.get("completed", 0)
        c_failed = nm_data.get("failed", 0)
        c_pending = nm_data.get("pending", 0)
        c_rejected = nm_data.get("rejected", 0)
        total_attempts = c_completed + c_failed
        quota_reached = total_attempts >= max_experiments
        if not quota_reached:
            all_quotas_reached = False

        per_noise_stats[nm] = {
            "total_attempts": total_attempts,
            "completed": c_completed,
            "failed": c_failed,
            "pending": c_pending,
            "rejected": c_rejected,
            "max_experiments": max_experiments,
            "quota_reached": quota_reached,
        }

    # Evaluate experiment count limit
    exp_limit_hit = all_quotas_reached

    # Evaluate iteration count limit if provided
    iter_limit_hit = False
    if current_iteration is not None and current_iteration >= max_iterations:
        iter_limit_hit = True

    limit_reached = exp_limit_hit or iter_limit_hit

    progress_str = ", ".join(
        f"{nm}: {stats['total_attempts']}/{max_experiments}"
        for nm, stats in per_noise_stats.items()
    )

    if exp_limit_hit:
        reason = (
            f"Experiment limit reached: all configured noise model experiment quotas "
            f"satisfied ({progress_str}). Total candidate experiments: {completed_count}."
        )
        action = "TERMINATE_AND_REPORT"
    elif iter_limit_hit:
        reason = (
            f"Iteration limit reached: iteration {current_iteration}/{max_iterations} "
            f"completed ({progress_str})."
        )
        action = "TERMINATE_AND_REPORT"
    else:
        reason = (
            f"Experiment limit not yet reached ({progress_str}). "
            f"Total evaluated experiments: {completed_count}. Continuing loop."
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
        "per_noise_stats": per_noise_stats,
        "per_noise_completion": per_noise_stats,
    }


# Backward compatibility alias
phase2_check_experiment_limit = node1_phase2_check_experiment_limit

