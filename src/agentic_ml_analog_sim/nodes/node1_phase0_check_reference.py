"""Node 1: Programmatic check for baseline reference design data completeness."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Mapping

from chia.base.ChiaFunction import ChiaFunction

logger = logging.getLogger("agentic_ml_analog_sim.node1")


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Extract attribute or dictionary key safely."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _find_reference_entry(db: Any, config: Any) -> Any:
    """Find reference design entry in DB across diverse DB interfaces or mocks."""
    # 1. Specialized helper method on ResultDatabase
    if hasattr(db, "get_reference_result") and callable(db.get_reference_result):
        try:
            res = db.get_reference_result()
            if res is not None:
                return res
        except Exception as exc:
            logger.debug("db.get_reference_result() raised: %s", exc)

    if hasattr(db, "get_reference") and callable(db.get_reference):
        try:
            res = db.get_reference()
            if res is not None:
                return res
        except Exception as exc:
            logger.debug("db.get_reference() raised: %s", exc)

    # 2. General query_proposals interface
    if hasattr(db, "query_proposals") and callable(db.query_proposals):
        try:
            results = db.query_proposals(is_reference=True)
            if results:
                # Return first matching reference proposal
                return results[0]
        except TypeError:
            try:
                # Fallback in case query_proposals has different signature
                results = db.query_proposals()
                for p in results or []:
                    if _get_val(p, "is_reference"):
                        return p
            except Exception as exc:
                logger.debug("db.query_proposals() raised: %s", exc)
        except Exception as exc:
            logger.debug("db.query_proposals(is_reference=True) raised: %s", exc)

    # 3. Direct SQLite connection access if present
    conn = getattr(db, "conn", None) or getattr(db, "connection", None) or getattr(db, "_conn", None)
    if conn is not None and isinstance(conn, sqlite3.Connection):
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT candidate_id, status, is_reference, relative_error, latency_ms, accuracy_fid
                FROM proposals
                WHERE is_reference = 1 OR is_reference = '1'
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row is not None:
                return dict(row)
        except Exception as exc:
            logger.debug("Direct SQLite query on proposals raised: %s", exc)

    # 4. Handle test mock structures (dict or list)
    if isinstance(db, dict):
        if "reference" in db:
            return db["reference"]
        for v in db.values():
            if isinstance(v, (dict, object)) and _get_val(v, "is_reference"):
                return v
    elif isinstance(db, list):
        for item in db:
            if _get_val(item, "is_reference"):
                return item

    return None


@ChiaFunction(resources={"control_worker": 1})
def node1_phase0_check_reference(db: Any, config: Any) -> bool:
    """Programmatically verify whether the reference design evaluation is complete in DB.

    Queries `db` for the reference design specified in `config.reference_design`.
    Verifies that a record exists with:
      - `status == 'COMPLETED'`
      - `is_reference == True` (or truthy)
      - `latency_ms is not None`

    Args:
        db: ResultDatabase instance (or compatible database/mock).
        config: SimulationConfig instance (or compatible configuration object).

    Returns:
        bool: True if reference design data is complete and valid (proceed to Node 2).
              False if reference data is missing or incomplete (route to Phase 1).
    """
    entry = _find_reference_entry(db, config)
    if entry is None:
        logger.info("Node 1: Reference design data not found in DB. Routing to Phase 1.")
        return False

    status = _get_val(entry, "status")
    is_ref = _get_val(entry, "is_reference")
    relative_error = _get_val(entry, "relative_error")
    latency_ms = _get_val(entry, "latency_ms")

    # Status must be strictly 'COMPLETED'
    if not status or str(status).strip().upper() != "COMPLETED":
        logger.info(
            "Node 1: Reference design entry present but status is '%s' (expected 'COMPLETED'). Routing to Phase 1.",
            status,
        )
        return False

    # is_reference must be True
    if not is_ref or str(is_ref).lower() in ("0", "false", "none", ""):
        logger.info("Node 1: Reference design flag is False or unset. Routing to Phase 1.")
        return False

    # Latency metric must be present
    if latency_ms is None:
        logger.info("Node 1: Reference design latency_ms is None. Routing to Phase 1.")
        return False

    logger.info(
        "Node 1: Reference design data complete (status=%s, relative_error=%s, latency_ms=%s). Proceeding to Node 2.",
        status,
        relative_error,
        latency_ms,
    )
    return True


# Backward compatibility aliases
check_reference_data = node1_phase0_check_reference
node1_check_reference = node1_phase0_check_reference

