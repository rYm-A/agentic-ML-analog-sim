#!/usr/bin/env python3
"""Sanitize results database by transitioning stale PENDING proposals to deterministic FAILED terminal states."""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanitize_db")


def sanitize_database(db_path: Path) -> int:
    if not db_path.is_file():
        logger.error("Database file not found: %s", db_path)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # 1. Inspect proposals currently in PENDING state
        cur.execute(
            """
            SELECT candidate_id, status, error_stage, error_message,
                   counter_1_compilation, counter_2_rewrite, counter_3_debug_gate, counter_4_debug_loop
            FROM proposals
            WHERE status = 'PENDING'
            """
        )
        pending_rows = cur.fetchall()
        logger.info("Found %d proposals in 'PENDING' state in %s", len(pending_rows), db_path)

        if not pending_rows:
            logger.info("No PENDING proposals to sanitize.")
            return 0

        updated_count = 0
        for row in pending_rows:
            cid = row["candidate_id"]
            c1 = row["counter_1_compilation"] or 0
            c2 = row["counter_2_rewrite"] or 0
            c3 = row["counter_3_debug_gate"] or 0
            c4 = row["counter_4_debug_loop"] or 0
            stage = row["error_stage"]
            msg = row["error_message"]

            if c1 > 3 or c2 > 3 or c3 > 2 or c4 > 5 or stage is not None:
                # Lowering or execution failure that exceeded retry counters
                target_stage = stage or ("compilation" if c1 > 3 else "rewrite")
                target_msg = msg or f"Exceeded retry threshold (c1={c1}, c2={c2}, c3={c3}, c4={c4})"
                cur.execute(
                    """
                    UPDATE proposals
                    SET status = 'FAILED',
                        error_stage = ?,
                        error_message = ?,
                        is_active_evaluation = 0
                    WHERE candidate_id = ?
                    """,
                    (target_stage, target_msg, cid),
                )
                logger.info("Transitioned %s -> FAILED (stage=%s, c1=%d, c2=%d)", cid, target_stage, c1, c2)
                updated_count += 1
            else:
                # Interrupted / abandoned candidate
                cur.execute(
                    """
                    UPDATE proposals
                    SET status = 'FAILED',
                        error_stage = 'interrupted_session',
                        error_message = 'Session interrupted before Phase 1 evaluation completed',
                        is_active_evaluation = 0
                    WHERE candidate_id = ?
                    """,
                    (cid,),
                )
                logger.info("Transitioned %s -> FAILED (interrupted session)", cid)
                updated_count += 1

        conn.commit()
        logger.info("Successfully sanitized %d proposals in database.", updated_count)

        # Print final distribution of statuses
        cur.execute("SELECT status, count(*) as count FROM proposals GROUP BY status")
        summary = cur.fetchall()
        logger.info("Final proposal status distribution:")
        for s in summary:
            logger.info("  %s: %d", s["status"], s["count"])

        return 0
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Sanitize results.db by transitioning stale PENDING proposals.")
    default_db = Path(__file__).resolve().parent.parent / "reports" / "first_run_trial_un0_analog_acceleration" / "results.db"
    parser.add_argument("--db-path", type=Path, default=default_db, help="Path to results.db file")
    args = parser.parse_args()

    sys.exit(sanitize_database(args.db_path))


if __name__ == "__main__":
    main()
