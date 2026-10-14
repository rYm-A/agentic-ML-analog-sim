#!/usr/bin/env python3
"""Backfill script to populate SQLite sensitivity_lookup table from JSON artifacts.

Usage:
    python scripts/backfill_sensitivity_lut.py [--json-path PATH] [--db-path PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sqlite3
import sys

# Add src to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.accuracy.sensitivity_lookup import SensitivityLookupTable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_sensitivity_lut")


def backfill_lut(json_path: Path, db_path: Path) -> int:
    if not json_path.exists():
        logger.warning("JSON file does not exist: %s", json_path)
        return 0

    if not db_path.exists():
        logger.warning("Database file does not exist: %s", db_path)
        return 0

    logger.info("Loading SensitivityLookupTable from %s", json_path)
    table = SensitivityLookupTable.load_from_file(json_path)

    logger.info(
        "Loaded %d entries for reference_id='%s', experiment_name='%s'",
        len(table),
        table.reference_id,
        table.experiment_name,
    )

    db = ResultDatabase(db_path=db_path)
    db.init_db()

    # Check if reference_id exists in proposals
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT candidate_id FROM proposals WHERE candidate_id = ?", (table.reference_id,))
        row = cur.fetchone()
        if not row:
            logger.warning(
                "Reference ID '%s' not found in proposals table of %s. Checking foreign key constraint...",
                table.reference_id,
                db_path,
            )
            # Create minimal reference proposal entry if missing so foreign key passes
            conn.execute(
                """
                INSERT OR IGNORE INTO proposals (candidate_id, solver, noise_model, is_reference, status)
                VALUES (?, 'rk4', 'none', 1, 'COMPLETED')
                """,
                (table.reference_id,),
            )
            conn.commit()
            logger.info("Inserted placeholder reference proposal for '%s'", table.reference_id)
    finally:
        conn.close()

    inserted = db.record_sensitivity_curve(
        reference_id=table.reference_id,
        curve=table.entries,
        experiment_name=table.experiment_name,
    )
    logger.info("Successfully recorded %d points into %s", inserted, db_path)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill sensitivity_lookup table in SQLite from JSON.")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=repo_root / "reports" / "un0_kuramoto_analog_acceleration" / "sensitivity_lookup.json",
        help="Path to sensitivity_lookup.json",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=repo_root / "reports" / "un0_kuramoto_analog_acceleration" / "results.db",
        help="Path to target results.db",
    )
    args = parser.parse_args()

    count = backfill_lut(args.json_path, args.db_path)
    print(f"Backfill complete: {count} points synchronized.")


if __name__ == "__main__":
    main()
