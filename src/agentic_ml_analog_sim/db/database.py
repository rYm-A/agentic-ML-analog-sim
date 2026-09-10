"""SQLite Result Database CRUD API and Pareto front computation for analog ML simulation."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union
import uuid

from agentic_ml_analog_sim.config import SimulationConfig

VALID_STATUSES = {"PENDING", "EVALUATING", "COMPLETED", "FAILED", "REJECTED"}
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _canonical_json(val: Any) -> str:
    """Normalize dictionary or JSON string to a sorted, canonical JSON string."""
    if val is None:
        return "{}"
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True)
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, sort_keys=True)
        except Exception:
            return stripped
    return str(val)


class ResultDatabase:
    """SQLite-backed results database for analog accelerator simulation proposals and runs."""

    def __init__(self, db_path: Union[str, Path] = "results.db", auto_init: bool = True):
        self.db_path = Path(db_path).resolve()
        if auto_init:
            self.init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Provide an active connection for callers expecting db.conn attribute."""
        return self._get_connection()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize database schema from schema.sql with idempotent migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not SCHEMA_PATH.is_file():
            raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = self._get_connection()
        try:
            conn.executescript(schema_sql)
            conn.execute("PRAGMA journal_mode = WAL;")

            # Ensure idempotent column additions for existing Phase 0 databases
            cur = conn.execute("PRAGMA table_info(proposals);")
            existing_cols = {row["name"] for row in cur.fetchall()}
            new_columns = [
                ("rewrite_solver_status", "TEXT DEFAULT 'PENDING'"),
                ("rewrite_noise_status", "TEXT DEFAULT 'PENDING'"),
                ("rewrite_compiler_status", "TEXT DEFAULT 'PENDING'"),
                ("compilation_status", "TEXT DEFAULT 'PENDING'"),
                ("simulation_status", "TEXT DEFAULT 'PENDING'"),
                ("error_stage", "TEXT"),
                ("error_message", "TEXT"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE proposals ADD COLUMN {col_name} {col_type};")

            conn.commit()
        finally:
            conn.close()

    def has_complete_reference(self, config: Optional[SimulationConfig] = None) -> bool:
        """Check whether a completed reference design evaluation exists in the database.

        If a SimulationConfig is provided, also verifies matching reference solver and noise_model.
        """
        conn = self._get_connection()
        try:
            if config is not None:
                cur = conn.execute(
                    """
                    SELECT COUNT(*) FROM proposals
                    WHERE is_reference = 1
                      AND status = 'COMPLETED'
                      AND solver = ?
                      AND noise_model = ?
                    """,
                    (config.reference_design.solver, config.reference_design.noise_model),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT COUNT(*) FROM proposals
                    WHERE is_reference = 1
                      AND status = 'COMPLETED'
                    """
                )
            count = cur.fetchone()[0]
            return count > 0
        finally:
            conn.close()

    def insert_proposal(
        self,
        candidate_id: Optional[str] = None,
        iteration: int = 0,
        solver: str = "rk4",
        noise_model: str = "none",
        sparsity_config: Union[Dict[str, Any], str] = "{}",
        solver_params: Union[Dict[str, Any], str] = "{}",
        noise_params: Union[Dict[str, Any], str] = "{}",
        status: str = "PENDING",
        is_reference: bool = False,
        is_active_evaluation: bool = False,
        accuracy_fid: Optional[float] = None,
        relative_error: Optional[float] = None,
        latency_ms: Optional[float] = None,
        wall_clock_s: Optional[float] = None,
        worker_utilization: Optional[float] = None,
        pareto_optimal: bool = False,
        selection_reason: str = "",
        phase_1_prompt: str = "",
    ) -> str:
        """Insert a new proposal into the proposals table.

        Returns:
            The generated or provided candidate_id.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

        if candidate_id is None:
            prefix = "ref" if is_reference else f"prop_iter{iteration:02d}"
            candidate_id = f"{prefix}_{solver}_{uuid.uuid4().hex[:8]}"

        canon_sparsity = _canonical_json(sparsity_config)
        canon_solver_params = _canonical_json(solver_params)
        canon_noise_params = _canonical_json(noise_params)

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO proposals (
                    candidate_id, iteration, solver, noise_model,
                    sparsity_config, solver_params, noise_params,
                    status, is_reference, is_active_evaluation,
                    accuracy_fid, relative_error, latency_ms, wall_clock_s,
                    worker_utilization, pareto_optimal, selection_reason, phase_1_prompt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    iteration,
                    solver,
                    noise_model,
                    canon_sparsity,
                    canon_solver_params,
                    canon_noise_params,
                    status,
                    1 if is_reference else 0,
                    1 if is_active_evaluation else 0,
                    accuracy_fid,
                    relative_error,
                    latency_ms,
                    wall_clock_s,
                    worker_utilization,
                    1 if pareto_optimal else 0,
                    selection_reason,
                    phase_1_prompt,
                ),
            )
            conn.commit()
            return candidate_id
        finally:
            conn.close()

    def get_proposals(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        is_reference: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve proposals optionally filtered by status and reference flag."""
        return self.query_proposals(status=status, limit=limit, is_reference=is_reference)

    def query_proposals(
        self,
        is_reference: Optional[bool] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query proposals with optional reference and status filters."""
        conn = self._get_connection()
        try:
            clauses = []
            params: List[Any] = []
            if is_reference is not None:
                clauses.append("is_reference = ?")
                params.append(1 if is_reference else 0)
            if status is not None:
                clauses.append("status = ?")
                params.append(status)

            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(limit)

            cur = conn.execute(
                f"""
                SELECT * FROM proposals
                {where}
                ORDER BY iteration DESC, created_at DESC, rowid DESC
                LIMIT ?
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def get_proposal(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single proposal by candidate_id."""
        conn = self._get_connection()
        try:
            cur = conn.execute("SELECT * FROM proposals WHERE candidate_id = ?", (candidate_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_latest_proposal(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recently created or highest iteration proposal."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM proposals
                ORDER BY iteration DESC, created_at DESC, rowid DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_reference_proposal(self) -> Optional[Dict[str, Any]]:
        """Retrieve the baseline reference proposal (preferring COMPLETED)."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM proposals
                WHERE is_reference = 1
                ORDER BY (CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) DESC,
                         created_at DESC,
                         rowid DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_reference_result(self) -> Optional[Dict[str, Any]]:
        """Alias for get_reference_proposal."""
        return self.get_reference_proposal()

    def get_reference(self) -> Optional[Dict[str, Any]]:
        """Alias for get_reference_proposal."""
        return self.get_reference_proposal()

    def is_duplicate_proposal(
        self,
        solver: str,
        noise_model: str,
        solver_params: Union[Dict[str, Any], str],
        noise_params: Union[Dict[str, Any], str],
        sparsity_config: Union[Dict[str, Any], str],
    ) -> bool:
        """Check if an existing proposal with identical solver, noise model, and configs exists."""
        canon_sparsity = _canonical_json(sparsity_config)
        canon_solver_params = _canonical_json(solver_params)
        canon_noise_params = _canonical_json(noise_params)

        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT sparsity_config, solver_params, noise_params
                FROM proposals
                WHERE solver = ? AND noise_model = ?
                """,
                (solver, noise_model),
            )
            rows = cur.fetchall()
            for r in rows:
                if (
                    _canonical_json(r["sparsity_config"]) == canon_sparsity
                    and _canonical_json(r["solver_params"]) == canon_solver_params
                    and _canonical_json(r["noise_params"]) == canon_noise_params
                ):
                    return True
            return False
        finally:
            conn.close()

    def add_comment(self, candidate_id: str, phase: str, agent_name: str, comment: str) -> int:
        """Add an agent commentary or review note to a candidate proposal."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                INSERT INTO comments (candidate_id, phase, agent_name, comment)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, phase, agent_name, comment),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_comments(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Retrieve all comments associated with candidate_id."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM comments
                WHERE candidate_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (candidate_id,),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def compute_pareto_front(self) -> List[Dict[str, Any]]:
        """Compute the Pareto front among COMPLETED evaluations and update pareto_optimal flags in DB.

        Objectives minimized:
            1. Error / quality: accuracy_fid (or relative_error if accuracy_fid is null)
            2. Cost / runtime: latency_ms (or wall_clock_s if latency_ms is null)

        Returns:
            List of Pareto-optimal proposal dicts.
        """
        conn = self._get_connection()
        try:
            cur = conn.execute("SELECT * FROM proposals WHERE status = 'COMPLETED'")
            completed = [dict(row) for row in cur.fetchall()]
            if not completed:
                return []

            # Extract objective vector (error, cost) for each completed candidate
            candidates_with_objs = []
            for item in completed:
                err = item["accuracy_fid"] if item["accuracy_fid"] is not None else item["relative_error"]
                cost = item["latency_ms"] if item["latency_ms"] is not None else item["wall_clock_s"]
                if err is not None and cost is not None:
                    candidates_with_objs.append((item, float(err), float(cost)))

            if not candidates_with_objs:
                return []

            pareto_front = []
            for item_a, err_a, cost_a in candidates_with_objs:
                is_dominated = False
                for item_b, err_b, cost_b in candidates_with_objs:
                    if item_a["candidate_id"] == item_b["candidate_id"]:
                        continue
                    # B dominates A if B is no worse than A in both objectives and strictly better in at least one
                    if err_b <= err_a and cost_b <= cost_a and (err_b < err_a or cost_b < cost_a):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_front.append(item_a)

            pareto_ids = {p["candidate_id"] for p in pareto_front}

            # Update DB flags
            conn.execute("UPDATE proposals SET pareto_optimal = 0 WHERE status = 'COMPLETED'")
            if pareto_ids:
                placeholders = ",".join("?" for _ in pareto_ids)
                conn.execute(
                    f"UPDATE proposals SET pareto_optimal = 1 WHERE candidate_id IN ({placeholders})",
                    list(pareto_ids),
                )
            conn.commit()

            # Refresh and return updated records
            for p in pareto_front:
                p["pareto_optimal"] = 1
            return pareto_front
        finally:
            conn.close()

    def mark_proposal_status(
        self,
        candidate_id: str,
        status: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update proposal status and optionally other metrics/fields.

        Supported extra fields: accuracy_fid, relative_error, latency_ms,
        wall_clock_s, worker_utilization, pareto_optimal, is_active_evaluation,
        selection_reason, phase_1_prompt, iteration, solver_params, noise_params,
        sparsity_config.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

        allowed_fields = {
            "accuracy_fid",
            "relative_error",
            "latency_ms",
            "wall_clock_s",
            "worker_utilization",
            "pareto_optimal",
            "is_active_evaluation",
            "selection_reason",
            "phase_1_prompt",
            "iteration",
            "solver_params",
            "noise_params",
            "sparsity_config",
            "rewrite_solver_status",
            "rewrite_noise_status",
            "rewrite_compiler_status",
            "compilation_status",
            "simulation_status",
            "error_stage",
            "error_message",
        }

        update_cols = ["status = ?"]
        values: List[Any] = [status]

        for k, v in kwargs.items():
            if k in allowed_fields:
                if k in {"solver_params", "noise_params", "sparsity_config"}:
                    v = _canonical_json(v)
                elif k in {"pareto_optimal", "is_active_evaluation"}:
                    v = 1 if v else 0
                update_cols.append(f"{k} = ?")
                values.append(v)

        values.append(candidate_id)

        conn = self._get_connection()
        try:
            sql = f"UPDATE proposals SET {', '.join(update_cols)} WHERE candidate_id = ?"
            cur = conn.execute(sql, values)
            if cur.rowcount == 0:
                raise KeyError(f"Proposal with candidate_id '{candidate_id}' not found.")
            conn.commit()

            cur = conn.execute("SELECT * FROM proposals WHERE candidate_id = ?", (candidate_id,))
            row = cur.fetchone()
            return dict(row)
        finally:
            conn.close()

    def update_proposal_status(
        self,
        candidate_id: str,
        new_status: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Alias for mark_proposal_status for seamless node integration."""
        return self.mark_proposal_status(candidate_id, new_status, **kwargs)

    def record_token_usage(
        self,
        phase: str,
        node_name: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        candidate_id: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> int:
        """Record token counts and estimated cost for an LLM execution.

        Returns:
            Inserted token_usage row id.
        """
        conn = self._get_connection()
        try:
            if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
                total_tokens = prompt_tokens + completion_tokens

            cur = conn.execute(
                """
                INSERT INTO token_usage (
                    candidate_id, phase, node_name, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost_usd, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    phase,
                    node_name,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cost_usd,
                    duration_seconds,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_token_usage_summary(self, phase: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate total tokens and cost across phases, nodes, and models."""
        conn = self._get_connection()
        try:
            where_clause = "WHERE phase = ?" if phase else ""
            params = (phase,) if phase else ()

            cur = conn.execute(
                f"""
                SELECT 
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                    COUNT(*) AS total_calls
                FROM token_usage
                {where_clause}
                """,
                params,
            )
            overall = dict(cur.fetchone())

            # Breakdown by phase
            cur = conn.execute(
                """
                SELECT phase, 
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(total_tokens) AS total_tokens,
                       SUM(cost_usd) AS cost_usd,
                       COUNT(*) AS num_calls
                FROM token_usage
                GROUP BY phase
                """
            )
            by_phase = {row["phase"]: dict(row) for row in cur.fetchall()}

            # Breakdown by model
            cur = conn.execute(
                """
                SELECT model, 
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(total_tokens) AS total_tokens,
                       SUM(cost_usd) AS cost_usd,
                       COUNT(*) AS num_calls
                FROM token_usage
                GROUP BY model
                """
            )
            by_model = {row["model"]: dict(row) for row in cur.fetchall()}

            # Breakdown by node
            cur = conn.execute(
                """
                SELECT node_name, phase,
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(total_tokens) AS total_tokens,
                       SUM(cost_usd) AS cost_usd,
                       COUNT(*) AS num_calls
                FROM token_usage
                GROUP BY node_name, phase
                """
            )
            by_node = {f"{row['phase']}.{row['node_name']}": dict(row) for row in cur.fetchall()}

            return {
                "overall": overall,
                "by_phase": by_phase,
                "by_model": by_model,
                "by_node": by_node,
            }
        finally:
            conn.close()

    def get_completed_experiments_count(self) -> int:
        """Count evaluated candidate experiments (completed or failed, non-reference)."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT COUNT(*) FROM proposals
                WHERE is_reference = 0
                  AND status IN ('COMPLETED', 'FAILED')
                """
            )
            return cur.fetchone()[0]
        finally:
            conn.close()

    def update_execution_status(
        self,
        candidate_id: str,
        rewrite_solver_status: Optional[str] = None,
        rewrite_noise_status: Optional[str] = None,
        rewrite_compiler_status: Optional[str] = None,
        compilation_status: Optional[str] = None,
        simulation_status: Optional[str] = None,
        error_stage: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update granular compilation, simulation, and rewrite statuses."""
        updates: Dict[str, Any] = {}
        if rewrite_solver_status is not None:
            updates["rewrite_solver_status"] = rewrite_solver_status
        if rewrite_noise_status is not None:
            updates["rewrite_noise_status"] = rewrite_noise_status
        if rewrite_compiler_status is not None:
            updates["rewrite_compiler_status"] = rewrite_compiler_status
        if compilation_status is not None:
            updates["compilation_status"] = compilation_status
        if simulation_status is not None:
            updates["simulation_status"] = simulation_status
        if error_stage is not None:
            updates["error_stage"] = error_stage
        if error_message is not None:
            updates["error_message"] = error_message

        if not updates:
            return self.get_proposal(candidate_id)

        conn = self._get_connection()
        try:
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [candidate_id]
            cur = conn.execute(f"UPDATE proposals SET {set_clause} WHERE candidate_id = ?", values)
            if cur.rowcount == 0:
                raise KeyError(f"Proposal '{candidate_id}' not found.")
            conn.commit()
            return self.get_proposal(candidate_id)
        finally:
            conn.close()

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Return breakdown by candidate and in total of succeeded vs failed compilations,
        simulations, and rewrites.
        """
        conn = self._get_connection()
        try:
            cur = conn.execute("SELECT * FROM proposals WHERE is_reference = 0")
            proposals = [dict(row) for row in cur.fetchall()]

            stats: Dict[str, Dict[str, List[str]]] = {
                "compilations": {"succeeded": [], "failed": [], "pending": []},
                "simulations": {"succeeded": [], "failed": [], "pending": []},
                "rewrites_solver": {"succeeded": [], "failed": [], "pending": []},
                "rewrites_noise": {"succeeded": [], "failed": [], "pending": []},
                "rewrites_compiler": {"succeeded": [], "failed": [], "pending": []},
            }

            for p in proposals:
                cid = p["candidate_id"]
                # Compilation
                c_stat = (p.get("compilation_status") or "PENDING").upper()
                if c_stat == "SUCCESS":
                    stats["compilations"]["succeeded"].append(cid)
                elif c_stat == "FAILED":
                    stats["compilations"]["failed"].append(cid)
                else:
                    stats["compilations"]["pending"].append(cid)

                # Simulation
                s_stat = (p.get("simulation_status") or "PENDING").upper()
                if s_stat == "SUCCESS":
                    stats["simulations"]["succeeded"].append(cid)
                elif s_stat == "FAILED":
                    stats["simulations"]["failed"].append(cid)
                else:
                    stats["simulations"]["pending"].append(cid)

                # Rewrites
                rw_s = (p.get("rewrite_solver_status") or "PENDING").upper()
                if rw_s == "SUCCESS":
                    stats["rewrites_solver"]["succeeded"].append(cid)
                elif rw_s == "FAILED":
                    stats["rewrites_solver"]["failed"].append(cid)
                else:
                    stats["rewrites_solver"]["pending"].append(cid)

                rw_n = (p.get("rewrite_noise_status") or "PENDING").upper()
                if rw_n == "SUCCESS":
                    stats["rewrites_noise"]["succeeded"].append(cid)
                elif rw_n == "FAILED":
                    stats["rewrites_noise"]["failed"].append(cid)
                else:
                    stats["rewrites_noise"]["pending"].append(cid)

                rw_c = (p.get("rewrite_compiler_status") or "PENDING").upper()
                if rw_c == "SUCCESS":
                    stats["rewrites_compiler"]["succeeded"].append(cid)
                elif rw_c == "FAILED":
                    stats["rewrites_compiler"]["failed"].append(cid)
                else:
                    stats["rewrites_compiler"]["pending"].append(cid)

            return {
                "total_candidates": len(proposals),
                "compilations_succeeded_count": len(stats["compilations"]["succeeded"]),
                "compilations_failed_count": len(stats["compilations"]["failed"]),
                "simulations_succeeded_count": len(stats["simulations"]["succeeded"]),
                "simulations_failed_count": len(stats["simulations"]["failed"]),
                "rewrites_solver_succeeded_count": len(stats["rewrites_solver"]["succeeded"]),
                "rewrites_solver_failed_count": len(stats["rewrites_solver"]["failed"]),
                "rewrites_noise_succeeded_count": len(stats["rewrites_noise"]["succeeded"]),
                "rewrites_noise_failed_count": len(stats["rewrites_noise"]["failed"]),
                "rewrites_compiler_succeeded_count": len(stats["rewrites_compiler"]["succeeded"]),
                "rewrites_compiler_failed_count": len(stats["rewrites_compiler"]["failed"]),
                "details": stats,
            }
        finally:
            conn.close()
