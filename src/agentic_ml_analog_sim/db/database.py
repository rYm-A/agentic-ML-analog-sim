"""SQLite Result Database CRUD API and Pareto front computation for analog ML simulation."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union
import uuid

from agentic_ml_analog_sim.config import SimulationConfig

VALID_STATUSES = {"PENDING", "EVALUATING", "COMPLETED", "FAILED", "REJECTED", "QUARANTINED"}
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

    def _resolve_db_path(self) -> Path:
        if self.db_path.exists():
            return self.db_path
        # Check if inside container where /workspace is the repo mount
        if Path("/workspace").is_dir():
            parts = self.db_path.parts
            if "reports" in parts:
                idx = parts.index("reports")
                cand = Path("/workspace").joinpath(*parts[idx:])
                if cand.exists() or cand.parent.exists():
                    self.db_path = cand
                    return cand
            cand = Path("/workspace") / self.db_path.name
            if cand.exists():
                self.db_path = cand
                return cand
        return self.db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection with WAL mode and foreign keys enabled."""
        resolved = self._resolve_db_path()
        conn = sqlite3.connect(str(resolved), timeout=30.0)
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
                ("absolute_error", "REAL"),
                ("is_debug", "BOOLEAN DEFAULT 0"),
                ("debug_parent_id", "TEXT"),
                ("counter_1_compilation", "INT DEFAULT 0"),
                ("counter_2_rewrite", "INT DEFAULT 0"),
                ("counter_3_debug_gate", "INT DEFAULT 0"),
                ("counter_4_debug_loop", "INT DEFAULT 0"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE proposals ADD COLUMN {col_name} {col_type};")

            conn.commit()
        finally:
            conn.close()

    def backup(self, target_path: Optional[Union[str, Path]] = None) -> str:
        """Create a consistent SQLite backup/snapshot of the database.

        Args:
            target_path: Optional destination file path. If omitted, generates a file in
                snapshots/ relative to db_path.

        Returns:
            str: Absolute path to the created backup snapshot file.
        """
        if target_path is None:
            base_dir = self.db_path.parent if self.db_path and str(self.db_path) != ":memory:" else Path.cwd()
            snapshots_dir = base_dir / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            target_path = snapshots_dir / f"backup_{uuid.uuid4().hex[:8]}.db"
        else:
            target_path = Path(target_path).resolve()
            target_path.parent.mkdir(parents=True, exist_ok=True)

        src_conn = self._get_connection()
        try:
            dst_conn = sqlite3.connect(str(target_path))
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()
        return str(target_path)

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
        absolute_error: Optional[float] = None,
        latency_ms: Optional[float] = None,
        wall_clock_s: Optional[float] = None,
        worker_utilization: Optional[float] = None,
        pareto_optimal: bool = False,
        selection_reason: str = "",
        phase_1_prompt: str = "",
        is_debug: bool = False,
        debug_parent_id: Optional[str] = None,
        counter_1_compilation: int = 0,
        counter_2_rewrite: int = 0,
        counter_3_debug_gate: int = 0,
        counter_4_debug_loop: int = 0,
    ) -> str:
        """Insert a new proposal into the proposals table.

        Returns:
            The generated or provided candidate_id.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

        if candidate_id is None:
            prefix = "debug" if is_debug else ("ref" if is_reference else f"prop_iter{iteration:02d}")
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
                    accuracy_fid, relative_error, absolute_error, latency_ms, wall_clock_s,
                    worker_utilization, pareto_optimal, selection_reason, phase_1_prompt,
                    is_debug, debug_parent_id,
                    counter_1_compilation, counter_2_rewrite, counter_3_debug_gate, counter_4_debug_loop
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    absolute_error,
                    latency_ms,
                    wall_clock_s,
                    worker_utilization,
                    1 if pareto_optimal else 0,
                    selection_reason,
                    phase_1_prompt,
                    1 if is_debug else 0,
                    debug_parent_id,
                    counter_1_compilation,
                    counter_2_rewrite,
                    counter_3_debug_gate,
                    counter_4_debug_loop,
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
        is_debug: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve proposals optionally filtered by status, reference flag, and debug flag."""
        return self.query_proposals(status=status, limit=limit, is_reference=is_reference, is_debug=is_debug)

    def query_proposals(
        self,
        is_reference: Optional[bool] = None,
        status: Optional[str] = None,
        is_debug: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query proposals with optional reference, debug, and status filters."""
        conn = self._get_connection()
        try:
            clauses = []
            params: List[Any] = []
            if is_reference is not None:
                clauses.append("is_reference = ?")
                params.append(1 if is_reference else 0)
            if is_debug is not None:
                clauses.append("is_debug = ?")
                params.append(1 if is_debug else 0)
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
            1. Error / quality: relative_error (or absolute_error / accuracy_fid fallback if relative_error is null)
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
                err = item["relative_error"] if item["relative_error"] is not None else (
                    item["absolute_error"] if item.get("absolute_error") is not None else item.get("accuracy_fid")
                )
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
            "absolute_error",
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
            "is_debug",
            "debug_parent_id",
            "counter_1_compilation",
            "counter_2_rewrite",
            "counter_3_debug_gate",
            "counter_4_debug_loop",
        }

        update_cols = ["status = ?"]
        values: List[Any] = [status]

        for k, v in kwargs.items():
            if k in allowed_fields:
                if k in {"solver_params", "noise_params", "sparsity_config"}:
                    v = _canonical_json(v)
                elif k in {"pareto_optimal", "is_active_evaluation", "is_debug"}:
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

    def get_completed_experiments_count(self, noise_model: Optional[str] = None) -> int:
        """Count evaluated candidate experiments (completed or failed, non-reference).

        Args:
            noise_model: Optional noise model name to filter by.
        """
        conn = self._get_connection()
        try:
            if noise_model is not None:
                cur = conn.execute(
                    """
                    SELECT COUNT(*) FROM proposals
                    WHERE is_reference = 0
                      AND status IN ('COMPLETED', 'FAILED')
                      AND noise_model = ?
                    """,
                    (noise_model,),
                )
            else:
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

    def get_proposal_counts_by_noise_model(
        self, exclude_reference: bool = True
    ) -> Dict[str, Dict[str, int]]:
        """Count proposals grouped by noise model and execution status.

        Args:
            exclude_reference: If True, excludes reference proposals (is_reference = 1).

        Returns:
            Dict mapping each noise model name to counts:
            {'total': int, 'completed': int, 'failed': int, 'pending': int, 'rejected': int}
        """
        conn = self._get_connection()
        try:
            where_clause = "WHERE is_reference = 0" if exclude_reference else ""
            cur = conn.execute(
                f"""
                SELECT noise_model, status, COUNT(*) as cnt
                FROM proposals
                {where_clause}
                GROUP BY noise_model, status
                """
            )
            rows = cur.fetchall()
            counts_by_model: Dict[str, Dict[str, int]] = {}
            for row in rows:
                nm = row["noise_model"]
                status = str(row["status"]).upper()
                cnt = int(row["cnt"])
                if nm not in counts_by_model:
                    counts_by_model[nm] = {
                        "total": 0,
                        "completed": 0,
                        "failed": 0,
                        "pending": 0,
                        "rejected": 0,
                    }
                counts_by_model[nm]["total"] += cnt
                if status == "COMPLETED":
                    counts_by_model[nm]["completed"] += cnt
                elif status == "FAILED":
                    counts_by_model[nm]["failed"] += cnt
                elif status in ("PENDING", "EVALUATING"):
                    counts_by_model[nm]["pending"] += cnt
                elif status in ("REJECTED", "QUARANTINED"):
                    counts_by_model[nm]["rejected"] += cnt
            return counts_by_model
        finally:
            conn.close()

    def get_next_unfulfilled_noise_model(
        self, noise_models: list[str], max_per_noise: int
    ) -> Optional[str]:
        """Find the first noise model whose candidate attempt count is less than max_per_noise.

        An attempt is an executed experiment (status COMPLETED or FAILED, non-reference).

        Args:
            noise_models: Ordered list of noise model names to check.
            max_per_noise: Maximum number of experiment attempts allowed per noise model.

        Returns:
            The first noise model in noise_models with attempt count < max_per_noise,
            or None if all noise models have satisfied their quota.
        """
        counts = self.get_proposal_counts_by_noise_model(exclude_reference=True)
        for nm in noise_models:
            nm_counts = counts.get(nm, {})
            attempts = nm_counts.get("completed", 0) + nm_counts.get("failed", 0)
            if attempts < max_per_noise:
                return nm
        return None


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

    def create_debug_candidate(
        self,
        debugee_id: str,
        deactivated_rewrites: Optional[Union[List[str], Dict[str, Any], str]] = None,
        candidate_id: Optional[str] = None,
        iteration: Optional[int] = None,
        solver: Optional[str] = None,
        noise_model: Optional[str] = None,
        sparsity_config: Optional[Union[Dict[str, Any], str]] = None,
        solver_params: Optional[Union[Dict[str, Any], str]] = None,
        noise_params: Optional[Union[Dict[str, Any], str]] = None,
        status: str = "PENDING",
        selection_reason: Optional[str] = None,
        phase_1_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Create a debug candidate derived from a debugee candidate.

        Args:
            debugee_id: The candidate_id of the candidate being debugged.
            deactivated_rewrites: Optional collection or string specifying deactivated rewrites.
            candidate_id: Optional custom candidate ID for the debug candidate.
            iteration: Optional iteration override (defaults to debugee's iteration).
            solver: Optional solver override (defaults to debugee's solver).
            noise_model: Optional noise_model override (defaults to debugee's noise_model).
            sparsity_config: Optional sparsity_config override (defaults to debugee's sparsity_config).
            solver_params: Optional solver_params override (defaults to debugee's solver_params).
            noise_params: Optional noise_params override (defaults to debugee's noise_params).
            status: Initial status for debug candidate (default "PENDING").
            selection_reason: Optional selection reason.
            phase_1_prompt: Optional prompt text.

        Returns:
            The generated or provided debug candidate_id.
        """
        debugee = self.get_proposal(debugee_id)
        if debugee is None:
            raise KeyError(f"Debugee candidate '{debugee_id}' not found.")

        if candidate_id is None:
            candidate_id = f"debug_{debugee_id}_{uuid.uuid4().hex[:6]}"

        eff_iteration = iteration if iteration is not None else debugee.get("iteration", 0)
        eff_solver = solver if solver is not None else debugee.get("solver", "rk4")
        eff_noise_model = noise_model if noise_model is not None else debugee.get("noise_model", "none")
        eff_sparsity_config = sparsity_config if sparsity_config is not None else debugee.get("sparsity_config", "{}")
        eff_solver_params = solver_params if solver_params is not None else debugee.get("solver_params", "{}")
        eff_noise_params = noise_params if noise_params is not None else debugee.get("noise_params", "{}")

        if selection_reason is None:
            reason_parts = [f"Debug candidate derived from {debugee_id}"]
            if deactivated_rewrites:
                reason_parts.append(f"Deactivated rewrites: {deactivated_rewrites}")
            selection_reason = "; ".join(reason_parts)

        eff_phase_1_prompt = phase_1_prompt if phase_1_prompt is not None else debugee.get("phase_1_prompt", "")

        counter_1 = kwargs.pop("counter_1_compilation", debugee.get("counter_1_compilation", 0))
        counter_2 = kwargs.pop("counter_2_rewrite", debugee.get("counter_2_rewrite", 0))
        counter_3 = kwargs.pop("counter_3_debug_gate", 0)
        # Counter 4 starts at 0 for a new debug session
        counter_4 = kwargs.pop("counter_4_debug_loop", 0)

        return self.insert_proposal(
            candidate_id=candidate_id,
            iteration=eff_iteration,
            solver=eff_solver,
            noise_model=eff_noise_model,
            sparsity_config=eff_sparsity_config,
            solver_params=eff_solver_params,
            noise_params=eff_noise_params,
            status=status,
            is_debug=True,
            debug_parent_id=debugee_id,
            selection_reason=selection_reason,
            phase_1_prompt=eff_phase_1_prompt,
            counter_1_compilation=counter_1,
            counter_2_rewrite=counter_2,
            counter_3_debug_gate=counter_3,
            counter_4_debug_loop=counter_4,
            **kwargs,
        )

    def quarantine_candidate(self, candidate_id: str) -> Dict[str, Any]:
        """Quarantine a candidate, updating its status to 'QUARANTINED'.

        Args:
            candidate_id: ID of candidate to quarantine.

        Returns:
            Updated proposal dictionary.
        """
        return self.mark_proposal_status(candidate_id, "QUARANTINED")

    def reactivate_candidate(self, candidate_id: str, new_status: str = "PENDING") -> Dict[str, Any]:
        """Reactivate a quarantined candidate, restoring status to 'PENDING' or 'EVALUATING'.

        Also restarts Counter ID 4 (counter_4_debug_loop) to 0 upon transitioning back to non-debug.

        Args:
            candidate_id: ID of candidate to reactivate.
            new_status: Desired new status ('PENDING' or 'EVALUATING').

        Returns:
            Updated proposal dictionary.
        """
        if new_status not in {"PENDING", "EVALUATING"}:
            raise ValueError(f"Invalid reactivation status '{new_status}'. Must be 'PENDING' or 'EVALUATING'.")
        return self.mark_proposal_status(candidate_id, new_status, counter_4_debug_loop=0)

    def reset_counters(
        self,
        candidate_id: str,
        counter_1_compilation: int = 0,
        counter_2_rewrite: int = 0,
        counter_3_debug_gate: int = 0,
        counter_4_debug_loop: int = 0,
    ) -> Dict[str, Any]:
        """Reset Phase 1 counters (counter 1 to 4) to 0 (or specified values) for a candidate.

        Args:
            candidate_id: Target candidate ID.
            counter_1_compilation: Compilation failure counter (default 0).
            counter_2_rewrite: Rewrite failure counter (default 0).
            counter_3_debug_gate: Invalid debug request counter (default 0).
            counter_4_debug_loop: Debug iteration loop counter (default 0).

        Returns:
            Updated proposal dictionary.
        """
        return self.update_counters(
            candidate_id=candidate_id,
            counter_1_compilation=counter_1_compilation,
            counter_2_rewrite=counter_2_rewrite,
            counter_3_debug_gate=counter_3_debug_gate,
            counter_4_debug_loop=counter_4_debug_loop,
        )

    def update_counters(
        self,
        candidate_id: str,
        counter_1_compilation: Optional[int] = None,
        counter_2_rewrite: Optional[int] = None,
        counter_3_debug_gate: Optional[int] = None,
        counter_4_debug_loop: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update counter values for a given candidate.

        Args:
            candidate_id: Target candidate ID.
            counter_1_compilation: Optional compilation counter value.
            counter_2_rewrite: Optional rewrite counter value.
            counter_3_debug_gate: Optional debug gate counter value.
            counter_4_debug_loop: Optional debug loop counter value.

        Returns:
            Updated proposal dictionary.
        """
        curr = self.get_proposal(candidate_id)
        if curr is None:
            raise KeyError(f"Proposal with candidate_id '{candidate_id}' not found.")

        kwargs: Dict[str, Any] = {}
        if counter_1_compilation is not None:
            kwargs["counter_1_compilation"] = counter_1_compilation
        if counter_2_rewrite is not None:
            kwargs["counter_2_rewrite"] = counter_2_rewrite
        if counter_3_debug_gate is not None:
            kwargs["counter_3_debug_gate"] = counter_3_debug_gate
        if counter_4_debug_loop is not None:
            kwargs["counter_4_debug_loop"] = counter_4_debug_loop

        if not kwargs:
            return curr

        return self.mark_proposal_status(candidate_id, curr["status"], **kwargs)

    def get_active_debug_candidate(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent active debug candidate (is_debug=1 and status in ('PENDING', 'EVALUATING'))."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM proposals
                WHERE is_debug = 1 AND status IN ('PENDING', 'EVALUATING')
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_quarantined_candidate(self, candidate_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a quarantined candidate by ID or the most recently quarantined candidate."""
        conn = self._get_connection()
        try:
            if candidate_id is not None:
                cur = conn.execute(
                    "SELECT * FROM proposals WHERE candidate_id = ? AND status = 'QUARANTINED'",
                    (candidate_id,),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT * FROM proposals
                    WHERE status = 'QUARANTINED'
                    ORDER BY updated_at DESC, created_at DESC, rowid DESC
                    LIMIT 1
                    """
                )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
