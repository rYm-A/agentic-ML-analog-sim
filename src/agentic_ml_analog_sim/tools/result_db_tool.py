"""ResultDB FastMCP tool server and API for the analog ML accelerator simulation loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ray
from chia.base.tools.ChiaTool import ChiaTool

from agentic_ml_analog_sim.db.database import ResultDatabase


class ResultDBTool(ChiaTool):
    """Result database ChiaTool (FastMCP server) for candidate proposal, Pareto tracking, and notes."""

    def __init__(
        self,
        name: str = "result_db",
        db: Optional[Union[ResultDatabase, str, Path]] = None,
        task_options: Optional[Dict[str, Any]] = None,
        logging_level: int = logging.INFO,
        deploy: Optional[bool] = None,
        read_only: bool = False,
    ):
        """Initialize ResultDBTool with database connection and register MCP tools.

        Args:
            name: MCP tool namespace prefix.
            db: Existing ResultDatabase instance, or path to SQLite file, or None for default.
            task_options: Ray placement/resource options for deploying the FastMCP actor.
            logging_level: Python logging level.
            deploy: If True, deploy Ray actor. If False, run in local direct mode.
                    If None, auto-deploy only if Ray is already initialized.
            read_only: If True, register ONLY read-only methods and block mutations.
        """
        super().__init__(name, task_options=task_options, logging_level=logging_level)
        self.read_only = read_only
        self.setup(db=db, read_only=read_only)

        should_deploy = deploy if deploy is not None else ray.is_initialized()
        if should_deploy:
            try:
                super().__post_init__()
            except Exception as e:
                self.logger.warning(
                    f"{self.__class__.__name__}: Ray deployment skipped or failed ({e}). "
                    "Operating in direct local execution mode."
                )

        if not getattr(self, "hostname", None):
            self.hostname = "localhost"
            self.port = getattr(self, "port", None) or 8000

    def setup(
        self,
        db: Optional[Union[ResultDatabase, str, Path]] = None,
        read_only: bool = False,
    ) -> None:
        """Register MCP tool methods and configure the database instance."""
        self.read_only = read_only
        if isinstance(db, ResultDatabase):
            self.db = db
        elif isinstance(db, (str, Path)):
            self.db = ResultDatabase(db_path=db)
        elif db is None:
            self.db = ResultDatabase()
        else:
            raise TypeError(f"Invalid type for db: {type(db)}")

        # Register read-only MCP tool methods (accessible in both read_only and read-write modes)
        self.mcp.add_tool(self.get_proposal, name=f"{self.name}_get_proposal")
        self.mcp.add_tool(self.read_prior_proposals, name=f"{self.name}_read_prior_proposals")
        self.mcp.add_tool(self.get_candidate_history_summary, name=f"{self.name}_get_candidate_history_summary")
        self.mcp.add_tool(self.get_active_evaluations, name=f"{self.name}_get_active_evaluations")
        self.mcp.add_tool(self.query_proposals, name=f"{self.name}_query_proposals")
        self.mcp.add_tool(self.get_pareto_front, name=f"{self.name}_get_pareto_front")
        self.mcp.add_tool(self.get_reference_result, name=f"{self.name}_get_reference_result")
        self.mcp.add_tool(self.get_proposal_comments, name=f"{self.name}_get_proposal_comments")
        self.mcp.add_tool(self.get_execution_statistics, name=f"{self.name}_get_execution_statistics")
        self.mcp.add_tool(self.get_token_usage_summary, name=f"{self.name}_get_token_usage_summary")
        self.mcp.add_tool(self.get_completed_experiments_count, name=f"{self.name}_get_completed_experiments_count")

        # Register mutation tools ONLY when NOT in read_only mode
        if not self.read_only:
            self.mcp.add_tool(self.insert_proposal, name=f"{self.name}_insert_proposal")
            self.mcp.add_tool(self.add_comment, name=f"{self.name}_add_comment")
            self.mcp.add_tool(self.record_token_usage, name=f"{self.name}_record_token_usage")
            self.mcp.add_tool(self.mark_proposal_status, name=f"{self.name}_mark_proposal_status")
            self.mcp.add_tool(self.update_execution_status, name=f"{self.name}_update_execution_status")

    def read_prior_proposals(self, status: Optional[str] = None, limit: int = 50) -> str:
        """Read simulation candidate proposals that have already been made from the database.

        Use this tool before generating a proposal to inspect previously explored combinations
        of solver, noise model, parameters, and sparsity configuration to avoid generating repeated proposals.

        Args:
            status: Optional status filter ('PENDING', 'EVALUATING', 'COMPLETED', 'FAILED', 'REJECTED').
                    If None, returns proposals across all statuses.
            limit: Maximum number of proposals to retrieve (default: 50).

        Returns:
            JSON string containing all existing proposals, their configurations, parameters,
            evaluation metrics (latency_ms, relative_error, absolute_error), status, and comments.
        """
        try:
            proposals = self.db.get_proposals(status=status, limit=limit)
            if not proposals:
                filter_str = f" with status='{status}'" if status else ""
                return json.dumps({
                    "status": "EMPTY",
                    "proposals": [],
                    "count": 0,
                    "message": f"No proposals found in database{filter_str}. The database has no prior proposals.",
                })

            enriched_proposals = []
            for p in proposals:
                item = dict(p)
                for json_col in ("solver_params", "noise_params", "sparsity_config"):
                    if isinstance(item.get(json_col), str):
                        try:
                            item[json_col] = json.loads(item[json_col])
                        except Exception:
                            pass
                cand_id = item.get("candidate_id")
                if cand_id and hasattr(self.db, "get_comments"):
                    try:
                        item["comments"] = self.db.get_comments(cand_id)
                    except Exception:
                        item["comments"] = []
                enriched_proposals.append(item)

            return json.dumps(
                {
                    "status": "SUCCESS",
                    "count": len(enriched_proposals),
                    "proposals": enriched_proposals,
                    "guidance": (
                        "Inspect the configurations above. Do NOT propose combinations of "
                        "(solver, num_steps, noise_model, sparsity_config) that already appear above."
                    ),
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error reading prior proposals: {e}"})

    def get_candidate_history_summary(self, limit: int = 50) -> str:
        """Retrieve a concise high-level summary of all candidate configurations already proposed.

        Specifically designed for Node 2 to quickly verify parameter sets already explored
        to avoid generating repeated proposals.

        Args:
            limit: Maximum number of recent candidates to summarize (default: 50).

        Returns:
            JSON string with a list of explored configurations (solver, num_steps, noise_model,
            sparsity_ratio, status, latency_ms, relative_error, absolute_error).
        """
        try:
            proposals = self.db.get_proposals(limit=limit)
            if not proposals:
                return json.dumps({
                    "status": "EMPTY",
                    "total_explored": 0,
                    "configurations": [],
                    "message": "No candidates have been proposed yet.",
                })

            configs = []
            for p in proposals:
                s_params = p.get("solver_params") or {}
                if isinstance(s_params, str):
                    try:
                        s_params = json.loads(s_params)
                    except Exception:
                        s_params = {}
                n_params = p.get("noise_params") or {}
                if isinstance(n_params, str):
                    try:
                        n_params = json.loads(n_params)
                    except Exception:
                        n_params = {}
                sp_cfg = p.get("sparsity_config") or {}
                if isinstance(sp_cfg, str):
                    try:
                        sp_cfg = json.loads(sp_cfg)
                    except Exception:
                        sp_cfg = {}

                configs.append({
                    "candidate_id": p.get("candidate_id"),
                    "solver": p.get("solver"),
                    "num_steps": s_params.get("num_steps") or s_params.get("steps"),
                    "noise_model": p.get("noise_model"),
                    "noise_sigma": n_params.get("sigma"),
                    "sparsity_ratio": sp_cfg.get("sparsity_ratio", 0.0),
                    "status": p.get("status"),
                    "is_reference": bool(p.get("is_reference")),
                    "is_active_evaluation": bool(p.get("is_active_evaluation")),
                    "latency_ms": p.get("latency_ms"),
                    "accuracy_fid": p.get("accuracy_fid"),
                    "relative_error": p.get("relative_error"),
                    "absolute_error": p.get("absolute_error"),
                })

            return json.dumps(
                {
                    "status": "SUCCESS",
                    "total_explored": len(configs),
                    "configurations": configs,
                    "guidance": (
                        "Select a candidate design that differs in solver, step count, "
                        "noise model, or sparsity ratio from all entries listed above."
                    ),
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error summarizing candidate history: {e}"})

    def get_active_evaluations(self) -> str:
        """Identify which candidate(s) is/are being evaluated in the actual loop.

        Per prompts/phase-0.md: 'There is also a way to identify which of the
        candidate(s) is/are being evaluated in the actual loop.'

        Returns:
            JSON string listing proposals where is_active_evaluation=1 or status='EVALUATING'.
        """
        try:
            conn = self.db._get_connection() if hasattr(self.db, "_get_connection") else None
            if conn is not None:
                try:
                    cur = conn.execute(
                        """
                        SELECT candidate_id, iteration, solver, noise_model, status,
                               is_active_evaluation, created_at, updated_at
                        FROM proposals
                        WHERE is_active_evaluation = 1 OR status = 'EVALUATING'
                        ORDER BY updated_at DESC
                        """
                    )
                    rows = [dict(r) for r in cur.fetchall()]
                    return json.dumps({
                        "status": "SUCCESS",
                        "active_evaluations_count": len(rows),
                        "active_proposals": rows,
                    }, indent=2, default=str)
                finally:
                    conn.close()
            else:
                props = self.db.get_proposals(status="EVALUATING")
                return json.dumps({
                    "status": "SUCCESS",
                    "active_evaluations_count": len(props),
                    "active_proposals": props,
                }, indent=2, default=str)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error fetching active evaluations: {e}"})

    def get_proposal_comments(self, candidate_id: str) -> str:
        """Retrieve cross-phase commentary notes for a specific proposal.

        Per prompts/phase-0.md: 'For each row (i.e., proposal, we can use some comment columns
        so that the agents present in the different phases of the Chia loop can leave some notes
        that can be used as interfaces of communication between different phase agents that
        the individual agents can query asynchronously).'

        Args:
            candidate_id: ID of the proposal to inspect notes for.

        Returns:
            JSON string containing list of notes/comments recorded for candidate_id.
        """
        try:
            if hasattr(self.db, "get_comments"):
                comments = self.db.get_comments(candidate_id=candidate_id)
                return json.dumps({
                    "status": "SUCCESS",
                    "candidate_id": candidate_id,
                    "count": len(comments),
                    "comments": comments,
                }, indent=2, default=str)
            return json.dumps({
                "status": "EMPTY",
                "candidate_id": candidate_id,
                "comments": [],
                "message": "Comments not supported by database instance.",
            })
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error fetching comments: {e}"})

    def query_proposals(self, status: Optional[str] = None, limit: int = 20) -> str:
        """Query simulation candidate proposals from the database, optionally filtered by status.

        Args:
            status: Status filter ('PENDING', 'EVALUATING', 'COMPLETED', 'FAILED', 'REJECTED') or None for all.
            limit: Maximum number of proposals to return (default: 20).

        Returns:
            JSON string containing a list of proposal records.
        """
        try:
            proposals = self.db.get_proposals(status=status, limit=limit)
            if not proposals:
                filter_str = f" with status='{status}'" if status else ""
                return json.dumps({
                    "status": "EMPTY",
                    "proposals": [],
                    "message": f"No proposals found{filter_str}.",
                })
            return json.dumps({"status": "SUCCESS", "proposals": proposals, "count": len(proposals)}, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error querying proposals: {e}"})

    def get_pareto_front(self) -> str:
        """Retrieve the current Pareto-optimal simulation proposals minimizing error and latency.

        Returns:
            JSON string containing Pareto-optimal proposals.
        """
        try:
            front = self.db.compute_pareto_front()
            if not front:
                return json.dumps({
                    "status": "EMPTY",
                    "pareto_front": [],
                    "message": "No Pareto-optimal candidates found among COMPLETED evaluations.",
                })
            return json.dumps({"status": "SUCCESS", "pareto_front": front, "count": len(front)}, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error computing Pareto front: {e}"})

    def get_reference_result(self) -> str:
        """Retrieve the baseline reference design simulation result.

        Returns:
            JSON string of reference proposal or a message if not found.
        """
        try:
            ref = self.db.get_reference_proposal()
            if ref is None:
                return json.dumps({
                    "status": "NOT_FOUND",
                    "reference": None,
                    "message": "No reference design proposal found in database.",
                })
            return json.dumps({"status": "SUCCESS", "reference": ref}, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error fetching reference result: {e}"})

    def insert_proposal(
        self,
        solver: str,
        noise_model: str,
        solver_params: str,
        noise_params: str,
        sparsity_config: str,
        selection_reason: str,
    ) -> str:
        """Insert a new simulation candidate proposal.

        Args:
            solver: Numerical ODE solver name (e.g. 'rk4', 'euler', 'euler_backward', 'parareal', 'par_ode').
            noise_model: Noise model name (e.g. 'none', 'L0_static_mismatch', 'L1_stochastic_parameter_noise', 'L2_functional_interface', 'L3_correlated_drift').
            solver_params: JSON string of solver parameters (e.g. '{"steps": 10}').
            noise_params: JSON string of noise parameters (e.g. '{"sigma": 0.05}').
            sparsity_config: JSON string of sparsity parameters (e.g. '{"sparsity_ratio": 0.2}').
            selection_reason: Text rationale for choosing this proposal.

        Returns:
            Status message string indicating success or duplicate detection.
        """
        if self.read_only:
            raise PermissionError("ResultDBReadOnlyTool provides read-only access. Insertion of proposals is not permitted.")
        try:
            # Check for duplicate configuration
            if self.db.is_duplicate_proposal(
                solver=solver,
                noise_model=noise_model,
                solver_params=solver_params,
                noise_params=noise_params,
                sparsity_config=sparsity_config,
            ):
                return json.dumps({
                    "status": "DUPLICATE",
                    "message": (
                        f"Duplicate proposal rejected: candidate with solver='{solver}', "
                        f"noise_model='{noise_model}', and identical parameters already exists."
                    ),
                })

            latest = self.db.get_latest_proposal()
            current_iter = (latest["iteration"] + 1) if latest and latest.get("iteration") is not None else 1

            candidate_id = self.db.insert_proposal(
                iteration=current_iter,
                solver=solver,
                noise_model=noise_model,
                solver_params=solver_params,
                noise_params=noise_params,
                sparsity_config=sparsity_config,
                status="PENDING",
                selection_reason=selection_reason,
            )

            return json.dumps({
                "status": "SUCCESS",
                "candidate_id": candidate_id,
                "iteration": current_iter,
                "message": f"Proposal successfully inserted with candidate_id '{candidate_id}' in iteration {current_iter}.",
            })
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error inserting proposal: {e}"})

    def add_comment(
        self,
        candidate_id: str,
        phase: str,
        comment: str,
        agent_name: str = "agent",
    ) -> str:
        """Add an agent commentary or review note to a candidate proposal.

        Args:
            candidate_id: ID of the proposal to comment on.
            phase: Phase or workflow step name (e.g. 'Phase_1', 'Phase_2', 'Evaluation').
            comment: Feedback, reasoning, or analysis note.
            agent_name: Name of the commenting agent (default: 'agent').

        Returns:
            Confirmation message string.
        """
        if self.read_only:
            raise PermissionError("ResultDBReadOnlyTool provides read-only access. Adding comments is not permitted.")
        try:
            prop = self.db.get_proposal(candidate_id)
            if prop is None:
                return json.dumps({
                    "status": "ERROR",
                    "message": f"Cannot add comment: proposal with candidate_id '{candidate_id}' does not exist.",
                })

            comment_id = self.db.add_comment(
                candidate_id=candidate_id,
                phase=phase,
                agent_name=agent_name,
                comment=comment,
            )
            return json.dumps({
                "status": "SUCCESS",
                "comment_id": comment_id,
                "candidate_id": candidate_id,
                "message": f"Comment recorded successfully (comment_id={comment_id}).",
            })
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error adding comment: {e}"})

    def get_execution_statistics(self) -> str:
        """Retrieve aggregated counts and breakdowns of succeeded vs failed compilations,
        simulations, and code rewrites across all candidate experiments.

        Returns:
            JSON string containing compilation, simulation, and rewrite statistics.
        """
        try:
            stats = self.db.get_execution_statistics()
            return json.dumps({"status": "SUCCESS", "statistics": stats}, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error retrieving execution statistics: {e}"})

    def get_token_usage_summary(self, phase: Optional[str] = None) -> str:
        """Retrieve total tokens consumed and estimated costs across all LLM calls,
        optionally filtered by phase ('phase_0', 'phase_1', 'phase_2').

        Returns:
            JSON string containing overall totals and breakdowns by phase, model, and node.
        """
        try:
            summary = self.db.get_token_usage_summary(phase=phase)
            return json.dumps({"status": "SUCCESS", "token_usage": summary}, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error retrieving token summary: {e}"})

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
    ) -> str:
        """Record token counts and estimated cost for an LLM execution turn.

        Returns:
            JSON string confirming recorded token usage with row id.
        """
        if self.read_only:
            raise PermissionError("ResultDBReadOnlyTool provides read-only access. Recording token usage is not permitted.")
        try:
            row_id = self.db.record_token_usage(
                phase=phase,
                node_name=node_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                candidate_id=candidate_id,
                duration_seconds=duration_seconds,
            )
            return json.dumps({
                "status": "SUCCESS",
                "token_usage_id": row_id,
                "message": f"Token usage recorded successfully (id={row_id}).",
            })
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error recording token usage: {e}"})

    def get_completed_experiments_count(self) -> str:
        """Retrieve the number of completed or failed candidate experiments.

        Returns:
            JSON string with completed_experiments_count.
        """
        try:
            count = self.db.get_completed_experiments_count()
            return json.dumps({"status": "SUCCESS", "completed_experiments_count": count})
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error getting completed experiments count: {e}"})

    def get_proposal(self, candidate_id: str) -> str:
        """Retrieve candidate proposal configuration and metadata by candidate_id.

        Args:
            candidate_id: Proposal candidate ID.

        Returns:
            JSON string containing candidate proposal details or error.
        """
        try:
            prop = self.db.get_proposal(candidate_id)
            if not prop:
                return json.dumps({"status": "NOT_FOUND", "message": f"Candidate '{candidate_id}' not found."})
            return json.dumps({"status": "SUCCESS", "proposal": prop}, default=str)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error retrieving proposal: {e}"})

    def mark_proposal_status(
        self,
        candidate_id: str,
        status: str,
        latency_ms: Optional[float] = None,
        accuracy_fid: Optional[float] = None,
        relative_error: Optional[float] = None,
        absolute_error: Optional[float] = None,
        wall_clock_s: Optional[float] = None,
        worker_utilization: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Update candidate proposal status and evaluation metrics.

        Args:
            candidate_id: Proposal candidate ID.
            status: New proposal status.
            latency_ms: Measured inference latency in milliseconds.
            accuracy_fid: Deprecated legacy FID metric (retained for backward compatibility).
            relative_error: Relative numerical error vs golden baseline trajectory.
            absolute_error: Absolute numerical error vs golden baseline trajectory.
            wall_clock_s: Wall clock elapsed time in seconds.
            worker_utilization: Estimated worker utilization.
            **kwargs: Extra fields passed to database mark_proposal_status.

        Returns:
            JSON string confirming update.
        """
        try:
            self.db.mark_proposal_status(
                candidate_id=candidate_id,
                status=status,
                latency_ms=latency_ms,
                accuracy_fid=accuracy_fid,
                relative_error=relative_error,
                absolute_error=absolute_error,
                wall_clock_s=wall_clock_s,
                worker_utilization=worker_utilization,
                **kwargs,
            )
            return json.dumps({"status": "SUCCESS", "candidate_id": candidate_id, "new_status": status})
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error updating proposal status: {e}"})

    def update_execution_status(
        self,
        candidate_id: str,
        simulation_status: Optional[str] = None,
        error_stage: Optional[str] = None,
        error_message: Optional[str] = None,
        compilation_status: Optional[str] = None,
        rewrite_solver_status: Optional[str] = None,
        rewrite_noise_status: Optional[str] = None,
        rewrite_compiler_status: Optional[str] = None,
    ) -> str:
        """Update granular stage execution status in the database.

        Args:
            candidate_id: Proposal candidate ID.
            simulation_status: Simulation status ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED').
            error_stage: Pipeline stage where error occurred.
            error_message: Description of the error.
            compilation_status: Compilation status.
            rewrite_solver_status: Solver rewrite status.
            rewrite_noise_status: Noise rewrite status.
            rewrite_compiler_status: Compiler rewrite status.

        Returns:
            JSON string confirming update.
        """
        try:
            self.db.update_execution_status(
                candidate_id=candidate_id,
                simulation_status=simulation_status,
                error_stage=error_stage,
                error_message=error_message,
                compilation_status=compilation_status,
                rewrite_solver_status=rewrite_solver_status,
                rewrite_noise_status=rewrite_noise_status,
                rewrite_compiler_status=rewrite_compiler_status,
            )
            return json.dumps({"status": "SUCCESS", "candidate_id": candidate_id, "simulation_status": simulation_status})
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Error updating execution status: {e}"})


# Direct helper functions for local/test execution without requiring tool server instantiation
def query_proposals(db: ResultDatabase, status: Optional[str] = None, limit: int = 20) -> str:
    """Helper function to query proposals directly against a ResultDatabase."""
    tool = ResultDBTool(db=db, deploy=False)
    return tool.query_proposals(status=status, limit=limit)


def get_pareto_front(db: ResultDatabase) -> str:
    """Helper function to get Pareto front directly against a ResultDatabase."""
    tool = ResultDBTool(db=db, deploy=False)
    return tool.get_pareto_front()


def get_reference_result(db: ResultDatabase) -> str:
    """Helper function to get reference result directly against a ResultDatabase."""
    tool = ResultDBTool(db=db, deploy=False)
    return tool.get_reference_result()


def insert_proposal(
    db: ResultDatabase,
    solver: str,
    noise_model: str,
    solver_params: str,
    noise_params: str,
    sparsity_config: str,
    selection_reason: str,
) -> str:
    """Helper function to insert a proposal directly against a ResultDatabase."""
    tool = ResultDBTool(db=db, deploy=False)
    return tool.insert_proposal(
        solver=solver,
        noise_model=noise_model,
        solver_params=solver_params,
        noise_params=noise_params,
        sparsity_config=sparsity_config,
        selection_reason=selection_reason,
    )


def add_comment(
    db: ResultDatabase,
    candidate_id: str,
    phase: str,
    comment: str,
    agent_name: str = "agent",
) -> str:
    """Helper function to add a comment directly against a ResultDatabase."""
    tool = ResultDBTool(db=db, deploy=False)
    return tool.add_comment(
        candidate_id=candidate_id,
        phase=phase,
        comment=comment,
        agent_name=agent_name,
    )


class ResultDBReadOnlyTool(ResultDBTool):
    """Read-only Result Database ChiaTool (FastMCP server) for inspecting prior proposals.

    Provides Node 2 with just read access to the database to inspect existing proposals,
    profiling metrics, comments, and the Pareto frontier, specifically avoiding repeated proposals.
    """

    def __init__(
        self,
        name: str = "result_db_read",
        db: Optional[Union[ResultDatabase, str, Path]] = None,
        task_options: Optional[Dict[str, Any]] = None,
        logging_level: int = logging.INFO,
        deploy: Optional[bool] = None,
    ):
        super().__init__(
            name=name,
            db=db,
            task_options=task_options,
            logging_level=logging_level,
            deploy=deploy,
            read_only=True,
        )


ResultDBReadTool = ResultDBReadOnlyTool


def read_prior_proposals(db: ResultDatabase, status: Optional[str] = None, limit: int = 50) -> str:
    """Helper function to read prior proposals directly from a ResultDatabase."""
    tool = ResultDBReadOnlyTool(db=db, deploy=False)
    return tool.read_prior_proposals(status=status, limit=limit)


def get_candidate_history_summary(db: ResultDatabase, limit: int = 50) -> str:
    """Helper function to summarize candidate history directly from a ResultDatabase."""
    tool = ResultDBReadOnlyTool(db=db, deploy=False)
    return tool.get_candidate_history_summary(limit=limit)


def get_active_evaluations(db: ResultDatabase) -> str:
    """Helper function to retrieve active evaluations directly from a ResultDatabase."""
    tool = ResultDBReadOnlyTool(db=db, deploy=False)
    return tool.get_active_evaluations()


def get_proposal_comments(db: ResultDatabase, candidate_id: str) -> str:
    """Helper function to retrieve proposal comments directly from a ResultDatabase."""
    tool = ResultDBReadOnlyTool(db=db, deploy=False)
    return tool.get_proposal_comments(candidate_id=candidate_id)


def get_proposal(db: ResultDatabase, candidate_id: str) -> str:
    """Helper function to retrieve candidate proposal directly from a ResultDatabase."""
    tool = ResultDBReadOnlyTool(db=db, deploy=False)
    return tool.get_proposal(candidate_id=candidate_id)


def mark_proposal_status(db: ResultDatabase, candidate_id: str, status: str, **kwargs: Any) -> str:
    """Helper function to mark proposal status directly on a ResultDatabase."""
    tool = ResultDBTool(db=db, deploy=False)
    return tool.mark_proposal_status(candidate_id=candidate_id, status=status, **kwargs)


def update_execution_status(db: ResultDatabase, candidate_id: str, **kwargs: Any) -> str:
    """Helper function to update execution status directly on a ResultDatabase."""
    tool = ResultDBTool(db=db, deploy=False)
    return tool.update_execution_status(candidate_id=candidate_id, **kwargs)

