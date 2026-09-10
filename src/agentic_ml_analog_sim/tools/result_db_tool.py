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
    ):
        """Initialize ResultDBTool with database connection and register MCP tools.

        Args:
            name: MCP tool namespace prefix.
            db: Existing ResultDatabase instance, or path to SQLite file, or None for default.
            task_options: Ray placement/resource options for deploying the FastMCP actor.
            logging_level: Python logging level.
            deploy: If True, deploy Ray actor. If False, run in local direct mode.
                    If None, auto-deploy only if Ray is already initialized.
        """
        super().__init__(name, task_options=task_options, logging_level=logging_level)
        self.setup(db=db)

        should_deploy = deploy if deploy is not None else ray.is_initialized()
        if should_deploy:
            try:
                super().__post_init__()
            except Exception as e:
                self.logger.warning(
                    f"ResultDBTool: Ray deployment skipped or failed ({e}). "
                    "Operating in direct local execution mode."
                )

    def setup(self, db: Optional[Union[ResultDatabase, str, Path]] = None) -> None:
        """Register MCP tool methods and configure the database instance."""
        if isinstance(db, ResultDatabase):
            self.db = db
        elif isinstance(db, (str, Path)):
            self.db = ResultDatabase(db_path=db)
        elif db is None:
            self.db = ResultDatabase()
        else:
            raise TypeError(f"Invalid type for db: {type(db)}")

        # Register methods with FastMCP
        self.mcp.add_tool(self.query_proposals, name=f"{self.name}_query_proposals")
        self.mcp.add_tool(self.get_pareto_front, name=f"{self.name}_get_pareto_front")
        self.mcp.add_tool(self.get_reference_result, name=f"{self.name}_get_reference_result")
        self.mcp.add_tool(self.insert_proposal, name=f"{self.name}_insert_proposal")
        self.mcp.add_tool(self.add_comment, name=f"{self.name}_add_comment")
        self.mcp.add_tool(self.get_execution_statistics, name=f"{self.name}_get_execution_statistics")
        self.mcp.add_tool(self.get_token_usage_summary, name=f"{self.name}_get_token_usage_summary")
        self.mcp.add_tool(self.record_token_usage, name=f"{self.name}_record_token_usage")
        self.mcp.add_tool(self.get_completed_experiments_count, name=f"{self.name}_get_completed_experiments_count")

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
