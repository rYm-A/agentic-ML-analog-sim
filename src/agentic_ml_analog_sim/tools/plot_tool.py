"""PlotTool ChiaTool (FastMCP server) for generating publication-grade co-design charts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ray
from chia.base.tools.ChiaTool import ChiaTool

from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.reporting.plot_generator import (
    generate_pareto_plot as _generate_pareto_plot,
    generate_breakdown_plot as _generate_breakdown_plot,
    generate_token_cost_plot as _generate_token_cost_plot,
)


class PlotTool(ChiaTool):
    """Plotting ChiaTool (FastMCP server) providing Matplotlib charts for Pareto, reliability, and token auditing."""

    def __init__(
        self,
        name: str = "plot_tool",
        db: Optional[Union[ResultDatabase, str, Path]] = None,
        default_output_dir: Union[str, Path] = "reports/figures",
        task_options: Optional[Dict[str, Any]] = None,
        logging_level: int = logging.INFO,
        deploy: Optional[bool] = None,
    ):
        """Initialize PlotTool and register FastMCP tools.

        Args:
            name: FastMCP tool prefix.
            db: ResultDatabase instance or path.
            default_output_dir: Default output directory for generated figure files.
            task_options: Ray placement/resource options.
            logging_level: Logging level.
            deploy: If True, deploy Ray actor. If False, run in local direct mode.
        """
        super().__init__(name, task_options=task_options, logging_level=logging_level)
        self.default_output_dir = Path(default_output_dir)
        self.setup(db=db)

        should_deploy = deploy if deploy is not None else ray.is_initialized()
        if should_deploy:
            try:
                super().__post_init__()
            except Exception as e:
                self.logger.warning(
                    f"PlotTool: Ray deployment skipped ({e}). Operating in local direct mode."
                )

    def setup(self, db: Optional[Union[ResultDatabase, str, Path]] = None) -> None:
        """Register MCP tool endpoints and configure database."""
        if isinstance(db, ResultDatabase):
            self.db = db
        elif isinstance(db, (str, Path)):
            self.db = ResultDatabase(db_path=db)
        elif db is None:
            self.db = ResultDatabase()
        else:
            raise TypeError(f"Invalid type for db: {type(db)}")

        # Register FastMCP tool methods
        self.mcp.add_tool(self.generate_pareto_plot, name=f"{self.name}_generate_pareto_plot")
        self.mcp.add_tool(self.generate_breakdown_plot, name=f"{self.name}_generate_breakdown_plot")
        self.mcp.add_tool(self.generate_token_cost_plot, name=f"{self.name}_generate_token_cost_plot")
        self.mcp.add_tool(self.generate_all_plots, name=f"{self.name}_generate_all_plots")

    def generate_pareto_plot(self, output_dir: Optional[str] = None) -> str:
        """Generate publication-grade Pareto frontier plot (PDF & PNG) comparing latency vs accuracy error.

        Args:
            output_dir: Optional directory to store generated plots. Defaults to tool's default_output_dir.

        Returns:
            JSON string containing status and file paths to generated 'pdf' and 'png'.
        """
        try:
            target_dir = Path(output_dir) if output_dir else self.default_output_dir
            paths = _generate_pareto_plot(self.db, target_dir)
            return json.dumps({
                "status": "SUCCESS",
                "plot_type": "pareto_frontier",
                "files": paths,
                "message": "Pareto frontier plot successfully generated.",
            }, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Failed to generate Pareto plot: {e}"})

    def generate_breakdown_plot(self, output_dir: Optional[str] = None) -> str:
        """Generate publication-grade execution reliability breakdown plot (PDF & PNG) for pipeline stages.

        Args:
            output_dir: Optional directory to store generated plots.

        Returns:
            JSON string containing status and file paths to generated 'pdf' and 'png'.
        """
        try:
            target_dir = Path(output_dir) if output_dir else self.default_output_dir
            paths = _generate_breakdown_plot(self.db, target_dir)
            return json.dumps({
                "status": "SUCCESS",
                "plot_type": "execution_breakdown",
                "files": paths,
                "message": "Execution breakdown plot successfully generated.",
            }, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Failed to generate breakdown plot: {e}"})

    def generate_token_cost_plot(self, output_dir: Optional[str] = None) -> str:
        """Generate publication-grade LLM token consumption and cost summary plot (PDF & PNG).

        Args:
            output_dir: Optional directory to store generated plots.

        Returns:
            JSON string containing status and file paths to generated 'pdf' and 'png'.
        """
        try:
            target_dir = Path(output_dir) if output_dir else self.default_output_dir
            paths = _generate_token_cost_plot(self.db, target_dir)
            return json.dumps({
                "status": "SUCCESS",
                "plot_type": "token_cost_summary",
                "files": paths,
                "message": "Token cost plot successfully generated.",
            }, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Failed to generate token cost plot: {e}"})

    def generate_all_plots(self, output_dir: Optional[str] = None) -> str:
        """Generate all publication-grade plots (Pareto, execution breakdown, token costs).

        Args:
            output_dir: Optional directory to store generated plots.

        Returns:
            JSON string containing all generated file paths.
        """
        try:
            target_dir = Path(output_dir) if output_dir else self.default_output_dir
            pareto = _generate_pareto_plot(self.db, target_dir)
            breakdown = _generate_breakdown_plot(self.db, target_dir)
            token_cost = _generate_token_cost_plot(self.db, target_dir)
            return json.dumps({
                "status": "SUCCESS",
                "plots": {
                    "pareto": pareto,
                    "breakdown": breakdown,
                    "token_cost": token_cost,
                },
                "message": "All plots successfully generated.",
            }, indent=2)
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Failed to generate plots: {e}"})
