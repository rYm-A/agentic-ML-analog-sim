"""Tools module for analog ML accelerator simulation loop."""

from agentic_ml_analog_sim.tools.result_db_tool import (
    ResultDBTool,
    add_comment,
    get_pareto_front,
    get_reference_result,
    insert_proposal,
    query_proposals,
)
from agentic_ml_analog_sim.tools.plot_tool import PlotTool

try:
    from agentic_ml_analog_sim.tools.un0_context_tool import (
        Un0Context,
        Un0ContextTool,
        create_un0_mcp_server,
    )
    _un0_exports = ["Un0Context", "Un0ContextTool", "create_un0_mcp_server"]
except ImportError:
    _un0_exports = []

__all__ = [
    "ResultDBTool",
    "PlotTool",
    "query_proposals",
    "get_pareto_front",
    "get_reference_result",
    "insert_proposal",
    "add_comment",
    *_un0_exports,
]
