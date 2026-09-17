"""Tools module for analog ML accelerator simulation loop."""

from agentic_ml_analog_sim.tools.result_db_tool import (
    ResultDBReadOnlyTool,
    ResultDBReadTool,
    ResultDBTool,
    add_comment,
    get_active_evaluations,
    get_candidate_history_summary,
    get_pareto_front,
    get_proposal_comments,
    get_reference_result,
    insert_proposal,
    query_proposals,
    read_prior_proposals,
)
from agentic_ml_analog_sim.tools.plot_tool import PlotTool

try:
    from agentic_ml_analog_sim.tools.un0_context_tool import (
        DEFAULT_UN0_ROOT,
        Un0Context,
        Un0ContextTool,
        create_un0_mcp_server,
        resolve_un0_root,
    )
    _un0_exports = [
        "Un0Context",
        "Un0ContextTool",
        "create_un0_mcp_server",
        "resolve_un0_root",
        "DEFAULT_UN0_ROOT",
    ]
except ImportError:
    _un0_exports = []

from agentic_ml_analog_sim.tools.compilation_info_tool import CompilationInfoTool
from agentic_ml_analog_sim.tools.noise_papers_tool import NoisePapersTool
from agentic_ml_analog_sim.tools.solver_papers_tool import SolverPapersTool
from agentic_ml_analog_sim.tools.un0_rewriter_tool import Un0RewriterTool

__all__ = [
    "ResultDBTool",
    "ResultDBReadOnlyTool",
    "ResultDBReadTool",
    "PlotTool",
    "CompilationInfoTool",
    "NoisePapersTool",
    "SolverPapersTool",
    "Un0RewriterTool",
    "query_proposals",
    "read_prior_proposals",
    "get_candidate_history_summary",
    "get_active_evaluations",
    "get_proposal_comments",
    "get_pareto_front",
    "get_reference_result",
    "insert_proposal",
    "add_comment",
    *_un0_exports,
]

