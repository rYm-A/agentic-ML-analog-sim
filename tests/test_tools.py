"""Tests for ResultDBTool and Un0ContextTool."""

import json
import tempfile
from pathlib import Path
import pytest

from agentic_ml_analog_sim.db import ResultDatabase
from agentic_ml_analog_sim.tools import (
    ResultDBTool,
    Un0Context,
    Un0ContextTool,
    resolve_un0_root,
)

REPO_ROOT = resolve_un0_root()


# ============================================================================
# ResultDBTool Tests
# ============================================================================


@pytest.fixture
def temp_result_db_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tools.db"
        db = ResultDatabase(db_path)
        tool = ResultDBTool(name="test_result_db", db=db, deploy=False)
        yield tool


def test_result_db_tool_lifecycle(temp_result_db_tool):
    """Test ResultDBTool methods: insert, duplicate check, query, pareto, comments."""
    tool = temp_result_db_tool
    db = tool.db

    # Reference initially missing
    ref_res = json.loads(tool.get_reference_result())
    assert ref_res["status"] == "NOT_FOUND"

    # Insert candidate proposal
    ins_res = json.loads(
        tool.insert_proposal(
            solver="euler",
            noise_model="L0_static_mismatch",
            solver_params='{"num_steps": 5}',
            noise_params='{"sigma": 0.01}',
            sparsity_config='{"sparsity_ratio": 0.4}',
            selection_reason="Fast exploration",
        )
    )
    assert ins_res["status"] == "SUCCESS"
    cand_id = ins_res["candidate_id"]

    # Duplicate detection rejection
    dup_res = json.loads(
        tool.insert_proposal(
            solver="euler",
            noise_model="L0_static_mismatch",
            solver_params='{"num_steps": 5}',
            noise_params='{"sigma": 0.01}',
            sparsity_config='{"sparsity_ratio": 0.4}',
            selection_reason="Duplicate attempt",
        )
    )
    assert dup_res["status"] == "DUPLICATE"

    # Add comment
    com_res = json.loads(
        tool.add_comment(
            candidate_id=cand_id,
            phase="Phase_0",
            comment="Approved by Node 3 validation.",
            agent_name="node3",
        )
    )
    assert com_res["status"] == "SUCCESS"

    # Mark completed and check queries
    db.mark_proposal_status(cand_id, "COMPLETED", accuracy_fid=11.2, latency_ms=35.0)

    q_res = json.loads(tool.query_proposals(status="COMPLETED"))
    assert q_res["status"] == "SUCCESS"
    assert q_res["count"] == 1

    pf_res = json.loads(tool.get_pareto_front())
    assert pf_res["status"] == "SUCCESS"
    assert len(pf_res["pareto_front"]) == 1


# ============================================================================
# Un0ContextTool Tests
# ============================================================================


@pytest.fixture
def un0_context():
    return Un0Context(repo_root=REPO_ROOT)


@pytest.fixture
def un0_tool():
    return Un0ContextTool(name="un0_test_tool", repo_root=REPO_ROOT, auto_start=False)


def test_un0_pagination_bounds_and_clamping(un0_context):
    """Test line-clamped reading with pagination notice."""
    total_lines = len((REPO_ROOT / "un0/model.py").read_text(encoding="utf-8").splitlines())

    # Requesting 300 lines should be clamped to 150 lines max
    read_slice = un0_context.un0_read_file("un0/model.py", start_line=1, end_line=300)
    assert f"[File: un0/model.py | Total lines: {total_lines} | Showing lines 1 to 150]" in read_slice
    assert f"[NOTE: {total_lines - 150} lines remaining. Call un0_read_file with start_line=151, end_line=300 to read further]" in read_slice


def test_un0_ast_symbol_extraction_kuramoto(un0_context):
    """Test AST-aware symbol extraction of ConditionalKuramotoDynamics."""
    symbol_src = un0_context.un0_get_symbol("un0/model.py", "ConditionalKuramotoDynamics")
    assert "[Symbol: ConditionalKuramotoDynamics | File: un0/model.py | Lines 28 to 155]" in symbol_src
    assert "class ConditionalKuramotoDynamics(nn.Module):" in symbol_src
    assert "def forward(self, state: Tensor, _time: Tensor, drive: Tensor) -> Tensor:" in symbol_src


def test_un0_model_summary(un0_context):
    """Test architecture summary of cifar10/n1024 oscillator network."""
    summary = un0_context.un0_get_model_summary("cifar10/n1024")
    assert "cifar10/n1024" in summary
    assert "1024" in summary
    assert "8" in summary
    assert "1032" in summary
    assert "rk4" in summary
    assert "25" in summary
    assert "K_drive" in summary
    assert "omega" in summary


def test_un0_code_search(un0_context):
    """Test case-insensitive code search within Un-0."""
    search_res = un0_context.un0_search_code("kuramoto", directory="un0")
    assert "Search results for 'kuramoto' in 'un0'" in search_res
    assert "un0/model.py" in search_res

    not_found = un0_context.un0_search_code("nonexistent_random_symbol_99999", directory="un0")
    assert "No matches found" in not_found


def test_un0_context_tool_delegation(un0_tool):
    """Verify Un0ContextTool delegates properly to Un0Context."""
    sym = un0_tool.un0_get_symbol("un0/model.py", "ConditionalKuramotoDynamics")
    assert "class ConditionalKuramotoDynamics(nn.Module):" in sym

    summary = un0_tool.un0_get_model_summary("cifar10/n1024")
    assert "1024" in summary
