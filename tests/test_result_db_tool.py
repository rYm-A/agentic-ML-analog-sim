"""Tests for ResultDBTool ChiaTool / FastMCP integration and direct execution."""

import json
import tempfile
from pathlib import Path
import pytest

from agentic_ml_analog_sim.db import ResultDatabase
from agentic_ml_analog_sim.tools import (
    ResultDBTool,
    add_comment,
    get_pareto_front,
    get_reference_result,
    insert_proposal,
    query_proposals,
)


@pytest.fixture
def temp_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tool.db"
        db = ResultDatabase(db_path)
        tool = ResultDBTool("test_tool", db=db, deploy=False)
        yield tool


def test_tool_methods_and_helpers(temp_tool):
    db = temp_tool.db

    # Reference not found initially
    ref_res = json.loads(temp_tool.get_reference_result())
    assert ref_res["status"] == "NOT_FOUND"

    # Insert proposal via tool
    p1_res = json.loads(
        temp_tool.insert_proposal(
            solver="euler",
            noise_model="none",
            solver_params='{"steps": 10}',
            noise_params="{}",
            sparsity_config='{"sparsity_ratio": 0.0}',
            selection_reason="Fast exploration",
        )
    )
    assert p1_res["status"] == "SUCCESS"
    p1_id = p1_res["candidate_id"]

    # Duplicate rejection
    dup_res = json.loads(
        temp_tool.insert_proposal(
            solver="euler",
            noise_model="none",
            solver_params='{"steps": 10}',
            noise_params="{}",
            sparsity_config='{"sparsity_ratio": 0.0}',
            selection_reason="Duplicate",
        )
    )
    assert dup_res["status"] == "DUPLICATE"

    # Add comment
    com_res = json.loads(
        temp_tool.add_comment(
            candidate_id=p1_id,
            phase="Phase_1",
            comment="Validation passed.",
            agent_name="TestAgent",
        )
    )
    assert com_res["status"] == "SUCCESS"

    # Complete evaluation
    db.mark_proposal_status(p1_id, "COMPLETED", accuracy_fid=14.0, latency_ms=45.0)

    # Query proposals
    q_res = json.loads(temp_tool.query_proposals(status="COMPLETED"))
    assert q_res["status"] == "SUCCESS"
    assert q_res["count"] == 1

    # Pareto front
    pf_res = json.loads(temp_tool.get_pareto_front())
    assert pf_res["status"] == "SUCCESS"
    assert len(pf_res["pareto_front"]) == 1

    # Direct helper function test
    direct_q = json.loads(query_proposals(db, status="COMPLETED"))
    assert direct_q["status"] == "SUCCESS"
