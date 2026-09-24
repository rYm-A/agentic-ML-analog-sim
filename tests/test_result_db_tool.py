"""Tests for ResultDBTool ChiaTool / FastMCP integration and direct execution."""

import json
import tempfile
from pathlib import Path
import pytest

from agentic_ml_analog_sim.db import ResultDatabase
from agentic_ml_analog_sim.tools import (
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


def test_read_only_tool_properties_and_methods():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ro.db"
        db = ResultDatabase(db_path)
        ro_tool = ResultDBReadOnlyTool(db=db, deploy=False)

        assert ro_tool.read_only is True
        assert ro_tool.name == "result_db_read"

        # Check empty proposals
        empty_res = json.loads(ro_tool.read_prior_proposals())
        assert empty_res["status"] == "EMPTY"
        assert empty_res["count"] == 0

        # Insert a proposal directly into the database
        cand_id = db.insert_proposal(
            iteration=1,
            solver="euler",
            noise_model="L0_static_mismatch",
            solver_params={"num_steps": 5, "dt": 0.2},
            noise_params={"sigma": 0.01},
            sparsity_config={"sparsity_ratio": 0.4},
            status="EVALUATING",
            selection_reason="Initial exploration",
            is_active_evaluation=True,
        )
        db.add_comment(
            candidate_id=cand_id,
            phase="Phase_0",
            agent_name="Node2_Agent",
            comment="Candidate proposed for Kuramoto simulation.",
        )

        # Test read_prior_proposals
        prior_res = json.loads(ro_tool.read_prior_proposals())
        assert prior_res["status"] == "SUCCESS"
        assert prior_res["count"] == 1
        assert prior_res["proposals"][0]["candidate_id"] == cand_id
        assert prior_res["proposals"][0]["solver"] == "euler"
        assert len(prior_res["proposals"][0]["comments"]) == 1

        # Test get_candidate_history_summary
        summary_res = json.loads(ro_tool.get_candidate_history_summary())
        assert summary_res["status"] == "SUCCESS"
        assert summary_res["total_explored"] == 1
        assert summary_res["configurations"][0]["solver"] == "euler"
        assert summary_res["configurations"][0]["noise_model"] == "L0_static_mismatch"
        assert summary_res["configurations"][0]["is_active_evaluation"] is True

        # Test get_active_evaluations
        active_res = json.loads(ro_tool.get_active_evaluations())
        assert active_res["status"] == "SUCCESS"
        assert active_res["active_evaluations_count"] == 1
        assert active_res["active_proposals"][0]["candidate_id"] == cand_id

        # Test get_proposal_comments
        comments_res = json.loads(ro_tool.get_proposal_comments(cand_id))
        assert comments_res["status"] == "SUCCESS"
        assert comments_res["count"] == 1
        assert "Kuramoto simulation" in comments_res["comments"][0]["comment"]

        # Test alias ResultDBReadTool
        alias_tool = ResultDBReadTool(db=db, deploy=False)
        assert alias_tool.read_only is True


def test_read_only_tool_blocks_mutations():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ro_block.db"
        db = ResultDatabase(db_path)
        ro_tool = ResultDBReadOnlyTool(db=db, deploy=False)

        # Calling insert_proposal must raise PermissionError
        with pytest.raises(PermissionError) as exc_info:
            ro_tool.insert_proposal(
                solver="rk4",
                noise_model="none",
                solver_params="{}",
                noise_params="{}",
                sparsity_config="{}",
                selection_reason="Attempted mutation",
            )
        assert "read-only access" in str(exc_info.value)

        # Calling add_comment must raise PermissionError
        with pytest.raises(PermissionError) as exc_info:
            ro_tool.add_comment(
                candidate_id="cand_test",
                phase="phase_0",
                comment="Attempted comment mutation",
            )
        assert "read-only access" in str(exc_info.value)

        # Calling record_token_usage must raise PermissionError
        with pytest.raises(PermissionError) as exc_info:
            ro_tool.record_token_usage(
                phase="phase_0",
                node_name="node2",
                model="test_model",
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
            )
        assert "read-only access" in str(exc_info.value)

        # Ensure mutation tools are NOT in MCP registered tools
        # FastMCP tool manager stores tools or tool keys
        if hasattr(ro_tool.mcp, "_tool_manager"):
            tool_names = list(getattr(ro_tool.mcp._tool_manager, "_tools", {}).keys())
            for t_name in tool_names:
                assert "insert_proposal" not in t_name
                assert "add_comment" not in t_name
                assert "record_token_usage" not in t_name


def test_read_only_helpers():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_helpers.db"
        db = ResultDatabase(db_path)

        cand_id = db.insert_proposal(
            iteration=1,
            solver="rk4",
            noise_model="L1_stochastic_parameter_noise",
            solver_params={"num_steps": 15},
            noise_params={"sigma": 0.05},
            sparsity_config={"sparsity_ratio": 0.2},
            status="COMPLETED",
            selection_reason="Helper test",
        )
        db.add_comment(cand_id, "phase_0", "Node2", "Helper comment")

        # Test read_prior_proposals helper
        proposals_json = read_prior_proposals(db)
        data = json.loads(proposals_json)
        assert data["status"] == "SUCCESS"
        assert data["count"] == 1

        # Test get_candidate_history_summary helper
        summary_json = get_candidate_history_summary(db)
        data_s = json.loads(summary_json)
        assert data_s["status"] == "SUCCESS"
        assert data_s["configurations"][0]["solver"] == "rk4"

        # Test get_proposal_comments helper
        com_json = get_proposal_comments(db, cand_id)
        data_c = json.loads(com_json)
        assert data_c["status"] == "SUCCESS"
        assert len(data_c["comments"]) == 1

        # Test get_active_evaluations helper
        active_json = get_active_evaluations(db)
        data_a = json.loads(active_json)
        assert data_a["status"] == "SUCCESS"


def test_node2_receives_read_only_tool_and_avoids_repeated_proposals():
    """Verify Node 2 is given just-read database access and avoids duplicate proposals."""
    from agentic_ml_analog_sim.config import load_simulation_config
    from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
    from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
        _bind_tools,
        node2_phase0_propose_candidate,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_node2_ro.db"
        db = ResultDatabase(db_path)

        # 1. Verify _bind_tools converts writable ResultDBTool to ResultDBReadOnlyTool
        writable_tool = ResultDBTool(db=db, deploy=False)
        bound = _bind_tools(db, [writable_tool])
        db_tools = [t for t in bound if isinstance(t, ResultDBReadOnlyTool)]
        assert len(db_tools) == 1
        assert db_tools[0].read_only is True

        # 2. Run iteration 1: generates candidate 1
        mock_llm = MockAntigravityLLM(mode="valid")
        config = load_simulation_config("config/simulation_config.yaml")

        res_iter1 = node2_phase0_propose_candidate(
            config=config,
            db=db,
            llm=mock_llm,
            tools=[db_tools[0]],
            iteration=1,
        )
        assert isinstance(res_iter1, dict)
        cand1_id = res_iter1["candidate_id"]
        prop1 = db.get_proposal(cand1_id)
        assert prop1 is not None
        assert prop1["solver"] == "euler"

        # Record candidate 1 into database to simulate completion of iteration 1
        db.mark_proposal_status(
            cand1_id,
            "COMPLETED",
            accuracy_fid=12.5,
            latency_ms=30.0,
        )

        # 3. Run iteration 2: Node 2 reads prior proposals via its tool and avoids generating duplicate!
        res_iter2 = node2_phase0_propose_candidate(
            config=config,
            db=db,
            llm=mock_llm,
            tools=[db_tools[0]],
            iteration=2,
        )
        assert isinstance(res_iter2, dict)
        cand2_id = res_iter2["candidate_id"]
        prop2 = db.get_proposal(cand2_id)
        assert prop2 is not None

        # Verify candidate 2 is a novel proposal that does not duplicate candidate 1
        assert cand2_id != cand1_id
        assert (prop2["solver"], prop2["solver_params"], prop2["sparsity_config"]) != (
            prop1["solver"],
            prop1["solver_params"],
            prop1["sparsity_config"],
        )


