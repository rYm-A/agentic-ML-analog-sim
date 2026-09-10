"""Unit tests for Phase 2 database extensions: token tracking, execution statistics, and count checks."""

import pytest
from pathlib import Path
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool


@pytest.fixture
def temp_db(tmp_path: Path) -> ResultDatabase:
    db_file = tmp_path / "test_phase2.db"
    return ResultDatabase(db_path=db_file)


def test_token_usage_recording_and_summary(temp_db: ResultDatabase):
    # Record some token usages across phases and models
    id1 = temp_db.record_token_usage(
        phase="phase_0",
        node_name="node2_propose_candidate",
        model="gemini-2.5-flash",
        prompt_tokens=500,
        completion_tokens=150,
        cost_usd=0.0003,
        duration_seconds=1.2,
    )
    assert id1 > 0

    id2 = temp_db.record_token_usage(
        phase="phase_1",
        node_name="node1_rewrite_solver",
        model="gemini-2.5-pro",
        prompt_tokens=1200,
        completion_tokens=400,
        cost_usd=0.004,
        duration_seconds=3.5,
    )
    assert id2 > id1

    id3 = temp_db.record_token_usage(
        phase="phase_2",
        node_name="phase2_generate_final_report",
        model="gemini-2.5-flash",
        prompt_tokens=3000,
        completion_tokens=1200,
        cost_usd=0.002,
        duration_seconds=5.0,
    )
    assert id3 > id2

    # Query overall summary
    summary = temp_db.get_token_usage_summary()
    assert summary["overall"]["total_prompt_tokens"] == 4700
    assert summary["overall"]["total_completion_tokens"] == 1750
    assert summary["overall"]["total_tokens"] == 6450
    assert pytest.approx(summary["overall"]["total_cost_usd"], 1e-5) == 0.0063
    assert summary["overall"]["total_calls"] == 3

    # Check phase breakdown
    assert "phase_0" in summary["by_phase"]
    assert summary["by_phase"]["phase_0"]["total_tokens"] == 650
    assert "phase_1" in summary["by_phase"]
    assert summary["by_phase"]["phase_1"]["total_tokens"] == 1600
    assert "phase_2" in summary["by_phase"]
    assert summary["by_phase"]["phase_2"]["total_tokens"] == 4200

    # Query phase_1 specific summary
    p1_summary = temp_db.get_token_usage_summary(phase="phase_1")
    assert p1_summary["overall"]["total_tokens"] == 1600
    assert p1_summary["overall"]["total_calls"] == 1


def test_completed_experiments_count(temp_db: ResultDatabase):
    assert temp_db.get_completed_experiments_count() == 0

    # Reference proposal (should not count as candidate experiment)
    temp_db.insert_proposal(
        solver="rk4",
        noise_model="none",
        is_reference=True,
        status="COMPLETED",
    )
    assert temp_db.get_completed_experiments_count() == 0

    # Candidate 1: COMPLETED
    cid1 = temp_db.insert_proposal(
        solver="euler",
        noise_model="none",
        is_reference=False,
    )
    temp_db.mark_proposal_status(cid1, "COMPLETED", latency_ms=10.0, relative_error=0.01)
    assert temp_db.get_completed_experiments_count() == 1

    # Candidate 2: FAILED
    cid2 = temp_db.insert_proposal(
        solver="parareal",
        noise_model="L1_stochastic_parameter_noise",
        is_reference=False,
    )
    temp_db.mark_proposal_status(cid2, "FAILED", error_stage="compilation", error_message="SyntaxError")
    assert temp_db.get_completed_experiments_count() == 2

    # Candidate 3: PENDING (not yet evaluated)
    temp_db.insert_proposal(
        solver="euler_backward",
        noise_model="L0_static_mismatch",
        is_reference=False,
    )
    assert temp_db.get_completed_experiments_count() == 2


def test_execution_statistics_and_updates(temp_db: ResultDatabase):
    cid1 = temp_db.insert_proposal(
        solver="euler",
        noise_model="L0_static_mismatch",
        is_reference=False,
    )

    temp_db.update_execution_status(
        cid1,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="SUCCESS",
        compilation_status="SUCCESS",
        simulation_status="SUCCESS",
    )

    cid2 = temp_db.insert_proposal(
        solver="parareal",
        noise_model="L3_correlated_drift",
        is_reference=False,
    )

    temp_db.update_execution_status(
        cid2,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="FAILED",
        rewrite_compiler_status="PENDING",
        compilation_status="FAILED",
        simulation_status="PENDING",
        error_stage="rewrite_noise",
        error_message="Drift tensor mismatch",
    )

    stats = temp_db.get_execution_statistics()
    assert stats["total_candidates"] == 2
    assert stats["compilations_succeeded_count"] == 1
    assert stats["compilations_failed_count"] == 1
    assert stats["simulations_succeeded_count"] == 1
    assert stats["simulations_failed_count"] == 0
    assert stats["rewrites_solver_succeeded_count"] == 2
    assert stats["rewrites_noise_failed_count"] == 1
    assert cid2 in stats["details"]["rewrites_noise"]["failed"]


def test_result_db_tool_phase2_tools(temp_db: ResultDatabase):
    import json
    tool = ResultDBTool(db=temp_db, deploy=False)

    # Test record_token_usage via tool
    res = tool.record_token_usage(
        phase="phase_2",
        node_name="test_node",
        model="gemini-2.5-flash",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.0001,
    )
    data = json.loads(res)
    assert data["status"] == "SUCCESS"
    assert "token_usage_id" in data

    # Test get_token_usage_summary via tool
    summary_res = tool.get_token_usage_summary()
    sdata = json.loads(summary_res)
    assert sdata["status"] == "SUCCESS"
    assert sdata["token_usage"]["overall"]["total_tokens"] == 150

    # Test get_completed_experiments_count via tool
    count_res = tool.get_completed_experiments_count()
    cdata = json.loads(count_res)
    assert cdata["status"] == "SUCCESS"
    assert cdata["completed_experiments_count"] == 0

    # Test get_execution_statistics via tool
    stats_res = tool.get_execution_statistics()
    stdata = json.loads(stats_res)
    assert stdata["status"] == "SUCCESS"
    assert stdata["statistics"]["total_candidates"] == 0
