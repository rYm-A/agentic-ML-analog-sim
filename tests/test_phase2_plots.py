"""Unit tests for Phase 2 Matplotlib plotting engine and PlotTool FastMCP server."""

import json
from pathlib import Path
import pytest

from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.reporting.plot_generator import (
    generate_pareto_plot,
    generate_breakdown_plot,
    generate_token_cost_plot,
)
from agentic_ml_analog_sim.tools.plot_tool import PlotTool


@pytest.fixture
def populated_db(tmp_path: Path) -> ResultDatabase:
    db_file = tmp_path / "plot_test.db"
    db = ResultDatabase(db_path=db_file)

    # Insert reference
    ref_id = db.insert_proposal(
        solver="rk4",
        noise_model="none",
        is_reference=True,
    )
    db.mark_proposal_status(
        ref_id,
        "COMPLETED",
        latency_ms=100.0,
        relative_error=0.0,
        accuracy_fid=0.0,
    )

    # Insert candidate 1: Fast, moderate error (Pareto)
    c1 = db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    db.mark_proposal_status(
        c1,
        "COMPLETED",
        latency_ms=12.5,
        relative_error=0.045,
        accuracy_fid=1.2,
        pareto_optimal=True,
    )
    db.update_execution_status(
        c1,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="SUCCESS",
        compilation_status="SUCCESS",
        simulation_status="SUCCESS",
    )

    # Insert candidate 2: Accurate, moderate latency (Pareto)
    c2 = db.insert_proposal(solver="parareal", noise_model="L0_static_mismatch", is_reference=False)
    db.mark_proposal_status(
        c2,
        "COMPLETED",
        latency_ms=45.0,
        relative_error=0.008,
        accuracy_fid=0.3,
        pareto_optimal=True,
    )
    db.update_execution_status(
        c2,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="SUCCESS",
        compilation_status="SUCCESS",
        simulation_status="SUCCESS",
    )

    # Insert candidate 3: Dominated
    c3 = db.insert_proposal(solver="euler_backward", noise_model="L1_stochastic_parameter_noise", is_reference=False)
    db.mark_proposal_status(
        c3,
        "COMPLETED",
        latency_ms=80.0,
        relative_error=0.05,
        accuracy_fid=1.5,
        pareto_optimal=False,
    )
    db.update_execution_status(
        c3,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="SUCCESS",
        compilation_status="SUCCESS",
        simulation_status="SUCCESS",
    )

    # Insert candidate 4: Failed at noise rewrite
    c4 = db.insert_proposal(solver="par_ode", noise_model="L3_correlated_drift", is_reference=False)
    db.mark_proposal_status(
        c4,
        "FAILED",
        error_stage="rewrite_noise",
        error_message="Nonlinear drift out of bound",
    )
    db.update_execution_status(
        c4,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="FAILED",
        compilation_status="FAILED",
        simulation_status="PENDING",
    )

    # Record some token usage
    db.record_token_usage(
        phase="phase_0",
        node_name="node2_propose_candidate",
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=300,
        cost_usd=0.0006,
    )
    db.record_token_usage(
        phase="phase_1",
        node_name="node1_rewrite_solver",
        model="gemini-2.5-pro",
        prompt_tokens=2500,
        completion_tokens=800,
        cost_usd=0.008,
    )
    db.record_token_usage(
        phase="phase_2",
        node_name="phase2_generate_final_report",
        model="gemini-2.5-flash",
        prompt_tokens=4000,
        completion_tokens=1500,
        cost_usd=0.003,
    )

    return db


def test_generate_pareto_plot(populated_db: ResultDatabase, tmp_path: Path):
    out_dir = tmp_path / "figures"
    res = generate_pareto_plot(populated_db, out_dir, "test_pareto")

    pdf_file = Path(res["pdf"])
    png_file = Path(res["png"])

    assert pdf_file.exists()
    assert png_file.exists()
    assert pdf_file.stat().st_size > 1000
    assert png_file.stat().st_size > 1000


def test_generate_breakdown_plot(populated_db: ResultDatabase, tmp_path: Path):
    out_dir = tmp_path / "figures"
    res = generate_breakdown_plot(populated_db, out_dir, "test_breakdown")

    pdf_file = Path(res["pdf"])
    png_file = Path(res["png"])

    assert pdf_file.exists()
    assert png_file.exists()
    assert pdf_file.stat().st_size > 1000
    assert png_file.stat().st_size > 1000


def test_generate_token_cost_plot(populated_db: ResultDatabase, tmp_path: Path):
    out_dir = tmp_path / "figures"
    res = generate_token_cost_plot(populated_db, out_dir, "test_tokens")

    pdf_file = Path(res["pdf"])
    png_file = Path(res["png"])

    assert pdf_file.exists()
    assert png_file.exists()
    assert pdf_file.stat().st_size > 1000
    assert png_file.stat().st_size > 1000


def test_plot_tool_mcp(populated_db: ResultDatabase, tmp_path: Path):
    out_dir = tmp_path / "mcp_figures"
    tool = PlotTool(db=populated_db, default_output_dir=out_dir, deploy=False)

    # Test all plots generated via MCP
    all_res = tool.generate_all_plots(str(out_dir))
    data = json.loads(all_res)

    assert data["status"] == "SUCCESS"
    assert "pareto" in data["plots"]
    assert "breakdown" in data["plots"]
    assert "token_cost" in data["plots"]

    for plot_type, files in data["plots"].items():
        assert Path(files["pdf"]).exists()
        assert Path(files["png"]).exists()
