"""Unit tests for Phase 2 Node 2: phase2_generate_final_report."""

from pathlib import Path
import pytest

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.nodes.node2_phase2_generate_final_report import (
    phase2_generate_final_report,
    _find_pdflatex,
)


@pytest.fixture
def populated_report_db(tmp_path: Path) -> ResultDatabase:
    db = ResultDatabase(db_path=tmp_path / "report_test.db")

    # Reference
    ref_id = db.insert_proposal(solver="rk4", noise_model="none", is_reference=True)
    db.mark_proposal_status(ref_id, "COMPLETED", latency_ms=120.0, relative_error=0.0)

    # Completed candidate 1
    c1 = db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    db.mark_proposal_status(c1, "COMPLETED", latency_ms=15.0, relative_error=0.03, pareto_optimal=True)
    db.update_execution_status(
        c1,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="SUCCESS",
        compilation_status="SUCCESS",
        simulation_status="SUCCESS",
    )

    # Completed candidate 2
    c2 = db.insert_proposal(solver="parareal", noise_model="L0_static_mismatch", is_reference=False)
    db.mark_proposal_status(c2, "COMPLETED", latency_ms=40.0, relative_error=0.009, pareto_optimal=True)
    db.update_execution_status(
        c2,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="SUCCESS",
        compilation_status="SUCCESS",
        simulation_status="SUCCESS",
    )

    # Failed candidate 3
    c3 = db.insert_proposal(solver="par_ode", noise_model="L3_correlated_drift", is_reference=False)
    db.mark_proposal_status(c3, "FAILED", error_stage="compilation", error_message="C compiler error")
    db.update_execution_status(
        c3,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="FAILED",
        compilation_status="FAILED",
        simulation_status="PENDING",
    )

    # Record tokens
    db.record_token_usage(
        phase="phase_0",
        node_name="node2_propose_candidate",
        model="gemini-2.5-flash",
        prompt_tokens=800,
        completion_tokens=250,
        cost_usd=0.0005,
    )
    db.record_token_usage(
        phase="phase_1",
        node_name="node1_rewrite_solver",
        model="gemini-2.5-pro",
        prompt_tokens=2000,
        completion_tokens=600,
        cost_usd=0.006,
    )

    return db


def test_phase2_generate_final_report_tex_and_figures(populated_report_db: ResultDatabase, tmp_path: Path):
    cfg = load_simulation_config("config/simulation_config.yaml")
    out_dir = tmp_path / "reports_test"
    mock_llm = MockAntigravityLLM(mode="valid")

    res = phase2_generate_final_report(
        db=populated_report_db,
        config=cfg,
        output_dir=out_dir,
        llm=mock_llm,
        compile_pdf=False,
    )

    assert res["status"] in ("SUCCESS", "PARTIAL")
    tex_path = Path(res["tex_path"])
    assert tex_path.exists()
    assert tex_path.stat().st_size > 1000

    # Read LaTeX and verify structure
    tex_text = tex_path.read_text(encoding="utf-8")
    assert "\\begin{document}" in tex_text
    assert "\\section{Executive Summary}" in tex_text
    assert "\\section{Multi-Objective Pareto Analysis}" in tex_text
    assert "\\section{Pipeline Execution Reliability}" in tex_text
    assert "\\section{Swarm Resource and Token Auditing}" in tex_text
    assert "figures/pareto_frontier.pdf" in tex_text
    assert "figures/execution_breakdown.pdf" in tex_text
    assert "figures/token_cost_summary.pdf" in tex_text

    # Verify experiment subfolder routing
    exp_out_dir = Path(res["output_dir"])
    assert exp_out_dir == out_dir / cfg.experiment_name
    assert res["experiment_name"] == cfg.experiment_name

    # Verify figures exist on disk in experiment subfolder
    assert (exp_out_dir / "figures" / "pareto_frontier.pdf").exists()
    assert (exp_out_dir / "figures" / "execution_breakdown.pdf").exists()
    assert (exp_out_dir / "figures" / "token_cost_summary.pdf").exists()

    # Check token recorded for Phase 2
    summary = populated_report_db.get_token_usage_summary(phase="phase_2")
    assert summary["overall"]["total_calls"] >= 1
    assert summary["overall"]["total_tokens"] > 0


def test_phase2_generate_final_report_dynamic_pareto_point_analysis(tmp_path: Path):
    """Verify that dynamic discussion analyzes Pareto point coordinates, spatial proximity,
    clustering/overlap, and frontier clarity without rigid solver counting heuristics.
    """
    db = ResultDatabase(db_path=tmp_path / "point_analysis_test.db")
    ref_id = db.insert_proposal(solver="rk4", noise_model="none", is_reference=True)
    db.mark_proposal_status(ref_id, "COMPLETED", latency_ms=100.0, relative_error=0.0)

    # Insert two candidates with closely clustered/overlapping performance points (mutually non-dominated)
    c1 = db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    db.mark_proposal_status(c1, "COMPLETED", latency_ms=20.0, relative_error=0.0205, pareto_optimal=True)
    c2 = db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    db.mark_proposal_status(c2, "COMPLETED", latency_ms=20.4, relative_error=0.0200, pareto_optimal=True)

    cfg = load_simulation_config("config/simulation_config.yaml")
    out_dir = tmp_path / "clustering_reports"

    # Run with llm=False to verify dynamic data-driven fallback discussion
    res = phase2_generate_final_report(
        db=db,
        config=cfg,
        output_dir=out_dir,
        llm=False,
        compile_pdf=False,
    )

    tex_text = Path(res["tex_path"]).read_text(encoding="utf-8")
    assert "[pregenerated template text]" in tex_text
    assert "clustering" in tex_text or "overlapping" in tex_text
    assert "diminishing dynamical separation" in tex_text or "localized accuracy-latency plateau" in tex_text

    # Now test well-separated, crystal-clear frontier
    db_clear = ResultDatabase(db_path=tmp_path / "clear_frontier_test.db")
    ref2 = db_clear.insert_proposal(solver="rk4", noise_model="none", is_reference=True)
    db_clear.mark_proposal_status(ref2, "COMPLETED", latency_ms=100.0, relative_error=0.0)
    c_fast = db_clear.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    db_clear.mark_proposal_status(c_fast, "COMPLETED", latency_ms=12.0, relative_error=0.05, pareto_optimal=True)
    c_acc = db_clear.insert_proposal(solver="rk4", noise_model="none", is_reference=False)
    db_clear.mark_proposal_status(c_acc, "COMPLETED", latency_ms=65.0, relative_error=0.005, pareto_optimal=True)

    out_dir_clear = tmp_path / "clear_reports"
    res_clear = phase2_generate_final_report(
        db=db_clear,
        config=cfg,
        output_dir=out_dir_clear,
        llm=False,
        compile_pdf=False,
    )
    tex_clear = Path(res_clear["tex_path"]).read_text(encoding="utf-8")
    assert "[pregenerated template text]" in tex_clear
    assert "crystal-clear, well-separated trade-off boundary" in tex_clear


def test_phase2_generate_final_report_pdf_compilation(populated_report_db: ResultDatabase, tmp_path: Path):
    pdflatex_bin = _find_pdflatex()
    if not pdflatex_bin:
        pytest.skip("pdflatex not found on system; skipping full PDF compilation test")

    cfg = load_simulation_config("config/simulation_config.yaml")
    out_dir = tmp_path / "reports_pdf_test"
    mock_llm = MockAntigravityLLM(mode="valid")

    res = phase2_generate_final_report(
        db=populated_report_db,
        config=cfg,
        output_dir=out_dir,
        llm=mock_llm,
        compile_pdf=True,
    )

    assert res["status"] == "SUCCESS"
    assert res["pdf_compiled"] is True
    assert res["pdf_path"] is not None
    pdf_path = Path(res["pdf_path"])
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 5000  # PDF should be non-trivial size
