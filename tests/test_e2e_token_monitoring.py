"""End-to-End Cross-Phase Token Monitoring and Auditing Integration Test.

Validates swarm-wide token tracking and cost auditing across Phase 0, Phase 1, and Phase 2:
  1. Phase 0: Proposal generation records tokens under phase="phase_0".
  2. Phase 1: Code rewriter nodes (solver, noise, compiler) record tokens under phase="phase_1".
  3. Phase 2: Report synthesis and auditing record tokens under phase="phase_2".
  4. Token Summary: ResultDatabase.get_token_usage_summary() produces accurate overall and per-phase sums.
  5. Report Content: Publication-grade LaTeX and PDF reports incorporate the cross-phase token and dollar audit table.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.phase_0 import run_phase_0
from agentic_ml_analog_sim.phase_1 import run_phase_1
from agentic_ml_analog_sim.phase_2 import run_phase_2
from agentic_ml_analog_sim.reporting.interactive_session import InteractiveReportSession


@pytest.fixture
def test_setup(tmp_path: Path):
    """Fixture providing configuration and clean temporary SQLite ResultDatabase."""
    repo_root = Path(__file__).parents[1]
    config_file = repo_root / "config/simulation_config.yaml"
    if not config_file.is_file():
        config_file = Path("agentic-ML-analog-sim/config/simulation_config.yaml")

    config = load_simulation_config(config_file)
    db_file = tmp_path / "e2e_swarm_audit.db"
    db = ResultDatabase(db_path=db_file)

    # Pre-populate completed baseline reference design in DB
    ref = config.reference_design
    ref_id = db.insert_proposal(
        solver=ref.solver,
        noise_model=ref.noise_model,
        solver_params={
            "num_steps": ref.num_steps,
            "integration_time": ref.integration_time,
        },
        noise_params={},
        sparsity_config={"sparsity_ratio": ref.sparsity_ratio},
        selection_reason="Baseline unperturbed Kuramoto reference design",
        is_reference=True,
        status="COMPLETED",
        accuracy_fid=28.5,
        relative_error=0.0,
        latency_ms=120.0,
    )

    return config, db, db_file, ref_id


def test_e2e_cross_phase_token_monitoring(test_setup, tmp_path: Path):
    """Execute full Phase 0 -> Phase 1 -> Phase 2 loop and verify token accounting."""
    config, db, db_file, ref_id = test_setup

    # --------------------------------------------------------------------------
    # Step 1: Run Phase 0 (Propose Candidate) with Mock LLM
    # --------------------------------------------------------------------------
    mock_llm_p0 = MockAntigravityLLM(mode="valid")
    result_p0 = run_phase_0(
        config_path=Path(__file__).parents[1] / "config/simulation_config.yaml",
        db_path=db_file,
        llm=mock_llm_p0,
    )

    assert result_p0.action == "PROCEED_TO_PHASE_1_CANDIDATE"
    candidate_id = result_p0.candidate_id
    assert candidate_id is not None
    assert candidate_id != ref_id

    # Verify Phase 0 token usage recorded in DB
    summary_p0 = db.get_token_usage_summary(phase="phase_0")
    assert summary_p0["overall"]["total_calls"] >= 1
    assert summary_p0["overall"]["total_tokens"] > 0
    assert "phase_0" in summary_p0["by_phase"]
    assert summary_p0["by_phase"]["phase_0"]["total_tokens"] > 0
    assert "phase_0.node2_phase0_propose_candidate" in summary_p0["by_node"]

    p0_tokens = summary_p0["by_phase"]["phase_0"]["total_tokens"]
    p0_cost = summary_p0["by_phase"]["phase_0"]["cost_usd"]
    assert p0_tokens > 0
    assert p0_cost > 0.0

    # --------------------------------------------------------------------------
    # Step 2: Run Phase 1 on the proposed candidate with Mock LLM
    # --------------------------------------------------------------------------
    mock_llm_p1 = MockAntigravityLLM(mode="valid")
    result_p1 = run_phase_1(
        config_path=config,
        db_path=db,
        candidate_id=candidate_id,
        trigger_source="phase_0",
        target_device="cpu",
        llm=mock_llm_p1,
        dry_run=True,
        force_rewrites=True,
    )

    assert result_p1.action == "PROCEED_TO_PHASE_2"
    assert result_p1.candidate_id == candidate_id

    # Verify candidate marked COMPLETED in database
    cand_row = db.get_proposal(candidate_id)
    assert cand_row["status"] == "COMPLETED"

    # Verify Phase 1 token usage recorded in DB across rewriters
    summary_p1 = db.get_token_usage_summary(phase="phase_1")
    assert summary_p1["overall"]["total_calls"] >= 3  # solver, noise, compiler rewriters
    assert summary_p1["overall"]["total_tokens"] > 0
    assert "phase_1" in summary_p1["by_phase"]
    assert summary_p1["by_phase"]["phase_1"]["total_tokens"] > 0
    assert "phase_1.node1_phase1_solver_rewriter" in summary_p1["by_node"]
    assert "phase_1.node2_phase1_noise_rewriter" in summary_p1["by_node"]
    assert "phase_1.node3_phase1_compiler_rewriter" in summary_p1["by_node"]

    p1_tokens = summary_p1["by_phase"]["phase_1"]["total_tokens"]
    p1_cost = summary_p1["by_phase"]["phase_1"]["cost_usd"]
    assert p1_tokens > 0
    assert p1_cost > 0.0

    # --------------------------------------------------------------------------
    # Step 3: Run Phase 2 (Limits & Report Synthesis) with Mock LLM
    # --------------------------------------------------------------------------
    reports_dir = tmp_path / "reports"
    mock_llm_p2 = MockAntigravityLLM(mode="valid")
    result_p2 = run_phase_2(
        config_path=config,
        db_path=db,
        candidate_id=candidate_id,
        output_dir=reports_dir,
        llm=mock_llm_p2,
        force_report=True,
        compile_pdf=True,
    )

    assert result_p2.action == "REPORT_GENERATED"
    assert result_p2.tex_path is not None
    assert Path(result_p2.tex_path).exists()

    # Verify Phase 2 token usage recorded in DB
    summary_p2 = db.get_token_usage_summary(phase="phase_2")
    assert summary_p2["overall"]["total_calls"] >= 1
    assert summary_p2["overall"]["total_tokens"] > 0
    assert "phase_2" in summary_p2["by_phase"]
    assert (
        "phase_2.node2_phase2_generate_final_report" in summary_p2["by_node"]
        or "phase_2.phase2_generate_final_report" in summary_p2["by_node"]
    )

    p2_tokens = summary_p2["by_phase"]["phase_2"]["total_tokens"]
    p2_cost = summary_p2["by_phase"]["phase_2"]["cost_usd"]
    assert p2_tokens > 0
    assert p2_cost > 0.0

    # --------------------------------------------------------------------------
    # Step 4: Cross-Phase Token Summary Verification
    # --------------------------------------------------------------------------
    summary = db.get_token_usage_summary()
    assert "overall" in summary
    assert "by_phase" in summary

    by_phase = summary["by_phase"]
    assert "phase_0" in by_phase
    assert "phase_1" in by_phase
    assert "phase_2" in by_phase

    assert by_phase["phase_0"]["total_tokens"] > 0
    assert by_phase["phase_1"]["total_tokens"] > 0
    assert by_phase["phase_2"]["total_tokens"] > 0

    expected_total_tokens = (
        by_phase["phase_0"]["total_tokens"]
        + by_phase["phase_1"]["total_tokens"]
        + by_phase["phase_2"]["total_tokens"]
    )
    assert summary["overall"]["total_tokens"] == expected_total_tokens

    expected_prompt_tokens = (
        by_phase["phase_0"]["prompt_tokens"]
        + by_phase["phase_1"]["prompt_tokens"]
        + by_phase["phase_2"]["prompt_tokens"]
    )
    assert summary["overall"]["total_prompt_tokens"] == expected_prompt_tokens

    expected_completion_tokens = (
        by_phase["phase_0"]["completion_tokens"]
        + by_phase["phase_1"]["completion_tokens"]
        + by_phase["phase_2"]["completion_tokens"]
    )
    assert summary["overall"]["total_completion_tokens"] == expected_completion_tokens

    expected_total_cost = (
        by_phase["phase_0"]["cost_usd"]
        + by_phase["phase_1"]["cost_usd"]
        + by_phase["phase_2"]["cost_usd"]
    )
    assert abs(summary["overall"]["total_cost_usd"] - expected_total_cost) < 1e-6

    expected_calls = (
        by_phase["phase_0"]["num_calls"]
        + by_phase["phase_1"]["num_calls"]
        + by_phase["phase_2"]["num_calls"]
    )
    assert summary["overall"]["total_calls"] == expected_calls

    # --------------------------------------------------------------------------
    # Step 5: Report Content Verification (LaTeX & PDF Artifacts)
    # --------------------------------------------------------------------------
    tex_path = Path(result_p2.tex_path)
    tex_text = tex_path.read_text(encoding="utf-8")

    # Document and section structure
    assert "\\begin{document}" in tex_text
    assert "\\section{Swarm Resource and Token Auditing}" in tex_text

    # Phase-specific table entries
    assert "Phase 0 (Candidate Exploration)" in tex_text
    assert (
        "Phase 1 (Code Rewrite \\& Simulation)" in tex_text
        or "Phase 1 (Code Rewrite & Simulation)" in tex_text
    )
    assert (
        "Phase 2 (Auditing \\& Final Reporting)" in tex_text
        or "Phase 2 (Auditing & Final Reporting)" in tex_text
    )
    assert "\\textbf{Swarm Total}" in tex_text

    # Token counts in LaTeX table
    assert f"{by_phase['phase_0']['total_tokens']:,}" in tex_text
    assert f"{by_phase['phase_1']['total_tokens']:,}" in tex_text
    assert f"{by_phase['phase_2']['total_tokens']:,}" in tex_text
    assert f"{expected_total_tokens:,}" in tex_text

    # Dollar cost in LaTeX table
    assert f"\\${expected_total_cost:.4f}" in tex_text

    # Figure generation verification
    fig_dir = Path(result_p2.tex_path).parent / "figures"
    assert (fig_dir / "pareto_frontier.pdf").exists()
    assert (fig_dir / "execution_breakdown.pdf").exists()
    assert (fig_dir / "token_cost_summary.pdf").exists()

    # PDF compilation verification (if pdflatex was executed)
    if result_p2.pdf_compiled and result_p2.pdf_path:
        pdf_file = Path(result_p2.pdf_path)
        assert pdf_file.exists()
        assert pdf_file.stat().st_size > 5000


def test_e2e_interactive_session_updates_token_monitoring(test_setup, tmp_path: Path):
    """Verify that post-report interactive consultant turns also update cross-phase token auditing."""
    config, db, db_file, ref_id = test_setup

    # Record some Phase 0 and Phase 1 tokens
    db.record_token_usage(
        phase="phase_0",
        node_name="node2_phase0_propose_candidate",
        model="gemini-2.5-pro",
        prompt_tokens=500,
        completion_tokens=150,
        total_tokens=650,
        cost_usd=0.001,
    )
    db.record_token_usage(
        phase="phase_1",
        node_name="node1_phase1_solver_rewriter",
        model="gemini-2.5-pro",
        prompt_tokens=800,
        completion_tokens=250,
        total_tokens=1050,
        cost_usd=0.002,
    )

    report_file = tmp_path / "test_report.tex"
    report_file.write_text("\\section{Executive Summary}\nOptimization succeeded.", encoding="utf-8")

    mock_llm = MockAntigravityLLM(
        responses=[
            "The Euler solver provided a 3.4x speedup over the reference baseline.",
            "Static mismatch noise (L0) degraded FID by less than 1.2%.",
        ]
    )

    session = InteractiveReportSession(
        db=db,
        report_tex_path=report_file,
        llm=mock_llm,
    )

    # Execute interactive turns
    resp1 = session.ask("What speedup was achieved?")
    assert "3.4x" in resp1

    resp2 = session.ask("How did L0 noise affect FID?")
    assert "1.2%" in resp2

    # Verify interactive turns were recorded in Phase 2
    summary = db.get_token_usage_summary(phase="phase_2")
    assert summary["overall"]["total_calls"] == 2
    assert "phase_2.interactive_session" in summary["by_node"]
    assert summary["by_node"]["phase_2.interactive_session"]["num_calls"] == 2

    # Verify swarm-wide total incorporates Phase 0, Phase 1, and Phase 2 (interactive)
    full_summary = db.get_token_usage_summary()
    assert full_summary["overall"]["total_calls"] == 4
    expected_tokens = 650 + 1050 + summary["overall"]["total_tokens"]
    assert full_summary["overall"]["total_tokens"] == expected_tokens
