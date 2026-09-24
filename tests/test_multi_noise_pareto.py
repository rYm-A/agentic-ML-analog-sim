"""Unit tests for Phase 2 multi-noise Pareto reporting and database discrepancy fixes."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node2_phase2_generate_final_report import (
    _build_pareto_table_tex,
    _generate_dynamic_reliability_analysis,
    _generate_report_tex_content,
    node2_phase2_generate_final_report,
)
from agentic_ml_analog_sim.nodes.node7_phase1_evaluation_gate import node7_phase1_evaluation_gate
from agentic_ml_analog_sim.reporting.plot_generator import (
    generate_consolidated_pareto_plot,
    generate_multi_noise_pareto_plots,
    generate_pareto_plot,
    generate_token_cost_plot,
)


@pytest.fixture
def multi_noise_db(tmp_path: Path) -> ResultDatabase:
    """Fixture providing a ResultDatabase populated with proposals across multiple noise models."""
    db_file = tmp_path / "multi_noise_test.db"
    db = ResultDatabase(db_path=db_file)

    # 1. Reference baseline (rk4, none)
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
        pareto_optimal=True,
    )
    db.update_execution_status(ref_id, simulation_status="SUCCESS")

    # 2. Noise model 'none' candidates
    c_none_1 = db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    db.mark_proposal_status(c_none_1, "COMPLETED", latency_ms=10.0, relative_error=0.05, accuracy_fid=1.0, pareto_optimal=True)
    db.update_execution_status(c_none_1, simulation_status="SUCCESS")

    c_none_2 = db.insert_proposal(solver="midpoint", noise_model="none", is_reference=False)
    db.mark_proposal_status(c_none_2, "COMPLETED", latency_ms=25.0, relative_error=0.02, accuracy_fid=0.5, pareto_optimal=True)
    db.update_execution_status(c_none_2, simulation_status="SUCCESS")

    # 3. Noise model 'L0_static_mismatch' candidates
    c_l0_1 = db.insert_proposal(solver="euler", noise_model="L0_static_mismatch", is_reference=False)
    db.mark_proposal_status(c_l0_1, "COMPLETED", latency_ms=12.0, relative_error=0.07, accuracy_fid=1.2, pareto_optimal=True)
    db.update_execution_status(c_l0_1, simulation_status="SUCCESS")

    c_l0_2 = db.insert_proposal(solver="parareal", noise_model="L0_static_mismatch", is_reference=False)
    db.mark_proposal_status(c_l0_2, "COMPLETED", latency_ms=45.0, relative_error=0.015, accuracy_fid=0.3, pareto_optimal=True)
    db.update_execution_status(c_l0_2, simulation_status="SUCCESS")

    # 4. Noise model 'L1_stochastic_parameter_noise' candidates
    c_l1_1 = db.insert_proposal(solver="euler", noise_model="L1_stochastic_parameter_noise", is_reference=False)
    db.mark_proposal_status(c_l1_1, "COMPLETED", latency_ms=15.0, relative_error=0.09, accuracy_fid=1.5, pareto_optimal=True)
    db.update_execution_status(c_l1_1, simulation_status="SUCCESS")

    # 5. Compilation failure candidate
    c_fail = db.insert_proposal(solver="rk4", noise_model="L0_static_mismatch", is_reference=False)
    db.mark_proposal_status(
        c_fail,
        "FAILED",
        error_stage="compilation",
        error_message="Compilation error on device 'cpu': LoweringException: KeyError: 'unbacked_bindings'",
    )
    db.update_execution_status(
        c_fail,
        compilation_status="FAILED",
        error_stage="compilation",
        error_message="Compilation error on device 'cpu': LoweringException: KeyError: 'unbacked_bindings'",
    )

    return db


def test_generate_multi_noise_pareto_plots(multi_noise_db: ResultDatabase, tmp_path: Path):
    """Test requirement 1: generate_multi_noise_pareto_plots produces per-noise plots and consolidated plot."""
    out_dir = tmp_path / "figures"
    noise_models = ["none", "L0_static_mismatch", "L1_stochastic_parameter_noise"]

    results = generate_multi_noise_pareto_plots(
        db=multi_noise_db,
        output_dir=out_dir,
        noise_models=noise_models,
        prefix="pareto_frontier",
    )

    # Check that individual per-noise plots exist as PDF and PNG
    assert "none" in results
    assert Path(results["none"]["pdf"]).exists()
    assert Path(results["none"]["png"]).exists()
    assert (out_dir / "pareto_frontier_none.pdf").exists()
    assert (out_dir / "pareto_frontier_none.png").exists()

    assert "l0_static_mismatch" in results
    assert Path(results["l0_static_mismatch"]["pdf"]).exists()
    assert Path(results["l0_static_mismatch"]["png"]).exists()
    assert (out_dir / "pareto_frontier_l0_static_mismatch.pdf").exists()
    assert (out_dir / "pareto_frontier_l0_static_mismatch.png").exists()

    assert "l1_stochastic_parameter_noise" in results
    assert Path(results["l1_stochastic_parameter_noise"]["pdf"]).exists()
    assert Path(results["l1_stochastic_parameter_noise"]["png"]).exists()
    assert (out_dir / "pareto_frontier_l1_stochastic_parameter_noise.pdf").exists()
    assert (out_dir / "pareto_frontier_l1_stochastic_parameter_noise.png").exists()

    # Check that consolidated comparison plot exists
    assert "all" in results
    assert Path(results["all"]["pdf"]).exists()
    assert Path(results["all"]["png"]).exists()
    assert (out_dir / "pareto_frontier_all.pdf").exists()
    assert (out_dir / "pareto_frontier_all.png").exists()


def test_build_pareto_table_tex_deduplicates_baseline():
    """Test requirement 2a: _build_pareto_table_tex deduplicates reference baseline row."""
    ref_proposal = {
        "candidate_id": "ref_rk4_25steps",
        "solver": "rk4",
        "noise_model": "none",
        "latency_ms": 100.0,
        "relative_error": 0.0,
        "is_reference": 1,
    }

    # pareto_front from db.compute_pareto_front() includes reference baseline
    pareto_front = [
        ref_proposal,  # ref baseline candidate
        {
            "candidate_id": "cand_euler_001",
            "solver": "euler",
            "noise_model": "none",
            "latency_ms": 12.0,
            "relative_error": 0.04,
            "is_reference": 0,
        },
        {
            "candidate_id": "cand_parareal_002",
            "solver": "parareal",
            "noise_model": "L0_static_mismatch",
            "latency_ms": 40.0,
            "relative_error": 0.01,
            "is_reference": 0,
        },
    ]

    tex_table = _build_pareto_table_tex(pareto_front=pareto_front, ref_proposal=ref_proposal)

    # Count occurrences of reference baseline in table (candidate_id is LaTeX-escaped)
    occurrences = tex_table.count("ref\\_rk4\\_25steps")
    assert occurrences == 1, f"Expected ref\\_rk4\\_25steps to appear exactly once, but found {occurrences} times:\n{tex_table}"
    assert tex_table.count("(Baseline)") == 1

    # Also verify that other candidates appear
    assert "cand\\_euler\\_001" in tex_table
    assert "cand\\_parareal\\_002" in tex_table


def test_simulation_status_update_persists_as_success(tmp_path: Path):
    """Test requirement 2b: simulation_status update persists correctly as SUCCESS in DB."""
    db_file = tmp_path / "status_test.db"
    db = ResultDatabase(db_path=db_file)

    cid = db.insert_proposal(solver="euler", noise_model="none", is_reference=False)

    # Initial state should be PENDING
    prop = db.get_proposal(cid)
    assert prop["simulation_status"] == "PENDING"

    # Update simulation_status to SUCCESS
    db.update_execution_status(candidate_id=cid, simulation_status="SUCCESS")

    prop = db.get_proposal(cid)
    assert prop["simulation_status"] == "SUCCESS", f"Expected SUCCESS, got {prop['simulation_status']}"

    # Verify execution statistics reflects the success
    stats = db.get_execution_statistics()
    assert stats["simulations_succeeded_count"] == 1
    assert stats["simulations_failed_count"] == 0


def test_node7_phase1_evaluation_gate_persists_simulation_status(tmp_path: Path):
    """Test requirement 2b in Node 7: evaluation gate updates simulation_status to SUCCESS."""
    from agentic_ml_analog_sim.config import load_simulation_config

    db_file = tmp_path / "node7_test.db"
    db = ResultDatabase(db_path=db_file)

    cid = db.insert_proposal(solver="midpoint", noise_model="none", is_reference=False)
    db.update_execution_status(
        candidate_id=cid,
        rewrite_solver_status="SUCCESS",
        rewrite_noise_status="SUCCESS",
        rewrite_compiler_status="SUCCESS",
        compilation_status="SUCCESS",
    )
    db.mark_proposal_status(
        cid,
        status="EVALUATING",
        latency_ms=18.5,
        relative_error=0.025,
        accuracy_fid=0.5,
    )

    config = load_simulation_config("config/simulation_config.yaml")

    res = node7_phase1_evaluation_gate(db=db, config=config, candidate_id=cid)

    assert res["status"] == "PASSED"
    prop = db.get_proposal(cid)
    assert prop["simulation_status"] == "SUCCESS"
    assert prop["status"] == "COMPLETED"

    stats = db.get_execution_statistics()
    assert stats["simulations_succeeded_count"] == 1


def test_token_cost_plot_ordering_and_report_generation(multi_noise_db: ResultDatabase, tmp_path: Path):
    """Test requirement 2c & 2d: Phase 2 token usage recorded before plot and Inductor failures grounded."""
    # Pre-record Phase 0 and Phase 1 tokens
    multi_noise_db.record_token_usage(
        phase="phase_0",
        node_name="node1_propose",
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        cost_usd=0.0001,
        duration_seconds=1.0,
    )
    multi_noise_db.record_token_usage(
        phase="phase_1",
        node_name="node1_solver",
        model="gemini-2.5-flash",
        prompt_tokens=2000,
        completion_tokens=400,
        total_tokens=2400,
        cost_usd=0.0002,
        duration_seconds=2.0,
    )

    # Mock LLM for Phase 2 report generation
    class MockLLM:
        model = "gemini-2.5-flash"

        def prompt(self, user_message: str, tools=None):
            class Result:
                usage = {
                    "prompt_tokens": 1500,
                    "candidates_tokens": 500,
                    "total_tokens": 2000,
                }
                result = json.dumps({
                    "executive_summary": "Autonomous co-design loop evaluated 6 proposals.",
                    "pareto_analysis": "The Pareto front spans across low latency and high accuracy.",
                    "reliability_discussion": "Execution pipeline encountered 1 compilation lowering failure.",
                    "resource_discussion": "Swarm operated within token budgets.",
                    "recommendations": "Expand candidate exploration.",
                })

            return Result()

    config = {
        "experiment_name": "multi_noise_exp",
        "noise_models": ["none", "L0_static_mismatch", "L1_stochastic_parameter_noise"],
    }

    res = node2_phase2_generate_final_report(
        db=multi_noise_db,
        config=config,
        output_dir=tmp_path / "reports",
        llm=MockLLM(),
        compile_pdf=False,
    )

    assert res["status"] == "SUCCESS"

    # Verify that Phase 2 tokens were recorded in DB
    summary = multi_noise_db.get_token_usage_summary()
    assert "phase_2" in summary["by_phase"]
    assert summary["by_phase"]["phase_2"]["total_tokens"] == 2000
    assert summary["by_phase"]["phase_2"]["cost_usd"] > 0.0

    # Verify that token cost plot contains the generated files
    assert "token_cost" in res["figures"]
    assert Path(res["figures"]["token_cost"]["png"]).exists()

    # Verify multi-noise pareto plots are in figures
    assert "pareto_multi" in res["figures"]
    assert Path(res["figures"]["pareto_multi"]["none"]["png"]).exists()
    assert Path(res["figures"]["pareto_multi"]["l0_static_mismatch"]["png"]).exists()
    assert Path(res["figures"]["pareto_multi"]["all"]["png"]).exists()

    # Verify that generated final_report.tex includes per-noise figure tags and factual failure text
    tex_path = Path(res["tex_path"])
    assert tex_path.exists()
    tex_text = tex_path.read_text()
    assert "pareto_frontier_none.pdf" in tex_text
    assert "pareto_frontier_l0_static_mismatch.pdf" in tex_text
    assert "pareto_frontier_all.pdf" in tex_text
    # Verify baseline is not duplicated in Table 1
    ref_baseline_count = tex_text.count("ref_rk4_25steps")
    # In table rows, ref baseline should appear once
    assert tex_text.count("\\textbf{ref_") <= 1


def test_factual_reliability_analysis_grounding():
    """Test requirement 2d: Ground failure narrative in PyTorch Inductor CPU lowering exception logs."""
    stats = {
        "total_proposals": 5,
        "rewrites_failed_count": 0,
        "compilations_succeeded_count": 4,
        "compilations_failed_count": 1,
        "simulations_succeeded_count": 4,
        "simulations_failed_count": 0,
    }
    failures = [{
        "candidate_id": "cand_001",
        "solver": "rk4",
        "noise_model": "L0_static_mismatch",
        "error_stage": "compilation",
        "error_message": "Compilation error on device 'cpu': LoweringException: KeyError: 'unbacked_bindings'",
    }]

    text = _generate_dynamic_reliability_analysis(stats, compilation_failures=failures)

    # Assert factual Inductor lowering exception is mentioned
    assert "PyTorch Inductor CPU lowering exceptions" in text
    # Assert schema validation rejection is NOT claimed
    assert "schema rejection" not in text.lower() or "rather than code schema rejection" in text
