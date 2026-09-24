"""Unit tests for Phase 2 end-to-end pipeline orchestrator and CLI."""

from pathlib import Path
import subprocess
import sys
import pytest

from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.phase_2 import Phase2Result, run_phase_2


@pytest.fixture
def pipeline_db(tmp_path: Path) -> ResultDatabase:
    db_file = tmp_path / "pipeline.db"
    db = ResultDatabase(db_path=db_file)

    # Reference baseline
    ref_id = db.insert_proposal(solver="rk4", noise_model="none", is_reference=True)
    db.mark_proposal_status(ref_id, "COMPLETED", latency_ms=100.0, relative_error=0.0)

    return db


def test_phase_2_loop_continuation(pipeline_db: ResultDatabase, tmp_path: Path):
    cfg = load_simulation_config("config/simulation_config.yaml")

    # Only 2 completed experiments (< 10)
    for i in range(2):
        cid = pipeline_db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
        pipeline_db.mark_proposal_status(cid, "COMPLETED", latency_ms=15.0 + i)

    res = run_phase_2(
        config_path=cfg,
        db_path=pipeline_db,
        output_dir=tmp_path / "reports",
        compile_pdf=False,
    )

    assert res.action == "CONTINUE_LOOP"
    assert res.completed_experiments == 2
    assert res.max_experiments == 10
    assert res.tex_path is None


def test_phase_2_force_report(pipeline_db: ResultDatabase, tmp_path: Path):
    cfg = load_simulation_config("config/simulation_config.yaml")
    mock_llm = MockAntigravityLLM(mode="valid")

    # Only 1 experiment, but forced
    cid = pipeline_db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    pipeline_db.mark_proposal_status(cid, "COMPLETED", latency_ms=12.0)

    res = run_phase_2(
        config_path=cfg,
        db_path=pipeline_db,
        output_dir=tmp_path / "forced_reports",
        llm=mock_llm,
        force_report=True,
        compile_pdf=False,
    )

    assert res.action == "REPORT_GENERATED"
    assert res.tex_path is not None
    assert Path(res.tex_path).exists()
    assert res.figures is not None
    assert "pareto" in res.figures


def test_phase_2_full_termination_and_pdf(pipeline_db: ResultDatabase, tmp_path: Path):
    cfg = load_simulation_config("config/simulation_config.yaml")
    mock_llm = MockAntigravityLLM(mode="valid")

    # Insert 10 completed experiments
    for i in range(10):
        cid = pipeline_db.insert_proposal(solver="euler", noise_model="none", is_reference=False, solver_params={"i": i})
        pipeline_db.mark_proposal_status(
            cid,
            "COMPLETED",
            latency_ms=10.0 + i * 2,
            relative_error=0.01 * (i + 1),
            pareto_optimal=(i < 3),
        )
        pipeline_db.update_execution_status(
            cid,
            rewrite_solver_status="SUCCESS",
            compilation_status="SUCCESS",
            simulation_status="SUCCESS",
        )

    res = run_phase_2(
        config_path=cfg,
        db_path=pipeline_db,
        output_dir=tmp_path / "full_reports",
        llm=mock_llm,
        compile_pdf=True,
    )

    assert res.action == "REPORT_GENERATED"
    assert res.completed_experiments == 10
    assert res.tex_path is not None
    assert Path(res.tex_path).exists()
    if res.pdf_compiled:
        assert res.pdf_path is not None
        assert Path(res.pdf_path).exists()


def test_phase_2_cli(pipeline_db: ResultDatabase, tmp_path: Path):
    # Insert 1 candidate
    cid = pipeline_db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    pipeline_db.mark_proposal_status(cid, "COMPLETED", latency_ms=10.0)

    out_dir = tmp_path / "cli_reports"
    cmd = [
        sys.executable,
        "-m",
        "agentic_ml_analog_sim.phase_2",
        "--config",
        "config/simulation_config.yaml",
        "--db",
        str(pipeline_db.db_path),
        "--output-dir",
        str(out_dir),
        "--mock",
        "--force-report",
        "--no-pdf",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0
    assert "REPORT_GENERATED" in proc.stdout
    cfg = load_simulation_config("config/simulation_config.yaml")
    assert (out_dir / cfg.experiment_name / "final_report.tex").exists()
