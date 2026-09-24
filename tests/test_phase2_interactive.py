"""Unit tests for Phase 2 Interactive Conversation Session."""

import io
from pathlib import Path
import pytest

from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.reporting.interactive_session import InteractiveReportSession


@pytest.fixture
def interactive_test_db(tmp_path: Path) -> ResultDatabase:
    db = ResultDatabase(db_path=tmp_path / "interactive_test.db")
    cid = db.insert_proposal(solver="euler", noise_model="none", is_reference=False)
    db.mark_proposal_status(cid, "COMPLETED", latency_ms=12.0, relative_error=0.02)
    return db


def test_interactive_session_ask(interactive_test_db: ResultDatabase, tmp_path: Path):
    report_file = tmp_path / "sample_report.tex"
    report_file.write_text("\\section{Executive Summary}\nSpeedup achieved is 8.5x.", encoding="utf-8")

    mock_llm = MockAntigravityLLM(
        responses=[
            "The Euler solver achieved an 8.5x speedup with only 2% error.",
            "L0 static mismatch had minimal impact on convergence.",
        ]
    )

    session = InteractiveReportSession(
        db=interactive_test_db,
        report_tex_path=report_file,
        llm=mock_llm,
    )

    # First turn
    resp1 = session.ask("What speedup did Euler achieve?")
    assert "8.5x" in resp1
    assert len(session.turns) == 1

    # Second turn
    resp2 = session.ask("What was the impact of L0 noise?")
    assert "minimal impact" in resp2
    assert len(session.turns) == 2

    # Check that token usage was recorded in DB
    summary = interactive_test_db.get_token_usage_summary(phase="phase_2")
    assert summary["overall"]["total_calls"] == 2
    assert summary["overall"]["total_tokens"] > 0


def test_interactive_session_repl(interactive_test_db: ResultDatabase, tmp_path: Path):
    mock_llm = MockAntigravityLLM(responses=["The optimal solver is Euler."])
    session = InteractiveReportSession(db=interactive_test_db, llm=mock_llm)

    simulated_input = io.StringIO("Which solver is best?\nquit\n")
    simulated_output = io.StringIO()

    session.run_repl(input_stream=simulated_input, output_stream=simulated_output)

    output_str = simulated_output.getvalue()
    assert "Un-0 Co-Design Interactive Consultant Session" in output_str
    assert "The optimal solver is Euler." in output_str
    assert "Ending interactive session. Goodbye!" in output_str
