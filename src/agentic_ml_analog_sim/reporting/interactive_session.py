"""Interactive multi-turn conversation session for user inquiry on the final co-design report."""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger("agentic_ml_analog_sim.interactive_session")

INTERACTIVE_SYSTEM_PROMPT = """
You are the Technical Consultant and Reporting Specialist for the Un-0 Analog Machine Learning Accelerator.
You have just finalized the comprehensive co-design report, evaluating physical Kuramoto oscillatory neural networks,
numerical differential equation solvers (Euler, RK4, Parareal), hardware noise resilience (L0 mismatch to L3 drift),
and crossbar coupling sparsity.

The user is now asking follow-up questions regarding the final report, specific candidate architectures,
numerical trade-offs, reliability failures, or deployment recommendations.
Answer their questions accurately, referring to the simulation database and generated findings whenever helpful.
Be concise, rigorous, and technically precise.
"""


class InteractiveReportSession:
    """Manages multi-turn conversation with the user post-report generation."""

    def __init__(
        self,
        db: Optional[Union[ResultDatabase, str, Path]] = None,
        report_tex_path: Optional[Union[str, Path]] = None,
        llm: Any = None,
        tools: Optional[List[Any]] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """Initialize interactive session with database context and tools.

        Args:
            db: ResultDatabase instance or path.
            report_tex_path: Path to the generated final_report.tex.
            llm: Optional pre-configured LLM instance (AntigravityLLM or MockAntigravityLLM).
            tools: Optional ChiaTool instances to bind.
            model_name: LLM model name.
        """
        if isinstance(db, ResultDatabase):
            self.db = db
        elif isinstance(db, (str, Path)):
            self.db = ResultDatabase(db_path=db)
        elif db is None:
            self.db = ResultDatabase()
        else:
            self.db = db

        self.report_tex_path = Path(report_tex_path) if report_tex_path else None
        self.model_name = model_name
        self.report_text = ""
        if self.report_tex_path and self.report_tex_path.exists():
            try:
                self.report_text = self.report_tex_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Could not read report tex file: %s", e)

        # Bind tools
        self.tools = list(tools) if tools is not None else []
        tool_names = {getattr(t, "name", str(t)) for t in self.tools}
        if "ResultDBTool" not in tool_names and "result_db" not in tool_names:
            try:
                from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool
                self.tools.append(ResultDBTool(db=self.db))
            except Exception:
                pass
        if "PlotTool" not in tool_names and "plot_tool" not in tool_names:
            try:
                from agentic_ml_analog_sim.tools.plot_tool import PlotTool
                self.tools.append(PlotTool(db=self.db, deploy=False))
            except Exception:
                pass

        self.llm = self._resolve_llm(llm)
        self.conversation_id: Optional[str] = None
        self.turns: List[Dict[str, str]] = []

    def _resolve_llm(self, llm: Any = None) -> Any:
        """Resolve LLM with multi-turn resume session capability."""
        if llm is not None:
            return llm

        try:
            from chia.models.antigravity import AntigravityLLM
            return AntigravityLLM(
                model=self.model_name,
                system_message=INTERACTIVE_SYSTEM_PROMPT,
                resume_session=True,
            )
        except Exception:
            from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
            return MockAntigravityLLM(
                mode="valid",
                system_message=INTERACTIVE_SYSTEM_PROMPT,
            )

    def ask(self, user_query: str) -> str:
        """Submit a user query in the multi-turn session and return the assistant response.

        Args:
            user_query: The question or inquiry from the user.

        Returns:
            The agent's text response.
        """
        # Formulate query context on the first turn if report is loaded
        if not self.turns and self.report_text:
            context_prefix = (
                f"[SYSTEM CONTEXT: The final co-design report has been generated. Excerpt below:]\n"
                f"{self.report_text[:3000]}\n...\n[END CONTEXT]\n\n"
                f"User Question: {user_query}"
            )
            prompt_to_send = context_prefix
        else:
            prompt_to_send = user_query

        start_time = time.time()
        # Prompt LLM with resume_session enabled
        try:
            query_result = self.llm.prompt(
                user_message=prompt_to_send,
                tools=self.tools,
                resume_session=True,
            )
        except TypeError:
            # Fallback if mock LLM doesn't accept resume_session keyword
            query_result = self.llm.prompt(
                user_message=prompt_to_send,
                tools=self.tools,
            )

        duration = time.time() - start_time
        response_text = getattr(query_result, "result", str(query_result))
        self.conversation_id = getattr(query_result, "conversation_id", self.conversation_id)

        # Record LLM token usage
        usage = getattr(query_result, "usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("candidates_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        cost_usd = (prompt_tokens * 0.075 / 1_000_000) + (completion_tokens * 0.30 / 1_000_000)

        if hasattr(self.db, "record_token_usage") and callable(self.db.record_token_usage):
            try:
                self.db.record_token_usage(
                    phase="phase_2",
                    node_name="interactive_session",
                    model=getattr(self.llm, "model", self.model_name),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    duration_seconds=duration,
                )
            except Exception as exc:
                logger.warning("Failed to record interactive token usage: %s", exc)

        self.turns.append({"user": user_query, "assistant": response_text})
        return response_text

    def run_repl(
        self,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
    ) -> None:
        """Run an interactive CLI read-eval-print loop with the user.

        Args:
            input_stream: Input text stream (defaults to sys.stdin).
            output_stream: Output text stream (defaults to sys.stdout).
        """
        out = output_stream or sys.stdout
        inp = input_stream or sys.stdin

        out.write("\n" + "=" * 70 + "\n")
        out.write("Un-0 Co-Design Interactive Consultant Session\n")
        out.write("Type your question about the final report or simulation results.\n")
        out.write("Type 'exit' or 'quit' to end the session.\n")
        out.write("=" * 70 + "\n\n")
        out.flush()

        while True:
            out.write("User > ")
            out.flush()
            try:
                line = inp.readline()
                if not line:  # EOF
                    break
                user_msg = line.strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ("exit", "quit", "q"):
                    out.write("Ending interactive session. Goodbye!\n")
                    out.flush()
                    break

                response = self.ask(user_msg)
                out.write(f"\nConsultant > {response}\n\n")
                out.flush()
            except (KeyboardInterrupt, EOFError):
                out.write("\nSession interrupted. Exiting.\n")
                out.flush()
                break
