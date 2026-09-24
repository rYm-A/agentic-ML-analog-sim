"""Tests for CompilationInfoTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ml_analog_sim.tools.compilation_info_tool import CompilationInfoTool


def test_compilation_info_tool_guides():
    tool = CompilationInfoTool(auto_start=False)

    guide_while = tool.get_torch_compile_guide("while_loop")
    assert "while_loop" in guide_while
    assert "cond_fn" in guide_while

    guide_scan = tool.get_torch_compile_guide("associative_scan")
    assert "associative_scan" in guide_scan

    guide_all = tool.get_torch_compile_guide("all")
    assert "while_loop" in guide_all
    assert "associative_scan" in guide_all
    assert "torch_compile" in guide_all

    guide_unknown = tool.get_torch_compile_guide("unknown_topic")
    assert "Topic 'unknown_topic' not found" in guide_unknown


def test_compilation_info_tool_regenerative_cornn(monkeypatch):
    import os
    if "REGENERATIVE_CORNN_ROOT" not in os.environ:
        workspace_cornn = Path(__file__).resolve().parents[2] / "RegenariveCoRNN"
        if workspace_cornn.is_dir():
            monkeypatch.setenv("REGENERATIVE_CORNN_ROOT", str(workspace_cornn))

    tool = CompilationInfoTool(auto_start=False)

    files = tool.list_regenerative_cornn_files()
    assert len(files) > 0
    assert any("implicit_parallel.py" in f for f in files)

    # Read lines from implicit_parallel.py
    content = tool.read_regenerative_cornn_file("model/implicit_parallel.py", start_line=1, end_line=30)
    assert "implicit_parallel.py" in content
    assert "while_loop" in content or "torch" in content

    # Search for while_loop
    search_res = tool.search_regenerative_cornn("while_loop")
    assert "implicit_parallel.py" in search_res or "matches" in search_res

    # Test path traversal protection
    traversal_res = tool.read_regenerative_cornn_file("../../etc/passwd")
    assert "Error: Path" in traversal_res


def test_compilation_info_tool_missing_repo_bypasses_and_warns(caplog):
    tool = CompilationInfoTool(repo_root="/nonexistent/RegenariveCoRNN", auto_start=False)

    assert tool.is_available is False
    assert any("RegenerativeCoRNN repository not found" in record.message for record in caplog.records)

    # All RegenerativeCoRNN logic is bypassed gracefully
    assert tool.list_regenerative_cornn_files() == []
    read_res = tool.read_regenerative_cornn_file("model/implicit_parallel.py")
    assert "Inspection bypassed" in read_res
    search_res = tool.search_regenerative_cornn("while_loop")
    assert "Search bypassed" in search_res

    # Static documentation continues to function
    guide = tool.get_torch_compile_guide("while_loop")
    assert "while_loop" in guide
