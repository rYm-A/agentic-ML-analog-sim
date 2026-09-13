"""Tests for Un0Context and Un0ContextTool."""

import pytest
from pathlib import Path

from agentic_ml_analog_sim.tools.un0_context_tool import (
    DEFAULT_UN0_ROOT,
    Un0Context,
    Un0ContextTool,
    create_un0_mcp_server,
    resolve_un0_root,
)

REPO_ROOT = resolve_un0_root()


@pytest.fixture
def un0_context():
    return Un0Context(repo_root=REPO_ROOT)


@pytest.fixture
def un0_tool():
    return Un0ContextTool(name="un0_test", repo_root=REPO_ROOT, auto_start=False)


def test_safe_path_resolution(un0_context):
    resolved = un0_context._resolve_safe_path("un0/model.py")
    assert resolved == (REPO_ROOT / "un0/model.py").resolve()

    # Strips "Un-0/" prefix if included
    resolved_with_prefix = un0_context._resolve_safe_path("Un-0/un0/model.py")
    assert resolved_with_prefix == (REPO_ROOT / "un0/model.py").resolve()

    # Directory traversal prevention
    with pytest.raises(PermissionError):
        un0_context._resolve_safe_path("../../etc/passwd")

    with pytest.raises(PermissionError):
        un0_context._resolve_safe_path("/etc/passwd")


def test_un0_read_file_header_and_notice(un0_context):
    total = len((REPO_ROOT / "un0/model.py").read_text(encoding="utf-8").splitlines())
    result = un0_context.un0_read_file("un0/model.py", start_line=1, end_line=50)
    assert f"[File: un0/model.py | Total lines: {total} | Showing lines 1 to 50]" in result
    assert f"[NOTE: {total - 50} lines remaining. Call un0_read_file with start_line=51, end_line=200 to read further]" in result
    assert "class ConditionalKuramotoDynamics" in result


def test_un0_read_file_clamping(un0_context):
    total = len((REPO_ROOT / "un0/model.py").read_text(encoding="utf-8").splitlines())
    # Requesting 300 lines should be clamped to 150 lines max
    result = un0_context.un0_read_file("un0/model.py", start_line=1, end_line=300)
    assert f"[File: un0/model.py | Total lines: {total} | Showing lines 1 to 150]" in result
    assert f"[NOTE: {total - 150} lines remaining. Call un0_read_file with start_line=151, end_line=300 to read further]" in result


def test_un0_read_file_full_content_no_notice(un0_context):
    # un0/common.py has fewer than 150 lines
    result = un0_context.un0_read_file("un0/common.py", start_line=1, end_line=150)
    assert "lines remaining" not in result
    assert "Showing lines 1 to" in result


def test_un0_get_symbol_classes_and_functions(un0_context):
    # Class symbol
    cls_result = un0_context.un0_get_symbol("un0/model.py", "ConditionalKuramotoDynamics")
    assert "[Symbol: ConditionalKuramotoDynamics | File: un0/model.py | Lines 28 to 155]" in cls_result
    assert "class ConditionalKuramotoDynamics(nn.Module):" in cls_result

    # Function symbol
    fn_result = un0_context.un0_get_symbol("un0/model.py", "_kuramoto_velocity")
    assert "[Symbol: _kuramoto_velocity | File: un0/model.py | Lines 19 to 25]" in fn_result
    assert "def _kuramoto_velocity" in fn_result

    # Another class symbol
    readout_result = un0_context.un0_get_symbol("un0/model.py", "ReadoutTransform")
    assert "[Symbol: ReadoutTransform | File: un0/model.py | Lines 158 to 195]" in readout_result
    assert "class ReadoutTransform(nn.Module):" in readout_result


def test_un0_get_symbol_not_found(un0_context):
    result = un0_context.un0_get_symbol("un0/model.py", "NonExistentSymbol")
    assert "Symbol 'NonExistentSymbol' not found in 'un0/model.py'." in result
    assert "Available classes:" in result
    assert "ConditionalKuramotoDynamics" in result
    assert "Available functions:" in result
    assert "_kuramoto_velocity" in result


def test_un0_search_code(un0_context):
    result = un0_context.un0_search_code("kuramoto", directory="un0")
    assert "Search results for 'kuramoto' in 'un0'" in result
    assert "un0/model.py" in result

    # Non existent query
    no_result = un0_context.un0_search_code("xyz_nonexistent_token_12345", directory="un0")
    assert "No matches found" in no_result


def test_un0_list_modules(un0_context):
    modules = un0_context.un0_list_modules()
    assert isinstance(modules, list)
    assert len(modules) >= 10
    assert "un0/model.py" in modules
    assert "un0/train_cifar10.py" in modules
    assert "un0/common.py" in modules


def test_un0_get_model_summary(un0_context):
    summary = un0_context.un0_get_model_summary("cifar10/n1024")
    assert "1024" in summary
    assert "8" in summary
    assert "1032" in summary
    assert "rk4" in summary
    assert "25" in summary
    assert "1.0" in summary
    assert "K" in summary
    assert "omega" in summary
    assert "K_drive" in summary


def test_un0_context_tool_delegation(un0_tool):
    # Verify tool delegates properly to context
    sym = un0_tool.un0_get_symbol("un0/model.py", "ConditionalKuramotoDynamics")
    assert "class ConditionalKuramotoDynamics" in sym

    read = un0_tool.un0_read_file("un0/model.py", 1, 20)
    assert "[File: un0/model.py" in read

    summary = un0_tool.un0_get_model_summary("cifar10/n1024")
    assert "cifar10/n1024" in summary

    mods = un0_tool.un0_list_modules()
    assert "un0/model.py" in mods


def test_create_un0_mcp_server():
    server = create_un0_mcp_server(repo_root=REPO_ROOT)
    assert server.name == "un0_context"


def test_dynamic_root_resolution():
    """Test dynamic resolution order: explicit, submodule, sibling, and fallback."""
    # 1. Explicit argument
    explicit = resolve_un0_root("/tmp")
    assert explicit == Path("/tmp").resolve()

    # 2. Default resolution (no argument passed)
    resolved = resolve_un0_root()
    assert resolved.is_dir()
    assert (resolved / "un0").is_dir() or (resolved / "pyproject.toml").is_file()

    # 3. Un0Context default init without explicit repo_root
    ctx_default = Un0Context()
    assert ctx_default.repo_root.is_dir()
    mods = ctx_default.un0_list_modules()
    assert len(mods) >= 5

    # 4. Un0ContextTool default init without explicit repo_root
    tool_default = Un0ContextTool(name="un0_dyn_test", auto_start=False)
    assert tool_default.context.repo_root.is_dir()

    # 5. create_un0_mcp_server default init without explicit repo_root
    server_default = create_un0_mcp_server()
    assert server_default.name == "un0_context"
