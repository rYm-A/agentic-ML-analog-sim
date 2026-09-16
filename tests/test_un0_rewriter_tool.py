"""Tests for Un0RewriterTool."""

from __future__ import annotations

import pytest

from agentic_ml_analog_sim.tools.un0_rewriter_tool import Un0RewriterTool
from un0.compiler.registry import COMPILER_REGISTRY, get_compiler_rewrite


def test_un0_rewriter_tool_inspection():
    tool = Un0RewriterTool(auto_start=False)

    modules = tool.un0_list_modules()
    assert len(modules) > 0

    # Search for Kuramoto in un0
    search_res = tool.un0_search_code("Kuramoto")
    assert "matches" in search_res or "model" in search_res

    # Read part of model.py
    content = tool.un0_read_file("un0/model.py", start_line=1, end_line=30)
    assert "un0/model.py" in content


def test_un0_rewriter_tool_register_compiler_rewrite():
    tool = Un0RewriterTool(auto_start=False)

    # 1. Test syntax error handling
    bad_code = "def invalid_syntax(:"
    res_bad = tool.un0_register_compiler_rewrite(
        rewrite_name="bad_syntax",
        candidate_id="cand_bad",
        code=bad_code,
    )
    assert res_bad["status"] == "ERROR"
    assert "SyntaxError" in res_bad["error"]

    # 2. Test valid registration
    valid_code = '''
from un0.compiler.functional_hop import functionalize_kuramoto_model

def my_test_hop_rewrite(model, solver_name="rk4", num_steps=5, dt=0.2, integration_time=1.0, **kwargs):
    return functionalize_kuramoto_model(
        model=model,
        solver_name=solver_name,
        num_steps=num_steps,
        dt=dt,
        integration_time=integration_time,
        use_hop=True,
    )
'''
    res_valid = tool.un0_register_compiler_rewrite(
        rewrite_name="my_test_hop_rewrite",
        candidate_id="cand_test_tool",
        code=valid_code,
        description="Test valid HOP rewrite via tool",
    )
    assert res_valid["status"] == "SUCCESS"
    assert res_valid["rewrite_name"] == "my_test_hop_rewrite"
    assert "my_test_hop_rewrite" in COMPILER_REGISTRY

    rewrite_fn = get_compiler_rewrite("my_test_hop_rewrite")
    assert callable(rewrite_fn)

    # 3. Test list_registered_rewrites
    compilers = tool.un0_list_registered_rewrites(axis="compiler")
    assert "my_test_hop_rewrite" in compilers
    assert "hop_while_loop" in compilers

    solvers = tool.un0_list_registered_rewrites(axis="solver")
    assert "rk4" in solvers

    noise_models = tool.un0_list_registered_rewrites(axis="noise")
    assert "L0_static_mismatch" in noise_models
