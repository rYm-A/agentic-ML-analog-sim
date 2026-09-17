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
    assert tool.last_registered_rewrite == "my_test_hop_rewrite"
    assert tool.get_registered_rewrite("cand_test_tool") == "my_test_hop_rewrite"
    assert tool.get_registered_rewrite() == "my_test_hop_rewrite"

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


def test_un0_rewriter_tool_register_solver_rewrite():
    tool = Un0RewriterTool(auto_start=False)

    # 1. Test syntax error handling
    bad_code = "def invalid_solver_syntax(:"
    res_bad = tool.un0_register_solver_rewrite(
        solver_name="bad_solver_syntax",
        candidate_id="cand_bad_solver",
        code=bad_code,
    )
    assert res_bad["status"] == "ERROR"
    assert "SyntaxError" in res_bad["error"]

    # 2. Test valid registration
    valid_code = '''from __future__ import annotations
import torch
from torch import Tensor
from typing import Callable
from un0.solver import register_solver

@register_solver("test_custom_heun")
def test_custom_heun(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    **kwargs,
) -> Tensor:
    trajectory = [state]
    curr = state
    for i in range(len(time_grid) - 1):
        dt = time_grid[i + 1] - time_grid[i]
        k1 = rhs(curr, time_grid[i], drive)
        k2 = rhs(curr + dt * k1, time_grid[i + 1], drive)
        curr = curr + 0.5 * dt * (k1 + k2)
        trajectory.append(curr)
    return torch.stack(trajectory, dim=0)
'''
    res_valid = tool.un0_register_solver_rewrite(
        solver_name="test_custom_heun",
        candidate_id="cand_test_heun",
        code=valid_code,
        description="Test 2nd-order Heun solver rewrite",
    )
    assert res_valid["status"] == "SUCCESS"
    assert res_valid["solver_name"] == "test_custom_heun"

    from un0.solver import SOLVER_REGISTRY, get_solver
    assert "test_custom_heun" in SOLVER_REGISTRY
    assert tool.get_registered_solver("cand_test_heun") == "test_custom_heun"

    solver_fn = get_solver("test_custom_heun")
    assert callable(solver_fn)

    # Check solver is present in un0_list_registered_rewrites
    solvers = tool.un0_list_registered_rewrites(axis="solver")
    assert "test_custom_heun" in solvers


def test_un0_rewriter_tool_register_noise_rewrite():
    tool = Un0RewriterTool(auto_start=False)

    # 1. Test syntax error handling
    bad_code = "class InvalidNoiseSyntax(:"
    res_bad = tool.un0_register_noise_rewrite(
        noise_name="bad_noise_syntax",
        candidate_id="cand_bad_noise",
        code=bad_code,
    )
    assert res_bad["status"] == "ERROR"
    assert "SyntaxError" in res_bad["error"]

    # 2. Test valid registration
    valid_code = '''from __future__ import annotations
import torch
from torch import Tensor, nn
from typing import Callable
from un0.noise import register_noise_model

@register_noise_model("test_custom_phase_jitter")
class TestCustomPhaseJitter(nn.Module):
    """Test custom phase jitter noise wrapper."""

    def __init__(self, dynamics: nn.Module | Callable, sigma: float = 0.02) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.sigma = float(sigma)

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        noise = torch.randn_like(vel) * self.sigma
        return vel + noise
'''
    res_valid = tool.un0_register_noise_rewrite(
        noise_name="test_custom_phase_jitter",
        candidate_id="cand_test_noise_jitter",
        code=valid_code,
        description="Test phase jitter noise wrapper rewrite",
    )
    assert res_valid["status"] == "SUCCESS"
    assert res_valid["noise_name"] == "test_custom_phase_jitter"

    from un0.noise import NOISE_REGISTRY, get_noise_model
    from torch import nn
    assert "test_custom_phase_jitter" in NOISE_REGISTRY
    assert tool.get_registered_noise("cand_test_noise_jitter") == "test_custom_phase_jitter"

    noise_cls = get_noise_model("test_custom_phase_jitter")
    assert issubclass(noise_cls, nn.Module)

    # Check noise is present in un0_list_registered_rewrites
    noises = tool.un0_list_registered_rewrites(axis="noise")
    assert "test_custom_phase_jitter" in noises


