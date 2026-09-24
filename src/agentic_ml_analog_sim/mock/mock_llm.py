"""Deterministic Mock Antigravity LLM for Phase 0 testing without external CLI/cloud calls."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union
from uuid import uuid4

try:
    from chia.models.antigravity import AntigravityLLM, AntigravityQueryResult
except ImportError:  # pragma: no cover
    # Standalone fallback if chia is not available
    AntigravityLLM = object  # type: ignore

    @dataclass
    class AntigravityQueryResult:  # type: ignore
        result: str
        returncode: int = 0
        stderr: str = ""
        stream_result: str = ""
        success: bool = True
        conversation_id: str | None = None
        usage: dict | None = None
        events: list = field(default_factory=list)
        diagnostics: str = ""


logger = logging.getLogger("agentic_ml_analog_sim.mock_llm")


DEFAULT_VALID_PROPOSAL: dict[str, Any] = {
    "candidate_id": "cand_euler_l0_01",
    "solver": "euler",
    "num_steps": 5,
    "solver_params": {"num_steps": 5, "dt": 0.2},
    "noise_model": "L0_static_mismatch",
    "noise_params": {"sigma": 0.01},
    "sparsity_config": {"sparsity_ratio": 0.4},
    "selection_reason": "[pregenerated template text] Explore Euler with low static mismatch and 40% crossbar sparsity",
    "phase_1_prompt": (
        "[pregenerated template text] # Phase 1 Build & Simulation Instructions\n"
        "Configure Un-0 model to use 1st-order explicit Euler solver with num_steps=5 (dt=0.2). "
        "Apply L0 static mismatch noise (sigma=0.01) to coupling matrix K. "
        "Apply 40% threshold-based crossbar sparsity pruning (|K_ij| < tau). "
        "Verify numerical accuracy against fp32 reference baseline."
    ),
}

VALID_CANDIDATE_POOL: list[dict[str, Any]] = [
    DEFAULT_VALID_PROPOSAL,
    {
        "candidate_id": "cand_euler_l0_02",
        "solver": "euler",
        "num_steps": 10,
        "solver_params": {"num_steps": 10, "dt": 0.1},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"sparsity_ratio": 0.2},
        "selection_reason": "[pregenerated template text] Explore Euler with 10 steps and 20% crossbar sparsity",
        "phase_1_prompt": "[pregenerated template text] Configure Euler solver with num_steps=10 and L0 noise.",
    },
    {
        "candidate_id": "cand_euler_l0_03",
        "solver": "euler",
        "num_steps": 2,
        "solver_params": {"num_steps": 2, "dt": 0.5},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"sparsity_ratio": 0.6},
        "selection_reason": "[pregenerated template text] Explore 2-step Euler with 60% crossbar sparsity",
        "phase_1_prompt": "[pregenerated template text] Configure Euler solver with num_steps=2 and L0 noise.",
    },
    {
        "candidate_id": "cand_rk4_l0_01",
        "solver": "rk4",
        "num_steps": 5,
        "solver_params": {"num_steps": 5, "dt": 0.2},
        "noise_model": "L0_static_mismatch",
        "noise_params": {"sigma": 0.01},
        "sparsity_config": {"sparsity_ratio": 0.4},
        "selection_reason": "[pregenerated template text] Explore RK4 with 5 steps and 40% crossbar sparsity",
        "phase_1_prompt": "[pregenerated template text] Configure RK4 solver with num_steps=5 and L0 noise.",
    },
]

DEFAULT_DUPLICATE_PROPOSAL: dict[str, Any] = {
    "candidate_id": "cand_dup_rk4_ref",
    "solver": "rk4",
    "num_steps": 25,
    "solver_params": {"num_steps": 25, "dt": 0.04},
    "noise_model": "none",
    "noise_params": {},
    "sparsity_config": {"sparsity_ratio": 0.0},
    "selection_reason": "[pregenerated template text] Simulate duplicate proposal of baseline reference design",
    "phase_1_prompt": "[pregenerated template text] Evaluate baseline RK4 without noise or sparsity.",
}

DEFAULT_INVALID_SOLVER_PROPOSAL: dict[str, Any] = {
    "candidate_id": "cand_inv_solver_001",
    "solver": "unsupported_leapfrog",
    "num_steps": 10,
    "solver_params": {"num_steps": 10},
    "noise_model": "L0_static_mismatch",
    "noise_params": {"sigma": 0.01},
    "sparsity_config": {"sparsity_ratio": 0.4},
    "selection_reason": "[pregenerated template text] Simulate invalid solver not in config",
    "phase_1_prompt": "[pregenerated template text] Test invalid solver rejection.",
}

DEFAULT_INVALID_NOISE_PROPOSAL: dict[str, Any] = {
    "candidate_id": "cand_inv_noise_001",
    "solver": "euler",
    "num_steps": 5,
    "solver_params": {"num_steps": 5},
    "noise_model": "quantum_vacuum_fluctuation",
    "noise_params": {"intensity": 99.0},
    "sparsity_config": {"sparsity_ratio": 0.4},
    "selection_reason": "[pregenerated template text] Simulate invalid noise model not in config",
    "phase_1_prompt": "[pregenerated template text] Test invalid noise rejection.",
}

DEFAULT_INVALID_SPARSITY_PROPOSAL: dict[str, Any] = {
    "candidate_id": "cand_inv_sparsity_001",
    "solver": "euler",
    "num_steps": 5,
    "solver_params": {"num_steps": 5},
    "noise_model": "L0_static_mismatch",
    "noise_params": {"sigma": 0.01},
    "sparsity_config": {"sparsity_ratio": 0.99},
    "selection_reason": "[pregenerated template text] Simulate invalid crossbar sparsity ratio 0.99",
    "phase_1_prompt": "[pregenerated template text] Test invalid sparsity rejection.",
}

DEFAULT_INVALID_TOLERANCE_PROPOSAL: dict[str, Any] = {
    "candidate_id": "cand_inv_tol_001",
    "solver": "euler",
    "num_steps": 1,
    "solver_params": {"num_steps": 1, "dt": 1.0, "rtol": 1e-12, "atol": 1e-12},
    "noise_model": "L0_static_mismatch",
    "noise_params": {"sigma": 0.01},
    "sparsity_config": {"sparsity_ratio": 0.4},
    "selection_reason": "Tolerance check failure: [pregenerated template text] Simulate invalid tight tolerances",
    "phase_1_prompt": "[pregenerated template text] Test invalid tolerance rejection.",
}


class MockAntigravityLLM(AntigravityLLM):
    """Deterministic mock for AntigravityLLM.

    Subclasses Chia's AntigravityLLM and overrides `_run_antigravity` so that calls
    to `prompt()` return deterministic responses without executing the `agy` binary,
    burning Google Cloud credits, or requiring network access.
    """

    def __init__(
        self,
        mode: str = "valid",
        responses: list[Union[dict[str, Any], str, AntigravityQueryResult]] | None = None,
        custom_proposal: dict[str, Any] | None = None,
        duplicate_proposal: dict[str, Any] | None = None,
        invalid_proposal: dict[str, Any] | None = None,
        model: str = "mock-antigravity",
        system_message: str = "",
        effort: str = "medium",
        **kwargs: Any,
    ):
        """Initialize MockAntigravityLLM with configurable behavior.

        Args:
            mode: Operating mode:
                - 'valid': Returns default valid proposal (euler, 5 steps, L0 noise, 40% sparsity).
                - 'duplicate': Returns duplicate proposal.
                - 'invalid_solver': Returns proposal with unsupported solver.
                - 'invalid_noise': Returns proposal with unsupported noise model.
                - 'invalid_sparsity': Returns proposal with invalid sparsity config.
                - 'invalid_tolerance': Returns proposal with unachievable fp32 tolerance.
                - 'custom': Returns `custom_proposal`.
            responses: Optional queue of sequential responses. When provided, each call
                to `prompt()` pops the next response.
            custom_proposal: Proposal dict to return when mode='custom'.
            duplicate_proposal: Specific duplicate proposal to return.
            invalid_proposal: Specific invalid proposal to return.
            model: Model identifier.
            system_message: System instructions.
            effort: Reasoning effort level ('low', 'medium', 'high', 'max').
            **kwargs: Extra arguments passed to AntigravityLLM.__init__.
        """
        # Parse effort from extra_cli_args if present and not explicitly overridden
        extra_cli_args = kwargs.get("extra_cli_args")
        if extra_cli_args and "--effort" in extra_cli_args:
            idx = extra_cli_args.index("--effort")
            if idx + 1 < len(extra_cli_args):
                effort = extra_cli_args[idx + 1]
        self.effort = effort

        # Ensure model is not None to avoid AntigravityLLM warning
        kwargs["model"] = model
        kwargs["system_message"] = system_message
        try:
            super().__init__(**kwargs)
        except Exception:
            # Fallback if base class __init__ requires specific files/environment
            self.model = model
            self.system_message = system_message

        # Preserve exact configured model for mocks
        if model is not None:
            self.model = model

        self.mode = mode
        self._responses = list(responses) if responses is not None else []
        self.custom_proposal = custom_proposal
        self.duplicate_proposal = duplicate_proposal or DEFAULT_DUPLICATE_PROPOSAL
        self.invalid_proposal = invalid_proposal
        self.call_count: int = 0
        self.history: list[dict[str, Any]] = []

    def set_mode(self, mode: str) -> None:
        """Switch behavior mode dynamically."""
        self.mode = mode

    def add_response(self, response: Union[dict[str, Any], str, AntigravityQueryResult]) -> None:
        """Queue another response for subsequent calls."""
        self._responses.append(response)

    def reset(self) -> None:
        """Reset call counters and history."""
        self.call_count = 0
        self.history.clear()

    @property
    def last_prompt(self) -> str | None:
        """Return the most recently received user prompt."""
        if not self.history:
            return None
        return self.history[-1]["user_message"]

    def _get_next_proposal_dict(self, tool_list: Optional[List[Any]] = None, target_noise: Optional[str] = None) -> dict[str, Any]:
        """Resolve proposal dict based on queue or active mode."""
        if self._responses:
            next_item = self._responses.pop(0)
            if isinstance(next_item, dict):
                return next_item
            elif isinstance(next_item, str):
                try:
                    return json.loads(next_item)
                except Exception:
                    return {"raw_text": next_item}
            elif isinstance(next_item, AntigravityQueryResult):
                try:
                    return json.loads(next_item.result)
                except Exception:
                    return {"raw_text": next_item.result}

        if self.mode == "valid":
            # If tool_list contains a database tool or database instance,
            # query existing proposals to avoid duplicates like a real LLM agent would
            db = None
            if tool_list:
                for t in tool_list:
                    if hasattr(t, "db"):
                        db = t.db
                        break
                    elif hasattr(t, "is_duplicate_proposal"):
                        db = t
                        break

            if db is not None and hasattr(db, "is_duplicate_proposal"):
                for candidate in VALID_CANDIDATE_POOL:
                    cand_copy = dict(candidate)
                    if target_noise:
                        cand_copy["noise_model"] = target_noise
                    is_dup = db.is_duplicate_proposal(
                        solver=cand_copy.get("solver", ""),
                        noise_model=cand_copy.get("noise_model", ""),
                        solver_params=cand_copy.get("solver_params", {}),
                        noise_params=cand_copy.get("noise_params", {}),
                        sparsity_config=cand_copy.get("sparsity_config", {}),
                    )
                    if not is_dup:
                        cand_copy["candidate_id"] = f"{cand_copy['candidate_id']}_{uuid4().hex[:6]}"
                        return cand_copy

            pool_idx = max(0, self.call_count - 1) % len(VALID_CANDIDATE_POOL)
            proposal = dict(VALID_CANDIDATE_POOL[pool_idx])
            if target_noise:
                proposal["noise_model"] = target_noise
            proposal["candidate_id"] = f"{proposal['candidate_id']}_{uuid4().hex[:6]}"
            return proposal
        elif self.mode == "duplicate":
            return dict(self.duplicate_proposal)
        elif self.mode == "invalid_solver":
            return dict(self.invalid_proposal or DEFAULT_INVALID_SOLVER_PROPOSAL)
        elif self.mode == "invalid_noise":
            return dict(self.invalid_proposal or DEFAULT_INVALID_NOISE_PROPOSAL)
        elif self.mode == "invalid_sparsity":
            return dict(self.invalid_proposal or DEFAULT_INVALID_SPARSITY_PROPOSAL)
        elif self.mode == "invalid_tolerance":
            return dict(self.invalid_proposal or DEFAULT_INVALID_TOLERANCE_PROPOSAL)
        elif self.mode == "custom" and self.custom_proposal is not None:
            return dict(self.custom_proposal)
        else:
            return dict(DEFAULT_VALID_PROPOSAL)

    def _run_antigravity(self, user_message: str, tool_list: list[Any]) -> AntigravityQueryResult:
        """Overrides AntigravityLLM CLI execution with deterministic response."""
        self.call_count += 1
        if not self._responses and ("technical synthesis" in user_message or "final co-design report" in user_message):
            proposal_dict = {
                "executive_summary": "[pregenerated template text] Empirical evaluation across all candidates demonstrated accelerated inference latencies with bounded relative error.",
                "pareto_analysis": "[pregenerated template text] Analysis of the Pareto front points reveals a distinct trade-off structure across numerical latency and error coordinates. Candidate points delineate an empirical boundary where accelerated step formulations achieve throughput improvements while maintaining bounded phase-locking deviations.",
                "reliability_discussion": "[pregenerated template text] All candidate proposals progressed through automated solver and noise rewrites, domain compilation, and numerical simulation with high pipeline reliability.",
                "resource_discussion": "[pregenerated template text] Swarm token consumption and financial expenditure remained well within allocated budgets across all exploration and verification phases.",
                "recommendations": "[pregenerated template text]\n1. Prioritize low-step schemes along the frontier for edge silicon deployment.\n2. Apply crossbar degree-bounding to reduce parasitic capacitance.\n3. Expand candidate exploration across orthogonal dimensions in subsequent sweeps.",
            }
            result_json = json.dumps(proposal_dict, indent=2)
        elif not self._responses and any(k in user_message.lower() for k in ("compiler rewriter", "torch._higher_order_ops", "functionalize the kuramoto")):
            if self.mode == "invalid_compiler":
                result_text = "I failed to synthesize a valid compiler rewrite for this model."
                proposal_dict = {"status": "FAILED", "error": result_text}
                result_json = result_text
            else:
                import re
                m = re.search(r"candidate(?:[ _]*(?:id|ID))?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                if not m:
                    m = re.search(r"candidate[_ ]*['\"]?(?!into\b|specifications\b)([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                cand = m.group(1) if m else f"cand_mock_{self.call_count}"
                rewrite_name = f"hop_while_loop_{cand}"
                custom_code = f'''from un0.compiler.functional_hop import functionalize_kuramoto_model

def {rewrite_name}(model, solver_name="rk4", num_steps=25, dt=0.04, integration_time=1.0, **kwargs):
    return functionalize_kuramoto_model(
        model=model,
        solver_name=solver_name,
        num_steps=num_steps,
        dt=dt,
        integration_time=integration_time,
        use_hop=False,
    )
'''
                registered = False
                for tool in tool_list:
                    if hasattr(tool, "un0_register_compiler_rewrite"):
                        tool.un0_register_compiler_rewrite(
                            rewrite_name=rewrite_name,
                            candidate_id=cand,
                            code=custom_code,
                            description="Mock agent synthesized HOP while_loop rewrite",
                        )
                        registered = True
                        break

                if not registered:
                    result_text = f"I failed to register a compiler rewrite for candidate '{cand}' because the un0_rewriter tool was not available or registration failed."
                    proposal_dict = {"status": "FAILED", "error": result_text}
                    result_json = result_text
                else:
                    # Return freeform natural-language agent explanation, NOT JSON!
                    result_text = (
                        f"I analyzed the Kuramoto dynamics model for candidate '{cand}' and synthesized "
                        f"a PyTorch Higher-Order Operator while_loop functional rewrite. "
                        f"I registered the rewrite via `un0_register_compiler_rewrite` under name '{rewrite_name}'. "
                        f"This avoids Python-side while-loops and dynamic graph breaks, supporting batch dimension B >= 1."
                    )
                    proposal_dict = {"raw_explanation": result_text}
                    result_json = result_text
        elif not self._responses and any(k in user_message.lower() for k in ("solver rewriter", "implement an ode solver", "register_solver", "kuramoto physical analog dynamics")):
            if self.mode in ("invalid_solver_rewrite", "invalid_solver"):
                result_text = "I failed to synthesize a valid ODE solver for this configuration."
                proposal_dict = {"status": "FAILED", "error": result_text}
                result_json = result_text
            else:
                import re
                m_cand = re.search(r"candidate(?:[ _]*(?:id|ID))?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                if not m_cand:
                    m_cand = re.search(r"candidate[_ ]*['\"]?(?!into\b|specifications\b)([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                cand = m_cand.group(1) if m_cand else f"cand_mock_{self.call_count}"

                m_solver = re.search(r"named\s+['\"]([a-zA-Z0-9_\-]+)['\"]", user_message, re.IGNORECASE)
                if not m_solver:
                    m_solver = re.search(r"solver\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                solver_name = m_solver.group(1) if m_solver else f"custom_solver_{cand}"

                solver_code = f'''from __future__ import annotations
import torch
from torch import Tensor
from typing import Callable
from un0.solver import register_solver

@register_solver("{solver_name}")
def {solver_name}_step(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    **kwargs,
) -> Tensor:
    """Agent synthesized 2nd-order explicit Runge-Kutta / Heun ODE solver."""
    trajectory = [state]
    current = state
    num_steps = len(time_grid) - 1
    for i in range(num_steps):
        t = time_grid[i]
        dt = time_grid[i + 1] - t
        k1 = rhs(current, t, drive)
        k2 = rhs(current + dt * k1, time_grid[i + 1], drive)
        current = current + 0.5 * dt * (k1 + k2)
        trajectory.append(current)
    return torch.stack(trajectory, dim=0)
'''
                registered = False
                for tool in tool_list:
                    if hasattr(tool, "un0_register_solver_rewrite"):
                        tool.un0_register_solver_rewrite(
                            solver_name=solver_name,
                            candidate_id=cand,
                            code=solver_code,
                            description=f"Agent synthesized {solver_name} ODE solver",
                        )
                        registered = True
                        break

                result_text = (
                    f"I have consulted `solver_papers` and implemented a continuous ODE solver named '{solver_name}' "
                    f"for candidate '{cand}'. The solver adheres to the Un-0 signature `(rhs, state, time_grid, drive, **kwargs) -> Tensor` "
                    f"and returns trajectory shape `(len(time_grid), batch_size, state_dim)`. "
                    f"I registered the solver via `un0_register_solver_rewrite` under name '{solver_name}'."
                )
                proposal_dict = {"status": "SUCCESS", "solver_name": solver_name, "raw_explanation": result_text}
                result_json = result_text
        elif not self._responses and any(k in user_message.lower() for k in ("noise rewriter", "hardware noise wrapper", "register_noise", "kuramoto physical analog simulation")):
            if self.mode in ("invalid_noise_rewrite", "invalid_noise"):
                result_text = "I failed to synthesize a valid hardware noise wrapper for this model."
                proposal_dict = {"status": "FAILED", "error": result_text}
                result_json = result_text
            else:
                import re
                m_cand = re.search(r"candidate(?:[ _]*(?:id|ID))?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                if not m_cand:
                    m_cand = re.search(r"candidate[_ ]*['\"]?(?!into\b|specifications\b)([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                cand = m_cand.group(1) if m_cand else f"cand_mock_{self.call_count}"

                m_noise = re.search(r"named\s+['\"]([a-zA-Z0-9_\-]+)['\"]", user_message, re.IGNORECASE)
                if not m_noise:
                    m_noise = re.search(r"noise\s*model\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
                noise_name = m_noise.group(1) if m_noise else f"custom_noise_{cand}"

                noise_code = f'''from __future__ import annotations
import torch
from torch import Tensor, nn
from typing import Callable
from un0.noise import register_noise_model

@register_noise_model("{noise_name}")
class {noise_name.replace("_", "").capitalize()}Noise(nn.Module):
    """Agent synthesized physical hardware noise wrapper."""

    def __init__(self, dynamics: nn.Module | Callable, sigma: float = 0.01) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.sigma = float(sigma)

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        noise = torch.randn_like(vel) * self.sigma
        return vel + noise
'''
                registered = False
                for tool in tool_list:
                    if hasattr(tool, "un0_register_noise_rewrite"):
                        tool.un0_register_noise_rewrite(
                            noise_name=noise_name,
                            candidate_id=cand,
                            code=noise_code,
                            description=f"Agent synthesized {noise_name} noise wrapper model",
                        )
                        registered = True
                        break

                result_text = (
                    f"I have consulted `noise_papers` and implemented a hardware noise wrapper model named '{noise_name}' "
                    f"for candidate '{cand}'. The wrapper inherits from `torch.nn.Module`, wraps the continuous dynamics, "
                    f"and implements `forward(state, t, drive) -> Tensor`. "
                    f"I registered the model via `un0_register_noise_rewrite` under name '{noise_name}'."
                )
                proposal_dict = {"status": "SUCCESS", "noise_name": noise_name, "raw_explanation": result_text}
                result_json = result_text
        elif not self._responses and any(k in user_message.lower() for k in ("diagnostic & root-cause analysis", "diagnose failure", "diagnostic proposal", "diagnostic agent")):
            import re
            m_cand = re.search(r"candidate[_ ]*(?:id)?[:=\s]*['\"]?([a-zA-Z0-9_\-]+)['\"]?", user_message, re.IGNORECASE)
            cand = m_cand.group(1) if m_cand else f"cand_mock_{self.call_count}"

            # Check if prompt mentions multiple rewrite retries or debug request condition
            m_c2 = re.search(r"counter\s*2\s*\(rewrite\)\s*:\s*(\d+)", user_message, re.IGNORECASE)
            c2_val = int(m_c2.group(1)) if m_c2 else 0

            if c2_val >= 2 or "multiple rewrites" in user_message.lower() or "debug mode" in user_message.lower() or "counter_2=2" in user_message.lower():
                proposal_dict = {
                    "action": "debug_request",
                    "nodes_to_rerun": [3],
                    "diagnosis": f"Multiple rewrite attempts failed for candidate {cand}. Requesting Debug Mode to isolate hardware noise versus solver dynamics.",
                    "reasoning": f"Candidate {cand} has accumulated repeated failures. Entering debug mode with deactivated noise isolates solver dynamics from analog parameter noise.",
                    "debug_request": {
                        "debugee_id": cand,
                        "deactivate": "deactivate_noise",
                        "deactivated_rewrites": ["noise"],
                    },
                }
            elif any(k in user_message.lower() for k in ("compil", "inductor", "hop", "dynamic shape")):
                proposal_dict = {
                    "action": "rewrite_correction",
                    "nodes_to_rerun": [3],
                    "diagnosis": f"Compilation failure detected for candidate {cand} during Inductor HOP lowering.",
                    "reasoning": "Rerunning compiler rewriter (Node 3) with dynamic shapes disabled to generate a compilable graph.",
                    "debug_request": None,
                }
            else:
                proposal_dict = {
                    "action": "rewrite_correction",
                    "nodes_to_rerun": [1, 2, 3],
                    "diagnosis": f"Out-of-tolerance simulation error for candidate {cand}.",
                    "reasoning": "Rerunning full rewrite pipeline (Nodes 1, 2, and 3) to co-optimize solver step size and noise wrapper scaling.",
                    "debug_request": None,
                }
            result_json = json.dumps(proposal_dict, indent=2)
        else:
            import re
            m_target_noise = re.search(r"noise_model to ['\"]([^'\"]+)['\"]", user_message, re.IGNORECASE)
            target_noise = m_target_noise.group(1) if (m_target_noise and self.mode not in ("invalid_noise", "invalid_proposal", "duplicate")) else None
            proposal_dict = self._get_next_proposal_dict(tool_list, target_noise=target_noise)
            if target_noise and self.mode not in ("invalid_noise", "invalid_proposal", "duplicate"):
                proposal_dict["noise_model"] = target_noise
            result_json = json.dumps(proposal_dict, indent=2)

        usage_data = {
            "prompt_tokens": 120,
            "candidates_tokens": 85,
            "completion_tokens": 85,
            "total_tokens": 205,
        }
        entry = {
            "call_index": self.call_count,
            "user_message": user_message,
            "tools": [getattr(t, "name", str(t)) for t in (tool_list or [])],
            "response": proposal_dict,
            "effort": self.effort,
            "usage": usage_data,
        }
        self.history.append(entry)

        return AntigravityQueryResult(
            result=result_json,
            returncode=0,
            stderr="",
            stream_result=result_json,
            success=True,
            conversation_id=f"mock-conv-{self.call_count}",
            usage=usage_data,
            diagnostics=f"effort={self.effort}",
        )

    def prompt(
        self, user_message: str, tools: Optional[List[Any]] = None
    ) -> AntigravityQueryResult:
        """Execute prompt deterministically in-process without Ray remote dispatch."""
        return self._run_antigravity(user_message, tools or [])
