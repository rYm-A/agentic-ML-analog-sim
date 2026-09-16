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
    "selection_reason": "[pregenerated template text] Simulate invalid tight tolerances",
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

    def _get_next_proposal_dict(self, tool_list: Optional[List[Any]] = None) -> dict[str, Any]:
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
                    is_dup = db.is_duplicate_proposal(
                        solver=cand_copy.get("solver", ""),
                        noise_model=cand_copy.get("noise_model", ""),
                        solver_params=cand_copy.get("solver_params", {}),
                        noise_params=cand_copy.get("noise_params", {}),
                        sparsity_config=cand_copy.get("sparsity_config", {}),
                    )
                    if not is_dup:
                        return cand_copy

            pool_idx = max(0, self.call_count - 1) % len(VALID_CANDIDATE_POOL)
            proposal = dict(VALID_CANDIDATE_POOL[pool_idx])
            proposal["candidate_id"] = f"{proposal['candidate_id']}_{self.call_count:03d}"
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
        else:
            proposal_dict = self._get_next_proposal_dict(tool_list)
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
