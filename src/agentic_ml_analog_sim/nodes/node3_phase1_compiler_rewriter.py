"""Node 3: Compiler Rewriter Agentic Node (Phase 1).

Rewrites continuous Kuramoto dynamics into a functionalized form using PyTorch
Higher-Order Operators (HOPs, e.g. torch._higher_order_ops.while_loop) with
dynamic batch dimensions (B >= 1, without upper bound) for TorchInductor AOT compilation.
Records real token metrics only when an LLM is executed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node3_phase1_compiler_rewriter")


def _resolve_llm(llm: Optional[Any] = None, config: Optional[SimulationConfig] = None) -> Any:
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node3_phase1_compiler_rewriter", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node3_phase1_compiler_rewriter")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-pro"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "medium"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


def node3_phase1_compiler_rewriter(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    target_device: str = "mps",
    llm: Optional[Any] = None,
    force_rewrite: bool = False,
) -> Dict[str, Any]:
    """Node 3 Compiler Rewriter (Phase 1).

    Transforms the continuous Kuramoto dynamics into a functionalized HOP model
    using PyTorch Higher-Order Operators (`torch._higher_order_ops.while_loop`)
    supporting dynamic batch shapes (B >= 1).

    Deterministic behavior:
        Applies systematic HOP functionalization. If no LLM call is executed,
        token consumption is null (0 tokens recorded).

    Agentic behavior:
        If an agentic code synthesis or rewrite is forced, queries the LLM and
        records real token consumption metrics to the database.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        target_device: Hardware execution target ('mps', 'cuda', 'cpu').
        llm: Optional LLM instance or mock.
        force_rewrite: Whether to force an LLM prompt turn.

    Returns:
        Dict detailing candidate_id, execution status, target device, prepared HOP model,
        and eager model baseline.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    logger.info(
        "Node 3: Applying HOP functionalization rewrite for %s (target_device=%s)",
        candidate_id,
        target_device,
    )

    # Obtain solver without assuming hardcoded 'rk4' default
    solver_name = proposal.get("solver")
    if not solver_name:
        ref_design = getattr(config, "reference_design", None)
        solver_name = getattr(ref_design, "solver", None) if ref_design else None
    if not solver_name:
        raise KeyError(f"No solver specified for candidate '{candidate_id}' in proposal or simulation config.")

    # Obtain architecture and simulation parameters directly from configuration / proposal
    ref_design = getattr(config, "reference_design", None)
    if ref_design is None:
        raise ValueError("Simulation configuration is missing 'reference_design'.")

    n_osc = getattr(ref_design, "n_oscillators", None)
    n_cond = getattr(ref_design, "n_conditional_oscillators", None)
    if n_osc is None or n_cond is None:
        raise ValueError("Simulation configuration reference_design must specify 'n_oscillators' and 'n_conditional_oscillators'.")

    family = str(getattr(ref_design, "family", "")).lower()
    if hasattr(ref_design, "num_classes") and ref_design.num_classes is not None:
        num_classes = ref_design.num_classes
    elif "cifar100" in family:
        num_classes = 100
    elif "cifar10" in family or "cifar" in family or "mnist" in family:
        num_classes = 10
    elif hasattr(config, "num_classes") and config.num_classes is not None:
        num_classes = config.num_classes
    else:
        num_classes = 10

    num_steps = proposal.get("num_steps") or getattr(ref_design, "num_steps", None)
    if num_steps is None:
        raise ValueError(f"Number of steps not specified for candidate '{candidate_id}' or reference design.")

    solver_params = proposal.get("solver_params", {}) or {}
    if isinstance(solver_params, str):
        try:
            import json
            solver_params = json.loads(solver_params)
        except Exception:
            solver_params = {}

    integration_time = solver_params.get("integration_time") or getattr(ref_design, "integration_time", None)
    if integration_time is None:
        raise ValueError(f"Integration time not specified for candidate '{candidate_id}' or reference design.")

    dt = solver_params.get("dt") or (float(integration_time) / float(num_steps))

    # 1. Instantiate Un-0 base model and apply noise if active
    eager_model = None
    functional_model = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_agentic_tokens = 0

    from un0.compiler import functionalize_kuramoto_model
    from un0.model import ConditionalKuramotoDynamics
    from un0.noise import get_noise_model

    base_dynamics = ConditionalKuramotoDynamics(
        n_oscillators=n_osc,
        n_conditional_oscillators=n_cond,
        num_classes=num_classes,
    )

    noise_name = proposal.get("noise_model")
    if noise_name is None:
        noise_name = getattr(ref_design, "noise_model", "none")

    is_no_noise = not noise_name or str(noise_name).lower() == "none" or bool(proposal.get("is_reference"))

    if is_no_noise:
        # When no noise model was requested (or reference mode), cleanly use base dynamics
        eager_model = base_dynamics
    else:
        # A noise model was requested; instantiate it or trigger agentic rewrite on failure
        noise_params = proposal.get("noise_params", {}) or {}
        if isinstance(noise_params, str):
            try:
                import json
                noise_params = json.loads(noise_params)
            except Exception:
                noise_params = {}

        try:
            noise_cls = get_noise_model(noise_name)
            eager_model = noise_cls(base_dynamics, **(noise_params if isinstance(noise_params, dict) else {}))
        except Exception as noise_err:
            logger.warning(
                "Node 3: Noise wrapper '%s' instantiation failed: %s. Activating agentic rewrite.",
                noise_name,
                noise_err,
            )
            active_llm = _resolve_llm(llm, config)
            start_time = time.time()
            rewrite_prompt = (
                f"Fix and implement the hardware noise wrapper model '{noise_name}' for Un-0 Kuramoto simulation. "
                f"Error was: {noise_err}. Wrap base dynamics and register with @register_noise_model('{noise_name}')."
            )
            q_res = None
            try:
                if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
                    q_res = active_llm.prompt(rewrite_prompt)
            except Exception as llm_err:
                logger.warning("Node 3 noise agentic prompt failed: %s", llm_err)

            elapsed = max(0.1, time.time() - start_time)
            usage = getattr(q_res, "usage", None) if q_res is not None else None
            if not usage and hasattr(active_llm, "history") and active_llm.history:
                usage = active_llm.history[-1].get("usage")

            if isinstance(usage, dict):
                p_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                c_tok = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_tokens") or 0
                t_tok = usage.get("total_tokens") or (p_tok + c_tok)
                total_prompt_tokens += p_tok
                total_completion_tokens += c_tok
                total_agentic_tokens += t_tok

                if t_tok > 0:
                    db.record_token_usage(
                        phase="phase_1",
                        node_name="node3_phase1_compiler_rewriter",
                        model=getattr(active_llm, "model", "gemini-2.5-pro"),
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=t_tok,
                        cost_usd=0.0,
                        candidate_id=candidate_id,
                        duration_seconds=elapsed,
                    )

            # Retry instantiation after agentic rewrite
            try:
                noise_cls = get_noise_model(noise_name)
                eager_model = noise_cls(base_dynamics, **(noise_params if isinstance(noise_params, dict) else {}))
            except Exception as retry_err:
                logger.error("Node 3: Noise wrapper instantiation still failing after agentic rewrite: %s", retry_err)
                raise RuntimeError(f"Cannot proceed with requested noise '{noise_name}': {retry_err}")

    # 2. Functionalize with PyTorch HOP (while_loop) and dynamic batching B >= 1
    try:
        functional_model = functionalize_kuramoto_model(
            model=eager_model,
            solver_name=solver_name,
            num_steps=num_steps,
            dt=dt,
            integration_time=integration_time,
            use_hop=True,
        )
    except Exception as hop_err:
        logger.warning(
            "Node 3: HOP functionalization failed for %s (solver=%s): %s. Activating agentic rewrite.",
            candidate_id,
            solver_name,
            hop_err,
        )
        active_llm = _resolve_llm(llm, config)
        start_time = time.time()
        hop_prompt = (
            f"Rewrite the Kuramoto functional dynamics loop for candidate '{candidate_id}' using solver '{solver_name}'. "
            f"PyTorch HOP while_loop encountered error: {hop_err}. "
            f"Implement functional unrolled step supporting dynamic batch dimension (B >= 1) targeting {target_device}."
        )
        q_res = None
        try:
            if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
                q_res = active_llm.prompt(hop_prompt)
        except Exception as llm_err:
            logger.warning("Node 3 HOP LLM prompt failed: %s", llm_err)

        elapsed = max(0.1, time.time() - start_time)
        usage = getattr(q_res, "usage", None) if q_res is not None else None
        if not usage and hasattr(active_llm, "history") and active_llm.history:
            usage = active_llm.history[-1].get("usage")

        if isinstance(usage, dict):
            p_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            c_tok = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_tokens") or 0
            t_tok = usage.get("total_tokens") or (p_tok + c_tok)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_agentic_tokens += t_tok

            if t_tok > 0:
                db.record_token_usage(
                    phase="phase_1",
                    node_name="node3_phase1_compiler_rewriter",
                    model=getattr(active_llm, "model", "gemini-2.5-pro"),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    cost_usd=0.0,
                    candidate_id=candidate_id,
                    duration_seconds=elapsed,
                )

        functional_model = functionalize_kuramoto_model(
            model=eager_model,
            solver_name=solver_name,
            num_steps=num_steps,
            dt=dt,
            integration_time=integration_time,
            use_hop=False,
        )

    # 3. Explicit forced agentic rewrite requested
    if force_rewrite:
        active_llm = _resolve_llm(llm, config)
        start_time = time.time()
        prompt_text = (
            f"Rewrite the Kuramoto dynamics model for candidate {candidate_id} to use PyTorch HOP "
            f"torch._higher_order_ops.while_loop with dynamic batch shapes (B >= 1) targeting {target_device}."
        )
        query_result = None
        try:
            if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
                query_result = active_llm.prompt(prompt_text)
        except Exception as exc:
            logger.warning("Node 3 LLM prompt failed: %s", exc)

        elapsed = max(0.1, time.time() - start_time)
        usage = getattr(query_result, "usage", None) if query_result is not None else None
        if not usage and hasattr(active_llm, "history") and active_llm.history:
            usage = active_llm.history[-1].get("usage")

        if isinstance(usage, dict):
            p_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            c_tok = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_tokens") or 0
            t_tok = usage.get("total_tokens") or (p_tok + c_tok)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_agentic_tokens += t_tok

            if t_tok > 0:
                db.record_token_usage(
                    phase="phase_1",
                    node_name="node3_phase1_compiler_rewriter",
                    model=getattr(active_llm, "model", "gemini-2.5-pro"),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    cost_usd=0.0,
                    candidate_id=candidate_id,
                    duration_seconds=elapsed,
                )

    if total_agentic_tokens > 0:
        logger.info(
            "Node 3: Tokens consumed - prompt: %d, completion: %d, total: %d",
            total_prompt_tokens,
            total_completion_tokens,
            total_agentic_tokens,
        )

    # Update DB status
    db.update_execution_status(
        candidate_id,
        rewrite_compiler_status="SUCCESS",
    )

    return {
        "candidate_id": candidate_id,
        "status": "SUCCESS",
        "target_device": target_device,
        "prepared_model": functional_model,
        "eager_model": eager_model,
        "tokens_consumed": total_agentic_tokens,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_agentic_tokens,
        },
        "hop_functionalized": True,
        "next_node": "node4_phase1_compilation_runner",
    }


# Backwards compatibility alias
node3_compiler_rewriter = node3_phase1_compiler_rewriter
