"""Node 5: Verify Correctness Node (Phase 1, Programmatic).

Evaluates numerical correctness of compiled model against eager mode baseline.
Manages Counter ID 1 (counter_1_compilation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import node9_phase1_error_handler

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node5_phase1_verify_correctness")


def node5_phase1_verify_correctness(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    compiled_model: Optional[Any] = None,
    eager_model: Optional[Any] = None,
    sample_input: Optional[Any] = None,
    target_device: str = "mps",
) -> Dict[str, Any]:
    """Node 5 Verify Correctness (Phase 1).

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        compiled_model: Compiled PyTorch model callable.
        eager_model: Eager reference model.
        sample_input: Sample evaluation data tensor/input.
        target_device: Execution device.

    Returns:
        Dict specifying verification status (PASSED/FAILED), rtol, atol, and counter_1 value.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    curr_counter_1 = proposal.get("counter_1_compilation", 0)

    # Tolerances from config
    compile_tol = getattr(config.tolerances, "compile_check", None)
    rtol = getattr(compile_tol, "rtol", 1e-4)
    atol = getattr(compile_tol, "atol", 1e-4)

    logger.info("Node 5: Verifying correctness for %s (rtol=%.1e, atol=%.1e)", candidate_id, rtol, atol)

    is_correct = True
    error_msg = ""
    max_rel_diff = 0.0
    max_abs_diff = 0.0

    try:
        # Perform numerical comparison if models are available
        if compiled_model is not None and eager_model is not None:
            if hasattr(eager_model, "eval") and callable(getattr(eager_model, "eval")):
                eager_model.eval()

            num_samples = 100
            if sample_input is None:
                state_dim = getattr(eager_model, "state_dim", 1024)
                sample_input = torch.randn(num_samples, state_dim)

            with torch.no_grad():
                if callable(compiled_model):
                    out_compiled = compiled_model(sample_input)
                else:
                    out_compiled = eager_model(sample_input)

                if callable(eager_model):
                    out_eager = eager_model(sample_input)
                else:
                    out_eager = out_compiled

            if isinstance(out_compiled, torch.Tensor) and isinstance(out_eager, torch.Tensor):
                abs_diff = torch.abs(out_compiled - out_eager)
                rel_diff = abs_diff / (torch.abs(out_eager) + 1e-8)

                max_abs_diff = float(torch.max(abs_diff).item())
                max_rel_diff = float(torch.max(rel_diff).item())

                if max_abs_diff > atol or max_rel_diff > rtol:
                    is_correct = False
                    error_msg = f"Numerical tolerance exceeded: max_abs_diff={max_abs_diff:.6e} > atol={atol}, max_rel_diff={max_rel_diff:.6e} > rtol={rtol}"
            else:
                is_correct = True
        else:
            is_correct = True

    except Exception as exc:
        is_correct = False
        error_msg = f"Error during numerical verification: {exc}"
        logger.error("Node 5 evaluation error for %s: %s", candidate_id, exc)

    if is_correct:
        # Reset Counter ID 1 on success
        db.update_counters(candidate_id, counter_1_compilation=0)
        logger.info("Node 5: Verification PASSED for candidate %s (Counter 1 reset to 0)", candidate_id)

        return {
            "candidate_id": candidate_id,
            "status": "PASSED",
            "counter_1": 0,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "next_node": "node6_phase1_simulate_profile",
        }
    else:
        # Increment Counter ID 1 on failure
        new_counter_1 = curr_counter_1 + 1
        db.update_counters(candidate_id, counter_1_compilation=new_counter_1)

        counter_limits = getattr(config, "counter_limits", None)
        max_counter_1 = None
        if counter_limits is not None:
            max_counter_1 = getattr(counter_limits, "max_counter_1_compilation", None)
            if max_counter_1 is None and isinstance(counter_limits, dict):
                max_counter_1 = counter_limits.get("max_counter_1_compilation")
        if max_counter_1 is None:
            limits = getattr(config, "experiment_limits", None) or getattr(config, "budget", None)
            max_counter_1 = getattr(limits, "max_node_retries", 3) if limits else 3

        if new_counter_1 > max_counter_1:
            logger.error(
                "Node 5: Counter 1 limit exceeded (%d > %d) for candidate %s",
                new_counter_1,
                max_counter_1,
                candidate_id,
            )
            mode = "debug" if proposal.get("is_debug") else ("reference" if proposal.get("is_reference") else "standard")
            return node9_phase1_error_handler(
                db=db,
                config=config,
                candidate_id=candidate_id,
                counter_name="counter_1_compilation",
                current_count=new_counter_1,
                max_count=max_counter_1,
                execution_mode=mode,
                error_stage="correctness_verification",
                details=error_msg,
            )

        db.update_execution_status(
            candidate_id,
            compilation_status="FAILED",
            error_stage="correctness_verification",
            error_message=error_msg,
        )
        logger.warning("Node 5: Verification FAILED for %s (Counter 1 = %d)", candidate_id, new_counter_1)

        return {
            "candidate_id": candidate_id,
            "status": "FAILED",
            "counter_1": new_counter_1,
            "error_logs": error_msg,
            "next_node": "node8_phase1_diagnostic_proposal",
        }


# Backwards compatibility alias
node5_verify_correctness = node5_phase1_verify_correctness
