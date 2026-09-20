"""Node 5: Verify Correctness Node (Phase 1, Programmatic).

Evaluates numerical correctness of compiled model against eager mode baseline.
Manages Counter ID 1 (counter_1_compilation).
Uses ChiaFunction for cluster worker orchestration and hardware device scheduling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from contextlib import contextmanager
try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:
    def ChiaFunction(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import node9_phase1_error_handler

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node5_phase1_verify_correctness")


@contextmanager
def _freeze_noise(models: list[Any]):
    """Temporarily disables stochastic noise parameters (e.g., sigma=0.0) during compilation verification."""
    saved: list[tuple[Any, Any]] = []
    seen_ids: set[int] = set()
    for m in models:
        if m is not None and hasattr(m, "modules"):
            for mod in m.modules():
                mod_id = id(mod)
                if mod_id not in seen_ids and hasattr(mod, "sigma") and isinstance(getattr(mod, "sigma"), (int, float)):
                    seen_ids.add(mod_id)
                    saved.append((mod, getattr(mod, "sigma")))
                    setattr(mod, "sigma", 0.0)
    try:
        yield
    finally:
        for mod, orig_val in saved:
            setattr(mod, "sigma", orig_val)


def _resolve_target_device(requested_device: str = "mps") -> str:
    """Resolve target device with fallback based on hardware availability."""
    req = str(requested_device).lower()
    if "cuda" in req and torch.cuda.is_available():
        return "cuda"
    if "mps" in req and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if req in ("cuda", "mps"):
        logger.info("Node 5: Requested device '%s' unavailable, falling back to 'cpu'.", requested_device)
        return "cpu"
    return "cpu"


def _sync_device(device: str) -> None:
    """Synchronize accelerator device queues to ensure execution completes."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


@ChiaFunction(resources={"sim_worker": 1})
def node5_phase1_verify_correctness(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    compiled_model: Optional[Any] = None,
    eager_model: Optional[Any] = None,
    sample_input: Optional[Any] = None,
    target_device: str = "mps",
    compilation_status: str = "SUCCESS",
    compilation_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Node 5 Verify Correctness (Phase 1).

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        compiled_model: Compiled PyTorch model callable.
        eager_model: Eager reference model.
        sample_input: Optional sample evaluation data tensor/input.
        target_device: Execution device ('mps', 'cuda', 'cpu').
        compilation_status: Compilation status from Node 4 ('SUCCESS' or 'FAILED').
        compilation_error: Error message/traceback from Node 4 if compilation failed.

    Returns:
        Dict specifying verification status (PASSED/FAILED/TERMINATED_MAX_CYCLES),
        details, error_logs, tolerances (rtol, atol), max_abs_diff, max_rel_diff,
        and counter_1 value.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    curr_counter_1 = proposal.get("counter_1_compilation", 0)

    # Tolerances from config
    compile_tol = getattr(config.tolerances, "compile_check", None)
    rtol = getattr(compile_tol, "rtol", 1e-4)
    atol = getattr(compile_tol, "atol", 1e-4)

    device = _resolve_target_device(target_device)
    logger.info("Node 5: Verifying correctness for %s on device %s (rtol=%.1e, atol=%.1e)", candidate_id, device, rtol, atol)

    def _handle_failure(
        error_stage: str,
        error_msg: str,
        max_abs: float = 0.0,
        max_rel: float = 0.0,
    ) -> Dict[str, Any]:
        """Helper to centralize Counter 1 increment, limit check, and DB updates."""
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
                "Node 5: Counter 1 limit exceeded (%d > %d) for candidate %s: %s",
                new_counter_1,
                max_counter_1,
                candidate_id,
                error_msg,
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
                error_stage=error_stage,
                details=error_msg,
            )

        db.update_execution_status(
            candidate_id,
            compilation_status="FAILED",
            error_stage=error_stage,
            error_message=error_msg,
        )
        logger.warning("Node 5: Verification FAILED for %s (Counter 1 = %d)", candidate_id, new_counter_1)

        # Attach telemetry to profiler if available
        try:
            from chia.base.profiling import get_profiler
            get_profiler().add_info({
                "candidate_id": candidate_id,
                "node": "node5_phase1_verify_correctness",
                "is_correct": False,
                "counter_1": new_counter_1,
                "error_stage": error_stage,
                "error_message": error_msg,
                "target_device": device,
            })
        except Exception:
            pass

        return {
            "candidate_id": candidate_id,
            "status": "FAILED",
            "counter_1": new_counter_1,
            "details": error_msg,
            "error_logs": error_msg,
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "rtol": rtol,
            "atol": atol,
            "target_device": device,
            "next_node": "node8_phase1_diagnostic_proposal",
        }

    # 1. Handle Compilation Failures or Missing Models upfront
    if compilation_status == "FAILED" or compiled_model is None or eager_model is None:
        if compilation_status == "FAILED" or compiled_model is None:
            err_stage = "compilation"
            if compilation_error:
                err_msg = compilation_error
            elif compiled_model is None and eager_model is None:
                err_msg = "Verification failed: missing model artifact(s): compiled_model, eager_model."
            elif compiled_model is None:
                err_msg = "Verification failed: missing model artifact(s): compiled_model."
            else:
                err_msg = "Compilation failed in Node 4 runner"
        else:
            err_stage = "correctness_verification"
            err_msg = "Verification failed: missing model artifact(s): eager_model."

        logger.error("Node 5: %s for candidate %s", err_msg, candidate_id)
        return _handle_failure(error_stage=err_stage, error_msg=err_msg)

    # 2. Pure Numerical Verification (both compiled and eager models present)
    max_rel_diff = 0.0
    max_abs_diff = 0.0

    try:
        # Place eager model on device and set evaluation mode if supported
        if hasattr(eager_model, "to") and callable(getattr(eager_model, "to")):
            try:
                eager_model = eager_model.to(device)
            except Exception as dev_err:
                logger.warning("Node 5: eager_model.to(%s) failed: %s; running on current device.", device, dev_err)

        if hasattr(eager_model, "eval") and callable(getattr(eager_model, "eval")):
            eager_model.eval()

        if hasattr(compiled_model, "to") and callable(getattr(compiled_model, "to")):
            try:
                compiled_model = compiled_model.to(device)
            except Exception as dev_err:
                logger.warning("Node 5: compiled_model.to(%s) failed: %s; running on current device.", device, dev_err)

        if hasattr(compiled_model, "eval") and callable(getattr(compiled_model, "eval")):
            compiled_model.eval()

        # Warmup and execute model on sample inputs under torch.no_grad() and frozen noise
        num_samples = 100
        with torch.no_grad(), _freeze_noise([compiled_model, eager_model]):
            if sample_input is not None:
                if isinstance(sample_input, (tuple, list)):
                    inp = [x.to(device) if hasattr(x, "to") and callable(getattr(x, "to")) else x for x in sample_input]
                    for _ in range(2):
                        if callable(compiled_model):
                            _ = compiled_model(*inp)
                        if callable(eager_model):
                            _ = eager_model(*inp)
                    _sync_device(device)

                    torch.manual_seed(42)
                    out_compiled = compiled_model(*inp)
                    _sync_device(device)
                    torch.manual_seed(42)
                    out_eager = eager_model(*inp)
                    _sync_device(device)
                else:
                    inp = sample_input.to(device) if hasattr(sample_input, "to") and callable(getattr(sample_input, "to")) else sample_input
                    for _ in range(2):
                        if callable(compiled_model):
                            _ = compiled_model(inp)
                        if callable(eager_model):
                            _ = eager_model(inp)
                    _sync_device(device)

                    torch.manual_seed(42)
                    out_compiled = compiled_model(inp)
                    _sync_device(device)
                    torch.manual_seed(42)
                    out_eager = eager_model(inp)
                    _sync_device(device)
            else:
                # Determine whether this is a multi-input Kuramoto model (state + drive)
                n_osc = (
                    getattr(eager_model, "n_oscillators", None)
                    or getattr(eager_model, "n", None)
                    or getattr(getattr(eager_model, "dynamics", None), "n", None)
                    or getattr(getattr(eager_model, "dynamics", None), "n_oscillators", None)
                    or getattr(getattr(config, "reference_design", None), "n_oscillators", None)
                )
                n_cond = (
                    getattr(eager_model, "n_conditional_oscillators", None)
                    or getattr(eager_model, "n_cond", None)
                    or getattr(getattr(eager_model, "dynamics", None), "n_cond", None)
                    or getattr(getattr(eager_model, "dynamics", None), "n_conditional_oscillators", None)
                    or getattr(getattr(config, "reference_design", None), "n_conditional_oscillators", None)
                )

                if n_osc is not None and n_cond is not None:
                    state_dim = getattr(eager_model, "state_dim", None) or getattr(getattr(eager_model, "dynamics", None), "state_dim", None) or (n_osc + n_cond)
                    batch_size = 4
                    sample_state = torch.randn(batch_size, state_dim, device=device)
                    sample_drive = torch.randn(batch_size, n_osc, n_cond, device=device)

                    # Warmup (2 passes under no_grad)
                    for _ in range(2):
                        try:
                            _ = compiled_model(sample_state, sample_drive)
                        except TypeError:
                            _ = compiled_model(sample_state)
                        try:
                            _ = eager_model(sample_state, sample_drive)
                        except TypeError:
                            _ = eager_model(sample_state)
                    _sync_device(device)

                    # Test invocation with state and drive
                    try:
                        out_compiled = compiled_model(sample_state, sample_drive)
                    except TypeError:
                        out_compiled = compiled_model(sample_state)
                    _sync_device(device)

                    try:
                        out_eager = eager_model(sample_state, sample_drive)
                    except TypeError:
                        out_eager = eager_model(sample_state)
                    _sync_device(device)
                else:
                    state_dim = getattr(eager_model, "state_dim", 1024)
                    inp = torch.randn(num_samples, state_dim, device=device)

                    # Warmup (2 passes under no_grad)
                    for _ in range(2):
                        if callable(compiled_model):
                            _ = compiled_model(inp)
                        if callable(eager_model):
                            _ = eager_model(inp)
                    _sync_device(device)

                    out_compiled = compiled_model(inp)
                    _sync_device(device)
                    out_eager = eager_model(inp)
                    _sync_device(device)

        if isinstance(out_compiled, torch.Tensor) and isinstance(out_eager, torch.Tensor):
            abs_diff = torch.abs(out_compiled - out_eager)
            rel_diff = abs_diff / (torch.abs(out_eager) + 1e-8)

            max_abs_diff = float(torch.max(abs_diff).item())
            max_rel_diff = float(torch.max(rel_diff).item())

            if max_abs_diff > atol or max_rel_diff > rtol:
                error_msg = f"Numerical tolerance exceeded: max_abs_diff={max_abs_diff:.6e} > atol={atol}, max_rel_diff={max_rel_diff:.6e} > rtol={rtol}"
                return _handle_failure(
                    error_stage="correctness_verification",
                    error_msg=error_msg,
                    max_abs=max_abs_diff,
                    max_rel=max_rel_diff,
                )
        elif type(out_compiled) != type(out_eager):
            error_msg = f"Output type mismatch: compiled returned {type(out_compiled)}, eager returned {type(out_eager)}"
            return _handle_failure(
                error_stage="correctness_verification",
                error_msg=error_msg,
                max_abs=max_abs_diff,
                max_rel=max_rel_diff,
            )

    except Exception as exc:
        error_msg = f"Error during numerical verification: {exc}"
        logger.error("Node 5 evaluation error for %s: %s", candidate_id, exc)
        return _handle_failure(
            error_stage="correctness_verification",
            error_msg=error_msg,
            max_abs=0.0,
            max_rel=0.0,
        )

    # Reset Counter ID 1 on success
    db.update_counters(candidate_id, counter_1_compilation=0)
    logger.info("Node 5: Verification PASSED for candidate %s (Counter 1 reset to 0)", candidate_id)

    # Attach verification metadata to Chia profiler if available
    try:
        from chia.base.profiling import get_profiler
        get_profiler().add_info({
            "candidate_id": candidate_id,
            "node": "node5_phase1_verify_correctness",
            "is_correct": True,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "target_device": device,
        })
    except Exception:
        pass

    return {
        "candidate_id": candidate_id,
        "status": "PASSED",
        "compiled_model": compiled_model,
        "counter_1": 0,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "rtol": rtol,
        "atol": atol,
        "target_device": device,
        "details": f"Verification passed: max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}",
        "error_logs": "",
        "next_node": "node6_phase1_simulate_profile",
    }


# Backwards compatibility alias
node5_verify_correctness = node5_phase1_verify_correctness
