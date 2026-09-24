"""Node 4: Compilation Runner Node (Phase 1, Programmatic).

Compiles the functionalized trajectory model (`prepared_model`) targeting the PyTorch
Inductor backend on target hardware (MPS / CUDA / CPU) dispatched to a Ray worker
using the Chia API (`@ChiaFunction`).
Numerical correctness verification and warmup passes are deferred to Node 5.

Model Taxonomy & Contracts:
- `prepared_model`: Discrete trajectory integrator model synthesized by Node 3
  (e.g., `FunctionalHOPKuramotoDynamics`), with call signature `forward(state, drive=None)`.
  This is the model compiled via TorchInductor.
- `eager_model`: Continuous physics dynamics model (e.g. `ConditionalKuramotoDynamics`
  with noise wrappers) with signature `forward(state, _time, drive)`. Node 4 adapts
  this into `eager_baseline` (an uncompiled trajectory model with signature `forward(state, drive)`)
  so Node 5 can verify numerical parity against `compiled_model`.
- Standalone Test Fallbacks: In normal pipeline execution, Node 3 always runs prior to Node 4
  and provides both models. Standalone fallbacks are strictly reserved for isolated unit tests.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

import torch

try:
    import ray
except ImportError:
    ray = None

try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:
    def ChiaFunction(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.tools.un0_context_tool import resolve_un0_root

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node4_phase1_compilation_runner")


def _resolve_target_device(requested_device: str = "mps") -> str:
    """Resolve target device with fallback based on hardware availability."""
    req = str(requested_device).lower()
    if "cuda" in req and torch.cuda.is_available():
        return "cuda"
    if "mps" in req and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if req in ("cuda", "mps"):
        logger.info("Requested device '%s' unavailable, falling back to 'cpu'.", requested_device)
        return "cpu"
    return "cpu"


def _sync_device(device: str) -> None:
    """Synchronize accelerator device queues to ensure accurate benchmarking."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _is_remote_or_readonly(db: Optional[Any] = None) -> tuple[bool, bool]:
    """Check whether node is executing on a remote worker or db is unavailable/read-only."""
    is_remote = False
    if ray is not None and hasattr(ray, "is_initialized") and ray.is_initialized():
        try:
            curr_ip = ray.util.get_node_ip_address()
            if curr_ip in ("127.0.0.1", "localhost"):
                is_remote = False
            else:
                ctx = ray.get_runtime_context()
                gcs_addr = getattr(ctx, "gcs_address", None)
                if gcs_addr:
                    head_ip = gcs_addr.split(":")[0]
                    is_remote = (curr_ip != head_ip)
                else:
                    is_remote = (curr_ip != "127.0.0.1")
        except Exception:
            is_remote = False

    is_readonly = False
    if db is None:
        is_readonly = True
    else:
        underlying_db = getattr(db, "_db", db)
        db_path = getattr(underlying_db, "db_path", None)
        if db_path is not None:
            try:
                import os
                p = str(db_path)
                if os.path.exists(p) and not os.access(p, os.W_OK):
                    is_readonly = True
            except Exception:
                pass
    return is_remote, is_readonly


@ChiaFunction(num_cpus=4, resources={"compile_worker": 1})
def node4_phase1_compilation_runner(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    target_device: str = "mps",
    un0_dir: Optional[str] = None,
    prepared_model: Optional[Any] = None,
    eager_model: Optional[Any] = None,
    registered_rewrite: Optional[str] = None,
    proposal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Node 4 Compilation Runner (Phase 1).

    Orchestrates compilation of the prepared model with Inductor backend on target hardware (MPS/CUDA/CPU)
    using the Chia API (@ChiaFunction(num_cpus=2)).
    Unconditionally routes to Node 5 (Verify Correctness).

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        target_device: Target execution device ('mps', 'cuda', 'cpu').
        un0_dir: Optional Un-0 workspace directory override.
        prepared_model: Functionalized HOP model from Node 3.
        eager_model: Eager reference model with solver & noise rewrites.
        registered_rewrite: Optional identifier of the registered rewrite.
        proposal: Optional pre-fetched proposal dictionary.

    Returns:
        Dict detailing candidate_id, status, compiled_model, eager_model, target_device,
        build_logs, error_logs, error_message, and next_node ('node5_phase1_verify_correctness').
    """
    if proposal is None and db is not None:
        try:
            proposal = db.get_proposal(candidate_id)
        except Exception as err:
            logger.warning("Node 4: Could not read proposal from db (%s)", err)
            proposal = None

    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database or arguments.")

    device = _resolve_target_device(target_device)
    logger.info("Node 4: Compiling model for candidate %s on target device '%s'", candidate_id, device)

    # Dynamic Un-0 root resolution and sys.path setup
    resolved_un0_dir = resolve_un0_root(un0_dir)
    if resolved_un0_dir.is_dir() and str(resolved_un0_dir) not in sys.path:
        sys.path.insert(0, str(resolved_un0_dir))

    # Resolve architecture and simulation parameters directly from configuration / proposal
    ref_design = getattr(config, "reference_design", None)
    if ref_design is None:
        raise ValueError("Simulation configuration is missing 'reference_design'.")

    n_osc = getattr(ref_design, "n_oscillators", None)
    n_cond = getattr(ref_design, "n_conditional_oscillators", None)
    if n_osc is None or n_cond is None:
        raise ValueError("Simulation configuration reference_design must specify 'n_oscillators' and 'n_conditional_oscillators'.")
    state_dim = n_osc + n_cond

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

    solver_name = proposal.get("solver") or getattr(ref_design, "solver", None)
    if not solver_name:
        raise KeyError(f"No solver specified for candidate '{candidate_id}' in proposal or simulation config.")

    solver_params = proposal.get("solver_params", {}) or {}
    if isinstance(solver_params, str):
        try:
            import json
            solver_params = json.loads(solver_params)
        except Exception:
            solver_params = {}

    num_steps = (
        proposal.get("num_steps")
        or solver_params.get("num_steps")
        or getattr(ref_design, "num_steps", None)
    )
    if num_steps is None:
        raise ValueError(f"num_steps not specified for candidate '{candidate_id}' or reference design.")

    integration_time = solver_params.get("integration_time") or getattr(ref_design, "integration_time", None)
    if integration_time is None:
        raise ValueError(f"integration_time not specified for candidate '{candidate_id}' or reference design.")

    dt = solver_params.get("dt") or (float(integration_time) / float(num_steps))

    # =========================================================================
    # STANDALONE / ISOLATED TEST FIXTURE RESOLUTION ONLY
    # In normal pipeline execution, Node 3 ALWAYS runs prior to Node 4 and provides
    # both `prepared_model` (functional trajectory model) and `eager_model`.
    # The blocks below exist strictly to support isolated unit tests and standalone
    # invocations where Node 4 is called without prior execution of Node 3.
    # =========================================================================
    if eager_model is None:
        logger.debug("Node 4: 'eager_model' not provided. Instantiating baseline for isolated test execution.")
        try:
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

            if noise_name and str(noise_name).lower() != "none" and not proposal.get("is_reference"):
                noise_cls = get_noise_model(noise_name)
                noise_params = proposal.get("noise_params", {}) or {}
                if isinstance(noise_params, str):
                    try:
                        import json
                        noise_params = json.loads(noise_params)
                    except Exception:
                        noise_params = {}
                eager_model = noise_cls(base_dynamics, **(noise_params if isinstance(noise_params, dict) else {}))
            else:
                eager_model = base_dynamics
        except Exception as exc:
            logger.warning("Node 4: Isolated test eager model resolution warning: %s", exc)

    if prepared_model is None:
        if eager_model is not None:
            if registered_rewrite:
                try:
                    from un0.compiler.registry import get_compiler_rewrite
                    rewrite_fn = get_compiler_rewrite(registered_rewrite)
                    prepared_model = rewrite_fn(
                        model=eager_model,
                        solver_name=solver_name,
                        num_steps=num_steps,
                        dt=dt,
                        integration_time=integration_time,
                        target_device=device,
                    )
                    logger.info("Node 4: Applied registered compiler rewrite '%s' on worker.", registered_rewrite)
                except Exception as rw_err:
                    logger.warning("Node 4: Failed to load rewrite '%s', falling back to functionalize: %s", registered_rewrite, rw_err)
            if prepared_model is None:
                logger.info(
                    "Node 4: Synthesizing unrolled functional model from 'eager_model' on worker."
                )
                try:
                    from un0.compiler import functionalize_kuramoto_model
                    prepared_model = functionalize_kuramoto_model(
                        model=eager_model,
                        solver_name=solver_name,
                        num_steps=num_steps,
                        dt=dt,
                        integration_time=integration_time,
                        use_hop=False,
                    )
                except Exception as synth_err:
                    logger.warning("Node 4: Functional model synthesis fallback error: %s", synth_err)
                    prepared_model = eager_model
        else:
            logger.error("Node 4: Neither 'prepared_model' nor 'eager_model' was provided.")

    model_to_compile = prepared_model if prepared_model is not None else eager_model

    # Construct sample dummy inputs for dry-run JIT lowering (batch size 4)
    sample_state = torch.randn(4, state_dim)
    sample_drive = torch.randn(4, n_osc, n_cond)

    # Prepare eager baseline trajectory model with matching (state, drive) interface for Node 5
    eager_baseline = eager_model
    if eager_model is not None:
        try:
            import inspect
            sig = inspect.signature(eager_model.forward)
            if "_time" in sig.parameters or "time" in sig.parameters or "t" in sig.parameters:
                from un0.compiler import functionalize_kuramoto_model
                eager_baseline = functionalize_kuramoto_model(
                    model=eager_model,
                    solver_name=solver_name,
                    num_steps=num_steps,
                    dt=dt,
                    integration_time=integration_time,
                    use_hop=False,
                )
        except Exception as align_err:
            logger.debug("Node 4: Eager baseline alignment skipped: %s", align_err)

    compiled_model = None
    build_logs = ""
    compilation_status = "SUCCESS"

    try:
        if model_to_compile is None:
            compilation_status = "FAILED"
            build_logs = "Compilation error: No model available to compile (prepared_model and eager_model are None)."
            logger.error("Node 4: %s", build_logs)
        else:
            if hasattr(model_to_compile, "eval") and callable(getattr(model_to_compile, "eval")):
                model_to_compile.eval()

            from un0.compiler import compile_model

            compiled_model = compile_model(
                model_to_compile,
                backend="inductor",
                target_device=device,
                dynamic=True,
            )

            # Trigger PyTorch lazy compilation via dry-run pass on target device
            if sample_state is not None:
                s = sample_state.to(device) if hasattr(sample_state, "to") and callable(getattr(sample_state, "to")) else sample_state
                d = sample_drive.to(device) if sample_drive is not None and hasattr(sample_drive, "to") and callable(getattr(sample_drive, "to")) else sample_drive
                with torch.inference_mode():
                    if d is not None:
                        try:
                            _ = compiled_model(s, d)
                        except TypeError:
                            _ = compiled_model(s)
                    else:
                        _ = compiled_model(s)
                _sync_device(device)

            build_logs = f"Torch Inductor compilation succeeded on {device} (dynamic batch shapes B>=1)."

    except Exception as exc:
        # DO NOT retry locally on failure; record failure cleanly for Node 5/8/9
        compilation_status = "FAILED"
        compiled_model = None
        build_logs = f"Compilation error on device '{device}': {exc}"
        logger.error("Node 4 compilation failed for candidate %s: %s", candidate_id, exc)

    # Update DB status (Node 4 never increments Counter 1; Counter 1 is owned by Node 5)
    is_remote, is_readonly = _is_remote_or_readonly(db)
    if db is not None and not is_remote and not is_readonly:
        try:
            db.update_execution_status(
                candidate_id,
                compilation_status=compilation_status,
                error_stage="compilation" if compilation_status == "FAILED" else None,
                error_message=build_logs if compilation_status == "FAILED" else None,
            )
        except Exception as db_exc:
            logger.debug("Node 4: Non-critical failure updating DB status (driver will sync): %s", db_exc)
    else:
        logger.debug(
            "Node 4: Skipping direct DB status update (remote=%s, readonly=%s, driver will sync)",
            is_remote,
            is_readonly,
        )

    # Attach compilation telemetry to Chia profiler if available
    try:
        from chia.base.profiling import get_profiler
        get_profiler().add_info({
            "node": "node4_phase1_compilation_runner",
            "candidate_id": candidate_id,
            "target_device": device,
            "compilation_status": compilation_status,
        })
    except Exception:
        pass

    # Node 4 UNCONDITIONALLY routes to Node 5 (Verify Correctness)
    if compilation_status == "SUCCESS":
        if ray is not None and hasattr(ray, "is_initialized") and ray.is_initialized():
            compiled_model_ref = ray.put(compiled_model)
            eager_model_ref = ray.put(eager_baseline)
        else:
            compiled_model_ref = compiled_model
            eager_model_ref = eager_baseline
        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "compiled_model": compiled_model_ref,
            "eager_model": eager_model_ref,
            "target_device": device,
            "build_logs": build_logs,
            "error_logs": "",
            "error_message": "",
            "next_node": "node5_phase1_verify_correctness",
        }
    else:
        if ray is not None and hasattr(ray, "is_initialized") and ray.is_initialized():
            eager_model_ref = ray.put(eager_baseline) if eager_baseline is not None else None
        else:
            eager_model_ref = eager_baseline
        return {
            "candidate_id": candidate_id,
            "status": "FAILED",
            "compiled_model": None,
            "eager_model": eager_model_ref,
            "target_device": device,
            "build_logs": build_logs,
            "error_logs": build_logs,
            "error_message": build_logs,
            "next_node": "node5_phase1_verify_correctness",
        }


# Backwards compatibility aliases and test introspection attributes
node4_compilation_runner = node4_phase1_compilation_runner
compile_model_worker_task = node4_phase1_compilation_runner
_compile_model_worker_task = node4_phase1_compilation_runner
