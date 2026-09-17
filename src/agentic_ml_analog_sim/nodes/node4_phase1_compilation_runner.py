"""Node 4: Compilation Runner Node (Phase 1, Programmatic).

Compiles the functionalized HOP model targeting Inductor backend on target hardware (MPS / CUDA / CPU)
dispatched in a Ray worker using the Chia API (@ChiaFunction).
Numerical correctness verification and warmup passes are deferred to Node 5.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, Optional

import torch

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


@ChiaFunction()
def _compile_model_worker_task(
    model: Any,
    backend: str = "inductor",
    target_device: str = "mps",
    dynamic: bool = True,
) -> Any:
    """ChiaFunction worker task executing PyTorch model compilation on cluster worker."""
    from un0.compiler import compile_model
    return compile_model(
        model,
        backend=backend,
        target_device=target_device,
        dynamic=dynamic,
    )


@ChiaFunction()
def node4_phase1_compilation_runner(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    target_device: str = "mps",
    un0_dir: Optional[str] = None,
    prepared_model: Optional[Any] = None,
    eager_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Node 4 Compilation Runner (Phase 1).

    Compiles the model with Inductor backend on target hardware (MPS/CUDA/CPU)
    in a Ray worker managed by the Chia API.
    Numerical correctness verification and warmup passes are deferred to Node 5.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        target_device: Target execution device ('mps', 'cuda', 'cpu').
        un0_dir: Optional Un-0 workspace directory override.
        prepared_model: Functionalized HOP model from Node 3.
        eager_model: Eager reference model with solver & noise rewrites.

    Returns:
        Dict detailing candidate_id, compilation_status, compiled model, eager model,
        and build logs.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

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

    num_steps = proposal.get("num_steps") or getattr(ref_design, "num_steps", None)
    if num_steps is None:
        raise ValueError(f"num_steps not specified for candidate '{candidate_id}' or reference design.")

    solver_params = proposal.get("solver_params", {}) or {}
    if isinstance(solver_params, str):
        try:
            import json
            solver_params = json.loads(solver_params)
        except Exception:
            solver_params = {}

    integration_time = solver_params.get("integration_time") or getattr(ref_design, "integration_time", None)
    if integration_time is None:
        raise ValueError(f"integration_time not specified for candidate '{candidate_id}' or reference design.")

    dt = solver_params.get("dt") or (float(integration_time) / float(num_steps))

    if eager_model is None:
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
            logger.warning("Node 4: Eager model resolution warning: %s", exc)

    if prepared_model is None:
        try:
            from un0.compiler import functionalize_kuramoto_model
            prepared_model = functionalize_kuramoto_model(
                model=eager_model,
                solver_name=solver_name,
                num_steps=num_steps,
                dt=dt,
                integration_time=integration_time,
                use_hop=True,
            )
        except Exception as exc:
            logger.warning("Node 4: Functionalized HOP model fallback issue: %s", exc)
            prepared_model = eager_model

    compiled_model = None
    build_logs = ""
    compilation_status = "SUCCESS"

    try:
        model_to_compile = prepared_model if prepared_model is not None else eager_model
        if model_to_compile is not None:
            # Task of Node 4 is compilation in a Ray worker using Chia API
            worker_resources: Dict[str, Any] = {}
            can_dispatch_remote = False

            import ray
            if ray.is_initialized():
                cluster_res = ray.cluster_resources()
                if device in ("mps", "cuda"):
                    target_res_key = "metal_gpu" if device == "mps" else "GPU"
                    if target_res_key in cluster_res and cluster_res[target_res_key] >= 1:
                        worker_resources = {target_res_key: 1}
                        can_dispatch_remote = True
                elif "python_worker" in cluster_res and cluster_res["python_worker"] >= 1:
                    worker_resources = {"python_worker": 1}
                    can_dispatch_remote = True

            if can_dispatch_remote:
                try:
                    compiled_model = _compile_model_worker_task.options(
                        resources=worker_resources
                    ).chia_remote_blocking(
                        model_to_compile,
                        backend="inductor",
                        target_device=device,
                        dynamic=None,
                    )
                except Exception as rem_exc:
                    logger.warning("Remote compilation on worker failed (%s); retrying locally.", rem_exc)
                    compiled_model = _compile_model_worker_task(
                        model_to_compile,
                        backend="inductor",
                        target_device=device,
                        dynamic=None,
                    )
            else:
                compiled_model = _compile_model_worker_task(
                    model_to_compile,
                    backend="inductor",
                    target_device=device,
                    dynamic=None,
                )

            build_logs = f"Torch Inductor compilation succeeded on {device} (dynamic batch shapes B>=1)."
        else:
            compiled_model = eager_model
            build_logs = f"Compilation bypassed: fallback to eager model on {device}."

    except Exception as exc:
        compilation_status = "FAILED"
        build_logs = f"Compilation error on device '{device}': {exc}"
        logger.error("Node 4 compilation failed for candidate %s: %s", candidate_id, exc)

    # Update DB status
    db.update_execution_status(
        candidate_id,
        compilation_status=compilation_status,
        error_stage="compilation" if compilation_status == "FAILED" else None,
        error_message=build_logs if compilation_status == "FAILED" else None,
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

    eager_baseline = prepared_model if prepared_model is not None else eager_model

    if compilation_status == "SUCCESS":
        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "compiled_model": compiled_model if compiled_model is not None else eager_baseline,
            "eager_model": eager_baseline,
            "target_device": device,
            "build_logs": build_logs,
            "next_node": "node5_phase1_verify_correctness",
        }
    else:
        return {
            "candidate_id": candidate_id,
            "status": "FAILED",
            "compiled_model": None,
            "eager_model": eager_baseline,
            "target_device": device,
            "error_logs": build_logs,
            "error_message": build_logs,
            "next_node": "node8_phase1_diagnostic_proposal",
        }


# Backwards compatibility alias
node4_compilation_runner = node4_phase1_compilation_runner
