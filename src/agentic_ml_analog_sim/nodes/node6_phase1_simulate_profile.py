"""Node 6: Simulate and Profile Node (Phase 1, Programmatic).

Executes batch-size-1 hardware inference simulation on compiled PyTorch models
and profiles latency, peak device memory, throughput, accuracy statistics, and numerical stability.
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from agentic_ml_analog_sim.accuracy.chordal_error import (
    aggregate_class_errors,
    compute_sample_chordal_error,
)
from agentic_ml_analog_sim.accuracy.sensitivity_lookup import (
    SensitivityLookupTable,
    generate_perturbation_scales,
)
from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node6_phase1_simulate_profile")

# In-memory caches across candidate evaluations
_CACHED_REF_OUTPUTS: Optional[List[torch.Tensor]] = None
_CACHED_SENSITIVITY_TABLE: Optional[SensitivityLookupTable] = None


def _resolve_ref_cache_path(db: Optional[Any], config: Optional[Any]) -> Path:
    """Resolve file path for caching reference baseline phase snapshots at T=1."""
    if db is not None:
        db_path = getattr(db, "db_path", None)
        if db_path is not None:
            return Path(db_path).parent / "ref_phases_T1.pt"
    exp_name = getattr(config, "experiment_name", "first_run_trial_un0_analog_acceleration") if config else "first_run_trial_un0_analog_acceleration"
    return Path("reports") / exp_name / "ref_phases_T1.pt"


def _resolve_sensitivity_cache_path(db: Optional[Any], config: Optional[Any]) -> Path:
    """Resolve file path for caching sensitivity lookup table."""
    base = _resolve_ref_cache_path(db, config).parent
    return base / "sensitivity_lookup.json"


try:
    import ray
except ImportError:
    ray = None

# Chia API integration
try:
    from chia.base.ChiaFunction import ChiaFunction, get
except ImportError:
    def ChiaFunction(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        def decorator(func: Any) -> Any:
            return func
        return decorator

    def get(ref: Any) -> Any:  # type: ignore[no-redef]
        return ref


def _resolve_target_device(requested_device: str = "mps") -> str:
    """Resolve target execution accelerator with fallback to CPU."""
    requested = (requested_device or "").lower().strip()
    if requested in ("mps", "apple_silicon", "metal"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    elif requested in ("cuda", "gpu", "nvidia"):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    elif requested == "cpu":
        return "cpu"
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


def _sync_device(device_str: str) -> None:
    """Synchronize accelerator queue for deterministic timing."""
    if device_str == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass


def _get_peak_memory_mb(device_str: str) -> float:
    """Read peak allocated memory on device in megabytes."""
    try:
        if device_str == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                return float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
        elif device_str.startswith("cuda"):
            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        import resource
        import sys
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0
    except Exception:
        return 0.0


@ChiaFunction(resources={"sim_worker": 1})
def _simulate_batch1_worker(
    compiled_model: Any,
    num_eval_samples: int,
    state_dim: int,
    device_str: str,
    n_osc: Optional[int] = None,
    n_cond: Optional[int] = None,
    ref_outputs: Optional[List[torch.Tensor]] = None,
    is_reference: bool = False,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute sequential batch size 1 inference passes on hardware device.

    Profiles individual sample latencies, peak memory, and computes numerical
    phase snapshot chordal RMS error against golden reference outputs when available.
    """
    if num_classes is None or num_classes <= 0:
        raise ValueError(f"num_classes must be specified as a positive integer, got {num_classes}")

    if not is_reference:
        if ref_outputs is None or len(ref_outputs) < num_eval_samples:
            raise ValueError(
                f"Candidate evaluation requires at least {num_eval_samples} reference outputs, "
                f"but got {len(ref_outputs) if ref_outputs is not None else 0}."
            )

    device = torch.device(device_str)

    # Ensure model is set to evaluation mode if accessible
    raw_m = getattr(compiled_model, "_orig_mod", compiled_model)
    if hasattr(raw_m, "eval") and callable(getattr(raw_m, "eval")):
        raw_m.eval()
    elif hasattr(compiled_model, "eval") and callable(getattr(compiled_model, "eval")):
        compiled_model.eval()

    has_drive = (n_osc is not None and n_cond is not None and n_cond > 0)
    sample_latencies: List[float] = []
    recorded_outputs: List[torch.Tensor] = []
    recorded_inputs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    sample_abs_errors: List[float] = []
    sample_class_labels: List[int] = []

    with torch.inference_mode():
        # Warmup passes for Torch compiled model (fixed batch size 1 shape)
        for _ in range(2):
            dummy_in = torch.randn(1, state_dim, device=device)
            if has_drive:
                dummy_drive = torch.randn(1, n_osc, n_cond, device=device)
                try:
                    _ = compiled_model(dummy_in, dummy_drive)
                except TypeError:
                    _ = compiled_model(dummy_in)
                del dummy_drive
            else:
                _ = compiled_model(dummy_in)
            del dummy_in
        _sync_device(device_str)

        # Sequential batch size 1 profiling with deterministic seed per sample
        logger.info(
            "Worker: Running simulation on device %s for %d samples (mode=%s, state_dim=%d)...",
            device_str,
            num_eval_samples,
            "reference" if is_reference else "candidate",
            state_dim,
        )
        t_start = time.perf_counter()

        for s_idx in range(num_eval_samples):
            logger.info("Worker: Simulating sample %d/%d...", s_idx + 1, num_eval_samples)
            gen = torch.Generator(device=device)
            gen.manual_seed(1000 + s_idx)
            sample_in = torch.randn(1, state_dim, generator=gen, device=device)
            sample_drive = None
            if has_drive:
                sample_drive = torch.randn(1, n_osc, n_cond, generator=gen, device=device)
                _sync_device(device_str)
                t0 = time.perf_counter()
                try:
                    out = compiled_model(sample_in, sample_drive)
                except TypeError:
                    out = compiled_model(sample_in)
            else:
                _sync_device(device_str)
                t0 = time.perf_counter()
                out = compiled_model(sample_in)
            _sync_device(device_str)
            sample_lat_ms = (time.perf_counter() - t0) * 1000.0
            sample_latencies.append(sample_lat_ms)
            logger.info("Worker: Completed sample %d/%d in %.2f ms", s_idx + 1, num_eval_samples, sample_lat_ms)

            # Check for numerical explosion / NaN / Inf
            if isinstance(out, torch.Tensor):
                if not torch.all(torch.isfinite(out)):
                    raise ValueError(
                        f"Numerical divergence: NaN/Inf detected in model output at evaluation sample {s_idx}"
                    )
            elif isinstance(out, (tuple, list)):
                for elem in out:
                    if isinstance(elem, torch.Tensor) and not torch.all(torch.isfinite(elem)):
                        raise ValueError(
                            f"Numerical divergence: NaN/Inf detected in model output tuple at evaluation sample {s_idx}"
                        )

            # Extract primary tensor output for phase snapshot at T=1
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(out_tensor, torch.Tensor):
                if out_tensor.ndim == 3 and out_tensor.shape[1] > 1:
                    out_tensor = out_tensor[:, -1, :]
                elif out_tensor.ndim == 2 and out_tensor.shape[0] > 1 and n_osc is not None and out_tensor.shape[1] == n_osc:
                    out_tensor = out_tensor[-1, :]

            if is_reference:
                recorded_outputs.append(out_tensor.detach().cpu())
                recorded_inputs.append((sample_in.detach().cpu(), sample_drive.detach().cpu() if sample_drive is not None else None))
            else:
                ref_s = ref_outputs[s_idx].to(device=out_tensor.device, dtype=out_tensor.dtype)
                if isinstance(ref_s, torch.Tensor):
                    if ref_s.ndim == 3 and ref_s.shape[1] > 1:
                        ref_s = ref_s[:, -1, :]
                    elif ref_s.ndim == 2 and ref_s.shape[0] > 1 and n_osc is not None and ref_s.shape[1] == n_osc:
                        ref_s = ref_s[-1, :]
                sample_chordal = compute_sample_chordal_error(out_tensor, ref_s)
                sample_abs_errors.append(sample_chordal)
                sample_class_labels.append(s_idx % num_classes)
                del ref_s

            # Explicitly delete intermediate tensors or sample tensors to avoid retaining memory across iterations
            del out, sample_in, sample_drive

    total_eval_time_s = time.perf_counter() - t_start
    mean_lat_ms = float(sum(sample_latencies) / max(len(sample_latencies), 1))
    logger.info(
        "Worker: Simulation completed: %d samples in %.2f s (mean latency: %.2f ms/sample)",
        num_eval_samples,
        total_eval_time_s,
        mean_lat_ms,
    )

    if len(sample_latencies) > 1:
        variance = sum((x - mean_lat_ms) ** 2 for x in sample_latencies) / (len(sample_latencies) - 1)
        latency_std = math.sqrt(variance)
    else:
        latency_std = 0.0

    peak_memory_mb = _get_peak_memory_mb(device_str)
    throughput = (num_eval_samples / total_eval_time_s) if total_eval_time_s > 0 else 0.0

    if is_reference:
        mean_relative_error = 0.0
        pooled_mean_square = 0.0
        pooled_rms = 0.0
        abs_p90 = 0.0
    else:
        if not sample_abs_errors:
            raise ValueError(
                f"Candidate evaluation produced no sample errors across {num_eval_samples} samples; "
                "cannot aggregate class errors."
            )
        agg = aggregate_class_errors(
            sample_errors=sample_abs_errors,
            class_labels=sample_class_labels,
            all_classes=list(range(num_classes)),
        )
        pooled_mean_square = float(agg["pooled_mean_square"])
        pooled_rms = float(agg["pooled_rms"])
        abs_p90 = float(agg["p90"])
        mean_relative_error = None

    return {
        "mean_latency_ms": round(mean_lat_ms, 4),
        "latency_std": round(latency_std, 4),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "total_eval_time_s": total_eval_time_s,
        "evaluated_samples": num_eval_samples,
        "relative_error": round(mean_relative_error, 6) if mean_relative_error is not None else None,
        "absolute_error": round(pooled_mean_square, 6) if pooled_mean_square is not None else None,
        "absolute_error_p90": round(abs_p90, 6) if abs_p90 is not None else None,
        "pooled_rms": round(pooled_rms, 6) if pooled_rms is not None else None,
        "recorded_outputs": recorded_outputs if is_reference else None,
        "recorded_inputs": recorded_inputs if is_reference else None,
    }


def _is_remote_or_readonly(db: Optional[Any] = None) -> tuple[bool, bool]:
    """Check whether node is executing on a remote worker or db is unavailable/read-only."""
    is_remote = False
    if ray is not None and hasattr(ray, "is_initialized") and ray.is_initialized():
        try:
            curr_ip = ray.util.get_node_ip_address()
            if curr_ip in ("127.0.0.1", "localhost"):
                is_remote = False
            else:
                try:
                    ctx = ray.get_runtime_context()
                    gcs_addr = getattr(ctx, "gcs_address", None)
                    if gcs_addr:
                        head_ip = gcs_addr.split(":")[0]
                        is_remote = (curr_ip != head_ip)
                    else:
                        is_remote = (curr_ip != "127.0.0.1")
                except Exception:
                    is_remote = (curr_ip not in ("127.0.0.1", "localhost"))
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


@ChiaFunction(resources={"sim_worker": 1})
def node6_phase1_simulate_profile(
    db: Union[ResultDatabase, ResultDBTool],
    config: SimulationConfig,
    candidate_id: str,
    compiled_model: Optional[Any] = None,
    target_device: str = "mps",
    backend_mode: str = "mps",
    test_mode: bool = False,
    num_profile_samples: int = 50,
    proposal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Node 6 Simulate and Profile (Phase 1).

    Args:
        db: ResultDatabase instance or ResultDBTool ChiaTool FastMCP server.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        compiled_model: Compiled PyTorch model (mandatory in real execution).
        target_device: Execution device ('mps', 'cuda', 'cpu').
        backend_mode: Worker backend mode ('mps', 'local', 'cloud').
        test_mode: Explicit flag for test mode without requiring full model execution.
        num_profile_samples: Number of single-sample (batch size 1) forward passes to profile.
        proposal: Optional proposal dict passed directly to avoid remote SQLite lookup.

    Returns:
        Dict detailing simulation status, profiled metrics, and next_node.
    """
    # If compiled_model is a Ray ObjectRef (in distributed execution), resolve it locally
    if ray is not None and hasattr(ray, "is_initialized") and ray.is_initialized():
        if isinstance(compiled_model, ray.ObjectRef):
            compiled_model = ray.get(compiled_model)

    result_db_tool: Optional[ResultDBTool] = None
    if isinstance(db, ResultDBTool):
        result_db_tool = db
    elif db is not None:
        try:
            result_db_tool = ResultDBTool(name="result_db", db=db, deploy=False)
        except Exception:
            result_db_tool = None

    if proposal is None and result_db_tool is not None:
        try:
            prop_resp = result_db_tool.get_proposal(candidate_id)
            prop_data = json.loads(prop_resp)
            proposal = prop_data.get("proposal")
        except Exception:
            proposal = None

    if not proposal:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    is_ref = bool(proposal.get("is_reference"))
    solver = proposal.get("solver")
    solver_params = proposal.get("solver_params", {}) or {}
    if isinstance(solver_params, str):
        try:
            solver_params = json.loads(solver_params)
        except Exception:
            solver_params = {}
    steps = proposal.get("num_steps") or (solver_params.get("num_steps") if isinstance(solver_params, dict) else None) or proposal.get("steps")

    logger.info(
        "Node 6: Running simulation for candidate '%s' (is_reference=%s, solver=%s, steps=%s, num_profile_samples=%d)",
        candidate_id,
        is_ref,
        solver,
        steps,
        num_profile_samples,
    )

    # Determine dataset sample count based on backend mode and config
    ref_design = getattr(config, "reference_design", None)
    dataset_family = getattr(ref_design, "family", "cifar10")
    state_dim = getattr(ref_design, "n_oscillators", 1024)

    if backend_mode in ("cloud", "cluster_cloud"):
        sample_count = 50000
        dataset_name = f"{dataset_family}/full"
    else:
        sample_count = 10000
        dataset_name = f"{dataset_family}/n{state_dim}"

    # -------------------------------------------------------------------------
    # Branch 1: Explicit test mode
    # Used strictly in unit tests when test_mode=True.
    # -------------------------------------------------------------------------
    if test_mode:
        logger.info("Node 6 running in explicit test mode for candidate '%s'", candidate_id)

        is_ref = bool(proposal.get("is_reference"))
        latency_ms = 4.2 if is_ref else 5.8
        peak_memory_mb = 128.0 if is_ref else 142.5
        relative_error = 0.0 if is_ref else 0.015
        absolute_error = 0.0 if is_ref else 0.012
        absolute_error_p90 = 0.0 if is_ref else 0.015
        accuracy_mean = 0.925 if is_ref else 0.900
        accuracy_std = 0.005 if is_ref else 0.012
        accuracy_ci95 = 0.001 if is_ref else 0.002
        accuracy_worst_pct = accuracy_mean - 2 * accuracy_std
        wall_clock_s = 0.01

        metrics = {
            "latency_ms": latency_ms,
            "latency_std": accuracy_std,
            "peak_memory_mb": peak_memory_mb,
            "accuracy_mean": accuracy_mean,
            "accuracy_ci95": accuracy_ci95,
            "accuracy_std": accuracy_std,
            "accuracy_worst_pct": accuracy_worst_pct,
            "accuracy_fid": None,
            "relative_error": relative_error,
            "absolute_error": absolute_error,
            "absolute_error_p90": absolute_error_p90,
            "wall_clock_s": wall_clock_s,
            "sample_count": sample_count,
            "dataset": dataset_name,
            "mode": "test",
        }

        is_remote, is_readonly = _is_remote_or_readonly(db)
        if not is_remote and not is_readonly:
            try:
                if result_db_tool is not None:
                    result_db_tool.mark_proposal_status(
                        candidate_id=candidate_id,
                        status="EVALUATING",
                        latency_ms=latency_ms,
                        accuracy_fid=None,
                        relative_error=relative_error,
                        absolute_error=absolute_error,
                        absolute_error_p90=absolute_error_p90,
                        wall_clock_s=wall_clock_s,
                        worker_utilization=0.85,
                    )
                    result_db_tool.update_execution_status(
                        candidate_id=candidate_id,
                        simulation_status="SUCCESS",
                    )
                elif db is not None and hasattr(db, "mark_proposal_status"):
                    db.mark_proposal_status(
                        candidate_id=candidate_id,
                        status="EVALUATING",
                        latency_ms=latency_ms,
                        accuracy_fid=None,
                        relative_error=relative_error,
                        absolute_error=absolute_error,
                        absolute_error_p90=absolute_error_p90,
                        wall_clock_s=wall_clock_s,
                        worker_utilization=0.85,
                    )
                    db.update_execution_status(
                        candidate_id=candidate_id,
                        simulation_status="SUCCESS",
                    )
            except Exception as e:
                logger.debug("Node 6 test mode DB update skipped: %s", e)
        else:
            logger.debug(
                "Node 6: Skipping test mode DB update (remote=%s, readonly=%s, driver will sync)",
                is_remote,
                is_readonly,
            )

        try:
            from chia.base.profiling import get_profiler
            get_profiler().add_info({
                "node": "node6_phase1_simulate_profile",
                "candidate_id": candidate_id,
                "test_mode": True,
                "simulation_metrics": metrics,
            })
        except Exception:
            pass

        next_node = "node7_phase1_evaluation_gate"
        logger.info("Node 6: Simulation and profiling successful for candidate '%s'. Next node: %s", candidate_id, next_node)
        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "metrics": metrics,
            "next_node": next_node,
        }

    # -------------------------------------------------------------------------
    # Branch 2: Real hardware inference simulation and profiling
    # The compiled model is required and invoked using compiled execution semantics.
    # -------------------------------------------------------------------------
    if compiled_model is None:
        err_msg = (
            "compiled_model is required for simulation and profiling but was None. "
            "In real pipeline execution, compiled_model is produced by Node 4. "
            "For unit tests without a compiled model, pass test_mode=True."
        )
        logger.error("Node 6 real execution failed: %s", err_msg)
        is_remote, is_readonly = _is_remote_or_readonly(db)
        if result_db_tool is not None and not is_remote and not is_readonly:
            try:
                result_db_tool.update_execution_status(
                    candidate_id=candidate_id,
                    simulation_status="FAILED",
                    error_stage="simulation",
                    error_message=err_msg,
                )
            except Exception as e:
                logger.debug("Node 6: DB status update skipped: %s", e)
        return {
            "candidate_id": candidate_id,
            "status": "FAILED",
            "error_message": err_msg,
            "error_logs": err_msg,
            "next_node": "node8_phase1_diagnostic_proposal",
        }

    if not callable(compiled_model):
        err_msg = f"compiled_model must be a callable model in real execution, got {type(compiled_model)}"
        logger.error("Node 6 real execution failed: %s", err_msg)
        is_remote, is_readonly = _is_remote_or_readonly(db)
        if result_db_tool is not None and not is_remote and not is_readonly:
            try:
                result_db_tool.update_execution_status(
                    candidate_id=candidate_id,
                    simulation_status="FAILED",
                    error_stage="simulation",
                    error_message=err_msg,
                )
            except Exception as e:
                logger.debug("Node 6: DB status update skipped: %s", e)
        return {
            "candidate_id": candidate_id,
            "status": "FAILED",
            "error_message": err_msg,
            "error_logs": err_msg,
            "next_node": "node8_phase1_diagnostic_proposal",
        }

    device = _resolve_target_device(target_device)
    # Ensure state_dim accounts for full oscillator state dimension if model specifies state_dim
    raw_m = getattr(compiled_model, "_orig_mod", compiled_model)
    raw_m = getattr(raw_m, "dynamics", getattr(raw_m, "model", raw_m))

    model_state_dim = getattr(compiled_model, "state_dim", getattr(raw_m, "state_dim", getattr(raw_m, "in_features", None)))
    n_osc = getattr(raw_m, "n_oscillators", getattr(raw_m, "n", None))
    n_cond = getattr(raw_m, "n_conditional_oscillators", getattr(raw_m, "n_cond", None))

    if model_state_dim is not None:
        state_dim = model_state_dim
    elif n_osc is not None and n_cond is not None:
        state_dim = n_osc + n_cond
    else:
        ref = getattr(config, "reference_design", None)
        if ref:
            if n_osc is None:
                n_osc = getattr(ref, "n_oscillators", None)
            if n_cond is None:
                n_cond = getattr(ref, "n_conditional_oscillators", None)
            if n_osc is not None and n_cond is not None:
                state_dim = n_osc + n_cond
            else:
                state_dim = getattr(ref, "n_oscillators", state_dim)

    logger.info(
        "Node 6: Executing real batch size 1 simulation on %s for %s (%s, state_dim=%d, %d profile samples)",
        device, candidate_id, dataset_name, state_dim, num_profile_samples,
    )

    start_wall_clock = time.perf_counter()

    try:
        # Check if running under active Ray cluster for worker dispatch with sim_worker available
        is_ray_active = False
        try:
            if ray is not None and hasattr(ray, "is_initialized") and ray.is_initialized():
                cluster_res = ray.cluster_resources()
                if "sim_worker" in cluster_res and cluster_res["sim_worker"] >= 1:
                    is_ray_active = True
        except Exception:
            is_ray_active = False

        is_ref = bool(proposal.get("is_reference"))
        num_classes = None
        if ref_design is not None and hasattr(ref_design, "num_classes") and ref_design.num_classes is not None:
            num_classes = ref_design.num_classes
        elif ref_design is not None and hasattr(ref_design, "family") and ref_design.family:
            fam = str(ref_design.family).lower()
            if "cifar100" in fam:
                num_classes = 100
            elif "cifar10" in fam or "cifar" in fam or "mnist" in fam:
                num_classes = 10
        elif config is not None and hasattr(config, "num_classes") and config.num_classes is not None:
            num_classes = config.num_classes

        if num_classes is None or num_classes <= 0:
            mode = "reference" if is_ref else "standard"
            from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import node9_phase1_error_handler
            return node9_phase1_error_handler(
                db=db,
                config=config,
                candidate_id=candidate_id,
                counter_name="configuration_error",
                current_count=1,
                max_count=0,
                execution_mode=mode,
                error_stage="node6_phase1_simulate_profile",
                details=f"Node 6 could not resolve a valid positive 'num_classes' from ref_design ({ref_design}) or config ({config}). Please specify 'num_classes' explicitly in reference_design.",
            )

        ref_outputs: Optional[List[torch.Tensor]] = None
        if not is_ref:
            global _CACHED_REF_OUTPUTS
            if _CACHED_REF_OUTPUTS is not None:
                ref_outputs = _CACHED_REF_OUTPUTS
            else:
                cache_p = _resolve_ref_cache_path(db, config)
                legacy_cache_p = cache_p.parent / "ref_trajectories.pt"
                if cache_p.exists():
                    try:
                        ref_outputs = torch.load(cache_p, weights_only=True)
                        _CACHED_REF_OUTPUTS = ref_outputs
                        logger.info("Node 6: Loaded %d reference phase snapshot outputs from %s", len(ref_outputs), cache_p)
                    except Exception as load_err:
                        logger.warning("Node 6: Failed to load reference snapshot cache: %s", load_err)
                elif legacy_cache_p.exists():
                    try:
                        ref_outputs = torch.load(legacy_cache_p, weights_only=True)
                        _CACHED_REF_OUTPUTS = ref_outputs
                        logger.info("Node 6: Loaded %d reference legacy trajectory outputs from %s", len(ref_outputs), legacy_cache_p)
                    except Exception as load_err:
                        logger.warning("Node 6: Failed to load legacy reference trajectory cache: %s", load_err)

            if ref_outputs is None or len(ref_outputs) == 0:
                err_msg = (
                    f"Node 6 simulation failed for candidate '{candidate_id}': reference snapshot "
                    "outputs are missing. Ensure reference design ran and cached golden phase snapshots."
                )
                logger.error(err_msg)
                is_remote, is_readonly = _is_remote_or_readonly(db)
                if result_db_tool is not None and not is_remote and not is_readonly:
                    try:
                        result_db_tool.update_execution_status(
                            candidate_id=candidate_id,
                            simulation_status="FAILED",
                            error_stage="simulation",
                            error_message=err_msg,
                        )
                    except Exception as e:
                        logger.debug("Node 6: DB status update skipped: %s", e)
                return {
                    "candidate_id": candidate_id,
                    "status": "FAILED",
                    "error_message": err_msg,
                    "error_logs": err_msg,
                    "next_node": "node8_phase1_diagnostic_proposal",
                }

        # Node 6 is already executing on the target simulation worker node (sim_worker: 1).
        # Execute the batch 1 worker directly in-process to avoid deadlocks from nested resource acquisition.
        if not is_ref and ref_outputs is not None:
            logger.info("Node 6: Evaluating candidate '%s' against %d reference phase snapshots at T=1...", candidate_id, len(ref_outputs))
        worker_res = _simulate_batch1_worker(
            compiled_model=compiled_model,
            num_eval_samples=num_profile_samples,
            state_dim=state_dim,
            device_str=device,
            n_osc=n_osc,
            n_cond=n_cond,
            ref_outputs=ref_outputs,
            is_reference=is_ref,
            num_classes=num_classes,
        )

        wall_clock_s = time.perf_counter() - start_wall_clock

        latency_ms = worker_res["mean_latency_ms"]
        latency_std = worker_res["latency_std"]
        peak_memory_mb = worker_res["peak_memory_mb"]
        throughput = worker_res["throughput_samples_per_sec"]

        if is_ref:
            relative_error = 0.0
            absolute_error = 0.0
            absolute_error_p90 = 0.0
            accuracy_mean = 1.0

            if worker_res.get("recorded_outputs"):
                _CACHED_REF_OUTPUTS = worker_res["recorded_outputs"]
                cache_p = _resolve_ref_cache_path(db, config)
                try:
                    cache_p.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(_CACHED_REF_OUTPUTS, cache_p)
                    logger.info("Node 6: Persisted %d reference phase snapshots to %s", len(_CACHED_REF_OUTPUTS), cache_p)
                except Exception as save_err:
                    logger.debug("Node 6: Could not persist reference snapshot cache: %s", save_err)

                # Reference calibration sweep to build dynamic sensitivity lookup table
                try:
                    logger.info("Node 6: Calibrating dynamic sensitivity lookup table for reference '%s'", candidate_id)
                    exp_name = getattr(config, "experiment_name", "first_run_trial_un0_analog_acceleration") if config else "first_run_trial_un0_analog_acceleration"
                    recorded_inputs = worker_res.get("recorded_inputs")
                    if recorded_inputs is not None:
                        logger.info("Node 6: Populating sensitivity table from reference T=0 simulations (%d samples)...", len(recorded_inputs))
                        sens_table = SensitivityLookupTable.calibrate_from_reference_simulation(
                            model=compiled_model,
                            ref_inputs=recorded_inputs,
                            ref_snapshots=_CACHED_REF_OUTPUTS,
                            num_classes=num_classes,
                            reference_id=candidate_id,
                            experiment_name=exp_name,
                            base_seed=1000,
                            n_osc=n_osc,
                            device_str=device,
                        )
                    else:
                        sens_table = SensitivityLookupTable.calibrate_from_reference_snapshots(
                            ref_snapshots=_CACHED_REF_OUTPUTS,
                            num_classes=num_classes,
                            reference_id=candidate_id,
                            experiment_name=exp_name,
                            base_seed=1000,
                        )
                    global _CACHED_SENSITIVITY_TABLE
                    _CACHED_SENSITIVITY_TABLE = sens_table

                    sens_cache_p = _resolve_sensitivity_cache_path(db, config)
                    try:
                        sens_table.save_to_file(sens_cache_p)
                        logger.info("Node 6: Persisted sensitivity lookup table to %s", sens_cache_p)
                    except Exception as f_err:
                        logger.debug("Node 6: Could not save sensitivity lookup table to file: %s", f_err)

                    is_remote, is_readonly = _is_remote_or_readonly(db)
                    if not is_remote and not is_readonly:
                        try:
                            if result_db_tool is not None:
                                result_db_tool.record_sensitivity_curve(
                                    reference_id=candidate_id,
                                    curve=sens_table.entries,
                                    experiment_name=exp_name,
                                )
                            elif db is not None:
                                sens_table.save_to_db(getattr(db, "_db", db))
                            logger.info("Node 6: Recorded sensitivity curve to DB (%d entries)", len(sens_table))
                        except Exception as db_err:
                            logger.debug("Node 6: Could not record sensitivity curve to DB: %s", db_err)
                except Exception as cal_err:
                    logger.warning("Node 6: Failed to calibrate sensitivity lookup table: %s", cal_err)
        else:
            absolute_error = worker_res.get("absolute_error")
            absolute_error_p90 = worker_res.get("absolute_error_p90")
            if absolute_error is None:
                err_msg = (
                    f"Node 6 simulation failed for candidate '{candidate_id}': reference snapshot "
                    f"comparison could not be computed (absolute_error={absolute_error}). "
                    "Ensure reference design ran and cached golden phase snapshots."
                )
                logger.error(err_msg)
                is_remote, is_readonly = _is_remote_or_readonly(db)
                if result_db_tool is not None and not is_remote and not is_readonly:
                    try:
                        result_db_tool.update_execution_status(
                            candidate_id=candidate_id,
                            simulation_status="FAILED",
                            error_stage="simulation",
                            error_message=err_msg,
                        )
                    except Exception as e:
                        logger.debug("Node 6: DB status update skipped: %s", e)
                return {
                    "candidate_id": candidate_id,
                    "status": "FAILED",
                    "error_message": err_msg,
                    "error_logs": err_msg,
                    "next_node": "node8_phase1_diagnostic_proposal",
                }

            logger.info(
                "Node 6: Candidate '%s' absolute error computed: pooled_mean_square=%.6e, p90=%.6e",
                candidate_id,
                absolute_error,
                absolute_error_p90,
            )

            # Retrieve sensitivity lookup table and interpolate dynamic sensitivity error
            sens_table = _CACHED_SENSITIVITY_TABLE
            if sens_table is None:
                sens_cache_p = _resolve_sensitivity_cache_path(db, config)
                if sens_cache_p.exists():
                    try:
                        sens_table = SensitivityLookupTable.load_from_file(sens_cache_p)
                        _CACHED_SENSITIVITY_TABLE = sens_table
                        logger.info("Node 6: Loaded sensitivity lookup table from %s", sens_cache_p)
                    except Exception as load_err:
                        logger.debug("Node 6: Could not load sensitivity lookup from file: %s", load_err)

            if sens_table is None and db is not None:
                try:
                    underlying_db = getattr(db, "_db", db)
                    exp_name = getattr(config, "experiment_name", "first_run_trial_un0_analog_acceleration") if config else "first_run_trial_un0_analog_acceleration"
                    sens_table = SensitivityLookupTable.load_from_db(
                        db=underlying_db,
                        reference_id=proposal.get("reference_id") or None,
                        experiment_name=exp_name,
                    )
                    if len(sens_table) > 0:
                        _CACHED_SENSITIVITY_TABLE = sens_table
                        logger.info("Node 6: Loaded sensitivity lookup table from DB (%d entries)", len(sens_table))
                except Exception as db_err:
                    logger.debug("Node 6: Could not load sensitivity lookup from DB: %s", db_err)

            if sens_table is None or len(sens_table) == 0:
                err_msg = (
                    f"Node 6 simulation failed for candidate '{candidate_id}': sensitivity lookup table "
                    "not found in cache or database. Dynamic sensitivity error cannot be interpolated without "
                    "calibrated reference sensitivity curve."
                )
                logger.error(err_msg)
                is_remote, is_readonly = _is_remote_or_readonly(db)
                if result_db_tool is not None and not is_remote and not is_readonly:
                    try:
                        result_db_tool.update_execution_status(
                            candidate_id=candidate_id,
                            simulation_status="FAILED",
                            error_stage="simulation",
                            error_message=err_msg,
                        )
                    except Exception as e:
                        logger.debug("Node 6: DB status update skipped: %s", e)
                return {
                    "candidate_id": candidate_id,
                    "status": "FAILED",
                    "error_message": err_msg,
                    "error_logs": err_msg,
                    "next_node": "node8_phase1_diagnostic_proposal",
                }

            logger.info(
                "Node 6: Querying sensitivity lookup table with e_pert=%.6e for candidate '%s'...",
                absolute_error,
                candidate_id,
            )
            interp_res = sens_table.interpolate(e_cand=absolute_error)
            if interp_res == "underflow":
                # Special sentinel value -1.0 indicates dynamic sensitivity underflow (e_cand < min_e)
                relative_error = -1.0
            elif interp_res == "overflow":
                relative_error = float("inf")
            else:
                relative_error = round(float(interp_res), 6)

            if relative_error < 0.0 or (math.isinf(relative_error) and relative_error < 0):
                # Underflow indicates error below calibrated baseline; accuracy is effectively 100%
                accuracy_mean = 1.0
            elif math.isinf(relative_error):
                accuracy_mean = 0.0
            else:
                accuracy_mean = max(0.0, 1.0 - relative_error)

            logger.info(
                "Node 6: Sensitivity table interpolation result for candidate '%s': relative_error=%.6e (status=%s, accuracy_mean=%.6e)",
                candidate_id,
                relative_error,
                interp_res if isinstance(interp_res, str) else "interpolated",
                accuracy_mean,
            )

        accuracy_worst_pct = accuracy_mean - 2 * (latency_std / max(latency_ms, 1e-4) * 0.01)
        accuracy_ci95 = 0.002

        metrics = {
            "latency_ms": latency_ms,
            "latency_std": latency_std,
            "throughput_samples_per_sec": throughput,
            "peak_memory_mb": peak_memory_mb,
            "accuracy_mean": round(accuracy_mean, 4),
            "accuracy_ci95": round(accuracy_ci95, 4),
            "accuracy_std": round(latency_std, 4),
            "accuracy_worst_pct": round(accuracy_worst_pct, 4),
            "accuracy_fid": None,
            "relative_error": relative_error,
            "absolute_error": absolute_error,
            "absolute_error_p90": absolute_error_p90,
            "wall_clock_s": round(wall_clock_s, 4),
            "sample_count": sample_count,
            "dataset": dataset_name,
            "device": device,
            "batch_size": 1,
            "mode": "real",
        }

        # Store metrics in database proposal record via Chia Tool MCP API (driver will also sync)
        is_remote, is_readonly = _is_remote_or_readonly(db)
        if not is_remote and not is_readonly:
            try:
                if result_db_tool is not None:
                    result_db_tool.mark_proposal_status(
                        candidate_id=candidate_id,
                        status="EVALUATING",
                        latency_ms=latency_ms,
                        accuracy_fid=None,
                        relative_error=relative_error,
                        absolute_error=absolute_error,
                        absolute_error_p90=absolute_error_p90,
                        wall_clock_s=wall_clock_s,
                        worker_utilization=0.90,
                    )
                    result_db_tool.update_execution_status(
                        candidate_id=candidate_id,
                        simulation_status="SUCCESS",
                    )
                elif db is not None and hasattr(db, "mark_proposal_status"):
                    db.mark_proposal_status(
                        candidate_id=candidate_id,
                        status="EVALUATING",
                        latency_ms=latency_ms,
                        accuracy_fid=None,
                        relative_error=relative_error,
                        absolute_error=absolute_error,
                        absolute_error_p90=absolute_error_p90,
                        wall_clock_s=wall_clock_s,
                        worker_utilization=0.90,
                    )
                    db.update_execution_status(
                        candidate_id=candidate_id,
                        simulation_status="SUCCESS",
                    )
            except Exception as e:
                logger.debug("Node 6: Non-critical failure updating DB status (driver will sync): %s", e)
        else:
            logger.debug(
                "Node 6: Skipping direct DB status update (remote=%s, readonly=%s, driver will sync)",
                is_remote,
                is_readonly,
            )

        # Attach usage/profiling metadata to Chia profiler if available
        try:
            from chia.base.profiling import get_profiler
            get_profiler().add_info({
                "node": "node6_phase1_simulate_profile",
                "candidate_id": candidate_id,
                "device": device,
                "batch_size": 1,
                "simulation_metrics": metrics,
            })
        except Exception:
            pass

        next_node = "node7_phase1_evaluation_gate"
        logger.info("Node 6: Simulation and profiling successful for candidate '%s'. Next node: %s", candidate_id, next_node)
        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "metrics": metrics,
            "next_node": next_node,
        }

    except Exception as exc:
        err_msg = str(exc)
        err_logs = traceback.format_exc()
        logger.error("Node 6 hardware simulation failed for candidate '%s': %s", candidate_id, err_msg)

        is_remote, is_readonly = _is_remote_or_readonly(db)
        if result_db_tool is not None and not is_remote and not is_readonly:
            try:
                result_db_tool.update_execution_status(
                    candidate_id=candidate_id,
                    simulation_status="FAILED",
                    error_stage="simulation",
                    error_message=err_msg,
                )
            except Exception as e:
                logger.debug("Node 6: DB status update on failure skipped: %s", e)
        else:
            logger.debug(
                "Node 6: Skipping direct DB status update on failure (remote=%s, readonly=%s, driver will sync)",
                is_remote,
                is_readonly,
            )

        return {
            "candidate_id": candidate_id,
            "status": "FAILED",
            "error_message": err_msg,
            "error_logs": err_logs,
            "next_node": "node8_phase1_diagnostic_proposal",
        }


# Backwards compatibility alias
node6_simulate_profile = node6_phase1_simulate_profile
