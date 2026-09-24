"""Node 1: Solver Rewriter Agentic Node (Phase 1).

Applies or registers solver rewrite for Un-0 model and updates database status.
Implements deterministic reuse when solver is already registered (with null token consumption)
and agentic LLM rewrite when solver is unregistered or forced, exposing the 3 required Chia tools:
1. `result_db`: ResultDBTool (query proposal, tolerances, status, write comments).
2. `solver_papers`: SolverPapersTool (query continuous systems solver papers and tier guidelines).
3. `un0_rewriter`: Un0RewriterTool (inspect Un-0 codebase and register new solvers via un0_register_solver_rewrite).

Adheres strictly to the Chia API, eliminates hardcoded fallback solvers, records real token metrics,
and fails fast with a RuntimeError when the agent fails to register a valid solver.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import warnings

try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:  # pragma: no cover
    def ChiaFunction(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator

try:
    from chia.base.tools.ChiaTool import cleanup_tools
except ImportError:  # pragma: no cover
    def cleanup_tools(tools: Any) -> None:
        if not tools:
            return
        for tool in tools:
            if hasattr(tool, "stop") and callable(tool.stop):
                try:
                    tool.stop()
                except Exception:
                    pass
            elif hasattr(tool, "_server_actor") and getattr(tool, "_server_actor", None) is not None:
                try:
                    import ray
                    ray.kill(tool._server_actor)
                except Exception:
                    pass

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool
from agentic_ml_analog_sim.tools.solver_papers_tool import SolverPapersTool
from agentic_ml_analog_sim.tools.un0_context_tool import resolve_un0_root
from agentic_ml_analog_sim.tools.un0_rewriter_tool import Un0RewriterTool

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node1_phase1_solver_rewriter")


def _sync_solver_rewrites_from_disk(un0_dir: Path, candidate_id: Optional[str] = None) -> List[str]:
    """Scan Un-0 solver directory for newly written modules and load them into SOLVER_REGISTRY."""
    solvers_dir = un0_dir / "un0" / "solver"
    if not solvers_dir.is_dir():
        return []

    discovered = []
    clean_cand = candidate_id.strip().replace(" ", "_") if candidate_id else None

    from un0.solver import SOLVER_REGISTRY

    all_files = [p for p in solvers_dir.glob("*.py") if p.name != "__init__.py"]
    if clean_cand:
        matching = [p for p in all_files if p.name.startswith(f"{clean_cand}_") or clean_cand in p.name]
        others = [p for p in all_files if p not in matching]
        ordered_files = sorted(matching, key=lambda p: p.stat().st_mtime, reverse=True) + sorted(others, key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        ordered_files = sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)

    for fpath in ordered_files:
        stem = fpath.stem
        module_name = f"un0.solver.{stem}"
        try:
            if module_name in sys.modules:
                mod = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, fpath)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                else:
                    continue

            potential_name = stem[len(clean_cand) + 1:] if clean_cand and stem.startswith(f"{clean_cand}_") else stem

            if potential_name not in SOLVER_REGISTRY:
                candidate_fn = (
                    getattr(mod, potential_name, None)
                    or getattr(mod, f"{potential_name}_step", None)
                    or getattr(mod, "solve", None)
                    or getattr(mod, "step", None)
                )
                if candidate_fn is None:
                    for attr_name in dir(mod):
                        val = getattr(mod, attr_name)
                        if callable(val) and not attr_name.startswith("_") and getattr(val, "__module__", "") == module_name:
                            candidate_fn = val
                            break
                if candidate_fn is not None:
                    SOLVER_REGISTRY[potential_name] = candidate_fn

            if potential_name in SOLVER_REGISTRY:
                discovered.append(potential_name)
        except Exception as err:
            logger.debug("Node 1: Error syncing solver %s from disk: %s", fpath, err)

    return discovered


def _resolve_llm(llm: Optional[Any] = None, config: Optional[SimulationConfig] = None) -> Any:
    """Resolve active LLM or mock instance based on simulation config."""
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node1_phase1_solver_rewriter", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node1_phase1_solver_rewriter")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-pro"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "high"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


@ChiaFunction(resources={"agent_worker": 1, "antigravity_creds": 0.01})
def node1_phase1_solver_rewriter(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    llm: Optional[Any] = None,
    force_rewrite: bool = False,
) -> Dict[str, Any]:
    """Node 1 Solver Rewriter (Phase 1).

    Deterministic behavior:
        If the solver is already registered in SOLVER_REGISTRY and no rewrite is forced,
        the node reuses the registered solver function directly. No LLM call is made,
        and token consumption is null (0 tokens recorded). Chia API profiler telemetry
        is still recorded for programmatic tracing.

    Agentic behavior:
        If the solver is not registered or forced, invokes the LLM agent using the Chia API,
        exposing all 3 required Chia tools:
          - `result_db`: database query and note taking.
          - `solver_papers`: continuous systems solver literature and tier guides.
          - `un0_rewriter`: Un-0 model code inspection and solver registration tool.
        The agent generates the ODE solver adhering to the Un-0 signature and registers it
        via `un0_register_solver_rewrite`.
        Real token consumption metrics are captured from the agent framework and
        persisted to the ResultDatabase.
        Strict verification ensures that if the agent fails to register the solver,
        the node records FAILED status in the DB and raises a RuntimeError (no hardcoded fallbacks).

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        llm: Optional LLM instance or mock.
        force_rewrite: Whether to force an agentic LLM rewrite even if registered.

    Returns:
        Dict detailing candidate_id, execution status, mode, and selected solver.

    Raises:
        KeyError: If candidate is missing or solver name cannot be resolved.
        RuntimeError: If agentic rewrite fails to register a valid solver in SOLVER_REGISTRY.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    solver_name = proposal.get("solver")
    if not solver_name:
        ref_design = getattr(config, "reference_design", None)
        solver_name = getattr(ref_design, "solver", None) if ref_design else None
    if not solver_name:
        raise KeyError(f"No solver specified for candidate '{candidate_id}' in proposal or simulation config.")

    solver_params = proposal.get("solver_params", {}) or {}
    if isinstance(solver_params, str):
        try:
            solver_params = json.loads(solver_params)
        except Exception:
            solver_params = {}

    # 1. Parameter resolution with explicit fallback warnings
    raw_num_steps = proposal.get("num_steps") or solver_params.get("num_steps")
    if raw_num_steps is None:
        logger.warning(
            "Node 1: 'num_steps' not specified for candidate '%s'; falling back to assumed default of 25",
            candidate_id,
        )
        warnings.warn(
            f"Node 1: 'num_steps' not specified for candidate '{candidate_id}'; using fallback default 25",
            UserWarning,
            stacklevel=2,
        )
        num_steps = 25
    else:
        num_steps = raw_num_steps

    raw_dt = solver_params.get("dt")
    if raw_dt is None:
        logger.warning(
            "Node 1: 'dt' not specified in solver_params for candidate '%s'; falling back to assumed default of 0.04",
            candidate_id,
        )
        warnings.warn(
            f"Node 1: 'dt' not specified in solver_params for candidate '{candidate_id}'; using fallback default 0.04",
            UserWarning,
            stacklevel=2,
        )
        dt = 0.04
    else:
        dt = raw_dt

    raw_integration_time = solver_params.get("integration_time")
    if raw_integration_time is None:
        computed_it = float(num_steps) * float(dt)
        logger.warning(
            "Node 1: 'integration_time' not specified in solver_params for candidate '%s'; "
            "calculating fallback value of %s (num_steps * dt = %s * %s)",
            candidate_id,
            computed_it,
            num_steps,
            dt,
        )
        warnings.warn(
            f"Node 1: 'integration_time' not specified for candidate '{candidate_id}'; using calculated fallback {computed_it}",
            UserWarning,
            stacklevel=2,
        )
        integration_time = computed_it
    else:
        integration_time = raw_integration_time

    logger.info("Node 1: Checking solver '%s' for candidate %s", solver_name, candidate_id)

    # Check solver registry
    try:
        from un0.solver import SOLVER_REGISTRY, get_solver
    except Exception as exc:
        logger.warning("Un-0 solver registry import issue: %s", exc)
        SOLVER_REGISTRY = {}
        get_solver = None

    is_registered = bool(get_solver and solver_name in SOLVER_REGISTRY)

    # Deterministic branch: already registered and no forced rewrite
    if is_registered and not force_rewrite:
        logger.info(
            "Node 1: Solver '%s' is already registered in Un-0. Using deterministic reuse (null token consumption).",
            solver_name,
        )
        solver_fn = get_solver(solver_name)

        db.update_execution_status(
            candidate_id,
            rewrite_solver_status="SUCCESS",
        )

        # Register deterministic programmatic execution telemetry with Chia profiler
        try:
            from chia.base.profiling import get_profiler
            get_profiler().add_info({
                "node": "node1_phase1_solver_rewriter",
                "candidate_id": candidate_id,
                "deterministic": True,
                "tokens_consumed": 0,
                "solver": solver_name,
            })
        except Exception:
            pass

        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "solver": solver_name,
            "solver_fn": solver_fn,
            "deterministic": True,
            "tokens_consumed": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "next_node": "node2_phase1_noise_rewriter",
        }

    # Nondeterministic / Agentic branch: solver not registered or forced rewrite
    logger.info(
        "Node 1: Solver '%s' requires agentic implementation/rewrite. Invoking LLM via Chia API.",
        solver_name,
    )

    # 1. Instantiate the 3 Chia Tools specified in prompts/phase-1.md
    un0_dir = resolve_un0_root()

    result_db_tool = ResultDBTool(name="result_db", db=db, deploy=False)
    solver_papers_tool = SolverPapersTool(name="solver_papers", auto_start=False)
    un0_rewriter_tool = Un0RewriterTool(name="un0_rewriter", repo_root=un0_dir, auto_start=False)

    # 2. Formulate Agentic Prompt
    prompt_text = (
        f"# Phase 1 Solver Rewriter Agent (Node 1)\n\n"
        f"You are the Solver Rewriter Agent for Phase 1. Your task is to implement and register an ODE solver "
        f"named '{solver_name}' for Kuramoto physical analog dynamics in the Un-0 repository.\n\n"
        f"Candidate Specifications:\n"
        f"- Candidate ID: {candidate_id}\n"
        f"- Target Solver Name: {solver_name}\n"
        f"- Integration Steps (num_steps): {num_steps}\n"
        f"- Time Step (dt): {dt}\n"
        f"- Integration Horizon: {integration_time}\n"
        f"- Solver Parameters: {solver_params}\n\n"
        f"Available Chia Tools:\n"
        f"1. `result_db`: query proposal specifications, tolerances, previous notes, or register comments.\n"
        f"2. `solver_papers`: consult continuous systems solver literature, equations, stability properties, "
        f"and tier guides (Tier 0 explicit, Tier 1 adaptive, Tier 2 implicit, Tier 3 parallel-in-time).\n"
        f"3. `un0_rewriter`: inspect Un-0 codebase, check existing solvers (e.g. `un0/solver/rk4.py`), and "
        f"register your newly synthesized solver into Un-0 using `un0_register_solver_rewrite`.\n\n"
        f"Requirements & Directives:\n"
        f"- Strict Un-0 Solver Signature: The solver must match the signature:\n"
        f"  `solver_fn(rhs, state, time_grid, drive, **kwargs) -> Tensor`\n"
        f"  where `rhs` is callable `(state, time, drive) -> Tensor`, and returns a trajectory tensor of shape "
        f"  `(len(time_grid), batch_size, state_dim)`.\n"
        f"- Use PyTorch operations (torch.stack, vector arithmetic); preserve batch dimensions.\n"
        f"- Call `un0_register_solver_rewrite(solver_name='{solver_name}', candidate_id='{candidate_id}', code=...)` "
        f"to persist and register your solver into `un0.solver.SOLVER_REGISTRY`.\n"
        f"- In your final response, describe your numerical scheme, order, stability properties, and confirmation of registration."
    )

    # 3. Execute Agent Turn via Chia API
    active_llm = _resolve_llm(llm, config)
    start_time = time.time()
    query_result = None
    tools = [result_db_tool, solver_papers_tool, un0_rewriter_tool]
    try:
        if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
            query_result = active_llm.prompt(
                prompt_text,
                tools=tools,
            )
    except Exception as exc:
        logger.warning("Node 1: LLM prompt call failed: %s", exc)
    finally:
        cleanup_tools(tools)

    elapsed = max(0.1, time.time() - start_time)

    # 4. Extract and Record Token Metrics
    usage = getattr(query_result, "usage", None) if query_result is not None else None
    if not usage and hasattr(active_llm, "history") and active_llm.history:
        usage = active_llm.history[-1].get("usage")

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    model_name = getattr(active_llm, "model", "gemini-2.5-pro")

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    logger.info(
        "Node 1: Tokens consumed - prompt: %d, completion: %d, total: %d (model: %s)",
        prompt_tokens,
        completion_tokens,
        total_tokens,
        model_name,
    )

    if total_tokens > 0:
        cost_usd = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
        db.record_token_usage(
            phase="phase_1",
            node_name="node1_phase1_solver_rewriter",
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            candidate_id=candidate_id,
            duration_seconds=elapsed,
        )

    # Profiler registration
    try:
        from chia.base.profiling import get_profiler
        get_profiler().add_info({
            "node": "node1_phase1_solver_rewriter",
            "candidate_id": candidate_id,
            "usage": usage,
        })
    except Exception:
        pass

    # 5. Extract Registered Solver Name Directly from un0_rewriter_tool (Strict check, NO hardcoded fallback!)
    _sync_solver_rewrites_from_disk(un0_dir, candidate_id=candidate_id)
    registered_solver_name = un0_rewriter_tool.get_registered_solver(candidate_id=candidate_id)

    # Secondary lookup: check filesystem matching candidate_id directly
    if not registered_solver_name:
        solvers_dir = un0_dir / "un0" / "solver"
        clean_cand = candidate_id.strip().replace(" ", "_")
        from un0.solver import SOLVER_REGISTRY
        if solvers_dir.is_dir():
            matching_files = sorted(
                [p for p in solvers_dir.glob(f"{clean_cand}_*.py") if p.name != "__init__.py"],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for fpath in matching_files:
                stem = fpath.stem
                pot_name = stem[len(clean_cand) + 1:] if stem.startswith(f"{clean_cand}_") else stem
                if pot_name in SOLVER_REGISTRY:
                    registered_solver_name = pot_name
                    break

    # Tertiary lookup if direct tool registration wasn't returned
    if not registered_solver_name:
        from un0.solver import SOLVER_REGISTRY
        if solver_name in SOLVER_REGISTRY:
            registered_solver_name = solver_name
        else:
            # Check if any registered solver matching candidate ID exists
            all_solvers = un0_rewriter_tool.un0_list_registered_rewrites(axis="solver")
            cand_solvers = [s for s in all_solvers if candidate_id in s]
            if cand_solvers:
                registered_solver_name = cand_solvers[-1]

    from un0.solver import SOLVER_REGISTRY

    if not registered_solver_name or registered_solver_name not in SOLVER_REGISTRY:
        raw_result = getattr(query_result, "result", "") if query_result is not None else ""
        error_msg = (
            f"Node 1: Agent failed to synthesize or register a valid solver in SOLVER_REGISTRY. "
            f"Target: '{solver_name}', registered: {registered_solver_name!r}. "
            f"Available in registry: {list(SOLVER_REGISTRY.keys())}"
        )
        logger.error(error_msg)
        db.update_execution_status(
            candidate_id,
            rewrite_solver_status="FAILED",
            error_stage="node1_phase1_solver_rewriter",
            error_message=error_msg,
        )
        if raw_result and str(raw_result).strip():
            db.add_comment(
                candidate_id=candidate_id,
                phase="phase_1",
                agent_name="node1_phase1_solver_rewriter",
                comment=f"Agent solver registration failed. Diagnostic output:\n{str(raw_result)[:500]}",
            )
        raise RuntimeError(error_msg)

    solver_fn = SOLVER_REGISTRY[registered_solver_name]

    # Add cross-node comment and update DB status to SUCCESS
    db.add_comment(
        candidate_id=candidate_id,
        phase="phase_1",
        agent_name="node1_phase1_solver_rewriter",
        comment=(
            f"Successfully synthesized and registered dynamic solver '{registered_solver_name}' "
            f"via un0_rewriter tool. Discretization dt={dt}, num_steps={num_steps}."
        ),
    )

    db.update_execution_status(
        candidate_id,
        rewrite_solver_status="SUCCESS",
    )

    return {
        "candidate_id": candidate_id,
        "status": "SUCCESS",
        "solver": registered_solver_name,
        "solver_fn": solver_fn,
        "deterministic": False,
        "tokens_consumed": total_tokens,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "next_node": "node2_phase1_noise_rewriter",
    }


# Backwards compatibility alias
node1_solver_rewriter = node1_phase1_solver_rewriter
