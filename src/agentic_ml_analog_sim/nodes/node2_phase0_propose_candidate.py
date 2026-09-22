"""Node 2: Agentic candidate proposal node utilizing AntigravityLLM with Un-0 and DB tools."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Any, Mapping, Optional
from uuid import uuid4

from chia.base.ChiaFunction import ChiaFunction

logger = logging.getLogger("agentic_ml_analog_sim.node2")


NODE2_SYSTEM_INSTRUCTIONS = """
You are the Phase 0 Exploration & Proposal Specialist for the Un-0 Analog Machine Learning Accelerator.
Un-0 is a physical oscillatory neural network (Kuramoto dynamics) mapped to analog silicon crossbars.

### Primary Objective:
REDUCE SIMULATION LATENCY WHILE PRESERVING INFERENCE ACCURACY.
Explore combinations of numerical solvers, physical hardware noise models, and coupling sparsity.

### Database Access (Read-Only MCP Tool):
You are equipped with the `result_db_read` MCP tool which provides JUST READ access to the simulator's database.
You CANNOT and MUST NOT perform write/mutation operations via this tool; candidate insertion is handled safely by programmatic validation.

Available Read-Only MCP Tools:
- `read_prior_proposals(status=None, limit=50)`: Reads all candidate proposals that have already been made in the database, including solver, noise model, parameters, sparsity, evaluation metrics (latency_ms, relative_error, absolute_error), status, and cross-phase comments.
- `get_candidate_history_summary(limit=50)`: Concise summary of all parameter configurations already explored.
- `get_pareto_front()`: Current Pareto-optimal frontier of completed evaluations (latency vs accuracy).
- `get_reference_result()`: Baseline reference simulation result.
- `get_active_evaluations()`: Candidate(s) currently being evaluated in the active loop.
- `get_proposal_comments(candidate_id)`: Cross-phase commentary notes for a candidate row.

### Hardware Noise Modeling Skill Integration (L0-L5 Ladder):
Based on the physical-neural-network-hardware-noise-modeling literature:
- L0 (Deterministic Static Mismatch): Fixed perturbation on coupling weights delta_K = sigma * z, where z ~ N(0, 1). Models fabrication and programming variation. Fast to compute in digital simulation.
- L1 (Stochastic Parameter Noise): Parameter perturbations resampled per sample or integration timestep.
- L2 (Functional Interface Noise): Input DAC quantization (e.g. 6, 8, 10 bits), clipping, output ADC quantization, and readout noise.
- L3 (Correlated Drift): Spatial process gradients and 1/f aging with power-law decay theta(t) = theta(t0) * (t/t0)^(-nu).
- Coupling Sparsity: Threshold pruning (|K_ij| < tau) or bounded oscillator degree to reduce crossbar capacitive load and digital GEMV latency.

### FP32 Tolerance Bounds & Mathematical Precision Verification:
- Single-precision float (fp32) has machine epsilon eps ~= 1.19e-7.
- Solvers with step size h accumulate local truncation error: Euler is O(h), RK4 is O(h^4).
- Verify that configuration tolerances (compile_check rtol/atol >= 1e-4, solver_accuracy rtol/atol >= 0.05) are mathematically attainable.

### Candidate Proposal Guidelines:
1. QUERY PRIOR PROPOSALS: Call `read_prior_proposals` or `get_candidate_history_summary` to inspect all proposals that have already been made.
2. AVOID REPEATED PROPOSALS: Never propose a combination of (solver, num_steps, noise_model, sparsity_config) that has already been proposed or evaluated in the database. Node 3 will automatically REJECT duplicate proposals.
3. If the database has no non-reference proposals, propose a fast baseline strictly conforming to the exploration config (e.g. Euler with 5 or 10 steps, L0 static mismatch sigma=0.01, dense connectivity).
4. If prior evaluations exist, balance exploration of new parameters with exploitation near the Pareto frontier.
5. If a prior proposal was rejected, avoid repeating the rejected parameter configuration.
6. Provide a clear rationale explaining expected latency reduction and accuracy impact.

### Response Format:
Respond with a valid JSON object matching this schema:
```json
{
  "candidate_id": "cand_<short_unique_id>",
  "solver": "<solver_name>",
  "num_steps": <int>,
  "solver_params": {"num_steps": <int>, "dt": <float>},
  "noise_model": "<noise_model_name>",
  "noise_params": {"sigma": <float>, ...},
  "sparsity_config": {"sparsity_ratio": <float>, ...},
  "selection_reason": "<detailed rationale for proposal>",
  "phase_1_prompt": "<structured instructions for Phase 1 to build and simulate this candidate>"
}
```
"""


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Extract attribute or dictionary key safely."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_allowed_solvers(config: Any) -> list[str]:
    """Extract list of allowed solver names from config."""
    configured_solvers = _get_val(config, "solvers", [])
    allowed: list[str] = []
    if isinstance(configured_solvers, Mapping):
        allowed = list(configured_solvers.keys())
    elif isinstance(configured_solvers, (list, tuple)):
        for s in configured_solvers:
            if isinstance(s, str):
                allowed.append(s)
            elif isinstance(s, Mapping):
                name = s.get("name")
                if name:
                    allowed.append(name)
            elif hasattr(s, "name"):
                name = getattr(s, "name")
                if name:
                    allowed.append(name)
    return allowed


def _parse_llm_json_response(
    raw_text: str,
    config: Any = None,
    db: Any = None,
    iteration: int = 1,
    target_noise_model: Optional[str] = None,
) -> dict[str, Any]:
    """Extract and parse JSON object from LLM response text with dynamic config-aware fallback."""
    clean_text = raw_text.strip() if raw_text else ""
    parsed_json: dict[str, Any] | None = None

    if clean_text:
        # Look for ```json ... ``` code fence
        fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
        match = fence_pattern.search(clean_text)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for raw JSON object {...}
        if parsed_json is None:
            brace_pattern = re.compile(r"(\{.*\})", re.DOTALL)
            match = brace_pattern.search(clean_text)
            if match:
                try:
                    parsed_json = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        if parsed_json is None:
            try:
                parsed_json = json.loads(clean_text)
            except json.JSONDecodeError:
                pass

    if isinstance(parsed_json, dict) and parsed_json:
        return parsed_json

    logger.warning("Could not parse JSON directly from LLM output. Synthesizing valid candidate from config.")
    if config is not None:
        synth = _synthesize_unexplored_candidate(config, db, iteration, target_noise_model)
        if synth is not None and isinstance(synth, dict):
            return synth

    # If _synthesize_unexplored_candidate returns None or config is None:
    # Dynamically inspect _get_val(config, "solvers", []): pick first allowed solver
    allowed_solvers = _get_allowed_solvers(config)
    fallback_solver = allowed_solvers[0] if allowed_solvers else "euler"

    configured_solvers = _get_val(config, "solvers", {}) if config is not None else {}
    steps = 5
    if isinstance(configured_solvers, Mapping) and fallback_solver in configured_solvers:
        spec = configured_solvers[fallback_solver]
        spec_steps = _get_val(spec, "steps") or _get_val(spec, "allowed_steps")
        if spec_steps and isinstance(spec_steps, (list, tuple)) and len(spec_steps) > 0:
            steps = int(spec_steps[0])
    elif isinstance(configured_solvers, (list, tuple)):
        for s in configured_solvers:
            s_name = s if isinstance(s, str) else (_get_val(s, "name") if isinstance(s, Mapping) or hasattr(s, "name") else None)
            if s_name == fallback_solver:
                spec_steps = _get_val(s, "steps") or _get_val(s, "allowed_steps")
                if spec_steps and isinstance(spec_steps, (list, tuple)) and len(spec_steps) > 0:
                    steps = int(spec_steps[0])
                break

    noise_model = target_noise_model
    if not noise_model:
        raw_noise_models = _get_val(config, "noise_models", ["L0_static_mismatch"]) if config is not None else ["L0_static_mismatch"]
        if isinstance(raw_noise_models, (list, tuple)) and len(raw_noise_models) > 0:
            first_n = raw_noise_models[0]
            if isinstance(first_n, str):
                noise_model = first_n
            elif isinstance(first_n, Mapping) and "name" in first_n:
                noise_model = first_n["name"]
            elif hasattr(first_n, "name"):
                noise_model = getattr(first_n, "name")
            else:
                noise_model = "L0_static_mismatch"
        elif isinstance(raw_noise_models, Mapping) and len(raw_noise_models) > 0:
            noise_model = list(raw_noise_models.keys())[0]
        else:
            noise_model = "L0_static_mismatch"

    dt = round(1.0 / steps, 6) if steps > 0 else 0.1
    noise_params = {} if noise_model == "none" else {"sigma": 0.01}
    cand_id = f"cand_{fallback_solver}{steps}_{uuid4().hex[:6]}"
    selection_reason = f"Fallback default: Explore {fallback_solver} with {noise_model}"
    sparsity_config = {"type": "dense", "sparsity_ratio": 0.0}
    solver_params = {"num_steps": steps, "dt": dt}

    phase_1_prompt = _format_phase_1_prompt(
        candidate_id=cand_id,
        solver=fallback_solver,
        solver_params=solver_params,
        noise_model=noise_model,
        noise_params=noise_params,
        sparsity_config=sparsity_config,
        selection_reason=selection_reason,
    )

    return {
        "candidate_id": cand_id,
        "iteration": iteration,
        "solver": fallback_solver,
        "num_steps": steps,
        "solver_params": solver_params,
        "noise_model": noise_model,
        "noise_params": noise_params,
        "sparsity_config": sparsity_config,
        "selection_reason": selection_reason,
        "phase_1_prompt": phase_1_prompt,
        "status": "PENDING",
    }


def _format_phase_1_prompt(
    candidate_id: str,
    solver: str,
    solver_params: dict[str, Any],
    noise_model: str,
    noise_params: dict[str, Any],
    sparsity_config: dict[str, Any],
    selection_reason: str,
) -> str:
    """Formulate actionable instructions for Phase 1 build and simulation."""
    return (
        f"# Phase 1 Build & Simulation Instructions\n"
        f"Candidate ID: {candidate_id}\n"
        f"Target Solver: {solver}\n"
        f"Solver Parameters: {json.dumps(solver_params)}\n"
        f"Noise Model: {noise_model}\n"
        f"Noise Parameters: {json.dumps(noise_params)}\n"
        f"Sparsity Config: {json.dumps(sparsity_config)}\n"
        f"Selection Rationale: {selection_reason}\n\n"
        f"## Action Items for AgentSolver:\n"
        f"1. Modify Un-0 ODE integration loop in `un0/model.py` to use '{solver}'.\n"
        f"2. Configure step parameters: {solver_params}.\n"
        f"3. Inject noise model nonidealities ('{noise_model}') with parameters: {noise_params}.\n"
        f"4. Apply crossbar coupling sparsity: {sparsity_config}.\n"
        f"5. Verify compile and simulation output against fp32 reference tolerances."
    )


def _insert_pending_candidate(db: Any, record: dict[str, Any]) -> None:
    """Insert candidate record into DB as PENDING across diverse DB types."""
    candidate_id = record["candidate_id"]
    iteration = record.get("iteration", 1)
    solver = record["solver"]
    noise_model = record["noise_model"]
    sparsity_config = record.get("sparsity_config", {})
    solver_params = record.get("solver_params", {})
    noise_params = record.get("noise_params", {})
    selection_reason = record.get("selection_reason", "")
    phase_1_prompt = record.get("phase_1_prompt", "")

    sparsity_json = json.dumps(sparsity_config) if isinstance(sparsity_config, (dict, list)) else str(sparsity_config)
    solver_params_json = json.dumps(solver_params) if isinstance(solver_params, (dict, list)) else str(solver_params)
    noise_params_json = json.dumps(noise_params) if isinstance(noise_params, (dict, list)) else str(noise_params)

    # 1. Specialized insert_proposal method on ResultDatabase
    if hasattr(db, "insert_proposal") and callable(db.insert_proposal):
        try:
            db.insert_proposal(
                solver=solver,
                noise_model=noise_model,
                solver_params=solver_params_json,
                noise_params=noise_params_json,
                sparsity_config=sparsity_json,
                selection_reason=selection_reason,
                phase_1_prompt=phase_1_prompt,
                candidate_id=candidate_id,
                status="PENDING",
                iteration=iteration,
            )
            logger.info("Inserted proposal %s via db.insert_proposal()", candidate_id)
            return
        except TypeError:
            try:
                db.insert_proposal(
                    solver=solver,
                    noise_model=noise_model,
                    solver_params=solver_params_json,
                    noise_params=noise_params_json,
                    sparsity_config=sparsity_json,
                    selection_reason=selection_reason,
                )
                logger.info("Inserted proposal %s via db.insert_proposal() (short signature)", candidate_id)
                return
            except Exception as exc:
                logger.debug("db.insert_proposal short call failed: %s", exc)
        except Exception as exc:
            logger.debug("db.insert_proposal call failed: %s", exc)

    # 2. Direct SQLite query
    conn = getattr(db, "conn", None) or getattr(db, "connection", None) or getattr(db, "_conn", None)
    if conn is not None and isinstance(conn, sqlite3.Connection):
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO proposals (
                    candidate_id, iteration, solver, noise_model, sparsity_config,
                    solver_params, noise_params, status, is_reference, selection_reason, phase_1_prompt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, ?, ?)
                """,
                (
                    candidate_id,
                    iteration,
                    solver,
                    noise_model,
                    sparsity_json,
                    solver_params_json,
                    noise_params_json,
                    selection_reason,
                    phase_1_prompt,
                ),
            )
            conn.commit()
            logger.info("Inserted proposal %s directly into SQLite proposals table", candidate_id)
            return
        except Exception as exc:
            logger.warning("Direct SQLite insert into proposals failed: %s", exc)

    # 3. In-memory / mock database support
    if isinstance(db, dict):
        db[candidate_id] = {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "solver": solver,
            "noise_model": noise_model,
            "sparsity_config": sparsity_config,
            "solver_params": solver_params,
            "noise_params": noise_params,
            "status": "PENDING",
            "is_reference": False,
            "selection_reason": selection_reason,
            "phase_1_prompt": phase_1_prompt,
        }
    elif isinstance(db, list):
        db.append(
            {
                "candidate_id": candidate_id,
                "iteration": iteration,
                "solver": solver,
                "noise_model": noise_model,
                "sparsity_config": sparsity_config,
                "solver_params": solver_params,
                "noise_params": noise_params,
                "status": "PENDING",
                "is_reference": False,
                "selection_reason": selection_reason,
                "phase_1_prompt": phase_1_prompt,
            }
        )
    elif hasattr(db, "proposals") and isinstance(db.proposals, dict):
        db.proposals[candidate_id] = record


def _bind_tools(db: Any, tools: list[Any] | None) -> list[Any]:
    """Bind ResultDBReadOnlyTool and Un0ContextTool to provide Node 2 with read-only database inspection."""
    from agentic_ml_analog_sim.tools.result_db_tool import ResultDBReadOnlyTool, ResultDBTool

    bound_tools: list[Any] = []
    has_db_read = False
    has_un0 = False

    if tools is not None:
        for t in tools:
            # If a writable ResultDBTool was provided, ensure Node 2 only gets read-only access
            if isinstance(t, ResultDBTool):
                if getattr(t, "read_only", False):
                    bound_tools.append(t)
                else:
                    read_tool = ResultDBReadOnlyTool(db=getattr(t, "db", db), deploy=False)
                    bound_tools.append(read_tool)
                has_db_read = True
            else:
                bound_tools.append(t)
                t_name = getattr(t, "name", str(t))
                if "result_db" in t_name or "db" in t_name:
                    has_db_read = True
                if "un0" in t_name:
                    has_un0 = True

    if not has_db_read:
        try:
            bound_tools.append(ResultDBReadOnlyTool(db=db, deploy=False))
        except Exception as e:
            logger.debug("Could not bind ResultDBReadOnlyTool: %s", e)

    if not has_un0:
        try:
            from agentic_ml_analog_sim.tools.un0_context_tool import Un0ContextTool
            bound_tools.append(Un0ContextTool(auto_start=False))
        except Exception as e:
            logger.debug("Could not bind Un0ContextTool: %s", e)

    return bound_tools



def _resolve_llm(llm: Any = None, config: Any = None) -> Any:
    """Resolve LLM instance, defaulting to MockAntigravityLLM if AntigravityLLM is unavailable."""
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node2_phase0_propose_candidate", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node2_phase0_propose_candidate")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-pro"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "high"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args, system_message=NODE2_SYSTEM_INSTRUCTIONS)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args, system_message=NODE2_SYSTEM_INSTRUCTIONS)


def _normalize_dict(val: Any) -> dict[str, Any]:
    """Safely convert dict-like, JSON-encoded, or object values to a standard dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    if hasattr(val, "model_dump") and callable(val.model_dump):
        return val.model_dump()
    if hasattr(val, "__dict__"):
        return {k: v for k, v in val.__dict__.items() if not k.startswith("_")}
    return {}


def _synthesize_unexplored_candidate(
    config: Any,
    db: Any,
    iteration: int = 1,
    target_noise_model: Optional[str] = None,
) -> dict[str, Any] | None:
    """Deterministically find and synthesize the next unexplored valid candidate configuration."""
    from agentic_ml_analog_sim.nodes.node3_phase0_validate_proposal import _check_duplicate

    solvers_dict = _get_val(config, "solvers", {})
    sparsity_configs_obj = _get_val(config, "sparsity_configs", {})

    # Extract solver list and steps
    solver_options: list[tuple[str, int]] = []
    if isinstance(solvers_dict, Mapping):
        for s_name, s_cfg in solvers_dict.items():
            steps = _get_val(s_cfg, "steps") or _get_val(s_cfg, "allowed_steps") or [5, 10, 20]
            for st in steps:
                solver_options.append((s_name, int(st)))
    elif isinstance(solvers_dict, (list, tuple)):
        for s_entry in solvers_dict:
            if isinstance(s_entry, str):
                s_name = s_entry
                steps = [5, 10, 20]
            else:
                s_name = _get_val(s_entry, "name", "euler")
                steps = _get_val(s_entry, "steps") or _get_val(s_entry, "allowed_steps") or [5, 10, 20]
            for st in steps:
                solver_options.append((s_name, int(st)))

    if not solver_options:
        solver_options = [("euler", 5), ("euler", 10), ("euler", 20), ("par_ode", 5), ("par_ode", 10)]

    # Extract noise models
    if target_noise_model:
        noise_models_list = [target_noise_model]
    else:
        raw_noise_models = _get_val(config, "noise_models", ["L0_static_mismatch"])
        noise_models_list = []
        if isinstance(raw_noise_models, (list, tuple)):
            for n in raw_noise_models:
                if isinstance(n, str):
                    noise_models_list.append(n)
                elif isinstance(n, Mapping) and "name" in n:
                    noise_models_list.append(n["name"])
                elif hasattr(n, "name"):
                    noise_models_list.append(getattr(n, "name"))
        elif isinstance(raw_noise_models, Mapping):
            noise_models_list = list(raw_noise_models.keys())
        if not noise_models_list:
            noise_models_list = ["L0_static_mismatch"]

    # Extract sparsity options
    sparsity_options: list[tuple[str, dict[str, Any]]] = []
    # Dense
    sparsity_options.append(("dense", {"type": "dense", "sparsity_ratio": 0.0}))

    # Degree bounded
    deg_cfg = _get_val(sparsity_configs_obj, "degree_bounded", {})
    max_degrees = _get_val(deg_cfg, "max_degree", [])
    if isinstance(max_degrees, (list, tuple)):
        for deg in max_degrees:
            sparsity_options.append((f"deg{deg}", {"type": "degree_bounded", "max_degree": int(deg), "sparsity_ratio": 0.0}))

    # Threshold pruned
    thresh_cfg = _get_val(sparsity_configs_obj, "threshold_pruned", {})
    thresh_ratios = _get_val(thresh_cfg, "allowed_sparsity_ratios", [])
    if isinstance(thresh_ratios, (list, tuple)):
        for r in thresh_ratios:
            sparsity_options.append((f"thresh{r}", {"type": "threshold_pruned", "sparsity_ratio": float(r)}))

    # Also handle list representation for sparsity_configs if present
    if isinstance(sparsity_configs_obj, (list, tuple)):
        for sp in sparsity_configs_obj:
            sp_name = _get_val(sp, "name")
            if sp_name == "degree_bounded":
                m_degs = _get_val(sp, "max_degree", [])
                if isinstance(m_degs, (list, tuple)):
                    for deg in m_degs:
                        sparsity_options.append((f"deg{deg}", {"type": "degree_bounded", "max_degree": int(deg), "sparsity_ratio": 0.0}))
            elif sp_name == "threshold_pruned":
                t_ratios = _get_val(sp, "allowed_sparsity_ratios", [])
                if isinstance(t_ratios, (list, tuple)):
                    for r in t_ratios:
                        sparsity_options.append((f"thresh{r}", {"type": "threshold_pruned", "sparsity_ratio": float(r)}))

    # Iterate through all grid combinations to find first unexplored
    for s_name, st in solver_options:
        for nm in noise_models_list:
            for sp_label, sp_dict in sparsity_options:
                dt_val = round(1.0 / st, 6)
                test_rec = {
                    "solver": s_name,
                    "solver_params": {"num_steps": st, "dt": dt_val},
                    "noise_model": nm,
                    "noise_params": {} if nm == "none" else {"sigma": 0.01},
                    "sparsity_config": sp_dict,
                }
                is_dup, _ = _check_duplicate("synth_check", test_rec, db)
                if not is_dup:
                    cand_id = f"cand_{s_name}{st}_{nm[:2]}_{sp_label}_{uuid4().hex[:4]}"
                    reason = f"Synthesized unexplored configuration: {s_name} ({st} steps, dt={dt_val}), {nm}, {sp_label}."
                    phase_1_prompt = _format_phase_1_prompt(
                        candidate_id=cand_id,
                        solver=s_name,
                        solver_params=test_rec["solver_params"],
                        noise_model=nm,
                        noise_params=test_rec["noise_params"],
                        sparsity_config=sp_dict,
                        selection_reason=reason,
                    )
                    return {
                        "candidate_id": cand_id,
                        "iteration": iteration,
                        "solver": s_name,
                        "num_steps": st,
                        "noise_model": nm,
                        "solver_params": test_rec["solver_params"],
                        "noise_params": test_rec["noise_params"],
                        "sparsity_config": sp_dict,
                        "selection_reason": reason,
                        "phase_1_prompt": phase_1_prompt,
                        "status": "PENDING",
                    }
    return None


@ChiaFunction(resources={"agent_worker": 1, "antigravity_creds": 0.01})
def node2_phase0_propose_candidate(
    db: Any,
    config: Any,
    llm: Any = None,
    tools: list[Any] | None = None,
    prior_rejection_reason: str | None = None,
    iteration: int = 1,
    target_noise_model: Optional[str] = None,
    prevalidate: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Agentic node proposing the next simulation candidate (solver + noise + sparsity).

    Utilizes AntigravityLLM (or MockAntigravityLLM), binds ResultDBTool and Un0ContextTool,
    incorporates hardware noise modeling skill guidelines, validates FP32 tolerances,
    generates proposal, stores Phase 1 prompt, inserts candidate into DB as PENDING,
    and returns structured candidate information.

    Args:
        db: ResultDatabase instance (or compatible database/mock).
        config: SimulationConfig instance (or compatible configuration object).
        llm: Optional AntigravityLLM or MockAntigravityLLM instance.
        tools: Optional list of ChiaTool instances.
        prior_rejection_reason: Optional rejection note from Node 3 to steer retry.
        iteration: Optimization loop iteration counter.
        target_noise_model: Optional target noise model to strictly enforce.

    Returns:
        dict: {
            "candidate_id": str,
            "solver": str,
            "noise_model": str,
            "phase_1_prompt": str
        }
    """
    active_llm = _resolve_llm(llm, config)
    active_tools = _bind_tools(db, tools)

    # Build user message with configuration context and rejection feedback
    config_summary = {
        "solvers": _get_val(config, "solvers", []),
        "noise_models": _get_val(config, "noise_models", []),
        "sparsity_configs": _get_val(config, "sparsity_configs", []),
        "tolerances": _get_val(config, "tolerances", {}),
        "budget": _get_val(config, "budget", {}),
    }

    # Retrieve all existing proposals from the DB across all statuses to inject into prompt
    all_proposals = []
    if hasattr(db, "get_all_proposals") and callable(db.get_all_proposals):
        try:
            all_proposals = db.get_all_proposals()
        except Exception:
            pass
    if not all_proposals and hasattr(db, "query_proposals") and callable(db.query_proposals):
        try:
            all_proposals = db.query_proposals(limit=200)
            if all_proposals:
                all_proposals = list(reversed(all_proposals))
        except Exception:
            pass
    if not all_proposals and hasattr(db, "get_proposals") and callable(db.get_proposals):
        try:
            all_proposals = db.get_proposals(limit=200)
            if all_proposals:
                all_proposals = list(reversed(all_proposals))
        except Exception:
            pass

    history_markdown = []
    if all_proposals:
        recent_proposals = all_proposals[-10:] if len(all_proposals) > 10 else all_proposals
        history_markdown.append("\n### Prohibited Configurations (Already Explored or In-Flight in DB):")
        history_markdown.append("| Candidate ID | Status | Solver | Steps | dt | Noise Model | Sparsity |")
        history_markdown.append("|---|---|---|---|---|---|---|")
        for p in recent_proposals:
            p_cid = _get_val(p, "candidate_id", "unknown")
            p_status = _get_val(p, "status", "UNKNOWN")
            p_solv = _get_val(p, "solver", "euler")
            p_sparams = _normalize_dict(_get_val(p, "solver_params"))
            p_steps = p_sparams.get("num_steps", "?")
            p_dt = p_sparams.get("dt", "?")
            p_noise = _get_val(p, "noise_model", "none")
            p_spar = _normalize_dict(_get_val(p, "sparsity_config"))
            spar_desc = p_spar.get("type", "")
            if p_spar.get("max_degree"):
                spar_desc = f"deg{p_spar.get('max_degree')}"
            elif "sparsity_ratio" in p_spar:
                spar_desc = f"ratio {p_spar.get('sparsity_ratio')}"
            history_markdown.append(f"| {p_cid} | {p_status} | {p_solv} | {p_steps} | {p_dt} | {p_noise} | {spar_desc} |")
        history_markdown.append(
            "CRITICAL CONSTRAINT: You are strictly forbidden from proposing any configuration that matches "
            "the (solver, steps, dt, noise, sparsity) tuples listed above. Proposing a duplicate will cause automatic rejection."
        )

    user_prompt_lines = [
        "Please propose the next simulation design candidate for Un-0 analog ML accelerator.",
        f"Exploration configuration: {json.dumps(config_summary, default=str)}",
        f"Current iteration: {iteration}",
    ]
    if target_noise_model:
        user_prompt_lines.append(
            f"CRITICAL REQUIREMENT: You MUST set noise_model to '{target_noise_model}'. Do NOT propose any other noise model."
        )
    if history_markdown:
        user_prompt_lines.extend(history_markdown)
    else:
        user_prompt_lines.extend([
            "IMPORTANT: You have read-only access to the database via the result_db_read MCP tool.",
            "Use read_prior_proposals or get_candidate_history_summary to inspect all proposals already made in the database.",
            "DO NOT propose a combination of solver, step count, noise model, and sparsity configuration that already exists.",
        ])

    if prior_rejection_reason:
        user_prompt_lines.append(
            f"\nATTENTION: A prior proposal was rejected by Node 3 with reason:\n"
            f"'{prior_rejection_reason}'\n"
            f"Do NOT propose the exact same configuration again. Select an alternative parameter combination."
        )

    user_prompt = "\n".join(user_prompt_lines)

    # Prompt the LLM
    start_time = time.time()
    query_result = active_llm.prompt(user_message=user_prompt, tools=active_tools)
    duration_seconds = time.time() - start_time
    response_text = getattr(query_result, "result", str(query_result))

    # Parse proposal JSON
    proposal_data = _parse_llm_json_response(
        response_text,
        config=config,
        db=db,
        iteration=iteration,
        target_noise_model=target_noise_model,
    )

    # In-node pre-validation: inspect parsed proposal fields
    solver = proposal_data.get("solver")
    noise_model = proposal_data.get("noise_model")
    solver_params = proposal_data.get("solver_params")
    sparsity_config = proposal_data.get("sparsity_config")

    allowed_solvers = _get_allowed_solvers(config)
    auto_corrected = False

    cfg_prev = _get_val(config, "prevalidate")
    should_prevalidate = prevalidate if cfg_prev is None else bool(cfg_prev)
    if should_prevalidate and allowed_solvers and solver not in allowed_solvers:
        logger.warning(
            "Node 2: Proposed solver '%s' is not in configured solvers %s. Auto-correcting to unexplored valid candidate.",
            solver,
            allowed_solvers,
        )
        synth_candidate = _synthesize_unexplored_candidate(config, db, iteration, target_noise_model)
        if synth_candidate is not None and isinstance(synth_candidate, dict):
            proposal_data = synth_candidate
        else:
            fallback_solver = allowed_solvers[0]
            proposal_data["solver"] = fallback_solver
            if isinstance(solver_params, dict):
                st = solver_params.get("num_steps", 5)
                proposal_data["solver_params"] = {"num_steps": st, "dt": round(1.0 / st, 6)}
            else:
                proposal_data["solver_params"] = {"num_steps": 5, "dt": 0.2}
        auto_corrected = True

    if target_noise_model and proposal_data.get("noise_model") != target_noise_model:
        proposal_data["noise_model"] = target_noise_model
        if target_noise_model == "none":
            proposal_data["noise_params"] = {}
        elif not proposal_data.get("noise_params"):
            proposal_data["noise_params"] = {"sigma": 0.01}
        auto_corrected = True

    # Extract or generate proposal fields
    candidate_id = proposal_data.get("candidate_id") or f"cand_{uuid4().hex[:8]}"
    solver = proposal_data.get("solver") or (allowed_solvers[0] if allowed_solvers else "euler")
    noise_model = proposal_data.get("noise_model", target_noise_model or "L0_static_mismatch")
    num_steps = proposal_data.get("num_steps", 5)

    solver_params = (
        proposal_data.get("solver_params")
        if proposal_data.get("solver_params") is not None
        else {"num_steps": num_steps, "dt": round(1.0 / num_steps, 6)}
    )
    noise_params = (
        proposal_data.get("noise_params")
        if proposal_data.get("noise_params") is not None
        else ({} if noise_model == "none" else {"sigma": 0.01})
    )
    sparsity_config = (
        proposal_data.get("sparsity_config")
        if proposal_data.get("sparsity_config") is not None
        else {"type": "dense", "sparsity_ratio": 0.0}
    )
    selection_reason = proposal_data.get(
        "selection_reason",
        f"Proposed {solver} with {noise_model} to reduce simulation latency while preserving accuracy",
    )

    # Formulate and store phase_1_prompt
    if auto_corrected or not proposal_data.get("phase_1_prompt"):
        phase_1_prompt = _format_phase_1_prompt(
            candidate_id=candidate_id,
            solver=solver,
            solver_params=solver_params,
            noise_model=noise_model,
            noise_params=noise_params,
            sparsity_config=sparsity_config,
            selection_reason=selection_reason,
        )
    else:
        phase_1_prompt = proposal_data["phase_1_prompt"]

    # Construct preliminary candidate record
    candidate_record = {
        "candidate_id": candidate_id,
        "iteration": iteration,
        "solver": solver,
        "noise_model": noise_model,
        "solver_params": solver_params,
        "noise_params": noise_params,
        "sparsity_config": sparsity_config,
        "selection_reason": selection_reason,
        "phase_1_prompt": phase_1_prompt,
        "status": "PENDING",
    }

    # Store candidate in DB as PENDING
    _insert_pending_candidate(db, candidate_record)

    # Extract token metrics and record to database and profiler
    usage = getattr(query_result, "usage", {}) or {}
    if not usage and hasattr(active_llm, "history") and active_llm.history:
        usage = active_llm.history[-1].get("usage", {}) or {}

    prompt_tokens = usage.get("prompt_tokens", 120) if isinstance(usage, Mapping) else 120
    completion_tokens = usage.get("candidates_tokens", usage.get("completion_tokens", 85)) if isinstance(usage, Mapping) else 85
    if not isinstance(prompt_tokens, (int, float)):
        prompt_tokens = 120
    if not isinstance(completion_tokens, (int, float)):
        completion_tokens = 85
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) if isinstance(usage, Mapping) else (prompt_tokens + completion_tokens)
    if not isinstance(total_tokens, (int, float)):
        total_tokens = prompt_tokens + completion_tokens
    cost_usd = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
    raw_model = getattr(active_llm, "model", "gemini-2.5-pro")
    model_name = raw_model if isinstance(raw_model, str) else "gemini-2.5-pro"

    if hasattr(db, "record_token_usage") and callable(db.record_token_usage):
        try:
            db.record_token_usage(
                phase="phase_0",
                node_name="node2_phase0_propose_candidate",
                model=model_name,
                candidate_id=candidate_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            logger.warning("Failed to record token usage to DB: %s", exc)
    else:
        conn = getattr(db, "conn", None) or getattr(db, "connection", None) or getattr(db, "_conn", None)
        if conn is not None and isinstance(conn, sqlite3.Connection):
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO token_usage (
                        candidate_id, phase, node_name, model,
                        prompt_tokens, completion_tokens, total_tokens,
                        cost_usd, duration_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        "phase_0",
                        "node2_phase0_propose_candidate",
                        model_name,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        cost_usd,
                        duration_seconds,
                    ),
                )
                conn.commit()
            except Exception as exc:
                logger.debug("Direct SQLite insert into token_usage failed: %s", exc)

    # Attach to Chia profiler
    try:
        from chia.base.profiling import get_profiler
        get_profiler().add_info({"usage": usage, "model": model_name, "phase": "phase_0"})
    except Exception:
        pass

    logger.info(
        "Node 2: Successfully generated and stored candidate %s (solver=%s, noise_model=%s)",
        candidate_id,
        solver,
        noise_model,
    )

    return {
        "candidate_id": candidate_id,
        "solver": solver,
        "noise_model": noise_model,
        "phase_1_prompt": phase_1_prompt,
    }


# Backward compatibility aliases
propose_candidate = node2_phase0_propose_candidate
node2_propose_candidate = node2_phase0_propose_candidate

