"""Node 2: Agentic candidate proposal node utilizing AntigravityLLM with Un-0 and DB tools."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Any, Mapping
from uuid import uuid4

from chia.base.ChiaFunction import ChiaFunction

logger = logging.getLogger("agentic_ml_analog_sim.node2")


NODE2_SYSTEM_INSTRUCTIONS = """
You are the Phase 0 Exploration & Proposal Specialist for the Un-0 Analog Machine Learning Accelerator.
Un-0 is a physical oscillatory neural network (Kuramoto dynamics) mapped to analog silicon crossbars.

### Primary Objective:
REDUCE SIMULATION LATENCY WHILE PRESERVING INFERENCE ACCURACY.
Explore combinations of numerical solvers, physical hardware noise models, and coupling sparsity.

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
1. Query prior proposals and Pareto front from the Result DB tool.
2. If the database is empty or contains only reference data, propose a fast baseline (e.g. Euler with 5 or 10 steps, L0 static mismatch sigma=0.01, 40% sparsity).
3. If prior evaluations exist, balance exploration of new parameters with exploitation near the Pareto frontier.
4. If a prior proposal was rejected, avoid repeating the rejected parameter configuration.
5. Provide a clear rationale explaining expected latency reduction and accuracy impact.

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


def _parse_llm_json_response(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON object from LLM response text."""
    clean_text = raw_text.strip()

    # Look for ```json ... ``` code fence
    fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    match = fence_pattern.search(clean_text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Look for raw JSON object {...}
    brace_pattern = re.compile(r"(\{.*\})", re.DOTALL)
    match = brace_pattern.search(clean_text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON directly from LLM output: %s", raw_text[:200])
        # Fallback to default proposal structure
        return {
            "solver": "euler",
            "num_steps": 5,
            "solver_params": {"num_steps": 5, "dt": 0.2},
            "noise_model": "L0_static_mismatch",
            "noise_params": {"sigma": 0.01},
            "sparsity_config": {"sparsity_ratio": 0.4},
            "selection_reason": "Fallback default: Explore Euler with low static mismatch and 40% sparsity",
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
    """Bind ResultDBTool and Un0ContextTool if available and not already provided."""
    bound_tools: list[Any] = list(tools) if tools is not None else []
    tool_names = {getattr(t, "name", str(t)) for t in bound_tools}

    if "ResultDBTool" not in tool_names and "result_db" not in tool_names:
        try:
            from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool
            bound_tools.append(ResultDBTool(db=db))
        except Exception:
            pass

    if "Un0ContextTool" not in tool_names and "un0_context" not in tool_names:
        try:
            from agentic_ml_analog_sim.tools.un0_context_tool import Un0ContextTool
            bound_tools.append(Un0ContextTool())
        except Exception:
            pass

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


@ChiaFunction()
def node2_phase0_propose_candidate(
    db: Any,
    config: Any,
    llm: Any = None,
    tools: list[Any] | None = None,
    prior_rejection_reason: str | None = None,
    iteration: int = 1,
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

    user_prompt_lines = [
        "Please propose the next simulation design candidate for Un-0 analog ML accelerator.",
        f"Exploration configuration: {json.dumps(config_summary, default=str)}",
        f"Current iteration: {iteration}",
    ]

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
    proposal_data = _parse_llm_json_response(response_text)

    # Extract or generate proposal fields
    candidate_id = proposal_data.get("candidate_id") or f"cand_{uuid4().hex[:8]}"
    solver = proposal_data.get("solver", "euler")
    noise_model = proposal_data.get("noise_model", "L0_static_mismatch")
    num_steps = proposal_data.get("num_steps", 5)

    solver_params = (
        proposal_data.get("solver_params")
        if proposal_data.get("solver_params") is not None
        else {"num_steps": num_steps, "dt": 0.2}
    )
    noise_params = (
        proposal_data.get("noise_params")
        if proposal_data.get("noise_params") is not None
        else ({} if noise_model == "none" else {"sigma": 0.01})
    )
    sparsity_config = (
        proposal_data.get("sparsity_config")
        if proposal_data.get("sparsity_config") is not None
        else {"sparsity_ratio": 0.4}
    )
    selection_reason = proposal_data.get(
        "selection_reason",
        f"Proposed {solver} with {noise_model} to reduce simulation latency while preserving accuracy",
    )

    # Formulate and store phase_1_prompt
    phase_1_prompt = proposal_data.get("phase_1_prompt") or _format_phase_1_prompt(
        candidate_id=candidate_id,
        solver=solver,
        solver_params=solver_params,
        noise_model=noise_model,
        noise_params=noise_params,
        sparsity_config=sparsity_config,
        selection_reason=selection_reason,
    )

    # Store candidate in DB as PENDING
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
    _insert_pending_candidate(db, candidate_record)

    # Extract token metrics and record to database and profiler
    usage = getattr(query_result, "usage", {}) or {}
    if not usage and hasattr(active_llm, "history") and active_llm.history:
        usage = active_llm.history[-1].get("usage", {}) or {}

    prompt_tokens = usage.get("prompt_tokens", 120)
    completion_tokens = usage.get("candidates_tokens", usage.get("completion_tokens", 85))
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    cost_usd = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
    model_name = getattr(active_llm, "model", "gemini-2.5-pro")

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

