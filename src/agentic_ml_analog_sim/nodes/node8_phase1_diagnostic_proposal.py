"""Node 8: Diagnostic Proposal Agentic Node (Phase 1).

Diagnoses compilation, verification, or evaluation failures from Node 5 or Node 7
and emits a corrective rewrite proposal or a debug mode request using agentic reasoning
over database error logs and MCP tools.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:  # pragma: no cover
    def ChiaFunction(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator

from agentic_ml_analog_sim.config import SimulationConfig
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool

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

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node8_phase1_diagnostic_proposal")


def _resolve_llm(llm: Optional[Any] = None, config: Optional[SimulationConfig] = None) -> Any:
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node8_phase1_diagnostic_proposal", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node8_phase1_diagnostic_proposal")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-flash"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "medium"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


def _parse_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON proposal from LLM output, supporting markdown fences and loose text."""
    if not raw_text or not raw_text.strip():
        return None

    # 1. Try direct json.loads
    try:
        parsed = json.loads(raw_text.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2. Extract markdown fenced code blocks (```json ... ``` or ``` ...)
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 3. Search for outermost JSON object with regex
    obj_match = re.search(r"(\{[\s\S]*\})", raw_text)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


@ChiaFunction(resources={"agent_worker": 1, "antigravity_creds": 0.01})
def node8_phase1_diagnostic_proposal(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    error_logs: Optional[Any] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Node 8 Diagnostic Proposal (Phase 1).

    Agentic node using the Chia API to systematically diagnose compilation,
    numerical verification, or evaluation failures and synthesize a structured
    proposal (either rewrite_correction with specific nodes_to_rerun or debug_request).

    Exposes ResultDBTool FastMCP tool server to the agent to inspect candidate
    history and error commentary.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Candidate ID that failed evaluation or compilation.
        error_logs: Log output or failure details from preceding nodes.
        llm: Optional LLM instance or mock.

    Returns:
        Dict specifying:
            - type: "DEBUG_MODE_REQUEST" or "REWRITE_CORRECTION"
            - action: "debug_request" or "rewrite_correction"
            - candidate_id: str
            - nodes_to_rerun: List[int]
            - debug_request: Optional[Dict[str, Any]]
            - diagnosis: str
            - reasoning: str
            - tokens_consumed: int
            - token_usage: Dict[str, int]
            - next_node: "node0_phase1_gate"
    """
    active_llm = _resolve_llm(llm, config)
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    # Instantiate ResultDBTool FastMCP interface
    result_db_tool = ResultDBTool(name="result_db", db=db, deploy=False)

    comments = db.get_comments(candidate_id)
    logger.info("Node 8: Formulating diagnostic proposal for candidate %s (%d previous comments)", candidate_id, len(comments))

    logs_str = str(error_logs or proposal.get("error_message") or "")
    c1 = proposal.get("counter_1_compilation", 0)
    c2 = proposal.get("counter_2_rewrite", 0)
    c4 = proposal.get("counter_4_debug_loop", 0)
    is_debug = bool(proposal.get("is_debug"))

    # Construct Agentic Diagnostic Prompt
    recent_comments_str = "\n".join(
        f"- [{c.get('phase', 'phase_1')}/{c.get('agent_name', 'agent')}]: {c.get('comment', '')}"
        for c in comments[-5:]
    ) if comments else "None"

    prompt_text = f"""You are the Phase 1 Diagnostic & Root-Cause Analysis Agent in an autonomous analog neural network co-design loop.

Candidate ID: {candidate_id}
Candidate Specification:
- Solver: {proposal.get('solver')}
- Noise Model: {proposal.get('noise_model')}
- Sparsity Config: {json.dumps(proposal.get('sparsity_config', {}))}
- Compilation Retries (Counter 1): {c1}
- Rewrite Retries (Counter 2): {c2}
- Debug Retries (Counter 4): {c4}
- Is Debug Candidate: {is_debug}

Failure Logs / Details:
{logs_str}

Recent Database Comments for Candidate:
{recent_comments_str}

Available Tools:
- result_db: You have access to the ResultDB MCP tool server to inspect candidate history or prior proposals.

Task Instructions:
1. Carefully reason about the failure root cause:
   - Was it a PyTorch / Inductor compilation failure (syntax, unsupported dynamic shape, lowering exception)?
   - Was it a numerical correctness verification failure between eager and compiled models?
   - Was it a simulation profiling failure or numerical divergence / out of tolerance on trajectory relative_error or absolute_error?
   - Has this candidate failed multiple rewrites already (counter_2 >= 2)?
2. Formulate a diagnostic strategy:
   - If multiple rewrites have failed (e.g. counter_2 >= 2) and the candidate is not already in debug mode, recommend "debug_request" to isolate whether solver or noise caused the issue by deactivating noise (`"deactivate": "deactivate_noise"`).
   - Otherwise, recommend "rewrite_correction" and select the minimal necessary nodes to rerun from:
     * Node 1: Solver Rewriter (rerun if solver formula, step size, or dynamics integration is broken)
     * Node 2: Noise Rewriter (rerun if noise model parameter, tensor shape, or thermal noise is causing failure)
     * Node 3: Compiler Rewriter (rerun if HOP functionalization or Inductor compilation failed)
     * Note: Node 3 is always rerun if Node 1 or Node 2 is rerun.
3. Respond with a valid JSON object matching this schema:
{{
  "action": "rewrite_correction" | "debug_request",
  "nodes_to_rerun": [1, 2, 3] | [3] | [2, 3],
  "diagnosis": "<Concise summary of the root cause>",
  "reasoning": "<In-depth technical reasoning explaining why the failure occurred and why the selected action resolves it>",
  "debug_request": {{
    "debugee_id": "{candidate_id}",
    "deactivate": "deactivate_noise",
    "deactivated_rewrites": ["noise"]
  }}
}}
Note: "debug_request" field must be null or omitted if action is "rewrite_correction".
"""

    start_time = time.time()
    query_result = None
    retry_query_result = None
    tools = [result_db_tool]
    try:
        try:
            if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
                query_result = active_llm.prompt(prompt_text, tools=tools)
        except Exception as exc:
            logger.warning("Node 8 LLM prompt failed: %s", exc)

        # Extract raw response text from attempt 1
        raw_response = ""
        if query_result is not None:
            raw_response = getattr(query_result, "result", None) or getattr(query_result, "stream_result", None) or str(query_result)
        elif hasattr(active_llm, "history") and active_llm.history:
            raw_response = str(active_llm.history[-1].get("response", ""))

        parsed_json = _parse_llm_json(raw_response)

        # Re-prompt trial: If attempt 1 returned None, empty, or non-parsable output, ask the LLM once more to return valid JSON
        if not (parsed_json and isinstance(parsed_json, dict) and "action" in parsed_json):
            logger.info(
                "Node 8: Initial LLM response for candidate %s was not valid JSON. Requesting a retry with strict JSON formatting...",
                candidate_id,
            )
            retry_prompt = (
                f"Your previous response for candidate {candidate_id} could not be parsed as valid JSON.\n"
                f"Previous output snippet: {raw_response[:200] if raw_response else '(empty / None)'}\n\n"
                f"Please re-generate your diagnostic proposal now. You MUST output ONLY a valid JSON object matching this schema:\n"
                f"{{\n"
                f'  "action": "rewrite_correction" | "debug_request",\n'
                f'  "nodes_to_rerun": [1, 2, 3] | [3] | [2, 3],\n'
                f'  "diagnosis": "<Concise summary of the root cause>",\n'
                f'  "reasoning": "<Technical reasoning explaining root cause and resolution>",\n'
                f'  "debug_request": null\n'
                f"}}\n"
                f'Do not include markdown or text outside the JSON object. "debug_request" must be null or omitted if action is "rewrite_correction".'
            )
            try:
                if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
                    retry_query_result = active_llm.prompt(retry_prompt, tools=tools)
            except Exception as exc:
                logger.warning("Node 8 LLM retry prompt failed: %s", exc)

            retry_raw_response = ""
            if retry_query_result is not None:
                retry_raw_response = getattr(retry_query_result, "result", None) or getattr(retry_query_result, "stream_result", None) or str(retry_query_result)
            elif hasattr(active_llm, "history") and active_llm.history:
                retry_raw_response = str(active_llm.history[-1].get("response", ""))

            if retry_raw_response:
                parsed_json = _parse_llm_json(retry_raw_response)
                if parsed_json and isinstance(parsed_json, dict) and "action" in parsed_json:
                    raw_response = retry_raw_response
    finally:
        cleanup_tools(tools)

    elapsed = max(0.1, time.time() - start_time)

    # Process parsed JSON or apply reasoning fallback with warning log
    if parsed_json and isinstance(parsed_json, dict) and "action" in parsed_json:
        action = str(parsed_json.get("action", "rewrite_correction")).lower()
        if action not in ("rewrite_correction", "debug_request"):
            action = "rewrite_correction"
        nodes_to_rerun = parsed_json.get("nodes_to_rerun") or ([1, 2, 3] if action == "rewrite_correction" else [3])
        if not isinstance(nodes_to_rerun, list):
            nodes_to_rerun = [1, 2, 3]
        diagnosis_text = str(parsed_json.get("diagnosis") or f"Diagnostic proposal for {candidate_id}")
        reasoning_text = str(parsed_json.get("reasoning") or diagnosis_text)
        debug_request_dict = parsed_json.get("debug_request")
    else:
        # LLM continues to generate non-parsable format after retry: alert with a warning log
        logger.warning(
            "Node 8: LLM failed to produce valid JSON diagnostic proposal for candidate %s after 2 trials. "
            "Falling back to rule-based diagnostic reasoning. Raw response: %s",
            candidate_id,
            raw_response[:300] if raw_response else "(empty / None)",
        )

        # Fallback reasoning
        if c2 >= 2 and not is_debug:
            action = "debug_request"
            nodes_to_rerun = [3]
            debug_request_dict = {
                "debugee_id": candidate_id,
                "deactivate": "deactivate_noise",
                "deactivated_rewrites": ["noise"],
            }
            diagnosis_text = f"Multiple rewrite attempts failed (counter_2={c2}). Entering Debug Mode with deactivated noise."
            reasoning_text = f"Candidate '{candidate_id}' exceeded normal rewrite tolerance with {c2} failed iterations. Deactivating noise to isolate underlying ODE solver behavior."
        elif "compil" in logs_str.lower() or "inductor" in logs_str.lower() or "hop" in logs_str.lower() or c1 > 0:
            action = "rewrite_correction"
            nodes_to_rerun = [3] if c2 <= 1 else [1, 2, 3]
            debug_request_dict = None
            diagnosis_text = f"Compilation / HOP lowering issue detected ({logs_str[:120]})."
            reasoning_text = f"Compilation failure indicated by error log: {logs_str[:200]}. Rerunning compiler rewrite node(s) {nodes_to_rerun}."
        else:
            action = "rewrite_correction"
            nodes_to_rerun = [1, 2, 3]
            debug_request_dict = None
            diagnosis_text = f"Evaluation / numerical tolerance discrepancy ({logs_str[:120]})."
            reasoning_text = f"Candidate failed numerical checks: {logs_str[:200]}. Rerunning full rewrite pipeline [1, 2, 3] to align solver and noise implementations."

    if action == "debug_request" and not debug_request_dict:
        debug_request_dict = {
            "debugee_id": candidate_id,
            "deactivate": "deactivate_noise",
            "deactivated_rewrites": ["noise"],
        }

    # Ensure Node 3 is always in nodes_to_rerun
    if 3 not in nodes_to_rerun:
        nodes_to_rerun.append(3)
    nodes_to_rerun = sorted(list(set(nodes_to_rerun)))

    # Formulate compiler rewrite steering instructions for Node 3
    is_compiler_failure = (
        "compil" in logs_str.lower()
        or "inductor" in logs_str.lower()
        or "hop" in logs_str.lower()
        or "node3" in logs_str.lower()
        or "compiler" in logs_str.lower()
        or c1 > 0
    )
    compiler_steering_instructions = None
    if parsed_json and isinstance(parsed_json, dict) and parsed_json.get("compiler_steering_instructions"):
        compiler_steering_instructions = str(parsed_json["compiler_steering_instructions"]).strip()
    elif is_compiler_failure:
        compiler_steering_instructions = (
            f"CRITICAL STEERING FROM NODE 8 DIAGNOSTICS (Compilation Cycle #{c1 + 1}):\n"
            f"The previous compiler rewrite attempt failed to register a valid rewrite in COMPILER_REGISTRY.\n"
            f"Failure context: {logs_str[:250]}\n"
            f"Diagnosis: {diagnosis_text}\n"
            f"STEERING DIRECTIVE FOR NODE 3: You MUST immediately call `un0_register_compiler_rewrite` with:\n"
            f"- candidate_id: '{candidate_id}'\n"
            f"- rewrite_name: '{candidate_id}_compiler_rewrite'\n"
            f"- implementation: Use `un0.compiler.functional_hop.functionalize_kuramoto_model` with `use_hop=False` "
            f"for Ahead-Of-Time TorchInductor compilation without graph breaks.\n"
            f"Do NOT execute background test runs or explore filesystem directories; call the registration tool directly."
        )

    if compiler_steering_instructions:
        db.add_comment(
            candidate_id=candidate_id,
            phase="phase_1",
            agent_name="node8_phase1_diagnostic_proposal",
            comment=f"[NODE8_STEERING_NODE3]\n{compiler_steering_instructions}",
        )
        if debug_request_dict:
            debug_request_dict["compiler_steering_instructions"] = compiler_steering_instructions

    # Record commentary in database
    db.add_comment(
        candidate_id=candidate_id,
        phase="phase_1",
        agent_name="node8_phase1_diagnostic_proposal",
        comment=f"Action: {action}. Diagnosis: {diagnosis_text}\nReasoning: {reasoning_text}",
    )

    # Extract Token Metrics per Node 1 convention (accumulating across initial and retry attempts if applicable)
    model_name = getattr(active_llm, "model", "gemini-2.5-flash")
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    def _extract_usage(res: Any) -> Dict[str, int]:
        u = getattr(res, "usage", None) if res is not None else None
        p, c, t = 0, 0, 0
        if isinstance(u, dict):
            p = u.get("prompt_tokens") or u.get("input_tokens") or 0
            c = u.get("completion_tokens") or u.get("output_tokens") or u.get("candidates_tokens") or 0
            t = u.get("total_tokens") or (p + c)
        return {"prompt": p, "completion": c, "total": t}

    u1 = _extract_usage(query_result)
    prompt_tokens += u1["prompt"]
    completion_tokens += u1["completion"]
    total_tokens += u1["total"]

    if retry_query_result is not None:
        u2 = _extract_usage(retry_query_result)
        prompt_tokens += u2["prompt"]
        completion_tokens += u2["completion"]
        total_tokens += u2["total"]
    elif total_tokens == 0 and hasattr(active_llm, "history") and active_llm.history:
        # Fallback to history if query_result did not directly attach usage dict
        u_hist = active_llm.history[-1].get("usage")
        if isinstance(u_hist, dict):
            prompt_tokens = u_hist.get("prompt_tokens") or u_hist.get("input_tokens") or 0
            completion_tokens = u_hist.get("completion_tokens") or u_hist.get("output_tokens") or u_hist.get("candidates_tokens") or 0
            total_tokens = u_hist.get("total_tokens") or (prompt_tokens + completion_tokens)

    logger.info(
        "Node 8: Tokens consumed - prompt: %d, completion: %d, total: %d (model: %s)",
        prompt_tokens,
        completion_tokens,
        total_tokens,
        model_name,
    )

    if total_tokens > 0:
        db.record_token_usage(
            phase="phase_1",
            node_name="node8_phase1_diagnostic_proposal",
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=0.0,  # Convention: do not estimate monetary cost; report token counts only
            candidate_id=candidate_id,
            duration_seconds=elapsed,
        )

    # Profiler registration
    aggregated_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    try:
        from chia.base.profiling import get_profiler
        get_profiler().add_info({
            "node": "node8_phase1_diagnostic_proposal",
            "candidate_id": candidate_id,
            "usage": aggregated_usage,
        })
    except Exception:
        pass

    return {
        "candidate_id": candidate_id,
        "type": "DEBUG_MODE_REQUEST" if action == "debug_request" else "REWRITE_CORRECTION",
        "action": action,
        "nodes_to_rerun": nodes_to_rerun,
        "debug_request": debug_request_dict if action == "debug_request" else None,
        "diagnosis": diagnosis_text,
        "reasoning": reasoning_text,
        "compiler_steering_instructions": compiler_steering_instructions,
        "tokens_consumed": total_tokens,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "next_node": "node0_phase1_gate",
    }


# Backwards compatibility alias
node8_diagnostic_proposal = node8_phase1_diagnostic_proposal
