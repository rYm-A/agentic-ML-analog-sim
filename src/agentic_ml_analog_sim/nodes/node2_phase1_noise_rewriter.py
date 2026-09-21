"""Node 2: Noise Rewriter Agentic Node (Phase 1).

Applies or registers noise wrapper rewrite for Un-0 model and updates database status.
Implements deterministic reuse when noise model is already registered (with null token consumption)
and agentic LLM rewrite using the Chia API and FastMCP tools (ResultDBTool, NoisePapersTool, Un0RewriterTool)
when noise model is unregistered or forced, capturing real token metrics.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

from chia.base.ChiaFunction import ChiaFunction

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
from agentic_ml_analog_sim.tools.noise_papers_tool import NoisePapersTool
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool
from agentic_ml_analog_sim.tools.un0_context_tool import resolve_un0_root
from agentic_ml_analog_sim.tools.un0_rewriter_tool import Un0RewriterTool

logger = logging.getLogger("agentic_ml_analog_sim.nodes.node2_phase1_noise_rewriter")


def _sync_noise_rewrites_from_disk(un0_dir: Path, candidate_id: Optional[str] = None) -> List[str]:
    """Scan Un-0 noise rewrites directory for newly written modules and load them into NOISE_REGISTRY."""
    noise_dir = un0_dir / "un0" / "noise" / "rewrites"
    if not noise_dir.is_dir():
        return []

    discovered = []
    clean_cand = candidate_id.strip().replace(" ", "_") if candidate_id else None

    try:
        from un0.noise import NOISE_REGISTRY
    except ImportError:
        return []

    all_files = [p for p in noise_dir.glob("*.py") if p.name != "__init__.py"]
    if clean_cand:
        matching = [p for p in all_files if p.name.startswith(f"{clean_cand}_") or clean_cand in p.name]
        others = [p for p in all_files if p not in matching]
        ordered_files = sorted(matching, key=lambda p: p.stat().st_mtime, reverse=True) + sorted(others, key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        ordered_files = sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)

    for fpath in ordered_files:
        stem = fpath.stem
        module_name = f"un0.noise.rewrites.{stem}"
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

            if potential_name not in NOISE_REGISTRY:
                for attr_name in dir(mod):
                    val = getattr(mod, attr_name)
                    if isinstance(val, type) and hasattr(val, "forward"):
                        NOISE_REGISTRY[potential_name] = val
                        break

            if potential_name in NOISE_REGISTRY:
                discovered.append(potential_name)
        except Exception as err:
            logger.debug("Node 2: Error syncing noise rewrite %s from disk: %s", fpath, err)

    return discovered


def _resolve_llm(llm: Optional[Any] = None, config: Optional[SimulationConfig] = None) -> Any:
    if llm is not None:
        return llm

    agent_efforts = getattr(config, "agent_efforts", None)
    node_cfg = None
    if agent_efforts is not None:
        node_cfg = getattr(agent_efforts, "node2_phase1_noise_rewriter", None)
        if node_cfg is None and isinstance(agent_efforts, dict):
            node_cfg = agent_efforts.get("node2_phase1_noise_rewriter")

    model = getattr(node_cfg, "model", None) or (node_cfg.get("model") if isinstance(node_cfg, dict) else None) or "gemini-2.5-pro"
    effort = getattr(node_cfg, "effort", None) or (node_cfg.get("effort") if isinstance(node_cfg, dict) else None) or "high"
    extra_cli_args = ["--effort", effort]

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, extra_cli_args=extra_cli_args)
    except Exception:
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


def _extract_registered_noise_from_text(raw_text: str) -> Optional[str]:
    """Parse JSON or string output from agent response to extract registered noise model name."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    # 1. Direct JSON parse
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, dict):
            val = data.get("noise_name") or data.get("rewrite_name") or data.get("registered_noise")
            if val:
                return str(val).strip()
    except Exception:
        pass

    # 2. Markdown fenced JSON block
    match_fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match_fenced:
        try:
            data = json.loads(match_fenced.group(1).strip())
            if isinstance(data, dict):
                val = data.get("noise_name") or data.get("rewrite_name") or data.get("registered_noise")
                if val:
                    return str(val).strip()
        except Exception:
            pass

    # 3. Try finding any JSON object in text
    match_obj = re.search(r"\{[^{}]*(?:noise_name|rewrite_name|registered_noise)[^{}]*\}", raw_text, re.DOTALL)
    if match_obj:
        try:
            data = json.loads(match_obj.group(0).strip())
            if isinstance(data, dict):
                val = data.get("noise_name") or data.get("rewrite_name") or data.get("registered_noise")
                if val:
                    return str(val).strip()
        except Exception:
            pass

    # 4. Regex string capture for noise_name / rewrite_name
    match_key = re.search(r'["\']?(?:noise_name|rewrite_name|registered_noise)["\']?\s*:\s*["\']([^"\']+)["\']', raw_text)
    if match_key:
        return match_key.group(1).strip()

    return None


@ChiaFunction(resources={"agent_worker": 1, "antigravity_creds": 0.01})
def node2_phase1_noise_rewriter(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    llm: Optional[Any] = None,
    force_rewrite: bool = False,
) -> Dict[str, Any]:
    """Node 2 Noise Rewriter (Phase 1).

    Deterministic behavior:
        If noise_model is 'none' or already registered in NOISE_REGISTRY and no rewrite is forced,
        the node reuses the registered noise class directly. No LLM call is made,
        and token consumption is null (0 tokens recorded).

    Agentic behavior:
        If the noise model is not registered or forced, invokes the LLM agent using the Chia API
        equipped with three MCP tools:
        1. result_db: query proposal details and record cross-phase comments.
        2. noise_papers: query physical hardware noise modeling skill (Tiers 0-5), ONN channels, and literature.
        3. un0_rewriter: inspect Un-0 codebase and register the synthesized noise wrapper directly
           via `un0_register_noise_rewrite`.

        Real token consumption metrics are captured from the agent framework and
        persisted to the ResultDatabase.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        llm: Optional LLM instance or mock.
        force_rewrite: Whether to force an agentic LLM rewrite even if registered.

    Returns:
        Dict detailing candidate_id, execution status, mode, and selected noise model.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    noise_name = proposal.get("noise_model")
    if noise_name is None:
        ref_design = getattr(config, "reference_design", None)
        noise_name = getattr(ref_design, "noise_model", "none") if ref_design else "none"

    is_none = not noise_name or noise_name.lower() == "none"
    logger.info("Node 2: Checking noise model '%s' for candidate %s", noise_name, candidate_id)

    # Check noise registry
    try:
        from un0.noise import NOISE_REGISTRY, get_noise_model, register_noise_model
    except Exception as exc:
        logger.warning("Un-0 noise registry import issue: %s", exc)
        NOISE_REGISTRY = {}
        get_noise_model = None
        register_noise_model = None

    is_registered = is_none or bool(get_noise_model and noise_name in NOISE_REGISTRY)

    # Deterministic branch: 'none' or already registered and not forced
    if is_registered and not force_rewrite:
        logger.info(
            "Node 2: Noise model '%s' is handled deterministically (null token consumption).",
            noise_name,
        )
        noise_cls = None if is_none else get_noise_model(noise_name)

        db.update_execution_status(
            candidate_id,
            rewrite_noise_status="SUCCESS",
        )

        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "noise_model": noise_name,
            "noise_cls": noise_cls,
            "deterministic": True,
            "tokens_consumed": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "next_node": "node3_phase1_compiler_rewriter",
        }

    # Nondeterministic / Agentic branch: unregistered noise model or forced rewrite
    logger.info(
        "Node 2: Noise model '%s' requires agentic implementation/rewrite. Invoking LLM via Chia API.",
        noise_name,
    )

    # 1. Instantiate the 3 Chia Tools
    un0_dir = resolve_un0_root()

    result_db_tool = ResultDBTool(name="result_db", db=db, deploy=False)
    noise_papers_tool = NoisePapersTool(name="noise_papers", auto_start=False)
    un0_rewriter_tool = Un0RewriterTool(name="un0_rewriter", repo_root=un0_dir, auto_start=False)

    # Extract proposal parameters and instructions from previous nodes
    noise_params = proposal.get("noise_params") or {}
    if isinstance(noise_params, str):
        try:
            noise_params = json.loads(noise_params)
        except Exception:
            noise_params = {}

    solver_name = proposal.get("solver", "euler")
    phase_1_prompt = proposal.get("phase_1_prompt") or ""

    # 2. Formulate Agentic Prompt
    prompt_text = (
        f"# Phase 1 Hardware Noise Rewriter Agent (Node 2)\n\n"
        f"You are the Hardware Noise Rewriter Agent for Phase 1 of the Un-0 Analog ML simulation loop.\n"
        f"Your task is to dynamically implement and register a hardware noise wrapper model named '{noise_name}' "
        f"for Kuramoto physical analog simulation in the Un-0 repository, matching the candidate specifications.\n\n"
        f"Candidate Specifications:\n"
        f"- Candidate ID: {candidate_id}\n"
        f"- Target Noise Model: {noise_name}\n"
        f"- Noise Parameters: {json.dumps(noise_params)}\n"
        f"- Solver Scheme: {solver_name}\n"
        f"- Exploration Prompt / Context: {phase_1_prompt}\n\n"
        f"Available Chia Tools:\n"
        f"1. `result_db`: query proposal details and record cross-phase comments via `add_comment`.\n"
        f"2. `noise_papers`: access hardware noise modeling skill guidance via `get_noise_tier_guide(tier)` "
        f"(Tier 0 static mismatch, Tier 1 stochastic/thermal, Tier 2 interface quantization/DAC/ADC, "
        f"Tier 3 composite drift/aging, Tier 4 physics-informed/SDE, Tier 5 behavioral twin), "
        f"`get_onn_noise_channels()`, and literature citations via `get_noise_literature_references()`.\n"
        f"3. `un0_rewriter`: inspect existing Un-0 noise wrappers (`un0/noise/`), dynamics definitions, "
        f"and register your new noise wrapper implementation into Un-0 using `un0_register_noise_rewrite`.\n\n"
        f"Implementation Directives:\n"
        f"- The wrapper must inherit from `torch.nn.Module`.\n"
        f"- Wrap the underlying dynamics (e.g. `self.dynamics = dynamics`).\n"
        f"- Implement `forward(self, state: torch.Tensor, t: torch.Tensor, drive: torch.Tensor) -> torch.Tensor`.\n"
        f"- Apply noise perturbations according to the requested physical tier/mechanism.\n"
        f"- Register your wrapper using `un0_rewriter.un0_register_noise_rewrite(\n"
        f"    noise_name='{noise_name}',\n"
        f"    candidate_id='{candidate_id}',\n"
        f"    code='...python code implementing the nn.Module and decorated with @register_noise_model...',\n"
        f"    description='...summary of physical noise modeling mechanism...'\n"
        f")`.\n"
        f"- Respond with a JSON object: {{\"status\": \"SUCCESS\", \"noise_name\": \"{noise_name}\", \"description\": \"...\"}}."
    )

    active_llm = _resolve_llm(llm, config)
    start_time = time.time()
    query_result = None
    tools = [result_db_tool, noise_papers_tool, un0_rewriter_tool]

    try:
        if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
            query_result = active_llm.prompt(
                prompt_text,
                tools=tools,
            )
    except Exception as exc:
        logger.warning("Node 2: LLM prompt call failed: %s", exc)
    finally:
        cleanup_tools(tools)

    elapsed = max(0.1, time.time() - start_time)

    # 3. Extract REAL token metrics from query_result or LLM history
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
        "Node 2: Tokens consumed - prompt: %d, completion: %d, total: %d (model: %s)",
        prompt_tokens,
        completion_tokens,
        total_tokens,
        model_name,
    )

    if total_tokens > 0:
        cost_usd = (prompt_tokens * 0.00000125) + (completion_tokens * 0.000005)
        db.record_token_usage(
            phase="phase_1",
            node_name="node2_phase1_noise_rewriter",
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            candidate_id=candidate_id,
            duration_seconds=elapsed,
        )

    # 4. Resolve registered noise model from tool, disk sync, or registry
    _sync_noise_rewrites_from_disk(un0_dir, candidate_id=candidate_id)
    registered_noise_name = un0_rewriter_tool.get_registered_noise(candidate_id=candidate_id)

    # Secondary lookup: check filesystem matching candidate_id directly
    if not registered_noise_name:
        noise_dir = un0_dir / "un0" / "noise" / "rewrites"
        clean_cand = candidate_id.strip().replace(" ", "_")
        try:
            from un0.noise import NOISE_REGISTRY
            if noise_dir.is_dir():
                matching_files = sorted(
                    [p for p in noise_dir.glob(f"{clean_cand}_*.py") if p.name != "__init__.py"],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for fpath in matching_files:
                    stem = fpath.stem
                    pot_name = stem[len(clean_cand) + 1:] if stem.startswith(f"{clean_cand}_") else stem
                    if pot_name in NOISE_REGISTRY:
                        registered_noise_name = pot_name
                        break
        except Exception:
            pass

    if not registered_noise_name and query_result is not None:
        raw_res = getattr(query_result, "result", str(query_result))
        registered_noise_name = _extract_registered_noise_from_text(raw_res)

    if not registered_noise_name:
        registered_noise_name = noise_name

    # Check if the class is now available in NOISE_REGISTRY
    noise_cls = None
    if get_noise_model and registered_noise_name in NOISE_REGISTRY:
        noise_cls = get_noise_model(registered_noise_name)
    elif get_noise_model and noise_name in NOISE_REGISTRY:
        noise_cls = get_noise_model(noise_name)

    if noise_cls is None:
        error_msg = (
            f"Node 2: Agent failed to register noise model '{noise_name}' "
            f"into NOISE_REGISTRY for candidate '{candidate_id}'."
        )
        logger.error(error_msg)
        db.update_execution_status(
            candidate_id,
            rewrite_noise_status="FAILED",
            error_stage="node2_phase1_noise_rewriter",
            error_message=error_msg,
        )
        raise RuntimeError(error_msg)

    # 5. Update DB status
    db.update_execution_status(
        candidate_id,
        rewrite_noise_status="SUCCESS",
    )

    return {
        "candidate_id": candidate_id,
        "status": "SUCCESS",
        "noise_model": registered_noise_name,
        "noise_cls": noise_cls,
        "deterministic": False,
        "tokens_consumed": total_tokens,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "next_node": "node3_phase1_compiler_rewriter",
    }


# Backwards compatibility alias
node2_noise_rewriter = node2_phase1_noise_rewriter
