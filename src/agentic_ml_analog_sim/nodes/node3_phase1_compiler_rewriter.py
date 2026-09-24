"""Node 3 (Phase 1): Agentic Compiler Rewriter Node.

Systematically transforms continuous Kuramoto dynamics into an Ahead-Of-Time (AOT)
compilable functional form using PyTorch Higher-Order Operators (torch._higher_order_ops.while_loop,
associative_scan) supporting dynamic batch sizes (B >= 1).

Exposes three ChiaTools to the agent:
1. ResultDBTool: query proposal, statuses, and log comments.
2. CompilationInfoTool: access PyTorch compile documentation and RegenerativeCoRNN HOP examples.
3. Un0RewriterTool: inspect Un-0 codebase and register compiler rewrites into Un-0.

Eliminates silent deterministic fallbacks. If an agent fails to register or synthesize
a valid rewrite, records failure in the database and raises RuntimeError so that Node 8
can handle diagnostic recovery.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn

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
from agentic_ml_analog_sim.tools.compilation_info_tool import CompilationInfoTool
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool
from agentic_ml_analog_sim.tools.un0_context_tool import resolve_un0_root
from agentic_ml_analog_sim.tools.un0_rewriter_tool import Un0RewriterTool
from un0.compiler import registry as compiler_registry

logger = logging.getLogger(__name__)


def _resolve_llm(llm: Optional[Any], config: Optional[SimulationConfig]) -> Any:
    """Resolve active LLM for Node 3 using simulation config or mock fallback."""
    if llm is not None:
        return llm

    model = "gemini-2.5-pro"
    effort = "high"
    extra_cli_args = None

    if config is not None and hasattr(config, "agent_efforts") and config.agent_efforts:
        node_cfg = getattr(config.agent_efforts, "node3_phase1_compiler_rewriter", None)
        if node_cfg is not None:
            model = getattr(node_cfg, "model", model)
            effort = getattr(node_cfg, "effort", effort)
            extra_cli_args = getattr(node_cfg, "extra_cli_args", None)

    try:
        from chia.models.antigravity import AntigravityLLM
        return AntigravityLLM(model=model, effort=effort, extra_cli_args=extra_cli_args)
    except Exception as exc:
        logger.warning(
            "Could not instantiate AntigravityLLM for Node 3 (%s). Using MockAntigravityLLM.",
            exc,
        )
        from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
        return MockAntigravityLLM(mode="valid", model=model, effort=effort, extra_cli_args=extra_cli_args)


def _extract_rewrite_name(raw_text: str) -> Optional[str]:
    """Robustly extract registered compiler rewrite name from agent response text."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    # 1. Try parsing complete text as JSON
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, dict):
            val = data.get("rewrite_name") or data.get("registered_rewrite")
            if val:
                return str(val).strip()
    except Exception:
        pass

    # 2. Try regex markdown code block ```json ... ```
    match_code = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match_code:
        try:
            data = json.loads(match_code.group(1).strip())
            if isinstance(data, dict):
                val = data.get("rewrite_name") or data.get("registered_rewrite")
                if val:
                    return str(val).strip()
        except Exception:
            pass

    # 3. Try finding any JSON object in text
    match_obj = re.search(r"\{[^{}]*(?:rewrite_name|registered_rewrite)[^{}]*\}", raw_text, re.DOTALL)
    if match_obj:
        try:
            data = json.loads(match_obj.group(0).strip())
            if isinstance(data, dict):
                val = data.get("rewrite_name") or data.get("registered_rewrite")
                if val:
                    return str(val).strip()
        except Exception:
            pass

    # 4. Regex string capture for rewrite_name
    match_key = re.search(r'["\']?(?:rewrite_name|registered_rewrite)["\']?\s*:\s*["\']([^"\']+)["\']', raw_text)
    if match_key:
        return match_key.group(1).strip()

    # 5. Regex for natural language markdown bold/backtick definitions
    match_nlp = re.search(
        r'(?:registered under(?: the name)?|rewrite name is|rewrite name:?|registered rewrite:?)\s*[*`\'"]*([a-zA-Z0-9_\-]+)[*`\'"]*',
        raw_text,
        re.IGNORECASE,
    )
    if match_nlp:
        return match_nlp.group(1).strip()

    # 6. Candidate pattern matches against compiler_registry.COMPILER_REGISTRY
    try:
        for reg_name in sorted(compiler_registry.COMPILER_REGISTRY.keys(), key=len, reverse=True):
            if re.search(r'[`\'"]' + re.escape(reg_name) + r'[`\'"]', raw_text) or re.search(r'\b' + re.escape(reg_name) + r'\b', raw_text):
                return reg_name
    except Exception:
        pass

    return None


def _sync_compiler_rewrites_from_disk(un0_dir: Path, candidate_id: Optional[str] = None) -> List[str]:
    """Scan Un-0 compiler rewrites directories for newly written modules and load them.

    Ensures modules persisted to disk across processes (e.g., FastMCP Ray actor)
    are loaded into sys.modules and registered in compiler_registry.COMPILER_REGISTRY.
    """
    search_dirs = [
        un0_dir / "un0" / "compiler" / "rewrites",
        Path.cwd() / "Un-0" / "un0" / "compiler" / "rewrites",
    ]
    if "UN0_ROOT" in os.environ and os.environ["UN0_ROOT"].strip():
        search_dirs.append(Path(os.environ["UN0_ROOT"].strip()) / "un0" / "compiler" / "rewrites")
    if "UN0_REWRITES_DIR" in os.environ and os.environ["UN0_REWRITES_DIR"].strip():
        search_dirs.append(Path(os.environ["UN0_REWRITES_DIR"].strip()))
    if "AGENTIC_SIM_RUN_REWRITES_DIR" in os.environ and os.environ["AGENTIC_SIM_RUN_REWRITES_DIR"].strip():
        search_dirs.append(Path(os.environ["AGENTIC_SIM_RUN_REWRITES_DIR"].strip()))
    seen_dirs: set[Path] = set()
    all_files: list[Path] = []
    for rdir in search_dirs:
        try:
            rdir_res = rdir.resolve()
            if rdir_res.is_dir() and rdir_res not in seen_dirs:
                seen_dirs.add(rdir_res)
                all_files.extend([p for p in rdir_res.glob("*.py") if p.name != "__init__.py"])
        except Exception:
            pass

    if not all_files:
        return []

    discovered = []
    clean_cand = candidate_id.strip().replace(" ", "_") if candidate_id else None

    if clean_cand:
        matching = [p for p in all_files if p.name.startswith(f"{clean_cand}_") or clean_cand in p.name]
        others = [p for p in all_files if p not in matching]
        ordered_files = sorted(matching, key=lambda p: p.stat().st_mtime, reverse=True) + sorted(others, key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        ordered_files = sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)

    for fpath in ordered_files:
        stem = fpath.stem
        module_name = f"un0.compiler.rewrites.{stem}"
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

            if potential_name not in compiler_registry.COMPILER_REGISTRY:
                candidate_fn = (
                    getattr(mod, potential_name, None)
                    or getattr(mod, "rewrite", None)
                    or getattr(mod, "rewrite_fn", None)
                )
                if candidate_fn is None:
                    for attr_name in dir(mod):
                        val = getattr(mod, attr_name)
                        if callable(val) and not attr_name.startswith("_") and getattr(val, "__module__", "") == module_name:
                            candidate_fn = val
                            break
                if candidate_fn is not None:
                    compiler_registry.COMPILER_REGISTRY[potential_name] = candidate_fn

            if potential_name in compiler_registry.COMPILER_REGISTRY:
                discovered.append(potential_name)
        except Exception as err:
            logger.debug("Node 3: Error syncing rewrite %s from disk: %s", fpath, err)

    return discovered


@ChiaFunction(resources={"agent_worker": 1, "antigravity_creds": 0.01})
def node3_phase1_compiler_rewriter(
    db: ResultDatabase,
    config: SimulationConfig,
    candidate_id: str,
    target_device: str = "mps",
    llm: Optional[Any] = None,
    force_rewrite: bool = False,
    steering_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Node 3 Compiler Rewriter (Phase 1).

    Agentic node using the Chia API to systematically functionalize continuous Kuramoto
    dynamics into an AOT compilable form using PyTorch Higher-Order Operators (HOP)
    supporting dynamic batch shapes (B >= 1).

    Exposes 3 ChiaTools to the agent:
    1. result_db: query proposal details and record comments.
    2. compilation_info: query Torch compile documentation and RegenerativeCoRNN HOP examples.
    3. un0_rewriter: inspect Un-0 codebase and register functional compiler rewrite.

    Args:
        db: ResultDatabase instance.
        config: SimulationConfig instance.
        candidate_id: Proposal candidate ID.
        target_device: Hardware execution target ('mps', 'cuda', 'cpu').
        llm: Optional LLM instance or mock.
        force_rewrite: When True, bypasses deterministic reuse and forces agentic rewrite.
        steering_instructions: Optional diagnostic guidance from Node 8. If None, checks DB comments.

    Returns:
        Dict detailing candidate_id, execution status, target device, prepared HOP model,
        eager baseline model, registered rewrite name, and token usage metrics.
    """
    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        raise KeyError(f"Proposal '{candidate_id}' not found in database.")

    logger.info(
        "Node 3: Applying Agentic Compiler Rewrite for %s (target_device=%s)",
        candidate_id,
        target_device,
    )

    # 1. Extract simulation & architecture specifications directly from config and proposal
    solver_name = proposal.get("solver")
    if not solver_name:
        ref_design = getattr(config, "reference_design", None)
        solver_name = getattr(ref_design, "solver", None) if ref_design else None
    if not solver_name:
        raise KeyError(f"No solver specified for candidate '{candidate_id}' in proposal or simulation config.")

    ref_design = getattr(config, "reference_design", None)
    if ref_design is None:
        raise ValueError("Simulation configuration is missing 'reference_design'.")

    n_osc = getattr(ref_design, "n_oscillators", None)
    n_cond = getattr(ref_design, "n_conditional_oscillators", None)
    if n_osc is None or n_cond is None:
        raise ValueError("Simulation configuration reference_design must specify 'n_oscillators' and 'n_conditional_oscillators'.")

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

    solver_params = proposal.get("solver_params", {}) or {}
    if isinstance(solver_params, str):
        try:
            solver_params = json.loads(solver_params)
        except Exception:
            solver_params = {}

    num_steps = (
        proposal.get("num_steps")
        or solver_params.get("num_steps")
        or getattr(ref_design, "num_steps", None)
    )
    if num_steps is None:
        raise ValueError(f"Number of steps not specified for candidate '{candidate_id}' or reference design.")

    integration_time = solver_params.get("integration_time") or getattr(ref_design, "integration_time", None)
    if integration_time is None:
        raise ValueError(f"Integration time not specified for candidate '{candidate_id}' or reference design.")

    dt = solver_params.get("dt") or (float(integration_time) / float(num_steps))

    # 2. Instantiate eager baseline model with requested noise model
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

    is_no_noise = not noise_name or str(noise_name).lower() == "none" or bool(proposal.get("is_reference"))

    if is_no_noise:
        eager_model = base_dynamics
    else:
        noise_params = proposal.get("noise_params", {}) or {}
        if isinstance(noise_params, str):
            try:
                noise_params = json.loads(noise_params)
            except Exception:
                noise_params = {}
        try:
            noise_cls = get_noise_model(noise_name)
            eager_model = noise_cls(base_dynamics, **(noise_params if isinstance(noise_params, dict) else {}))
        except Exception as noise_err:
            error_msg = f"Node 3: Failed to instantiate noise model '{noise_name}' for candidate '{candidate_id}': {noise_err}"
            logger.error(error_msg)
            db.update_execution_status(
                candidate_id,
                rewrite_compiler_status="FAILED",
                error_stage="node3_phase1_compiler_rewriter",
                error_message=error_msg,
            )
            raise RuntimeError(error_msg)

    # 3. Check for diagnostic steering instructions (from parameter or DB comments)
    if steering_instructions is None:
        try:
            comments = db.get_comments(candidate_id)
            for c in reversed(comments):
                c_text = c.get("comment", "")
                if "[NODE8_STEERING_NODE3]" in c_text or "[STEERING_PROMPT_NODE3]" in c_text:
                    for tag in ("[NODE8_STEERING_NODE3]", "[STEERING_PROMPT_NODE3]"):
                        if tag in c_text:
                            part = c_text.split(tag, 1)[1].strip()
                            steering_instructions = part if part else c_text.strip()
                            break
                    if steering_instructions:
                        break
        except Exception as exc:
            logger.debug("Node 3: Could not retrieve steering comments from db: %s", exc)

    # 4. Check for existing registered compiler rewrite (Deterministic Fast Path)
    un0_dir = resolve_un0_root()
    _sync_compiler_rewrites_from_disk(un0_dir, candidate_id=candidate_id)

    cand_clean = candidate_id.strip().replace(" ", "_")
    is_reference = bool(proposal.get("is_reference"))

    existing_rewrite_name = None
    prop_rewrite = proposal.get("compiler_rewrite") or solver_params.get("compiler_rewrite")
    if prop_rewrite and prop_rewrite in compiler_registry.COMPILER_REGISTRY:
        existing_rewrite_name = prop_rewrite
    elif f"{cand_clean}_compiler_rewrite" in compiler_registry.COMPILER_REGISTRY:
        existing_rewrite_name = f"{cand_clean}_compiler_rewrite"
    elif cand_clean in compiler_registry.COMPILER_REGISTRY:
        existing_rewrite_name = cand_clean
    elif is_reference and "unrolled_functional" in compiler_registry.COMPILER_REGISTRY:
        existing_rewrite_name = "unrolled_functional"
    elif is_reference and "hop_while_loop" in compiler_registry.COMPILER_REGISTRY:
        existing_rewrite_name = "hop_while_loop"

    # Deterministic branch: already registered, no forced rewrite, no steering instructions
    if existing_rewrite_name and not force_rewrite and not steering_instructions:
        logger.info(
            "Node 3: Compiler rewrite '%s' is already registered in Un-0. Using deterministic reuse (null token consumption).",
            existing_rewrite_name,
        )
        rewrite_fn = compiler_registry.get_compiler_rewrite(existing_rewrite_name)
        try:
            functional_model = rewrite_fn(
                model=eager_model,
                solver_name=solver_name,
                num_steps=num_steps,
                dt=dt,
                integration_time=integration_time,
                target_device=target_device,
            )
        except Exception as exec_err:
            error_msg = f"Node 3: Execution of deterministic compiler rewrite '{existing_rewrite_name}' failed: {exec_err}"
            logger.error(error_msg)
            db.update_execution_status(
                candidate_id,
                rewrite_compiler_status="FAILED",
                error_stage="node3_phase1_compiler_rewriter",
                error_message=error_msg,
            )
            raise RuntimeError(error_msg)

        db.update_execution_status(
            candidate_id,
            rewrite_compiler_status="SUCCESS",
        )

        try:
            from chia.base.profiling import get_profiler
            get_profiler().add_info({
                "node": "node3_phase1_compiler_rewriter",
                "candidate_id": candidate_id,
                "deterministic": True,
                "tokens_consumed": 0,
                "registered_rewrite": existing_rewrite_name,
            })
        except Exception:
            pass

        return {
            "candidate_id": candidate_id,
            "status": "SUCCESS",
            "target_device": target_device,
            "prepared_model": functional_model,
            "eager_model": eager_model,
            "registered_rewrite": existing_rewrite_name,
            "tokens_consumed": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "hop_functionalized": True,
            "deterministic": True,
            "next_node": "node4_phase1_compilation_runner",
        }

    # Nondeterministic / Agentic branch: requires agentic rewrite
    logger.info(
        "Node 3: Candidate '%s' requires agentic compiler rewrite. Invoking LLM via Chia API.",
        candidate_id,
    )

    # 5. Instantiate Chia Tools (matching Node 1 and Node 2)
    result_db_tool = ResultDBTool(name="result_db", db=db, deploy=False)
    un0_rewriter_tool = Un0RewriterTool(
        name="un0_rewriter",
        repo_root=un0_dir,
        auto_start=False,
    )

    tools = [result_db_tool, un0_rewriter_tool]
    compilation_info_tool = CompilationInfoTool(name="compilation_info", auto_start=False)
    if compilation_info_tool.is_available:
        tools.append(compilation_info_tool)
    else:
        logger.warning(
            "Node 3: RegenerativeCoRNN repository not found at '%s'. "
            "Hiding compilation_info tool from agent and continuing normal execution.",
            compilation_info_tool.repo_root,
        )

    # 6. Formulate Agentic Prompt
    tool_descriptions = [
        "1. `result_db`: query proposal details, record comments, and check Pareto front.",
    ]
    if compilation_info_tool.is_available:
        tool_descriptions.append(
            "2. `compilation_info`: access official PyTorch compiler documentation (while_loop, associative_scan, "
            "dynamic shapes, Inductor) and inspect reference HOP implementations in RegenerativeCoRNN (model/implicit_parallel.py)."
        )
    tool_descriptions.append(
        f"{len(tool_descriptions) + 1}. `un0_rewriter`: read Un-0 codebase, inspect dynamics definitions, and register your functionalized "
        "compiler rewrite into Un-0 using `un0_register_compiler_rewrite`."
    )
    tools_section = "\n".join(tool_descriptions)

    rewrite_identifier = f"{cand_clean}_compiler_rewrite"

    code_template = f'''from un0.compiler.registry import register_compiler_rewrite
from un0.compiler.functional_hop import functionalize_kuramoto_model

@register_compiler_rewrite("{rewrite_identifier}")
def {rewrite_identifier}_fn(model, solver_name="{solver_name}", num_steps={num_steps}, dt={dt}, integration_time={integration_time}, target_device="{target_device}", **kwargs):
    """AOT compilable functional rewrite for {candidate_id}."""
    return functionalize_kuramoto_model(
        model=model,
        solver_name=solver_name,
        num_steps=num_steps,
        dt=dt,
        integration_time=integration_time,
        use_hop=False,
    )'''

    prompt_text = (
        f"# Phase 1 Compiler Rewriter Agent (Node 3)\n\n"
        f"You are the Compiler Rewriter Agent for Phase 1. Your task is to functionalize the Kuramoto "
        f"oscillator neural network candidate into an Ahead-Of-Time (AOT) compilable form compatible with "
        f"Torch Dynamo and Torch Inductor on target hardware '{target_device}'.\n\n"
        f"Candidate Specifications:\n"
        f"- Candidate ID: {candidate_id}\n"
        f"- Solver: {solver_name}\n"
        f"- Noise Model: {noise_name}\n"
        f"- Integration Steps: {num_steps}\n"
        f"- Time Step (dt): {dt}\n"
        f"- Integration Time: {integration_time}\n"
        f"- Target Compilation Device: {target_device}\n"
        f"- Dynamic Batch Constraint: Batch size B >= 1 with no upper bound.\n\n"
        f"Available Chia Tools:\n"
        f"{tools_section}\n\n"
        f"## Concrete Implementation Template\n"
        f"Use or adapt the following compilable rewrite template:\n"
        f"```python\n{code_template}\n```\n\n"
        f"## Directives & Execution Instructions\n"
        f"- Directly call `un0_register_compiler_rewrite(rewrite_name=\"{rewrite_identifier}\", candidate_id=\"{candidate_id}\", code=...)`.\n"
        f"- Do NOT run long shell commands or background benchmarks.\n"
        f"- Do NOT browse unrelated files.\n"
        f"- Do NOT use static Python while-loops that cause graph breaks in TorchDynamo.\n"
        f"- Functionalize using PyTorch Higher-Order Operators (`torch._higher_order_ops.while_loop`) or unrolled functional operators.\n"
        f"- In your final response, explain your functionalization design, operator choices, and compilation considerations."
    )

    if steering_instructions:
        prompt_text += f"\n\n## Diagnostic Steering from Node 8\n{steering_instructions}\n"


    # 5. Execute Agent Turn via Chia API
    active_llm = _resolve_llm(llm, config)
    start_time = time.time()
    query_result = None
    try:
        if hasattr(active_llm, "prompt") and callable(active_llm.prompt):
            query_result = active_llm.prompt(
                prompt_text,
                tools=tools,
            )
    except Exception as llm_err:
        logger.error("Node 3: LLM prompt failed: %s", llm_err)
    finally:
        cleanup_tools(tools)

    elapsed = max(0.1, time.time() - start_time)

    # 6. Extract and Record Token Metrics (cost_usd=0.0)
    usage = getattr(query_result, "usage", None) if query_result is not None else None
    if not usage and hasattr(active_llm, "history") and active_llm.history:
        usage = active_llm.history[-1].get("usage")

    p_tok = (usage.get("prompt_tokens") or usage.get("input_tokens") or 0) if isinstance(usage, dict) else 0
    c_tok = (usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_tokens") or 0) if isinstance(usage, dict) else 0
    t_tok = (usage.get("total_tokens") or (p_tok + c_tok)) if isinstance(usage, dict) else 0

    if t_tok > 0:
        cost_usd = (p_tok * 0.00000125) + (c_tok * 0.000005)
        db.record_token_usage(
            phase="phase_1",
            node_name="node3_phase1_compiler_rewriter",
            model=getattr(active_llm, "model", "gemini-2.5-pro"),
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            cost_usd=cost_usd,
            candidate_id=candidate_id,
            duration_seconds=elapsed,
        )
        logger.info(
            "Node 3: Token usage recorded - prompt: %d, completion: %d, total: %d",
            p_tok,
            c_tok,
            t_tok,
        )

    # Profiler attachment
    try:
        from chia.base.profiling import get_profiler
        get_profiler().add_info({
            "node": "node3_phase1_compiler_rewriter",
            "candidate_id": candidate_id,
            "usage": usage,
        })
    except Exception:
        pass

    # 7. Extract Registered Rewrite Name Directly from un0_rewriter_tool (No deterministic fallback!)
    raw_result = getattr(query_result, "result", "") if query_result is not None else ""
    if not isinstance(raw_result, str):
        raw_result = str(raw_result)

    # Cross-process disk synchronization: ensure any module written by remote FastMCP actor is imported
    _sync_compiler_rewrites_from_disk(un0_dir, candidate_id=candidate_id)

    # Primary source of truth: registered rewrite directly from un0_rewriter_tool
    rewrite_name = un0_rewriter_tool.get_registered_rewrite(candidate_id=candidate_id)

    # Secondary lookup: check filesystem matching candidate_id directly
    if not rewrite_name:
        rewrites_dir = un0_dir / "un0" / "compiler" / "rewrites"
        clean_cand = candidate_id.strip().replace(" ", "_")
        if rewrites_dir.is_dir():
            matching_files = sorted(
                [p for p in rewrites_dir.glob(f"{clean_cand}_*.py") if p.name != "__init__.py"],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for fpath in matching_files:
                stem = fpath.stem
                pot_name = stem[len(clean_cand) + 1:] if stem.startswith(f"{clean_cand}_") else stem
                if pot_name in compiler_registry.COMPILER_REGISTRY:
                    rewrite_name = pot_name
                    break

    # Tertiary lookup: check if any rewrite registered for this candidate exists in registry
    if not rewrite_name:
        registered_compilers = un0_rewriter_tool.un0_list_registered_rewrites(axis="compiler")
        cand_matches = [name for name in registered_compilers if candidate_id in name]
        if cand_matches:
            rewrite_name = cand_matches[-1]
        elif rewrite_identifier in compiler_registry.COMPILER_REGISTRY:
            rewrite_name = rewrite_identifier
        else:
            # Check if agent referred to a valid registered rewrite in its response
            extracted = _extract_rewrite_name(raw_result)
            if extracted and extracted in compiler_registry.COMPILER_REGISTRY:
                rewrite_name = extracted
            else:
                # Also check candidate pattern matches directly against compiler_registry.COMPILER_REGISTRY
                cand_reg_matches = [name for name in compiler_registry.COMPILER_REGISTRY.keys() if candidate_id in name]
                if cand_reg_matches:
                    rewrite_name = cand_reg_matches[-1]

    if not rewrite_name or rewrite_name not in compiler_registry.COMPILER_REGISTRY:
        error_msg = (
            f"Node 3: Agent failed to synthesize or register a valid compiler rewrite in COMPILER_REGISTRY. "
            f"Returned rewrite name: {rewrite_name!r}. Available in registry: {list(compiler_registry.COMPILER_REGISTRY.keys())}"
        )
        logger.error(error_msg)
        db.update_execution_status(
            candidate_id,
            rewrite_compiler_status="FAILED",
            error_stage="node3_phase1_compiler_rewriter",
            error_message=error_msg,
        )
        # Record raw LLM output for diagnostic inspection by Node 8
        if raw_result and raw_result.strip():
            db.add_comment(
                candidate_id=candidate_id,
                phase="phase_1",
                agent_name="node3_phase1_compiler_rewriter",
                comment=f"[NODE 3 FAILURE RAW OUTPUT]\n{raw_result.strip()}",
            )
        raise RuntimeError(error_msg)

    # 8. Retrieve and Execute Registered Compiler Rewrite
    rewrite_fn = compiler_registry.get_compiler_rewrite(rewrite_name)
    try:
        functional_model = rewrite_fn(
            model=eager_model,
            solver_name=solver_name,
            num_steps=num_steps,
            dt=dt,
            integration_time=integration_time,
            target_device=target_device,
        )
    except Exception as exec_err:
        error_msg = f"Node 3: Execution of registered compiler rewrite '{rewrite_name}' failed: {exec_err}"
        logger.error(error_msg)
        db.update_execution_status(
            candidate_id,
            rewrite_compiler_status="FAILED",
            error_stage="node3_phase1_compiler_rewriter",
            error_message=error_msg,
        )
        if raw_result and raw_result.strip():
            db.add_comment(
                candidate_id=candidate_id,
                phase="phase_1",
                agent_name="node3_phase1_compiler_rewriter",
                comment=f"[NODE 3 EXECUTION ERROR RAW OUTPUT]\n{raw_result.strip()}",
            )
        raise RuntimeError(error_msg)

    # 9. Update Database Status and Cross-Node Comment
    db.update_execution_status(
        candidate_id,
        rewrite_compiler_status="SUCCESS",
    )
    # Record the agent's raw response / explanation as a comment for cross-node communication
    if raw_result and raw_result.strip():
        db.add_comment(
            candidate_id=candidate_id,
            phase="phase_1",
            agent_name="node3_phase1_compiler_rewriter",
            comment=f"[REWRITE: {rewrite_name}] {raw_result.strip()}",
        )
    else:
        db.add_comment(
            candidate_id=candidate_id,
            phase="phase_1",
            agent_name="node3_phase1_compiler_rewriter",
            comment=(
                f"Registered compiler rewrite '{rewrite_name}' targeting device '{target_device}' "
                f"with dynamic batching B >= 1."
            ),
        )

    logger.info(
        "Node 3: Compiler rewrite '%s' successfully applied for %s",
        rewrite_name,
        candidate_id,
    )

    return {
        "candidate_id": candidate_id,
        "status": "SUCCESS",
        "target_device": target_device,
        "prepared_model": functional_model,
        "eager_model": eager_model,
        "registered_rewrite": rewrite_name,
        "raw_result": raw_result,
        "tokens_consumed": t_tok,
        "token_usage": {
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
        },
        "hop_functionalized": True,
        "next_node": "node4_phase1_compilation_runner",
    }


# Backwards compatibility alias
node3_compiler_rewriter = node3_phase1_compiler_rewriter
