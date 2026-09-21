"""Phase 1 Pipeline Orchestrator for the analog ML accelerator simulation loop.

Coordinates Nodes 0 through 8:
  - Node 0: Programmatic gate orchestrating Reference Profiling, Rewrite Correction, and Debug Mode.
  - Node 1: Agentic solver rewriter applying integrator changes under un0/solver.
  - Node 2: Agentic noise rewriter applying hardware noise model wrappers under un0/noise.
  - Node 3: Agentic compiler rewriter functionalizing model for torch.compile and HOPs.
  - Node 4: Programmatic AOT compilation runner on target device worker (MPS / CUDA / CPU).
  - Node 5: Programmatic numerical verification gate comparing compiled vs eager mode outputs on 100 samples.
  - Node 6: Programmatic simulation & profiling (latency, peak memory, accuracy + CI + std + worst-percentile).
  - Node 7: Programmatic evaluation gate checking tolerances, retry counters, debug loop invariants, and Pareto front.
  - Node 8: Agentic diagnostic agent emitting fix rewrite proposals or debug mode requests on error/out-of-tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.nodes.node0_phase1_gate import node0_gate, node0_phase1_gate
from agentic_ml_analog_sim.nodes.node1_phase1_solver_rewriter import (
    node1_phase1_solver_rewriter,
    node1_solver_rewriter,
)
from agentic_ml_analog_sim.nodes.node2_phase1_noise_rewriter import (
    node2_phase1_noise_rewriter,
    node2_noise_rewriter,
)
from agentic_ml_analog_sim.nodes.node3_phase1_compiler_rewriter import (
    node3_phase1_compiler_rewriter,
    node3_compiler_rewriter,
)
from agentic_ml_analog_sim.nodes.node4_phase1_compilation_runner import (
    node4_phase1_compilation_runner,
    node4_compilation_runner,
)
from agentic_ml_analog_sim.nodes.node5_phase1_verify_correctness import (
    node5_phase1_verify_correctness,
    node5_verify_correctness,
)
from agentic_ml_analog_sim.nodes.node6_phase1_simulate_profile import (
    node6_phase1_simulate_profile,
    node6_simulate_profile,
)
from agentic_ml_analog_sim.nodes.node7_phase1_evaluation_gate import (
    node7_phase1_evaluation_gate,
    node7_evaluation_gate,
)
from agentic_ml_analog_sim.nodes.node8_phase1_diagnostic_proposal import (
    node8_phase1_diagnostic_proposal,
    node8_diagnostic_proposal,
)
from agentic_ml_analog_sim.nodes.node9_phase1_error_handler import (
    node9_error_handler,
    node9_phase1_error_handler,
)

try:
    from chia.base.ChiaFunction import get
except ImportError:
    get = None
import ray

logger = logging.getLogger("agentic_ml_analog_sim.phase_1")


def _dispatch_node(node_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Dispatch a node function via chia_remote if Ray is initialized, else local."""
    if ray.is_initialized() and hasattr(node_fn, "chia_remote") and get is not None:
        ref = node_fn.chia_remote(*args, **kwargs)
        return get(ref)
    return node_fn(*args, **kwargs)


@dataclass
class Phase1Result:
    """Outcome of the Phase 1 execution, compilation, profiling, and evaluation pipeline."""

    action: Literal["PROCEED_TO_PHASE_2", "TERMINATED_MAX_CYCLES", "ERROR"]
    candidate_id: Optional[str]
    is_reference: bool
    execution_mode: str
    metrics: Dict[str, Any]
    cycle_count: int
    details: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def run_phase_1(
    config_path: Union[str, Path, SimulationConfig],
    db_path: Union[str, Path, ResultDatabase],
    candidate_id: Optional[str] = None,
    trigger_source: str = "phase_0",
    target_device: str = "mps",
    llm: Any = None,
    un0_dir: Union[str, Path, None] = None,
    dry_run: bool = False,
    max_loop_cycles: int = 15,
    force_rewrites: bool = False,
) -> Phase1Result:
    """Execute the end-to-end Phase 1 execution, compilation, profiling, and diagnostic loop.

    1. Loads config and initializes database.
    2. Identifies active candidate (or latest proposal from Phase 0).
    3. Runs Node 0 Gate to set execution mode (Reference Profiling, Rewrite Correction, Debug Mode).
    4. Executes required rewriters (Node 1 solver, Node 2 noise, Node 3 compiler).
    5. Node 4 compiles model on target hardware device (MPS default, CUDA, CPU).
    6. Node 5 verifies numerical correctness vs eager mode (100 samples). On failure -> Node 8 diagnostic -> Node 0.
    7. Node 6 runs single-batch simulation and profiles metrics (latency, memory, accuracy).
    8. Node 7 evaluates metrics vs tolerances. On pass -> PROCEED_TO_PHASE_2. On failure -> Node 8 diagnostic -> Node 0.

    Args:
        config_path: Path to YAML config or SimulationConfig object.
        db_path: Path to SQLite database file or ResultDatabase object.
        candidate_id: Optional target proposal candidate ID.
        trigger_source: Source activating Phase 1 ('phase_0', 'node_8', 'cli').
        target_device: Hardware target device ('mps', 'cuda', 'cpu').
        llm: Optional LLM instance or MockAntigravityLLM.
        un0_dir: Optional path to Un-0 repository root.
        dry_run: If True, uses MockAntigravityLLM and deterministic mock models.
        max_loop_cycles: Safety upper bound on internal Phase 1 loop cycles.

    Returns:
        Phase1Result.
    """
    if dry_run and llm is None:
        llm = MockAntigravityLLM(mode="valid")

    from agentic_ml_analog_sim.tools.un0_context_tool import resolve_un0_root
    resolved_un0_dir = resolve_un0_root(un0_dir)

    # 1. Resolve Config
    if isinstance(config_path, SimulationConfig):
        config = config_path
    else:
        config = load_simulation_config(config_path)

    # 2. Resolve Database
    if isinstance(db_path, ResultDatabase):
        db = db_path
    else:
        db = ResultDatabase(db_path=db_path)

    # 3. Identify Candidate
    if candidate_id is None:
        latest = db.get_latest_proposal()
        if latest is None:
            return Phase1Result(
                action="ERROR",
                candidate_id=None,
                is_reference=False,
                execution_mode="UNKNOWN",
                metrics={},
                cycle_count=0,
                details="No proposal found in database to evaluate in Phase 1.",
            )
        candidate_id = latest["candidate_id"]

    proposal = db.get_proposal(candidate_id)
    if proposal is None:
        return Phase1Result(
            action="ERROR",
            candidate_id=candidate_id,
            is_reference=False,
            execution_mode="UNKNOWN",
            metrics={},
            cycle_count=0,
            details=f"Candidate ID '{candidate_id}' not found in database.",
        )

    is_reference = bool(proposal.get("is_reference", False))
    current_trigger = trigger_source
    previous_node_id = "phase_0_node_3" if current_trigger == "phase_0" else "node_8"
    debug_request: Optional[Dict[str, Any]] = None

    cycle_count = 0
    active_candidate_id = candidate_id

    # Every time we enter Phase 1 coming from Phase 0, all four counters must be reset to 0
    if current_trigger in ("phase_0", "phase0_node3") or previous_node_id.startswith("phase_0") or previous_node_id.startswith("phase0"):
        logger.info("run_phase_1: Entering Phase 1 from %s for candidate '%s'. Resetting all 4 counters to 0.", current_trigger, active_candidate_id)
        db.reset_counters(active_candidate_id)

    logger.info("Phase 1 pipeline starting for candidate '%s' (is_reference=%s)", active_candidate_id, is_reference)

    while cycle_count < max_loop_cycles:
        cycle_count += 1
        logger.info("Phase 1 loop cycle %d starting for candidate '%s'", cycle_count, active_candidate_id)

        # STEP 1: Node 0 Programmatic Gate
        gate_res = _dispatch_node(
            node0_phase1_gate,
            db=db,
            config=config,
            trigger_source=current_trigger,
            previous_node_id=previous_node_id,
            proposal_id=active_candidate_id,
            debug_request=debug_request,
        )

        gate_action = gate_res.get("action")
        gate_status = gate_res.get("status")
        if gate_action == "TERMINATED_MAX_CYCLES" or gate_status == "TERMINATED_MAX_CYCLES":
            logger.error("Node 0 gate terminated execution: %s", gate_res.get("reason", gate_res.get("details")))
            return Phase1Result(
                action="TERMINATED_MAX_CYCLES",
                candidate_id=active_candidate_id,
                is_reference=is_reference,
                execution_mode=gate_res.get("execution_mode", gate_res.get("mode", "UNKNOWN")),
                metrics={},
                cycle_count=cycle_count,
                details=f"Node 0 gate terminated: {gate_res.get('details') or gate_res.get('reason') or gate_res.get('counter_name')}",
            )

        execution_mode = gate_res["mode"]
        nodes_to_run = gate_res["nodes_to_run"]
        active_candidate_id = gate_res.get("candidate_id", active_candidate_id)

        logger.info("Node 0 configured mode '%s' for candidate '%s', nodes to run: %s", execution_mode, active_candidate_id, nodes_to_run)

        # STEP 2: Execute Rewriters (Nodes 1, 2, 3 as specified)
        if 1 in nodes_to_run:
            logger.info("Executing Node 1 (Solver Rewriter) for '%s'", active_candidate_id)
            node1_res = _dispatch_node(
                node1_phase1_solver_rewriter,
                db=db,
                config=config,
                candidate_id=active_candidate_id,
                llm=llm,
                force_rewrite=force_rewrites,
            )
            if node1_res.get("status") == "FAILED":
                logger.warning("Node 1 solver rewriter failed: %s", node1_res.get("error"))

        if 2 in nodes_to_run:
            logger.info("Executing Node 2 (Noise Rewriter) for '%s'", active_candidate_id)
            node2_res = _dispatch_node(
                node2_phase1_noise_rewriter,
                db=db,
                config=config,
                candidate_id=active_candidate_id,
                llm=llm,
                force_rewrite=force_rewrites,
            )
            if node2_res.get("status") == "FAILED":
                logger.warning("Node 2 noise rewriter failed: %s", node2_res.get("error"))

        if 3 in nodes_to_run:
            logger.info("Executing Node 3 (Compiler Rewriter) for '%s'", active_candidate_id)
            try:
                node3_res = _dispatch_node(
                    node3_phase1_compiler_rewriter,
                    db=db,
                    config=config,
                    candidate_id=active_candidate_id,
                    target_device=target_device,
                    llm=llm,
                )
            except Exception as err:
                logger.error("Node 3 encountered exception: %s. Routing to Node 9 error handler / Node 8 diagnostic recovery.", err)
                node3_res = {"status": "FAILED", "error": str(err)}

            if node3_res.get("status") == "FAILED":
                node3_err = node3_res.get("error", "Unknown Node 3 failure")
                logger.warning("Node 3 compiler rewriter failed: %s", node3_err)

                # Check / increment compilation retry counter (Counter 1)
                proposal = db.get_proposal(active_candidate_id) or {}
                curr_counter_1 = proposal.get("counter_1_compilation", 0) + 1
                db.update_counters(active_candidate_id, counter_1_compilation=curr_counter_1)

                counter_limits = getattr(config, "counter_limits", None)
                max_counter_1 = getattr(counter_limits, "max_counter_1_compilation", 3) if counter_limits else 3

                if curr_counter_1 > max_counter_1:
                    logger.error(
                        "Node 3 failure caused counter_1_compilation to exceed limit (%d > %d) for candidate '%s'",
                        curr_counter_1,
                        max_counter_1,
                        active_candidate_id,
                    )
                    node9_res = _dispatch_node(
                        node9_phase1_error_handler,
                        db=db,
                        config=config,
                        candidate_id=active_candidate_id,
                        counter_name="counter_1_compilation",
                        current_count=curr_counter_1,
                        max_count=max_counter_1,
                        execution_mode=execution_mode,
                        error_stage="node3_phase1_compiler_rewriter",
                        details=str(node3_err),
                    )
                    return Phase1Result(
                        action="TERMINATED_MAX_CYCLES",
                        candidate_id=active_candidate_id,
                        is_reference=is_reference,
                        execution_mode=execution_mode,
                        metrics={},
                        cycle_count=cycle_count,
                        details=f"Node 3 compiler rewrite exceeded retry limit: {node3_err}",
                    )

                # Route cleanly to Node 8 Diagnostic Agent
                diag_res = _dispatch_node(
                    node8_phase1_diagnostic_proposal,
                    db=db,
                    config=config,
                    candidate_id=active_candidate_id,
                    error_logs=f"Compiler rewriter failed: {node3_err}",
                    llm=llm,
                )
                current_trigger = "node_8"
                previous_node_id = "node8_phase1_diagnostic_proposal"
                if diag_res.get("type") == "DEBUG_MODE_REQUEST" or diag_res.get("action") == "debug_request":
                    debug_request = diag_res.get("debug_request") or diag_res
                else:
                    debug_request = {"nodes_to_rerun": diag_res.get("nodes_to_rerun", [1, 2, 3])}
                continue

        # STEP 3: Node 4 Compilation Runner
        logger.info("Executing Node 4 (Compilation Runner) on '%s' for '%s'", target_device, active_candidate_id)
        comp_res = _dispatch_node(
            node4_phase1_compilation_runner,
            db=db,
            config=config,
            candidate_id=active_candidate_id,
            target_device=target_device,
            un0_dir=str(resolved_un0_dir),
            prepared_model=node3_res.get("prepared_model") if "node3_res" in locals() and isinstance(node3_res, dict) else None,
            eager_model=node3_res.get("eager_model") if "node3_res" in locals() and isinstance(node3_res, dict) else None,
        )

        compiled_model = comp_res.get("compiled_model")
        eager_model = comp_res.get("eager_model")
        compilation_status = comp_res.get("status", "SUCCESS")
        compilation_error = comp_res.get("error_message") or comp_res.get("error_logs")

        # STEP 4: Node 5 Verify Correctness (Eager vs Compiled)
        logger.info("Executing Node 5 (Verify Correctness) for '%s'", active_candidate_id)
        verify_res = _dispatch_node(
            node5_phase1_verify_correctness,
            db=db,
            config=config,
            candidate_id=active_candidate_id,
            compiled_model=compiled_model,
            eager_model=eager_model,
            target_device=target_device,
            compilation_status=compilation_status,
            compilation_error=compilation_error,
        )

        if verify_res.get("status") == "TERMINATED_MAX_CYCLES":
            logger.error("Node 5 terminated loop due to cycle limits for '%s'", active_candidate_id)
            return Phase1Result(
                action="TERMINATED_MAX_CYCLES",
                candidate_id=active_candidate_id,
                is_reference=is_reference,
                execution_mode=verify_res.get("execution_mode", execution_mode),
                metrics={},
                cycle_count=cycle_count,
                details=f"Node 5 retry limit exceeded: {verify_res.get('counter_name')} ({verify_res.get('current_count')}/{verify_res.get('max_count')})",
            )

        if verify_res.get("status") == "FAILED":
            logger.warning("Node 5 verification failed for candidate '%s': %s", active_candidate_id, verify_res.get("details"))
            # Route to Node 8 Diagnostic Agent
            diag_res = _dispatch_node(
                node8_phase1_diagnostic_proposal,
                db=db,
                config=config,
                candidate_id=active_candidate_id,
                error_logs=f"Numerical verification failed: {verify_res.get('details')}",
                llm=llm,
            )
            current_trigger = "node_8"
            previous_node_id = "node8_phase1_diagnostic_proposal"
            if diag_res.get("type") == "DEBUG_MODE_REQUEST" or diag_res.get("action") == "debug_request":
                debug_request = diag_res.get("debug_request") or diag_res
            else:
                debug_request = {"nodes_to_rerun": diag_res.get("nodes_to_rerun", [1, 2, 3])}
            continue

        # STEP 5: Node 6 Simulate and Profile
        logger.info("Executing Node 6 (Simulate & Profile) for '%s'", active_candidate_id)
        model_to_profile = verify_res.get("compiled_model") if verify_res.get("compiled_model") is not None else compiled_model
        profile_res = _dispatch_node(
            node6_phase1_simulate_profile,
            db=db,
            config=config,
            candidate_id=active_candidate_id,
            compiled_model=model_to_profile,
            target_device=target_device,
            backend_mode=target_device,
        )

        if profile_res.get("status") == "FAILED":
            logger.warning("Node 6 simulation failed for candidate '%s': %s", active_candidate_id, profile_res.get("error_message"))
            # Route to Node 8 Diagnostic Agent
            diag_res = _dispatch_node(
                node8_phase1_diagnostic_proposal,
                db=db,
                config=config,
                candidate_id=active_candidate_id,
                error_logs=f"Simulation profiling failed: {profile_res.get('error_message')}",
                llm=llm,
            )
            current_trigger = "node_8"
            previous_node_id = "node8_phase1_diagnostic_proposal"
            if diag_res.get("type") == "DEBUG_MODE_REQUEST" or diag_res.get("action") == "debug_request":
                debug_request = diag_res.get("debug_request") or diag_res
            else:
                debug_request = {"nodes_to_rerun": diag_res.get("nodes_to_rerun", [1, 2, 3])}
            continue

        metrics = profile_res.get("metrics", {})

        # STEP 6: Node 7 Evaluation Gate
        logger.info("Executing Node 7 (Evaluation Gate) for '%s'", active_candidate_id)
        eval_res = _dispatch_node(
            node7_phase1_evaluation_gate,
            db=db,
            config=config,
            candidate_id=active_candidate_id,
        )

        eval_status = eval_res.get("status")
        next_node = eval_res.get("next_node")

        if eval_status == "PASSED" or next_node == "phase2_node1":
            logger.info("Node 7 approved candidate '%s'. Proceeding to Phase 2.", active_candidate_id)
            return Phase1Result(
                action="PROCEED_TO_PHASE_2",
                candidate_id=active_candidate_id,
                is_reference=is_reference,
                execution_mode=execution_mode,
                metrics=metrics,
                cycle_count=cycle_count,
                details="Phase 1 evaluation passed.",
            )
        elif eval_status == "TERMINATED_MAX_CYCLES" or (eval_status == "FAILED" and eval_res.get("reason") in ("counter_2_rewrite_exceeded", "counter_4_debug_loop_exceeded")):
            logger.error("Node 7 terminated loop due to cycle limits for '%s'", active_candidate_id)
            return Phase1Result(
                action="TERMINATED_MAX_CYCLES",
                candidate_id=active_candidate_id,
                is_reference=is_reference,
                execution_mode=eval_res.get("execution_mode", execution_mode),
                metrics=metrics,
                cycle_count=cycle_count,
                details=f"Node 7 retry limit exceeded: {eval_res.get('counter_name', eval_res.get('reason', ''))}",
            )
        else:
            # Out of tolerance -> Route to Node 8 Diagnostic Agent
            logger.warning("Node 7 marked candidate '%s' out of tolerance. Triggering Node 8 diagnostic.", active_candidate_id)
            diag_res = _dispatch_node(
                node8_phase1_diagnostic_proposal,
                db=db,
                config=config,
                candidate_id=active_candidate_id,
                error_logs=f"Evaluation out of tolerance: {eval_res.get('reason', '')}",
                llm=llm,
            )
            current_trigger = "node_8"
            previous_node_id = "node8_phase1_diagnostic_proposal"
            if diag_res.get("type") == "DEBUG_MODE_REQUEST" or diag_res.get("action") == "debug_request":
                debug_request = diag_res.get("debug_request") or diag_res
            else:
                debug_request = {"nodes_to_rerun": diag_res.get("nodes_to_rerun", [1, 2, 3])}
            continue

    # Exceeded loop bound
    logger.error("Exceeded maximum internal Phase 1 loop cycles (%d)", max_loop_cycles)
    node9_res = _dispatch_node(
        node9_phase1_error_handler,
        db=db,
        config=config,
        candidate_id=active_candidate_id,
        counter_name="max_loop_cycles",
        current_count=max_loop_cycles,
        max_count=max_loop_cycles,
        execution_mode="EXCEEDED_BOUND",
        error_stage="phase1_orchestrator",
        details=f"Exceeded maximum internal Phase 1 loop cycles ({max_loop_cycles}).",
    )
    return Phase1Result(
        action="TERMINATED_MAX_CYCLES",
        candidate_id=active_candidate_id,
        is_reference=is_reference,
        execution_mode="EXCEEDED_BOUND",
        metrics={},
        cycle_count=max_loop_cycles,
        details=f"Exceeded maximum internal Phase 1 loop cycles ({max_loop_cycles}). Snapshot: {node9_res.get('db_snapshot')}",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Phase 1."""
    parser = argparse.ArgumentParser(
        description="Phase 1 CLI: Model execution, compilation, profiling, and evaluation loop."
    )
    parser.add_argument(
        "--config",
        default="config/simulation_config.yaml",
        help="Path to YAML configuration file (default: config/simulation_config.yaml)",
    )
    parser.add_argument(
        "--db",
        default="results.db",
        help="Path to SQLite results database (default: results.db)",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Optional specific candidate ID to execute",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Target hardware compilation & simulation device (default: mps)",
    )
    parser.add_argument(
        "--un0-dir",
        default=None,
        help="Optional path to Un-0 repository root (defaults to dynamic resolution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Phase 1 with deterministic MockAntigravityLLM and offline mock models",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG level logging",
    )
    return parser


def main() -> None:
    """CLI entrypoint for Phase 1 pipeline execution."""
    parser = build_arg_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    llm = MockAntigravityLLM(mode="valid") if args.dry_run else None

    result = run_phase_1(
        config_path=args.config,
        db_path=args.db,
        candidate_id=args.candidate_id,
        target_device=args.device,
        llm=llm,
        un0_dir=args.un0_dir,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("PHASE 1 EXECUTION SUMMARY")
    print("=" * 60)
    print(result.to_json())


if __name__ == "__main__":
    main()
