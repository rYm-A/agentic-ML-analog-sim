"""Phase 0 Pipeline Orchestrator for the analog ML accelerator simulation loop.

Coordinates:
  - Node 1: Programmatic check for baseline reference design data.
  - Node 2: Agentic candidate proposal via AntigravityLLM with ResultDB and Un0 tools.
  - Node 3: Programmatic validation, deduplication, and retry cycle bounding.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from agentic_ml_analog_sim.config import SimulationConfig, load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.mock.mock_llm import MockAntigravityLLM
from agentic_ml_analog_sim.nodes.node1_phase0_check_reference import (
    check_reference_data,
    node1_phase0_check_reference,
)
from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
    node2_phase0_propose_candidate,
    propose_candidate,
)
from agentic_ml_analog_sim.nodes.node3_phase0_validate_proposal import (
    node3_phase0_validate_proposal,
    validate_proposal,
)

try:
    from chia.base.ChiaFunction import get
except ImportError:
    get = None
import ray

logger = logging.getLogger("agentic_ml_analog_sim.phase_0")


def _dispatch_node(node_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Dispatch a node function via chia_remote if Ray is initialized, else local."""
    if ray.is_initialized() and hasattr(node_fn, "chia_remote") and get is not None:
        ref = node_fn.chia_remote(*args, **kwargs)
        return get(ref)
    return node_fn(*args, **kwargs)


@dataclass
class Phase0Result:
    """Outcome of the Phase 0 exploration and validation pipeline."""

    action: Literal[
        "PROCEED_TO_PHASE_1_REFERENCE",
        "PROCEED_TO_PHASE_1_CANDIDATE",
        "TERMINATED_MAX_CYCLES",
        "ERROR",
    ]
    candidate_id: Optional[str]
    phase_1_instructions: Optional[str]
    cycle_count: int
    details: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _format_reference_instructions(config: SimulationConfig, ref_id: str) -> str:
    """Format instructions for running baseline reference simulation in Phase 1."""
    ref = config.reference_design
    return (
        f"# Phase 1 Baseline Reference Design Evaluation\n"
        f"Candidate ID: {ref_id}\n"
        f"Design Name: {ref.name} (Family: {ref.family})\n"
        f"Oscillators: {ref.n_oscillators} main, {ref.n_conditional_oscillators} conditional\n"
        f"Numerical Solver: {ref.solver}\n"
        f"Integration Steps: {ref.num_steps} (Time: {ref.integration_time}, Precision: {ref.precision})\n"
        f"Hardware Noise Model: {ref.noise_model}\n"
        f"Crossbar Sparsity: {ref.sparsity_ratio}\n\n"
        f"## Action Items for Phase 1:\n"
        f"1. Run unperturbed full-precision reference evaluation on Un-0 Kuramoto dynamics.\n"
        f"2. Record baseline numerical trajectory accuracy (relative_error = 0.0 against golden RK4) and simulation latency (ms).\n"
        f"3. Mark candidate '{ref_id}' as COMPLETED in the results database."
    )


def run_phase_0(
    config_path: Union[str, Path],
    db_path: Union[str, Path],
    llm: Any = None,
    un0_dir: Union[str, Path, None] = None,
    iteration: int = 1,
    enable_safety_net: bool = True,
) -> Phase0Result:
    """Execute the end-to-end Phase 0 pipeline.

    1. Loads SimulationConfig from config_path.
    2. Initializes ResultDatabase at db_path.
    3. Runs Node 1 check:
       - If reference design data is incomplete -> returns PROCEED_TO_PHASE_1_REFERENCE.
    4. If reference design data is complete -> enters Node 2 / Node 3 loop:
       - Loop while cycle_count <= config.budget.max_node_retries:
         - Node 2 proposes next candidate (solver + noise + sparsity).
         - Node 3 validates candidate against config and prior DB records.
         - If valid -> returns PROCEED_TO_PHASE_1_CANDIDATE.
         - If invalid/duplicate -> increments cycle_count and re-triggers Node 2.
       - If cycle cap is reached -> returns TERMINATED_MAX_CYCLES.

    Args:
        config_path: Path to simulation configuration YAML.
        db_path: Path to SQLite results database file.
        llm: Optional LLM instance (AntigravityLLM or MockAntigravityLLM).
        un0_dir: Optional path to Un-0 repository root.

    Returns:
        Phase0Result containing action, candidate_id, instructions, cycle_count, and details.
    """
    cycle_count = 0
    try:
        # 1. Load Simulation Configuration
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            return Phase0Result(
                action="ERROR",
                candidate_id=None,
                phase_1_instructions=None,
                cycle_count=0,
                details=f"Configuration file not found at: {config_path}",
            )

        config: SimulationConfig = load_simulation_config(cfg_path)
        logger.info("Loaded simulation configuration for experiment '%s'", config.experiment_name)

        # 2. Initialize Result Database
        database = ResultDatabase(db_path=db_path)
        logger.info("Initialized ResultDatabase at %s", db_path)

        # 3. Setup Tools in direct local mode (avoiding unwanted background Ray actor starts)
        tools: list[Any] = []
        try:
            from agentic_ml_analog_sim.tools.result_db_tool import ResultDBReadOnlyTool
            tools.append(ResultDBReadOnlyTool(db=database, deploy=False))
        except Exception as e:
            logger.debug("Could not bind ResultDBReadOnlyTool: %s", e)

        from agentic_ml_analog_sim.tools.un0_context_tool import (
            Un0ContextTool,
            resolve_un0_root,
        )
        repo_root = resolve_un0_root(un0_dir)
        if repo_root.exists() and repo_root.is_dir():
            try:
                tools.append(Un0ContextTool(repo_root=repo_root, auto_start=False))
            except Exception as e:
                logger.debug("Could not bind Un0ContextTool for %s: %s", repo_root, e)

        # 4. Node 1: Check Baseline Reference Data Completeness
        has_reference = _dispatch_node(node1_phase0_check_reference, db=database, config=config)
        if not has_reference:
            logger.info("Node 1: Reference data incomplete. Preparing Phase 1 reference evaluation.")
            existing_ref = database.get_reference_proposal()
            if existing_ref is not None:
                ref_id = existing_ref["candidate_id"]
                instructions = existing_ref.get("phase_1_prompt") or _format_reference_instructions(config, ref_id)
            else:
                ref_cfg = config.reference_design
                ref_id = f"ref_{ref_cfg.solver}_{ref_cfg.num_steps}steps"
                instructions = _format_reference_instructions(config, ref_id)
                database.insert_proposal(
                    solver=ref_cfg.solver,
                    noise_model=ref_cfg.noise_model,
                    solver_params={"num_steps": ref_cfg.num_steps, "integration_time": ref_cfg.integration_time},
                    noise_params={},
                    sparsity_config={"sparsity_ratio": ref_cfg.sparsity_ratio},
                    candidate_id=ref_id,
                    status="PENDING",
                    is_reference=True,
                    selection_reason="Baseline reference design initial setup",
                    phase_1_prompt=instructions,
                )

            return Phase0Result(
                action="PROCEED_TO_PHASE_1_REFERENCE",
                candidate_id=ref_id,
                phase_1_instructions=instructions,
                cycle_count=0,
                details="Baseline reference design data is missing or incomplete in database. Routing to Phase 1.",
            )

        # 5. Programmatically determine target noise model based on per-noise quota budget
        configured_noise = getattr(config, "noise_models", ["L0_static_mismatch"])
        noise_model_names: list[str] = []
        if isinstance(configured_noise, dict):
            noise_model_names = list(configured_noise.keys())
        elif isinstance(configured_noise, (list, tuple)):
            for nm in configured_noise:
                if isinstance(nm, str):
                    noise_model_names.append(nm)
                elif hasattr(nm, "name"):
                    noise_model_names.append(getattr(nm, "name"))
                elif isinstance(nm, dict) and "name" in nm:
                    noise_model_names.append(nm["name"])
        if not noise_model_names:
            noise_model_names = ["L0_static_mismatch"]

        max_experiments = getattr(config.budget, "max_experiments", 10)
        target_noise_model: Optional[str] = None
        if hasattr(database, "get_next_unfulfilled_noise_model") and callable(database.get_next_unfulfilled_noise_model):
            target_noise_model = database.get_next_unfulfilled_noise_model(
                noise_models=noise_model_names,
                max_per_noise=max_experiments,
            )
        else:
            if hasattr(database, "get_proposal_counts_by_noise_model"):
                counts = database.get_proposal_counts_by_noise_model(exclude_reference=True)
            else:
                counts = {}
            for nm in noise_model_names:
                nm_counts = counts.get(nm, {})
                attempts = nm_counts.get("completed", 0) + nm_counts.get("failed", 0)
                if attempts < max_experiments:
                    target_noise_model = nm
                    break

        if target_noise_model is None:
            logger.info(
                "All configured noise models have completed their experiment budget (%d each).",
                max_experiments,
            )
            return Phase0Result(
                action="TERMINATED_MAX_CYCLES",
                candidate_id=None,
                phase_1_instructions=None,
                cycle_count=0,
                details="All configured noise models have completed their experiment budget.",
            )

        logger.info(
            "Target noise model for Phase 0 proposal: '%s' (quota: %d experiments per noise model)",
            target_noise_model,
            max_experiments,
        )

        # 6. Reference is complete: Enter Node 2 / Node 3 Optimization Loop
        max_retries = config.budget.max_node_retries
        cycle_count = 1
        prior_rejection_reason: Optional[str] = None
        last_candidate_id: Optional[str] = None

        logger.info(
            "Node 1 check passed. Entering Node 2 / Node 3 candidate proposal loop (max retries: %d)",
            max_retries,
        )

        while cycle_count <= max_retries:
            logger.info("Phase 0 loop cycle %d/%d starting", cycle_count, max_retries)

            # Node 2: Propose Candidate
            proposal_dict = _dispatch_node(
                node2_phase0_propose_candidate,
                db=database,
                config=config,
                llm=llm,
                tools=tools if tools else None,
                prior_rejection_reason=prior_rejection_reason,
                iteration=iteration,
                target_noise_model=target_noise_model,
            )

            candidate_id = proposal_dict["candidate_id"]
            last_candidate_id = candidate_id
            phase_1_prompt = proposal_dict.get("phase_1_prompt")

            # Node 3: Validate Candidate
            is_valid, reason, updated_cycle = _dispatch_node(
                node3_phase0_validate_proposal,
                candidate_id=candidate_id,
                db=database,
                config=config,
                cycle_count=cycle_count,
                target_noise_model=target_noise_model,
            )

            if is_valid:
                logger.info(
                    "Node 3 approved candidate '%s' on cycle %d. Proceeding to Phase 1.",
                    candidate_id,
                    cycle_count,
                )
                return Phase0Result(
                    action="PROCEED_TO_PHASE_1_CANDIDATE",
                    candidate_id=candidate_id,
                    phase_1_instructions=phase_1_prompt,
                    cycle_count=cycle_count,
                    details="Candidate approved for Phase 1",
                )

            # Invalid or duplicate candidate
            logger.warning(
                "Proposal '%s' rejected by Node 3 on cycle %d: %s",
                candidate_id,
                cycle_count,
                reason,
            )
            prior_rejection_reason = reason

            if reason == "MAX_CYCLES_EXCEEDED" or updated_cycle > max_retries or cycle_count >= max_retries:
                logger.error("Max proposal retry cycles exceeded (%d/%d)", cycle_count, max_retries)
                if enable_safety_net:
                    from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
                        _insert_pending_candidate,
                        _synthesize_unexplored_candidate,
                    )

                    fallback_proposal = _synthesize_unexplored_candidate(
                        config, database, iteration, target_noise_model
                    )
                    if fallback_proposal is not None:
                        logger.warning(
                            "Phase 0: Max proposal retries reached with LLM rejections. Engaging emergency programmatic synthesis safety net to proceed with candidate '%s'.",
                            fallback_proposal["candidate_id"],
                        )
                        _insert_pending_candidate(database, fallback_proposal)
                        fb_valid, _, _ = _dispatch_node(
                            node3_phase0_validate_proposal,
                            candidate_id=fallback_proposal["candidate_id"],
                            db=database,
                            config=config,
                            cycle_count=cycle_count,
                            target_noise_model=target_noise_model,
                        )
                        if fb_valid:
                            return Phase0Result(
                                action="PROCEED_TO_PHASE_1_CANDIDATE",
                                candidate_id=fallback_proposal["candidate_id"],
                                phase_1_instructions=fallback_proposal.get("phase_1_prompt"),
                                cycle_count=cycle_count,
                                details="Candidate generated via Phase 0 programmatic safety net",
                            )

                return Phase0Result(
                    action="TERMINATED_MAX_CYCLES",
                    candidate_id=candidate_id,
                    phase_1_instructions=None,
                    cycle_count=cycle_count,
                    details=f"Exceeded maximum proposal retry cycles ({max_retries}): {reason}",
                )

            cycle_count = max(cycle_count + 1, updated_cycle)

        # Exceeded loop bound
        if enable_safety_net:
            from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
                _insert_pending_candidate,
                _synthesize_unexplored_candidate,
            )

            fallback_proposal = _synthesize_unexplored_candidate(
                config, database, iteration, target_noise_model
            )
            if fallback_proposal is not None:
                logger.warning(
                    "Phase 0: Max proposal retries reached with LLM rejections. Engaging emergency programmatic synthesis safety net to proceed with candidate '%s'.",
                    fallback_proposal["candidate_id"],
                )
                _insert_pending_candidate(database, fallback_proposal)
                fb_valid, _, _ = _dispatch_node(
                    node3_phase0_validate_proposal,
                    candidate_id=fallback_proposal["candidate_id"],
                    db=database,
                    config=config,
                    cycle_count=cycle_count,
                    target_noise_model=target_noise_model,
                )
                if fb_valid:
                    return Phase0Result(
                        action="PROCEED_TO_PHASE_1_CANDIDATE",
                        candidate_id=fallback_proposal["candidate_id"],
                        phase_1_instructions=fallback_proposal.get("phase_1_prompt"),
                        cycle_count=cycle_count,
                        details="Candidate generated via Phase 0 programmatic safety net",
                    )

        return Phase0Result(
            action="TERMINATED_MAX_CYCLES",
            candidate_id=last_candidate_id,
            phase_1_instructions=None,
            cycle_count=max_retries,
            details=f"Exceeded maximum proposal retry cycles ({max_retries})",
        )

    except Exception as exc:
        logger.exception("Phase 0 execution encountered an unhandled error: %s", exc)
        return Phase0Result(
            action="ERROR",
            candidate_id=None,
            phase_1_instructions=None,
            cycle_count=cycle_count,
            details=f"Unhandled error in Phase 0: {type(exc).__name__}: {str(exc)}",
        )


def main() -> None:
    """CLI entrypoint for running Phase 0 pipeline directly."""
    import sys

    parser = argparse.ArgumentParser(
        description="Phase 0 Pipeline Orchestrator for Analog ML Accelerator Co-Design Loop"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/simulation_config.yaml",
        help="Path to simulation YAML configuration file (default: config/simulation_config.yaml)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results.db",
        help="Path to SQLite results database (default: results.db)",
    )
    parser.add_argument(
        "--un0-dir",
        type=str,
        default=None,
        help="Optional path to Un-0 repository root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Phase 0 with deterministic MockAntigravityLLM (no external calls or cloud credits)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG level logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    llm = MockAntigravityLLM(mode="valid") if args.dry_run else None

    result = run_phase_0(
        config_path=args.config,
        db_path=args.db,
        llm=llm,
        un0_dir=args.un0_dir,
    )

    print(result.to_json())

    if result.action in ("PROCEED_TO_PHASE_1_REFERENCE", "PROCEED_TO_PHASE_1_CANDIDATE"):
        sys.exit(0)
    elif result.action == "TERMINATED_MAX_CYCLES":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
