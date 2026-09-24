"""Chia Nodes for the analog ML accelerator simulation loop (Phase 0, Phase 1, Phase 2)."""

# Phase 0 Nodes (Standardized: node<n>_phase0_*)
from agentic_ml_analog_sim.nodes.node1_phase0_check_reference import (
    check_reference_data,
    node1_check_reference,
    node1_phase0_check_reference,
)
from agentic_ml_analog_sim.nodes.node2_phase0_propose_candidate import (
    node2_phase0_propose_candidate,
    node2_propose_candidate,
    propose_candidate,
)
from agentic_ml_analog_sim.nodes.node3_phase0_validate_proposal import (
    node3_phase0_validate_proposal,
    node3_validate_proposal,
    validate_proposal,
)

# Phase 1 Nodes (Standardized: node<n>_phase1_*)
from agentic_ml_analog_sim.nodes.node0_phase1_gate import (
    node0_gate,
    node0_phase1_gate,
)
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

# Phase 2 Nodes (Standardized: node<n>_phase2_*)
from agentic_ml_analog_sim.nodes.node1_phase2_check_experiment_limit import (
    node1_phase2_check_experiment_limit,
    phase2_check_experiment_limit,
)
from agentic_ml_analog_sim.nodes.node2_phase2_generate_final_report import (
    node2_phase2_generate_final_report,
    phase2_generate_final_report,
)

__all__ = [
    # Phase 0 Nodes
    "node1_phase0_check_reference",
    "check_reference_data",
    "node1_check_reference",
    "node2_phase0_propose_candidate",
    "propose_candidate",
    "node2_propose_candidate",
    "node3_phase0_validate_proposal",
    "validate_proposal",
    "node3_validate_proposal",
    # Phase 1 Nodes
    "node0_phase1_gate",
    "node0_gate",
    "node1_phase1_solver_rewriter",
    "node1_solver_rewriter",
    "node2_phase1_noise_rewriter",
    "node2_noise_rewriter",
    "node3_phase1_compiler_rewriter",
    "node3_compiler_rewriter",
    "node4_phase1_compilation_runner",
    "node4_compilation_runner",
    "node5_phase1_verify_correctness",
    "node5_verify_correctness",
    "node6_phase1_simulate_profile",
    "node6_simulate_profile",
    "node7_phase1_evaluation_gate",
    "node7_evaluation_gate",
    "node8_phase1_diagnostic_proposal",
    "node8_diagnostic_proposal",
    "node9_phase1_error_handler",
    "node9_error_handler",
    # Phase 2 Nodes
    "node1_phase2_check_experiment_limit",
    "phase2_check_experiment_limit",
    "node2_phase2_generate_final_report",
    "phase2_generate_final_report",
]
