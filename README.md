# Agentic Hardware/Software Co-Design Loop for Analog ML Accelerator Simulation

An autonomous agentic AI framework powered by [Chia](https://github.com/ucb-bar/chia) and Ray for co-designing and evaluating analog machine learning accelerators executing physical neural network models (continuous-time oscillatory neural networks and memristive crossbar dynamics based on the Un-0 modular baseline).

---

## Table of Contents

- [Overview](#overview)
- [Agentic Loop Architecture (Phases 0 – 2)](#agentic-loop-architecture-phases-0--2)
  - [Phase 0: Candidate Exploration & Gatekeeping](#phase-0-candidate-exploration--gatekeeping)
  - [Phase 1: Code Rewriting, Simulation & Verification](#phase-1-code-rewriting-simulation--verification)
  - [Phase 2: Experiment Auditing, Reporting & Consultation](#phase-2-experiment-auditing-reporting--consultation)
- [Configuration Guide: `simulation_config.yaml`](#configuration-guide-simulation_configyaml)
  - [Reference Design (`reference_design`)](#reference-design-reference_design)
  - [Solvers & Discretization Steps (`solvers`)](#solvers--discretization-steps-solvers)
  - [Hardware Noise Models (`noise_models`)](#hardware-noise-models-noise_models)
  - [Coupling Matrix Sparsity (`sparsity_configs`)](#coupling-matrix-sparsity-sparsity_configs)
  - [Numerical Tolerances (`tolerances`)](#numerical-tolerances-tolerances)
  - [Campaign Budget vs. Internal Counter Limits](#campaign-budget-vs-internal-counter-limits)
  - [Agent Model & Effort Allocation (`agent_efforts`)](#agent-model--effort-allocation-agent_efforts)
- [Un-0 Co-Design Engine Integration](#un-0-co-design-engine-integration)
- [Token & Resource Auditing](#token--resource-auditing)
- [Cluster Execution Architecture](#cluster-execution-architecture)
  - [Bare-Metal Native macOS (Apple Silicon)](#bare-metal-native-macos-apple-silicon)
  - [Cloud Burst Profile (GCP + Tailnet)](#cloud-burst-profile-gcp--tailnet)
- [Installation & Quickstart](#installation--quickstart)
- [Running the Loop](#running-the-loop)
- [Testing](#testing)

---

## Overview

Deploying neural networks directly onto analog silicon substrates enables ultra-low-latency and high-energy-efficiency inference by leveraging intrinsic circuit physics (e.g., Kuramoto oscillator phase locking, non-volatile memristive conductance states). However, analog hardware introduces non-idealities such as device mismatch, thermal noise, phase jitter, non-linear I-V characteristics, and line resistance.

This repository implements an autonomous **agentic hardware/software co-design loop** that searches the multi-objective Pareto frontier across three primary dimensions:
1. **Numerical Solver Discretization**: Evaluating low-order vs. high-order ODE solvers (`euler`, `rk4`, `euler_backward`, `parareal`, `par_ode`) across variable step counts ($dt = T / N$).
2. **Coupling Matrix Sparsity**: Evaluating crossbar interconnect pruning ratios (from dense $0\%$ up to $80\%$ pruned) to reduce analog MAC energy and physical routing area.
3. **Hardware Noise Robustness**: Simulating circuit non-idealities across multi-fidelity noise tiers (L0 static mismatch, L1 parameter noise, L2 interface noise, and L3 correlated drift).

---

## Agentic Loop Architecture (Phases 0 – 2)

The autonomous loop is organized into three synchronized phases governed by SQLite telemetry and standardized node contracts:

```
+----------------------------------------------------------------------------------------------------+
|                                      PHASE 0: CANDIDATE EXPLORATION                                |
|                                                                                                    |
|  [Node 1: Check Reference] ---> [Node 2: Propose Candidate (LLM)] ---> [Node 3: Validate Proposal]   |
|         | (missing reference)                       ^                              | (passed)      |
|         v                                           +------ (retry if rejected) <--+               |
|  Route Reference to Phase 1                                                        v               |
+-----------------------------------------------------------------------------+----------------------+
                                                                              |
+-----------------------------------------------------------------------------v----------------------+
|                                 PHASE 1: REWRITE, COMPILE & SIMULATE                               |
|                                                                                                    |
|  [Node 0: Gate & Debug] ---> [Node 1: Solver Rewriter] ---> [Node 2: Noise Rewriter]               |
|           ^                                                                |                       |
|           |                                                                v                       |
|           |                 [Node 4: Compilation Runner] <--- [Node 3: Compiler Rewriter]          |
|           |                                |                                                       |
|           |                                v                                                       |
|           |                 [Node 5: Verify Correctness] (Counter 1 check)                         |
|           |                                |                                                       |
|           |                                v                                                       |
|           |                 [Node 6: Simulate & Profile (MPS/CUDA)]                                |
|           |                                |                                                       |
|           |                                v                                                       |
|  [Node 8: Diagnostic] <--- [Node 7: Evaluation Gate] ---> [Passed]                                |
|  (Reruns / Debug Mode)      (Counter 2 & 4 checks)           |                                     |
|           |                            |                     |                                     |
|           v                            v (max cycles hit)    |                                     |
|  [Node 9: Error Handler & SQLite Live Snapshot]              |                                     |
+--------------------------------------------------------------|-------------------------------------+
                                                               |
+--------------------------------------------------------------v-------------------------------------+
|                                 PHASE 2: AUDITING & REPORTING                                      |
|                                                                                                    |
|  [Node 1: Check Limits] ---> (Target reached?) ---> [Node 2: Generate Final LaTeX/PDF Report]      |
|         | (continue loop)                                    |                                     |
|         v                                                    v                                     |
|  Return to Phase 0                                   [Interactive Consultant Session REPL]        |
+----------------------------------------------------------------------------------------------------+
```

### Phase 0: Candidate Exploration & Gatekeeping
- **Node 1 (`node1_phase0_check_reference`)**: Checks SQLite database for the baseline golden reference design. If missing, routes reference parameters to Phase 1 to establish baseline accuracy and latency.
- **Node 2 (`node2_phase0_propose_candidate`)**: LLM agent (Gemini 2.5 Pro with high reasoning effort) queries past evaluations and Pareto frontier, proposing novel combinations of solver, steps, noise model, and sparsity ratio.
- **Node 3 (`node3_phase0_validate_proposal`)**: Gatekeeper validating proposals against allowed solver steps, noise models, and sparsity ratios; enforces uniqueness against duplicate proposals.

### Phase 1: Code Rewriting, Simulation & Verification
- **Node 0 (`node0_phase1_gate`)**: Enforces debug mode isolation invariants and guards against invalid debug requests (Counter 3).
- **Nodes 1–3 (`node1_phase1_solver_rewriter`, `node2_phase1_noise_rewriter`, `node3_phase1_compiler_rewriter`)**: Agents rewriting solver registries, noise parameters, and model compilation definitions.
- **Node 4 (`node4_phase1_compilation_runner`)**: Programmatically compiles the model using the Un-0 AOT compiler (`un0.compiler.compile_model`).
- **Node 5 (`node5_phase1_verify_correctness`)**: Verifies numerical stability and outputs between compiled and eager models against tolerances; manages Counter 1.
- **Node 6 (`node6_phase1_simulate_profile`)**: Executes batch inference simulation on hardware (Metal/MPS or CUDA), profiling latency, peak memory, FID, and relative error.
- **Node 7 (`node7_phase1_evaluation_gate`)**: Checks relative error and FID degradation against `tolerances.solver_accuracy`. Marks successful designs `COMPLETED` and updates the Pareto frontier; manages Counters 2 and 4.
- **Node 8 (`node8_phase1_diagnostic_proposal`)**: Formulates targeted corrective rewrites or requests isolated Debug Mode when candidates fail.
- **Node 9 (`node9_phase1_error_handler`)**: Standardized stop handler executed when any retry counter is exhausted. Performs graceful termination, updates proposal status to `FAILED`, and generates an atomic SQLite live snapshot (`snapshots/<candidate_id>_error_snapshot.db`).

### Phase 2: Experiment Auditing, Reporting & Consultation
- **Node 1 (`node1_phase2_check_experiment_limit`)**: Checks completed experiment counts against `budget.max_experiments` and loop iterations against `budget.max_iterations`.
- **Node 2 (`node2_phase2_generate_final_report`)**: Compiles publication-grade Pareto frontier plots, generates a LaTeX report with candidate comparison tables and swarm token audits, and compiles the final PDF artifact.
- **Interactive Consultant**: A CLI REPL (`InteractiveReportSession`) backed by an LLM to answer technical questions and analyze Pareto trade-offs from the generated report.

---

## Configuration Guide: `simulation_config.yaml`

All simulation parameters, campaign budgets, tolerances, and agent models are centralized in [`config/simulation_config.yaml`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/config/simulation_config.yaml).

```yaml
experiment_name: "un0_kuramoto_analog_acceleration"

# 1. Baseline Reference Design
reference_design:
  name: "cifar10/n1024"
  family: "cifar10"
  n_oscillators: 1024
  n_conditional_oscillators: 8
  solver: "rk4"
  num_steps: 25
  integration_time: 1.0
  precision: "fp32"
  noise_model: "none"
  sparsity_ratio: 0.0

# 2. Solvers Exploration Grid
solvers:
  euler:
    steps: [1, 2, 5, 10]
  rk4:
    steps: [5, 10, 15, 25]
  euler_backward:
    steps: [2, 5, 10]
  parareal:
    steps: [10, 20]
  par_ode:
    steps: [10, 25]

# 3. Hardware Noise Models
noise_models:
  - "L0_static_mismatch"
  - "L1_stochastic_parameter_noise"
  - "L2_functional_interface"
  - "L3_correlated_drift"

# 4. Sparsity Configurations
sparsity_configs:
  dense:
    sparsity_ratio: 0.0
  threshold_pruned:
    allowed_sparsity_ratios: [0.2, 0.4, 0.6, 0.8]
  degree_bounded:
    max_degree: [64, 128, 256]

# 5. Numerical Tolerances
tolerances:
  compile_check:
    rtol: 1e-4
    atol: 1e-4
  solver_accuracy:
    rtol: 5e-2
    atol: 5e-2
    max_fid_degradation: 2.0

# 6. Campaign Budget
budget:
  max_experiments: 10
  max_iterations: 15
  max_node_retries: 3
  timeout_seconds_per_eval: 300

# 7. Internal Phase 1 Counter Limits
counter_limits:
  max_counter_1_compilation: 3
  max_counter_2_rewrite: 3
  max_counter_3_debug_gate: 3
  max_counter_4_debug_loop: 5

# 8. Agent Reasoning Effort
agent_efforts:
  node2_phase0_propose_candidate:
    model: "gemini-2.5-pro"
    effort: "high"
  node1_phase1_solver_rewriter:
    model: "gemini-2.5-pro"
    effort: "high"
  node2_phase1_noise_rewriter:
    model: "gemini-2.5-pro"
    effort: "high"
  node3_phase1_compiler_rewriter:
    model: "gemini-2.5-pro"
    effort: "high"
  node8_phase1_diagnostic_proposal:
    model: "gemini-2.5-flash"
    effort: "medium"
  node2_phase2_generate_final_report:
    model: "gemini-2.5-flash"
    effort: "medium"
```

### Reference Design (`reference_design`)
The golden ground truth baseline. It is strictly dense (`sparsity_ratio: 0.0`), noise-free (`noise_model: "none"`), and integrated using high-precision 4th-order Runge-Kutta with 25 steps ($\Delta t = 0.04$). All candidate accuracy degradation, relative errors, and speedup multipliers ($1.00\times$) are measured relative to this design.

### Solvers & Discretization Steps (`solvers`)
Defines the numerical ODE algorithms available to candidate proposals. For each solver, `steps` specifies the allowed integration step grid. The step size is computed as $\Delta t = \text{integration\_time} / \text{num\_steps}$. Lower steps reduce simulation and MAC operation counts at the expense of truncation error.

### Hardware Noise Models (`noise_models`)
The physical noise ladder representing silicon non-idealities:
- **`L0_static_mismatch`**: Time-invariant spatial threshold variation across crossbar cells.
- **`L1_stochastic_parameter_noise`**: Dynamic thermal noise and white parameter fluctuations.
- **`L2_functional_interface`**: ADC/DAC quantization and driver line parasitics.
- **`L3_correlated_drift`**: Temperature drift and low-frequency $1/f$ phase noise.

### Coupling Matrix Sparsity (`sparsity_configs`)
Governs analog crossbar array connectivity:
- **`dense`**: Unpruned matrix ($0\%$ sparsity).
- **`threshold_pruned`**: Explores pruned coupling weights ($|K_{ij}| < \tau$) at ratios `[0.2, 0.4, 0.6, 0.8]`.
- **`degree_bounded`**: Constrains oscillator fan-in degrees to `[64, 128, 256]`.

### Numerical Tolerances (`tolerances`)
- **`compile_check`**: Strict tolerance (`rtol=1e-4`, `atol=1e-4`) enforced in Node 5 when comparing compiled AOT model forward passes against eager Un-0 models.
- **`solver_accuracy`**: Accuracy criteria enforced in Node 7:
  - `rtol = 0.05` ($5\%$ maximum relative state trajectory error vs. reference).
  - `max_fid_degradation = 2.0` (maximum permissible FID degradation).

### Campaign Budget vs. Internal Counter Limits
The configuration separates macro campaign bounds from micro execution retry bounds:

| Section | Parameter | Governs | Destination Upon Exceedance |
| :--- | :--- | :--- | :--- |
| **`budget`** | `max_experiments` (10) | Total completed candidate experiments | Phase 2 Final Report Generation |
| **`budget`** | `max_iterations` (15) | Hard ceiling on outer exploration cycles | Phase 2 Final Report Generation |
| **`budget`** | `max_node_retries` (3) | Phase 0 candidate rejection retries | Halt Phase 0 (`MAX_CYCLES_EXCEEDED`) |
| **`counter_limits`** | `max_counter_1_compilation` (3) | Compilation / ODE numerical checks (Node 5) | `node9_phase1_error_handler` |
| **`counter_limits`** | `max_counter_2_rewrite` (3) | Out-of-tolerance candidate rewrites (Node 7) | `node9_phase1_error_handler` |
| **`counter_limits`** | `max_counter_3_debug_gate` (3) | Invalid debug mode requests (Node 0) | `node9_phase1_error_handler` |
| **`counter_limits`** | `max_counter_4_debug_loop` (5) | Debug isolation iterations (Node 7) | `node9_phase1_error_handler` |

*Note: Bidirectional fallbacks ensure that if `counter_limits` is omitted from older configuration files, Counters 1–3 default automatically to `budget.max_node_retries`.*

### Agent Model & Effort Allocation (`agent_efforts`)
Configures the LLM backend and reasoning effort tier for each agentic node:
- **Proposer & Rewriters** (`node2_phase0`, `node1_phase1`, `node2_phase1`, `node3_phase1`): Use `gemini-2.5-pro` with `high` reasoning effort (`--effort high`) for complex architectural parameter selection and code rewriting.
- **Diagnostics & Reporting** (`node8_phase1`, `node2_phase2`): Use `gemini-2.5-flash` with `medium` reasoning effort (`--effort medium`) for fast root-cause analysis and report summarization.

---

## Un-0 Co-Design Engine Integration

The physical analog simulation baseline is provided by the `Un-0` project, integrated as a **Git submodule** (`Un-0/`) and installed in editable mode:
- **Modular Solvers** (`un0/solver/`): Pluggable numerical ODE integrators registered in `SOLVER_REGISTRY`.
- **Multi-Fidelity Noise** (`un0/noise/`): Hardware noise models implementing the physical noise ladder.
- **AOT Compiler Integration** (`un0/compiler/`): Dynamic AOT model compilation via `compile_model(eager_model)`.
- **Dynamic Root Discovery**: [`resolve_un0_root()`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/src/agentic_ml_analog_sim/tools/un0_context_tool.py#L25) resolves submodule and sibling directory paths dynamically, avoiding hardcoded filesystem paths.

---

## Token & Resource Auditing

In accordance with swarm-wide auditing guidelines, every LLM invocation across all phases extracts usage metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_usd`, and `duration_seconds`).
1. **Persistent SQLite Telemetry**: Stored in the `token_usage` table in `ResultDatabase`.
2. **Ray Profiler Registration**: Attached to the Chia profiling system via `get_profiler().add_info()`.
3. **Report Audit Table**: Automatically tabulated and rendered into the final LaTeX and PDF report.

---

## Cluster Execution Architecture

### Bare-Metal Native macOS (Apple Silicon)
- **Profile**: [`cluster.yaml`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/cluster.yaml) / [`cluster.bare.yaml`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/cluster.bare.yaml)
- **Runtime**: Native macOS `aarch64` within `conda activate chia_env` (no Docker/Colima emulation overhead).
- **Acceleration**: Direct Apple Silicon Metal/MPS unified memory allocation.
- **Ray Resources**: Tagged with `antigravity_creds: 1`, `metal_gpu: 1`, `sqlite_db: 1`, and `python_worker: 4`.

### Cloud Burst Profile (GCP + Tailnet)
- **Profile**: [`cluster.cloud.yaml`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/cluster.cloud.yaml)
- **Runtime**: Hybrid macOS head + GCP Linux worker instances (`g2-standard-8` with NVIDIA L4 GPUs).
- **Networking**: Userspace Tailscale WireGuard mesh securing inter-node Ray gRPC traffic.

---

## Installation & Quickstart

### Prerequisites
- macOS (Apple Silicon recommended) with Python 3.10+
- Conda / Miniconda

### Setup
```bash
# 1. Clone repository with submodules
git clone --recurse-submodules <repo-url>
cd agentic-ML-analog-sim

# 2. Activate conda environment
conda activate chia_env

# 3. Install packages in editable mode
pip install -e .
pip install -e Un-0

# 4. Optional: Configure external reference repository in shell (~/.zshrc)
export REGENERATIVE_CORNN_ROOT="/path/to/RegenariveCoRNN"
```

### Environment Variables

Configure optional environment variables in your shell (e.g., `~/.zshrc`):

| Variable | Description | Behavior when Unset / Missing |
| :--- | :--- | :--- |
| `REGENERATIVE_CORNN_ROOT` | Path to the local `RegenariveCoRNN` repository. Used by `CompilationInfoTool` in Phase 1 (Node 3) for reference code inspection and keyword searching (`read_regenerative_cornn_file`, `list_regenerative_cornn_files`, `search_regenerative_cornn`). | If unset or pointing to a non-existent directory, `CompilationInfoTool` logs a warning and gracefully bypasses all reference repository inspection tools, while continuing to serve static PyTorch 2.x compile guides (`get_torch_compile_guide`). Node 3 hides reference tools from the LLM agent and continues normal loop execution. |

---

## Running the Loop

### Running Individual Phases via Python API

```python
from agentic_ml_analog_sim.config import load_simulation_config
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.phase_0 import run_phase_0
from agentic_ml_analog_sim.phase_1 import run_phase_1
from agentic_ml_analog_sim.phase_2 import run_phase_2

# 1. Load centralized configuration
config = load_simulation_config("config/simulation_config.yaml")
db = ResultDatabase("results.db")

# 2. Execute Phase 0 (Candidate Exploration)
p0_result = run_phase_0(config_path=config, db_path=db)
print(f"Phase 0 Action: {p0_result.action}, Candidate: {p0_result.candidate_id}")

# 3. Execute Phase 1 (Rewrite, Compile & Simulate)
p1_result = run_phase_1(candidate_id=p0_result.candidate_id, config_path=config, db_path=db)
print(f"Phase 1 Status: {p1_result['status']}")

# 4. Execute Phase 2 (Audit & Final Report)
p2_result = run_phase_2(config_path=config, db_path=db, compile_pdf=True)
print(f"Phase 2 Report Generated: {p2_result.pdf_path}")
```

### Starting the Bare-Metal Ray Cluster

```bash
chia up cluster.yaml
```

---

## Testing

Run the full pytest regression suite across all unit, pipeline, and cross-phase integration tests:

```bash
pytest tests/ -v
```

The test suite covers:
- Cluster configuration syntax and resource tags.
- Simulation configuration schema validation, fallbacks, and limits synchronization.
- Phase 0 candidate generation, deduplication, and validation.
- Phase 1 rewriters, compiler execution, and `node9_phase1_error_handler` stop conditions.
- Phase 2 limit checks, LaTeX/PDF report generation, and interactive consultant sessions.
- End-to-end cross-phase token and dollar cost tracking (`test_e2e_token_monitoring.py`).
