# Agentic Hardware/Software Co-Design Loop for Analog ML Accelerator Simulation

An agentic AI framework powered by [Chia](https://github.com/ucb-bar/chia) and Ray for co-designing and evaluating analog machine learning accelerators executing physical neural network models (e.g. continuous-time oscillatory neural networks and memristive crossbar dynamics based on the Un-0 modular baseline).

---

## Table of Contents

- [Overview](#overview)
- [Architecture (Phase 0)](#architecture-phase-0)
  - [Co-Design Loop Concept](#co-design-loop-concept)
  - [Resource & Node Roles](#resource--node-roles)
- [Cluster Configurations](#cluster-configurations)
  - [Bare-Metal Native macOS (Apple Silicon)](#bare-metal-native-macos-apple-silicon)
  - [Cloud Burst Profile (GCP + Tailnet)](#cloud-burst-profile-gcp--tailnet)
  - [Profile Comparison](#profile-comparison)
- [Bare Worker Execution Model](#bare-worker-execution-model)
  - [Why Bare-Metal on Apple Silicon?](#why-bare-metal-on-apple-silicon)
  - [Execution Flow](#execution-flow)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Environment Configuration](#environment-configuration)
  - [Package Installation](#package-installation)
- [Cluster Operations](#cluster-operations)
  - [Verifying Cluster Configuration](#verifying-cluster-configuration)
  - [Starting the Bare-Metal Cluster](#starting-the-bare-metal-cluster)
  - [Stopping the Cluster](#stopping-the-cluster)
  - [Bursting to Cloud (Phase 2)](#bursting-to-cloud-phase-2)
- [Testing](#testing)
- [Roadmap](#roadmap)

---

## Overview

Deploying neural networks directly onto analog silicon substrates enables ultra-low-latency and high-energy-efficiency inference by leveraging intrinsic circuit physics (e.g., oscillator phase locking, non-volatile conductance states). However, analog hardware introduces non-idealities such as thermal noise, phase jitter, device mismatch, non-linear I-V characteristics, and line resistance.

This repository implements an **agentic hardware/software co-design loop** that couples:
1. **Agentic Reasoning Swarm**: LLM agents (powered by Antigravity / Gemini / Claude via Model Context Protocol tools) that formulate circuit topologies, parameter search spaces, noise tolerance budgets, and quantization/mapping strategies.
2. **Multi-Fidelity Simulation Hierarchy**:
   - *Fidelity 0 (Analytical / Behavioral)*: Fast ODE and transfer-function solvers.
   - *Fidelity 1 (Phenomenological Noise)*: Dynamic models injecting thermal noise, phase-noise PSDs, and device variations.
   - *Fidelity 2 (Circuit / SPICE / GPU Batched)*: High-throughput batch inference simulations on GPU clusters.
3. **Hardware-Noise Calibration**: Automated parameter tuning bridging simulated physical dynamics with silicon measurements.

---

## Architecture (Phase 0)

Phase 0 establishes the packaging, cluster configurations, and bare-metal native execution runtime for local development on Apple Silicon, alongside cloud burst specifications for GPU acceleration.

### Co-Design Loop Concept

```
  +-----------------------------------------------------------------+
  |                       Ray Cluster Head                          |
  |             (Local Orchestrator / Apple Silicon)                |
  +-------------------------------+---------------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
  +-----------------------+                   +-----------------------+
  |    Local Bare Worker  |                   |  Cloud GPU Pool (P2)  |
  |  (Apple Silicon Host) |                   | (GCP g2-standard-8)   |
  |  - ODE Simulation     |                   | - Batched Circuit Sim |
  |  - SQLite Telemetry   |                   | - PyTorch CUDA Models |
  |  - Agent Credentials  |                   | - Tailscale Mesh      |
  +-----------------------+                   +-----------------------+
```

### Resource & Node Roles

Ray resources are tagged to enforce strict scheduling constraints:

| Resource Tag | Default Capacity | Purpose |
| :--- | :---: | :--- |
| `antigravity_creds` | `1` | Permits agentic reasoning and LLM API calls via MCP tools |
| `metal_gpu` | `1` | Allocates Apple Silicon GPU (Metal/MPS) for local acceleration |
| `sqlite_db` | `1` | Serializes state tracking, metrics, and trial history |
| `python_worker` | `4` | Parallel CPU workers for simulation and task execution |
| `nvidia_gpu` | `1` (Cloud) | NVIDIA L4 Tensor Core GPU slots on GCP worker instances |
| `sim_worker` | `1` (Cloud) | GPU simulation batch runners in Phase 2 |

---

## Cluster Configurations

The repository provides three cluster configuration profiles in YAML:

### Bare-Metal Native macOS (Apple Silicon)

- **Default profile**: [`cluster.yaml`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/cluster.yaml)
- **Dedicated bare alias**: [`cluster.bare.yaml`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/cluster.bare.yaml)

Configured for macOS (aarch64) without container virtualization:
- **Head IP**: `${THIS_MACHINE}` (auto-expanded from environment)
- **SSH Authentication**: User `${USER}`, key `~/.ssh/id_ed25519`
- **Environment**: Automatically activates `conda activate chia_env` via `.zshrc`
- **Ray Head**: Port `6379`, dashboard disabled to minimize overhead
- **Local Worker**: Runs natively with `antigravity_creds: 1` and `sqlite_db: 1`

### Cloud Burst Profile (GCP + Tailnet)

- **Cloud profile**: [`cluster.cloud.yaml`](file:///Users/chia/ChiaWorkspace/agentic-ML-analog-sim/cluster.cloud.yaml)

Configured for Phase 2 high-throughput parallel simulation bursts:
- **Tailnet Overlay**: `tailnet: manage_all: true` automatically installs userspace Tailscale daemons across all nodes and secures inter-node Ray gRPC traffic over WireGuard without exposing public ports.
- **Compute Provider**: Google Cloud Platform (GCP) Compute Engine
- **Instance Type**: `g2-standard-8` (8 vCPUs, 32 GB RAM, 1x NVIDIA L4 24GB GPU)
- **Cost Optimization**: `spot: true` (GCP Spot instances for ~60-80% cost reduction)
- **Dynamic Node Resolution**: `@sim_gpu_pool:0`, `@sim_gpu_pool:1` dynamically mapped to provisioned VM IPs.

### Profile Comparison

| Feature | `cluster.yaml` / `cluster.bare.yaml` | `cluster.cloud.yaml` |
| :--- | :--- | :--- |
| **Target Runtime** | macOS on Apple Silicon (bare-metal) | macOS (Head) + GCP Linux (Workers) |
| **Virtualization** | None (native bare-metal) | Tailnet overlay mesh + GCP VMs |
| **Worker Hardware** | Apple Silicon Unified Memory (16 GB) | NVIDIA L4 Tensor Core GPUs (24 GB) |
| **Networking** | Local host loopback / LAN | Tailscale WireGuard mesh (userspace) |
| **Phase Target** | Phase 0 & 1 (Development & Fast ODE) | Phase 2 (Large-scale batch sim) |

---

## Bare Worker Execution Model

### Why Bare-Metal on Apple Silicon?

Standard Chia workflows typically run workers inside Linux Docker containers. On macOS with Apple Silicon (aarch64), running x86_64 container images requires QEMU/Rosetta 2 emulation (via Colima or Docker Desktop). Previous benchmarking revealed major bottlenecks:
1. **Emulation Latency**: Severe CPU overhead under QEMU emulation for numerical solvers.
2. **Fork Latency & Health Checks**: Apple Silicon emulation latency can trip Ray's pipe-based parent-liveness checks, prematurely killing worker processes.
3. **Bridged Networking Quirks**: Virtual network interfaces (e.g. `col0` vs `eth0`) complicate Ray GCS address binding.
4. **Hardware Access**: Docker on macOS cannot directly expose Apple Silicon Metal/MPS GPU acceleration to Linux guests.

The **bare-metal execution model** eliminates containers completely:
- Node definitions in `available_node_types` omit the `docker:` section (`docker: None`).
- Worker processes run directly in the host's native `chia_env` Conda environment.
- Access to 16 GB Apple Silicon unified memory and MPS acceleration is direct and unthrottled.

### Execution Flow

```
1. chia up cluster.yaml
   └─> SSH to ${THIS_MACHINE}
   └─> Execute head_env_commands ("source ~/.zshrc && conda activate chia_env")
   └─> Execute head_start_ray_commands (starts Ray GCS with local resource tags)
   └─> Query node assignments
   └─> Launch local_worker (executes directly in native shell)
```

---

## Installation & Setup

### Prerequisites

1. **Operating System**: macOS (Apple Silicon recommended)
2. **Conda**: Miniconda or Anaconda with Python 3.10+
3. **SSH Access**: Local SSH server enabled (`System Settings > General > Sharing > Remote Login`) and SSH key generated:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
   cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

### Environment Configuration

Export the required environment variables in your `~/.zshrc`:

```bash
# Local development
export THIS_MACHINE=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || echo "127.0.0.1")
export USER=$(whoami)

# Cloud burst (Phase 2 - GCP & Tailscale)
export GCP_PROJECT="your-gcp-project-id"
export TS_AUTHKEY="tskey-auth-your-reusable-tailscale-key"
```

Activate the Conda environment:

```bash
conda activate chia_env
```

### Package Installation

Install `agentic-ml-analog-sim` in editable mode:

```bash
cd /Users/chia/ChiaWorkspace/agentic-ML-analog-sim
python -m pip install -e .
```

Verify the installation:

```bash
python -c "import agentic_ml_analog_sim; print(agentic_ml_analog_sim.__name__, agentic_ml_analog_sim.__version__)"
```

---

## Cluster Operations

### Verifying Cluster Configuration

Verify that `cluster.yaml` parses cleanly into a Chia `ClusterConfig`:

```bash
python -c "from chia.cluster.config import load_config; cfg = load_config('cluster.yaml'); print('Cluster name:', cfg.cluster_name)"
```

Expected output:
```
Cluster name: agentic-ml-analog-sim-mac
```

### Starting the Bare-Metal Cluster

Bring up the cluster using Chia:

```bash
chia up cluster.yaml
```

Or start the Ray head directly:

```bash
source ~/.zshrc && conda activate chia_env && \
ray start --head --port=6379 --node-ip-address=${THIS_MACHINE} \
  --resources='{"antigravity_creds": 1, "metal_gpu": 1, "sqlite_db": 1, "python_worker": 4}' \
  --include-dashboard=False
```

Check cluster status:

```bash
ray status
```

### Stopping the Cluster

Tear down the cluster cleanly:

```bash
chia down cluster.yaml
```

Or via Ray:

```bash
ray stop
```

### Bursting to Cloud (Phase 2)

When transitioning to large-scale GPU simulation runs:

1. Authenticate with Google Cloud Application Default Credentials:
   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project $GCP_PROJECT
   ```
2. Launch the hybrid cloud cluster:
   ```bash
   chia up cluster.cloud.yaml
   ```
3. After simulation completion, tear down all cloud VMs:
   ```bash
   chia down cluster.cloud.yaml
   ```

---

## Testing

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

Tests verify:
- Syntax and schema validation for `cluster.yaml` and `cluster.bare.yaml`.
- Resource and node configuration integrity for `local_worker`.
- GCP and Tailnet configuration parsing for `cluster.cloud.yaml`.
- Cloud node placeholder resolution (`@sim_gpu_pool:N`).

---

## Roadmap

- [x] **Phase 0: Environment & Cluster Foundation**
  - Packaging (`pyproject.toml`) and source layout.
  - Native Apple Silicon bare-metal execution profile (`cluster.yaml`, `cluster.bare.yaml`).
  - Phase 2 GPU cloud burst profile (`cluster.cloud.yaml`).
  - Unit test suite for configuration verification.
- [ ] **Phase 1: Simulation Engine & Noise Modeling**
  - Implement continuous-time dynamical system ODE solvers for Un-0 modular physical model.
  - Integrate phenomenological hardware noise models (thermal, phase noise, device mismatch).
  - Expose simulation routines as Chia/MCP tools.
- [ ] **Phase 2: Agentic Co-Design Swarm & Multi-Fidelity Calibration**
  - Deploy autonomous co-design loop with LLM reasoning agents.
  - High-throughput parallel GPU batching on GCP cluster.
  - Automated pareto-optimal exploration of accuracy vs energy/noise trade-offs.
