---
name: phase-1-rewriter
description: Un-0 Model Rewrite Registry Infrastructure & Skill for Phase 1 agentic loop nodes (solvers, noise models, compiler wrappers).
---

# Phase 1 Model Rewrite Registry Skill

This skill documents the modular registry infrastructure in the `Un-0` codebase designed for Phase 1 agentic model rewrites. The infrastructure provides isolated, extensible axes for solvers, noise models, and AOT compilation wrappers.

---

## 1. Solver Registry (`un0.solver`)

Location: `un0/solver/`

### Registry API

```python
from un0.solver import SOLVER_REGISTRY, register_solver, get_solver

# Decorator to register a new solver
@register_solver("my_custom_solver")
def my_solver_step(rhs, state, time_grid, drive, **kwargs):
    ...

# Lookup a solver by string ID
solver_fn = get_solver("rk4")
```

### Signature Contract
All solver functions MUST implement the following signature:
```python
def solver_func(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    **kwargs
) -> Tensor
```
- **Inputs**:
  - `rhs`: Callable `(state, t, drive) -> velocity` computing $\frac{d\text{state}}{dt}$.
  - `state`: Initial phase state tensor shaped `(batch, state_dim)`.
  - `time_grid`: 1D tensor of time steps $t_0, t_1, \dots, t_N$.
  - `drive`: Class drive matrix tensor shaped `(batch, n, n_cond)`.
- **Output**: Trajectory tensor shaped `(len(time_grid), batch, state_dim)`.

### Registered Solvers
- `"euler"`: Explicit first-order Euler method.
- `"rk4"`: Explicit fourth-order Runge-Kutta method.
- `"euler_backward"`: Implicit backward Euler method via fixed-point iteration.
- `"parareal"`: Parallel-in-time predictor-corrector Parareal algorithm.
- `"par_ode"`: Parallelized ODE integration scheme across time steps.

---

## 2. Noise Model Registry (`un0.noise`)

Location: `un0/noise/`

### Registry API

```python
from un0.noise import NOISE_REGISTRY, register_noise_model, get_noise_model

# Decorator to register a noise wrapper class or builder
@register_noise_model("custom_noise")
class CustomNoiseWrapper(nn.Module):
    def __init__(self, dynamics, **kwargs):
        super().__init__()
        self.dynamics = dynamics

    def forward(self, state, t, drive):
        vel = self.dynamics(state, t, drive)
        return vel + noise

# Lookup noise model by string ID
noise_cls = get_noise_model("L0_static_mismatch")
noisy_dynamics = noise_cls(base_dynamics, sigma=0.01)
```

### Signature Contract
All noise model wrappers MUST be `nn.Module` instances or callables wrapping a base dynamics module and exposing:
```python
forward(state: Tensor, t: Tensor, drive: Tensor) -> Tensor
```

### Registered Noise Models
- `"L0_static_mismatch"`: Static spatial parameter mismatch noise ($\delta_K = \sigma \cdot z$).
- `"L1_stochastic_parameter_noise"`: Independent stochastic/thermal parameter noise redrawn per step.
- `"L2_functional_interface"`: Functional interface DAC/ADC quantization (bits) and readout clipping.
- `"L3_correlated_drift"`: Spatially correlated and time-dependent parameter drift.

---

## 3. Compiler Wrapper (`un0.compiler`)

Location: `un0/compiler/`

### Compiler API

```python
from un0.compiler import compile_model

# Wrap PyTorch model in torch.compile
compiled_model = compile_model(model, backend="inductor", mode="default", dynamic=True)
```

### Options
- `backend`: Compilation target backend (`"inductor"`, `"aot_eager"`, etc.).
- `mode`: Torch compile mode (`"default"`, `"reduce-overhead"`, `"max-autotune"`).
- `dynamic`: Enable dynamic shapes support (e.g., dynamic batch dimension).

---

## 4. Phase 1 Agentic Rewriter Guidelines

When executing rewrites in Phase 1:
1. **Agentic Node 1 (Solvers)**: Write or select solver functions under `un0/solver/<name>.py` and register via `@register_solver`.
2. **Agentic Node 2 (Noise Models)**: Wrap dynamics using noise modules in `un0/noise/<level>.py` registered via `@register_noise_model`.
3. **Agentic Node 3 (Compilation)**: Functionalize control flow and wrap model execution via `un0.compiler.compile_model`.
4. **Database Record**: Log registered solver name, noise model name, and compile target into `proposals` DB table.
