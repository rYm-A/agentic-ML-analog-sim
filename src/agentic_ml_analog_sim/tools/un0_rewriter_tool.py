"""Un-0 Model Rewriter FastMCP tool and API for Phase 1 agentic loop nodes.

Provides inspection of the Un-0 codebase, symbol extraction, and the ability to register
custom compiler functionalization rewrites, solvers, and noise models directly into
Un-0 registries.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import ray

try:
    from chia.base.tools.ChiaTool import ChiaTool
except ImportError:  # pragma: no cover
    class ChiaTool:  # type: ignore
        def __init__(
            self,
            name: str,
            task_options: Optional[Dict[str, Any]] = None,
            logging_level: int = logging.INFO,
        ) -> None:
            self.name = name
            self.task_options = task_options
            from mcp.server.fastmcp import FastMCP
            self.mcp = FastMCP(name)

        def __post_init__(self) -> None:
            pass

from mcp.server.fastmcp import FastMCP

from agentic_ml_analog_sim.tools.un0_context_tool import Un0Context, resolve_un0_root

logger = logging.getLogger(__name__)


class Un0RewriterTool(ChiaTool):
    """ChiaTool wrapper providing Un-0 inspection and rewrite registration over FastMCP."""

    def __init__(
        self,
        name: str = "un0_rewriter",
        repo_root: Optional[Union[str, Path]] = None,
        task_options: Optional[Dict[str, Any]] = None,
        auto_start: Optional[bool] = None,
        logging_level: int = logging.INFO,
    ) -> None:
        super().__init__(name, task_options=task_options, logging_level=logging_level)
        self.setup(repo_root=repo_root)

        should_start = auto_start is True or (auto_start is None and ray.is_initialized())
        if should_start:
            try:
                super().__post_init__()
            except Exception as e:
                self.logger.warning(
                    f"{self.__class__.__name__}: Ray deployment skipped or failed ({e}). "
                    "Operating in direct local execution mode."
                )

        if not getattr(self, "hostname", None):
            self.hostname = "localhost"
            self.port = getattr(self, "port", None) or 8000

    def setup(self, repo_root: Optional[Union[str, Path]] = None) -> None:
        """Initialize repository context and register FastMCP tool methods per Chia documentation."""
        self.repo_root = resolve_un0_root(repo_root)
        self.context = Un0Context(repo_root=self.repo_root)

        # Track registered rewrites directly on the tool
        self.last_registered_rewrite: Optional[str] = None
        self.last_registered_solver: Optional[str] = None
        self.last_registered_noise: Optional[str] = None
        self.last_registration_details: Optional[Dict[str, Any]] = None
        self.registrations_by_candidate: Dict[str, List[Dict[str, Any]]] = {}

        # Register MCP inspection tools
        self.mcp.add_tool(self.un0_read_file, name="un0_read_file")
        self.mcp.add_tool(self.un0_get_symbol, name="un0_get_symbol")
        self.mcp.add_tool(self.un0_search_code, name="un0_search_code")
        self.mcp.add_tool(self.un0_list_modules, name="un0_list_modules")
        self.mcp.add_tool(self.un0_get_model_summary, name="un0_get_model_summary")

        # Register MCP rewriter mutation tools
        self.mcp.add_tool(self.un0_register_compiler_rewrite, name="un0_register_compiler_rewrite")
        self.mcp.add_tool(self.un0_register_solver_rewrite, name="un0_register_solver_rewrite")
        self.mcp.add_tool(self.un0_register_noise_rewrite, name="un0_register_noise_rewrite")
        self.mcp.add_tool(self.un0_list_registered_rewrites, name="un0_list_registered_rewrites")

    # --- Inspection methods delegated to Un0Context ---

    def un0_read_file(
        self,
        relative_path: str,
        start_line: int = 1,
        end_line: int = 150,
    ) -> str:
        """Safely read lines from a file in Un-0 codebase (bounded to 150 lines max)."""
        return self.context.un0_read_file(
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
        )

    def un0_get_symbol(self, relative_path: str, symbol_name: str) -> str:
        """Extract exact source code for a class or function using AST analysis."""
        return self.context.un0_get_symbol(
            relative_path=relative_path,
            symbol_name=symbol_name,
        )

    def un0_search_code(self, query: str, directory: str = "un0") -> str:
        """Case-insensitive search in Python files in Un-0 (capped at 20 snippets)."""
        return self.context.un0_search_code(
            query=query,
            directory=directory,
        )

    def un0_list_modules(self) -> List[str]:
        """List Python modules inside Un-0/un0 directory."""
        return self.context.un0_list_modules()

    def un0_get_model_summary(self, model_variant: str = "cifar10/n1024") -> str:
        """Return structured summary of state dimensions, oscillator counts, solver, and steps."""
        return self.context.un0_get_model_summary(model_variant=model_variant)

    # --- Rewriter Registration methods ---

    def un0_register_compiler_rewrite(
        self,
        rewrite_name: str,
        candidate_id: str,
        code: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Validate and register a compiler functionalization rewrite in Un-0.

        Saves code to un0/compiler/rewrites/<candidate_id>_<clean_name>.py,
        executes module, and registers the callable into un0.compiler.COMPILER_REGISTRY.

        Args:
            rewrite_name: Target registry identifier (e.g. 'hop_while_loop_cand01').
            candidate_id: Proposal candidate ID.
            code: Python source code implementing the compiler rewrite.
            description: Optional summary of compiler optimizations performed.

        Returns:
            Dict containing registration status, rewrite_name, and file_path.
        """
        clean_name = rewrite_name.strip().replace(" ", "_")
        clean_cand = candidate_id.strip().replace(" ", "_")

        # 1. AST syntax validation
        try:
            ast.parse(code)
        except SyntaxError as syn_err:
            return {
                "status": "ERROR",
                "error": f"SyntaxError in compiler rewrite code: {syn_err}",
                "rewrite_name": clean_name,
                "candidate_id": clean_cand,
            }

        # 2. Persist code into Un-0 compiler rewrites directory
        rewrites_dir = self.repo_root / "un0" / "compiler" / "rewrites"
        rewrites_dir.mkdir(parents=True, exist_ok=True)
        init_file = rewrites_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Dynamic compiler rewrites."""\n', encoding="utf-8")

        if clean_name.startswith(f"{clean_cand}_"):
            module_base = clean_name
        else:
            module_base = f"{clean_cand}_{clean_name}"

        target_file = rewrites_dir / f"{module_base}.py"
        try:
            target_file.write_text(code, encoding="utf-8")
        except Exception as write_err:
            return {
                "status": "ERROR",
                "error": f"Failed to write rewrite file: {write_err}",
                "rewrite_name": clean_name,
                "candidate_id": clean_cand,
            }

        # 3. Import and register into COMPILER_REGISTRY
        from un0.compiler.registry import COMPILER_REGISTRY, register_compiler_rewrite

        module_name = f"un0.compiler.rewrites.{module_base}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, target_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for module {module_name}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Check if module registered via decorator
            if clean_name not in COMPILER_REGISTRY:
                # Find suitable callable inside the module
                candidate_fn = getattr(mod, clean_name, None) or getattr(mod, "rewrite", None) or getattr(mod, "rewrite_fn", None)
                if candidate_fn is None:
                    # Look for any callable defined in the module
                    for attr_name in dir(mod):
                        val = getattr(mod, attr_name)
                        if callable(val) and not attr_name.startswith("_") and getattr(val, "__module__", "") == module_name:
                            candidate_fn = val
                            break

                if candidate_fn is not None:
                    COMPILER_REGISTRY[clean_name] = candidate_fn
                else:
                    return {
                        "status": "ERROR",
                        "error": f"Module {target_file.name} did not register '{clean_name}' or define a callable rewrite.",
                        "rewrite_name": clean_name,
                        "candidate_id": clean_cand,
                    }

        except Exception as load_err:
            logger.error("Failed to load and register compiler rewrite: %s", load_err)
            return {
                "status": "ERROR",
                "error": f"Exception loading rewrite module: {load_err}",
                "rewrite_name": clean_name,
                "candidate_id": clean_cand,
            }

        logger.info(
            "Successfully registered compiler rewrite '%s' for candidate '%s' into COMPILER_REGISTRY",
            clean_name,
            clean_cand,
        )
        res = {
            "status": "SUCCESS",
            "rewrite_name": clean_name,
            "candidate_id": clean_cand,
            "file_path": str(target_file),
            "description": description,
        }
        self.last_registered_rewrite = clean_name
        self.last_registration_details = res
        self.registrations_by_candidate.setdefault(clean_cand, []).append(res)
        return res

    def un0_register_solver_rewrite(
        self,
        solver_name: str,
        candidate_id: str,
        code: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Validate and register an ODE solver function into un0.solver.SOLVER_REGISTRY.

        Saves code to un0/solver/<candidate_id>_<clean_name>.py, executes module,
        and ensures the callable is registered in un0.solver.SOLVER_REGISTRY.

        Args:
            solver_name: Target solver registry identifier (e.g. 'adaptive_heun', 'custom_rk4').
            candidate_id: Proposal candidate ID.
            code: Python source code implementing the ODE solver.
            description: Optional summary of the solver scheme and discretization.

        Returns:
            Dict containing registration status, solver_name, candidate_id, and file_path.
        """
        clean_name = solver_name.strip().replace(" ", "_")
        clean_cand = candidate_id.strip().replace(" ", "_")

        # 1. AST syntax validation
        try:
            ast.parse(code)
        except SyntaxError as syn_err:
            return {
                "status": "ERROR",
                "error": f"SyntaxError in solver code: {syn_err}",
                "solver_name": clean_name,
                "candidate_id": clean_cand,
            }

        # 2. Persist code into Un-0 solver directory
        solvers_dir = self.repo_root / "un0" / "solver"
        solvers_dir.mkdir(parents=True, exist_ok=True)
        target_file = solvers_dir / f"{clean_cand}_{clean_name}.py"
        try:
            target_file.write_text(code, encoding="utf-8")
        except Exception as write_err:
            return {
                "status": "ERROR",
                "error": f"Failed to write solver file: {write_err}",
                "solver_name": clean_name,
                "candidate_id": clean_cand,
            }

        # 3. Import and register into SOLVER_REGISTRY
        from un0.solver import SOLVER_REGISTRY, register_solver

        module_name = f"un0.solver.{clean_cand}_{clean_name}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, target_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for module {module_name}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Check if registered via decorator or directly
            if clean_name not in SOLVER_REGISTRY:
                candidate_fn = (
                    getattr(mod, clean_name, None)
                    or getattr(mod, f"{clean_name}_step", None)
                    or getattr(mod, "solve", None)
                    or getattr(mod, "step", None)
                )
                if candidate_fn is None:
                    for attr_name in dir(mod):
                        val = getattr(mod, attr_name)
                        if callable(val) and not attr_name.startswith("_") and getattr(val, "__module__", "") == module_name:
                            candidate_fn = val
                            break

                if candidate_fn is not None:
                    SOLVER_REGISTRY[clean_name] = candidate_fn
                else:
                    return {
                        "status": "ERROR",
                        "error": f"Module {target_file.name} did not register '{clean_name}' or define a callable solver function.",
                        "solver_name": clean_name,
                        "candidate_id": clean_cand,
                    }
        except Exception as load_err:
            logger.error("Failed to load and register solver rewrite: %s", load_err)
            return {
                "status": "ERROR",
                "error": f"Exception loading solver module: {load_err}",
                "solver_name": clean_name,
                "candidate_id": clean_cand,
            }

        logger.info(
            "Successfully registered solver '%s' for candidate '%s' into SOLVER_REGISTRY",
            clean_name,
            clean_cand,
        )
        res = {
            "status": "SUCCESS",
            "solver_name": clean_name,
            "candidate_id": clean_cand,
            "file_path": str(target_file),
            "description": description,
        }
        self.last_registered_solver = clean_name
        self.last_registration_details = res
        self.registrations_by_candidate.setdefault(clean_cand, []).append(res)
        return res

    def un0_register_noise_rewrite(
        self,
        noise_name: str,
        candidate_id: str,
        code: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Validate and register a hardware noise wrapper model into un0.noise.NOISE_REGISTRY.

        Saves code to un0/noise/rewrites/<candidate_id>_<clean_name>.py, executes module,
        and ensures the nn.Module class is registered in un0.noise.NOISE_REGISTRY.

        Args:
            noise_name: Target noise registry identifier (e.g. 'custom_phase_jitter', 'drift_tier3').
            candidate_id: Proposal candidate ID.
            code: Python source code implementing the noise wrapper model (inheriting from nn.Module).
            description: Optional summary of the noise mechanism and physical rationale.

        Returns:
            Dict containing registration status, noise_name, candidate_id, and file_path.
        """
        clean_name = noise_name.strip().replace(" ", "_")
        clean_cand = candidate_id.strip().replace(" ", "_")

        # 1. AST syntax validation
        try:
            ast.parse(code)
        except SyntaxError as syn_err:
            return {
                "status": "ERROR",
                "error": f"SyntaxError in noise wrapper code: {syn_err}",
                "noise_name": clean_name,
                "candidate_id": clean_cand,
            }

        # 2. Persist code into Un-0 noise rewrites directory
        noise_rewrites_dir = self.repo_root / "un0" / "noise" / "rewrites"
        noise_rewrites_dir.mkdir(parents=True, exist_ok=True)
        init_file = noise_rewrites_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Dynamic noise rewrites."""\n', encoding="utf-8")

        target_file = noise_rewrites_dir / f"{clean_cand}_{clean_name}.py"
        try:
            target_file.write_text(code, encoding="utf-8")
        except Exception as write_err:
            return {
                "status": "ERROR",
                "error": f"Failed to write noise file: {write_err}",
                "noise_name": clean_name,
                "candidate_id": clean_cand,
            }

        # 3. Import and register into NOISE_REGISTRY
        from un0.noise import NOISE_REGISTRY, register_noise_model

        module_name = f"un0.noise.rewrites.{clean_cand}_{clean_name}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, target_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for module {module_name}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Check if registered via decorator or directly
            if clean_name not in NOISE_REGISTRY:
                candidate_cls = (
                    getattr(mod, clean_name, None)
                    or getattr(mod, f"{clean_name}Noise", None)
                    or getattr(mod, "NoiseModel", None)
                    or getattr(mod, "NoiseWrapper", None)
                )
                if candidate_cls is None:
                    # Look for any class or callable defined in the module that is not private
                    for attr_name in dir(mod):
                        val = getattr(mod, attr_name)
                        if isinstance(val, type) and not attr_name.startswith("_") and getattr(val, "__module__", "") == module_name:
                            candidate_cls = val
                            break

                if candidate_cls is not None:
                    NOISE_REGISTRY[clean_name] = candidate_cls
                else:
                    return {
                        "status": "ERROR",
                        "error": f"Module {target_file.name} did not register '{clean_name}' or define a noise wrapper class.",
                        "noise_name": clean_name,
                        "candidate_id": clean_cand,
                    }
        except Exception as load_err:
            logger.error("Failed to load and register noise rewrite: %s", load_err)
            return {
                "status": "ERROR",
                "error": f"Exception loading noise module: {load_err}",
                "noise_name": clean_name,
                "candidate_id": clean_cand,
            }

        logger.info(
            "Successfully registered noise rewrite '%s' for candidate '%s' into NOISE_REGISTRY",
            clean_name,
            clean_cand,
        )
        res = {
            "status": "SUCCESS",
            "noise_name": clean_name,
            "candidate_id": clean_cand,
            "file_path": str(target_file),
            "description": description,
        }
        self.last_registered_noise = clean_name
        self.last_registration_details = res
        self.registrations_by_candidate.setdefault(clean_cand, []).append(res)
        return res

    def get_registered_noise(self, candidate_id: Optional[str] = None) -> Optional[str]:
        """Retrieve registered noise model name directly from tool state or filesystem fallback."""
        if candidate_id:
            cand_clean = candidate_id.strip().replace(" ", "_")
            cand_records = self.registrations_by_candidate.get(cand_clean)
            if cand_records:
                for rec in reversed(cand_records):
                    if "noise_name" in rec:
                        return rec["noise_name"]

            # Cross-process fallback: check filesystem in self.repo_root / "un0" / "noise" / "rewrites"
            noise_dir = self.repo_root / "un0" / "noise" / "rewrites"
            if noise_dir.is_dir():
                from un0.noise import NOISE_REGISTRY
                matching_files = sorted(
                    [p for p in noise_dir.glob(f"{cand_clean}_*.py") if p.name != "__init__.py"],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for fpath in matching_files:
                    try:
                        stem = fpath.stem
                        potential_name = stem[len(cand_clean) + 1:] if stem.startswith(f"{cand_clean}_") else stem
                        module_name = f"un0.noise.rewrites.{stem}"
                        spec = importlib.util.spec_from_file_location(module_name, fpath)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            if potential_name in NOISE_REGISTRY:
                                return potential_name
                            for attr_name in dir(mod):
                                val = getattr(mod, attr_name)
                                if isinstance(val, type) and hasattr(val, "forward"):
                                    NOISE_REGISTRY[potential_name] = val
                                    return potential_name
                    except Exception as err:
                        logger.debug("Failed loading noise from disk %s: %s", fpath, err)
            return None

        return self.last_registered_noise

    def get_registered_solver(self, candidate_id: Optional[str] = None) -> Optional[str]:
        """Retrieve registered solver name directly from tool state or filesystem fallback."""
        if candidate_id:
            cand_clean = candidate_id.strip().replace(" ", "_")
            cand_records = self.registrations_by_candidate.get(cand_clean)
            if cand_records:
                for rec in reversed(cand_records):
                    if "solver_name" in rec:
                        return rec["solver_name"]

            # Cross-process fallback: check filesystem in self.repo_root / "un0" / "solver"
            solvers_dir = self.repo_root / "un0" / "solver"
            if solvers_dir.is_dir():
                from un0.solver import SOLVER_REGISTRY
                matching_files = sorted(
                    [p for p in solvers_dir.glob(f"{cand_clean}_*.py") if p.name != "__init__.py"],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for fpath in matching_files:
                    try:
                        stem = fpath.stem
                        potential_name = stem[len(cand_clean) + 1:] if stem.startswith(f"{cand_clean}_") else stem
                        module_name = f"un0.solver.{stem}"
                        spec = importlib.util.spec_from_file_location(module_name, fpath)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            if potential_name in SOLVER_REGISTRY:
                                return potential_name
                            candidate_fn = (
                                getattr(mod, potential_name, None)
                                or getattr(mod, f"{potential_name}_step", None)
                                or getattr(mod, "solve", None)
                                or getattr(mod, "step", None)
                            )
                            if candidate_fn is not None:
                                SOLVER_REGISTRY[potential_name] = candidate_fn
                                return potential_name
                    except Exception as err:
                        logger.debug("Failed loading solver from disk %s: %s", fpath, err)
            return None

        return self.last_registered_solver

    def get_registered_rewrite(self, candidate_id: Optional[str] = None) -> Optional[str]:
        """Retrieve registered compiler rewrite name directly from tool state or filesystem fallback.

        Args:
            candidate_id: Optional proposal candidate ID.

        Returns:
            Registered rewrite name string if registered, else None.
        """
        if candidate_id:
            cand_clean = candidate_id.strip().replace(" ", "_")
            cand_records = self.registrations_by_candidate.get(cand_clean)
            if cand_records:
                for rec in reversed(cand_records):
                    if "rewrite_name" in rec:
                        return rec["rewrite_name"]

            # Cross-process fallback: check filesystem in self.repo_root / "un0" / "compiler" / "rewrites"
            rewrites_dir = self.repo_root / "un0" / "compiler" / "rewrites"
            if rewrites_dir.is_dir():
                from un0.compiler.registry import COMPILER_REGISTRY
                matching_files = sorted(
                    [p for p in rewrites_dir.glob(f"{cand_clean}_*.py") if p.name != "__init__.py"],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for fpath in matching_files:
                    try:
                        stem = fpath.stem
                        potential_name = stem[len(cand_clean) + 1:] if stem.startswith(f"{cand_clean}_") else stem
                        module_name = f"un0.compiler.rewrites.{stem}"
                        spec = importlib.util.spec_from_file_location(module_name, fpath)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            if potential_name in COMPILER_REGISTRY:
                                return potential_name
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
                                COMPILER_REGISTRY[potential_name] = candidate_fn
                                return potential_name
                    except Exception as err:
                        logger.debug("Failed loading compiler rewrite from disk %s: %s", fpath, err)
            return None

        if self.last_registered_rewrite:
            return self.last_registered_rewrite

        # If still None, check any recently modified rewrite files on disk
        rewrites_dir = self.repo_root / "un0" / "compiler" / "rewrites"
        if rewrites_dir.is_dir():
            from un0.compiler.registry import COMPILER_REGISTRY
            recent_files = sorted(
                [p for p in rewrites_dir.glob("*.py") if p.name != "__init__.py"],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for fpath in recent_files[:5]:
                try:
                    module_name = f"un0.compiler.rewrites.{fpath.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, fpath)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        for k in COMPILER_REGISTRY:
                            if k in fpath.stem:
                                return k
                except Exception as err:
                    logger.debug("Failed inspecting rewrite file %s: %s", fpath, err)

        return None

    def un0_list_registered_rewrites(self, axis: str = "compiler") -> List[str]:
        """List currently registered rewrites in Un-0 across solver, noise, or compiler.

        Scans the filesystem as a cross-process fallback if rewrites were persisted
        by a remote FastMCP server process.

        Args:
            axis: One of 'compiler', 'solver', or 'noise'.

        Returns:
            List of registered string names.
        """
        clean_axis = axis.strip().lower()
        if clean_axis == "compiler":
            from un0.compiler.registry import COMPILER_REGISTRY
            rewrites_dir = self.repo_root / "un0" / "compiler" / "rewrites"
            if rewrites_dir.is_dir():
                for fpath in rewrites_dir.glob("*.py"):
                    if fpath.name == "__init__.py":
                        continue
                    try:
                        module_name = f"un0.compiler.rewrites.{fpath.stem}"
                        spec = importlib.util.spec_from_file_location(module_name, fpath)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                    except Exception:
                        pass
            return sorted(list(COMPILER_REGISTRY.keys()))
        elif clean_axis == "solver":
            from un0.solver import SOLVER_REGISTRY
            solvers_dir = self.repo_root / "un0" / "solver"
            if solvers_dir.is_dir():
                for fpath in solvers_dir.glob("*.py"):
                    if fpath.name == "__init__.py":
                        continue
                    try:
                        module_name = f"un0.solver.{fpath.stem}"
                        spec = importlib.util.spec_from_file_location(module_name, fpath)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                    except Exception:
                        pass
            return sorted(list(SOLVER_REGISTRY.keys()))
        elif clean_axis == "noise":
            from un0.noise import NOISE_REGISTRY
            noise_dir = self.repo_root / "un0" / "noise" / "rewrites"
            if noise_dir.is_dir():
                for fpath in noise_dir.glob("*.py"):
                    if fpath.name == "__init__.py":
                        continue
                    try:
                        module_name = f"un0.noise.rewrites.{fpath.stem}"
                        spec = importlib.util.spec_from_file_location(module_name, fpath)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                    except Exception:
                        pass
            return sorted(list(NOISE_REGISTRY.keys()))
        else:
            return []
