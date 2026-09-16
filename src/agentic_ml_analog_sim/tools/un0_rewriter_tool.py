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
        self.repo_root = resolve_un0_root(repo_root)
        self.context = Un0Context(repo_root=self.repo_root)

        # Register MCP inspection tools
        self.mcp.add_tool(self.un0_read_file, name="un0_read_file")
        self.mcp.add_tool(self.un0_get_symbol, name="un0_get_symbol")
        self.mcp.add_tool(self.un0_search_code, name="un0_search_code")
        self.mcp.add_tool(self.un0_list_modules, name="un0_list_modules")
        self.mcp.add_tool(self.un0_get_model_summary, name="un0_get_model_summary")

        # Register MCP rewriter mutation tools
        self.mcp.add_tool(self.un0_register_compiler_rewrite, name="un0_register_compiler_rewrite")
        self.mcp.add_tool(self.un0_list_registered_rewrites, name="un0_list_registered_rewrites")

        should_start = auto_start is True or (auto_start is None and ray.is_initialized())
        if should_start:
            super().__post_init__()

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

        target_file = rewrites_dir / f"{clean_cand}_{clean_name}.py"
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

        module_name = f"un0.compiler.rewrites.{clean_cand}_{clean_name}"
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
        return {
            "status": "SUCCESS",
            "rewrite_name": clean_name,
            "candidate_id": clean_cand,
            "file_path": str(target_file),
            "description": description,
        }

    def un0_list_registered_rewrites(self, axis: str = "compiler") -> List[str]:
        """List currently registered rewrites in Un-0 across solver, noise, or compiler.

        Args:
            axis: One of 'compiler', 'solver', or 'noise'.

        Returns:
            List of registered string names.
        """
        clean_axis = axis.strip().lower()
        if clean_axis == "compiler":
            from un0.compiler.registry import COMPILER_REGISTRY
            return sorted(list(COMPILER_REGISTRY.keys()))
        elif clean_axis == "solver":
            from un0.solver import SOLVER_REGISTRY
            return sorted(list(SOLVER_REGISTRY.keys()))
        elif clean_axis == "noise":
            from un0.noise import NOISE_REGISTRY
            return sorted(list(NOISE_REGISTRY.keys()))
        else:
            return []
