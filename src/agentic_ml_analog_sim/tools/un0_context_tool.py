"""Safe, bounded onboarding and context exploration tool for the Un-0 repository.

Provides AST-aware symbol extraction, line-clamped file reading, module enumeration,
code searching, and architecture summarization for Kuramoto oscillator neural networks.
Can be used standalone via `Un0Context` or as a ChiaTool / FastMCP server via `Un0ContextTool`.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ray

try:
    from chia.base.tools.ChiaTool import ChiaTool
except ImportError:  # pragma: no cover
    # Fallback stub for environments where chia package is not available
    class ChiaTool:  # type: ignore
        """Minimal fallback stub for ChiaTool."""

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

UN0_ROOT_ENV = os.environ.get("UN0_ROOT")
FALLBACK_UN0_ROOT = (
    Path(UN0_ROOT_ENV).resolve()
    if UN0_ROOT_ENV
    else (Path(__file__).resolve().parents[3] / "Un-0").resolve()
)


def resolve_un0_root(repo_root: Optional[Union[str, Path]] = None) -> Path:
    """Dynamically resolve the root directory of the Un-0 repository.

    Search order:
    1. Explicit argument passed to Un0Context(repo_root=...)
    2. UN0_ROOT environment variable (if set and valid)
    3. Submodule directory inside agentic-ML-analog-sim/Un-0 (or submodules/Un-0)
    4. Sibling directory ../Un-0 relative to agentic-ML-analog-sim or CWD
    5. Default fallback.

    Args:
        repo_root: Optional explicit path provided by caller.

    Returns:
        Resolved absolute Path to the Un-0 repository root.
    """
    # 1. Explicit argument passed
    if repo_root is not None:
        return Path(repo_root).resolve()

    # 2. UN0_ROOT environment variable
    if "UN0_ROOT" in os.environ and os.environ["UN0_ROOT"].strip():
        env_cand = Path(os.environ["UN0_ROOT"].strip()).resolve()
        if env_cand.is_dir() and ((env_cand / "un0").is_dir() or (env_cand / "pyproject.toml").is_file()):
            return env_cand

    # Determine package / workspace root for agentic-ML-analog-sim
    current_file = Path(__file__).resolve()
    # Path hierarchy: .../agentic-ML-analog-sim/src/agentic_ml_analog_sim/tools/un0_context_tool.py
    pkg_root = current_file.parents[3] if len(current_file.parents) > 3 else current_file.parent

    # 2. Submodule directory inside agentic-ML-analog-sim/Un-0 or submodules/Un-0
    submodule_candidates = [
        pkg_root / "Un-0",
        pkg_root / "submodules" / "Un-0",
        Path.cwd() / "Un-0",
        Path.cwd() / "submodules" / "Un-0",
    ]
    for cand in submodule_candidates:
        if cand.is_dir() and ((cand / "un0").is_dir() or (cand / "pyproject.toml").is_file()):
            return cand.resolve()

    # 3. Sibling directory ../Un-0
    sibling_candidates = [
        pkg_root.parent / "Un-0",
        Path.cwd().parent / "Un-0",
        Path.cwd() / ".." / "Un-0",
    ]
    for cand in sibling_candidates:
        if cand.is_dir() and ((cand / "un0").is_dir() or (cand / "pyproject.toml").is_file()):
            return cand.resolve()

    # Any candidate that exists as a directory
    for cand in submodule_candidates + sibling_candidates:
        if cand.is_dir():
            return cand.resolve()

    # 4. Default fallback
    if FALLBACK_UN0_ROOT.is_dir():
        return FALLBACK_UN0_ROOT.resolve()

    return (pkg_root / "Un-0").resolve()


DEFAULT_UN0_ROOT = resolve_un0_root()


class Un0Context:
    """Core inspection engine for the Un-0 repository.

    Provides safe, bounded, read-only methods for examining code, symbols,
    modules, and architecture parameters.
    """

    def __init__(self, repo_root: Optional[Union[str, Path]] = None) -> None:
        """Initialize with path to Un-0 repository root.

        Args:
            repo_root: Optional absolute or relative path to the Un-0 repository.
                       If None, dynamically resolves via submodule, sibling, or fallback.
        """
        self.repo_root = resolve_un0_root(repo_root)
        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"Un-0 repository not found at {self.repo_root}")

    def _resolve_safe_path(self, relative_path: Union[str, Path]) -> Path:
        """Resolve a path safely, preventing directory traversal outside the repo root.

        Args:
            relative_path: File or directory path relative to repo root (or within repo).

        Returns:
            Resolved absolute Path guaranteed to be inside self.repo_root.

        Raises:
            PermissionError: If the resolved path lies outside self.repo_root.
        """
        raw_str = str(relative_path).strip()
        # Strip redundant prefix if caller included the repo folder name
        if raw_str.startswith("Un-0/"):
            raw_str = raw_str[5:]
        elif raw_str == "Un-0":
            raw_str = "."

        candidate = Path(raw_str)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.repo_root / candidate).resolve()

        try:
            resolved.relative_to(self.repo_root)
        except ValueError as err:
            raise PermissionError(
                f"Access denied: Path '{relative_path}' traverses outside repository root '{self.repo_root}'"
            ) from err

        return resolved

    def un0_read_file(
        self,
        relative_path: str,
        start_line: int = 1,
        end_line: int = 150,
    ) -> str:
        """Safely read a slice of lines from a file in the Un-0 codebase.

        Restricts paths within Un-0/, clamps reading to at most 150 lines per call,
        prepends a status header, and appends a continuation notice if lines remain.

        Args:
            relative_path: Path relative to repository root.
            start_line: 1-indexed starting line number (default: 1).
            end_line: 1-indexed ending line number (default: 150, clamped to start + 149).

        Returns:
            Formatted string containing header, file content, and optional continuation notice.
        """
        target_path = self._resolve_safe_path(relative_path)
        if not target_path.exists():
            return f"Error: File '{relative_path}' not found at {target_path}."
        if not target_path.is_file():
            return f"Error: Path '{relative_path}' is a directory, not a file."

        rel_display = target_path.relative_to(self.repo_root).as_posix()
        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading '{relative_path}': {e}"

        total = len(lines)
        if total == 0:
            return f"[File: {rel_display} | Total lines: 0 | Showing lines 0 to 0]\n(Empty file)"

        start = max(1, int(start_line))
        if start > total:
            return (
                f"[File: {rel_display} | Total lines: {total} | Showing lines {start} to {total}]\n"
                f"[NOTE: start_line={start} is past end of file (total lines: {total})]"
            )

        # Clamp line slice to at most 150 lines per call
        max_slice = 150
        requested_end = max(start, int(end_line))
        end = min(start + max_slice - 1, requested_end, total)

        slice_lines = lines[start - 1 : end]
        content = "".join(slice_lines)
        if not content.endswith("\n"):
            content += "\n"

        header = f"[File: {rel_display} | Total lines: {total} | Showing lines {start} to {end}]\n"
        result = header + content

        if end < total:
            remaining = total - end
            next_start = end + 1
            next_end = end + max_slice
            result += (
                f"[NOTE: {remaining} lines remaining. Call un0_read_file with "
                f"start_line={next_start}, end_line={next_end} to read further]"
            )

        return result

    def un0_get_symbol(self, relative_path: str, symbol_name: str) -> str:
        """Extract exact source code lines for a class or function using AST analysis.

        If found, returns the line range and the exact source code block.
        If not found, lists available top-level classes and functions in that file.

        Args:
            relative_path: Path to the Python file relative to repository root.
            symbol_name: Name of the class, function, or ClassName.method.

        Returns:
            Source code block with line range or directory listing of symbols.
        """
        target_path = self._resolve_safe_path(relative_path)
        if not target_path.exists():
            return f"Error: File '{relative_path}' not found at {target_path}."
        if not target_path.is_file():
            return f"Error: Path '{relative_path}' is a directory, not a regular file."

        rel_display = target_path.relative_to(self.repo_root).as_posix()
        try:
            source = target_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(target_path))
        except Exception as e:
            return f"Error parsing Python AST for '{relative_path}': {e}"

        source_lines = source.splitlines(keepends=True)

        top_classes: List[ast.ClassDef] = []
        top_functions: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                top_classes.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top_functions.append(node)

        # Search for exact symbol match
        target_node: Optional[Union[ast.AST, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]] = None

        # 1. Exact top-level class match
        for cls_node in top_classes:
            if cls_node.name == symbol_name:
                target_node = cls_node
                break

        # 2. Exact top-level function match
        if target_node is None:
            for fn_node in top_functions:
                if fn_node.name == symbol_name:
                    target_node = fn_node
                    break

        # 3. ClassName.method_name match
        if target_node is None and "." in symbol_name:
            cls_name, method_name = symbol_name.split(".", 1)
            for cls_node in top_classes:
                if cls_node.name == cls_name:
                    for item in cls_node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                            target_node = item
                            break
                    if target_node:
                        break

        # 4. Method name inside any class
        if target_node is None:
            for cls_node in top_classes:
                for item in cls_node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == symbol_name:
                        target_node = item
                        break
                if target_node:
                    break

        if target_node is not None:
            decorators = getattr(target_node, "decorator_list", [])
            start = min([target_node.lineno] + [d.lineno for d in decorators])
            end = getattr(target_node, "end_lineno", target_node.lineno)
            code_block = "".join(source_lines[start - 1 : end])
            return (
                f"[Symbol: {symbol_name} | File: {rel_display} | Lines {start} to {end}]\n"
                f"{code_block}"
            )

        # Symbol not found: compile listing of available classes and functions
        class_items = [
            f"  - class {c.name} (lines {min([c.lineno] + [d.lineno for d in getattr(c, 'decorator_list', [])])} to {c.end_lineno})"
            for c in top_classes
        ]
        func_items = [
            f"  - def {f.name} (lines {min([f.lineno] + [d.lineno for d in getattr(f, 'decorator_list', [])])} to {f.end_lineno})"
            for f in top_functions
        ]

        sections = [f"Symbol '{symbol_name}' not found in '{rel_display}'."]
        if class_items:
            sections.append("Available classes:\n" + "\n".join(class_items))
        if func_items:
            sections.append("Available functions:\n" + "\n".join(func_items))
        if not class_items and not func_items:
            sections.append("No top-level classes or functions found in file.")

        return "\n\n".join(sections)

    def un0_search_code(self, query: str, directory: str = "un0") -> str:
        """Case-insensitive search for a string in Python files inside Un-0/<directory>.

        Args:
            query: Substring to search for.
            directory: Subdirectory within Un-0 to search (default: "un0").

        Returns:
            Snippets with file path and line numbers (capped at 20 snippets total).
        """
        search_dir = self._resolve_safe_path(directory)
        if not search_dir.exists():
            return f"Error: Directory '{directory}' does not exist at {search_dir}."
        if not search_dir.is_dir():
            return f"Error: Path '{directory}' is not a directory."

        q_lower = query.lower()
        snippets: List[str] = []
        total_matches = 0
        max_snippets = 20

        py_files = sorted(search_dir.rglob("*.py"))
        for py_file in py_files:
            rel = py_file.relative_to(self.repo_root).as_posix()
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for idx, line in enumerate(content.splitlines(), start=1):
                if q_lower in line.lower():
                    total_matches += 1
                    if len(snippets) < max_snippets:
                        snippets.append(f"{rel}:{idx}: {line.strip()}")

        if not snippets:
            return f"No matches found for '{query}' in '{directory}'."

        header = (
            f"Search results for '{query}' in '{directory}' "
            f"(found {total_matches} match{'es' if total_matches != 1 else ''}):\n"
        )
        body = "\n".join(snippets)
        if total_matches > max_snippets:
            body += (
                f"\n\n[NOTE: Showing first {max_snippets} of {total_matches} matches. "
                "Narrow your query to see more specific results.]"
            )
        return header + body

    def un0_list_modules(self) -> List[str]:
        """Return relative paths of all .py files inside Un-0/un0.

        Returns:
            List of repository-relative POSIX paths to Python modules.
        """
        un0_dir = self.repo_root / "un0"
        if not un0_dir.exists() or not un0_dir.is_dir():
            return []
        py_files = sorted(un0_dir.rglob("*.py"))
        return [p.relative_to(self.repo_root).as_posix() for p in py_files]

    def un0_get_model_summary(self, model_variant: str = "cifar10/n1024") -> str:
        """Return structured summary of state dimensions, oscillator counts, solver, and steps.

        Args:
            model_variant: Identifier for model architecture (default: "cifar10/n1024").

        Returns:
            Formatted architecture specification and dimensions.
        """
        variant = model_variant.lower().strip()
        if "n1024" in variant or variant in ("cifar10/n1024", "cifar10"):
            return (
                "Model Variant: cifar10/n1024\n"
                "======================================================================\n"
                "Oscillator Counts & Classification:\n"
                "  - Main Oscillators (n): 1024\n"
                "  - Conditioning Oscillators (n_cond): 8\n"
                "  - Total Joint Oscillators: 1032\n"
                "  - Classes: 10 (CIFAR-10)\n\n"
                "State Dimensions & Parameter Matrices:\n"
                "  - Phase state vector (theta): [batch_size, 1032]\n"
                "    * Main block phases: theta[:, :1024]\n"
                "    * Conditioning block phases: theta[:, 1024:]\n"
                "  - Main internal coupling matrix (K): shape (1024, 1024), zero diagonal\n"
                "  - Conditioning coupling matrix (K_cond): shape (8, 8), zero diagonal\n"
                "  - Class drive matrix (K_drive): shape (10, 1024, 8) [one-way drive cond -> main]\n"
                "  - Main natural frequencies (omega): shape (1, 1024)\n"
                "  - Conditioning natural frequencies (omega_cond): shape (1, 8)\n\n"
                "ODE Dynamics & Numerical Solver:\n"
                "  - Default Solver: rk4 (4th-order Runge-Kutta; CLI checkpoint uses euler)\n"
                "  - Number of Steps (num_steps): 25 (step size dt = 0.04; CLI checkpoint uses 10)\n"
                "  - Integration Time: 1.0\n"
                "  - Kuramoto Velocity Equations:\n"
                "      dtheta_main/dt = omega + cos(theta_main)*(sin(theta_main)@K.T)\n"
                "                       - sin(theta_main)*(cos(theta_main)@K.T)\n"
                "                       + cos(theta_main)*(K_drive[c]@sin(theta_cond))\n"
                "                       - sin(theta_main)*(K_drive[c]@cos(theta_cond))\n"
                "      dtheta_cond/dt = omega_cond + cos(theta_cond)*(sin(theta_cond)@K_cond.T)\n"
                "                       - sin(theta_cond)*(cos(theta_cond)@K_cond.T)\n\n"
                "Readout & Decoder Architecture:\n"
                "  - Readout Phases: First 1024 phases (main block)\n"
                "  - Relativization: ref_oscillator (CLI training default: mean_relative)\n"
                "  - Encoding: sin_cos -> Feature dimension: 2048 (2 * 1024)\n"
                "  - Decoder: ResizeConvDecoder\n"
                "      * Input: [batch_size, 128, 4, 4] (128 channels = 2048 / 16 spatial)\n"
                "      * Architecture: 3 resize-conv upsampling stages (4x4 -> 8x8 -> 16x16 -> 32x32)\n"
                "      * Output: [batch_size, 3, 32, 32] RGB image with tanh activation (gain=0.5)\n"
                "  - Parameterization: standard\n"
                "  - Total Parameters: ~1.3M"
            )
        elif "n2048" in variant:
            return (
                "Model Variant: cifar10/n2048\n"
                "======================================================================\n"
                "Oscillator Counts: 2048 main, 8 cond, 10 classes (Total: 2056 oscillators)\n"
                "State Dimensions:\n"
                "  - K: (2048, 2048), K_cond: (8, 8), K_drive: (10, 2048, 8)\n"
                "  - Feature Dimension: 4096 (sin_cos)\n"
                "  - Decoder In Channels: 256 (4x4 spatial)\n"
                "ODE Dynamics: Solver rk4, 25 steps, Integration Time 1.0\n"
                "Total Parameters: ~4.9M"
            )
        elif "n4096" in variant:
            return (
                "Model Variant: cifar10/n4096\n"
                "======================================================================\n"
                "Oscillator Counts: 4096 main, 8 cond, 10 classes (Total: 4104 oscillators)\n"
                "State Dimensions:\n"
                "  - K: (4096, 4096), K_cond: (8, 8), K_drive: (10, 4096, 8)\n"
                "  - Feature Dimension: 8192 (sin_cos)\n"
                "  - Decoder In Channels: 512 (4x4 spatial)\n"
                "ODE Dynamics: Solver rk4, 25 steps, Integration Time 1.0\n"
                "Total Parameters: ~19.4M"
            )
        elif "imagenet" in variant:
            return (
                f"Model Variant: {model_variant}\n"
                "======================================================================\n"
                "Oscillator Counts: 16384 (n16384) / 10240 (n10240) / 6656 (n6656) main, 1 cond, 1000 classes\n"
                "ODE Dynamics: Solver euler, 10 steps, Integration Time 1.0, Parameterization: mup\n"
                "Output: [batch_size, 3, 64, 64]"
            )
        else:
            return self.un0_get_model_summary("cifar10/n1024")


class Un0ContextTool(ChiaTool):
    """ChiaTool wrapper exposing Un0Context inspection tools over MCP / Ray."""

    def __init__(
        self,
        name: str = "un0_context",
        repo_root: Optional[Union[str, Path]] = None,
        task_options: Optional[Dict[str, Any]] = None,
        auto_start: Optional[bool] = None,
        logging_level: int = logging.INFO,
    ) -> None:
        """Initialize the Un0ContextTool and register MCP tools.

        Args:
            name: Tool name for logging and MCP endpoint routing.
            repo_root: Optional path to the Un-0 repository (defaults to dynamic resolution).
            task_options: Optional Ray task placement and resource options.
            auto_start: If True, deploy server actor immediately via Ray.
                        If False, defer deployment. If None, auto-start only if
                        Ray is already initialized.
            logging_level: Logging level for tool logger.
        """
        super().__init__(name, task_options=task_options, logging_level=logging_level)
        self.setup(repo_root=repo_root)

        # Auto-start Ray actor if requested or if Ray cluster is active
        should_start = auto_start is True or (auto_start is None and ray.is_initialized())
        if should_start:
            try:
                super().__post_init__()
            except Exception as e:
                self.logger.warning(
                    f"{self.__class__.__name__}: Ray deployment skipped or failed ({e}). "
                    "Operating in direct local execution mode."
                )

    def setup(self, repo_root: Optional[Union[str, Path]] = None) -> None:
        """Register MCP tool endpoints with FastMCP according to Chia documentation."""
        self.context = Un0Context(repo_root=repo_root)

        # Register tools with FastMCP instance
        self.mcp.add_tool(self.un0_read_file, name="un0_read_file")
        self.mcp.add_tool(self.un0_get_symbol, name="un0_get_symbol")
        self.mcp.add_tool(self.un0_search_code, name="un0_search_code")
        self.mcp.add_tool(self.un0_list_modules, name="un0_list_modules")
        self.mcp.add_tool(self.un0_get_model_summary, name="un0_get_model_summary")

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


def create_un0_mcp_server(
    repo_root: Optional[Union[str, Path]] = None,
    server_name: str = "un0_context",
) -> FastMCP:
    """Create and configure a standalone FastMCP server for Un-0 tools.

    Args:
        repo_root: Optional path to the Un-0 repository (defaults to dynamic resolution).
        server_name: MCP server name.

    Returns:
        Configured FastMCP instance with all un0_* tools registered.
    """
    ctx = Un0Context(repo_root=repo_root)
    mcp_app = FastMCP(server_name)

    @mcp_app.tool()
    def un0_read_file(relative_path: str, start_line: int = 1, end_line: int = 150) -> str:
        return ctx.un0_read_file(relative_path, start_line=start_line, end_line=end_line)

    @mcp_app.tool()
    def un0_get_symbol(relative_path: str, symbol_name: str) -> str:
        return ctx.un0_get_symbol(relative_path, symbol_name)

    @mcp_app.tool()
    def un0_search_code(query: str, directory: str = "un0") -> str:
        return ctx.un0_search_code(query, directory)

    @mcp_app.tool()
    def un0_list_modules() -> List[str]:
        return ctx.un0_list_modules()

    @mcp_app.tool()
    def un0_get_model_summary(model_variant: str = "cifar10/n1024") -> str:
        return ctx.un0_get_model_summary(model_variant)

    return mcp_app
