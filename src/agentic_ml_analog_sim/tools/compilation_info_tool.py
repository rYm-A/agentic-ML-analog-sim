"""Compilation Information FastMCP tool and API for PyTorch AOT compilation.

Provides access to Torch Compile documentation, PyTorch Higher-Order Operators (HOP)
guides (torch._higher_order_ops.while_loop, associative_scan), and reference examples
from the RegenerativeCoRNN repository.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

logger = logging.getLogger(__name__)

DEFAULT_REGENERATIVE_CORNN_ROOT: Optional[Path] = (
    Path(os.environ["REGENERATIVE_CORNN_ROOT"]).resolve()
    if "REGENERATIVE_CORNN_ROOT" in os.environ
    else None
)

TORCH_COMPILE_GUIDES: Dict[str, str] = {
    "while_loop": (
        "# PyTorch Higher-Order Operator: while_loop\n"
        "Reference: torch._higher_order_ops.while_loop(cond_fn, body_fn, carried_inputs, additional_inputs=())\n\n"
        "Key Rules for TorchDynamo and AOTInductor Whole-Graph Tracing:\n"
        "1. Signature:\n"
        "   - `cond_fn(*carried_inputs, *additional_inputs) -> torch.Tensor (scalar boolean)`\n"
        "   - `body_fn(*carried_inputs, *additional_inputs) -> tuple matching carried_inputs types/shapes`\n"
        "2. Functional purity:\n"
        "   - No in-place mutations (`x += y`) on captured tensors. Use out-of-place arithmetic (`x = x + y`).\n"
        "   - Carried inputs must remain pure PyTorch tensors.\n"
        "3. Loop condition:\n"
        "   - The predicate returned by `cond_fn` must be a scalar boolean Tensor (`step < max_steps`).\n"
        "   - Do NOT use Python built-in `bool(step < max_steps)` as it triggers a graph break.\n"
        "4. Dynamic Batching:\n"
        "   - All tensor operations inside `body_fn` must support arbitrary batch size (B >= 1).\n"
        "   - Avoid hardcoding batch dimension constants.\n"
    ),
    "associative_scan": (
        "# PyTorch Higher-Order Operator: associative_scan\n"
        "Reference: torch._higher_order_ops.associative_scan(operator, inputs, dim)\n\n"
        "Key Rules:\n"
        "1. Operator must be strictly associative: (a * b) * c == a * (b * c).\n"
        "2. Useful for parallelizing linear recurrence forms: h_{t} = A_t * h_{t-1} + B_t * x_t.\n"
        "3. Supported under Inductor with tree reduction codegen on GPU/accelerators.\n"
    ),
    "torch_compile": (
        "# PyTorch 2.x torch.compile Documentation\n"
        "URL: https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html\n\n"
        "1. Dynamic Shapes:\n"
        "   - Pass `dynamic=True` to compile models without recompiling when batch size varies.\n"
        "   - Guard assertions are generated dynamically for tensor dimensions.\n"
        "2. Higher-Order Operators:\n"
        "   - Python loops with data-dependent steps cause graph breaks in TorchDynamo.\n"
        "   - Wrapping loop dynamics in HOPs (`while_loop`) allows TorchDynamo to capture the entire\n"
        "     trajectory calculation in a single FX Graph and compile fused kernels in Inductor.\n"
        "3. Hardware Targets:\n"
        "   - 'mps': Metal Performance Shaders backend on Apple Silicon.\n"
        "   - 'cuda': NVIDIA CUDA compiler backend with Triton kernel generation.\n"
        "   - 'cpu': AOT Inductor C++ code generator.\n"
    ),
    "dynamo_optimizations": (
        "# TorchDynamo and Inductor Configuration for Analog Simulation\n"
        "- Enable scalar capture: `torch._dynamo.config.capture_scalar_outputs = True`\n"
        "- Suppress fallbacks: Set backend='inductor', mode='default'\n"
        "- Avoid Python closures over non-leaf tensor parameters; pass module weights as additional inputs.\n"
    ),
}


class CompilationInfoTool(ChiaTool):
    """ChiaTool wrapper providing compilation guides and RegenerativeCoRNN reference inspection."""

    def __init__(
        self,
        name: str = "compilation_info",
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

    def setup(self, repo_root: Optional[Union[str, Path]] = None) -> None:
        """Register MCP tool endpoints and configure reference repository root per Chia documentation."""
        target_path = repo_root or os.environ.get("REGENERATIVE_CORNN_ROOT") or DEFAULT_REGENERATIVE_CORNN_ROOT
        if target_path:
            self.repo_root: Optional[Path] = Path(target_path).resolve()
            self.is_available: bool = self.repo_root.is_dir()
        else:
            self.repo_root = None
            self.is_available = False

        if not self.is_available:
            logger.warning(
                "RegenerativeCoRNN repository not found (path: %s). "
                "Bypassing all RegenerativeCoRNN reference inspection logic and disabling reference tools.",
                self.repo_root,
            )
        else:
            # Register RegenerativeCoRNN reference inspection tools only if repository exists
            self.mcp.add_tool(self.read_regenerative_cornn_file, name="read_regenerative_cornn_file")
            self.mcp.add_tool(self.list_regenerative_cornn_files, name="list_regenerative_cornn_files")
            self.mcp.add_tool(self.search_regenerative_cornn, name="search_regenerative_cornn")

        # Static PyTorch compile documentation is always available
        self.mcp.add_tool(self.get_torch_compile_guide, name="get_torch_compile_guide")

    def get_torch_compile_guide(self, topic: str = "all") -> str:
        """Query official PyTorch compile, Dynamo, Inductor, and HOP while_loop documentation.

        Args:
            topic: Topic key ('while_loop', 'associative_scan', 'torch_compile', 'dynamo_optimizations', or 'all').

        Returns:
            Formatted documentation string.
        """
        clean_topic = topic.strip().lower()
        if clean_topic in TORCH_COMPILE_GUIDES:
            return TORCH_COMPILE_GUIDES[clean_topic]
        if clean_topic == "all":
            return "\n\n---\n\n".join(TORCH_COMPILE_GUIDES.values())

        available = list(TORCH_COMPILE_GUIDES.keys()) + ["all"]
        return f"Topic '{topic}' not found. Available topics: {available}\n\n" + TORCH_COMPILE_GUIDES["torch_compile"]

    def read_regenerative_cornn_file(
        self,
        relative_path: str = "model/implicit_parallel.py",
        start_line: int = 1,
        end_line: int = 100,
    ) -> str:
        """Safely read lines from a reference file in the RegenerativeCoRNN repository.

        Args:
            relative_path: Path relative to RegenerativeCoRNN root (e.g. 'model/implicit_parallel.py').
            start_line: 1-indexed starting line.
            end_line: 1-indexed ending line (max 150 lines per query).

        Returns:
            Line-numbered file contents.
        """
        if not self.is_available or self.repo_root is None or not self.repo_root.is_dir():
            logger.warning("RegenerativeCoRNN repository not configured or directory not found at '%s'; read operation bypassed.", self.repo_root)
            return f"Warning: RegenerativeCoRNN repository not configured or directory not found at '{self.repo_root}'. Inspection bypassed."

        target = (self.repo_root / relative_path).resolve()

        try:
            target.relative_to(self.repo_root)
        except ValueError:
            return f"Error: Path '{relative_path}' traverses outside RegenerativeCoRNN repository."

        if not target.is_file():
            return f"Error: File '{relative_path}' does not exist in RegenerativeCoRNN repository."

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file '{relative_path}': {e}"

        total_lines = len(lines)
        clamped_start = max(1, start_line)
        clamped_end = min(total_lines, max(clamped_start, end_line))
        if clamped_end - clamped_start > 150:
            clamped_end = clamped_start + 150

        selected = lines[clamped_start - 1 : clamped_end]
        header = f"=== File: {relative_path} (Lines {clamped_start}-{clamped_end} of {total_lines}) ===\n"
        numbered_body = "".join(
            f"{i + clamped_start:4d}: {line}" for i, line in enumerate(selected)
        )
        return header + numbered_body

    def list_regenerative_cornn_files(self) -> List[str]:
        """List relevant Python and documentation files in the RegenerativeCoRNN repository.

        Returns:
            List of relative file paths.
        """
        if not self.is_available or self.repo_root is None or not self.repo_root.is_dir():
            logger.warning("RegenerativeCoRNN repository not configured or directory not found at '%s'; listing bypassed.", self.repo_root)
            return []

        results: List[str] = []
        for root, dirs, files in os.walk(self.repo_root):
            # Skip hidden and cache dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in sorted(files):
                if f.endswith((".py", ".md", ".txt", ".json", ".sh")):
                    full = Path(root) / f
                    results.append(str(full.relative_to(self.repo_root)))
        return sorted(results)

    def search_regenerative_cornn(self, query: str) -> str:
        """Search Python files in RegenerativeCoRNN for occurrences of a substring.

        Args:
            query: Substring to search for (e.g. 'while_loop', 'associative_scan').

        Returns:
            Matching file paths and line snippets.
        """
        if not self.is_available or self.repo_root is None or not self.repo_root.is_dir():
            logger.warning("RegenerativeCoRNN repository not configured or directory not found at '%s'; search operation bypassed.", self.repo_root)
            return f"Warning: RegenerativeCoRNN repository not configured or directory not found at '{self.repo_root}'. Search bypassed."

        matches: List[str] = []
        q_lower = query.lower()

        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                file_path = Path(root) / f
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        for idx, line in enumerate(fh, start=1):
                            if q_lower in line.lower():
                                rel = file_path.relative_to(self.repo_root)
                                matches.append(f"{rel}:{idx}: {line.strip()[:120]}")
                                if len(matches) >= 25:
                                    break
                except Exception:
                    continue
                if len(matches) >= 25:
                    break
            if len(matches) >= 25:
                break

        if not matches:
            return f"No matches found for '{query}' in RegenerativeCoRNN."
        return f"Found {len(matches)} matches in RegenerativeCoRNN:\n" + "\n".join(matches)
