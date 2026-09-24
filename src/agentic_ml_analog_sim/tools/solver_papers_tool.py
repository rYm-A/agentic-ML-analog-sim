"""Solvers Related Papers and Tiers FastMCP tool and API for Phase 1 agentic loop.

Provides access to continuous systems solver literature, mathematical formulations,
solver tiers (Tier 0 to Tier 3), implementation guidelines, and failure modes
from the solvers-for-continuous-systems skill.
"""

from __future__ import annotations

import logging
import re
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

# Relative path from agentic-ML-analog-sim/ repository root
SKILL_RELATIVE_PATH = Path("skills/solvers-for-continuous-systems/SKILL.md")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SKILL_PATH = REPO_ROOT / SKILL_RELATIVE_PATH


def _resolve_skill_path(skill_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve skill file path relative to agentic-ML-analog-sim root or current working dir."""
    if skill_path is not None:
        p = Path(skill_path)
        if not p.is_absolute():
            for base in (REPO_ROOT, Path.cwd()):
                candidate = (base / p).resolve()
                if candidate.exists():
                    return candidate
        p = p.resolve()
        if p.exists():
            return p

    candidates = [
        DEFAULT_SKILL_PATH,
        (Path.cwd() / SKILL_RELATIVE_PATH).resolve(),
        (REPO_ROOT / SKILL_RELATIVE_PATH).resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    return DEFAULT_SKILL_PATH


class SolverPapersTool(ChiaTool):
    """ChiaTool wrapper providing continuous systems solver literature, equations, and tier guides."""

    def __init__(
        self,
        name: str = "solver_papers",
        skill_path: Optional[Union[str, Path]] = None,
        task_options: Optional[Dict[str, Any]] = None,
        auto_start: Optional[bool] = None,
        logging_level: int = logging.INFO,
    ) -> None:
        super().__init__(name, task_options=task_options, logging_level=logging_level)
        self.setup(skill_path=skill_path)

        should_start = auto_start is True or (auto_start is None and ray.is_initialized())
        if should_start:
            super().__post_init__()

    def setup(self, skill_path: Optional[Union[str, Path]] = None) -> None:
        """Register MCP tool methods with FastMCP according to Chia documentation."""
        self.skill_path = _resolve_skill_path(skill_path)
        self._content: Optional[str] = None

        # Register MCP tools in setup() function per Chia documentation
        self.mcp.add_tool(self.get_solver_skill_summary, name="get_solver_skill_summary")
        self.mcp.add_tool(self.get_solver_tier_guide, name="get_solver_tier_guide")
        self.mcp.add_tool(self.search_solver_papers, name="search_solver_papers")
        self.mcp.add_tool(self.read_solver_skill_section, name="read_solver_skill_section")

    def _load_content(self) -> str:
        """Cache and return the markdown content of the solver skill."""
        if self._content is None:
            if self.skill_path.exists():
                self._content = self.skill_path.read_text(encoding="utf-8")
            else:
                self._content = (
                    "# Solvers for Continuous Systems\n\n"
                    "Error: Skill document could not be located at " + str(self.skill_path)
                )
        return self._content

    def get_solver_skill_summary(self) -> str:
        """Return high-level summary of solver tiers, role of rewriter agent, and shared vocabulary.

        Returns:
            Structured summary covering Tier 0 (explicit), Tier 1 (adaptive), Tier 2 (implicit/stiff),
            Tier 3 (parallel-in-time), and vocabulary (stiffness, phase error, work vs span).
        """
        content = self._load_content()
        summary_sections = [
            "# Solvers for Continuous Systems - Reference Summary",
            "This guide provides mathematical foundations and PyTorch implementation patterns "
            "for physical-neural-network simulations (Kuramoto oscillators on analog silicon).",
            "",
            "## Core Rewriter Signature in Un-0:",
            "All solvers in Un-0 must match the signature:",
            "  `solver_fn(rhs, state, time_grid, drive, **kwargs) -> Tensor`",
            "and return a trajectory tensor of shape `(len(time_grid), batch_size, state_dim)`.",
            "",
            "## Available Solver Tiers:",
            "1. **TIER 0 - Fixed-step explicit transient integration**:",
            "   - Methods: Euler, Midpoint/Heun, RK4.",
            "   - Suitability: Non-stiff ODEs, phase-reduced coupled oscillators.",
            "   - Cost: Sequential span O(N_steps), 1 to 4 RHS evaluations per step.",
            "",
            "2. **TIER 1 - Adaptive explicit integration & event-driven stopping**:",
            "   - Methods: RK45, Tsit5, embedded Runge-Kutta pairs.",
            "   - Suitability: Non-stiff ODEs with varying activity or uncertain settling time.",
            "",
            "3. **TIER 2 - Implicit / Stiff Solvers**:",
            "   - Methods: Backward Euler (L-stable), Crank-Nicolson / Trapezoidal (A-stable).",
            "   - Suitability: Stiff decaying modes, tightly coupled oscillators, DAE formulations.",
            "",
            "4. **TIER 3 - Parallel-in-time / Recurrence Solvers**:",
            "   - Methods: Associative scan, Parareal, DEER, MGRIT.",
            "   - Suitability: Accelerating long horizons when batch parallelism is saturated.",
            "",
            "## Shared Numerical Vocabulary:",
            "- **Stiffness**: Fast stable modes forcing explicit methods to take tiny dt.",
            "- **Phase Error**: Timing displacement of an oscillator; accumulates over time.",
            "- **Work vs Span**: Total operations vs critical path length under parallel execution.",
        ]
        return "\n".join(summary_sections)

    def get_solver_tier_guide(self, tier: str = "tier_0") -> str:
        """Retrieve detailed theory, PyTorch implementation path, starting settings, and failure modes for a solver tier.

        Args:
            tier: One of 'tier_0' (or 'fixed_step'), 'tier_1' (or 'adaptive'),
                  'tier_2' (or 'implicit'/'stiff'), 'tier_3' (or 'parallel_in_time'/'parareal').

        Returns:
            Extracted markdown section for the requested tier.
        """
        content = self._load_content()
        tier_clean = tier.strip().lower()

        tier_header_map = {
            "0": "TIER 0 — Fixed-step explicit transient integration",
            "tier_0": "TIER 0 — Fixed-step explicit transient integration",
            "fixed_step": "TIER 0 — Fixed-step explicit transient integration",
            "explicit": "TIER 0 — Fixed-step explicit transient integration",
            "euler": "TIER 0 — Fixed-step explicit transient integration",
            "rk4": "TIER 0 — Fixed-step explicit transient integration",
            "1": "TIER 1 — Adaptive explicit integration",
            "tier_1": "TIER 1 — Adaptive explicit integration",
            "adaptive": "TIER 1 — Adaptive explicit integration",
            "2": "TIER 2",
            "tier_2": "TIER 2",
            "implicit": "TIER 2",
            "stiff": "TIER 2",
            "3": "TIER 3",
            "tier_3": "TIER 3",
            "parallel": "TIER 3",
            "parareal": "TIER 3",
        }

        target_pattern = tier_header_map.get(tier_clean)
        if not target_pattern:
            for k, pattern in tier_header_map.items():
                if k in tier_clean:
                    target_pattern = pattern
                    break

        if not target_pattern:
            return (
                f"Unknown tier: {tier!r}. Valid options: 'tier_0' (fixed-step explicit), "
                f"'tier_1' (adaptive explicit), 'tier_2' (implicit/stiff), 'tier_3' (parallel-in-time)."
            )

        # Extract tier block
        pattern = rf"(#+\s+{re.escape(target_pattern)}.*?)(?=\n#+\s+TIER|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback search if exact heading regex misses
        idx = content.find(target_pattern)
        if idx != -1:
            next_tier = content.find("# TIER", idx + len(target_pattern))
            if next_tier != -1:
                return content[idx:next_tier].strip()
            return content[idx : idx + 4000].strip()

        return f"Could not extract section for {target_pattern}."

    def search_solver_papers(self, query: str) -> str:
        """Case-insensitive search across solver literature, equations, and papers in the skill document.

        Args:
            query: Keyword or phrase to search for (e.g. 'Heun', 'stiffness', 'rk4', 'backward euler').

        Returns:
            Formatted snippets containing matches.
        """
        content = self._load_content()
        lines = content.splitlines()
        q_lower = query.lower()

        matches: List[str] = []
        for i, line in enumerate(lines):
            if q_lower in line.lower():
                start = max(0, i - 2)
                end = min(len(lines), i + 4)
                snippet = "\n".join(lines[start:end])
                matches.append(f"--- Match around line {i+1} ---\n{snippet}")
                if len(matches) >= 5:
                    break

        if not matches:
            return f"No matches found in solver papers skill for query: {query!r}"

        return f"Found {len(matches)} match(es) in solver papers skill for {query!r}:\n\n" + "\n\n".join(matches)

    def read_solver_skill_section(self, section_name: str) -> str:
        """Retrieve a specific section of the solver skill by its markdown heading.

        Args:
            section_name: Heading title (e.g. 'Role of the rewriter agent', 'Shared vocabulary', 'Solver problem statement').

        Returns:
            Extracted section content.
        """
        content = self._load_content()
        clean_name = section_name.strip()
        pattern = rf"(#+\s+.*{re.escape(clean_name)}.*?)(?=\n#+\s+|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return f"Section '{section_name}' not found in solver skill document."
