"""Noise Related Papers and Physical Hardware Noise Modeling FastMCP tool and API.

Provides access to the physical-neural-network-hardware-noise-modeling skill,
including fidelity tiers (Tier 0 to Tier 5), ONN-specific noise channels,
and curated literature references.
"""

from __future__ import annotations

import logging
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

# Relative path from agentic-ML-analog-sim/ repository root
SKILL_RELATIVE_PATH = Path("skills/physical-neural-network-hardware-noise-modeling/SKILL.md")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SKILL_PATH = REPO_ROOT / SKILL_RELATIVE_PATH


class NoisePapersTool(ChiaTool):
    """ChiaTool wrapper providing hardware noise modeling guidance and literature over FastMCP."""

    def __init__(
        self,
        name: str = "noise_papers",
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
        if skill_path is not None:
            p = Path(skill_path)
            if not p.is_absolute():
                for base in (REPO_ROOT, Path.cwd()):
                    cand = (base / p).resolve()
                    if cand.exists():
                        p = cand
                        break
            self.skill_path = p.resolve()
        else:
            self.skill_path = DEFAULT_SKILL_PATH.resolve()

        self._content: Optional[str] = None

        # Register MCP tools in setup() function per Chia documentation
        self.mcp.add_tool(self.get_noise_tier_guide, name="get_noise_tier_guide")
        self.mcp.add_tool(self.get_onn_noise_channels, name="get_onn_noise_channels")
        self.mcp.add_tool(self.get_noise_literature_references, name="get_noise_literature_references")
        self.mcp.add_tool(self.search_noise_skill, name="search_noise_skill")

    def _load_content(self) -> str:
        if self._content is None:
            if self.skill_path.exists():
                self._content = self.skill_path.read_text(encoding="utf-8")
            else:
                self._content = ""
        return self._content

    def get_noise_tier_guide(self, tier: Union[int, str]) -> str:
        """Query physical hardware noise modeling guidance for a specific tier (0 to 5).

        Args:
            tier: Tier index or identifier ('0', '1', '2', '3', '4', '5' or 'tier_0', etc.).

        Returns:
            Extracted markdown section containing concept, PyTorch recipe, calibration defaults, and cost.
        """
        content = self._load_content()
        tier_str = str(tier).strip().lower().replace("tier_", "").replace("tier", "").replace("l", "").strip()

        tier_headers = {
            "0": "## Tier 0 — Static deterministic perturbation",
            "1": "## Tier 1 — Independent stochastic parameter noise",
            "2": "## Tier 2 — Placement-aware block + interface model",
            "3": "## Tier 3 — Composite phenomenological device model",
            "4": "## Tier 4 — Physics-informed dynamic noise",
            "5": "## Tier 5 — Full-stack calibrated behavioral twin",
        }

        header = tier_headers.get(tier_str)
        if not header or header not in content:
            available = list(tier_headers.keys())
            return (
                f"Tier '{tier}' not recognized or section missing in SKILL.md. "
                f"Available tiers: {available}."
            )

        start_idx = content.find(header)
        next_sep = content.find("\n---", start_idx)
        if next_sep != -1:
            return content[start_idx:next_sep].strip()
        return content[start_idx:].strip()

    def get_onn_noise_channels(self) -> str:
        """Query ONN-specific noise channels (natural frequency mismatch dOmega, phase jitter/diffusion, coupling variation dK, and readout quantization).

        Returns:
            Extracted markdown section for oscillatory neural network noise channels.
        """
        content = self._load_content()
        header = "## ONN-specific noise channels"
        if header not in content:
            return "ONN-specific noise channels section not found in SKILL.md."

        start_idx = content.find(header)
        next_sep = content.find("\n---", start_idx)
        if next_sep != -1:
            return content[start_idx:next_sep].strip()
        return content[start_idx:].strip()

    def get_noise_literature_references(self) -> str:
        """Query curated literature references and citations for analog physical neural network noise modeling.

        Returns:
            Extracted key references list with paper titles, DOIs, and URLs.
        """
        content = self._load_content()
        header = "## Key references (curated)"
        if header not in content:
            return "Key references section not found in SKILL.md."

        start_idx = content.find(header)
        next_sep = content.find("\n---", start_idx)
        if next_sep != -1:
            return content[start_idx:next_sep].strip()
        return content[start_idx:].strip()

    def search_noise_skill(self, query: str) -> str:
        """Search the hardware noise modeling skill text for keywords or concepts.

        Args:
            query: Case-insensitive search query.

        Returns:
            Matching lines or context paragraphs.
        """
        content = self._load_content()
        if not content:
            return "SKILL.md is empty or not found."

        q = query.lower()
        paragraphs = content.split("\n\n")
        matched = [p.strip() for p in paragraphs if q in p.lower()]

        if not matched:
            return f"No matches found for '{query}' in physical hardware noise modeling skill."

        return "\n\n---\n\n".join(matched[:5])
