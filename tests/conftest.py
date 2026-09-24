"""Pytest configuration and environment fixtures for agentic_ml_analog_sim."""

from __future__ import annotations

import os
from pathlib import Path

# If REGENERATIVE_CORNN_ROOT is not explicitly exported, discover sibling in workspace
if "REGENERATIVE_CORNN_ROOT" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[2]
    candidate = workspace_root / "RegenariveCoRNN"
    if candidate.is_dir():
        os.environ["REGENERATIVE_CORNN_ROOT"] = str(candidate)
