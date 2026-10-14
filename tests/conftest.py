import sys
import os
from pathlib import Path

# Ensure local worktree src/ takes precedence in sys.path
_src_path = str(Path(__file__).resolve().parents[1] / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# If REGENERATIVE_CORNN_ROOT is not explicitly exported, discover sibling in workspace
if "REGENERATIVE_CORNN_ROOT" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[2]
    candidate = workspace_root / "RegenariveCoRNN"
    if candidate.is_dir():
        os.environ["REGENERATIVE_CORNN_ROOT"] = str(candidate)
