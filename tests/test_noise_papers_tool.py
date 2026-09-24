"""Tests for NoisePapersTool."""

from __future__ import annotations

import pytest
from agentic_ml_analog_sim.tools.noise_papers_tool import NoisePapersTool


def test_noise_papers_tool_methods():
    tool = NoisePapersTool(auto_start=False)

    # 1. Test tier lookup
    tier1 = tool.get_noise_tier_guide(1)
    assert "Tier 1" in tier1
    assert "noisy_forward" in tier1 or "parameter noise" in tier1.lower()

    tier2 = tool.get_noise_tier_guide(2)
    assert "Tier 2" in tier2
    assert "quantize" in tier2 or "DAC" in tier2 or "ADC" in tier2

    # Test unrecognized tier
    unrec = tool.get_noise_tier_guide(99)
    assert "not recognized" in unrec

    # 2. Test ONN noise channels
    onn = tool.get_onn_noise_channels()
    assert "ONN-specific noise channels" in onn
    assert "dOmega" in onn or "mismatch" in onn.lower()

    # 3. Test literature references
    refs = tool.get_noise_literature_references()
    assert "Key references" in refs
    assert "Momeni" in refs or "Joshi" in refs

    # 4. Test search
    res = tool.search_noise_skill("quantization")
    assert "quantize" in res.lower() or "quantization" in res.lower()
