"""Tests for SolverPapersTool."""

from __future__ import annotations

import pytest

from agentic_ml_analog_sim.tools.solver_papers_tool import SolverPapersTool


def test_solver_papers_tool_summary():
    tool = SolverPapersTool(auto_start=False)

    summary = tool.get_solver_skill_summary()
    assert "Solvers for Continuous Systems" in summary
    assert "TIER 0" in summary
    assert "TIER 1" in summary
    assert "TIER 2" in summary
    assert "TIER 3" in summary
    assert "solver_fn(rhs, state, time_grid, drive, **kwargs)" in summary


def test_solver_papers_tool_tier_guides():
    tool = SolverPapersTool(auto_start=False)

    # Tier 0 (explicit fixed-step)
    t0 = tool.get_solver_tier_guide("tier_0")
    assert "TIER 0" in t0
    assert "Euler" in t0 or "RK4" in t0

    # Tier 1 (adaptive)
    t1 = tool.get_solver_tier_guide("tier_1")
    assert "TIER 1" in t1
    assert "Adaptive" in t1 or "solve_ivp" in t1 or "RK45" in t1

    # Tier 2 (implicit)
    t2 = tool.get_solver_tier_guide("tier_2")
    assert "TIER 2" in t2

    # Tier 3 (parallel-in-time)
    t3 = tool.get_solver_tier_guide("tier_3")
    assert "TIER 3" in t3

    # Unknown tier
    t_unk = tool.get_solver_tier_guide("tier_99")
    assert "Unknown tier" in t_unk


def test_solver_papers_tool_search():
    tool = SolverPapersTool(auto_start=False)

    # Search for RK4
    res = tool.search_solver_papers("RK4")
    assert "Found" in res
    assert "RK4" in res

    # Search for non-existent keyword
    res_none = tool.search_solver_papers("completely_absent_query_term_12345")
    assert "No matches found" in res_none


def test_solver_papers_tool_read_section():
    tool = SolverPapersTool(auto_start=False)

    sec = tool.read_solver_skill_section("Role of the rewriter agent")
    assert "rewriter" in sec.lower() or "Role" in sec

    sec_none = tool.read_solver_skill_section("NonexistentSectionTitle12345")
    assert "not found" in sec_none
