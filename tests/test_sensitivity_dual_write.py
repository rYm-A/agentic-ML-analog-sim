"""Unit tests for ResultDBTool sensitivity curve dual format support and DB persistence."""

import json
import pytest
from agentic_ml_analog_sim.db.database import ResultDatabase
from agentic_ml_analog_sim.tools.result_db_tool import ResultDBTool


@pytest.fixture
def test_db_and_tool(tmp_path):
    db_file = tmp_path / "test_sensitivity.db"
    db = ResultDatabase(db_path=db_file)
    db.init_db()

    # Create reference proposal so foreign key constraint is satisfied
    db.insert_proposal(
        candidate_id="ref_design_001",
        solver="rk4",
        noise_model="none",
        is_reference=True,
        status="COMPLETED",
    )

    tool = ResultDBTool(db=db)
    return db, tool


def test_record_sensitivity_curve_with_curve_list(test_db_and_tool):
    db, tool = test_db_and_tool

    curve_entries = [(0.0, 0.0), (0.01, 0.005), (0.05, 0.025), (0.1, 0.08)]
    res_raw = tool.record_sensitivity_curve(
        reference_id="ref_design_001",
        curve=curve_entries,
        experiment_name="test_exp",
    )
    res = json.loads(res_raw)
    assert res["status"] == "SUCCESS"
    assert res["reference_id"] == "ref_design_001"
    assert res["points_recorded"] == 4

    # Verify directly in SQLite DB
    db_curve = db.get_sensitivity_curve(reference_id="ref_design_001", experiment_name="test_exp")
    assert len(db_curve) == 4
    assert db_curve[0] == (0.0, 0.0)
    assert db_curve[-1] == (0.1, 0.08)


def test_record_sensitivity_curve_with_curve_json(test_db_and_tool):
    db, tool = test_db_and_tool

    curve_json = json.dumps([[0.0, 0.0], [0.02, 0.01], [0.04, 0.03]])
    res_raw = tool.record_sensitivity_curve(
        reference_id="ref_design_001",
        curve_json=curve_json,
        experiment_name="test_exp_json",
    )
    res = json.loads(res_raw)
    assert res["status"] == "SUCCESS"
    assert res["points_recorded"] == 3

    # Test retrieval via MCP get_sensitivity_curve tool
    get_raw = tool.get_sensitivity_curve(reference_id="ref_design_001", experiment_name="test_exp_json")
    get_res = json.loads(get_raw)
    assert get_res["status"] == "SUCCESS"
    assert len(get_res["curve"]) == 3


def test_interpolate_sensitivity_via_tool(test_db_and_tool):
    db, tool = test_db_and_tool

    curve_entries = [(0.0, 0.0), (0.1, 0.05), (0.2, 0.10)]
    tool.record_sensitivity_curve(
        reference_id="ref_design_001",
        curve=curve_entries,
        experiment_name="test_interp",
    )

    # Interpolate mid-point (e_pert = 0.025 -> epsilon_d = 0.05)
    interp_raw = tool.interpolate_sensitivity(
        e_pert=0.025,
        reference_id="ref_design_001",
        experiment_name="test_interp",
    )
    interp_res = json.loads(interp_raw)
    assert interp_res["status"] == "SUCCESS"
    assert pytest.approx(interp_res["epsilon_d"], abs=1e-4) == 0.05
