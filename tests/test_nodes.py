"""Comprehensive unit and integration tests for Phase 0 Chia nodes and MockAntigravityLLM."""

import json
import sqlite3
import pytest
from dataclasses import dataclass, field
from typing import Any, Optional

from agentic_ml_analog_sim.mock import MockAntigravityLLM
from agentic_ml_analog_sim.nodes import (
    check_reference_data,
    propose_candidate,
    validate_proposal,
)


@dataclass
class DummyBudget:
    max_node_retries: int = 3
    max_iterations: int = 15


@dataclass
class DummySimulationConfig:
    solvers: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "euler", "allowed_steps": [1, 2, 5, 10]},
            {"name": "rk4", "allowed_steps": [5, 10, 15, 25]},
            {"name": "euler_backward", "allowed_steps": [2, 5, 10]},
        ]
    )
    noise_models: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "none"},
            {"name": "L0_static_mismatch", "sigma_range": [0.001, 0.05]},
            {"name": "L1_stochastic_parameter_noise", "sigma_range": [0.001, 0.02]},
        ]
    )
    sparsity_configs: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "dense", "sparsity_ratio": 0.0},
            {"name": "threshold_pruned", "allowed_sparsity_ratios": [0.2, 0.4, 0.6, 0.8]},
        ]
    )
    budget: DummyBudget = field(default_factory=DummyBudget)
    reference_design: dict[str, Any] = field(
        default_factory=lambda: {
            "model_name": "cifar10/n1024",
            "solver": "rk4",
            "num_steps": 25,
            "noise_model": "none",
        }
    )


@pytest.fixture
def test_db():
    """In-memory SQLite database matching schema.sql."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE proposals (
            candidate_id TEXT PRIMARY KEY,
            iteration INTEGER NOT NULL,
            solver TEXT NOT NULL,
            noise_model TEXT NOT NULL,
            sparsity_config TEXT NOT NULL DEFAULT '{}',
            solver_params TEXT NOT NULL,
            noise_params TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('PENDING', 'EVALUATING', 'COMPLETED', 'FAILED', 'REJECTED')),
            is_reference BOOLEAN NOT NULL DEFAULT 0,
            is_active_evaluation BOOLEAN NOT NULL DEFAULT 0,
            accuracy_fid REAL,
            relative_error REAL,
            latency_ms REAL,
            wall_clock_s REAL,
            worker_utilization REAL,
            pareto_optimal BOOLEAN DEFAULT 0,
            selection_reason TEXT,
            phase_1_prompt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            comment TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(candidate_id) REFERENCES proposals(candidate_id)
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT,
            phase TEXT NOT NULL,
            node_name TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0.0,
            duration_seconds REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(candidate_id) REFERENCES proposals(candidate_id)
        );
        """
    )
    conn.commit()

    class MockResultDB:
        def __init__(self, connection):
            self.conn = connection

        def record_token_usage(
            self,
            phase: str,
            node_name: str,
            model: str,
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            total_tokens: int = 0,
            cost_usd: float = 0.0,
            candidate_id: Optional[str] = None,
            duration_seconds: float = 0.0,
        ) -> int:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO token_usage (
                    candidate_id, phase, node_name, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost_usd, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    phase,
                    node_name,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cost_usd,
                    duration_seconds,
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    return MockResultDB(conn)


@pytest.fixture
def config():
    return DummySimulationConfig()


# ==============================================================================
# Tests for MockAntigravityLLM
# ==============================================================================

def test_mock_llm_valid_mode():
    llm = MockAntigravityLLM(mode="valid")
    res = llm.prompt("Please propose next candidate")
    assert res.success is True
    data = json.loads(res.result)
    assert data["solver"] == "euler"
    assert data["noise_model"] == "L0_static_mismatch"
    assert data["sparsity_config"]["sparsity_ratio"] == 0.4
    assert "Explore Euler" in data["selection_reason"]
    assert llm.call_count == 1
    assert len(llm.history) == 1


def test_mock_llm_modes_and_responses():
    llm = MockAntigravityLLM(mode="invalid_solver")
    res = llm.prompt("test")
    data = json.loads(res.result)
    assert data["solver"] == "unsupported_leapfrog"

    llm.set_mode("invalid_noise")
    res = llm.prompt("test")
    data = json.loads(res.result)
    assert data["noise_model"] == "quantum_vacuum_fluctuation"

    llm.set_mode("invalid_sparsity")
    res = llm.prompt("test")
    data = json.loads(res.result)
    assert data["sparsity_config"]["sparsity_ratio"] == 0.99

    llm.set_mode("duplicate")
    res = llm.prompt("test")
    data = json.loads(res.result)
    assert data["solver"] == "rk4"
    assert data["noise_model"] == "none"


def test_mock_llm_response_queue():
    seq = [
        {"solver": "rk4", "noise_model": "none", "selection_reason": "dup"},
        {"solver": "euler", "noise_model": "L0_static_mismatch", "selection_reason": "valid"},
    ]
    llm = MockAntigravityLLM(responses=seq)
    res1 = llm.prompt("Call 1")
    assert json.loads(res1.result)["solver"] == "rk4"
    res2 = llm.prompt("Call 2")
    assert json.loads(res2.result)["solver"] == "euler"


# ==============================================================================
# Tests for Node 1: check_reference_data
# ==============================================================================

def test_node1_phase0_check_reference_empty_db(test_db, config):
    # Empty DB -> should return False
    assert check_reference_data(test_db, config) is False


def test_node1_phase0_check_reference_incomplete(test_db, config):
    cursor = test_db.conn.cursor()
    # Insert reference with status PENDING
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, solver_params, noise_params,
            status, is_reference
        ) VALUES ('ref_01', 0, 'rk4', 'none', '{}', '{}', 'PENDING', 1)
        """
    )
    test_db.conn.commit()
    assert check_reference_data(test_db, config) is False

    # Update to COMPLETED but missing latency_ms
    cursor.execute(
        "UPDATE proposals SET status = 'COMPLETED', accuracy_fid = 32.5 WHERE candidate_id = 'ref_01'"
    )
    test_db.conn.commit()
    assert check_reference_data(test_db, config) is False


def test_node1_phase0_check_reference_completed(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, solver_params, noise_params,
            status, is_reference, accuracy_fid, latency_ms
        ) VALUES ('ref_01', 0, 'rk4', 'none', '{}', '{}', 'COMPLETED', 1, 32.5, 12.4)
        """
    )
    test_db.conn.commit()
    # Now reference is COMPLETED, is_reference=True, with accuracy_fid and latency_ms
    assert check_reference_data(test_db, config) is True


def test_node1_phase0_check_reference_completed_without_fid(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, solver_params, noise_params,
            status, is_reference, relative_error, latency_ms
        ) VALUES ('ref_02', 0, 'rk4', 'none', '{}', '{}', 'COMPLETED', 1, 0.0, 12.4)
        """
    )
    test_db.conn.commit()
    # Reference is COMPLETED, is_reference=True, with relative_error and latency_ms, accuracy_fid is None
    assert check_reference_data(test_db, config) is True


# ==============================================================================
# Tests for Node 2: propose_candidate
# ==============================================================================

def test_node2_phase0_propose_candidate(test_db, config):
    llm = MockAntigravityLLM(mode="valid")
    result = propose_candidate(db=test_db, config=config, llm=llm, iteration=1)

    assert "candidate_id" in result
    assert result["solver"] == "euler"
    assert result["noise_model"] == "L0_static_mismatch"
    assert "Phase 1 Build & Simulation Instructions" in result["phase_1_prompt"]

    # Verify that candidate was inserted into DB as PENDING
    cursor = test_db.conn.cursor()
    cursor.execute("SELECT status, solver, noise_model FROM proposals WHERE candidate_id = ?", (result["candidate_id"],))
    row = cursor.fetchone()
    assert row is not None
    assert row["status"] == "PENDING"
    assert row["solver"] == "euler"
    assert row["noise_model"] == "L0_static_mismatch"

    # Verify that token usage was recorded in token_usage table
    cursor.execute("SELECT * FROM token_usage WHERE candidate_id = ?", (result["candidate_id"],))
    t_row = cursor.fetchone()
    assert t_row is not None
    assert t_row["phase"] == "phase_0"
    assert t_row["node_name"] == "node2_phase0_propose_candidate"
    assert t_row["total_tokens"] > 0
    assert t_row["prompt_tokens"] > 0
    assert t_row["completion_tokens"] > 0


def test_node2_phase0_propose_candidate_with_rejection_feedback(test_db, config):
    llm = MockAntigravityLLM(mode="valid")
    result = propose_candidate(
        db=test_db,
        config=config,
        llm=llm,
        prior_rejection_reason="DUPLICATE_PROPOSAL: Matches candidate 'cand_01'",
        iteration=2,
    )
    assert result["candidate_id"] is not None
    # Verify that LLM received the rejection note in the user prompt
    last_prompt = llm.last_prompt
    assert "DUPLICATE_PROPOSAL" in last_prompt


# ==============================================================================
# Tests for Node 3: validate_proposal
# ==============================================================================

def test_node3_validate_valid_proposal(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_valid', 1, 'euler', 'L0_static_mismatch', '{"sparsity_ratio": 0.4}',
            '{"num_steps": 5}', '{"sigma": 0.01}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    is_valid, reason, cycle = validate_proposal("cand_valid", test_db, config, cycle_count=0)
    assert is_valid is True
    assert reason == "VALID"
    assert cycle == 0


def test_node3_validate_invalid_solver(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_bad_solver', 1, 'unsupported_solver', 'none', '{"sparsity_ratio": 0.0}',
            '{"num_steps": 5}', '{}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    is_valid, reason, next_cycle = validate_proposal("cand_bad_solver", test_db, config, cycle_count=0)
    assert is_valid is False
    assert "INVALID_SOLVER" in reason
    assert next_cycle == 1

    # Verify status changed to REJECTED in DB
    cursor.execute("SELECT status FROM proposals WHERE candidate_id = 'cand_bad_solver'")
    assert cursor.fetchone()["status"] == "REJECTED"


def test_node3_validate_invalid_noise_model(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_bad_noise', 1, 'euler', 'unknown_noise_level', '{"sparsity_ratio": 0.0}',
            '{"num_steps": 5}', '{}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    is_valid, reason, next_cycle = validate_proposal("cand_bad_noise", test_db, config, cycle_count=0)
    assert is_valid is False
    assert "INVALID_NOISE_MODEL" in reason
    assert next_cycle == 1


def test_node3_validate_invalid_sparsity(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_bad_sparsity', 1, 'euler', 'L0_static_mismatch', '{"sparsity_ratio": 0.99}',
            '{"num_steps": 5}', '{"sigma": 0.01}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    is_valid, reason, next_cycle = validate_proposal("cand_bad_sparsity", test_db, config, cycle_count=0)
    assert is_valid is False
    assert "INVALID_SPARSITY" in reason
    assert next_cycle == 1


def test_node3_validate_duplicate_proposal(test_db, config):
    cursor = test_db.conn.cursor()
    # Insert existing evaluated proposal
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_orig', 1, 'euler', 'L0_static_mismatch', '{"sparsity_ratio": 0.4}',
            '{"num_steps": 5}', '{"sigma": 0.01}', 'COMPLETED'
        )
        """
    )
    # Insert duplicate pending proposal
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_duplicate', 2, 'euler', 'L0_static_mismatch', '{"sparsity_ratio": 0.4}',
            '{"num_steps": 5}', '{"sigma": 0.01}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    is_valid, reason, next_cycle = validate_proposal("cand_duplicate", test_db, config, cycle_count=1)
    assert is_valid is False
    assert "DUPLICATE_PROPOSAL" in reason
    assert next_cycle == 2

    # Verify candidate marked REJECTED
    cursor.execute("SELECT status FROM proposals WHERE candidate_id = 'cand_duplicate'")
    assert cursor.fetchone()["status"] == "REJECTED"


def test_node3_max_cycles_exceeded(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_retry3', 3, 'invalid_solver_again', 'none', '{}',
            '{}', '{}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    # When cycle_count >= max_node_retries (3), should return MAX_CYCLES_EXCEEDED
    is_valid, reason, cycle = validate_proposal("cand_retry3", test_db, config, cycle_count=3)
    assert is_valid is False
    assert reason == "MAX_CYCLES_EXCEEDED"
    assert cycle == 3

    # Verify marked REJECTED and termination comment logged
    cursor.execute("SELECT status FROM proposals WHERE candidate_id = 'cand_retry3'")
    assert cursor.fetchone()["status"] == "REJECTED"

    cursor.execute("SELECT comment FROM comments WHERE candidate_id = 'cand_retry3'")
    comment_row = cursor.fetchone()
    assert comment_row is not None
    assert "max_node_retries" in comment_row["comment"]


def test_node3_validate_solver_steps(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_bad_steps', 1, 'euler', 'none', '{"sparsity_ratio": 0.0}',
            '{"num_steps": 99}', '{}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    is_valid, reason, next_cycle = validate_proposal("cand_bad_steps", test_db, config, cycle_count=0)
    assert is_valid is False
    assert "INVALID_SOLVER_STEPS" in reason
    assert next_cycle == 1


def test_node3_validate_noise_sigma(test_db, config):
    cursor = test_db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO proposals (
            candidate_id, iteration, solver, noise_model, sparsity_config,
            solver_params, noise_params, status
        ) VALUES (
            'cand_bad_sigma', 1, 'euler', 'L0_static_mismatch', '{"sparsity_ratio": 0.4}',
            '{"num_steps": 5}', '{"sigma": 0.99}', 'PENDING'
        )
        """
    )
    test_db.conn.commit()

    is_valid, reason, next_cycle = validate_proposal("cand_bad_sigma", test_db, config, cycle_count=0)
    assert is_valid is False
    assert "INVALID_NOISE_SIGMA" in reason
    assert next_cycle == 1


def test_node1_phase0_with_dict_and_list_db(config):
    # Dict mock with completed reference (accuracy_fid can be None)
    dict_db = {
        "ref": {
            "is_reference": True,
            "status": "COMPLETED",
            "accuracy_fid": None,
            "relative_error": 0.0,
            "latency_ms": 10.2,
        }
    }
    assert check_reference_data(dict_db, config) is True

    # List mock with incomplete reference (missing latency_ms)
    list_db = [
        {"is_reference": True, "status": "COMPLETED", "accuracy_fid": None, "latency_ms": None}
    ]
    assert check_reference_data(list_db, config) is False


# Backwards compatibility alias
test_node1_with_dict_and_list_db = test_node1_phase0_with_dict_and_list_db


def test_mock_llm_tolerance_mode():
    llm = MockAntigravityLLM(mode="invalid_tolerance")
    res = llm.prompt("test tolerance")
    data = json.loads(res.result)
    assert "Tolerance check failure" in data["selection_reason"]
    assert data["solver_params"]["rtol"] == 1e-12


def test_integration_with_actual_db_and_config(tmp_path):
    from agentic_ml_analog_sim.config import load_simulation_config
    from agentic_ml_analog_sim.db.database import ResultDatabase

    cfg = load_simulation_config("config/simulation_config.yaml")
    db_path = tmp_path / "integration.db"
    db = ResultDatabase(db_path=str(db_path))

    # 1. Node 1 check on empty DB
    assert check_reference_data(db, cfg) is False

    # 2. Insert completed reference design
    db.insert_proposal(
        solver="rk4",
        noise_model="none",
        solver_params={"steps": 25, "dt": 0.04},
        noise_params={},
        sparsity_config={"sparsity_ratio": 0.0},
        selection_reason="Baseline reference",
        is_reference=True,
        status="COMPLETED",
        accuracy_fid=31.2,
        latency_ms=15.4,
    )
    assert check_reference_data(db, cfg) is True

    # 3. Node 2 propose candidate with MockAntigravityLLM (pass tools=[] to avoid spinning up external tool servers)
    llm = MockAntigravityLLM(mode="valid")
    cand_info = propose_candidate(db=db, config=cfg, llm=llm, tools=[], iteration=1)
    cand_id = cand_info["candidate_id"]
    assert cand_id is not None
    assert cand_info["solver"] == "euler"

    # 4. Node 3 validate proposal
    is_valid, reason, cycle = validate_proposal(cand_id, db, cfg, cycle_count=0)
    assert is_valid is True
    assert reason == "VALID"
    assert cycle == 0

    # 5. Verify token usage recorded
    summary = db.get_token_usage_summary(phase="phase_0")
    assert summary["overall"]["total_calls"] >= 1
    assert summary["overall"]["total_tokens"] > 0
    assert "phase_0.node2_phase0_propose_candidate" in summary["by_node"]
