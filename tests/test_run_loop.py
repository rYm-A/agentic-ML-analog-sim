"""Unit tests for run_loop.py verification, profiling lifecycle, and CLI integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run_loop import (
    ensure_ray_initialized,
    execute_loop,
    parse_args,
    setup_experiment_storage,
)


def test_parse_args_defaults() -> None:
    """Verify default CLI arguments for cluster and profiling parameters."""
    parser = parse_args()
    args = parser.parse_args([])
    assert args.config == Path("config/simulation_config.yaml")
    assert args.reports_dir == Path("reports")
    assert args.db_path is None
    assert args.max_iterations is None
    assert args.max_experiments is None
    assert args.no_pdf is False
    assert args.ray_address == "auto"
    assert args.no_ray_init is False
    assert args.no_profile is False
    assert args.profile_dir is None
    assert args.verbose is False


def test_parse_args_custom() -> None:
    """Verify custom CLI argument parsing."""
    parser = parse_args()
    args = parser.parse_args([
        "--config", "custom_config.yaml",
        "--reports-dir", "/tmp/reports",
        "--db-path", "/tmp/test.db",
        "--max-iterations", "5",
        "--max-experiments", "3",
        "--no-pdf",
        "--ray-address", "100.64.0.2:6379",
        "--no-ray-init",
        "--no-profile",
        "--profile-dir", "/tmp/profiles",
        "--verbose",
    ])
    assert args.config == Path("custom_config.yaml")
    assert args.reports_dir == Path("/tmp/reports")
    assert args.db_path == Path("/tmp/test.db")
    assert args.max_iterations == 5
    assert args.max_experiments == 3
    assert args.no_pdf is True
    assert args.ray_address == "100.64.0.2:6379"
    assert args.no_ray_init is True
    assert args.no_profile is True
    assert args.profile_dir == Path("/tmp/profiles")
    assert args.verbose is True


def test_ensure_ray_initialized_already_active() -> None:
    """Verify ensure_ray_initialized detects already running Ray."""
    mock_ctx = MagicMock()
    mock_ctx.get_node_id.return_value = "node_123"

    with patch("ray.is_initialized", return_value=True), \
         patch("ray.get_runtime_context", return_value=mock_ctx), \
         patch("ray.cluster_resources", return_value={"control_worker": 1, "sim_worker": 1, "CPU": 4}):
        result = ensure_ray_initialized()
        assert result is True


def test_ensure_ray_initialized_connects_cluster() -> None:
    """Verify ensure_ray_initialized connects when Ray is uninitialized."""
    mock_ctx = MagicMock()
    mock_ctx.get_node_id.return_value = "node_cluster_1"

    is_init = False

    def mock_is_initialized() -> bool:
        return is_init

    def mock_ray_init(*args, **kwargs):
        nonlocal is_init
        is_init = True
        return None

    with patch("ray.is_initialized", side_effect=mock_is_initialized), \
         patch("ray.init", side_effect=mock_ray_init) as mock_init, \
         patch("ray.get_runtime_context", return_value=mock_ctx), \
         patch("ray.cluster_resources", return_value={"control_worker": 1}):
        result = ensure_ray_initialized(address="auto")
        assert result is True
        mock_init.assert_called_once_with(address="auto", ignore_reinit_error=True)


def test_ensure_ray_initialized_fallback_local() -> None:
    """Verify ensure_ray_initialized falls back to local Ray if cluster connection fails."""
    mock_ctx = MagicMock()
    mock_ctx.get_node_id.return_value = "local_node"

    is_init = False

    def mock_is_initialized() -> bool:
        return is_init

    def mock_ray_init(*args, **kwargs):
        nonlocal is_init
        if "address" in kwargs and kwargs["address"] == "auto":
            raise ConnectionError("Cluster unavailable")
        is_init = True
        return None

    with patch("ray.is_initialized", side_effect=mock_is_initialized), \
         patch("ray.init", side_effect=mock_ray_init) as mock_init, \
         patch("ray.get_runtime_context", return_value=mock_ctx), \
         patch("ray.cluster_resources", return_value={"CPU": 2}):
        result = ensure_ray_initialized(address="auto", auto_init_local=True)
        assert result is True
        assert mock_init.call_count == 2


def test_ensure_ray_initialized_fails_cleanly() -> None:
    """Verify ensure_ray_initialized returns False if all init attempts fail."""
    with patch("ray.is_initialized", return_value=False), \
         patch("ray.init", side_effect=RuntimeError("Fatal Ray error")):
        result = ensure_ray_initialized(address="auto", auto_init_local=True)
        assert result is False


def test_setup_experiment_storage(tmp_path: Path) -> None:
    """Verify experiment directory and SQLite DB resolution."""
    config_mock = MagicMock()
    config_mock.experiment_name = "test_exp"

    reports_base = tmp_path / "reports"
    exp_dir, db_path = setup_experiment_storage(config_mock, reports_base=reports_base)

    assert exp_dir == reports_base / "test_exp"
    assert exp_dir.exists()
    assert db_path == exp_dir / "results.db"


def test_execute_loop_profiling_lifecycle_and_summary(tmp_path: Path) -> None:
    """Verify execute_loop handles top-level profiling lifecycle and returns a summary dict."""
    config_file = tmp_path / "sim_config.yaml"
    config_file.write_text("experiment_name: test_run\n", encoding="utf-8")

    reports_dir = tmp_path / "reports"

    # Mock phases and database
    mock_config = MagicMock()
    mock_config.experiment_name = "test_run"
    mock_config.budget.max_iterations = 1
    mock_config.budget.max_experiments = 1

    mock_db = MagicMock()
    mock_db.get_completed_experiments_count.return_value = 1
    mock_db.get_token_usage_summary.return_value = {
        "overall": {"total_tokens": 500, "total_cost_usd": 0.005}
    }

    mock_p0 = MagicMock()
    mock_p0.action = "PROPOSED"
    mock_p0.candidate_id = "cand_001"
    mock_p0.details = "Candidate proposed"

    mock_p1 = MagicMock()
    mock_p1.action = "SIMULATION_SUCCESS"
    mock_p1.candidate_id = "cand_001"
    mock_p1.execution_mode = "analog_circuit"
    mock_p1.details = "Success"

    mock_p2 = MagicMock()
    mock_p2.action = "REPORT_GENERATED"
    mock_p2.completed_experiments = 1
    mock_p2.max_experiments = 1
    mock_p2.details = "Goal reached"
    mock_p2.tex_path = reports_dir / "test_run" / "report.tex"
    mock_p2.pdf_path = reports_dir / "test_run" / "report.pdf"
    mock_p2.pdf_compiled = True

    with patch("run_loop.load_simulation_config", return_value=mock_config), \
         patch("run_loop.ResultDatabase", return_value=mock_db), \
         patch("run_loop.run_phase_0", return_value=mock_p0), \
         patch("run_loop.run_phase_1", return_value=mock_p1), \
         patch("run_loop.run_phase_2", return_value=mock_p2), \
         patch("run_loop.ensure_ray_initialized", return_value=True), \
         patch("run_loop._PROFILER_AVAILABLE", True), \
         patch("ray.is_initialized", return_value=True), \
         patch("run_loop.start_collector") as mock_start_collector, \
         patch("run_loop.stop_collector") as mock_stop_collector:

        summary = execute_loop(
            config_path=config_file,
            reports_base=reports_dir,
            auto_init_ray=True,
            enable_profiling=True,
        )

        # Profiling collector started & stopped
        mock_start_collector.assert_called_once()
        mock_stop_collector.assert_called_once()

        # Summary assertions
        assert summary["status"] == "SUCCESS_REPORT_GENERATED"
        assert summary["iterations"] == 1
        assert summary["completed_experiments"] == 1
        assert summary["report_tex_path"] == str(reports_dir / "test_run" / "report.tex")
        assert summary["report_pdf_path"] == str(reports_dir / "test_run" / "report.pdf")
        assert summary["token_usage"]["overall"]["total_tokens"] == 500


def test_execute_loop_no_profiling_flag(tmp_path: Path) -> None:
    """Verify execute_loop respects enable_profiling=False."""
    config_file = tmp_path / "sim_config.yaml"
    config_file.write_text("experiment_name: test_run\n", encoding="utf-8")
    reports_dir = tmp_path / "reports"

    mock_config = MagicMock()
    mock_config.experiment_name = "test_run"
    mock_config.budget.max_iterations = 1
    mock_config.budget.max_experiments = 1

    mock_db = MagicMock()
    mock_db.get_completed_experiments_count.return_value = 0
    mock_db.get_token_usage_summary.return_value = {}

    mock_p0 = MagicMock()
    mock_p0.action = "NO_CANDIDATE"
    mock_p0.candidate_id = None
    mock_p0.details = "No candidates found"

    with patch("run_loop.load_simulation_config", return_value=mock_config), \
         patch("run_loop.ResultDatabase", return_value=mock_db), \
         patch("run_loop.run_phase_0", return_value=mock_p0), \
         patch("run_loop.ensure_ray_initialized", return_value=True), \
         patch("run_loop._PROFILER_AVAILABLE", True), \
         patch("ray.is_initialized", return_value=True), \
         patch("run_loop.start_collector") as mock_start_collector, \
         patch("run_loop.stop_collector") as mock_stop_collector:

        summary = execute_loop(
            config_path=config_file,
            reports_base=reports_dir,
            enable_profiling=False,
        )

        mock_start_collector.assert_not_called()
        mock_stop_collector.assert_not_called()
        assert summary["status"] == "NO_CANDIDATE"
        assert summary["profile_dir"] is None
