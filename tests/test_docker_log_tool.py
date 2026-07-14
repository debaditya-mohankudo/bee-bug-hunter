"""Tests for docker_log_tool.py's --since window derivation: EXPLAIN-style
schema queries aren't involved here, but the same "derive from what we
already know instead of a flat guess" idea applies -- the lookback window
scales with each flow's own duration_seconds instead of a flat 5m, so a flow
expected to run long (e.g. one built around a slow query) gets a
proportionally longer window automatically. subprocess.Popen/time.sleep are
mocked so no real docker CLI or wait is needed.
"""
from unittest.mock import MagicMock, patch

from bee_bug_hunter.tools import docker_log_tool
from bee_bug_hunter.tools.docker_log_tool import (
    SINCE_MIN_SECONDS,
    SINCE_MULTIPLIER,
    _capture_sync,
    _since_window_seconds,
)


def test_since_window_scales_with_duration():
    assert _since_window_seconds(30) == 30 * SINCE_MULTIPLIER


def test_since_window_floors_short_durations():
    # 5 * SINCE_MULTIPLIER (15) would be below SINCE_MIN_SECONDS (60).
    assert _since_window_seconds(5) == SINCE_MIN_SECONDS


def test_since_window_scales_up_for_long_flows():
    assert _since_window_seconds(400) == 400 * SINCE_MULTIPLIER


def test_capture_sync_passes_computed_since_to_docker_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_log_tool, "CAPTURE_DIR", tmp_path)
    monkeypatch.setattr(docker_log_tool.time, "sleep", lambda _seconds: None)

    fake_proc = MagicMock()
    fake_proc.wait.return_value = None

    with patch("bee_bug_hunter.tools.docker_log_tool.subprocess.Popen", return_value=fake_proc) as mock_popen:
        _capture_sync("demo-api", 30, "test_run", None)

    cmd = mock_popen.call_args.args[0]
    assert cmd == ["docker", "logs", "-f", "--since", f"{30 * SINCE_MULTIPLIER}s", "demo-api"]


def test_capture_sync_uses_floor_for_short_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_log_tool, "CAPTURE_DIR", tmp_path)
    monkeypatch.setattr(docker_log_tool.time, "sleep", lambda _seconds: None)

    fake_proc = MagicMock()
    fake_proc.wait.return_value = None

    with patch("bee_bug_hunter.tools.docker_log_tool.subprocess.Popen", return_value=fake_proc) as mock_popen:
        _capture_sync("demo-api", 5, "test_run", None)

    cmd = mock_popen.call_args.args[0]
    assert cmd == ["docker", "logs", "-f", "--since", f"{SINCE_MIN_SECONDS}s", "demo-api"]


def test_docker_capture_started_logs_the_computed_since_seconds(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setattr(docker_log_tool, "CAPTURE_DIR", tmp_path)
    monkeypatch.setattr(docker_log_tool.time, "sleep", lambda _seconds: None)

    fake_proc = MagicMock()
    fake_proc.wait.return_value = None

    with patch("bee_bug_hunter.tools.docker_log_tool.subprocess.Popen", return_value=fake_proc):
        with caplog.at_level(logging.INFO, logger="bee_bug_hunter.tools.docker_log_tool"):
            _capture_sync("demo-api", 30, "test_run", None)

    started = [r for r in caplog.records if r.getMessage() == "docker_capture_started"]
    assert len(started) == 1
    assert started[0].extra_fields["since_seconds"] == 30 * SINCE_MULTIPLIER
