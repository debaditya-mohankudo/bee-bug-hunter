"""Tests for the FrameworkError handling in orchestrator.run_flow_once /
run_batch_once: a FrameworkError raised by supervisor.run() must produce a
structured log line (error_type/is_fatal/is_retryable/error_chain), not just
a generic traceback, and must still propagate/be skipped the same way a bare
Exception always has.
"""
import logging

import pytest
from beeai_framework.agents.errors import AgentError

from bee_bug_hunter import orchestrator
from bee_bug_hunter.logging_config import new_run_context


class _RaisingSupervisor:
    def __init__(self, error: Exception):
        self._error = error

    def run(self, _prompt):
        raise self._error


def _patch_build_supervisor(monkeypatch, error: Exception):
    def _fake_build_supervisor(*_args, **_kwargs):
        return _RaisingSupervisor(error), "prompt"

    monkeypatch.setattr(orchestrator, "build_supervisor", _fake_build_supervisor)


def test_framework_error_fields_extracts_explain_chain():
    inner = AgentError("root cause", cause=ValueError("bad input"))
    outer = AgentError("agent failed", cause=inner)

    fields = orchestrator._framework_error_fields(outer)

    assert fields["error_type"] == "AgentError"
    assert fields["is_fatal"] is True
    assert fields["is_retryable"] is False
    assert "agent failed" in fields["error_chain"]
    assert "root cause" in fields["error_chain"]


def test_run_flow_once_logs_structured_fields_for_framework_error(monkeypatch, caplog):
    _patch_build_supervisor(monkeypatch, AgentError("boom", cause=ValueError("root")))

    flow_cfg = {"name": "demo_login", "containers": ["demo_app-web-1"]}

    with caplog.at_level(logging.ERROR, logger="bee_bug_hunter.orchestrator"):
        with pytest.raises(AgentError):
            orchestrator.run_flow_once(flow_cfg, duration_seconds=5)

    failed_records = [r for r in caplog.records if r.getMessage() == "supervisor_run_failed"]
    assert len(failed_records) == 1
    fields = failed_records[0].extra_fields
    assert fields["error_type"] == "AgentError"
    assert fields["is_fatal"] is True
    assert fields["is_retryable"] is False
    assert "boom" in fields["error_chain"]


def test_run_flow_once_logs_plain_for_non_framework_error(monkeypatch, caplog):
    _patch_build_supervisor(monkeypatch, ValueError("not a framework error"))

    flow_cfg = {"name": "demo_login", "containers": ["demo_app-web-1"]}

    with caplog.at_level(logging.ERROR, logger="bee_bug_hunter.orchestrator"):
        with pytest.raises(ValueError):
            orchestrator.run_flow_once(flow_cfg, duration_seconds=5)

    failed_records = [r for r in caplog.records if r.getMessage() == "supervisor_run_failed"]
    assert len(failed_records) == 1
    assert "extra_fields" not in failed_records[0].__dict__ or failed_records[0].extra_fields == {
        "elapsed_s": failed_records[0].extra_fields.get("elapsed_s")
    }


def test_run_batch_once_skips_flow_and_logs_structured_fields(monkeypatch, caplog):
    _patch_build_supervisor(monkeypatch, AgentError("boom", cause=ValueError("root")))
    manifest = {
        "duration_seconds": 5,
        "flows": [{"name": "demo_login", "containers": ["demo_app-web-1"]}],
    }

    with caplog.at_level(logging.ERROR, logger="bee_bug_hunter.orchestrator"):
        results = orchestrator.run_batch_once(manifest)

    assert results == []
    skipped = [r for r in caplog.records if r.getMessage() == "flow_run_skipped_after_failure"]
    assert len(skipped) == 1
    assert skipped[0].extra_fields["error_type"] == "AgentError"


@pytest.fixture(autouse=True)
def _run_id_context():
    new_run_context("test-flow")
    yield
