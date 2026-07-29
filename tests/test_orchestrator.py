"""Tests for orchestrator.run_flow_once / run_batch_once / monitor_loop.

FrameworkError handling: a FrameworkError raised by supervisor.run() must
produce a structured log line (error_type/is_fatal/is_retryable/error_chain),
not just a generic traceback, and must still propagate/be skipped the same
way a bare Exception always has.

Success path: extends the same fake-build_supervisor pattern with a fake
supervisor.run() that succeeds, seeding tool_capture/delegation_capture from
inside the fake's own run() method (where the run_id contextvar set by
new_run_context() is actually visible, same as real tool code) to exercise
anomaly detection, report saving, and known_issues recording end to end.
"""
import logging
from types import SimpleNamespace

import pytest
from beeai_framework.agents.errors import AgentError

from bee_bug_hunter import delegation_capture, orchestrator, tool_capture
from bee_bug_hunter.known_issues import record_issue
from bee_bug_hunter.logging_config import new_run_context, run_id_var


class _RaisingSupervisor:
    def __init__(self, error: Exception):
        self._error = error

    def run(self, _prompt):
        raise self._error


class _SuccessfulSupervisor:
    """seed, if given, is called with the active run_id from inside run() --
    the only point at which the run_id contextvar new_run_context() set is
    actually visible, matching how real tool code populates tool_capture/
    delegation_capture during a real supervisor.run()."""

    def __init__(self, response_text: str, seed=None):
        self.response_text = response_text
        self.seed = seed
        self.received_prompt = None

    async def run(self, prompt):
        self.received_prompt = prompt
        if self.seed:
            self.seed(run_id_var.get())
        return SimpleNamespace(last_message=SimpleNamespace(text=self.response_text))


def _patch_build_supervisor(monkeypatch, error: Exception):
    def _fake_build_supervisor(*_args, **_kwargs):
        return _RaisingSupervisor(error), "prompt"

    monkeypatch.setattr(orchestrator, "build_supervisor", _fake_build_supervisor)


def _patch_build_supervisor_success(monkeypatch, response_text: str, seed=None, capture_kwargs: dict | None = None):
    def _fake_build_supervisor(*args, **kwargs):
        if capture_kwargs is not None:
            capture_kwargs.update(kwargs)
        return _SuccessfulSupervisor(response_text, seed), "prompt"

    monkeypatch.setattr(orchestrator, "build_supervisor", _fake_build_supervisor)


def _patch_save_report(monkeypatch, path: str = "/fake/report.md"):
    calls = []

    def _fake_save_report(result):
        calls.append(result)
        return path

    monkeypatch.setattr(orchestrator, "save_report", _fake_save_report)
    return calls


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


def test_run_flow_once_success_clean_run_falls_back_to_manager_summary(monkeypatch):
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nnothing found")
    save_calls = _patch_save_report(monkeypatch)

    result = orchestrator.run_flow_once({"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5)

    assert result["flow"] == "demo_login"
    assert result["response"] == "SUMMARY: clean\n\nnothing found"
    assert result["anomaly"]["bug_signal"] is False
    assert result["bug_report"] is None
    assert result["perf_report"] is None
    assert result["report_path"] == "/fake/report.md"
    assert len(save_calls) == 1
    assert save_calls[0]["response"] == "SUMMARY: clean\n\nnothing found"


def test_run_flow_once_success_populates_anomaly_from_tool_capture(monkeypatch):
    def _seed(run_id):
        tool_capture._captures[run_id] = {
            "flow": ['{"network_log": [{"status": 500}], "step_results": []}'],
        }

    _patch_build_supervisor_success(monkeypatch, "SUMMARY: bug found\n\ndetails", seed=_seed)
    save_calls = _patch_save_report(monkeypatch)

    result = orchestrator.run_flow_once({"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5)

    assert result["anomaly"]["bug_signal"] is True
    assert result["anomaly"]["http_errors"] == [{"status": 500}]
    # tool_capture/delegation_capture must be cleared after use, not left for the next run.
    assert tool_capture.get_flow_raw(result["run_id"]) is None
    assert save_calls[0]["anomaly"]["bug_signal"] is True


def test_run_flow_once_bug_and_perf_reports_populated_from_delegation_capture(monkeypatch):
    def _seed(run_id):
        delegation_capture._captures[run_id] = [
            delegation_capture.Delegation(coworker="Bug Analyst", task="t", result="root cause: X"),
            delegation_capture.Delegation(coworker="SQL Performance Agent", task="t", result="missing index"),
        ]

    _patch_build_supervisor_success(monkeypatch, "SUMMARY: found\n\ndetails", seed=_seed)
    save_calls = _patch_save_report(monkeypatch)

    result = orchestrator.run_flow_once({"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5)

    assert result["bug_report"] == "root cause: X"
    assert result["perf_report"] == "missing index"
    assert save_calls[0]["bug_report"] == "root cause: X"


def test_run_flow_once_flow_raw_falls_back_to_delegation_capture_when_tool_capture_empty(monkeypatch):
    # tool_capture is the preferred source; delegation_capture's prose is only
    # a last resort when the flow-runner tool itself never succeeded.
    def _seed(run_id):
        delegation_capture._captures[run_id] = [
            delegation_capture.Delegation(
                coworker="API Flow Runner", task="t",
                result='{"network_log": [{"status": 500}], "step_results": []}',
            ),
        ]

    _patch_build_supervisor_success(monkeypatch, "SUMMARY: found\n\ndetails", seed=_seed)
    _patch_save_report(monkeypatch)

    result = orchestrator.run_flow_once({"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5)

    assert result["anomaly"]["bug_signal"] is True


def test_run_flow_once_records_known_issue_when_anomaly_present(monkeypatch):
    def _seed(run_id):
        tool_capture._captures[run_id] = {"flow": ['{"network_log": [{"status": 500}], "step_results": []}']}

    _patch_build_supervisor_success(monkeypatch, "SUMMARY: passwd column bug\n\ndetails", seed=_seed)
    _patch_save_report(monkeypatch)
    known_issues: list = []

    orchestrator.run_flow_once(
        {"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5, known_issues=known_issues,
    )

    assert len(known_issues) == 1
    assert known_issues[0]["flow_name"] == "demo_login"
    assert known_issues[0]["summary"] == "passwd column bug"


def test_run_flow_once_does_not_record_known_issue_on_clean_run(monkeypatch):
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nnothing found")
    _patch_save_report(monkeypatch)
    known_issues: list = []

    orchestrator.run_flow_once(
        {"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5, known_issues=known_issues,
    )

    assert known_issues == []


def test_run_flow_once_does_not_record_when_known_issues_is_none(monkeypatch):
    def _seed(run_id):
        tool_capture._captures[run_id] = {"flow": ['{"network_log": [{"status": 500}], "step_results": []}']}

    _patch_build_supervisor_success(monkeypatch, "SUMMARY: bug\n\ndetails", seed=_seed)
    _patch_save_report(monkeypatch)

    # known_issues=None (the default) means "standalone run" -- must not raise
    # even though the anomaly is non-clean, and nothing exists to record into.
    result = orchestrator.run_flow_once({"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5)
    assert result["anomaly"]["bug_signal"] is True


def test_run_flow_once_applies_and_logs_known_issue_note(monkeypatch, caplog):
    known_issues: list = []
    record_issue(known_issues, "other_flow", "already found this bug")
    captured_kwargs = {}
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok", capture_kwargs=captured_kwargs)
    _patch_save_report(monkeypatch)

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.orchestrator"):
        orchestrator.run_flow_once(
            {"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5, known_issues=known_issues,
        )

    applied = [r for r in caplog.records if r.getMessage() == "known_issue_note_applied"]
    assert len(applied) == 1
    assert "other_flow" in applied[0].extra_fields["note"]
    assert captured_kwargs["known_issue_note"] is not None
    assert "already found this bug" in captured_kwargs["known_issue_note"]


def test_run_flow_once_warns_on_missing_container_stack(monkeypatch, caplog):
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok")
    _patch_save_report(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="bee_bug_hunter.orchestrator"):
        orchestrator.run_flow_once(
            {"name": "demo_login", "containers": ["demo_app-web-1", "demo_app-db-1"]},
            duration_seconds=5,
            container_stacks={"demo_app-web-1": "flask"},
        )

    warnings = [r for r in caplog.records if r.getMessage() == "container_stack_missing"]
    assert len(warnings) == 1
    assert warnings[0].extra_fields["containers"] == ["demo_app-db-1"]


def test_run_flow_once_passes_filtered_container_stacks_to_build_supervisor(monkeypatch):
    captured_kwargs = {}
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok", capture_kwargs=captured_kwargs)
    _patch_save_report(monkeypatch)

    orchestrator.run_flow_once(
        {"name": "demo_login", "containers": ["demo_app-web-1"]},
        duration_seconds=5,
        container_stacks={"demo_app-web-1": "flask", "unrelated-container": "django"},
    )

    assert captured_kwargs["container_stacks"] == {"demo_app-web-1": "flask"}


def test_run_flow_once_docker_host_override_from_flow_cfg_takes_precedence(monkeypatch):
    captured_kwargs = {}
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok", capture_kwargs=captured_kwargs)
    _patch_save_report(monkeypatch)
    monkeypatch.setenv("BEE_DEFAULT_DOCKER_HOST", "tcp://env-default:2375")

    orchestrator.run_flow_once(
        {"name": "demo_login", "containers": ["demo_app-web-1"], "docker_host": "tcp://flow-override:2375"},
        duration_seconds=5,
    )

    assert captured_kwargs["docker_host"] == "tcp://flow-override:2375"


def test_run_flow_once_docker_host_falls_back_to_env_default(monkeypatch):
    captured_kwargs = {}
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok", capture_kwargs=captured_kwargs)
    _patch_save_report(monkeypatch)
    monkeypatch.setenv("BEE_DEFAULT_DOCKER_HOST", "tcp://env-default:2375")

    orchestrator.run_flow_once({"name": "demo_login", "containers": ["demo_app-web-1"]}, duration_seconds=5)

    assert captured_kwargs["docker_host"] == "tcp://env-default:2375"


def test_run_batch_once_clears_mysql_and_source_scratch_caches(monkeypatch):
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok")
    _patch_save_report(monkeypatch)
    mysql_calls = []
    scratch_calls = []
    monkeypatch.setattr(orchestrator, "clear_schema_cache", lambda: mysql_calls.append(1))
    monkeypatch.setattr(orchestrator, "clear_scratch_cache", lambda: scratch_calls.append(1))

    manifest = {"duration_seconds": 5, "flows": [{"name": "demo_login", "containers": ["demo_app-web-1"]}]}
    orchestrator.run_batch_once(manifest)

    assert mysql_calls == [1]
    assert scratch_calls == [1]


def test_run_batch_once_clears_claude_cli_sessions_when_that_provider_is_active(monkeypatch):
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok")
    _patch_save_report(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "claude_cli")
    calls = []
    monkeypatch.setattr("bee_bug_hunter.claude_cli_llm.clear_persisted_sessions", lambda: calls.append(1))

    manifest = {"duration_seconds": 5, "flows": [{"name": "demo_login", "containers": ["demo_app-web-1"]}]}
    orchestrator.run_batch_once(manifest)

    assert calls == [1]


def test_run_batch_once_clears_copilot_cli_sessions_when_that_provider_is_active(monkeypatch):
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok")
    _patch_save_report(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "copilot_cli")
    calls = []
    monkeypatch.setattr("bee_bug_hunter.copilot_cli_llm.clear_persisted_sessions", lambda: calls.append(1))

    manifest = {"duration_seconds": 5, "flows": [{"name": "demo_login", "containers": ["demo_app-web-1"]}]}
    orchestrator.run_batch_once(manifest)

    assert calls == [1]


def test_run_batch_once_skips_flow_on_bare_exception_not_just_framework_error(monkeypatch, caplog):
    _patch_build_supervisor(monkeypatch, ValueError("not a framework error"))
    manifest = {"duration_seconds": 5, "flows": [{"name": "demo_login", "containers": ["demo_app-web-1"]}]}

    with caplog.at_level(logging.ERROR, logger="bee_bug_hunter.orchestrator"):
        results = orchestrator.run_batch_once(manifest)

    assert results == []
    skipped = [r for r in caplog.records if r.getMessage() == "flow_run_skipped_after_failure"]
    assert len(skipped) == 1
    assert skipped[0].extra_fields == {"flow": "demo_login"}


def test_run_batch_once_returns_results_for_all_successful_flows(monkeypatch):
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok")
    _patch_save_report(monkeypatch)

    manifest = {
        "duration_seconds": 5,
        "flows": [
            {"name": "flow_a", "containers": ["c1"]},
            {"name": "flow_b", "containers": ["c2"]},
        ],
    }
    results = orchestrator.run_batch_once(manifest)

    assert [r["flow"] for r in results] == ["flow_a", "flow_b"]


def test_run_batch_once_uses_default_duration_when_not_specified(monkeypatch):
    captured_durations = []

    def _fake_run_flow_once(flow_cfg, duration_seconds, known_issues=None, container_stacks=None):
        captured_durations.append(duration_seconds)
        return {"flow": flow_cfg["name"], "response": ""}

    monkeypatch.setattr(orchestrator, "run_flow_once", _fake_run_flow_once)

    orchestrator.run_batch_once({"flows": [{"name": "demo_login", "containers": ["c1"]}]})

    assert captured_durations == [30]


def test_monitor_loop_once_returns_without_sleeping(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("flows:\n  - name: demo_login\n    containers: [demo_app-web-1]\n")

    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok")
    _patch_save_report(monkeypatch)

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("time.sleep must not be called when once=True")

    monkeypatch.setattr(orchestrator.time, "sleep", _fail_if_called)

    results = orchestrator.monitor_loop(str(manifest_path), once=True)

    assert len(results) == 1
    assert results[0]["flow"] == "demo_login"


def test_monitor_loop_logs_response_report_when_response_present(monkeypatch, tmp_path, caplog):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("flows:\n  - name: demo_login\n    containers: [demo_app-web-1]\n")

    _patch_build_supervisor_success(monkeypatch, "SUMMARY: bug found\n\nfull report text")
    _patch_save_report(monkeypatch)

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.orchestrator"):
        orchestrator.monitor_loop(str(manifest_path), once=True)

    reported = [r for r in caplog.records if r.getMessage() == "response_report"]
    assert len(reported) == 1
    assert reported[0].extra_fields["flow"] == "demo_login"


def test_monitor_loop_sleeps_and_loops_again_when_not_once(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "poll_interval_seconds: 42\nflows:\n  - name: demo_login\n    containers: [demo_app-web-1]\n"
    )
    _patch_build_supervisor_success(monkeypatch, "SUMMARY: clean\n\nok")
    _patch_save_report(monkeypatch)

    sleep_calls = []

    class _StopTheLoop(Exception):
        pass

    def _fake_sleep(seconds):
        sleep_calls.append(seconds)
        raise _StopTheLoop  # escape the infinite while True after the first pass

    monkeypatch.setattr(orchestrator.time, "sleep", _fake_sleep)

    with pytest.raises(_StopTheLoop):
        orchestrator.monitor_loop(str(manifest_path), once=False)

    assert sleep_calls == [42]


@pytest.fixture(autouse=True)
def _cleanup_capture_state():
    yield
    with delegation_capture._lock:
        delegation_capture._captures.clear()
    with tool_capture._lock:
        tool_capture._captures.clear()


@pytest.fixture(autouse=True)
def _run_id_context():
    new_run_context("test-flow")
    yield
