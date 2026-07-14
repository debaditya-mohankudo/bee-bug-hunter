"""Tests for CopilotCLIChatModel: CLI invocation shape (excluded-tools, session
resume/mint) and the copilot-specific JSONL event parsing in _invoke_cli. The
shared text-bridge protocol (tool-call JSON extraction, message flattening) is
covered by claude_cli_llm's usage of the same cli_tool_protocol module and
isn't re-tested here.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from bee_bug_hunter.copilot_cli_llm import CopilotCLIChatModel, clear_persisted_sessions


@pytest.fixture(autouse=True)
def _clear_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_CLI_SESSION_STORE", str(tmp_path / "sessions.json"))
    clear_persisted_sessions()
    yield
    clear_persisted_sessions()


def _fake_proc(stdout_lines: list[dict], returncode: int = 0, stderr: str = ""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = "\n".join(json.dumps(line) for line in stdout_lines)
    proc.stderr = stderr
    return proc


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_first_call_mints_session_id_via_session_id_flag(mock_which):
    model = CopilotCLIChatModel(model="claude-sonnet-4.5", flow_key="demo", role_key="Bug Analyst")
    assert model._session_id is None
    assert model._session_started is False

    events = [
        {"type": "assistant.message", "data": {"content": "hello", "model": "gpt-5.3-codex"}},
        {"type": "result", "sessionId": "minted-123", "usage": {"premiumRequests": 0}},
    ]
    with patch("bee_bug_hunter.copilot_cli_llm.subprocess.run", return_value=_fake_proc(events)) as mock_run:
        text, session_id, resolved_model, premium_requests = model._invoke_cli("system framing", "do the thing")

    assert text == "hello"
    assert session_id == "minted-123"
    assert resolved_model == "gpt-5.3-codex"
    assert premium_requests == 0
    cmd = mock_run.call_args.args[0]
    assert "--session-id" in cmd
    assert "--resume" not in cmd
    assert "--excluded-tools" in cmd
    assert "--disable-builtin-mcps" in cmd
    assert "system framing\n\ndo the thing" in cmd


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_known_session_resumes_via_resume_flag(mock_which):
    model = CopilotCLIChatModel(
        model="claude-sonnet-4.5", flow_key="demo", role_key="Bug Analyst", session_id="known-abc",
    )
    events = [
        {"type": "assistant.message", "data": {"content": "continuing", "model": "claude-sonnet-4.5"}},
        {"type": "result", "sessionId": "known-abc", "usage": {"premiumRequests": 1}},
    ]
    with patch("bee_bug_hunter.copilot_cli_llm.subprocess.run", return_value=_fake_proc(events)) as mock_run:
        text, session_id, _resolved_model, _premium_requests = model._invoke_cli("", "next turn")

    assert text == "continuing"
    cmd = mock_run.call_args.args[0]
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == "known-abc"
    assert "--session-id" not in cmd


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_last_assistant_message_wins_over_earlier_turns(mock_which):
    model = CopilotCLIChatModel(model="claude-sonnet-4.5", flow_key="demo", role_key="Bug Analyst")
    events = [
        {"type": "assistant.message", "data": {"content": "first turn (e.g. a hallucinated tool attempt)"}},
        {"type": "assistant.message", "data": {"content": "final reply"}},
        {"type": "result", "sessionId": "sess-1"},
    ]
    with patch("bee_bug_hunter.copilot_cli_llm.subprocess.run", return_value=_fake_proc(events)):
        text, _session_id, _resolved_model, _premium_requests = model._invoke_cli("", "prompt")

    assert text == "final reply"


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_nonzero_exit_raises_with_stderr_detail(mock_which):
    model = CopilotCLIChatModel(model="claude-sonnet-4.5", flow_key="demo", role_key="Bug Analyst")
    proc = _fake_proc([], returncode=1, stderr="Error: Model not available.")
    with patch("bee_bug_hunter.copilot_cli_llm.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="Model not available"):
            model._invoke_cli("", "prompt")


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_no_assistant_message_event_raises(mock_which):
    model = CopilotCLIChatModel(model="claude-sonnet-4.5", flow_key="demo", role_key="Bug Analyst")
    events = [{"type": "result", "sessionId": "sess-1"}]
    with patch("bee_bug_hunter.copilot_cli_llm.subprocess.run", return_value=_fake_proc(events)):
        with pytest.raises(RuntimeError, match="no assistant.message"):
            model._invoke_cli("", "prompt")


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_auto_model_reports_resolved_model_and_premium_requests(mock_which):
    """model="auto" leaves self._model as the literal string "auto" -- the
    caller needs the CLI's own per-event `model` field (and premiumRequests)
    to know what actually ran and whether it cost quota."""
    model = CopilotCLIChatModel(model="auto", flow_key="demo", role_key="Bug Analyst")
    events = [
        {"type": "assistant.message", "data": {"content": "hello", "model": "gpt-5.3-codex"}},
        {"type": "result", "sessionId": "sess-auto", "usage": {"premiumRequests": 0}},
    ]
    with patch("bee_bug_hunter.copilot_cli_llm.subprocess.run", return_value=_fake_proc(events)):
        _text, _session_id, resolved_model, premium_requests = model._invoke_cli("", "prompt")

    assert model._model == "auto"
    assert resolved_model == "gpt-5.3-codex"
    assert premium_requests == 0


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_for_role_gives_distinct_instances_per_role_same_flow(mock_which):
    a = CopilotCLIChatModel.for_role("Bug Analyst", model="claude-sonnet-4.5", flow_name="demo")
    b = CopilotCLIChatModel.for_role("SQL Performance Agent", model="claude-sonnet-4.5", flow_name="demo")
    same_role_again = CopilotCLIChatModel.for_role("Bug Analyst", model="claude-sonnet-4.5", flow_name="demo")

    assert a is not b
    assert a is same_role_again


@patch("bee_bug_hunter.copilot_cli_llm.shutil.which", return_value="/usr/bin/copilot")
def test_clone_returns_self(mock_which):
    import asyncio

    model = CopilotCLIChatModel(model="claude-sonnet-4.5", flow_key="demo", role_key="Bug Analyst")
    assert asyncio.run(model.clone()) is model
