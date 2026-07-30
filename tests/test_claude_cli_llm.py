"""Tests for ClaudeCLIChatModel's manager+fork session topology: the
Investigation Manager (MANAGER_ROLE_NAME) gets its own reserved session,
bootstrapped once per flow and never forked from anything; every worker role
forks its own session off the manager's session on first use instead of off a
separate, generic root. Whichever role's for_role() runs first for a flow
(worker or manager -- build_agents() actually constructs workers before the
manager in manager.py) must still end up bootstrapping the SAME manager
session that the manager's own for_role() call later resumes.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from bee_bug_hunter.claude_cli_llm import (
    ClaudeCLIChatModel,
    _ensure_manager_session,
    _load_persisted_sessions,
    clear_persisted_sessions,
)
from bee_bug_hunter.config import MANAGER_ROLE_NAME


@pytest.fixture(autouse=True)
def _clear_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CLI_SESSION_STORE", str(tmp_path / "sessions.json"))
    clear_persisted_sessions()
    yield
    clear_persisted_sessions()


def _fake_bootstrap_proc(returncode: int = 0, is_error: bool = False):
    """The bootstrap call passes --session-id explicitly (not --resume), so the
    CLI is expected to honor that exact id -- _ensure_manager_session never
    reads a session_id back out of the response, it just returns the uuid it
    generated and told the CLI to use. The fake response's own session_id
    field is irrelevant here; only is_error/returncode matter."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = json.dumps({"result": "Acknowledged.", "is_error": is_error})
    proc.stderr = ""
    return proc


def _session_id_arg(mock_run, flag: str) -> str:
    cmd = mock_run.call_args[0][0]
    assert flag in cmd
    return cmd[cmd.index(flag) + 1]


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_ensure_manager_session_bootstraps_and_persists_under_manager_role(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        session_id = _ensure_manager_session("demo_flow", "demo_flow", "demo-api,demo-db", "sonnet")

    assert session_id == _session_id_arg(mock_run, "--session-id")

    persisted = _load_persisted_sessions()
    assert persisted["demo_flow"]["roles"][MANAGER_ROLE_NAME] == session_id


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_ensure_manager_session_is_idempotent(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        first = _ensure_manager_session("demo_flow", "demo_flow", "demo-api", "sonnet")
    assert mock_run.call_count == 1

    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run") as mock_run_again:
        second = _ensure_manager_session("demo_flow", "demo_flow", "demo-api", "sonnet")

    assert second == first
    mock_run_again.assert_not_called()


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_worker_for_role_forks_from_manager_session_not_a_separate_root(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        worker = ClaudeCLIChatModel.for_role(
            "Bug Analyst", model="sonnet", flow_name="demo_flow", containers="demo-api",
        )
        manager_session_id = _session_id_arg(mock_run, "--session-id")

    assert worker._role_key == "Bug Analyst"
    assert worker._session_id is None  # not yet forked -- lazy, happens on first real _create()
    assert worker._fork_from == manager_session_id

    persisted = _load_persisted_sessions()
    assert "root" not in persisted["demo_flow"]  # the old generic-root key no longer exists
    assert persisted["demo_flow"]["roles"][MANAGER_ROLE_NAME] == manager_session_id


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_manager_for_role_resumes_its_own_bootstrapped_session_without_forking(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        manager = ClaudeCLIChatModel.for_role(
            MANAGER_ROLE_NAME, model="sonnet", flow_name="demo_flow", containers="demo-api",
        )
        manager_session_id = _session_id_arg(mock_run, "--session-id")

    # The bootstrap call IS the manager's real session -- __init__ must adopt
    # it as session_id directly (already-started, never forked), not treat it
    # as merely a fork_from target.
    assert manager._session_id == manager_session_id
    assert manager._session_started is True
    assert manager._fork_from is None


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_worker_built_before_manager_still_forks_off_the_same_manager_session(mock_which):
    """Mirrors manager.py's real construction order: build_agents() constructs
    every worker's ChatModel before build_supervisor() constructs the manager's
    own. The manager session must already exist (bootstrapped by whichever
    worker asks first) so that when the manager's own for_role() call happens
    later, it resumes that exact session rather than minting a second one."""
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        worker = ClaudeCLIChatModel.for_role(
            "API Flow Runner", model="sonnet", flow_name="demo_flow", containers="demo-api",
        )
        manager_session_id = _session_id_arg(mock_run, "--session-id")
        manager = ClaudeCLIChatModel.for_role(
            MANAGER_ROLE_NAME, model="sonnet", flow_name="demo_flow", containers="demo-api",
        )

    assert worker._fork_from == manager_session_id
    assert manager._session_id == manager_session_id
    assert manager._fork_from is None


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_clear_persisted_sessions_wipes_manager_and_worker_sessions(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()):
        ClaudeCLIChatModel.for_role("Bug Analyst", model="sonnet", flow_name="demo_flow", containers="demo-api")

    assert _load_persisted_sessions()

    clear_persisted_sessions()

    assert _load_persisted_sessions() == {}
    assert ClaudeCLIChatModel._instances == {}
