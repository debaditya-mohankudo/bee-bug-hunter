"""Tests for ClaudeCLIChatModel's session topology: the Investigation Manager
(MANAGER_ROLE_NAME) gets its own reserved session, minted directly on its own
first call and never forked from anything; every worker instead forks its own
session off a separate, content-free "root" session.

An earlier version had workers fork directly off the manager's own live
session instead of this root, on the theory that richer shared context would
help. Verified live against a real flow (2026-07-30) that this backfires: the
manager's system prompt carries all six handoff tool schemas, and that leaked
into every worker's forked history, causing workers to hallucinate handoff-tool
calls they were never given (e.g. the API Flow Runner worker attempting to
call "docker_log_capturer"), each one wasting a retry and once cascading into
a fatal CLI error. These tests guard the corrected separation: the manager's
own session must never be used as anything's fork source.
"""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.backend.types import ChatModelInput

from bee_bug_hunter.claude_cli_llm import (
    ClaudeCLIChatModel,
    _ensure_root_session,
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
    CLI is expected to honor that exact id -- _ensure_root_session never reads
    a session_id back out of the response, it just returns the uuid it
    generated and told the CLI to use."""
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
def test_ensure_root_session_bootstraps_and_persists_under_root_key(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        session_id = _ensure_root_session("demo_flow", "demo_flow", "demo-api,demo-db", "sonnet")

    assert session_id == _session_id_arg(mock_run, "--session-id")

    persisted = _load_persisted_sessions()
    assert persisted["demo_flow"]["root"] == session_id
    # The root session is content-free framing only -- it must never be keyed
    # under roles[MANAGER_ROLE_NAME], or a worker's fork_from lookup and the
    # manager's own session resolution could collide.
    assert persisted["demo_flow"].get("roles", {}).get(MANAGER_ROLE_NAME) is None


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_ensure_root_session_is_idempotent(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        first = _ensure_root_session("demo_flow", "demo_flow", "demo-api", "sonnet")
    assert mock_run.call_count == 1

    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run") as mock_run_again:
        second = _ensure_root_session("demo_flow", "demo_flow", "demo-api", "sonnet")

    assert second == first
    mock_run_again.assert_not_called()


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_worker_for_role_forks_from_content_free_root_not_the_manager(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        worker = ClaudeCLIChatModel.for_role(
            "Bug Analyst", model="sonnet", flow_name="demo_flow", containers="demo-api",
        )
        root_id = _session_id_arg(mock_run, "--session-id")

    assert worker._role_key == "Bug Analyst"
    assert worker._session_id is None  # not yet forked -- lazy, happens on first real _create()
    assert worker._fork_from == root_id

    persisted = _load_persisted_sessions()
    assert persisted["demo_flow"]["root"] == root_id
    assert MANAGER_ROLE_NAME not in persisted["demo_flow"].get("roles", {})


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_manager_for_role_mints_its_own_standalone_session_without_forking(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run") as mock_run:
        manager = ClaudeCLIChatModel.for_role(
            MANAGER_ROLE_NAME, model="sonnet", flow_name="demo_flow", containers="demo-api",
        )

    # Constructing the manager must not itself invoke the CLI (no root
    # bootstrap needed for the manager's own path) -- the manager mints its
    # session lazily on its own first real _create()/_invoke_cli() call.
    mock_run.assert_not_called()
    assert manager._session_id is None
    assert manager._session_started is False
    assert manager._fork_from is None


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_worker_built_before_manager_does_not_leak_a_fork_target_to_the_manager(mock_which):
    """Mirrors manager.py's real construction order: build_agents() constructs
    every worker's ChatModel before build_supervisor() constructs the manager's
    own. A worker's for_role() call bootstraps root as a side effect -- the
    manager's own later for_role() call must still end up with fork_from=None,
    not accidentally pick up root as its fork target."""
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()) as mock_run:
        worker = ClaudeCLIChatModel.for_role(
            "API Flow Runner", model="sonnet", flow_name="demo_flow", containers="demo-api",
        )
        root_id = _session_id_arg(mock_run, "--session-id")
        manager = ClaudeCLIChatModel.for_role(
            MANAGER_ROLE_NAME, model="sonnet", flow_name="demo_flow", containers="demo-api",
        )

    assert worker._fork_from == root_id
    assert manager._fork_from is None
    assert manager._session_id is None


@patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
def test_clear_persisted_sessions_wipes_root_and_worker_sessions(mock_which):
    with patch("bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_bootstrap_proc()):
        ClaudeCLIChatModel.for_role("Bug Analyst", model="sonnet", flow_name="demo_flow", containers="demo-api")

    assert _load_persisted_sessions()

    clear_persisted_sessions()

    assert _load_persisted_sessions() == {}


def _fake_cli_call_proc(session_id: str, result: str = "ack"):
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = json.dumps({"result": result, "session_id": session_id, "is_error": False})
    proc.stderr = ""
    return proc


class TestMessageDedupAcrossRepeatDelegations:
    """A worker role can be delegated to more than once in one investigation
    (e.g. DB Query Agent, run multiple times for several queries). Each
    delegation, HandoffTool resets and reseeds that worker's BeeAI memory --
    the reseeded input.messages on delegation 2 is unrelated in length to
    whatever was sent to the CLI on delegation 1, since it reflects the
    manager's conversation *at that later point* rather than continuing where
    delegation 1's own internal reasoning left off. The old
    `messages[sent_count:]` position cursor broke on this: if delegation 2's
    reseeded list happened to be shorter than delegation 1's final message
    count, the slice went out of range and silently sent an empty prompt.
    Content-signature dedup (cli_tool_protocol.select_new_messages) fixes
    this -- these tests reproduce the exact shape of the bug directly through
    _create(), not just the pure helper function in isolation."""

    @patch("bee_bug_hunter.claude_cli_llm.shutil.which", return_value="/usr/bin/claude")
    def test_shorter_reseeded_message_list_still_sends_a_nonempty_prompt(self, mock_which):
        model = ClaudeCLIChatModel(model="sonnet", flow_key="demo", role_key="DB Query Agent")

        # Delegation 1: a long-ish internal reasoning trace accumulates.
        delegation_1_messages = [
            UserMessage("original manager instructions"),
            UserMessage("run this query"),
            AssistantMessage("tool call 1"),
            AssistantMessage("tool call 2"),
            AssistantMessage("final answer 1"),
        ]
        with patch(
            "bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_cli_call_proc("sess-1"),
        ):
            asyncio.run(model._create(ChatModelInput(messages=delegation_1_messages, tools=[]), run=None))

        assert len(model._sent_message_signatures) == 5

        # Delegation 2: HandoffTool reset+reseeded this worker's memory to a
        # SHORTER list reflecting the manager's state at this later point --
        # same original instructions (already sent) plus one new task.
        delegation_2_messages = [
            UserMessage("original manager instructions"),
            UserMessage("run a different query this time"),
        ]
        with patch(
            "bee_bug_hunter.claude_cli_llm.subprocess.run", return_value=_fake_cli_call_proc("sess-1"),
        ) as mock_run:
            asyncio.run(model._create(ChatModelInput(messages=delegation_2_messages, tools=[]), run=None))

        sent_prompt = mock_run.call_args.kwargs.get("input") or mock_run.call_args[1].get("input")
        assert sent_prompt, "delegation 2's prompt must not be empty"
        assert "run a different query this time" in sent_prompt
        # The already-sent original instructions must NOT be resent verbatim.
        assert sent_prompt.count("original manager instructions") == 0
    assert ClaudeCLIChatModel._instances == {}
