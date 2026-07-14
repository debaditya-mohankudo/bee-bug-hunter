"""BeeAI-compatible ChatModel that shells out to `copilot -p` (GitHub Copilot
CLI) instead of an API/SDK -- the same text-bridge idea as claude_cli_llm.py,
adapted to the Copilot CLI's different flags. Reuses the caller's existing
`copilot` login (no ANTHROPIC_API_KEY needed); --model claude-sonnet-4.5
(configurable) routes through Copilot's own Claude access.

Tool calling is bridged by hand exactly as in claude_cli_llm.py: tool schemas
are described in the prompt, the model is asked to reply with a small JSON
protocol ({"tool": ..., "args": {...}} or {"final_answer": ...}), and
_create() translates a tool-call reply into a native MessageToolCallContent
so BeeAI's own agent loop executes the tool and feeds the result back. The
CLI-agnostic half of that bridge (prompt text, JSON extraction, message
flattening) lives in cli_tool_protocol.py, shared with claude_cli_llm.py.

Two real capability gaps versus the `claude` CLI, both worked around here:

1. No `--tools none` equivalent. Copilot CLI always has *some* tool surface
   (bash, file edit, its built-in github-mcp-server, ...) unless explicitly
   excluded -- `--available-tools=` (empty) alone does NOT disable it (verified
   empirically: the model still executed a real shell command). The fix is
   `--excluded-tools=<every built-in tool name>` (_EXCLUDED_TOOLS below) plus
   `--disable-builtin-mcps`, which does genuinely prevent real execution (the
   model instead hallucinates a `<run_command>`-shaped block in prose, which
   the same JSON tool-protocol parsing below simply ignores as non-JSON text).

2. No `--append-system-prompt` / `--fork-session` equivalent. Unlike
   claude_cli_llm.py's shared-root-then-fork topology, there's no way to seed
   a role-specific session from a shared generic-framing parent here -- system
   and user text are folded into one combined -p prompt argument, and each
   role's session starts cold on its first call rather than forking from a
   root. This loses some of the cross-role cache-priming claude_cli_llm.py
   gets, but per-role session reuse (via --resume) on every later call within
   that role still holds, which is the main cache win.
"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, MessageToolCallContent
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput

from bee_bug_hunter.cli_tool_protocol import (
    FORCED_TOOL_INSTRUCTIONS as _FORCED_TOOL_INSTRUCTIONS,
    REQUIRED_TOOL_CHOICE_INSTRUCTIONS as _REQUIRED_TOOL_CHOICE_INSTRUCTIONS,
    REQUIRED_TOOL_CHOICE_WITH_FINAL_ANSWER_INSTRUCTIONS as _REQUIRED_TOOL_CHOICE_WITH_FINAL_ANSWER_INSTRUCTIONS,
    TOOL_PROTOCOL_INSTRUCTIONS as _TOOL_PROTOCOL_INSTRUCTIONS,
    describe_tools as _describe_tools,
    extract_json_object as _extract_json_object,
    flatten_messages as _flatten_messages,
)
from bee_bug_hunter.config import DEFAULT_COPILOT_CLI_SESSION_STORE
from bee_bug_hunter.logging_config import get_logger, log

logger = get_logger(__name__)

_session_store_lock = threading.Lock()

# Every built-in tool `copilot -p` currently offers (queried live via `copilot -p
# "list all tool names available to you"`), excluded so the CLI can never actually
# execute anything on this host -- BeeAI's own agent loop is the only thing that
# should be running tools, via the hand-rolled JSON protocol below. Only the
# github-mcp-server toolset is covered by --disable-builtin-mcps separately.
_EXCLUDED_TOOLS = (
    "bash,read_bash,stop_bash,list_bash,view,create,edit,web_fetch,"
    "fetch_copilot_cli_documentation,skill,sql,session_store_sql,read_agent,"
    "list_agents,grep,glob,task,web_search"
)


def _session_store_path() -> Path:
    return Path(os.getenv("COPILOT_CLI_SESSION_STORE", DEFAULT_COPILOT_CLI_SESSION_STORE))


def _load_persisted_sessions() -> dict[str, dict]:
    """{flow_key: {role_key: session_id}}, persisted to disk so sessions survive
    process restarts -- same rationale as claude_cli_llm.py's
    _load_persisted_sessions()."""
    path = _session_store_path()
    if not path.exists():
        return {}
    try:
        with _session_store_lock, path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(logger, logging.WARNING, "copilot_cli_session_store_read_failed", path=str(path), error=str(e))
        return {}


def _persist_role_session(flow_key: str, role_key: str, session_id: str) -> None:
    path = _session_store_path()
    with _session_store_lock:
        try:
            sessions = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            sessions = {}
        sessions.setdefault(flow_key, {})[role_key] = session_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sessions, indent=2))


def clear_persisted_sessions() -> None:
    """Wipes the on-disk session store (and any in-memory CopilotCLIChatModel
    instances) -- same crash-recovery rationale as claude_cli_llm.py's
    clear_persisted_sessions(), called at the top of every
    orchestrator.run_batch_once iteration."""
    path = _session_store_path()
    with _session_store_lock:
        path.unlink(missing_ok=True)
    CopilotCLIChatModel._instances.clear()


class CopilotCLIChatModel(ChatModel):
    """Shells to `copilot -p` with every built-in tool excluded. Each role
    (worker or manager) gets its own persistent, per-flow CLI session (see
    for_role()), resumed via --resume on every later turn.

    No API key needed -- reuses the caller's `copilot` CLI login. No native
    streaming; _create_stream just yields the full _create result once.
    """

    # (flow_key, role_key) -> singleton instance, mirroring
    # ClaudeCLIChatModel._instances -- see for_role()'s docstring.
    _instances: dict[tuple[str, str], "CopilotCLIChatModel"] = {}

    @classmethod
    def for_role(
        cls, role: str | None, *, model: str, flow_name: str = "default", **kwargs: Any,
    ) -> "CopilotCLIChatModel":
        """Returns the singleton CopilotCLIChatModel for this (flow, role) pair,
        constructing it on first use. One instance per role keeps each worker's
        conversation isolated from the others' (same "private per-agent memory"
        design as claude_cli_llm.py's for_role()) while letting repeat
        delegations to the *same* role resume their one ongoing session (see
        clone())."""
        flow_key = flow_name or "default"
        role_key = role or "default"
        cache_key = (flow_key, role_key)
        if cache_key not in cls._instances:
            persisted = _load_persisted_sessions()
            existing_session = persisted.get(flow_key, {}).get(role_key)
            cls._instances[cache_key] = cls(
                model=model, flow_key=flow_key, role_key=role_key,
                session_id=existing_session, **kwargs,
            )
        return cls._instances[cache_key]

    def __init__(
        self, model: str = "claude-sonnet-4.5", *, flow_key: str = "default", role_key: str = "default",
        session_id: str | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        copilot_path = shutil.which("copilot")
        if not copilot_path:
            raise RuntimeError("copilot_cli provider: `copilot` binary not found on PATH")
        self._copilot_path = copilot_path
        self._flow_key = flow_key
        self._role_key = role_key
        self._sent_message_count = 0
        self._session_id = session_id
        self._session_started = session_id is not None

    async def clone(self) -> "CopilotCLIChatModel":
        # Deliberately returns self -- see ClaudeCLIChatModel.clone()'s docstring
        # for why (HandoffTool clones the target agent's llm on every delegation;
        # returning self here means "resume this role's one ongoing session," not
        # "start a fresh cold one per delegation").
        return self

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_id(self) -> str:
        return "copilot_cli"

    async def _create(self, input: ChatModelInput, run) -> ChatModelOutput:
        new_messages = input.messages[self._sent_message_count:] if self._session_started else input.messages
        system_prompt, prompt = _flatten_messages(new_messages)
        if input.tools:
            system_prompt = (
                system_prompt + "\n\n" +
                _TOOL_PROTOCOL_INSTRUCTIONS.format(tool_descriptions=_describe_tools(input.tools))
            )

        final_answer_allowed = any(t.name == "final_answer" for t in (input.tools or []))
        tool_choice = input.tool_choice
        if tool_choice == "required":
            system_prompt += "\n\n" + (
                _REQUIRED_TOOL_CHOICE_WITH_FINAL_ANSWER_INSTRUCTIONS
                if final_answer_allowed
                else _REQUIRED_TOOL_CHOICE_INSTRUCTIONS
            )
        elif tool_choice is not None and hasattr(tool_choice, "name"):
            system_prompt += "\n\n" + _FORCED_TOOL_INSTRUCTIONS.format(tool_name=tool_choice.name)

        tool_choice_str = "single" if hasattr(tool_choice, "name") else str(tool_choice)
        is_new_session = not self._session_started
        started = time.monotonic()
        raw, returned_session_id, resolved_model, premium_requests = await asyncio.to_thread(
            self._invoke_cli, system_prompt, prompt,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        if self._session_id is None:
            self._session_id = returned_session_id
            _persist_role_session(self._flow_key, self._role_key, self._session_id)
        self._session_started = True
        self._sent_message_count = len(input.messages)
        parsed = _extract_json_object(raw)

        # See claude_cli_llm.py's identical block for the full rationale: a
        # legitimate "I'm done" reply wrapped in final_answer on a turn that
        # demanded a tool call is converted locally rather than failed/retried.
        if (
            parsed and "tool" not in parsed and "final_answer" in parsed
            and tool_choice == "required" and final_answer_allowed
        ):
            answer = parsed["final_answer"]
            if not isinstance(answer, str):
                answer = json.dumps(answer)
            parsed = {"tool": "final_answer", "args": {"response": answer}}
            log(
                logger, logging.INFO, "copilot_cli_final_answer_converted",
                model=self._model, session_id=self._session_id, flow=self._flow_key,
            )

        if parsed and "tool" in parsed and input.tools:
            log(
                logger, logging.INFO, "copilot_cli_call",
                model=self._model, resolved_model=resolved_model, premium_requests=premium_requests,
                tool_choice=tool_choice_str, elapsed_ms=elapsed_ms,
                outcome="tool_call", tool_name=parsed["tool"],
                session_id=self._session_id, new_session=is_new_session, flow=self._flow_key,
            )
            content = MessageToolCallContent(
                id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name=parsed["tool"],
                args=json.dumps(parsed.get("args", {})),
            )
            return ChatModelOutput(
                output=[AssistantMessage(content)], finish_reason="tool_calls",
            )

        outcome = "text"
        if tool_choice_str != "auto" and tool_choice_str != "None" and (not parsed or "tool" not in parsed):
            outcome = "missed_required_tool_call"
        log(
            logger, logging.INFO if outcome == "text" else logging.WARNING, "copilot_cli_call",
            model=self._model, resolved_model=resolved_model, premium_requests=premium_requests,
            tool_choice=tool_choice_str, elapsed_ms=elapsed_ms, outcome=outcome,
            session_id=self._session_id, new_session=is_new_session, flow=self._flow_key,
            **({"raw_preview": raw[:500]} if outcome == "missed_required_tool_call" else {}),
        )

        text = parsed["final_answer"] if parsed and "final_answer" in parsed else raw
        if not isinstance(text, str):
            text = json.dumps(text)
        return ChatModelOutput(output=[AssistantMessage(text)], finish_reason="stop")

    async def _create_stream(self, input: ChatModelInput, run) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)

    def _invoke_cli(self, system_prompt: str, user_prompt: str) -> tuple[str, str, str, int | None]:
        """Returns (result_text, session_id, resolved_model, premium_requests).
        resolved_model/premium_requests come straight off the CLI's own JSONL
        event stream -- mainly useful when self._model == "auto", since our own
        model= log field would otherwise just say the literal string "auto"
        forever with no way to tell which model actually ran. No
        --append-system-prompt exists on this CLI (see module docstring), so
        system_prompt and user_prompt are folded into one combined -p argument
        here rather than split across a flag and stdin. --session-id mints a
        fresh session on first call for this role; --resume continues an
        already-known one (this process or a persisted prior one)."""
        combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        cmd = [
            self._copilot_path, "-p", combined_prompt,
            "--model", self._model,
            "--output-format", "json",
            "--excluded-tools", _EXCLUDED_TOOLS,
            "--disable-builtin-mcps",
        ]
        if self._session_id is not None:
            cmd += ["--resume", self._session_id]
        else:
            self._session_id = str(uuid.uuid4())
            cmd += ["--session-id", self._session_id]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout[:500]
            raise RuntimeError(f"copilot CLI exited {proc.returncode}: {detail}")

        result_text = None
        session_id = self._session_id
        resolved_model = self._model
        premium_requests = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "assistant.message":
                # Later turns overwrite earlier ones -- the last assistant.message
                # in the stream is the model's actual final reply for this call
                # (a hallucinated <run_command> block from an excluded tool, if
                # any, shows up as *prose inside* this same content, not a
                # separate real tool-execution event -- see module docstring).
                result_text = event["data"]["content"]
                # Only meaningful when self._model == "auto": the CLI's own
                # per-message `model` field reports what it actually resolved
                # to (e.g. "gpt-5.3-codex"), which the JSONL log needs -- our
                # own model= field would otherwise just say the literal
                # string "auto" forever, useless for telling which model
                # actually ran a given call.
                resolved_model = event["data"].get("model", resolved_model)
            elif event.get("type") == "result":
                session_id = event.get("sessionId", session_id)
                premium_requests = event.get("usage", {}).get("premiumRequests")

        if result_text is None:
            raise RuntimeError(f"copilot CLI produced no assistant.message event: {proc.stdout[:500]}")
        return result_text, session_id, resolved_model, premium_requests
