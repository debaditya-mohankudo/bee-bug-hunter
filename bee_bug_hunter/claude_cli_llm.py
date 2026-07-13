"""BeeAI-compatible ChatModel that shells out to `claude -p` instead of an API/SDK.

Reuses the caller's existing Claude Code OAuth login (no ANTHROPIC_API_KEY needed).
Since `--tools none` disables the CLI's own tool use, tool calling is bridged by
hand: tool schemas are described in the system prompt, the model is asked to reply
with a small JSON protocol ({"tool": ..., "args": {...}} or {"final_answer": ...}),
and _create() translates a tool-call reply into a native MessageToolCallContent so
BeeAI's own agent loop executes the tool and feeds the result back. Unlike the
CrewAI port this was adapted from, no hand-rolled tool loop is needed here — BeeAI
drives the loop; this backend only translates one reasoning step at a time.

Session-based prompt caching, root+fork topology: one shared "root" `claude -p`
session per flow investigation holds only generic, role-agnostic framing (flow
name, containers -- no tool schemas). Each role (worker or manager) forks its own
session off that root on its first call (`claude -p --resume <root_id>
--fork-session`), then reuses its own forked session (`--resume <its_id>`) on
every later call, sending only the messages appended since the previous call.

This is deliberately not one shared session across every role: each role's
system prompt carries a *different* tool schema (_describe_tools(input.tools)),
so mixing roles into one session would leave stale tool schemas from other roles
sitting in the history, confusing the model about which tools are actually
callable this turn. Forking from a common root instead gives every role the same
shared starting context (cache-friendly, and a true analogue of organs sharing
one nervous system) while keeping each role's own tool-specific continuation
isolated from the others (see ClaudeCLIChatModel.for_role()'s docstring).

A prior version called `claude -p` fresh every step with the entire conversation
re-sent as one new prompt each time -- a brand-new sessionless subprocess every
call has no identical/growing prefix for Anthropic's prompt cache to hit against,
so every call was cold.
"""
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
)
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.backend.utils import parse_broken_json

from bee_bug_hunter.config import DEFAULT_CLAUDE_CLI_SESSION_STORE
from bee_bug_hunter.logging_config import get_logger, log

logger = get_logger(__name__)

_session_store_lock = threading.Lock()


def _session_store_path() -> Path:
    return Path(os.getenv("CLAUDE_CLI_SESSION_STORE", DEFAULT_CLAUDE_CLI_SESSION_STORE))


def _load_persisted_sessions() -> dict[str, dict]:
    """{flow_name: {"root": session_id, "roles": {role: session_id}}}, persisted to
    disk so sessions survive process restarts (a crash, or a fresh --once
    invocation) instead of living only in ClaudeCLIChatModel._instances, an
    in-memory dict that resets every run."""
    path = _session_store_path()
    if not path.exists():
        return {}
    try:
        with _session_store_lock, path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(logger, logging.WARNING, "claude_cli_session_store_read_failed", path=str(path), error=str(e))
        return {}


def _persist_root_session(flow_key: str, session_id: str) -> None:
    _update_session_store(lambda sessions: sessions.setdefault(flow_key, {}).update(root=session_id))


def _persist_role_session(flow_key: str, role_key: str, session_id: str) -> None:
    _update_session_store(
        lambda sessions: sessions.setdefault(flow_key, {}).setdefault("roles", {}).update({role_key: session_id})
    )


def _update_session_store(mutate) -> None:
    path = _session_store_path()
    with _session_store_lock:
        try:
            sessions = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            sessions = {}
        mutate(sessions)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sessions, indent=2))


def clear_persisted_sessions() -> None:
    """Wipes the on-disk session store (and any in-memory ClaudeCLIChatModel
    instances) so every process invocation starts with fresh root+role
    sessions. A killed/crashed run leaves a persisted session whose CLI-side
    history can be ahead of the BeeAI-side RequirementAgentRunState (which is
    always in-memory and starts empty) -- e.g. the manager's session
    "remembers" a completed flow_runner call from an aborted prior run, tries
    to skip straight to log_capturer, and gets structurally rejected by
    ConditionalRequirement without a clean way to recover. Called at the top
    of every orchestrator.run_batch_once iteration (covers both a fresh
    --once process and every poll cycle of a long-running monitor_loop)
    rather than trying to reconcile the two on the fly."""
    path = _session_store_path()
    with _session_store_lock:
        path.unlink(missing_ok=True)
    ClaudeCLIChatModel._instances.clear()


_ROOT_SESSION_PROMPT = (
    "You are one of several specialized agents collaborating on an automated bug-hunting "
    "investigation of the '{flow_name}' flow (containers: {containers}). You will each be given "
    "your own specific role and task separately -- this message is only shared background context, "
    "not a task. Acknowledge briefly."
)


def _ensure_root_session(flow_key: str, flow_name: str, containers: str, model: str) -> str:
    """Creates (once) the shared root session for a flow investigation -- a
    generic, role-agnostic framing (no tool schemas) that every role forks its
    own session from (see ClaudeCLIChatModel.for_role()). A raw standalone `claude
    -p` call rather than a ChatModel instance, since it isn't itself a worker/
    manager role and has no BeeAI-side conversation of its own."""
    persisted = _load_persisted_sessions()
    existing = persisted.get(flow_key, {}).get("root")
    if existing:
        return existing

    claude_path = shutil.which("claude")
    if not claude_path:
        raise RuntimeError("claude_cli provider: `claude` binary not found on PATH")
    root_id = str(uuid.uuid4())
    prompt = _ROOT_SESSION_PROMPT.format(flow_name=flow_name, containers=containers)
    proc = subprocess.run(
        [claude_path, "-p", "--safe-mode", "--output-format", "json", "--tools", "none",
         "--model", model, "--session-id", root_id],
        input=prompt, capture_output=True, text=True, timeout=240,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exited {proc.returncode} creating root session: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude CLI reported an error creating root session: {data.get('result')}")

    _persist_root_session(flow_key, root_id)
    log(logger, logging.INFO, "claude_cli_root_session_created", flow=flow_name, session_id=root_id)
    return root_id

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_TOOL_PROTOCOL_INSTRUCTIONS = """
You have access to tools. To call one, respond with ONLY this JSON (no other text):
{{"tool": "<tool_name>", "args": {{...}}}}

When you have the final answer and don't need any more tools, respond with ONLY this JSON:
{{"final_answer": "<your answer>"}}

Available tools:
{tool_descriptions}
""".strip()

# BeeAI sometimes forces ChatModelInput.tool_choice ("required", or a specific Tool)
# to guarantee the agent takes an action this step. Left unhandled, the model often
# just answers in prose since the base protocol above always offers a final_answer
# escape hatch -- BeeAI's own Retryable then catches the resulting ChatModelToolCallError
# and re-asks with a corrective nudge, which works but costs a wasted `claude -p`
# subprocess call (real latency) per miss. Stating the constraint up front avoids that.
#
# When BeeAI says "required" it usually *includes* its final_answer tool in the
# allowed list (final_answer_as_tool forces the final answer to arrive as a tool
# call too), so "required" does not mean "you may not finish" -- it means "your
# reply must be a tool call, and finishing is done by calling the final_answer
# tool". A previous version of this instruction forbade the final_answer format
# outright, which punished the model for legitimately being done: raw_preview
# logging showed the misses were well-formed {"final_answer": ...} replies, i.e.
# the model wanting to finish and the protocol giving it no sanctioned way to say
# so. Hence two variants, picked by whether final_answer is actually allowed.
_REQUIRED_TOOL_CHOICE_INSTRUCTIONS = """
IMPORTANT: A tool call is required this turn. You MUST respond with ONLY the
{{"tool": "<tool_name>", "args": {{...}}}} JSON for one of the tools listed above.
Do NOT respond with plain text and do NOT use the final_answer format this turn.
""".strip()

_REQUIRED_TOOL_CHOICE_WITH_FINAL_ANSWER_INSTRUCTIONS = """
IMPORTANT: A tool call is required this turn. You MUST respond with ONLY the
{{"tool": "<tool_name>", "args": {{...}}}} JSON for one of the tools listed above.
Do NOT respond with plain text. If you are done and want to give your final answer,
do it AS A TOOL CALL: {{"tool": "final_answer", "args": {{"response": "<your answer>"}}}}
""".strip()

_FORCED_TOOL_INSTRUCTIONS = """
IMPORTANT: You MUST call the '{tool_name}' tool this turn. Respond with ONLY this JSON
(no other text): {{"tool": "{tool_name}", "args": {{...}}}}
""".strip()


def _find_balanced_json_objects(text: str) -> list[str]:
    """Scans for every top-level {...} span via brace-depth counting (string/escape
    aware), instead of a regex spanning first-'{' to last-'}' -- that greedy-regex
    approach swallows the whole response into one unparseable blob whenever the
    model's surrounding prose happens to quote braces of its own (e.g. an HTTP
    error body), which was silently producing 'no tool call' failures."""
    spans = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start:i + 1])
    return spans


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _CODE_FENCE_PATTERN.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Several brace spans can be in play at once -- the real tool-call JSON
    # plus prose that happens to quote its own (possibly also-valid-JSON)
    # braces, e.g. an HTTP error body. Parsing "first span that merely
    # parses" is not enough: prose braces are often valid JSON in their own
    # right (see test_ignores_leading_prose_that_quotes_braces). Require the
    # object to actually look like our protocol (has a "tool" or
    # "final_answer" key) before accepting it; only fall back to "first
    # parseable" if nothing matches the protocol shape.
    parsed_candidates = []
    for candidate in _find_balanced_json_objects(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("tool" in parsed or "final_answer" in parsed):
            return parsed
        parsed_candidates.append(parsed)
    if parsed_candidates:
        return parsed_candidates[0]

    # Last resort: genuinely malformed JSON (not just extra surrounding prose) --
    # e.g. a long markdown-heavy final_answer response with an unescaped quote or
    # raw newline breaking strict json.loads on every balanced-brace candidate
    # above. Observed in practice on the Bug Analyst's longer responses. Use
    # beeai_framework's own repair (json_repair, the same library its runner uses
    # for the identical problem) rather than hand-rolling another parser.
    repaired = parse_broken_json(text, fallback=None)
    if isinstance(repaired, dict) and ("tool" in repaired or "final_answer" in repaired):
        return repaired
    return None


def _describe_tools(tools) -> str:
    if not tools:
        return "(none)"
    lines = []
    for t in tools:
        try:
            schema = json.dumps(t.input_schema.model_json_schema())
        except Exception:
            schema = "{}"
        lines.append(f"- {t.name}: {t.description}\n  parameters schema: {schema}")
    return "\n".join(lines)


def _flatten_messages(messages) -> tuple[str, str]:
    """Returns (system_prompt, conversation_prompt) — the CLI takes one system
    string and one user string per invocation, so the given slice of message
    history is flattened into role-prefixed text. Callers pass only the messages
    new since the last invocation once a session is underway (see
    ClaudeCLIChatModel._create), not the full history every time."""
    system_parts = []
    convo_parts = []
    for m in messages:
        role = getattr(m, "role", "user")
        role = role.value if hasattr(role, "value") else str(role)
        if role == "system":
            system_parts.append(m.text)
            continue
        chunks = []
        for c in m.content:
            if isinstance(c, MessageToolResultContent):
                chunks.append(f"Tool '{c.tool_name}' result:\n{c.result}")
            elif isinstance(c, MessageToolCallContent):
                chunks.append(f'{{"tool": "{c.tool_name}", "args": {c.args}}}')
            else:
                chunks.append(getattr(c, "text", ""))
        convo_parts.append(f"{role.upper()}: " + "\n".join(chunks))
    return "\n\n".join(system_parts), "\n\n".join(convo_parts)


class ClaudeCLIChatModel(ChatModel):
    """Shells to `claude -p --safe-mode --tools none` per reasoning step. Each role
    (worker or manager) forks its own persistent CLI session off a shared per-flow
    root session on first use (see for_role()), then reuses that fork on every
    later turn -- see the module docstring for the full root+fork rationale.

    No API key needed — reuses the caller's Claude Code login. No native
    streaming; _create_stream just yields the full _create result once.
    """

    # (flow_key, role) -> singleton instance, so each worker/manager role gets its
    # own persistent, forked `claude -p` session (see for_role()) instead of a
    # fresh cold one per delegation. Keyed by role (not global) so workers stay
    # isolated from each other's sessions -- see for_role()'s docstring.
    _instances: dict[tuple[str, str], "ClaudeCLIChatModel"] = {}

    @classmethod
    def for_role(
        cls, role: str | None, *, model: str, flow_name: str, containers: str = "", **kwargs: Any,
    ) -> "ClaudeCLIChatModel":
        """Returns the singleton ClaudeCLIChatModel for this (flow, role) pair (e.g.
        flow_name="example_login_api", role="Bug Analyst"), constructing it on
        first use. A single shared instance across all roles would mix every
        worker's conversation into one Claude session, defeating the "private
        per-agent memory" design (see manager.py's module docstring) -- one
        instance per role keeps that isolation while still letting repeat
        delegations to the *same* role resume their one ongoing session (see
        clone()) rather than starting cold each time. Lives for the process's
        lifetime, so in continuous `monitor_loop` mode a role's session keeps
        growing across poll cycles too -- more cache reuse, but unbounded growth
        is an accepted known limitation for now (no periodic reset yet).

        The session_id itself must never be lost even across process restarts (a
        crash, or a fresh --once invocation starting a brand-new _instances dict)
        -- _instances alone is in-memory only. So construction here always checks
        the on-disk store (see _load_persisted_sessions) for this (flow, role)
        pair first; a known id is reused (always --resume, since it already
        exists remotely) rather than minting -- and losing -- a fresh one every
        process."""
        flow_key = flow_name or "default"
        role_key = role or "default"
        cache_key = (flow_key, role_key)
        if cache_key not in cls._instances:
            root_id = _ensure_root_session(flow_key, flow_name or flow_key, containers, model)
            persisted = _load_persisted_sessions()
            existing_role_session = persisted.get(flow_key, {}).get("roles", {}).get(role_key)
            cls._instances[cache_key] = cls(
                model=model, flow_key=flow_key, role_key=role_key,
                session_id=existing_role_session, fork_from=root_id, **kwargs,
            )
        return cls._instances[cache_key]

    def __init__(
        self, model: str = "sonnet", *, flow_key: str = "default", role_key: str = "default",
        session_id: str | None = None, fork_from: str | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        claude_path = shutil.which("claude")
        if not claude_path:
            raise RuntimeError("claude_cli provider: `claude` binary not found on PATH")
        self._claude_path = claude_path
        self._flow_key = flow_key
        self._role_key = role_key
        self._sent_message_count = 0
        if session_id:
            # This role already forked its own session in a prior process run --
            # always resume it, never re-fork, so we never lose/orphan it.
            self._session_id = session_id
            self._session_started = True
            self._fork_from = None
        else:
            # Not yet forked: the real session_id isn't known until the first
            # `--resume <fork_from> --fork-session` call returns one (see
            # _create()/_invoke_cli()), which is also when it gets persisted.
            self._session_id = None
            self._session_started = False
            self._fork_from = fork_from

    async def clone(self) -> "ClaudeCLIChatModel":
        # Deliberately returns self rather than a fresh instance: HandoffTool clones
        # the target agent (and its llm, via RequirementAgent.clone()'s
        # `llm=await self._llm.clone()`) on every single delegation, which would
        # otherwise start a brand-new cold `claude -p` session per delegation --
        # even repeat delegations to the same worker role. get_chat_model()'s
        # per-(flow, role) singleton cache (llm.py/for_role()) is what keeps this
        # safe: each role gets its own ClaudeCLIChatModel instance, so returning
        # self here means "resume this role's one ongoing session," not "share a
        # session across roles." Only correct because delegations are sequential
        # (allow_parallel_tool_calls is False for every provider, see llm.py) --
        # concurrent delegations to the same role would race on
        # _sent_message_count/_session_started.
        return self

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_id(self) -> str:
        return "claude_cli"

    async def _create(self, input: ChatModelInput, run) -> ChatModelOutput:
        # Only flatten what's new since the last call in this session -- the CLI
        # session already holds everything sent on prior turns, and re-sending the
        # full history both wastes tokens and defeats prompt-cache prefix matching.
        new_messages = input.messages[self._sent_message_count:] if self._session_started else input.messages
        system_prompt, prompt = _flatten_messages(new_messages)
        if input.tools:
            system_prompt = (
                system_prompt + "\n\n" +
                _TOOL_PROTOCOL_INSTRUCTIONS.format(tool_descriptions=_describe_tools(input.tools))
            )

        # "required" usually still allows finishing -- via the final_answer *tool*
        # (see the instruction constants above). Only forbid the final_answer
        # format outright when that tool genuinely isn't on the menu this turn.
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
        raw, returned_session_id = await asyncio.to_thread(self._invoke_cli, system_prompt, prompt)
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        if self._session_id is None:
            # First call for this role was the fork-from-root; adopt and persist
            # the session_id the CLI assigned to it so every later call (this
            # process and any future one) resumes it correctly.
            self._session_id = returned_session_id
            _persist_role_session(self._flow_key, self._role_key, self._session_id)
        self._session_started = True
        self._sent_message_count = len(input.messages)
        parsed = _extract_json_object(raw)

        # The model replied {"final_answer": ...} on a turn that demanded a tool
        # call, but final_answer *is* an allowed tool this turn -- the reply is a
        # legitimate "I'm done", just in the wrong wrapper. Convert it locally into
        # the equivalent final_answer tool call instead of failing the whole call
        # and paying for a retry: raw_preview logging showed this exact shape was
        # the dominant missed_required_tool_call failure mode (well-formed JSON,
        # premature-looking but valid final answer -- never corrupt syntax).
        if (
            parsed and "tool" not in parsed and "final_answer" in parsed
            and tool_choice == "required" and final_answer_allowed
        ):
            answer = parsed["final_answer"]
            if not isinstance(answer, str):
                answer = json.dumps(answer)
            parsed = {"tool": "final_answer", "args": {"response": answer}}
            log(
                logger, logging.INFO, "claude_cli_final_answer_converted",
                model=self._model, session_id=self._session_id, flow=self._flow_key,
            )

        if parsed and "tool" in parsed and input.tools:
            log(
                logger, logging.INFO, "claude_cli_call",
                model=self._model, tool_choice=tool_choice_str, elapsed_ms=elapsed_ms,
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

        # A required/forced tool_choice with no tool call in the parsed response means
        # this call is about to fail beeai_framework.backend.chat._assert_tool_response's
        # check downstream -- log it here as the wasted-`claude -p`-call signal, since
        # that raise site logs an ERROR but doesn't identify which run/agent/step it was.
        outcome = "text"
        if tool_choice_str != "auto" and tool_choice_str != "None" and (not parsed or "tool" not in parsed):
            outcome = "missed_required_tool_call"
        log(
            logger, logging.INFO if outcome == "text" else logging.WARNING, "claude_cli_call",
            model=self._model, tool_choice=tool_choice_str, elapsed_ms=elapsed_ms, outcome=outcome,
            session_id=self._session_id, new_session=is_new_session, flow=self._flow_key,
            # Only on the failure path -- distinguishes "model never attempted a
            # tool call" (raw has no {"tool": ...} at all, pure prose) from
            # "attempted but unparseable" (raw has a JSON-*looking* fragment that
            # _extract_json_object still couldn't parse, e.g. truncated/malformed).
            # Without this we can only speculate about which failure mode we're
            # actually hitting.
            **({"raw_preview": raw[:500]} if outcome == "missed_required_tool_call" else {}),
        )

        text = parsed["final_answer"] if parsed and "final_answer" in parsed else raw
        if not isinstance(text, str):
            text = json.dumps(text)
        return ChatModelOutput(output=[AssistantMessage(text)], finish_reason="stop")

    async def _create_stream(self, input: ChatModelInput, run) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)

    def _invoke_cli(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        """Returns (result_text, session_id). Three possible modes, checked in this
        order: (1) first-ever call for this role -> fork off the shared root
        (--resume <root> --fork-session), letting the CLI assign the new
        session_id, which _create() then adopts and persists; (2) resuming a
        session already known (from this process or a persisted prior one)
        (--resume <id>); (3) no fork_from and no session_id -- only reachable via
        direct construction outside for_role() (e.g. tests) -- starts a plain
        standalone session (--session-id <fresh uuid>), matching the old
        no-forking behavior."""
        cmd = [
            self._claude_path, "-p",
            "--safe-mode",
            "--output-format", "json",
            "--tools", "none",
            "--model", self._model,
        ]
        if self._session_id is None and self._fork_from:
            cmd += ["--resume", self._fork_from, "--fork-session"]
        elif self._session_id is not None:
            cmd += ["--resume", self._session_id]
        else:
            self._session_id = str(uuid.uuid4())
            cmd += ["--session-id", self._session_id]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]

        proc = subprocess.run(
            cmd, input=user_prompt, capture_output=True, text=True, timeout=240,
        )
        if proc.returncode != 0:
            # API-level failures (rate limits, auth, etc.) land in stdout's JSON
            # `result` field with is_error=true and an empty stderr -- surface
            # whichever actually has content instead of always showing stderr.
            detail = proc.stderr.strip()
            if not detail:
                try:
                    detail = json.loads(proc.stdout).get("result", proc.stdout[:500])
                except json.JSONDecodeError:
                    detail = proc.stdout[:500]
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {detail}")

        data = json.loads(proc.stdout)
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI reported an error: {data.get('result')}")
        return data["result"], data.get("session_id", self._session_id)
