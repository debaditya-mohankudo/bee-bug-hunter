"""BeeAI-compatible ChatModel that shells out to `claude -p` instead of an API/SDK.

Reuses the caller's existing Claude Code OAuth login (no ANTHROPIC_API_KEY needed).
Since `--tools none` disables the CLI's own tool use, tool calling is bridged by
hand: tool schemas are described in the system prompt, the model is asked to reply
with a small JSON protocol ({"tool": ..., "args": {...}} or {"final_answer": ...}),
and _create() translates a tool-call reply into a native MessageToolCallContent so
BeeAI's own agent loop executes the tool and feeds the result back. Unlike the
CrewAI port this was adapted from, no hand-rolled tool loop is needed here — BeeAI
drives the loop; this backend only translates one reasoning step at a time.
"""
import asyncio
import json
import re
import shutil
import subprocess
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
)
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput

_JSON_OBJECT_PATTERN = re.compile(r"\{.*}", re.DOTALL)

_TOOL_PROTOCOL_INSTRUCTIONS = """
You have access to tools. To call one, respond with ONLY this JSON (no other text):
{{"tool": "<tool_name>", "args": {{...}}}}

When you have the final answer and don't need any more tools, respond with ONLY this JSON:
{{"final_answer": "<your answer>"}}

Available tools:
{tool_descriptions}
""".strip()


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
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
    string and one user string per invocation, so the whole message history is
    flattened into role-prefixed text."""
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
    """Shells to `claude -p --safe-mode --tools none` per reasoning step.

    No API key needed — reuses the caller's Claude Code login. No native
    streaming; _create_stream just yields the full _create result once.
    """

    def __init__(self, model: str = "sonnet", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = model
        claude_path = shutil.which("claude")
        if not claude_path:
            raise RuntimeError("claude_cli provider: `claude` binary not found on PATH")
        self._claude_path = claude_path

    async def clone(self) -> "ClaudeCLIChatModel":
        # HandoffTool clones the target agent (and its llm) per delegation;
        # this backend is stateless apart from the model name, so a fresh
        # instance is a faithful clone.
        return ClaudeCLIChatModel(model=self._model)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_id(self) -> str:
        return "claude_cli"

    async def _create(self, input: ChatModelInput, run) -> ChatModelOutput:
        system_prompt, prompt = _flatten_messages(input.messages)
        if input.tools:
            system_prompt = (
                system_prompt + "\n\n" +
                _TOOL_PROTOCOL_INSTRUCTIONS.format(tool_descriptions=_describe_tools(input.tools))
            )

        raw = await asyncio.to_thread(self._invoke_cli, system_prompt, prompt)
        parsed = _extract_json_object(raw)

        if parsed and "tool" in parsed and input.tools:
            content = MessageToolCallContent(
                id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name=parsed["tool"],
                args=json.dumps(parsed.get("args", {})),
            )
            return ChatModelOutput(
                output=[AssistantMessage(content)], finish_reason="tool_calls",
            )

        text = parsed["final_answer"] if parsed and "final_answer" in parsed else raw
        if not isinstance(text, str):
            text = json.dumps(text)
        return ChatModelOutput(output=[AssistantMessage(text)], finish_reason="stop")

    async def _create_stream(self, input: ChatModelInput, run) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)

    def _invoke_cli(self, system_prompt: str, user_prompt: str) -> str:
        cmd = [
            self._claude_path, "-p",
            "--safe-mode",
            "--output-format", "json",
            "--tools", "none",
            "--model", self._model,
        ]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]

        proc = subprocess.run(
            cmd, input=user_prompt, capture_output=True, text=True, timeout=240,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:500]}")

        data = json.loads(proc.stdout)
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI reported an error: {data.get('result')}")
        return data["result"]
