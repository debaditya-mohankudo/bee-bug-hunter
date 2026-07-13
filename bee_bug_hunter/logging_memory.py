"""UnconstrainedMemory subclass that logs every message added to or removed
from an agent's context, into the same JSONL stream as everything else.

Without this, "what actually went into an agent's context" (its own reasoning
turns, tool results, and -- for workers -- the conversation HandoffTool copied
in) is only inspectable by attaching a debugger to memory.messages mid-run.
add_many() funnels through add() (see BaseMemory's default impl), so
overriding add/delete/reset alone covers every mutation path.
"""
import logging

from beeai_framework.backend.message import (
    AnyMessage,
    MessageToolCallContent,
    MessageToolResultContent,
)
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

from bee_bug_hunter.logging_config import get_logger, log

logger = get_logger(__name__)


def _role_of(message: AnyMessage) -> str:
    role = getattr(message, "role", "")
    return role.value if hasattr(role, "value") else str(role)


def _preview(message: AnyMessage, limit: int = 200) -> str:
    # message.text only covers plain-text content -- tool calls/results (the
    # most interesting messages to see here) render as empty via .text, so
    # each content chunk is rendered by its actual type instead.
    parts = []
    for c in getattr(message, "content", None) or []:
        if isinstance(c, MessageToolCallContent):
            parts.append(f"tool_call={c.tool_name} args={c.args}")
        elif isinstance(c, MessageToolResultContent):
            parts.append(f"tool_result={c.tool_name}: {c.result}")
        else:
            parts.append(getattr(c, "text", "") or "")
    text = " ".join(" ".join(parts).split())
    return text[:limit] + ("…" if len(text) > limit else "")


class LoggingMemory(UnconstrainedMemory):
    """Same as UnconstrainedMemory, plus a memory_message_added/removed/reset
    log line per mutation, tagged with which agent's memory this is (and, via
    the ambient run_id contextvar, which investigation run)."""

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        await super().add(message, index)
        log(
            logger, logging.INFO, "memory_message_added",
            agent=self.agent_name, role=_role_of(message),
            preview=_preview(message), memory_size=len(self._messages),
        )

    async def delete(self, message: AnyMessage) -> bool:
        removed = await super().delete(message)
        if removed:
            log(
                logger, logging.INFO, "memory_message_removed",
                agent=self.agent_name, role=_role_of(message),
                preview=_preview(message), memory_size=len(self._messages),
            )
        return removed

    def reset(self) -> None:
        cleared = len(self._messages)
        super().reset()
        if cleared:
            log(logger, logging.INFO, "memory_reset", agent=self.agent_name, cleared=cleared)

    async def clone(self) -> "LoggingMemory":
        cloned = LoggingMemory(self.agent_name)
        cloned._messages = self._messages.copy()
        return cloned
