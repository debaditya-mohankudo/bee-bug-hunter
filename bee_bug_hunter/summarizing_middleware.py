"""Middleware that logs every message an agent's live memory accumulates and
collapses that memory once its running token estimate crosses a threshold.

Replaces LoggingMemory for both jobs. An agent's memory= constructor param is
only synced from the runner's internal live UnconstrainedMemory at the very
end of a run (agent.py:190-205: reset() + add_many(final_state.memory.messages)
happens once, after the loop exits) -- so a BaseMemory subclass passed as
memory= never sees per-iteration mutations, and both "log every message as it
happens" and "collapse mid-run before the next iteration" are impossible to
build on top of it. RunMiddlewareProtocol.bind(ctx) instead registers on the
runner's own per-iteration "success" event (RequirementAgentSuccessEvent,
_runner.py:262-269), whose state.memory IS the runner's live memory object, by
reference (_runner.py:67-68) -- so logging each new message and, on threshold
crossing, reset()-ing and reseeding that same object are both visible to
every subsequent iteration's LLM call.
"""
import logging

from beeai_framework.agents.requirement.events import RequirementAgentSuccessEvent
from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import (
    AnyMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    SystemMessage,
)
from beeai_framework.context import RunContext, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions

from bee_bug_hunter.logging_config import get_logger, log

logger = get_logger(__name__)

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None

_SUMMARIZE_PROMPT = (
    "Summarize the following agent transcript for continued use as its own context. "
    "Preserve every concrete fact needed to keep investigating -- commands run, files "
    "read, values observed, conclusions reached -- and drop only redundant back-and-forth. "
    "Be concise but do not omit evidence."
)


def _role_of(message: AnyMessage) -> str:
    role = getattr(message, "role", "")
    return role.value if hasattr(role, "value") else str(role)


def _full_text(message: AnyMessage) -> str:
    # Mirrors logging_memory._preview's content-type branching, without the
    # 200-char truncation -- tool calls/results render as empty via .text, so
    # undercounting here would silently defeat the whole mechanism for
    # exactly the tool-call-heavy investigations it exists to protect against.
    parts = []
    for c in getattr(message, "content", None) or []:
        if isinstance(c, MessageToolCallContent):
            parts.append(f"tool_call={c.tool_name} args={c.args}")
        elif isinstance(c, MessageToolResultContent):
            parts.append(f"tool_result={c.tool_name}: {c.result}")
        else:
            parts.append(getattr(c, "text", "") or "")
    return " ".join(" ".join(parts).split())


def _preview(text: str, limit: int = 200) -> str:
    return text[:limit] + ("…" if len(text) > limit else "")


def _count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


class ThresholdSummarizingMiddleware(RunMiddlewareProtocol):
    """Attach via RequirementAgent(..., middlewares=[ThresholdSummarizingMiddleware(...)]).

    Every iteration: logs each message newly added to the runner's live memory
    since the last iteration (a memory_message_added JSONL line per message,
    same shape LoggingMemory used to produce -- see logging_memory.py), and
    tracks a running token estimate. Once that estimate crosses
    summarize_at_tokens, the live memory is collapsed in place to a single
    system message produced by summarizer_llm.
    """

    def __init__(self, agent_name: str, summarizer_llm: ChatModel, summarize_at_tokens: int) -> None:
        self.agent_name = agent_name
        self._summarizer_llm = summarizer_llm
        self._summarize_at_tokens = summarize_at_tokens
        self._logged_count = 0
        self._token_estimate = 0

    def bind(self, ctx: RunContext) -> None:
        self._logged_count = 0
        self._token_estimate = 0

        @ctx.emitter.on(
            "success",
            options=EmitterOptions(is_blocking=True, match_nested=False, priority=-1),
        )
        async def handle_success(data: object, meta: object) -> None:
            if not isinstance(data, RequirementAgentSuccessEvent):
                return
            await self._on_success(data)

    async def _on_success(self, data: RequirementAgentSuccessEvent) -> None:
        memory = data.state.memory
        messages = memory.messages

        for message in messages[self._logged_count :]:
            text = _full_text(message)
            self._token_estimate += _count_tokens(text)
            log(
                logger, logging.INFO, "memory_message_added",
                agent=self.agent_name, role=_role_of(message),
                preview=_preview(text), memory_size=len(messages),
                token_estimate=self._token_estimate,
            )
        self._logged_count = len(messages)

        if self._token_estimate < self._summarize_at_tokens:
            return

        summary = await self._summarize(messages)
        cleared = len(messages)
        memory.reset()
        await memory.add_many([SystemMessage(summary)])
        self._logged_count = 1
        self._token_estimate = _count_tokens(summary)
        log(
            logger, logging.INFO, "memory_summarized",
            agent=self.agent_name, cleared=cleared, token_estimate=self._token_estimate,
        )

    async def _summarize(self, messages: list[AnyMessage]) -> str:
        transcript = "\n".join(f"{_role_of(m)}: {_full_text(m)}" for m in messages)
        response = await self._summarizer_llm.run([SystemMessage(_SUMMARIZE_PROMPT), SystemMessage(transcript)])
        return response.get_text_content()
