"""Tests for ThresholdSummarizingMiddleware: replaces LoggingMemory for the
manager, since memory= is only synced from the runner's live UnconstrainedMemory
once, at the very end of a run (agent.py:190-205) -- this middleware instead
hooks the runner's per-iteration "success" event, whose state.memory IS that
live object, by reference. Covers per-iteration logging, threshold-crossing
collapse, and that the middleware ignores events of the wrong type (RunContext's
own generic "success" event shares the event name but a different payload)."""
import logging

import pytest
from beeai_framework.agents.requirement.events import RequirementAgentSuccessEvent
from beeai_framework.agents.requirement.types import RequirementAgentRunState
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from beeai_framework.backend.types import ChatModelOutput
from beeai_framework.emitter import Emitter
from beeai_framework.memory import UnconstrainedMemory

from bee_bug_hunter.summarizing_middleware import ThresholdSummarizingMiddleware, _count_tokens, _full_text


class _FakeCtx:
    def __init__(self) -> None:
        self.emitter = Emitter()


class _FakeSummarizerLLM:
    def __init__(self, summary: str = "summary of the transcript") -> None:
        self.summary = summary
        self.calls: list[list] = []

    async def run(self, messages, **kwargs):
        self.calls.append(messages)
        return ChatModelOutput(output=[AssistantMessage(self.summary)])


def _success_event(memory: UnconstrainedMemory) -> RequirementAgentSuccessEvent:
    state = RequirementAgentRunState(answer=None, result=None, memory=memory, steps=[], iteration=1)
    response = ChatModelOutput(output=[AssistantMessage("ok")])
    return RequirementAgentSuccessEvent(state=state, response=response)


@pytest.mark.asyncio
async def test_new_messages_are_logged_once_per_iteration(caplog):
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager", summarizer_llm=_FakeSummarizerLLM(), summarize_at_tokens=1_000_000
    )
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("hello"))

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.summarizing_middleware"):
        await mw._on_success(_success_event(memory))

    added = [r for r in caplog.records if r.getMessage() == "memory_message_added"]
    assert len(added) == 1
    assert added[0].extra_fields["agent"] == "Investigation Manager"
    assert added[0].extra_fields["role"] == "user"
    assert "hello" in added[0].extra_fields["preview"]
    assert mw._logged_count == 1


@pytest.mark.asyncio
async def test_only_newly_added_messages_are_logged_on_next_iteration(caplog):
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager", summarizer_llm=_FakeSummarizerLLM(), summarize_at_tokens=1_000_000
    )
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("one"))
    await mw._on_success(_success_event(memory))

    await memory.add(AssistantMessage("two"))
    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.summarizing_middleware"):
        await mw._on_success(_success_event(memory))

    added = [r for r in caplog.records if r.getMessage() == "memory_message_added"]
    assert len(added) == 1
    assert "two" in added[0].extra_fields["preview"]
    assert mw._logged_count == 2


@pytest.mark.asyncio
async def test_token_estimate_accumulates_across_iterations():
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager", summarizer_llm=_FakeSummarizerLLM(), summarize_at_tokens=1_000_000
    )
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("hello"))
    await mw._on_success(_success_event(memory))
    first_estimate = mw._token_estimate
    assert first_estimate > 0

    await memory.add(AssistantMessage("world"))
    await mw._on_success(_success_event(memory))
    assert mw._token_estimate > first_estimate


@pytest.mark.asyncio
async def test_crossing_threshold_collapses_memory_to_single_system_message(caplog):
    summarizer = _FakeSummarizerLLM(summary="collapsed summary")
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager", summarizer_llm=summarizer, summarize_at_tokens=1
    )
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("hello"))
    await memory.add(AssistantMessage("world"))

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.summarizing_middleware"):
        await mw._on_success(_success_event(memory))

    assert len(memory.messages) == 1
    assert isinstance(memory.messages[0], SystemMessage)
    assert memory.messages[0].text == "collapsed summary"
    assert summarizer.calls, "summarizer_llm.run should have been called"

    summarized = [r for r in caplog.records if r.getMessage() == "memory_summarized"]
    assert len(summarized) == 1
    assert summarized[0].extra_fields["agent"] == "Investigation Manager"
    assert summarized[0].extra_fields["cleared"] == 2


@pytest.mark.asyncio
async def test_state_after_collapse_reflects_the_summary_only():
    summary = "s"
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager",
        summarizer_llm=_FakeSummarizerLLM(summary),
        # High enough that the short summary + one short follow-up message
        # don't immediately re-trigger a collapse, low enough that one long
        # message alone crosses it.
        summarize_at_tokens=50,
    )
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("hello world " * 50))
    await mw._on_success(_success_event(memory))

    assert mw._logged_count == 1
    assert mw._token_estimate == _count_tokens(summary)
    assert len(memory.messages) == 1

    # Next iteration's new message is logged relative to the collapsed state,
    # not re-logging the summary message itself.
    await memory.add(UserMessage("new question"))
    await mw._on_success(_success_event(memory))
    assert mw._logged_count == 2


@pytest.mark.asyncio
async def test_below_threshold_does_not_call_summarizer():
    summarizer = _FakeSummarizerLLM()
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager", summarizer_llm=summarizer, summarize_at_tokens=1_000_000
    )
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("hello"))
    await mw._on_success(_success_event(memory))

    assert not summarizer.calls
    assert len(memory.messages) == 1


@pytest.mark.asyncio
async def test_bind_resets_counters_and_wires_into_ctx_emitter():
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager", summarizer_llm=_FakeSummarizerLLM(), summarize_at_tokens=1_000_000
    )
    mw._logged_count = 5
    mw._token_estimate = 500

    ctx = _FakeCtx()
    mw.bind(ctx)
    assert mw._logged_count == 0
    assert mw._token_estimate == 0

    memory = UnconstrainedMemory()
    await memory.add(UserMessage("hello"))
    await ctx.emitter.emit("success", _success_event(memory))

    assert mw._logged_count == 1


@pytest.mark.asyncio
async def test_bind_ignores_non_requirement_agent_success_events():
    """RunContext's own generic lifecycle events share the "success" event
    name with RequirementAgentSuccessEvent but carry a different payload type
    -- the handler must not mistake one for the other."""
    mw = ThresholdSummarizingMiddleware(
        agent_name="Investigation Manager", summarizer_llm=_FakeSummarizerLLM(), summarize_at_tokens=1_000_000
    )
    ctx = _FakeCtx()
    mw.bind(ctx)

    await ctx.emitter.emit("success", object())

    assert mw._logged_count == 0
    assert mw._token_estimate == 0


def test_full_text_renders_tool_call_and_tool_result_content():
    call_msg = AssistantMessage([MessageToolCallContent(tool_name="run_mysql_query", args='{"query": "SELECT 1"}', id="1")])
    assert "tool_call=run_mysql_query" in _full_text(call_msg)

    result_msg = ToolMessage([MessageToolResultContent(tool_name="run_mysql_query", tool_call_id="1", result="3 rows")])
    text = _full_text(result_msg)
    assert "tool_result=run_mysql_query" in text
    assert "3 rows" in text


def test_count_tokens_is_positive_for_nonempty_text_and_zero_free_for_empty():
    assert _count_tokens("hello world") > 0
    assert _count_tokens("") >= 0
