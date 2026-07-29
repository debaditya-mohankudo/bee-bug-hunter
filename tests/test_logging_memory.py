"""Tests for LoggingMemory: an UnconstrainedMemory subclass that logs every
message added/removed/reset into the JSONL stream, tagged with which agent's
memory this is. add_many() funnels through add() (BaseMemory's default impl),
so overriding add/delete/reset alone covers every mutation path."""
import logging

import pytest
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    ToolMessage,
    UserMessage,
)

from bee_bug_hunter.logging_memory import LoggingMemory


@pytest.mark.asyncio
async def test_add_logs_memory_message_added(caplog):
    memory = LoggingMemory(agent_name="Bug Analyst")
    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        await memory.add(UserMessage("hello"))

    records = [r for r in caplog.records if r.getMessage() == "memory_message_added"]
    assert len(records) == 1
    assert records[0].extra_fields["agent"] == "Bug Analyst"
    assert records[0].extra_fields["role"] == "user"
    assert "hello" in records[0].extra_fields["preview"]
    assert records[0].extra_fields["memory_size"] == 1


@pytest.mark.asyncio
async def test_add_many_funnels_through_add_and_logs_each(caplog):
    memory = LoggingMemory(agent_name="DB Query Agent")
    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        await memory.add_many([UserMessage("one"), AssistantMessage("two")])

    added = [r for r in caplog.records if r.getMessage() == "memory_message_added"]
    assert len(added) == 2
    assert added[0].extra_fields["memory_size"] == 1
    assert added[1].extra_fields["memory_size"] == 2


@pytest.mark.asyncio
async def test_delete_logs_memory_message_removed(caplog):
    memory = LoggingMemory(agent_name="SQL Performance Agent")
    msg = UserMessage("to be removed")
    await memory.add(msg)

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        removed = await memory.delete(msg)

    assert removed is True
    records = [r for r in caplog.records if r.getMessage() == "memory_message_removed"]
    assert len(records) == 1
    assert records[0].extra_fields["agent"] == "SQL Performance Agent"
    assert records[0].extra_fields["memory_size"] == 0


@pytest.mark.asyncio
async def test_delete_of_absent_message_does_not_log(caplog):
    memory = LoggingMemory(agent_name="Docker Log Capturer")
    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        removed = await memory.delete(UserMessage("never added"))

    assert removed is False
    assert not [r for r in caplog.records if r.getMessage() == "memory_message_removed"]


@pytest.mark.asyncio
async def test_reset_logs_memory_reset_with_cleared_count(caplog):
    memory = LoggingMemory(agent_name="API Flow Runner")
    await memory.add(UserMessage("one"))
    await memory.add(UserMessage("two"))

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        memory.reset()

    records = [r for r in caplog.records if r.getMessage() == "memory_reset"]
    assert len(records) == 1
    assert records[0].extra_fields["agent"] == "API Flow Runner"
    assert records[0].extra_fields["cleared"] == 2
    assert memory.messages == []


@pytest.mark.asyncio
async def test_reset_on_empty_memory_does_not_log(caplog):
    memory = LoggingMemory(agent_name="Source Code Analyst")
    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        memory.reset()

    assert not [r for r in caplog.records if r.getMessage() == "memory_reset"]


@pytest.mark.asyncio
async def test_clone_preserves_agent_name_and_messages():
    memory = LoggingMemory(agent_name="Bug Analyst")
    await memory.add(UserMessage("hello"))

    cloned = await memory.clone()

    assert isinstance(cloned, LoggingMemory)
    assert cloned.agent_name == "Bug Analyst"
    assert len(cloned.messages) == 1
    # Must be a copy, not the same list object, or mutating one affects the other.
    await cloned.add(UserMessage("only in clone"))
    assert len(memory.messages) == 1
    assert len(cloned.messages) == 2


@pytest.mark.asyncio
async def test_preview_renders_tool_call_content_in_log(caplog):
    memory = LoggingMemory(agent_name="DB Query Agent")
    msg = AssistantMessage([MessageToolCallContent(tool_name="run_mysql_query", args='{"query": "SELECT 1"}', id="1")])

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        await memory.add(msg)

    added = [r for r in caplog.records if r.getMessage() == "memory_message_added"][0]
    assert "tool_call=run_mysql_query" in added.extra_fields["preview"]


@pytest.mark.asyncio
async def test_preview_renders_tool_result_content_in_log(caplog):
    memory = LoggingMemory(agent_name="DB Query Agent")
    msg = ToolMessage([MessageToolResultContent(tool_name="run_mysql_query", tool_call_id="1", result="3 rows")])

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        await memory.add(msg)

    added = [r for r in caplog.records if r.getMessage() == "memory_message_added"][0]
    assert "tool_result=run_mysql_query" in added.extra_fields["preview"]
    assert "3 rows" in added.extra_fields["preview"]


@pytest.mark.asyncio
async def test_preview_truncates_long_text(caplog):
    memory = LoggingMemory(agent_name="Bug Analyst")
    long_text = "x" * 500

    with caplog.at_level(logging.INFO, logger="bee_bug_hunter.logging_memory"):
        await memory.add(UserMessage(long_text))

    added = [r for r in caplog.records if r.getMessage() == "memory_message_added"][0]
    assert len(added.extra_fields["preview"]) <= 201  # 200 chars + ellipsis
    assert added.extra_fields["preview"].endswith("…")
