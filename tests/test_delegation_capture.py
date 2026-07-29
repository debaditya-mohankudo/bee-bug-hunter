"""Tests for delegation_capture.py: CapturingHandoffTool records each
worker's own returned text keyed by run_id, because the manager's final
answer is a synthesized summary, not the workers' raw output.

CapturingHandoffTool._run() is tested directly against a lightweight fake
worker rather than a real RequirementAgent -- HandoffTool._run only needs
(a) context.context["state"]["memory"] to be a BaseMemory and (b) the target
to expose an async .run(messages) returning something with .last_message.text;
it does not require the target to be a BaseAgent or Cloneable."""
from types import SimpleNamespace

import pytest
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools.handoff import HandoffSchema

from bee_bug_hunter import delegation_capture
from bee_bug_hunter.delegation_capture import (
    CapturingHandoffTool,
    Delegation,
    _normalize_role,
    all_for_run,
    clear,
    get_by_role,
)
from bee_bug_hunter.logging_config import new_run_context


class _FakeWorker:
    """Minimal stand-in for a Runnable target: not a BaseAgent, not
    Cloneable, just an async .run(messages) -> object with .last_message.text."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.received_messages = None

    async def run(self, messages):
        self.received_messages = messages
        return SimpleNamespace(last_message=SimpleNamespace(text=self.response_text))


class _FakeContext:
    def __init__(self, memory):
        self.context = {"state": {"memory": memory}}


@pytest.fixture(autouse=True)
def _run_id_context():
    run_id = new_run_context("test-flow")
    yield run_id
    clear(run_id)


async def _make_memory(*messages) -> UnconstrainedMemory:
    memory = UnconstrainedMemory()
    if messages:
        await memory.add_many(messages)
    return memory


class TestNormalizeRole:
    def test_case_insensitive(self):
        assert _normalize_role("Bug Analyst") == _normalize_role("bug analyst")

    def test_underscore_and_space_equivalent(self):
        assert _normalize_role("api_flow_runner") == _normalize_role("API Flow Runner")

    def test_collapses_repeated_whitespace(self):
        assert _normalize_role("Bug   Analyst") == _normalize_role("Bug Analyst")

    def test_strips_quotes(self):
        assert _normalize_role('"Bug Analyst"') == _normalize_role("Bug Analyst")

    def test_empty_string(self):
        assert _normalize_role("") == ""

    def test_none_like_falsy(self):
        assert _normalize_role(None) == ""


class TestCapturingHandoffToolRun:
    @pytest.mark.asyncio
    async def test_records_delegation_and_returns_worker_output(self, _run_id_context):
        worker = _FakeWorker("root cause: passwd column typo")
        tool = CapturingHandoffTool(worker, role="Bug Analyst", name="bug_analyst", description="analyzes bugs")
        memory = await _make_memory(UserMessage("investigate the login flow"))
        ctx = _FakeContext(memory)

        output = await tool._run(HandoffSchema(task="find the root cause"), None, ctx)

        assert output.get_text_content() == "root cause: passwd column typo"
        delegations = all_for_run(_run_id_context)
        assert len(delegations) == 1
        assert delegations[0] == Delegation(
            coworker="Bug Analyst", task="find the root cause", result="root cause: passwd column typo",
        )

    @pytest.mark.asyncio
    async def test_role_defaults_to_tool_name_when_not_given(self, _run_id_context):
        worker = _FakeWorker("done")
        tool = CapturingHandoffTool(worker, name="db_query_agent", description="runs queries")
        memory = await _make_memory()
        ctx = _FakeContext(memory)

        await tool._run(HandoffSchema(task="query the db"), None, ctx)

        delegations = all_for_run(_run_id_context)
        assert delegations[0].coworker == "db_query_agent"

    @pytest.mark.asyncio
    async def test_repeat_delegation_to_same_role_appends_not_overwrites(self, _run_id_context):
        worker = _FakeWorker("first result")
        tool = CapturingHandoffTool(worker, role="Bug Analyst", name="bug_analyst", description="d")
        memory = await _make_memory()
        ctx = _FakeContext(memory)

        await tool._run(HandoffSchema(task="task 1"), None, ctx)
        worker.response_text = "second result"
        await tool._run(HandoffSchema(task="task 2"), None, ctx)

        delegations = all_for_run(_run_id_context)
        assert len(delegations) == 2
        assert [d.result for d in delegations] == ["first result", "second result"]


class TestGetByRole:
    @pytest.mark.asyncio
    async def test_returns_none_when_role_never_delegated_to(self, _run_id_context):
        assert get_by_role(_run_id_context, "Bug Analyst") is None

    @pytest.mark.asyncio
    async def test_returns_single_result(self, _run_id_context):
        worker = _FakeWorker("the finding")
        tool = CapturingHandoffTool(worker, role="Bug Analyst", name="bug_analyst", description="d")
        await tool._run(HandoffSchema(task="t"), None, _FakeContext(await _make_memory()))

        assert get_by_role(_run_id_context, "Bug Analyst") == "the finding"

    @pytest.mark.asyncio
    async def test_role_lookup_is_normalized(self, _run_id_context):
        worker = _FakeWorker("finding")
        tool = CapturingHandoffTool(worker, role="Bug Analyst", name="bug_analyst", description="d")
        await tool._run(HandoffSchema(task="t"), None, _FakeContext(await _make_memory()))

        assert get_by_role(_run_id_context, "bug_analyst") == "finding"
        assert get_by_role(_run_id_context, "BUG   ANALYST") == "finding"

    @pytest.mark.asyncio
    async def test_concatenates_repeat_delegations_with_separator(self, _run_id_context):
        worker = _FakeWorker("first")
        tool = CapturingHandoffTool(worker, role="Bug Analyst", name="bug_analyst", description="d")
        ctx = _FakeContext(await _make_memory())
        await tool._run(HandoffSchema(task="t1"), None, ctx)
        worker.response_text = "second"
        await tool._run(HandoffSchema(task="t2"), None, ctx)

        result = get_by_role(_run_id_context, "Bug Analyst")
        assert result == "first\n\n---\n\nsecond"

    @pytest.mark.asyncio
    async def test_distinguishes_roles(self, _run_id_context):
        bug_worker = _FakeWorker("bug finding")
        perf_worker = _FakeWorker("perf finding")
        bug_tool = CapturingHandoffTool(bug_worker, role="Bug Analyst", name="bug_analyst", description="d")
        perf_tool = CapturingHandoffTool(perf_worker, role="SQL Performance Agent", name="perf_agent", description="d")
        ctx = _FakeContext(await _make_memory())

        await bug_tool._run(HandoffSchema(task="t"), None, ctx)
        await perf_tool._run(HandoffSchema(task="t"), None, ctx)

        assert get_by_role(_run_id_context, "Bug Analyst") == "bug finding"
        assert get_by_role(_run_id_context, "SQL Performance Agent") == "perf finding"


class TestClearAndAllForRun:
    @pytest.mark.asyncio
    async def test_clear_removes_all_delegations_for_run(self, _run_id_context):
        worker = _FakeWorker("finding")
        tool = CapturingHandoffTool(worker, role="Bug Analyst", name="bug_analyst", description="d")
        await tool._run(HandoffSchema(task="t"), None, _FakeContext(await _make_memory()))
        assert all_for_run(_run_id_context) != []

        clear(_run_id_context)

        assert all_for_run(_run_id_context) == []
        assert get_by_role(_run_id_context, "Bug Analyst") is None

    def test_clear_of_unknown_run_id_is_a_no_op(self):
        clear("never-existed")  # must not raise

    def test_all_for_run_returns_empty_list_for_unknown_run_id(self):
        assert all_for_run("never-existed") == []

    @pytest.mark.asyncio
    async def test_captures_are_isolated_per_run_id(self, _run_id_context):
        worker = _FakeWorker("finding for run A")
        tool = CapturingHandoffTool(worker, role="Bug Analyst", name="bug_analyst", description="d")
        await tool._run(HandoffSchema(task="t"), None, _FakeContext(await _make_memory()))

        other_run_id = new_run_context("other-flow")
        try:
            assert get_by_role(other_run_id, "Bug Analyst") is None
            assert get_by_role(_run_id_context, "Bug Analyst") == "finding for run A"
        finally:
            clear(other_run_id)


@pytest.fixture(autouse=True)
def _cleanup_module_state():
    yield
    with delegation_capture._lock:
        delegation_capture._captures.clear()
