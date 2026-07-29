"""Tests for tool_capture.py: subscribes once to BeeAI's root Emitter and
records the unfiltered StringToolOutput of the flow-runner/log-capture tools
on ToolSuccessEvent, keyed by run_id -- the raw-JSON input anomaly_detector.
detect() needs, independent of what a worker's LLM chose to say about it.

install()/_installed/_captures are process-global (not per-test isolated by
the module itself), so every test here resets that state via an autouse
fixture -- same isolation pattern test_mysql_tool.py uses for its module-
level schema cache."""
import pytest
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool
from pydantic import BaseModel

from bee_bug_hunter import tool_capture
from bee_bug_hunter.logging_config import new_run_context


class _EmptyInput(BaseModel):
    pass


def _make_tool(tool_name: str, output_text: str):
    class _FakeTool(Tool):
        name = tool_name
        description = "test double"
        input_schema = _EmptyInput

        def _create_emitter(self) -> Emitter:
            return Emitter.root().child(namespace=["tool", tool_name], creator=self)

        async def _run(self, input, options, context) -> StringToolOutput:
            return StringToolOutput(output_text)

    return _FakeTool()


@pytest.fixture(autouse=True)
def _isolated_tool_capture_state(monkeypatch):
    """Emitter.root() is a functools.cache'd, genuinely process-wide singleton
    (also subscribed to once at orchestrator.py's import time) -- simply
    resetting tool_capture._installed per test would make install() truly
    re-subscribe onto that SAME immortal root emitter every time, leaking
    permanent duplicate subscriptions across the whole pytest session (not
    just this file), silently double/triple-capturing in unrelated tests
    elsewhere in the suite. Isolate instead: give each test its own fresh
    Emitter root that vanishes when monkeypatch reverts, so install()'s
    subscription never touches the real shared singleton at all."""
    fresh_root = Emitter(creator=object())
    monkeypatch.setattr(Emitter, "root", staticmethod(lambda: fresh_root))
    tool_capture._installed = False
    tool_capture._captures.clear()
    yield
    tool_capture._captures.clear()


@pytest.mark.asyncio
async def test_install_captures_flow_runner_tool_output():
    tool_capture.install()
    run_id = new_run_context("flow-1")
    tool = _make_tool("run_playwright_flow", '{"flow_name": "x", "step_results": []}')

    await tool.run({})

    assert tool_capture.get_flow_raw(run_id) == '{"flow_name": "x", "step_results": []}'
    assert tool_capture.get_log_raw(run_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["run_playwright_flow", "run_playwright_script", "run_api_flow"])
async def test_all_three_flow_runner_tools_share_the_flow_slot(tool_name):
    tool_capture.install()
    run_id = new_run_context("flow-1")
    tool = _make_tool(tool_name, '{"flow_name": "shared"}')

    await tool.run({})

    assert tool_capture.get_flow_raw(run_id) == '{"flow_name": "shared"}'


@pytest.mark.asyncio
async def test_docker_log_tool_captured_in_logs_slot():
    tool_capture.install()
    run_id = new_run_context("flow-1")
    tool = _make_tool("capture_docker_logs", '{"web": {"content": "log lines"}}')

    await tool.run({})

    assert tool_capture.get_log_raw(run_id) == '{"web": {"content": "log lines"}}'
    assert tool_capture.get_flow_raw(run_id) is None


@pytest.mark.asyncio
async def test_untracked_tool_name_is_not_captured():
    tool_capture.install()
    run_id = new_run_context("flow-1")
    tool = _make_tool("run_mysql_query", '{"query": "SELECT 1"}')

    await tool.run({})

    assert tool_capture.get_flow_raw(run_id) is None
    assert tool_capture.get_log_raw(run_id) is None


@pytest.mark.asyncio
async def test_repeat_calls_keep_the_most_recent_output():
    tool_capture.install()
    run_id = new_run_context("flow-1")
    tool = _make_tool("run_playwright_flow", '{"attempt": 1}')
    await tool.run({})
    tool2 = _make_tool("run_playwright_flow", '{"attempt": 2}')
    await tool2.run({})

    assert tool_capture.get_flow_raw(run_id) == '{"attempt": 2}'


@pytest.mark.asyncio
async def test_captures_are_isolated_per_run_id():
    tool_capture.install()
    run_id_a = new_run_context("flow-a")
    tool_a = _make_tool("run_playwright_flow", '{"run": "a"}')
    await tool_a.run({})

    run_id_b = new_run_context("flow-b")
    tool_b = _make_tool("run_playwright_flow", '{"run": "b"}')
    await tool_b.run({})

    assert tool_capture.get_flow_raw(run_id_a) == '{"run": "a"}'
    assert tool_capture.get_flow_raw(run_id_b) == '{"run": "b"}'


def test_get_flow_raw_and_log_raw_return_none_for_unknown_run_id():
    assert tool_capture.get_flow_raw("never-existed") is None
    assert tool_capture.get_log_raw("never-existed") is None


def test_clear_removes_captures_for_run_id():
    tool_capture._captures["run-x"] = {"flow": ["cached"]}
    tool_capture.clear("run-x")
    assert tool_capture.get_flow_raw("run-x") is None


def test_clear_of_unknown_run_id_is_a_no_op():
    tool_capture.clear("never-existed")  # must not raise


@pytest.mark.asyncio
async def test_install_is_idempotent_second_call_does_not_duplicate_subscription():
    tool_capture.install()
    tool_capture.install()  # second call should be a no-op, not a second subscription

    run_id = new_run_context("flow-1")
    tool = _make_tool("run_playwright_flow", '{"n": 1}')
    await tool.run({})

    # If install() had subscribed twice, the single StringToolOutput would
    # still just be appended twice to the same list -- get_flow_raw only
    # returns the last one regardless, so assert on the full capture list
    # length via all captures for this run/slot instead.
    with tool_capture._lock:
        outputs = tool_capture._captures[run_id]["flow"]
    assert len(outputs) == 1
