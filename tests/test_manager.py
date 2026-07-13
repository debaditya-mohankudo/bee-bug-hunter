"""Tests for bee_bug_hunter.manager.build_supervisor: handoff wiring, the
prompt's step ordering, and the ConditionalRequirements that structurally
enforce "log_capturer must run before db_query_agent/bug_analyst/
sql_performance_agent are delegated to." No live LLM call or docker
containers needed.
"""
import pytest
from beeai_framework.agents.requirement.types import RequirementAgentRunState, RequirementAgentRunStateStep
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import StringToolOutput

from bee_bug_hunter.delegation_capture import CapturingHandoffTool
from bee_bug_hunter.manager import build_supervisor


def _empty_state() -> RequirementAgentRunState:
    return RequirementAgentRunState(answer=None, result=None, memory=UnconstrainedMemory(), steps=[], iteration=0)


def _state_after(tool) -> RequirementAgentRunState:
    step = RequirementAgentRunStateStep(
        id="1", iteration=0, input={}, output=StringToolOutput("ok"), tool=tool, error=None
    )
    return RequirementAgentRunState(answer=None, result=None, memory=UnconstrainedMemory(), steps=[step], iteration=1)


@pytest.fixture
def supervisor():
    supervisor, _prompt = build_supervisor("demo_login", "demo_app-web-1", 5)
    return supervisor


def test_supervisor_has_no_domain_tools_only_handoffs(supervisor):
    assert len(supervisor._tools) == 5
    assert all(isinstance(t, CapturingHandoffTool) for t in supervisor._tools)


def test_supervisor_handoff_names(supervisor):
    names = {t.name for t in supervisor._tools}
    assert names == {
        "api_flow_runner",
        "docker_log_capturer",
        "db_query_agent",
        "bug_analyst",
        "sql_performance_agent",
    }


def test_supervisor_has_three_ordering_requirements(supervisor):
    assert len(supervisor._requirements) == 3


def test_prompt_lists_steps_in_order():
    _supervisor, prompt = build_supervisor("demo_login", "demo_app-web-1", 5)
    assert prompt.index("1. Hand off to 'API Flow Runner'") < prompt.index("2. Hand off to 'Docker Log Capturer'")
    assert prompt.index("2. Hand off to 'Docker Log Capturer'") < prompt.index("3. Hand off to 'DB Query Agent'")


def test_api_flow_kind_selects_run_api_flow_tool():
    _supervisor, prompt = build_supervisor(
        "demo_login", "demo_app-web-1", 5, flow_kind="api", api_flow_name="login_flow"
    )
    assert "run_api_flow" in prompt
    assert "login_flow" in prompt


def test_ui_flow_kind_selects_run_playwright_flow_tool():
    _supervisor, prompt = build_supervisor("demo_login", "demo_app-web-1", 5)
    assert "run_playwright_flow" in prompt
    assert "demo_login" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff_name", ["db_query_agent", "bug_analyst", "sql_performance_agent"])
async def test_handoff_blocked_until_log_capturer_ran(supervisor, handoff_name):
    tools = list(supervisor._tools)
    for requirement in supervisor._requirements:
        await requirement.init(tools=tools, ctx=None)

    log_tool = next(t for t in tools if t.name == "docker_log_capturer")
    requirement = next(r for r in supervisor._requirements if r.source.name == handoff_name)

    rules_before = await requirement.run(_empty_state())
    assert rules_before[0].allowed is False

    rules_after = await requirement.run(_state_after(log_tool))
    assert rules_after[0].allowed is True


@pytest.mark.asyncio
async def test_flow_runner_handoff_is_never_blocked_by_ordering(supervisor):
    """api_flow_runner has no ConditionalRequirement targeting it -- it should
    always be runnable, since it's the very first step."""
    tools = list(supervisor._tools)
    for requirement in supervisor._requirements:
        await requirement.init(tools=tools, ctx=None)

    assert not any(r.source.name == "api_flow_runner" for r in supervisor._requirements)
