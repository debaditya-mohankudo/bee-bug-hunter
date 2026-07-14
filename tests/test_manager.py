"""Tests for bee_bug_hunter.manager.build_supervisor: handoff wiring, the
prompt's step ordering, and the ConditionalRequirements that structurally
enforce "flow_runner must run before log_capturer" and "log_capturer must
run before db_query_agent/bug_analyst/sql_performance_agent/
source_code_analyst are delegated to." No live LLM call or docker
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
    assert len(supervisor._tools) == 6
    assert all(isinstance(t, CapturingHandoffTool) for t in supervisor._tools)


def test_supervisor_handoff_names(supervisor):
    names = {t.name for t in supervisor._tools}
    assert names == {
        "api_flow_runner",
        "docker_log_capturer",
        "db_query_agent",
        "bug_analyst",
        "sql_performance_agent",
        "source_code_analyst",
    }


def test_supervisor_has_five_ordering_requirements(supervisor):
    assert len(supervisor._requirements) == 5


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


def test_script_flow_kind_selects_run_playwright_script_tool():
    _supervisor, prompt = build_supervisor(
        "demo_login", "demo_app-web-1", 5, flow_kind="script", api_flow_name="example_login_script"
    )
    assert "run_playwright_script" in prompt
    assert "example_login_script" in prompt


def test_known_issue_note_is_prepended_to_prompt():
    _supervisor, prompt = build_supervisor(
        "demo_login", "demo_app-web-1", 5, known_issue_note="passwd vs password column bug in api_login"
    )
    assert prompt.index("passwd vs password column bug") < prompt.index("Investigate flow 'demo_login'")


def test_no_known_issue_note_by_default():
    _supervisor, prompt = build_supervisor("demo_login", "demo_app-web-1", 5)
    assert "earlier in this same run" not in prompt


def test_prompt_requires_summary_line_in_final_answer():
    _supervisor, prompt = build_supervisor("demo_login", "demo_app-web-1", 5)
    assert "SUMMARY:" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handoff_name", ["db_query_agent", "bug_analyst", "sql_performance_agent", "source_code_analyst"]
)
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
    assert not any(r.source.name == "api_flow_runner" for r in supervisor._requirements)


@pytest.mark.asyncio
async def test_log_capturer_blocked_until_flow_runner_ran(supervisor):
    tools = list(supervisor._tools)
    for requirement in supervisor._requirements:
        await requirement.init(tools=tools, ctx=None)

    flow_tool = next(t for t in tools if t.name == "api_flow_runner")
    requirement = next(r for r in supervisor._requirements if r.source.name == "docker_log_capturer")

    rules_before = await requirement.run(_empty_state())
    assert rules_before[0].allowed is False

    rules_after = await requirement.run(_state_after(flow_tool))
    assert rules_after[0].allowed is True
