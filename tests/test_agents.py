"""Tests for bee_bug_hunter.agents.build_agents: worker wiring and the Bug
Analyst's check_anomalies-first requirement. No live LLM call or docker
containers needed -- RequirementAgent construction doesn't touch the network,
and the requirement is exercised directly against a constructed run state.
"""
import pytest
from beeai_framework.agents.requirement.types import RequirementAgentRunState
from beeai_framework.memory import UnconstrainedMemory

from bee_bug_hunter.agents import AGENT_SUMMARIES, agent_summaries, build_agents
from bee_bug_hunter.tools.anomaly_tool import AnomalyCheckTool
from bee_bug_hunter.tools.api_request_tool import ApiRequestFlowTool
from bee_bug_hunter.tools.docker_log_tool import DockerLogCaptureTool
from bee_bug_hunter.tools.mysql_tool import MySQLQueryTool
from bee_bug_hunter.tools.playwright_script_tool import RunPlaywrightScriptTool
from bee_bug_hunter.tools.playwright_tool import PlaywrightFlowTool
from bee_bug_hunter.tools.read_source_tool import ReadSourceFileTool


def _empty_state() -> RequirementAgentRunState:
    return RequirementAgentRunState(answer=None, result=None, memory=UnconstrainedMemory(), steps=[], iteration=0)


@pytest.fixture
def workers():
    return build_agents()


def test_build_agents_returns_all_six_workers(workers):
    assert set(workers) == {
        "flow_runner",
        "log_capturer",
        "db_query_agent",
        "bug_analyzer",
        "sql_performance_agent",
        "source_code_analyst",
    }


def test_flow_runner_has_playwright_and_api_tools(workers):
    tool_types = {type(t) for t in workers["flow_runner"]._tools}
    assert tool_types == {PlaywrightFlowTool, RunPlaywrightScriptTool, ApiRequestFlowTool}


def test_log_capturer_has_docker_log_tool(workers):
    tool_types = {type(t) for t in workers["log_capturer"]._tools}
    assert tool_types == {DockerLogCaptureTool}


def test_log_capturer_honors_docker_host_override():
    workers = build_agents(docker_host="tcp://example:2375")
    docker_tool = workers["log_capturer"]._tools[0]
    assert docker_tool.docker_host == "tcp://example:2375"


def test_db_and_sql_agents_have_mysql_tool(workers):
    assert {type(t) for t in workers["db_query_agent"]._tools} == {MySQLQueryTool}
    assert {type(t) for t in workers["sql_performance_agent"]._tools} == {MySQLQueryTool}


def test_bug_analyzer_has_anomaly_tool_and_one_requirement(workers):
    assert {type(t) for t in workers["bug_analyzer"]._tools} == {AnomalyCheckTool}
    assert len(workers["bug_analyzer"]._requirements) == 1


@pytest.mark.asyncio
async def test_bug_analyzer_must_check_anomalies_at_step_one(workers):
    bug_analyzer = workers["bug_analyzer"]
    tools = list(bug_analyzer._tools)
    requirement = bug_analyzer._requirements[0]
    await requirement.init(tools=tools, ctx=None)

    rules = await requirement.run(_empty_state())
    assert rules[0].forced is True
    assert rules[0].allowed is True


def test_source_code_analyst_has_read_tool(workers):
    tool_types = {type(t) for t in workers["source_code_analyst"]._tools}
    assert tool_types == {ReadSourceFileTool}
    assert workers["source_code_analyst"]._requirements == []


def test_other_workers_have_no_requirements(workers):
    for key in ("flow_runner", "log_capturer", "db_query_agent", "sql_performance_agent", "source_code_analyst"):
        assert workers[key]._requirements == []


def test_agent_summaries_cover_manager_and_all_workers():
    roles = {summary["role"] for summary in agent_summaries()}
    assert roles == {
        "Investigation Manager",
        "API Flow Runner",
        "Docker Log Capturer",
        "DB Query Agent",
        "Bug Analyst",
        "SQL Performance Agent",
        "Source Code Analyst",
    }
    assert agent_summaries() is AGENT_SUMMARIES
