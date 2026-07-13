"""Worker agent definitions. Each worker is a RequirementAgent holding its own
tools and memory; the Investigation Manager (see manager.py) holds no domain
tools at all — only HandoffTools targeting these workers, which structurally
enforces the 'manager only delegates' rule the CrewAI version needed a
hierarchical-process gotcha workaround for."""
from beeai_framework.agents.requirement import RequirementAgent

from bee_bug_hunter.llm import get_chat_model
from bee_bug_hunter.logging_memory import LoggingMemory
from bee_bug_hunter.tools.anomaly_tool import AnomalyCheckTool
from bee_bug_hunter.tools.api_request_tool import ApiRequestFlowTool
from bee_bug_hunter.tools.docker_log_tool import DockerLogCaptureTool
from bee_bug_hunter.tools.mysql_tool import MySQLQueryTool
from bee_bug_hunter.tools.playwright_tool import PlaywrightFlowTool


def build_agents(docker_host: str | None = None, mysql_cfg: dict | None = None) -> dict:
    """docker_host/mysql_cfg are per-flow overrides from flows_manifest.yaml (see
    that file's comments): None means "use the .env default for everything"."""
    llm = get_chat_model()
    mysql_cfg = mysql_cfg or {}

    flow_runner = RequirementAgent(
        llm=llm,
        name="API Flow Runner",
        description=(
            "Executes the target flow -- via Playwright for UI flows, or via a registered "
            "Python/requests API flow for pure JSON API flows -- and reports every "
            "request/response and step failure."
        ),
        role="API Flow Runner",
        instructions=(
            "You drive real user/API flows, following the flow's declared steps exactly. For a UI flow "
            "you use run_playwright_flow to drive a real browser; for a pure JSON API flow (no rendering "
            "involved) you use run_api_flow instead, which calls a registered Python/requests function "
            "directly. You are always told which tool to use and which flow_name to pass. You never skip "
            "steps and you report network activity faithfully — include the full network log and step "
            "results in your answer."
        ),
        tools=[PlaywrightFlowTool(), ApiRequestFlowTool()],
        memory=LoggingMemory(agent_name="API Flow Runner"),
    )

    log_capturer = RequirementAgent(
        llm=llm,
        name="Docker Log Capturer",
        description="Captures container logs for the exact window the flow ran in, so downstream analysis has full context.",
        role="Docker Log Capturer",
        instructions=(
            "You know exactly which containers back this app and capture their logs during a flow run "
            "with capture_docker_logs, without altering timing or interfering with the flow itself. "
            "Include the captured log content per container in your answer."
        ),
        tools=[DockerLogCaptureTool(docker_host=docker_host)],
        memory=LoggingMemory(agent_name="Docker Log Capturer"),
    )

    db_query_agent = RequirementAgent(
        llm=llm,
        name="DB Query Agent",
        description="Finds SQL statements referenced in logs and re-runs equivalent read-only queries to inspect actual data state.",
        role="DB Query Agent",
        instructions=(
            "You read raw application/container logs, spot SQL fragments or ORM-generated queries, "
            "reconstruct a safe read-only SELECT equivalent, and run it with run_mysql_query to see "
            "what the data actually looks like."
        ),
        tools=[MySQLQueryTool(**mysql_cfg)],
        memory=LoggingMemory(agent_name="DB Query Agent"),
    )

    bug_analyzer = RequirementAgent(
        llm=llm,
        name="Bug Analyst",
        description="Synthesizes the flow result, captured logs, and DB query output into a root-cause bug analysis.",
        role="Bug Analyst",
        instructions=(
            "You are a senior engineer who reads flow execution results, container logs, and database "
            "query output together, and produces a precise, evidence-backed root-cause analysis with a "
            "concrete recommended fix — not vague speculation. You may use check_anomalies as a quick "
            "heuristic first pass over raw flow/log output, but you make the final call yourself."
        ),
        tools=[AnomalyCheckTool()],
        memory=LoggingMemory(agent_name="Bug Analyst"),
    )

    sql_performance_agent = RequirementAgent(
        llm=llm,
        name="SQL Performance Agent",
        description="Investigates slow queries flagged during a flow run and recommends concrete performance fixes.",
        role="SQL Performance Agent",
        instructions=(
            "You are a database performance specialist. Given queries observed to run slowly during a "
            "flow, you run EXPLAIN on them with the run_mysql_query tool, look for missing indexes, "
            "full table scans, or N+1 patterns, and recommend a specific fix (index, query rewrite, "
            "caching) backed by the EXPLAIN output — never a generic 'optimize your queries' answer."
        ),
        tools=[MySQLQueryTool(**mysql_cfg)],
        memory=LoggingMemory(agent_name="SQL Performance Agent"),
    )

    return {
        "flow_runner": flow_runner,
        "log_capturer": log_capturer,
        "db_query_agent": db_query_agent,
        "bug_analyzer": bug_analyzer,
        "sql_performance_agent": sql_performance_agent,
    }


# Static role/goal metadata, kept separate from build_agents() so listing the
# roster (e.g. the TUI's home screen) doesn't require a configured LLM provider.
AGENT_SUMMARIES: list[dict[str, str]] = [
    {
        "role": "Investigation Manager",
        "goal": "Decides run order and escalation itself, then delegates to the specialists below.",
    },
    {
        "role": "API Flow Runner",
        "goal": "Executes the target API flow via Playwright and reports every request/response and step failure.",
    },
    {
        "role": "Docker Log Capturer",
        "goal": "Captures container logs for the exact window the flow ran in.",
    },
    {
        "role": "DB Query Agent",
        "goal": "Finds SQL referenced in captured logs and re-runs read-only equivalents to inspect actual data state.",
    },
    {
        "role": "Bug Analyst",
        "goal": "Synthesizes flow/log/DB evidence into a root-cause bug analysis with a concrete recommended fix.",
    },
    {
        "role": "SQL Performance Agent",
        "goal": "Runs EXPLAIN on slow queries and recommends a concrete index/query/caching fix.",
    },
]


def agent_summaries() -> list[dict[str, str]]:
    return AGENT_SUMMARIES
