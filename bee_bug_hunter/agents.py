"""Worker agent definitions. Each worker is a RequirementAgent holding its own
tools and memory; the Investigation Manager (see manager.py) holds no domain
tools at all — only HandoffTools targeting these workers, which structurally
enforces the 'manager only delegates' rule the CrewAI version needed a
hierarchical-process gotcha workaround for."""
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement

from bee_bug_hunter.llm import get_chat_model
from bee_bug_hunter.logging_memory import LoggingMemory
from bee_bug_hunter.tools.anomaly_tool import AnomalyCheckTool
from bee_bug_hunter.tools.api_request_tool import ApiRequestFlowTool
from bee_bug_hunter.tools.docker_log_tool import DockerLogCaptureTool
from bee_bug_hunter.tools.mysql_tool import MySQLQueryTool
from bee_bug_hunter.tools.playwright_script_tool import RunPlaywrightScriptTool
from bee_bug_hunter.tools.playwright_tool import PlaywrightFlowTool
from bee_bug_hunter.tools.read_source_tool import ReadSourceFileTool


def build_agents(
    docker_host: str | None = None, mysql_cfg: dict | None = None,
    flow_name: str = "default", containers: str = "",
) -> dict:
    """docker_host/mysql_cfg are per-flow overrides from manifest.yaml (see
    that file's comments): None means "use the .env default for everything".
    flow_name/containers are only meaningful for the CLI-shelling providers
    (claude_cli's per-flow root+fork session topology, see claude_cli_llm.py;
    copilot_cli's per-(flow, role) session, see copilot_cli_llm.py -- containers
    is unused there since it has no shared-root session to seed); every other
    provider ignores them. MySQLQueryTool's EXPLAIN cache is module-level (see
    mysql_tool.py), not threaded through here -- every instance built below
    shares it automatically."""
    # A separate get_chat_model(role=...) call per worker rather than one shared llm:
    # for the CLI-shelling providers this gives each role its own session singleton
    # (see llm.py), keeping workers isolated from each other instead of bleeding
    # into one shared session. Other providers ignore `role` and just construct fresh each
    # call, same as before.
    mysql_cfg = mysql_cfg or {}

    flow_runner = RequirementAgent(
        llm=get_chat_model(role="API Flow Runner", flow_name=flow_name, containers=containers),
        name="API Flow Runner",
        description=(
            "Executes the target flow -- via Playwright for YAML-defined UI flows, via a plain-Python "
            "Playwright script for UI flows needing real control flow, or via a registered Python/requests "
            "API flow for pure JSON API flows -- and reports every request/response and step failure."
        ),
        role="API Flow Runner",
        instructions=(
            "You drive real user/API flows, following the flow's declared steps exactly. For a UI flow "
            "defined as flows/<name>.yaml you use run_playwright_flow; for a UI flow that needs real "
            "control flow (loops, conditionals, multi-page interaction) beyond the YAML step DSL, it's "
            "registered in playwright_flows.py and you use run_playwright_script instead; for a pure "
            "JSON API flow (no rendering involved) you use run_api_flow, which calls a registered "
            "Python/requests function directly. You are always told which tool to use and which "
            "flow_name to pass. You never skip steps and you report network activity faithfully — "
            "include the full network log and step results in your answer."
        ),
        tools=[PlaywrightFlowTool(), RunPlaywrightScriptTool(), ApiRequestFlowTool()],
        memory=LoggingMemory(agent_name="API Flow Runner"),
    )

    log_capturer = RequirementAgent(
        llm=get_chat_model(role="Docker Log Capturer", flow_name=flow_name, containers=containers),
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
        llm=get_chat_model(role="DB Query Agent", flow_name=flow_name, containers=containers),
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

    anomaly_check_tool = AnomalyCheckTool()
    bug_analyzer = RequirementAgent(
        llm=get_chat_model(role="Bug Analyst", flow_name=flow_name, containers=containers),
        name="Bug Analyst",
        description="Synthesizes the flow result, captured logs, and DB query output into a root-cause bug analysis.",
        role="Bug Analyst",
        instructions=(
            "You are a senior engineer who reads flow execution results, container logs, and database "
            "query output together, and produces a precise, evidence-backed root-cause analysis with a "
            "concrete recommended fix — not vague speculation. You must use check_anomalies as a quick "
            "heuristic first pass over raw flow/log output before you make the final call yourself."
        ),
        tools=[anomaly_check_tool],
        requirements=[ConditionalRequirement(anomaly_check_tool, force_at_step=1)],
        memory=LoggingMemory(agent_name="Bug Analyst"),
    )

    sql_performance_agent = RequirementAgent(
        llm=get_chat_model(role="SQL Performance Agent", flow_name=flow_name, containers=containers),
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

    source_code_analyst = RequirementAgent(
        llm=get_chat_model(role="Source Code Analyst", flow_name=flow_name, containers=containers),
        name="Source Code Analyst",
        description="Reads the app's own source code (copied out of its container) to confirm a hypothesis against the real implementation.",
        role="Source Code Analyst",
        instructions=(
            "You confirm or refute a suspected root cause by reading the application's actual source "
            "code with read_source_file — e.g. checking the exact column name in a SQL query, or the "
            "exact loop shape behind a suspected N+1 pattern — rather than relying only on log/DB "
            "evidence. Quote the specific line(s) that confirm or refute the hypothesis in your answer."
        ),
        tools=[ReadSourceFileTool(docker_host=docker_host)],
        memory=LoggingMemory(agent_name="Source Code Analyst"),
    )

    return {
        "flow_runner": flow_runner,
        "log_capturer": log_capturer,
        "db_query_agent": db_query_agent,
        "bug_analyzer": bug_analyzer,
        "sql_performance_agent": sql_performance_agent,
        "source_code_analyst": source_code_analyst,
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
    {
        "role": "Source Code Analyst",
        "goal": "Reads the app's own source (copied out of its container) to confirm a hypothesis against the real implementation.",
    },
]


def agent_summaries() -> list[dict[str, str]]:
    return AGENT_SUMMARIES
