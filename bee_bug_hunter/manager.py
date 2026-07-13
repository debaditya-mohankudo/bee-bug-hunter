"""Investigation Manager supervisor: a RequirementAgent whose only tools are
CapturingHandoffTools targeting the five workers — the BeeAI equivalent of the
old CrewAI Process.hierarchical crew. The manager decides sequencing and
escalation itself; handoffs propagate the conversation so far to each worker
(cross-delegation context comes free from HandoffTool's memory propagation,
replacing CrewAI's Crew(memory=True) embedder setup)."""
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement

from bee_bug_hunter.agents import build_agents
from bee_bug_hunter.config import DEFAULT_FLOW_KIND
from bee_bug_hunter.delegation_capture import CapturingHandoffTool
from bee_bug_hunter.llm import get_chat_model
from bee_bug_hunter.logging_memory import LoggingMemory


def build_supervisor(
    flow_name: str,
    containers: str,
    duration_seconds: int,
    docker_host: str | None = None,
    mysql_cfg: dict | None = None,
    flow_kind: str = DEFAULT_FLOW_KIND,
    api_flow_name: str | None = None,
    known_issue_note: str | None = None,
) -> tuple[RequirementAgent, str]:
    """Returns (supervisor agent, investigation prompt). flow_kind selects which
    Flow Runner tool the manager is told to have the worker use: 'ui' (default)
    drives flows/<flow_name>.yaml via Playwright; 'api' calls a registered
    bee_bug_hunter/api_flows.py function (api_flow_name) via requests instead.
    known_issue_note, if given (see known_issues.py), summarizes what another
    flow already found this same batch pass on a container this flow also
    touches -- prepended as context, not a shortcut: the manager still
    investigates from real evidence every time and only uses the note to judge
    whether this run reproduces the same issue or something new."""
    workers = build_agents(
        docker_host=docker_host, mysql_cfg=mysql_cfg, flow_name=flow_name, containers=containers,
    )

    handoffs = {
        key: CapturingHandoffTool(
            worker,
            role=worker.meta.name,
            description=f"Delegate a task to the {worker.meta.name}: {worker.meta.description}",
        )
        for key, worker in workers.items()
    }

    # Structural ordering guarantee, replacing "trust the prompt": the Bug Analyst,
    # SQL Performance Agent, and DB Query Agent must not be delegated to until the
    # Docker Log Capturer has actually run, since they're each handed "the logs
    # from step 2" that would otherwise not exist yet. Likewise the Docker Log
    # Capturer must not run until the Flow Runner has, since `capture_docker_logs`'s
    # `--since 5m` window is only a hedge for capturing a flow's already-emitted
    # output, not a substitute for real ordering (see docker_log_tool.py).
    requirements = [
        ConditionalRequirement(handoffs["log_capturer"], only_after=handoffs["flow_runner"]),
        ConditionalRequirement(handoffs["db_query_agent"], only_after=handoffs["log_capturer"]),
        ConditionalRequirement(handoffs["bug_analyzer"], only_after=handoffs["log_capturer"]),
        ConditionalRequirement(handoffs["sql_performance_agent"], only_after=handoffs["log_capturer"]),
        ConditionalRequirement(handoffs["source_code_analyst"], only_after=handoffs["log_capturer"]),
    ]

    supervisor = RequirementAgent(
        llm=get_chat_model(role="Investigation Manager", flow_name=flow_name, containers=containers),
        name="Investigation Manager",
        description=(
            "Runs the flow, captures its logs and DB state, decides whether anything is wrong, and if so "
            "delegates to the right specialist to produce a root-cause or performance report."
        ),
        role="Investigation Manager",
        instructions=(
            "You lead a bug-hunting investigation end to end. You have no domain tools of your own — "
            "you only delegate via the handoff tools, one worker at a time. Do not fabricate any "
            "evidence (logs, query output, stack traces); every fact in your report must come from a "
            "worker's real delegated output. If nothing looks wrong, say so plainly instead of "
            "manufacturing a finding."
        ),
        tools=list(handoffs.values()),
        requirements=requirements,
        memory=LoggingMemory(agent_name="Investigation Manager"),
    )

    if flow_kind == "api":
        step1 = (
            "1. Hand off to 'API Flow Runner': run the flow using its run_api_flow "
            f"tool with flow_name='{api_flow_name}'. Get back its full network log and step results.\n"
        )
    else:
        step1 = (
            "1. Hand off to 'API Flow Runner': run the flow using its run_playwright_flow "
            f"tool with flow_name='{flow_name}'. Get back its full network log and step results.\n"
        )

    known_issue_block = (
        f"Note: earlier in this same run, {known_issue_note}\n"
        "Treat this only as a hint, not a conclusion -- confirm from this flow's own real "
        "evidence whether the same issue reproduces here too or something new/different is "
        "happening.\n\n"
        if known_issue_note
        else ""
    )

    prompt = (
        f"{known_issue_block}"
        f"Investigate flow '{flow_name}'. Delegate in this order:\n"
        f"{step1}"
        "2. Hand off to 'Docker Log Capturer': capture logs using its capture_docker_logs tool "
        f"for containers [{containers}] over {duration_seconds} seconds, run_name='{flow_name}'. "
        "Get back the captured content per container.\n"
        "3. Hand off to 'DB Query Agent': point it at the captured logs from step 2 and have it "
        "find and run any referenced SQL with its run_mysql_query tool.\n"
        "4. From the real evidence gathered in steps 1-3, decide whether anything is wrong "
        "(functional failure and/or slow query). If so, hand off to 'Bug Analyst' and/or "
        "'SQL Performance Agent' as appropriate, passing them the real evidence you gathered. "
        "If the run is clean, report that plainly instead of inventing a finding.\n"
        "5. If 'Bug Analyst' or 'SQL Performance Agent' formed a specific hypothesis (e.g. a "
        f"suspected column name or query pattern), hand off to 'Source Code Analyst' with the "
        f"relevant container name from [{containers}] and the hypothesis, so it can confirm or "
        "refute it against the real source before you finalize the report.\n\n"
        "Final answer: begin with a single line 'SUMMARY: <one sentence>' stating either the "
        "confirmed root cause (e.g. 'passwd vs password column bug in api_login') or 'clean' if "
        "nothing was found, then a blank line, then the full markdown report — run summary, "
        "whether any issue was found, and (if so) the specialist findings and recommended fix."
    )

    return supervisor, prompt
