"""Investigation Manager supervisor: a RequirementAgent whose only tools are
CapturingHandoffTools targeting the five workers — the BeeAI equivalent of the
old CrewAI Process.hierarchical crew. The manager decides sequencing and
escalation itself; handoffs propagate the conversation so far to each worker
(cross-delegation context comes free from HandoffTool's memory propagation,
replacing CrewAI's Crew(memory=True) embedder setup)."""
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory

from bee_bug_hunter.agents import build_agents
from bee_bug_hunter.config import DEFAULT_FLOW_KIND
from bee_bug_hunter.delegation_capture import CapturingHandoffTool
from bee_bug_hunter.llm import get_chat_model


def build_supervisor(
    flow_name: str,
    containers: str,
    duration_seconds: int,
    docker_host: str | None = None,
    mysql_cfg: dict | None = None,
    flow_kind: str = DEFAULT_FLOW_KIND,
    api_flow_name: str | None = None,
) -> tuple[RequirementAgent, str]:
    """Returns (supervisor agent, investigation prompt). flow_kind selects which
    Flow Runner tool the manager is told to have the worker use: 'ui' (default)
    drives flows/<flow_name>.yaml via Playwright; 'api' calls a registered
    bee_bug_hunter/api_flows.py function (api_flow_name) via requests instead."""
    workers = build_agents(docker_host=docker_host, mysql_cfg=mysql_cfg)

    handoffs = [
        CapturingHandoffTool(
            worker,
            role=worker.meta.name,
            description=f"Delegate a task to the {worker.meta.name}: {worker.meta.description}",
        )
        for worker in workers.values()
    ]

    supervisor = RequirementAgent(
        llm=get_chat_model(),
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
        tools=handoffs,
        memory=UnconstrainedMemory(),
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

    prompt = (
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
        "If the run is clean, report that plainly instead of inventing a finding.\n\n"
        "Final answer: a markdown report — run summary, whether any issue was found, and (if so) "
        "the specialist findings and recommended fix. If clean, a short statement saying so."
    )

    return supervisor, prompt
