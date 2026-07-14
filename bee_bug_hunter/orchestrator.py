"""Batch-of-flows monitor loop: each flow is handed to the Investigation Manager
supervisor, which decides sequencing and escalation itself (see manager.py)."""
import asyncio
import logging
import os
import re
import time

import yaml
from beeai_framework.errors import FrameworkError

from bee_bug_hunter import delegation_capture, tool_capture
from bee_bug_hunter.anomaly_detector import detect
from bee_bug_hunter.config import DEFAULT_FLOW_KIND
from bee_bug_hunter.known_issues import compute_fingerprint, note_for, record_issue
from bee_bug_hunter.logging_config import get_logger, log, new_run_context
from bee_bug_hunter.manager import build_supervisor
from bee_bug_hunter.reports import save_report

_SUMMARY_LINE_PATTERN = re.compile(r"^SUMMARY:\s*(.+)$", re.MULTILINE)

logger = get_logger(__name__)


def _framework_error_fields(error: FrameworkError) -> dict:
    """error.explain() walks the full predecessor/__cause__ chain (agent ->
    tool -> backend errors nest via `cause=`), giving a diagnosable trail
    without a re-run. context dicts are attacker/PII-free today (nothing in
    this codebase constructs a FrameworkError with context=), but any future
    context= must stay query-shape-only per the mysql logging rule above."""
    return {
        "error_type": error.name(),
        "is_fatal": error.fatal,
        "is_retryable": error.retryable,
        "error_chain": error.explain(),
    }

# Installed once at import time: from here on, every successful flow-runner /
# log-capture tool call (for any flow run in this process) has its raw JSON
# output recorded, keyed by the run_id contextvar — see tool_capture.py.
tool_capture.install()


def run_flow_once(flow_cfg: dict, duration_seconds: int, known_issues: list | None = None) -> dict:
    """known_issues, if given, is a per-batch-pass registry (see known_issues.py)
    shared across every flow in the same run_batch_once call -- every flow gets
    told what any earlier flow in this same pass found, and the manager decides
    for itself whether it's relevant. None means "standalone run" (e.g. a
    single-flow --manifest for testing): no known-issue note is used at all,
    and nothing gets recorded anywhere."""
    flow_name = flow_cfg["name"]
    containers = ",".join(flow_cfg["containers"])
    run_id = new_run_context(flow_name)

    log(logger, logging.INFO, "flow_run_started", containers=containers, duration_seconds=duration_seconds)
    started = time.monotonic()

    known_issue_note = note_for(known_issues) if known_issues is not None else None
    if known_issue_note:
        log(logger, logging.INFO, "known_issue_note_applied", note=known_issue_note)

    try:
        supervisor, prompt = build_supervisor(
            flow_name, containers, duration_seconds,
            docker_host=flow_cfg.get("docker_host"),
            mysql_cfg=flow_cfg.get("mysql"),
            flow_kind=flow_cfg.get("kind", DEFAULT_FLOW_KIND),
            api_flow_name=flow_cfg.get("api_flow"),
            known_issue_note=known_issue_note,
        )
        # asyncio.run copies the current context, so the run_id contextvar set
        # above is visible to every tool call inside the supervisor's loop.
        # supervisor.run() returns a BeeAI Run (awaitable, not a coroutine),
        # so it must be awaited inside a real coroutine for asyncio.run.
        async def _drive():
            return await supervisor.run(prompt)

        output = asyncio.run(_drive())
        result = output.last_message.text
    except FrameworkError as e:
        # AgentError/ToolError/BackendError/ChatModelError etc. all derive from
        # FrameworkError -- catch it first so the structured fields below are
        # available; fall through to the bare Exception branch for anything
        # BeeAI itself doesn't wrap (e.g. a raw httpx error before wrapping).
        log(
            logger, logging.ERROR, "supervisor_run_failed",
            elapsed_s=round(time.monotonic() - started, 2),
            **_framework_error_fields(e),
        )
        logger.exception("supervisor run raised a FrameworkError")
        delegation_capture.clear(run_id)
        tool_capture.clear(run_id)
        raise
    except Exception:
        log(logger, logging.ERROR, "supervisor_run_failed", elapsed_s=round(time.monotonic() - started, 2))
        logger.exception("supervisor run raised")
        delegation_capture.clear(run_id)
        tool_capture.clear(run_id)
        raise

    log(logger, logging.INFO, "flow_run_completed", elapsed_s=round(time.monotonic() - started, 2), run_id=run_id)

    # Independent of whatever the manager chose to quote in its final answer:
    # feed the anomaly detector the *raw tool JSON* recorded by tool_capture
    # (network log, step results, per-container log content) — the workers'
    # delegation answers are LLM prose and usually not parseable JSON, so they
    # are only a last-resort fallback here.
    flow_raw = (
        tool_capture.get_flow_raw(run_id)
        or delegation_capture.get_by_role(run_id, "API Flow Runner")
        or ""
    )
    log_raw = (
        tool_capture.get_log_raw(run_id)
        or delegation_capture.get_by_role(run_id, "Docker Log Capturer")
        or ""
    )
    bug_report = delegation_capture.get_by_role(run_id, "Bug Analyst")
    perf_report = delegation_capture.get_by_role(run_id, "SQL Performance Agent")
    anomaly = detect(flow_raw, log_raw).to_dict()
    delegation_capture.clear(run_id)
    tool_capture.clear(run_id)

    log(logger, logging.INFO, "anomaly_signals_computed", **anomaly)

    report_result = {
        "flow": flow_name,
        "run_id": run_id,
        "response": str(result),
        "anomaly": anomaly,
        "bug_report": bug_report,
        "perf_report": perf_report,
    }
    report_path = save_report(report_result)
    log(logger, logging.INFO, "report_saved", path=report_path)
    report_result["report_path"] = report_path

    if known_issues is not None and compute_fingerprint(anomaly) != "clean":
        summary_match = _SUMMARY_LINE_PATTERN.search(str(result))
        summary = summary_match.group(1).strip() if summary_match else str(result)[:200]
        record_issue(known_issues, flow_name, summary)

    return report_result


def run_batch_once(manifest: dict) -> list[dict]:
    duration_seconds = manifest.get("duration_seconds", 30)

    if os.getenv("LLM_PROVIDER") == "claude_cli":
        # Same reason main.py clears sessions at process startup: each flow run gets a
        # fresh, empty RequirementAgentRunState, but claude_cli's per-(flow, role)
        # sessions are cached in-memory in ClaudeCLIChatModel._instances for the life of
        # the process. In monitor_loop's continuous mode, that cache otherwise survives
        # across poll cycles within this same process, so cycle 2+ would resume a
        # session whose CLI-side history already has cycle 1's completed steps baked
        # in -- the same crash-recovery mismatch, just without a crash needed.
        from bee_bug_hunter.claude_cli_llm import clear_persisted_sessions

        clear_persisted_sessions()
    elif os.getenv("LLM_PROVIDER") == "copilot_cli":
        # Same rationale, same fix -- see the claude_cli branch above.
        from bee_bug_hunter.copilot_cli_llm import clear_persisted_sessions

        clear_persisted_sessions()

    # Fresh every batch pass, discarded at the end -- never persisted across poll
    # cycles or process restarts (see known_issues.py's module docstring for why).
    known_issues: list = []

    results = []
    for flow_cfg in manifest["flows"]:
        try:
            results.append(run_flow_once(flow_cfg, duration_seconds, known_issues))
        except FrameworkError as e:
            # one flow's failure shouldn't take down the rest of the batch;
            # the full chain is already logged in run_flow_once -- this line
            # just needs enough to grep alongside the flow name.
            log(
                logger, logging.ERROR, "flow_run_skipped_after_failure",
                flow=flow_cfg.get("name"), **_framework_error_fields(e),
            )
            continue
        except Exception:
            # the exception is already logged with full context in run_flow_once.
            log(logger, logging.ERROR, "flow_run_skipped_after_failure", flow=flow_cfg.get("name"))
            continue
    return results


def monitor_loop(manifest_path: str, once: bool = False):
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    poll_interval = manifest.get("poll_interval_seconds", 300)
    log(logger, logging.INFO, "monitor_loop_started", manifest=manifest_path, poll_interval_seconds=poll_interval, once=once)

    while True:
        results = run_batch_once(manifest)
        for r in results:
            if r["response"]:
                log(logger, logging.INFO, "response_report", flow=r["flow"], run_id=r["run_id"], report=r["response"])

        if once:
            return results

        log(logger, logging.INFO, "monitor_loop_sleeping", seconds=poll_interval)
        time.sleep(poll_interval)
