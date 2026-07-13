"""Captures every supervisor -> worker delegation via CapturingHandoffTool, a
HandoffTool subclass that records each worker's own returned text.

This is the reliable hook into what each specialist agent (API Flow Runner,
Docker Log Capturer, Bug Analyst, SQL Performance Agent) actually reported back
to the Investigation Manager. The supervisor's own final answer is a synthesized
summary written by its own LLM call, not each worker's raw output.

Keyed by run_id (the same contextvar-based correlation id used by JSONL logging),
so callers can pull out, after a flow's supervisor run returns, what each worker
actually said -- independent of whether the manager chose to quote it in its
final markdown report. In particular this lets the TUI/orchestrator show:
  - anomaly signals computed from the *actual* API Flow Runner / Docker Log
    Capturer output, and
  - the Bug Analyst / SQL Performance Agent's own report text, separately.
"""
import logging
import threading
from typing import NamedTuple

from beeai_framework.tools import StringToolOutput
from beeai_framework.tools.handoff import HandoffTool

from bee_bug_hunter.logging_config import get_logger, log, run_id_var

logger = get_logger(__name__)

_lock = threading.Lock()
_captures: dict[str, list["Delegation"]] = {}


class Delegation(NamedTuple):
    coworker: str
    task: str
    result: str


def _normalize_role(name: str) -> str:
    """Whitespace/underscore/case-insensitive role matching, so lookups by the
    human role name ("API Flow Runner") match the safe-word tool/agent name
    ("api_flow_runner") HandoffTool derives from it."""
    if not name:
        return ""
    return " ".join(name.replace("_", " ").split()).replace('"', "").casefold()


class CapturingHandoffTool(HandoffTool):
    """HandoffTool that additionally records (role, task, worker result) under
    the current run_id before handing the result back to the supervisor."""

    def __init__(self, *args, role: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._role = role or self._name

    async def _run(self, input, options, context) -> StringToolOutput:
        output = await super()._run(input, options, context)
        result = output.get_text_content()
        run_id = run_id_var.get()
        with _lock:
            _captures.setdefault(run_id, []).append(
                Delegation(coworker=self._role, task=input.task, result=result)
            )
        log(
            logger, logging.INFO, "delegation_captured",
            coworker=self._role, result_chars=len(result or ""),
        )
        return output


def get_by_role(run_id: str, role: str) -> str | None:
    """Concatenates every result a given worker role returned during this
    run_id (a worker can be delegated to more than once in one investigation).
    Returns None if that role was never delegated to, so callers can distinguish
    "not invoked" from "invoked but said nothing"."""
    target = _normalize_role(role)
    with _lock:
        matches = [
            d.result for d in _captures.get(run_id, [])
            if _normalize_role(d.coworker) == target
        ]
    if not matches:
        return None
    return "\n\n---\n\n".join(matches)


def all_for_run(run_id: str) -> list[Delegation]:
    with _lock:
        return list(_captures.get(run_id, []))


def clear(run_id: str) -> None:
    with _lock:
        _captures.pop(run_id, None)
