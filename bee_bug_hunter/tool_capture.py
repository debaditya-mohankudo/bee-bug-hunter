"""Captures the raw JSON output of the evidence-collecting tools, keyed by run_id.

delegation_capture.py records what each *worker agent* reported back to the
supervisor — prose, filtered through the worker's own LLM summarization. That is
the right source for the Bug Analyst / SQL Performance report text, but the wrong
input for anomaly_detector.detect(), which parses the tools' raw JSON (network
log, step results, per-container log content). This module subscribes to BeeAI's
global emitter and records the unfiltered StringToolOutput of the flow-runner and
log-capture tools whenever they succeed, so the deterministic signals are computed
from what the tools actually saw rather than what an LLM chose to mention.

Emitter callbacks fire synchronously inside the tool's own asyncio task, so the
run_id contextvar set by new_run_context() is visible here — same correlation
mechanism as JSONL logging and delegation_capture.
"""
import threading

from beeai_framework.emitter import Emitter
from beeai_framework.tools.events import ToolSuccessEvent

from bee_bug_hunter.logging_config import run_id_var

# tool name -> capture slot; both flow-runner tools share the "flow" slot since
# anomaly_detector treats their (identically shaped) output the same way.
_SLOT_BY_TOOL = {
    "run_playwright_flow": "flow",
    "run_api_flow": "flow",
    "capture_docker_logs": "logs",
}

_lock = threading.Lock()
_captures: dict[str, dict[str, list[str]]] = {}
_installed = False


def install() -> None:
    """Idempotent: subscribes to the root emitter exactly once per process even
    if called from multiple entry points (orchestrator.py and tui.py both do)."""
    global _installed
    if _installed:
        return

    def _on_event(data, event) -> None:
        if not isinstance(data, ToolSuccessEvent):
            return
        slot = _SLOT_BY_TOOL.get(getattr(event.creator, "name", ""))
        if slot is None:
            return
        run_id = run_id_var.get()
        with _lock:
            _captures.setdefault(run_id, {}).setdefault(slot, []).append(
                data.output.get_text_content()
            )

    Emitter.root().on("*.*", _on_event)
    _installed = True


def _get(run_id: str, slot: str) -> str | None:
    """Most recent raw output for this run/slot — a tool can run more than once
    in one investigation; the last run reflects the state the report describes.
    None means the tool never succeeded this run."""
    with _lock:
        outputs = _captures.get(run_id, {}).get(slot)
    return outputs[-1] if outputs else None


def get_flow_raw(run_id: str) -> str | None:
    return _get(run_id, "flow")


def get_log_raw(run_id: str) -> str | None:
    return _get(run_id, "logs")


def clear(run_id: str) -> None:
    with _lock:
        _captures.pop(run_id, None)
