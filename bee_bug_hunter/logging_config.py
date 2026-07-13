"""Structured JSON logging to stdout, with a per-flow-run correlation id.

Enterprise deployments scrape stdout with a log aggregator (ELK, Datadog, CloudWatch)
rather than reading terminal output directly, so every line is a single JSON object.
run_id is a contextvar rather than a threaded-through parameter: tool _run() methods
are invoked by BeeAI's agent runtime, which doesn't let us pass extra positional args,
so context-local state is the only way tool-level logs pick up the correlation id.
BeeAI's asyncio loop propagates contextvars across awaits, so tools running inside
the supervisor's event loop see the run_id set by new_run_context() in the caller.
"""
import json
import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

run_id_var: ContextVar[str] = ContextVar("run_id", default="-")
flow_name_var: ContextVar[str] = ContextVar("flow_name", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": run_id_var.get(),
            "flow": flow_name_var.get(),
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO, log_file: str | None = "logs/bee_bug_hunter.jsonl") -> None:
    """stdout handler is always on (for log aggregators). log_file additionally
    writes the same JSONL lines to disk with rotation — pass None to disable."""
    handlers: list[logging.Handler] = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(JsonFormatter())
    handlers.append(stdout_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)

    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log(logger: logging.Logger, level: int, message: str, **fields) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


def new_run_context(flow_name: str) -> str:
    """Call once per flow run (e.g. at the top of run_flow_once) to scope all
    subsequent log lines — from the orchestrator down through tool calls — to
    this run via run_id/flow contextvars."""
    run_id = uuid.uuid4().hex[:12]
    run_id_var.set(run_id)
    flow_name_var.set(flow_name)
    return run_id
