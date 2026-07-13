"""Deterministic, non-LLM anomaly detection over collected flow/log output.

Decides routing: HTTP errors / step failures / ERROR-ish log lines -> bug_signal
(Bug Analyst); query durations over threshold -> perf_signal (SQL Performance Agent).
Kept as plain code, not an agent, so routing is cheap, predictable, and testable.
"""
import json
import re
from dataclasses import dataclass, field

SLOW_QUERY_MS_THRESHOLD = 500
ERROR_LOG_PATTERN = re.compile(r"\b(ERROR|Exception|Traceback|FATAL)\b")
DURATION_MS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*ms\b")


@dataclass
class AnomalySignals:
    bug_signal: bool = False
    perf_signal: bool = False
    http_errors: list = field(default_factory=list)
    failed_steps: list = field(default_factory=list)
    error_containers: list = field(default_factory=list)
    slow_containers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bug_signal": self.bug_signal,
            "perf_signal": self.perf_signal,
            "http_errors": self.http_errors,
            "failed_steps": self.failed_steps,
            "error_containers": self.error_containers,
            "slow_containers": self.slow_containers,
        }


def detect(flow_output_raw: str, log_output_raw: str) -> AnomalySignals:
    signals = AnomalySignals()

    try:
        flow = json.loads(flow_output_raw)
    except (json.JSONDecodeError, TypeError):
        flow = {}

    try:
        logs = json.loads(log_output_raw)
    except (json.JSONDecodeError, TypeError):
        logs = {}

    signals.http_errors = [
        r for r in flow.get("network_log", []) if r.get("status", 0) >= 400
    ]
    signals.failed_steps = [
        s for s in flow.get("step_results", []) if s.get("status") == "failed"
    ]

    for container, data in logs.items():
        content = data.get("content", "") if isinstance(data, dict) else ""
        if ERROR_LOG_PATTERN.search(content):
            signals.error_containers.append(container)
        for match in DURATION_MS_PATTERN.finditer(content):
            if float(match.group(1)) >= SLOW_QUERY_MS_THRESHOLD:
                signals.slow_containers.append(container)
                break

    signals.slow_containers = sorted(set(signals.slow_containers))
    signals.bug_signal = bool(
        signals.http_errors or signals.failed_steps or signals.error_containers
    )
    signals.perf_signal = bool(signals.slow_containers)

    return signals
