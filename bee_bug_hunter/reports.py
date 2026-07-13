"""Persists one markdown report per flow run. Without this the Bug Analyst /
SQL Performance Agent findings would only exist in-memory (returned from
run_flow_once) or transiently in stdout / the supervisor's summary line in the
JSONL log -- nothing would capture the actual deliverable to disk."""
import os
from datetime import datetime, timezone

from bee_bug_hunter.config import DEFAULT_REPORTS_DIR


def save_report(result: dict, reports_dir: str = DEFAULT_REPORTS_DIR) -> str:
    """Writes result (as returned by orchestrator.run_flow_once) to a markdown
    file under reports_dir, named so runs sort and correlate with the JSONL
    log by run_id. Returns the file path written."""
    os.makedirs(reports_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = result["run_id"]
    flow = result["flow"]
    path = os.path.join(reports_dir, f"{ts}_{flow}_{run_id}.md")

    anomaly = result["anomaly"]
    lines = [
        f"# {flow}",
        "",
        f"- run_id: {run_id}",
        f"- generated: {ts}",
        f"- bug_signal: {anomaly.get('bug_signal')}",
        f"- perf_signal: {anomaly.get('perf_signal')}",
        "",
    ]

    if result.get("bug_report"):
        lines += ["## Bug Analyst report", "", result["bug_report"], ""]
    if result.get("perf_report"):
        lines += ["## SQL Performance Agent report", "", result["perf_report"], ""]
    if not result.get("bug_report") and not result.get("perf_report"):
        lines += ["## Manager summary", "", result["response"], ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return path
