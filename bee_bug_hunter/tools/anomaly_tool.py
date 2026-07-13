"""Exposes the deterministic anomaly_detector as an optional tool the Bug Analyst
can call for a cheap first-pass read on flow/log output. The analyst isn't required
to use it or act on its verdict — it's a shortcut, not a gate."""
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from pydantic import BaseModel, Field

from bee_bug_hunter.anomaly_detector import detect


class CheckAnomaliesInput(BaseModel):
    flow_output_raw: str = Field(..., description="Raw JSON output from the run_playwright_flow tool")
    log_output_raw: str = Field(..., description="Raw JSON output from the capture_docker_logs tool")


class AnomalyCheckTool(Tool[CheckAnomaliesInput, ToolRunOptions, StringToolOutput]):
    name = "check_anomalies"
    description = (
        "Runs a fast, deterministic regex/threshold check over flow and log output and reports "
        "any HTTP errors, failed steps, ERROR-ish log lines, or slow (>=500ms) queries it finds. "
        "This is a cheap heuristic hint, not a verdict — use your own judgment on the raw output too."
    )
    input_schema = CheckAnomaliesInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "check_anomalies"], creator=self)

    async def _run(self, input: CheckAnomaliesInput, options, context) -> StringToolOutput:
        signals = detect(input.flow_output_raw, input.log_output_raw)
        return StringToolOutput(repr(signals.to_dict()))
