"""Runs a registered Python/requests API flow (bee_bug_hunter/api_flows.py) --
the non-browser counterpart to PlaywrightFlowTool, for flows that are pure
JSON API calls with nothing to render or click."""
import asyncio
import json
import logging

from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from pydantic import BaseModel, Field

from bee_bug_hunter.api_flows import API_FLOW_REGISTRY
from bee_bug_hunter.logging_config import get_logger, log

logger = get_logger(__name__)


class RunApiFlowInput(BaseModel):
    flow_name: str = Field(..., description="Registry name of the API flow function to run (see bee_bug_hunter/api_flows.py)")


class ApiRequestFlowTool(Tool[RunApiFlowInput, ToolRunOptions, StringToolOutput]):
    name = "run_api_flow"
    description = (
        "Runs a named Python/requests API flow registered in bee_bug_hunter/api_flows.py. "
        "Use this instead of run_playwright_flow when the target flow is a pure JSON API call "
        "with no browser rendering involved. Returns a JSON summary of every HTTP request/response "
        "observed during the flow, plus any step failures -- same shape as run_playwright_flow."
    )
    input_schema = RunApiFlowInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "run_api_flow"], creator=self)

    async def _run(self, input: RunApiFlowInput, options, context) -> StringToolOutput:
        flow_name = input.flow_name
        log(logger, logging.INFO, "api_flow_started", flow_name=flow_name)
        fn = API_FLOW_REGISTRY.get(flow_name)
        if fn is None:
            log(logger, logging.ERROR, "api_flow_not_registered", flow_name=flow_name, known=list(API_FLOW_REGISTRY))
            return StringToolOutput(json.dumps({"error": f"no API flow registered under name '{flow_name}'"}))

        try:
            result = await asyncio.to_thread(fn)
        except Exception as e:
            log(logger, logging.ERROR, "api_flow_raised", flow_name=flow_name, error=str(e))
            return StringToolOutput(json.dumps({"error": f"API flow '{flow_name}' raised: {e}"}))

        failed_steps = sum(1 for s in result.get("step_results", []) if s.get("status") == "failed")
        log(
            logger, logging.INFO, "api_flow_finished",
            flow_name=flow_name, step_count=len(result.get("step_results", [])), failed_steps=failed_steps,
            network_requests=len(result.get("network_log", [])),
        )
        return StringToolOutput(json.dumps(result, indent=2))
