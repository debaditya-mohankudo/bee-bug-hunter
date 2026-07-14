"""Runs a registered plain-Python Playwright flow (bee_bug_hunter/playwright_flows.py) --
the scripted counterpart to PlaywrightFlowTool's YAML step DSL. Uses Playwright's
async API directly for the same reason PlaywrightFlowTool does: BeeAI tools run
inside the agent's asyncio loop, and the sync Playwright API refuses to run when
a loop is already running."""
import json
import logging

from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

from bee_bug_hunter.logging_config import get_logger, log
from bee_bug_hunter.playwright_flows import PLAYWRIGHT_FLOW_REGISTRY

logger = get_logger(__name__)


class RunPlaywrightScriptInput(BaseModel):
    flow_name: str = Field(..., description="Registry name of the Playwright flow function to run (see bee_bug_hunter/playwright_flows.py)")
    headless: bool = Field(True, description="Whether to launch the browser headless")


async def _run_script(flow_name: str, headless: bool) -> str:
    log(logger, logging.INFO, "playwright_script_started", flow_name=flow_name)
    fn = PLAYWRIGHT_FLOW_REGISTRY.get(flow_name)
    if fn is None:
        log(logger, logging.ERROR, "playwright_script_not_registered", flow_name=flow_name, known=list(PLAYWRIGHT_FLOW_REGISTRY))
        return json.dumps({"error": f"no Playwright flow registered under name '{flow_name}'"})

    network_log: list = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        def on_response(response):
            try:
                network_log.append({
                    "method": response.request.method,
                    "url": response.url,
                    "status": response.status,
                })
            except Exception:
                pass

        page.on("response", on_response)

        try:
            step_results = await fn(page, network_log)
        except Exception as e:
            log(logger, logging.ERROR, "playwright_script_raised", flow_name=flow_name, error=str(e))
            await browser.close()
            return json.dumps({"error": f"Playwright flow '{flow_name}' raised: {e}"})

        await browser.close()

    failed_steps = sum(1 for s in step_results if s.get("status") == "failed")
    log(
        logger, logging.INFO, "playwright_script_finished",
        flow_name=flow_name, step_count=len(step_results), failed_steps=failed_steps,
        network_requests=len(network_log),
    )

    return json.dumps({
        "flow_name": flow_name,
        "step_results": step_results,
        "network_log": network_log,
    }, indent=2)


class RunPlaywrightScriptTool(Tool[RunPlaywrightScriptInput, ToolRunOptions, StringToolOutput]):
    name = "run_playwright_script"
    description = (
        "Runs a named plain-Python Playwright flow registered in bee_bug_hunter/playwright_flows.py. "
        "Use this instead of run_playwright_flow when the target flow needs real control flow "
        "(loops, conditionals, multi-page interaction) that the YAML step DSL can't express. "
        "Returns a JSON summary of every HTTP request/response observed during the flow, plus any "
        "step failures -- same shape as run_playwright_flow and run_api_flow."
    )
    input_schema = RunPlaywrightScriptInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "run_playwright_script"], creator=self)

    async def _run(self, input: RunPlaywrightScriptInput, options, context) -> StringToolOutput:
        result = await _run_script(input.flow_name, input.headless)
        return StringToolOutput(result)
