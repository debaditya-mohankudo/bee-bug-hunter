"""Executes a YAML-defined API/UI flow with Playwright and records the result.

Uses Playwright's async API directly -- BeeAI tools run inside the agent's
asyncio loop, and the sync Playwright API refuses to run when a loop is
already running, so async is the natural (and required) fit here."""
import json
import logging
from pathlib import Path

import yaml
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

from bee_bug_hunter.logging_config import get_logger, log

FLOWS_DIR = Path(__file__).parent.parent / "flows"
logger = get_logger(__name__)


class RunFlowInput(BaseModel):
    flow_name: str = Field(..., description="Name of the flow YAML file (without .yaml) in the flows/ directory")


async def _run_flow(flow_name: str) -> str:
    log(logger, logging.INFO, "playwright_flow_started", flow_name=flow_name)
    flow_path = FLOWS_DIR / f"{flow_name}.yaml"
    if not flow_path.exists():
        log(logger, logging.ERROR, "playwright_flow_file_missing", flow_path=str(flow_path))
        return json.dumps({"error": f"flow file not found: {flow_path}"})

    flow = yaml.safe_load(flow_path.read_text())
    base_url = flow.get("base_url", "")
    steps = flow.get("steps", [])

    network_log = []
    step_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=flow.get("headless", True))
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

        for step in steps:
            action = step.get("action")
            try:
                if action == "goto":
                    await page.goto(base_url + step.get("path", ""))
                elif action == "click":
                    await page.click(step["selector"])
                elif action == "fill":
                    await page.fill(step["selector"], step.get("value", ""))
                elif action == "wait_for_selector":
                    await page.wait_for_selector(step["selector"], timeout=step.get("timeout_ms", 10000))
                elif action == "wait_for_response":
                    await page.wait_for_event(
                        "response",
                        predicate=lambda r: step["url_contains"] in r.url,
                        timeout=step.get("timeout_ms", 10000),
                    )
                else:
                    step_results.append({"step": step, "status": "skipped_unknown_action"})
                    continue
                step_results.append({"step": step, "status": "ok"})
            except Exception as e:
                step_results.append({"step": step, "status": "failed", "error": str(e)})
                log(logger, logging.WARNING, "playwright_step_failed", action=action, error=str(e))

        await browser.close()

    failed_steps = sum(1 for s in step_results if s["status"] == "failed")
    log(
        logger, logging.INFO, "playwright_flow_finished",
        flow_name=flow_name, step_count=len(step_results), failed_steps=failed_steps,
        network_requests=len(network_log),
    )

    return json.dumps({
        "flow_name": flow_name,
        "step_results": step_results,
        "network_log": network_log,
    }, indent=2)


class PlaywrightFlowTool(Tool[RunFlowInput, ToolRunOptions, StringToolOutput]):
    name = "run_playwright_flow"
    description = (
        "Runs a named Playwright flow defined in flows/<flow_name>.yaml against a browser. "
        "Each step is one of: goto, click, fill, wait_for_selector, wait_for_response, expect_status. "
        "Returns a JSON summary of every HTTP request/response observed during the flow, "
        "plus any step failures."
    )
    input_schema = RunFlowInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "run_playwright_flow"], creator=self)

    async def _run(self, input: RunFlowInput, options, context) -> StringToolOutput:
        result = await _run_flow(input.flow_name)
        return StringToolOutput(result)
