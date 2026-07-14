"""Registry of plain-Python Playwright flows -- the scripted alternative to a
YAML flow (bee_bug_hunter/flows/<name>.yaml), for cases where the declarative
step DSL (goto/click/fill/wait_for_selector/wait_for_response/expect_status/
expect_text) can't express what's needed -- loops, conditionals, multi-page
interaction, or anything else real control flow buys you that a step list
can't. The counterpart to api_flows.py's @api_flow registry, but for browser
flows instead of pure JSON API calls.

Each registered function takes the already-launched Playwright `page` (async
API -- see playwright_script_tool.py for why) plus a `network_log` list to
append {method, url, status} dicts to, and returns just `step_results`
(list[dict], same {"step", "status", ["error"]} shape YAML flows produce).
RunPlaywrightScriptTool wraps the result into the same {flow_name,
step_results, network_log} dict PlaywrightFlowTool and run_api_flow both
produce, so anomaly_detector.py and everything downstream of the Flow Runner
keep working unmodified regardless of which of the three ways a flow ran.

Add a new scripted flow by writing an async function and decorating it with
@playwright_flow; its registry name becomes the value manifest.yaml's
`playwright_flow:` key and RunPlaywrightScriptTool's flow_name argument both
refer to.
"""
from typing import Awaitable, Callable

from playwright.async_api import Page

PLAYWRIGHT_FLOW_REGISTRY: dict[str, Callable[[Page, list], Awaitable[list[dict]]]] = {}


def playwright_flow(name: str):
    def decorator(fn: Callable[[Page, list], Awaitable[list[dict]]]) -> Callable[[Page, list], Awaitable[list[dict]]]:
        PLAYWRIGHT_FLOW_REGISTRY[name] = fn
        return fn
    return decorator


@playwright_flow("example_login_script")
async def example_login_script(page: Page, network_log: list) -> list[dict]:
    """Plain-Playwright equivalent of flows/example_login.yaml -- same six
    steps, same seeded-bug endpoint, written as an ordinary async function
    instead of a step list. Kept step-for-step identical to the YAML version
    so the two are a direct side-by-side comparison of the same flow in both
    formats."""
    base_url = "http://localhost:3000"
    step_results = []

    async def _run_step(step: dict) -> None:
        action = step["action"]
        try:
            if action == "goto":
                await page.goto(base_url + step["path"])
            elif action == "fill":
                await page.fill(step["selector"], step["value"])
            elif action == "click":
                await page.click(step["selector"])
            elif action == "wait_for_response":
                await page.wait_for_event(
                    "response", predicate=lambda r: step["url_contains"] in r.url, timeout=step["timeout_ms"],
                )
            elif action == "wait_for_selector":
                await page.wait_for_selector(step["selector"], timeout=step["timeout_ms"])
            step_results.append({"step": step, "status": "ok"})
        except Exception as e:
            step_results.append({"step": step, "status": "failed", "error": str(e)})

    await _run_step({"action": "goto", "path": "/login"})
    await _run_step({"action": "fill", "selector": "#email", "value": "test@example.com"})
    await _run_step({"action": "fill", "selector": "#password", "value": "password123"})
    await _run_step({"action": "click", "selector": "#login-submit"})
    await _run_step({"action": "wait_for_response", "url_contains": "/api/auth/login", "timeout_ms": 10000})
    await _run_step({"action": "wait_for_selector", "selector": "#dashboard", "timeout_ms": 10000})

    return step_results
