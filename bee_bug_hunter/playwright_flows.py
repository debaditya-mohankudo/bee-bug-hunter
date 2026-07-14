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
    # Local import: tools/playwright_script_tool.py imports PLAYWRIGHT_FLOW_REGISTRY
    # from this module at module scope, so importing run_step back at module
    # scope here would be circular. See run_step's own docstring.
    from bee_bug_hunter.tools.playwright_script_tool import run_step

    base_url = "http://localhost:3000"
    step_results: list[dict] = []

    await run_step(page, {"action": "goto", "path": "/login"}, step_results, base_url)
    await run_step(page, {"action": "fill", "selector": "#email", "value": "test@example.com"}, step_results, base_url)
    await run_step(page, {"action": "fill", "selector": "#password", "value": "password123"}, step_results, base_url)
    await run_step(page, {"action": "click", "selector": "#login-submit"}, step_results, base_url)
    await run_step(page, {"action": "wait_for_response", "url_contains": "/api/auth/login", "timeout_ms": 10000}, step_results, base_url)
    await run_step(page, {"action": "wait_for_selector", "selector": "#dashboard", "timeout_ms": 10000}, step_results, base_url)

    return step_results
