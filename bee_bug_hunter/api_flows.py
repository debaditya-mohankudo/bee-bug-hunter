"""Registry of Python/requests-based API flows -- the Tavern-style alternative
to a YAML/Playwright flow (bee_bug_hunter/flows/<name>.yaml), for cases where
the target is a pure JSON API and browser automation is unnecessary overhead.

Each registered function takes no arguments and returns a dict shaped exactly
like PlaywrightFlowTool's return value ({"flow_name", "step_results",
"network_log"}), so anomaly_detector.py and everything downstream of the Flow
Runner keep working unmodified regardless of which tool actually ran the flow.

Add a new API flow by writing a function and decorating it with @api_flow;
its registry name becomes the value manifest.yaml's `api_flow:` key and
ApiRequestFlowTool's flow_name argument both refer to.
"""
from typing import Callable

import requests

API_FLOW_REGISTRY: dict[str, Callable[[], dict]] = {}


def api_flow(name: str):
    def decorator(fn: Callable[[], dict]) -> Callable[[], dict]:
        API_FLOW_REGISTRY[name] = fn
        return fn
    return decorator


@api_flow("example_login_api")
def example_login_api() -> dict:
    """Direct-request equivalent of flows/example_login.yaml: hits the same
    seeded-bug endpoint without a browser, since it's a pure JSON POST/response."""
    base_url = "http://localhost:3000"
    step_results = []
    network_log = []

    resp = requests.post(
        f"{base_url}/api/auth/login",
        json={"email": "test@example.com", "password": "password123"},
        timeout=10,
    )
    network_log.append({"method": "POST", "url": resp.url, "status": resp.status_code})
    step_results.append({
        "step": {"action": "post", "path": "/api/auth/login"},
        "status": "ok" if resp.ok else "failed",
        **({} if resp.ok else {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}),
    })

    return {
        "flow_name": "example_login_api",
        "step_results": step_results,
        "network_log": network_log,
    }
