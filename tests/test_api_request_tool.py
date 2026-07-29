"""Tests for ApiRequestFlowTool: runs a named function out of
bee_bug_hunter.api_flows.API_FLOW_REGISTRY. Tested via fake functions
injected directly into the registry rather than exercising the real
network-calling example_login_api -- that's api_flows.py's own concern
(see test_api_flows.py)."""
import json

import pytest

from bee_bug_hunter.api_flows import API_FLOW_REGISTRY
from bee_bug_hunter.tools.api_request_tool import ApiRequestFlowTool, RunApiFlowInput


@pytest.fixture(autouse=True)
def _clean_registry():
    """Registry is module-level/shared -- don't let a test-injected fake flow
    leak into other tests (or collide with the real example_login_api)."""
    before = dict(API_FLOW_REGISTRY)
    yield
    API_FLOW_REGISTRY.clear()
    API_FLOW_REGISTRY.update(before)


@pytest.mark.asyncio
async def test_unregistered_flow_name_returns_error():
    tool = ApiRequestFlowTool()
    output = await tool._run(RunApiFlowInput(flow_name="does_not_exist"), None, None)

    body = json.loads(output.get_text_content())
    assert "no API flow registered" in body["error"]
    assert "does_not_exist" in body["error"]


@pytest.mark.asyncio
async def test_registered_flow_that_raises_returns_error():
    def _raising_flow():
        raise RuntimeError("connection refused")

    API_FLOW_REGISTRY["raising_flow"] = _raising_flow
    tool = ApiRequestFlowTool()

    output = await tool._run(RunApiFlowInput(flow_name="raising_flow"), None, None)

    body = json.loads(output.get_text_content())
    assert "raising_flow" in body["error"]
    assert "connection refused" in body["error"]


@pytest.mark.asyncio
async def test_successful_flow_returns_its_result_as_json():
    def _ok_flow():
        return {
            "flow_name": "ok_flow",
            "step_results": [{"step": {"action": "post"}, "status": "ok"}],
            "network_log": [{"method": "POST", "url": "http://x", "status": 200}],
        }

    API_FLOW_REGISTRY["ok_flow"] = _ok_flow
    tool = ApiRequestFlowTool()

    output = await tool._run(RunApiFlowInput(flow_name="ok_flow"), None, None)

    body = json.loads(output.get_text_content())
    assert body["flow_name"] == "ok_flow"
    assert body["step_results"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_run_through_real_tool_run_wrapper_exercises_emitter():
    # Unlike the other tests here (which call _run() directly), this goes
    # through the real Tool.run() wrapper -- the only path that exercises
    # _create_emitter() (a cached_property accessed lazily on first .run()).
    def _ok_flow():
        return {"flow_name": "ok_flow", "step_results": [], "network_log": []}

    API_FLOW_REGISTRY["ok_flow"] = _ok_flow
    tool = ApiRequestFlowTool()

    output = await tool.run({"flow_name": "ok_flow"})

    body = json.loads(output.get_text_content())
    assert body["flow_name"] == "ok_flow"


@pytest.mark.asyncio
async def test_flow_with_failed_steps_still_returns_full_result():
    def _partial_fail_flow():
        return {
            "flow_name": "partial_fail",
            "step_results": [
                {"step": {"action": "post"}, "status": "ok"},
                {"step": {"action": "get"}, "status": "failed", "error": "HTTP 404"},
            ],
            "network_log": [],
        }

    API_FLOW_REGISTRY["partial_fail"] = _partial_fail_flow
    tool = ApiRequestFlowTool()

    output = await tool._run(RunApiFlowInput(flow_name="partial_fail"), None, None)

    body = json.loads(output.get_text_content())
    assert len(body["step_results"]) == 2
    assert body["step_results"][1]["status"] == "failed"
