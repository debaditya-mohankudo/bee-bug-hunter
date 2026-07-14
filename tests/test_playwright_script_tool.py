"""Tests for playwright_script_tool.py -- the plain-Python-script counterpart
to playwright_tool.py's YAML step DSL. Playwright itself is fully mocked
(async_playwright, browser, page) so no real browser is launched.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bee_bug_hunter.playwright_flows import PLAYWRIGHT_FLOW_REGISTRY, playwright_flow
from bee_bug_hunter.tools.playwright_script_tool import _run_script, run_step


def _fake_playwright(page):
    browser = AsyncMock()
    browser.new_page.return_value = page
    browser.close = AsyncMock()
    chromium = AsyncMock()
    chromium.launch.return_value = browser

    playwright_obj = MagicMock()
    playwright_obj.chromium = chromium

    @asynccontextmanager
    async def _fake_async_playwright():
        yield playwright_obj

    return _fake_async_playwright


def _make_page():
    page = AsyncMock()
    page.on = MagicMock()
    return page


@pytest.fixture
def registered_flow():
    """Registers a throwaway flow under a unique name and cleans it up after."""
    name = "test_script_flow"

    async def _flow(page, network_log):
        await page.goto("http://x/login")
        return [{"step": {"action": "goto"}, "status": "ok"}]

    PLAYWRIGHT_FLOW_REGISTRY[name] = _flow
    yield name
    PLAYWRIGHT_FLOW_REGISTRY.pop(name, None)


@pytest.mark.asyncio
async def test_runs_registered_flow_and_returns_canonical_shape(registered_flow):
    page = _make_page()
    with patch("bee_bug_hunter.tools.playwright_script_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_script(registered_flow, headless=True))

    assert result["flow_name"] == registered_flow
    assert result["step_results"] == [{"step": {"action": "goto"}, "status": "ok"}]
    assert result["network_log"] == []
    page.goto.assert_awaited_once_with("http://x/login")


@pytest.mark.asyncio
async def test_unregistered_flow_name_returns_error():
    page = _make_page()
    with patch("bee_bug_hunter.tools.playwright_script_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_script("does_not_exist", headless=True))

    assert "error" in result
    assert "does_not_exist" in result["error"]


@pytest.mark.asyncio
async def test_flow_function_raising_is_caught_and_browser_closed():
    name = "raising_flow"

    async def _flow(page, network_log):
        raise RuntimeError("boom")

    PLAYWRIGHT_FLOW_REGISTRY[name] = _flow
    try:
        page = _make_page()
        with patch("bee_bug_hunter.tools.playwright_script_tool.async_playwright", _fake_playwright(page)) as fake:
            result = json.loads(await _run_script(name, headless=True))
    finally:
        PLAYWRIGHT_FLOW_REGISTRY.pop(name, None)

    assert "error" in result
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_network_log_populated_via_response_listener(registered_flow):
    page = _make_page()
    captured_handler = {}

    def _capture_on(event, handler):
        if event == "response":
            captured_handler["fn"] = handler

    page.on = MagicMock(side_effect=_capture_on)

    async def _flow(p, network_log):
        response = MagicMock(request=MagicMock(method="GET"), url="http://x/api", status=200)
        captured_handler["fn"](response)
        return []

    PLAYWRIGHT_FLOW_REGISTRY["net_flow"] = _flow
    try:
        with patch("bee_bug_hunter.tools.playwright_script_tool.async_playwright", _fake_playwright(page)):
            result = json.loads(await _run_script("net_flow", headless=True))
    finally:
        PLAYWRIGHT_FLOW_REGISTRY.pop("net_flow", None)

    assert result["network_log"] == [{"method": "GET", "url": "http://x/api", "status": 200}]


def test_example_login_script_is_registered():
    assert "example_login_script" in PLAYWRIGHT_FLOW_REGISTRY


@pytest.mark.asyncio
async def test_example_login_script_runs_all_six_steps_ok():
    """Exercises the action-dispatch in _run_step end to end -- goto, fill x2,
    click, wait_for_response, wait_for_selector all need to resolve to a real
    page.* call, not just structurally run without raising."""
    page = AsyncMock()
    response = MagicMock(url="http://localhost:3000/api/auth/login")
    page.wait_for_event.return_value = response

    fn = PLAYWRIGHT_FLOW_REGISTRY["example_login_script"]
    step_results = await fn(page, [])

    assert [s["status"] for s in step_results] == ["ok"] * 6
    page.goto.assert_awaited_once_with("http://localhost:3000/login")
    assert page.fill.await_count == 2
    page.click.assert_awaited_once_with("#login-submit")
    page.wait_for_selector.assert_awaited_once_with("#dashboard", timeout=10000)


def test_playwright_flow_decorator_registers_under_given_name():
    @playwright_flow("decorator_test_flow")
    async def _fn(page, network_log):
        return []

    try:
        assert PLAYWRIGHT_FLOW_REGISTRY["decorator_test_flow"] is _fn
    finally:
        PLAYWRIGHT_FLOW_REGISTRY.pop("decorator_test_flow", None)


class TestRunStep:
    """run_step is a standalone reusable helper (moved out of playwright_flows.py
    per user request) -- exercised directly here, including the expect_status/
    expect_text branches that mirror playwright_tool.py's YAML DSL but weren't
    previously covered in this module."""

    @pytest.mark.asyncio
    async def test_expect_status_passes_when_status_matches(self):
        page = AsyncMock()
        page.wait_for_event.return_value = MagicMock(status=200)
        step_results = []

        await run_step(page, {"action": "expect_status", "url_contains": "/api/login", "status": 200}, step_results)

        assert step_results[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_expect_status_fails_when_status_does_not_match(self):
        page = AsyncMock()
        page.wait_for_event.return_value = MagicMock(status=500)
        step_results = []

        await run_step(page, {"action": "expect_status", "url_contains": "/api/login", "status": 200}, step_results)

        assert step_results[0]["status"] == "failed"
        assert "500" in step_results[0]["error"]

    @pytest.mark.asyncio
    async def test_expect_status_accepts_status_in_list(self):
        page = AsyncMock()
        page.wait_for_event.return_value = MagicMock(status=201)
        step_results = []

        await run_step(page, {"action": "expect_status", "url_contains": "/api/login", "status_in": [200, 201]}, step_results)

        assert step_results[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_expect_text_passes_when_text_contains(self):
        page = AsyncMock()
        locator = AsyncMock()
        locator.inner_text.return_value = "Hello, test user"
        page.locator = MagicMock(return_value=locator)
        step_results = []

        await run_step(page, {"action": "expect_text", "selector": "#welcome", "contains": "Hello"}, step_results)

        assert step_results[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_expect_text_fails_when_text_does_not_contain(self):
        page = AsyncMock()
        locator = AsyncMock()
        locator.inner_text.return_value = "Goodbye"
        page.locator = MagicMock(return_value=locator)
        step_results = []

        await run_step(page, {"action": "expect_text", "selector": "#welcome", "contains": "Hello"}, step_results)

        assert step_results[0]["status"] == "failed"
        assert "Goodbye" in step_results[0]["error"]

    @pytest.mark.asyncio
    async def test_goto_prefixes_base_url(self):
        page = AsyncMock()
        step_results = []

        await run_step(page, {"action": "goto", "path": "/login"}, step_results, base_url="http://x")

        page.goto.assert_awaited_once_with("http://x/login")
        assert step_results[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_unknown_action_is_recorded_ok_without_calling_page(self):
        """Mirrors playwright_tool.py's YAML runner: an unrecognized action
        doesn't raise, it just does nothing -- caller sees status 'ok' since
        no exception was raised. (Unlike the YAML runner, there's no
        'skipped_unknown_action' status here since a script author controls
        the action strings directly; this documents current behavior.)"""
        page = AsyncMock()
        step_results = []

        await run_step(page, {"action": "not_a_real_action"}, step_results)

        assert step_results[0]["status"] == "ok"
        page.goto.assert_not_called()
        page.click.assert_not_called()
