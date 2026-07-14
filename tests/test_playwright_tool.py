"""Tests for playwright_tool.py's `expect_status`/`expect_text` assertion
steps: both are new step types that let a flow assert on a real response
status or on rendered page text, not just "did a selector eventually
appear" -- a functional bug that renders wrong content or the wrong status
code without ever failing a selector wait would otherwise go undetected by
anomaly_detector.detect()'s failed_steps signal. Playwright itself is fully
mocked (async_playwright, browser, page) so no real browser is launched.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bee_bug_hunter.tools.playwright_tool import _run_flow


def _fake_playwright(page):
    browser = AsyncMock()
    browser.new_page.return_value = page
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
def flows_dir(tmp_path, monkeypatch):
    from bee_bug_hunter.tools import playwright_tool

    monkeypatch.setattr(playwright_tool, "FLOWS_DIR", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_expect_status_passes_when_status_matches(flows_dir):
    (flows_dir / "flow.yaml").write_text(
        "base_url: http://x\nsteps:\n"
        "  - action: expect_status\n    url_contains: /api/login\n    status: 200\n"
    )
    page = _make_page()
    response = MagicMock(status=200)
    page.wait_for_event.return_value = response

    with patch("bee_bug_hunter.tools.playwright_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_flow("flow"))

    assert result["step_results"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_expect_status_fails_when_status_does_not_match(flows_dir):
    (flows_dir / "flow.yaml").write_text(
        "base_url: http://x\nsteps:\n"
        "  - action: expect_status\n    url_contains: /api/login\n    status: 200\n"
    )
    page = _make_page()
    response = MagicMock(status=500)
    page.wait_for_event.return_value = response

    with patch("bee_bug_hunter.tools.playwright_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_flow("flow"))

    assert result["step_results"][0]["status"] == "failed"
    assert "500" in result["step_results"][0]["error"]


@pytest.mark.asyncio
async def test_expect_status_accepts_status_in_list(flows_dir):
    (flows_dir / "flow.yaml").write_text(
        "base_url: http://x\nsteps:\n"
        "  - action: expect_status\n    url_contains: /api/login\n    status_in: [200, 201]\n"
    )
    page = _make_page()
    response = MagicMock(status=201)
    page.wait_for_event.return_value = response

    with patch("bee_bug_hunter.tools.playwright_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_flow("flow"))

    assert result["step_results"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_expect_text_passes_when_text_contains(flows_dir):
    (flows_dir / "flow.yaml").write_text(
        "base_url: http://x\nsteps:\n"
        "  - action: expect_text\n    selector: '#welcome'\n    contains: Hello\n"
    )
    page = _make_page()
    locator = AsyncMock()
    locator.inner_text.return_value = "Hello, test user"
    page.locator = MagicMock(return_value=locator)

    with patch("bee_bug_hunter.tools.playwright_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_flow("flow"))

    assert result["step_results"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_expect_text_fails_when_text_does_not_contain(flows_dir):
    (flows_dir / "flow.yaml").write_text(
        "base_url: http://x\nsteps:\n"
        "  - action: expect_text\n    selector: '#welcome'\n    contains: Hello\n"
    )
    page = _make_page()
    locator = AsyncMock()
    locator.inner_text.return_value = "Goodbye"
    page.locator = MagicMock(return_value=locator)

    with patch("bee_bug_hunter.tools.playwright_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_flow("flow"))

    assert result["step_results"][0]["status"] == "failed"
    assert "Goodbye" in result["step_results"][0]["error"]


@pytest.mark.asyncio
async def test_expect_text_equals_mismatch_fails(flows_dir):
    (flows_dir / "flow.yaml").write_text(
        "base_url: http://x\nsteps:\n"
        "  - action: expect_text\n    selector: '#status'\n    equals: Active\n"
    )
    page = _make_page()
    locator = AsyncMock()
    locator.inner_text.return_value = "Inactive"
    page.locator = MagicMock(return_value=locator)

    with patch("bee_bug_hunter.tools.playwright_tool.async_playwright", _fake_playwright(page)):
        result = json.loads(await _run_flow("flow"))

    assert result["step_results"][0]["status"] == "failed"
