"""Tests for api_flows.py: the @api_flow decorator's registry, and
example_login_api()'s direct-request equivalent of the seeded-bug demo flow.
requests.post is monkeypatched -- no live demo_app needed."""
from unittest.mock import MagicMock, patch

from bee_bug_hunter.api_flows import API_FLOW_REGISTRY, api_flow, example_login_api


def test_example_login_api_is_registered_under_its_name():
    assert "example_login_api" in API_FLOW_REGISTRY
    assert API_FLOW_REGISTRY["example_login_api"] is example_login_api


def test_api_flow_decorator_registers_under_given_name():
    @api_flow("my_custom_flow")
    def my_flow():
        return {"flow_name": "my_custom_flow", "step_results": [], "network_log": []}

    assert API_FLOW_REGISTRY["my_custom_flow"] is my_flow
    del API_FLOW_REGISTRY["my_custom_flow"]  # don't leak into other tests


def test_api_flow_decorator_returns_the_original_function_unchanged():
    def my_flow():
        return {}

    decorated = api_flow("passthrough_test")(my_flow)
    assert decorated is my_flow
    del API_FLOW_REGISTRY["passthrough_test"]


def _fake_response(status_code: int, ok: bool, url: str, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok
    resp.url = url
    resp.text = text
    return resp


def test_example_login_api_success_shape():
    with patch("bee_bug_hunter.api_flows.requests.post") as mock_post:
        mock_post.return_value = _fake_response(200, True, "http://localhost:3000/api/auth/login")

        result = example_login_api()

    assert result["flow_name"] == "example_login_api"
    assert result["network_log"] == [{"method": "POST", "url": "http://localhost:3000/api/auth/login", "status": 200}]
    assert result["step_results"] == [{"step": {"action": "post", "path": "/api/auth/login"}, "status": "ok"}]


def test_example_login_api_failure_includes_error_detail():
    with patch("bee_bug_hunter.api_flows.requests.post") as mock_post:
        mock_post.return_value = _fake_response(500, False, "http://localhost:3000/api/auth/login", text="server error")

        result = example_login_api()

    assert result["step_results"][0]["status"] == "failed"
    assert "HTTP 500" in result["step_results"][0]["error"]
    assert "server error" in result["step_results"][0]["error"]


def test_example_login_api_posts_expected_payload():
    with patch("bee_bug_hunter.api_flows.requests.post") as mock_post:
        mock_post.return_value = _fake_response(200, True, "http://localhost:3000/api/auth/login")

        example_login_api()

    mock_post.assert_called_once_with(
        "http://localhost:3000/api/auth/login",
        json={"email": "test@example.com", "password": "password123"},
        timeout=10,
    )
