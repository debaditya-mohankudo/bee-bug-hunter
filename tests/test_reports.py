"""Tests for reports.save_report(): persists one markdown file per flow run,
named so runs sort chronologically and correlate with the JSONL log by
run_id. Specialist sections (Bug Analyst / SQL Performance) take precedence
over the manager's own summary."""
from bee_bug_hunter.reports import save_report


def _base_result(**overrides) -> dict:
    result = {
        "flow": "demo_login",
        "run_id": "abc123",
        "response": "SUMMARY: clean\n\nfull manager report text",
        "anomaly": {"bug_signal": False, "perf_signal": False},
        "bug_report": None,
        "perf_report": None,
    }
    result.update(overrides)
    return result


def test_writes_to_reports_dir_and_returns_path(tmp_path):
    path = save_report(_base_result(), reports_dir=str(tmp_path))
    assert path.startswith(str(tmp_path))
    assert path.endswith(".md")


def test_creates_reports_dir_if_missing(tmp_path):
    target = tmp_path / "nested" / "reports"
    save_report(_base_result(), reports_dir=str(target))
    assert target.is_dir()


def test_filename_contains_flow_and_run_id(tmp_path):
    path = save_report(_base_result(flow="my_flow", run_id="run42"), reports_dir=str(tmp_path))
    assert "my_flow" in path
    assert "run42" in path


def test_header_includes_run_id_and_signals(tmp_path):
    result = _base_result(anomaly={"bug_signal": True, "perf_signal": False})
    path = save_report(result, reports_dir=str(tmp_path))
    content = open(path).read()
    assert "# demo_login" in content
    assert "- run_id: abc123" in content
    assert "- bug_signal: True" in content
    assert "- perf_signal: False" in content


def test_bug_report_section_written_when_present(tmp_path):
    result = _base_result(bug_report="root cause: passwd column typo")
    path = save_report(result, reports_dir=str(tmp_path))
    content = open(path).read()
    assert "## Bug Analyst report" in content
    assert "root cause: passwd column typo" in content
    assert "## Manager summary" not in content


def test_perf_report_section_written_when_present(tmp_path):
    result = _base_result(perf_report="missing index on orders.user_id")
    path = save_report(result, reports_dir=str(tmp_path))
    content = open(path).read()
    assert "## SQL Performance Agent report" in content
    assert "missing index on orders.user_id" in content
    assert "## Manager summary" not in content


def test_both_specialist_sections_written_when_both_present(tmp_path):
    result = _base_result(bug_report="bug finding", perf_report="perf finding")
    path = save_report(result, reports_dir=str(tmp_path))
    content = open(path).read()
    assert "## Bug Analyst report" in content
    assert "## SQL Performance Agent report" in content
    assert "## Manager summary" not in content


def test_manager_summary_written_when_neither_specialist_present(tmp_path):
    result = _base_result(bug_report=None, perf_report=None, response="SUMMARY: clean\n\nnothing found")
    path = save_report(result, reports_dir=str(tmp_path))
    content = open(path).read()
    assert "## Manager summary" in content
    assert "nothing found" in content
    assert "## Bug Analyst report" not in content


def test_falsy_but_present_report_strings_are_treated_as_absent(tmp_path):
    # Empty string is falsy -- same branch as None, per the `if result.get(...)` check.
    result = _base_result(bug_report="", perf_report="")
    path = save_report(result, reports_dir=str(tmp_path))
    content = open(path).read()
    assert "## Manager summary" in content
