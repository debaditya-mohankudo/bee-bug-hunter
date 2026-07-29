"""Tests for anomaly_detector.detect() -- pure, deterministic routing logic:
HTTP >=400 / failed steps / ERROR-ish log lines set bug_signal; a '<n> ms'
duration >= SLOW_QUERY_MS_THRESHOLD sets perf_signal. No I/O, no framework
dependency."""
import json

from bee_bug_hunter.anomaly_detector import (
    SLOW_QUERY_MS_THRESHOLD,
    AnomalySignals,
    detect,
)


def _flow(network_log=None, step_results=None) -> str:
    return json.dumps({"network_log": network_log or [], "step_results": step_results or []})


def _logs(**container_content: str) -> str:
    return json.dumps({name: {"content": content} for name, content in container_content.items()})


def test_clean_run_has_no_signals():
    signals = detect(_flow(), _logs())
    assert signals.bug_signal is False
    assert signals.perf_signal is False
    assert signals.http_errors == []
    assert signals.failed_steps == []
    assert signals.error_containers == []
    assert signals.slow_containers == []


def test_http_error_sets_bug_signal():
    flow = _flow(network_log=[{"status": 200}, {"status": 500}])
    signals = detect(flow, _logs())
    assert signals.bug_signal is True
    assert signals.http_errors == [{"status": 500}]


def test_http_status_below_400_is_not_an_error():
    flow = _flow(network_log=[{"status": 399}])
    signals = detect(flow, _logs())
    assert signals.bug_signal is False
    assert signals.http_errors == []


def test_failed_step_sets_bug_signal():
    flow = _flow(step_results=[{"status": "ok"}, {"status": "failed"}])
    signals = detect(flow, _logs())
    assert signals.bug_signal is True
    assert signals.failed_steps == [{"status": "failed"}]


def test_error_log_line_sets_bug_signal_and_error_containers():
    logs = _logs(web="normal line\nERROR something broke")
    signals = detect(_flow(), logs)
    assert signals.bug_signal is True
    assert signals.error_containers == ["web"]


def test_exception_traceback_fatal_all_match_error_pattern():
    for keyword in ("Exception", "Traceback", "FATAL"):
        logs = _logs(web=f"line containing {keyword} in it")
        signals = detect(_flow(), logs)
        assert signals.error_containers == ["web"], f"expected {keyword!r} to match"


def test_duration_at_or_above_threshold_sets_perf_signal():
    logs = _logs(db=f"query took {SLOW_QUERY_MS_THRESHOLD} ms")
    signals = detect(_flow(), logs)
    assert signals.perf_signal is True
    assert signals.slow_containers == ["db"]


def test_duration_below_threshold_does_not_set_perf_signal():
    logs = _logs(db=f"query took {SLOW_QUERY_MS_THRESHOLD - 1} ms")
    signals = detect(_flow(), logs)
    assert signals.perf_signal is False
    assert signals.slow_containers == []


def test_slow_containers_deduplicated_and_sorted():
    logs = _logs(
        web=f"took {SLOW_QUERY_MS_THRESHOLD} ms then took {SLOW_QUERY_MS_THRESHOLD + 100} ms",
        api=f"took {SLOW_QUERY_MS_THRESHOLD} ms",
    )
    signals = detect(_flow(), logs)
    # "web" would appear twice without the dedup (two matching durations in
    # one container's content) -- sorted() also confirms ordering isn't
    # container-insertion-order dependent.
    assert signals.slow_containers == ["api", "web"]


def test_malformed_flow_json_degrades_to_empty_not_raise():
    signals = detect("not json at all", _logs())
    assert signals.bug_signal is False
    assert signals.http_errors == []


def test_malformed_log_json_degrades_to_empty_not_raise():
    signals = detect(_flow(), "not json at all")
    assert signals.bug_signal is False
    assert signals.error_containers == []


def test_non_string_input_degrades_to_empty_not_raise():
    # TypeError path in the try/except (json.loads on a non-str/bytes input).
    signals = detect(None, None)
    assert signals.bug_signal is False
    assert signals.perf_signal is False


def test_log_entry_that_is_not_a_dict_is_ignored_not_raise():
    logs = json.dumps({"web": "not a dict, just a string"})
    signals = detect(_flow(), logs)
    assert signals.error_containers == []
    assert signals.slow_containers == []


def test_both_bug_and_perf_signal_can_fire_together():
    flow = _flow(network_log=[{"status": 500}])
    logs = _logs(db=f"took {SLOW_QUERY_MS_THRESHOLD} ms")
    signals = detect(flow, logs)
    assert signals.bug_signal is True
    assert signals.perf_signal is True


def test_to_dict_has_all_six_stable_keys():
    signals = AnomalySignals()
    d = signals.to_dict()
    assert set(d.keys()) == {
        "bug_signal", "perf_signal", "http_errors", "failed_steps",
        "error_containers", "slow_containers",
    }
