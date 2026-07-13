"""Tests for bee_bug_hunter.known_issues: fingerprint computation and the
in-memory, per-batch-pass registry that lets a later flow know what an
earlier flow in the same pass already found. No disk persistence, no
container indexing -- the registry is a plain list the caller creates fresh
per run_batch_once pass, and every entry is surfaced to every later flow;
relevance judgment is left to the manager, not pre-filtered here.
"""
from bee_bug_hunter import known_issues


def test_compute_fingerprint_is_clean_when_no_signals():
    anomaly = {"http_errors": [], "error_containers": [], "slow_containers": []}
    assert known_issues.compute_fingerprint(anomaly) == "clean"


def test_compute_fingerprint_reflects_real_signals():
    anomaly = {
        "http_errors": [{"method": "POST", "url": "/api/auth/login", "status": 500}],
        "error_containers": ["demo-api"],
        "slow_containers": [],
    }
    fp = known_issues.compute_fingerprint(anomaly)
    assert "http:POST:/api/auth/login:500" in fp
    assert "errlog:demo-api" in fp


def test_compute_fingerprint_is_order_independent():
    a = {"http_errors": [], "error_containers": ["b", "a"], "slow_containers": []}
    b = {"http_errors": [], "error_containers": ["a", "b"], "slow_containers": []}
    assert known_issues.compute_fingerprint(a) == known_issues.compute_fingerprint(b)


def test_note_for_returns_none_when_registry_empty():
    assert known_issues.note_for([]) is None


def test_note_for_surfaces_an_earlier_flows_finding():
    registry: list = []
    known_issues.record_issue(registry, "example_login_api", "passwd bug in api_login")

    note = known_issues.note_for(registry)

    assert "example_login_api" in note
    assert "passwd bug in api_login" in note


def test_note_for_combines_multiple_entries_without_filtering():
    registry: list = []
    known_issues.record_issue(registry, "example_login_api", "passwd bug")
    known_issues.record_issue(registry, "example_orders", "N+1 query bug")

    note = known_issues.note_for(registry)

    assert "passwd bug" in note
    assert "N+1 query bug" in note


def test_record_issue_appends_once_per_call_not_per_container():
    """Regression test: the earlier container-indexed design appended the same
    entry once per container a flow touched, so a two-container flow's finding
    showed up twice in the note. record_issue no longer takes containers at
    all, so this can't happen -- one call, one entry."""
    registry: list = []
    known_issues.record_issue(registry, "example_login", "passwd bug")

    assert len(registry) == 1
    assert known_issues.note_for(registry).count("passwd bug") == 1
