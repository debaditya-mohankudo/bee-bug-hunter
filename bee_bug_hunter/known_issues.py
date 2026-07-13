"""In-memory (never persisted to disk) registry of issues confirmed earlier in
*this same* run_batch_once pass, so a later flow can be told "flow X found
this a moment ago" -- context only, never a skip: every flow still gets a full
investigation, and the manager decides for itself whether an earlier finding
is actually relevant (proven in practice: given the same note, one flow
correctly said "yes, this reproduces" and another correctly said "no, this is
unrelated").

Deliberately not indexed by container: a flow's list of containers is an
implementation detail of how it happens to be run, not a reliable signal for
whether two investigations are related -- pre-filtering by container match
just risks silently hiding a genuinely relevant finding (or double-counting
one recorded under several containers) instead of letting the manager, which
has already shown it can tell the difference, make that call itself.

Deliberately not persisted across process runs / poll cycles either: the demo
app's source isn't git-checkpointed, so nothing here can assume the code (or
the bug in it) is the same from one run to the next -- only within a single
batch pass is "the same investigation, moments apart" a safe assumption.
"""
import logging

from bee_bug_hunter.logging_config import get_logger, log

logger = get_logger(__name__)


def compute_fingerprint(anomaly: dict) -> str:
    """A stable, order-independent digest of this cycle's real signals -- not a
    hash, just a sorted pipe-joined string, so it's readable directly in logs."""
    parts = [
        f"http:{e.get('method')}:{e.get('url')}:{e.get('status')}" for e in anomaly.get("http_errors", [])
    ]
    parts += [f"errlog:{c}" for c in anomaly.get("error_containers", [])]
    parts += [f"slow:{c}" for c in anomaly.get("slow_containers", [])]
    return "|".join(sorted(parts)) or "clean"


def record_issue(registry: list, flow_name: str, summary: str) -> None:
    """The only place entries are added -- logged right here, at the append
    itself, so the JSONL log's line order is the true chronology of what was
    known when, regardless of which caller triggered it."""
    registry.append({"flow_name": flow_name, "summary": summary})
    log(logger, logging.INFO, "known_issue_recorded", flow=flow_name, summary=summary, registry_size=len(registry))


def note_for(registry: list) -> str | None:
    """Everything recorded so far this batch pass, regardless of which flow
    is asking -- there's no "own" entry to exclude, since a flow only runs
    once per batch pass."""
    if not registry:
        return None
    return " ".join(f"flow '{e['flow_name']}' found: {e['summary']}" for e in registry)
