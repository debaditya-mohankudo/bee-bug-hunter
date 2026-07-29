"""Tests for ConceptStore.upsert()'s merge semantics: a call that only
supplies a subset of fields must not wipe the rest of an existing entry.
Regression coverage for a real incident where an upsert({"name": ..., "related": [...]})
call silently reset an existing concept's description/invariants/evidence/
confidence to empty."""
from concept_store.store import ConceptStore


def _full_concept(name: str = "some-concept") -> dict:
    return {
        "name": name,
        "module": "pkg/mod.py",
        "description": "does a thing",
        "invariants": ["must always X"],
        "contracts": ["returns Y"],
        "confidence": 0.9,
        "evidence": ["pkg/mod.py:1-10"],
        "related": ["other-concept"],
    }


def test_upsert_creates_new_concept_with_all_fields(tmp_path):
    store = ConceptStore(tmp_path / "concepts.json")
    store.upsert(_full_concept())

    saved = store.get("some-concept")
    assert saved["description"] == "does a thing"
    assert saved["invariants"] == ["must always X"]
    assert saved["related"] == ["other-concept"]
    assert saved["confidence"] == 0.9


def test_upsert_full_replace_still_overwrites_every_field(tmp_path):
    store = ConceptStore(tmp_path / "concepts.json")
    store.upsert(_full_concept())

    replacement = _full_concept()
    replacement["description"] = "does a different thing now"
    replacement["invariants"] = []
    store.upsert(replacement)

    saved = store.get("some-concept")
    assert saved["description"] == "does a different thing now"
    assert saved["invariants"] == []  # explicitly cleared -- present in the call, so it's honored


def test_upsert_partial_call_does_not_wipe_omitted_fields(tmp_path):
    """The actual regression: a caller upserting only `related` (e.g. to link
    two concepts after the fact) must not reset description/invariants/
    contracts/evidence/confidence to empty."""
    store = ConceptStore(tmp_path / "concepts.json")
    store.upsert(_full_concept())

    store.upsert({"name": "some-concept", "related": ["other-concept", "third-concept"]})

    saved = store.get("some-concept")
    assert saved["related"] == ["other-concept", "third-concept"]
    assert saved["description"] == "does a thing"
    assert saved["invariants"] == ["must always X"]
    assert saved["contracts"] == ["returns Y"]
    assert saved["evidence"] == ["pkg/mod.py:1-10"]
    assert saved["confidence"] == 0.9


def test_upsert_partial_call_on_new_concept_defaults_omitted_fields(tmp_path):
    """A partial upsert for a name that doesn't exist yet still needs sane
    defaults for whatever wasn't supplied -- there's no existing entry to
    merge onto."""
    store = ConceptStore(tmp_path / "concepts.json")
    store.upsert({"name": "brand-new", "description": "just created"})

    saved = store.get("brand-new")
    assert saved["description"] == "just created"
    assert saved["invariants"] == []
    assert saved["contracts"] == []
    assert saved["evidence"] == []
    assert saved["related"] == []
    assert saved["confidence"] == 0.0


def test_upsert_preserves_created_at_across_updates(tmp_path):
    store = ConceptStore(tmp_path / "concepts.json")
    store.upsert(_full_concept())
    created_at = store.get("some-concept")["created_at"]

    store.upsert({"name": "some-concept", "confidence": 0.99})

    assert store.get("some-concept")["created_at"] == created_at
    assert store.get("some-concept")["confidence"] == 0.99
