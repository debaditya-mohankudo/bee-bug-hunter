"""ConceptStore — JSON-backed store for architectural concepts extracted from bee-bug-hunter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_FILENAME = "concepts.json"


class ConceptStore:
    """Stores architectural concepts as a JSON file keyed by concept name.

    Each concept:
        name        — unique slug (e.g. "manager-delegation-only-supervisor")
        module      — source file (e.g. "bee_bug_hunter/manager.py")
        description — what this module/concept does architecturally
        invariants  — list[str] of constraints that must always hold
        contracts   — list[str] of promises to callers
        confidence  — float 0.0–1.0
        evidence    — list[str] of "file:line" references
        related     — list[str] of other concept names this one is coupled to
                      (same relationship the "related" column plays for memories
                      in MEMORY.sqlite — see export_graph.py)
        last_validated — ISO timestamp
        created_at     — ISO timestamp
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        # Store-level metadata (commit the concepts were extracted at, extraction
        # timestamp) -- same top-level {"meta": ..., "concepts": ...} wrapper as the
        # ACME_Cert_Life_Cycle concept store, so tooling can treat both alike.
        self._meta: dict = {"commit": "", "extracted_at": "", "note": ""}
        if self._path.exists():
            text = self._path.read_text(encoding="utf-8").strip()
            raw = json.loads(text) if text else {}
            if "concepts" in raw:
                self._data = raw["concepts"]
                self._meta.update(raw.get("meta", {}))
            else:
                # Legacy flat layout ({name: concept}) from before the meta wrapper.
                self._data = raw

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    _FIELDS = ("module", "description", "invariants", "contracts", "confidence", "evidence", "related")
    _FIELD_DEFAULTS = {"invariants": [], "contracts": [], "evidence": [], "related": [], "confidence": 0.0, "module": "", "description": ""}

    def upsert(self, concept: dict) -> None:
        """Merges onto any existing entry for this name -- only fields actually
        present in `concept` are overwritten; a field simply omitted from the
        call (as opposed to explicitly passed as "" / [] / 0.0) keeps its prior
        value rather than being reset to empty. A full-replace call (all fields
        supplied) still behaves exactly as a full overwrite, matching this
        format's documented "upsert() fully overwrites" semantics for in-place
        correction -- this only fixes the case where a caller only means to
        touch a subset of fields (e.g. a `related` link-up pass) and would
        otherwise silently wipe the rest of the concept."""
        name = concept["name"]
        now = datetime.now(timezone.utc).isoformat()
        existing = self._data.get(name, {})
        merged = dict(existing)
        merged["name"] = name
        for field in self._FIELDS:
            if field in concept:
                merged[field] = concept[field]
            elif field not in merged:
                merged[field] = self._FIELD_DEFAULTS[field]
        merged["last_validated"] = now
        merged["created_at"] = existing.get("created_at", now)
        self._data[name] = merged
        self.save()

    def delete(self, name: str) -> None:
        self._data.pop(name, None)
        self.save()

    def save(self) -> None:
        payload = {"meta": self._meta, "concepts": self._data}
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def set_meta(self, **fields) -> None:
        """Merge store-level metadata (e.g. commit=<sha>, extracted_at=<iso>) and persist."""
        self._meta.update(fields)
        self.save()

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[dict]:
        return self._data.get(name)

    def list(self, module: Optional[str] = None) -> list[dict]:
        concepts = list(self._data.values())
        if module is not None:
            concepts = [c for c in concepts if c.get("module") == module]
        return concepts

    def modules(self) -> list[str]:
        return sorted({c.get("module", "") for c in self._data.values() if c.get("module")})

    def __len__(self) -> int:
        return len(self._data)
