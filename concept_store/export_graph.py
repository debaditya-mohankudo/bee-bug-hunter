"""Export bee-bug-hunter concepts to concept_store/graph/<name>.json — one file
per node. Mirrors claude-hooks' scripts/export_memory_graph.py, but reads from
concept_store/concepts.json (via ConceptStore) instead of MEMORY.sqlite.

Each file is a self-contained concept node. The `related` list contains
concept names that resolve to sibling files in the same directory.

Also writes concept_store/_graph.json — adjacency list for fast traversal.

Usage:
    python3 concept_store/export_graph.py
"""
from __future__ import annotations

import json
from pathlib import Path

from concept_store.store import ConceptStore

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_STORE_PATH  = _REPO_ROOT / "concept_store" / "concepts.json"
_GRAPH_DIR   = _REPO_ROOT / "concept_store" / "graph"
_GRAPH_OUT   = _REPO_ROOT / "concept_store" / "_graph.json"


def export(store_path: Path | None = None) -> dict:
    store_path = store_path or _STORE_PATH
    store = ConceptStore(store_path)
    concepts = store.list()

    _GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    names = {c["name"] for c in concepts}
    edges: list[dict] = []

    for concept in concepts:
        related = list(concept.get("related") or [])
        node = {
            "name":           concept["name"],
            "module":         concept.get("module", ""),
            "description":    concept.get("description", ""),
            "invariants":     concept.get("invariants", []),
            "contracts":      concept.get("contracts", []),
            "confidence":     concept.get("confidence", 0.0),
            "evidence":       concept.get("evidence", []),
            "related":        related,
            "last_validated": concept.get("last_validated", ""),
        }
        path = _GRAPH_DIR / f"{concept['name']}.json"
        path.write_text(json.dumps(node, indent=2, ensure_ascii=False) + "\n")

        for target in related:
            edges.append({"source": concept["name"], "target": target, "target_exists": target in names})

    # Adjacency list for graph traversal
    adjacency: dict[str, list[str]] = {c["name"]: [] for c in concepts}
    for e in edges:
        if e["target_exists"]:
            adjacency[e["source"]].append(e["target"])

    graph = {
        "repo":       store.meta.get("commit", "")[:8] or "bee-bug-hunter",
        "commit":     store.meta.get("commit", ""),
        "node_count": len(concepts),
        "edge_count": len([e for e in edges if e["target_exists"]]),
        "adjacency":  adjacency,
    }
    _GRAPH_OUT.write_text(json.dumps(graph, indent=2) + "\n")

    print(f"✓ {graph['node_count']} nodes, {graph['edge_count']} edges")
    print(f"  nodes → {_GRAPH_DIR.relative_to(_REPO_ROOT)}/<name>.json")
    print(f"  graph → {_GRAPH_OUT.relative_to(_REPO_ROOT)}")

    dangling = [e for e in edges if not e["target_exists"]]
    if dangling:
        print(f"  WARNING: {len(dangling)} dangling edge(s) to unknown concepts:")
        for e in dangling:
            print(f"    {e['source']} -> {e['target']}")

    return graph


def load_graph() -> dict:
    """Load adjacency list from _graph.json for traversal."""
    return json.loads(_GRAPH_OUT.read_text())


def neighbours(name: str, hops: int = 1) -> set[str]:
    """Return all concept names within N hops of name."""
    adj = load_graph()["adjacency"]
    visited, frontier = {name}, {name}
    for _ in range(hops):
        next_frontier = set()
        for s in frontier:
            for t in adj.get(s, []):
                if t not in visited:
                    next_frontier.add(t)
        visited |= next_frontier
        frontier = next_frontier
    return visited - {name}


def most_connected(top_n: int = 10) -> list[tuple[str, int]]:
    """Return top-N concepts by out-degree (number of related links)."""
    adj = load_graph()["adjacency"]
    ranked = sorted(adj.items(), key=lambda x: -len(x[1]))
    return [(name, len(links)) for name, links in ranked[:top_n]]


if __name__ == "__main__":
    export()
