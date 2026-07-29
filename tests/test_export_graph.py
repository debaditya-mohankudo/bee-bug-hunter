"""Tests for concept_store/export_graph.py: exports concepts.json to one
JSON file per concept plus an adjacency-list _graph.json.

CRITICAL: export()/load_graph()/neighbours()/most_connected() all reference
module-level _GRAPH_DIR/_GRAPH_OUT Path constants computed ONCE at import
time from the REAL repo root -- export(store_path=...) only overrides the
INPUT concepts.json location, not the OUTPUT graph/ directory. Every test
here monkeypatches _GRAPH_DIR and _GRAPH_OUT to tmp_path locations FIRST,
or it would silently overwrite this repo's real checked-in
concept_store/graph/*.json and _graph.json."""
import pytest

from concept_store import export_graph
from concept_store.store import ConceptStore


@pytest.fixture(autouse=True)
def _isolated_graph_output(tmp_path, monkeypatch):
    # _REPO_ROOT is a THIRD module-level constant beyond _GRAPH_DIR/_GRAPH_OUT
    # -- export()'s own status prints do _GRAPH_DIR.relative_to(_REPO_ROOT),
    # which raises ValueError unless _REPO_ROOT is also redirected under the
    # same tmp_path tree. Found by running this test file for real, not by
    # reading the source alone -- the failure mode was a loud exception, not
    # a silent real-file write, which is the good outcome of a mistake here.
    graph_dir = tmp_path / "graph"
    graph_out = tmp_path / "_graph.json"
    monkeypatch.setattr(export_graph, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(export_graph, "_GRAPH_DIR", graph_dir)
    monkeypatch.setattr(export_graph, "_GRAPH_OUT", graph_out)
    return graph_dir, graph_out


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "concepts.json"


def _seed_store(path, *concepts: dict) -> ConceptStore:
    store = ConceptStore(path)
    for c in concepts:
        store.upsert(c)
    return store


class TestExport:
    def test_writes_one_file_per_concept(self, store_path, tmp_path):
        _seed_store(
            store_path,
            {"name": "concept-a", "module": "a.py", "description": "does a"},
            {"name": "concept-b", "module": "b.py", "description": "does b"},
        )

        export_graph.export(store_path=store_path)

        graph_dir = tmp_path / "graph"
        assert (graph_dir / "concept-a.json").exists()
        assert (graph_dir / "concept-b.json").exists()

    def test_written_node_file_has_expected_shape(self, store_path, tmp_path):
        _seed_store(store_path, {
            "name": "concept-a", "module": "a.py", "description": "does a",
            "invariants": ["always true"], "contracts": ["returns X"],
            "confidence": 0.9, "evidence": ["a.py:1-5"], "related": ["concept-b"],
        })

        export_graph.export(store_path=store_path)

        import json
        node = json.loads((tmp_path / "graph" / "concept-a.json").read_text())
        assert node["name"] == "concept-a"
        assert node["module"] == "a.py"
        assert node["invariants"] == ["always true"]
        assert node["related"] == ["concept-b"]
        assert node["confidence"] == 0.9

    def test_creates_graph_dir_if_missing(self, store_path, tmp_path):
        assert not (tmp_path / "graph").exists()
        _seed_store(store_path, {"name": "concept-a", "module": "a.py", "description": "d"})

        export_graph.export(store_path=store_path)

        assert (tmp_path / "graph").is_dir()

    def test_writes_graph_json_with_node_and_edge_counts(self, store_path, tmp_path):
        _seed_store(
            store_path,
            {"name": "concept-a", "module": "a.py", "description": "d", "related": ["concept-b"]},
            {"name": "concept-b", "module": "b.py", "description": "d", "related": []},
        )

        graph = export_graph.export(store_path=store_path)

        assert graph["node_count"] == 2
        assert graph["edge_count"] == 1
        assert graph["adjacency"] == {"concept-a": ["concept-b"], "concept-b": []}

    def test_dangling_edge_excluded_from_adjacency_and_edge_count(self, store_path, capsys):
        _seed_store(store_path, {
            "name": "concept-a", "module": "a.py", "description": "d",
            "related": ["concept-that-does-not-exist"],
        })

        graph = export_graph.export(store_path=store_path)

        assert graph["edge_count"] == 0
        assert graph["adjacency"] == {"concept-a": []}
        printed = capsys.readouterr().out
        assert "1 dangling edge" in printed
        assert "concept-a -> concept-that-does-not-exist" in printed

    def test_no_dangling_edges_no_warning_printed(self, store_path, capsys):
        _seed_store(store_path, {"name": "concept-a", "module": "a.py", "description": "d"})

        export_graph.export(store_path=store_path)

        assert "dangling" not in capsys.readouterr().out

    def test_repo_label_falls_back_to_bee_bug_hunter_when_no_commit_set(self, store_path):
        _seed_store(store_path, {"name": "concept-a", "module": "a.py", "description": "d"})

        graph = export_graph.export(store_path=store_path)

        assert graph["repo"] == "bee-bug-hunter"
        assert graph["commit"] == ""

    def test_repo_label_uses_short_commit_when_set(self, store_path):
        store = _seed_store(store_path, {"name": "concept-a", "module": "a.py", "description": "d"})
        store.set_meta(commit="1234567890abcdef")

        graph = export_graph.export(store_path=store_path)

        assert graph["repo"] == "12345678"
        assert graph["commit"] == "1234567890abcdef"

    def test_empty_store_exports_zero_nodes(self, store_path):
        _seed_store(store_path)  # no concepts at all

        graph = export_graph.export(store_path=store_path)

        assert graph["node_count"] == 0
        assert graph["edge_count"] == 0
        assert graph["adjacency"] == {}


class TestLoadGraph:
    def test_reads_back_what_export_wrote(self, store_path):
        _seed_store(store_path, {"name": "concept-a", "module": "a.py", "description": "d"})
        export_graph.export(store_path=store_path)

        loaded = export_graph.load_graph()

        assert loaded["node_count"] == 1
        assert loaded["adjacency"] == {"concept-a": []}


class TestNeighbours:
    def test_one_hop(self, store_path):
        _seed_store(
            store_path,
            {"name": "a", "module": "a.py", "description": "d", "related": ["b"]},
            {"name": "b", "module": "b.py", "description": "d", "related": ["c"]},
            {"name": "c", "module": "c.py", "description": "d", "related": []},
        )
        export_graph.export(store_path=store_path)

        assert export_graph.neighbours("a", hops=1) == {"b"}

    def test_two_hops_includes_transitive_neighbour(self, store_path):
        _seed_store(
            store_path,
            {"name": "a", "module": "a.py", "description": "d", "related": ["b"]},
            {"name": "b", "module": "b.py", "description": "d", "related": ["c"]},
            {"name": "c", "module": "c.py", "description": "d", "related": []},
        )
        export_graph.export(store_path=store_path)

        assert export_graph.neighbours("a", hops=2) == {"b", "c"}

    def test_does_not_include_self(self, store_path):
        _seed_store(
            store_path,
            {"name": "a", "module": "a.py", "description": "d", "related": ["a"]},  # self-reference
        )
        export_graph.export(store_path=store_path)

        assert export_graph.neighbours("a", hops=1) == set()

    def test_node_with_no_relations_has_no_neighbours(self, store_path):
        _seed_store(store_path, {"name": "isolated", "module": "a.py", "description": "d"})
        export_graph.export(store_path=store_path)

        assert export_graph.neighbours("isolated", hops=1) == set()


class TestMostConnected:
    def test_ranks_by_out_degree_descending(self, store_path):
        _seed_store(
            store_path,
            {"name": "hub", "module": "a.py", "description": "d", "related": ["a", "b", "c"]},
            {"name": "a", "module": "a.py", "description": "d", "related": []},
            {"name": "b", "module": "b.py", "description": "d", "related": ["a"]},
            {"name": "c", "module": "c.py", "description": "d", "related": []},
        )
        export_graph.export(store_path=store_path)

        ranked = export_graph.most_connected(top_n=2)

        assert ranked[0] == ("hub", 3)
        assert ranked[1][1] == 1  # "b" -> ["a"]

    def test_top_n_limits_results(self, store_path):
        _seed_store(
            store_path,
            {"name": "a", "module": "a.py", "description": "d", "related": ["b"]},
            {"name": "b", "module": "b.py", "description": "d", "related": ["c"]},
            {"name": "c", "module": "c.py", "description": "d", "related": []},
        )
        export_graph.export(store_path=store_path)

        assert len(export_graph.most_connected(top_n=1)) == 1
