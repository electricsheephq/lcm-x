"""#635: summary projections must be independent of physical column order."""

import re
import sqlite3
from pathlib import Path

import pytest

import hermes_lcm.dag
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.scope_storage import ensure_scope_columns


LEGACY_SUMMARY_NODES = """
CREATE TABLE summary_nodes (
    node_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    depth INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    source_token_count INTEGER DEFAULT 0,
    source_ids TEXT NOT NULL DEFAULT '[]',
    source_type TEXT NOT NULL DEFAULT 'messages',
    created_at REAL NOT NULL,
    expand_hint TEXT DEFAULT ''
);
"""


def _legacy_db(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(LEGACY_SUMMARY_NODES)
        conn.execute(
            "INSERT INTO summary_nodes (session_id, depth, summary, token_count,"
            " source_token_count, source_ids, source_type, created_at, expand_hint)"
            " VALUES ('s1', 0, 'pre-upgrade summary', 5, 50, '[1,2]', 'messages',"
            " 1000.0, 'Expand for details about: legacy')"
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _window_node(**kw):
    values = dict(
        session_id="s1", depth=0, summary="deploy rollback notes",
        token_count=5, source_token_count=50, source_ids=[3, 4],
        created_at=30.0, earliest_at=10.0, latest_at=20.0,
        expand_hint="Expand details",
    )
    values.update(kw)
    return SummaryNode(**values)


def test_new_node_round_trips_on_migrated_legacy_table(tmp_path):
    dag = SummaryDAG(_legacy_db(tmp_path))
    try:
        node_id = dag.add_node(SummaryNode(
            session_id="s1", depth=0, summary="post-upgrade summary",
            token_count=5, source_token_count=50, source_ids=[3, 4],
            created_at=30.0, earliest_at=10.0, latest_at=20.0,
            expand_hint="Expand details",
        ))
        node = dag.get_node(node_id)
        assert (node.earliest_at, node.latest_at, node.expand_hint) == (
            10.0, 20.0, "Expand details"
        )
    finally:
        dag.close()


def test_pre_upgrade_node_keeps_its_expand_hint(tmp_path):
    dag = SummaryDAG(_legacy_db(tmp_path))
    try:
        node = dag.get_node(1)
        assert node.expand_hint == "Expand for details about: legacy"
        assert node.earliest_at is None and node.latest_at is None
    finally:
        dag.close()


@pytest.mark.parametrize("reader", [
    "session", "session_depth", "depth_samples", "uncondensed",
    "uncondensed_newest", "sources", "search_session", "search_all",
    "like", "describe",
])
def test_every_reader_decodes_by_name_on_migrated_legacy_table(tmp_path, reader):
    dag = SummaryDAG(_legacy_db(tmp_path))
    try:
        a = dag.add_node(_window_node(summary="deploy alpha", expand_hint="Expand alpha"))
        b = dag.add_node(_window_node(
            summary="deploy beta", created_at=31.0, earliest_at=11.0,
            latest_at=21.0, expand_hint="Expand beta",
        ))
        parent = _window_node(
            depth=1, summary="deploy parent", created_at=32.0,
            source_type="nodes", source_ids=[a, b], latest_at=21.0,
            expand_hint="Expand parent",
        )
        parent_id = dag.add_node(parent)
        expected = {
            1: (None, None, "Expand for details about: legacy"),
            a: (10.0, 20.0, "Expand alpha"),
            b: (11.0, 21.0, "Expand beta"),
            parent_id: (10.0, 21.0, "Expand parent"),
        }
        if reader == "describe":
            tree = dag.describe_subtree(parent_id)
            assert (tree["earliest_at"], tree["latest_at"], tree["expand_hint"]) == expected[parent_id]
            # Child dictionaries expose hints only; child windows are covered
            # independently by the sources reader without changing the API.
            assert {child["node_id"] for child in tree["children"]} == {a, b}
            for child in tree["children"]:
                assert child["expand_hint"] == expected[child["node_id"]][2]
            return

        if reader == "session":
            nodes = dag.get_session_nodes("s1")
            expected_ids = set(expected)
        elif reader == "session_depth":
            nodes = dag.get_session_nodes("s1", depth=0)
            expected_ids = {1, a, b}
        elif reader == "depth_samples":
            samples = dag.get_session_depth_samples("s1")
            assert set(samples) == {0, 1}
            nodes = [node for group in samples.values() for node in group]
            expected_ids = set(expected)
        elif reader in {"uncondensed", "uncondensed_newest"}:
            nodes = dag.get_uncondensed_at_depth("s1", 1, newest=reader.endswith("newest"))
            expected_ids = {parent_id}
        elif reader == "sources":
            nodes = dag.get_source_nodes(parent)
            expected_ids = {a, b}
        elif reader == "search_session":
            nodes = dag.search("deploy", session_id="s1")
            expected_ids = {a, b, parent_id}
        elif reader == "search_all":
            nodes = dag.search("deploy")
            expected_ids = {a, b, parent_id}
        else:
            nodes = dag._search_like("deploy", session_id="s1")
            expected_ids = {a, b, parent_id}

        assert {node.node_id for node in nodes} == expected_ids
        for node in nodes:
            assert (node.earliest_at, node.latest_at, node.expand_hint) == expected[node.node_id]
            if reader in {"search_session", "search_all", "like"}:
                assert isinstance(node.search_rank, float)
            else:
                assert node.search_rank is None
    finally:
        dag.close()


@pytest.mark.parametrize("reader", ["get", "search_session", "search_relevance"])
def test_appended_access_scope_column_does_not_break_search(tmp_path, reader):
    dag = SummaryDAG(tmp_path / "s.db")
    try:
        result = ensure_scope_columns(dag.connection, tables=("summary_nodes",))
        assert result["added"] == ["summary_nodes"]
        dag.connection.commit()
        node_id = dag.add_node(_window_node(summary="deploy rollback notes"))
        dag.connection.execute("UPDATE summary_nodes SET access_scope = 'owner:alice'")
        dag.connection.commit()

        if reader == "get":
            node = dag.get_node(node_id)
            assert node.search_rank is None
        else:
            nodes = (
                dag.search("deploy", session_id="s1") if reader == "search_session"
                else dag.search("deploy", sort="relevance")
            )
            assert [node.node_id for node in nodes] == [node_id]
            node = nodes[0]
            assert isinstance(node.search_rank, float)
        assert (node.earliest_at, node.latest_at, node.expand_hint) == (10.0, 20.0, "Expand details")
    finally:
        dag.close()


def test_dag_has_no_star_select_on_summary_nodes():
    pattern = r"SELECT\s+(?:n\.)?\*"
    assert re.search(pattern, "SELECT n.*, rank", re.IGNORECASE)
    assert re.search(pattern, "SELECT * FROM summary_nodes", re.IGNORECASE)
    source = Path(hermes_lcm.dag.__file__).read_text()
    assert re.search(pattern, source, re.IGNORECASE) is None


def test_current_shape_round_trip_and_column_order_unchanged(tmp_path):
    dag = SummaryDAG(tmp_path / "current.db")
    try:
        written = _window_node()
        node_id = dag.add_node(written)
        node = dag.get_node(node_id)
        assert node == written
        assert node.search_rank is None
        hits = dag.search("deploy")
        assert [hit.node_id for hit in hits] == [node_id]
        assert isinstance(hits[0].search_rank, float)
        assert (hits[0].earliest_at, hits[0].latest_at, hits[0].expand_hint) == (10.0, 20.0, "Expand details")
    finally:
        dag.close()

    dag = SummaryDAG(_legacy_db(tmp_path))
    try:
        columns = [row[1] for row in dag.connection.execute("PRAGMA table_info(summary_nodes)")]
        assert columns == [
            "node_id", "session_id", "depth", "summary", "token_count",
            "source_token_count", "source_ids", "source_type", "created_at",
            "expand_hint", "earliest_at", "latest_at",
        ]
        assert dag.connection.row_factory is None
    finally:
        dag.close()
