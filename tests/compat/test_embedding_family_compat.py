"""A5 embedding-family forward and rollback proof using pinned release code."""

from collections.abc import Callable
from contextlib import closing
from dataclasses import FrozenInstanceError
from pathlib import Path
import sqlite3
import struct
import typing
from typing import Any, cast

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.schema_families import COMPAT_REGISTRY, FINDING_KINDS, SchemaFinding
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
from tests.compat.helpers import PREVIOUS_GA, READERS, run_old
from tests.test_embedding_schema_golden import CASES, Case, schema_for

EMBEDDING_READERS = {
    **READERS,
    "v0.20.0": "49e99a272d2d461e5c90732e7ef2bc20e96f0826",
}
run_reader = cast(Callable[..., dict[str, Any]], run_old)
OLD_EMBEDDING_BOOTSTRAP = '''
import importlib.util, json, sys
from pathlib import Path
old, db, home, host_root, mode, data = sys.argv[1:]
spec = importlib.util.spec_from_file_location("hermes_lcm", old + "/__init__.py", submodule_search_locations=[old])
sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
from hermes_lcm import db_bootstrap, vector_store
from hermes_lcm.config import LCMConfig
assert Path(db_bootstrap.__file__).resolve().parent == Path(old).resolve()
assert Path(vector_store.__file__).resolve().parent == Path(old).resolve()
'''
OLD_SETUP = '''
from hermes_lcm.vector_store import VectorStore, EmbeddingIdentity
from hermes_lcm.db_bootstrap import verify_embedding_schema
store = VectorStore(db, config=LCMConfig(embeddings_enabled=False))
identity = EmbeddingIdentity.canonical('fixture', 'fixture-model', '', 2, 'float32', 'little', 'summary')
assert verify_embedding_schema(store._conn) == []
'''
OLD_WRITE = OLD_SETUP + '''
from hermes_lcm.dag import SummaryDAG, SummaryNode
dag = SummaryDAG(db)
for i in (1, 2, 3):
    assert dag.add_node(SummaryNode(session_id='fixture', summary='synthetic', source_token_count=10,
                                   created_at=1.0, earliest_at=1.0, latest_at=1.0)) == i
dag.close()
store.register_profile('fixture-model', 'fixture', 2)
for key, vec in [('1', [1.0, 0.0]), ('2', [0.0, 1.0])]:
    store.record_embedding(key, 'summary', 'fixture-model', vec, identity=identity)
assert store._migration_step_present('embeddings_v1')
store.close()
print(json.dumps({'written': 2}))
'''
OLD_ROLLBACK = OLD_SETUP + '''
rows = store._load_vectors_for_ids(identity.identity_hash, 2, ['1', '2'])
assert dict(zip(rows[1], rows[3])) == {'1': [1.0, 0.0], '2': [0.0, 1.0]}
store._now = lambda: '2000-01-01T00:00:00+00:00'
store.record_embedding('3', 'summary', 'fixture-model', [1.0, 0.0], identity=identity)
rows = store._load_vectors_for_ids(identity.identity_hash, 2, ['1', '2', '3'])
assert dict(zip(rows[1], rows[3])) == {'1': [1.0, 0.0], '2': [0.0, 1.0], '3': [1.0, 0.0]}
assert verify_embedding_schema(store._conn) == []
assert store._migration_step_present('embeddings_v1')
store.close()
print(json.dumps({'verifier': [], 'read': 2, 'after_write': 3}))
'''


def snapshot(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    with closing(sqlite3.connect(path)) as conn:
        return {
            "tables": conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall(),
            "profile": conn.execute("SELECT * FROM lcm_embedding_profile ORDER BY identity_hash").fetchall(),
            "meta": conn.execute("SELECT * FROM lcm_embedding_meta ORDER BY embedded_id").fetchall(),
            "vectors": conn.execute("SELECT * FROM lcm_embedding_vectors ORDER BY embedded_id").fetchall(),
            "binary": conn.execute("SELECT * FROM lcm_embedding_binary ORDER BY embedded_id").fetchall(),
            "marker": conn.execute("SELECT * FROM lcm_migration_state WHERE step_name='embeddings_v1'").fetchall(),
        }


@pytest.mark.parametrize("reader", tuple(EMBEDDING_READERS))
def test_embedding_forward(
    reader: str, tmp_path: Path, old_tree: Callable[..., Path], host_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert run_reader(old_tree(reader, EMBEDDING_READERS), tmp_path, host_root, OLD_WRITE,
                      bootstrap=OLD_EMBEDDING_BOOTSTRAP) == {"written": 2}
    path = tmp_path / "lcm.db"
    before = snapshot(path)
    assert len(before["profile"]) == 1 and len(before["vectors"]) == 2 and len(before["marker"]) == 1
    drops: list[str | None] = []
    connect = sqlite3.connect

    def authorize(action: int, name: str | None, column: str | None, database: str | None,
                  trigger: str | None) -> int:
        if action == sqlite3.SQLITE_DROP_TABLE:
            drops.append(name)
        return sqlite3.SQLITE_OK

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = cast(sqlite3.Connection, connect(*args, **kwargs))
        conn.set_authorizer(authorize)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    with closing(VectorStore(path, config=LCMConfig(embeddings_enabled=False))) as store:
        assert store._conn is not None
        assert db_bootstrap.embedding_schema_findings(store._conn) == []
        assert snapshot(path) == before
    assert drops == []


@pytest.mark.parametrize("reader", tuple(EMBEDDING_READERS))
@pytest.mark.parametrize("binary", (False, True), ids=("empty-binary", "populated-binary"))
def test_embedding_rollback(
    reader: str, binary: bool, tmp_path: Path, old_tree: Callable[..., Path], host_root: Path,
) -> None:
    path = tmp_path / "lcm.db"
    with closing(SummaryDAG(path)) as dag:
        for i in (1, 2, 3):
            assert dag.add_node(SummaryNode(session_id="fixture", summary="synthetic", source_token_count=10,
                                            created_at=1.0, earliest_at=1.0, latest_at=1.0)) == i
    with closing(VectorStore(path, config=LCMConfig(embeddings_enabled=False))) as store:
        store.register_profile("fixture-model", "fixture", 2)
        identity = EmbeddingIdentity.canonical("fixture", "fixture-model", "", 2, "float32", "little", "summary")
        for key, vector in (("1", [1.0, 0.0]), ("2", [0.0, 1.0])):
            store.record_embedding(key, "summary", "fixture-model", vector, identity=identity)
        if binary:
            # An inactive int8 corpus supplies real pre-existing binary rows.
            store.register_profile("binary-fixture-model", "fixture", 2, dtype="int8")
            binary_identity = EmbeddingIdentity.canonical(
                "fixture", "binary-fixture-model", "", 2, "int8", "little", "summary",
            )
            store.record_embedding("1", "summary", "binary-fixture-model", [1.0, 0.0], identity=binary_identity)
            store.register_profile("fixture-model", "fixture", 2)
    before = snapshot(path)
    assert run_reader(old_tree(reader, EMBEDDING_READERS), tmp_path, host_root, OLD_ROLLBACK,
                      bootstrap=OLD_EMBEDDING_BOOTSTRAP) == {
        "verifier": [], "read": 2, "after_write": 3,
    }
    after = snapshot(path)
    assert after["tables"] == before["tables"] and after["marker"] == before["marker"]
    if not binary:
        assert before["binary"] == after["binary"] == []
    else:
        assert len(before["binary"]) == 1
    primary_keys = {
        "profile": ("identity_hash",),
        "meta": ("embedded_id", "embedded_kind", "identity_hash"),
        "vectors": ("embedded_id", "identity_hash"),
        "binary": ("embedded_id", "identity_hash"),
    }
    expected_new = {
        "profile": {},
        "meta": {("3", "summary", identity.identity_hash): {
            "embedded_id": "3", "embedded_kind": "summary", "identity_hash": identity.identity_hash,
            "embedded_at": "2000-01-01T00:00:00+00:00", "source_token_count": 10, "archived": 0,
        }},
        "vectors": {("3", identity.identity_hash): {
            "embedded_id": "3", "identity_hash": identity.identity_hash, "vec": struct.pack("<2f", 1.0, 0.0),
        }},
        "binary": {},
    }
    with closing(sqlite3.connect(path)) as conn:
        for table, keys in primary_keys.items():
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info(lcm_embedding_{table})")]
            keyed = []
            for state in (before, after):
                rows = [dict(zip(columns, row, strict=True)) for row in state[table]]
                keyed.append({tuple(row[key] for key in keys): row for row in rows})
            prior, current = keyed
            assert current.keys() == prior.keys() | expected_new[table].keys(), table
            for key, row in prior.items():
                expected = dict(row)
                if table == "profile" and key == (identity.identity_hash,):
                    # Every pinned reader bumps only this counter for a write.
                    expected["data_version"] += 1
                assert current[key] == expected, (table, key)
            for key, row in expected_new[table].items():
                assert current[key] == row, (table, key)


def test_embedding_schema_findings_type_hints() -> None:
    assert typing.get_type_hints(db_bootstrap.embedding_schema_findings)["return"] == list[SchemaFinding]


def test_embedding_registry(tmp_path: Path) -> None:
    assert tuple(COMPAT_REGISTRY) == ("embedding",)
    family = COMPAT_REGISTRY["embedding"]
    assert family.name == "embedding" and family.prefix == "lcm_embedding"
    assert family.marker == "embeddings_v1" and family.removal_version is None
    assert family.tables == db_bootstrap._REQUIRED_EMBEDDING_TABLES
    assert family.indexes == db_bootstrap._REQUIRED_EMBEDDING_INDEXES
    # Compare release versions, including prereleases, rather than insertion order
    # or git ancestry: the oldest embedding release is on a separate lineage.
    def version(tag: str) -> tuple[int, int, int, bool, int]:
        release, _, rc = tag.removeprefix("v").partition("-rc")
        major, minor, patch = map(int, release.split("."))
        return major, minor, patch, not rc, int(rc or "0")

    assert family.oldest_reader == min(EMBEDDING_READERS, key=version)
    assert "v0.20.0" in EMBEDDING_READERS
    assert PREVIOUS_GA in EMBEDDING_READERS
    for test in (test_embedding_forward, test_embedding_rollback):
        marks = getattr(test, "pytestmark")
        assert any(mark.name == "parametrize" and mark.args[0] == "reader"
                   and set(mark.args[1]) == set(EMBEDDING_READERS) for mark in marks)
    assert any(row["name"] == family.name and row["prefix"] == family.prefix
               for row in db_bootstrap._INTERIM_FEATURE_FAMILIES)
    with closing(VectorStore(tmp_path / "lcm.db", config=LCMConfig(embeddings_enabled=False))) as store:
        assert store._migration_step_present(family.marker)
    with pytest.raises(FrozenInstanceError):
        setattr(family, "marker", "changed")


@pytest.mark.parametrize("kind", ("missing object", "malformed table", "malformed constraints", "malformed index"))
def test_finding_string_round_trip(kind: str) -> None:
    assert FINDING_KINDS == ("missing object", "malformed table", "malformed constraints", "malformed index")
    finding = SchemaFinding(kind, "object")
    assert str(finding) == f"{kind}: object"
    assert SchemaFinding(*str(finding).split(": ", 1)) == finding
    assert len({finding, SchemaFinding(kind, "object")}) == 1
    with pytest.raises(FrozenInstanceError):
        setattr(finding, "name", "changed")


def test_unknown_finding_kind_rejected() -> None:
    with pytest.raises(ValueError):
        SchemaFinding("unknown", "object")


@pytest.mark.parametrize("case", [CASES[i] for i in (3, 7, 11, 12)], ids=lambda case: case[0])
def test_typed_findings(case: Case) -> None:
    with closing(schema_for(case)) as conn:
        findings = db_bootstrap.embedding_schema_findings(conn)
        assert findings and all(isinstance(finding, SchemaFinding) for finding in findings)
        assert [str(finding) for finding in findings] == list(case[3])
