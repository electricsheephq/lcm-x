"""Frozen embedding verifier strings, including ordering and the family facade."""

import sqlite3

import pytest

from hermes_lcm import db_bootstrap

# name, DDL replacements, objects to drop, exact verifier output at the base.
Case = tuple[str, tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]]
CASES: tuple[Case, ...] = (
    ("clean", (), (), ()),
    ("missing-profile", (), ("TABLE lcm_embedding_profile",), (
        "missing object: idx_lcm_embedding_profile_model",
        "missing object: lcm_embedding_profile",
    )),
    ("missing-meta", (), ("TABLE lcm_embedding_meta",), (
        "missing object: idx_lcm_embedding_meta_identity_embedded_at",
        "missing object: lcm_embedding_meta",
    )),
    ("missing-vectors", (), ("TABLE lcm_embedding_vectors",), (
        "missing object: lcm_embedding_vectors",
    )),
    ("missing-binary", (), ("TABLE lcm_embedding_binary",), (
        "missing object: lcm_embedding_binary",
    )),
    ("missing-profile-index", (), ("INDEX idx_lcm_embedding_profile_model",), (
        "missing object: idx_lcm_embedding_profile_model",
    )),
    ("missing-meta-index", (), ("INDEX idx_lcm_embedding_meta_identity_embedded_at",), (
        "missing object: idx_lcm_embedding_meta_identity_embedded_at",
    )),
    ("column-type", (("vec BLOB", "vec TEXT"),), (), (
        "malformed table: lcm_embedding_vectors",
    )),
    ("column-notnull", (("vec BLOB NOT NULL", "vec BLOB"),), (), (
        "malformed table: lcm_embedding_vectors",
    )),
    ("column-default", (("active INTEGER DEFAULT 1", "active INTEGER DEFAULT 0"),), (), (
        "malformed table: lcm_embedding_profile",
    )),
    ("pk-order", (("PRIMARY KEY(embedded_id, embedded_kind, identity_hash)",
                   "PRIMARY KEY(identity_hash, embedded_kind, embedded_id)"),), (), (
        "malformed table: lcm_embedding_meta",
    )),
    ("check-expression", (("dim BETWEEN 1 AND 4096", "dim BETWEEN 1 AND 4095"),), (), (
        "malformed constraints: lcm_embedding_profile",
    )),
    ("index-column", (("(model_name, provider)", "(revision, provider)"),), (), (
        "malformed index: idx_lcm_embedding_profile_model",
    )),
    ("index-desc", (("embedded_at DESC", "embedded_at ASC"),), (), (
        "malformed index: idx_lcm_embedding_meta_identity_embedded_at",
    )),
    ("index-unique", (("CREATE INDEX idx_lcm_embedding_profile_model",
                       "CREATE UNIQUE INDEX idx_lcm_embedding_profile_model"),), (), (
        "malformed index: idx_lcm_embedding_profile_model",
    )),
    ("index-predicate", (("WHERE archived = 0", "WHERE archived = 1"),), (), (
        "malformed index: idx_lcm_embedding_meta_identity_embedded_at",
    )),
    ("two-defects", (("vec BLOB", "vec TEXT"), ("dim BETWEEN 1 AND 4096", "dim > 0")), (), (
        "malformed constraints: lcm_embedding_profile",
        "malformed table: lcm_embedding_vectors",
    )),
    ("index-multiple-defects", (("embedded_at DESC", "embedded_at ASC"),
                                ("WHERE archived = 0", "WHERE archived = 1")), (), (
        "malformed index: idx_lcm_embedding_meta_identity_embedded_at",
    )),
    ("missing-returns-early", (("vec BLOB", "vec TEXT"),), ("TABLE lcm_embedding_binary",), (
        "missing object: lcm_embedding_binary",
    )),
)


def schema_for(case: Case) -> sqlite3.Connection:
    """Replay real DDL with one explicit corruption; never edit schema metadata."""
    source = sqlite3.connect(":memory:")
    try:
        db_bootstrap.ensure_embedding_tables(source)
        ddl = ";\n".join(str(row[0]) for row in source.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid"
        )) + ";"
    finally:
        source.close()
    for before, after in case[1]:
        assert before in ddl
        ddl = ddl.replace(before, after)
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    for object_name in case[2]:
        conn.execute(f"DROP {object_name}")
    return conn


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_embedding_verifier_golden(case: Case) -> None:
    conn = schema_for(case)
    try:
        assert db_bootstrap.verify_embedding_schema(conn) == list(case[3])
    finally:
        conn.close()


@pytest.mark.parametrize("case", [CASES[0], CASES[16]], ids=["clean", "two-defects"])
def test_embedding_family_facade_golden(case: Case) -> None:
    conn = schema_for(case)
    try:
        verifier = db_bootstrap._family_verifier("lcm_embedding")
        assert verifier is db_bootstrap.verify_embedding_schema
        assert verifier(conn) == list(case[3])
    finally:
        conn.close()
