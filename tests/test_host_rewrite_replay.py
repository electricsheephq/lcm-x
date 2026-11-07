"""Synthetic persisted-batch proof for the approved host rewrite replay arms."""
from collections import Counter
from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX as HEADER
from tests.harness import engine_context

TODO = HEADER + "\n- [ ] synthetic task"


def row(role, content, stamp=None):
    return dict(role=role, content=content, **({"timestamp": stamp} if stamp else {}))


A, U, B, V, C = (row("assistant", "alpha", 100), row("user", "prompt", 101),
                  row("assistant", "beta", 102), row("user", "next", 103),
                  row("assistant", "gamma", 104))
FOLD = row("assistant", "alpha\nbeta", 100)
SNAP = row("user", TODO)
COMPOSITE = row("user", TODO + "\n\nprompt", 101)


def multiset(engine):
    return Counter((r["session_id"], r["role"], r["content"], r["observed_at"])
                   for r in engine._store.get_session_messages("S") + engine._store.get_session_messages("other"))


def seed(engine, rows, session="S"):
    return engine._store.append_batch(session, deepcopy(rows), conversation_id="conv")


@pytest.mark.parametrize("parts", [2, 3])
def test_m_red(tmp_path, parts):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        original = [A, U, B] if parts == 2 else [A, U, B, V, C]
        seed(engine, original)
        before = multiset(engine)
        incoming = [row("assistant", "\n".join(r["content"] for r in original if r["role"] == "assistant"), 100)]
        assert engine._ingest_cursor == 0
        assert engine._ingest_messages(deepcopy(incoming)) == 0
        assert multiset(engine) == before
        assert engine._ingest_messages(deepcopy(incoming)) == 0
        assert multiset(engine) == before


@pytest.mark.parametrize("stored", [COMPOSITE, row("user", TODO, 101)], ids=["scaffold-span", "standalone"])
def test_s_red(tmp_path, stored):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [stored])
        before = multiset(engine)
        assert engine._ingest_messages([deepcopy(SNAP)]) == 0
        assert multiset(engine) == before
        assert engine._ingest_messages([deepcopy(SNAP)]) == 0
        assert multiset(engine) == before


def test_r6_acp_334_335_cursor_zero_persisted_batch(tmp_path, caplog):
    with engine_context(tmp_path, window=200000, platform="acp", conversation_id="conv",
                        embeddings_enabled=False) as engine:
        seed(engine, [A, COMPOSITE, B])  # host pops the composite, exposing A and B
        before = multiset(engine)
        with caplog.at_level("INFO"):
            assert engine._ingest_cursor == 0
            assert engine._ingest_messages(deepcopy([FOLD, SNAP])) == 0
        assert multiset(engine) == before  # includes the prompt the host popped
        assert engine._ingest_messages(deepcopy([FOLD, SNAP])) == 0
        assert multiset(engine) == before


# Literal main expectations, including stamps; no candidate-derived oracle.
BASE = [("S", "assistant", "alpha", 100), ("S", "user", "prompt", 101),
        ("S", "assistant", "beta", 102)]
CONTROLS = [
    ("new-stamp", [A, U, B], [row("assistant", "alpha\nbeta", 110)], [],
     BASE + [("S", "assistant", "alpha\nbeta", 110)]),
    ("missing-part", [A, U, B], [row("assistant", "alpha\nmissing", 100)], [],
     BASE + [("S", "assistant", "alpha\nmissing", 100)]),
    ("tool-barrier", [A, row("tool", "result", 101), B], [FOLD], [],
     [("S", "assistant", "alpha", 100), ("S", "tool", "result", 101),
      ("S", "assistant", "beta", 102), ("S", "assistant", "alpha\nbeta", 100)]),
    ("separator", [A, U, B], [row("assistant", "alpha\n\nbeta", 100)], [],
     BASE + [("S", "assistant", "alpha\n\nbeta", 100)]),
    ("consumed-part", [A, U, B], [A, FOLD], [],
     BASE + [("S", "assistant", "alpha\nbeta", 100)]),
    ("ambiguous", [A, A, U, B], [FOLD], [],
     BASE + [("S", "assistant", "alpha", 100), ("S", "assistant", "alpha\nbeta", 100)]),
    ("foreign-fold", [U, B], [FOLD], [A],
     [("S", "user", "prompt", 101), ("S", "assistant", "beta", 102),
      ("other", "assistant", "alpha", 100), ("S", "assistant", "alpha\nbeta", 100)]),
    ("changed-item", [COMPOSITE], [row("user", HEADER + "\n- [ ] changed")], [],
     [("S", "user", TODO + "\n\nprompt", 101), ("S", "user", HEADER + "\n- [ ] changed", None)]),
    ("suffix", [COMPOSITE], [row("user", TODO + "\n\nextra")], [],
     [("S", "user", TODO + "\n\nprompt", 101), ("S", "user", TODO + "\n\nextra", None)]),
    ("stamped-tui", [COMPOSITE], [row("user", TODO, 110)], [],
     [("S", "user", TODO + "\n\nprompt", 101), ("S", "user", TODO, 110)]),
    ("foreign-snapshot", [], [SNAP], [COMPOSITE],
     [("other", "user", TODO + "\n\nprompt", 101), ("S", "user", TODO, None)]),
]


@pytest.mark.parametrize("name,stored,incoming,foreign,expected", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_control_equals_main(tmp_path, caplog, name, stored, incoming, foreign, expected):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, stored)
        seed(engine, foreign, "other")
        with caplog.at_level("INFO"):
            engine._ingest_messages(deepcopy(incoming))
        assert multiset(engine) == Counter(expected)
        assert not any("LCM host-rewrite replay:" in r.message for r in caplog.records)
