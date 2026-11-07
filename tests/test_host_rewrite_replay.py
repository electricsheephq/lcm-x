"""Synthetic persisted-batch proof for the approved host rewrite replay arms."""
from collections import Counter
from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX as HEADER
from tests.harness import engine_context

TODO = HEADER + "\n- [ ] synthetic task (pending)"


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


def ingest_added(engine, rows):
    before = engine._store.get_session_count("S")
    engine._ingest_messages(deepcopy(rows))
    return engine._store.get_session_count("S") - before


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
        assert ingest_added(engine, incoming) == 0
        assert multiset(engine) == before
        assert ingest_added(engine, incoming) == 0
        assert multiset(engine) == before


@pytest.mark.parametrize("stored", [COMPOSITE, row("user", TODO, 101)], ids=["scaffold-span", "standalone"])
def test_s_red(tmp_path, stored):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [stored])
        before = multiset(engine)
        assert ingest_added(engine, [SNAP]) == 0
        assert multiset(engine) == before
        assert ingest_added(engine, [SNAP]) == 0
        assert multiset(engine) == before


def test_r6_acp_334_335_cursor_zero_persisted_batch(tmp_path, caplog):
    with engine_context(tmp_path, window=200000, platform="acp", conversation_id="conv",
                        embeddings_enabled=False) as engine:
        seed(engine, [A, COMPOSITE, B])  # host pops the composite, exposing A and B
        before = multiset(engine)
    with engine_context(tmp_path, window=200000, platform="acp", conversation_id="conv",
                        embeddings_enabled=False) as engine:
        with caplog.at_level("INFO"):
            assert engine._ingest_cursor == 0
            assert ingest_added(engine, [FOLD, SNAP]) == 0
        assert engine._last_ingest_reconciliation["action"] == "persisted batch"
        assert engine._last_ingest_reconciliation["reason"] == "persisted ambiguous delta"
        assert multiset(engine) == before  # includes the prompt the host popped
        assert ingest_added(engine, [FOLD, SNAP]) == 0
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
    ("ambiguous", [A, row("assistant", "alpha\nalpha", 100), row("assistant", "alpha", 102)],
     [row("assistant", "alpha\nalpha\nalpha", 100)], [],
     [("S", "assistant", "alpha", 100), ("S", "assistant", "alpha\nalpha", 100),
      ("S", "assistant", "alpha", 102), ("S", "assistant", "alpha\nalpha\nalpha", 100)]),
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


def test_recorder_exact_content_free_lines(tmp_path, caplog):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        ids = seed(engine, [A, COMPOSITE, B])
        with caplog.at_level("INFO"):
            assert ingest_added(engine, [FOLD, SNAP]) == 0
        records = [r for r in caplog.records if r.message.startswith("LCM host-rewrite replay:")]
        assert [(r.levelname, r.message) for r in records] == [
            ("INFO", f"LCM host-rewrite replay: arm=M parts=2 consumed={[ids[0], ids[2]]} session=S"),
            ("INFO", "LCM host-rewrite replay: arm=S parts=1 consumed=[] session=S")]


@pytest.mark.parametrize("incoming", [[row("assistant", "alpha\nbeta", 999)],
                                     [row("user", "ordinary prompt")], [row("assistant", "single", 100)]])
def test_no_candidate_zero_extra_queries(tmp_path, incoming):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [A, U, B])
        sql = []
        engine._store._conn.set_trace_callback(sql.append)
        ingest_added(engine, incoming)
        engine._store._conn.set_trace_callback(None)
        assert not any("INDEXED BY idx_msg_session_ts" in q or "substr(content, 1," in q for q in sql)


@pytest.mark.parametrize("barrier", [row("tool", "unstamped result"), row("system", "system", 99),
                                    row("summary", "summary", 90)])
def test_barriers_outside_stamp_scan(tmp_path, barrier):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [A, barrier, B])
        before = multiset(engine)
        assert ingest_added(engine, [FOLD]) == 1
        assert multiset(engine) == before + Counter([("S", "assistant", "alpha\nbeta", 100)])


def test_two_fold_views_consume_parts_once(tmp_path):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [A, U, B])
        assert ingest_added(engine, [FOLD, FOLD]) == 1
        assert multiset(engine) == Counter(BASE + [("S", "assistant", "alpha\nbeta", 100)])


@pytest.mark.parametrize("text", ["alpha\nbeta ", " alpha\nbeta", "alpha\nmissing", "alpha\n\nbeta"])
def test_inexact_fold_bytes_stay_new(tmp_path, text):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [A, U, B])
        assert ingest_added(engine, [row("assistant", text, 100)]) == 1


def test_full_scan_budget_stays_new(tmp_path):
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [A, U, B] + [row("user", f"padding {i}", 200+i) for i in range(61)])
        assert ingest_added(engine, [FOLD]) == 1


def test_unique_decomposition_with_repeated_anchor_stamp(tmp_path):
    # The first head cannot fit; the second is the unique contiguous decomposition.
    with engine_context(tmp_path, window=200000, conversation_id="conv", embeddings_enabled=False) as engine:
        seed(engine, [A, A, U, B])
        before = multiset(engine)
        assert ingest_added(engine, [FOLD]) == 0
        assert multiset(engine) == before
        assert ingest_added(engine, [FOLD]) == 0
