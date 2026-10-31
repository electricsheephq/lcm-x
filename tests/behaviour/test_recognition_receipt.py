"""Store-derived observations of public ingest, replay and coverage (#436/#1000)."""

from copy import deepcopy
from dataclasses import dataclass

import pytest

from tests.harness import ScriptedSummarizer, assistant, call, engine_context, make_engine, pin_char_counter, stored_rows, tool, user
from tests import test_issue_650_survival_fit_keeps_summary as survival


@dataclass(frozen=True)
class RecognitionReceipt:
    input_positions: tuple[int, ...]
    appended_rows: tuple[tuple[int, str, str, float | None], ...]
    matched_ids: tuple[tuple[int, tuple[int, ...]], ...]
    relations: tuple[tuple[int, str, int, int | None], ...]
    coverage: tuple[tuple[int, ...], ...]


def receipt(engine, view):
    """Observe effects without invoking the recognizer's private mapper.

    matched_ids are byte-equal stored occurrences (consumed once) or a recorded
    composite decomposition, not an assertion about the engine's internal cursor.
    """
    original = deepcopy(view)
    before = {row["store_id"] for row in stored_rows(engine)}
    engine.ingest(view)
    assert view == original
    rows = stored_rows(engine)
    conn = engine._store.connection
    has_relations = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_relations'"
    ).fetchone()
    relations = tuple(tuple(row) for row in conn.execute(
        "SELECT store_id, kind, related_store_id, ordinal "
        "FROM message_relations ORDER BY relation_id"
    )) if has_relations else ()
    by_id = {row["store_id"]: row for row in rows}
    composites = {}
    for owner, kind, related, ordinal in relations:
        if kind == "composite":
            composites.setdefault(owner, []).append((ordinal, related))
    available = list(rows)
    matched = []
    for position, message in enumerate(view):
        ids = ()
        for row in available:
            if (row["role"], row["content"]) == (message["role"], message["content"]):
                ids = (row["store_id"],)
                available.remove(row)
                break
        if not ids:
            for parts in composites.values():
                candidates = tuple(related for _, related in sorted(parts))
                if "\n\n".join(by_id[sid]["content"] for sid in candidates) == message["content"]:
                    ids = candidates
                    break
        matched.append((position, ids))
    return RecognitionReceipt(
        tuple(range(len(view))),
        tuple((r["store_id"], r["role"], r["content"], r.get("observed_at"))
              for r in rows if r["store_id"] not in before),
        tuple(matched), relations,
        tuple(tuple(node.source_ids) for node in engine._dag.get_session_nodes(engine.current_session_id)
              if node.source_type == "messages"),
    )


def stamped(role, text, timestamp):
    return {"role": role, "content": text, "timestamp": timestamp}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    ScriptedSummarizer().install(monkeypatch)


def test_nested_tool_input_is_immutable_on_ingest_and_replay(tmp_path):
    view = [user("probe"), assistant("", calls=[call("c1", arguments='{"value": [1, 2]}')]),
            tool("c1", "probe result")]
    with engine_context(tmp_path, window=200_000, embeddings_enabled=False) as engine:
        observed = receipt(engine, view)
        assert len(observed.appended_rows) == 3
        assert observed.matched_ids == ((0, (1,)), (1, (2,)), (2, (3,)))
        assert receipt(engine, deepcopy(view)).appended_rows == ()


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_identical_new_occurrence_and_replay(tmp_path, cold):
    first = [stamped("user", "ok", 700.0), stamped("assistant", "reply one", 701.0)]
    second = [*deepcopy(first), stamped("user", "ok", 700.0), stamped("assistant", "reply two", 702.0)]
    engine = make_engine(tmp_path, window=200_000, embeddings_enabled=False)
    try:
        initial = receipt(engine, first)
        assert initial.input_positions == (0, 1)
        assert initial.appended_rows == ((1, "user", "ok", 700.0), (2, "assistant", "reply one", 701.0))
        assert initial.matched_ids == ((0, (1,)), (1, (2,)))
        assert initial.relations == initial.coverage == ()
        if cold:
            engine.shutdown()
            engine = make_engine(tmp_path, window=200_000, embeddings_enabled=False)
        added = receipt(engine, second)
        assert added.appended_rows == ((3, "user", "ok", 700.0), (4, "assistant", "reply two", 702.0))
        assert added.matched_ids == ((0, (1,)), (1, (2,)), (2, (3,)), (3, (4,)))
        assert receipt(engine, deepcopy(second)).appended_rows == ()
    finally:
        engine.shutdown()
    with engine_context(tmp_path, window=200_000, embeddings_enabled=False) as engine:
        replay = receipt(engine, deepcopy(second))
        assert replay.appended_rows == ()
        assert replay.matched_ids == added.matched_ids
        assert replay.relations == added.relations == ()


def test_cold_replay_with_missing_middle_keeps_rows(tmp_path):
    history = [stamped(role, f"turn {i} {role}", (i + 1) * 10 + offset)
               for i in range(4) for offset, role in enumerate(("user", "assistant"))]
    with engine_context(tmp_path, window=200_000, embeddings_enabled=False) as engine:
        assert len(receipt(engine, history).appended_rows) == 8
    with engine_context(tmp_path, window=200_000, embeddings_enabled=False) as engine:
        observed = receipt(engine, [*history[:2], *history[4:], user("new"), assistant("reply")])
        assert [row[2] for row in observed.appended_rows] == ["new", "reply"]
        assert observed.matched_ids == tuple(enumerate((
            (1,), (2,), (5,), (6,), (7,), (8,), (9,), (10,),
        )))
        assert [row["content"] for row in stored_rows(engine)][:8] == [m["content"] for m in history]


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_composite_remainder_receipt_and_summary_coverage(tmp_path, cold):
    head = stamped("user", "head R\n\n  indented line \n\n\nthree newlines\t", 500.0)
    remainder = "  remainder U \n\nsecond  paragraph\n\n\n  tail \t"
    options = dict(window=200_000, embeddings_enabled=False, fresh_tail_count=2, leaf_chunk_tokens=1)
    engine = make_engine(tmp_path, **options)
    try:
        receipt(engine, [head])
        if cold:
            engine.shutdown()
            engine = make_engine(tmp_path, **options)
        composite = stamped("user", head["content"] + "\n\n" + remainder, 500.0)
        live = [composite, stamped("assistant", "reply", 501.0)]
        observed = receipt(engine, live)
        assert observed.appended_rows == ((2, "user", remainder, None), (3, "assistant", "reply", 501.0))
        assert observed.matched_ids == ((0, (1, 2)), (1, (3,)))
        assert [(kind, related, ordinal) for _, kind, related, ordinal in observed.relations] == [
            ("composite", 1, 0), ("composite", 2, 1),
        ]
        assert receipt(engine, deepcopy(live)).appended_rows == ()
        live.extend([stamped("user", "later " * 60, 600.0), stamped("assistant", "last reply", 601.0)])
        engine.compress(live)
        covered = receipt(engine, live)
        assert {1, 2} <= {sid for group in covered.coverage for sid in group}
        assert len(stored_rows(engine)) == 5
    finally:
        engine.shutdown()


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    # Reuse #1000's host scenario and held-fit setup; keep public ingest below.
    host = survival.host.__wrapped__(monkeypatch)

    def factory(path, window=24_000):
        return make_engine(path, window=window, platform="telegram", conversation_id="conv",
                           embeddings_enabled=False, fresh_tail_count=4, leaf_chunk_tokens=1600,
                           context_threshold=0.75, threshold_full_sweep_enabled=True)

    monkeypatch.setattr(survival, "_engine", factory)
    engine, view, rest = survival._held_fit_after_inplace_boundary(tmp_path, host, monkeypatch)
    try:
        yield engine, view, rest
    finally:
        engine.shutdown()


class DuplicateRetained(AssertionError):
    """Only the final copy count may satisfy the #923 strict xfail."""


@pytest.mark.parametrize("when", ["same-pass", "next-turn"])
@pytest.mark.parametrize("stamp", ["original-stamp", "unstamped"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["as-returned", "turn-end-rewrite"])
def test_1000_carrier_replay_receipt(fitted, when, stamp, rewrite, request):
    if when == "same-pass" and stamp == "unstamped":
        request.applymarker(pytest.mark.xfail(strict=True, raises=DuplicateRetained, reason=(
            "#923: unstamped U beside the carrier is retained as a duplicate rather than lost"
        )))
    engine, view, rest = fitted
    u = next(row for row in stored_rows(engine) if row["content"] == rest)
    replayed = user(rest)
    if stamp == "original-stamp":
        replayed["timestamp"] = u["observed_at"]
    if rewrite:
        view[-1] = {**view[-1], "content": "[CUR] persist variant"}
    reply = stamped("assistant", "reply CUR", 30_001.0)
    if when == "next-turn":
        claimed = receipt(engine, [*view, reply])
        assert all("Summary (d" not in row[2][:40] for row in claimed.appended_rows)
        summary = view[0]["content"].split("\n\n" + rest)[0]
        observed = receipt(engine, [user(summary), replayed, *view[1:], reply])
        assert observed.appended_rows == ()
    else:
        observed = receipt(engine, [view[0], replayed, *view[1:], reply])
    assert all("Summary (d" not in row[2][:40] for row in observed.appended_rows)
    assert observed.coverage
    copies = sum(row["content"] == rest for row in stored_rows(engine))
    if copies != 1:
        raise DuplicateRetained(copies)


@pytest.mark.parametrize("when", ["same-pass", "next-turn"])
def test_1000_fresh_stamped_identical_turn_is_appended(fitted, when):
    engine, view, rest = fitted
    view[0]["timestamp"] = 40_000.0
    reply = stamped("assistant", "reply CUR", 30_001.0)
    new = stamped("user", rest, 40_010.0)
    if when == "next-turn":
        assert [r[2] for r in receipt(engine, [*view, reply]).appended_rows] == ["reply CUR"]
    observed = receipt(engine, [*view, reply, new])
    assert [r[2] for r in observed.appended_rows] == (["reply CUR", rest] if when == "same-pass" else [rest])
    assert sum(row["content"] == rest for row in stored_rows(engine)) == 2
