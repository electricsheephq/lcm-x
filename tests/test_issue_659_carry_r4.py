"""#659 round 4 probes: a new user copy of the emitted context is the user's, byte for byte.

Ported from the third cross-model review's probes (R3-N1 lossless user copy, R3-N2 replay identity), with the
controls it kept green: the #922 objective-only guard and legacy conversation ids. Round 6 stores a no-proof
rotated head whole (no storage normalisation), a recorded F1 deviation from base.
"""

import hashlib

import pytest

from hermes_lcm.engine import _render_omitted_summaries
import tests.test_issue_659_user_carry_packet as carry_packet
from tests.test_issue_659_user_carry_packet import CARRY, SEP, _compact, _history, _host_merge, _tools

make = carry_packet.make  # the shared fixture

# The rotated heads the feature base (ce186095) stores for the same sequence: sha256 and UTF-8 length (REVIEW3).
BASE_HEADS = {"summary": ("d5b7184a46a3668e5493d5e87012ec3253eab08f078168149ffab7206a74f049", 125),
              "objective": ("752a4c1f451199428b72c731056762b5f84b73dc9695529adbe81889c5341059", 204)}


def _generated(engine, objective=False):
    """(compress output, the emitted pure scaffold head)."""
    ending = ([{"role": "user", "content": "ACTIVE request"}, *_tools(70), *_tools(71)] if objective
              else [{"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}])
    out = _compact(engine, [*_history(8), *ending])
    row = next(m for m in out if m.get("role") == "user")
    return out, row["content"] if objective else row["content"][:engine._verified_lcm_summary_prefix_end(row["content"])]


@pytest.mark.parametrize("hops", [1, 2, 32, 33])
@pytest.mark.parametrize("kind", ["summary", "objective"])
def test_no_proof_rotated_head_is_stored_whole_a_recorded_deviation_from_base(make, hops, kind):
    """Round 6: the head arrives without its emitted window or a session end, so no commit proof applies and it is
    stored as submitted, carry included. Base stores BASE_HEADS here; this F1 deviation is intentional. On the
    commit-proof path storage equals base (test_issue_659_carry_r6)."""
    engine = make(session="P0")
    _out, pure = _generated(engine, objective=kind == "objective")
    assert CARRY in pure
    for i in range(1, hops + 1):
        engine.on_session_start(f"P{i}", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression", old_session_id=f"P{i - 1}")
    engine.ingest([{"role": "user", "content": pure}, {"role": "user", "content": "new request"},
                   {"role": "assistant", "content": "ok"}])
    rows = engine._store.get_session_messages(f"P{hops}")
    heads = [r for r in rows if "[Recent Summary" in r["content"]]
    assert len(rows) == 3 and len(heads) == 1 and heads[0]["content"].encode("utf-8") == pure.encode("utf-8")
    assert (hashlib.sha256(pure.encode("utf-8")).hexdigest(), len(pure.encode("utf-8"))) != BASE_HEADS[kind]


@pytest.mark.parametrize("kind", ["summary", "objective"])
def test_cold_bind_replay_of_the_rotated_head_matches_the_base_shape_replay(make, kind, tmp_path):
    """Full identities stay as they were: the emitted head (with carry) replays exactly like the base-shape head."""
    results = []
    for host in ("emitted", "base-shape"):
        engine = make(session="P0", home=tmp_path / host)
        _out, pure = _generated(engine, objective=kind == "objective")
        engine.on_session_start("P1", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression", old_session_id="P0")
        tail = [{"role": "user", "content": "new request"}, {"role": "assistant", "content": "ok"}]
        engine.ingest([{"role": "user", "content": pure}, *tail])
        before = len(engine._store.get_session_messages("P1"))
        engine.shutdown()
        resumed = make(session="P1", home=tmp_path / host)
        head = pure if host == "emitted" else pure[:pure.index(SEP + CARRY)]
        resumed.ingest([{"role": "user", "content": head}, *tail, {"role": "user", "content": "next"}])
        results.append(([r["content"] for r in resumed._store.get_session_messages("P1")[before:]],
                        resumed._last_ingest_reconciliation["cursor"]))
    assert results[0] == results[1]


def _user_copy(engine, form):
    out, pure = _generated(engine)
    base = pure.split(SEP + CARRY)[0]
    if form == "exact":
        text = pure
    elif form == "prefixed":
        text = "Please inspect this copied context:" + SEP + pure
    elif form == "own-separator":
        text = "MY authored section" + SEP + base + SEP + pure.split(SEP, 1)[1]
    elif form == "manifest":
        node = engine._dag.get_node(int(engine._LCM_SUMMARY_PART_HEADER_RE.match(base).group(2)))
        text = "Please inspect this omitted node:" + SEP + base + SEP + _render_omitted_summaries([node], 0)
    elif form == "edited-carry":
        text = base + SEP + "[Earlier user messages in this session, verbatim, for reference only: MY OWN TEXT]"
    elif form == "edited-manifest":
        text = base + SEP + "[Summary parts omitted for space: MY OWN TEXT]"
    else:
        text = pure + "\n\nMY REAL ACTIVE REQUEST — keep every byte.\n"
    return out, pure, text


@pytest.mark.parametrize("form", ["exact", "prefixed", "own-separator", "manifest",
                                  "edited-carry", "edited-manifest", "glued-request"])
def test_new_user_copy_of_the_emitted_context_is_stored_whole(make, form):
    """R3-N1: a NEW user occurrence after the emitted context is the user's, even when it equals LCM's text."""
    engine = make()
    out, _pure, text = _user_copy(engine, form)
    last = max(r["store_id"] for r in engine._store.get_session_messages("S"))
    engine.ingest([*out, {"role": "user", "content": text}])
    added = [r for r in engine._store.get_session_messages("S") if r["store_id"] > last]
    if form == "glued-request":
        assert any(r["content"].endswith("MY REAL ACTIVE REQUEST — keep every byte.\n") for r in added)
    else:
        assert [r["content"].encode("utf-8") for r in added] == [text.encode("utf-8")]


@pytest.mark.parametrize("source", ["real-quote", "scaffold"])
def test_distinct_user_rows_keep_distinct_identities_across_a_cold_bind(make, source):
    """R3-N2: two different user rows never share a replay identity; the second is stored after a restart."""
    engine = make()
    out, pure, _text = _user_copy(engine, "exact")
    base = pure.split(SEP + CARRY)[0]
    if source == "real-quote":
        first, second = "Please inspect this copied context:" + SEP + base, "Please inspect this copied context:" + SEP + pure
    else:
        first, second = pure, base
    a, b = {"role": "user", "content": first}, {"role": "user", "content": second}
    assert engine._message_replay_identity(a, strip_carrier=False) != engine._message_replay_identity(b, strip_carrier=False)
    engine.ingest([*out, a])
    before = len(engine._store.get_session_messages("S"))
    engine.shutdown()
    resumed = make()
    resumed.ingest([*out, b])
    assert any(r["content"] == second for r in resumed._store.get_session_messages("S")[before:])


def test_legacy_conversation_ids_stay_in_lineage_scope(make):
    from hermes_lcm.db_bootstrap import refresh_legacy_conversation_ids
    engine = make()
    store = engine._store
    values = ["conv", "", None, " \t\n", " conv ", "other", " other "]
    engine.ingest([{"role": "user", "content": f"row-{i}"} for i in range(len(values))])
    ids = [r["store_id"] for r in store.get_session_messages("S")]
    store.connection.executemany("UPDATE messages SET conversation_id=? WHERE store_id=?", list(zip(values, ids)))
    store.connection.commit()
    refresh_legacy_conversation_ids(store.connection)
    readers = [sorted(r["content"] for r in store.load_lineage_user_rows(["S"], " conv ", cursor, 100, newest_first=newest))
               for newest, cursor in ((False, 0), (True, 100000))]
    existing = sorted(r["content"] for r in store.get_range("S", conversation_id="conv", include_blank_conversation=True))
    assert readers[0] == readers[1] == existing == [f"row-{i}" for i in range(5)]


def test_922_objective_only_guard_matches_carrier_and_plain_request(make, tmp_path):
    results = []
    for name in ("carrier", "plain"):
        engine = make(home=tmp_path / name, threshold_full_sweep_enabled=False,
                      fresh_tail_pressure_yield_enabled=False, fresh_tail_max_tokens=12000)
        current = "CURRENT request " + "specific work " * 1600
        out = _compact(engine, [*_history(8), {"role": "user", "content": current},
                               *_tools(70, 50), {"role": "assistant", "content": "a"}])
        row = next(m for m in _host_merge(out) if m.get("role") == "user")
        assert engine._generated_context_carrier_remainder(row) == current
        before = engine.compression_count
        _compact(engine, [row if name == "carrier" else {"role": "user", "content": current},
                          *_tools(80, 50), *_tools(81, 50)])
        results.append((engine.compression_count - before, engine._objective_only_noop, engine._last_compression_status))
    assert results == [(0, True, "noop"), (0, True, "noop")]
