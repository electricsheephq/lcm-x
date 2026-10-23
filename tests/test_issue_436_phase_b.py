"""#436 rc1 Phase B: B-ID-1 (a repeated user text merged into a dangling R) and B-ID-2 (a profile
rebind reusing the identity-anchor caches against another database)."""

from __future__ import annotations

from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_436_identity_anchor import (
    PAD,
    _a,
    _assert_claims_are_in_the_input,
    _engine,
    _relations,
    _rows,
    _turns,
    _u,
    summaries,  # noqa: F401 (fixture)
)


def _stored(rows) -> Counter:
    return Counter((row["role"], str(row["content"])) for row in rows)


def _merge_turn(engine, tmp_path, new_u, *, host_object, restart, u_stamp=1.0, reply=True):
    """[U@1, A@2, R@3], then Hermes' _merge_consecutive_users folds the new U into the dangling R
    (R's dict keeps its stamp): [U@1, A@2, "R\\n\\nU"@3, reply@5]."""
    u, a, r = _u(new_u[0], u_stamp), _a("reply to U" + PAD, 2.0), _u("[R] failed turn" + PAD, 3.0)
    r_text = r["content"]
    engine.ingest([u, a, r] if reply else [u, r])
    if restart:
        engine.shutdown()
        engine = _engine(tmp_path)
    if host_object == "in-place":
        composite = r
        composite["content"] = r_text + "\n\n" + new_u[1]
        live = [u, a, composite, _a("reply to R+U", 5.0)]
    else:
        live = [dict(u), dict(a), _u(r_text + "\n\n" + new_u[1], 3.0), _a("reply to R+U", 5.0)]
    if not reply:
        live.remove(a)
    engine.ingest(live)
    return engine, live, r_text


def _assert_shown_row_is_no_constituent(engine, text):
    """The host view shows the earlier U as its own occurrence: it is never a composite constituent."""
    shown = min(int(row["store_id"]) for row in _rows(engine) if row["content"] == text)
    assert not [rel for rel in _relations(engine) if rel[1] == "composite" and rel[2] == shown]


LONG = "[U] please go on with the report" + PAD


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart"])
@pytest.mark.parametrize("host_object", ["in-place", "new-dict"])
@pytest.mark.parametrize("text, second", [
    ("continue", "continue"), (LONG, LONG),  # B-ID-1: the new U repeats an earlier user row
    ("continue", "continue (again)"), (LONG, LONG + " again"),  # control: distinct text
], ids=["repeat-continue", "repeat-long", "control-continue", "control-long"])
def test_bid1_merged_user_turn_is_stored_even_when_its_text_repeats_an_earlier_row(tmp_path, text, second, host_object, restart):
    engine = _engine(tmp_path)
    try:
        engine, _live, r_text = _merge_turn(engine, tmp_path, (text, second), host_object=host_object, restart=restart)
        expected = Counter([("user", text), ("assistant", "reply to U" + PAD), ("user", r_text),
                            ("user", second), ("assistant", "reply to R+U")])
        assert _stored(_rows(engine)) == expected
        _assert_shown_row_is_no_constituent(engine, text)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart"])
@pytest.mark.parametrize("reply", [True, False], ids=["replied", "unreplied"])
@pytest.mark.parametrize("text", ["continue", LONG], ids=["continue", "long"])
def test_bid1_an_unstamped_earlier_row_is_reserved_too(tmp_path, text, reply, restart):
    """Mixed legacy/current history: the earlier U has no host stamp (stored NULL, shown unstamped).
    It is still the view's own occurrence, so the new repeated U is stored, not absorbed."""
    engine = _engine(tmp_path)
    try:
        engine, _live, r_text = _merge_turn(engine, tmp_path, (text, text), host_object="in-place", restart=restart,
                                            u_stamp=None, reply=reply)
        expected = Counter([("user", text), ("user", r_text), ("user", text), ("assistant", "reply to R+U")]
                           + [("assistant", "reply to U" + PAD)] * reply
                           # A restart whose list no longer proves the stored tail re-stores an unstamped row
                           # (pre-existing: nothing keys it; flag off re-stores the whole list): duplication, never loss.
                           + [("user", text)] * restart)
        assert _stored(_rows(engine)) == expected
        _assert_shown_row_is_no_constituent(engine, text)
    finally:
        engine.shutdown()


def test_bid1_merge_turn_compacts_with_contiguous_coverage(tmp_path, summaries):  # noqa: F811
    """After the B-ID-1 merge turn a forced compaction commits, covers a contiguous prefix that
    includes the repeated U, and never claims a source whose text is not in its input."""
    engine = _engine(tmp_path)
    try:
        engine, live, _r_text = _merge_turn(engine, tmp_path, ("continue", "continue"), host_object="in-place", restart=False)
        live = [*live, *_turns(10, 4, 600.0)]
        engine.ingest(live)
        statuses = []
        for _ in range(3):
            live = engine.compress(live, force=True)
            statuses.append(engine._last_compression_status)
        assert "error" not in statuses and "compacted" in statuses, engine._last_compression_noop_reason
        claimed = set(_assert_claims_are_in_the_input(engine, summaries))
        frontier = int(engine._last_compacted_store_id or 0)
        owned = [int(row["store_id"]) for row in _rows(engine)]
        assert [sid for sid in owned if sid <= frontier] == sorted(claimed & set(owned))
        repeats = [int(row["store_id"]) for row in _rows(engine) if row["content"] == "continue"]
        assert len(repeats) == 2 and all(sid <= frontier for sid in repeats)
    finally:
        engine.shutdown()


def _home_engine(home) -> LCMEngine:
    home.mkdir(parents=True, exist_ok=True)
    return LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1), hermes_home=str(home))


def _start(engine, home) -> None:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv", hermes_home=str(home))


@pytest.mark.parametrize("engine_kind", ["reused-from-profile-A", "fresh-engine"])
def test_bid2_profile_rebind_never_aliases_another_databases_row(tmp_path, engine_kind):
    """A / S stores "R\\n\\nU"@20 as row 1; B / S holds X@5 (row 1) and R@10 (row 2). A's engine
    rebinds to B, replays B, then sees [X@5, "R\\n\\nU"@10, reply@30]: U is stored in B, and no
    alt_stamp points at B's unrelated row 1."""
    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    r_text, u_text = "[R] prompt in profile B" + PAD, "[U] follow-up"
    x, r = _a("[X] assistant row in B" + PAD, 5.0), _u(r_text, 10.0)
    seed = _home_engine(home_b)
    try:
        _start(seed, home_b)
        seed.ingest([x, r])
    finally:
        seed.shutdown()
    engine = _home_engine(home_a if engine_kind == "reused-from-profile-A" else home_b)
    try:
        if engine_kind == "reused-from-profile-A":
            _start(engine, home_a)
            engine.ingest([_u(r_text + "\n\n" + u_text, 20.0)])
        _start(engine, home_b)
        engine.ingest([dict(x), dict(r)])
        engine.ingest([dict(x), _u(r_text + "\n\n" + u_text, 10.0), _a("reply to R+U", 30.0)])
        rows = _rows(engine)
        assert _stored(rows) == Counter([("assistant", x["content"]), ("user", r_text), ("user", u_text),
                                         ("assistant", "reply to R+U")])
        assert not [rel for rel in _relations(engine) if rel[1] == "alt_stamp"]
    finally:
        engine.shutdown()


# -- round 3: a row answers any of its forms (exact, host-rewrite): the reservation, R1 and gap-fill match as a
# whole (maximum matching), never greedily per row. The adversarial form order is forced, not hash-seed luck.

def _forms_first(engine, monkeypatch, first: dict) -> None:
    """``_stored_row_forms`` of a row whose content is a key of ``first`` lists those texts, in that order."""
    original = engine._stored_row_forms

    def forms(row):
        exact = next(iter(original(row)))
        texts = first.get(str(row.get("content")))
        return original(row) if texts is None else [(exact[0], text, *exact[2:]) for text in texts]

    monkeypatch.setattr(engine, "_stored_row_forms", forms)


def _ids(engine, text):
    return [int(row["store_id"]) for row in _rows(engine) if row["content"] == text]


@pytest.mark.parametrize("stamp, a_text, b_text, restart", [
    (1.0, "alpha", "beta", False),  # one stamp: A answers {alpha, beta}, B only beta; the view shows both
    (None, "foo ", "foo", True),  # NULL stamps after a restart: A "foo " (override "foo"), B "foo"
], ids=["stamped-steady", "null-restart"])
def test_r3_a_multi_form_row_never_frees_a_shown_row_into_a_composite(tmp_path, monkeypatch, stamp, a_text, b_text, restart):
    engine = _engine(tmp_path)
    try:
        a, b, r = _u(a_text, stamp), _u(b_text, stamp), _u("[R] failed turn" + PAD, 3.0)
        head = [a, b, _a("reply" + PAD, 2.0)]
        engine.ingest([*head, r])
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        _forms_first(engine, monkeypatch, {a_text: [b_text, a_text]})
        shown = set(_ids(engine, a_text) + _ids(engine, b_text))
        r["content"] += "\n\n" + b_text  # the new U repeats B's text, merged into the dangling R
        engine.ingest([*head, r, _a("reply to R+U", 5.0)])
        members = [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        assert not shown & set(members), (shown, members)
        assert set(_ids(engine, b_text)) & set(members)  # the new U: stored, its own occurrence
    finally:
        engine.shutdown()


def test_r3_gap_fill_never_rehydrates_a_row_the_view_shows(tmp_path, monkeypatch):
    """R4 gap-fill: A answers {alpha, beta}, B only beta, the view shows alpha and beta: both rows are shown,
    none is rehydrated into the summarizer input."""
    engine = _engine(tmp_path)
    try:
        x = _u("[X] later user row" + PAD, 2.0)
        engine.ingest([_u("alpha", 1.0), _u("beta", 1.0), x])
        _forms_first(engine, monkeypatch, {"alpha": ["beta", "alpha"]})
        [b_id], [x_id] = _ids(engine, "beta"), _ids(engine, x["content"])
        out = engine._identity_anchor_summary_input([x], {id(x): x_id}, view=[_u("alpha", 1.0), _u("beta", 1.0), x])
        assert not [ids for _row, ids in out or () if b_id in ids], out
    finally:
        engine.shutdown()


def test_r3_r1_a_multi_form_row_leaves_the_single_form_row_its_occurrence(tmp_path, monkeypatch):
    """R1: A answers {xval, yval} (listed first in store order), B only xval; the host re-issues both. A takes
    yval and B xval: nothing is stored again."""
    engine = _engine(tmp_path)
    try:
        a, b = _u("yval", 1.0), _u("xval", 1.0)
        engine.ingest([_u("[P] dropped prefix" + PAD, 0.5), a, b, _a("reply" + PAD, 2.0)])
        _forms_first(engine, monkeypatch, {"yval": ["xval", "yval"]})
        engine.ingest([dict(b), dict(a), _a("reply" + PAD, 2.0), _u("[N] new" + PAD, 5.0)])  # the prefix row left
        assert [row["content"] for row in _rows(engine)].count("yval") == 1
        assert [row["content"] for row in _rows(engine)].count("xval") == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart"])
@pytest.mark.parametrize("text", ["foo", LONG], ids=["short", "long"])
def test_r3f_a_null_stored_row_the_host_shows_stamped_is_reserved(tmp_path, text, restart):
    """Bot P1: LCM stored U before the host stamped it (NULL); the host then shows U at its state.db stamp below
    the cursor and folds a new U repeating its text into R: stored U@NULL, R@3; shown U@1, "R\\n\\nU"@3."""
    engine = _engine(tmp_path)
    try:
        r = _u("[R] failed turn" + PAD, 3.0)
        engine.ingest([_u(text, None), r])
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        engine.ingest([_u(text, 1.0), _u(r["content"] + "\n\n" + text, 3.0), _a("reply to R+U", 5.0)])
        expected = [("user", text)] * 2 + [("user", r["content"]), ("assistant", "reply to R+U")]
        assert _stored(_rows(engine)) == Counter(expected + [("user", text)] * restart)  # restart re-store: as above
        _assert_shown_row_is_no_constituent(engine, text)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shown_alone", [1, 2], ids=["one-shown", "both-shown"])
@pytest.mark.parametrize("text", ["foo", LONG], ids=["short", "long"])
def test_r3f_two_null_rows_one_stamped_occurrence_reserves_the_first_in_store_order(tmp_path, text, shown_alone):
    """#583 (a): stored U1@NULL, R@3, U2@NULL (same text); the host shows U alone ``shown_alone`` times, stamped,
    and "R\\n\\nU"@3. One occurrence reserves U1 (store order) and the composite absorbs U2; a genuine second
    occurrence reserves U2 too, so the composite's U is new and stored."""
    engine = _engine(tmp_path)
    try:
        r = _u("[R] failed turn" + PAD, 3.0)
        engine.ingest([_u(text, None), r, _u(text, None)])
        u1, u2 = _ids(engine, text)
        engine.ingest([_u(text, 1.0 + k) for k in range(shown_alone)]
                      + [_u(r["content"] + "\n\n" + text, 3.0), _a("reply to R+U", 5.0)])
        members = [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        assert u1 not in members
        assert (u2 in members) == (shown_alone == 1), members
        # #1403: an unproven moved NULL occurrence is retained, not consumed.
        assert _stored(_rows(engine))[("user", text)] == 2 + 2 * (shown_alone - 1)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("other", [False, True], ids=["alone", "other-stamped-row"])
def test_r3f_a_null_row_shown_only_inside_a_composite_is_no_reservation(tmp_path, other):
    """#583 (b): stored U@NULL, R@3; the host shows U only inside "R\\n\\nU"@3 (a stamped row of another text
    left over does not reserve it): the composite still decomposes into R and U, nothing is stored again."""
    engine = _engine(tmp_path)
    try:
        r, x = _u("[R] failed turn" + PAD, 3.0), _u("[X] other row" + PAD, 1.0)
        engine.ingest([x, _u("foo", None), r] if other else [_u("foo", None), r])
        [u] = _ids(engine, "foo")
        engine.ingest([dict(x)] * other + [_u(r["content"] + "\n\n" + "foo", 3.0), _a("reply to R+U", 5.0)])
        assert u in [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        assert _stored(_rows(engine))[("user", "foo")] == 1
    finally:
        engine.shutdown()


P_TEXT, Q_TEXT = "[P] first part" + PAD, "[Q] second part" + PAD


@pytest.mark.parametrize("text", [("P", "Q"), (P_TEXT, Q_TEXT)], ids=["short", "long"])
def test_bid3_a_row_the_host_shows_under_its_recorded_alias_stamp_is_reserved(tmp_path, text):
    """B-ID-3 (fresh host dicts each ingest, U = "P\\n\\nQ"): [P@10, U@20]; then [U@10, A@21, R@30] (R5 records
    alt_stamp 10 on U's row, observed_at 20); then a NEW U merged into the failed R: [U@10, A@21, "R\\n\\nU"@30,
    A@40]. U@10 reserves U's row through its alias: the composite's U is new and stored, never absorbed
    into the old row (loss)."""
    engine = _engine(tmp_path)
    try:
        u_text, r = "\n\n".join(text), _u("[R] failed turn" + PAD, 30.0)
        engine.ingest([_u(text[0], 10.0), _u(u_text, 20.0)])
        engine.ingest([_u(u_text, 10.0), _a("reply to U" + PAD, 21.0), dict(r)])
        [u] = _ids(engine, u_text)
        assert [rel for rel in _relations(engine) if rel[1] == "alt_stamp"] == [(u, "alt_stamp", None, None)]
        before, composite = max(int(row["store_id"]) for row in _rows(engine)), r["content"] + "\n\n" + u_text
        engine.ingest([_u(u_text, 10.0), _a("reply to U" + PAD, 21.0), _u(composite, 30.0), _a("reply to R+U", 40.0)])
        members = [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        assert u not in members
        if members:  # R3: stored constituents (P, not shown on its own, may head it) plus a remainder stored anew
            content = {int(row["store_id"]): row["content"] for row in _rows(engine)}
            assert "\n\n".join(content[m] for m in members) == composite and max(members) > before, members
        else:
            stored = _stored(_rows(engine))
            assert stored[("user", u_text)] == 2 or stored[("user", composite)] == 1, stored
    finally:
        engine.shutdown()


def test_bid3_an_alias_reserves_only_its_own_row(tmp_path):
    """B-ID-3 precision: U's first row carries alias 10; a second stored U@35 (LCM stored it before the host
    folded it into R) has no alias. U@10 reserves the first row only; the composite still absorbs U@35."""
    engine = _engine(tmp_path)
    try:
        u_text, r = "\n\n".join((P_TEXT, Q_TEXT)), _u("[R] failed turn" + PAD, 30.0)
        engine.ingest([_u(P_TEXT, 10.0), _u(u_text, 20.0)])
        engine.ingest([_u(u_text, 10.0), _a("reply to U" + PAD, 21.0), dict(r)])
        engine.ingest([_u(u_text, 10.0), _a("reply to U" + PAD, 21.0), dict(r), _u(u_text, 35.0)])
        first, second = _ids(engine, u_text)
        engine.ingest([_u(u_text, 10.0), _a("reply to U" + PAD, 21.0), _u(r["content"] + "\n\n" + u_text, 30.0),
                       _a("reply to R+U", 40.0)])
        members = [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        assert first not in members and second in members, members
        assert _stored(_rows(engine))[("user", u_text)] == 2
    finally:
        engine.shutdown()



def _r2_history(engine, variant):
    """Stored U and R@30; the view then shows U alone at @10 (stored at 10, NULL, or -- U = "P\\n\\nQ" -- at 20
    with the recorded R5 alias 10) and a NEW U merged into the failed R: ``(u_text, u_old, view, composite)``."""
    r = _u("[R] failed turn" + PAD, 30.0)
    if variant == "alias":
        u_text = P_TEXT + "\n\n" + Q_TEXT
        engine.ingest([_u(P_TEXT, 10.0), _u(u_text, 20.0)])
        engine.ingest([_u(u_text, 10.0), _a("reply to U" + PAD, 21.0), dict(r)])
        assert [rel for rel in _relations(engine) if rel[1] == "alt_stamp"]
        head = [_u(u_text, 10.0), _a("reply to U" + PAD, 21.0)]
    else:
        u_text = "[U] please go on" + PAD
        engine.ingest([_u(u_text, 10.0 if variant == "stamped" else None), dict(r)])
        head = [_u(u_text, 10.0)]
    [u_old] = _ids(engine, u_text)
    composite = _u(r["content"] + "\n\n" + u_text, 30.0)
    return u_text, u_old, [*head, composite, _a("reply to R+U", 40.0)], composite



def _assert_new_occurrence_kept(engine, u_text, u_old, composite):
    """U_new is stored: a second U (or the composite whole), or R3's held constituents -- never U_old -- plus a
    remainder stored anew (the alias variant's U = "P\\n\\nQ" decomposes into R, P and a new Q)."""
    stored = _stored(_rows(engine))
    if stored[("user", u_text)] + stored[("user", composite)] >= 2:
        return
    content = {int(row["store_id"]): row["content"] for row in _rows(engine)}
    groups: dict = {}
    for head, kind, member, ordinal in _relations(engine):
        if kind == "composite":
            groups.setdefault(head, []).append((ordinal, member))
    members = [[sid for _o, sid in sorted(group)] for group in groups.values()]
    assert any("\n\n".join(content[m] for m in group) == composite and u_old not in group
               and max(group) > u_old for group in members), (stored, members)

@pytest.mark.parametrize("mapped", [True, False], ids=["mapped", "unmapped"])
@pytest.mark.parametrize("variant", ["stamped", "null", "alias"])
def test_r2_publication_never_composes_a_row_the_view_shows_on_its_own(tmp_path, variant, mapped):
    """PR #590 r2 (codex sol-high item 6): the publication-side composition (form i) of the unmapped composite
    "R\\n\\nU_new"@30 never takes U, the row the view shows on its own at @10 -- mapped by the live row, or
    (unmapped) held by that occurrence under its stamp, NULL or alias key. Else the witness [R, U_old] makes the
    next ingest replay the composite once R1 consumed U: U_new lost."""
    engine = _engine(tmp_path)
    try:
        u_text, u_old, view, composite = _r2_history(engine, variant)
        full_map = engine._get_store_id_map_for_messages(view)
        if not mapped:
            full_map = {key: sid for key, sid in full_map.items() if sid != u_old}
        engine._identity_anchor_summary_input([composite], full_map, view=view)
        assert u_old not in [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        engine.ingest([dict(m) for m in view])
        _assert_new_occurrence_kept(engine, u_text, u_old, composite["content"])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("variant", ["stamped", "null", "alias"])
def test_r2_compress_before_ingest_keeps_the_new_occurrence(tmp_path, summaries, variant):  # noqa: F811
    """The same history through compress() before any further ingest (its preflight ingest runs first, so the
    ingest site's reservation records the witness [R, U_new]): the new occurrence is stored, claims stay in the
    input."""
    engine = _engine(tmp_path)
    try:
        u_text, u_old, view, composite = _r2_history(engine, variant)
        live = [dict(m) for m in view] + _turns(10, 4, 600.0)
        for _ in range(4):
            live = engine.compress(live, force=True)
            assert engine._last_compression_status != "error", engine._last_compression_noop_reason
        assert u_old not in [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        engine.ingest([dict(m) for m in live] + [_u("[N] next" + PAD, 900.0)])
        _assert_new_occurrence_kept(engine, u_text, u_old, composite["content"])
        _assert_claims_are_in_the_input(engine, summaries)
    finally:
        engine.shutdown()


def test_r2_a_null_row_is_reserved_when_a_multi_form_stamped_row_can_move(tmp_path):
    """PR #590 thread PRRT_kwDOT3S3V86mov1Z, REAL forms (no monkeypatch): N "foo " stored NULL; S "foo "@5, which
    the host rewrote in place to "foo" (edge whitespace: the override form), so S answers {"foo ", "foo"}. The
    host shows "foo "@5, S "foo"@5 and a new "foo " merged into the failed R. S moves to "foo", N keeps "foo "@5:
    the composite's "foo " is new and stored, never absorbed into N."""
    engine = _engine(tmp_path)
    try:
        n, s, r = _u("foo ", None), _u("foo ", 5.0), _u("[R] failed turn" + PAD, 30.0)
        engine.ingest([n])
        engine.ingest([n, s])
        s["content"] = "foo"  # the host rewrites S's own object in place
        engine.ingest([n, s, r])
        n_id, s_id = _ids(engine, "foo ")
        assert {form[1] for form in engine._stored_row_forms(_rows(engine)[1])} == {"foo ", "foo"}
        engine.ingest([_u("foo ", 5.0), dict(s), _u(r["content"] + "\n\nfoo ", 30.0), _a("reply to R+U", 40.0)])
        members = [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        assert n_id not in members and s_id not in members, members
        stored = _stored(_rows(engine))
        assert stored[("user", "foo ")] + stored[("user", r["content"] + "\n\nfoo ")] >= 3, stored
    finally:
        engine.shutdown()
