"""#2895: keep verified summaries when a bound literal prefix row cannot fit whole."""

import pytest

from tests.test_issue_2895_nested_carrier import (
    _leaf,
    _summary,
    carrier_session as carrier_session,
    stand_ins as stand_ins,
)


def _append(engine, row):
    return engine._store.append(engine._session_id, row, conversation_id=engine._conversation_id)


def _tail(session, size):
    return [
        session.row(role="assistant" if index % 2 == 0 else "user", content=f"[t{index}] " + "pine " * 35)
        for index in range(size)
    ]


def _assert_fit(engine, rows, budget, summary_parts, kept_parts, expected_ids, *, exit_fit=False, earlier_ids=()):
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(rows), False
    before = engine._survival_measure(rows)
    plan, _ = engine._plan_survival_fit(
        rows, rows, before, before, budget, "exit_fit:test" if exit_fit else "test", exit_fit=exit_fit,
    )

    assert plan is not None
    assert plan.emergency is False
    fitted_texts = [row["content"] for row in plan.fitted]
    assert tuple(index for index, part in enumerate(summary_parts) if any(part in text for text in fitted_texts)) == kept_parts

    stored = engine._store.get_session_messages(engine._session_id, limit=len(rows))
    omitted_ids = tuple(sorted(
        row["store_id"] for row in stored if not any(row["content"] in text for text in fitted_texts)
    ))
    assert tuple(plan.ids) == expected_ids
    assert omitted_ids == (*earlier_ids, *expected_ids)
    assert plan.count == len(expected_ids)
    # T1-341 (2): the per-cut notice names this cut's own count and store-id span.
    per_cut = plan.notice.endswith("; lcm_grep / lcm_load_session reach them.]")
    named_ids = expected_ids if per_cut else omitted_ids
    first, last = named_ids[0], named_ids[-1]
    assert f"{len(named_ids)} earlier messages (store ids {first}..{last})" in plan.notice
    # T1-341 (2): a cumulative notice still names the explicit loader call.
    if not per_cut:
        assert f"session_id={engine._session_id} after_store_id={first - 1} limit={len(omitted_ids)} to {last}" in plan.notice
    assert engine._survival_measure(plan.fitted) <= budget


@pytest.mark.parametrize(("lead", "words", "share", "expected_ids"), [
    pytest.param("scaffold", 120, 0.6, (1, 2, 3, 4, 5), id="scaffold-120-60"),
    pytest.param("scaffold", 120, 0.5, (1, 2, 3, 4, 5, 6, 7), id="scaffold-120-50"),
    pytest.param("scaffold", 120, 0.4, (1, 2, 3, 4, 5, 6, 7, 8), id="scaffold-120-40"),
    pytest.param("scaffold", 300, 0.6, (1, 2, 3, 4), id="scaffold-300-60"),
    pytest.param("scaffold", 300, 0.5, (1, 2, 3, 4, 5, 6), id="scaffold-300-50"),
    pytest.param("scaffold", 300, 0.4, (1, 2, 3, 4, 5, 6, 7), id="scaffold-300-40"),
    pytest.param("carrier", 120, 0.6, (2, 3, 4, 5, 6), id="carrier-120-60"),
    pytest.param("carrier", 120, 0.5, (2, 3, 4, 5, 6, 7, 8), id="carrier-120-50"),
    pytest.param("carrier", 120, 0.4, (2, 3, 4, 5, 6, 7, 8, 9), id="carrier-120-40"),
    pytest.param("carrier", 300, 0.6, (2, 3, 4, 5), id="carrier-300-60"),
    pytest.param("carrier", 300, 0.5, (2, 3, 4, 5, 6, 7), id="carrier-300-50"),
    pytest.param("carrier", 300, 0.4, (2, 3, 4, 5, 6, 7, 8), id="carrier-300-40"),
])
def test_literal_after_verified_prefix(carrier_session, lead, words, share, expected_ids):
    session = carrier_session
    engine = session.engine
    first, second = (_summary([_leaf(engine, label)]) for label in ("a2 leaf zero", "a2 leaf one"))
    leading_text = first
    if lead == "carrier":
        carried_text = "[R0] carried ask " + "ash " * 20
        _append(engine, session.row(role="user", content=carried_text))
        leading_text = f"{first}\n\n{carried_text}"
    leading = {"role": "user", "content": leading_text}
    literal = session.row(role="user", content=f"{second}\n\n[R1] pasted ask " + "oak " * words)
    # A shorter tail makes the 120-word row exceed the 60% whole-row budget too.
    tail = _tail(session, 7 if words == 120 and share == 0.6 else 9)
    for row in [literal, *tail]:
        _append(engine, row)

    rows = [{"role": "system", "content": "sys"}, leading, literal, *tail]
    budget = int(engine._survival_measure(rows) * share)
    _assert_fit(engine, rows, budget, (first, second), (0, 1), expected_ids)


def test_split_literal_notice_also_names_an_earlier_omission(carrier_session):
    session = carrier_session
    engine = session.engine
    first, second = (_summary([_leaf(engine, label)]) for label in ("a2 leaf zero", "a2 leaf one"))
    earlier_id = _append(engine, session.row(role="assistant", content="Earlier omitted reply"))
    literal = session.row(role="user", content=f"{second}\n\n[R1] pasted ask " + "oak " * 300)
    tail = _tail(session, 9)
    for row in [literal, *tail]:
        _append(engine, row)

    rows = [{"role": "system", "content": "sys"}, {"role": "user", "content": first}, literal, *tail]
    budget = int(engine._survival_measure(rows) * 0.6)
    _assert_fit(engine, rows, budget, (first, second), (0, 1), (2, 3, 4, 5), earlier_ids=(earlier_id,))


@pytest.mark.parametrize(("share", "expected_ids"), [
    pytest.param(0.4, (1, 2, 3, 4, 5), id="exit-40"),
    pytest.param(0.5, (1, 2, 3, 4), id="exit-50"),
    pytest.param(0.6, (1, 2), id="exit-60"),
])
def test_exit_fit_splits_covered_literal_at_head(carrier_session, share, expected_ids):
    session = carrier_session
    engine = session.engine
    engine._config.fresh_tail_count = 2
    part = _summary([_leaf(engine, "x leaf one")])
    literal = session.row(role="user", content=f"{part}\n\n[R1] pasted ask " + "oak " * 300)
    tail = _tail(session, 7)
    source_ids = [_append(engine, row) for row in [literal, *tail]]
    _leaf(engine, "x cover", source_ids=source_ids)

    rows = [{"role": "system", "content": "sys"}, literal, *tail]
    budget = int(engine._survival_measure(rows) * share)
    _assert_fit(engine, rows, budget, (part,), (0,), expected_ids, exit_fit=True)


@pytest.mark.parametrize("head_tokens", [480, 660], ids=["0.8x-budget", "1.1x-budget"])
def test_large_literal_head_keeps_its_verified_span(carrier_session, head_tokens):
    session = carrier_session
    engine = session.engine
    first, second = (_summary([_leaf(engine, label)]) for label in ("v2 zero", "v2 one"))
    literal = session.row(role="user", content=f"{first}\n\n{second}\n\nreal ask v2: ")
    while engine._survival_measure([literal]) < head_tokens:
        literal["content"] += "cedar "
    tail = [
        session.row(role="assistant" if index % 2 == 0 else "user", content=f"<v2-{index}> " + "maple " * 24)
        for index in range(9)
    ]
    for row in [literal, *tail]:
        _append(engine, row)

    rows = [{"role": "system", "content": "sys v2"}, literal, *tail]
    _assert_fit(engine, rows, 600, (first, second), (0,), (1, 2))
