"""I1 cold and warm replay cells for a bound UID whose host presentation changes (#2910, LCM_HOST_MESSAGE_UID=on).

Measured on the v0.30.2 release line and on v0.30.1 with identical results. A cold bind over a tool result that the
host replaced by its own persisted-output stub, while the store holds LCM's externalized stub, stops the uid walk at
the call row and stores the call and its result again (2 copies). That cell is a strict expected failure documenting
#2910. Cleared-output placeholders and a real changed version store no copy. The warm changed-version cell documents
the separate #2116 limit: an unstamped edit below the resume point is not stored on the warm path."""

import json
import logging
from collections import Counter
from copy import deepcopy

import pytest

from hermes_lcm import reconcile
from hermes_lcm.config import uid_recognition_enabled
from tests.invariants.host_uid_fixtures import non_uid_host  # noqa: F401
from tests.invariants.test_issue_2781_placeholder_walk import (
    PRUNED,
    _bindings,
    _group,
    _msg,
    _open,
)
from tests.test_issue_10_persisted_output_spillover import FULL_OUTPUT, _write_output
from tests.test_issue_581_582_store_complete_survival import _rows

pytestmark = pytest.mark.issue(2910)

_PREFIX_ROWS = 64
_BUDGET_EXHAUSTED = "LCM UID cut budget exhausted:"


class _UnchangedHostCopy(AssertionError):
    pass


class _MissingEditedVersion(AssertionError):
    pass


def _stored_identity(engine, row):
    return (
        row["role"],
        row.get("content") or "",
        row.get("tool_call_id"),
        row.get("tool_name") or row.get("name"),
        engine._stable_tool_calls_identity(row.get("tool_calls")),
    )


def _seed(engine, path, shape):
    head = [_msg(f"prefix-{i}", "user" if i % 2 == 0 else "assistant") for i in range(_PREFIX_ROWS)]
    suffix = [_msg(f"suffix-{i}", "user" if i % 2 == 0 else "assistant") for i in range(_PREFIX_ROWS + 4)]
    if shape == "changed-version":
        # #2116 applies to replies without a host timestamp.
        original = [_msg("edited")]
        edited = dict(original[0], content="Edited assistant answer with new facts")
        offset = 0
    else:
        original = _group("target", 1, payload=FULL_OUTPUT)
        if shape == "cleared-output":
            content = PRUNED
        else:
            content = _write_output(path / "cache" / "spillover" / "target.txt", FULL_OUTPUT)
        edited = dict(original[1], content=content)
        offset = 1

    engine._config.large_output_externalization_enabled = shape == "persisted-output"
    engine._config.large_output_externalization_threshold_chars = 200
    engine._config.large_output_externalization_path = str(path / "externalized")
    history = [*head, *original, *suffix]
    engine.ingest(deepcopy(history))
    baseline = _rows(engine)
    assert len(baseline) == len(history)
    assert _bindings(engine, history) == {
        message["message_uid"]: [(row["store_id"], "canonical")]
        for message, row in zip(history, baseline)
    }
    stop = len(head) + offset
    target = baseline[stop]
    if shape == "persisted-output":
        assert target["content"].startswith("[Externalized tool output:")
        payloads = list((path / "externalized").glob("*.json"))
        assert len(payloads) == 1
        assert json.loads(payloads[0].read_text())["content"] == FULL_OUTPUT
    else:
        assert target["content"] == original[offset]["content"]
    assert target["content"] != edited["content"]
    assert all(
        _stored_identity(engine, message) == _stored_identity(engine, row)
        for message, row in zip(history[stop + 1:], baseline[stop + 1:])
    )

    if shape == "cleared-output":
        for _ in range(2):
            engine._store.append("S", original[0], conversation_id="conv")
            engine._store.append("S", edited, conversation_id="conv")
    # Keep a durable row outside the host snapshot so the UID-located durable
    # tail cannot short-circuit the content-cut search.
    engine._store.append("S", _msg("outside-host-view"), conversation_id="conv")
    fresh = _msg("never-stored", "user")
    view = [*history[:stop], edited, *history[stop + 1:], fresh]
    return view, stop, target


def _measure(engine, view, stop, target, shape, temperature, cut_budget, monkeypatch, caplog):
    assert uid_recognition_enabled(engine)
    before = _rows(engine)
    prefix_cursors = []
    searches = []
    resolve = engine._host_uid_resolve
    search = engine._find_reconciled_cursor_for_store_tail

    def observe_prefix(messages, *args, **kwargs):
        matches = resolve(messages, *args, **kwargs)
        if kwargs.get("prefix_only"):
            prefix_cursors.append(next((i for i in range(len(messages)) if i not in matches), len(messages)))
        return matches

    def observe_search(*args, **kwargs):
        cursor = search(*args, **kwargs)
        searches.append({
            "uid_cursor": kwargs.get("uid_cursor"),
            "cut_budget": kwargs.get("cut_budget"),
            "content_cut": kwargs.get("content_cut"),
            "result": cursor,
        })
        return cursor

    caplog.clear()
    with (
        monkeypatch.context() as observe,
        caplog.at_level(logging.INFO, logger="hermes_lcm.reconcile"),
        caplog.at_level(logging.DEBUG, logger="hermes_lcm.engine"),
    ):
        if cut_budget is not None:
            observe.setattr(reconcile, "_UID_ON_CURSOR_CUT_BUDGET", cut_budget)
        observe.setattr(engine, "_host_uid_resolve", observe_prefix)
        observe.setattr(engine, "_find_reconciled_cursor_for_store_tail", observe_search)
        engine.ingest(deepcopy(view))

    exhausted = [msg for msg in caplog.messages if msg.startswith(_BUDGET_EXHAUSTED)]
    ingest_cursors = [
        record.args[1] for record in caplog.records
        if record.name == "hermes_lcm.engine" and record.msg == "Ingest: session=%s cursor=%d incoming=%d"
    ]
    after = _rows(engine)
    before_ids = {row["store_id"] for row in before}
    missing_ids = before_ids - {row["store_id"] for row in after}
    inserted = [row for row in after if row["store_id"] not in before_ids]
    before_counts = Counter(_stored_identity(engine, row) for row in before)
    after_counts = Counter(_stored_identity(engine, row) for row in after)
    host_indices = {_stored_identity(engine, row): i for i, row in enumerate(view)}
    replacement = view[stop]
    unchanged_keys = {
        _stored_identity(engine, row)
        for row in view[:-1]
        if row["message_uid"] != replacement["message_uid"]
    }
    added_copies = [
        row for row in inserted if _stored_identity(engine, row) in before_counts
    ]
    unchanged_copies = [row for row in added_copies if _stored_identity(engine, row) in unchanged_keys]
    fresh_count = after_counts[_stored_identity(engine, view[-1])]
    version_count = after_counts[_stored_identity(engine, replacement)]
    version_ids = [
        sid for sid, kind in _bindings(engine, [replacement]).get(replacement["message_uid"], ())
        if kind == "version" and sid not in before_ids
    ]
    missing_new_host_indices = []
    if not fresh_count:
        missing_new_host_indices.append(len(view) - 1)
    if shape == "changed-version" and not version_count:
        missing_new_host_indices.append(stop)
    print(json.dumps({
        "shape": shape,
        "temperature": temperature,
        "cut_budget_override": cut_budget,
        "host_rows": len(view),
        "stop_row_index": stop,
        "stop_row_canonical_id": target["store_id"],
        "uid_prefix_cursors": prefix_cursors,
        "content_searches": searches,
        "cut_budget_exhausted": exhausted,
        "reconciled_cursor": engine._last_ingest_reconciliation.get("cursor") if prefix_cursors else None,
        "ingest_cursors": ingest_cursors,
        "before": len(before),
        "after": len(after),
        "inserted": [{
            "store_id": row["store_id"],
            "role": row["role"],
            "host_index": host_indices.get(_stored_identity(engine, row)),
            "content": (row.get("content") or "")[:96],
            "copy": _stored_identity(engine, row) in before_counts,
        } for row in inserted],
        "copies": len(added_copies),
        "unchanged_copies": len(unchanged_copies),
        "unchanged_suffix_copies": sum(host_indices[_stored_identity(engine, row)] > stop for row in unchanged_copies),
        "missing_store_ids": sorted(missing_ids),
        "missing_new_host_indices": missing_new_host_indices,
        "fresh_count": fresh_count,
        "replacement_count": version_count,
        "new_version_store_ids": version_ids,
    }), flush=True)

    lost_store_rows = missing_ids or len(after) < len(before)
    if temperature == "cold" and (lost_store_rows or missing_new_host_indices):
        pytest.fail(  # Failed is not an AssertionError, so the xfail marks cannot absorb a loss
            f"#2910 release stop: cold-{shape}, cut budget override {cut_budget}; see the cell just printed"
        )

    assert not lost_store_rows
    assert fresh_count == 1
    if temperature == "cold" and cut_budget is not None:
        assert any(search["cut_budget"] == cut_budget for search in searches)
        assert exhausted
    if unchanged_copies:
        raise _UnchangedHostCopy(f"Saved {len(unchanged_copies)} extra unchanged host row(s)")
    if shape == "changed-version":
        if not version_count:
            raise _MissingEditedVersion("The edited assistant reply was not saved")
        assert version_count == 1
        assert len(version_ids) == 1
        assert _stored_identity(engine, engine._store.get(version_ids[0])) == _stored_identity(engine, replacement)


@pytest.mark.parametrize("cut_budget", [pytest.param(None, id="default-budget"), pytest.param(1, id="cut-budget-1")])
@pytest.mark.parametrize(
    ("temperature", "shape"),
    [
        pytest.param("cold", "cleared-output", id="cold-cleared-output"),
        pytest.param(
            "cold", "persisted-output", id="cold-persisted-output",
            marks=pytest.mark.xfail(
                strict=True, raises=_UnchangedHostCopy,
                reason="#2910 cursor 64, 135 to 138 saved rows, 2 copies including 1 unchanged host row",
            ),
        ),
        pytest.param("cold", "changed-version", id="cold-changed-version"),
        pytest.param("warm", "cleared-output", id="warm-cleared-output"),
        pytest.param("warm", "persisted-output", id="warm-persisted-output"),
        pytest.param(
            "warm", "changed-version", id="warm-changed-version",
            marks=pytest.mark.xfail(strict=True, raises=_MissingEditedVersion, reason="#2116 known limit"),
        ),
    ],
)
def test_uid_walk_stop(tmp_path, monkeypatch, caplog, shape, temperature, cut_budget):
    # Supply the host UIDs ourselves, even though the shared fixture disables
    # generated UID emission for the host stub. Recognition is a separate gate.
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    path = tmp_path / "store"
    with _open(path) as seed:
        view, stop, target = _seed(seed, path, shape)
        if temperature == "warm":
            _measure(seed, view, stop, target, shape, temperature, cut_budget, monkeypatch, caplog)
            return

    with _open(path, restart=True) as cold:
        cold._config.large_output_externalization_enabled = shape == "persisted-output"
        cold._config.large_output_externalization_threshold_chars = 200
        cold._config.large_output_externalization_path = str(path / "externalized")
        _measure(cold, view, stop, target, shape, temperature, cut_budget, monkeypatch, caplog)
