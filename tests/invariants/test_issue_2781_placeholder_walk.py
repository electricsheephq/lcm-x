"""Synthetic store and host-list regressions for UID-bound pruned tool results."""

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from hermes_lcm import host_uid_emit, store_complete
from hermes_lcm.config import LCMConfig, uid_recognition_enabled
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER as PRUNED
from hermes_lcm.summary_input_clip import CLIP_MARKER
from hermes_lcm.tokens import count_messages_tokens
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _covered, _frontier, _rows, summaries as summaries

pytestmark = pytest.mark.issue(2781)
PAD = " amber azure lilac ochre" * 16
LOG = "hermes_lcm.store_complete"


def _mode(monkeypatch, mode):
    if mode == "unset":
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)


@contextmanager
def _open(path, *, restart=False):
    if not restart:
        path.mkdir()
        _state_db(path, [("S", None, None)])
    engine = LCMEngine(config=LCMConfig(
        database_path=str(path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=1,
        dynamic_leaf_chunk_enabled=False, threshold_full_sweep_enabled=False,
        temporal_rollups_enabled=False, embeddings_enabled=False, survival_fit=False,
        large_output_active_replay_stubbing_enabled=False,
    ), hermes_home=str(path))
    engine.on_session_start("S", platform="telegram", context_length=48_000, conversation_id="conv")
    engine.threshold_tokens = 100
    try:
        yield engine
    finally:
        engine.shutdown()


def _msg(uid, role="assistant", content=None, **fields):
    return {"role": role, "content": uid + PAD if content is None else content, "message_uid": uid, **fields}


def _group(tag, count=2, *, ids=None, payload=None):
    ids = ids or [f"{tag}-{i}" for i in range(count)]
    caller = _msg(tag, content="", tool_calls=[
        {"id": call, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"tag": tag})}}
        for call in ids
    ])
    return [caller, *[_msg(f"{tag}-output-{i}", "tool", payload, tool_call_id=call, tool_name="lookup")
                      for i, call in enumerate(ids)]]


def _pruned(group):
    return [group[0], *(dict(row, content=PRUNED) for row in group[1:])]


def _no_uid(row):
    return {key: value for key, value in row.items() if key != "message_uid"}


def _lineage(engine):
    lineage, reason = engine._host_uid_lineage_key()
    assert lineage is not None, reason
    return lineage


def _bindings(engine, rows):
    return engine._store.host_uid_bindings_for(_lineage(engine), {row["message_uid"] for row in rows})


def _canonical(engine, row):
    sid, = [sid for sid, kind in _bindings(engine, [row])[row["message_uid"]] if kind == "canonical"]
    return sid


def _pass(engine, view, *, force=True):
    # The running Hermes list can replace a tool's output in place and reorder
    # already-ingested rows. Its new presentation has no un-ingested suffix.
    engine._ingest_cursor = len(view)
    engine.note_turn_complete()
    return engine.compress(view, current_tokens=count_messages_tokens(view), force=force)


def _opening(engine, prefix):
    raw = [_msg("opening", "user"), *prefix, _msg("seed-capstone")]
    engine.ingest(raw)
    engine._config.fresh_tail_count = 0
    result = _pass(engine, raw)
    engine._config.fresh_tail_count = 2
    assert engine._last_compress_leaves[1] == 1, engine._last_compression_noop_reason
    return raw, result[0]


def _copy(group, offset):
    return [_no_uid(group[0]), _no_uid(dict(group[offset], content=PRUNED))]


def _far_store(engine, *, count=2, groups=1, defeat=True, version_name="lookup"):
    units = [_group(f"call-{i}", count) for i in range(groups)]
    offset = 2 if count == 3 else 1
    earlier = [row for i, group in enumerate(units) for row in
               _group(f"earlier-{i}", ids=[group[offset]["tool_call_id"]])]
    engine._config.leaf_chunk_tokens = 1
    prefix, summary = _opening(engine, earlier if defeat else [_msg("earlier")])
    if defeat:
        bindings = _bindings(engine, earlier)
        assert {sid for row in earlier for sid, _ in bindings[row["message_uid"]]} <= set(_covered(engine))

    results = [row for group in units for row in group]
    retained = _msg("retained", "user")
    copies = [row for group in units for row in _copy(group, offset)]
    for copied in copies[1::2]:
        copied["tool_name"] = version_name
    after = _msg("after")
    tail = _msg("tail")
    engine.ingest([*prefix, *results, retained, *copies, after, tail])
    stored = _rows(engine)
    assert len(stored) == len(prefix) + len(results) + len(copies) + 3
    first_caller_copy = len(prefix) + len(results) + 1
    proofs = {}
    canonical, resolution = [], []
    for i, group in enumerate(units):
        caller_copy, version = [row["store_id"] for row in stored[first_caller_copy + 2*i:][:2]]
        assert stored[first_caller_copy + 2*i + 1]["content"] == PRUNED
        members = [_canonical(engine, row) for row in group]
        canonical.append(members)
        assert members[-1] < _canonical(engine, retained) < caller_copy < version
        uid = group[offset]["message_uid"]
        engine._store.add_host_uid_bindings(_lineage(engine), [(version, uid, "version", "version_new")])
        assert _bindings(engine, [group[offset]])[uid] == [(members[offset], "canonical"), (version, "version")]
        resolution.append((_pruned(group)[offset], members[offset], version))
        proofs.update({caller_copy: members[0], version: members[offset]})
    view = [summary, *[row for group in units for row in _pruned(group)], after, retained, tail]
    return view, units, canonical, resolution, proofs, after


def _observe_resolution(engine, monkeypatch, cases):
    seen = []
    original = engine._store_complete_input

    def walk(*args, **kwargs):
        assert uid_recognition_enabled(engine)
        observed = [engine._host_uid_resolve([row]).get(0, {}).get("store_id") for row, _, _ in cases]
        seen.append(observed)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_store_complete_input", walk)
    return seen


def _present_node_sources(engine, output):
    from hermes_lcm.engine import _summary_part_text

    return {sid for node in engine._dag.get_session_nodes("S") if node.source_type == "messages"
            and any(_summary_part_text(node) in (message.get("content") or "") for message in output)
            for sid in node.source_ids}


def _assert_no_orphan_or_stub(output):
    active = set()
    for row in output:
        if row.get("tool_calls"):
            active.update(call["id"] for call in row["tool_calls"])
        if row["role"] == "tool":
            assert row["tool_call_id"] in active
            active.remove(row["tool_call_id"])
    assert not active


def _assert_resolution(seen, cases, *, head=False):
    assert seen
    wanted = [version if head else canonical for _, canonical, version in cases]
    assert seen[0] == wanted


@pytest.mark.parametrize(("count", "defeat"), [(2, True), (2, False), (3, True)],
                         ids=["split", "alias-works", "three-middle"])
def test_complete_group(tmp_path, monkeypatch, summaries, count, defeat):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        view, units, canonical, resolve, _, _ = _far_store(engine, count=count, defeat=defeat)
        seen = _observe_resolution(engine, monkeypatch, resolve)
        summaries.clear()
        output = _pass(engine, view)
        if os.environ.get("C16_HEAD") == "1" and defeat:
            _assert_resolution(seen, resolve, head=True)
        assert engine._last_compress_leaves[1] == 1, engine._last_compression_noop_reason
        assert set(canonical[0]) <= _present_node_sources(engine, output)
        assert all(row["message_uid"] not in {m.get("message_uid") for m in output} for row in units[0])
        text, = summaries
        assert PRUNED not in text
        assert all(row["content"] in text for row in units[0][1:])
        _assert_no_orphan_or_stub(output)
        _assert_resolution(seen, resolve)


def test_stalled_conversation(tmp_path, monkeypatch, summaries, caplog):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        view, units, canonical, resolve, copies, _ = _far_store(engine, groups=40)
        assert len(resolve) == 40
        engine._config.leaf_chunk_tokens = 110
        assert engine._context_aware_leaf_cap() is None
        seen = _observe_resolution(engine, monkeypatch, resolve)
        with caplog.at_level(logging.INFO, logger=LOG):
            output = _pass(engine, view, force=False)
        if os.environ.get("C16_HEAD") == "1":
            _assert_resolution(seen, resolve, head=True)
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert _frontier(engine) >= canonical[0][-1]
        assert not engine._sweep_budget_hold_until and not engine._no_progress_hold_status()
        sources = _present_node_sources(engine, output)
        remaining = {row.get("message_uid") for row in output}
        assert all(sid in sources for group, ids in zip(units, canonical) for row, sid in zip(group, ids)
                   if row["message_uid"] not in remaining)
        assert not (set(copies) & set(_covered(engine)))
        assert not engine._last_compression_noop_reason
        assert "contiguity hole" not in caplog.text
        _assert_no_orphan_or_stub(output)
        _assert_resolution(seen, resolve)


def _cross_copy(engine, view, after, *, extra=()):
    # Once the retained row is no longer the last turn, the walk can cross it
    # and prove the copy exclusions below the next summarisable store row.
    retained, tail = [next(row for row in view if row.get("message_uid") == uid) for uid in ("retained", "tail")]
    later = [_msg("later", "user"), _msg("latest")]
    engine._ingest_cursor = len(view)
    engine.ingest([*view, *extra, *later])
    active = [*(row for row in view if row.get("message_uid") not in {"retained", "after", "tail"}),
              retained, after, tail, *extra, *later]
    return _pass(engine, active)


def test_far_copy_publication(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        view, _, _, resolve, proofs, after = _far_store(engine)
        seen = _observe_resolution(engine, monkeypatch, resolve)
        first = _pass(engine, view)
        if os.environ.get("C16_HEAD") == "1":
            _assert_resolution(seen, resolve, head=True)
        output = _cross_copy(engine, first, after)
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert engine._store_complete_copy_exclusion_proofs == proofs
        assert max(proofs) < _frontier(engine)
        assert not set(proofs).intersection(_covered(engine))
        for _ in range(2):
            output = _pass(engine, output)
            assert max(proofs) < _frontier(engine)
            assert not engine._store_complete_blocked or engine._store_complete_blocked[0] not in proofs
        _assert_no_orphan_or_stub(output)


def _snapshot(engine, output, summaries):
    rows = [{k: v for k, v in row.items() if k not in {"timestamp", "ingested_at", "observed_at"}}
            for row in _rows(engine)]
    bindings = engine._store.connection.execute(
        "SELECT store_id, uid, kind, proof_kind, binding_version FROM host_uid_bindings ORDER BY rowid").fetchall()
    # Generated UID bytes depend on the absolute state.db path. Keep each
    # occurrence and binding, using an ordinal in place of the path hash.
    engine_uids = {uid: f"generated-{i}" for i, (sid, uid, kind, *_rest) in enumerate(bindings) if kind == "engine"}

    def replace(value):
        if isinstance(value, str):
            return engine_uids.get(value, value)
        if isinstance(value, list | tuple):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    return dict(output=replace(output), rows=rows, bindings=replace(bindings), frontier=_frontier(engine),
                status=engine._last_compression_status, reason=engine._last_compression_noop_reason,
                summaries=summaries)


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _placement_digest(shape, snapshots):
    digest = _digest(snapshots)
    if directory := os.environ.get("C16_PLACEMENT_ACTUAL"):
        (Path(directory) / f"{shape}.txt").write_text(digest + "\n")
    if directory := os.environ.get("C16_PLACEMENT_EXPECT"):
        assert digest == (Path(directory) / f"{shape}.txt").read_text().strip()


def test_unset_shadow_digests(tmp_path, monkeypatch, summaries):
    digests = {}
    for mode in ("unset", "shadow", "on"):
        _mode(monkeypatch, mode)
        for cell in (1, 4, 5):
            name = f"{mode}-{cell}"
            summaries.clear()
            with _open(tmp_path / name) as engine, monkeypatch.context() as control:
                if mode == "on":
                    control.setattr(engine, "_uid_recognition_allowed", lambda: False)
                view, _, _, _, _, after = _far_store(engine, groups=40 if cell == 4 else 1)
                if cell == 4:
                    engine._config.leaf_chunk_tokens = 110
                output = _pass(engine, view, force=cell != 4)
                if cell == 5:
                    output = _cross_copy(engine, output, after)
                    for _ in range(2):
                        output = _pass(engine, output)
                digests[name] = _digest(_snapshot(engine, output, summaries))
    assert {cell: digests[f"unset-{cell}"] for cell in (1, 4, 5)} == {
        cell: digests[f"shadow-{cell}"] for cell in (1, 4, 5)}
    assert {cell: digests[f"on-{cell}"] for cell in (1, 4, 5)} == {
        cell: digests[f"shadow-{cell}"] for cell in (1, 4, 5)}


def test_complete_group_followed_by_version(tmp_path, monkeypatch, summaries, caplog):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        view, _, canonical, cases, copies, after = _far_store(engine, defeat=False)
        seen = _observe_resolution(engine, monkeypatch, cases)
        # Make the first pass reach the copy and a later source.
        with caplog.at_level(logging.INFO, logger=LOG):
            output = _cross_copy(engine, view, after, extra=[_msg("extra", content=PAD * 4)])
        _assert_resolution(seen, cases)
        assert set(canonical[0]) <= _present_node_sources(engine, output)
        assert engine._store_complete_copy_exclusion_proofs == copies
        assert "first tool group remains open" not in caplog.text
        _assert_no_orphan_or_stub(output)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="first-group exception gap, pre-existing, issue 2934")
def test_first_open_group_keeps_its_real_result(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, summary = _opening(engine, [_msg("initial")])
        caller, result = _group("open-call", 1)
        retained, later, tail = [_msg(name) for name in ("retained", "later", "tail")]
        engine.ingest([*prefix, caller, retained, result, later, tail])
        live = dict(result, content=PRUNED)
        result_id = _canonical(engine, result)
        seen = _observe_resolution(engine, monkeypatch, [(live, result_id, None)])
        output = _pass(engine, [summary, caller, live, later, retained, tail])
        _assert_resolution(seen, [(live, result_id, None)])
        assert result_id in _present_node_sources(engine, output) or any(
            row.get("role") == "tool" and row.get("tool_call_id") == result["tool_call_id"]
            and row.get("content") == result["content"] for row in output)


@pytest.mark.parametrize("exit_at", ["retained", "hole", "scan"])
@pytest.mark.parametrize("first", [False, True], ids=["after-source", "first-source"])
def test_group_closure(tmp_path, monkeypatch, summaries, caplog, exit_at, first):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, summary = _opening(engine, [_msg("initial")])
        group = _group("open-call", 2 if exit_at == "hole" else 1)
        caller, result = group[0], group[-1]
        live_group = _pruned(group) if first else group
        prior = [] if first else [_msg("prior")]
        retained, hidden, later, tail = [_msg(name) for name in ("retained", "hidden", "later", "tail")]
        if exit_at == "retained":
            stored = [*prior, caller, retained, result, later, tail]
            view = [summary, *prior, *live_group, later, retained, tail]
        elif exit_at == "hole":
            stored = [*prior, caller, result, hidden, retained, group[1], tail]
            view = [summary, *prior, *live_group, retained, tail]
        else:
            stored = [*prior, caller, hidden, result, later, retained, tail]
            view = [summary, *prior, *live_group, later, retained, tail]
        engine.ingest([*prefix, *stored])
        caller_id, result_id = [_canonical(engine, row) for row in (caller, result)]
        assert engine._host_uid_resolve([caller, result])[1]["store_id"] == result_id
        engine._config.fresh_tail_count = 2
        if exit_at == "scan":
            monkeypatch.setattr(store_complete, "_SCAN_LIMIT", len(prior) + 2)
        cases = [(live_group[-1], result_id, None)]
        seen = _observe_resolution(engine, monkeypatch, cases)
        with caplog.at_level(logging.INFO, logger=LOG):
            output = _pass(engine, view)
        _assert_resolution(seen, cases)
        sources = _present_node_sources(engine, output)
        if first:
            assert caller_id in sources
            lines = [r.getMessage() for r in caplog.records if "first tool group remains open" in r.getMessage()]
            assert len(lines) == 1 and str(caller_id) in lines[0]
            _assert_no_orphan_or_stub(output)
            before = _frontier(engine)
            monkeypatch.setattr(store_complete, "_SCAN_LIMIT", 2000)
            _pass(engine, output)
            assert _frontier(engine) > before
            assert engine._store.get(result_id)["content"] == result["content"]
        else:
            assert result_id not in sources
            assert _canonical(engine, prior[0]) in sources
            assert caller_id not in sources
            assert all(row in output for row in group)
            _assert_no_orphan_or_stub(output)


def test_covered_canonical_drops_group(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        view, units, canonical, resolve, proofs, after = _far_store(engine)
        full_view = [view[0], *units[0], *view[-3:]]
        first = _pass(engine, full_view)
        assert set(canonical[0]) <= _present_node_sources(engine, first)
        covered = dict(resolve[0][0])
        group = [units[0][0], covered, *(_pruned(units[0])[2:])]
        replay = [first[0], *group, *first[1:]]
        guard = engine._guard_uid_covered_drop
        verdicts = []

        def observe(rows):
            assert uid_recognition_enabled(engine)
            output = guard(rows)
            verdicts.append(rows is output)
            return output

        monkeypatch.setattr(engine, "_guard_uid_covered_drop", observe)
        output = _pass(engine, replay)
        assert verdicts == [True]
        assert not {row["message_uid"] for row in group}.intersection(row.get("message_uid") for row in output)
        output = _cross_copy(engine, output, after)
        assert engine._store_complete_copy_exclusion_proofs == proofs
        _assert_no_orphan_or_stub(output)


@pytest.mark.parametrize("shape", ["non-string", "wrong-name", "placeholder-canonical", "recognition-disabled"])
def test_placement_controls(tmp_path, monkeypatch, summaries, shape):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        if shape == "wrong-name":
            view, _, _, cases, _, _ = _far_store(engine, version_name="different")
            row, canonical, version = cases[0]
            wrong = dict(row, tool_name="different")
            view[view.index(row)] = wrong
            cases = [(wrong, version, version)]
            assert engine._store.get(version)["tool_name"] == "different"
            seen = _observe_resolution(engine, monkeypatch, cases)
            summaries.clear()
            output = _pass(engine, view)
            assert engine._store.get(canonical)["tool_name"] == "lookup"
            _assert_resolution(seen, cases)
            _placement_digest(shape, [_snapshot(engine, output, summaries)])
            return

        tail = [_msg("latest-user", "user"), _msg("latest-reply")]
        if shape == "recognition-disabled":
            group = _group(shape, 1)
            engine.ingest([*group, *tail])
            sid = _canonical(engine, group[1])
            engine._config.ignore_message_patterns = ["never-matches-a-c16-row"]
            walk = engine._store_complete_input
            observed = []

            def observe(chunk, claims, full_map, *args, **kwargs):
                assert not uid_recognition_enabled(engine)
                observed.extend(full_map.get(id(m)) for m in chunk if m.get("content") == PRUNED)
                return walk(chunk, claims, full_map, *args, **kwargs)

            monkeypatch.setattr(engine, "_store_complete_input", observe)
            summaries.clear()
            output = _pass(engine, [*_pruned(group), *tail])
            assert observed == [sid], (observed, sid, summaries)
            assert not getattr(engine, "_store_complete_substitutes", {})
            assert summaries and PRUNED in summaries[0] and group[1]["content"] not in summaries[0]
            _placement_digest(shape, [_snapshot(engine, output, summaries)])
            return

        prefix, summary = _opening(engine, [_msg("initial")])
        group = _group(shape, 1, payload={"answer": 42} if shape == "non-string" else PRUNED)
        raw = group.copy()
        if shape == "placeholder-canonical":
            full = _group("earlier-full", ids=[group[1]["tool_call_id"]])
            raw = [*map(_no_uid, full), *group]
        engine.ingest([*prefix, *raw, *tail])
        sid = _canonical(engine, group[1])
        cases = [(_pruned(group)[1], sid, None)]
        seen = _observe_resolution(engine, monkeypatch, cases)
        summaries.clear()
        output = _pass(engine, [summary, *_pruned(group), *tail])
        _assert_resolution(seen, cases)
        if shape == "non-string":
            assert sid in _present_node_sources(engine, output)
            assert any('{"answer": 42}' in text and PRUNED not in text for text in summaries)
        else:
            first = _snapshot(engine, output, list(summaries))
            assert sid not in _present_node_sources(engine, output)
            output = _pass(engine, output)
            assert engine._store.get(sid)["content"] == PRUNED
            assert any(PRUNED in text for text in summaries)
            assert sid in _present_node_sources(engine, output)
            _placement_digest(shape, [first, _snapshot(engine, output, summaries)])
        _assert_no_orphan_or_stub(output)


def test_new_uid_with_reused_call_id(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        old = _group("old-call", 1)
        prefix, _ = _opening(engine, old)
        new = _group("new-call", ids=[old[1]["tool_call_id"]], payload=PRUNED)
        real_resolve = engine._host_uid_resolve
        checked = []

        def observe(rows, *args, **kwargs):
            result = real_resolve(rows, *args, **kwargs)
            for i, row in enumerate(rows):
                if row.get("message_uid") == new[1]["message_uid"]:
                    assert uid_recognition_enabled(engine)
                    checked.append(i in result)
            return result

        monkeypatch.setattr(engine, "_host_uid_resolve", observe)
        before = len(_rows(engine))
        engine.ingest([*prefix, *new])
        assert checked and not any(checked)
        assert _bindings(engine, new).get(new[1]["message_uid"]) == [(_rows(engine)[-1]["store_id"], "canonical")]
        sid = _canonical(engine, new[1])
        assert len(_rows(engine)) == before + 2 and sid > _canonical(engine, old[1])
        assert engine._store.get(sid)["content"] == PRUNED


def test_oversized_first_group(tmp_path, monkeypatch, summaries):
    _mode(monkeypatch, "on")
    with _open(tmp_path / "store") as engine:
        prefix, summary = _opening(engine, [_msg("initial")])
        engine._config.large_output_externalization_enabled = False
        engine._config.leaf_chunk_tokens = 110
        huge = _group("huge", 1, payload="long output " * 10_000)
        tail = [_msg("later-user", "user"), _msg("later-reply")]
        more = _msg("more")
        engine.ingest([*prefix, *huge, more, *tail])
        sid = _canonical(engine, huge[1])
        cases = [(_pruned(huge)[1], sid, None)]
        seen = _observe_resolution(engine, monkeypatch, cases)
        summaries.clear()
        output = _pass(engine, [summary, *_pruned(huge), more, *tail])
        assert len(engine._store.get(sid)["content"]) == 120_000
        _assert_resolution(seen, cases)
        assert sid in _present_node_sources(engine, output)
        text, = summaries
        assert "long output" in text and PRUNED not in text
        assert len(text) < 3_200 and CLIP_MARKER in text
        assert more["content"] not in text
        _assert_no_orphan_or_stub(output)


@pytest.mark.parametrize("copies", [1, 3])
def test_cold_uid_prefix(tmp_path, monkeypatch, summaries, copies):
    _mode(monkeypatch, "on")
    path = tmp_path / "store"
    neighbor = _group("neighbor", 1, payload="N" * 321)
    group = _group("target", 1, payload="F" * 321)
    with _open(path) as engine:
        filler = [_msg(f"history-{i}") for i in range(67)]
        prefix, _ = _opening(engine, [*filler, *neighbor, *group])
        caller_id, result_id = [_canonical(engine, row) for row in group]
        neighbor_ids = [_canonical(engine, row) for row in neighbor]
        assert {caller_id, result_id} <= set(_covered(engine))
        copy_rows = _copy(group, 1) * copies
        engine.ingest([*prefix, *copy_rows])
        seeded = _rows(engine)
        assert [row["content"] for row in seeded[-(2 * copies):]] == ["", PRUNED] * copies
        assert engine._store.host_uid_uids_of_stores(_lineage(engine), [row["store_id"] for row in seeded[-(2 * copies):]]) == {}
    view = [*prefix[:68], *_pruned(neighbor), *_pruned(group)]
    assert view[70:] == _pruned(group)
    for attempt in range(2):
        with _open(path, restart=True) as engine:
            assert [_canonical(engine, row) for row in group] == [caller_id, result_id]
            assert [row["store_id"] for row in engine._host_uid_resolve(
                _pruned(neighbor), prefix_only=True).values()] == neighbor_ids
            real = engine._reconcile_ingest_cursor_from_store
            records = []

            def observe(*args, **kwargs):
                assert uid_recognition_enabled(engine)
                cursor = real(*args, **kwargs)
                records.append((cursor, deepcopy(engine._last_ingest_reconciliation)))
                return cursor

            with monkeypatch.context() as observation:
                observation.setattr(engine, "_reconcile_ingest_cursor_from_store", observe)
                engine.ingest(view)
            assert any(cursor == len(view) and record.get("reason") == "replayed UID-bound prefix"
                       for cursor, record in records), (attempt, records, len(_rows(engine)) - len(seeded))
            assert _rows(engine) == seeded
