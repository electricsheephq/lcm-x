"""#2890: compaction keeps stored content reachable through the returned view."""

import base64
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from textwrap import wrap

import pytest

from hermes_lcm import engine as engine_module, escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import extract_externalized_ref, externalize_ingest_payload, load_externalized_payload
from hermes_lcm.ingest_content import _EXTERNALIZED_PLACEHOLDER_PREFIX, _INGEST_PLACEHOLDER_RE
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX

SESSION = "lineage"
PAD = " the quick brown fox jumps over"
SUMMARY = "Summary of earlier work.\nExpand for details about: earlier"
BODY = base64.b64encode(bytes((i * 131 + 7) % 256 for i in range(3600))).decode("ascii")


def _summary(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
    deadline = time.monotonic() + timeout if timeout is not None else float("inf")
    if time.monotonic() >= deadline:
        raise TimeoutError("synthetic summary deadline expired")
    return SUMMARY


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(escalation, "_call_llm_for_summary", _summary)
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    engines = []

    def create(*, rollups=False, **config):
        root = tmp_path / str(len(engines))
        root.mkdir()
        engine = LCMEngine(
            config=LCMConfig(database_path=str(root / "lcm.db"), embeddings_enabled=False,
                             temporal_rollups_enabled=rollups, **config),
            hermes_home=str(root),
        )
        engines.append(engine)
        engine.on_session_start(SESSION, platform="cli", context_length=200_000, conversation_id="conv")
        return engine

    yield create
    for engine in engines:
        engine.shutdown()


class History:
    def __init__(self, engine, system=False):
        self.engine = engine
        self.view = [{"role": "system", "content": "You are a helpful agent."}] if system else []
        self.timestamp = 1_800_000_000.0

    def add(self, **fields):
        self.timestamp += 2
        row = {**fields, "timestamp": self.timestamp}
        self.view.append(row)
        return row

    def turn(self, number, *, words=1, body=None, result_count=1):
        self.add(role="user", content=f"question {number}" + PAD * 3)
        if body is not None:
            calls = [
                {"id": f"call_{number}_{i}", "type": "function",
                 "function": {"name": "write_file", "arguments": json.dumps(
                     {"path": f"/srv/cert{i}.pem", "content": body})}}
                for i in range(result_count)
            ]
            self.add(role="assistant", content="", tool_calls=calls)
            for call in calls:
                self.add(role="tool", tool_call_id=call["id"],
                         content=json.dumps({"bytes_written": len(body), "ok": True}),
                         name="write_file", tool_name="write_file")
        self.add(role="assistant", content=f"reply {number}" + PAD * words)
        self.engine.ingest(deepcopy(self.view))

    def compress(self, *, force=True):
        before = json.loads(json.dumps(self.view))
        tokens = len(json.dumps(before)) // 4 if force else 190_000
        self.view = self.engine.compress(before, current_tokens=tokens, focus_topic=None, force=force)
        return self.view

    def boundary(self):
        self.engine.on_session_start(SESSION, old_session_id=SESSION, boundary_reason="compression",
                                     platform="cli", conversation_id="conv")


def _verifier_history(engine, *, columns=76, pem=False, system=False, turns=17, words=230):
    history = History(engine, system)
    body = "\n".join(wrap(BODY, columns)) if columns else BODY
    if pem:
        body = f"-----BEGIN CERTIFICATE-----\n{body}\n-----END CERTIFICATE-----\n"
    for i in range(1, turns + 1):
        history.turn(i, words=words)
    history.turn(turns + 1)
    history.turn(turns + 2, body=body)
    for i in range(turns + 3, turns + 16):
        history.turn(i)
    return history


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _content_digest(engine, row):
    # Recover actual bytes from our payload files independently of the identity code under test.
    def restore(match):
        payload = load_externalized_payload(match.group(1), config=engine._config, hermes_home=engine._hermes_home)
        return payload["content"]

    calls = []
    for call in row.get("tool_calls") or ():
        function = call["function"]
        args = function["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        args = {key: _INGEST_PLACEHOLDER_RE.sub(restore, value) if isinstance(value, str) else value
                for key, value in args.items()}
        calls.append((call["id"], function["name"], sorted(args.items())))
    content = _INGEST_PLACEHOLDER_RE.sub(restore, row.get("content") or "")
    if ref := extract_externalized_ref(content):
        payload = load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home)
        if payload and isinstance(payload.get("content"), str):
            content = payload["content"]
    return _digest((row["role"], content, row.get("tool_call_id") or "", calls))


def _stored(engine):
    return engine._store.get_session_messages(SESSION)


def _nodes(engine):
    return engine._dag.get_session_nodes(SESSION)


def _rendered_sources(engine, node_ids):
    reachable = set()
    seen = set()
    pending = list(node_ids)
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = engine._dag.get_node(node_id)
        if node.source_type == "messages":
            reachable.update(node.source_ids)
        else:
            pending.extend(node.source_ids)
    return reachable


def _summary_spans(engine, content):
    """Verify leading summaries and summaries appended to a preserved objective."""
    separator = "\n\n---\n\n"
    starts = [0]
    if content.startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX + "\n"):
        starts.extend(match.end() for match in re.finditer(separator, content))
    previous_end = -1
    for start in starts:
        if start < previous_end:
            continue
        end, node_ids = engine._verified_lcm_summary_prefix(content[start:])
        if end is not None:
            previous_end = start + end
            yield start, previous_end, node_ids


def _covered_digests(engine, stored, returned):
    present = {_content_digest(engine, row) for row in returned}
    rendered_nodes = set()
    for index, row in enumerate(returned):
        content = row.get("content") or ""
        if index == 0 and row["role"] == "system":
            original = content.split("\n\n[Note: This conversation uses Lossless Context Management (LCM). ", 1)[0]
            present.add(_content_digest(engine, {**row, "content": original}))
        spans = list(_summary_spans(engine, content))
        for _, _, node_ids in spans:
            rendered_nodes.update(node_ids)
        if spans:
            suffix = content[spans[-1][1]:]
            for separator in ("\n\n---\n\n", "\n\n"):
                if suffix.startswith(separator):
                    suffix = suffix[len(separator):]
                    if row["role"] == "user" and suffix:
                        present.add(_content_digest(engine, {**row, "content": suffix}))
                    content = suffix if spans[0][0] == 0 else content
                    break
        if content.startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX + "\n"):
            objective_end = spans[0][0] - len("\n\n---\n\n") if spans and spans[0][0] else len(content)
            original = content[len(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX) + 1:objective_end]
            present.add(_content_digest(engine, {"role": "user", "content": original}))
    sources = _rendered_sources(engine, rendered_nodes)
    present.update(_content_digest(engine, row) for row in stored if row["store_id"] in sources)
    return present


def _uncovered(engine, returned):
    stored = _stored(engine)
    present = _covered_digests(engine, stored, returned)
    return [(row["store_id"], row["role"], row.get("tool_call_id") or "",
             _content_digest(engine, row)) for row in stored if _content_digest(engine, row) not in present]


def _assert_pairs(before, returned):
    expected = Counter(row["tool_call_id"] for row in before if row["role"] == "tool")
    opened = set()
    actual = Counter()
    for row in returned:
        opened.update(call["id"] for call in row.get("tool_calls") or ())
        if row["role"] == "tool":
            assert row["tool_call_id"] in opened
            actual[row["tool_call_id"]] += 1
    assert opened <= expected.keys()
    assert actual == Counter({key: expected[key] for key in opened})


@pytest.mark.issue(2890)
@pytest.mark.parametrize(("config", "shape"), [
    pytest.param({}, {}, id="A-gateway"),
    pytest.param({}, {"system": True}, id="B-system"),
    pytest.param({}, {"columns": 64, "pem": True}, id="B-pem"),
    pytest.param({}, {"turns": 10, "words": 400}, id="B-10-by-400"),
    pytest.param({}, {"turns": 25, "words": 150}, id="B-25-by-150"),
    pytest.param({"rollups": True}, {}, id="B-rollups"),
])
def test_wrapped_content_remains_reachable(session, config, shape):
    history = _verifier_history(session(**config), **shape)
    before = history.view
    returned = history.compress()
    _assert_pairs(before, returned)
    assert _uncovered(history.engine, returned) == []


def test_one_line_control_matches_main(session):
    engine = session()
    history = _verifier_history(engine, columns=0)
    before = history.view
    returned = history.compress()
    _assert_pairs(before, returned)
    assert _uncovered(engine, returned) == []
    assert _digest((returned, [node.source_ids for node in _nodes(engine)])) == MAIN_CONTROL


def _old_restore(text, *, config, hermes_home="", session_id=""):
    if not isinstance(text, str) or _EXTERNALIZED_PLACEHOLDER_PREFIX not in text:
        return text

    def replace(match):
        payload = load_externalized_payload(match.group(1).strip(), config=config, hermes_home=hermes_home)
        if payload is None or payload.get("kind") != "ingest_payload":
            return match.group(0)
        if session_id and payload.get("session_id") and payload["session_id"] != session_id:
            return match.group(0)
        content = payload.get("content")
        return content if isinstance(content, str) else match.group(0)

    return _INGEST_PLACEHOLDER_RE.sub(replace, text)


def test_guard_keeps_unmapped_group_and_keeps_progress(session, monkeypatch):
    monkeypatch.setattr(engine_module, "restore_ingest_payload_placeholders", _old_restore)
    engine = session()
    history = _verifier_history(engine)
    original = history.view
    call = next(row for row in original if row.get("tool_calls"))
    tool = next(row for row in original if row["role"] == "tool")
    assert id(call) not in engine._get_store_id_map_for_messages(original)
    coverage, progress, first_group, pass_two_digests = [], [], [], set()
    for index in range(5):
        before = history.view
        node_count = len(_nodes(engine))
        returned = history.compress()
        _assert_pairs(before, returned)
        coverage.append(_uncovered(engine, returned))
        progress.append(len(_nodes(engine)) > node_count)
        if index == 0:
            first_group = [row for row in returned if row.get("tool_calls") or row["role"] == "tool"]
        if index == 1:
            pass_two_digests = _covered_digests(engine, _stored(engine), returned)
        history.boundary()
    print("D progress on five passes:", progress)
    assert first_group == [call, tool]
    assert {_content_digest(engine, call), _content_digest(engine, tool)} <= pass_two_digests
    assert coverage == [[] for _ in progress]
    # Filled from the five node-publication observations on the untouched archive.
    assert all(actual or not baseline for actual, baseline in zip(progress, MAIN_PROGRESS, strict=True))


# Observed on an untouched archive of b289240a with the same synthetic histories.
MAIN_CONTROL = "11b20fc4fd7bd4702a3acbaf52d7965d1b7388cac7e87328cab0a2e520c1f19a"
MAIN_PROGRESS = (True, False, False, False, False)
MAIN_SEEDED = "c68174d8f19796fafe9be0b437aa5e1d5611c571345879322f0a3b0dffbc23f0"


@pytest.mark.parametrize("value", ["first\nsecond", 'write "quoted"', r"C:\path\file", "a\tb",
                                   "a\r\u0000\u001fb", "café 日本語 😀"])
def test_parsed_field_restore_round_trips_json(session, value):
    engine = session()
    ref = externalize_ingest_payload(
        value, config=engine._config, hermes_home=engine._hermes_home, session_id=SESSION,
        role="assistant", field_path="tool_calls[0].function.arguments.content",
    )
    live = {"path": "/srv/cert.pem", "content": f"before {value} after"}
    stored_args = json.dumps({**live, "content": f"before {ref['placeholder']} after"})
    restored = engine._restore_ingest_payload_placeholders_in_value(stored_args, session_id=SESSION)
    assert json.loads(restored) == live


def test_wrapped_arguments_map_to_stored_row(session):
    engine = session()
    history = _verifier_history(engine)
    call = next(row for row in history.view if row.get("tool_calls"))
    stored = next(row for row in _stored(engine) if row.get("tool_calls"))
    assert "Externalized LCM ingest payload:" in json.dumps(stored["tool_calls"])
    assert engine._message_replay_identity(stored, stored_row=True) == engine._message_replay_identity(call)
    assert engine._get_store_id_map_for_messages(history.view)[id(call)] == stored["store_id"]


def test_forced_map_gap_does_not_consume_unnamed_row(session, monkeypatch):
    # Leave the tool result at the cut, between the historical prefix and tail.
    engine = session(fresh_tail_count=27)
    history = _verifier_history(engine, columns=0)
    before = history.view
    target = next(row for row in before if row["role"] == "tool")
    original_map = engine._get_store_id_map_for_messages
    target_id = original_map(before)[id(target)]
    consume = engine._leaf_consumption
    injected = []

    def map_with_gap(messages, *args, **kwargs):
        mapped = original_map(messages, *args, **kwargs)
        for row in messages:
            if row.get("tool_call_id") == target["tool_call_id"]:
                mapped.pop(id(row), None)
        return mapped

    def consume_with_map_gap(selected, compacted, sources, anchored, claims, dependent):
        # Inject the map failure at the cut, after the walk's separate claim recovery.
        cached = engine._compress_run.store_ids_by_message_id
        gap = next((id(row) for row in selected if cached.get(id(row)) == target_id), None)
        if gap is None:
            return consume(selected, compacted, sources, anchored, claims, dependent)
        injected.append(cached.pop(gap))
        try:
            with monkeypatch.context() as patch:
                patch.setattr(engine, "_get_store_id_map_for_messages", map_with_gap)
                return consume(selected, compacted, sources, anchored, claims, dependent)
        finally:
            cached[gap] = target_id

    monkeypatch.setattr(engine, "_leaf_consumption", consume_with_map_gap)
    returned = history.compress()
    _assert_pairs(before, returned)
    assert injected == [target_id]
    assert _nodes(engine), "the injected gap must exercise leaf publication"
    assert _uncovered(engine, returned) == []


def test_mapped_sessions_match_main(session):
    outcomes = []
    for seed in range(20):
        rng = random.Random(seed)
        engine = session(fresh_tail_count=8, leaf_chunk_tokens=2_000)
        history = History(engine, system=seed % 3 == 0)
        for number in range(1, 13):
            history.turn(number, words=rng.randint(10, 40),
                         body=f"synthetic contents for seed {seed}, turn {number}" if rng.choice((True, False)) else None,
                         result_count=rng.randint(1, 3))
        for index in range(5):
            if index:
                history.turn(12 + index, words=rng.randint(10, 40))
            before = history.view
            returned = history.compress(force=(seed + index) % 2 == 0)
            _assert_pairs(before, returned)
            assert _uncovered(engine, returned) == [], (seed, index)
            outcomes.append((returned, [asdict(node) for node in _nodes(engine)], _stored(engine)))
            history.boundary()
    assert _digest(outcomes) == MAIN_SEEDED


_SEEDED_MODE = os.environ.get("LCMX_2890_SWEEP")


# Seeds 119, 194, 114, 26, 61 and 40 control for the behavior main already
# supports. Seeds 98 and 39 expose losses on main under the repaired oracle.
@pytest.mark.issue(2890)
@pytest.mark.parametrize("seed", range(200 if _SEEDED_MODE == "clean" else 300) if _SEEDED_MODE else
                         [119, 194, 114, 26, 61, 98, 39, 40])
def test_seeded_lineage(seed, tmp_path, monkeypatch):
    from tests.lineage_guard_sessions import save_session, simulate

    edits = _SEEDED_MODE == "edits" if _SEEDED_MODE else seed in {114, 26, 61}
    drain = 16 if not _SEEDED_MODE and seed in {26, 61} else 4
    grow = 10 if not _SEEDED_MODE and seed == 61 else 2
    records = simulate(seed, tmp_path, monkeypatch, _summary, (_content_digest, _covered_digests, _nodes, _stored),
                       edits=edits, drain=drain, grow=grow)
    save_session(seed, records)
    if _SEEDED_MODE or os.environ.get("LCMX_2890_RECORD_ONLY"):
        return
    assert not [i for i, record in enumerate(records) if record["publication_error"]]
    if not edits:
        assert not [(i, record["missing"]) for i, record in enumerate(records) if record["missing"]]
    if seed in {114, 26, 61}:
        # The main replays publish at these passes. The 26 drain reaches a
        # different bar afterwards: its main leaf loses a content digest.
        expected_progress = {114: (4,), 26: (8,), 61: (7, 12, 13, 15)}[seed]
        assert not [i for i in expected_progress if not records[i]["leaves"] or records[i]["refused"]]
    if seed == 61:
        # Issue 2920: the tool-pair repair drops a real tool result when a user row sits between its call and the
        # result. An equal-state replay (round 5) shows main does the same from the same state, so it is pinned by
        # row id: only stored row 116 (call_35_0), only at passes 19 to 21. Any other late loss fails here.
        late = {i: r["missing"] for i, r in enumerate(records) if i >= 13 and r["missing"]}
        assert {i: list(m.values()) for i, m in late.items()} == {
            19: [[116]], 20: [[116]], 21: [[116]],
        }, f"#2920: {late}"


# Main handles these name-only shapes. They catch regressions in the proposed guard.
@pytest.mark.issue(2890)
@pytest.mark.parametrize(("tail", "budget"), [(2, 2000), (2, 400), (6, 800), (24, 24000)])
def test_p5_name_only_results(session, tail, budget):
    engine = session(fresh_tail_count=tail, leaf_chunk_tokens=budget)
    history = History(engine, system=True)

    def turn(n, results):
        history.add(role="user", content=f"question {n}" + PAD * 10)
        if results:
            calls = [
                {"id": f"c{n}_{i}", "type": "function",
                 "function": {"name": "read_file", "arguments": json.dumps({"path": f"/f{n}_{i}"})}}
                for i in range(results)
            ]
            history.add(role="assistant", content=f"calling {n}", tool_calls=calls)
            for call in calls:
                history.add(role="tool", tool_call_id=call["id"], content=f"result {call['id']}" + PAD * 20,
                            name="read_file")
        history.add(role="assistant", content=f"reply {n}" + PAD * 30)
        engine.ingest(deepcopy(history.view))

    for n in range(1, 13):
        turn(n, n % 3)
    for index in range(10):
        before = history.view
        returned = history.compress()
        assert engine._last_compression_status != "error", (index, engine._last_compression_noop_reason)
        assert _uncovered(engine, returned) == [], index
        _assert_pairs(before, returned)
        history.boundary()
        turn(100 + index, 1 + index % 2)
