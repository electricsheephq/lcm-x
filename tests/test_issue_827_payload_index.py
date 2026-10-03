"""#827: set buckets and one digest listing across a scoped trial/final pair."""

import json
import os
import random
from pathlib import Path

import pytest

from hermes_lcm import externalize
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def config(tmp_path):
    storage = tmp_path / "payloads"
    storage.mkdir()
    return LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(storage),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_active_replay_stubbing_enabled=True,
        large_output_active_replay_stub_threshold_tokens=10_000,
        large_output_active_replay_stub_aged_threshold_tokens=5,
        fresh_tail_count=2,
    )


def write_payload(config, payload_content, name="a", **fields):
    prefix = externalize._content_digest_prefix(payload_content)
    path = Path(config.large_output_externalization_path) / f"{name}_{prefix}_x.json"
    payload = dict(kind="tool_result", role="tool", tool_call_id="call", session_id="s", content=payload_content)
    payload.update(fields)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def lookup(config, content, **fields):
    return externalize.find_externalized_payload_for_message(content, config=config, **fields)


def count_listings(monkeypatch, config):
    storage = Path(config.large_output_externalization_path)
    counts = {"scandir": 0, "glob": 0}
    original_scandir, original_glob = os.scandir, Path.glob
    in_glob = []

    def scandir(path):
        # Path.glob's own scandir calls vary with the Python version.
        if Path(path) == storage and not in_glob:
            counts["scandir"] += 1
        return original_scandir(path)

    def glob(path, pattern, *args, **kwargs):
        if path == storage:
            counts["glob"] += 1
        in_glob.append(True)
        try:
            return iter(list(original_glob(path, pattern, *args, **kwargs)))
        finally:
            in_glob.pop()

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(Path, "glob", glob)
    return counts


def test_membership_is_not_linear():
    class CountedName(str):
        comparisons = 0
        __hash__ = str.__hash__

        def __eq__(self, other):
            type(self).comparisons += 1
            return super().__eq__(other)

    n = 400
    digest = "0123456789ab"
    names = [CountedName(f"{i:04d}_{digest}_x.json") for i in range(n)]
    index = {}
    for name in names:
        externalize._index_payload_name(index, name)
    assert CountedName.comparisons < 4 * n
    for name in names:
        externalize._index_payload_name(index, CountedName(name))
    assert len(index[digest]) == n
    assert CountedName.comparisons < 4 * n


def test_order_and_lookup_equivalence(config):
    content = "shared payload"
    order = list(range(40))
    random.Random(827).shuffle(order)
    for i in order:
        write_payload(config, content, f"{i:03d}", tool_call_id=f"call-{i % 3}", session_id=f"s-{i % 2}")
    write_payload(config, content, "a-kind", kind="other")
    write_payload(config, content, "b-role", role="assistant")
    write_payload(config, content, "c-content", content="wrong content")
    digest = externalize._content_digest_prefix(content)
    write_payload(config, content, f"d_{digest}")
    cases = [
        dict(tool_call_id="call-0", session_id="s-0", role="tool"),
        dict(tool_call_id="call-1", role="tool"),
        dict(tool_call_id="call-2", session_id="absent"),
        dict(tool_call_id="absent"),
        dict(tool_call_id="call", kind="other"),
        dict(tool_call_id="call", role="assistant"),
        dict(tool_call_id="call", kind=None),
    ]
    baseline = [lookup(config, content, **fields) for fields in cases]
    storage = Path(config.large_output_externalization_path)
    expected = sorted(storage.glob(f"*_{digest}_*.json"))
    with externalize.payload_lookup_scope():
        assert next(externalize._payload_lookup_candidates(storage, digest)) == expected
        assert [lookup(config, content, **fields) for fields in cases] == baseline


@pytest.mark.parametrize("rejected_candidate", [False, True])
def test_one_glob_per_digest_per_scope(config, monkeypatch, rejected_candidate):
    content = "missing or rejected payload"
    if rejected_candidate:
        write_payload(config, content)
    counts = count_listings(monkeypatch, config)
    with externalize.payload_lookup_scope():
        assert lookup(config, content, tool_call_id="reject-1") is None
        assert lookup(config, content, tool_call_id="reject-2") is None
        assert counts == {"scandir": 1, "glob": 1}
        if rejected_candidate:
            # Listing reuse must not cache the rejecting message's verdict.
            assert lookup(config, content, tool_call_id="call", session_id="s") is not None
            assert counts == {"scandir": 1, "glob": 1}


@pytest.mark.parametrize("ingest", [False, True])
def test_process_write_after_glob_is_visible(config, monkeypatch, ingest):
    content = "new payload after a miss"
    fields = (dict(kind="ingest_payload", role="user", session_id="s") if ingest
              else dict(kind="raw_payload", tool_call_id="new-call", session_id="s"))
    counts = count_listings(monkeypatch, config)
    with externalize.payload_lookup_scope():
        assert lookup(config, content, **fields) is None
        assert counts == {"scandir": 1, "glob": 1}
        if ingest:
            written = externalize.externalize_ingest_payload(content, role="user", session_id="s", config=config)
        else:
            written = externalize.maybe_externalize_payload(content, tool_call_id="new-call", session_id="s", config=config)
        assert written is not None
        assert lookup(config, content, **fields)["ref"] == written["path"].name
        assert counts == {"scandir": 1, "glob": 1}


def test_external_writer_after_glob_waits_for_next_scope(config, monkeypatch):
    content = "another process payload"
    counts = count_listings(monkeypatch, config)
    with externalize.payload_lookup_scope():
        assert lookup(config, content, tool_call_id="call", session_id="s") is None
        path = write_payload(config, content)
        assert lookup(config, content, tool_call_id="call", session_id="s") is None
        assert counts == {"scandir": 1, "glob": 1}
    with externalize.payload_lookup_scope():
        assert lookup(config, content, tool_call_id="call", session_id="s")["ref"] == path.name
        assert counts == {"scandir": 2, "glob": 1}


def test_stub_first_trial_and_final_share_listing(config, tmp_path, monkeypatch):
    # Minimal #671 setup; call the exit directly and force only its host measure.
    config.leaf_chunk_tokens = 1_000
    config.fresh_tail_pressure_yield_enabled = False
    config.large_output_externalization_threshold_chars = 1_000_000
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start("s", conversation_id="issue-827", context_length=200_000)
    engine.threshold_tokens = 12_000
    engine._compress_occurrences = None  # Normally initialized by compress().
    monkeypatch.setattr(engine, "_survival_measure", lambda messages: 0)
    rows = [{"role": "system", "content": "system prompt"}]
    m = 3
    for i in range(m):
        rows.extend([
            {"role": "user", "content": f"turn {i}"},
            {"role": "assistant", "content": "running tool", "tool_calls": [
                {"id": f"call-{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": f"call-{i}", "content": f"payload {i} " + "alpha beta gamma delta " * 20},
        ])
    rows.extend([{"role": "user", "content": "latest ask"}, {"role": "assistant", "content": "latest answer"}])
    storage = Path(config.large_output_externalization_path)
    assert not list(storage.iterdir())
    counts = count_listings(monkeypatch, config)
    try:
        result = engine._stub_first_exit(rows, rows, rows, rows, count_messages_tokens(rows))
        assert result is not None and engine._last_stub_first_exit is not None
        stubs = [row for row in result if row.get("role") == "tool"]
        assert len(stubs) == m
        assert all(row["content"].startswith("[Externalized tool output:") for row in stubs)
        assert result[-2:] == rows[-2:]
        assert counts == {"scandir": 1, "glob": m}
    finally:
        engine.shutdown()
