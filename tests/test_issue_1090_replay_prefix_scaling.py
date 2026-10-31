"""#1090: exact replay proofs must not rebuild every prefix's identities."""
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import random

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    e = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        embeddings_enabled=False, temporal_rollups_enabled=False,
        empty_lifecycle_gc_enabled=False,
    ), hermes_home=str(tmp_path))
    e.on_session_start("synthetic-1090")
    yield e
    e.shutdown()


def rows(n):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"synthetic row {i}: العربية 🪷 " + "payload " * 10}
            for i in range(n)]


def find(e, messages, *, session_end=False):
    tail = [("user", "unrelated durable row", "", "", "")]
    return e._find_reconciled_cursor_for_store_tail(
        messages, tail, allow_empty_prefix=True, session_count=1,
        raw_session_count=1, allow_session_end_replay_proof=session_end,
    )


@pytest.mark.parametrize("proof", [False, True])
@pytest.mark.parametrize("n", [40, 80])
def test_store_tail_snapshot_identity_work_is_linear(engine, monkeypatch, n, proof):
    messages = rows(n)
    if proof:
        # A valid but nonmatching proof must still compare every eligible cut.
        engine._remember_native_recovery_replay_snapshot_digest(
            engine._native_recovery_replay_snapshot_digest(rows(3)))
        messages[0]["content"] += " host rejected snapshot"
    calls = Counter()
    original = engine._message_replay_identity

    def counted(message, **kw):
        calls["identity"] += 1
        return original(message, **kw)

    monkeypatch.setattr(engine, "_message_replay_identity", counted)
    assert find(engine, messages) == 0
    # Projection and other matcher mechanics also visit rows. The old native
    # proof scan alone visits n*(n+1)/2 even when its proof namespace is empty.
    assert calls["identity"] <= 6 * n


def test_empty_session_snapshot_identity_work_is_linear(engine, monkeypatch):
    messages = rows(80)
    engine._remember_native_recovery_replay_snapshot_digest(
        engine._native_recovery_replay_snapshot_digest(rows(3)))
    messages[0]["content"] += " host rejected snapshot"
    calls = Counter()
    original = engine._message_replay_identity

    def counted(message, **kw):
        calls["identity"] += 1
        return original(message, **kw)

    monkeypatch.setattr(engine, "_message_replay_identity", counted)
    assert engine._reconcile_ingest_cursor_from_store(messages) == 0
    assert calls["identity"] <= len(messages)


NOTE = {"role": "system", "content": (
    "[Note: This conversation uses Lossless Context Management (LCM). "
    "Earlier turns have been compacted into hierarchical summaries below.]"
)}
SUMMARY = {"role": "user", "content": (
    "[Recent Summary (d0, node 7)]\nالعربية 🪷\n[Expand for details: synthetic]"
)}
LOSSY = {"role": "tool", "tool_call_id": "lossy", "content": (
    "[LCM sensitive redaction: name=password_assignment; chars=8; bytes=8]"
)}


def rich_rows():
    return [
        {"role": "user", "content": [{"type": "text", "text": 'عربية 🪷 "\\\n'}]},
        {"role": "assistant", "content": None, "reasoning": "synthetic thought",
         "tool_calls": [{"id": "c1", "type": "function", "function": {
             "name": "synthetic", "arguments": '{"x":"🪷","n":1}'}}]},
        {"role": "tool", "content": "synthetic result", "tool_call_id": "c1", "name": "synthetic"},
        {"role": "assistant", "content": "[LCM sensitive redaction: api_key; sha256=" + "a" * 64 + "; chars=8; bytes=8]"},
        NOTE, SUMMARY, LOSSY, {"role": "user", "content": "after lossy row"},
    ]


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
@pytest.mark.parametrize("shape", ["empty", "rich", "summary_only", "late_note", "random"])
def test_prefix_digests_equal_legacy_oracle_at_every_cut(engine, flags, shape):
    messages = {
        "empty": [], "rich": rich_rows(), "summary_only": [SUMMARY] + rows(4),
        "late_note": [SUMMARY] + rows(4) + [NOTE] + rows(2),
        "random": random.Random(1090).choices(rich_rows() + rows(8), k=40),
    }[shape]
    before = deepcopy(messages)
    actual = engine._replay_snapshot_prefix_digests(
        messages, native_recovery=flags[0], compacted_active=flags[1], session_end=flags[2])
    # These unchanged single-snapshot readers are the persisted version-1 oracle.
    readers = (engine._native_recovery_replay_snapshot_digest,
               engine._compacted_active_replay_snapshot_digest,
               engine._session_end_replay_snapshot_digest)
    for enabled, digests, reader in zip(flags, actual, readers):
        expected = {}
        if enabled:
            for cursor in range(len(messages) + 1):
                digest = reader(messages[:cursor])
                if digest:
                    expected[cursor] = digest
        assert digests == expected
    assert messages == before


def test_streamed_bytes_equal_legacy_serialization(engine, monkeypatch):
    import hermes_lcm.reconcile as reconcile

    messages = rich_rows()
    streams = []
    original = hashlib.sha256

    class RecordedHash:
        def __init__(self, data=b""):
            self.data = data
            self.inner = original(data)

        def update(self, data):
            self.data += data
            self.inner.update(data)

        def copy(self):
            other = RecordedHash()
            other.data = self.data
            other.inner = self.inner.copy()
            return other

        def hexdigest(self):
            streams.append(self.data)
            return self.inner.hexdigest()

    monkeypatch.setattr(reconcile.hashlib, "sha256", RecordedHash)
    # Every cut is eligible for native proof before the lossy row. Session-end
    # proof admits the lossy suffix, as the unchanged legacy reader does.
    engine._replay_snapshot_prefix_digests(messages, native_recovery=True, session_end=True)
    expected = [json.dumps(
        {"version": 1, "messages": [list(engine._message_replay_identity(m)) for m in messages[:cut]]},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8") for cut in range(1, len(messages) + 1)]
    assert streams == expected


@pytest.mark.parametrize("namespace", ["native", "active", "end"])
@pytest.mark.parametrize("adopted", [False, True])
def test_snapshot_match_and_host_rejection_keep_cursor_rules(engine, namespace, adopted):
    snapshot = ([NOTE] if namespace == "active" else []) + [SUMMARY] + rows(2)
    if namespace == "native":
        engine._remember_native_recovery_replay_snapshot_digest(
            engine._native_recovery_replay_snapshot_digest(snapshot))
    elif namespace == "active":
        engine._remember_compacted_active_replay_snapshot(snapshot)
    else:
        engine._remember_session_end_replay_snapshot(engine._session_id, snapshot)
    messages = deepcopy(snapshot) + [{"role": "user", "content": "new unique delta"}]
    if not adopted:
        messages[1 if namespace == "active" else 0]["content"] += " host rejected"
    tail = [("user", "unrelated", "", "", "")] + [
        engine._message_replay_identity(m) for m in [SUMMARY] + rows(2)]

    def cursor(allowed):
        return engine._find_reconciled_cursor_for_store_tail(
            messages, tail, allow_empty_prefix=True, session_count=len(tail),
            raw_session_count=100, allow_session_end_replay_proof=allowed)

    # The bare LCM system note is an empty/scaffold prefix, not a durable row.
    rejected_cursor = 1 if namespace == "active" else 0
    assert cursor(namespace == "end") == (len(snapshot) if adopted else rejected_cursor)
    if namespace == "end":
        assert cursor(False) == 0  # ordinary ingest cannot consume end proof


@pytest.mark.parametrize("lossy", [False, True])
def test_native_empty_session_durable_proof_preserves_new_delta(engine, lossy):
    snapshot = rows(3) + ([LOSSY] if lossy else [])
    digest = engine._native_recovery_replay_snapshot_digest(snapshot)
    assert bool(digest) is not lossy
    engine._remember_native_recovery_replay_snapshot_digest(digest)
    before = engine._store.read_metadata_json(engine._replay_snapshot_metadata_key(
        "native_recovery_replay_snapshot_digests"))
    assert engine._reconcile_ingest_cursor_from_store(snapshot + rows(1)) == (0 if lossy else len(snapshot))
    assert engine._store.read_metadata_json(engine._replay_snapshot_metadata_key(
        "native_recovery_replay_snapshot_digests")) == before


@pytest.mark.parametrize("n", [32, 64, 128])
def test_prefix_hash_and_serialization_work_is_linear(engine, monkeypatch, n):
    import hermes_lcm.reconcile as reconcile

    calls = Counter()
    original_hash, original_dumps = hashlib.sha256, json.dumps

    class CountedHash:
        def __init__(self, data=b""):
            calls["bytes"] += len(data)
            self.inner = original_hash(data)

        def update(self, data):
            calls["bytes"] += len(data)
            self.inner.update(data)

        def copy(self):
            other = CountedHash()
            other.inner = self.inner.copy()
            return other

        def hexdigest(self):
            return self.inner.hexdigest()

    def counted_dumps(*args, **kwargs):
        calls["serializations"] += 1
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(reconcile.hashlib, "sha256", CountedHash)
    monkeypatch.setattr(reconcile.json, "dumps", counted_dumps)
    messages = [{"role": "user", "content": "fixed-size synthetic payload 🪷"}] * n
    digests, _, _ = engine._replay_snapshot_prefix_digests(messages, native_recovery=True)
    assert len(digests) == n
    assert calls["serializations"] == n
    one = original_dumps(list(engine._message_replay_identity(messages[0])),
                         ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    assert calls["bytes"] == len(b'{"messages":[') + n * len(one) + (n - 1) + n * len(b'],"version":1}')


@pytest.mark.parametrize("field", ["content", "tool_calls"])
def test_native_lossy_fence_is_prefix_local_and_identity_wide(engine, field):
    message = {"role": "assistant", "content": "synthetic", "tool_calls": []}
    if field == "content":
        message["content"] = LOSSY["content"]
    else:
        message["tool_calls"] = [{"id": "lossy", "type": "function", "function": {
            "name": "synthetic", "arguments": LOSSY["content"]}}]
    messages = rows(2) + [message] + rows(2)
    native, _, _ = engine._replay_snapshot_prefix_digests(messages, native_recovery=True)
    assert set(native) == {1, 2}
    assert native[2] == engine._native_recovery_replay_snapshot_digest(messages[:2])
    assert engine._native_recovery_replay_snapshot_digest(messages[:3]) == ""


@pytest.mark.parametrize("kind", ["tool_result", "media_payload"])
def test_externalized_payload_digests_match_and_cache_is_call_local(engine, monkeypatch, kind):
    from pathlib import Path
    import hermes_lcm.reconcile as reconcile

    directory = Path(engine._config.large_output_externalization_path)
    directory.mkdir(parents=True, exist_ok=True)
    payload_path = directory / "synthetic-1090.json"
    content = "data:image/png;base64," + "QUFB" * 65536 if kind == "media_payload" else "🪷 result " * 4096
    payload = {"kind": kind, "content": content, "session_id": engine._session_id,
               "role": "tool", "field_path": "content", "tool_call_id": "c1"}
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    message = {"role": "tool", "tool_call_id": "c1", "name": "synthetic", "content": (
        f"[Externalized tool output: tool_call_id=c1; chars={len(content)}; ref=synthetic-1090.json]"
    )}
    messages = [message] + rows(24)
    expected = {cut: engine._native_recovery_replay_snapshot_digest(messages[:cut])
                for cut in range(1, len(messages) + 1)}
    calls = Counter()
    original = reconcile.load_externalized_payload

    def counted(*args, **kw):
        calls["loads"] += 1
        return original(*args, **kw)

    monkeypatch.setattr(reconcile, "load_externalized_payload", counted)
    native, _, _ = engine._replay_snapshot_prefix_digests(messages, native_recovery=True)
    assert native == expected
    assert calls["loads"] == 1
    payload["content"] += " changed generation"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    changed, _, _ = engine._replay_snapshot_prefix_digests(messages, native_recovery=True)
    assert changed[1] != native[1]
    assert calls["loads"] == 2
    assert changed[1] == engine._native_recovery_replay_snapshot_digest(messages[:1])
