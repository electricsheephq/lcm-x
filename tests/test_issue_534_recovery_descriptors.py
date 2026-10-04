"""#534 Candidate A: recovery provenance, restart alignment and loss probes."""
import copy
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine, _OVERFLOW_RECOVERY_PLACEHOLDER
from hermes_lcm.reconcile import _emission_identity, _finalize_emission_descriptors
from tests.test_compression_boundary import (
    _compacted_engine, _durable_commit_proof, _V0240_DURABLE_PROOF_KEYS,
)
from tests.test_issue_529_overflow_followups import CAP, SYSTEM, OVERSIZED, ORPHAN, _note

_RC1 = "c36b46e3a7b1bff23cfa45fa117565ab8624441d"  # the v0.25.0-rc1 tag commit


def _engine(path, sid="S0"):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(path / "lcm.db"), fresh_tail_count=10,
        large_output_externalization_path=str(path / "externalized"),
    ), hermes_home=str(path / "home"))
    engine.on_session_start(sid, platform="acp", context_length=200_000)
    return engine


def _recover(engine, monkeypatch, *, system=True, older="older request", stamp=False, prefix=None):
    newest = {"role": "user", "content": "newest request " * 300}
    pre = (copy.deepcopy(prefix) if prefix is not None else ([dict(SYSTEM)] if system else [])) + [
        {"role": "user", "content": older},
        {"role": "assistant", "content": OVERSIZED}, newest, dict(ORPHAN),
    ]
    if stamp:
        for i, row in enumerate(pre):
            row["timestamp"] = 1000.0 + i
    engine._config.max_assembly_tokens = CAP
    monkeypatch.setattr(engine, "_summary_route_stop_applies", lambda *_a, **_kw: True)
    # Force the existing no-leaf overflow path, not a mock compress() result.
    out = engine.compress(pre, force=True)
    assert engine._last_compression_status == "overflow_recovery"
    assert out[-1]["content"] == _note(newest)["content"]
    return pre, out


def _commit(engine, pre, sid="S0"):
    old = engine._session_id
    engine.on_session_end(old, pre)
    engine.on_session_start(sid, boundary_reason="compression", old_session_id=old, platform="acp")


def _contents(engine):
    return [r[0] for r in engine._store._conn.execute("SELECT content FROM messages")]


def _merged(out):
    rows = copy.deepcopy(out)
    assert rows[-2]["role"] == rows[-1]["role"] == "user"
    rows[-2]["content"] += "\n\n" + rows.pop()["content"]
    return rows


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("merged", [False, True])
def test_recovery_restart_after_reply(tmp_path, monkeypatch, prior, merged):
    engine = _compacted_engine(tmp_path, monkeypatch)[0] if prior else _engine(tmp_path)
    if prior:
        get_nodes = engine._dag.get_uncondensed_at_depth
        def oversized_nodes(*args, **kwargs):
            nodes = copy.deepcopy(get_nodes(*args, **kwargs))
            for node in nodes:
                node.summary = OVERSIZED
            return nodes
        monkeypatch.setattr(engine._dag, "get_uncondensed_at_depth", oversized_nodes)
    pre, out = _recover(engine, monkeypatch)
    note = out[-1]["content"]
    _commit(engine, pre)
    host = _merged(out) if merged else copy.deepcopy(out)
    reply = {"role": "assistant", "content": "recovery reply"}
    engine.ingest(host + [reply])
    engine.shutdown()
    replay = _engine(tmp_path)
    try:
        replay.ingest(host + [reply, {"role": "user", "content": "next request"}])
        contents = _contents(replay)
        assert not any(note in content for content in contents)
        for content in ("older request", "recovery reply", "next request"):
            assert contents.count(content) == 1
    finally:
        replay.shutdown()


@pytest.mark.parametrize("retained,merged", [(False, False), (True, False), (True, True)])
def test_placeholder_restart(tmp_path, monkeypatch, retained, merged):
    engine = _engine(tmp_path)
    pre = [dict(SYSTEM), dict(ORPHAN)]
    engine._ingest_messages(pre)
    retained_row = {"role": "user", "content": "retained request"} if retained else None
    if retained:
        engine._ingest_messages([dict(SYSTEM), retained_row, dict(ORPHAN)])
    out = engine._assemble_overflow_recovery_context(
        pre[0], pre[1:], assembly_cap_override=CAP,
    )
    if retained:
        # A retained-prefix descriptor fixture: current assembler returns the retained row
        # early; use its actual generated placeholder to cover the suffix-binding shape.
        out.insert(-1, retained_row)
    out = engine._finalize_forced_overflow_result(pre, out, assembly_cap_override=CAP)
    engine._record_compress_commit_proof(pre, out)
    assert out[-1]["content"] == _OVERFLOW_RECOVERY_PLACEHOLDER
    host = _merged(out) if merged else out
    engine.shutdown()
    replay = _engine(tmp_path)
    try:
        assert replay._cursor_from_durable_commit_proof(host) == len(host)
        replay.ingest(host + [{"role": "assistant", "content": "placeholder reply"}])
        assert not any(_OVERFLOW_RECOVERY_PLACEHOLDER in c for c in _contents(replay))
        assert _contents(replay).count("placeholder reply") == 1
        if retained:
            assert _contents(replay).count("retained request") == 1
    finally:
        replay.shutdown()


@pytest.mark.parametrize("system", [False, True])
def test_note_only_restart(tmp_path, monkeypatch, system):
    engine = _engine(tmp_path)
    _pre, out = _recover(engine, monkeypatch, system=system, older="older ask " * 40)
    assert [r["role"] for r in out] == (["system", "user"] if system else ["user"])
    before = _contents(engine)
    engine.shutdown()
    replay = _engine(tmp_path)
    try:
        assert replay._cursor_from_durable_commit_proof(out) == len(out)
        replay.ingest(out + [{"role": "assistant", "content": "note-only reply"},
                             {"role": "user", "content": "next request"}])
        assert _contents(replay) == before + ["note-only reply", "next request"]
    finally:
        replay.shutdown()


def test_rotation_rekeys_recovery_proof(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    pre, out = _recover(engine, monkeypatch)
    _commit(engine, pre, "S1")
    child = _durable_commit_proof(engine, "S1")
    assert any(d["kind"] == "recovery" for d in child["emissions"])
    assert all(d["scope"]["session_id"] == "S1" for d in child["emissions"])
    before = _contents(engine)
    engine.shutdown()
    replay = _engine(tmp_path, "S1")
    try:
        replay.ingest(_merged(out))
        assert _contents(replay) == before
    finally:
        replay.shutdown()


def test_published_compaction_carries_recovery_descriptor(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        pre, out = _recover(engine, monkeypatch)
        _commit(engine, pre)
        host = out + [{"role": "assistant", "content": "reply"}]
        engine.ingest(host)
        engine._config.max_assembly_tokens = 200_000
        # Exercise compress()'s real proof carry-forward without requiring another summary route.
        def published(_messages, **_kw):
            engine._last_compression_status = "compacted"
            engine._ingest_cursor = len(out)
            return out
        monkeypatch.setattr(engine, "_compress_impl", published)
        engine.compress(host, force=True)
        assert any(d["kind"] == "recovery" for d in _durable_commit_proof(engine, "S0")["emissions"])
        assert engine._cursor_from_durable_commit_proof(out) == len(out)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("merged", [False, True])
def test_stamped_in_process_merge_has_no_anchor_remainder(tmp_path, monkeypatch, merged):
    engine = _engine(tmp_path)
    try:
        pre, out = _recover(engine, monkeypatch, stamp=True)
        proof = copy.deepcopy(engine._compress_commit_proof)
        host = _merged(out) if merged else copy.deepcopy(out)
        assert engine._remap_cursor_through_host_merge(host, proof) == len(host)
        _commit(engine, pre)
        host += [{"role": "assistant", "content": "stamped reply", "timestamp": 1100.0}]
        plan = engine._identity_anchor_prematch(host, host, 0)
        assert not plan["remainders"]
        assert 1 in plan["replayed"]
        if not merged:
            assert 2 in plan["replayed"]
        engine.ingest(host)
        assert not any(out[-1]["content"] in c for c in _contents(engine))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True])
def test_later_same_bytes_user_turn_is_stored(tmp_path, monkeypatch, restart):
    engine = _engine(tmp_path)
    pre, out = _recover(engine, monkeypatch)
    _commit(engine, pre)
    note = out[-1]["content"]
    host = _merged(out) + [{"role": "assistant", "content": "reply"}]
    engine.ingest(host)
    if restart:
        engine.shutdown()
        engine = _engine(tmp_path)
    try:
        engine.ingest(host + [{"role": "user", "content": note}])
        assert _contents(engine).count(note) == 1
    finally:
        engine.shutdown()


def test_loss_probe_same_bytes_left_and_right_of_recorded_index(tmp_path):
    engine = _engine(tmp_path)
    try:
        real = {"role": "user", "content": _OVERFLOW_RECOVERY_PLACEHOLDER}
        emitted = dict(real)
        out = [real, {"role": "assistant", "content": "held reply"}, emitted]
        candidate = {"kind": "recovery", "span": emitted["content"],
                     "row": emitted, "full_identity": _emission_identity(emitted)}
        proof = {"version": 4, **engine._emission_binding(),
                 "output": [_emission_identity(m) for m in out],
                 "emissions": _finalize_emission_descriptors(out, [candidate], engine._emission_binding())}
        host = out + [{"role": "assistant", "content": "later reply"}, dict(real)]
        _, identities = engine._occurrence_replay_identities(host, proof)
        assert identities[0] == identities[4] == engine._message_replay_identity(real, strip_carrier=False)
        assert identities[2] is None
        # With the generated row dropped, the real row may take the `recovery` binding; N6 restores its full identity.
        _, identities = engine._occurrence_replay_identities(host[:2] + host[3:], proof)
        assert all(identity is not None for identity in identities)
    finally:
        engine.shutdown()


def test_loss_probe_merge_base_mismatch_after_acp_strip(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        _pre, out = _recover(engine, monkeypatch, older="  older request  ")
        proof = engine._compress_commit_proof
        host = _merged(out)
        host[-1]["content"] = host[-1]["content"].strip()
        assert engine._occurrence_replay_identities(host, proof)[1][-1][1].strip() == "older request"
        host[-1]["content"] = host[-1]["content"].replace("older request", "different request", 1)
        projection, identities = engine._occurrence_replay_identities(host, proof)
        assert projection.entries[-1].generated_span is None
        assert identities[-1][1] == host[-1]["content"]
        assert engine._cursor_from_durable_commit_proof(host) is None
    finally:
        engine.shutdown()


def test_loss_probe_rc1_rollback_reader(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{_RC1}^{{commit}}"], capture_output=True).returncode:
        pytest.skip("the v0.25.0-rc1 commit c36b46e3 is not in this checkout (shallow clone)")
    engine = _engine(tmp_path)
    try:
        _pre, out = _recover(engine, monkeypatch)
        payload = _durable_commit_proof(engine, "S0")
        assert payload["version"] == 3 and payload["descriptor_version"] == 4
        assert _V0240_DURABLE_PROOF_KEYS <= payload.keys()
        # Load the actual immutable rc1 reconcile module, including its kind filter and cursor.
        source = subprocess.run(["git", "-C", str(root), "show", f"{_RC1}:reconcile.py"],
                                capture_output=True, text=True, check=True).stdout
        module = types.ModuleType("hermes_lcm.rc1_reconcile")
        module.__package__ = "hermes_lcm"
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, "v0.25.0-rc1:reconcile.py", "exec"), module.__dict__)
        host = out + [{"role": "assistant", "content": "rollback reply"}]
        engine.ingest(host)
        for name in ("_occurrence_replay_identities", "_durable_commit_proof_payload", "_cursor_from_durable_commit_proof"):
            monkeypatch.setattr(engine, name, types.MethodType(getattr(module.ReconcileMixin, name), engine))
        assert engine._cursor_from_durable_commit_proof(host) is None
        assert engine._cursor_from_durable_commit_proof(_merged(out) + host[-1:]) is None
        assert engine._cursor_from_durable_commit_proof(_merged(out)) is None
        assert engine._occurrence_replay_identities(out, engine._durable_commit_proof_payload())[1][-1] is not None
        before = _contents(engine)
        engine._ingest_cursor = 0
        engine._ingest_cursor_needs_reconcile = True
        engine.ingest(host + [{"role": "user", "content": "rollback request"}])
        assert all(c in _contents(engine) for c in before)
        assert "rollback request" in _contents(engine)
    finally:
        engine.shutdown()


def test_loss_probe_two_recoveries_in_one_session(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        pre, first = _recover(engine, monkeypatch)
        _commit(engine, pre)
        engine.ingest(_merged(first) + [{"role": "assistant", "content": "first reply"}])
        pre, second = _recover(engine, monkeypatch, older="second older request",
                               prefix=_merged(first) + [{"role": "assistant", "content": "first reply"}])
        assert any(d["kind"] == "recovery" for d in _durable_commit_proof(engine, "S0")["emissions"])
        _commit(engine, pre)
        engine.ingest(_merged(second) + [{"role": "assistant", "content": "second reply"}])
        engine.shutdown()
        engine = _engine(tmp_path)
        engine.ingest(_merged(second) + [{"role": "assistant", "content": "second reply"},
                                        {"role": "user", "content": "third request"}])
        contents = _contents(engine)
        assert not any("[LCM overflow recovery]" in c for c in contents)
        for c in ("older request", "second older request", "first reply", "second reply", "third request"):
            assert contents.count(c) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("content, emitted", [
    ("<think>PRIVATE</think>visible", "visible"),
    ("<think>PRIVATE</think>", None),
])
def test_kept_reply_is_cleaned_like_other_active_rows(tmp_path, monkeypatch, content, emitted):
    engine = _engine(tmp_path)
    newest = {"role": "user", "content": "newest request " * 300}
    older = {"role": "user", "content": "older request"}
    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "oversized-call", "function": {"name": "write_file", "arguments": OVERSIZED}},
    ]}
    pre = [older, {"role": "assistant", "content": content}, newest, call, dict(ORPHAN)]
    engine._config.max_assembly_tokens = CAP
    monkeypatch.setattr(engine, "_summary_route_stop_applies", lambda *_a, **_kw: True)
    try:
        out = engine.compress(pre, force=True)
        assert engine._last_compression_status == "overflow_recovery"
        assert not any("PRIVATE" in (m.get("content") or "") for m in out)
        if emitted is None:
            assert out == [older, _note(newest)]
        else:
            assert out == [older, {"role": "assistant", "content": emitted}, _note(newest)]
    finally:
        engine.shutdown()


def test_fitting_last_reply_keeps_recovery_at_its_recorded_index(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    newest = {"role": "user", "content": "newest request " * 300}
    older = {"role": "user", "content": "older request"}
    reply = {"role": "assistant", "content": "already delivered reply"}
    # The oversized tool-call row must not displace the latest visible reply.
    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "oversized-call", "function": {"name": "write_file", "arguments": OVERSIZED}},
    ]}
    pre = [older, reply, newest, call, dict(ORPHAN)]
    engine._config.max_assembly_tokens = CAP
    monkeypatch.setattr(engine, "_summary_route_stop_applies", lambda *_a, **_kw: True)
    try:
        out = engine.compress(pre, force=True)
        assert engine._last_compression_status == "overflow_recovery"
        assert out == [older, reply, _note(newest)]
        descriptor = next(d for d in engine._compress_commit_proof["emissions"] if d["kind"] == "recovery")
        assert descriptor["output_occurrence"]["index"] == 2
        assert engine._occurrence_replay_identities(copy.deepcopy(out), engine._compress_commit_proof)[1][-1] is None
        _commit(engine, pre)
        host = copy.deepcopy(out) + [{"role": "assistant", "content": "new reply"}]
        engine.ingest(host)
        engine.shutdown()
        engine = _engine(tmp_path)
        engine.ingest(host + [{"role": "user", "content": "next request"}])
        contents = _contents(engine)
        assert not any("[LCM overflow recovery]" in (c or "") for c in contents)
        for c in (older["content"], reply["content"], "new reply", "next request"):
            assert contents.count(c) == 1
    finally:
        engine.shutdown()
