"""B1 emit carry: exact dictionaries, legacy parity and persisted restart replay."""

import copy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import is_externalized_placeholder, maybe_externalize_tool_output
from hermes_lcm.message_analysis import _merge_adjacent_assistant_messages
from hermes_lcm.tokens import count_messages_tokens
from tests.test_host_uid_shadow import _bindings, _counts, _engine, _m, _rows, _state_db

try:
    from hermes_lcm import host_uid_emit as emit
except ImportError:  # Red-at-base must exercise the sites, not fail collection.
    emit = None

IDENTITY = ("message_uid", "_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid")
ADDRESS = ("_row_id", "_db_row_snapshot", "_canonical_row")
# Hermes agent.message_metadata.PERSISTENCE_ONLY_MESSAGE_FIELDS.
PERSISTENCE_ONLY = set(IDENTITY + ADDRESS) | {
    "timestamp", "display_kind", "display_metadata", "_submit_row_session_id", "_merged_turn_prefix",
}


@pytest.fixture(autouse=True)
def carry_mode(monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    if emit is not None:
        monkeypatch.setattr(emit, "_host_uid_capability", True)


@pytest.fixture
def build(tmp_path):
    engines = []

    def make(**overrides):
        settings = dict(database_path=str(tmp_path / f"emit-{len(engines)}.db"),
                        fresh_tail_count=10, leaf_chunk_tokens=20_000)
        settings.update(overrides)
        engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "host"))
        engine.on_session_start("S", conversation_id="conv", context_length=200_000)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.shutdown()


def _metadata(uid="u"):
    return dict(message_uid=uid, _absorbed_message_uids=[uid + "-absorbed"],
                _tool_call_uids={"c": "call-uid"}, _tool_call_uid="result-uid",
                _row_id=10, _db_row_snapshot="snapshot", _canonical_row=True)


def _call(call_id):
    return {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}


def _provider(rows):
    return [{k: v for k, v in row.items() if k not in PERSISTENCE_ONLY} for row in rows]


def _off_receipt(name, value):
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    print(f"OFF_PARITY {name} {digest}")


@pytest.mark.parametrize("case", ["text", "calls", "no-first-uid"])
@pytest.mark.parametrize("off", [False, True])
def test_site1_exact_merge(case, off, monkeypatch):
    first = _m("assistant", "  first \n", uid="first", _absorbed_message_uids=["old", "first"], _row_id=11)
    second = _m("assistant", "\tsecond ", uid="second", _absorbed_message_uids=["old", "third", "first"], _row_id=22)
    expected = dict(first, content="first\nsecond")
    if case == "no-first-uid":
        first.pop("message_uid")
        first.pop("_absorbed_message_uids")
        expected = dict(first, content="first\nsecond")
    if case == "calls":
        first.update(tool_calls=[_call("same"), _call("same"), _call("only-a")],
                     _tool_call_uids={"same": "a", "only-a": "a-only"})
        second.update(tool_calls=[_call("same"), _call("only-b"), _call("list")],
                      _tool_call_uids={"same": "b", "only-b": "b-only", "list": ["l1", "l2"]})
        expected.update(tool_calls=first["tool_calls"] + second["tool_calls"],
                        _tool_call_uids=first["_tool_call_uids"] if off else {
                            "same": ["a", "a", "b"], "only-a": "a-only", "only-b": "b-only", "list": ["l1", "l2"],
                        })
    if off:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    else:
        expected["_absorbed_message_uids"] = ["second", "old", "third", "first"] if case == "no-first-uid" else ["old", "second", "third"]
    original = copy.deepcopy([first, second])
    out = _merge_adjacent_assistant_messages([first, second])
    assert out == [expected]
    assert [first, second] == original
    if off:
        _off_receipt("merge-" + case, out)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
        assert _provider(out) == _provider(_merge_adjacent_assistant_messages([first, second]))


def test_site1_absent_capability_equals_base(monkeypatch):
    if emit is not None:
        monkeypatch.setattr(emit, "_host_uid_capability", False)
    rows = [_m("assistant", "a", uid="a", tool_calls=[_call("c")], _tool_call_uids={"c": "a-c"}),
            _m("assistant", "b", uid="b", tool_calls=[_call("c")], _tool_call_uids={"c": "b-c"})]
    expected = [dict(rows[0], content="a\nb", tool_calls=[_call("c"), _call("c")])]
    assert _merge_adjacent_assistant_messages(rows) == expected
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    assert _merge_adjacent_assistant_messages(rows) == expected


@pytest.mark.parametrize("role", ["user", "assistant", "tool", "system"])
@pytest.mark.parametrize("off", [False, True])
def test_site3_each_placeholder(role, off, build, monkeypatch):
    engine = build(ignore_message_patterns=["IGNORE_THIS"])
    row = dict(role=role, content="IGNORE_THIS payload", tool_call_id="c", tool_calls=[_call("c")], **_metadata())
    if off:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    elif emit is not None:
        monkeypatch.setattr(emit, "_host_uid_capability", False)  # Copies do not require capability.
    out = engine._apply_ignored_active_replay_placeholders([row], [dict(row)])
    expected = dict(role=role, content=engine._ignored_active_replay_placeholder(row["content"]))
    if role == "tool":
        expected["tool_call_id"] = "c"
    if not off:
        expected.update(message_uid="u", _absorbed_message_uids=["u-absorbed"])
        if role == "tool":
            expected["_tool_call_uid"] = "result-uid"
    assert out == [expected]
    assert not set(ADDRESS) & out[0].keys()
    if off:
        _off_receipt("placeholder-" + role, out)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
        base = engine._apply_ignored_active_replay_placeholders([row], [dict(row)])
        assert _provider(out) == base


def _tail():
    return [_m("user", "run tools"), _m("assistant", "running tools", tool_calls=[_call("c")]),
            dict(role="tool", tool_call_id="c", content="oversized payload " * 1000, **_metadata())]


def _rich(engine, row):
    assert maybe_externalize_tool_output(row["content"], tool_call_id="c", session_id="S",
                                        config=engine._config, hermes_home=engine._hermes_home,
                                        tool_name="read_file") is not None
    stub = engine._over_cap_tool_result_stub(row, "read_file")
    assert is_externalized_placeholder(stub["content"])
    return stub


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("off", [False, True])
def test_site8_plain_and_rich_stub(rich, off, build, monkeypatch):
    engine = build(large_output_externalization_enabled=rich, large_output_externalization_threshold_chars=100)
    row = _tail()[-1]
    if off:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    elif emit is not None:
        monkeypatch.setattr(emit, "_host_uid_capability", False)
    stub = _rich(engine, row) if rich else engine._over_cap_tool_result_stub(row)
    expected = dict(engine._missing_tool_result_stub("c"), content=stub["content"])
    if not off:
        expected.update(message_uid="u", _tool_call_uid="result-uid")
    assert stub == expected
    assert engine._missing_tool_result_stub("c").keys() == {"role", "content", "tool_call_id"}
    if off:
        # Payload ref names are minted; provider text is independently compared below.
        _off_receipt("stub-" + str(rich), {**stub, "content": "rich-ref" if rich else stub["content"]})
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
        assert _provider([stub]) == [engine._over_cap_tool_result_stub(row, "read_file")]


@pytest.mark.parametrize("off", [False, True])
def test_site8_pending_downgrade(off, build, monkeypatch):
    engine = build(large_output_externalization_enabled=True, large_output_externalization_threshold_chars=100)
    system, tail = _m("system", "system"), _tail()
    if off:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    rich = _rich(engine, tail[-1])
    plain = engine._missing_tool_result_stub("c")
    cap = count_messages_tokens([system, tail[1], plain])
    assert count_messages_tokens([system, tail[1], rich]) > cap
    out = engine._assemble_overflow_recovery_context(system, tail, assembly_cap_override=cap)
    expected = dict(plain) if off else dict(plain, message_uid="u", _tool_call_uid="result-uid")
    assert out == [system, tail[1], expected]
    if off:
        _off_receipt("downgrade", out)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
        assert _provider(out) == engine._assemble_overflow_recovery_context(system, tail, assembly_cap_override=cap)


@pytest.mark.parametrize("appended", [False, True])
@pytest.mark.parametrize("off", [False, True])
def test_site17_cache_refresh_and_removal(appended, off, build, monkeypatch):
    engine = build()
    initial = [_m("user", "ask", **_metadata("old-user")), _m("assistant", "reply", **_metadata("old-reply"))]
    engine._ingest_messages(initial)
    host = [_m("user", "ask", **_metadata("new-user")), _m("assistant", "reply")]
    host[0].update(_row_id=20, _db_row_snapshot="new-snapshot", _canonical_row=False)
    if appended:
        host += [_m("user", "next", uid="next")]
    if off:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    out = engine._ingest_messages(host)
    expected = [*initial, *host[2:]] if off else host
    assert out == expected
    assert len(out) == len(host)
    if off:
        _off_receipt("cache-" + str(appended), out)
    else:
        assert _provider(out) == _provider(expected)


@pytest.mark.parametrize("off", [False, True])
def test_site17_generated_placeholder_never_keeps_address(off, build, monkeypatch):
    engine = build(ignore_message_patterns=["IGNORE_THIS"])
    source = [_m("user", "IGNORE_THIS", **_metadata("old"))]
    cached = engine._apply_ignored_active_replay_placeholders(source, copy.deepcopy(source))
    cached[0].update(_metadata("stale"))  # Assert removal even if an old cache holds addresses.
    engine._remember_active_replay_messages(source, cached)
    host = [_m("user", "IGNORE_THIS", message_uid="new", _row_id=20, _db_row_snapshot="new")]
    if off:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    out = engine._cached_active_replay_messages(host)
    expected = cached[0] if off else {"role": "user", "content": cached[0]["content"], "message_uid": "new"}
    assert out == [expected]
    assert id(out[0]) in engine._generated_ignored_active_replay_placeholder_message_ids
    if off:
        _off_receipt("cache-placeholder", out)


def test_site17_misaligned_cache_is_not_synced(build):
    engine = build()
    host = [_m("user", "a", uid="current"), _m("assistant", "b", uid="current-b")]
    cached = [_m("user", "a", uid="stale", _row_id=99)]
    engine._last_active_replay_source_identities = [engine._message_replay_identity(m, strip_carrier=False) for m in host]
    engine._last_active_replay_messages = cached
    assert engine._cached_active_replay_messages(host) == cached  # Historical zip behavior, no sync/count.


@pytest.mark.parametrize("failure", ["missing-module", "missing-attribute", "exception", "absent", "present"])
def test_capability_probe_cached_and_fail_closed(failure, monkeypatch):
    module = importlib.import_module("hermes_lcm.host_uid_emit")
    monkeypatch.setattr(module, "_host_uid_capability", None)
    calls = []

    def probe(name):
        calls.append(name)
        if failure == "missing-module":
            raise ImportError("not installed")
        if failure == "exception":
            raise RuntimeError("broken module")
        if failure == "missing-attribute":
            return SimpleNamespace()
        return SimpleNamespace(PERSISTENCE_ONLY_MESSAGE_FIELDS={"message_uid"} if failure == "present" else set())

    monkeypatch.setattr(module, "import_module", probe)
    assert module.host_uid_capable() is (failure == "present")
    assert module.host_uid_capable() is (failure == "present")
    assert calls == ["agent.message_metadata"]


@pytest.mark.parametrize("capable", [False, True])
@pytest.mark.parametrize("site", ["merge", "placeholder", "overflow", "cache"])
def test_p7_uid_free_compress_and_provider_parity(site, capable, build, monkeypatch):
    if emit is not None:
        monkeypatch.setattr(emit, "_host_uid_capability", capable)
    if site == "merge":
        rows = [_m("user", "ask"), _m("assistant", "a", tool_calls=[_call("c")]),
                _m("assistant", "b", tool_calls=[_call("c")]), _m("tool", "result", tool_call_id="c")]
    elif site == "placeholder":
        rows = [_m("user", "IGNORE_THIS"), _m("assistant", "reply")]
    elif site == "overflow":
        rows = [_m("system", "system"), *_tail()]
        rows[-1] = {k: v for k, v in rows[-1].items() if k not in IDENTITY + ADDRESS}
    else:
        rows = [_m("user", "ask"), _m("assistant", "reply")]
    settings = dict(ignore_message_patterns=["IGNORE_THIS"], max_assembly_tokens=300)
    shadow, base = build(**settings), build(**settings)
    if site == "cache":
        shadow._ingest_messages(copy.deepcopy(rows))
        base._ingest_messages(copy.deepcopy(rows))
    carried = shadow.compress(copy.deepcopy(rows), current_tokens=1000)
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    off = base.compress(copy.deepcopy(rows), current_tokens=1000)
    assert carried == off  # Nothing can be constructed from uid-free input, even on capable hosts.
    assert _provider(carried) == _provider(off)
    assert not any(set(IDENTITY) & row.keys() for row in carried)
    assert shadow.last_compression_status == base.last_compression_status
    assert shadow.compression_count == base.compression_count
    _off_receipt("compress-" + site + "-" + str(capable), off)


@pytest.mark.parametrize("capable", [False, True])
def test_p7_capability_gates_merge_construction_through_compress(capable, build, monkeypatch):
    if emit is not None:
        monkeypatch.setattr(emit, "_host_uid_capability", capable)
    host = [_m("user", "ask", 10.0, "ask"),
            _m("assistant", " first ", 11.0, "first", tool_calls=[_call("c"), _call("c")], _tool_call_uids={"c": "a"}),
            _m("assistant", " second ", 12.0, "second", tool_calls=[_call("c")], _tool_call_uids={"c": "b"}),
            _m("tool", "result", 13.0, "result", tool_call_id="c", _tool_call_uid="b")]
    engine, baseline = build(), build()
    out = engine.compress(copy.deepcopy(host), current_tokens=1000)
    merged = next(row for row in out if row.get("role") == "assistant")
    assert merged["message_uid"] == "first"
    assert merged.get("_absorbed_message_uids") == (["second"] if capable else None)
    assert merged["_tool_call_uids"] == ({"c": ["a", "a", "b"]} if capable else {"c": "a"})
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    off = baseline.compress(copy.deepcopy(host), current_tokens=1000)
    assert _provider(out) == _provider(off)
    assert (engine.last_compression_status, engine.compression_count) == (baseline.last_compression_status, baseline.compression_count)
    if not capable:
        assert out == off
    _off_receipt("compress-uid-merge-" + str(capable), off)


@pytest.mark.parametrize("site", ["placeholder", "overflow", "merge"])
def test_p6_restart_persisted_rows_do_not_rebind(tmp_path, monkeypatch, site):
    _state_db(tmp_path, [("S", None, None)])
    # Use slice A's real-engine helpers and stores; only the emit transformation differs.
    first = _engine(tmp_path)
    first._hermes_home = str(tmp_path)
    first._config.max_assembly_tokens = 300 if site == "overflow" else 10_000
    if site == "placeholder":
        from hermes_lcm.message_patterns import compile_message_patterns
        first._compiled_ignore_message_patterns = compile_message_patterns(["IGNORE_THIS"])
        host = [_m("user", "IGNORE_THIS payload", 10.0, "ignored"), _m("assistant", "reply", 11.0, "reply")]
        target_uid = "ignored"
    elif site == "overflow":
        first._config.large_output_externalization_enabled = True
        first._config.large_output_externalization_threshold_chars = 100
        host = [_m("system", "system", 9.0, "system"), *_tail()]
        for i, row in enumerate(host):
            row["message_uid"] = "host-" + str(i)
            row["timestamp"] = 10.0 + i
        target_uid = host[-1]["message_uid"]
    else:
        host = [_m("user", "ask", 10.0, "ask"), _m("assistant", "first", 11.0, "first"),
                _m("assistant", "second", 12.0, "second")]
        target_uid = "first"
    try:
        out = first.compress(copy.deepcopy(host), current_tokens=1000)
        carried = next(row for row in out if row.get("message_uid") == target_uid)
        original = next(row for row in host if row.get("message_uid") == target_uid)
        assert carried["content"] != original["content"]
        if site == "merge":
            assert carried["content"] == "first\nsecond"
            assert carried["_absorbed_message_uids"] == ["second"]
        if site == "overflow":
            assert is_externalized_placeholder(carried["content"])
        bindings = _bindings(first)
        before_rows = _rows(first)
        persisted = [dict(row, message_uid=row.get("message_uid") or "minted-" + str(i)) for i, row in enumerate(out)]
        assert len({row["message_uid"] for row in persisted}) == len(persisted)
    finally:
        first.shutdown()
    second = _engine(tmp_path)
    second._hermes_home = str(tmp_path)
    try:
        second._ingest_messages(persisted)
        counts = _counts(second)
        assert not any("disagree" in k.lower() for k in counts), counts
        assert not any(b[1] == target_uid for b in _bindings(second) if b not in bindings)
        assert _rows(second) == before_rows
    finally:
        second.shutdown()


def _merge_restart(tmp_path, mutate=None):
    """Persist LCM's site-1 merge, re-present it after a restart; return (counts, gate rows, doctor lines)."""
    from hermes_lcm.host_uid import host_uid_doctor_lines
    from tests.test_host_uid_shadow import _gate

    tmp_path.mkdir(exist_ok=True)
    _state_db(tmp_path, [("S", None, None)])
    host = [_m("user", "ask", 10.0, "ask"), _m("assistant", "first", 11.0, "first"),
            _m("assistant", "second", 12.0, "second")]
    first = _engine(tmp_path)
    first._hermes_home = str(tmp_path)
    try:
        out = first.compress(copy.deepcopy(host), current_tokens=1000)
    finally:
        first.shutdown()
    persisted = [dict(row, message_uid=row.get("message_uid") or "minted-" + str(i)) for i, row in enumerate(out)]
    merged = next(row for row in persisted if row.get("message_uid") == "first")
    assert merged["_absorbed_message_uids"] == ["second"]
    if mutate:
        mutate(persisted, merged)
    second = _engine(tmp_path)
    second._hermes_home = str(tmp_path)
    try:
        second._ingest_messages(persisted)
        return _counts(second), _gate(second), host_uid_doctor_lines(second)
    finally:
        second.shutdown()


def test_lcm_merge_gate_checks_both_constituents_as_agree(tmp_path):
    counts, gate, doctor = _merge_restart(tmp_path)
    assert counts.get("replay.composite.agree.lcm_merge") == 1, counts
    assert {uid: (checked, disagree) for uid, checked, disagree in gate if uid in ("first", "second")} == {
        "first": ("agree", 0), "second": ("agree", 0)}
    assert "host_uid_gate_disagree: 0" in doctor


def _classify_replayed(tmp_path, message):
    """Seed bound assistants, then classify ``message`` as an unmapped prefix replay (no #436 row map)."""
    from tests.test_host_uid_shadow import _classify, _seed

    _state_db(tmp_path, [("S", None, None)])
    _seed(tmp_path, [_m("user", "ask", 10.0, "ask"), _m("assistant", "  first ", 11.0, "first"),
                     _m("assistant", "second", 12.0, "second")])
    engine = _engine(tmp_path)
    try:
        return _classify(engine, [message], {"replayed": {0}, "matched": {}})
    finally:
        engine.shutdown()


@pytest.mark.parametrize("role, content, absorbed, expected", [
    ("assistant", "first\nsecond", ["second"], "replay.composite.agree.lcm_merge"),  # positive control
    ("assistant", "first\nsecomd", ["second"], "replay.bound.disagree.prefix_replay"),  # one byte changed
    ("assistant", "first\nsecond", ["never-bound"], "replay.bound.disagree.prefix_replay"),  # absorbed unbound
    ("assistant", "first\nsecond", [], "replay.bound.disagree.prefix_replay"),  # no absorbed uids
    ("user", "first\nsecond", ["second"], "replay.bound.disagree.prefix_replay"),  # user role: rule not applied
    ("user", "first\nsecond", [], "replay.bound.disagree.prefix_replay"),
])
def test_lcm_merge_rule_negative_controls(tmp_path, role, content, absorbed, expected):
    message = _m(role, content, 11.0, "first", _absorbed_message_uids=absorbed)
    assert _classify_replayed(tmp_path, message) == {expected: 1}
