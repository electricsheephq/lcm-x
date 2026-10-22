"""I1/I2: anchor-rewound rows receive fresh ignore and quarantine decisions.

Synthetic shapes from the #1628 probes and the #633 before-cursor steer.
"""

from copy import deepcopy
import json
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_host_uid_emit_carry import _ContainsPattern

SESSION = "s1628"
MARK = "IGNORE_THIS_ROW"
LOOP = ("the same line again and again " * 3000).strip()


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=False,
        large_output_externalization_path=str(tmp_path / "home" / "payloads"),
    )
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    instance.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    yield instance
    instance.shutdown()


def _rows(engine):
    return [row for row in engine._store.get_session_messages(SESSION)
            if row.get("role") != "system"]


def _contents(engine):
    return [row["content"] for row in _rows(engine)]


def _pattern(engine):
    engine._compiled_ignore_message_patterns = [_ContainsPattern(MARK)]


def _first():
    return [{"role": "user", "content": "Run the migration", "timestamp": 100.0},
            {"role": "assistant", "content": "Migration done.", "timestamp": 101.0}]


def _r6(engine, new_text, *, role, in_place):
    host = _first()
    engine.ingest(host)
    if in_place:
        host[0 if role == "user" else 1]["content"] = new_text
    host.append({"role": "user", "content": new_text if role == "user" and not in_place
                 else "now deploy it", "timestamp": 200.0})
    if role == "assistant" and not in_place:
        host.append({"role": "assistant", "content": new_text, "timestamp": 200.5})
    host.append({"role": "assistant", "content": "Deploying.", "timestamp": 201.0})
    engine.ingest(host)
    return host


@pytest.mark.parametrize("in_place", [True, False], ids=["r6_user_rewound", "control_appended"])
def test_r6_user_ignore_pattern_drops_row(engine, in_place):
    _pattern(engine)
    _r6(engine, f"{MARK} revised prompt", role="user", in_place=in_place)
    assert "Deploying." in _contents(engine)
    assert not any(MARK in text for text in _contents(engine))


def test_r6_user_control_no_pattern_stores_version(engine):
    host = _r6(engine, f"{MARK} revised prompt", role="user", in_place=True)
    expected = ["Run the migration", "Migration done.", f"{MARK} revised prompt",
                "now deploy it", "Deploying."]
    assert _contents(engine) == expected
    engine.ingest(host)
    assert _contents(engine) == expected


@pytest.mark.parametrize("in_place", [True, False], ids=["r6_rewound", "control_appended"])
def test_r6_quarantined_assistant_row_is_stored(engine, tmp_path, in_place):
    ignored = engine._ignored_message_count
    host = _r6(engine, LOOP, role="assistant", in_place=in_place)
    assistants = [row["content"] for row in _rows(engine) if row["role"] == "assistant"]
    assert len(assistants) == 3
    assert assistants[0] == "Migration done." and assistants[-1] == "Deploying."
    assert "assistant output quarantined" in assistants[1] and "ref=" in assistants[1]
    assert engine._ignored_message_count == ignored
    payloads = list((tmp_path / "home" / "payloads").glob("*.json"))
    assert len(payloads) == 1
    assert json.loads(payloads[0].read_text())["content"] == LOOP
    payload_bytes = payloads[0].read_bytes()
    stored = _contents(engine)
    engine.ingest(host)
    engine.ingest(host)
    assert _contents(engine) == stored
    assert list((tmp_path / "home" / "payloads").glob("*.json")) == payloads
    assert payloads[0].read_bytes() == payload_bytes
    assert engine._ignored_message_count == ignored


def _steer_first():
    return [
        {"role": "user", "content": "Run the migration", "timestamp": 100.0},
        {"role": "assistant", "content": "", "timestamp": 101.0, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"output": "migrated", "exit_code": 0}',
         "timestamp": 102.0},
        {"role": "assistant", "content": "Migration done.", "timestamp": 103.0},
    ]


@pytest.mark.parametrize("pattern", [False, True], ids=["no_pattern", "pattern"])
def test_633_steer_before_cursor(engine, pattern):
    if pattern:
        _pattern(engine)
    host = _steer_first()
    engine.ingest(host)
    steer = {"role": "user", "content": f"{MARK} synthetic inserted steer", "display_kind": "steer"}
    host.insert(3, steer)
    host += [{"role": "user", "content": "now deploy it", "timestamp": 200.0},
             {"role": "assistant", "content": "Deploying.", "timestamp": 201.0}]
    engine.ingest(host)
    expected = [row["content"] for row in _steer_first()]
    expected += ([] if pattern else [steer["content"]])
    expected += ["now deploy it", "Deploying."]
    assert _contents(engine) == expected
    engine.ingest(host)
    assert _contents(engine) == expected


def test_ordinary_rewind_preserves_active_and_stored_bytes(engine):
    host = _first()
    engine._ingest_messages(host)
    host[0]["content"] = "Revised migration prompt\n  keep whitespace  "
    host += [{"role": "user", "content": "now deploy it", "timestamp": 200.0},
             {"role": "assistant", "content": "Deploying.", "timestamp": 201.0}]
    expected_active = json.dumps(host, ensure_ascii=False, sort_keys=True).encode()
    active = engine._ingest_messages(host)
    assert json.dumps(active, ensure_ascii=False, sort_keys=True).encode() == expected_active
    expected_stored = [row["content"] for row in _first()] + [row["content"] for row in host if row is not host[1]]
    assert json.dumps(_contents(engine)).encode() == json.dumps(expected_stored).encode()
    stored = deepcopy(_rows(engine))
    active = engine._ingest_messages(host)
    assert json.dumps(active, ensure_ascii=False, sort_keys=True).encode() == expected_active
    assert _rows(engine) == stored


def test_rescan_preserves_anchored_replay_and_reconcile_input(engine, monkeypatch):
    host = _first()
    host.insert(1, {"role": "user", "content": f"{MARK} already stored", "timestamp": 100.5})
    engine.ingest(host)
    _pattern(engine)
    host[0]["content"] = "Revised migration prompt"
    host.append({"role": "assistant", "content": "Deploying.", "timestamp": 201.0})
    build = engine._build_ingest_replay
    prematch = engine._identity_anchor_prematch
    capture = engine._host_uid_capture
    observed = {}

    def record_build(**kwargs):
        result = build(**kwargs)
        observed.setdefault("builds", []).append((kwargs["cursor"], deepcopy(result)))
        return result

    def record_prematch(messages, identity_messages, *args, **kwargs):
        observed["reconcile"] = identity_messages
        observed["reconcile_before"] = deepcopy(identity_messages)
        return prematch(messages, identity_messages, *args, **kwargs)

    def record_capture(messages, reconcile_messages, *args, **kwargs):
        assert reconcile_messages is observed["reconcile"]
        assert reconcile_messages == observed["reconcile_before"]
        return capture(messages, reconcile_messages, *args, **kwargs)

    monkeypatch.setattr(engine, "_build_ingest_replay", record_build)
    monkeypatch.setattr(engine, "_identity_anchor_prematch", record_prematch)
    monkeypatch.setattr(engine, "_host_uid_capture", record_capture)
    active = engine._ingest_messages(host)
    # The newly configured pattern must not rewrite an already explained row.
    assert active == host
    assert _contents(engine).count(host[1]["content"]) == 1
    if len(observed["builds"]) == 2:  # Also a base-green preservation control.
        (scan_start, first), (rewound, second) = observed["builds"]
        assert rewound < scan_start
        assert first[1][1] is False and second[1][1] is True
        assert second[2][1] != active[1]
        assert active[scan_start:] == first[2][scan_start:]


def test_faulted_prematch_does_not_rescan_and_retry_recovers(engine, monkeypatch):
    host = _first()
    engine.ingest(host)
    host[0]["content"] = "Revised migration prompt"
    host.append({"role": "assistant", "content": "Deploying.", "timestamp": 201.0})
    build = engine._build_ingest_replay
    calls = []

    def record_build(**kwargs):
        calls.append(kwargs["cursor"])
        return build(**kwargs)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic prematch fault")

    monkeypatch.setattr(engine, "_build_ingest_replay", record_build)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_identity_anchor_prematch", fail)
        engine.ingest(host)
    assert calls == [2]
    assert "Revised migration prompt" not in _contents(engine)
    engine.ingest(host)
    assert _contents(engine).count("Revised migration prompt") == 1
    assert _contents(engine).count("Deploying.") == 1


def test_steady_and_reconcile_paths_do_not_rescan(engine, monkeypatch):
    build = engine._build_ingest_replay
    calls = []

    def record_build(**kwargs):
        result = build(**kwargs)
        calls.append((kwargs["cursor"], result[0]))
        return result

    monkeypatch.setattr(engine, "_build_ingest_replay", record_build)
    host = _first()
    engine.ingest(host)
    engine.ingest(host)
    engine._ingest_cursor_needs_reconcile = True
    engine.ingest(host)
    assert calls == [(0, 0), (2, 2), (2, 0)]
    assert _contents(engine) == [row["content"] for row in host]
