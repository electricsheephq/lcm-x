"""#808: directory enumeration is shared by replay rows, with live miss fallback."""

import json
import os
from pathlib import Path

import pytest

from hermes_lcm import externalize
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


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
        # Count only the index's own listing: whether Path.glob reaches the
        # patched os.scandir differs between Python versions.
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


def test_i1_scoped_results_match_unscoped(config):
    content = "shared payload"
    write_payload(config, content, "a-kind", kind="other")
    write_payload(config, content, "b-role", role="assistant")
    write_payload(config, content, "c-call", tool_call_id="other")
    write_payload(config, content, "d-content", content="wrong content")
    first = write_payload(config, content, "e-other-session", session_id="other")
    preferred = write_payload(config, content, "f-current-session")
    # Adjacent digest segments and arbitrary names must match the old glob too.
    prefix = externalize._content_digest_prefix(content)
    write_payload(config, content, f"g_{prefix}")
    write_payload(config, "raw", kind="raw_payload", role="user", tool_call_id="")
    cases = [
        (content, dict(tool_call_id="call", session_id="s", role="tool")),
        (content, dict(tool_call_id="call", role="tool")),
        (content, dict(tool_call_id="call", session_id="absent")),
        (content, dict(tool_call_id="absent")),
        (content, dict(tool_call_id="call", kind="absent")),
        (content, dict(tool_call_id="call", role="user")),
        (content, dict(tool_call_id="call", kind=None)),
        ("raw", dict(kind="raw_payload", role="user")),
        ("missing", dict()),
        ("", dict()),
    ]
    baseline = [lookup(config, text, **fields) for text, fields in cases]
    assert baseline[0]["ref"] == preferred.name
    assert baseline[1]["ref"] == first.name
    assert baseline[2] is None
    with externalize.payload_lookup_scope():
        assert [lookup(config, text, **fields) for text, fields in cases] == baseline


def assembly_fixture(config, rows=6):
    engine = LCMEngine(config=config)
    engine._session_id = "s"
    tail = [{"role": "user", "content": "use tools"}]
    for i in range(rows):
        call_id = f"call-{i}"
        content = f"payload {i} " + "alpha beta gamma delta " * 20
        write_payload(config, content, str(i), tool_call_id=call_id)
        tail.extend([
            {"role": "assistant", "content": "running", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": call_id, "content": content},
        ])
    tail.extend([{"role": "user", "content": "fresh question"}, {"role": "assistant", "content": "fresh answer"}])
    return engine, tail


@pytest.mark.parametrize("persist", [False, True])
def test_i2_assembly_lists_payload_directory_once(config, monkeypatch, persist):
    engine, tail = assembly_fixture(config)
    counts = count_listings(monkeypatch, config)
    try:
        result = engine._assemble_context(None, tail, persist=persist)
    finally:
        engine.shutdown()
    stubs = [m for m in result if m.get("role") == "tool"]
    assert len(stubs) == 6
    assert all(m["content"].startswith("[Externalized tool output:") for m in stubs)
    assert counts == {"scandir": 1, "glob": 0}


@pytest.mark.parametrize("rejected_candidate", [False, True])
def test_i3_external_writer_found_on_miss(config, monkeypatch, rejected_candidate):
    write_payload(config, "warm")
    if rejected_candidate:
        write_payload(config, "new", "a-wrong-session", session_id="other")
    counts = count_listings(monkeypatch, config)
    with externalize.payload_lookup_scope():
        assert lookup(config, "warm", tool_call_id="call", session_id="s")
        # Bypass the process write helper and hence the active memo.
        path = write_payload(config, "new", "z-new")
        found = lookup(config, "new", tool_call_id="call", session_id="s")
        assert found["ref"] == path.name
        assert counts == {"scandir": 1, "glob": 1}
        assert lookup(config, "new", tool_call_id="call", session_id="s") == found
        assert counts == {"scandir": 1, "glob": 1}


@pytest.mark.parametrize("ingest", [False, True])
def test_i4_process_write_updates_memo(config, monkeypatch, ingest):
    write_payload(config, "warm")
    counts = count_listings(monkeypatch, config)
    with externalize.payload_lookup_scope():
        assert lookup(config, "warm", tool_call_id="call")
        if ingest:
            written = externalize.externalize_ingest_payload("new payload", role="user", session_id="s", config=config)
            fields = dict(kind="ingest_payload", role="user", session_id="s")
        else:
            written = externalize.maybe_externalize_payload("new payload", tool_call_id="new-call", session_id="s", config=config)
            fields = dict(kind="raw_payload", tool_call_id="new-call", session_id="s")
        before = dict(counts)
        assert lookup(config, "new payload", **fields)["ref"] == written["path"].name
        assert counts == before


def test_i5_unscoped_lookup_keeps_per_call_glob(config, monkeypatch):
    write_payload(config, "hit")
    counts = count_listings(monkeypatch, config)
    found = lookup(config, "hit", tool_call_id="call")
    assert found is not None
    for _ in range(4):
        assert lookup(config, "hit", tool_call_id="call") == found
    assert lookup(config, "miss") is None
    assert counts == {"scandir": 0, "glob": 6}


def test_nested_scope_reuses_index_and_resets_on_exception(config, monkeypatch):
    write_payload(config, "hit")
    counts = count_listings(monkeypatch, config)
    with pytest.raises(RuntimeError), externalize.payload_lookup_scope():
        assert lookup(config, "hit", tool_call_id="call")
        with externalize.payload_lookup_scope():
            assert lookup(config, "hit", tool_call_id="call")
        assert counts == {"scandir": 1, "glob": 0}
        raise RuntimeError("leave scope")
    assert lookup(config, "hit", tool_call_id="call")
    assert counts == {"scandir": 1, "glob": 1}
