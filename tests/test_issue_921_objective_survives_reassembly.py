"""#921: the sole user objective survives another assembly in the same turn."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_host_uid_shadow import ROLLBACK_READERS, _skip_or_fail_in_ci


PREFIX = "[Current user objective preserved from compacted history]"
SEPARATOR = "\n\n---\n\n"
PROMPT = "Keep this sole request verbatim: café, 日本語.\n  indented detail\t\n" + "plan " * 3670


@pytest.fixture
def engine(tmp_path, monkeypatch):
    # Use the supported offline estimator: approximately 4.6k/20k tokens,
    # independent of an optional tokenizer or a tokenizer download.
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(
        engine_module, "summarize_with_escalation",
        lambda **kwargs: ("Synthetic tool-round summary; no verbatim user text.", 1),
    )
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=4,
        fresh_tail_max_tokens=2000,
        leaf_chunk_tokens=8000,
        threshold_full_sweep_enabled=True,
    ), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("S", platform="cli", context_length=128_000)
    try:
        yield instance
    finally:
        instance.shutdown()
        tokens._count_tokens_cached.cache_clear()


def _round(index):
    call_id = f"call_{index}"
    return [
        {"role": "assistant", "content": f"Read synthetic item {index}.", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }], "timestamp": 10.0 + index * 2},
        {"role": "tool", "tool_call_id": call_id,
         "content": f"Synthetic output {index}: " + "data " * 16_000,
         "timestamp": 11.0 + index * 2},
    ]


def _first_assembly(engine):
    messages = [{"role": "user", "content": PROMPT, "timestamp": 1.0}]
    for index in range(5):
        messages.extend(_round(index))
    assert 4500 < tokens.count_tokens(PROMPT) < 4700
    assert 20_000 < tokens.count_tokens(messages[2]["content"]) < 20_100
    first = engine.compress(messages)
    assert first[0]["content"].startswith(PREFIX + "\n" + PROMPT)
    assert first[0]["content"].partition(SEPARATOR)[2]  # also carries DAG summaries
    assert not any(row.get("content") == PROMPT for row in first[1:])
    return first


@pytest.mark.parametrize("more_rounds", [0, 2])
def test_second_same_turn_compress_keeps_objective(engine, more_rounds):
    first = _first_assembly(engine)
    continued = list(first)
    for index in range(5, 5 + more_rounds):
        continued.extend(_round(index))
    second = engine.compress(continued)
    assert second[0]["content"].startswith(PREFIX)
    assert PROMPT in second[0]["content"]
    assert second[0]["content"].partition(SEPARATOR)[0] == first[0]["content"].partition(SEPARATOR)[0]
    assert second[0]["content"].count(PREFIX) == 1
    assert second[-2:] == continued[-2:]  # preserve the active call/result pair


def test_newer_real_user_wins_over_carried_scaffold(engine):
    first = _first_assembly(engine)
    newer = {"role": "user", "content": "The newer request is the current objective.", "timestamp": 100.0}
    assert engine._latest_user_context_anchor([*first, newer], []) == PREFIX + "\n" + newer["content"]
    assert engine._latest_user_context_anchor([*first, newer], [newer]) is None
    second = engine.compress([*first, newer, *_round(50), *_round(51)])
    assert second[0]["content"].partition(SEPARATOR)[0] == PREFIX + "\n" + newer["content"]
    assert PROMPT not in second[0]["content"]


def test_assistant_only_tail_does_not_invent_objective(engine):
    messages = [{"role": "assistant", "content": "An assistant-only continuation."}]
    assert engine._latest_user_context_anchor(messages, []) is None
    assert engine.compress(messages) == messages


def test_carried_objective_part_is_byte_identical(engine):
    objective = " \t" + PREFIX + "\n  preserve whitespace\t\n日本語 café  "
    scaffold = {"role": "user", "content": objective + SEPARATOR + "old summary" + SEPARATOR + "another summary"}
    stale = {"role": "user", "content": "An older request must not win."}
    assert engine._latest_user_context_anchor([stale, scaffold], []) == objective
    assert engine._latest_user_context_anchor([{"role": "user", "content": objective}], []) == objective


_OLD_READER = r'''
import importlib.util, json, sys
old, db, root, host_file = sys.argv[1:]
sys.path.insert(0, root)  # git-ignored ContextEngine import stub, not a host
spec = importlib.util.spec_from_file_location("hermes_lcm", old + "/__init__.py", submodule_search_locations=[old])
sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
with open(host_file) as stream:
    host = json.load(stream)
engine = LCMEngine(config=LCMConfig(database_path=db))
try:
    engine.on_session_start("S", platform="cli", context_length=128_000)
    before = engine._store.get_session_messages("S")
    assert engine._legacy_objective_head(host[0]) is None
    merged = {**host[0], "content": host[0]["content"] + "\n\nnew user suffix"}
    assert engine._legacy_objective_head(merged) == host[0]["content"]
    engine.ingest(host)
    engine.ingest(host)
    after = engine._store.get_session_messages("S")
    assert [(r["role"], r["content"]) for r in after] == [(r["role"], r["content"]) for r in before]
    assert engine._store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print("ROLLBACK_READER_OK v0.25.1 objective recognized; replay adds no rows")
finally:
    engine.shutdown()
'''


def test_v0251_rollback_reader_accepts_reassembled_objective(engine, tmp_path):
    root = Path(__file__).resolve().parents[1]
    commit = ROLLBACK_READERS["v0.25.1"]
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
        _skip_or_fail_in_ci("the v0.25.1 rollback reader commit is not available")
    emitted = engine.compress(_first_assembly(engine))
    engine.shutdown()
    archive = subprocess.run(["git", "-C", str(root), "archive", commit], check=True, capture_output=True).stdout
    old = tmp_path / "old-reader"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter="data")
    host_file = tmp_path / "emitted.json"
    host_file.write_text(json.dumps(emitted))
    result = subprocess.run(
        [sys.executable, "-B", "-c", _OLD_READER, str(old), str(tmp_path / "lcm.db"), str(root), str(host_file)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "ROLLBACK_READER_OK v0.25.1" in result.stdout
