"""Shared fixtures; old code is always imported from an archived, pinned tree."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

from tests.compat import codec_adapter

# Same pins as test_host_uid_shadow.ROLLBACK_READERS (CI fetches these commits).
READERS = {
    "v0.25.0-rc1": "c36b46e3a7b1bff23cfa45fa117565ab8624441d",
    "v0.25.1": "f47b55e031b507b424ff5f480d8f2a80d358f1f0",
    "v0.26.2": "6800cf45ad05280e6cf6fa9e1ad96c62b7132646",
}
FORMS = ("mark_only", "head_mark", "head_mark_tail")
VARIANTS = ("text", "structured", "tool_call", "tool_result", "named_tool_result")
CASES = tuple(f"{form}-{variant}" for form in FORMS for variant in VARIANTS) + (
    "bounded_calls-empty", "bounded_calls-text", "bounded_calls-structured",
    "stored_calls-empty", "stored_calls-text", "stored_calls-structured",
    "head_mark-user", "head_mark_tail-user", "followers-text", "keep_reply-text",
)


def config_for(directory, **overrides):
    from hermes_lcm.config import LCMConfig

    return LCMConfig(database_path=str(directory / "lcm.db"), embeddings_enabled=False,
                     temporal_rollups_enabled=False,
                     large_output_externalization_path=str(directory / "payloads"), **overrides)


def open_engine(directory):
    from hermes_lcm.engine import LCMEngine

    engine = LCMEngine(config=config_for(directory), hermes_home=str(directory / "home"))
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    return engine


def call(arguments):
    return {"id": "call", "type": "function", "function": {"name": "tool", "arguments": arguments}}


def emit_case(engine, case):
    """Ingest real source rows, then exercise emission (not just the codec helper)."""
    from hermes_lcm.message_content import normalize_content_value

    if case.startswith(("followers-", "keep_reply-")):
        return emit_fit_case(engine, case)
    form, variant = case.split("-")
    text = "word " * (1200 if form == "head_mark_tail" else 400)
    content = [{"type": "text", "text": text}] if variant == "structured" else text
    source = {"role": "assistant", "content": content, "timestamp": 2.0}
    view = [{"role": "user", "content": "Run the requested task.", "timestamp": 1.0}, source]
    if variant == "tool_call" or form in ("bounded_calls", "stored_calls"):
        arguments = "argument " * 1500
        if form in ("bounded_calls", "stored_calls"):
            content = "" if variant == "empty" else "Calling the tool."
            source["content"] = [{"type": "text", "text": content}] if variant == "structured" else content
        if form == "stored_calls":
            image = base64.b64encode(b"synthetic image " * 1000).decode("ascii")
            arguments = json.dumps({"image": "data:image/png;base64," + image})
        source["tool_calls"] = [call(arguments)]
        view.append({"role": "tool", "content": "done", "tool_call_id": "call", "timestamp": 3.0})
    elif variant in ("tool_result", "named_tool_result"):
        source.update(role="tool", tool_call_id="call", name="tool", timestamp=3.0)
        if variant == "tool_result":
            source["tool_name"] = "tool"
        view.insert(1, {"role": "assistant", "content": "", "tool_calls": [call("{}")], "timestamp": 2.0})
    if variant == "user":
        source["role"] = "user"
        view = [source]
    engine.ingest(view)
    before = engine._store.get_session_messages("S")
    row = next(row for row in before if row["role"] == source["role"] and row["observed_at"] == source["timestamp"])
    text = normalize_content_value(row["content"]) or ""
    fields = codec_adapter.projected_fields(engine, row, source, stored_calls=form == "stored_calls")
    expected = {**source, **{key: value for key, value in fields.items() if value is not None}}
    limit = 0 if form == "mark_only" else codec_adapter.measure(engine, [expected])
    projected = codec_adapter.project(engine, source, row["store_id"], limit)
    assert projected != source, case
    projected_text = normalize_content_value(projected["content"]) or ""
    if form == "mark_only":
        assert codec_adapter.is_mark_only(projected_text), case
    elif form == "head_mark":
        assert projected_text.startswith(text[:1200] + "\n...\n") and projected_text.endswith("1200/600)]"), case
    elif form == "head_mark_tail":
        assert projected_text.startswith(text[:1200] + "\n...\n") and projected_text.endswith(text[-600:]), case
    elif form == "bounded_calls":
        assert projected["content"] == source["content"]
        assert "lcm_survival_fit" in projected["tool_calls"][0]["function"]["arguments"]
    else:
        assert "[LCM" in projected_text and projected["tool_calls"] == row["tool_calls"]
        if variant == "empty":
            assert codec_adapter.is_mark_only(projected_text), "#959 mark-only content plus stored calls"
    emitted = [projected if message is source else message for message in view]
    return json.loads(json.dumps(emitted)), before


def emit_fit_case(engine, case):
    """Exercise user projection/followers and the separate keep_reply emission site."""
    if case == "followers-text":
        view = [{"role": "user", "content": "request " * 1200, "timestamp": 1.0},
                {"role": "assistant", "content": "", "tool_calls": [call("{}")]},
                {"role": "tool", "content": "done", "tool_call_id": "call", "tool_name": "tool"},
                {"role": "assistant", "content": "Task complete."}]
        limit = 1000
    else:
        view = [{"role": "user", "content": "old request " * 1600, "timestamp": 1.0},
                {"role": "assistant", "content": "reply " * 1200, "timestamp": 2.0},
                {"role": "user", "content": "Next request.", "timestamp": 3.0}]
        limit = 650
    engine.ingest(view)
    before = engine._store.get_session_messages("S")
    emitted = codec_adapter.fit(engine, view, limit)
    assert codec_adapter.measure(engine, emitted) <= limit
    if case == "followers-text":
        assert emitted[0] != view[0] and emitted[0]["role"] == "user"
        assert emitted[1:] == view[1:] and all("timestamp" not in row for row in emitted[1:])
    else:
        assert [row["role"] for row in emitted] == ["assistant", "user"]
        assert emitted[0]["content"].startswith(view[1]["content"][:1200] + "\n...\n")
        assert emitted[0]["content"].endswith(view[1]["content"][-600:])
        assert emitted[1] == view[-1]
    assert engine._store.get_session_messages("S") == before
    return json.loads(json.dumps(emitted)), before


OLD_BOOTSTRAP = r'''
import importlib.util, json, sys
from pathlib import Path
old, db, home, host_root, mode, data = sys.argv[1:]
sys.path.insert(0, host_root)
spec = importlib.util.spec_from_file_location("hermes_lcm", old + "/__init__.py", submodule_search_locations=[old])
sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
from hermes_lcm import engine as engine_mod, store as store_mod, tokens, survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
assert Path(store_mod.__file__).resolve().parent == Path(old).resolve()
assert Path(engine_mod.__file__).resolve().parent == Path(old).resolve()
''' + codec_adapter.OLD_RUNTIME_SETUP + r'''
engine_mod.summarize_with_escalation = lambda **kwargs: ("Synthetic summary preserving the task and tool result.", 1)
options = dict(database_path=db, embeddings_enabled=False, temporal_rollups_enabled=False,
               large_output_externalization_path=str(Path(home).parent / "payloads"))
'''


def run_old(old, directory, host_root, script, mode="read", data=""):
    env = {key: value for key, value in os.environ.items() if not key.startswith("LCM_")}
    env.update(HERMES_HOME=str(directory / "home"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", OLD_BOOTSTRAP + script, str(old), str(directory / "lcm.db"),
         str(directory / "home"), str(host_root), mode, str(data)],
        cwd=directory, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr[-6000:]
    return json.loads(result.stdout)


# Frozen protocol fixture: add keys only; changed emissions require a new golden file.
GOLDEN = Path(__file__).with_name("golden") / "projections.json"
