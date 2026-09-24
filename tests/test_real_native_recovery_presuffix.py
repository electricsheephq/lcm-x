"""Real-Hermes proof that native recovery cannot rewrite LCM-X's fresh tail (#482)."""

from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_PROBE = textwrap.dedent(
    r"""
    import copy, importlib, json, os, sys, types
    from pathlib import Path

    plugin_root = Path(sys.argv[1])
    home = Path(sys.argv[2])
    package = types.ModuleType("hermes_lcm")
    package.__path__ = [str(plugin_root)]
    package.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = package

    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    import agent.context_compressor as cc

    CC = cc.ContextCompressor
    original_generate = CC._generate_summary

    def ident(engine, message):
        return engine._message_replay_identity(message)

    def tool_round(i, body):
        return [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call-{i}", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": f"call-{i}",
             "name": "read_file", "content": body},
        ]

    def history(n=40):
        return [{
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"old turn {i}: " + "synthetic history " * 90,
        } for i in range(n)]

    def fixture(duplicate=False):
        rows = history()
        for i in range(12):
            body = "Repeated output. " * 30 if duplicate else f"Unique output {i}. " * 30
            rows += tool_round(i, body)
        return rows

    def user_in_suffix():
        rows = history() + [{"role": "user", "content": "start the long run"}]
        for i in range(30):
            rows += tool_round(i, f"long run output {i} " * 120)
        rows += [
            {"role": "assistant", "content": "Long run done."},
            {"role": "user", "content": "now do the follow-up"},
        ]
        for i in range(30, 36):
            rows += tool_round(i, f"follow-up output {i} " * 120)
        return rows

    def handoff_in_suffix():
        rows = fixture()
        summary = {
            "role": "user",
            "content": getattr(cc, "SUMMARY_PREFIX", "[CONTEXT COMPACTION]")
            + "\nPrior native summary.\n\n" + getattr(cc, "_SUMMARY_END_MARKER", ""),
        }
        metadata_key = getattr(cc, "COMPRESSED_SUMMARY_METADATA_KEY", None)
        if metadata_key:
            summary[metadata_key] = True
        rows.insert(len(rows) - 12, summary)
        rows.insert(len(rows) - 12, {"role": "assistant", "content": "Acknowledged."})
        return rows

    def images_in_suffix():
        rows = fixture()
        def image(text):
            return {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + "A" * 4000,
                }},
            ]}
        rows.insert(len(rows) - 14, image("first screenshot"))
        rows.insert(len(rows) - 14, {"role": "assistant", "content": "Seen."})
        rows.insert(len(rows) - 4, image("second screenshot"))
        rows.insert(len(rows) - 4, {"role": "assistant", "content": "Seen again."})
        return rows

    def make_engine(name):
        cfg = LCMConfig(
            database_path=str(home / f"{name}.db"), native_recovery=True,
            fresh_tail_count=24, fresh_tail_max_tokens=24000,
            embeddings_enabled=False, temporal_rollups_enabled=False,
            empty_lifecycle_gc_enabled=False,
        )
        engine = LCMEngine(config=cfg, hermes_home=str(home))
        engine.on_session_start(name, conversation_id=name)
        engine.model = "gpt-6-astra"
        engine.provider = "openai-codex"
        engine.api_mode = "codex_responses"
        engine.context_length = 272000
        engine.threshold_tokens = 204000
        engine._compression_cancelled_check = lambda: False
        return engine

    def run_case(name, rows):
        engine = make_engine(name)
        try:
            boundary = engine._fresh_tail_boundary(rows)
            source_tail = copy.deepcopy(rows[boundary.start:])
            for row in rows[boundary.start:]:
                row[getattr(cc, "_DB_PERSISTED_MARKER")] = True
            before_messages = engine._store._conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            result = engine._compress_native_recovery(rows, current_tokens=250000)
            after_messages = engine._store._conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            tail = result[-len(source_tail):]
            return {
                "status": engine.last_compression_status,
                "smaller": len(result) < len(rows),
                "tail_equal": [ident(engine, row) for row in tail]
                == [ident(engine, row) for row in source_tail],
                "tail_marked": all(row.get(getattr(cc, "_COMPACTION_TAIL_MARKER")) is True for row in tail),
                "persisted_removed": all(getattr(cc, "_DB_PERSISTED_MARKER") not in row for row in tail),
                "store_unchanged": before_messages == after_messages,
            }
        finally:
            engine.shutdown()

    CC._generate_summary = lambda self, turns, **kwargs: f"Synthetic summary of {len(turns)} turns."
    cases = {
        "fixture_unique": fixture(False),
        "fixture_duplicate": fixture(True),
        "user_in_suffix": user_in_suffix(),
        "handoff_in_suffix": handoff_in_suffix(),
        "images_in_suffix": images_in_suffix(),
    }
    results = {name: run_case(name, rows) for name, rows in cases.items()}

    CC._generate_summary = lambda self, turns, **kwargs: None
    failure_engine = make_engine("generation_failure")
    failure_rows = fixture()
    failure_result = failure_engine._compress_native_recovery(failure_rows, current_tokens=250000)
    results["generation_failure"] = {
        "unchanged": failure_result is failure_rows,
        "reason": getattr(failure_engine, "_last_native_recovery_rejection", None),
    }
    failure_engine.shutdown()

    cancelled = {"value": False}
    def cancel_after_summary(self, turns, **kwargs):
        cancelled["value"] = True
        return "Synthetic summary before cancellation."
    CC._generate_summary = cancel_after_summary
    cancel_engine = make_engine("cancelled")
    cancel_engine._compression_cancelled_check = lambda: cancelled["value"]
    cancel_rows = fixture()
    cancel_result = cancel_engine._compress_native_recovery(cancel_rows, current_tokens=250000)
    results["cancelled"] = {
        "unchanged": cancel_result is cancel_rows,
        "reason": getattr(cancel_engine, "_last_native_recovery_rejection", None),
    }
    cancel_engine.shutdown()
    CC._generate_summary = original_generate
    print(json.dumps(results, sort_keys=True))
    """
)


def _hermes_python() -> tuple[str, str] | None:
    python = os.environ.get("LCM_REAL_HERMES_PYTHON")
    src = os.environ.get("LCM_REAL_HERMES_SRC", "")
    if python:
        return python, src
    if importlib.util.find_spec("agent.context_compressor") is not None:
        return sys.executable, src
    return None


HERMES = _hermes_python()
pytestmark = pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")


def test_real_native_recovery_preserves_protected_suffix(tmp_path):
    python, src = HERMES
    plugin_root = Path(os.environ.get("LCM_PLUGIN_ROOT", REPO_ROOT))
    home = tmp_path / "hermes-home"
    home.mkdir()
    completed = subprocess.run(
        [python, "-c", _PROBE, str(plugin_root), str(home)],
        cwd=src or None,
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "HERMES_HOME": str(home),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    for name in (
        "fixture_unique", "fixture_duplicate", "user_in_suffix",
        "handoff_in_suffix", "images_in_suffix",
    ):
        assert result[name] == {
            "persisted_removed": True,
            "smaller": True,
            "status": "host_native",
            "store_unchanged": True,
            "tail_equal": True,
            "tail_marked": True,
        }, (name, result[name])
    assert result["generation_failure"] == {
        "reason": "native_aborted", "unchanged": True,
    }
    assert result["cancelled"] == {"reason": "cancelled", "unchanged": True}
