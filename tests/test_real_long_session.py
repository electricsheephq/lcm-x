"""300 real Hermes turns, cold resume, deterministic provider, no external network."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.real_long_session_probe import assert_ordered_tail, assert_restored_history, canonical

REPO_ROOT = Path(__file__).resolve().parents[1]


def _hermes_python() -> tuple[str, str] | None:
    python = os.environ.get("LCM_REAL_HERMES_PYTHON")
    src = os.environ.get("LCM_REAL_HERMES_SRC", "")
    if python:
        return python, src
    if importlib.util.find_spec("hermes_cli") is not None:
        try:
            if importlib.util.find_spec("hermes_cli.plugins") is not None:
                return sys.executable, src
        except (ImportError, ValueError):
            return None
    return None


HERMES = _hermes_python()


@pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")
def test_real_long_session(tmp_path: Path) -> None:
    assert HERMES is not None
    python, src = HERMES
    out = Path(os.environ.get("LCM_LONG_SESSION_ARTIFACT", str(tmp_path / "long-session.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins/hermes-lcm-x").symlink_to(REPO_ROOT)
    user_home = tmp_path / "home"
    files = user_home / "long-session"
    files.mkdir(parents=True)
    for turn in range(10, 301, 10):
        for k in range(2 if turn % 20 == 0 else 1):
            text = "".join(f"record {turn}-{k} line {i}: amber birch cedar delta elm fjord\n"
                           for i in range(120 if turn % 20 == 0 else 3))
            (files / f"record-{turn}-{k}.txt").write_text(text)
    env = {"HOME": str(user_home), "PATH": "/usr/bin:/bin", "HERMES_HOME": str(home),
           "PYTHONDONTWRITEBYTECODE": "1", "HERMES_DISABLE_LAZY_INSTALLS": "1",
           "LCM_SUMMARY_MODEL": "rel/lcm-summary", "LCM_SUMMARY_SPEND_MAX_CALLS": "100000",
           "LCM_CONTEXT_THRESHOLD": "0.5", "LCM_FRESH_TAIL_COUNT": "8", "LCM_FRESH_TAIL_MAX_TOKENS": "0",
           "LCM_LEAF_CHUNK_TOKENS": "2000", "LCM_CONDENSATION_FANIN": "4",
           "LCM_EMBEDDINGS_ENABLED": "false", "LCM_RERANK_ENABLED": "false", "LCM_NATIVE_RECOVERY": "false",
           "LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS": "4000",
           "LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH": str(home / "lcm_externalized"),
           "FASTEMBED_CACHE_PATH": str(tmp_path / "empty-cache")}
    phases = []
    for first, last in ((1, 150), (151, 300)):
        completed = subprocess.run([python, str(REPO_ROOT / "tests/real_long_session_probe.py"),
                                    str(first), str(last), str(evidence)],
                                   cwd=src or None, env=env, capture_output=True, text=True, timeout=175)
        (evidence / f"host-{first}.log").write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-5000:]
        phases.append(json.loads((evidence / f"phase-{first}.json").read_text()))
        if first == 1:
            terminal = json.loads((evidence / "terminal-history.json").read_text())
            assert terminal["history"] and terminal["tail"] == terminal["history"][-8:]
    combined = {key: [row for phase in phases for row in phase[key]]
                for key in ("checkpoints", "turns", "compactions")}
    out.write_text(json.dumps(combined, indent=2) + "\n")
    assert len(combined["turns"]) == 300, "I1 exactly-once: incomplete scenario"
    assert sum(c["status"] == "compacted" for c in combined["compactions"]) >= 5, "I4 DAG: fewer than five compactions"
    assert all(c["recalled"] > 0 for c in combined["checkpoints"]), "I3 recall: no aged facts exercised"


def test_ordered_tail_rejects_reordering_and_substrings() -> None:
    tail = [{"role": "user", "content": "inventory"}, {"role": "assistant", "content": "noted"}]
    protected = [canonical(m) for m in tail]
    assert_ordered_tail(tail, protected)
    with pytest.raises(AssertionError, match="ordered fresh tail changed"):
        assert_ordered_tail(list(reversed(tail)), protected)
    with pytest.raises(AssertionError, match="ordered fresh tail changed"):
        assert_ordered_tail([tail[0], {**tail[1], "content": "noted extra"}], protected)
    carrier = [{**tail[0], "content": "summary\n\ninventory"}, tail[1]]
    assert_ordered_tail(carrier, protected, carrier_remainder="inventory")
    with pytest.raises(AssertionError, match="ordered fresh tail changed"):
        assert_ordered_tail(carrier, protected)
    with pytest.raises(AssertionError, match="ordered fresh tail changed"):
        assert_ordered_tail([tail[0], {**tail[1], "content": "summary\n\nnoted"}], protected,
                            carrier_remainder="inventory")


def test_ordered_tail_canonicalizes_tool_argument_json() -> None:
    def call(arguments: str) -> dict:
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "read_20_0", "type": "function", "function": {"name": "read_file", "arguments": arguments}}]}

    protected = [canonical(call('{"path": "record.txt"}'))]
    assert_ordered_tail([call('{"path":"record.txt"}')], protected)
    with pytest.raises(AssertionError, match="ordered fresh tail changed"):
        assert_ordered_tail([call('{"path":"other.txt"}')], protected)


def test_restored_history_rejects_missing_or_reordered_rows() -> None:
    history = [{"role": "user", "content": "inventory"}, {"role": "assistant", "content": "noted"}]
    terminal = [canonical(m) for m in history]
    assert_restored_history(history, terminal)
    with pytest.raises(AssertionError, match="restored history changed"):
        assert_restored_history(history[:-1], terminal)
    with pytest.raises(AssertionError, match="restored history changed"):
        assert_restored_history(list(reversed(history)), terminal)
