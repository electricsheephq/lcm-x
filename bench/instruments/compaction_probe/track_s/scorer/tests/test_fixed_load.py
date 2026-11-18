"""Synthetic replay and share checks for the declared fixed prompt load."""
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import fixtures as fx
import pytest
import summary_share
from test_eval2_harness import TRACK, module

FIXED_LOAD = 18_768
SUMMARY = "[Recent Summary (d0, node 1)]\nSynthetic memory\n[Expand for details: node 1]"
MODES = {
    "probes": [],
    "full-stream": ["--full-stream"],
    "prefix60k": ["--prefix60k"],
    "checkpoints": ["--checkpoints", "2,4"],
    "unsupported": [],
}
# SHA-256 over the ordered file names and file-byte digests from main at 4dd077a6.
MAIN_OUTPUTS = {
    "probes": "83d0cbfe444133f8253107806a898c6f68e2f4ac35657c36ee77abfbee5a5094",
    "full-stream": "28eacd455effcdc048f88f05108d04ced440cb84f9f582a3b680e39dd2f9b528",
    "prefix60k": "e34794f1e2b3be3a118e01c23a36e24fe31c327a6ae8b3edd490a1d31d73e833",
    "checkpoints": "8519fc355c7503413ed150e000bbac0900b53d467eb8c9fecd62de0ef34e61a6",
    "unsupported": "f2425c246e8dde4d7abdafb0c7a89485372daa05cba784cce6851425cf788920",
}


def view_tokens(view):
    return sum(len(message["content"]) for message in view)


class FakeEngine:
    threshold_tokens = 250
    last_compression_status = "compacted"
    last_compression_noop_reason = None

    def __init__(self, db, sid):
        self.db, self.sid = db, sid
        self.gates, self.calls = [], []
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.executescript("""
                CREATE TABLE messages (store_id, role, content, session_id);
                CREATE TABLE summary_nodes (node_id, depth, source_type, source_ids, token_count,
                                            source_token_count, summary, session_id);
                CREATE TABLE summary_node_provenance (node_id, escalation_level);
            """)

    def ingest(self, view):
        pass

    def should_compress_preflight(self, view):
        self.gates.append(tuple(message["content"] for message in view))
        return view_tokens(view) >= self.threshold_tokens

    def should_compress(self, *args, **kwargs):
        pytest.fail("the replay must only use the preflight gate")

    def compress(self, view, *, current_tokens, force):
        self.calls.append((current_tokens, force))
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO summary_nodes VALUES (?, 0, 'messages', '[]', ?, ?, ?, ?)",
                         (len(self.calls), len(SUMMARY), view_tokens(view), SUMMARY, self.sid))
        return [dict(role="assistant", content=SUMMARY)]

    def shutdown(self):
        pass


@pytest.fixture
def replay(monkeypatch, tmp_path):
    runner = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    material = tmp_path / "seed-1"
    material.mkdir()
    rows = [dict(id=f"R{i}", turn=i, role=role, content=content) for i, (role, content) in enumerate([
        ("system", "Rules"), ("user", "x" * 120), ("assistant", "y" * 180),
        ("user", "z" * 110), ("assistant", "v" * 80), ("user", "w" * 90),
    ])]
    fx.wl(material / "transcript.jsonl", rows)
    fx.wj(material / "material.manifest.json", dict(continuity=[], decision_checkpoint=dict(row_index=4, tokens=500),
                                                  smoke_suffix=dict(action="force_compaction_after_row", row_id="R4")))
    fx.wj(material / "prefix60k.json", dict(freeze_row_index=4))
    fx.wj(material / "facts.json", [])

    tokens = SimpleNamespace(count_messages_tokens=view_tokens, count_tokens=len, _get_encoder=lambda: None)
    config_type = type("FakeConfig", (), {"from_env": staticmethod(lambda: None)})
    config = SimpleNamespace(LCMConfig=config_type)
    monkeypatch.setattr(runner.seam, "load_engine", lambda arm: SimpleNamespace(tokens=tokens, config=config))
    monkeypatch.setattr(runner.seam, "set_lane", lambda *args: None)
    monkeypatch.setattr(runner.seam, "WORKTREE", Path("synthetic-engine"))
    monkeypatch.setattr(runner.seam, "PINNED", "synthetic-head")
    monkeypatch.setattr(runner, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=lambda: 1000, monotonic=lambda: 0))
    monkeypatch.setattr(runner, "config_dict", lambda cfg: {})
    monkeypatch.setattr(runner, "schema_engine", lambda m: None)
    monkeypatch.setattr(runner, "probe", lambda *args: [])
    # SQLite headers include the library version. Keep receipt bytes portable.
    monkeypatch.setattr(runner, "sha", lambda path: "synthetic-db-hash" if path.exists() else None)
    monkeypatch.setattr(runner.A, "SECRET_ENV", {})
    monkeypatch.setattr(runner.A, "FLEET_EXCLUDED", {})
    monkeypatch.setattr(runner.A, "HARNESS_OVERRIDES", {})
    logger = SimpleNamespace(addHandler=lambda handler: None, setLevel=lambda level: None)
    monkeypatch.setattr(runner, "logging", SimpleNamespace(FileHandler=lambda path: None, getLogger=lambda name: logger, INFO=20))

    def run(mode, *extra):
        engines = []

        def new_engine(m, home, sid, ctx):
            engine = FakeEngine(home.parent / "lcm.db", sid)
            engines.append(engine)
            return engine

        arm = dict(name="fixture", kind="plain", env={}, note="synthetic arm",
                   unsupported="fixture arm unavailable" if mode == "unsupported" else None)
        monkeypatch.setattr(runner.A, "resolve", lambda name: arm)
        monkeypatch.setattr(runner, "new_engine", new_engine)
        root = Path("declared" if extra else "default")
        monkeypatch.setattr(runner, "RUNS", root)
        monkeypatch.setattr(sys, "argv", ["run_s_lcmx.py", "--seed", "1", "--arm", "fixture",
                                          "--lane", "fake", "--reader", "fake", *MODES[mode], *extra])
        assert runner.main() in (None, 0)
        out = root / "fixture/seed-1/1"
        files = {str(path.relative_to(out)): path.read_bytes() for path in sorted(out.rglob("*.json"))
                 if path.name == "summary.json" or path.parent.name == "continuity"}
        return files, engines[0] if engines else None

    return runner, run


@pytest.mark.parametrize("mode", MODES)
def test_zero_load_matches_main(replay, mode):
    _, run = replay
    files, engine = run(mode)
    record = json.loads(files["summary.json"])
    if engine is not None:
        assert record["events"] and any(forced for _, forced in engine.calls)
        assert engine.calls == [(event["tokens_at_trigger"], event["trigger"].startswith("forced"))
                                for event in record["events"]]
    receipts = [(name, hashlib.sha256(data).hexdigest()) for name, data in files.items()]
    assert hashlib.sha256(json.dumps(receipts).encode()).hexdigest() == MAIN_OUTPUTS[mode]


@pytest.mark.parametrize("mode", MODES)
def test_declared_load_keeps_view_counts_and_preflight(replay, mode):
    _, run = replay
    original, before = run(mode)
    declared, after = run(mode, "--fixed-load-tokens", str(FIXED_LOAD))
    assert declared.keys() == original.keys()
    for name, data in declared.items():
        if not name.endswith("summary.json"):
            assert data == original[name]
            continue
        record, baseline = json.loads(data), json.loads(original[name])
        assert record["fixed_load_tokens"] == FIXED_LOAD
        if mode == "unsupported":
            assert record["status"] == baseline["status"] == "UNSUPPORTED"
            assert "prompt_tokens" not in record
            continue
        assert record["final_context"] == baseline["final_context"]
        assert record["events"] == baseline["events"]  # includes both measured event token counts
        assert record.get("summary_tokens") == baseline.get("summary_tokens")
        assert record["prompt_tokens"] == baseline["final_context"]["tokens"] + FIXED_LOAD
    if before is not None:
        assert len(before.gates) > 1 and before.calls
        assert after.gates == before.gates
        assert after.calls == [(tokens + FIXED_LOAD, force) for tokens, force in before.calls]


def test_dry_run_reports_declared_load(replay, monkeypatch, capsys):
    runner, run = replay
    (runner.TS / "s2").mkdir(parents=True)
    monkeypatch.setattr(runner, "tempfile", SimpleNamespace(mkdtemp=lambda **kwargs: "dry-engine"))
    monkeypatch.setattr(runner, "shutil", SimpleNamespace(rmtree=lambda *args, **kwargs: None))
    run("probes", "--dry-run", "--fixed-load-tokens", str(FIXED_LOAD))
    assert f"fixed_load_tokens={FIXED_LOAD}" in capsys.readouterr().out


@pytest.mark.parametrize("load", [0, FIXED_LOAD], ids=["legacy-record", "declared-load"])
def test_share_uses_recorded_prompt_tokens(tmp_path, capsys, load):
    record = dict(arm={"name": "fixture"}, seed="seed-1", run="1", summary_tokens=25, prompt_tokens=100 + load)
    if load:
        record["fixed_load_tokens"] = load
    fx.wj(tmp_path / "summary.json", record)
    assert summary_share.main([str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fixed_load_tokens"] == load
    assert out["prompt_tokens"] == record["prompt_tokens"]
    assert out["share"] == record["summary_tokens"] / record["prompt_tokens"]


def test_negative_load_is_argparse_error(replay, capsys):
    _, run = replay
    with pytest.raises(SystemExit) as exc:
        run("probes", "--fixed-load-tokens", "-1")
    assert exc.value.code == 2
    assert "--fixed-load-tokens must be non-negative" in capsys.readouterr().err
