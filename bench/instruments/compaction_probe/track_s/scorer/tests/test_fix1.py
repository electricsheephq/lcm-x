"""Review probes converted to offline regressions."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import fixtures as fx
import pytest
import score_s as sc
from test_eval2_harness import TRACK, module


@pytest.fixture(scope="module")
def real_seed(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-seed")
    spec = importlib.util.spec_from_file_location("fix1_gen", TRACK.parent / "gen_material.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.generate(1, root, v4=True)
    return root


@pytest.mark.parametrize("reply,expected", [
    ("There is no record of {task} in the conversation.", False),
    ("No information about {task}.", False), ("No mention of {task}.", False), ("No.", False),
    ("I don't know.", False), ("No; do {repl}.", True),
    ("No - {task} was dropped; you no longer need to do {task}. Do {repl}.", True),
    ("No, there is no need to do {task}; {repl} replaced it.", True),
    ("No, {task} is not still to be done; it was replaced by {repl}.", True),
    ("It should not be done; the user cancelled it in favour of {repl}.", True),
    ("No. We shouldn't do {task}; do {repl}.", True),
    ("Yes, {task} is pending.", False), ("It is pending: run {task} next.", False),
    ("No; do {repl}. Also do {task}.", False),
])
def test_real_stale_task_contract(real_seed, reply, expected):
    man, facts = sc.jload(real_seed / "material.manifest.json"), sc.jload(real_seed / "facts.json")
    p = next(p for p in sc.jlines(real_seed / "lifecycle_probes.jsonl") if p["kind"] == "stale_task")
    task = next(r["task"] for r in man["lifecycle"] if r.get("resolution", {}).get("row_id") == p["row_id"])
    repl = p["answer"].split("do ", 1)[1].rstrip(".")
    assert sc.lifecycle_correct(p, reply.format(task=task, repl=repl), man, facts) is expected


def test_continuity_only_after_presentation():
    man = dict(continuity=[dict(id="late", value="late", row_index=45)])
    run = dict(events=[dict(row_index=i, cont={"late": False}, text="", label=str(i), timing="full-stream") for i in (10, 44, 45)])
    result = sc.continuity(run, man)
    assert len(result["grid"]) == 1 and result["grid"][0]["event"] == 2


@pytest.mark.parametrize("status", ["ERROR", "TIMEOUT"])
def test_reader_failure_is_incomplete_not_miss(tmp_path, status):
    mat, run = fx.material(tmp_path), fx.s2_run(tmp_path)
    rows = sc.jlines(run / "results.jsonl")
    for r in rows:
        if r["batch_id"] == rows[0]["batch_id"]:
            r.update(status=status, answer=None, answered=False)
    fx.wl(run / "results.jsonl", rows)
    scored = sc.score(mat, run, "LCMX-a")
    assert not scored["metrics"]["facts_kept"]["complete"]
    assert scored["reader_errors"] > 0
    assert all(p["class"] == "INCOMPLETE" for p in scored["probes"].values() if p.get("batch") == rows[0]["batch_id"])


def test_eval2_pins_required(monkeypatch, tmp_path):
    m = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    monkeypatch.delenv("S2_ARM_PINS", raising=False)
    for arm in ("L0", "L1", "L1-H", "L1-noptr", "H"):
        with pytest.raises(SystemExit, match="S2_ARM_PINS"):
            m.A.resolve(arm)


def test_latency_distributions_separate():
    a = sc.accounting(dict(reader_calls=[dict(calls=[dict(latency_s=1)])],
                           events=[dict(is_compaction=True, compress_wall_s=20, summariser_calls=[dict(latency_s=10)])]), {})
    assert a["reader_latency"]["p50"] == 1
    assert a["summariser_latency"]["p50"] == 10
    assert a["compaction_wall"]["p50"] == 20


def test_h_threshold_cooldown_system_and_returned_sections(monkeypatch, tmp_path):
    s2 = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "run_s_lcmx", s2)
    cc = SimpleNamespace(_LEAN_USER_MESSAGES_HEADING="## User", _LEAN_ANCHOR_HEADING="## Anchors")
    def compressor(**kwargs):
        assert kwargs["threshold_percent"] == .75
        return SimpleNamespace(threshold_tokens=48000, threshold_percent=.75, tail_mode="lean")
    cc.ContextCompressor = compressor
    monkeypatch.setitem(sys.modules, "agent", SimpleNamespace(context_compressor=cc))
    monkeypatch.setitem(sys.modules, "agent.context_compressor", cc)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", SimpleNamespace(estimate_messages_tokens_rough=lambda v: len(v), estimate_tokens_rough=len))
    monkeypatch.setenv("S2_H_SANDBOX", "1")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("HERMES_SRC", str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    fx.wj(tmp_path.parent / "HOST.json", dict(sha="fixture"))
    monkeypatch.setitem(sys.modules, "hosts", SimpleNamespace(verify=lambda *a: {"method": "fixture"}))
    h = module(TRACK / "s2/run_s_hermes_builtin.py", monkeypatch, tmp_path)
    engine = h.Engine(64000)
    calls = []
    def compress(view, **kwargs):
        assert view[0]["role"] == "system" and kwargs["bypass_cooldown"]
        calls.append(kwargs)
        engine.real.compression_count += 1
        return [view[0], dict(role="assistant", content="## User\nkept\n## Anchors\nid")]
    engine.real.compress, engine.real.compression_count, engine.real._previous_summary = compress, 0, ""
    run = h.Run.__new__(h.Run)
    run.engine, run.events, run.sysmsg = engine, [], dict(role="system", content="fixture")
    run.man, run.system, run.ntok = dict(continuity=[]), "fixture", len
    view = run.event([dict(role="user", content="fixture")], 48000, 45, dict(turn=1), False)
    assert calls and view[0]["role"] == "assistant"
    ev = run.events[0]
    assert ev["threshold_percent"] == .75 and ev["tail_mode"] == "lean"
    assert ev["user_section_bytes"] > 0 and ev["identifier_index_bytes"] > 0
    # Exercise H's checkpoint writer with a mismatched reader, never a real host/model.
    cp = dict(id="CP1", row_index=45)
    monkeypatch.setattr(s2.CP, "select", lambda *a: [cp])
    monkeypatch.setattr(s2, "jload", lambda p: dict(continuity=[], decision_checkpoint=dict(tokens=600000)))
    monkeypatch.setattr(s2, "jlines", lambda p: [])
    monkeypatch.setattr(h.arms, "resolve", lambda *a: dict(name="H"))
    monkeypatch.setattr(h.seam, "set_lane", lambda *a: None)
    monkeypatch.setattr(s2, "probe", lambda *a: [])
    from s2lib import fakes
    monkeypatch.setattr(fakes.Reader, "readback", dict(model="FAKE", pin_ok=False))
    monkeypatch.setattr(h.hosts, "real_hermes_dir", lambda: tmp_path / ".hermes", raising=False)
    denied = []
    real_open = Path.open
    def opened(path, *args, **kwargs):
        if path.name == ".eval2-write-denied-check":
            denied.append(path.parent)
            raise PermissionError("fixture denial")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    def make_run(*args):
        run.dir, run.seed, run.sdir = args[-1], "seed-1", tmp_path
        run.reader_calls = []
        def replay():
            out = run.dir / "cp-CP1"
            out.mkdir()
            run.snaps, run.events = [dict(dir=out, row=45, checkpoint=cp, view=[], n_events=0)], []
        run.replay = replay
        return run
    monkeypatch.setattr(h, "Run", make_run)
    monkeypatch.setattr(sys, "argv", ["h", "--seed", "1", "--lane", "fake", "--reader", "fake"])
    assert h.main() == 1
    summary = sc.jload(run.dir / "cp-CP1/summary.json")
    assert summary["status"] == "FAILED" and summary["threshold_percent"] == .75
    assert set(denied) == {s2.S2, h.src, tmp_path / ".hermes"}
    monkeypatch.setattr(h.hosts, "verify", lambda *a: (_ for _ in ()).throw(ValueError("tree hash mismatch")))
    with pytest.raises(ValueError, match="tree hash mismatch"):
        module(TRACK / "s2/run_s_hermes_builtin.py", monkeypatch, tmp_path)


def test_actual_horizons( monkeypatch, tmp_path):
    m = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    events = [dict(row_index=i, is_compaction=comp) for i, comp in ((0, True), (2, True), (3, False), (4, True), (9, True))]
    assert m.actual_horizon(events, 2, 8) == 2
    assert m.actual_horizon(events, 9, 8) == 0


def test_empty_carry_has_unknown_room(monkeypatch, tmp_path):
    m = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    assert m.carry_room({}) is None
    assert m.carry_room(None) is None
    assert m.carry_room({"empty_no_room": False}) is False


def test_material_timeout_scales_and_is_at_least_triple(monkeypatch, tmp_path):
    m = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    assert m.replay_timeout(600000, multiplier=3) >= 3 * m.replay_estimate(600000)
    assert m.replay_timeout(1200000) > m.replay_timeout(600000)
    with pytest.raises(ValueError):
        m.replay_timeout(600000, multiplier=2)
