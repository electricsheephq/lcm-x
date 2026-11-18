"""Synthetic coverage for the Track S env-file reader and summary share."""

import inspect
import json
import runpy
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TRACK = Path(__file__).resolve().parents[1] / "bench/instruments/compaction_probe/track_s"
PLAIN, OPEN = "LCMX-env-file", "LCMX-env-file-open"


@pytest.fixture
def load(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACK_S_OUT", str(tmp_path))
    monkeypatch.setenv("TRACK_S_MATERIAL", str(tmp_path / "material"))
    monkeypatch.delenv("S2_RUN_ROOT", raising=False)
    monkeypatch.syspath_prepend(str(TRACK / "scorer"))
    monkeypatch.setattr(sys, "path", sys.path.copy())

    def module(path):
        return SimpleNamespace(**runpy.run_path(str(TRACK / path)))

    return module


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    file = tmp_path / "arm.env"
    file.write_text(
        "LCM_EMBEDDINGS_ENABLED=true\nLCM_LEAF_CHUNK_TOKENS=2000\n"
        "LCM_DATABASE_PATH=ignored\nVOYAGE_API_KEY=synthetic-key\nOTHER=ignored\n"
    )
    monkeypatch.setenv("S2_ARM_ENV_FILE", str(file))


def test_resolve_open_env_file(load, env_file):
    arms = load("s2/s2lib/arms.py")
    expected_all = [name for base in arms.BASE_ARMS for name in (base, base + "-open")] + list(arms.CONTROLS)
    plain = arms.resolve(PLAIN)
    assert arms.SECRET_ENV == {"VOYAGE_API_KEY": "synthetic-key"}
    opened = arms.resolve(OPEN)
    assert opened == dict(plain, name=OPEN, open=True, kind="open")
    assert opened["base"] == PLAIN
    assert arms.SECRET_ENV == {"VOYAGE_API_KEY": "synthetic-key"}
    assert arms.all_arms() == expected_all
    assert PLAIN not in expected_all and OPEN not in expected_all
    assert all(arms.is_env_file_arm(name) for name in (PLAIN, OPEN))
    assert not arms.is_env_file_arm("LCMX-fleet-open")


def test_report_classifies_env_file_reader_arms(load):
    report = load("scorer/report_s.py")
    metric = {"status": "OK", "value": 1, "gate_events": {"p90": 0, "max": 0, "count": 0}}
    metrics = {k: metric for k in ("facts_kept", "stale_rate", "trap_abstention", "trap_failure_rate",
                                   "continuation", "recall", "continuity", "latency", "level3")}
    timing = dict(p50=0, p90=0, max=0)
    behaviour = dict(events=[], compactions=0, wall=timing, wall_label="synthetic", summariser_wall=timing,
                     levels=[], tokens_kept_at_checkpoint=10)
    row = dict(run="1", metrics=metrics, behaviour=behaviour, label_map={}, writer="synthetic", gaps=[])
    _, result = report.render({(name, "seed-1"): [row] for name in (PLAIN, OPEN)}, 1)
    gates = result["gates"]
    assert f"quality_vs_lc/{PLAIN}/facts_kept" in gates
    assert f"quality_vs_lc/{OPEN}/facts_kept" not in gates
    assert f"recall/{OPEN}/lossless-claw-open" in gates
    assert f"recall/{PLAIN}/lossless-claw-open" not in gates


def test_score_decision_selects_env_file_reader_arms(load, tmp_path):
    score = load("decision/score_decision.py")
    args = SimpleNamespace(material=tmp_path / "material", run_root=tmp_path / "runs",
                           external_root=tmp_path / "external", arms=[PLAIN, OPEN], checkpoints=[304], runs=[1])
    runs = {arm: (label, directory, cps) for arm, label, directory, cps in score.runs(1, args)
            if arm in args.arms}
    assert runs == {
        PLAIN: ("d1-r1", args.run_root / PLAIN / "seed-1/d1-r1", {304: "cp-304"}),
        OPEN: ("d1-open", args.run_root / OPEN / "seed-1/d1-open", {304: "cp-304"}),
    }


def test_render_decision_preserves_open_recall_source(load, tmp_path, monkeypatch):
    render = load("decision/render_decision.py")
    data = {name: {"seed-1": []} for name in (PLAIN, OPEN)}
    monkeypatch.setitem(render.main.__globals__, "load", lambda _: data)
    render.H.mkdir()
    (tmp_path / "report-unit-notes.md").write_text("Synthetic run")
    (tmp_path / "report-unit-notes.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["render_decision.py", "--report-notes", str(tmp_path)])
    render.main()
    rows = json.loads((render.H / "DECISION-RUN.json").read_text())["cp-304"]
    assert rows["quality"][PLAIN]["recall_source"] == OPEN
    assert rows["open"][OPEN]["recall_source"] == OPEN
    assert "one run per seed" in rows["open"][OPEN]["note"]

    # Data without the env-file arms renders no env-file rows (the existing arms' output is unchanged).
    monkeypatch.setitem(render.main.__globals__, "load", lambda _: {"LCMX-fleet": {"seed-1": []}})
    render.main()
    assert PLAIN not in (render.H / "DECISION-RUN.md").read_text()
    assert PLAIN not in (render.H / "DECISION-RUN.json").read_text()


@pytest.fixture
def stubbed_runner(load, tmp_path, monkeypatch):
    runner = load("s2/run_s_lcmx.py")
    monkeypatch.setattr(runner.A, "SECRET_ENV", {})
    (tmp_path / "s2").mkdir()
    (tmp_path / "material/seed-1").mkdir(parents=True)
    (tmp_path / "material/seed-1/material.manifest.json").write_text("{}")
    rows = [{"role": "system", "content": "Fixed system load"}]
    man = {"continuity": [], "decision_checkpoint": {"tokens": 300}}
    engine = SimpleNamespace(threshold_tokens=100, shutdown=lambda: None)
    tokens = SimpleNamespace(count_messages_tokens=lambda msgs: sum(4 + len(m["content"]) for m in msgs),
                             count_tokens=len, _get_encoder=lambda: None)
    fake = SimpleNamespace(tokens=tokens, engine=SimpleNamespace(LCMEngine=type("Engine", (), {})),
                           config=SimpleNamespace(LCMConfig=lambda: None))
    fake.config.LCMConfig.from_env = lambda: None
    monkeypatch.setattr(runner.seam, "load_engine", lambda _: fake)
    monkeypatch.setattr(runner.seam, "set_lane", lambda *_: None)
    monkeypatch.setattr(runner.seam, "set_fault", lambda *_: None)
    ns = runner.main.__globals__
    monkeypatch.setitem(ns, "os", SimpleNamespace(environ={"LCM_UNUSED": "synthetic"}))
    logger = SimpleNamespace(addHandler=lambda _: None, setLevel=lambda _: None)
    monkeypatch.setitem(ns, "logging", SimpleNamespace(FileHandler=lambda _: None, getLogger=lambda _: logger, INFO=20))
    monkeypatch.setitem(ns, "select_rows", lambda *_: (rows, man, 0))
    monkeypatch.setitem(ns, "new_engine", lambda *_: engine)
    monkeypatch.setitem(ns, "config_dict", lambda _: {})
    monkeypatch.setattr(runner.Run, "receipts", lambda self: {})
    monkeypatch.setitem(ns, "probe", lambda *args: [])
    monkeypatch.setitem(ns, "tempfile", SimpleNamespace(mkdtemp=lambda **_: str(tmp_path / "dry")))
    monkeypatch.setitem(ns, "shutil", SimpleNamespace(rmtree=lambda *args, **kwargs: None))
    return runner


@pytest.mark.parametrize("name", [PLAIN, OPEN])
def test_env_file_dry_run_consumers(stubbed_runner, env_file, name, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_s_lcmx.py", "--arm", name, "--seed", "1", "--dry-run"])
    stubbed_runner.main()
    stdout = capsys.readouterr().out
    assert f"{name}: effective LCM_* env (4 keys) == env file LCM_* set (2)" in stdout
    assert ": True; ignored file keys:" in stdout
    assert "fleet keys excluded:" not in stdout


@pytest.mark.parametrize("name", [PLAIN, OPEN])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_replay_records_env_file_and_prompt_tokens(stubbed_runner, env_file, name, checkpoint, tmp_path, monkeypatch):
    summary_block = "[Recent Summary (d0, node 1)]\nSynthetic memory\n[Expand for details: node 1]"
    tail = "Fixed reader load"
    view = [{"role": "user", "content": summary_block + "\n\n" + tail},
            {"role": "assistant", "content": "Fresh turn"}]
    def replay(run):
        if not checkpoint:
            return view
        out = run.dir / "cp-S1-CP340000"
        out.mkdir()
        db = out / "lcm.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE summary_nodes (session_id TEXT)")
        run.snaps.append(dict(dir=out, db=db, view=view, row=0, checkpoint={"id": "S1-CP340000"},
                              n_events=0, receipts=[], admission={}))
        return view + [{"role": "user", "content": "After checkpoint"}]

    monkeypatch.setattr(stubbed_runner.Run, "replay", replay)
    monkeypatch.setattr(sys, "argv", ["run_s_lcmx.py", "--arm", name, "--seed", "1",
                                      "--reader", "fake", "--lane", "fake"])
    assert stubbed_runner.main() == 0
    out = tmp_path / "lcmx-runs" / name / "seed-1/1"
    if checkpoint:
        assert "summary_tokens" not in json.loads((out / "summary.json").read_text())
        out /= "cp-S1-CP340000"
    record = json.loads((out / "summary.json").read_text())
    assert record["fleet_keys_excluded"] == record["harness_overrides"] == {}
    assert record["effective_lcm_env"]["LCM_LEAF_CHUNK_TOKENS"] == "2000"
    assert record["summary_tokens"] == len(summary_block)
    assert record["prompt_tokens"] == 12 + len("Fixed system load") + sum(len(m["content"]) for m in view)


def test_summary_share_cli(tmp_path):
    paths = [tmp_path / name for name in (PLAIN, OPEN, "missing-count")]
    for path in paths:
        path.mkdir()
    base = {"seed": "seed-1", "run": "1", "checkpoint": 304}
    receipts = [
        dict(base, arm={"name": PLAIN}, summary_tokens=35, prompt_tokens=100),
        dict(base, arm={"name": OPEN}, summary_tokens=80, prompt_tokens=160, checkpoint_id="S1-CP340000"),
        dict(base, arm={"name": OPEN}, prompt_tokens=160),
    ]
    for name, bad in (("zero-prompt", {"summary_tokens": 1, "prompt_tokens": 0}),
                      ("bool-count", {"summary_tokens": True, "prompt_tokens": 10})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(dict(base, arm={"name": PLAIN}, **bad)))
        rejected = subprocess.run([sys.executable, str(TRACK / "scorer/summary_share.py"), tmp_path / name],
                                  capture_output=True, text=True)
        assert rejected.returncode == 2 and not rejected.stdout, name
    for path, record in zip(paths, receipts):
        (path / "summary.json").write_text(json.dumps(record))
    cmd = [sys.executable, str(TRACK / "scorer/summary_share.py")]
    result = subprocess.run([*cmd, *paths[:2]], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        dict(base, arm=PLAIN, summary_tokens=35, prompt_tokens=100, share=0.35),
        dict(base, arm=OPEN, summary_tokens=80, prompt_tokens=160, share=0.5, checkpoint="S1-CP340000"),
    ]
    result = subprocess.run([*cmd, paths[2], paths[0]], capture_output=True, text=True)
    assert result.returncode == 2
    assert str(paths[2]) in result.stderr
    assert len(result.stdout.splitlines()) == 1


def test_summary_blocks_match_the_engine_render(load):
    """The block pattern against the engine's own renderer: objective part first, parts joined as assembly joins
    them, a body with its own separator and hint line, and a row glued behind the prefix as Hermes merges it."""
    import hermes_lcm.engine as lcm_engine

    runner = load("s2/run_s_lcmx.py")
    nodes = [SimpleNamespace(depth=3, node_id=40, summary="Oldest arc.", expand_hint="the first week"),
             SimpleNamespace(depth=2, node_id=31, summary="Durable facts.\n\n---\n\nA second section.\n"
                             "Expand for details about: decisions", expand_hint="decisions [and dates]"),
             SimpleNamespace(depth=1, node_id=22, summary="Session arc.", expand_hint="the arc"),
             SimpleNamespace(depth=0, node_id=13, summary="Recent work.", expand_hint="recent turns")]
    parts = [lcm_engine._summary_part_text(node) for node in nodes]
    objective = f"{lcm_engine._PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nKeep the objective."
    glued = "\n\n---\n\n".join([objective, *parts]) + "\n\nThe next real user row."
    assert runner.SUMMARY_BLOCK_RE.findall(glued) == parts
    assert runner.SUMMARY_BLOCK_RE.findall("\n\n---\n\n".join(parts)) == parts
    run = SimpleNamespace(m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=len)))
    view = [{"role": "system", "content": "System."},
            {"role": "user", "content": [{"type": "text", "text": glued}, {"type": "image_url", "image_url": {}}]},
            {"role": "assistant", "content": "Quoting [Recent Summary (d0, node 13)] in a reply is not a block."}]
    assert runner.summary_tokens(run, view) == sum(len(p) for p in parts)
    # A block counts only at the start or after the part separator, and only when a paragraph break or the end follows.
    part = parts[-1]
    for text in ("Quoted earlier:\n\n" + part, "Quoted inline: " + part, part + "\nthe same paragraph goes on"):
        assert runner.SUMMARY_BLOCK_RE.findall(text) == [], text[:20]


def test_schema_engine_serves_the_current_engine(load, monkeypatch):
    """The open reader's schema lookup on this tree: a bare shell raises, the configured shell lists the public tools,
    and the arm's env decides what it lists."""
    import os

    import hermes_lcm.config as lcm_config
    import hermes_lcm.engine as lcm_engine
    from hermes_lcm.recall_session import session_schema

    for key in [key for key in os.environ if key.startswith("LCM_")]:  # no ambient settings
        monkeypatch.delenv(key)
    runner = load("s2/run_s_lcmx.py")
    m = SimpleNamespace(engine=lcm_engine, config=lcm_config)

    def listed():
        return {schema.get("name"): schema for schema in lcm_engine.LCMEngine.get_tool_schemas(runner.schema_engine(m))}

    with pytest.raises(AttributeError):
        lcm_engine.LCMEngine.get_tool_schemas(lcm_engine.LCMEngine.__new__(lcm_engine.LCMEngine))
    default = listed()
    assert set(runner.R.PUBLIC_TOOLS) <= set(default)
    # The env applied before the lookup reaches the shell: disabled tools, session delivery and guidance v2.
    monkeypatch.setenv("LCM_DISABLED_TOOLS", "lcm_describe")
    assert "lcm_describe" not in listed() and set(runner.R.PUBLIC_TOOLS) - {"lcm_describe"} <= set(listed())
    monkeypatch.delenv("LCM_DISABLED_TOOLS")
    monkeypatch.setenv("LCM_RECALL_DELIVERY_UNIT", "session")
    assert listed()["lcm_recall"] == session_schema(default["lcm_recall"]) != default["lcm_recall"]
    monkeypatch.delenv("LCM_RECALL_DELIVERY_UNIT")
    monkeypatch.setenv("LCM_RECALL_GUIDANCE", "v2")
    assert listed()["lcm_recall"]["description"].startswith(default["lcm_recall"]["description"] + " ")
    # main() builds the lookup shell through the helper, not a bare shell (probe() is stubbed in the main() tests).
    source = inspect.getsource(runner.main)
    assert "schema_engine(m)" in source and "LCMEngine.__new__" not in source
