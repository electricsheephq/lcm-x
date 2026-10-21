"""Offline synthetic deep scorer contracts; no provider, live stores or credentials."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scorer"))
from loss_class import classify_loss  # noqa: E402
from pmw_read import engine_turn_wait  # noqa: E402
from score_deep import KINDS, aggregate, continuity_events  # noqa: E402
from score_s import checkpoint_facts, score  # noqa: E402
from score_decision import runs  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def cell(tmp_path):
    mat = tmp_path / "material" / "seed-1"
    fact = dict(id="F1", value="new-value", answer="new-value", stale="old-value", placement="head", **{"class": "decision"},
                row_index=2, versions=[dict(value="old-value", row_index=0), dict(value="new-value", row_index=2)])
    early = dict(fact, value="old-value", answer="old-value", stale=None, row_index=0, versions=fact["versions"][:1])
    future = dict(fact, id="F2", value="future", row_index=3)
    write(mat / "facts.json", [fact, future])
    write(mat / "deep_metadata.json", dict(checkpoint_answers={"1": [early], "3": [fact, future]}))
    write(mat / "traps.json", [])
    write(mat / "continuation.json", dict(id="STATE", row_index=10))
    write(mat / "material.manifest.json", dict(mode="full", continuity=[], receipt_targets=[]))
    (mat / "probe_batches.jsonl").write_text(json.dumps(dict(id="B1", probes=[dict(id="F1"), dict(id="F2")])) + "\n")
    (mat / "transcript.jsonl").write_text("\n".join(json.dumps(dict(turn=t)) for t in (0, 1, 1, 2)))
    directory = tmp_path / "run"
    write(directory / "summary.json", dict(status="DONE", arm=dict(kind="builtin"), checkpoint=1, stop_row_index=1,
        events=[], admission=dict(facts_not_admitted=[]), receipts=[]))
    (directory / "results.jsonl").write_text(json.dumps(dict(schema="s2-result-v1", probe_id="F1", arm="H", kind="probe",
        answer="old-value", status="OK", batch_id="B1")) + "\n")
    return mat, directory


def test_checkpoint_answer_and_loss_filtering(cell):
    mat, directory = cell
    scored = score(mat, directory, "H")
    assert scored["metrics"]["facts_kept"]["denominator"] == 1
    assert scored["metrics"]["facts_kept"]["correct"] == 1
    assert "F2" not in scored["probes"]
    assert scored["metrics"]["deep_corrected"]["denominator"] == 0
    assert classify_loss(scored)["lost"] == 0
    with pytest.raises(ValueError, match="answer key absent"):
        checkpoint_facts(mat, 2)


def test_corrected_answer_rejects_stale_assertion_and_h_needs_no_store(cell):
    mat, directory = cell
    summary = json.loads((directory / "summary.json").read_text())
    write(directory / "summary.json", dict(summary, checkpoint=3, stop_row_index=3))
    (directory / "results.jsonl").write_text(json.dumps(dict(schema="s2-result-v1", probe_id="F1", arm="H", kind="probe",
        answer="new-value, also old-value", status="OK", batch_id="B1")) + "\n")
    scored = score(mat, directory, "H")
    assert scored["metrics"]["deep_corrected"]["correct"] == 0
    assert scored["metrics"]["deep_corrected"]["stale_ids"] == ["F1"]
    losses = classify_loss(scored)
    assert losses["counts"] == {"incomplete": 1}


def event(passed=True, wall=2):
    return dict(event=0, turn=1, row_index=2, is_compaction=True, compress_wall_s=wall,
        nodes=[dict(depth=2)], summariser_calls=[dict(prompt_tokens=100, completion_tokens=20)],
        deep_probes=[dict(kind=kind, passed=passed, present_verbatim=passed, reader_pass=passed) for kind in KINDS])


def test_one_event_failure_is_fail_and_missing_is_incomplete():
    assert continuity_events(dict(events=[event(), event(False)]))["result"] == "FAIL"
    assert continuity_events(dict(events=[]))["result"] == "INCOMPLETE"
    missing = event()
    missing["deep_probes"].pop()
    assert continuity_events(dict(events=[missing]))["result"] == "INCOMPLETE"
    missing = event()
    del missing["deep_probes"][0]["reader_pass"]
    assert continuity_events(dict(events=[missing]))["result"] == "INCOMPLETE"


def test_wait_groups_events_per_turn_and_includes_zero_turns():
    rows = [dict(turn=t) for t in (0, 1, 1, 2)]
    result = engine_turn_wait(dict(stop_row_index=3, events=[event(wall=.6), event(wall=.6)]), rows)
    assert result["turn_waits_s"] == [0, 1.2, 0]
    assert result["mean_s"] == pytest.approx(.4)
    assert result["waited_share_ge_1s"] == pytest.approx(1 / 3)
    assert result["compactions_per_100_messages"] == 50
    assert not engine_turn_wait(dict(stop_row_index=3, events=[event(wall=None)]), rows)["complete"]


@pytest.mark.parametrize("passed,expected", [(False, "FAIL"), (None, "INCOMPLETE"), (True, "COMPLETE")])
def test_aggregate_fails_entire_arm_on_one_compaction_miss(cell, tmp_path, passed, expected):
    import hashlib
    mat, directory = cell
    run = tmp_path / "runs" / "H" / "seed-1" / "d1-r1"
    digest = hashlib.sha256((mat / "material.manifest.json").read_bytes()).hexdigest()
    summary = json.loads((directory / "summary.json").read_text())
    summary.update(material_sha256=digest, host_sha="host", events=[event(passed)])
    write(run / "summary.json", summary)
    write(run / "cp-1" / "summary.json", summary)
    (run / "cp-1" / "results.jsonl").write_text((directory / "results.jsonl").read_text())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "s2-H-d1-r1.log.wall").write_text("exit 0 end now\n")
    args = SimpleNamespace(material=mat.parent, run_root=tmp_path / "runs", logs=logs, out=tmp_path / "scores",
                           arms=["H"], seeds=[1], checkpoints=["1"], expected_host_sha="host")
    result = aggregate(args)
    assert result["per_arm"]["H"]["result"] == expected
    for population in ("scores", "loss"):
        assert bool(list((args.out / population).glob("cp-*/*.json"))) == (expected == "COMPLETE")
    from score_deep import exit_status
    assert exit_status(result) == {"FAIL": 1, "INCOMPLETE": 2, "COMPLETE": 0}[expected]
    assert result["per_arm"]["H"]["summary_tokens"] == dict(input=100, output=20, estimated_calls=0)
    assert result["per_run"]["H|d1-r1"]["dag"]["status"] == "N/A"
    assert result["per_arm"]["H"]["checkpoints"]["1"]["facts_kept"] == 1
    assert list(runs(1, args))[0][0] == "H"
    args.checkpoints = ["last"]
    assert list(runs(1, args))[0][3] == {3: "cp-3"}


def test_round2_rejects_every_stale_history_value(cell):
    mat, directory = cell
    metadata = json.loads((mat / "deep_metadata.json").read_text())
    fact = metadata["checkpoint_answers"]["1"][0]
    fact["stale_history"] = [dict(value="obsolete-origin")]
    fact["stale"] = "intermediate-value"
    write(mat / "deep_metadata.json", metadata)
    rows = [json.loads(r) for r in (directory / "results.jsonl").read_text().splitlines()]
    rows[0]["answer"] = "old-value and obsolete-origin"
    (directory / "results.jsonl").write_text(json.dumps(rows[0]) + "\n")
    corrected = score(mat, directory, "H")["metrics"]["deep_corrected"]
    assert corrected["correct"] == 0 and corrected["stale_ids"] == ["F1"]


@pytest.mark.parametrize("answer", [None, {}, {"R": None}, {"R": []}, {"R": ""}])
def test_round2_missing_event_answer_is_incomplete(tmp_path, monkeypatch, answer):
    sys.path.insert(0, str(HERE.parent / "s2"))
    from s2lib import deep
    from score_deep import exit_status
    specs = [dict(id="R", kind="current_request", text="request?", answer="keep request")]
    run = SimpleNamespace(dir=tmp_path, man=dict(material_version="track-s-deep-v2", continuity_probes=specs),
                          system="", m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=len)))
    monkeypatch.setattr(deep.R, "SolLowReader", lambda _: SimpleNamespace(readback={}))
    monkeypatch.setattr(deep.R, "answer", lambda *a, **k: (answer, {}))
    monkeypatch.setattr(deep.R, "flatten", lambda system, view: "earlier: keep request")
    ev = event()
    deep.event_probes(run, ev, [])
    assert ev["deep_probes"][0]["reader_pass"] is None and ev["deep_probes"][0]["passed"] is None
    # A missing read of a value that is in context is incomplete evidence, not a miss.
    result = continuity_events(dict(events=[ev]))
    assert result["result"] == "INCOMPLETE"
    assert exit_status(dict(per_arm={"H": result})) == 2
    # A definite miss elsewhere is still a definite FAIL: the gate is a conjunction.
    assert continuity_events(dict(events=[ev, event(False)]))["result"] == "FAIL"
    assert exit_status(dict(per_arm={"H": result, "L": dict(result="FAIL")})) == 1
    # A value absent from context fails whatever the reader returned.
    monkeypatch.setattr(deep.R, "flatten", lambda system, view: "")
    absent = event()
    deep.event_probes(run, absent, [])
    assert absent["deep_probes"][0]["passed"] is False


@pytest.mark.parametrize("arm,field", [("H", "host_sha"), ("LCMX-fleet-rc3", "worktree_head")])
@pytest.mark.parametrize("recorded,expected", [(None, "pinned"), ("other", "pinned"), ("pinned", None)])
def test_round2_run_pins_required(cell, tmp_path, arm, field, recorded, expected):
    import hashlib
    mat, directory = cell
    run = tmp_path / "runs" / arm / "seed-1/d1-r1"
    summary = json.loads((directory / "summary.json").read_text())
    summary.update(material_sha256=hashlib.sha256((mat / "material.manifest.json").read_bytes()).hexdigest(),
                   events=[event()], **{field: recorded})
    write(run / "summary.json", summary)
    write(run / "cp-1/summary.json", summary)
    (run / "cp-1/results.jsonl").write_text((directory / "results.jsonl").read_text())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"s2-{arm}-d1-r1.log.wall").write_text("exit 0 end now\n")
    args = SimpleNamespace(material=mat.parent, run_root=tmp_path / "runs", logs=logs, out=tmp_path / "out",
                           arms=[arm], seeds=[1], checkpoints=["1"],
                           expected_host_sha=expected, expected_product_sha=expected)
    from score_deep import exit_status
    result = aggregate(args)
    assert result["per_run"][f"{arm}|d1-r1"]["result"] == "INCOMPLETE"
    assert "pin" in result["per_run"][f"{arm}|d1-r1"]["reason"]
    assert exit_status(result) == 2
    assert not args.out.exists()


@pytest.mark.parametrize("stop", [-1, 4, 100, True, "3", 1.5])
def test_round2_wait_stop_must_be_in_range_integer(stop):
    result = engine_turn_wait(dict(stop_row_index=stop, events=[event()]), [dict(turn=t) for t in (0, 1, 1, 2)])
    assert not result["complete"] and "stop_row_index" in result["reason"]


def test_round2_checkpoint_pin_must_match_run_pin(cell, tmp_path):
    import hashlib
    mat, directory = cell
    run = tmp_path / "runs" / "H" / "seed-1/d1-r1"
    summary = json.loads((directory / "summary.json").read_text())
    summary.update(material_sha256=hashlib.sha256((mat / "material.manifest.json").read_bytes()).hexdigest(),
                   events=[event()], host_sha="pinned")
    write(run / "summary.json", summary)
    write(run / "cp-1/summary.json", dict(summary, host_sha="other"))
    (run / "cp-1/results.jsonl").write_text((directory / "results.jsonl").read_text())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "s2-H-d1-r1.log.wall").write_text("exit 0 end now\n")
    args = SimpleNamespace(material=mat.parent, run_root=tmp_path / "runs", logs=logs, out=tmp_path / "out",
                           arms=["H"], seeds=[1], checkpoints=["1"], expected_host_sha="pinned")
    result = aggregate(args)
    run_result = result["per_run"]["H|d1-r1"]
    assert run_result["result"] == "INCOMPLETE" and "1" not in run_result["checkpoints"]
    assert any("cp-1" in r and "pin" in r for r in run_result["reasons"])


def test_round2_stale_rate_counts_stale_history(cell):
    mat, directory = cell
    metadata = json.loads((mat / "deep_metadata.json").read_text())
    fact = metadata["checkpoint_answers"]["1"][0]
    fact["stale_history"] = [dict(value="obsolete-origin")]
    fact["stale"] = "intermediate-value"
    write(mat / "deep_metadata.json", metadata)
    rows = [json.loads(r) for r in (directory / "results.jsonl").read_text().splitlines()]
    rows[0]["answer"] = "obsolete-origin"
    (directory / "results.jsonl").write_text(json.dumps(rows[0]) + "\n")
    metrics = score(mat, directory, "H")["metrics"]
    assert metrics["stale_rate"]["stale_ids"] == ["F1"]
    assert metrics["deep_corrected"]["stale_ids"] == ["F1"]


@pytest.mark.parametrize("arm,field", [("H", "host_sha"), ("LCMX-fleet-rc3", "worktree_head")])
@pytest.mark.parametrize("rejection", ["run-pin", "missing-run-pin", "expected-pin", "checkpoint-pin",
                                      "missing-checkpoint-pin", "wall", "missing-wall", "summary", "checkpoint"])
def test_rejected_rerun_removes_only_its_prior_scores(cell, tmp_path, arm, field, rejection):
    import hashlib
    from score_deep import exit_status, write_deep

    mat, directory = cell
    run = tmp_path / "runs" / arm / "seed-1/d1-r1"
    summary = json.loads((directory / "summary.json").read_text())
    summary.update(material_sha256=hashlib.sha256((mat / "material.manifest.json").read_bytes()).hexdigest(),
                   events=[event()], **{field: "pinned"})
    write(run / "summary.json", summary)
    write(run / "cp-1/summary.json", summary)
    rows = [dict(json.loads(line), arm=arm) for line in (directory / "results.jsonl").read_text().splitlines()]
    (run / "cp-1/results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    wall = tmp_path / "logs" / f"s2-{arm}-d1-r1.log.wall"
    wall.parent.mkdir()
    wall.write_text("exit 0 end now\n")
    args = SimpleNamespace(material=mat.parent, run_root=tmp_path / "runs", logs=wall.parent,
                           out=tmp_path / "out", arms=[arm], seeds=[1], checkpoints=["1"],
                           expected_host_sha="pinned", expected_product_sha="pinned")
    first = write_deep(args)
    assert exit_status(first) == 0
    assert first["per_run"][f"{arm}|d1-r1"]["continuity"]["result"] == "PASS"
    name = f"{arm}.seed-1.d1-r1.json"
    for population in ("scores", "loss"):
        assert (args.out / population / "cp-1" / name).is_file()
        # A previously selected checkpoint must also be invalidated on this rerun.
        write(args.out / population / "cp-999" / name, {"old": True})
    preserved = [args.out / "keep.json", args.out / "other" / "cp-1" / name]
    for population in ("scores", "loss"):
        preserved.extend(args.out / population / "cp-1" / filename for filename in
                         (f"{arm}.seed-2.d2-r1.json", f"{arm}.seed-1.d1-r2.json", "other.seed-1.d1-r1.json"))
        preserved.append(args.out / population / "not-a-checkpoint" / name)
    for path in preserved:
        write(path, {"keep": True})
    before = {path: path.read_bytes() for path in preserved}

    if rejection in ("run-pin", "missing-run-pin", "summary"):
        if rejection == "summary":
            (run / "summary.json").unlink()
        else:
            write(run / "summary.json", dict(summary, **{field: "other" if rejection == "run-pin" else None}))
    elif rejection == "expected-pin":
        setattr(args, "expected_host_sha" if arm == "H" else "expected_product_sha", None)
    elif rejection in ("checkpoint-pin", "missing-checkpoint-pin", "checkpoint"):
        if rejection == "checkpoint":
            (run / "cp-1/summary.json").unlink()
        else:
            write(run / "cp-1/summary.json", dict(summary, **{field: "other" if rejection == "checkpoint-pin" else None}))
    elif rejection == "wall":
        wall.write_text("exit 1 end now\n")
    else:
        wall.unlink()

    result = write_deep(args)
    assert exit_status(result) == 2
    assert result["per_run"][f"{arm}|d1-r1"]["result"] == "INCOMPLETE"
    assert json.loads((args.out / "score-deep.json").read_text()) == result
    for population in ("scores", "loss"):
        assert not list((args.out / population).glob(f"cp-*/{name}"))
    assert {path: path.read_bytes() for path in preserved} == before


@pytest.mark.parametrize("arm,field", [("H", "host_sha"), ("LCMX-fleet-rc3", "worktree_head")])
@pytest.mark.parametrize("rejection", ["missing", "pin", "facts"])
def test_later_checkpoint_rejection_publishes_no_run_files(cell, tmp_path, arm, field, rejection):
    import hashlib
    from score_deep import exit_status, write_deep

    mat, directory = cell
    run = tmp_path / "runs" / arm / "seed-1/d1-r1"
    summary = json.loads((directory / "summary.json").read_text())
    summary.update(material_sha256=hashlib.sha256((mat / "material.manifest.json").read_bytes()).hexdigest(),
                   events=[event()], **{field: "pinned"})
    write(run / "summary.json", summary)
    write(run / "cp-1/summary.json", summary)
    rows = [dict(json.loads(line), arm=arm) for line in (directory / "results.jsonl").read_text().splitlines()]
    (run / "cp-1/results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    if rejection != "missing":
        later = dict(summary, checkpoint=3, stop_row_index=3)
        if rejection == "pin":
            later[field] = "other"
        write(run / "cp-3/summary.json", later)
        # F2 is absent, so the later checkpoint has incomplete facts.
        (run / "cp-3/results.jsonl").write_text(json.dumps(dict(rows[0], answer="new-value")) + "\n")
    wall = tmp_path / "logs" / f"s2-{arm}-d1-r1.log.wall"
    wall.parent.mkdir()
    wall.write_text("exit 0 end now\n")
    args = SimpleNamespace(material=mat.parent, run_root=tmp_path / "runs", logs=wall.parent,
                           out=tmp_path / "out", arms=[arm], seeds=[1], checkpoints=["1", "3"],
                           expected_host_sha="pinned", expected_product_sha="pinned")
    name = f"{arm}.seed-1.d1-r1.json"
    for population in ("scores", "loss"):
        for cp in (1, 3, 999):
            write(args.out / population / f"cp-{cp}" / name, {"old": True})

    result = write_deep(args)
    run_result = result["per_run"][f"{arm}|d1-r1"]
    assert run_result["checkpoints"]["1"]["facts"]["complete"]
    if rejection == "facts":
        assert not run_result["checkpoints"]["3"]["facts"]["complete"]
    assert run_result["result"] == "INCOMPLETE" and exit_status(result) == 2
    assert json.loads((args.out / "score-deep.json").read_text()) == result
    for population in ("scores", "loss"):
        assert not list((args.out / population).glob(f"cp-*/{name}"))
