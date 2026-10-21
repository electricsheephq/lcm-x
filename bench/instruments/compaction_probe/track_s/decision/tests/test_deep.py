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


def test_aggregate_fails_entire_arm_on_one_compaction_miss(cell, tmp_path):
    import hashlib
    mat, directory = cell
    run = tmp_path / "runs" / "H" / "seed-1" / "d1-r1"
    digest = hashlib.sha256((mat / "material.manifest.json").read_bytes()).hexdigest()
    summary = json.loads((directory / "summary.json").read_text())
    summary.update(material_sha256=digest, events=[event(False)])
    write(run / "summary.json", summary)
    write(run / "cp-1" / "summary.json", summary)
    (run / "cp-1" / "results.jsonl").write_text((directory / "results.jsonl").read_text())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "s2-H-d1-r1.log.wall").write_text("exit 0 end now\n")
    args = SimpleNamespace(material=mat.parent, run_root=tmp_path / "runs", logs=logs, out=tmp_path / "scores",
                           arms=["H"], seeds=[1], checkpoints=["1"])
    result = aggregate(args)
    assert result["per_arm"]["H"]["result"] == "FAIL"
    assert result["per_arm"]["H"]["summary_tokens"] == dict(input=100, output=20, estimated_calls=0)
    assert result["per_run"]["H|d1-r1"]["dag"]["status"] == "N/A"
    assert result["per_arm"]["H"]["checkpoints"]["1"]["facts_kept"] == 1
    assert list(runs(1, args))[0][0] == "H"
    args.checkpoints = ["last"]
    assert list(runs(1, args))[0][3] == {3: "cp-3"}
