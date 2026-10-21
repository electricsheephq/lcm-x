"""Declaration v2 coverage and calibration regressions; synthetic stores only."""
from argparse import Namespace as NS
import json
import pytest
from bench.instruments.local_catchup import floor, rescore
pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


def events(seconds=.572):
    return [dict(event="_embedding_backfill_summary_run", embedded=32, at=i, end=i+seconds) for i in range(5)]


@pytest.mark.parametrize("seconds,normalized", [(.267, True), (.700, False)])
def test_normalized_rate(seconds, normalized):
    result = floor.rate(dict(state="INCOMPLETE", rows_s=2, audit=dict(complete=False)), 7200, 250000, events(seconds))
    assert result["implied_hours"] == 250000/2/3600
    assert result["reference_calibration_s"] == .572
    assert (result["implied_hours_gated"] > result["implied_hours"]) is normalized
    assert result["implied_hours_gated"] == max(result["implied_hours"], result["implied_hours_normalized"])


def test_missing_calibration_is_invalid():
    result = floor.rate(dict(state="INCOMPLETE", rows_s=10, audit=dict(complete=False)), 7200, 250000)
    assert (result["state"], result["reason"]) == ("INVALID", "no_calibration")


def test_only_exact_32_summary_batches_calibrate():
    other = [dict(event="_chunk_backfill_run", embedded=32, at=0, end=999),
             dict(event="_embedding_backfill_summary_run", embedded=31, at=0, end=999),
             dict(event="_embedding_backfill_summary_run", foreground=False)]
    assert floor.calibration(events()+other) == pytest.approx(.572)
    assert floor.calibration(events()[:4]+other) is None


@pytest.mark.parametrize("shape,order", [("short", ["empty", "upgrade"]), ("mixed", ["upgrade", "empty"])])
def test_pair_order_and_separate_evidence(tmp_path, monkeypatch, shape, order):
    a = NS(out=tmp_path, scratch=tmp_path/"scratch", cgroup=tmp_path, cell="coverage-pair", shape=shape, baseline="empty")
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **kw: "CPU fixture\n")
    monkeypatch.setattr(floor, "standalone", lambda _: {})
    def measure(a):
        floor.dump(a.out/"summary.json", dict(state="FAIL" if a.baseline == "upgrade" else "PASS", baseline=a.baseline))
    monkeypatch.setattr(floor, "coverage", measure)
    floor.limited(a)
    result = json.loads((tmp_path/"summary.json").read_text())
    assert [r["baseline"] for r in result["cells"]] == order
    assert result["state"] == "FAIL"
    assert all((tmp_path/b/"summary.json").exists() for b in order)


def test_rescore_recomputes_and_checks_coverage(tmp_path):
    (tmp_path/"catchup").mkdir()
    summary = dict(state="INCOMPLETE", rows_s=2, audit=dict(complete=False))
    floor.dump(tmp_path/"catchup/summary.json", summary)
    floor.dump(tmp_path/"manifest.json", dict(rows=250000))
    (tmp_path/"catchup/events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events()))
    floor.dump(tmp_path/"summary.json", floor.rate(summary, 7200, 250000, events()))
    assert rescore.v2(tmp_path)["state"] == "PASS"
    recorded = json.loads((tmp_path/"summary.json").read_text())
    recorded["implied_hours_gated"] = 0
    floor.dump(tmp_path/"summary.json", recorded)
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


def test_standalone_five_fresh_32_leaf_batches(corpus, tmp_path, monkeypatch):
    from bench.instruments.local_catchup import calibration
    _, _, mod = corpus
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    def backfill(cfg, db, **kw):
        import sqlite3
        with sqlite3.connect(db) as c:
            assert c.execute("SELECT count(*) FROM summary_nodes WHERE depth=0").fetchone()[0] == 32
            assert c.execute("SELECT count(*) FROM lcm_embedding_vectors").fetchone()[0] == 0
        calls.append((db.name, kw))
        return dict(embedded=32)
    monkeypatch.setattr(mod("command"), "_embedding_backfill_summary_run", backfill)
    monkeypatch.setattr(mod("embedding_provider"), "resolve_provider", lambda *a, **kw: NS(_ensure_local=lambda **kw: None))
    a = NS(scratch=tmp_path/"cal", out=tmp_path, model_cache=tmp_path, lcmx_repo=tmp_path, hermes_source=tmp_path)
    monkeypatch.setattr(calibration, "product", lambda *a: mod)
    value = calibration.standalone(a)
    assert len(set(n for n, _ in calls)) == len(value["core_seconds"]) == 5
    assert all(k == dict(apply=True, limit=32, retry_uncertain=False, include_next_hint=False) for _, k in calls)
    assert value["reference_standalone_s"] is None
