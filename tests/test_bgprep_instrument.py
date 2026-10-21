"""Red-on-wrong scoring fixtures: no product code or provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bench.instruments.compaction_probe.bgprep.scoring import (
    at_least, coverage, intact, promotion, score_cell, waits,
)
from bench.instruments.compaction_probe.bgprep import replay, run


@pytest.fixture
def probe():
    return dict(expected=["user A", "reply A", "user B"],
                rows=[dict(store_id=i, identity=v) for i, v in enumerate(
                    ["user A", "reply A", "user B"], 1)],
                leaf_sources=[[1, 2]], frontier=2, active_view=["summary", "user B"])


def test_active_unsummarized_tail_is_covered(probe):
    score = coverage(probe)
    assert intact(score) and score["active_tail_ids"] == [3] and score["lost"] == 0


@pytest.mark.parametrize("frontier", [0, 1])
@pytest.mark.parametrize("live,excess", [(["old", "tail"], 1), (["tail"], 0)])
def test_covered_rows_in_active_view_fail(frontier, live, excess):
    probe = dict(expected=["old", "tail"],
                 rows=[dict(store_id=1, identity="old"), dict(store_id=2, identity="tail")],
                 leaf_sources=[[1]], frontier=frontier, active_view=["system", *live, "summary"])
    score = coverage(probe)
    assert intact(score) is (excess == 0)
    assert score["covered_in_active"] == excess
    assert score["active_tail_ids"] == [2]


@pytest.mark.parametrize("copies", [1, 2])
def test_covered_and_tail_shared_active_identity(copies):
    probe = dict(expected=["old stored", "tail stored"],
                 rows=[dict(store_id=i, identity=v, active_identity="same content")
                       for i, v in enumerate(["old stored", "tail stored"], 1)],
                 leaf_sources=[[1]], frontier=1, active_view=["same content"] * copies)
    score = coverage(probe)
    assert intact(score) is (copies == 1)
    assert score["covered_in_active"] == copies - 1
    assert score["active_tail_ids"] == [2]


def test_frontier_row_in_active_view_is_reported_without_leaf():
    probe = dict(expected=["old"], rows=[dict(store_id=1, identity="old")],
                 leaf_sources=[], frontier=1, active_view=["old"])
    score = coverage(probe)
    assert not intact(score) and score["covered_in_active"] == 1


@pytest.mark.parametrize("live,ordered,duplicates", [
    (["b", "a"], False, 0), (["a", "b", "b"], False, 1),
    (["a", "b"], True, 0), (["summary", "a", "b"], True, 0),
])
def test_active_tail_order_and_multiplicity(live, ordered, duplicates):
    probe = dict(expected=["stored a", "stored b"],
                 rows=[dict(store_id=i, identity="stored " + v, active_identity=v)
                       for i, v in enumerate(["a", "b"], 1)],
                 leaf_sources=[], frontier=0, active_view=live)
    score = coverage(probe)
    assert score["tail_order_preserved"] is ordered
    assert score["duplicate_tail"] == duplicates
    assert intact(score) is ordered


@pytest.mark.parametrize("leaves,ordered", [
    ([[2, 1]], False), ([[1, 2]], True), ([[3, 4], [1, 2]], False),
])
def test_ordered_leaf_provenance(leaves, ordered):
    count = sum(map(len, leaves))
    probe = dict(expected=list(range(1, count + 1)),
                 rows=[dict(store_id=i, identity=i) for i in range(1, count + 1)],
                 leaf_sources=leaves, frontier=count, active_view=[])
    score = coverage(probe)
    assert score["leaf_order_preserved"] is ordered
    assert intact(score) is ordered


@pytest.mark.parametrize("corrupt,field", [
    ("missing_tail", "lost"), ("missing_stored", "missing_rows"),
    ("duplicate_leaf", "duplicate_coverage"), ("duplicate_stored", "duplicate_stored"),
    ("duplicate_id", "duplicate_ids"), ("dangling", "dangling_sources"),
])
def test_wrong_coverage_fails(probe, corrupt, field):
    if corrupt == "missing_tail":
        probe["active_view"] = ["summary"]
    elif corrupt == "missing_stored":
        probe["rows"].pop()
    elif corrupt == "duplicate_leaf":
        probe["leaf_sources"].append([2])
    elif corrupt == "duplicate_stored":
        probe["rows"].append(dict(store_id=4, identity="user A"))
    elif corrupt == "duplicate_id":
        probe["rows"].append(deepcopy(probe["rows"][0]))
    elif corrupt == "dangling":
        probe["leaf_sources"].append([99])
    score = coverage(probe)
    assert score[field] and not intact(score)


@pytest.mark.parametrize("mutation", ["reverse", "rewrite", "frontier"])
def test_identity_chronology_and_frontier_fail_closed(probe, mutation):
    if mutation == "reverse":
        probe["rows"].reverse()
    elif mutation == "rewrite":
        probe["rows"][0]["identity"] = "changed"
    else:
        probe["frontier"] = 3  # live raw row behind published frontier is not an active tail
    assert not intact(coverage(probe))


def test_promotion_counts_only_threshold_crossings():
    score = promotion([dict(crossing=True, promoted=2, leaf_compactions=3),
                       dict(crossing=True, promoted=1, leaf_compactions=2),
                       dict(crossing=False, promoted=9, leaf_compactions=9)])
    assert score == dict(numerator=3, denominator=5, rate=.6)
    assert promotion([])["rate"] is None
    assert not at_least(dict(numerator=0, denominator=0), score)
    with pytest.raises(ValueError):
        promotion([dict(crossing=True, promoted=3, leaf_compactions=2)])


def test_failed_foreground_leaf_still_counts_as_ran():
    score = promotion([dict(crossing=True, promoted=1, leaves=1, leaf_compactions=2)])
    assert score["rate"] == .5 and score["denominator"] == 2


def test_pooled_rate_is_not_seed_rate_mean_and_equality_passes():
    # Seeds 1/1 and 0/9 pool to 1/10, not the misleading mean 1/2.
    pooled = dict(numerator=1, denominator=10)
    reference = dict(numerator=1, denominator=5)
    assert not at_least(pooled, reference)
    assert at_least(reference, reference)
    assert at_least(dict(numerator=3, denominator=10), reference)


def test_wait_nearest_rank_zeros_and_denominators():
    values = [99, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
    score = waits([dict(turn=i, crossing=i == 1, visible_wait_s=v) for i, v in enumerate(values)])
    assert score["turns"] == 10 and score["waited_turns"] == 2
    assert score["p90_s"] == 1 and score["waited_turn_share"] == .2


@pytest.fixture
def runner_main(tmp_path, monkeypatch):
    host = tmp_path / "host"
    (host / "agent").mkdir(parents=True)
    (host / "agent/context_engine.py").write_text("# host contract\n")
    argv = ["run.py", "background-prep-thinktime-on-off", "--out", str(tmp_path / "out"),
            "--product-source", str(tmp_path / "product"), "--product-sha", run.BASE,
            "--hermes-source", str(host), "--jobs", "6", "--freeze-reference"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(run, "git", lambda path, *words: "" if words[0] == "status" else run.BASE)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **kw: None)
    summary = dict(verdict="PASS", pooled_on=dict(numerator=1, denominator=1, rate=1),
                   cells={f"on-{s}": dict(promotion={}) for s in (1, 2, 3)})
    monkeypatch.setattr(run, "summarize", lambda root: summary)
    monkeypatch.setattr(run, "card", lambda *a: None)
    return tmp_path / "out", host


@pytest.mark.parametrize("checkout", ["product", "host"])
@pytest.mark.parametrize("status", [" M tracked.py", "?? untracked.py"])
def test_dirty_checkout_refused_before_replay(runner_main, monkeypatch, checkout, status):
    out, host = runner_main
    dirty = host if checkout == "host" else host.parent / "product"

    def git(path, *words):
        if words[0] == "status":
            assert words == ("status", "--porcelain", "--untracked-files=normal")
            return status if path == dirty else ""
        return run.BASE

    monkeypatch.setattr(run, "git", git)
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2 and not out.exists()


def test_reference_published_only_after_card_success(runner_main, monkeypatch):
    out, _ = runner_main

    def failed_card(*a):
        assert not (out / "REFERENCE.json").exists()
        raise subprocess.CalledProcessError(1, "card")

    monkeypatch.setattr(run, "card", failed_card)
    with pytest.raises(subprocess.CalledProcessError):
        run.main()
    assert not (out / "REFERENCE.json").exists()


def test_successful_reference_uses_snapshot_cells(runner_main, monkeypatch):
    out, _ = runner_main
    commands = []
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    assert run.main() == 0
    assert (out / "REFERENCE.json").exists()
    assert len(commands) == 6
    for cmd in commands:
        assert Path(cmd[2]) == out / "instrument/run.py"
        assert cmd[cmd.index("--host-sha") + 1] == run.BASE
    # Import the copied entry point from elsewhere: both siblings must resolve to the copy.
    code = ("import run, replay, scoring; from pathlib import Path; "
            "assert all(Path(m.__file__).parent == Path('" + str(out / "instrument") + "') "
            "for m in (run, replay, scoring))")
    subprocess.check_call([sys.executable, "-B", "-c", code], cwd=out,
                          env={"PYTHONPATH": str(out / "instrument")})


def test_protocol_binds_host_git_sha(runner_main, monkeypatch):
    out, host = runner_main
    monkeypatch.setattr(run, "git", lambda path, *words: "" if words[0] == "status" else
                        "host-sha" if path == host else run.BASE)
    assert run.main() == 0
    protocol = json.loads((out / "run.json").read_text())["protocol"]
    assert protocol["host_sha"] == "host-sha"
    assert protocol["host_context_sha256"]


@pytest.mark.parametrize("pin", ["changed-host", None, run.BASE])
def test_cell_checks_pinned_host_before_replay(runner_main, monkeypatch, pin):
    out, _ = runner_main
    sys.argv.extend(["--cell", "on", "1"])
    if pin is not None:
        sys.argv.extend(["--host-sha", pin])
    replays = []
    monkeypatch.setattr(run, "run_cell", lambda *a: replays.append(a))
    if pin == run.BASE:
        assert run.main() == 0
        assert len(replays) == 1
    else:
        with pytest.raises(ValueError, match="host HEAD"):
            run.main()
        assert not replays and not out.exists()


@pytest.mark.parametrize("field", [None, "pooled_on", "protocol", "seeds", "per_seed"])
def test_reference_matches_card_covered_files(runner_main, monkeypatch, field):
    out, host = runner_main
    sys.argv.remove("--freeze-reference")
    refroot = host.parent / "reference"
    refroot.mkdir()
    ref = dict(product_sha=run.BASE, seeds=[1, 2, 3], protocol=run.protocol(SimpleNamespace(
        gap_seconds=60, real_route=False, summary_model="", jobs=6,
        seeds=[1, 2, 3], hermes_source=host)), pooled_on=dict(numerator=3, denominator=6, rate=.5),
        per_seed={str(s): dict(numerator=1, denominator=2, rate=.5) for s in (1, 2, 3)})
    (refroot / "run.json").write_text(json.dumps(dict(seeds=ref["seeds"], protocol=ref["protocol"])))
    (refroot / "summary.json").write_text(json.dumps(dict(pooled_on=ref["pooled_on"], cells={
        f"on-{s}": dict(promotion=ref["per_seed"][str(s)]) for s in (1, 2, 3)})))
    if field == "pooled_on":
        ref[field] = dict(numerator=0, denominator=6, rate=0)
    elif field == "protocol":
        ref[field]["jobs"] = 1
    elif field == "seeds":
        ref[field] = [1]
    elif field == "per_seed":
        ref[field]["1"]["rate"] = 0
    path = refroot / "REFERENCE.json"
    path.write_text(json.dumps(ref))
    sys.argv.extend(["--reference", str(path)])
    if field is None:
        assert run.main() == 0
    else:
        with pytest.raises(ValueError, match="reference.*mismatch"):
            run.main()
        assert not out.exists()


@pytest.mark.parametrize("source", ["product", "host", "instrument"])
def test_output_inside_source_refused_before_creation(runner_main, monkeypatch, source):
    out, host = runner_main
    instrument = host.parent / "instrument"
    monkeypatch.setattr(run, "HERE", instrument)
    root = {"product": host.parent / "product", "host": host, "instrument": instrument}[source]
    output = root / "nested/output"
    sys.argv[sys.argv.index("--out") + 1] = str(output)
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2 and not output.exists() and not out.exists()


def test_card_command_has_no_local_paths(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    flags = ["background-prep-off-replay", "--out", str(tmp_path / "out"),
             "--product-source=" + str(tmp_path / "product"), "--hermes-source", str(tmp_path / "host"),
             "--reference", str(tmp_path / "ref"), "--card-tool", str(tmp_path / "tool"),
             "--gap-seconds", "0", "--seeds", "1", "--smoke"]
    monkeypatch.setattr(sys, "argv", [str(run.HERE / "run.py"), *flags])
    args = SimpleNamespace(card_tool=tmp_path / "tool", product_source=tmp_path / "product",
                           runner=flags[0], seeds=[1])
    run.card(args, tmp_path, dict(verdict="PASS", bars={}, pooled_on=dict(rate=None), not_proven="release"))
    command = commands[0][commands[0].index("--command") + 1]
    assert str(tmp_path) not in command and str(run.HERE) not in command
    assert str(Path(sys.executable).parent) not in command
    assert "bench/instruments/compaction_probe/bgprep/run.py" in command
    assert args.runner in command and "--gap-seconds 0 --seeds 1 --smoke" in command
    assert all(f"<{p}>" in command for p in ("out", "product", "hermes", "reference", "card-tool"))


@pytest.mark.parametrize("timestamp", [2.0, None])
def test_timestamp_only_change_fails_integrity(timestamp):
    message = dict(role="user", content="same", timestamp=1.0)
    expected = replay.identity(message)
    stored = replay.identity({**message, "timestamp": timestamp})
    probe = dict(expected=[expected], rows=[dict(store_id=1, identity=stored)],
                 leaf_sources=[[1]], frontier=1, active_view=[], activity={})
    assert not coverage(probe)["ordered_identity_preserved"]
    keys = ("worker_starts", "worker_constructions", "enqueue_attempts", "jobs_enqueued",
            "worker_summary_calls", "foreground_summary_calls", "promotions", "promotion_lookups")
    probe["activity"] = dict.fromkeys(keys, 0)
    assert not score_cell(dict(probes=[probe], events=[]))["integrity"]


def test_carrier_timestamp_omission_does_not_hide_stored_timestamp_change():
    message = dict(role="user", content="carrier remainder", timestamp=1.0)
    active = replay.identity(message, timestamp=False)
    row = dict(store_id=1, identity=replay.identity(message), active_identity=active)
    probe = dict(expected=[row["identity"]], rows=[row], leaf_sources=[], frontier=0,
                 active_view=[active])
    assert intact(coverage(probe))
    row["identity"] = replay.identity({**message, "timestamp": None})
    assert not intact(coverage(probe))


def test_estimator_pinned_before_consumer_imports(tmp_path, monkeypatch):
    def original(text):
        return 999

    tokens = SimpleNamespace(count_tokens=original, _fallback_token_estimate=lambda text: len(text))

    class ImportObserved(Exception):
        pass

    def import_module(name):
        if name == "hermes_lcm.tokens":
            return tokens
        assert tokens.count_tokens is not original
        assert tokens.count_tokens("abcd") == 4 and tokens.count_tokens("") == 0
        raise ImportObserved

    monkeypatch.setattr(replay.importlib, "import_module", import_module)
    monkeypatch.setattr(replay.os, "environ", dict(replay.os.environ))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "hermes_lcm", SimpleNamespace())
    args = SimpleNamespace(hermes_source=tmp_path, product_source=tmp_path, real_route=True)
    with pytest.raises(ImportObserved):
        replay.run_cell(args, tmp_path, "on", 1)
