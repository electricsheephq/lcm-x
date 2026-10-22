"""Deterministic producer processes: no credentials, network or summariser."""
import copy
import json

import pytest

from test_d2_calibrated import load


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Producer:
    def __init__(self, failures=(), rate=False, low=False, rate_exit=False):
        self.failures, self.rate, self.low, self.rate_exit = failures, rate, low, rate_exit
        self.active, self.maximum, self.started = set(), 0, []

    def __call__(self, row, attempt, path):
        assert row["arm"] not in self.active
        assert (path.parents[2] / "pins.json").is_file()
        assert (path.parents[2] / "ledger.json").is_file()
        self.active.add(row["arm"])
        self.maximum = max(self.maximum, len(self.active))
        self.started.append((row["arm"], row["seed"], row["k"], row["slot"], attempt))
        ticks, done = 2, False
        owner = self
        if self.rate:
            (path / "producer.log").write_text("usage-limit reached\n")

        class Process:
            def poll(self):
                nonlocal ticks, done
                ticks -= 1
                if ticks > 0 and not done:
                    return None
                if not done:
                    owner.active.remove(row["arm"])
                    done = True
                return 3 if owner.rate_exit else 0

            def terminate(self):
                nonlocal ticks
                ticks = 0
                self.poll()

            kill = terminate

            def wait(self, timeout=None):
                self.terminate()

        return Process()

    def validate(self, row, attempt, path):
        if row["arm"] == "candidate" and row["k"] in self.failures and attempt == 1:
            return "invalid", "synthetic partial", None
        share = .1 if self.low and row["arm"] == "candidate" else .6
        return "valid", "complete", {str(cp): {"share": share, "facts": {"f": True}} for cp in (176, 304)}


def run(tmp_path, producer=None, option="B", aa=False, clock=None, **extra):
    driver = load("d2_driver")
    clock, producer = clock or Clock(), producer or Producer()
    pins = {"option": option, "aa": aa, "synthetic": True}
    result = driver.run_comparison(pins, tmp_path / "comparison", producer=producer,
                                   validate=producer.validate, clock=clock, sleep=clock.sleep, **extra)
    data = json.loads((tmp_path / "comparison/ledger.json").read_text())
    return result, data, producer


@pytest.mark.parametrize("option,aa,count", [("B", False, 54), ("B", True, 54), ("A", False, 24)])
def test_schedule_slots_concurrency_pins_and_artifacts(tmp_path, option, aa, count):
    result, data, producer = run(tmp_path, option=option, aa=aa)
    assert result["raw_verdict"] == "PASS"
    assert len(data["rows"]) == len(producer.started) == count
    assert producer.maximum == 2 and not producer.active
    paths = set()
    for row in data["rows"]:
        want = (1 if row["k"] % 2 else 2) if row["arm"] == "candidate" else (2 if row["k"] % 2 else 1)
        assert row["slot"] == want
        assert row["state"] == "valid" and row["started"] <= row["finished"]
        assert row["artifact"] not in paths
        paths.add(row["artifact"])
    assert [r[3] for r in producer.started] == [1, 2] * (count // 2)
    assert (tmp_path / "comparison/result.json").is_file()
    card = (tmp_path / "comparison/result.md").read_text()
    assert "PASS" in card and "disposition" in card and "mcnemar" in card


def test_replacement_keeps_denominator_and_both_attempts(tmp_path):
    producer = Producer(failures=(1,))
    result, data, _ = run(tmp_path, producer)
    # One invalid per seed; third is refused, first two remain valid replacements.
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert len(data["rows"]) == 54
    assert sum(len(r["attempts"]) - 1 for r in data["rows"] if r["attempts"]) == 2
    assert data["stop_reason"] == "replacement budget"


def test_one_invalid_replay_replaced_once(tmp_path):
    producer = Producer()
    original = producer.validate

    def validate(row, attempt, path):
        if row["arm"] == "candidate" and row["seed"] == row["k"] == attempt == 1:
            return "invalid", "synthetic partial", None
        return original(row, attempt, path)
    producer.validate = validate
    result, data, _ = run(tmp_path, producer)
    assert result["raw_verdict"] == "PASS"
    row = next(r for r in data["rows"] if len(r["attempts"]) == 2)
    assert [a["id"] for a in row["attempts"]] == [1, 2]
    assert [a["state"] for a in row["attempts"]] == ["replaced", "valid"]
    assert "partial" in row["attempts"][0]["validity_reason"]
    assert row["attempts"][0]["artifact"] != row["attempts"][1]["artifact"]


def test_no_second_replacement_for_one_slot(tmp_path):
    producer = Producer()
    producer.validate = lambda *args: ("invalid", "still invalid", None)
    result, data, _ = run(tmp_path, producer)
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert all(len(r["attempts"]) <= 2 for r in data["rows"])


@pytest.mark.parametrize("exit_only", [False, True])
def test_first_rate_marker_stops_and_never_replaces(tmp_path, exit_only):
    result, data, producer = run(tmp_path, Producer(rate=not exit_only, rate_exit=exit_only))
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert data["stop_reason"] == "rate_limited"
    assert len(producer.started) <= 2
    assert all(len(r["attempts"]) <= 1 for r in data["rows"])
    assert any(r["state"] == "rate_limited" for r in data["rows"])
    assert all(r["state"] not in ("planned", "running") for r in data["rows"])
    assert not producer.active


def test_elapsed_includes_replays_and_marks_unfinished(tmp_path):
    clock = Clock()
    clock.sleep = lambda seconds: setattr(clock, "now", clock.now + 21600)
    result, data, producer = run(tmp_path, clock=clock)
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert data["elapsed_s"] >= 21600
    assert len(producer.started) <= 2
    assert all(r["state"] not in ("planned", "running") for r in data["rows"])
    assert not producer.active


def test_elapsed_includes_setup_and_scoring(tmp_path):
    driver, scorer, clock = load("d2_driver"), load("d2_score"), Clock()

    def slow_score(data):
        result = scorer.score(copy.deepcopy(data))
        clock.now += 21600
        return result
    result, data, _ = run(tmp_path, clock=clock, scorer=slow_score)
    assert result["raw_verdict"] == "INCONCLUSIVE" and data["elapsed_s"] >= 21600
    assert driver.LIMIT_SECONDS == 21600


def test_valid_low_score_is_never_rerun(tmp_path):
    result, data, producer = run(tmp_path, Producer(low=True))
    assert result["raw_verdict"] == "INCONCLUSIVE"  # also trips cp176; never requests a replay
    assert len(producer.started) == 54
    assert all(len(r["attempts"]) == 1 for r in data["rows"])


def test_existing_comparison_refused_without_changes(tmp_path):
    run(tmp_path)
    before = {str(p): p.read_bytes() for p in (tmp_path / "comparison").rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert before == {str(p): p.read_bytes() for p in (tmp_path / "comparison").rglob("*") if p.is_file()}
