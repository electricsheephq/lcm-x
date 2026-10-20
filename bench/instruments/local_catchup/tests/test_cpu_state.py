from bench.instruments.local_catchup.catchup import state
from bench.instruments.local_catchup.cpu import Supervisor, percentile, rolling, statistics


def test_nearest_rank_and_rolling():
    assert percentile(list(range(1, 101))) == 95
    samples = [(i, 1, .5) for i in range(1, 121)]
    assert len(rolling(samples)) == 61
    assert rolling(samples) == [.5]*61
    assert statistics(samples)["p95_rolling60_cores"] == .5
    assert statistics(samples)["mean_cores"] == .5
    assert rolling([(30, 30, 1), (60, 30, 0), (90, 30, 0)]) == [(30-i)/60 for i in range(31)]


def test_abort_cpu_and_swap(monkeypatch, tmp_path):
    from bench.instruments.local_catchup import cpu
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    tick = iter(range(12))
    monkeypatch.setattr(cpu.time, "monotonic", lambda: next(tick))
    values = iter((i*13., 100, 2) for i in range(12))
    monkeypatch.setattr(cpu, "process", lambda _: next(values))
    meter = Supervisor(tmp_path/"cpu.jsonl")
    for _ in range(10):
        assert meter.sample([1]) is None
    assert meter.sample([1]) == "sustained_cpu_breach"
    monkeypatch.setattr(cpu, "swap", lambda: 1)
    assert meter.sample([1]) == "swap_increased"


def test_state_machine():
    assert state({"complete": True}, 1, 0, True, 1200) == "COMPLETE"
    assert state({"complete": False}, 700, 600, True, 1200) == "STALLED"
    assert state({"complete": False}, 700, 600, False, 1200) == "RUNNING"
    assert state({"complete": False}, 1200, 0, True, 1200) == "INCOMPLETE"
