"""Explicit grid tests for capturing and replaying synthetic pre-pass checkpoints."""
import hashlib
import json
import os
import random
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from shutil import copytree

import pytest

from hermes_lcm import escalation
from hermes_lcm.externalize import DEFAULT_LARGE_OUTPUT_DIRNAME
from tests.lineage_guard_sessions import Sim, simulate
from tests.test_issue_2890_lineage_guard import _content_digest, _covered_digests, _digest, _nodes, _stored, _summary

HELPERS = (_content_digest, _covered_digests, _nodes, _stored)
CASES = {
    7: (9, 8), 59: (12,), 69: (14,), 75: (5,), 77: (3, 5), 78: (9,), 80: (9,),
    84: (6, 4), 85: (14,), 87: (13,), 95: (7,), 105: (11,), 117: (6,), 125: (2,),
    131: (8,), 144: (5,), 150: (3,), 158: (4,), 170: (9,), 174: (6,), 199: (3,),
    211: (5,), 214: (3,), 216: (6,), 252: (7,), 279: (4,), 280: (7,), 283: (19,),
}
# Old grid's first event divergence, or first loss if schedules agree. Also
# replay the first loss itself when those indices differ. Each index is zero-based.
SEEDS = tuple(CASES)


def _file_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def _input_digest(engine, view):
    return _digest([view, _stored(engine), [asdict(node) for node in _nodes(engine)]])


@pytest.mark.parametrize("seed", SEEDS)
def test_capture(seed, tmp_path, monkeypatch):
    root = Path(os.environ["LCMX_2890_CHECKPOINTS"])
    indices = CASES[seed]
    original = Sim.run_pass
    index = 0

    def capture(sim, before, kind, tokens, step, events):
        nonlocal index
        if index in indices:
            case = root / f"{seed}-{index}"
            case.mkdir()
            state = case / "state"
            state.mkdir()
            with (closing(sqlite3.connect(sim.root / "lcm.db")) as source,
                  closing(sqlite3.connect(state / "lcm.db")) as target):
                source.backup(target)
            payloads = sim.root / DEFAULT_LARGE_OUTPUT_DIRNAME
            if payloads.is_dir():
                copytree(payloads, state / payloads.name)
            data = dict(config=sim.cfg, window=sim.window, view=before, tokens=tokens, kind=kind, step=step,
                        events=events, clock=time.time(), input_digest=_input_digest(sim.engine, before),
                        files=_file_hashes(state))
            (case / "input.json").write_text(json.dumps(data) + "\n")
        index += 1
        return original(sim, before, kind, tokens, step, events)

    monkeypatch.setattr(Sim, "run_pass", capture)
    simulate(seed, tmp_path, monkeypatch, _summary, HELPERS, edits=True, stop_after=max(indices) + 1)
    assert all((root / f"{seed}-{index}" / "input.json").is_file() for index in indices)


@pytest.mark.parametrize(("seed", "index"), [(seed, i) for seed, indices in CASES.items() for i in indices])
def test_replay(seed, index, tmp_path, monkeypatch):
    case = Path(os.environ["LCMX_2890_CHECKPOINTS"]) / f"{seed}-{index}"
    raw = (case / "input.json").read_bytes()
    data = json.loads(raw)
    state = tmp_path / "state"
    copytree(case / "state", state)
    assert data["files"] == _file_hashes(state)
    clock = data["clock"]

    def now():
        nonlocal clock
        clock += 0.5
        return clock

    monkeypatch.setattr(time, "time", now)
    outage = random.Random(seed).random() < 0.15

    def summarize(*args, **kwargs):
        return None if outage else _summary(*args, **kwargs)

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summarize)
    sim = Sim(random.Random(seed), state, data["window"], data["config"], True, HELPERS)
    try:
        view = data["view"]
        assert _input_digest(sim.engine, view) == data["input_digest"]
        result = sim.run_pass(view, data["kind"], data["tokens"], data["step"], data["events"])
        result["checkpoint"] = hashlib.sha256(raw).hexdigest()
        result["input_digest"] = data["input_digest"]
        output = Path(os.environ["LCMX_2890_REPLAYS"]) / f"{seed}-{index}.json"
        output.write_text(json.dumps(result) + "\n")
    finally:
        sim.engine.shutdown()
