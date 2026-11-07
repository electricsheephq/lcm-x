"""Seeded true-prefix/continuation differential through real cold ingests (#2377).

LCM_DIFF_CASES defaults to 2,000. Each case uses random.Random(case_seed), so a
failure's seed alone reconstructs its conversation, projection, cursor and ledger.
I-c is separately skipped until the frozen main/candidate skip-set hook exists.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time

import pytest

from tests.harness import make_engine
from tests.invariants._ingest_shapes_2377 import U, V, call, ingest_shape

pytestmark = pytest.mark.issue(2377)


@dataclass
class Case:
    seed: int
    true: list
    source: list
    host: list
    cursor: int
    bindings: dict
    new: set
    replay: set


class DifferentialFailure(AssertionError):
    """Only a completed invariant census may satisfy the strict xfail."""


def generate(seed):
    """Every stored row belongs to a true prefix; projections never invent ownership.

    Each segment has 1-3 result rows with distinct parallel tool ids. Repeated
    occurrences reuse the same call identity and bytes. Stamp changes and result
    rewrites apply only to known replay occurrences; appended segments are new.
    No dropped/undo branch (the design's named non-goal) enters this generator.
    """
    rng = random.Random(seed)
    alphabet = []
    for identity in range(rng.randint(1, 3)):
        ids = tuple(f"id-{identity}-{i}" for i in range(rng.randint(1, 3)))
        alphabet.append((ids, json.dumps({"identity": identity})))

    def segment(identity, occurrence):
        ids, args = alphabet[identity]
        stamp = rng.choice((None, float(occurrence + 2)))
        return [call(stamp, ids, args), *[
            dict(role="tool", tool_call_id=cid, content=f"result {cid}",
                 timestamp=rng.choice((None, stamp))) for cid in ids]]

    old = [segment(rng.randrange(len(alphabet)), i) for i in range(rng.randint(1, 4))]
    new_segments = [segment(rng.randrange(len(alphabet)), i + len(old))
                    for i in range(rng.randint(1, 2))]
    source = [deepcopy(U), *[r for seg in old for r in seg]]
    true = [*deepcopy(source), *[r for seg in new_segments for r in seg], deepcopy(V)]
    host = deepcopy(source)
    boundaries, offset = [1], 1
    for seg in old:
        # Identity and occurrence stay fixed even when Hermes changes presentation.
        mode = rng.choice(("same", "absent", "restamped"))
        for row in host[offset:offset + len(seg)]:
            if mode == "absent":
                row.pop("timestamp", None)
            elif mode == "restamped":
                row["timestamp"] = float(100 + offset)
        if rng.randrange(3) == 0:
            at = offset + rng.randrange(1, len(seg))
            host[at]["content"] = f"rewritten occurrence {offset} result {at}"
        boundaries.extend((offset + 1, offset + len(seg)))  # inside call/results, or segment end
        offset += len(seg)
    host += deepcopy(true[len(source):])
    cursor = rng.choice(boundaries)
    bindings = {}
    # At most one prefix row, with its actual original occurrence, never an equal twin.
    eligible = [i for i in range(1, cursor) if host[i] == source[i]]
    if eligible and rng.choice((False, True)):
        at = rng.choice(eligible)
        bindings[at] = at
    new = set(range(len(source), len(host)))
    replay = set(range(cursor, len(source)))
    assert true[:len(source)] == source
    assert new.isdisjoint(replay) and new | replay == set(range(cursor, len(host)))
    return Case(seed, true, source, host, cursor, bindings, new, replay)


def execute(case, root):
    folder = root / f"seed-{case.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    engines = []
    def build():
        engine = make_engine(folder, window=200_000, session_id="S", conversation_id="conversation",
                             fresh_tail_count=9, leaf_chunk_tokens=400,
                             large_output_externalization_enabled=False,
                             large_output_active_replay_stubbing_enabled=False,
                             temporal_rollups_enabled=False, embeddings_enabled=False,
                             deferred_maintenance_enabled=False)
        engines.append(engine)
        return engine
    try:
        with pytest.MonkeyPatch.context() as patch:
            run = ingest_shape(build, patch, case.source, case.host, case.cursor, case.bindings)
            errors = run.violations(case.new)
            # I-b is about surplus replay rows, whose occurrences are known by construction.
            replay_ids = {run.inserted[i] for i in case.replay if i in run.inserted}
            assert set(errors["silent"]) <= replay_ids
            ceiling = getattr(run.engine, "_last_tool_skip_sets", None)
            if ceiling is not None:
                main_skips, v7_skips = ceiling
                assert set(v7_skips) <= set(main_skips), (case.seed, "I-c")
            return errors
    finally:
        for engine in engines:
            engine.shutdown()


@pytest.mark.parametrize("seed", [2377], ids=["seed-2377"])
@pytest.mark.xfail(strict=True, raises=DifferentialFailure, reason="#2377 v7: main loses new repeated occurrences and stores silent replay copies")
def test_seeded_differential_all_cases(tmp_path, seed, record_property):
    cases = int(os.environ.get("LCM_DIFF_CASES", "2000"))
    assert cases > 0
    started = time.perf_counter()
    failures = {"missing": [], "silent": [], "invalid": []}
    examples = {}
    green = []
    # A consecutive case-seed stream supports -k seed-2377 and exact regeneration.
    for case_seed in range(seed, seed + cases):
        errors = execute(generate(case_seed), tmp_path)
        if not any(errors.values()):
            green.append(case_seed)
        for invariant, values in errors.items():
            if values:
                failures[invariant].append(case_seed)
                if len(examples.get(invariant, [])) < 5:
                    examples.setdefault(invariant, []).append(dict(seed=case_seed, rows=values))
    result = dict(seed=seed, cases=cases, seconds=round(time.perf_counter() - started, 3),
                  failed_cases={k: len(v) for k, v in failures.items()},
                  first_five={k: v[:5] for k, v in failures.items()}, examples=examples, green_first_ten=green[:10])
    record_property("differential_result", json.dumps(result, sort_keys=True))
    print("\nDIFFERENTIAL_RESULT " + json.dumps(result, sort_keys=True))
    if any(failures.values()):
        raise DifferentialFailure(result)


# Pinned by the unmarked main census, including inside-segment cursors.
@pytest.mark.parametrize("seed", [2380, 2381, 2383, 2384, 2390], ids=lambda seed: f"seed-{seed}")
def test_seeded_differential_green_subset(tmp_path, seed):
    errors = execute(generate(seed), tmp_path)
    assert not any(errors.values()), (seed, errors)


@pytest.mark.skip(reason="#2377 v7: needs implementation hook engine._last_tool_skip_sets")
def test_seeded_differential_ceiling_subset(tmp_path):
    case = generate(2377)
    folder = Path(tmp_path) / "ceiling"
    folder.mkdir()
    engines = []
    def build():
        engine = make_engine(folder, window=200_000, session_id="S", conversation_id="conversation",
                             embeddings_enabled=False, temporal_rollups_enabled=False,
                             deferred_maintenance_enabled=False)
        engines.append(engine)
        return engine
    try:
        with pytest.MonkeyPatch.context() as patch:
            run = ingest_shape(build, patch, case.source, case.host, case.cursor, case.bindings)
            main_skips, v7_skips = run.engine._last_tool_skip_sets
            assert set(v7_skips) <= set(main_skips)
    finally:
        for engine in engines:
            engine.shutdown()
