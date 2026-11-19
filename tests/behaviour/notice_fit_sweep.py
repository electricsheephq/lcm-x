"""#2896's two-tree sweep: the same synthetic no-system-row fits on one tree, written as JSON for a row-by-row compare.

    python3 -I -B notice_fit_sweep.py --tree <tree> --sweep band|d1|random --out <json> --workdir <dir>
                                      [--syspath <dir> ...] [--limit <n>]

The engine is imported ONLY from ``--tree`` (registered as ``hermes_lcm``, as catchup_grid_runner.py does); the cases
come from this tree's tests/notice_fit_cases.py, which uses only engine surface main has too. Sweeps:
  band    the band sweep: [C(summary + U0), A0, ...] at 2, 3 and 4 turns, every third cap up to 1,500 below the list.
  d1      the direct-fit sweep: carrier, carrier_sep2, two_carriers and twofits, 6 seeded cases x 45 caps each.
  random  the randomised differential: every direct-fit shape plus pasted blocks and ignore patterns; per case a
          fit, then the host's replay (as returned, merged, a new turn, session end) and a cold process.
Synthetic data only; one engine and one store per cap, under --workdir. ``--limit``: at most n caps (or cases) per
shape, for a smoke run of the script itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D1_SHAPES = ["carrier", "carrier_sep2", "two_carriers", "twofits"]
RANDOM_CASES = 240  # iteration cap: one engine pair (warm, cold) per case


def load(tree: Path, extra_paths: list[str]):
    sys.dont_write_bytecode = True
    for entry in extra_paths:
        if entry not in sys.path:
            sys.path.append(entry)
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", str(tree / "__init__.py"), submodule_search_locations=[str(tree)])
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(tree)]
    package.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = package  # not executed: it registers with the host
    cases_spec = importlib.util.spec_from_file_location("notice_fit_cases", str(HERE.parent / "notice_fit_cases.py"))
    cases = importlib.util.module_from_spec(cases_spec)
    cases_spec.loader.exec_module(cases)
    cases.install_host_estimator()
    return cases


def band(cases, work: Path, limit: int):
    records = []
    for nturns, sep in cases.BAND_SHAPES:
        probe = cases.engine(_dir(work, f"band-{nturns}-{len(sep)}-probe"))
        try:
            full = probe._survival_measure(cases.build_band(probe, nturns, sep))
        finally:
            probe.shutdown()
        for cap in cases.band_caps(full)[:limit or None]:
            e = cases.engine(_dir(work, f"band-{nturns}-{len(sep)}-{cap}"))
            try:
                _, record = cases.fit(e, cases.build_band(e, nturns, sep), cap, True)
            finally:
                e.shutdown()
            records.append({"key": f"band/{nturns}/{len(sep)}/{cap}", **record})
    return records


def d1(cases, work: Path, limit: int):
    records = []
    for shape in D1_SHAPES:
        for case in range(6):
            seed = cases.SHAPES.index(shape) * 1000 + case
            probe = cases.engine(_dir(work, f"{shape}-{case}-probe"))
            try:
                full = probe._survival_measure(cases.build_case(probe, random.Random(seed), shape))
            finally:
                probe.shutdown()
            for k, cap in enumerate(cases.d1_caps(seed, full)[:limit or None]):
                after_exception = bool((case + k) % 2)
                e = cases.engine(_dir(work, f"{shape}-{case}-{k}"))
                try:
                    result, record = cases.fit(e, cases.build_case(e, random.Random(seed), shape), cap, after_exception)
                    if shape == "twofits" and record["fitted"]:
                        grown, cap2 = _grown(cases, e, result, random.Random(seed + k))
                        record["second"] = cases.fit(e, grown, cap2, after_exception)[1]
                finally:
                    e.shutdown()
                records.append({"key": f"d1/{shape}/{case}/{cap}", **record})
    return records


def _grown(cases, e, result, rng):
    more = []
    for i in range(3):
        user = {"role": "user", "content": f"[G{i}] " + cases.words(rng, rng.randint(5, 200)), "timestamp": 900.0 + i}
        reply = {"role": "assistant", "content": f"[GA{i}] " + cases.words(rng, rng.randint(5, 300))}
        for message in (user, reply):
            e._store.append("S", dict(message), conversation_id="conv")
        more += [user, reply]
    grown = [*result, *more]
    plain = [{**m, "content": cases.BLOCK_RE.sub("", m["content"])} if isinstance(m.get("content"), str) else m
             for m in grown]  # the cap from main's bytes, so both trees fit against the same cap
    return grown, max(50, e._survival_measure(plain) - rng.randint(50, 1500))


def randomised(cases, work: Path, limit: int):
    records = []
    for n in range(limit or RANDOM_CASES):
        rng = random.Random(20_000 + n)
        shape = cases.SHAPES[n % len(cases.SHAPES)]
        patterns = rng.choice([(), (), (r"^ok thanks$",), (r"LCM survival fit",), (r"\Aheartbeat\Z", r"hello\n\nworld")])
        pasted = rng.random() < 0.3
        path = _dir(work, f"random-{n}")
        e = cases.engine(path, patterns)
        record = {"key": f"random/{n}/{shape}", "patterns": list(patterns), "pasted": pasted}
        try:
            active = cases.build_case(e, random.Random(30_000 + n), shape)
            if pasted:  # a user who pastes a digest-shaped block into a turn
                at = rng.choice([i for i, m in enumerate(active) if m["role"] == "user" and isinstance(m["content"], str)
                                 and not m["content"].startswith("[Recent")] or [len(active) - 1])
                active[at] = {**active[at], "content": active[at]["content"] + cases.PASTED}
            full = e._survival_measure(active)
            cap = rng.randint(max(50, full // 5), full - 1)
            after_exception = rng.random() < 0.5
            result, record["fit"] = cases.fit(e, active, cap, after_exception)
            if shape == "twofits" and record["fit"]["fitted"]:
                grown, cap2 = _grown(cases, e, result, rng)
                result, record["second"] = cases.fit(e, grown, cap2, after_exception)
            new = [{"role": "user", "content": rng.choice(["[NEW] newest question", "ok thanks", "heartbeat",
                                                           "hello\n\nworld"]), "timestamp": 9_000.0},
                   {"role": "assistant", "content": "[NEWA] answer"}]
            merged = cases.merge(result)
            stages = {}
            e.ingest(result)
            stages["returned"] = cases.stored(e)
            e.ingest(merged)
            stages["merged"] = cases.stored(e)
            e.ingest([*merged, *new])
            stages["new"] = cases.stored(e)
            e.on_session_end("S", [*merged, *new])
            stages["end"] = cases.stored(e)
            record["counts_warm"] = cases.store_counts(e)
        finally:
            e.shutdown()
        cold = cases.engine(path, patterns)
        try:
            cold.ingest([*merged, *new])
            stages["cold"] = cases.stored(cold)
            record["counts_cold"] = cases.store_counts(cold)
        finally:
            cold.shutdown()
        record["stages"] = stages
        records.append(record)
    return records


def _dir(work: Path, name: str) -> Path:
    path = work / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--sweep", required=True, choices=["band", "d1", "random"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--syspath", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    logging.disable(logging.WARNING)
    cases = load(tree, args.syspath)
    work = Path(args.workdir)
    records = {"band": band, "d1": d1, "random": randomised}[args.sweep](cases, work, args.limit)
    modules = sorted({str(Path(m.__file__).resolve()) for name, m in sys.modules.items()
                      if name.startswith("hermes_lcm.") and getattr(m, "__file__", None)})
    Path(args.out).write_text(json.dumps({"tree": str(tree), "modules": modules, "records": records}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
