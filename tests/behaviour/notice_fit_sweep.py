"""#2896's two-tree sweep: the same synthetic no-system-row fits on one tree, written as JSON for a row-by-row compare.

    python3 -I -B notice_fit_sweep.py --tree <tree> --sweep band|d1|random --out <json> --workdir <dir>
                                      [--syspath <dir> ...] [--limit <n>]

The engine is imported ONLY from ``--tree`` (registered as ``hermes_lcm``, as catchup_grid_runner.py does); the cases
come from this tree's tests/notice_fit_cases.py, which uses only engine surface main has too. Sweeps:
  band    the band sweep: [C(summary + U0), A0, ...] at 2, 3 and 4 turns, every third cap up to 1,500 below the list.
  d1      the direct-fit sweep: carrier, carrier_sep2, two_carriers and twofits, 6 seeded cases x 45 caps each.
  random  the randomised differential: every direct-fit shape plus pasted blocks and ignore patterns; per case a
          fit bound as the exception path binds it, then the host's replay (as returned, merged, a new turn, session
          end) and a cold process.
  review  the full-diff review's scenarios R1 to R6 (one record each, keyed by scenario and cap).
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
            e._bind_survival_fit_emission(active, result)  # as the exception path binds it (#1000)
            generated = list(record["fit"]["generated"])
            if shape == "twofits" and record["fit"]["fitted"]:
                grown, cap2 = _grown(cases, e, result, rng)
                result, record["second"] = cases.fit(e, grown, cap2, after_exception)
                e._bind_survival_fit_emission(grown, result)
                generated += record["second"]["generated"]
            new = [{"role": "user", "content": rng.choice(["[NEW] newest question", "ok thanks", "heartbeat",
                                                           "hello\n\nworld"]), "timestamp": 9_000.0},
                   {"role": "assistant", "content": "[NEWA] answer"}]
            merged = cases.merge(result)
            stages = {}
            e.ingest(result)
            stages["returned"] = cases.stored(e, generated)
            e.ingest(merged)
            stages["merged"] = cases.stored(e, generated)
            e.ingest([*merged, *new])
            stages["new"] = cases.stored(e, generated)
            e.on_session_end("S", [*merged, *new])
            stages["end"] = cases.stored(e, generated)
            record["counts_warm"] = cases.store_counts(e)
        finally:
            e.shutdown()
        cold = cases.engine(path, patterns)
        try:
            cold.ingest([*merged, *new])
            stages["cold"] = cases.stored(cold, generated)
            record["counts_cold"] = cases.store_counts(cold)
        finally:
            cold.shutdown()
        record["stages"] = stages
        records.append(record)
    return records


def _replay(cases, e, result, generated, new_text="[NEW] newest question"):
    """The host's replay of a fitted list: as returned, merged, a new turn, session end. The stored rows after each."""
    merged = cases.merge(result)
    new = [{"role": "user", "content": new_text, "timestamp": 9_000.0}, {"role": "assistant", "content": "[NEWA] ok"}]
    stages = {}
    for stage, rows in (("returned", result), ("merged", merged), ("new", [*merged, *new])):
        e.ingest(rows)
        stages[stage] = cases.stored(e, generated)
    return stages


def _map_of(e, rows):
    ids = e._get_store_id_map_for_messages(rows)
    return [ids.get(id(row)) for row in rows]


def review(cases, work: Path, limit: int):
    records = []
    # R1: a second fit at one token under the first fit's cap, the first fit's rows persisted
    for cap in (1108, 1150, 1200)[:limit or None]:
        e = cases.engine(_dir(work, f"r1-{cap}"))
        try:
            first, one = cases.fit(e, cases.build_case(e, random.Random(0), "carrier"), cap, True)
            e.ingest(first)
            _, two = cases.fit(e, first, cap - 1, False)
            records.append({"key": f"r1/{cap}", "first": one, "second": two})
        finally:
            e.shutdown()
    # R2: the complete carrier summary + U stored right after the distinct row U, the carrier ahead of an objective
    for seed in range(limit or 3):
        for room in (60, 200):
            rng = random.Random(seed)
            e = cases.engine(_dir(work, f"r2-{seed}-{room}"))
            try:
                raw = [m for i in range(6) for m in (
                    {"role": "user", "content": f"[U{i}] " + cases.words(rng, rng.randint(40, 200)),
                     "timestamp": 10.0 * i},
                    {"role": "assistant", "content": f"[A{i}] " + cases.words(rng, rng.randint(40, 300))})]
                summary = cases.part(e, "summary " + cases.words(rng, 30))
                carrier = {"role": "user", "content": f"{summary}\n\n{raw[0]['content']}"}
                for message in [raw[0], carrier, *raw[1:]]:  # a host stored the composite too
                    e._store.append("S", dict(message), conversation_id="conv")
                active = [carrier, {"role": "user", "content": f"{cases.OBJ}\nfinish it"}, *[dict(m) for m in raw[1:]]]
                before = _map_of(e, active)
                result, record = cases.fit(e, active, e._survival_measure(active) - room, True)
                e._bind_survival_fit_emission(active, result)
                emissions = (getattr(e, "_last_emission_descriptors", None) or {}).get("emissions") or []
                records.append({"key": f"r2/{seed}/{room}", "fit": record, "map_before": before[0],
                                "map_after": _map_of(e, result)[0],
                                "retained": [(d.get("retained_source") or {}).get("store_id") for d in emissions],
                                "stages": _replay(cases, e, result, record["generated"])})
            finally:
                e.shutdown()
    # R3: [carrier A, carrier B, objective] on the exception path, no prior proof
    for case in range(limit or 3):
        seed = cases.SHAPES.index("two_carriers") * 1000 + case
        e = cases.engine(_dir(work, f"r3-{case}"))
        try:
            built = cases.build_case(e, random.Random(seed), "two_carriers")
            active = [*built[:2], {"role": "user", "content": f"{cases.OBJ}\nfinish it"}, *built[2:]]
            before = _map_of(e, active)
            result, record = cases.fit(e, active, e._survival_measure(active) - 60, True)
            e._bind_survival_fit_emission(active, result)
            records.append({"key": f"r3/{case}", "fit": record, "map_before": before[:2],
                            "map_after": _map_of(e, result)[:2], "stages": _replay(cases, e, result, record["generated"])})
        finally:
            e.shutdown()
    # R4: patterns "session_id=S " and "confirm\Z"; a fresh turn ending in confirm plus a pasted counted notice,
    # merged behind the fitted carrier
    for cap in (1200, 1400)[:limit or None]:
        e = cases.engine(_dir(work, f"r4-{cap}"), [r"session_id=S ", r"confirm\Z"])
        try:
            result, record = cases.fit(e, cases.build_case(e, random.Random(0), "carrier"), cap, True)
            fresh = "[FRESH] please confirm" + cases.PASTED
            merged = {"role": "user", "content": result[0]["content"] + "\n\n" + fresh, "timestamp": 9e3}
            ignored = e._matches_ignore_message_patterns(merged)
            e.ingest([merged, {"role": "assistant", "content": "[FRESHA] ok"}])
            kept = any("[FRESH] please confirm" in json.dumps(r["content"])
                       for r in e._store.get_session_messages(e._session_id, limit=100_000))
            records.append({"key": f"r4/{cap}", "fit": record, "ignored": ignored, "fresh_stored": kept})
        finally:
            e.shutdown()
    # R5: a leading system row and [marked carrier, objective, old turns, newest]; caps that keep the head
    for seed in range(limit or 3):
        rng = random.Random(100 + seed)
        e = cases.engine(_dir(work, f"r5-{seed}"))
        try:
            system = {"role": "system", "content": "synthetic system prompt"}
            raw = [m for i in range(5) for m in (
                {"role": "user", "content": f"[U{i}] " + cases.words(rng, rng.randint(40, 200)), "timestamp": 10.0 * i},
                {"role": "assistant", "content": f"[A{i}] " + cases.words(rng, rng.randint(40, 300))})]
            raw.append({"role": "user", "content": "[NEWEST] " + cases.words(rng, 30), "timestamp": 99.0})
            e.ingest([system, *raw])
            summary = cases.part(e, "summary " + cases.words(rng, 30))
            nodes = e._verified_lcm_summary_prefix(summary)[1]
            carrier = {"role": "user", "content": summary + cases.marked_block(summary, nodes, cases.OLD_NOTICE)
                       + "\n\n" + raw[0]["content"]}
            active = [dict(system), carrier, {"role": "user", "content": f"{cases.OBJ}\nfinish it"},
                      *[dict(m) for m in raw[1:]]]
            result, record = cases.fit(e, active, e._survival_measure(active) - 80, False)
            records.append({"key": f"r5/{seed}", "fit": record, "head": [json.dumps(m, sort_keys=True, default=str)
                                                                       for m in result[:3]]})
        finally:
            e.shutdown()
    # R6 (the harness): a user-pasted marker kept in view, at a cap that cuts and at one that is a no-op
    for cap in (900, 100_000)[:limit or None]:
        e = cases.engine(_dir(work, f"r6-{cap}"))
        try:
            active = cases.build_case(e, random.Random(0), "carrier")
            active[-2] = {**active[-2], "content": active[-2]["content"] + cases.PASTED}
            _, record = cases.fit(e, active, cap, True)
            records.append({"key": f"r6/{cap}", "fit": record,
                            "raw_markers": sum(str(m.get("content")).count(cases.OPEN) for m in _)})
        finally:
            e.shutdown()
    return records


def _dir(work: Path, name: str) -> Path:
    path = work / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--sweep", required=True, choices=["band", "d1", "random", "review"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--syspath", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    logging.disable(logging.WARNING)
    cases = load(tree, args.syspath)
    work = Path(args.workdir)
    records = {"band": band, "d1": d1, "random": randomised, "review": review}[args.sweep](cases, work, args.limit)
    modules = sorted({str(Path(m.__file__).resolve()) for name, m in sys.modules.items()
                      if name.startswith("hermes_lcm.") and getattr(m, "__file__", None)})
    Path(args.out).write_text(json.dumps({"tree": str(tree), "modules": modules, "records": records}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
