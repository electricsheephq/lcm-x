"""Build the kit's synthetic store by driving one tree's public ingest (the v0.30.0 tree for the kit store).

    python -B -m bench.instruments.gate_kit.live_rows.build --tree TREE --out NEW_DIR [--kit kit.json] [--scale 0.1]

Writes NEW_DIR/store/lcm.db (+ externalized payloads) and NEW_DIR/build-manifest.json: the kit's input hash, the
generated counts, a census of the stored rows (counts, ids and a digest; no content) and the store's sha256 after a
WAL checkpoint. `--scale` shrinks every turn count for a local smoke; a scaled store never qualifies.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from bench.instruments.gate_kit.live_rows import rt, shape  # noqa: E402

CHUNK_ROWS = 600


def scaled(kit: dict[str, Any], scale: float) -> dict[str, Any]:
    if scale == 1.0:
        return kit
    kit = copy.deepcopy(kit)
    kit["target"]["turns"] = max(8, int(kit["target"]["turns"] * scale))
    kit["others"]["turns"] = max(4, int(kit["others"]["turns"] * scale))
    kit["others"]["conversations"] = max(1, int(kit["others"]["conversations"] * scale))
    kit["target"]["context_length"] = max(32_000, int(kit["target"]["context_length"] * scale))
    kit["scale"] = scale
    return kit


def ingest_history(engine_module: Any, config: Any, kit: dict[str, Any], home: Path, rows: list[dict[str, Any]],
                   session: str, conversation: str) -> dict[str, Any]:
    engine = rt.engine(engine_module, config, kit, home, session=session, conversation=conversation)
    calls, started = 0, time.monotonic()
    try:
        for end in range(CHUNK_ROWS, len(rows) + CHUNK_ROWS, CHUNK_ROWS):
            engine.ingest(copy.deepcopy(rows[:min(end, len(rows))]))
            calls += 1
    finally:
        engine.shutdown()
    return {"session": session, "rows": len(rows), "ingest_calls": calls, "seconds": round(time.monotonic() - started, 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kit", type=Path, default=rt.KIT)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    kit = scaled(rt.load_kit(args.kit), args.scale)
    rt.isolate(args.out / "home")
    config_module, engine_module, _lines = rt.load(args.tree, kit)
    rt.aux.configure(floor_s=0.05, rate_tps=1e9)  # building only: earlier passes need no realistic latency
    db = args.out / "store" / "lcm.db"
    db.parent.mkdir()
    config = rt.config_for(config_module, db)
    generated = shape.build(kit)
    started = time.monotonic()
    report = [ingest_history(engine_module, config, kit, args.out / "home", rows, f"other{index}-s",
                             f"other{index}-conv") for index, rows in enumerate(generated["others"])]
    target = kit["target"]
    report.append(ingest_history(engine_module, config, kit, args.out / "home", generated["target_history"],
                                 target["session_id"], target["conversation_id"]))
    # Earlier restarts: each binds the host's cold-rebuilt list once (no pass), as the reference store's earlier
    # builds did; on the v0.30.0 tree such a bind stores copies of unstamped rows (#1674), which the store keeps.
    for number in range(int(target.get("cold_restarts", 0))):
        begun = time.monotonic()
        engine = rt.engine(engine_module, config, kit, args.out / "home")
        passed = None
        try:
            engine.ingest(copy.deepcopy(generated["restart_list"]))
            if target.get("pass_at_restart"):  # each earlier start also ran a pass (fast stand-in; list not adopted)
                view = generated["restart_list"]
                passed = len(engine.compress(copy.deepcopy(view), current_tokens=sum(
                    len(str(r.get("content") or "")) for r in view) // shape.CHARS_PER_TOKEN) or [])
        finally:
            engine.shutdown()
        report.append({"cold_restart": number + 1, "seconds": round(time.monotonic() - begun, 2), "pass_rows_out": passed,
                       "stored": rt.store_census(db)["rows"]})
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    externalized = db.parent / "externalized"
    manifest = {
        "kit": str(args.kit.name), "kit_body": kit, "scale": args.scale, "qualifies": args.scale == 1.0,
        "seed": kit["seed"], "input_hash": generated["input_hash"], "generated": generated["counts"],
        "stored": rt.store_census(db),
        "externalized_files": sum(1 for p in externalized.rglob("*") if p.is_file()) if externalized.exists() else 0,
        "tree_identity": rt.driver.tree_identity(args.tree.resolve()), "ingest": report,
        "build_seconds": round(time.monotonic() - started, 1), "store_sha256": rt.sha256_file(db),
    }
    (args.out / "build-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({k: manifest[k] for k in ("input_hash", "generated", "stored", "externalized_files",
                                                "build_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
