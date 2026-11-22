"""Build the kit's synthetic store by driving one tree's public ingest (the v0.30.0 tree for the kit store).

    python -B -m bench.instruments.gate_kit.live_rows.build --tree TREE --out NEW_DIR [--kit kit.json] [--scale 0.1]

Writes NEW_DIR/store/lcm.db (+ externalized payloads) and NEW_DIR/build-manifest.json: the kit's input hash, the
generated counts, a census of the stored rows (counts, ids and a digest; no content) and the store's sha256 after a
WAL checkpoint. `--scale` shrinks every turn count for a local smoke; a scaled store never qualifies. `--extend BUILD --tree T
--env K=V --restarts N` adds later restarts by another tree (recorded in the manifest, with the new store hash).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
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


MARKER = ".build-store-sha256"


def marker(work: Path) -> str | None:
    """The store hash of the build last restored (or snapshotted) at this working path."""
    path = work / "current" / MARKER
    return path.read_text().strip() if path.exists() else None


def set_aside(path: Path) -> None:
    """Move a stale working file or folder out of the way under a unique name (never deleted, never reused)."""
    if path.exists() or path.is_symlink():
        path.rename(path.with_name(f"{path.name}.stale-{time.time_ns()}"))


def clear_working(work: Path, build: Path | None = None) -> None:
    """Before a build or a restore: no store, payload, host store or sidecar of another build stays in use. A
    payload folder holding only names the build also has is overwritten in place (no copy per run); one with any
    other file is set aside. A host store the build replaces is overwritten in place; its sidecars are set aside."""
    for name in ("lcm.db", "lcm.db-wal", "lcm.db-shm", "lcm.db-journal", MARKER):
        (work / "current" / name).unlink(missing_ok=True)
    payloads, source = work / "current" / "externalized", (build / "store" / "externalized") if build else None
    if payloads.exists() and not payloads.is_symlink():
        mine = {p.relative_to(payloads) for p in payloads.rglob("*") if p.is_file()}
        theirs = {p.relative_to(source) for p in source.rglob("*") if p.is_file()} if source and source.exists() else set()
        if not mine <= theirs:
            set_aside(payloads)
    else:
        set_aside(payloads)
    for name in ("state.db-wal", "state.db-shm", "state.db-journal"):
        set_aside(work / "home" / name)
    if not (build and (build / "host" / "state.db").exists()):
        set_aside(work / "home" / "state.db")


def snapshot(work: Path, out: Path) -> None:
    """Checkpoint the store at the working path and copy it, its payloads and the host store into the build."""
    live = work / "current" / "lcm.db"
    with closing(sqlite3.connect(live)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    (out / "store").mkdir(parents=True, exist_ok=True)
    (out / "host").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(live, out / "store" / "lcm.db")
    if (live.parent / "externalized").exists():
        shutil.copytree(live.parent / "externalized", out / "store" / "externalized", dirs_exist_ok=True)
    if (work / "home" / "state.db").exists():
        with closing(sqlite3.connect(work / "home" / "state.db")) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        shutil.copyfile(work / "home" / "state.db", out / "host" / "state.db")
    (work / "current" / MARKER).write_text(rt.sha256_file(out / "store" / "lcm.db"))


def restore(build: Path, work: Path) -> Path:
    """Exactly the build's store, payloads and host store at the working path: whatever another build or run left
    there is set aside first, and the marker records which build's store this copy is."""
    work = rt.plain_dir(work)
    current = work / "current"
    current.mkdir(parents=True, exist_ok=True)
    (work / "home").mkdir(parents=True, exist_ok=True)
    clear_working(work, build)
    shutil.copyfile(build / "store" / "lcm.db", current / "lcm.db")
    if (build / "store" / "externalized").exists():
        shutil.copytree(build / "store" / "externalized", current / "externalized", dirs_exist_ok=True)
    if (build / "host" / "state.db").exists():
        shutil.copyfile(build / "host" / "state.db", work / "home" / "state.db")
    sha = json.loads((build / "build-manifest.json").read_text()).get("store_sha256")
    if sha and rt.sha256_file(current / "lcm.db") == sha:  # the copy is the store the manifest names
        (current / MARKER).write_text(sha)
    return current / "lcm.db"


def run_lock(work: Path) -> int:
    from bench.instruments.gate_kit.live_rows.run import claim
    return claim(work)


def extend(args: argparse.Namespace) -> int:
    """Later restarts by a newer tree on an existing build (the reference store's last starts ran the hotfix line
    with the opt-in mode, which wrote its host-uid bindings): each binds the restart list and runs one pass."""
    manifest_path = args.extend / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    kit = manifest["kit_body"]
    env = dict(item.split("=", 1) for item in args.env)
    work = rt.plain_dir(Path(manifest["work"]))
    lock = run_lock(work)  # noqa: F841 - held until the process exits, as a run's is
    rt.isolate(work / "home", env)
    config_module, engine_module, _lines = rt.load(args.tree, kit)
    rt.aux.configure(floor_s=0.05, rate_tps=1e9)
    db = restore(args.extend, work)
    config = rt.config_for(config_module, db)
    view = shape.build(kit)["restart_list"]
    steps = []
    for number in range(args.restarts):
        begun = time.monotonic()
        engine = rt.engine(engine_module, config, kit, work / "home")
        outs = []
        try:
            engine.ingest(copy.deepcopy(view))
            # Earlier passes after this start (the reference target's 33 summaries and non-zero frontier came from
            # its own earlier compactions).
            # Earlier passes as a host runs them: each is given the list the previous one returned (the host
            # adopts it), and a turn ends between them (a hold the product arms lasts until turn end). They stop
            # at the first pass that publishes nothing. The later restart's cold-rebuilt list is the kit's.
            current = copy.deepcopy(view)
            nodes_before = None
            for _ in range(max(1, args.passes)):
                tokens = sum(len(str(r.get("content") or "")) for r in current) // shape.CHARS_PER_TOKEN
                usage = {"prompt_tokens": tokens, "completion_tokens": 64, "total_tokens": tokens + 64}
                with closing(sqlite3.connect(rt.ro_uri(db), uri=True)) as conn:
                    nodes_before = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
                out = engine.compress(copy.deepcopy(current), current_tokens=tokens)
                current = out if isinstance(out, list) else current
                outs.append(len(current))
                engine.update_from_response(dict(usage))
                try:
                    engine.on_turn_complete(copy.deepcopy(current), usage=dict(usage), turn_id="build",
                                            task_id="build", api_call_count=1, interrupted=False, failed=False,
                                            turn_exit_reason="completed")
                except TypeError:
                    engine.on_turn_complete(copy.deepcopy(current))
                note = getattr(engine, "note_turn_complete", None)
                if callable(note):  # the host's post-LLM-call hook ends turn-paced holds through this notification
                    note()
                with closing(sqlite3.connect(rt.ro_uri(db), uri=True)) as conn:
                    if conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == nodes_before:
                        break
        finally:
            engine.shutdown()
        with closing(sqlite3.connect(rt.ro_uri(db), uri=True)) as conn:
            nodes = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
        steps.append({"restart": number + 1, "seconds": round(time.monotonic() - begun, 2),
                      "pass_rows_out": outs, "summary_nodes": nodes, "stored": rt.store_census(db)["rows"]})
    snapshot(work, args.extend)
    db = args.extend / "store" / "lcm.db"
    manifest.setdefault("extended", []).append({"tree_identity": rt.driver.tree_identity(args.tree.resolve()),
                                                "env": env, "steps": steps})
    manifest["stored"] = rt.store_census(db)
    manifest["store_sha256"] = rt.sha256_file(db)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({"extended": steps, "stored": manifest["stored"]["rows"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--kit", type=Path, default=rt.KIT)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--work", type=Path, default=None,
                        help="the runs' working directory (the store is written at its current/ path); default OUT/work")
    parser.add_argument("--extend", type=Path, default=None, help="an existing build to extend with later restarts")
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--passes", type=int, default=1, help="passes after each extending restart")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for the extending tree")
    args = parser.parse_args(argv)
    if args.extend:
        return extend(args)
    if args.out is None:
        parser.error("--out is required unless --extend is given")
    args.out.mkdir(parents=True, exist_ok=False)
    kit = scaled(rt.load_kit(args.kit), args.scale)
    work = rt.plain_dir((args.work or args.out / "work").absolute()).resolve()  # recorded as runs compare it
    lock = run_lock(work)  # noqa: F841 - held until the process exits
    home = work / "home"
    rt.isolate(home)
    config_module, engine_module, _lines = rt.load(args.tree, kit)
    rt.aux.configure(floor_s=0.05, rate_tps=1e9)  # building only: earlier passes need no realistic latency
    # The store is written at the runs' own working path, so the host-uid lineage key (it hashes the host
    # store's path) is the one the runs see; the result is then snapshotted into the build directory.
    db = work / "current" / "lcm.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    clear_working(work)
    config = rt.config_for(config_module, db)
    generated = shape.build(kit)
    target = kit["target"]
    started = time.monotonic()
    report = [ingest_history(engine_module, config, kit, home, rows, f"other{index}-s",
                             f"other{index}-conv") for index, rows in enumerate(generated["others"])]
    report.append(ingest_history(engine_module, config, kit, home, generated["target_history"],
                                 target["session_id"], target["conversation_id"]))
    # Earlier restarts: each binds the host's cold-rebuilt list once (no pass), as the reference store's earlier
    # builds did; on the v0.30.0 tree such a bind stores copies of unstamped rows (#1674), which the store keeps.
    restarts, bound = int(target.get("cold_restarts", 0)), int(target.get("bound_restarts", 0))
    for number in range(restarts):
        if bound and number == restarts - bound:
            # From here the host store resolves the target's lineage, so these last starts write its host-uid
            # bindings (the reference store held bindings for only part of its rows); other sessions stay unbound.
            rt.host_state_db(home, [target["session_id"]])
        begun = time.monotonic()
        engine = rt.engine(engine_module, config, kit, home)
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
    snapshot(work, args.out)
    db = args.out / "store" / "lcm.db"
    externalized = db.parent / "externalized"
    manifest = {
        "work": str(work), "host_sessions": rt.store_census(db)["conversations"],
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
