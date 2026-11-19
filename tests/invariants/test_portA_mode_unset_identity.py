"""Two-tree UID differential. Set PORTA_MAIN_REF to the detached 5948585c checkout.

The lane checkout is discovered automatically. PORTA_UID_ON_REF optionally supplies the
detached cf36ccb7 checkout for the same workload in mode on.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
from contextlib import closing
from copy import deepcopy

import pytest


ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT.parent / "refs"
MAIN = "5948585c4f807d9bdef14f54261d25f5079d94d8"
RELEASE = "cf36ccb79adbac1c2cbcbe94b371de4260ae9a5d"
SEED = 2815005
CASE_COUNT = 208


def _digest(value):
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8", errors="surrogatepass")).hexdigest()


def _turns(rng, count, *, offset=0):
    rows = []
    for i in range(offset, offset + count):
        turn = [
            {"role": "user", "content": f"Request {i}: " + "alpha beta " * rng.randrange(12, 35)},
            {"role": "assistant", "content": f"Reply {i}: " + "gamma delta " * rng.randrange(8, 24)},
        ]
        if i == 1 or rng.random() < 0.45:
            call_id = f"call-{i}"
            turn[1:1] = [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": call_id, "content": f"Output {i}: " + "epsilon " * 120},
            ]
        for j, row in enumerate(turn):
            row.update(message_uid=f"uid-{i}-{j}", timestamp=float(1000 + 10 * i + j))
        rows.extend(turn)
    return rows


def _snapshot(engine, returned=None):
    conn = engine._store._conn
    rows = [[sid, session, role, _digest(content), stamp] for sid, session, role, content, stamp in conn.execute(
        "SELECT store_id, session_id, role, content, observed_at FROM messages ORDER BY store_id")]
    bindings = list(conn.execute(
        "SELECT * FROM host_uid_bindings ORDER BY store_id, uid, lineage_key, kind, binding_version, "
        "proof_kind, created_at, first_check, disagree_seen")) if engine._store._host_uid_table_exists() else None
    lifecycle = engine._lifecycle.get_by_conversation("port-a")
    reconciliation = engine._last_ingest_reconciliation
    return {
        "rows": rows,
        "bindings": None if bindings is None else [list(row) for row in bindings],
        "session": engine._session_id,
        "ingest_cursor": engine._ingest_cursor,
        "reconciliation": dict(reconciliation),
        "compaction_cursor": engine._last_compacted_store_id,
        "frontier": lifecycle.current_frontier_store_id if lifecycle else None,
        "compression_status": engine._last_compression_status,
        "compression_reason": engine._last_compression_noop_reason,
        "returned": None if returned is None else _digest(returned),
    }


def _case(home, index, rng):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER

    settings = LCMConfig(
        database_path=str(home / "lcm.db"), fresh_tail_count=rng.randrange(2, 5),
        leaf_chunk_tokens=rng.choice((260, 400, 650)), context_threshold=0.001,
        threshold_full_sweep_enabled=False, embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
        large_output_externalization_enabled=True, large_output_externalization_threshold_chars=500,
        large_output_externalization_path=str(home / "payloads"),
    )
    count = rng.randrange(5, 13)
    rows = _turns(rng, count)
    if index % 4 == 0:
        rows[0]["content"] = [
            {"type": "text", "text": rows[0]["content"]},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "AAAB" * 270}},
        ]
    incoming = deepcopy(rows)
    for row in incoming:
        if row["role"] == "tool" and index % 3 != 0 and rng.random() < 0.8:
            row["content"] = _HERMES_PRUNED_TOOL_PLACEHOLDER
        if row["role"] != "user" and index % 3 == 1:
            row["timestamp"] += 0.5
        elif row["role"] != "user" and index % 3 == 2:
            row.pop("timestamp")
    incoming += _turns(rng, rng.randrange(1, 4), offset=count)
    if index % 8 == 0:
        # Exercise a long fresh suffix at the UID cursor cut-budget boundary.
        incoming += _turns(rng, 28, offset=count + 3)

    binding = dict(platform="telegram", conversation_id="port-a", context_length=200_000)
    engine = LCMEngine(config=settings, hermes_home=str(home))
    try:
        engine.on_session_start("S", **binding)
        engine.ingest(rows)
        snapshot = {"seed": _snapshot(engine)}
    finally:
        engine.shutdown()
    assert snapshot["seed"]["bindings"], "fixture must seed real UID evidence"

    engine = LCMEngine(config=settings, hermes_home=str(home))
    try:
        engine.on_session_start("S", **binding)
        engine.ingest(incoming)
        snapshot["cold"] = _snapshot(engine)
        returned = engine.compress(incoming, current_tokens=24_000)
        snapshot["compact"] = _snapshot(engine, returned)
        if index % 2 == 0:
            engine.on_session_end("S", returned)
            engine.on_session_start("R", old_session_id="S", boundary_reason="compression", **binding)
            snapshot["rotate"] = _snapshot(engine)
        engine.note_turn_complete()
        returned = engine.compress(returned, current_tokens=rng.choice((10, 24_000)))
        snapshot["second"] = _snapshot(engine, returned)
    finally:
        engine.shutdown()
    return snapshot


def _worker(tree, home, mode, output):
    sys.path.insert(0, str(tree))
    import tests.conftest  # noqa: F401
    import hermes_lcm.engine as engine_module
    from tests.behaviour.ingest_trace_harness import install_clock
    from tests.harness import pin_char_counter
    from tests.test_host_uid_shadow import _state_db

    assert Path(engine_module.__file__).resolve().parent == tree
    logging.disable(logging.CRITICAL)
    home.mkdir(exist_ok=True)
    _state_db(home, [("S", None, "compression"), ("R", "S", None)])
    rng = random.Random(SEED)
    with pytest.MonkeyPatch.context() as patch, closing(sqlite3.connect(":memory:")) as empty:
        patch.setenv("HERMES_HOME", str(home))
        if mode == "unset":
            patch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
        else:
            patch.setenv("LCM_HOST_MESSAGE_UID", mode)
        pin_char_counter(patch)
        install_clock(patch)
        patch.setattr(engine_module, "summarize_with_escalation",
                      lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1))
        with output.open("w") as stream:
            for index in range(CASE_COUNT):
                # Reset the single synthetic store in place between isolated cases.
                with closing(sqlite3.connect(home / "lcm.db")) as target:
                    empty.backup(target)
                snapshot = _case(home, index, rng)
                stream.write(json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n")


def _run(tree, home, mode, output):
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "TZ": "UTC"}
    # The selected checkout supplies both product imports and test helpers.
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p).resolve() != ROOT]
    env["PYTHONPATH"] = os.pathsep.join([str(tree), *paths])
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tree), str(home), mode, str(output)],
        cwd=tree, env=env, check=True, timeout=1080,
    )
    return [json.loads(line) for line in output.read_text().splitlines()]


def _compare(tmp_path, mode, variable, name, sha):
    explicit = os.environ.get(variable)
    ref = Path(explicit).resolve() if explicit else REFS / name
    if not ref.is_dir() and not explicit:
        pytest.skip(f"provide the pinned detached checkout via {variable}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ref, text=True).strip()
    assert head == sha
    assert subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=ref).returncode == 1
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ref, text=True)
    home = tmp_path / "sandbox"
    expected = _run(ref, home, mode, tmp_path / "reference.jsonl")
    actual = _run(ROOT, home, mode, tmp_path / "candidate.jsonl")
    assert len(expected) == len(actual) == CASE_COUNT
    differences = [
        {"case": i, "at": stage, "fields": fields}
        for i, (before, after) in enumerate(zip(expected, actual, strict=True))
        for stage in sorted(before.keys() | after.keys())
        if (fields := sorted(k for k in before.get(stage, {}).keys() | after.get(stage, {}).keys()
                             if before.get(stage, {}).get(k) != after.get(stage, {}).get(k)))
    ]
    (tmp_path / "differences.json").write_text(json.dumps(differences, indent=2) + "\n")
    assert not differences, f"{mode}: {len(differences)} differing checkpoints; see {tmp_path}/differences.json"


@pytest.mark.parametrize("mode", ("unset", "shadow"))
def test_mode_unset_identity(tmp_path, mode):
    _compare(tmp_path, mode, "PORTA_MAIN_REF", "portA-main", MAIN)


def test_mode_on_release_identity(tmp_path):
    _compare(tmp_path, "on", "PORTA_UID_ON_REF", "portA-release", RELEASE)


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
