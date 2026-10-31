"""P-S2: v0.29.0 stale cleanup proof across upgrade and two cold starts (I1/I5)."""
import hashlib
import json
import os
from collections import Counter
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

V0290 = "c13dea3783bdbd1aab52d83dff5d009bfade0bd2"
SID = "cleanup-rewrite-proof"
PAD = " alpha beta gamma delta" * 30


def _save_database(source, saved):
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(saved)) as dst:
        src.backup(dst)


def _restore_database(saved, target):
    for suffix in ("-wal", "-shm"):
        Path(str(target) + suffix).unlink(missing_ok=True)
    shutil.copy2(saved, target)


def _identity(role, content, tool_call_id, tool_calls):
    """Keep in sync with the probe's SQLite identity projection."""
    calls = json.loads(tool_calls) if tool_calls else []
    call_ids = tuple((call.get("id"), call.get("function", {}).get("name"),
                      call.get("function", {}).get("arguments")) for call in calls) if role == "assistant" else ()
    return role, hashlib.sha256((content or "").encode()).hexdigest(), call_ids, tool_call_id if role == "tool" else None


def _stored_rows(database):
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        return [_identity(*row) for row in conn.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM messages WHERE session_id = ? ORDER BY store_id",
            (SID,)).fetchall()]


def _freeze(value):
    return tuple(_freeze(item) for item in value) if isinstance(value, list) else value


def _turn_rows(restart):
    return [_identity(role, f"P-S2 restart {restart} {role}{PAD}", None, None) for role in ("user", "assistant")]


def _duplicates(before, after, new_turn):
    """Multiset of newly added occurrences whose identity was already stored."""
    added = Counter(after) - Counter(before)
    return Counter({row: count for row, count in added.items() if row in set(before) and row not in new_turn})


def _in_order(before, after):
    remaining = iter(after)
    return all(any(row == candidate for candidate in remaining) for row in before)


def _identity_hash(row):
    return hashlib.sha256(json.dumps(row, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _hashed_counts(counts):
    return [{"identity_sha256": _identity_hash(row), "count": count}
            for row, count in sorted(counts.items(), key=lambda item: _identity_hash(item[0]))]


@pytest.mark.parametrize("redact", ["off", "on"])
@pytest.mark.parametrize("variant", ["stub", "reasoning"])
@pytest.mark.parametrize("change", ["insert", "replace"])
def test_stale_cleanup_proof_upgrade(tmp_path, old_tree, host_root, variant, change, redact):
    base = old_tree("v0.29.0", {"v0.29.0": V0290})
    candidate = Path(__file__).resolve().parents[2]
    probe = tmp_path / "stale_cleanup_proof_upgrade_probe.py"
    shutil.copy2(Path(__file__).with_name(probe.name), probe)
    home, result = tmp_path / "writer", tmp_path / "result.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("LCM_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", P2S_HOME=str(home), P2S_OUT=str(result),
               P2S_VARIANT=variant, P2S_CHANGE=change, P2S_REDACT=redact)

    def phase(tree, action, side, restart=0):
        env.update(PYTHONPATH=str(tree) + os.pathsep + str(host_root), P2S_PHASE=action,
                   P2S_RESTART=str(restart))
        result.unlink(missing_ok=True)
        command = [sys.executable, "-B", "-c",
                   "import tests.conftest; import pytest, sys; sys.exit(pytest.main(sys.argv[1:]))",
                   str(probe), "-q", "-p", "no:cacheprovider", "--confcutdir=" + str(probe.parent),
                   "--basetemp=" + str(tmp_path / f"{side}-{action}-{restart}")]
        run = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, f"{variant}/{change} {side} {action} restart {restart}:\n{run.stdout}{run.stderr}"
        if action == "read":
            data = json.loads(result.read_text())
            diagnostics.append({"side": side, "restart": restart, "reconciliation": data.get("reconciliation"),
                                "proof_log": data.get("proof_log", [])})
            return [_freeze(row) for row in data["rows"]]

    diagnostics = []
    phase(base, "write", "base")
    saved = tmp_path / "old-writer.db"
    _save_database(home / "lcm.db", saved)
    initial = _stored_rows(saved)
    sides, summary, preservation_errors = {}, {}, []
    for side, tree in (("base", base), ("candidate", candidate)):
        # Both readers see the same old-writer DB at the same absolute path.
        _restore_database(saved, home / "lcm.db")
        before = initial
        duplicates, receipts = [], []
        for restart in (1, 2):
            after = phase(tree, "read", side, restart)
            dup = _duplicates(before, after, _turn_rows(restart))
            duplicates.append(dup)
            kept = _in_order(initial, after)
            if not kept:
                preservation_errors.append(f"{side} restart {restart}: pre-restart-1 rows missing or reordered")
            counts = Counter(after)
            added = counts - Counter(before)
            turns = Counter(row for n in range(1, restart + 1) for row in _turn_rows(n))
            # This restart's own turn is stored exactly once; an earlier restart's turn only has to be present
            # (a repeat of it at this restart is a duplicate, counted in dupN, not a preservation error).
            for row in _turn_rows(restart):
                if counts[row] != 1 or added[row] != 1:
                    preservation_errors.append(
                        f"{side} restart {restart}: new-turn row {_identity_hash(row)} stored {counts[row]} times")
            for earlier in range(1, restart):
                for row in _turn_rows(earlier):
                    if counts[row] < 1:
                        preservation_errors.append(
                            f"{side} restart {restart}: restart {earlier}'s turn row {_identity_hash(row)} missing")
            receipts.append({"restart": restart, "rows_before": len(before), "rows_after": len(after),
                             "added_count": sum((counts - Counter(before)).values()),
                             "duplicate_count": sum(dup.values()), "duplicates": _hashed_counts(dup),
                             "initial_rows_in_order": kept,
                             "new_turn_counts": _hashed_counts(Counter({row: counts[row] for row in turns}))})
            before = after
        sides[side], summary[side] = duplicates, receipts

    subset = sides["candidate"][0] <= sides["base"][0]
    no_repeat = not sides["candidate"][1]
    base_repeats = bool(sides["base"][1])
    if result_dir := os.environ.get("P2S_RESULT_DIR"):
        directory = Path(result_dir)
        directory.mkdir(parents=True, exist_ok=True)
        receipt = {"variant": variant, "change": change, "redact": redact, "initial_count": len(initial),
                   "initial_identity_hashes": [_identity_hash(row) for row in initial], "sides": summary,
                   "checks": {"candidate_dup1_subset": subset, "candidate_dup2_empty": no_repeat,
                              "base_dup2_nonempty": base_repeats, "rows_preserved": not preservation_errors},
                   "diagnostics": diagnostics}
        (directory / f"p-s2-{variant}-{change}-{redact}.json").write_text(json.dumps(receipt, indent=2) + "\n")

    cell = f"{variant}/{change}/redact-{redact}"
    assert subset, f"(a) {cell}: candidate dup1 is not a sub-multiset of base dup1"
    assert no_repeat, f"(b) {cell}: candidate dup2 is not empty"
    assert base_repeats, f"(c) {cell}: base did not repeat"
    assert not preservation_errors, f"(d) {cell}: " + "; ".join(preservation_errors)
