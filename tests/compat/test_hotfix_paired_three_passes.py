"""Pair the v0.30.2 and v0.30.3 compaction paths on the same stored head copy."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
from subprocess import TimeoutExpired

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.head_copy import known_head_copy_ids
from tests.compat.helpers import run_old
from tests.compat.test_hotfix_head_copy_rollback import BINDING, PINS, _append, _seed_copy, _view
from tests.test_issue_671_stub_first_exit import summaries  # noqa: F401


SETTINGS = dict(
    embeddings_enabled=False,
    temporal_rollups_enabled=False,
    fresh_tail_count=2,
    leaf_chunk_tokens=1_000,
    fresh_tail_pressure_yield_enabled=False,
    threshold_full_sweep_enabled=False,
    l3_truncate_tokens=1,
    large_output_externalization_enabled=True,
    large_output_externalization_threshold_chars=1_000_000,
    large_output_active_replay_stubbing_enabled=True,
)
PASS_THRESHOLD = 85_000
PASS_LEAF_CEILING = 26_000
CANDIDATE = "8ed5afc407aed67ca8527f817a69e5cc39f0d8af"


PAIRED_PASSES = r'''
import io
import logging
import os
import re

from hermes_lcm.config import host_message_uid_mode
from tests.invariants.test_hotfix_stored_summary_head import _host_flush, _view

if mode == "on":
    os.environ["LCM_HOST_MESSAGE_UID"] = "on"
assert os.environ.get("LCM_HOST_MESSAGE_UID") == ("on" if mode == "on" else None)
assert host_message_uid_mode() == ("on" if mode == "on" else "shadow")

inputs = json.loads(data)
stream = io.StringIO()
logger = logging.getLogger("hermes_lcm")
logger.addHandler(logging.StreamHandler(stream))
logger.setLevel(logging.INFO)
engine_mod.summarize_with_escalation = lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1)
engine = LCMEngine(config=LCMConfig(**{**options, **inputs["settings"]}), hermes_home=home)


def leaf_sources(conn):
    return {node: json.loads(ids) for node, ids in conn.execute(
        "SELECT node_id, source_ids FROM summary_nodes WHERE session_id = ? AND source_type = 'messages'",
        (inputs["binding"]["session_id"],),
    )}


def hold_reasons():
    reasons = []
    if hold := engine._no_progress_hold_status():
        reasons.append(hold["reason"])
    if engine._sweep_budget_hold_active():
        reasons.append("sweep_budget")
    if engine._retained_stop_hold_active():
        reasons.append("retained_stop")
    return sorted(set(reasons))


selections = []
select_leaf = engine._select_leaf_candidate


def observe_selection(state, leaf, dependent_reply_message_ids):
    outcome = select_leaf(state, leaf, dependent_reply_message_ids)
    mapping = engine._compress_run.store_ids_by_message_id
    if outcome[0] == "next":
        selections.append({
            "store_ids": [mapping.get(id(row), row.get("store_id")) for row in leaf.to_compact],
            "fresh_tail_ids": [mapping.get(id(row), row.get("store_id"))
                               for row in state.working_messages[leaf.fresh_tail_start:]],
        })
    return outcome


try:
    engine.on_session_start(**inputs["binding"])
    engine.threshold_tokens = inputs["threshold"]
    conn = engine._store.connection
    view = _view(engine)
    # The committed rollback cell presents the already-durable rows as a loaded host view.
    engine._ingest_cursor = len(view)
    engine._ingest_cursor_needs_reconcile = False
    engine._compress_commit_proof = None
    engine._select_leaf_candidate = observe_selection
    flush = _host_flush()
    passes = []
    for index in range(3):
        stream.seek(0)
        stream.truncate()
        if index:
            view = view + flush([
                {"role": "user", "content": f"Please check adjustment {index}.", "message_uid": f"paired-user-{index}"},
                {"role": "assistant", "content": f"Adjustment {index} reconciles.",
                 "message_uid": f"paired-assistant-{index}"},
            ])
            engine.ingest(view)
        before = engine._store_complete_frontier()
        leaves_before = leaf_sources(conn)
        selections.clear()
        prompt_before = tokens.count_messages_tokens(view)
        view = engine.compress(view, current_tokens=prompt_before)
        added = {node: ids for node, ids in leaf_sources(conn).items() if node not in leaves_before}
        status = engine.last_compression_status
        noop_reason = engine.last_compression_noop_reason
        logs = stream.getvalue()
        stop_lines = [line for line in logs.splitlines() if line.startswith("LCM compaction stop:")]
        stop_line = stop_lines[-1] if stop_lines else ""
        refusal = re.search(r"\bfailure=(.*?)(?: escaped=|$)", stop_line)
        blocked = re.search(r"\bblocked=store_id:(\d+)", stop_line)
        stop_id = int(blocked[1]) if blocked else None
        if selections and not added and stop_id is None:
            stop_id = engine._store_complete_hole
        holds = hold_reasons()
        if not added and not selections and "retained_stop" in holds:
            signature, _, _ = engine._retained_stop_backoffs[engine._hold_conversation_key()]
            stop_id = signature[1]
        chunks = [[sid for sid in selection["store_ids"] if sid is not None] for selection in selections]
        passes.append(dict(
            number=index + 1, status=status, noop_reason=noop_reason,
            refusal_reason=refusal[1] if refusal else "", leaves=len(added), leaf_sources=added,
            frontier_before=before, frontier_after=engine._store_complete_frontier(),
            chunk_bounds=[[ids[0], ids[-1]] for ids in chunks if ids], stop_store_id=stop_id,
            selections=list(selections), prompt_input_tokens=prompt_before,
            prompt_tokens=tokens.count_messages_tokens(view),
            prompt_bytes=len(json.dumps(view, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
            hold_armed=bool(holds), hold_reasons=holds, stop_line=stop_line, logs=logs,
        ))
finally:
    engine.shutdown()
    if sys.exc_info()[0] is not None:
        print(stream.getvalue(), file=sys.stderr)
print(json.dumps(passes))
'''


def _bar_failures(mode, base, candidate):
    failures = []
    stopped_at = {}
    for baseline, current in zip(base, candidate, strict=True):
        index = current["number"]
        if current["leaves"] < baseline["leaves"]:
            failures.append((mode, index, 1, f"leaves CAND={current['leaves']} BASE={baseline['leaves']}"))
        if current["prompt_tokens"] > baseline["prompt_tokens"]:
            failures.append((mode, index, 2,
                             f"tokens CAND={current['prompt_tokens']} BASE={baseline['prompt_tokens']}"))
        extra_holds = set(current["hold_reasons"]) - set(baseline["hold_reasons"])
        extra_reasons = {current["noop_reason"], current["refusal_reason"]} - {
            "", baseline["noop_reason"], baseline["refusal_reason"],
        }
        if extra_holds or extra_reasons:
            failures.append((mode, index, 3, f"added holds={sorted(extra_holds)} reasons={sorted(extra_reasons)}"))
        row_id = current["stop_store_id"]
        if current["leaves"] == 0 and row_id is not None:
            if row_id in stopped_at:
                failures.append((mode, index, 4, f"stopping row={row_id} also stopped pass {stopped_at[row_id]}"))
            stopped_at.setdefault(row_id, index)
    return failures


@pytest.mark.usefixtures("summaries")
def test_hotfix_paired_three_passes(tmp_path, old_tree, host_root, caplog):
    base_tree = old_tree("v0.30.2", PINS)
    worktree = Path(__file__).resolve().parents[2]
    home = tmp_path / "seed-home"
    seed_path = tmp_path / "lcm.db"
    seed = LCMEngine(config=LCMConfig(database_path=str(seed_path), **SETTINGS), hermes_home=str(home))
    try:
        _, copy_id, _ = _seed_copy(seed, caplog)
        for index in range(24):
            _append(seed, {"role": "user", "content": f"Unrelated question {index}." +
                           " Please reconcile the itemized invoice expenses." * 140})
            _append(seed, {"role": "assistant", "content": f"Unrelated reply {index}." +
                           " The accounting ledger confirms the itemized expenses." * 140})
        frontier = seed._store_complete_frontier()
        window = _view(seed)
        window_ids = [row["store_id"] for row in window]
        # The stub-first seed can already have stored a carrier on its way to the explicitly appended copy.
        known_copy_ids = sorted(known_head_copy_ids(seed._store.connection, BINDING["session_id"]))
        assert copy_id in known_copy_ids
        assert frontier < copy_id < window_ids[seed._fresh_tail_start(window)]
    finally:
        seed.shutdown()

    settings = dict(SETTINGS, large_output_active_replay_stubbing_enabled=False,
                    dynamic_leaf_chunk_max=PASS_LEAF_CEILING)
    data = json.dumps(dict(binding=BINDING, settings=settings, threshold=PASS_THRESHOLD))
    seed_files = [path for path in (seed_path, tmp_path / "lcm.db-wal", tmp_path / "lcm.db-shm") if path.exists()]
    seed_hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in seed_files}
    report = dict(
        revisions={"BASE": PINS["v0.30.2"], "CAND": CANDIDATE},
        seed=dict(copy_id=copy_id, known_copy_ids=known_copy_ids, frontier=frontier,
                  rows_after_frontier_before_copy=window_ids.index(copy_id),
                  window_first=window_ids[0], window_last=window_ids[-1], sha256=seed_hashes),
        ingest="One identical host-stamped turn before each of passes 2 and 3.",
        prompt_bytes="UTF-8 byte length of the returned message list as compact JSON.",
        settings=settings, threshold=PASS_THRESHOLD, modes={}, failures=[], errors={},
    )
    for mode in ("unset", "on"):
        arms = report["modes"][mode] = {}
        for arm, tree in (("BASE", base_tree), ("CAND", worktree)):
            directory = tmp_path / f"{mode}-{arm}"
            directory.mkdir()
            for path in seed_files:
                clone = shutil.copy2(path, directory / path.name)
                assert sha256(Path(clone).read_bytes()).hexdigest() == seed_hashes[path.name]
            try:
                arms[arm] = run_old(tree, directory, host_root, PAIRED_PASSES, mode=mode, data=data)
            except (AssertionError, TimeoutExpired) as exc:
                report["errors"][f"{mode}/{arm}"] = str(exc)
        if "BASE" in arms:
            first = arms["BASE"][0]
            if not any(copy_id in selection["store_ids"] or copy_id in selection["fresh_tail_ids"]
                       for selection in first["selections"]):
                report["errors"][f"{mode}/placement"] = f"BASE pass 1 selection did not reach copy {copy_id}"
        if {"BASE", "CAND"} <= arms.keys():
            report["failures"].extend(_bar_failures(mode, arms["BASE"], arms["CAND"]))

    print("PAIRED_RESULT " + json.dumps(report, sort_keys=True))
    assert not report["failures"], report["failures"]
    assert not report["errors"], report["errors"]
