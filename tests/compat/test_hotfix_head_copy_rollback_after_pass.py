"""v0.30.2 rollback across a hotfix frontier that skips a retained head copy (I1, I2, I11)."""

from hashlib import sha256
from inspect import getsource
import json
import logging
from pathlib import Path
import shutil
from subprocess import TimeoutExpired

import pytest

from hermes_lcm.config import host_message_uid_mode
from hermes_lcm.tokens import count_messages_tokens
from tests.compat.test_hotfix_head_copy_rollback import (
    BINDING,
    GREP,
    PINS,
    ROLLBACK_TURN,
    _append,
    _host_flush,
    _view,
    run_old,
)
from tests.invariants.test_head_copy_subset import head_copy_subset  # noqa: F401
from tests.invariants.test_hotfix_stored_summary_head import make_engine, summaries, twins, uid_mode  # noqa: F401


SUMMARIZED_PHRASE = "alpha beta gamma delta"
LINEAGE_STOP = "selected leaf chunk lacks raw store lineage"


def message_hashes(conn):
    return [
        [store_id, role, sha256(content.encode()).hexdigest()]
        for store_id, role, content in conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id")
    ]


def compact_loaded_turn(engine, turn):
    for row in turn:
        _append(engine, row)
    view = _view(engine)
    # These already-stored rows are presented as the host's loaded view.
    engine._ingest_cursor = len(view)
    engine._ingest_cursor_needs_reconcile = False
    engine._compress_commit_proof = None
    assembled = engine.compress(view, current_tokens=count_messages_tokens(view), force=True)
    return assembled, [engine.last_compression_status, engine.last_compression_noop_reason,
                       engine._last_compacted_store_id]


OLD_COMPACT_CLEAN_TWIN = r'''
import os
from hashlib import sha256

from hermes_lcm.config import host_message_uid_mode
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_hotfix_stored_summary_head import NEW_TURN, _append, _host_flush, _view
from tests.test_issue_671_stub_first_exit import THRESHOLD, text_view, tool_view
''' + getsource(message_hashes) + "\n\n" + getsource(compact_loaded_turn) + r'''

if mode == "on":
    os.environ["LCM_HOST_MESSAGE_UID"] = "on"
assert host_message_uid_mode() == ("on" if mode == "on" else "shadow")
inputs = json.loads(data)
# Match the twins fixture's seed settings and summarizer for the pre-resume passes.
engine_mod.summarize_with_escalation = lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1)
engine = LCMEngine(config=LCMConfig(
    database_path=db,
    fresh_tail_count=2,
    leaf_chunk_tokens=1_000,
    fresh_tail_pressure_yield_enabled=False,
    threshold_full_sweep_enabled=False,
    l3_truncate_tokens=1,
    large_output_externalization_enabled=True,
    large_output_externalization_threshold_chars=1_000_000,
    large_output_active_replay_stubbing_enabled=True,
), hermes_home=home)
try:
    engine.on_session_start("issue-671-0", conversation_id="issue-671-conversation-0", context_length=200_000)
    engine.on_session_start(**inputs["binding"])
    engine.threshold_tokens = THRESHOLD
    seed = text_view()[1:]
    seed[0]["content"] = "Earlier turns. " + seed[0]["content"]
    first = engine.compress(seed, current_tokens=count_messages_tokens(seed))
    head_input = {**first[0], "content": first[0]["content"] + "\n\n---\n\n" + first[1]["content"]}
    supplied = tool_view(prefix=[head_input, *first[2:]])[1:]
    returned = engine.compress(supplied, current_tokens=count_messages_tokens(supplied))
    assert len(returned) == len(supplied) and returned[0]["content"] != head_input["content"]
    engine.ingest(_host_flush()(returned + NEW_TURN))
    # Occupy the copy's global store ID outside the session, just as the clean twin does.
    occupied_id = _append(engine, NEW_TURN[0], session="unrelated")
    engine._config.large_output_active_replay_stubbing_enabled = False
    conn = engine._store.connection
    seed_rows = message_hashes(conn)
    seed_frontier = engine._last_compacted_store_id
    passes = []
    for turn in inputs["turns"]:
        view, result = compact_loaded_turn(engine, turn)
        passes.append(result)
    report = dict(
        occupied_id=occupied_id, seed_rows=seed_rows, seed_frontier=seed_frontier, head=returned[0]["content"],
        passes=passes, view=view, messages=message_hashes(conn), frontier=engine._last_compacted_store_id,
    )
finally:
    engine.shutdown()
print(json.dumps(report))
'''


OLD_RESUME_AFTER_PASS = r'''
import io
import logging
import os
from hashlib import sha256

from hermes_lcm.command import _doctor_source_text
from hermes_lcm.config import host_message_uid_mode
from hermes_lcm.tools import lcm_grep
''' + getsource(message_hashes) + r'''

if mode == "on":
    os.environ["LCM_HOST_MESSAGE_UID"] = "on"
assert host_message_uid_mode() == ("on" if mode == "on" else "shadow")
log = io.StringIO()
logging.getLogger("hermes_lcm").addHandler(logging.StreamHandler(log))
logging.getLogger("hermes_lcm").setLevel(logging.INFO)

inputs = json.loads(data)
engine = LCMEngine(config=LCMConfig(database_path=db, **inputs["settings"]), hermes_home=home)
try:
    engine.on_session_start(**inputs["binding"])
    conn = engine._store.connection
    bound_frontier = engine._last_compacted_store_id
    view = inputs["view"]
    engine.ingest(view)
    passes = []
    for _ in range(2):
        last_node_id = conn.execute("SELECT MAX(node_id) FROM summary_nodes").fetchone()[0]
        view = engine.compress(view, current_tokens=tokens.count_messages_tokens(view), force=True)
        passes.append({
            "status": engine.last_compression_status,
            "noop_reason": engine.last_compression_noop_reason,
            "frontier": engine._last_compacted_store_id,
            "new_leaf_ids": [row[0] for row in conn.execute(
                "SELECT node_id FROM summary_nodes WHERE source_type = 'messages' AND node_id > ? ORDER BY node_id",
                (last_node_id,),
            )],
        })
    grep = {}
    for name, args in inputs["grep"].items():
        result = json.loads(lcm_grep(args, engine=engine))
        grep[name] = {
            "store_ids": [hit["store_id"] for hit in result["results"] if "store_id" in hit],
            "node_ids": [hit["node_id"] for hit in result["results"] if "node_id" in hit],
        }
    report = {
        "messages": message_hashes(conn),
        "nodes": [
            [node_id, source_type, json.loads(source_ids)] for node_id, source_type, source_ids in conn.execute(
                "SELECT node_id, source_type, source_ids FROM summary_nodes ORDER BY node_id"
            )
        ],
        "bound_frontier": bound_frontier,
        "passes": passes,
        "grep": grep,
        "doctor_source": _doctor_source_text(engine).splitlines(),
        "integrity_check": [row[0] for row in conn.execute("PRAGMA integrity_check")],
    }
finally:
    engine.shutdown()
    relevant = [
        line for line in log.getvalue().splitlines()
        if any(term in line.lower() for term in ("lineage", "conflict", "coverage"))
    ]
    if sys.exc_info()[0] is not None:
        print("\n".join(relevant), file=sys.stderr)
report["logs"] = relevant
print(json.dumps(report, sort_keys=True))
'''


@pytest.mark.parametrize("uid_mode", ["unset", "on"], indirect=True)
def test_v0302_rollback_after_hotfix_compacts_across_known_head_copy(
    twins, tmp_path, old_tree, host_root, caplog,  # noqa: F811
):
    old = old_tree("v0.30.2", PINS)
    mode = "on" if host_message_uid_mode() == "on" else "unset"
    (clean, clean_head, _), (copied, head, copy_id) = twins
    assert clean_head == head
    copy_conn = copied._store.connection
    copy_before = copy_conn.execute("SELECT * FROM messages WHERE store_id = ?", (copy_id,)).fetchone()
    assert copy_before is not None
    assert copy_conn.execute("SELECT content FROM messages WHERE store_id = ?", (copy_id,)).fetchone()[0] == head["content"]
    assert all(engine._last_compacted_store_id < copy_id for engine in (clean, copied))

    stores = {"clean": clean, "copy": copied}
    clean_seed_rows = message_hashes(clean._store.connection)
    clean_seed_frontier = clean._last_compacted_store_id
    views, before, frontiers, paths = {}, {}, {}, {}
    hotfix_passes = []
    appended_turns = []
    # Put enough unrelated history outside the two-message fresh tail to meet the leaf threshold.
    question = " Please reconcile the itemized invoice expenses." * 140
    reply = " The accounting ledger confirms the itemized expenses." * 140
    try:
        for turn in range(20):
            appended_turn = [
                {"role": "user", "content": f"Unrelated question {turn}.{question}"},
                {"role": "assistant", "content": f"Unrelated reply {turn}.{reply}"},
            ]
            appended_turns.append(appended_turn)
            passes = {}
            with caplog.at_level(logging.INFO, logger="hermes_lcm"):
                for name, engine in stores.items():
                    views[name], passes[name] = compact_loaded_turn(engine, appended_turn)
            hotfix_passes.append(passes)
            if copied._last_compacted_store_id > copy_id:
                break
        print(json.dumps({"uid_mode": mode, "copy_id": copy_id, "hotfix_passes": hotfix_passes}))
        assert copied._last_compacted_store_id > copy_id
        assert clean._last_compacted_store_id == copied._last_compacted_store_id
        assert copy_conn.execute("SELECT * FROM messages WHERE store_id = ?", (copy_id,)).fetchone() == copy_before
        covered_ids = {row[0] for row in copy_conn.execute(
            "SELECT source.value FROM summary_nodes AS node, json_each(node.source_ids) AS source "
            "WHERE node.source_type = 'messages'"
        )}
        assert copy_id not in covered_ids
        assert any(SUMMARIZED_PHRASE in content for store_id, content in copy_conn.execute(
            "SELECT store_id, content FROM messages"
        ) if store_id in covered_ids)
        assert views["copy"] == views["clean"]
        for name, engine in stores.items():
            paths[name] = Path(engine._store.db_path)
            before[name] = message_hashes(engine._store.connection)
            frontiers[name] = engine._last_compacted_store_id
    finally:
        for engine in stores.values():
            engine.shutdown()

    settings = dict(
        embeddings_enabled=False,
        temporal_rollups_enabled=False,
        fresh_tail_count=2,
        leaf_chunk_tokens=1_000,
        fresh_tail_pressure_yield_enabled=False,
        threshold_full_sweep_enabled=False,
        l3_truncate_tokens=1,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1_000_000,
        large_output_active_replay_stubbing_enabled=False,
    )
    pure_base_compaction = tmp_path / "pure_base_compaction"
    pure_base_compaction.mkdir()
    pure_base = run_old(
        old, pure_base_compaction, host_root, OLD_COMPACT_CLEAN_TWIN, mode=mode,
        data=json.dumps(dict(binding=BINDING, turns=appended_turns)),
    )
    print(json.dumps({"uid_mode": mode, "pure_base_passes": pure_base["passes"]}))
    before["pure_base"] = pure_base["messages"]
    views["pure_base"] = pure_base["view"]
    frontiers["pure_base"] = pure_base["frontier"]
    paths["pure_base"] = pure_base_compaction / "lcm.db"

    reports, replay_duplicates, errors = {}, {}, {}
    for name, path in paths.items():
        # Resume every arm from a copy of its closed store in a fresh profile directory.
        directory = tmp_path / name
        directory.mkdir()
        for suffix in ("", "-wal", "-shm"):
            source = path.with_name(path.name + suffix)
            if source.exists():
                shutil.copy2(source, directory / ("lcm.db" + suffix))
        data = json.dumps(dict(
            settings=settings, binding=BINDING, view=_host_flush()(views[name] + ROLLBACK_TURN),
            grep={"head": GREP, "summarized_turn": {"query": SUMMARIZED_PHRASE, "limit": 100}},
        ))
        try:
            reports[name] = run_old(old, directory, host_root, OLD_RESUME_AFTER_PASS, mode=mode, data=data)
        except (AssertionError, TimeoutExpired) as exc:
            errors[name] = str(exc)
        report = reports.get(name)
        duplicates = {}
        if report is not None:
            for store_id, role, digest in report["messages"][len(before[name]):]:
                matches = [row_id for row_id, old_role, old_digest in before[name]
                           if (old_role, old_digest) == (role, digest)]
                if matches:
                    duplicates[store_id] = matches
        replay_duplicates[name] = duplicates
        print(json.dumps({"uid_mode": mode, "store": name, "report": report,
                          "replay_duplicates": duplicates, "error": errors.get(name)}))
    assert not errors, errors
    assert pure_base["occupied_id"] == copy_id
    assert pure_base["seed_rows"] == clean_seed_rows
    assert pure_base["seed_frontier"] == clean_seed_frontier
    assert pure_base["head"] == clean_head["content"]
    assert before["pure_base"] == before["clean"]

    clean_report, copy_report = reports["clean"], reports["copy"]
    pure_base_report = reports["pure_base"]
    differences = {}
    for key, clean_value in clean_report.items():
        copy_value = copy_report[key]
        if clean_value == copy_value:
            continue
        if key in ("messages", "nodes"):
            clean_by_id = {row[0]: row for row in clean_value}
            copy_by_id = {row[0]: row for row in copy_value}
            differences[key] = {
                row_id: {"clean": clean_by_id.get(row_id), "copy": copy_by_id.get(row_id)}
                for row_id in sorted(clean_by_id.keys() | copy_by_id.keys())
                if clean_by_id.get(row_id) != copy_by_id.get(row_id)
            }
        else:
            differences[key] = {"clean": clean_value, "copy": copy_value}
    print(json.dumps({"uid_mode": mode, "twin_differences": differences}))

    expected_added = [[row["role"], sha256(row["content"].encode()).hexdigest()] for row in ROLLBACK_TURN]
    expected_second_pass = ("compacted", "") if mode == "on" else ("noop", LINEAGE_STOP)
    for name, report in reports.items():
        assert report["integrity_check"] == ["ok"], name
        assert "status: ok" in report["doctor_source"], name
        assert report["bound_frontier"] == frontiers[name], name
        assert report["messages"][:len(before[name])] == before[name], name
        first_pass, second_pass = report["passes"]
        assert (first_pass["status"], first_pass["noop_reason"]) == ("compacted", ""), (name, first_pass)
        assert first_pass["new_leaf_ids"], (name, first_pass)
        assert (second_pass["status"], second_pass["noop_reason"]) == expected_second_pass, (name, second_pass)
        assert not any("conflict" in line.lower() for line in report["logs"]), name
        added = [row[1:] for row in report["messages"][len(before[name]):] if row[0] not in replay_duplicates[name]]
        assert added == expected_added, name

    assert set(differences) == {"messages", "grep"}
    assert set(differences["messages"]) == {copy_id}
    assert copy_report["nodes"] == clean_report["nodes"]
    assert copy_report["passes"] == clean_report["passes"] == pure_base_report["passes"]
    for name, hits in copy_report["grep"].items():
        assert [store_id for store_id in hits["store_ids"] if store_id != copy_id] == clean_report["grep"][name]["store_ids"]
        assert hits["node_ids"] == clean_report["grep"][name]["node_ids"]
    assert replay_duplicates["pure_base"] == {56: [53, 55], 57: [54]}
    assert replay_duplicates["copy"] == replay_duplicates["clean"] == replay_duplicates["pure_base"], replay_duplicates
