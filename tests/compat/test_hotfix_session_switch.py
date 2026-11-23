"""Compare first ingest after a session switch with a pending stub-first proof."""

import json
from pathlib import Path
from subprocess import TimeoutExpired

from tests.compat.helpers import run_old
from tests.compat.test_hotfix_head_copy_rollback import PINS


CANDIDATE = "8ed5afc407aed67ca8527f817a69e5cc39f0d8af"

SESSION_SWITCH = r'''
from dataclasses import replace
from hashlib import sha256
from io import StringIO
import logging
import os

from hermes_lcm import host_uid_emit, reconcile
from hermes_lcm.config import host_message_uid_mode
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_1116_stub_first_exit_commit_proof import (
    CID, CONTEXT, SID, TAIL_COUNT, Host, committed_session, grow_over_threshold, stub_first_result,
)
from tests.test_refused_compaction_rollback import _drain_session_end_maintenance


class Capture(StringIO):
    @property
    def text(self):
        return self.getvalue()

    def clear(self):
        self.seek(0)
        self.truncate()


class SwitchHost(Host):
    def _config(self):
        return replace(super()._config(), embeddings_enabled=False, temporal_rollups_enabled=False)


summary_calls = 0


def summarize(**kwargs):
    global summary_calls
    summary_calls += 1
    return f"Summary {summary_calls}: " + "lorem ipsum " * 200 + "\nExpand for details about: turns", 1


def fingerprint(row):
    return {"role": row["role"], "sha256": sha256(row["content"].encode("utf-8")).hexdigest()}


if mode == "on":
    os.environ["LCM_HOST_MESSAGE_UID"] = "on"
assert os.environ.get("LCM_HOST_MESSAGE_UID") == ("on" if mode == "on" else None)
assert host_message_uid_mode() == ("on" if mode == "on" else "shadow")
host_uid_emit._host_uid_capability = True
engine_mod.summarize_with_escalation = summarize
capture = Capture()
logger = logging.getLogger("hermes_lcm")
logger.addHandler(logging.StreamHandler(capture))
logger.setLevel(logging.INFO)

child = "switch-session-2"
Path(home).mkdir(exist_ok=True)
_state_db(Path(home), [(SID, None, None)])
host = SwitchHost(Path(db).parent, TAIL_COUNT["carrier"])
engine = host.engine
try:
    committed_session(host, "carrier")
    flush = _host_flush()
    host.live = flush(host.live)
    grow_over_threshold(host)
    result, original = stub_first_result(host, capture)
    head = result[0]
    # The archived release has a separate recognition gate. On main, both
    # configured modes use shadow behavior and allow UID emission.
    recognition_enabled = getattr(engine, "_uid_recognition_enabled", None)
    if recognition_enabled is not None:
        assert recognition_enabled() == (mode == "on")
    else:
        assert host_uid_emit.identity_emit_enabled()
    assert len(result) == len(original), (len(result), len(original))
    assert engine._last_compress_leaves[1] == engine._compress_host_rows_consumed == 0
    assert head["role"] == "user" and "timestamp" not in head
    assert engine._verified_lcm_summary_prefix_end(head["content"])
    assert engine._generated_context_carrier_remainder(head)
    assert engine._compress_commit_publication == "local"
    pending = engine._compress_commit_proof
    assert pending["session_id"] == SID and not pending.get("consulted")
    assert pending["stub_first"] and not pending["end_consumed"]
    s1_binding = [pending["session_id"], pending["conversation_id"]]
    head_digest = pending.get("emitted_user_digests", {}).get(0)

    _drain_session_end_maintenance(engine)
    if data == "rotation":
        host.rotate(result, original, child)
    else:
        assert data == "plain"
        engine.on_session_start(child, platform="cli", conversation_id=CID, context_length=CONTEXT)
    assert engine._session_id == child
    session_before = engine._store.get_session_messages(child)
    assert not session_before, "S2 must be empty before its first ingest"
    bound_proof = engine._compress_commit_proof or {}
    ingest_entry = dict(proof_session=bound_proof.get("session_id"), proof_rows=len(bound_proof.get("output") or ()),
                        cursor=engine._ingest_cursor, needs_reconcile=engine._ingest_cursor_needs_reconcile)

    # The host stamps the emitted row at a new address and preserves its UID fields.
    replay_head = host_uid_emit.carry_identity(head, {"role": "user", "content": head["content"]})
    supplied = flush([
        replay_head,
        {"role": "user", "content": "New user request in the second session."},
        {"role": "assistant", "content": "New assistant reply in the second session."},
    ])
    assert all("timestamp" in row and "_row_id" in row for row in supplied)
    digest_matches_input = (reconcile._emitted_user_replay_digest(supplied[0]) == head_digest) if head_digest else None
    engine.ingest(supplied)
    rows = engine._store.get_session_messages(child)
    observations = []
    for row in rows:
        relations = engine._store.connection.execute(
            "SELECT DISTINCT kind FROM message_relations WHERE store_id = ? ORDER BY kind", (row["store_id"],)
        )
        observations.append(dict(fingerprint(row), relation_kinds=[kind for kind, in relations],
                                 equals_s1_head=row["content"] == head["content"]))
    print(json.dumps(dict(
        count=len(observations), rows=observations, has_s1_head=any(row["equals_s1_head"] for row in observations),
        supplied=list(map(fingerprint, supplied)), stub_rows=len(result), digest_matches_input=digest_matches_input,
        s1_binding=s1_binding, s2_binding=[engine._session_id, engine._conversation_id],
        ingest_entry=ingest_entry, proof_consulted=bool(pending.get("consulted")),
    )))
finally:
    engine.shutdown()
    if sys.exc_info()[0] is not None:
        print(capture.text, file=sys.stderr)
'''


def _ordered_rows(observation):
    return [(row["role"], row["sha256"]) for row in observation["rows"]]


def test_hotfix_session_switch(tmp_path, old_tree, host_root):
    base = old_tree("v0.30.2", PINS)
    candidate = Path(__file__).resolve().parents[2]
    report = dict(revisions={"BASE": PINS["v0.30.2"], "CAND": CANDIDATE}, cells={}, failures=[], errors={})
    for shape in ("rotation", "plain"):
        report["cells"][shape] = {}
        for mode in ("unset", "on"):
            arms = report["cells"][shape][mode] = {}
            for arm, tree in (("BASE", base), ("CAND", candidate)):
                directory = tmp_path / f"{shape}-{mode}-{arm}"
                directory.mkdir()
                try:
                    arms[arm] = run_old(tree, directory, host_root, SESSION_SWITCH, mode=mode, data=shape)
                except (AssertionError, TimeoutExpired) as exc:
                    report["errors"][f"{shape}/{mode}/{arm}"] = str(exc)
            if {"BASE", "CAND"} <= arms.keys():
                baseline, current = arms["BASE"], arms["CAND"]
                if _ordered_rows(current) != _ordered_rows(baseline):
                    report["failures"].append(dict(shape=shape, mode=mode, BASE=_ordered_rows(baseline),
                                                   CAND=_ordered_rows(current)))
                if current["supplied"] != baseline["supplied"]:
                    report["errors"][f"{shape}/{mode}/input"] = "S2 host content differs between arms"
                if current["digest_matches_input"] is not True:
                    report["errors"][f"{shape}/{mode}/proof"] = "S2's stamped head did not match CAND's pending digest"

    print("SWITCH_RESULT " + json.dumps(report, sort_keys=True))
    assert not report["failures"], report["failures"]
    assert not report["errors"], report["errors"]
