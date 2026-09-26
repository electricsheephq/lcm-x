"""#519: a rotation (compression.in_place=false) commits, then the process dies before the in-flight reply.

The restored child list ends on the dangling user prompt: the end row of the child proof's carry
range. On the next turn Hermes glues the new prompt into that row and its persist step rewrites the
row to the new prompt alone (session_persistence.py:327-339 via turn_finalizer.py:264-270) before
post_llm_call ingests. R1 accepts that replaced last position (the carry range's end row is exactly
the vanished parent row) and trims it from the child's own carry; R2 voids the carry when the empty
child still re-stores its list; R3 logs the publication failure's ids. Controls store what the base
stores (duplicates, never loss).
"""
from __future__ import annotations

import json
import subprocess
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _config, _host_merge_consecutive_users, _stub_summarizer, _turn
from tests.test_issue_524_gen2_replay import _hermes_python

P, C, G = "P519", "C519", "G519"
T16, T17 = _turn(16), _turn(17)


def _base_behaviour(monkeypatch, *, no_r1=True, no_r2=True):
    """The base: R1 never accepts (``no_r1``) and no carry is ever rewritten (``no_r2``)."""
    if no_r1:
        monkeypatch.setattr(LCMEngine, "_replaced_carry_tail", lambda self, payload, digest_of: None)
    if no_r2:
        monkeypatch.setattr(LCMEngine, "_rewrite_own_carry_ranges", lambda self, *args: None)


def _real_repair(messages):
    """Hermes' own repair_message_sequence (pinned host), in its interpreter."""
    python, src = _hermes_python()
    done = subprocess.run(
        [python, "-c", "import json, sys\nfrom agent.agent_runtime_helpers import repair_message_sequence\n"
         "m = json.load(sys.stdin)\nrepair_message_sequence(None, m)\nprint(json.dumps(m))"],
        input=json.dumps(messages), cwd=src or "/", capture_output=True, text=True, timeout=300, check=False,
        env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


MERGES = {"emulated": _host_merge_consecutive_users, "real-hermes": _real_repair}
_REAL = pytest.mark.skipif(_hermes_python() is None, reason="no real Hermes runtime available")
MERGE_PARAMS = [pytest.param("emulated"), pytest.param("real-hermes", marks=_REAL)]


class _Rotation:
    """Parent P: 16 turns, then a dangling T16 (ACP stores T15/T16 with a trailing newline). A forced
    compaction with fresh tail 16 emits a user-role summary S + P18..P33; the host rotates to C."""

    def __init__(self, tmp_path, monkeypatch, *, dangling=True, tail=16):
        monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
        self.tmp_path = tmp_path
        self.engine, host = self.make(P, tail), []
        for i in range(16):
            user, reply = _turn(i)
            host.append({**user, "content": user["content"] + ("\n" if i == 15 else "")})
            self.engine.ingest(host)
            host.append(dict(reply))
            self.engine.ingest(host)
        if dangling:
            host.append({**T16[0], "content": T16[0]["content"] + "\n"})
            self.engine.ingest(host)
        self.output = deepcopy(self.engine.compress(list(host), force=True))
        assert self.engine._last_compression_status == "compacted"
        self.engine.on_session_end(P, list(host))
        self.engine.on_session_start(C, boundary_reason="compression", old_session_id=P, platform="acp")
        self.engine._config.fresh_tail_count = 6  # the child's compaction reaches past the carried rows
        self.proof = self.raw_proof()

    def make(self, session_id, tail=6):
        engine = LCMEngine(config=_config(self.tmp_path, fresh_tail_count=tail), hermes_home=str(self.tmp_path / "home"))
        engine.on_session_start(session_id, platform="acp", context_length=200_000)
        return engine

    def restart(self, session_id=C):
        self.engine.shutdown()
        self.engine = self.make(session_id)

    def raw_proof(self, session_id=C):
        return self.engine._store.read_metadata_json(f"compaction_commit_proof:{session_id}")

    def rows(self, session_id=C):
        return self.engine._store._conn.execute(
            "SELECT store_id, role, content FROM messages WHERE session_id = ? ORDER BY store_id", (session_id,)
        ).fetchall()

    def restored(self):
        """The host's restored child list: compress()'s output, user rows stripped (state.db)."""
        host = deepcopy(self.output)
        for message in host:
            if message["role"] == "user":
                message["content"] = message["content"].strip()
        return host

    def replaced(self, merge="emulated"):
        """The next turn: T17 glued into the dangling T16, then the persist step's in-place rewrite."""
        host = MERGES[merge](self.restored() + [dict(T17[0])])
        assert len(host) == 17 and host[16]["content"] == T16[0]["content"].strip() + "\n\n" + T17[0]["content"]
        host[16] = {**host[16], "content": T17[0]["content"]}
        return host + [dict(T17[1])]

    def turns(self, host, first, n=6):
        for i in range(first, first + n):
            for message in _turn(i):
                host.append(dict(message))
                self.engine.ingest(host)
        return host

    def nodes(self):
        return [json.loads(ids) for (ids,) in self.engine._dag._conn.execute(
            "SELECT source_ids FROM summary_nodes WHERE source_type = 'messages' ORDER BY node_id")]


def _conflicts(caplog):
    return [r.getMessage() for r in caplog.records if "publication_invariant_conflict" in r.getMessage()]


def _t1(tmp_path, monkeypatch, caplog, merge, *, restart=True):
    r = _Rotation(tmp_path, monkeypatch)
    assert r.proof["carry_ranges"] == [[P, 17, 33]] and r.proof["last_store_id"] == 0 and not r.rows()
    if restart:
        r.restart()
    host = r.replaced(merge)
    r.engine.ingest(host)
    first = [(role, content) for _id, role, content in r.rows()]
    proof = r.raw_proof()
    r.turns(host, 18)
    status = r.engine.compress(list(host), force=True) and r.engine._last_compression_status
    return r, host, first, proof, status


@pytest.mark.parametrize("merge", MERGE_PARAMS)
@pytest.mark.parametrize("restart", [True, False], ids=["restart", "in-process"])
def test_replaced_last_carried_row_stores_only_the_new_turn(tmp_path, monkeypatch, caplog, merge, restart):
    """T1 (restart) / T2 (in-process): the first child ingest stores T17 + r17 only; the child's own
    proof trims row 33 from its carry; its next compaction commits over 18..32 and never claims 33."""
    r, _host, first, proof, status = _t1(tmp_path, monkeypatch, caplog, merge, restart=restart)
    try:
        assert first == [("user", T17[0]["content"]), ("assistant", T17[1]["content"])]
        assert proof == {**r.proof, "carry_ranges": [[P, 17, 32]]}  # version 3 / descriptor 4 as written
        assert (r.proof["version"], r.proof["descriptor_version"]) == (3, 4)
        assert status == "compacted" and not _conflicts(caplog), r.engine._last_compression_noop_reason
        assert set(range(18, 33)) <= set(sum(r.nodes(), [])) and 33 not in sum(r.nodes(), [])
    finally:
        r.engine.shutdown()



def test_list_ending_before_the_replaced_position(tmp_path, monkeypatch):
    """R1's loop-exit arm: the host list ends on P32 (the dangling T16 is gone, nothing new yet):
    cursor 16, nothing stored, row 33 trimmed; the next turn's rows are then stored once."""
    r = _Rotation(tmp_path, monkeypatch)
    try:
        r.restart()
        host = r.restored()[:16]
        r.engine.ingest(host)
        assert r.engine._last_ingest_reconciliation["cursor"] == 16 and not r.rows()
        assert r.raw_proof()["carry_ranges"] == [[P, 17, 32]]
        r.engine.ingest(host + [dict(T17[0]), dict(T17[1])])
        assert [(role, content) for _id, role, content in r.rows()] == [("user", T17[0]["content"]), ("assistant", T17[1]["content"])]
    finally:
        r.engine.shutdown()


def test_a_failed_carry_write_keeps_the_record(tmp_path, monkeypatch):
    """The rewrite is fail-soft: a store error leaves the proof as written and the ingest proceeds."""
    r = _Rotation(tmp_path, monkeypatch)
    try:
        r.restart()

        def locked(*args, **kwargs):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(r.engine._store, "write_metadata_json", locked)
        r.engine.ingest(r.replaced())
        monkeypatch.undo()
        assert len(r.rows()) == 2 and r.raw_proof() == r.proof
    finally:
        r.engine.shutdown()

@pytest.mark.parametrize("restart", [True, False], ids=["restart", "in-process"])
def test_positive_control_the_base_re_stores_and_errors(tmp_path, monkeypatch, caplog, restart):
    """T1-red in the suite: without R1/R2 the child re-stores S + 15 carried rows + the new pair and its
    next compaction fails; R3's detail carries the publication's ids (authoritative 18..33 included)."""
    _base_behaviour(monkeypatch)
    r, _host, first, proof, status = _t1(tmp_path, monkeypatch, caplog, "emulated", restart=restart)
    try:
        assert len(first) == 18 and proof == r.proof
        assert status == "error" and len(_conflicts(caplog)) == 1
        detail = _conflicts(caplog)[0].split("detail=", 1)[1]
        assert "authoritative=[" + ", ".join(map(str, range(18, 34))) in detail, detail
    finally:
        r.engine.shutdown()


def _reply_flushed(tmp_path, monkeypatch, *, fixed):
    if not fixed:
        _base_behaviour(monkeypatch)
    r = _Rotation(tmp_path, monkeypatch)
    try:
        r.restart()
        host = r.restored() + [{"role": "assistant", "content": "reply to T16"}]
        r.engine.ingest(host)
        return r.engine._last_ingest_reconciliation["cursor"], r.rows(), r.raw_proof()["carry_ranges"]
    finally:
        r.engine.shutdown()


def test_negative_control_kill_after_the_reply_is_flushed(tmp_path, monkeypatch):
    """T3: the restored list [S, P18..P33, r16] walks exactly: cursor 17 on the base and the head."""
    fixed = _reply_flushed(tmp_path / "fixed", monkeypatch, fixed=True)
    monkeypatch.undo()
    base = _reply_flushed(tmp_path / "base", monkeypatch, fixed=False)
    assert fixed == base and fixed[0] == 17 and len(fixed[1]) == 1 and fixed[2] == [[P, 17, 33]]



@pytest.mark.xfail(strict=True, reason="#519 T4: after R1 stored T17 as the child's first row, a second restart restores "
                   "T16 + '\\n\\n' + T17 (the host's restore-time merge). #535 N1's composite arm accepts it at the last "
                   "output position, but the durable walk's post-proof row check (proof_identity(messages[index]) != "
                   "proof_identity(row, stored_row=True, with_host_rewrite=True)) compares that composite with stored row "
                   "T17 alone and returns None: no predicate admits a merge-append of the trimmed carry end row behind the "
                   "child's first stored row. The list is re-stored (18 rows, as on the base) and the compaction errors")
def test_second_restart_restores_the_composite(tmp_path, monkeypatch, caplog):
    """T4: after T1's first ingest the process restarts again. state.db restores [S, P18..P32, T16,
    T17, r17] and the restore-time repair glues T16 + T17 into one composite row. Target: 0 re-store
    and a committing compaction (#535 N1 + N2's owned-id guard)."""
    r = _Rotation(tmp_path, monkeypatch)
    try:
        r.restart()
        r.engine.ingest(r.replaced())
        before = r.rows()
        r.restart()
        host = _host_merge_consecutive_users(
            r.restored()[:16] + [{"role": "user", "content": T16[0]["content"].strip()}, dict(T17[0]), dict(T17[1])]
        )
        host.append(dict(_turn(18)[0]))
        r.engine.ingest(host)
        assert r.rows() == before + r.rows()[len(before):] and len(r.rows()) == len(before) + 1  # T18 only
        host.append(dict(_turn(18)[1]))
        r.engine.ingest(host)
        r.turns(host, 19)
        r.engine.compress(list(host), force=True)
        assert r.engine._last_compression_status == "compacted" and not _conflicts(caplog)
    finally:
        r.engine.shutdown()


def test_second_restart_stores_no_more_than_the_base(tmp_path, monkeypatch, caplog):
    """T4's pinned count: the head's two restarts store fewer rows than the base's (2 + 18 vs 18 + 18)."""
    def run(path, fixed):
        if not fixed:
            _base_behaviour(monkeypatch)
        r = _Rotation(path, monkeypatch)
        try:
            r.restart()
            r.engine.ingest(r.replaced())
            first = len(r.rows())
            r.restart()
            r.engine.ingest(_host_merge_consecutive_users(
                r.restored()[:16] + [{"role": "user", "content": T16[0]["content"].strip()}, dict(T17[0]), dict(T17[1])]
            ) + [dict(_turn(18)[0])])
            return first, len(r.rows()) - first
        finally:
            r.engine.shutdown()

    fixed = run(tmp_path / "fixed", True)
    base = run(tmp_path / "base", False)
    assert fixed == (2, 18) and base == (18, 18)


def _middle_rewrite(r):
    host = r.restored()
    host[5] = {**host[5], "content": host[5]["content"] + " (edited by host)"}
    return host + [{"role": "assistant", "content": "reply to T16"}]


def _backstop(tmp_path, monkeypatch, caplog, *, fixed):
    if not fixed:
        _base_behaviour(monkeypatch)
    r = _Rotation(tmp_path, monkeypatch)
    try:
        r.restart()
        host = _middle_rewrite(r)
        r.engine.ingest(host)
        first, carry = len(r.rows()), r.raw_proof()["carry_ranges"]
        r.turns(host, 17)
        out = deepcopy(r.engine.compress(list(host), force=True))
        status, nodes = r.engine._last_compression_status, r.nodes()
        r.engine.on_session_end(C, list(host))  # the host rotates whatever the engine reported (29->29 live)
        r.engine.on_session_start(G, boundary_reason="compression", old_session_id=C, platform="acp")
        r.engine.ingest(out + [dict(_turn(30)[0])])
        return first, carry, status, len(_conflicts(caplog)), nodes, r.rows(G)
    finally:
        r.engine.shutdown()


def test_r2_voids_the_carry_when_the_empty_child_re_stores(tmp_path, monkeypatch, caplog):
    """T5: a host rewrite of a MIDDLE carried row fails the walk (cursor 0, as on the base: the stub
    plus 16 carried copies plus the reply). R2 voids the child's carry, so its next compaction commits
    over its own copies only and the following host rotation re-stores nothing into the grandchild."""
    first, carry, status, conflicts, nodes, grandchild = _backstop(tmp_path / "fixed", monkeypatch, caplog, fixed=True)
    assert (first, carry, status, conflicts) == (18, [], "compacted", 0)
    assert all(store_id > 33 for store_id in nodes[-1])  # no voided parent row is claimed
    assert [(role, content) for _id, role, content in grandchild] == [("user", _turn(30)[0]["content"])]
    caplog.clear()
    monkeypatch.undo()
    base = _backstop(tmp_path / "base", monkeypatch, caplog, fixed=False)
    assert base[:4] == (18, [[P, 17, 33]], "error", 1) and len(base[5]) > 1  # the base re-stores again


def _assistant_last(r):
    host = r.restored()
    host[16] = {"role": "assistant", "content": "an assistant row where T16 was"}
    return host


def _tool_bearing_last(r):
    host = r.restored()
    host[16] = {"role": "user", "content": T17[0]["content"], "tool_call_id": "call_519"}
    return host + [dict(T17[1])]


def _non_last(r):
    host = r.replaced()
    host[5] = {**host[5], "content": host[5]["content"] + " (edited by host)"}
    return host


NEGATIVE = {  # (dangling T16, host list) -- each is today's None: the walk, then R2
    "assistant_last_row": (True, _assistant_last),
    "tool_bearing_last_row": (True, _tool_bearing_last),
    "non_last_position": (True, _non_last),
    "carry_ends_on_assistant_row": (False, lambda r: r.restored()[:-1] + [dict(T17[0]), dict(T17[1])]),
    "non_empty_child": (True, None),
}


def _negative_run(tmp_path, monkeypatch, case, **base):
    if base:
        _base_behaviour(monkeypatch, **base)
    dangling, shape = NEGATIVE[case]
    r = _Rotation(tmp_path, monkeypatch, dangling=dangling)
    try:
        r.restart()
        if shape is None:  # the child already owns r16, then restarts before the next turn's reply
            r.engine.ingest(r.restored() + [{"role": "assistant", "content": "reply to T16"}])
            r.restart()
            shape = lambda r: r.replaced()  # noqa: E731
        r.engine.ingest(shape(r))
        return r.rows(), json.dumps({**r.raw_proof(), "created_at": None}, sort_keys=True).replace(str(tmp_path), "")
    finally:
        r.engine.shutdown()


@pytest.mark.parametrize("case", sorted(NEGATIVE))
def test_r1_negative_controls_take_todays_path(tmp_path, monkeypatch, case):
    """R1 never fires: the head equals R1-disabled exactly and stores what the base stores."""
    fixed = _negative_run(tmp_path / "fixed", monkeypatch, case)
    monkeypatch.undo()
    no_r1 = _negative_run(tmp_path / "no-r1", monkeypatch, case, no_r1=True, no_r2=False)
    monkeypatch.undo()
    base = _negative_run(tmp_path / "base", monkeypatch, case, no_r1=True, no_r2=True)
    assert fixed == no_r1 and fixed[0] == base[0]
    carry = json.loads(fixed[1])["carry_ranges"]
    assert carry == ([] if case != "non_empty_child" else json.loads(base[1])["carry_ranges"])


def test_r1_requires_the_carry_end_digest_and_an_empty_child(tmp_path, monkeypatch):
    """Conditions (3) and (4) directly: a carry end row that is not the proof's last target, or a
    child row after the proof's last_store_id, rejects the replaced tail."""
    r = _Rotation(tmp_path, monkeypatch)
    try:
        payload = r.engine._durable_commit_proof_payload()
        exact = lambda row: payload["effective_sha256"][-1]  # noqa: E731
        assert r.engine._replaced_carry_tail(payload, exact) == (P, 17, 33)
        assert r.engine._replaced_carry_tail(payload, lambda row: "another digest") is None
        assert r.engine._replaced_carry_tail({**payload, "carry_ranges": []}, exact) is None
        r.engine.ingest(r.restored() + [{"role": "assistant", "content": "reply to T16"}])
        assert r.engine._replaced_carry_tail(payload, exact) is None
    finally:
        r.engine.shutdown()


def test_n2_never_inserts_the_trimmed_row(tmp_path, monkeypatch):
    """#535 N2 x R1: the composite stored as the child's first row (N1) consumes its base, parent
    row 33, only while the carry owns it; once 33 is trimmed it is unowned and never inserted."""
    r = _Rotation(tmp_path, monkeypatch)
    try:
        r.restart()
        r.engine.ingest(_host_merge_consecutive_users(r.restored() + [dict(T17[0])]))
        ((composite, content),) = r.engine._store._conn.execute(
            "SELECT store_id, content FROM messages WHERE session_id = ?", (C,)).fetchall()
        assert content.endswith("\n\n" + T17[0]["content"])
        assert r.engine._with_merge_append_bases([composite], set()) == [33, composite]
        r.engine._rewrite_own_carry_ranges([(P, 17, 32)], "test: R1's trim", [33])
        assert r.engine._with_merge_append_bases([composite], set()) == [composite]
    finally:
        r.engine.shutdown()


@pytest.mark.parametrize("shape", ["trimmed", "voided"])
def test_n3_never_maps_back_into_the_parent_range(tmp_path, monkeypatch, caplog, shape):
    """#535 N3 x R1/R2: after the child's committed compaction, the lineage the #524 term and the #457
    resume replay (collapsed or not) never holds the trimmed row 33 or a voided parent row."""
    r = _Rotation(tmp_path, monkeypatch)
    try:
        r.restart()
        host = r.replaced() if shape == "trimmed" else _middle_rewrite(r)
        r.engine.ingest(host)
        r.turns(host, 18)
        r.engine.compress(list(host), force=True)
        assert r.engine._last_compression_status == "compacted" and not _conflicts(caplog)
        frontier = int(r.engine._last_compacted_store_id)
        lineage = r.engine._head_lineage_rows(host[0], frontier, 200)
        collapsed = r.engine._collapse_merge_append_bases(lineage + r.engine._store.get_session_messages_after(C, frontier))
        for rows in (lineage, collapsed):
            ids = {int(row["store_id"]) for row in rows}
            assert 33 not in ids and (shape == "trimmed" or not ids & set(range(18, 34))), sorted(ids)
        assert (shape == "trimmed") == bool({int(row["store_id"]) for row in lineage} & set(range(18, 33)))
    finally:
        r.engine.shutdown()
