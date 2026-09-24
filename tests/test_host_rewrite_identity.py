"""A host's in-place rewrite of a stored user row keeps its replay identity (#498).

Hermes' ACP adapter persists ``prompt.strip()``: at turn end it rewrites the current
user dict in place, AFTER LCM already stored the raw prompt -- from the preflight
maintenance ingest on nearly every turn (no compaction, no commit proof), or from a
same-turn compaction. LCM records a store_id-keyed identity override for the row it
stored; the stored content itself is never modified.

The driver mirrors the host sequence with the real engine and a stub summarizer.
"""

import json
import sqlite3
import sys

import pytest

import hermes_lcm.engine as lcm_engine_module
import hermes_lcm.reconcile as lcm_reconcile_module
from hermes_lcm.engine import LCMEngine

from hermes_lcm.ingest_protection import scan_externalized_payload_integrity
from tests.test_compression_boundary import (
    _config,
    _stub_summarizer,
    _summary_carrier_fixture,
)


def _user(i, trail=True, text=None):
    body = text if text is not None else f"[T{i:02d}] user turn {i}: " + ("alpha beta gamma delta " * 40) + "end."
    return {"role": "user", "content": body + ("\n" if trail else "")}


def _reply(i):
    return {"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}."}


class _AcpHost:
    """Hermes' ACP turn: preflight ingest of the raw prompt, optional same-turn
    compaction commit, then the persist override rewrites the current user dict."""

    def __init__(self, tmp_path, monkeypatch, **config):
        monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
        self.tmp_path, self.config = tmp_path, config
        self.engine = self._start()
        self.history = []

    def _start(self):
        engine = LCMEngine(config=_config(self.tmp_path, **self.config), hermes_home=str(self.tmp_path / "home"))
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        return engine

    def turn(self, i, *, trail=True, compact=False, rewrite=True, text=None, crash=False):
        self.history.append(_user(i, trail, text))
        self.engine.should_compress_preflight(self.history)  # ingests the RAW row
        if compact:
            compressed = self.engine.compress(list(self.history), force=True)
            assert self.engine._last_compression_status == "compacted", self.engine._last_compression_noop_reason
            self.engine.on_session_end("S0", self.history)
            self.engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
            self.history = compressed
        if rewrite:
            self.history[-1]["content"] = self.history[-1]["content"].strip()  # finalize_turn, in place
        self.history.append(_reply(i))
        if not crash:  # a crash dies before the post_llm_call ingest
            self.engine.ingest(self.history)

    def compact_turn(self, i):
        """A turn whose compaction the host adopts only when it published."""
        self.history.append(_user(i))
        self.engine.should_compress_preflight(self.history)
        compressed = self.engine.compress(list(self.history), force=True)
        status = self.engine._last_compression_status
        if status == "compacted":
            self.engine.on_session_end("S0", self.history)
            self.engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
            self.history = compressed
        self.history[-1]["content"] = self.history[-1]["content"].strip()
        self.history.append(_reply(i))
        self.engine.ingest(self.history)
        return status

    def restart(self, raw_indexes=(), *, graceful=True, drop_last_reply=False):
        """A new process. The host resumes its transcript: the next turn's preflight
        ingest resends it plus the new prompt. Hermes' state.db keeps a same-turn
        compaction's prompt in the RAW form (the commit persisted it before the
        persist override), so ``raw_indexes`` of the history are resent untrimmed."""
        if graceful:
            self.engine.shutdown()
        self.engine = self._start()
        self.history = [dict(m) for m in self.history]  # rebuilt from state.db
        if drop_last_reply:  # interrupted turn: no reply, post_llm_call never fired
            self.history.pop()
        for index in raw_indexes:
            self.history[index] = dict(self.history[index], content=self.history[index]["content"].strip() + "\n")

    def rows(self):
        return self.engine._store._conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall()

    def overrides(self):
        return dict(
            self.engine._store._conn.execute(
                "SELECT key, value FROM metadata WHERE key LIKE 'host_rewrite_identity:%' ORDER BY key"
            ).fetchall()
        )

    def compact(self, i):
        self.history.append(_user(i, trail=False))
        self.engine.ingest(self.history)
        self.engine.compress(list(self.history), force=True)
        return self.engine._last_compression_status, self.engine._last_compression_noop_reason


def _no_duplicates(rows):
    normalized = [(role, (content or "").strip()) for _sid, role, content in rows]
    return len(normalized) == len(set(normalized))


@pytest.mark.parametrize("trail", [True, False], ids=["acp-trailing-newline", "no-whitespace"])
def test_preflight_ingest_then_host_trim_then_compaction_publishes(tmp_path, monkeypatch, trail):
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 13):
            host.turn(i, trail=trail)
        assert len(host.rows()) == 24
        assert len(host.overrides()) == (12 if trail else 0)
        assert host.compact(13) == ("compacted", "")
        assert _no_duplicates(host.rows())
    finally:
        host.engine.shutdown()


def test_restart_before_the_first_compaction_stores_no_duplicates(tmp_path, monkeypatch):
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 13):
            host.turn(i)
        host.restart()
        host.turn(13)
        assert len(host.rows()) == 26
        assert host.compact(14) == ("compacted", "")
        assert _no_duplicates(host.rows())
    finally:
        host.engine.shutdown()


@pytest.mark.parametrize("state_db_form", [False, True], ids=["live-form", "state-db-form"])
@pytest.mark.parametrize("commits", [1, 2], ids=["after-commit-1", "after-commit-2"])
def test_restart_after_a_same_turn_compaction_commit_stores_no_duplicates(
    tmp_path, monkeypatch, commits, state_db_form
):
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 13):
            host.turn(i)
        host.turn(13, compact=True)
        if commits == 2:
            for i in range(14, 20):
                host.turn(i)
            host.turn(20, compact=True)
        rows = len(host.rows())
        last = 20 if commits == 2 else 13
        host.restart(raw_indexes=(len(host.history) - 2,) if state_db_form else ())
        for i in range(last + 1, last + 7):
            host.turn(i)
        assert len(host.rows()) == rows + 12
        assert host.compact(last + 7) == ("compacted", "")
        assert _no_duplicates(host.rows())
    finally:
        host.engine.shutdown()


def test_repeated_identical_prompts_each_keep_their_row(tmp_path, monkeypatch):
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 10):
            host.turn(i)
        for i in range(10, 13):
            host.turn(i, text="continue")
        host.turn(13, text="continue", compact=True)
        rows = len(host.rows())
        # state.db resends the compaction turn's "continue\n" raw, the others trimmed
        host.restart(raw_indexes=(len(host.history) - 2,))
        for i in range(14, 20):
            host.turn(i)
        assert len(host.rows()) == rows + 12
        roles = [role for _sid, role, content in host.rows() if (content or "").strip() == "continue"]
        assert roles == ["user"] * 4
        assert host.compact(20) == ("compacted", "")
        assert _no_duplicates([row for row in host.rows() if (row[2] or "").strip() != "continue"])
    finally:
        host.engine.shutdown()


def test_override_leaves_stored_content_and_fts_unchanged_and_is_idempotent(tmp_path, monkeypatch):
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 7):
            host.turn(i)
        conn = host.engine._store._conn
        users = [(sid, content) for sid, role, content in host.rows() if role == "user"]
        assert all(content.endswith("end.\n") for _sid, content in users)  # the raw form LCM stored
        fts = conn.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'alpha' ORDER BY rowid").fetchall()
        overrides = host.overrides()
        assert set(overrides) == {f"host_rewrite_identity:{sid}" for sid, _content in users}
        payload = json.loads(next(iter(overrides.values())))
        assert payload["stripped"] == ["", "\n"] and not payload["content"].endswith("\n")
        host.restart()
        host.engine.ingest(list(host.history) + [_user(7)])
        host.engine.ingest(list(host.history) + [_user(7)])
        assert host.overrides() == overrides
        assert [(sid, content) for sid, role, content in host.rows() if role == "user"][:-1] == users
        assert conn is not host.engine._store._conn
        fts_after = host.engine._store._conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'alpha' ORDER BY rowid"
        ).fetchall()
        assert fts_after[: len(fts)] == fts  # only the new prompt's row was added
    finally:
        host.engine.shutdown()


def test_host_that_keeps_the_raw_form_gets_no_override(tmp_path, monkeypatch):
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 13):
            host.turn(i, rewrite=False)
        assert host.overrides() == {}
        host.restart()
        host.turn(13, rewrite=False)
        assert len(host.rows()) == 26
        assert host.compact(14) == ("compacted", "")
    finally:
        host.engine.shutdown()


def test_externalized_large_user_row_keeps_its_identity(tmp_path, monkeypatch):
    host = _AcpHost(
        tmp_path,
        monkeypatch,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
    )
    try:
        for i in range(1, 13):
            host.turn(i)
        users = [content for _sid, role, content in host.rows() if role == "user"]
        assert all(not content.startswith("[T") for content in users)  # stored as externalized refs
        conn, engine = host.engine._store._conn, host.engine
        stored = [dict(zip(("store_id", "role", "content"), row)) for row in host.rows() if row[1] == "user"]
        live = [m for m in host.history if m["role"] == "user"]
        assert [engine._message_replay_identity(dict(r, session_id="S0"), stored_row=True, with_host_rewrite=True)
                for r in stored] == [engine._message_replay_identity(m) for m in live]
        # The override reuses the stored payload: no second payload file per prompt (#498 lri7i).
        scan = scan_externalized_payload_integrity(conn, engine._config, hermes_home=str(tmp_path / "home"))
        assert scan["externalized_payload_files_unreferenced"] == 0, scan
        host.restart()
        host.turn(13)
        assert len(host.rows()) == 26
        assert host.compact(14) == ("compacted", "")
        assert _no_duplicates(host.rows())
    finally:
        host.engine.shutdown()


@pytest.mark.parametrize(("raw", "trimmed"), [(" retry\n", "retry"), ("  \n", "")], ids=["retry", "whitespace-only"])
def test_override_never_turns_a_new_identical_exchange_into_replay(tmp_path, raw, trimmed):
    """#498 r2 P0: the override proves the old row's rewrite, not that a later
    identical exchange is replay. A restarted host that sends just
    [user, assistant] again gets a full, unanchored replay decision: exact."""
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        user = {"role": "user", "content": raw}
        engine.ingest([user])
        user["content"] = trimmed  # in-place host rewrite
        engine.ingest([user, {"role": "assistant", "content": "OK"}])
        assert len(engine._store._conn.execute("SELECT key FROM metadata WHERE key LIKE 'host_rewrite%'").fetchall()) == 1
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        resumed.ingest([{"role": "user", "content": trimmed}, {"role": "assistant", "content": "OK"}])
        assert resumed._store.get_session_count("S0") == 4
    finally:
        resumed.shutdown()


def test_mapper_binds_each_occurrence_to_either_form_of_its_row(tmp_path, monkeypatch):
    """#498 r2 P1: two overridden "continue\n" rows, resent as [raw, trimmed]
    (state.db raw for one, live trimmed for the other): both map, in order."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        host.turn(1)
        host.turn(2, text="continue")
        host.turn(3, text="continue")
        assert len(host.overrides()) == 3
        stored = [sid for sid, role, content in host.rows() if content == "continue\n"]
        active = [dict(m) for m in host.history]
        active[2]["content"] = "continue\n"
        mapped = host.engine._get_store_id_map_for_messages(active)
        assert [mapped.get(id(active[i])) for i in (2, 4)] == stored
    finally:
        host.engine.shutdown()


def test_rekey_declines_an_ambiguous_input_occurrence(tmp_path, monkeypatch):
    """#498 r2 P2: two watched "continue\n" inputs, one output copy. The copy's
    row is ambiguous, so no watch acquires it: only a row whose own object was
    rewritten may get an override."""
    host = _AcpHost(tmp_path, monkeypatch, fresh_tail_count=4)
    try:
        for i in range(1, 5):
            host.turn(i)
        host.turn(5, text="continue", rewrite=False)
        for i in range(6, 9):
            host.turn(i, rewrite=False)
        before = set(host.overrides())
        host.turn(9, text="continue", compact=True)
        continue_rows = [sid for sid, role, content in host.rows() if content == "continue\n"]
        assert len(continue_rows) == 2
        assert f"host_rewrite_identity:{continue_rows[0]}" not in set(host.overrides()) - before
    finally:
        host.engine.shutdown()


def test_rekey_host_rewrite_watch_tracks_emitted_summary_carrier(tmp_path):
    """The carrier has the historical row's replay identity, so #498's watch
    transfers that exact store occurrence to the carrier object."""
    engine, _compacted, tail, tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        engine._watch_stored_user_rows([(tail[0], tail[0], tail_ids[0])])
        assembled = engine._assemble_context(None, tail)
        carrier = assembled[0]

        engine._rekey_host_rewrite_watch(tail, assembled)

        _identity, _stored, objects = engine._host_rewrite_state()[0][tail_ids[0]]
        assert objects[0] is tail[0]
        assert objects[-1] is carrier
        assert engine._generated_context_carrier_remainder(carrier) == tail[0]["content"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("prior_commit", "same_turn", "graceful", "interrupted"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (False, True, False, False),
        (False, False, True, True),
    ],
    ids=[
        "a-crash-no-compaction",
        "b1-crash-after-earlier-commit",
        "b2-crash-same-turn-commit-live-form",
        "b3-crash-first-commit-same-turn",
        "a-interrupted-then-restart",
    ],
)
def test_restart_after_a_rewrite_lcm_never_saw(tmp_path, monkeypatch, prior_commit, same_turn, graceful, interrupted):
    """#498 lrfow: the process dies (or the turn is interrupted, so post_llm_call
    never fires) after the persist rewrite: that row has no override. The resumed
    session loses nothing, stores no duplicate and every compaction publishes."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        turn = 12
        for i in range(1, turn + 1):
            host.turn(i)
        if prior_commit:
            host.turn(turn + 1, compact=True)
            for i in range(turn + 2, turn + 5):
                host.turn(i)
            turn += 4
        turn += 1
        host.turn(turn, compact=same_turn, crash=True)
        host.restart(graceful=graceful, drop_last_reply=interrupted)
        statuses = []
        for block in range(3):
            for i in range(turn + 1, turn + 7):
                host.turn(i)
            statuses.append(host.compact_turn(turn + 7))
            turn += 7
        assert statuses == ["compacted"] * 3
        rows = [(role, (content or "").strip()) for _sid, role, content in host.rows()]
        assert len(rows) == len(set(rows))
        wanted = {("user", _user(i)["content"].strip()) for i in range(1, turn + 1)}
        wanted |= {("assistant", _reply(i)["content"]) for i in range(1, turn + 1)}
        wanted -= {("assistant", _reply(13 if not prior_commit else 17)["content"])} if interrupted else set()
        assert wanted <= set(rows)
    finally:
        host.engine.shutdown()


@pytest.mark.parametrize("fails", [0, 1], ids=["no-failure", "one-transient-failure"])
def test_override_write_failure_retries_on_the_next_ingest(tmp_path, monkeypatch, fails):
    """#498 lri7e: a transient "database is locked" on the override write keeps the
    watch, so the next ingest records the override and a restart stays exact."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 5):
            host.turn(i)
        real, left = host.engine._store.write_metadata_json, [fails]

        def flaky(keys, serialized, **kwargs):
            if left[0] and any(key.startswith("host_rewrite_identity:") for key in keys):
                left[0] -= 1
                raise sqlite3.OperationalError("database is locked")
            return real(keys, serialized, **kwargs)

        monkeypatch.setattr(host.engine._store, "write_metadata_json", flaky)
        for i in range(5, 13):
            host.turn(i)
        assert len(host.overrides()) == 12
        host.restart()
        host.turn(13)
        assert len(host.rows()) == 26
        assert _no_duplicates(host.rows())
        assert host.compact(14) == ("compacted", "")
    finally:
        host.engine.shutdown()


def _tool_pair(tag):
    return [
        {"role": "assistant", "content": "step " + "c" * 160,
         "tool_calls": [{"id": tag, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": tag, "content": f"result {tag} " + "d" * 160},
    ]


def test_trimmed_retained_anchor_stays_retained_through_later_compactions(tmp_path, monkeypatch):
    """#498 r3 (Sol P1): the sole user prompt retained behind the system prompt is
    trimmed in place by the host after a fresh_tail_count=0 compaction kept it. It
    must stay the retained anchor (raw or override form, one occurrence)."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=0, leaf_chunk_tokens=1), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        user = {"role": "user", "content": "the only real user prompt\n"}
        messages = [{"role": "system", "content": "stable system prompt"}, user]
        for k in range(4):
            messages += _tool_pair(f"a{k}")
        engine.should_compress_preflight(messages)
        out = engine.compress(list(messages), force=True)
        engine.on_session_end("S0", messages)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        assert out[1]["content"] == "the only real user prompt\n"
        out[1]["content"] = out[1]["content"].strip()  # finalize_turn, in place
        for round_ in ("b", "c"):
            messages = list(out) + [m for k in range(4) for m in _tool_pair(f"{round_}{k}")]
            engine.ingest(messages)
            out = engine.compress(list(messages), force=True)
            assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
            assert engine._leading_anchor_count(messages) == 2
            assert out[1]["content"] == "the only real user prompt"
    finally:
        engine.shutdown()


def test_head_replay_wins_over_a_shorter_tool_anchored_cursor(tmp_path, monkeypatch):
    """#498 r3 (Sol P2): the greatest independently proven cursor wins."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        host.history += _tool_pair("t1")  # a durable tool pair proves cursor 2 on its own
        host.history.append({"role": "user", "content": " run the tool\n"})
        host.engine.should_compress_preflight(host.history)
        host.history[-1]["content"] = host.history[-1]["content"].strip()
        host.history.append({"role": "assistant", "content": "done"})
        host.engine.ingest(host.history)
        assert len(host.rows()) == 4
        host.restart()
        host.history.append({"role": "user", "content": "next prompt"})
        host.engine.should_compress_preflight(host.history)
        assert len(host.rows()) == 5
    finally:
        host.engine.shutdown()


def test_mapper_surplus_shares_one_capacity_across_a_rows_forms(tmp_path, monkeypatch):
    """#498 r3 (Astra P2): [extra "continue", original "continue\\n", later "continue"]
    against [1 "continue\\n" (override), 2, 3 "continue", 4]: the extra occurrence is
    the surplus; the raw original keeps row 1."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        host.turn(1, text="continue")
        host.turn(2, text="continue", trail=False, rewrite=False)
        stored = [sid for sid, role, _content in host.rows() if role == "user"]
        active = [
            {"role": "user", "content": "continue"},
            {"role": "user", "content": "continue\n"},
            {"role": "user", "content": "continue"},
        ]
        mapped = host.engine._get_store_id_map_for_messages(active)
        assert [mapped.get(id(m)) for m in active] == [None, stored[0], stored[1]]
    finally:
        host.engine.shutdown()


def test_restart_with_mixed_raw_and_trimmed_forms_stores_only_the_new_prompt(tmp_path, monkeypatch):
    """#498 r3 (Sol P2): no commit proof; the resumed transcript carries one raw
    state.db row and trimmed rows elsewhere. Head replay compares both forms."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        host.turn(1)
        host.turn(2)
        host.restart(raw_indexes=(0,))
        host.history.append(_user(3))
        host.engine.should_compress_preflight(host.history)
        assert len(host.rows()) == 5
    finally:
        host.engine.shutdown()


@pytest.mark.parametrize("path", ["per-turn-ingest", "session-end"])
@pytest.mark.parametrize("fault", ["protect", "write"])
def test_capture_failure_never_blocks_the_ingest(tmp_path, monkeypatch, fault, path):
    """#498 r5 (ls6kv): capture is best-effort. A raise while protecting or writing
    an override must not stop the ingest; the watch stays and the next ingest retries."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        host.turn(1)
        failing = [True]
        if fault == "protect":
            real_protect = lcm_reconcile_module.protect_messages_for_ingest

            def broken(*args, **kwargs):
                if failing[0]:
                    raise RuntimeError("protect failed")
                return real_protect(*args, **kwargs)

            monkeypatch.setattr(lcm_reconcile_module, "protect_messages_for_ingest", broken)
        else:
            real_write = host.engine._store.write_metadata_json

            def broken(keys, serialized, **kwargs):
                if failing[0] and any(key.startswith("host_rewrite_identity:") for key in keys):
                    raise sqlite3.OperationalError("database is locked")
                return real_write(keys, serialized, **kwargs)

            monkeypatch.setattr(host.engine._store, "write_metadata_json", broken)
        host.history.append(_user(2))
        host.engine.should_compress_preflight(host.history)
        host.history[-1]["content"] = host.history[-1]["content"].strip()
        host.history.append(_reply(2))
        if path == "session-end":
            host.engine.on_session_end("S0", host.history)
        else:
            host.engine.ingest(host.history)
        assert len(host.rows()) == 4  # the final reply is stored despite the failed capture
        assert len(host.overrides()) == 1
        failing[0] = False
        if path == "session-end":
            host.engine.on_session_start("S0", platform="acp", context_length=200_000)
        host.turn(3)
        assert len(host.overrides()) == 3  # the kept watch recorded T2's override on retry
        assert len(host.rows()) == 6
        assert _no_duplicates(host.rows())
    finally:
        host.engine.shutdown()


def test_retained_anchor_trimmed_before_capture_survives_a_crash(tmp_path, monkeypatch):
    """#498 r5 (lszgN): the host trims the sole user prompt, then the process dies
    before any override. After restart the head replay binds the trimmed resend to the
    raw row; the retained anchor must accept that proof-normalized form too."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    config = _config(tmp_path, fresh_tail_count=0, leaf_chunk_tokens=1)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    messages = [{"role": "system", "content": "stable system prompt"},
                {"role": "user", "content": "the only real user prompt\n"}]
    for k in range(4):
        messages += _tool_pair(f"a{k}")
    engine.should_compress_preflight(messages)  # stores the RAW prompt, then the crash
    messages[1]["content"] = messages[1]["content"].strip()
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        messages = [dict(m) for m in messages] + [m for k in range(4) for m in _tool_pair(f"b{k}")]
        engine.ingest(messages)
        assert not engine._store._conn.execute(
            "SELECT 1 FROM metadata WHERE key LIKE 'host_rewrite_identity:%'").fetchall()
        out = engine.compress(list(messages), force=True)
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert engine._leading_anchor_count(messages) == 2
        assert out[1] == {"role": "user", "content": "the only real user prompt"}
        users = engine._store._conn.execute("SELECT content FROM messages WHERE role = 'user'").fetchall()
        assert users == [("the only real user prompt\n",)]  # stored once, never modified
    finally:
        engine.shutdown()


def _stored_user_rows(host):
    return [{"store_id": sid, "session_id": "S0", "role": role, "content": content}
            for sid, role, content in host.rows() if role == "user"]


def test_override_cache_stays_bounded_across_reads(tmp_path, monkeypatch):
    """#498 r6 (ltrlc): the read-through cache is FIFO-capped, whatever a long-lived
    process reads: at most the cap plus the batch being loaded."""
    monkeypatch.setattr(lcm_reconcile_module, "_HOST_REWRITE_OVERRIDE_CACHE_CAP", 4, raising=False)
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 13):
            host.turn(i)
        assert len(host.overrides()) == 12
        cache = host.engine._host_rewrite_state()[1]
        assert len(cache) <= 4 + 1
        rows = _stored_user_rows(host)
        for start in range(0, len(rows), 2):
            host.engine._load_host_rewrite_overrides(rows[start:start + 2])
            assert len(cache) <= 4 + 2
    finally:
        host.engine.shutdown()


def test_evicted_override_reloads_and_matches_as_before(tmp_path, monkeypatch):
    """#498 r6: eviction is safe. A positive override pushed out of the cache reloads
    from metadata for its consumers: the replay identity and a restart's head replay."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 13):
            host.turn(i)
        rows = _stored_user_rows(host)
        target = rows[0]
        key = f"host_rewrite_identity:{target['store_id']}"
        override_form = json.loads(host.overrides()[key])["content"]
        assert override_form == target["content"].strip() != target["content"]
        host.restart()
        monkeypatch.setattr(lcm_reconcile_module, "_HOST_REWRITE_OVERRIDE_CACHE_CAP", 2, raising=False)
        engine, reads = host.engine, []
        cache, real_read = engine._host_rewrite_state()[1], engine._store.read_metadata_json_many

        def spy(keys, *args, **kwargs):
            reads.append((sys._getframe(2).f_code.co_name, list(keys)))  # (loader's caller, keys)
            return real_read(keys, *args, **kwargs)

        monkeypatch.setattr(engine._store, "read_metadata_json_many", spy)

        def evict_target(others):
            engine._load_host_rewrite_overrides([target])
            assert isinstance(cache.get(target["store_id"]), dict)  # the positive override is cached
            engine._load_host_rewrite_overrides(others)  # past the cap: the oldest entries go
            assert target["store_id"] not in cache
            reads.clear()

        evict_target(rows[1:3])
        identity = engine._message_replay_identity(target, stored_row=True, with_host_rewrite=True)
        assert identity[1] == override_form
        assert ("_host_rewrite_override_content", [key]) in reads  # reloaded from metadata
        evict_target(rows[3:5])
        host.turn(13)  # the restart's head replay reads the stored session's overrides
        assert any(caller == "_cursor_from_host_rewrite_head" and key in batch for caller, batch in reads)
        assert len(host.rows()) == 26
        assert _no_duplicates(host.rows())
        assert host.compact(14) == ("compacted", "")
    finally:
        host.engine.shutdown()


def test_override_batch_larger_than_the_cap_serves_every_row(tmp_path, monkeypatch):
    """#498 r6: a batch larger than the cap is kept whole for the matcher that asked;
    only older entries are evicted."""
    host = _AcpHost(tmp_path, monkeypatch)
    try:
        for i in range(1, 13):
            host.turn(i)
        rows = _stored_user_rows(host)
        host.restart()
        monkeypatch.setattr(lcm_reconcile_module, "_HOST_REWRITE_OVERRIDE_CACHE_CAP", 3, raising=False)
        cache = host.engine._host_rewrite_state()[1]
        host.engine._load_host_rewrite_overrides(rows[:1])
        host.engine._load_host_rewrite_overrides(rows[1:])
        assert set(cache) == {row["store_id"] for row in rows[1:]}  # the batch whole, the older entry gone
        for row in rows[1:]:
            assert host.engine._host_rewrite_override_content(row) == row["content"].strip()
        assert set(cache) == {row["store_id"] for row in rows[1:]}  # served without a reload
    finally:
        host.engine.shutdown()
