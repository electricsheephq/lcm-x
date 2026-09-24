"""A host's in-place rewrite of a stored user row keeps its replay identity (#498).

Hermes' ACP adapter persists ``prompt.strip()``: at turn end it rewrites the current
user dict in place, AFTER LCM already stored the raw prompt -- from the preflight
maintenance ingest on nearly every turn (no compaction, no commit proof), or from a
same-turn compaction. LCM records a store_id-keyed identity override for the row it
stored; the stored content itself is never modified.

The driver mirrors the host sequence with the real engine and a stub summarizer.
"""

import json

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.engine import LCMEngine

from tests.test_compression_boundary import _config, _stub_summarizer


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

    def turn(self, i, *, trail=True, compact=False, rewrite=True, text=None):
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
        self.engine.ingest(self.history)

    def restart(self, raw_indexes=()):
        """A new process. The host resumes its transcript: the next turn's preflight
        ingest resends it plus the new prompt. Hermes' state.db keeps a same-turn
        compaction's prompt in the RAW form (the commit persisted it before the
        persist override), so ``raw_indexes`` of the history are resent untrimmed."""
        self.engine.shutdown()
        self.engine = self._start()
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
