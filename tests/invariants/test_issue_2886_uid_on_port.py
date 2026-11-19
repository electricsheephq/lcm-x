"""#2886 on this line: the gateway name-only cold bind with LCM_HOST_MESSAGE_UID=on (port 2916)."""

from collections import Counter
from contextlib import closing
import sqlite3
import weakref

import pytest

from hermes_lcm.config import uid_recognition_enabled
from hermes_lcm.engine import LCMEngine
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_2886_cold_bind_tool_name import (
    EXCHANGES,
    SESSION,
    TOOL,
    ZERO_BINDS,
    Transcript,
    _bind_counts,
    _seed,
    bind,  # noqa: F401
)

pytestmark = pytest.mark.issue(2886)

SEED_ROWS = (EXCHANGES + 1) * 4
ROWS_PER_BIND = 8  # _bind_counts runs two turns per bind

# Modes: "env" sets only LCM_HOST_MESSAGE_UID=on (no host state.db, so no UID binding);
# "active" adds the host state.db lineage, but the transcript hands each new user row to the
# preflight before its message_uid, so user rows stay unbound and the UID-bound prefix stops at
# row 0; "stamped" gives the user row its UID first, so every row is bound and the UID prefix
# stops at the assistant call that owns the first nameless tool result (exchange 60 under
# "one", exchange 0 under "all"), which sends a UID cursor and cut budget into the store-tail
# search.
MODES = ("env", "active", "stamped")
BINDINGS = {"env": 0, "active": 3 * (EXCHANGES + 1), "stamped": 4 * (EXCHANGES + 1)}

# Pinned per (mode, name_only): the cold UID prefix, which is also the UID cursor that reaches
# the store-tail search (None when no cut budget does), and one reason per cold bind. The
# durable tail wins in every case: the search returns the full stored count through port
# 2916's name-only early return, which is greater than the UID-bound prefix.
PINNED_UID_PREFIXES = {
    ("env", "one"): 0, ("env", "all"): 0, ("active", "one"): 0, ("active", "all"): 0,
    ("stamped", "one"): 241, ("stamped", "all"): 1,
}
PINNED_SEARCH_UID_CURSORS = {
    ("env", "one"): None, ("env", "all"): None, ("active", "one"): None, ("active", "all"): None,
    ("stamped", "one"): 241, ("stamped", "all"): 1,
}
DURABLE_TAIL = ["replayed durable tail"] * len(ZERO_BINDS)
PINNED_REASONS = {(mode, name_only): DURABLE_TAIL for mode in MODES for name_only in ("one", "all")}


@pytest.fixture(autouse=True)
def _enable_uid_recognition(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")


def _first_ingests(monkeypatch):
    """Record each engine's first ingest: its row count and the reconciliation it recorded."""
    records, seen = [], weakref.WeakSet()
    original = LCMEngine.ingest

    def ingest(self, messages, *args, **kwargs):
        first = self not in seen
        seen.add(self)
        result = original(self, messages, *args, **kwargs)
        if first:
            records.append((list(messages), dict(self._last_ingest_reconciliation)))
        return result

    monkeypatch.setattr(LCMEngine, "ingest", ingest)
    return records


def _tail_searches(monkeypatch):
    """Record each engine's first store-tail search: the UID cursor it was given and the cursor it found."""
    searches, seen = [], weakref.WeakSet()
    original = LCMEngine._find_reconciled_cursor_for_store_tail

    def search(self, messages, *args, **kwargs):
        result = original(self, messages, *args, **kwargs)
        if self not in seen:
            seen.add(self)
            searches.append((kwargs.get("uid_cursor"), result))
        return result

    monkeypatch.setattr(LCMEngine, "_find_reconciled_cursor_for_store_tail", search)
    return searches


def _uid_prefixes(monkeypatch):
    """Record each engine's first cold UID prefix: rows matched by UID before the first unmatched one."""
    prefixes, seen = [], weakref.WeakSet()
    original = LCMEngine._host_uid_resolve

    def resolve(self, messages, *args, **kwargs):
        matches = original(self, messages, *args, **kwargs)
        if kwargs.get("prefix_only") and self not in seen:
            seen.add(self)
            prefixes.append(next((i for i in range(len(messages)) if i not in matches), len(messages)))
        return matches

    monkeypatch.setattr(LCMEngine, "_host_uid_resolve", resolve)
    return prefixes


def _host_binding_count(tmp_path):
    with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'host_uid_bindings'"
        ).fetchone():
            return 0
        return conn.execute("SELECT count(*) FROM host_uid_bindings").fetchone()[0]


def _stamp_user_before_preflight(monkeypatch, transcript):
    """Give the new user row the UID the transcript will assign before the engine first sees it."""
    original = LCMEngine.should_compress_preflight

    def preflight(self, messages, *args, **kwargs):
        if messages and messages[-1].get("role") == "user" and not messages[-1].get("message_uid"):
            messages[-1]["message_uid"] = f"message-{len(transcript.rows) + 1}"
        return original(self, messages, *args, **kwargs)

    monkeypatch.setattr(LCMEngine, "should_compress_preflight", preflight)


def _run_cell(bind, tmp_path, monkeypatch, mode, name_only):  # noqa: F811
    transcript = Transcript(name_only)
    if mode != "env":
        _state_db(tmp_path, [(SESSION, None, None)])
    if mode == "stamped":
        _stamp_user_before_preflight(monkeypatch, transcript)
    _seed(bind, transcript, gateway=True)
    bindings = _host_binding_count(tmp_path)
    with bind() as engine:
        recognition = uid_recognition_enabled(engine)
    records = _first_ingests(monkeypatch)
    searches = _tail_searches(monkeypatch)
    prefixes = _uid_prefixes(monkeypatch)
    counts = _bind_counts(bind, transcript)
    print(f"{mode} {name_only}: counts={counts} bindings={bindings} recognition={recognition}")
    print(f"{mode} {name_only}: searches (uid_cursor, found)={searches} uid prefixes={prefixes}")
    assert counts == ZERO_BINDS
    assert prefixes == [PINNED_UID_PREFIXES[(mode, name_only)]] * len(ZERO_BINDS)
    # The search finds the whole stored history, past any UID cursor it was given.
    assert searches == [
        (PINNED_SEARCH_UID_CURSORS[(mode, name_only)], SEED_ROWS + k * ROWS_PER_BIND)
        for k in range(len(ZERO_BINDS))
    ]

    # Each cold bind's first ingest carries the full prior history plus the new turn's rows.
    assert len(records) == len(ZERO_BINDS)
    reasons = []
    for k, (messages, recon) in enumerate(records):
        history = SEED_ROWS + k * ROWS_PER_BIND
        assert messages[history - 1]["message_uid"] == f"message-{history}"
        assert messages[history]["message_uid"] == f"message-{history + 1}"
        print(f"bind {k}: incoming={len(messages)} history={history} recon={recon}")
        # The bind reconciles before its first ingest, on a view that may hold only the new
        # user row; either way the cursor sits exactly at the end of the stored history.
        assert history < recon["incoming"] <= len(messages)
        assert recon["cursor"] == history
        reasons.append(recon["reason"])

    with closing(sqlite3.connect(tmp_path / "lcm.db")) as conn:
        names = dict(conn.execute("SELECT tool_call_id, tool_name FROM messages WHERE role = 'tool'"))
        stored = Counter(conn.execute(
            "SELECT role, content, tool_call_id FROM messages WHERE session_id = ?", (SESSION,),
        ).fetchall())
    assert names == {
        f"call_{i:012x}": None if name_only == "all" or i == 60 else TOOL
        for i in range(len(transcript.rows) // 4)
    }
    # Content-hash coverage: each transcript occurrence is stored exactly once.
    expected = Counter((row["role"], row["content"], row.get("tool_call_id")) for row in transcript.rows)
    assert stored == expected
    return reasons, recognition, bindings


@pytest.mark.parametrize("name_only", ["one", "all"])
def test_gateway_name_only_cold_binds_uid_on(bind, tmp_path, monkeypatch, name_only):  # noqa: F811
    reasons, recognition, bindings = _run_cell(bind, tmp_path, monkeypatch, "env", name_only)
    # The mode alone: no host state.db lineage beside the store, so the seed binds no UID.
    assert recognition
    assert bindings == BINDINGS["env"]
    assert reasons == PINNED_REASONS[("env", name_only)]


@pytest.mark.parametrize("mode", ["active", "stamped"])
@pytest.mark.parametrize("name_only", ["one", "all"])
def test_gateway_name_only_cold_binds_uid_active(bind, tmp_path, monkeypatch, mode, name_only):  # noqa: F811
    reasons, recognition, bindings = _run_cell(bind, tmp_path, monkeypatch, mode, name_only)
    # Control: UID recognition is on and the seed wrote host UID bindings into the store
    # ("active": each exchange's assistant call, tool result and reply; "stamped": all four rows).
    assert recognition
    assert bindings == BINDINGS[mode]
    assert reasons == PINNED_REASONS[(mode, name_only)]
