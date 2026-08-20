"""The store-backed owner resolver fails CLOSED, on both of its faults.

`TeamsPolicy` already refuses when its `session_owner` callback RAISES (#68).
The resolver `resolution._session_owner_for_engine` binds to the store, and it
had two ways to answer "unclaimed" when it did not actually know:

* every read raised `sqlite3.OperationalError` -- a lock, a schema fault, an
  un-migrated store -- and the loop returned `None`, which the policy is
  DELIBERATELY permissive about because an unclaimed session cannot belong to
  anyone else;
* rows for one session carried more than one owner stamp, and `LIMIT 1` picked
  one arbitrarily and reported it as authoritative -- enough for the chosen
  principal to pass a session-level reset, compaction or lifecycle gate over
  another principal's rows.

Both are answers the store cannot support, so neither is `None` any more.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timezone

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import FailClosedPolicy, TeamsPolicy, policy_for_engine
from hermes_lcm.access_policy.resolution import (
    OwnerLookupError,
    _session_owner_for_engine,
)
from hermes_lcm.teams import catalog


_OWNER_TABLES = """
CREATE TABLE messages (
    store_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    content      TEXT,
    access_scope TEXT
);
CREATE TABLE summary_nodes (
    node_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    summary      TEXT,
    access_scope TEXT
);
"""


class _Store:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection


class _Engine:
    def __init__(self, connection, context) -> None:
        self._store = _Store(connection)
        self.lcm_teams_enabled = True
        self.get_lcm_access_context = lambda: context


def _context(principal: str = "principal-a") -> AccessContextV1:
    return AccessContextV1.from_host(
        authenticated_transport="host-session",
        context_id="ctx-1",
        request_id="req-1",
        source_kind="human",
        deployment_id="dep-1",
        tenant_id="tenant-1",
        principal_id=principal,
        profile_id="profile-a",
        profile_incarnation="incarnation-1",
        session_id="session-own",
        session_owner_principal_id=principal,
        conversation_id="conversation-a",
        conversation_lane="lane-a",
        read_policy_ref="policy-a",
        lease_id="lease-a",
        issued_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    db_bootstrap.configure_connection(conn)
    conn.executescript(_OWNER_TABLES)
    catalog.ensure_teams_catalog(conn)
    catalog.provision_principal(
        conn, principal_id="principal-a", tenant_id="tenant-1", now=0.0
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _stamp(conn: sqlite3.Connection, table: str, session: str, owner: str) -> None:
    conn.execute(
        f"INSERT INTO {table}(session_id, access_scope) VALUES(?, ?)",
        (session, owner),
    )
    conn.commit()


def _minted(store: sqlite3.Connection) -> AccessContextV1:
    """A context at the catalog's CURRENT revisions.

    Provisioning the principal bumps `membership_revision`, so a context minted
    at zero is stale and would fail closed before reaching the owner lookup.
    """

    current = catalog.read_revisions(store, "tenant-1")
    return dataclasses.replace(
        _context(),
        policy_revision=current.policy_revision,
        membership_revision=current.membership_revision,
        revocation_epoch=current.revocation_epoch,
    )


# --- the resolver itself ---------------------------------------------------


def test_a_single_stamped_owner_resolves(store: sqlite3.Connection) -> None:
    """POSITIVE CONTROL: raising on everything also passes the two below."""
    _stamp(store, "messages", "session-x", "principal-a")

    resolve = _session_owner_for_engine(_Engine(store, _context()))

    assert resolve("session-x") == "principal-a"


def test_an_unclaimed_session_still_resolves_to_none(store: sqlite3.Connection) -> None:
    """Unclaimed is a real answer and stays permissive -- the write stamps it."""
    resolve = _session_owner_for_engine(_Engine(store, _context()))

    assert resolve("session-nobody") is None


def test_conflicting_owner_stamps_are_not_an_answer(store: sqlite3.Connection) -> None:
    _stamp(store, "messages", "session-x", "principal-a")
    _stamp(store, "summary_nodes", "session-x", "principal-b")

    resolve = _session_owner_for_engine(_Engine(store, _context()))

    with pytest.raises(OwnerLookupError):
        resolve("session-x")


def test_conflicting_stamps_within_one_table_are_not_an_answer(
    store: sqlite3.Connection,
) -> None:
    _stamp(store, "messages", "session-x", "principal-a")
    _stamp(store, "messages", "session-x", "principal-b")

    resolve = _session_owner_for_engine(_Engine(store, _context()))

    with pytest.raises(OwnerLookupError):
        resolve("session-x")


def test_an_unreadable_store_is_not_an_unclaimed_session(
    store: sqlite3.Connection,
) -> None:
    """Neither owner table can be read, so ownership is UNKNOWN, not absent."""
    store.execute("DROP TABLE messages")
    store.execute("DROP TABLE summary_nodes")
    store.commit()

    resolve = _session_owner_for_engine(_Engine(store, _context()))

    with pytest.raises(OwnerLookupError):
        resolve("session-x")


def test_one_readable_table_is_enough(store: sqlite3.Connection) -> None:
    """Most stores have no summary_nodes; that absence is not a fault."""
    store.execute("DROP TABLE summary_nodes")
    _stamp(store, "messages", "session-x", "principal-a")

    resolve = _session_owner_for_engine(_Engine(store, _context()))

    assert resolve("session-x") == "principal-a"


# --- through the policy ----------------------------------------------------


def test_a_conflicted_target_session_is_denied_end_to_end(
    store: sqlite3.Connection,
) -> None:
    """The reason this matters: the arbitrary winner passed the gate."""
    _stamp(store, "messages", "session-shared", "principal-a")
    _stamp(store, "messages", "session-shared", "principal-b")
    policy = policy_for_engine(_Engine(store, _minted(store)))
    assert isinstance(policy, TeamsPolicy), "expected the enforcing policy"

    decision = policy.authorize_operation(
        None, "read", {"session_id": "session-shared"}
    )

    assert not decision.allowed
    assert decision.denial_reason is not None
    assert decision.denial_reason.value == "context_invalid"


def test_an_owned_target_session_is_still_allowed_end_to_end(
    store: sqlite3.Connection,
) -> None:
    """POSITIVE CONTROL for the end-to-end path."""
    _stamp(store, "messages", "session-mine", "principal-a")
    policy = policy_for_engine(_Engine(store, _minted(store)))
    assert not isinstance(policy, FailClosedPolicy)

    assert policy.authorize_operation(
        None, "read", {"session_id": "session-mine"}
    ).allowed
