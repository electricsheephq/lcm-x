"""The store-backed owner resolver fails CLOSED, on both of its faults.

`TeamsPolicy` already refuses when its `session_owner` callback RAISES (#68).
The resolver `resolution._session_owner_for_engine` binds to the store, and it
had two ways to answer "unclaimed" when it did not actually know:

* every read raised `sqlite3.OperationalError` -- a lock, a schema fault, an
  un-migrated store -- and the loop returned `None`, which the policy is
  DELIBERATELY permissive about because an unclaimed session cannot belong to
  anyone else. The chain now covers FIVE owner-stamped tables (`messages`,
  `summary_nodes` and the three rollup tables), and only a table genuinely
  absent from `sqlite_master` counts as a tolerable absence -- one that
  exists and cannot be read may hold the deciding stamp;
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
from hermes_lcm.access_policy import (
    FailClosedPolicy,
    TeamsPolicy,
    TrustedOwnerPolicy,
    policy_for_engine,
)
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
    """NO owner table can be read, so ownership is UNKNOWN, not absent.

    The resolver reads five owner-stamped tables: `messages`,
    `summary_nodes`, and the three rollup tables. The fixture creates only
    the first two, so dropping both leaves zero readable owner tables. A
    fixture that later adds a rollup table must drop it here too, or this
    test stops meaning what its name says.
    """
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
    # `isinstance(policy, TeamsPolicy)`, not merely "not FailClosedPolicy":
    # `policy_for_engine` can also return `TrustedOwnerPolicy`, which allows
    # unconditionally, so the weaker assertion would report enforcement that
    # never ran.
    assert isinstance(policy, TeamsPolicy), "expected the enforcing policy"

    assert policy.authorize_operation(
        None, "read", {"session_id": "session-mine"}
    ).allowed


def test_a_rollup_only_session_resolves_to_its_owner(store: sqlite3.Connection) -> None:
    """Rollup tables are owner-stamped session data keyed by ``scope``.

    A session whose only remaining rows are rollups (messages pruned or
    externalized) previously resolved to None — and None is deliberately
    permissive, so another principal's session-level operation was ALLOWED
    against it."""
    store.execute(
        "CREATE TABLE lcm_rollups ("
        " rollup_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " scope TEXT NOT NULL,"
        " access_scope TEXT)"
    )
    store.execute(
        "INSERT INTO lcm_rollups(scope, access_scope) VALUES(?, ?)",
        ("session-rollup-only", "principal-a"),
    )
    store.commit()

    resolve = _session_owner_for_engine(_Engine(store, _context()))

    assert resolve("session-rollup-only") == "principal-a"


def test_an_existing_but_unreadable_owner_table_is_not_unclaimed(
    store: sqlite3.Connection,
) -> None:
    """A table that EXISTS and cannot be read may hold the deciding stamp.

    Every `sqlite3.OperationalError` was treated as "this table is absent on
    this store", and the raise required that NO table had been read. So a
    partially migrated store -- `messages` present but without `access_scope`,
    alongside a rollup table that exists and is empty -- made `readable`
    nonzero, `owners` empty, and the answer `None`, which the policy
    DELIBERATELY allows. Absence and unreadability are now separate: only a
    table that is not in `sqlite_master` is a tolerable absence.
    """
    store.execute("ALTER TABLE messages DROP COLUMN access_scope")
    store.execute(
        "CREATE TABLE lcm_rollups (scope TEXT NOT NULL, access_scope TEXT)"
    )
    store.commit()

    resolve = _session_owner_for_engine(_Engine(store, _context()))

    with pytest.raises(OwnerLookupError):
        resolve("session-x")


def test_a_disabled_engine_never_reads_the_context_accessor() -> None:
    """Teams-off must not depend on the host callback succeeding.

    `resolve_mode` returns STANDARD_UNMANAGED for `teams_enabled=False` whatever
    the context is, so reading one decides nothing -- but the read happened
    unconditionally, and an accessor raising during teardown or partial wiring
    took the whole unmanaged request down with it.
    """

    class _Disabled:
        lcm_teams_enabled = False

        @staticmethod
        def get_lcm_access_context():
            raise RuntimeError("host carrier is torn down")

    policy = policy_for_engine(_Disabled())

    assert isinstance(policy, TrustedOwnerPolicy)


def test_a_foreign_tenants_session_owner_is_refused(store: sqlite3.Connection) -> None:
    """The EFFECTIVE session owner is the row scope, so it must be validated.

    `_catalog_principal_denial` checked only `principal_id`, while
    `TeamsPolicy.principal_of()` scopes rows by
    `session_owner_principal_id or principal_id`. An active actor in tenant-1
    carrying a session owner from another tenant therefore resolved to
    `TeamsPolicy` and authorized rows stamped for that other tenant's owner.
    """
    catalog.provision_principal(
        store, principal_id="foreign", tenant_id="tenant-2", now=0.0
    )
    context = dataclasses.replace(
        _minted(store), session_owner_principal_id="foreign"
    )

    policy = policy_for_engine(_Engine(store, context))

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason.value == "context_invalid"


def test_a_suspended_session_owner_is_refused(store: sqlite3.Connection) -> None:
    """Same rule, the revocation half: a suspended owner is not an owner."""
    catalog.provision_principal(
        store, principal_id="dormant", tenant_id="tenant-1", now=0.0
    )
    catalog.suspend_principal(
        store, principal_id="dormant", tenant_id="tenant-1", now=1.0
    )
    context = dataclasses.replace(
        _minted(store), session_owner_principal_id="dormant"
    )

    policy = policy_for_engine(_Engine(store, context))

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason.value == "context_revoked"
