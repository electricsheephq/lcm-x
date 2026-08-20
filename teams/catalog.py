"""Schema and accessors for the Teams catalog.

Created only on an explicit enable. Ordinary startup must leave a store with no
Teams tables at all -- the same discipline the access_scope columns follow, and
the reason a single-user install is unaffected by any of this.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..db_bootstrap import mark_migration_step_complete


TEAMS_CATALOG_MIGRATION = "teams_catalog_v1"

# Every table shares the lcm_teams prefix so db_bootstrap's feature-family
# allowlist can recognise the whole family with one entry. Without that,
# repair tooling refuses to repair a Teams-enabled store.
TEAMS_TABLES = (
    "lcm_teams_tenants",
    "lcm_teams_principals",
    "lcm_teams_collections",
    "lcm_teams_memberships",
    "lcm_teams_revisions",
    "lcm_teams_audit",
    "lcm_teams_requests",
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS lcm_teams_tenants (
    tenant_id  TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS lcm_teams_principals (
    principal_id TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    status       TEXT NOT NULL CHECK(status IN ('active', 'suspended')),
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lcm_teams_principals_tenant
    ON lcm_teams_principals(tenant_id, status);

CREATE TABLE IF NOT EXISTS lcm_teams_collections (
    collection_id TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK(kind IN ('own', 'shared')),
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lcm_teams_collections_tenant
    ON lcm_teams_collections(tenant_id, kind);

CREATE TABLE IF NOT EXISTS lcm_teams_memberships (
    principal_id  TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    grants        TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (principal_id, collection_id)
);

CREATE TABLE IF NOT EXISTS lcm_teams_revisions (
    tenant_id           TEXT PRIMARY KEY,
    policy_revision     INTEGER NOT NULL DEFAULT 0,
    membership_revision INTEGER NOT NULL DEFAULT 0,
    revocation_epoch    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lcm_teams_audit (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at   REAL NOT NULL,
    tenant_id     TEXT,
    principal_id  TEXT,
    operation     TEXT NOT NULL,
    allowed       INTEGER NOT NULL CHECK(allowed IN (0, 1)),
    denial_reason TEXT,
    detail        TEXT
);

CREATE INDEX IF NOT EXISTS idx_lcm_teams_audit_time
    ON lcm_teams_audit(occurred_at);

-- Connector request ledger. #497 requires that a duplicate management request
-- produce ONE effect, and that the same id carrying a DIFFERENT payload be
-- rejected rather than silently applied on top of the first. Both need the
-- original digest kept, so the id alone is not enough.
--
-- `result_json` stores what the first execution returned, so a replay can be
-- answered from the ledger instead of re-running the effect. That is the whole
-- point: idempotency by replaying the ANSWER, not by re-doing the work and
-- hoping it is harmless.
--
-- Keyed by (tenant_id, request_id), not by the id alone: an idempotency key is
-- the CONTROL PLANE's, and two tenants choosing the same one are two requests.
-- With a global key the second tenant's request collided with the first's row,
-- which is either a spurious conflict or -- before the identity check in
-- `_prior` -- the first tenant's cached answer.
CREATE TABLE IF NOT EXISTS lcm_teams_requests (
    request_id     TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    capability     TEXT NOT NULL,
    tenant_id      TEXT NOT NULL DEFAULT '',
    principal_id   TEXT,
    recorded_at    REAL NOT NULL,
    result_json    TEXT,
    PRIMARY KEY (tenant_id, request_id)
);
"""


# What each catalog table must actually carry. A table can exist with the wrong
# shape -- an older build, a partial restore, a hand-edited store -- and
# `CREATE TABLE IF NOT EXISTS` cannot repair it, so verifying the NAME alone
# hands an operator a clean structural result for a store on which every policy
# lookup fails.
_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "lcm_teams_tenants": ("tenant_id", "created_at"),
    "lcm_teams_principals": (
        "principal_id",
        "tenant_id",
        "status",
        "created_at",
        "updated_at",
    ),
    "lcm_teams_collections": ("collection_id", "tenant_id", "kind", "created_at"),
    "lcm_teams_memberships": (
        "principal_id",
        "collection_id",
        "grants",
        "created_at",
    ),
    "lcm_teams_revisions": (
        "tenant_id",
        "policy_revision",
        "membership_revision",
        "revocation_epoch",
    ),
    "lcm_teams_audit": (
        "event_id",
        "occurred_at",
        "tenant_id",
        "principal_id",
        "operation",
        "allowed",
        "denial_reason",
        "detail",
    ),
    "lcm_teams_requests": (
        "request_id",
        "payload_digest",
        "capability",
        "tenant_id",
        "principal_id",
        "recorded_at",
        "result_json",
    ),
}


class InconsistentCatalogError(RuntimeError):
    """The catalog holds a tenant's rows but not the state that governs them.

    Distinct from "unknown tenant", which is an ordinary absence with an
    ordinary answer. This one is a store that cannot be reasoned about, and
    ``resolution._catalog_revisions`` turns it into a fail-closed policy.
    """


@dataclass(frozen=True)
class CatalogRevisions:
    """The revisions a context is validated against.

    Held by the catalog rather than supplied by the host: under the narrow-shim
    carrier the host authenticates a principal and nothing more, so a context
    arriving with its own revision numbers proves only that someone wrote them
    into it.
    """

    policy_revision: int = 0
    membership_revision: int = 0
    revocation_epoch: int = 0


def teams_catalog_exists(conn: sqlite3.Connection) -> bool:
    """True when the catalog has been materialised on this store.

    Reads sqlite_master and creates nothing. Every caller of this runs on paths
    that must not mutate a store they are only inspecting.
    """

    placeholders = ",".join("?" for _ in TEAMS_TABLES)
    found = conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ({placeholders})",
        TEAMS_TABLES,
    ).fetchone()[0]
    return int(found) == len(TEAMS_TABLES)


def ensure_teams_catalog(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Create the catalog if absent and return the tables it now owns.

    Idempotent: every statement is CREATE ... IF NOT EXISTS, so a second enable
    is a no-op rather than an error. The core numeric schema_version is
    deliberately NOT bumped -- this is an opt-in feature family, and bumping it
    would make every Teams store look newer than a stock build to a downgrade
    check that is entitled to refuse.
    """

    conn.executescript(_SCHEMA)
    mark_migration_step_complete(conn, TEAMS_CATALOG_MIGRATION)
    conn.commit()
    return TEAMS_TABLES


def verify_teams_catalog(conn: sqlite3.Connection) -> list[str]:
    """Return structural defects, without mutating anything.

    Checks COLUMNS as well as table names. A table that exists with a missing
    or renamed column is exactly the store `ensure_teams_catalog` cannot repair
    -- every statement there is CREATE ... IF NOT EXISTS -- so reporting it
    clean is reporting the one defect an operator most needs to see.
    """

    errors: list[str] = []
    for table in TEAMS_TABLES:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if row is None:
            errors.append(f"missing table:{table}")
            continue
        present = {
            str(column[1]) for column in conn.execute(f'PRAGMA table_info("{table}")')
        }
        for column in _REQUIRED_COLUMNS.get(table, ()):
            if column not in present:
                errors.append(f"missing column:{table}.{column}")
    return sorted(errors)


def read_revisions(conn: sqlite3.Connection, tenant_id: str) -> CatalogRevisions:
    """Return the tenant's current revisions.

    An UNKNOWN tenant reads as all-zero rather than raising. A context that
    genuinely belongs to no tenant will fail the ownership stage on its
    principal, which is a more precise answer than a lookup error here.

    A KNOWN tenant with no revision row is a different thing wearing the same
    absence, and it must not read as zero. Zero is a real revision value, so a
    tenant whose row was lost to a partial restore or corruption handed every
    previously revoked context a match again -- undoing every policy,
    membership and revocation bump ever made, at once. The tenant is known
    exactly when the catalog still holds principals for it.
    """

    row = conn.execute(
        "SELECT policy_revision, membership_revision, revocation_epoch "
        "FROM lcm_teams_revisions WHERE tenant_id = ?",
        (tenant_id,),
    ).fetchone()
    if row is None:
        known = conn.execute(
            "SELECT 1 FROM lcm_teams_principals WHERE tenant_id = ? LIMIT 1",
            (str(tenant_id),),
        ).fetchone()
        if known is not None:
            raise InconsistentCatalogError(
                f"tenant {tenant_id!r} has catalog principals but no revision row; "
                "reading it as revision zero would revalidate every revoked context"
            )
        return CatalogRevisions()
    return CatalogRevisions(
        policy_revision=int(row[0] or 0),
        membership_revision=int(row[1] or 0),
        revocation_epoch=int(row[2] or 0),
    )


def record_audit_event(
    conn: sqlite3.Connection,
    *,
    occurred_at: float,
    tenant_id: str,
    principal_id: str,
    operation: str,
    allowed: bool,
    denial_reason: str | None = None,
    detail: str | None = None,
) -> None:
    """Append one authorization outcome to the audit trail.

    The denial reason stored here is the PUBLIC projection, never the internal
    one. #497 exposes an ``audit.*`` family, so these rows can leave the store
    and reach a tenant admin; an internal reason distinguishes "forbidden" from
    "does not exist", which is exactly the distinction the public projection
    exists to collapse. An operator debugging a denial has the operation, the
    principal and the timestamp, which is enough to correlate.

    Best-effort by construction: auditing must never be the reason an
    authorized operation fails. A store whose audit table is missing or locked
    still serves its principals.

    Transaction ownership stays with the CALLER. ``sqlite3.Connection.commit()``
    is connection-wide -- it does not commit "only" the audit insert -- so an
    unconditional commit here promoted every pending write on that connection
    to durable state, including a failed handler's partial ones. The caller was
    told the request failed while the store kept half the effect. When the
    connection is already inside a transaction the row is left for the caller
    to commit or roll back; only an otherwise-idle connection is committed
    here, which is the case for every audit-only writer.
    """

    try:
        in_caller_transaction = bool(getattr(conn, "in_transaction", False))
        conn.execute(
            "INSERT INTO lcm_teams_audit("
            "occurred_at, tenant_id, principal_id, operation, allowed, "
            "denial_reason, detail) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                float(occurred_at),
                str(tenant_id or ""),
                str(principal_id or ""),
                str(operation or ""),
                1 if allowed else 0,
                str(denial_reason) if denial_reason is not None else None,
                str(detail) if detail is not None else None,
            ),
        )
        if not in_caller_transaction:
            conn.commit()
    except sqlite3.Error:
        return


def read_audit_events(
    conn: sqlite3.Connection, *, tenant_id: str | None = None, limit: int = 100
) -> list[dict[str, object]]:
    """Read the audit trail, newest first."""

    if tenant_id is None:
        rows = conn.execute(
            "SELECT occurred_at, tenant_id, principal_id, operation, allowed, "
            "denial_reason, detail FROM lcm_teams_audit "
            "ORDER BY event_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT occurred_at, tenant_id, principal_id, operation, allowed, "
            "denial_reason, detail FROM lcm_teams_audit WHERE tenant_id = ? "
            "ORDER BY event_id DESC LIMIT ?",
            (str(tenant_id), int(limit)),
        ).fetchall()
    return [
        {
            "occurred_at": row[0],
            "tenant_id": row[1],
            "principal_id": row[2],
            "operation": row[3],
            "allowed": bool(row[4]),
            "denial_reason": row[5],
            "detail": row[6],
        }
        for row in rows
    ]


def set_revisions(
    conn: sqlite3.Connection, tenant_id: str, revisions: CatalogRevisions
) -> None:
    """Write a tenant's revisions outright, never backwards.

    Provisioning needs the outright write: a tenant is created at whatever
    revisions its control plane already issued contexts against, which is not
    necessarily zero. Bumping from zero would only reach those numbers by
    accident.

    But a counter that can DECREASE un-revokes. Every context invalidated by a
    prior policy, membership or revocation bump matches the rolled-back value
    again, so stale control-plane state replayed onto an existing tenant hands
    back access that was deliberately withdrawn. Arbitrary counters are
    therefore accepted only on initial creation, and a decrease for an existing
    tenant is refused rather than silently clamped -- a control plane sending
    one is out of step with the store and needs to know.

    The refusal is a predicate on the WRITE, not only on the read below. A
    control plane runs more than one reconciler: worker A could read 5 here,
    worker B commit 10, and A then overwrite it with 6 unconditionally. The
    read-then-write pair is not serialized, so the check has to travel with the
    statement -- ``ON CONFLICT ... DO UPDATE ... WHERE`` re-evaluates the
    stored row at write time, and a worker that lost the race changes nothing
    and is told so. The read below stays because it names WHICH counter went
    backwards, which the write-time predicate cannot.
    """

    current = conn.execute(
        "SELECT policy_revision, membership_revision, revocation_epoch "
        "FROM lcm_teams_revisions WHERE tenant_id = ?",
        (str(tenant_id),),
    ).fetchone()
    if current is not None:
        requested = (
            int(revisions.policy_revision),
            int(revisions.membership_revision),
            int(revisions.revocation_epoch),
        )
        names = ("policy_revision", "membership_revision", "revocation_epoch")
        regressions = [
            f"{name}: {int(stored or 0)} -> {new}"
            for name, stored, new in zip(names, current, requested)
            if new < int(stored or 0)
        ]
        if regressions:
            raise ValueError(
                f"revisions may not move backwards for tenant {tenant_id!r}: "
                + ", ".join(regressions)
            )

    cursor = conn.execute(
        "INSERT INTO lcm_teams_revisions("
        "tenant_id, policy_revision, membership_revision, revocation_epoch"
        ") VALUES(?, ?, ?, ?) "
        "ON CONFLICT(tenant_id) DO UPDATE SET "
        "policy_revision = excluded.policy_revision, "
        "membership_revision = excluded.membership_revision, "
        "revocation_epoch = excluded.revocation_epoch "
        "WHERE excluded.policy_revision >= lcm_teams_revisions.policy_revision "
        "AND excluded.membership_revision >= lcm_teams_revisions.membership_revision "
        "AND excluded.revocation_epoch >= lcm_teams_revisions.revocation_epoch",
        (
            tenant_id,
            int(revisions.policy_revision),
            int(revisions.membership_revision),
            int(revisions.revocation_epoch),
        ),
    )
    if cursor.rowcount < 1:
        # The conflict update's predicate was false, which the read above said
        # it would not be: another worker advanced the counters between the
        # two statements. Nothing was written -- refuse rather than retry, so
        # the control plane resolves the disagreement instead of racing again.
        conn.rollback()
        raise ValueError(
            f"revisions for tenant {tenant_id!r} were advanced concurrently; "
            "this update would have moved them backwards"
        )
    conn.commit()


def bump_revision(conn: sqlite3.Connection, tenant_id: str, field: str) -> int:
    """Advance one revision counter and return its new value.

    Revoking access is a revision bump, so this is the operation that makes a
    previously-issued context stale. The field name is checked against the
    known set rather than interpolated blindly.
    """

    _bump_revision_uncommitted(conn, tenant_id, field)
    conn.commit()
    return getattr(read_revisions(conn, tenant_id), field)


def _bump_revision_uncommitted(
    conn: sqlite3.Connection, tenant_id: str, field: str
) -> None:
    """The statements of :func:`bump_revision`, without the commit.

    Exists so a mutation and the bump that invalidates the contexts depending
    on it can share ONE transaction. Committing the mutation first left a
    window in which the membership was gone and the old counters were still
    current, so every already-issued context kept validating -- which is
    precisely the next-operation revocation guarantee #498 requires.
    """

    if field not in {"policy_revision", "membership_revision", "revocation_epoch"}:
        raise ValueError(f"unknown revision field: {field}")
    conn.execute(
        "INSERT INTO lcm_teams_revisions(tenant_id) VALUES(?) "
        "ON CONFLICT(tenant_id) DO NOTHING",
        (str(tenant_id),),
    )
    conn.execute(
        f"UPDATE lcm_teams_revisions SET {field} = {field} + 1 WHERE tenant_id = ?",
        (str(tenant_id),),
    )


# ---------------------------------------------------------------------------
# Principals, collections and memberships
#
# These were the accessors the catalog promised and did not have, which is why
# TeamsPolicy decides from the CONTEXT rather than from the catalog: with no way
# to ask "which collections may this principal read", a shared collection could
# not be modelled at all, and the policy could only answer the private case.
#
# Every write bumps the matching revision. That is not bookkeeping -- the narrow
# host carrier deliberately does NOT send revisions, so the catalog is the only
# thing that can say a membership changed, and a stale context is detected by
# comparing against these counters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    principal_id: str
    tenant_id: str
    status: str


@dataclass(frozen=True)
class Membership:
    principal_id: str
    collection_id: str
    grants: tuple[str, ...]


# ``status`` is CHECK-constrained to these two. Note what is MISSING: there is
# no 'archived'. The ratified contract calls removal "disable-then-archive,
# never destructive delete", so archive needs either a third status or an
# archived_at column, and adding it is a schema migration rather than an
# accessor. Suspend is what exists today, and it is what `principals.suspend`
# maps to; `principals.archive` cannot be honoured until that migration lands.
PRINCIPAL_STATUSES = ("active", "suspended")


# Grants are stored as ONE comma-joined string and read back by splitting on
# the same character, so the delimiter is not decoration -- it is the only
# thing separating one capability from the next. A grant name carrying it
# turns one requested value into several: a control-plane payload asking for
# `"custom,write"` produced a `write` grant nobody requested, and
# `authorized_collections(..., grant="write")` then returned the collection.
#
# Named once, used by the join and the split alike, so the two cannot drift.
GRANT_DELIMITER = ","


def provision_principal(
    conn: sqlite3.Connection,
    *,
    principal_id: str,
    tenant_id: str,
    now: float,
    status: str = "active",
) -> Principal:
    """Create or re-activate a principal. Idempotent by primary key.

    ``principal_id`` is the primary key, so the upsert cannot move a principal
    between tenants -- and it did not try to, which is the problem: it left the
    stored tenant alone, bumped the SUPPLIED tenant's revision, and returned a
    Principal naming a tenant the row does not have. One tenant re-provisioning
    another's principal id therefore reactivated it while reporting success in
    its own name. A cross-tenant collision is refused instead -- and refused in
    the WRITE as well as in the read, because two workers provisioning the same
    previously-absent id for different tenants both saw "absent" and the loser
    then updated the winner's row.

    The bump is CONDITIONAL, for the reason ``grant_membership`` records: an
    unchanged principal is a logical no-op, and bumping ``membership_revision``
    anyway made every context already issued for the tenant fail revision
    validation. A desired-state reconciler re-asserting its inventory therefore
    locked out principals nothing had changed.
    """

    if status not in PRINCIPAL_STATUSES:
        raise ValueError(f"status must be one of {PRINCIPAL_STATUSES}")
    existing = read_principal(conn, principal_id)
    if existing is not None and existing.tenant_id != str(tenant_id):
        raise LookupError(
            f"principal {principal_id!r} belongs to tenant {existing.tenant_id!r}, "
            f"not {tenant_id!r}"
        )
    if existing is not None and existing.status == status:
        return Principal(str(principal_id), str(tenant_id), status)
    cursor = conn.execute(
        "INSERT INTO lcm_teams_principals(principal_id, tenant_id, status,"
        " created_at, updated_at) VALUES(?,?,?,?,?)"
        " ON CONFLICT(principal_id) DO UPDATE SET status=excluded.status,"
        " updated_at=excluded.updated_at"
        " WHERE lcm_teams_principals.tenant_id = excluded.tenant_id",
        (str(principal_id), str(tenant_id), status, float(now), float(now)),
    )
    if cursor.rowcount < 1:
        # The row exists and is not this tenant's -- it was created between the
        # read above and this statement. Same refusal as the read's, reached by
        # the only check that can still be true at write time.
        conn.rollback()
        owner = read_principal(conn, principal_id)
        owner_tenant = owner.tenant_id if owner is not None else "<unreadable>"
        raise LookupError(
            f"principal {principal_id!r} belongs to tenant {owner_tenant!r}, "
            f"not {tenant_id!r}"
        )
    _bump_revision_uncommitted(conn, str(tenant_id), "membership_revision")
    conn.commit()
    return Principal(str(principal_id), str(tenant_id), status)


def suspend_principal(
    conn: sqlite3.Connection, *, principal_id: str, tenant_id: str, now: float
) -> Principal:
    """Suspend without unstamping anything.

    The principal's rows keep their access_scope, exactly as `disable_teams`
    keeps stamps: attribution is what a later re-provision and every audit
    answer depend on. Suspension also bumps the revocation epoch, so contexts
    already issued to this principal stop validating rather than running to
    their natural expiry.

    The tenant is a PREDICATE, not bookkeeping. It arrived as an argument, drove
    the epoch bump, and was then dropped before the UPDATE, so one tenant could
    suspend another tenant's principal while invalidating its own contexts. A
    principal that does not belong to the named tenant is not found.
    """

    cursor = conn.execute(
        "UPDATE lcm_teams_principals SET status='suspended', updated_at=?"
        " WHERE principal_id=? AND tenant_id=?",
        (float(now), str(principal_id), str(tenant_id)),
    )
    if cursor.rowcount < 1:
        conn.rollback()
        raise LookupError(
            f"no principal {principal_id!r} in tenant {tenant_id!r}"
        )
    # One transaction with the suspension: a committed suspension whose epoch
    # bump then failed leaves the principal's already-issued contexts valid.
    _bump_revision_uncommitted(conn, str(tenant_id), "revocation_epoch")
    conn.commit()
    return Principal(str(principal_id), str(tenant_id), "suspended")


def read_principal(conn: sqlite3.Connection, principal_id: str) -> Principal | None:
    row = conn.execute(
        "SELECT principal_id, tenant_id, status FROM lcm_teams_principals"
        " WHERE principal_id = ?",
        (str(principal_id),),
    ).fetchone()
    return Principal(str(row[0]), str(row[1]), str(row[2])) if row else None


def _resolve_collection_conflict(
    conn: sqlite3.Connection, collection_id: str, tenant_id: str, kind: str
) -> str:
    """Decide what an existing row with this id means for this request.

    Only an identical row in the SAME tenant is a replay. A row belonging to
    another tenant is a collision, and a different kind in this tenant is a
    conflicting redefinition -- neither is a success.
    """

    row = conn.execute(
        "SELECT tenant_id, kind FROM lcm_teams_collections WHERE collection_id = ?",
        (str(collection_id),),
    ).fetchone()
    if row is None:
        raise LookupError(
            f"collection {collection_id!r} conflicted but could not be read back"
        )
    stored_tenant, stored_kind = str(row[0]), str(row[1])
    if stored_tenant != str(tenant_id):
        raise LookupError(
            f"collection {collection_id!r} belongs to tenant {stored_tenant!r}, "
            f"not {tenant_id!r}"
        )
    if stored_kind != str(kind):
        raise ValueError(
            f"collection {collection_id!r} already exists in tenant {tenant_id!r} "
            f"as kind {stored_kind!r}, not {str(kind)!r}"
        )
    return str(collection_id)


def create_collection(
    conn: sqlite3.Connection,
    *,
    collection_id: str,
    tenant_id: str,
    kind: str,
    now: float,
) -> str:
    """Create a collection within ONE tenant. Idempotent for an identical row.

    The tenant is a predicate here too. This was ``INSERT OR IGNORE``, so a
    collection id already owned by ANOTHER tenant silently skipped the insert
    and the accessor still reported the id as created for the caller. The
    management ledger then caches a false success, while the caller's next
    membership grant fails against a collection it does not own.

    A plain INSERT rather than a pre-read: the primary-key conflict is what
    detects the collision, so the check cannot be raced past.
    """

    if kind not in ("own", "shared"):
        raise ValueError("kind must be 'own' or 'shared'")
    try:
        conn.execute(
            "INSERT INTO lcm_teams_collections(collection_id, tenant_id,"
            " kind, created_at) VALUES(?,?,?,?)",
            (str(collection_id), str(tenant_id), kind, float(now)),
        )
    except sqlite3.IntegrityError:
        conn.rollback()
        return _resolve_collection_conflict(conn, collection_id, tenant_id, kind)
    conn.commit()
    return str(collection_id)


def grant_membership(
    conn: sqlite3.Connection,
    *,
    principal_id: str,
    collection_id: str,
    grants: "tuple[str, ...] | list[str]",
    tenant_id: str,
    now: float,
) -> Membership:
    """Grant a principal access to a collection, within ONE tenant. Idempotent.

    Both sides are verified against the requested tenant. Neither the schema
    nor this accessor used to check either: the membership row names only a
    principal and a collection, so a grant across two tenants inserted
    cleanly and ``authorized_collections()`` then returned the foreign
    collection. The supplied tenant only chose which revision to bump, so the
    call could invalidate one tenant's contexts while leaving a cross-tenant
    grant live in another.

    The bump is CONDITIONAL. An unchanged grant is a logical no-op, but the
    unconditional bump advanced ``membership_revision`` anyway, so every
    context already issued for the tenant failed revision validation. A
    periodic desired-state reconciler replaying its grants therefore locked out
    principals nothing had changed.
    """

    normalized = tuple(sorted({str(g).strip() for g in grants if str(g).strip()}))
    delimited = tuple(name for name in normalized if GRANT_DELIMITER in name)
    if delimited:
        raise ValueError(
            f"a grant name may not contain {GRANT_DELIMITER!r}, the encoding "
            f"delimiter -- it would be read back as several grants: {delimited!r}"
        )

    principal = read_principal(conn, principal_id)
    if principal is None or principal.tenant_id != str(tenant_id):
        raise LookupError(
            f"no principal {principal_id!r} in tenant {tenant_id!r}"
        )
    collection_tenant = conn.execute(
        "SELECT tenant_id FROM lcm_teams_collections WHERE collection_id = ?",
        (str(collection_id),),
    ).fetchone()
    if collection_tenant is None or str(collection_tenant[0]) != str(tenant_id):
        raise LookupError(
            f"no collection {collection_id!r} in tenant {tenant_id!r}"
        )

    existing = conn.execute(
        "SELECT grants FROM lcm_teams_memberships"
        " WHERE principal_id=? AND collection_id=?",
        (str(principal_id), str(collection_id)),
    ).fetchone()
    unchanged = existing is not None and str(existing[0]) == GRANT_DELIMITER.join(normalized)
    if unchanged:
        return Membership(str(principal_id), str(collection_id), normalized)

    conn.execute(
        "INSERT INTO lcm_teams_memberships(principal_id, collection_id, grants,"
        " created_at) VALUES(?,?,?,?)"
        " ON CONFLICT(principal_id, collection_id) DO UPDATE SET grants=excluded.grants",
        (str(principal_id), str(collection_id), GRANT_DELIMITER.join(normalized), float(now)),
    )
    _bump_revision_uncommitted(conn, str(tenant_id), "membership_revision")
    conn.commit()
    return Membership(str(principal_id), str(collection_id), normalized)


def revoke_membership(
    conn: sqlite3.Connection, *, principal_id: str, collection_id: str, tenant_id: str
) -> bool:
    """Remove a grant and bump the revocation epoch, ATOMICALLY.

    #498 requires revocation to block the NEXT operation, not eventually. The
    epoch bump is what does that: a context issued before it stops validating
    immediately, rather than remaining good until its lease expires.

    All three statements are therefore one transaction. Committing the deletion
    first left a window -- however short -- in which the membership was gone
    and both counters were still current, so every already-issued context kept
    validating; and a bump that then failed left that state permanently. The
    guarantee is that no observer sees the deletion without the invalidation.

    The tenant is a predicate here too: a membership whose collection belongs
    to another tenant is not this tenant's to revoke, and revoking it while
    bumping the caller's counters is the cross-tenant write the grant path now
    refuses to create.

    And the bumps are CONDITIONAL on the deletion, which is the other half of
    that predicate. It was added to the DELETE alone, so a foreign or
    already-gone membership removed nothing and still advanced the supplied
    tenant's revocation epoch and membership revision -- invalidating every
    context that tenant had issued for a deletion that never happened. The
    suspension path already rolls back and refuses on the same condition.
    """

    cursor = conn.execute(
        "DELETE FROM lcm_teams_memberships WHERE principal_id=? AND collection_id=?"
        " AND collection_id IN ("
        "  SELECT collection_id FROM lcm_teams_collections WHERE tenant_id=?"
        " )",
        (str(principal_id), str(collection_id), str(tenant_id)),
    )
    if cursor.rowcount < 1:
        conn.rollback()
        return False
    _bump_revision_uncommitted(conn, str(tenant_id), "revocation_epoch")
    _bump_revision_uncommitted(conn, str(tenant_id), "membership_revision")
    conn.commit()
    return True


def read_memberships(
    conn: sqlite3.Connection, principal_id: str
) -> tuple[Membership, ...]:
    rows = conn.execute(
        "SELECT principal_id, collection_id, grants FROM lcm_teams_memberships"
        " WHERE principal_id = ? ORDER BY collection_id",
        (str(principal_id),),
    ).fetchall()
    return tuple(
        Membership(
            str(r[0]),
            str(r[1]),
            tuple(g for g in str(r[2]).split(GRANT_DELIMITER) if g),
        )
        for r in rows
    )


def authorized_collections(
    conn: sqlite3.Connection, principal_id: str, *, grant: str = "read"
) -> tuple[str, ...]:
    """The collections this principal may act on -- the question the policy asks.

    A SUSPENDED principal gets nothing, regardless of surviving membership rows.
    Suspension is deliberately non-destructive, so the rows are still there; if
    this read went straight to memberships, suspending someone would remove
    their status and leave their access intact.
    """

    principal = read_principal(conn, principal_id)
    if principal is None or principal.status != "active":
        return ()
    return tuple(
        m.collection_id for m in read_memberships(conn, principal_id) if grant in m.grants
    )
