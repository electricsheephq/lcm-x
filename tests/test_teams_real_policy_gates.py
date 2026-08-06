"""Every gate, asserted against the REAL TeamsPolicy -- never a stub.

Four separate defects this session shared one root cause: the test injected a
policy that denies, then asserted the handler propagated the denial. That proves
the plumbing and says nothing about whether the real policy ever produces one.
Three apply-mode gates (#207) and the doctor gate (#218) each shipped inert with
a green test sitting next to them.

So this file takes NO policy argument. It builds a real AccessContextV1 for a
real principal, constructs the real TeamsPolicy, and asks what it decides.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import TeamsPolicy


def _context(principal: str = "acorn") -> AccessContextV1:
    now = datetime.now(timezone.utc)
    return AccessContextV1.from_host(
        authenticated_transport="test", context_id="ctx", request_id="req",
        source_kind="human", deployment_id="dep", tenant_id="tenant",
        principal_id=principal, profile_id=principal, profile_incarnation="inc",
        session_id=f"openclaw-lcm:agent:{principal}:s1",
        session_owner_principal_id=principal, conversation_id="conv",
        conversation_lane="lane", default_write_collection_id="col",
        read_policy_ref="policy", lease_id="lease",
        issued_at=now - timedelta(minutes=1), expires_at=now + timedelta(hours=1),
    )


@pytest.mark.parametrize("command", ["doctor", "doctor clean", "doctor retention"])
def test_a_principal_is_denied_the_doctor_command(command: str) -> None:
    """#218. `scan_session_cleanup_stats` takes no filter -- allowing this hands
    one principal every other principal's session ids, message counts and tokens."""
    decision = TeamsPolicy(_context()).authorize_operation(
        None, "admin",
        {"kind": "slash_command", "command": command, "required_scope": "admin"},
    )
    assert not decision.allowed, f"/lcm {command} is allowed under Teams"


def test_a_principal_may_still_reset_its_OWN_session() -> None:
    """The other half of #218, and the reason the fix keys on "admin" alone.

    `on_session_reset` (engine.py:3803) also asks for `owner_only`, but there it
    means owner OF THE TARGET and carries the principal's own session id. Denying
    every `owner_only` would break a principal resetting its own session -- the
    same conflation that once denied principal A its own session load.
    """
    context = _context()
    decision = TeamsPolicy(context).authorize_operation(
        None, "owner_only",
        {"kind": "session_reset", "session_id": context.session_id,
         "conversation_id": "conv", "required_scope": "owner_only"},
    )
    assert decision.allowed, "a principal was denied a reset of its own session"


@pytest.mark.parametrize(
    "kind", ["backup", "assertions_rebuild", "embedding_backfill", "chunk_backfill"]
)
def test_store_wide_kinds_stay_denied(kind: str) -> None:
    """Regression guard for #207 -- these were allowed for everyone."""
    decision = TeamsPolicy(_context()).authorize_operation(
        None, "owner_only", {"kind": kind, "entry_point": "x", "required_scope": "owner_only"}
    )
    assert not decision.allowed, f"{kind} is allowed under Teams"


def test_no_gate_site_asks_for_an_authority_the_policy_ignores() -> None:
    """Structural: every required_scope value in the source is one the policy acts on.

    This is the check that would have caught #218 on the day it was written. A new
    gate asking for an authority TeamsPolicy silently ignores fails here rather
    than shipping permissive with a green stub test beside it.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    # Every authority a gate site may request, and what the policy does with it:
    #   admin       -> DENIED. Store-wide administration; the connector holds it.
    #   owner_only  -> overloaded, decided by KIND and by target ownership, never
    #                  by the word itself (see the own-session test above).
    #   write       -> deliberately ALLOWED to fall through. A principal writing
    #                  its OWN data is the ordinary case; the session/target checks
    #                  are what stop it writing someone else's.
    # A value NOT listed here means a new gate is asking for something the policy
    # has never been taught to act on -- which is how #218 shipped. Add it here
    # only after deciding, in authorize_operation, what it should mean.
    known = {"admin", "owner_only", "write"}
    found: set[str] = set()
    for path in root.rglob("*.py"):
        if "tests" in path.parts or ".venv" in path.parts:
            continue
        for match in re.finditer(r'"required_scope":\s*"([a-z_]+)"', path.read_text(encoding="utf-8")):
            found.add(match.group(1))

    unhandled = found - known
    assert not unhandled, (
        f"gate sites request authorities TeamsPolicy does not act on: {sorted(unhandled)}. "
        "Add handling to authorize_operation, or the gate is inert."
    )
