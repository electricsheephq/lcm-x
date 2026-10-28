"""Pure ownership decisions for slash-style /lcm commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# Unknown paths default to WRITE. Plain doctor uses SQLite's FTS integrity-check
# INSERT. Arguments remain part of the path except for preset show.
READ_LCM_SUBCOMMANDS = frozenset({
    "status", "help", "doctor clean", "doctor clean lifecycle", "doctor source",
    "doctor retention", "rotate", "rollups", "preset show", "preset suggest",
})
WRITE_LCM_SUBCOMMANDS = frozenset({
    "doctor", "doctor clean apply", "doctor clean lifecycle apply",
    "doctor repair", "doctor repair apply", "doctor repair level3",
    "doctor repair level3 apply", "doctor repair schema-stamp",
    "doctor repair schema-stamp apply", "doctor source apply", "backup",
    "rotate apply", "rollups rebuild", "assertions rebuild", "preset apply",
    "embed warmup", "embed backfill",
})


@dataclass(frozen=True)
class CommandInvocation:
    """Parsed command and caller facts; None marks an observation not yet made."""

    command: tuple[str, ...]
    session_engine_resolved: bool = True
    session_context_present: bool = False
    session_profile: str = ""
    engine_profile: str = ""
    context_session_id: str = ""
    context_conversation_id: str = ""
    bound_session_id: str | None = None
    bound_conversation_id: str = ""
    resident_engine_present: bool | None = None


class Decision(Enum):
    ALLOW = "allow"
    ALLOW_WITH_STORE = "allow_with_store"
    READ_BINDINGS = "read_bindings"
    CHECK_RESIDENT = "check_resident"
    PROFILE_MISMATCH = (
        "The command could not resolve this session's LCM engine in this process; "
        "run it from a process launched for this profile."
    )
    OWNERSHIP_UNPROVEN = (
        "The command could not prove ownership of this session's LCM engine; "
        "retry after the session has started."
    )


def authorize(invocation: CommandInvocation) -> Decision:
    """Decide using values only; request missing observations in their old order.

    The dispatcher supplies bindings only after profile validation, and reads
    the resident registry only for a same-profile, unbound fallback write.
    Re-evaluating with those facts keeps I/O and exceptions out of this policy.
    """
    profiles_known = bool(invocation.session_profile and invocation.engine_profile)
    if profiles_known and invocation.session_profile != invocation.engine_profile:
        return Decision.PROFILE_MISMATCH
    if invocation.session_context_present:
        path = " ".join(part.lower() for part in invocation.command) or "status"
        if path.startswith("preset show "):
            path = "preset show"
        if invocation.bound_session_id is None:
            return Decision.READ_BINDINGS
        bound_to_invoker = bool(
            (invocation.context_session_id and invocation.bound_session_id == invocation.context_session_id)
            or (invocation.context_conversation_id and invocation.bound_conversation_id == invocation.context_conversation_id)
        )
        # Resolved engines accept either binding when profiles are unknown.
        # Unresolved engines are session-first, matching the resolver.
        bound_to_invoker_strict = bool(
            invocation.bound_session_id == invocation.context_session_id if invocation.context_session_id
            else invocation.context_conversation_id and invocation.bound_conversation_id == invocation.context_conversation_id
        )
        if path not in READ_LCM_SUBCOMMANDS:
            if invocation.session_engine_resolved:
                if not (profiles_known or bound_to_invoker):
                    return Decision.OWNERSHIP_UNPROVEN
            elif not profiles_known:
                return Decision.OWNERSHIP_UNPROVEN
            elif not bound_to_invoker_strict:
                if invocation.bound_session_id or invocation.bound_conversation_id:
                    return Decision.OWNERSHIP_UNPROVEN
                if invocation.resident_engine_present is None:
                    return Decision.CHECK_RESIDENT
                if invocation.resident_engine_present:
                    return Decision.OWNERSHIP_UNPROVEN
    if not invocation.session_engine_resolved and not profiles_known:
        return Decision.ALLOW_WITH_STORE
    return Decision.ALLOW
