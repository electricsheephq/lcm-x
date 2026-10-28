"""Check routing metadata against the existing authorization policy."""

import pytest

from hermes_lcm import command_auth
from hermes_lcm.command_auth import CommandInvocation, Decision, authorize
from hermes_lcm.command_router import ROUTES


# Literal pre-refactor policy, including the two reads parsed by preset itself.
READ_PATHS = {
    "status", "help", "doctor clean", "doctor clean lifecycle", "doctor source",
    "doctor retention", "rotate", "rollups", "preset show", "preset suggest",
}
REGISTERED_PATHS = tuple(" ".join(entry.path) for entry in ROUTES)
EXTRA_PATHS = (
    "", "doctor clean foo", "doctor repair level3 foo", "Unknown",
    "preset show", "preset show Example", "preset suggest", "preset apply Example",
    "assertions rebuild", "rollups rebuild", "embed backfill --apply",
)


@pytest.mark.parametrize("path", REGISTERED_PATHS + EXTRA_PATHS)
@pytest.mark.parametrize("mixed_case", [False, True])
def test_registry_access_matches_existing_authorization(path, mixed_case):
    # Unknown profiles and no matching binding distinguish reads from writes.
    normalized = path or "status"
    if normalized.startswith("preset show "):
        normalized = "preset show"
    expected = Decision.ALLOW if normalized in READ_PATHS else Decision.OWNERSHIP_UNPROVEN
    invocation = CommandInvocation(
        command=tuple((path.upper() if mixed_case else path).split()),
        session_context_present=True,
        context_session_id="invoker",
        bound_session_id="other",
    )
    assert authorize(invocation) is expected


@pytest.mark.parametrize("path", REGISTERED_PATHS)
def test_registered_paths_with_extra_arguments_default_to_write(path):
    assert authorize(CommandInvocation(
        command=tuple((path + " Extra").split()),
        session_context_present=True,
        context_session_id="invoker",
        bound_session_id="other",
    )) is Decision.OWNERSHIP_UNPROVEN


def test_registry_write_metadata_gaps_are_characterized():
    # Check, do not derive or repair the policy sets in this slice.
    assert set(REGISTERED_PATHS) - command_auth.WRITE_LCM_SUBCOMMANDS == {
        "status", "help", "doctor clean", "doctor clean lifecycle", "doctor source",
        "doctor retention", "rotate", "rollups", "assertions", "preset",
    }
    assert {path for path in REGISTERED_PATHS if path not in READ_PATHS} - (
        command_auth.WRITE_LCM_SUBCOMMANDS
    ) == {"assertions", "preset"}


def test_write_metadata_is_not_consulted(monkeypatch):
    monkeypatch.setattr(command_auth, "WRITE_LCM_SUBCOMMANDS", frozenset({"status"}))
    for path in REGISTERED_PATHS:
        expected = Decision.ALLOW if path in READ_PATHS else Decision.OWNERSHIP_UNPROVEN
        assert authorize(CommandInvocation(
            command=tuple(path.split()),
            session_context_present=True,
            bound_session_id="",
        )) is expected
