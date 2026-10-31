"""The unchanged public-handler truth table also specifies pure authorization."""

import pytest

from hermes_lcm.command_auth import CommandInvocation, Decision, authorize
from .test_command_authorization import CALLERS, COMMANDS, PROFILE_REFUSAL


@pytest.mark.parametrize("raw,is_read", COMMANDS)
@pytest.mark.parametrize("case", CALLERS, ids=lambda case: case.name)
def test_authorize_ownership_table(raw, is_read, case):
    invocation = CommandInvocation(
        command=tuple((raw or "").split()),
        bound_session_id=case.bound_session,
        bound_conversation_id=case.bound_conversation,
        resident_engine_present=case.resident,
        **case.kwargs,
    )
    if case.refusal == PROFILE_REFUSAL:
        expected = Decision.PROFILE_MISMATCH
    elif not is_read and not case.write_allowed:
        expected = Decision.OWNERSHIP_UNPROVEN
    elif case.disclose:
        expected = Decision.ALLOW_WITH_STORE
    else:
        expected = Decision.ALLOW
    assert authorize(invocation) is expected
