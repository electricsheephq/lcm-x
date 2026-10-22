"""Public-dispatch characterization of command ownership, including #852."""

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
import types

import pytest

from hermes_lcm import command


PROFILE_REFUSAL = (
    "The command could not resolve this session's LCM engine in this process; "
    "run it from a process launched for this profile."
)
OWNERSHIP_REFUSAL = (
    "The command could not prove ownership of this session's LCM engine; "
    "retry after the session has started."
)

# Literal classification: unknown paths and extra arguments default to writes;
# preset show is the sole argument-bearing read exception.
READ_COMMANDS = (
    None, "", "status", "help", "doctor clean", "doctor clean lifecycle",
    "doctor source", "doctor retention", "rotate", "rollups", "preset show",
    "preset suggest", "  PRESET  SHOW  example  ",
)
WRITE_COMMANDS = (
    "doctor", "doctor clean apply", "doctor clean lifecycle apply",
    "doctor repair", "doctor repair apply", "doctor repair level3",
    "doctor repair level3 apply", "doctor repair schema-stamp",
    "doctor repair schema-stamp apply", "doctor source apply", "backup",
    "rotate apply", "rollups rebuild", "assertions rebuild", "preset apply",
    "embed warmup", "embed backfill", "embed backfill --apply",
    "assertions rebuild --dry-run", "preset apply example --dry-run",
    "future-command", "status extra", "help extra", "doctor clean extra",
    "rollups extra", "preset suggest extra", "  EMBED  WARMUP  ",
)
COMMANDS = tuple((raw, True) for raw in READ_COMMANDS) + tuple(
    (raw, False) for raw in WRITE_COMMANDS
)


@dataclass(frozen=True)
class Caller:
    name: str
    write_allowed: bool = False
    refusal: str = ""
    disclose: bool = False
    resident: bool = False
    reads_resident: bool = False
    bound_session: str = ""
    bound_conversation: str = ""
    kwargs: dict = field(default_factory=dict)


def caller(name, write_allowed=False, *, resolved=False, profile="same", context=True,
           session="invoker", conversation="lane", **facts):
    profiles = {
        "same": ("profile", "profile"), "different": ("caller", "engine"),
        "unknown-session": ("", "profile"), "unknown-engine": ("profile", ""),
        "unknown-both": ("", ""),
    }
    session_profile, engine_profile = profiles[profile]
    return Caller(
        name, write_allowed, disclose=not resolved and profile.startswith("unknown"),
        kwargs=dict(session_engine_resolved=resolved, session_context_present=context,
                    session_profile=session_profile, engine_profile=engine_profile,
                    context_session_id=session, context_conversation_id=conversation),
        **facts,
    )


CALLERS = (
    caller("cli", True, context=False),
    caller("cli-unknown", True, context=False, profile="unknown-both"),
    caller("cli-resolved", True, context=False, resolved=True),
    caller("cli-mismatch", context=False, profile="different", refusal=PROFILE_REFUSAL),
    caller("unresolved-mismatch", profile="different", refusal=PROFILE_REFUSAL),
    caller("resolved-mismatch", resolved=True, profile="different", refusal=PROFILE_REFUSAL),
    caller("cold-same-profile", True, reads_resident=True),
    caller("cold-with-resident", resident=True, reads_resident=True),
    caller("fallback-bound-session", True, bound_session="invoker", resident=True),
    caller("fallback-bound-elsewhere", bound_session="other"),
    caller("fallback-key-only-match", True, session="", bound_conversation="lane"),
    caller("fallback-key-only-miss", session="", bound_conversation="other"),
    caller("fallback-session-beats-key", bound_session="other", bound_conversation="lane"),
    caller("fallback-matching-session-key-differs", True,
           bound_session="invoker", bound_conversation="other"),
    caller("fallback-unknown-session", profile="unknown-session", bound_session="invoker"),
    caller("fallback-unknown-engine", profile="unknown-engine", bound_session="invoker"),
    caller("fallback-unknown-both", profile="unknown-both"),
    caller("resolved-same-unbound", True, resolved=True, resident=True),
    caller("resolved-same-bound-elsewhere", True, resolved=True, bound_session="other"),
    caller("resolved-unknown-session-match", True, resolved=True,
           profile="unknown-session", bound_session="invoker"),
    caller("resolved-unknown-engine-match", True, resolved=True,
           profile="unknown-engine", bound_session="invoker"),
    caller("resolved-unknown-both-match", True, resolved=True,
           profile="unknown-both", bound_session="invoker"),
    caller("resolved-unknown-key-over-session", True, resolved=True,
           profile="unknown-both", bound_session="other", bound_conversation="lane"),
    caller("resolved-unknown-key-only", True, resolved=True, session="",
           profile="unknown-both", bound_conversation="lane"),
    caller("resolved-unknown-unbound", resolved=True, profile="unknown-both"),
    caller("resolved-unknown-bound-elsewhere", resolved=True,
           profile="unknown-both", bound_session="other", bound_conversation="other"),
    caller("context-without-ids-cold", True, session="", conversation="", reads_resident=True),
    caller("context-without-ids-bound", session="", conversation="", bound_session="other"),
)


# Patch only effectful renderers. Authorization and the public dispatcher run.
RENDERERS = (
    "_status_text", "_help_text", "_doctor_text", "_doctor_clean_text",
    "_doctor_clean_lifecycle_text", "_doctor_source_text", "_doctor_retention_text",
    "_doctor_clean_apply_text", "_doctor_clean_lifecycle_apply_text",
    "_doctor_repair_text", "_doctor_repair_apply_text", "_doctor_repair_level3_text",
    "_doctor_repair_level3_apply_text", "_doctor_repair_schema_stamp_text",
    "_doctor_repair_schema_stamp_apply_text", "_doctor_source_apply_text",
    "_backup_text", "_rotate_text", "_rotate_apply_text", "_rollups_text",
    "_assertions_text", "_preset_text", "_embedding_warmup_text", "_embedding_backfill_text",
)


@pytest.mark.parametrize("raw,is_read", COMMANDS)
@pytest.mark.parametrize("case", CALLERS, ids=lambda case: case.name)
def test_public_handler_ownership_table(monkeypatch, raw, is_read, case):
    events = []

    class Engine:
        @property
        def _session_id(self):
            events.append("session")
            return case.bound_session

        @property
        def _conversation_id(self):
            events.append("conversation")
            return case.bound_conversation

        @property
        def _store(self):
            events.append("store")
            return types.SimpleNamespace(db_path="opened.db")

    def render(*_args):
        events.append("dispatch")
        return "command output"

    def resident():
        events.append("resident")
        return case.resident

    for name in RENDERERS:
        monkeypatch.setattr(command, name, render)
    monkeypatch.setattr(command, "has_resident_lcm_engine", resident)
    result = command.handle_lcm_command(raw, Engine(), **case.kwargs)
    expected_events = []
    if case.refusal:
        expected = case.refusal
    else:
        if case.kwargs["session_context_present"]:
            expected_events += ["session", "conversation"]
        if not is_read and case.reads_resident:
            expected_events.append("resident")
        if not is_read and not case.write_allowed:
            expected = OWNERSHIP_REFUSAL
        else:
            expected = "command output"
            expected_events.append("dispatch")
            if case.disclose:
                expected += "\nstore: opened.db"
                expected_events.append("store")
    assert result == expected
    assert events == expected_events


@pytest.mark.parametrize("failure", ["binding", "resident"])
def test_observation_errors_propagate(monkeypatch, failure):
    def fail(*_args):
        raise RuntimeError("observation failed")

    class Engine:
        _session_id = property(fail) if failure == "binding" else ""
        _conversation_id = ""

    monkeypatch.setattr(command, "has_resident_lcm_engine", fail)
    with pytest.raises(RuntimeError, match="^observation failed$"):
        command.handle_lcm_command("backup", Engine(), **CALLERS[6].kwargs)


def test_task_local_context_failure_does_not_read_global_environment(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("authorization_context_plugin", path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    context = types.ModuleType("gateway.session_context")

    def fail(*_args):
        raise RuntimeError("task context unavailable")

    context.get_session_env = fail
    monkeypatch.setitem(sys.modules, "gateway", types.ModuleType("gateway"))
    monkeypatch.setitem(sys.modules, "gateway.session_context", context)
    monkeypatch.setenv("HERMES_SESSION_ID", "process-session")
    assert plugin._session_context_value("HERMES_SESSION_ID") == ""
