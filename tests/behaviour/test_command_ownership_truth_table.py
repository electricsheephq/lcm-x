"""Public command authorization decisions, including the five #852 corrections."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
# The expected access column is deliberately independent of READ_LCM_SUBCOMMANDS.
COMMANDS = [
    ("", True), ("status", True), ("help", True), ("doctor clean", True),
    ("doctor clean lifecycle", True), ("doctor source", True),
    ("doctor retention", True), ("rotate", True), ("rollups", True),
    ("preset show", True), ("preset show balanced", True), ("preset suggest", True),
    ("doctor", False), ("doctor clean apply", False),
    ("doctor clean lifecycle apply", False), ("doctor repair", False),
    ("doctor repair apply", False), ("doctor repair level3", False),
    ("doctor repair level3 apply", False), ("doctor repair schema-stamp", False),
    ("doctor repair schema-stamp apply", False), ("doctor source apply", False),
    ("backup", False), ("rotate apply", False), ("rollups rebuild day", False),
    ("assertions rebuild --apply", False), ("embed warmup", False),
    ("embed backfill", False), ("embed backfill --apply", False),
    ("preset apply balanced --dry-run", False), ("future-command", False),
    ("status extra", False), (" HELP ", True), ("PRESET SHOW balanced", True),
]
# name, resolved, caller profile, engine profile, caller id/key, bound id/key,
# resident, expected write permission. Reads pass unless profiles mismatch.
CONTEXTS = [
    ("resolved-equal", True, "p", "p", "s", "k", "other", "other", True, True),
    ("resolved-mismatch", True, "p", "q", "s", "k", "s", "k", True, False),
    ("fallback-mismatch", False, "p", "q", "s", "k", "s", "k", False, False),
    ("fallback-cold", False, "p", "p", "s", "k", "", "", False, True),
    ("fallback-cold-resident", False, "p", "p", "s", "k", "", "", True, False),
    ("fallback-bound-id", False, "p", "p", "s", "k", "s", "other", True, True),
    ("fallback-id-wins", False, "p", "p", "s", "k", "other", "k", True, False),
    ("fallback-key-only", False, "p", "p", "", "k", "other", "k", True, True),
    ("fallback-conversation-bound", False, "p", "p", "", "k", "", "other", False, False),
    ("fallback-unknown-caller", False, "", "p", "s", "k", "s", "k", True, False),
    ("fallback-unknown-engine", False, "p", "", "s", "k", "s", "k", True, False),
    ("fallback-both-unknown", False, "", "", "s", "k", "s", "k", True, False),
    ("resolved-unknown-caller-bound", True, "", "p", "s", "k", "s", "other", True, True),
    ("resolved-unknown-engine-bound", True, "p", "", "s", "k", "s", "other", True, True),
    ("resolved-both-unknown-key", True, "", "", "s", "k", "other", "k", True, True),
    ("resolved-unknown-unbound", True, "", "p", "s", "k", "other", "other", True, False),
    ("no-context", False, "", "p", "", "", "other", "other", True, True),
    ("no-context-mismatch", False, "p", "q", "", "", "", "", False, False),
]


@pytest.mark.parametrize("raw,read", COMMANDS, ids=[x[0] or "default" for x in COMMANDS])
@pytest.mark.parametrize("context", CONTEXTS, ids=[x[0] for x in CONTEXTS])
def test_command_ownership(raw, read, context, monkeypatch):
    _, resolved, caller, profile, sid, key, bound, conversation, resident, write = context
    engine = SimpleNamespace(
        _session_id=bound, _conversation_id=conversation,
        _store=SimpleNamespace(db_path="opened.db"),
        _config=SimpleNamespace(database_path="configured-but-not-opened.db"),
    )
    dispatched = []

    def render(*args, **kwargs):
        dispatched.append(True)
        return "dispatched"

    # Fence effects after the real public authorization/grammar dispatcher.
    renderers = (
        "_status_text _help_text _doctor_text _doctor_clean_text "
        "_doctor_clean_apply_text _doctor_clean_lifecycle_text "
        "_doctor_clean_lifecycle_apply_text _doctor_source_text _doctor_source_apply_text "
        "_doctor_retention_text _doctor_repair_text _doctor_repair_apply_text "
        "_doctor_repair_level3_text _doctor_repair_level3_apply_text "
        "_doctor_repair_schema_stamp_text _doctor_repair_schema_stamp_apply_text "
        "_backup_text _rotate_text _rotate_apply_text _rollups_text _assertions_text "
        "_preset_text _embedding_warmup_text _embedding_backfill_text"
    )
    for name in renderers.split():
        monkeypatch.setattr(command, name, render)
    monkeypatch.setattr(command, "has_resident_lcm_engine", lambda: resident)
    result = command.handle_lcm_command(
        raw, engine, session_engine_resolved=resolved, session_context_present=bool(sid or key),
        session_profile=caller, engine_profile=profile,
        context_session_id=sid, context_conversation_id=key,
    )
    if caller and profile and caller != profile:
        assert result == PROFILE_REFUSAL
        assert dispatched == []
    elif not read and not write:
        assert result == OWNERSHIP_REFUSAL
        assert dispatched == []
    else:
        disclosure = not resolved and not (caller and caller == profile)
        assert result == "dispatched" + ("\nstore: opened.db" if disclosure else "")
        assert dispatched == [True]


@pytest.mark.parametrize("failure", ["task-local", "profile-import", "profile-helper", "profile-home"])
def test_host_adapter_failure_is_unknown_not_process_identity(monkeypatch, failure):
    path = Path(__file__).resolve().parents[2] / "__init__.py"
    spec = importlib.util.spec_from_file_location("ownership_characterization_plugin", path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    host = ModuleType("hermes_constants")
    host.profile_name_for_home = lambda home: "p"
    monkeypatch.setitem(sys.modules, "hermes_constants", host)
    session = ModuleType("gateway.session_context")
    context = {"HERMES_SESSION_ID": "s", "HERMES_SESSION_PROFILE": "p"}

    def get_session_env(name, default=""):
        if failure == "task-local":
            raise RuntimeError("task-local lookup failed")
        return context.get(name, default)

    session.get_session_env = get_session_env
    monkeypatch.setitem(sys.modules, "gateway.session_context", session)
    for key in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY", "HERMES_SESSION_PROFILE"):
        monkeypatch.setenv(key, "process-global-must-not-leak")
    engine = SimpleNamespace(_hermes_home="captured-home")
    if failure == "profile-import":
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
    elif failure == "profile-helper":
        def broken(home):
            raise RuntimeError("profile helper failed")
        host.profile_name_for_home = broken
    elif failure == "profile-home":
        engine._hermes_home = ""
    resolutions, invocations = [], []

    def resolve(**kwargs):
        resolutions.append(kwargs)
        return None

    def dispatch(raw, selected, **kwargs):
        invocations.append((raw, selected, kwargs))
        return "observed"

    handler = plugin._make_command_handler(dispatch, engine, resolve)
    assert handler("status") == "observed"
    assert invocations == [("status", engine, {
        "session_engine_resolved": False,
        "session_context_present": failure != "task-local",
        "session_profile": "" if failure == "task-local" else "p",
        "engine_profile": "p" if failure == "task-local" else "",
        "context_session_id": "" if failure == "task-local" else "s",
        "context_conversation_id": "",
    })]
    assert resolutions == ([] if failure == "task-local" else [{"session_id": "s", "conversation_id": ""}])
