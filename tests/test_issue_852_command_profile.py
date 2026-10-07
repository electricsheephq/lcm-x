"""Slash commands must respect profile and bound-session ownership (#852)."""

import importlib.util
from pathlib import Path
import sqlite3
import sys
import types
import weakref

import pytest

from hermes_lcm import command, engine as engine_module, engine_registry
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import VectorStore


REFUSAL = (
    "The command could not resolve this session's LCM engine in this process; "
    "run it from a process launched for this profile."
)
OWNERSHIP_REFUSAL = (
    "The command could not prove ownership of this session's LCM engine; "
    "retry after the session has started."
)


@pytest.fixture
def plugin(monkeypatch):
    host = types.ModuleType("hermes_constants")
    host.profile_name_for_home = lambda home: Path(home).name
    monkeypatch.setitem(sys.modules, "hermes_constants", host)
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("issue_852_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engines(tmp_path, monkeypatch):
    for name in ("_ACTIVE_ENGINES_BY_SESSION_ID", "_ACTIVE_ENGINES_BY_CONVERSATION_ID"):
        registry = weakref.WeakValueDictionary()
        monkeypatch.setattr(engine_registry, name, registry)
        monkeypatch.setattr(engine_module, name, registry)
    class OfflineProvider:
        provider_id = "ollama"
        model_id = "test-model"

        def embed_query(self, _text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(command, "resolve_provider", lambda _config: OfflineProvider())
    result = []
    for profile in ("launch", "session"):
        config = LCMConfig(
            database_path=str(tmp_path / profile / "lcm.db"),
            embeddings_enabled=True,
            embedding_provider="ollama",
            embedding_model="test-model",
        )
        result.append(LCMEngine(config=config, hermes_home=str(tmp_path / profile)))
    yield result
    for engine in result:
        engine.shutdown()


def _snapshot(engine):
    with sqlite3.connect(engine._store.db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        return {
            name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for (name,) in tables
        }


def _handler(plugin, monkeypatch, prototype, context, resolved=None):
    monkeypatch.setattr(plugin, "_session_context_value", lambda key: context.get(key, ""))
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return resolved

    return plugin._make_command_handler(command.handle_lcm_command, prototype, resolve), calls


@pytest.mark.parametrize("raw_args", ["embed backfill --apply", "embed warmup"])
@pytest.mark.parametrize("context_key", ["HERMES_SESSION_ID", "HERMES_SESSION_KEY"])
def test_unresolved_session_refuses_writes(plugin, engines, monkeypatch, raw_args, context_key):
    prototype, _session = engines
    if "backfill" in raw_args:
        vectors = VectorStore(prototype._store.db_path)
        vectors.register_profile("test-model", "ollama", 3)
        vectors.close()
    before = _snapshot(prototype)
    # review F1: refusal requires a known different session profile.
    handler, calls = _handler(plugin, monkeypatch, prototype, {
        context_key: "other-profile", "HERMES_SESSION_PROFILE": "session",
    })

    result = handler(raw_args)

    assert _snapshot(prototype) == before
    assert result == REFUSAL
    assert len(calls) == 1


@pytest.mark.parametrize("raw_args", [
    "", "status", "help", "doctor clean", "doctor clean lifecycle", "doctor source",
    "doctor retention", "rotate", "rollups", "preset show", "preset suggest",
])
def test_unresolved_session_read_names_fallback_store(plugin, engines, monkeypatch, raw_args):
    prototype, _session = engines
    expected = command.handle_lcm_command(raw_args, prototype)
    before = _snapshot(prototype)
    handler, _calls = _handler(
        plugin, monkeypatch, prototype, {"HERMES_SESSION_ID": "other-profile"}
    )

    assert handler(raw_args) == expected + f"\nstore: {prototype._config.database_path}"
    assert _snapshot(prototype) == before


def test_no_session_context_write_uses_prototype(plugin, engines, monkeypatch):
    prototype, session = engines
    expected = command.handle_lcm_command("embed warmup", session)
    before = _snapshot(session)
    handler, calls = _handler(plugin, monkeypatch, prototype, {})

    # review F1: an unknown session profile now discloses the fallback store.
    assert handler("embed warmup") == expected + f"\nstore: {prototype._config.database_path}"
    assert _snapshot(prototype)["lcm_embedding_profile"] == 2
    assert _snapshot(session) == before
    assert calls == []


def test_resolved_session_write_uses_session_store(plugin, engines, monkeypatch):
    prototype, session = engines
    before = _snapshot(prototype)
    context = {
        "HERMES_SESSION_ID": "session-id", "HERMES_SESSION_KEY": "session-key",
        "HERMES_SESSION_PROFILE": "session",
    }
    handler, calls = _handler(plugin, monkeypatch, prototype, context, session)

    assert "status: ready" in handler("embed warmup")
    assert _snapshot(session)["lcm_embedding_profile"] == 2
    assert _snapshot(prototype) == before
    assert calls == [{"session_id": "session-id", "conversation_id": "session-key"}]


def test_resolver_legacy_return_stays_an_engine(plugin, engines, monkeypatch):
    prototype, _session = engines
    monkeypatch.setattr(plugin, "_session_context_value", lambda _key: "")
    assert plugin._command_engine_for_current_session(prototype, lambda **_kw: None) is prototype


@pytest.mark.parametrize("raw_args", [
    "doctor", "doctor clean apply", "doctor clean lifecycle apply", "doctor repair",
    "doctor repair apply", "doctor repair level3", "doctor repair level3 apply",
    "doctor repair schema-stamp", "doctor repair schema-stamp apply", "doctor source apply",
    "backup", "rotate apply", "rollups rebuild day", "assertions rebuild --apply",
    "preset apply codex_gpt_long_context --dry-run", "embed backfill", "future-command",
])
def test_other_write_paths_fail_closed(plugin, engines, monkeypatch, raw_args):
    prototype, _session = engines
    before = _snapshot(prototype)
    # review F1: refusal requires a known different session profile.
    handler, _calls = _handler(plugin, monkeypatch, prototype, {
        "HERMES_SESSION_KEY": "other-profile", "HERMES_SESSION_PROFILE": "session",
    })
    assert handler(raw_args) == REFUSAL
    assert _snapshot(prototype) == before


@pytest.mark.parametrize("raw_args", ["embed warmup", "doctor", "status"])
def test_cold_start_same_profile_matches_resolved_output(
    plugin, engines, monkeypatch, raw_args,
):
    prototype, _session = engines
    context = {"HERMES_SESSION_ID": "fresh-session", "HERMES_SESSION_PROFILE": "launch"}
    resolved, _ = _handler(plugin, monkeypatch, prototype, context, prototype)
    expected = resolved(raw_args)
    fallback, _ = _handler(plugin, monkeypatch, prototype, context)

    assert fallback(raw_args).encode() == expected.encode()
    if raw_args == "embed warmup":
        assert _snapshot(prototype)["lcm_embedding_profile"] == 2


def test_different_profile_read_names_store(plugin, engines, monkeypatch):
    prototype, _session = engines
    before = _snapshot(prototype)
    handler, _ = _handler(plugin, monkeypatch, prototype, {
        "HERMES_SESSION_ID": "fresh-session", "HERMES_SESSION_PROFILE": "session",
    })

    assert handler("status") == REFUSAL
    assert _snapshot(prototype) == before


@pytest.mark.parametrize("raw_args", ["status", "embed warmup"])
def test_resolved_engine_profile_mismatch_refuses(plugin, engines, monkeypatch, raw_args):
    prototype, session = engines
    before = [_snapshot(engine) for engine in engines]
    handler, calls = _handler(plugin, monkeypatch, prototype, {
        "HERMES_SESSION_KEY": "shared-lane", "HERMES_SESSION_PROFILE": "launch",
    }, session)

    assert handler(raw_args) == REFUSAL
    assert [_snapshot(engine) for engine in engines] == before
    assert calls == [{"session_id": "", "conversation_id": "shared-lane"}]


@pytest.mark.parametrize("profile_state", ["same", "unset", "import", "helper", "home"])
@pytest.mark.parametrize("context_key", ["HERMES_SESSION_ID", "HERMES_SESSION_KEY"])
@pytest.mark.parametrize("raw_args", ["embed warmup", "rotate apply"])
def test_unresolved_other_bound_session_refuses_write_allows_read(
    plugin, engines, monkeypatch, profile_state, context_key, raw_args,
):
    prototype, _session = engines
    prototype.on_session_start("bound-session")
    context = {context_key: "invoking-session", "HERMES_SESSION_PROFILE": "launch"}
    if profile_state == "unset":
        context.pop("HERMES_SESSION_PROFILE")
    elif profile_state == "import":
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
    elif profile_state == "helper":
        def broken(_home):
            raise RuntimeError("profile lookup unavailable")
        monkeypatch.setattr(sys.modules["hermes_constants"], "profile_name_for_home", broken)
    elif profile_state == "home":
        monkeypatch.setattr(prototype, "_hermes_home", "")
    expected = command.handle_lcm_command("status", prototype)
    if profile_state != "same":
        expected += f"\nstore: {prototype._config.database_path}"
    before = _snapshot(prototype)
    handler, _ = _handler(plugin, monkeypatch, prototype, context)

    assert handler(raw_args) == OWNERSHIP_REFUSAL
    assert handler("status") == expected
    assert prototype._session_id == "bound-session"
    assert _snapshot(prototype) == before


@pytest.mark.parametrize("profile_known", [True, False])
def test_unresolved_matching_bound_session_allows_write(
    plugin, engines, monkeypatch, profile_known,
):
    prototype, session = engines
    prototype.on_session_start("invoking-session")
    context = {"HERMES_SESSION_ID": "invoking-session"}
    if profile_known:
        context["HERMES_SESSION_PROFILE"] = "launch"
    expected = command.handle_lcm_command("embed warmup", session)
    if not profile_known:
        expected = OWNERSHIP_REFUSAL
    before = _snapshot(prototype)
    handler, _ = _handler(plugin, monkeypatch, prototype, context)

    assert handler("embed warmup") == expected
    if profile_known:
        assert _snapshot(prototype)["lcm_embedding_profile"] == 2
    else:
        assert _snapshot(prototype) == before


def test_session_profile_unset_write_is_refused(plugin, engines, monkeypatch):
    prototype, _session = engines
    before = _snapshot(prototype)
    handler, _ = _handler(plugin, monkeypatch, prototype, {"HERMES_SESSION_ID": "fresh"})

    assert handler("embed warmup") == OWNERSHIP_REFUSAL
    assert _snapshot(prototype) == before


@pytest.mark.parametrize("failure", ["import", "helper", "home"])
def test_host_profile_helper_failure_is_unknown(plugin, engines, monkeypatch, failure):
    prototype, _session = engines
    before = _snapshot(prototype)
    if failure == "import":
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
    elif failure == "helper":
        def broken(_home):
            raise RuntimeError("profile lookup unavailable")
        monkeypatch.setattr(sys.modules["hermes_constants"], "profile_name_for_home", broken)
    else:
        monkeypatch.setattr(prototype, "_hermes_home", "")
    handler, _ = _handler(plugin, monkeypatch, prototype, {
        "HERMES_SESSION_ID": "fresh", "HERMES_SESSION_PROFILE": "session",
    })

    assert handler("embed warmup") == OWNERSHIP_REFUSAL
    assert _snapshot(prototype) == before


@pytest.mark.parametrize("resident", [False, True])
def test_unbound_same_profile_write_requires_no_resident_engine(
    plugin, engines, monkeypatch, resident,
):
    prototype, session = engines
    if resident:
        session.on_session_start("resident-session")
    assert engine_registry.has_resident_lcm_engine() is resident
    before = _snapshot(prototype)
    handler, _ = _handler(plugin, monkeypatch, prototype, {
        "HERMES_SESSION_ID": "fresh", "HERMES_SESSION_PROFILE": "launch",
    })

    result = handler("embed warmup")

    if resident:
        assert result == OWNERSHIP_REFUSAL
        assert _snapshot(prototype) == before
    else:
        assert "status: ready" in result
        assert _snapshot(prototype)["lcm_embedding_profile"] == 2


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("matching", [False, True])
def test_key_only_context_checks_bound_conversation(
    plugin, engines, monkeypatch, resident, matching,
):
    prototype, _session = engines
    prototype.on_session_start(
        "bound-session", conversation_id="invoking-key" if matching else "other-key",
    )
    # A conversation binding alone is still bound, never a cold prototype.
    monkeypatch.setattr(prototype, "_session_id", "")
    if not resident:
        engine_registry._ACTIVE_ENGINES_BY_SESSION_ID.clear()
        engine_registry._ACTIVE_ENGINES_BY_CONVERSATION_ID.clear()
    assert engine_registry.has_resident_lcm_engine() is resident
    before = _snapshot(prototype)
    handler, calls = _handler(plugin, monkeypatch, prototype, {
        "HERMES_SESSION_KEY": "invoking-key", "HERMES_SESSION_PROFILE": "launch",
    })

    result = handler("embed warmup")

    if matching:
        assert "status: ready" in result
        assert _snapshot(prototype)["lcm_embedding_profile"] == 2
    else:
        assert result == OWNERSHIP_REFUSAL
        assert _snapshot(prototype) == before
    assert calls == [{"session_id": "", "conversation_id": "invoking-key"}]


@pytest.mark.parametrize("unknown_profile", ["session", "engine", "both"])
@pytest.mark.parametrize("context_key", ["HERMES_SESSION_ID", "HERMES_SESSION_KEY"])
@pytest.mark.parametrize("matching", [False, True])
def test_resolved_unknown_profiles_require_invoker_binding(
    plugin, engines, monkeypatch, unknown_profile, context_key, matching,
):
    prototype, _session = engines
    bound_id = "invoker" if matching else "elsewhere"
    prototype.on_session_start(bound_id, conversation_id=bound_id)
    context = {context_key: "invoker"}
    if unknown_profile == "engine":
        context["HERMES_SESSION_PROFILE"] = "launch"
    if unknown_profile in {"engine", "both"}:
        monkeypatch.setattr(prototype, "_hermes_home", "")
    before = _snapshot(prototype)
    handler, _ = _handler(plugin, monkeypatch, prototype, context, prototype)

    result = handler("embed warmup")

    if matching:
        assert "status: ready" in result
        assert _snapshot(prototype)["lcm_embedding_profile"] == 2
    else:
        assert result == OWNERSHIP_REFUSAL
        assert _snapshot(prototype) == before


@pytest.mark.parametrize("config_path", ["", "not-the-opened.db"])
def test_store_line_names_opened_database(plugin, engines, monkeypatch, config_path):
    prototype, _session = engines
    monkeypatch.setattr(prototype._config, "database_path", config_path)
    expected = command.handle_lcm_command("status", prototype)
    handler, _ = _handler(plugin, monkeypatch, prototype, {"HERMES_SESSION_KEY": "fresh"})

    assert handler("status") == expected + f"\nstore: {prototype._store.db_path}"


def test_engine_profile_seam_uses_captured_home(plugin, engines, monkeypatch):
    prototype, _session = engines
    seen = []
    def profile_for_home(home):
        seen.append(home)
        return "launch"
    monkeypatch.setattr(sys.modules["hermes_constants"], "profile_name_for_home", profile_for_home)
    assert plugin._command_engine_profile(prototype) == "launch"
    assert seen == [prototype._hermes_home]


def test_plugin_import_does_not_require_host(monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("issue_852_no_host_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module._command_engine_profile)
