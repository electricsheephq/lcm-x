"""Slash-command fallback must not mutate another profile's store (#852)."""

import importlib.util
from pathlib import Path
import sqlite3

import pytest

from hermes_lcm import command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import VectorStore


REFUSAL = (
    "The command could not resolve this session's LCM engine in this process; "
    "run it from a process launched for this profile."
)


@pytest.fixture
def plugin():
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("issue_852_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engines(tmp_path, monkeypatch):
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
    handler, calls = _handler(plugin, monkeypatch, prototype, {context_key: "other-profile"})

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

    assert handler("embed warmup") == expected
    assert _snapshot(prototype)["lcm_embedding_profile"] == 2
    assert _snapshot(session) == before
    assert calls == []


def test_resolved_session_write_uses_session_store(plugin, engines, monkeypatch):
    prototype, session = engines
    before = _snapshot(prototype)
    context = {"HERMES_SESSION_ID": "session-id", "HERMES_SESSION_KEY": "session-key"}
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
    handler, _calls = _handler(
        plugin, monkeypatch, prototype, {"HERMES_SESSION_KEY": "other-profile"}
    )
    assert handler(raw_args) == REFUSAL
    assert _snapshot(prototype) == before
