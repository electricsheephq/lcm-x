"""#900: a session-scoped Hermes prompt section without replay scaffold grammar."""

import importlib
import logging
import re

import pytest

from tests.test_packaging_install import (
    _ensure_agent_context_engine_importable,
    _load_plugin_entrypoint_module,
)


PHRASE = "This conversation uses Lossless Context Management (LCM)"


class LegacyContext:
    context_engine_tool_handlers_receive_messages = True

    def __init__(self):
        self.engine = None
        self.calls = []

    def register_context_engine(self, engine):
        self.engine = engine
        self.calls.append(("engine", engine.name))

    def register_hook(self, name, callback):
        self.calls.append(("hook", name))

    def register_skill(self, name, path, description=""):
        self.calls.append(("skill", name))

    def register_tool(self, **kwargs):
        self.calls.append(("tool", kwargs["name"]))

    def register_command(self, name, callback, description=""):
        self.calls.append(("command", name))


class SectionContext(LegacyContext):
    def __init__(self):
        super().__init__()
        self.sections = []

    def register_system_prompt_section(self, section_id, content, position="after_memory"):
        self.sections.append((section_id, content, position))


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    _ensure_agent_context_engine_importable(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "true")
    module = _load_plugin_entrypoint_module("hermes_lcm_issue_900")
    contexts = []

    def register(ctx):
        contexts.append(ctx)
        module.register(ctx)
        return ctx

    yield module, register
    for ctx in contexts:
        if ctx.engine is not None:
            ctx.engine.shutdown()


def test_u1_registers_one_callable_section(plugin):
    module, register = plugin
    ctx = register(SectionContext())
    assert len(ctx.sections) == 1
    section_id, content, position = ctx.sections[0]
    assert section_id == "lcm-x"
    assert re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", section_id)
    assert position == "after_memory"
    assert callable(content)
    ctx.engine.on_session_start("bound", platform="cli")
    text = content({"session_id": "bound"})
    assert text == module.LCM_SYSTEM_PROMPT_NOTE
    assert len(text) <= 700 < 4000
    assert PHRASE in text
    assert "[Note:" not in text
    assert "Earlier turns have been compacted into hierarchical summaries below." not in text


def test_u2_resolves_only_lcm_sessions_and_fails_empty(plugin, monkeypatch):
    module, register = plugin
    ctx = register(SectionContext())
    content = ctx.sections[0][1]
    assert content({"session_id": "bound"}) == ""
    assert content({}) == ""
    ctx.engine.on_session_start("bound", platform="cli")
    assert content({"session_id": "bound"}) == module.LCM_SYSTEM_PROMPT_NOTE
    assert content({"session_id": "other"}) == ""
    # Keep a strong reference: the registry holds weak values.
    class OtherEngine:
        pass
    other_engine = OtherEngine()
    other_engine.name = "other-context-engine"
    other_engine._session_id = "other"
    other_engine.ingest = None
    registry = importlib.import_module(f"{module.__name__}.engine_registry")
    monkeypatch.setitem(registry._ACTIVE_ENGINES_BY_SESSION_ID, "other", other_engine)
    assert content({"session_id": "other"}) == ""

    def broken_resolution(*args, **kwargs):
        raise RuntimeError("resolution failed")

    monkeypatch.setattr(registry, "resolve_active_lcm_engine", broken_resolution)
    assert content({"session_id": "bound"}) == ""


def test_u3_older_host_keeps_all_other_registrations(plugin, caplog):
    _, register = plugin
    modern = register(SectionContext())
    with caplog.at_level(logging.INFO):
        legacy = register(LegacyContext())
    assert legacy.calls == modern.calls
    assert ("hook", "pre_llm_call") in legacy.calls
    assert ("hook", "post_llm_call") in legacy.calls
    assert ("command", "lcm") in legacy.calls
    assert sum("system-prompt section unavailable" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_registration_rejection_logs_once_and_continues(plugin, caplog, error):
    _, register = plugin

    class RejectingContext(SectionContext):
        def register_system_prompt_section(self, *args, **kwargs):
            raise error("unsupported section")

    modern = register(SectionContext())
    with caplog.at_level(logging.INFO):
        rejected = register(RejectingContext())
    assert rejected.calls == modern.calls
    assert sum("system-prompt section unavailable" in record.message for record in caplog.records) == 1


def test_inactive_plugin_does_not_register_section(plugin, monkeypatch):
    module, register = plugin
    monkeypatch.setattr(module, "_engine_took_slot", lambda *args: False)
    assert register(SectionContext()).sections == []


@pytest.mark.parametrize("blocks", [False, True])
def test_u4_host_prompt_with_section_is_not_replay_scaffold(plugin, blocks):
    module, register = plugin
    ctx = register(SectionContext())
    ctx.engine.on_session_start("bound", platform="cli")
    text = "Ordinary host instructions. " * 200 + "\n## Plugin Context: lcm-x\n" + module.LCM_SYSTEM_PROMPT_NOTE
    content = [{"type": "text", "text": text}] if blocks else text
    system = {"role": "system", "content": content}
    engine = ctx.engine
    assert not engine._is_replayed_context_scaffold_message(system)
    assert not engine._is_lcm_emitted_head_row(system, 10**9)
    summary = {"role": "user", "content": "[Recent Summary (d0, node 1)]\nHistory\n[Expand for details: node 1]"}
    # A summary is present so a missing summary cannot explain the empty digest.
    assert engine._replay_snapshot_digest([system, summary], require_lcm_system_note=False)
    assert engine._replay_snapshot_digest([system, summary], require_lcm_system_note=True) == ""


@pytest.mark.parametrize("blocks", [False, True])
def test_u5_assembly_does_not_append_second_note(plugin, blocks):
    module, register = plugin
    ctx = register(SectionContext())
    ctx.engine.on_session_start("bound", platform="cli")
    text = "Ordinary host prompt.\n\n" + module.LCM_SYSTEM_PROMPT_NOTE
    content = [{"type": "text", "text": text}] if blocks else text
    system = {"role": "system", "content": content}
    assembled = ctx.engine._assemble_context(
        system, [{"role": "user", "content": "A fresh turn."}], persist=False,
    )
    assert assembled[0]["content"] == content
    assert text.count(PHRASE) == 1
    assert system["content"] == content
