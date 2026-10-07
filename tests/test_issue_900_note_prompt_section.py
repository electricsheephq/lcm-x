"""#900: a session-scoped Hermes prompt section without replay scaffold grammar."""

import importlib
import logging
import re
import threading

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
    monkeypatch.delenv("LCM_DISABLED_TOOLS", raising=False)
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
    registry = importlib.import_module(f"{module.__name__}.engine_registry")
    assert registry.resolve_active_lcm_engine(session_id="bound") is None
    assert content({"session_id": "bound"}) == ""
    assert content({}) == ""
    ctx.engine.on_session_start("bound", platform="cli")
    assert registry.resolve_active_lcm_engine(session_id="bound") is ctx.engine
    assert content({"session_id": "bound"}) == module.LCM_SYSTEM_PROMPT_NOTE
    assert content({"session_id": "other"}) == ""
    # Keep a strong reference: the registry holds weak values.
    class OtherEngine:
        pass
    other_engine = OtherEngine()
    other_engine.name = "other-context-engine"
    other_engine._session_id = "other"
    other_engine.ingest = None
    monkeypatch.setitem(registry._ACTIVE_ENGINES_BY_SESSION_ID, "other", other_engine)
    assert content({"session_id": "other"}) == ""

    def broken_resolution(*args, **kwargs):
        raise RuntimeError("resolution failed")

    monkeypatch.setattr(registry, "resolve_active_lcm_engine", broken_resolution)
    assert content({"session_id": "bound"}) == ""


@pytest.mark.parametrize("setting,pattern,session_id", [
    ("LCM_IGNORE_SESSION_PATTERNS", "ignored", "ignored"),
    ("LCM_STATELESS_SESSION_PATTERNS", "stateless", "stateless"),
    ("LCM_IGNORE_SESSION_PATTERNS", "cli:ignored", "ignored"),
    ("LCM_STATELESS_SESSION_PATTERNS", "cli:stateless", "stateless"),
])
def test_section_omits_bypassed_session_by_requested_id(plugin, monkeypatch, setting, pattern, session_id):
    module, register = plugin
    monkeypatch.setenv(setting, pattern)
    ctx = register(SectionContext())
    content = ctx.sections[0][1]
    registry = importlib.import_module(f"{module.__name__}.engine_registry")
    ctx.engine.on_session_start("ordinary", platform="cli")
    assert content({"session_id": "ordinary"}) == module.LCM_SYSTEM_PROMPT_NOTE
    ctx.engine.on_session_start(session_id, platform="cli")
    assert registry.resolve_active_lcm_engine(session_id=session_id) is ctx.engine
    assert ctx.engine.side_channel_active
    assert not ctx.engine.current_session_ignored
    assert not ctx.engine.current_session_stateless
    assert content({"session_id": session_id}) == ""
    assert content({"session_id": "ordinary"}) == module.LCM_SYSTEM_PROMPT_NOTE


def test_all_enabled_note_is_byte_identical(plugin):
    module, _ = plugin
    assert module.lcm_system_prompt_note(set()) == module.LCM_SYSTEM_PROMPT_NOTE


def test_section_omits_active_auxiliary_session(plugin):
    module, register = plugin
    ctx = register(SectionContext())
    ctx.engine.on_session_start("auxiliary-child", platform="cli")
    ctx.engine._mark_thread_context_stateless("auxiliary-child")
    registry = importlib.import_module(f"{module.__name__}.engine_registry")
    assert registry.resolve_active_lcm_engine(session_id="auxiliary-child") is ctx.engine
    assert ctx.engine._thread_context_stateless()
    assert not ctx.engine._session_id_matches_lcm_bypass_filters("auxiliary-child", platform="cli")

    assert ctx.sections[0][1]({"session_id": "auxiliary-child"}) == ""


@pytest.mark.parametrize("setting", ["LCM_IGNORE_SESSION_PATTERNS", "LCM_STATELESS_SESSION_PATTERNS"])
def test_section_omits_rotated_bypass_session(plugin, monkeypatch, setting):
    _, register = plugin
    monkeypatch.setenv(setting, "bypassed-old")
    ctx = register(SectionContext())
    ctx.engine.on_session_start("bypassed-old", platform="cli")
    ctx.engine.on_session_start(
        "rotated-child", boundary_reason="compression", old_session_id="bypassed-old", platform="cli",
    )
    assert ctx.engine._has_lcm_bypass_lineage_session("rotated-child", platform="cli")
    assert ctx.engine._lcm_session_last_bypassed["rotated-child"]
    assert not ctx.engine._session_id_matches_lcm_bypass_filters("rotated-child", platform="cli")

    assert ctx.sections[0][1]({"session_id": "rotated-child"}) == ""


def test_section_keeps_foreground_note_while_auxiliary_child_is_active(plugin):
    module, register = plugin
    ctx = register(SectionContext())
    ctx.engine.on_session_start("foreground", platform="cli")
    ctx.engine._mark_thread_context_stateless("auxiliary-child")
    assert ctx.engine._thread_context_stateless()
    assert "auxiliary-child" in ctx.engine._active_auxiliary_session_ids()

    assert ctx.sections[0][1]({"session_id": "foreground"}) == module.LCM_SYSTEM_PROMPT_NOTE


@pytest.mark.parametrize("disabled", [
    {"lcm_expand"}, {"lcm_grep"}, {"lcm_describe"},
    {"lcm_grep", "lcm_describe", "lcm_expand"},
])
def test_section_names_only_enabled_tools(plugin, monkeypatch, disabled):
    module, register = plugin
    monkeypatch.setenv("LCM_DISABLED_TOOLS", ",".join(sorted(disabled)))
    ctx = register(SectionContext())
    ctx.engine.on_session_start("ordinary", platform="cli")
    text = ctx.sections[0][1]({"session_id": "ordinary"})
    assert PHRASE in text
    assert text == module.lcm_system_prompt_note(disabled)
    for name in {"lcm_grep", "lcm_describe", "lcm_expand"}:
        assert (name in text) == (name not in disabled)
    assert ('An "Externalized tool output"' in text) == ("lcm_expand" not in disabled)
    assert ("externalized_ref=" in text) == ("lcm_expand" not in disabled)
    assert ("Tools: " in text) == (len(disabled) < 3)


def test_prompt_section_does_not_wait_for_stable_use_lock(plugin):
    module, register = plugin
    ctx = register(SectionContext())
    ctx.engine.on_session_start("bound", platform="cli")
    content = ctx.sections[0][1]
    locked = threading.Event()
    release = threading.Event()
    rendered = threading.Event()
    results = []

    def hold_lock():
        with ctx.engine._stable_use_lock:
            locked.set()
            release.wait(10)

    def render():
        results.append(content({"session_id": "bound"}))
        rendered.set()

    holder = threading.Thread(target=hold_lock)
    caller = threading.Thread(target=render)
    holder.start()
    try:
        assert locked.wait(1), "lock holder did not start"
        caller.start()
        assert rendered.wait(0.5), "prompt section waited on the stable-use lock"
        assert results == [module.LCM_SYSTEM_PROMPT_NOTE]
    finally:
        release.set()
        holder.join(2)
        if caller.ident is not None:
            caller.join(6)
    assert not holder.is_alive()
    assert not caller.is_alive()


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
