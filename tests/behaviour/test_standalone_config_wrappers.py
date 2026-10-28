"""Plugin wrappers must use their own config without a sys.path entry."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("unrelated", [False, True])
@pytest.mark.parametrize("evicted", [False, True])
def test_wrappers_without_tree_on_path(monkeypatch, unrelated, evicted):
    name = "standalone_wrapper_plugin"
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p).resolve() != ROOT])
    monkeypatch.delitem(sys.modules, "config", raising=False)
    foreign = ModuleType("config")
    if unrelated:
        monkeypatch.setitem(sys.modules, "config", foreign)
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    if evicted:
        monkeypatch.setitem(sys.modules, name, plugin)
    spec.loader.exec_module(plugin)
    if evicted:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "true")
    monkeypatch.setenv("LCM_DISABLED_TOOLS", " lcm_recall, lcm_expand ")
    try:
        assert plugin._env_flag_enabled("LCM_ENABLE_SLASH_COMMAND") is True
        assert plugin._disabled_tool_names() == {"lcm_recall", "lcm_expand"}
        note = plugin.lcm_system_prompt_note(plugin._disabled_tool_names())
        assert "lcm_grep" in note and "lcm_recall" not in note and "lcm_expand" not in note
        monkeypatch.setenv("LCM_DISABLED_TOOLS", "lcm_grep")
        monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "flase")
        assert plugin._disabled_tool_names() == {"lcm_grep"}
        assert plugin._env_flag_enabled("LCM_ENABLE_SLASH_COMMAND", True) is False
        assert sys.modules.get("config") is (foreign if unrelated else None)
    finally:
        for key in list(sys.modules):
            if key.startswith(name + "."):
                sys.modules.pop(key)
