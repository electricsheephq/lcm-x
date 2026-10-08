"""#998: both LCM notes name lcm_recall, the cross-session recall tool, first."""

import pytest

from hermes_lcm.engine import LCMEngine
from tests.test_packaging_install import (
    _ensure_agent_context_engine_importable,
    _load_plugin_entrypoint_module,
)

TOOLS = ("lcm_recall", "lcm_grep", "lcm_describe", "lcm_expand")
CLAUSES = {
    "lcm_recall": "lcm_recall finds facts from any earlier session (ranked, cited)",
    "lcm_grep": "lcm_grep searches text (this session by default)",
    "lcm_describe": "lcm_describe inspects the summary DAG",
    "lcm_expand": "lcm_expand recovers details",
}
RECOVERY = (
    'An "Externalized tool output" stub ending in ref=R means the full output is stored; '
    'lcm_expand(externalized_ref="R") returns it.'
)


@pytest.fixture
def module(monkeypatch):
    _ensure_agent_context_engine_importable(monkeypatch)
    return _load_plugin_entrypoint_module("hermes_lcm_issue_998")


def test_nothing_disabled_note_equals_constant_and_names_recall_first(module):
    note = module.lcm_system_prompt_note(set())
    assert note == module.LCM_SYSTEM_PROMPT_NOTE
    assert "Tools: " + ", ".join(CLAUSES[name] for name in TOOLS) + ". " in note
    assert note.index("lcm_recall") < note.index("lcm_grep")


@pytest.mark.parametrize("name", TOOLS)
def test_single_disabled_tool_removes_exactly_its_clause(module, name):
    note = module.lcm_system_prompt_note({name})
    full_tools = "Tools: " + ", ".join(CLAUSES[n] for n in TOOLS) + ". "
    kept_tools = "Tools: " + ", ".join(CLAUSES[n] for n in TOOLS if n != name) + ". "
    expected = module.LCM_SYSTEM_PROMPT_NOTE.replace(full_tools, kept_tools, 1)
    if name == "lcm_expand":
        expected = expected.replace(RECOVERY, "", 1).rstrip()
    assert note == expected
    assert name not in note


def test_all_disabled_drops_tools_clause_and_recovery_sentence(module):
    note = module.lcm_system_prompt_note(set(TOOLS))
    assert "Tools:" not in note
    assert "externalized_ref" not in note
    assert note == module.LCM_SYSTEM_PROMPT_NOTE.split(" Tools: ", 1)[0]


def test_in_context_note_names_lcm_recall_first():
    note = LCMEngine._append_lcm_note_to_content("")
    assert "Tools: lcm_recall find facts from any earlier session, lcm_grep search, " in note
    assert note.index("lcm_recall") < note.index("lcm_grep")
