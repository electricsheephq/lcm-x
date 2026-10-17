import logging
from pathlib import Path
import re

from tests.test_packaging_install import _load_plugin_entrypoint_module


class _Ctx:
    def __init__(self):
        self.engine = None
        self.skill_path = None

    def register_context_engine(self, engine):
        self.engine = engine

    def register_skill(self, name, path, description=""):
        assert isinstance(path, Path)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SKILL.md not found at {p}")
        self.skill_path = p

    def register_hook(self, name, callback):
        pass


def test_register_skill_receives_skill_md_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    module = _load_plugin_entrypoint_module("hermes_lcm_issue_571_1")
    ctx = _Ctx()
    try:
        module.register(ctx)
        path = ctx.skill_path
        assert path.is_file()
        assert path.name == "SKILL.md"
        assert path.parent.name == "hermes-lcm"
    finally:
        if ctx.engine is not None:
            ctx.engine.shutdown()


def test_registered_skill_is_readable_under_host_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    module = _load_plugin_entrypoint_module("hermes_lcm_issue_571_2")
    ctx = _Ctx()
    try:
        module.register(ctx)
        text = ctx.skill_path.read_text(encoding="utf-8")
        assert text.startswith("---")
        assert "name: hermes-lcm" in text.splitlines()
    finally:
        if ctx.engine is not None:
            ctx.engine.shutdown()


def test_registered_skill_references_resolve_from_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    module = _load_plugin_entrypoint_module("hermes_lcm_issue_571_3")
    ctx = _Ctx()
    try:
        module.register(ctx)
        path = ctx.skill_path
        text = path.read_text(encoding="utf-8")
        references = set(re.findall(r"references/[A-Za-z0-9_.-]+\.md", text))
        assert references
        for token in references:
            linked_file = path.parent / token
            assert linked_file.is_file()
            assert linked_file.read_text(encoding="utf-8")
        shipped_files = sorted(p.name for p in (path.parent / "references").glob("*.md"))
        assert {
            "architecture.md",
            "configuration.md",
            "diagnostics.md",
            "recall-policy.md",
            "recall-tools.md",
            "session-lifecycle.md",
        }.issubset(shipped_files)
    finally:
        if ctx.engine is not None:
            ctx.engine.shutdown()


def test_register_skill_failure_stays_nonfatal(tmp_path, monkeypatch, caplog):
    # Guard: a broken host registration must remain non-fatal before and after #571.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    module = _load_plugin_entrypoint_module("hermes_lcm_issue_571_4")

    class _FailingCtx(_Ctx):
        def register_skill(self, name, path, description=""):
            raise FileNotFoundError("SKILL.md not found in broken install")

    ctx = _FailingCtx()
    try:
        with caplog.at_level(logging.WARNING):
            module.register(ctx)
        assert ctx.engine is not None
        assert any(
            record.levelno == logging.WARNING
            and "LCM bundled skill registration did not complete" in record.getMessage()
            for record in caplog.records
        )
    finally:
        if ctx.engine is not None:
            ctx.engine.shutdown()
