"""Plugin/engine identity rename (#471): hermes-lcm-x / lcm-x with a loud legacy alias."""

from pathlib import Path
import importlib.util
import json
import logging
import sys
import types

import pytest

from hermes_lcm import plugin_identity
from hermes_lcm.engine_registry import _is_usable_lcm_engine


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)


def _install_hermes_config(monkeypatch, config):
    """Serve *config* through the host's ``hermes_cli.config.load_config``."""
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []
    config_module = types.ModuleType("hermes_cli.config")
    config_module.load_config = lambda: config
    config_module.get_hermes_home = lambda: Path(sys.modules["os"].environ["HERMES_HOME"])
    hermes_cli.config = config_module
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)


def _register(module_name):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    class _Ctx:
        engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _Ctx()
    module.register(ctx)
    return module, ctx.engine


def test_identity_constants():
    assert plugin_identity.PLUGIN_NAME == "hermes-lcm-x"
    assert plugin_identity.ENGINE_NAME == "lcm-x"
    assert plugin_identity.ENGINE_NAMES == {"lcm-x", "lcm"}
    manifest = (Path(__file__).resolve().parent.parent / "plugin.yaml").read_text(encoding="utf-8")
    assert "name: hermes-lcm-x\n" in manifest


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"context": {"engine": "lcm-x"}, "plugins": {"enabled": ["hermes-lcm-x"]}},
        {"context": {"engine": "compressor"}},
    ],
)
def test_no_notice_for_current_or_unrelated_config(config):
    assert plugin_identity.identity_migration_notice(config) is None


def test_notice_names_every_change_for_the_old_config():
    notice = plugin_identity.identity_migration_notice(
        {"context": {"engine": "lcm"}, "plugins": {"enabled": ["hermes-lcm"]}}
    )
    assert notice["legacy_engine_alias_active"] is True
    assert notice["legacy_plugin_enabled"] is True
    message = notice["message"]
    assert "`context.engine: lcm-x`" in message
    assert "add `hermes-lcm-x` to `plugins.enabled`" in message
    assert "remove `hermes-lcm` from `plugins.enabled`" in message
    assert "lcm.db is untouched" in message
    # #477: the alias is fixed at register(), so the notice leads with stop/restart.
    assert message.startswith("Stop Hermes, then edit config.yaml: ")
    assert "then start Hermes again" in message
    assert "while Hermes runs makes new sessions fall back to the built-in compressor" in message
    assert notice["change"][0] == "stop Hermes" and notice["change"][-1] == "start Hermes again"


def test_notice_for_stale_enabled_entry_only_keeps_engine_name():
    notice = plugin_identity.identity_migration_notice(
        {"context": {"engine": "lcm-x"}, "plugins": {"enabled": ["hermes-lcm", "hermes-lcm-x"]}}
    )
    assert notice["legacy_engine_alias_active"] is False
    assert notice["change"] == [
        "stop Hermes",
        "remove `hermes-lcm` from `plugins.enabled` once `hermes-lcm-x` is enabled",
        "start Hermes again",
    ]


def test_registry_accepts_current_and_legacy_engine_names():
    for name in ("lcm-x", "lcm"):
        assert _is_usable_lcm_engine(types.SimpleNamespace(name=name, ingest=lambda _m: None))
    assert not _is_usable_lcm_engine(types.SimpleNamespace(name="compressor", ingest=lambda _m: None))


def test_current_config_selects_lcm_x_without_warning(monkeypatch, caplog):
    _install_hermes_config(
        monkeypatch, {"context": {"engine": "lcm-x"}, "plugins": {"enabled": ["hermes-lcm-x"]}}
    )
    with caplog.at_level(logging.INFO):
        _module, engine = _register("hermes_lcm_identity_current")

    assert engine.name == "lcm-x"
    assert engine.identity_migration is None
    assert "DEPRECATED LCM-X config" not in caplog.text
    assert "LCM plugin loaded — lossless context management active" in caplog.text
    status = engine.get_status()
    assert status["engine"] == "lcm-x"
    assert status["identity_migration"] is None


def test_legacy_engine_config_keeps_selecting_lcm_with_one_loud_warning(monkeypatch, caplog):
    _install_hermes_config(
        monkeypatch,
        {"context": {"engine": "lcm"}, "plugins": {"enabled": ["hermes-lcm", "hermes-lcm-x"]}},
    )
    with caplog.at_level(logging.INFO):
        module, engine = _register("hermes_lcm_identity_legacy")

    # Hermes selects the engine whose name equals context.engine.
    assert engine.name == "lcm"
    assert engine.clone_for_agent().name == "lcm"
    warnings = [r for r in caplog.records if "DEPRECATED LCM-X config" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "`context.engine: lcm-x`" in warnings[0].getMessage()
    assert "LCM plugin loaded — lossless context management active" in caplog.text

    status = engine.get_status()
    assert status["engine"] == "lcm-x"
    assert status["runtime_identity"]["engine_selected_as"] == "lcm"
    assert status["identity_migration"]["legacy_engine_alias_active"] is True

    # Once per process: a second registration in the same module does not re-warn.
    caplog.clear()
    ctx = types.SimpleNamespace(register_context_engine=lambda _engine: None)
    module.register(ctx)
    assert "DEPRECATED LCM-X config" not in caplog.text


def test_lcm_status_and_doctor_surface_the_migration(monkeypatch):
    _install_hermes_config(monkeypatch, {"context": {"engine": "lcm"}})
    _module, engine = _register("hermes_lcm_identity_tools")
    tools = sys.modules["hermes_lcm_identity_tools.tools"]

    status = json.loads(tools.lcm_status({}, engine=engine))
    assert status["identity_migration"]["configured_engine"] == "lcm"

    doctor = json.loads(tools.lcm_doctor({}, engine=engine))
    check = next(c for c in doctor["checks"] if c["check"] == "identity_migration")
    assert check["status"] == "warn"
    assert "`context.engine: lcm-x`" in check["detail"]["message"]


class _ForeignLcmEngine:
    """Stands in for a separate pre-rename copy's engine (different module)."""

    name = "lcm"


class _HookCtx:
    def __init__(self, existing):
        self._manager = types.SimpleNamespace(_context_engine=existing)
        self.engine = None
        self.hooks = {}

    def register_context_engine(self, engine):
        self.engine = engine

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)


def _load_module(module_name):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        module_name, str(repo_root / "__init__.py"), submodule_search_locations=[str(repo_root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_second_lcm_generation_stays_inert(caplog):
    """#471 fix round 1: a separate hermes-lcm copy loaded first must stay the only writer."""
    module = _load_module("hermes_lcm_identity_dual_guard")
    ctx = _HookCtx(_ForeignLcmEngine())
    with caplog.at_level(logging.INFO):
        module.register(ctx)

    assert ctx.engine is None
    assert ctx.hooks == {}
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "Another LCM generation is already loaded" in errors[0].getMessage()
    assert "replace `hermes-lcm` with `hermes-lcm-x` in plugins.enabled" in errors[0].getMessage()
    assert "LCM plugin loaded" not in caplog.text


@pytest.mark.parametrize("existing_name", [None, "compressor-plus"])
def test_guard_ignores_no_engine_or_non_lcm_engine(existing_name):
    module = _load_module(f"hermes_lcm_identity_guard_{existing_name}")
    existing = None if existing_name is None else types.SimpleNamespace(name=existing_name)
    ctx = _HookCtx(existing)
    module.register(ctx)
    try:
        assert ctx.engine is not None
        assert "pre_llm_call" in ctx.hooks
    finally:
        ctx.engine.shutdown()
