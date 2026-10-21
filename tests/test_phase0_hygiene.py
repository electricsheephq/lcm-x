"""Acceptance checks for import reporting and per-test environment isolation."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_lcm.config import ENV_FIELD_SPECS
from tests.phase0_hygiene import product_env_keys

ROOT = Path(__file__).resolve().parents[1]
HARNESS_KEYS = (
    "LCM_REAL_HERMES_SRC", "LCM_REAL_HERMES_PYTHON", "LCM_REAL_HERMES_LONG",
    "LCM_DIFFERENTIAL", "LCM_DIFFERENTIAL_ENABLED",
)
TRACE_HARNESSES = ("assembly", "compress", "reconcile")


@pytest.mark.parametrize("name", TRACE_HARNESSES)
def test_trace_regeneration_isolates_settings_from_interpreter_start(tmp_path, name):
    destination = tmp_path / "trace.json"
    env = dict(os.environ)
    for key in product_env_keys(ROOT):
        env.pop(key, None)
    env.update(LCM_IDENTITY_ANCHOR="false", LCM_FRESH_TAIL_COUNT="9999",
               LCM_DIFFERENTIAL="1", LCM_TRACE_UNKNOWN_SETTING="inherited")
    code = f"""
import importlib, json, os, sys
from pathlib import Path
harness = importlib.import_module('tests.behaviour.{name}_trace_harness')
expected = json.loads(harness.GOLDEN.read_text())
destination = Path({str(destination)!r})
harness.GOLDEN = destination
sys.argv = ['trace', '--regenerate'] if {name!r} == 'compress' else [
    'trace', '--repo', {str(ROOT)!r}, '--out', str(destination)]
harness.main()
assert json.loads(destination.read_text()) == expected, 'product settings changed trace'
assert os.environ.get('LCM_DIFFERENTIAL') == '1', 'harness control removed'
assert os.environ.get('LCM_TRACE_UNKNOWN_SETTING') == 'inherited', 'unknown key removed'
"""
    result = subprocess.run(
        ["nice", "-n", "10", sys.executable, "-c", code],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", TRACE_HARNESSES)
@pytest.mark.parametrize("raises", [False, True], ids=["return", "raise"])
def test_trace_regeneration_restores_environment(tmp_path, monkeypatch, name, raises):
    harness = importlib.import_module(f"tests.behaviour.{name}_trace_harness")
    keys = product_env_keys(ROOT)
    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "9999")
    monkeypatch.setenv("LCM_DIFFERENTIAL", "1")
    monkeypatch.setenv("LCM_TRACE_UNKNOWN_SETTING", "inherited")
    monkeypatch.setenv("TRACE_EXISTING_SETTING", "before")
    monkeypatch.delenv("TRACE_ADDED_SETTING", raising=False)
    before = dict(os.environ)
    destination = tmp_path / "trace.json"
    monkeypatch.setattr(harness, "GOLDEN", destination)
    monkeypatch.setattr(harness, "SCENARIOS", harness.SCENARIOS[:1])
    args = ["trace", "--regenerate"] if name == "compress" else [
        "trace", "--repo", str(ROOT), "--out", str(destination)]
    monkeypatch.setattr(sys, "argv", args)

    def observe_clean(case, root):
        assert not keys & os.environ.keys()
        control = os.environ.get("LCM_DIFFERENTIAL")
        unknown = os.environ.get("LCM_TRACE_UNKNOWN_SETTING")
        assert control == "1"
        assert unknown == "inherited"
        os.environ["TRACE_ADDED_SETTING"] = "during"
        del os.environ["TRACE_EXISTING_SETTING"]
        if raises:
            raise RuntimeError("trace observer failed")
        return ["clean settings"]

    monkeypatch.setattr(harness, "observe", observe_clean)
    if raises:
        with pytest.raises(RuntimeError, match="trace observer failed"):
            harness.main()
    else:
        harness.main()
        assert json.loads(destination.read_text()) == {harness.SCENARIOS[0].name: ["clean settings"]}
    restored = dict(os.environ) == before
    assert restored, "trace regeneration did not restore the exact environment"


@pytest.mark.parametrize("name", ["assembly", "reconcile"])
def test_trace_regeneration_scrubs_selected_tree_settings(tmp_path, monkeypatch, name):
    harness = importlib.import_module(f"tests.behaviour.{name}_trace_harness")
    key = "LCM_SELECTED_TREE_ONLY"
    assert key not in product_env_keys(ROOT)
    selected = tmp_path / "selected"
    selected.mkdir()
    module = selected / "reconcile.py"
    module.write_text(f"import os\nSETTING = os.environ.get({key!r})\n", encoding="utf-8")
    monkeypatch.setenv(key, "inherited")
    before = dict(os.environ)
    destination = tmp_path / "trace.json"
    monkeypatch.setattr(harness, "SCENARIOS", harness.SCENARIOS[:1])
    monkeypatch.setattr(sys, "argv", [
        "trace", "--repo", str(selected), "--out", str(destination)])
    if name == "reconcile":
        monkeypatch.setattr(harness.reconcile, "__file__", str(module))

    def observe_clean(case, root):
        assert key not in os.environ, "selected-tree product setting survived scrub"
        return ["clean settings"]

    monkeypatch.setattr(harness, "observe", observe_clean)
    harness.main()
    assert json.loads(destination.read_text()) == {harness.SCENARIOS[0].name: ["clean settings"]}
    assert dict(os.environ) == before, "trace regeneration did not restore the exact environment"


def test_product_env_derivation_covers_config_and_root_reads():
    keys = product_env_keys(ROOT)
    assert {spec.env_key for spec in ENV_FIELD_SPECS} <= keys
    assert {
        "LCM_LEAF_CHUNK_TOKENS", "LCM_HOST_MESSAGE_UID", "LCM_MODEL_THRESHOLDS",
        "LCM_DISABLED_TOOLS", "LCM_HERMES_BASE_DIR", "LCM_IDENTITY_ANCHOR",
        "LCM_ABSOLUTE_THRESHOLD_TOKENS", "LCM_EMBEDDING_BACKFILL_BATCH_SIZE",
        "LCM_SENSITIVE_PATTERNS", "LCM_SUMMARY_FALLBACK_MODELS",
    } <= keys
    assert not set(HARNESS_KEYS) & keys


def _run_pytest(tmp_path, source, *, injection="", env=None):
    probe = tmp_path / "test_probe.py"
    probe.write_text(source, encoding="utf-8")
    code = (
        "import sys, pytest\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        + injection
        + "\nraise SystemExit(pytest.main(['-p', 'tests.conftest', '-p', 'no:cacheprovider', "
        + f"'--rootdir', {str(tmp_path)!r}, '--confcutdir', {str(tmp_path)!r}, '-v', {str(probe)!r}]))\n"
    )
    return subprocess.run(
        ["nice", "-n", "19", sys.executable, "-c", code],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,  # correctness, not latency: loaded hosts are slow
    )


@pytest.mark.parametrize("module,error,allowed", [
    ("engine", "ModuleNotFoundError('missing host', name='agent')", True),
    ("engine", "ModuleNotFoundError('missing host', name='agent.context_engine')", True),
    ("engine", "ModuleNotFoundError('missing product dependency', name='broken_dependency')", False),
    ("engine", "ValueError('broken engine')", False),
    ("presets", "RuntimeError('broken presets')", False),
])
def test_bootstrap_reports_failures_and_rejects_unexpected_imports(tmp_path, module, error, allowed):
    injection = f"""
import importlib.machinery
original = importlib.machinery.SourceFileLoader.exec_module
def injected(loader, mod):
    if mod.__name__ == 'hermes_lcm.{module}':
        raise {error}
    return original(loader, mod)
importlib.machinery.SourceFileLoader.exec_module = injected
"""
    result = _run_pytest(tmp_path, "def test_probe(): pass\n", injection=injection)
    output = result.stdout + result.stderr
    assert result.returncode == (0 if allowed else pytest.ExitCode.USAGE_ERROR), output
    assert f"LCM import failure: hermes_lcm.{module}:" in output
    assert error.split("(")[0] in output
    if allowed:
        assert "missing host" in output
        assert "1 passed" in output
    else:
        assert "Unexpected plugin import failures" in output
        assert "1 passed" not in output


def test_ambient_product_env_is_removed_but_harness_controls_and_test_overrides_survive(tmp_path):
    keys = sorted(product_env_keys(ROOT))
    env = dict(os.environ)
    env.update(dict.fromkeys(keys, "ambient-product-sentinel"))
    env.update(dict.fromkeys(HARNESS_KEYS, "harness-sentinel"))
    ambient_home = tmp_path / "ambient-home"
    env["HERMES_HOME"] = str(ambient_home)
    result = _run_pytest(tmp_path, f"""
import os
from pathlib import Path

HOMES = []

def _home(tmp_path_factory):
    home = Path(os.environ['HERMES_HOME'])
    assert home.parent == tmp_path_factory.getbasetemp() / 'hermes-home'
    HOMES.append(home)
    return home

def test_first(monkeypatch, tmp_path_factory):
    assert not set({keys!r}) & os.environ.keys()
    assert all(os.environ[k] == 'harness-sentinel' for k in {HARNESS_KEYS!r})
    _home(tmp_path_factory)
    monkeypatch.setenv('LCM_LEAF_CHUNK_TOKENS', '1234')
    from hermes_lcm.config import LCMConfig
    assert LCMConfig.from_env().leaf_chunk_tokens == 1234

def test_second(tmp_path_factory):
    assert not set({keys!r}) & os.environ.keys()
    _home(tmp_path_factory)
    assert HOMES[0] != HOMES[1]  # each test gets its own home
""", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert not ambient_home.exists()
