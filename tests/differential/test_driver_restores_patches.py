"""Differential replay must not leak its deterministic module bindings."""
from importlib import import_module
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _package_modules() -> dict[str, ModuleType]:
    return {name: module for name, module in sys.modules.items()
            if name == "hermes_lcm" or name.startswith("hermes_lcm.")}


@pytest.fixture(autouse=True)
def restore_package_imports():
    """load_tree rebinds the hermes_lcm package; later tests resolve dotted paths through it."""
    saved = _package_modules()
    yield
    for name in set(_package_modules()) - set(saved):
        del sys.modules[name]
    sys.modules.update(saved)


def bindings() -> list[tuple[ModuleType, str, Any]]:
    result = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("hermes_lcm.") or module is None:
            continue
        for attr in ("time", "_call_llm_for_summary", "_host_estimate",
                     "summarize_with_escalation", "_encoder_ready", "_encoder", "_encoder_generation"):
            if hasattr(module, attr):
                result.append((module, attr, getattr(module, attr)))
    return result


@pytest.mark.parametrize("exit_path", ["success", "early-return", "setup-error", "case-error"])
def test_run_case_restores_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_path: str,
) -> None:
    driver = import_module("tests.differential.driver")
    modules, missing = driver.load_tree(Path(__file__).resolve().parents[2])
    # Snapshot after loading, so modules load_tree imports for the first time are checked too.
    originals = bindings()
    assert originals
    engines = []
    engine_class = modules["engine"].LCMEngine
    start = engine_class.on_session_start

    def capture_engine(self, *args, **kwargs):
        engines.append(self)
        return start(self, *args, **kwargs)

    monkeypatch.setattr(engine_class, "on_session_start", capture_engine)
    # Keep a red regression from contaminating the tests following it.
    for module, attr, original in originals:
        monkeypatch.setattr(module, attr, original)
    case: dict[str, Any] = dict(
        id="binding-restoration", layer="projection", summarizer="healthy", window=32768,
        limit=1000, overrides={}, fixture_summarizer=True,
        rows=[dict(message=dict(role="user", content="⟦r0⟧ request"),
                   stored={}, is_stored=False, ext=False)],
    )
    if exit_path == "early-return":
        case["overrides"] = dict(unavailable_setting=True)
    elif exit_path == "setup-error":
        del case["overrides"]
    elif exit_path == "case-error":
        del case["rows"]
    if exit_path.endswith("error"):
        key = "overrides" if exit_path == "setup-error" else "rows"
        with pytest.raises(KeyError, match=key):
            driver.run_case(case, modules, tmp_path, missing)
    else:
        result = driver.run_case(case, modules, tmp_path, missing)
        if exit_path == "early-return":
            assert "unavailable_setting" in result["error"]
        else:
            assert "error" not in result
            assert result["out"] == [case["rows"][0]["message"]]
    leaked = [f"{module.__name__}.{attr}" for module, attr, original in originals
              if getattr(module, attr) is not original]
    assert leaked == []
    for engine in engines:
        assert (engine._summarize_leaf_chunk_with_rescue.__func__
                is engine_class._summarize_leaf_chunk_with_rescue)


@pytest.mark.parametrize("generator", ["generated_cases", "projection_cases"])
def test_case_generation_restores_counter(
    monkeypatch: pytest.MonkeyPatch, generator: str,
) -> None:
    cases = import_module("tests.differential.cases")
    # Materialize the counter module first, then check its original bindings.
    getattr(cases, generator)(1, 1227)
    tokens = import_module("differential_counter.tokens")
    encoder = object()
    monkeypatch.setattr(tokens, "_encoder_ready", False)
    monkeypatch.setattr(tokens, "_encoder", encoder)
    assert len(getattr(cases, generator)(1, 1227)) == 1
    assert tokens._encoder_ready is False
    assert tokens._encoder is encoder


def _install_encoder(tokens: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    # A byte encoder makes exact counts differ from fallback without loading a tokenizer.
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    monkeypatch.setattr(tokens, "_encoder", SimpleNamespace(encode=lambda text: list(text.encode())))
    monkeypatch.setattr(tokens, "_count_tokens_cached",
                        lru_cache(maxsize=2048)(tokens._count_tokens_cached.__wrapped__))


def test_run_case_retires_scoped_token_estimates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = import_module("tests.differential.driver")
    modules, missing = driver.load_tree(Path(__file__).resolve().parents[2])
    tokens = modules["tokens"]
    _install_encoder(tokens, monkeypatch)
    text = "driver scoped token estimate"

    def count_in_scope(*args):
        assert tokens._encoder is None
        assert tokens.count_tokens(text) != len(text.encode())

    monkeypatch.setattr(driver, "_run_case", count_in_scope)
    driver.run_case({}, modules, tmp_path, missing)
    assert tokens.count_tokens(text) == tokens._count_tokens_core(text) == len(text.encode())


def test_token_counter_retires_scoped_token_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = import_module("tests.differential.cases")
    with cases.token_counter():
        pass
    tokens = import_module("differential_counter.tokens")
    _install_encoder(tokens, monkeypatch)
    text = "generator scoped token estimate"
    with cases.token_counter() as scoped:
        assert scoped._encoder is None
        assert scoped.count_tokens(text) != len(text.encode())
    assert tokens.count_tokens(text) == tokens._count_tokens_core(text) == len(text.encode())
