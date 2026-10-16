"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import sys
import importlib.util
from pathlib import Path

# Make the repo root importable (for agent.context_engine etc.)
repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Register the plugin directory as a proper package
plugin_dir = Path(__file__).resolve().parent.parent
pkg_name = "hermes_lcm"

if pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(plugin_dir / "__init__.py"),
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(plugin_dir)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod
    # Don't exec the module (it tries to register with ctx)
    # Just make submodules importable

    # Register each submodule
    for py_file in plugin_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"{pkg_name}.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name, str(py_file),
                submodule_search_locations=[],
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__package__ = pkg_name
            sys.modules[sub_name] = sub_mod
            setattr(mod, py_file.stem, sub_mod)
            try:
                sub_spec.loader.exec_module(sub_mod)
            except Exception:
                pass  # some modules may fail (e.g. engine needs agent)


def load_cli():
    script = Path(__file__).resolve().parents[1] / "scripts" / "lcm_longmemeval.py"
    spec = importlib.util.spec_from_file_location("lcm_longmemeval_test_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import pytest  # noqa: E402

_MAINTENANCE_SCHEDULERS = ("_ROLLUP_MAINTENANCE_SCHEDULER", "_EMBEDDING_MAINTENANCE_SCHEDULER")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_local_model: use the real locally cached FastEmbed model"
    )


@pytest.fixture(autouse=True)
def _isolate_fastembed_cache(tmp_path, request):
    """#1196: cached models must not start real inference in ordinary unit tests."""
    # Replay controls call the shared monkeypatch fixture's undo() mid-test.
    with pytest.MonkeyPatch.context() as monkeypatch:
        if request.node.get_closest_marker("real_local_model") is None:
            # Some tests require tmp_path itself to remain an empty output directory.
            cache_dir = tmp_path.with_name(tmp_path.name + "-fastembed-cache")
            cache_dir.mkdir()
            monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_dir))
        yield


def _wait_scheduler_idle(scheduler, timeout: float) -> bool:
    import time

    deadline = time.monotonic() + timeout
    with scheduler._condition:
        while (scheduler._queued_keys or scheduler._active_keys or scheduler._follow_up_key) and \
                time.monotonic() < deadline:
            scheduler._condition.wait(max(0.0, deadline - time.monotonic()))
        return not (scheduler._queued_keys or scheduler._active_keys or scheduler._follow_up_key)


@pytest.fixture(autouse=True)
def _no_background_model_prepare():
    """#1186: the local model prepare is process-wide state; a test that wants it opts in.

    Local embeddings are on by default, so every bound session would otherwise start a
    download thread that outlives the test and calls a later test's fakes.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        maintenance = sys.modules.get("hermes_lcm.embedding_maintenance")
        provider = sys.modules.get("hermes_lcm.embedding_provider")
        if provider is not None:
            monkeypatch.setattr(provider, "_FASTEMBED_MODELS", {}, raising=False)
            monkeypatch.setattr(provider, "_FASTEMBED_BUILD_LOCKS", {}, raising=False)
        if maintenance is not None:
            monkeypatch.setattr(maintenance, "_PREPARE", {})
            monkeypatch.setattr(maintenance, "_PREPARE_THREAD", None, raising=False)
            monkeypatch.setattr(maintenance, "start_local_model_prepare", lambda *_args, **_kwargs: False)
        yield


@pytest.fixture(autouse=True)
def _drain_background_maintenance(monkeypatch, _isolate_fastembed_cache, _no_background_model_prepare):
    """#1013 part B: temporal rollups are on by default, and their worker is process-wide.

    A job a test schedules must finish inside that test, while its fakes are still installed;
    otherwise it runs during a later test and calls that test's monkeypatched provider.
    """
    yield
    engine = sys.modules.get("hermes_lcm.engine")
    for name in _MAINTENANCE_SCHEDULERS:
        scheduler = getattr(engine, name, None)
        if scheduler is not None and not _wait_scheduler_idle(scheduler, timeout=30.0):
            # Fail closed: a job still running here would call a later test's fakes.
            pytest.fail(f"{name} still had background work 30 s after the test ended")
