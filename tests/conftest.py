"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import hashlib
import sys
import importlib.util
from pathlib import Path

from tests.phase0_hygiene import product_env_keys

_IMPORT_FAILURES = {}
pytest_plugins = ["tests.diag_1753_capture"]
# Measured with/without the CI host stub: only engine needs agent.context_engine.
# Allow that missing host dependency, never arbitrary errors in engine itself.
_OPTIONAL_HOST_IMPORTS = {"hermes_lcm.engine": {"agent", "agent.context_engine"}}

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
            # A plain module spec: its parent is the package, matching __package__ below (a
            # submodule_search_locations spec made each module a package and warned on every relative import).
            sub_spec = importlib.util.spec_from_file_location(sub_name, str(py_file))
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__package__ = pkg_name
            sys.modules[sub_name] = sub_mod
            setattr(mod, py_file.stem, sub_mod)
            try:
                sub_spec.loader.exec_module(sub_mod)
            except Exception as exc:
                _IMPORT_FAILURES[sub_name] = exc
                sys.modules.pop(sub_name, None)
                delattr(mod, py_file.stem)


def load_cli():
    script = Path(__file__).resolve().parents[1] / "scripts" / "lcm_longmemeval.py"
    spec = importlib.util.spec_from_file_location("lcm_longmemeval_test_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import pytest  # noqa: E402


@pytest.fixture(params=[False, pytest.param(True, marks=pytest.mark.latency, id="latency")], ids=["correctness", "latency"])
def latency_check(request):
    """Keep correctness in PR CI; enable elapsed bounds only in the nightly cell."""
    return request.param


def pytest_report_header(config):
    return [
        f"LCM import failure: {name}: {type(exc).__name__}: {exc}"
        for name, exc in sorted(_IMPORT_FAILURES.items())
    ]


def pytest_sessionstart(session):
    unexpected = [
        name for name, exc in _IMPORT_FAILURES.items()
        if not (
            isinstance(exc, ModuleNotFoundError)
            and exc.name in _OPTIONAL_HOST_IMPORTS.get(name, ())
        )
    ]
    if unexpected:
        raise pytest.UsageError("Unexpected plugin import failures:\n" + "\n".join(pytest_report_header(None)))


@pytest.fixture(autouse=True)
def _isolate_product_environment(monkeypatch, request, tmp_path_factory):
    for key in product_env_keys(plugin_dir):
        monkeypatch.delenv(key, raising=False)
    # A unique, not-yet-created home per test; requesting tmp_path here would create ~7,600 directories.
    test_id = hashlib.sha1(request.node.nodeid.encode()).hexdigest()[:16]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path_factory.getbasetemp() / "hermes-home" / test_id))


_MAINTENANCE_SCHEDULERS = ("_ROLLUP_MAINTENANCE_SCHEDULER", "_EMBEDDING_MAINTENANCE_SCHEDULER")


def _wait_scheduler_idle(scheduler, timeout: float) -> bool:
    import time

    deadline = time.monotonic() + timeout
    with scheduler._condition:
        # The same background work the scheduler's own drain() waits for: deferred jobs are promoted
        # once the active one finishes. Exclusive keys belong to the test's own synchronous rebuilds.
        def busy():
            return (scheduler._queued_keys or scheduler._deferred_jobs or scheduler._active_keys
                    or scheduler._follow_up_key)

        while busy() and time.monotonic() < deadline:
            scheduler._condition.wait(max(0.0, deadline - time.monotonic()))
        return not busy()


@pytest.fixture(autouse=True)
def _isolate_embedding_process_state(request):
    """Keep process caches and model preparation inside the test that owns them."""
    if "tmp_path" in request.fixturenames:
        # Set tmp_path up first so it is torn down last: tmp_path_retention_policy=failed deletes a
        # passing test's directory at its teardown, which must follow the model-preparation and
        # background-maintenance drains below, not precede them.
        request.getfixturevalue("tmp_path")
    with pytest.MonkeyPatch.context() as patch:
        provider = sys.modules.get("hermes_lcm.embedding_provider")
        maintenance = sys.modules.get("hermes_lcm.embedding_maintenance")
        engine = sys.modules.get("hermes_lcm.engine")
        retrieval = sys.modules.get("hermes_lcm.retrieval_core")
        # These caches are keyed by database path. tmp_path_retention_policy=failed deletes a
        # passing test's tmp_path, so a later test can be handed the same path again.
        if engine is not None:
            patch.setattr(engine, "_EMBEDDING_SCHEMA_READY", set())
        if retrieval is not None:
            retrieval._reset_vector_store_pool()
        if provider is not None:
            patch.setattr(provider, "_FASTEMBED_MODELS", {})
            patch.setattr(provider, "_FASTEMBED_BUILD_LOCKS", {})
        if maintenance is not None:
            patch.setattr(maintenance, "_PREPARE", {})
            patch.setattr(maintenance, "_PREPARE_RETRY", {}, raising=False)
            patch.setattr(maintenance, "_PREPARE_READY", {})
            patch.setattr(maintenance, "_PREPARE_THREAD", None)
            if request.node.path.name != "test_issue_1586_local_precursor.py":
                patch.setattr(maintenance, "start_local_model_prepare", lambda *_a, **_k: False)
        yield
        thread = getattr(maintenance, "_PREPARE_THREAD", None)
        if thread is not None:
            thread.join(5)
            assert not thread.is_alive(), "model preparation outlived its test"


@pytest.fixture(autouse=True)
def _drain_background_maintenance(monkeypatch, _isolate_embedding_process_state):
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
