"""Profile config isolation across registration, turns and unscoped workers (#1603).

The host stub models secret_scope's context-local accessors and home-only plugin
discovery. Its get_secret deliberately raises: the plugin must not call it.
"""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def host(monkeypatch):
    home_var = ContextVar("home", default=None)
    scope_var = ContextVar("scope", default=None)
    bound_var = ContextVar("bound_home", default=None)
    host = SimpleNamespace(routed=True, builds=[])
    scope = ModuleType("agent.secret_scope")
    scope.set_multiplex_active = lambda active: setattr(host, "routed", active)
    scope.set_secret_scope = scope_var.set
    scope.reset_secret_scope = scope_var.reset
    scope.get_secret_str = lambda name, default="": raising_get_secret(name, default)
    scope.load_env_file = lambda path: build(path.parent)
    scope.current_secret_scope = scope_var.get
    scope.current_secret_scope_home = bound_var.get
    scope.serves_routed_profile = lambda: host.routed
    scope.is_multiplex_active = lambda: host.routed

    def build(home):
        host.builds.append(home)
        path = home / ".env"
        return dict(line.split("=", 1) for line in path.read_text().splitlines()) if path.exists() else {}

    def raising_get_secret(*args):
        raise RuntimeError("UnscopedSecretError: must never reach this accessor")

    scope.build_profile_secret_scope = build
    scope.get_secret = raising_get_secret
    monkeypatch.setitem(sys.modules, "agent.secret_scope", scope)
    import agent
    monkeypatch.setattr(agent, "secret_scope", scope, raising=False)
    cli = ModuleType("hermes_cli")
    cli.__path__ = []
    config = ModuleType("hermes_cli.config")
    config.get_hermes_home = lambda: home_var.get() or Path(os.environ["HERMES_HOME"])
    config.load_config = lambda: {"context": {"engine": "lcm-x"}}
    cli.config = config
    monkeypatch.setitem(sys.modules, "hermes_cli", cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config)
    constants = ModuleType("hermes_constants")
    constants.get_hermes_home_override = home_var.get
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)

    @contextmanager
    def bind(home, mapping=None, *, bound_home=None):
        tokens = (home_var.set(home), scope_var.set(mapping),
                  bound_var.set(str(bound_home or home) if mapping is not None else None))
        try:
            yield
        finally:
            home_var.reset(tokens[0])
            scope_var.reset(tokens[1])
            bound_var.reset(tokens[2])

    host.bind = bind
    return host


@pytest.fixture
def load_profile():
    names = []
    engines = []

    def load(module=None, *, register=True):
        if module is not None:
            module.register(SimpleNamespace(register_context_engine=engines.append))
            return module, engines[-1]
        name = f"issue1603_profile_{len(names)}"
        names.append(name)
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py",
                                                    submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if register:
            module.register(SimpleNamespace(register_context_engine=engines.append))
        return module, engines[-1] if register else None

    yield load
    for engine in engines:
        engine.shutdown()
    for key in list(sys.modules):
        if any(key == name or key.startswith(name + ".") for name in names):
            sys.modules.pop(key, None)


@pytest.mark.parametrize("discovery_scope", [True, False])
def test_two_homes_keep_registration_values(host, load_profile, tmp_path, monkeypatch, discovery_scope):
    """Different profiles stay isolated through model switches and worker reads."""
    monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "120000")
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture-launch")
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    profiles = []
    for index, absolute, ratio, embeddings in [(0, 250000, 0.61, True), (1, 160000, 0.82, False)]:
        home = tmp_path / f"profile-{index}"
        home.mkdir()
        mapping = {"LCM_ABSOLUTE_THRESHOLD_TOKENS": str(absolute),
                   "LCM_CONTEXT_THRESHOLD": str(ratio),
                   "LCM_EMBEDDINGS_ENABLED": str(embeddings).lower(),
                   "VOYAGE_API_KEY": f"fixture-routed-{index}",
                   "LCM_EMBEDDING_PROVIDER": "voyage", "LCM_EMBEDDING_MODEL": "voyage-4"}
        (home / ".env").write_text("\n".join(f"{key}={value}" for key, value in mapping.items()))
        with host.bind(home, mapping if discovery_scope else None):
            module, engine = load_profile()
        profiles.append((home, module, engine, mapping, absolute, ratio, embeddings))

    monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "99000")
    for home, module, engine, mapping, absolute, ratio, embeddings in profiles:
        assert engine._hermes_home == str(home)
        assert engine._config.context_threshold == ratio
        assert engine._config.embeddings_enabled is embeddings
        # A routed miss must retain the default, not the launch profile's true.
        assert engine._config.preanswer_evidence_enabled is False
        mapping["LCM_ABSOLUTE_THRESHOLD_TOKENS"] = "1"  # registration copied it
        config = sys.modules[module.__name__ + ".config"]
        provider_module = sys.modules[module.__name__ + ".embedding_provider"]

        def later():
            assert config._hermes_config_path() == home / "config.yaml"
            assert config.lcm_env_get("VOYAGE_API_KEY") == f"fixture-routed-{home.name[-1]}"
            assert config.lcm_env_get("LCM_PREANSWER_EVIDENCE_ENABLED", "missing") == "missing"
            for window in (262144, 500000, 1000000):
                engine._set_context_length(window, source="issue1603", model="fixture", provider="custom")
                assert engine.threshold_tokens == absolute
                assert engine.context_threshold == ratio
            provider = provider_module.resolve_provider(engine._config)
            headers_seen = []

            def transport(*args, **kwargs):
                headers_seen.append(kwargs["headers"]["Authorization"])
                return provider_module.HttpResponse(200, {}, json.dumps({"data": [{"embedding": [0.1, 0.2]}]}).encode())

            provider._transport = transport
            provider.embed_query("fixture query")
            assert headers_seen == [f"Bearer fixture-routed-{home.name[-1]}"]

        # Home-only workers may reuse only the pin for their resolved home.
        def worker():
            with host.bind(home):
                later()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker).result()
        with host.bind(home):
            later()


def test_yaml_and_unscoped_misses(host, load_profile, tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture-launch")
    monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "120000")
    home = tmp_path / "yaml-home"
    home.mkdir()
    (home / "config.yaml").write_text("lcm:\n  context_threshold: 0.67\n")
    with host.bind(home):
        module, engine = load_profile()
    assert engine._config.context_threshold == 0.67
    config = sys.modules[module.__name__ + ".config"]
    assert config.lcm_env_get("VOYAGE_API_KEY", "missing") == "missing"
    engine._set_context_length(200000, source="issue1603", model="fixture", provider="custom")
    assert engine.threshold_tokens == 134000


def test_home_only_discovery_rejects_foreign_bound_mapping(host, load_profile, tmp_path):
    home = tmp_path / "wanted"
    home.mkdir()
    (home / ".env").write_text("LCM_CONTEXT_THRESHOLD=0.69\n")
    with host.bind(home, {"LCM_CONTEXT_THRESHOLD": "0.12"}, bound_home=tmp_path / "foreign"):
        _, engine = load_profile()
    assert engine._config.context_threshold == 0.69


def test_scope_failure_uses_defaults(host, load_profile, tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture-launch")

    def fail(*args):
        raise RuntimeError("host scope unavailable")

    monkeypatch.setattr(sys.modules["agent.secret_scope"], "build_profile_secret_scope", fail)
    with host.bind(tmp_path):
        module, engine = load_profile()
    assert engine._config.context_threshold == 0.75
    assert sys.modules[module.__name__ + ".config"].lcm_env_get("VOYAGE_API_KEY", "missing") == "missing"


@pytest.mark.parametrize("api", ["full", "floor", "absent", "nonrouting_failure"])
def test_no_scope_snapshot_matches_main(host, load_profile, monkeypatch, tmp_path, api):
    """All dataclass fields equal the main d411e404 fixture, not selected values."""
    fixture = json.loads((ROOT / "tests/fixtures/issue_1603_single_profile.json").read_text())
    for key, value in fixture["env"].items():
        monkeypatch.setenv(key, value)
    # Config-only: the artificial containment base is deliberately not a real
    # home. Loading an engine here would conflate config parsing with storage.
    host.routed = False
    if api == "floor":
        _floor_api(monkeypatch)
    if api == "nonrouting_failure":
        def fail():
            raise RuntimeError("host predicate unavailable")
        monkeypatch.setattr(sys.modules["agent.secret_scope"], "serves_routed_profile", fail)
    if api == "absent":
        monkeypatch.setitem(sys.modules, "agent.secret_scope", None)
        import agent
        monkeypatch.delattr(agent, "secret_scope")
    module, _ = load_profile(register=False)
    config = importlib.import_module(module.__name__ + ".config")
    with host.bind(tmp_path):
        assert asdict(config.LCMConfig.from_env()) == fixture["expected"]
        for key in ("VOYAGE_API_KEY", "SILICONFLOW_API_KEY", "LCM_EMBEDDING_API_KEY"):
            assert config.lcm_env_get(key) == fixture["env"][key]
        command = importlib.import_module(module.__name__ + ".command")
        assert command._embedding_backfill_batch_size() == 23
        assert command._embedding_backfill_lease_ttl_s() == 100
        assert command._embedding_backfill_heartbeat_s() == 10
        assert command._embedding_backfill_budget_s() == 30
        bootstrap = importlib.import_module(module.__name__ + ".db_bootstrap")
        assert bootstrap._integrity_check_interval_hours() == 24
        assert config.host_message_uid_mode() == "off"
        assert config._disabled_tool_names() == {"lcm_expand", "lcm_grep"}
        diagnostics = importlib.import_module(module.__name__ + ".diagnostics")
        base = Path("fixture-base").resolve()
        assert diagnostics._enforce_state_db_containment(base / "state.db", description="fixture") == base / "state.db"
        with pytest.raises(ValueError):
            diagnostics._enforce_state_db_containment(base.parent / "outside.db", description="fixture")
        # Exercise the actual engine setter, including live edits, outside the
        # artificial containment/config fixture's unrelated startup settings.
        monkeypatch.delenv("LCM_HERMES_BASE_DIR")
        engine_module = importlib.import_module(module.__name__ + ".engine")
        engine = engine_module.LCMEngine(config=config.LCMConfig(
            database_path=str(tmp_path / "parity.db"), temporal_rollups_enabled=False), hermes_home=str(tmp_path))
        try:
            for value in ("250000", "130000"):
                monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", value)
                engine._set_context_length(1000000, source="parity", model="fixture", provider="custom")
                assert engine.threshold_tokens == int(value)
        finally:
            engine.shutdown()


def _floor_api(monkeypatch):
    """Exactly the public functions shipped by floor-0.21.3 secret_scope.py."""
    scope = sys.modules["agent.secret_scope"]
    monkeypatch.delattr(scope, "serves_routed_profile")
    monkeypatch.delattr(scope, "current_secret_scope_home")
    assert {name for name in vars(scope) if not name.startswith("_")} == {
        "set_multiplex_active", "is_multiplex_active", "set_secret_scope",
        "reset_secret_scope", "current_secret_scope", "get_secret", "get_secret_str",
        "load_env_file", "build_profile_secret_scope",
    }


@pytest.mark.parametrize("api", ["full", "floor"])
def test_one_module_two_homes_and_clones(host, load_profile, tmp_path, monkeypatch, api):
    if api == "floor":
        _floor_api(monkeypatch)
    module = None
    profiles = []
    for index in range(2):
        home = tmp_path / str(index)
        mapping = {"LCM_CONTEXT_THRESHOLD": str(0.5 + index / 10),
                   "LCM_ABSOLUTE_THRESHOLD_TOKENS": str(150000 + index * 50000),
                   "VOYAGE_API_KEY": f"fixture-key-{index}"}
        with host.bind(home, mapping):
            module, engine = load_profile(module)
            config = importlib.import_module(module.__name__ + ".config")
            assert engine._config.context_threshold == 0.5 + index / 10
            assert config.lcm_env_get("VOYAGE_API_KEY") == mapping["VOYAGE_API_KEY"]
        profiles.append((home, engine, dict(mapping)))
        mapping["VOYAGE_API_KEY"] = "changed-after-registration"
    for home, engine, mapping in profiles:
        # No bound scope; only a pin with this exact home can be used.
        with host.bind(home):
            assert config.lcm_env_get("VOYAGE_API_KEY") == mapping["VOYAGE_API_KEY"]
            assert config._hermes_config_path() == home / "config.yaml"
            clone = engine.clone_for_agent()
            try:
                clone._set_context_length(1000000, source="clone", model="fixture", provider="custom")
                assert clone.context_threshold == float(mapping["LCM_CONTEXT_THRESHOLD"])
                assert clone.threshold_tokens == int(mapping["LCM_ABSOLUTE_THRESHOLD_TOKENS"])
                assert config.lcm_env_get("VOYAGE_API_KEY") == mapping["VOYAGE_API_KEY"]
            finally:
                clone.shutdown()
    with host.bind(tmp_path / "unknown"):
        assert config.lcm_env_get("VOYAGE_API_KEY") is None
        assert config.lcm_env_get("LCM_CONTEXT_THRESHOLD") is None


@pytest.mark.parametrize("composition", ["gateway_files", "serve_launch_overlay"])
def test_launch_profile_multiplex_composition(host, load_profile, tmp_path, monkeypatch, composition):
    launch = tmp_path / "launch"
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.44")
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture-injected")
    # gateway._profile_runtime_scope: file sources only. hermes serve's
    # launch_secret_scope: captured env, then files override it.
    mapping = {"LCM_CONTEXT_THRESHOLD": "0.66"}
    if composition == "serve_launch_overlay":
        mapping = {"VOYAGE_API_KEY": "fixture-injected", **mapping}
    with host.bind(tmp_path / "other", {"VOYAGE_API_KEY": "fixture-other"}):
        module, _ = load_profile()
    with host.bind(launch, mapping):
        _, engine = load_profile(module)
    config = importlib.import_module(module.__name__ + ".config")
    with host.bind(launch):
        assert engine._config.context_threshold == 0.66
        assert config.lcm_env_get("VOYAGE_API_KEY") == mapping.get("VOYAGE_API_KEY")
        assert config.lcm_env_get("LCM_CONTEXT_THRESHOLD") == "0.66"


def test_routed_profile_keeps_process_containment_root(host, load_profile, tmp_path, monkeypatch):
    base = (tmp_path / "contained").resolve()
    monkeypatch.setenv("LCM_HERMES_BASE_DIR", str(base))
    home = base / "routed"
    home.mkdir(parents=True)
    for mapping in ({}, {"LCM_HERMES_BASE_DIR": "/"}):
        with host.bind(home, mapping):
            module, _ = load_profile(register=False)
            diagnostics = importlib.import_module(module.__name__ + ".diagnostics")
            externalize = importlib.import_module(module.__name__ + ".externalize")
            assert diagnostics._enforce_state_db_containment(home / "lcm.db", description="fixture") == home / "lcm.db"
            with pytest.raises(ValueError):
                diagnostics._enforce_state_db_containment(tmp_path / "outside.db", description="fixture")
            with pytest.raises(ValueError):
                externalize.get_large_output_storage_dir(
                    SimpleNamespace(large_output_externalization_path=str(tmp_path / "outside")), str(home), create=False)


def test_environment_read_inventory():
    """AST inventory, including aliased getenv/environ access in plugin modules."""
    allowed = {
        ("config.py", "lcm_env"): "sole resolver: explicit single-profile/old-host fallback",
        ("config.py", "from_env"): "identity check to distinguish routed snapshot from process mapping",
        ("config.py", "containment_base_dir"): "LCM_HERMES_BASE_DIR: operator containment root stays process-wide",
        ("embedding_provider.py", "__init__"): "FASTEMBED_CACHE_PATH: process-wide model download cache",
        ("inactive_record.py", "_process_start"): "LC_ALL=C subprocess for ps process identity, not profile configuration",
    }
    seen = set()
    failures = []
    excluded = {"tests", "bench", "benchmarking", "benchmarks", "scripts"}
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if any(part in excluded or part.startswith(".") for part in relative.parts) or relative.parts[0] == "agent":
            continue
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        os_names = {"os"}
        env_names, getenv_names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                os_names.update(alias.asname or "os" for alias in node.names if alias.name == "os")
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                env_names.update(alias.asname or alias.name for alias in node.names if alias.name == "environ")
                getenv_names.update(alias.asname or alias.name for alias in node.names if alias.name == "getenv")
        for node in ast.walk(tree):
            is_read = (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                       and node.value.id in os_names and node.attr in {"environ", "getenv"})
            is_read |= isinstance(node, ast.Name) and node.id in env_names | getenv_names
            if not is_read:
                continue
            parent = parents.get(node)
            while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = parents.get(parent)
            key = (relative.as_posix(), parent.name if parent else "<module>")
            seen.add(key)
            if key not in allowed:
                failures.append(f"{relative}:{node.lineno}: unclassified environment read")
            if key == ("config.py", "lcm_env"):
                assert isinstance(parents[node], ast.Return), "only the legacy mapping fallback is allowed"
            if key == ("config.py", "from_env"):
                assert isinstance(parents[node], ast.Compare) and isinstance(parents[node].ops[0], ast.IsNot)
            if key == ("embedding_provider.py", "__init__"):
                call = parents.get(parents.get(node))
                assert isinstance(call, ast.Call) and ast.literal_eval(call.args[0]) == "FASTEMBED_CACHE_PATH"
    assert not failures, "\n".join(failures)
    assert seen == set(allowed), "remove stale allowlist entries or check newly unclassified reads"


@pytest.fixture
def api():
    return "full"


@pytest.fixture
def scheduled_profiles(host, load_profile, tmp_path, monkeypatch, api):
    """One module, two pinned homes; workers must inherit B without binding it."""
    if api == "floor":
        _floor_api(monkeypatch)
    module = None
    for name, batch in [("launch", "11"), ("B", "23")]:
        home = tmp_path / name
        mapping = {
            "VOYAGE_API_KEY": f"fixture-{name}",
            "LCM_EMBEDDING_BACKFILL_BATCH_SIZE": batch,
            "LCM_EMBEDDINGS_ENABLED": "false",
            "LCM_EMBEDDING_PROVIDER": "voyage",
            "LCM_EMBEDDING_MODEL": "voyage-4",
            "LCM_ASSERTIONS_ENABLED": "true",
            "LCM_ASSERTION_EXTRACTION_ENABLED": "true",
            "LCM_ASSERTION_EXTRACTION_MODEL": f"fixture/{name}",
        }
        home.mkdir()
        (home / ".env").write_text("\n".join(f"{key}={value}" for key, value in mapping.items()))
        with host.bind(home, mapping):
            module, engine = load_profile(module)
        if name == "launch":
            monkeypatch.setenv("HERMES_HOME", str(home))
    return module, engine, home, mapping


@pytest.mark.parametrize("api", ["full", "floor"])
def test_contextless_query_worker(host, scheduled_profiles, api):
    module, engine, home, mapping = scheduled_profiles
    provider_module = importlib.import_module(module.__name__ + ".embedding_provider")
    tools = importlib.import_module(module.__name__ + ".tools")
    provider = provider_module.resolve_provider(engine._config)
    headers = []

    def transport(*args, **kwargs):
        headers.append(kwargs["headers"]["Authorization"])
        return provider_module.HttpResponse(
            200, {}, json.dumps({"data": [{"embedding": [0.1, 0.2]}]}).encode())

    provider._transport = transport
    with host.bind(home, mapping):
        tools._run_within_deadline(
            lambda: provider_module._run_blocking_with_deadline(
                lambda: provider.embed_query("fixture query"), timeout=3, provider="fixture"),
            remaining_s=3, name="fixture-query")
    assert headers == ["Bearer fixture-B"]


@pytest.mark.parametrize("api", ["full", "floor"])
def test_contextless_maintenance_worker(host, scheduled_profiles, monkeypatch, api):
    module, engine, home, mapping = scheduled_profiles
    config = importlib.import_module(module.__name__ + ".config")
    command = importlib.import_module(module.__name__ + ".command")
    engine_module = importlib.import_module(module.__name__ + ".engine")
    release = threading.Event()
    seen = []

    def maintain(*args, **kwargs):
        assert release.wait(3)
        seen.append((command._embedding_backfill_batch_size(), config.lcm_env_get("VOYAGE_API_KEY")))

    monkeypatch.setattr(engine_module, "run_incremental_embedding_pass", maintain)
    engine._config.embeddings_enabled = True
    try:
        with host.bind(home, mapping):
            engine._schedule_embedding_maintenance()
    finally:
        release.set()
    assert engine.drain_embedding_maintenance(timeout=3)
    assert seen == [(23, "fixture-B")]


@pytest.mark.parametrize("background", [False, True])
def test_stamped_scope_without_home_override(host, scheduled_profiles, monkeypatch, background):
    """A foreign stamp is routing provenance even without a home override."""
    module, engine, home, mapping = scheduled_profiles
    config = importlib.import_module(module.__name__ + ".config")
    command = importlib.import_module(module.__name__ + ".command")
    engine_module = importlib.import_module(module.__name__ + ".engine")
    release = threading.Event()
    seen = []

    def read(*args, **kwargs):
        if background:
            assert release.wait(3)
        seen.append((config.lcm_env_get("HERMES_HOME"),
                     config.lcm_env_get("VOYAGE_API_KEY"),
                     command._embedding_backfill_batch_size()))

    monkeypatch.setattr(engine_module, "run_incremental_embedding_pass", read)
    engine._config.embeddings_enabled = True
    try:
        with host.bind(None, mapping, bound_home=home):
            from hermes_constants import get_hermes_home_override
            assert get_hermes_home_override() is None
            if background:
                engine._schedule_embedding_maintenance()
            else:
                read()
    finally:
        release.set()
    if background:
        assert engine.drain_embedding_maintenance(timeout=3)
    assert seen == [(str(home), "fixture-B", 23)]


@pytest.mark.parametrize("accessor", ["missing", "raises"])
def test_stamped_scope_override_accessor_unavailable(host, scheduled_profiles, monkeypatch, accessor):
    module, _, home, mapping = scheduled_profiles
    config = importlib.import_module(module.__name__ + ".config")
    import hermes_constants

    def fail():
        raise RuntimeError("override accessor unavailable")

    if accessor == "missing":
        monkeypatch.delattr(hermes_constants, "get_hermes_home_override")
    else:
        monkeypatch.setattr(hermes_constants, "get_hermes_home_override", fail)
    with host.bind(None, mapping, bound_home=home):
        assert config.lcm_env_get("HERMES_HOME") == str(home.parent / "launch")
        assert config.lcm_env_get("VOYAGE_API_KEY") == "fixture-launch"


def test_profile_bound_reentrance(host, scheduled_profiles):
    module, _, home, mapping = scheduled_profiles
    config = importlib.import_module(module.__name__ + ".config")
    barrier = threading.Barrier(2)

    def read(*, concurrent=False):
        if concurrent:
            barrier.wait(timeout=3)
        return config.lcm_env_get("VOYAGE_API_KEY")

    with host.bind(home, mapping):
        wrapped = config.profile_bound(read)
    assert [wrapped(), wrapped()] == ["fixture-B", "fixture-B"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(wrapped, concurrent=True) for _ in range(2)]
        assert [job.result(timeout=4) for job in jobs] == ["fixture-B", "fixture-B"]


def test_profile_bound_resets_lcm_turn_state(host, scheduled_profiles):
    module, _, home, mapping = scheduled_profiles
    config = importlib.import_module(module.__name__ + ".config")
    externalize = importlib.import_module(module.__name__ + ".externalize")

    def read():
        return (config.lcm_env_get("VOYAGE_API_KEY"), externalize._payload_lookup_memo.get(),
                externalize.ingest_payload_writes.get())

    with host.bind(home, mapping), externalize.payload_lookup_scope():
        token = externalize.ingest_payload_writes.set(False)
        try:
            wrapped = config.profile_bound(read)
        finally:
            externalize.ingest_payload_writes.reset(token)
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        assert worker.submit(wrapped).result(timeout=3) == ("fixture-B", None, True)
    finally:
        worker.shutdown()


def test_assertion_worker_profile_model(host, scheduled_profiles, monkeypatch):
    module, engine, home, mapping = scheduled_profiles
    config = importlib.import_module(module.__name__ + ".config")
    extraction = importlib.import_module(module.__name__ + ".assertion_extraction")
    seen = []

    def model_call(prompt, model, timeout):
        seen.append((model, config.lcm_env_get("LCM_ASSERTION_EXTRACTION_MODEL")))
        source = json.loads(prompt.split("EXACT_SOURCE:\n", 1)[1])
        return json.dumps({"schema_version": 1, "source_store_id": source["store_id"],
                           "source_content_sha256": source["content_sha256"],
                           "assertions": [], "relations": []}), 0, 0

    monkeypatch.setattr(extraction, "_call_structured_assertion_llm", model_call)
    engine._session_id = "fixture-session"
    messages = [{"role": "user", "content": "I prefer tea."}]
    engine._store.append(engine._session_id, messages[0])
    with host.bind(home, mapping):
        assert engine._schedule_pre_compaction_assertions(messages)
    assert engine._assertion_extraction_idle.wait(3)
    assert seen == [("fixture/B", "fixture/B")]
    assert engine._assertion_extraction_sources_failed == 0
