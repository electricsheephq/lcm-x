from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_guidance_module():
    package_name = "hermes_lcm_recall_guidance"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    package = importlib.util.module_from_spec(package_spec)
    assert package_spec is not None
    assert package_spec.loader is not None
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    return __import__(f"{package_name}.guidance", fromlist=["guidance"])


def test_recall_policy_is_canonical_bounded_and_benchmark_neutral():
    guidance = _load_guidance_module()
    first = guidance.get_recall_policy()
    second = guidance.get_recall_policy()

    assert first == second
    assert first == guidance.RECALL_POLICY_PATH.read_text(encoding="utf-8").strip()
    assert 0 < len(first.encode("utf-8")) <= 8 * 1024
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == guidance.recall_policy_sha256()
    lowered = first.lower()
    for forbidden in ("longmemeval", "question_id", "reference answer", "judge output"):
        assert forbidden not in lowered


def test_recall_policy_covers_evidence_safety_and_stable_tool_routing():
    guidance = _load_guidance_module()
    policy = guidance.get_recall_policy()

    for phrase in (
        "summaries are recall cues",
        "newer source-backed evidence",
        "1-3 distinctive terms",
        "lcm_grep",
        "lcm_describe",
        "lcm_expand_query",
        "lcm_recall",
        "lcm_load_session",
        "lcm_recent",
        "lcm_compile_evidence",
        "lcm_evidence_pack",
        "lcm_compute",
        "Open-cardinality evidence remains incomplete",
    ):
        assert phrase in policy


def test_bundled_skill_has_valid_minimal_frontmatter_and_matching_tool_references():
    skill_root = REPO_ROOT / "skills" / "hermes-lcm"
    skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    assert skill_text.startswith("---\nname: hermes-lcm\ndescription:")
    frontmatter = skill_text.split("---", 2)[1]
    keys = {
        line.split(":", 1)[0].strip()
        for line in frontmatter.splitlines()
        if ":" in line
    }
    assert keys == {"name", "description"}

    recall_reference = (skill_root / "references" / "recall-tools.md").read_text(
        encoding="utf-8"
    )
    documented = {
        "lcm_grep",
        "lcm_recall",
        "lcm_recent",
        "lcm_load_session",
        "lcm_describe",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_compute",
        "lcm_compile_evidence",
        "lcm_evidence_pack",
    }
    for tool_name in documented:
        assert f"`{tool_name}`" in recall_reference

    schemas_text = (REPO_ROOT / "schemas.py").read_text(encoding="utf-8")
    for tool_name in documented:
        assert f'"name": "{tool_name}"' in schemas_text


def test_v2_served_identically_on_every_path(monkeypatch):
    import json
    from types import SimpleNamespace
    from hermes_lcm import mcp_reader
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    from bench.fmp.delivery import schemas as fmp_schemas

    golden = json.loads((REPO_ROOT / 'tests/fixtures/recall_anchor_main_golden.json').read_text())
    guidance = _load_guidance_module()
    plugin = sys.modules[guidance.__package__]
    cfg = LCMConfig()
    shim = SimpleNamespace(_config=cfg, _disabled_tool_names=lambda: set(),
                           _omitted_tool_names=lambda: set(), apply_identity_migration=lambda notice: None)
    monkeypatch.setattr(LCMConfig, 'from_env', classmethod(lambda cls: cfg))
    # register() imports within its independently loaded package.
    engine_module = __import__(f'{plugin.__name__}.engine', fromlist=['LCMEngine'])
    config_module = __import__(f'{plugin.__name__}.config', fromlist=['LCMConfig'])
    monkeypatch.setattr(engine_module, 'LCMEngine', lambda **kw: shim)
    monkeypatch.setattr(config_module.LCMConfig, 'from_env', classmethod(lambda cls: cfg))
    monkeypatch.setattr(plugin, '_engine_took_slot', lambda *a: True)
    for variant in ('v1', 'v2'):
        cfg.recall_guidance = variant
        engine = LCMEngine.get_tool_schemas(shim)
        captured = []
        ctx = SimpleNamespace(context_engine_tool_handlers_receive_messages=True,
                              register_context_engine=lambda e: None,
                              register_tool=lambda **kw: captured.append(kw['schema']))
        plugin.register(ctx)
        # Default invocation remains a backwards-compatible reader API.
        reader = mcp_reader.tool_definitions(set(), cfg) if variant == 'v2' else mcp_reader.tool_definitions(set())
        if variant == 'v1':
            assert json.dumps(engine) == golden['engine_schemas']
            assert json.dumps(captured) == golden['engine_schemas']
            assert json.dumps(reader) == golden['reader_schemas']
        by_name = {s['name']: s for s in engine}
        for name in ('lcm_recall', 'lcm_load_session'):
            host = next(s for s in captured if s['name'] == name)
            served = next(s for s in reader if s['name'] == name)
            assert host == by_name[name]
            assert served['description'] == host['description'] + mcp_reader.READER_NOTE
            assert served['inputSchema'] == host['parameters']
        if variant == 'v2':
            assert 'For questions that combine or compare' in by_name['lcm_recall']['description']
            assert 'min_sessions' not in json.dumps(engine)
            props = by_name['lcm_load_session']['parameters']['properties']
            assert props['anchor_store_id']['type'] == 'integer'
            assert props['before']['default'] == 6 and props['after']['default'] == 12
            assert fmp_schemas(shim)[0]['type'] == 'function'
            assert next(s['function'] for s in fmp_schemas(shim)
                        if s['function']['name'] == 'lcm_recall') == by_name['lcm_recall']
