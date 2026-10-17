"""Real assembly observations for I3/I4/I9/I10/I13, before phase extraction.

Frozen after the stamping regroup (Refs #1271, #1398). Regenerate intentionally::

    python -m tests.behaviour.assembly_trace_harness --repo "$TREE" --out tests/fixtures/assembly_trace.json

Only the offline counter, clocks, host capabilities and provider boundary are
pinned. Store faults use SQLite triggers. Observers call the real implementation;
private helper names and source lines never enter the golden trace.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import gc
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import CodeType, FrameType, ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--repo")
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.repo == "":
        pre_parser.error("argument --repo: expected a non-empty path")
    selected = Path(pre_args.repo).resolve() if pre_args.repo is not None else Path(__file__).resolve().parents[2]
    package = ModuleType("hermes_lcm")
    package.__path__ = [str(selected)]
    sys.modules["hermes_lcm"] = package

import tests.conftest  # noqa: F401
from tests.phase0_hygiene import product_env_keys
from tests.harness import engine_context, pin_char_counter

engine_module: Any = importlib.import_module("hermes_lcm.engine")
dag_module: Any = importlib.import_module("hermes_lcm.dag")
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "assembly_trace.json"
Message = dict[str, Any]
Trace = list[str]
OBJECTIVE = "[Current user objective preserved from compacted history]"
PAD = " alpha beta gamma delta epsilon" * 40
# Independent frozen data inventory, including external reconciliation side effects.
OBSERVED_ATTRIBUTES = frozenset({
    '_LCM_SUMMARY_PART_HEADER_RE', '_compiled_ignore_message_patterns', '_compress_run',
    '_config', '_conversation_id', '_current_compress_placeholder_identity_counts',
    '_dag', '_hermes_home', '_kept_reply_mark', '_last_compacted_store_id', '_last_gate_tokens',
    '_pending_context_anchor_messages', '_pending_emission_candidates', '_pressure_yield_tail_token_limit',
    '_proactive_recall_injected_count', '_proactive_recall_privacy_error_count',
    '_proactive_recall_privacy_warned', '_proactive_recall_skipped_count',
    '_proactive_recall_timeout_count', '_session_id', '_store', 'compression_count',
    'context_length', 'last_prompt_tokens',
    '_host_rewrite_overrides', '_host_rewrite_store', '_host_rewrite_watch',
    '_engine_uids_pending', '_host_uid_lineage_cache',
})
EXTERNAL_ATTRIBUTES = frozenset({'_host_rewrite_overrides', '_host_rewrite_store', '_host_rewrite_watch',
                                 '_engine_uids_pending', '_host_uid_lineage_cache'})


TRACE_GUIDANCE = (
    "Regenerate from repo root: python -m tests.behaviour.assembly_trace_harness --repo . --out tests/fixtures/assembly_trace.json\n"
    "Attribute every changed event in the PR description to its commit or behaviour change.\n"
    "See docs/process/turn-path-probe-packet.md#golden-traces."
)


def engine_attributes(source: str) -> set[str]:
    """Follow in-file self methods (including passed callables) to a fixed point."""
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LCMEngine')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    pending, visited, found = ['_assemble_context'], set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(methods[name]):
            attr = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                attr = node.attr
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {'getattr', 'setattr', 'hasattr'} and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == 'self'):
                key = node.args[1]
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise AssertionError(
                        "Dynamic self lookup: update engine_attributes and OBSERVED_ATTRIBUTES/EXTERNAL_ATTRIBUTES.\n"
                        + TRACE_GUIDANCE)
                attr = key.value
            if attr in methods:
                pending.append(attr)
            elif attr is not None and not callable(getattr(engine_module.LCMEngine, attr, None)):
                found.add(attr)
    return found


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class Canonical:
    def __init__(self, root: Path) -> None:
        self.root = str(root)
        self.nonces: dict[str, str] = {}
        self.objects: dict[int, str] = {}
        self.held: list[Any] = []

    def text(self, value: str) -> str:
        tag = digest(str(Path(self.root) / 'home' / 'state.db'))[:16]
        value = value.replace(tag, '<home-tag>').replace(self.root, '<sandbox>')
        def nonce(match: re.Match[str]) -> str:
            return self.nonces.setdefault(match[0], f'<nonce-{len(self.nonces)}>')
        return re.sub(r'(?<![a-zA-Z0-9])[0-9a-f]{32}(?![a-zA-Z0-9])|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b', nonce, value)

    def alias(self, value: Any) -> str:
        key = id(value)
        if key not in self.objects:
            self.objects[key] = f'object-{len(self.objects)}'
            self.held.append(value)  # prevent allocator identity reuse
        return self.objects[key]

    def value(self, value: Any, key: str = '') -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (str, Path)):
            return self.text(str(value))
        if isinstance(value, bytes):
            return {'bytes': hashlib.sha256(value).hexdigest()}
        if isinstance(value, re.Pattern):
            return {'pattern': value.pattern}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: self.value(getattr(value, f.name), f.name) for f in dataclasses.fields(value)}
        if isinstance(value, dict):
            if key == 'store_ids_by_message_id':
                return {self.objects[k] if isinstance(k, int) else k: self.value(v) for k, v in value.items()}
            return {(self.objects.get(k, str(k)) if isinstance(k, int) else self.text(str(k))): self.value(v, str(k))
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(v, key) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted((self.value(v, key) for v in value), key=encoded)
        if callable(value):
            return {'callback': True}
        if isinstance(value, sqlite3.Row):
            return self.value(dict(value))
        raise TypeError(f'Unclassified observation {key}: {type(value).__name__}')

    def compact(self, value: Any) -> str:
        text = encoded(self.value(value))
        return text if len(text) <= 180 else f'sha256={digest(text)} chars={len(text)}'

    def graph(self, value: Any) -> Any:
        """Record nested input/output aliases separately from their full values."""
        if isinstance(value, (list, tuple)):
            return [self.alias(value), [self.graph(v) for v in value]]
        if isinstance(value, dict):
            return [self.alias(value), {str(k): self.graph(v) for k, v in value.items()
                                      if isinstance(v, (dict, list, tuple))}]
        return None


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    family: int
    shape: str = 'turns'
    leading: str = 'system'
    cap: int | None = None
    depths: tuple[int, ...] = ()
    persist: bool = True
    forced: bool = False
    retained: bool = False
    stored: bool = True
    note: bool = False
    count: int = 0
    window: int = 0
    identity: bool = True
    metadata: bool = True
    recall: bool = False
    external: bool = False
    caller: str = ''
    fault: bool = False
    effective_cap: int = 0


SCENARIOS = (
    Scenario('empty', 1, shape='empty', leading='none'),
    Scenario('system-note', 1, note=True),
    Scenario('system-existing-note', 1, leading='noted', note=True),
    Scenario('system-after-compression', 1, note=True, count=1),
    Scenario('system-no-note', 1),
    Scenario('leading-user', 6, leading='user', depths=(0,)),
    Scenario('leading-assistant', 6, leading='assistant', depths=(0,)),
    Scenario('no-leading', 6, leading='none', depths=(0,)),
    Scenario('uncapped', 2, depths=(0,)),
    Scenario('effective-cap', 2, depths=(0,), effective_cap=120),
    Scenario('override-cap', 2, depths=(0,), cap=120, effective_cap=500),
    *[Scenario(f'forced-{shape}-{cap}', 3, shape=shape, forced=True, cap=cap)
      for shape in ('pair', 'huge-result', 'missing', 'gap', 'mismatch') for cap in (30, 140, 500)],
    Scenario('external-rich-stub', 3, shape='huge-result', forced=True, cap=170, external=True),
    Scenario('external-plain-stub', 3, shape='huge-result', forced=True, cap=80, external=True),
    *[Scenario(f'reply-{cap}', 4, shape='reply', leading='none', cap=cap, depths=(0,), forced=True)
      for cap in (200, 560, 800, 900)],
    Scenario('reply-fold-rejections', 4, shape='reply-fold', cap=700, depths=(0,), retained=True),
    *[Scenario(f'anchor-choice-{cap}', 5, shape='scaffold', cap=cap, depths=(0,))
      for cap in (30, 100, 400, 2000)],
    Scenario('zero-depths', 7),
    Scenario('one-depth', 7, depths=(0,)),
    Scenario('many-depths', 7, depths=(0, 1, 3)),
    *[Scenario(f'default-prefix-{window}', 8, shape='scaffold', depths=(0, 1), window=window)
      for window in (0, 50, 200, 400, 1600, 8000)],
    Scenario('default-retained', 8, retained=True, depths=(0,), window=100),
    *[Scenario(f'prefix-{cap}', 9, cap=cap, depths=(0, 0, 1)) for cap in (10, 150, 3000)],
    Scenario('objective', 9, shape='objective', cap=500, depths=(0,)),
    Scenario('recall-off', 10),
    Scenario('recall-on', 10, recall=True),
    Scenario('recall-trial', 10, recall=True, persist=False),
    Scenario('recall-retained', 10, recall=True, retained=True),
    Scenario('fold-string', 11, shape='fold', retained=True, depths=(0,)),
    Scenario('fold-structured', 11, shape='fold-blocks', retained=True, depths=(0,)),
    Scenario('fold-no-lineage', 11, shape='fold', retained=True, depths=(0,), stored=False),
    Scenario('fold-standalone', 11, retained=True, depths=(0,)),
    Scenario('carrier', 12, leading='none', depths=(0,)),
    Scenario('carrier-mismatch', 12, leading='none', depths=(0,), shape='carrier-mismatch'),
    Scenario('carrier-no-later-user', 12, leading='none', depths=(0,), shape='one-user'),
    Scenario('carrier-leading', 12, depths=(0,)),
    Scenario('identity-off', 13, leading='none', depths=(0,), identity=False),
    Scenario('identity-on', 13, leading='none', depths=(0,)),
    Scenario('leading-strip', 14, leading='none', shape='fold'),
    Scenario('objective-overcap-without-objective', 14, shape='objective', cap=1),
    Scenario('objective-trim', 14, shape='trim', cap=1),
    Scenario('objective-trim-parts', 14, shape='trim-parts', cap=1, depths=(0,)),
    Scenario('metadata-absent', 15, depths=(0,), metadata=False),
    Scenario('metadata-present', 15, depths=(0,)),
    Scenario('trial-fold', 16, shape='fold', retained=True, depths=(0,), persist=False),
    Scenario('trial-standalone', 16, shape='pair', depths=(0,), persist=False),
    Scenario('trial-external', 16, shape='huge-result', external=True, persist=False, stored=False),
    Scenario('lineage-write-failure', 17, shape='fold', retained=True, depths=(0,), fault=True),
    Scenario('lineage-clear', 17),
    Scenario('lineage-existing', 17, shape='lineage-existing'),
    Scenario('committed', 18, caller='committed', depths=(0,)),
    Scenario('committed-trial', 18, caller='committed', depths=(0,), persist=False),
    Scenario('compress-scaffold', 18, caller='compress', shape='compress', depths=(0,)),
    Scenario('overflow-normal', 18, caller='overflow', shape='pair', cap=140),
    Scenario('overflow-scaffold', 18, caller='overflow', shape='overflow', depths=(0,), retained=True),
    Scenario('stub-restore', 18, caller='stub-restore', shape='huge-result', external=True, window=4000),
    Scenario('stub-first', 18, caller='stub', shape='huge-result', external=True, depths=(0,), window=200000),
)


def message(role: str, content: Any) -> Message:
    return {'role': role, 'content': content}


def pair(index: int = 1, size: int = 20) -> list[Message]:
    return [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': f'call-{index}', 'type': 'function',
             'function': {'name': 'inspect', 'arguments': '{"path":"synthetic"}'}}]},
            {'role': 'tool', 'tool_call_id': f'call-{index}', 'content': 'data' * size}]


class Provider:
    provider_id, model_id, dim = 'mock', 'mock-model', 2

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class Logs(logging.Handler):
    def __init__(self, events: Trace, canonical: Canonical) -> None:
        super().__init__()
        self.events, self.canonical = events, canonical

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(f'log {record.levelname} {self.canonical.text(record.getMessage())}')


def install_clock(patch: pytest.MonkeyPatch) -> None:
    clock = SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith('__')})
    clock.time = lambda: 1700001000.0
    clock.monotonic = clock.perf_counter = lambda: 1000.0
    clock.time_ns = lambda: 1700001000000000000
    clock.gmtime = lambda stamp=1700001000: time.gmtime(stamp)
    # Patch each product module's time binding, never pytest's or the process clock.
    for name, module in tuple(sys.modules.items()):
        if name.startswith('hermes_lcm.') and module is not None and getattr(module, 'time', None) is time:
            patch.setattr(module, 'time', clock)
        if name.startswith('hermes_lcm.') and module is not None:
            for key, value in tuple(vars(module).items()):
                if value is time.monotonic:
                    patch.setattr(module, key, clock.monotonic)


def setup(engine: Any, case: Scenario, patch: pytest.MonkeyPatch) -> tuple[Message | None, list[Message], Message | None]:
    system = None if case.leading == 'none' else message(case.leading, 'Stable system prompt.')
    if case.leading == 'noted':
        system = message('system', 'This conversation uses Lossless Context Management (LCM)')
    retained = message('user', 'kanban dashboard sprint plan') if case.retained else None
    tail = [message('user', 'Earlier request'), message('assistant', 'Earlier response'),
            message('user', 'kanban dashboard sprint plan')]
    if case.shape == 'empty':
        tail = []
    elif case.shape in {'pair', 'huge-result', 'missing', 'gap', 'mismatch'}:
        tail = pair(size=4000 if case.shape == 'huge-result' else 20)
        if case.shape == 'missing':
            tail = tail[:1]
        elif case.shape == 'gap':
            tail = [message('user', 'old'), message('assistant', 'huge' * 1000), *pair()]
        elif case.shape == 'mismatch':
            tail.insert(1, message('user', 'intervening user'))
    elif case.shape in {'fold', 'fold-blocks', 'lineage-existing'}:
        content: Any = 'durable assistant text'
        if case.shape == 'fold-blocks':
            content = [{'type': 'text', 'text': content}]
        tail = [message('assistant', content), message('user', 'next request')]
    elif case.shape == 'one-user':
        tail = tail[:2]
    elif case.shape == 'carrier-mismatch':
        tail[0]['content'] = '   '
    elif case.shape in {'reply', 'reply-fold'}:
        tail = [message('assistant', 'Visible reply ' + PAD), message('user', 'continue' + PAD * 3),
                *pair(1, 180), message('user', 'continue' + PAD), *pair(2, 180)]
        if case.shape == 'reply-fold':
            tail[1]['content'] = '[Your active task list was preserved across context compression]\n- [ ] ship it'
    elif case.shape in {'objective', 'scaffold'}:
        tail = pair()
    elif case.shape == 'compress':
        tail = [message('user' if i % 2 == 0 else 'assistant', f'Turn {i}' + PAD) for i in range(8)]
    if case.shape in {'trim', 'trim-parts'}:
        system = message('system', OBJECTIVE + '\nquoted objective' + ('\n\n---\n\nkeep this' if case.shape == 'trim-parts' else ''))
        tail = pair()
    if case.external:
        tail.extend(pair(2))
    for i, row in enumerate(tail):
        row['message_uid'] = f'host-{i}'
    node_ids = []
    for i, depth in enumerate(case.depths):
        node_ids.append(engine._dag.add_node(dag_module.SummaryNode(
            session_id='S', depth=depth, summary=f'Seed {i} ' + PAD, token_count=300,
            source_token_count=600, source_ids=[], source_type='messages' if depth == 0 else 'summaries',
            created_at=100.0 + i)))
    if case.shape == 'overflow':
        tail = [message('assistant', engine_module._summary_part_text(engine._dag.get_node(node_ids[0]))), *pair()]
    if case.caller == 'stub-restore':
        engine._config.large_output_externalization_enabled = False
        engine._config.survival_fit = False
    if case.stored:
        engine.ingest(([system] if system else []) + ([retained] if retained else []) + tail)
    if case.shape in {'objective', 'scaffold', 'trim', 'trim-parts'}:
        objective = OBJECTIVE + '\n' + 'old objective ' * 50
        if case.shape == 'scaffold':
            part = engine_module._summary_part_text(engine._dag.get_node(node_ids[0]))
            objective += '\n\n---\n\n' + part + '\n\nnew request'
        engine._pending_context_anchor_messages = [message('user', objective), *tail]
    elif case.shape == 'reply':
        engine._pending_context_anchor_messages = [message('user', 'Earlier objective'), *tail]
    if case.recall:
        tools: Any = importlib.import_module('hermes_lcm.tools')
        vectors: Any = importlib.import_module('hermes_lcm.vector_store')
        patch.setattr(tools, 'resolve_provider', lambda _config: Provider())
        node_id = engine._dag.add_node(dag_module.SummaryNode(
            session_id='earlier-session', depth=0, summary='kanban dashboard sprint plan', token_count=20,
            source_token_count=40, source_ids=[], source_type='messages', created_at=1700000000.0))
        store = vectors.VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile('mock-model', 'mock', 2)
            identity = store.capture_identity('mock-model', provider='mock')
            store.record_embedding(str(node_id), 'summary', 'mock-model', [1.0, 0.0], identity=identity)
        finally:
            store.close()
        engine._config.embeddings_enabled = True
    if case.shape == 'lineage-existing':
        source = engine._get_store_id_map_for_messages(tail)[id(tail[0])]
        assert engine._write_folded_tail_lineage(tail[0], source)
    if case.fault:
        engine._store._conn.execute("""CREATE TEMP TRIGGER fail_fold BEFORE INSERT ON metadata
            WHEN NEW.key LIKE 'folded_tail_lineage:%' AND instr(NEW.value, '"source_store_id": 0') = 0
            BEGIN SELECT RAISE(ABORT, 'synthetic lineage failure'); END""")
    if case.caller == 'stub-restore':
        engine._config.large_output_externalization_enabled = True
        payloads = Path(engine._config.large_output_externalization_path)
        payloads.mkdir(exist_ok=True)
        # A directory occupies the would-be payload filename: a real EEXIST
        # at publication, while the no-write trial can still form a placeholder.
        filename = f'20231114_223000_call-1_{digest(str(tail[1]["content"]))[:12]}_17979de70acf1000.json'
        (payloads / filename).mkdir()
    if case.caller in {'stub', 'stub-restore'}:
        engine._compress_occurrences = None  # public compress entry initializes this channel
    engine.compression_count = case.count
    engine._pending_emission_candidates = [{'sentinel': 'untouched trial'}]
    engine._kept_reply_mark = (tail, 7)
    return system, tail, retained


def snapshot(engine: Any, canonical: Canonical, result: Any = None) -> dict[str, Any]:
    state = {}
    for name in sorted(OBSERVED_ATTRIBUTES):
        value = getattr(engine, name, {'absent': True})
        if name == '_store':
            value = {table: [tuple(row) for row in engine._store._conn.execute(f'SELECT * FROM {table} ORDER BY 1')]
                     for table in ('messages', 'metadata')}
        elif name == '_dag':
            value = engine._dag.get_session_nodes('S', limit=10000)
        elif name == '_host_rewrite_store':
            value = {'present': value is not None}
        elif name == '_kept_reply_mark' and value is not None:
            value = {'tokens': value[1], 'is_result': value[0] is result, 'rows': value[0]}
        elif name == '_pending_emission_candidates':
            value = [{**candidate, 'row_index': next((i for i, row in enumerate(result or [])
                      if row is candidate.get('row')), None)} for candidate in (value or [])]
        state[name] = canonical.value(value, name)
    return state


def observe(case: Scenario, root: Path) -> Trace:
    root.mkdir(parents=True, exist_ok=True)
    canonical, events = Canonical(root), []
    active = False
    conns: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = connect(*args, **kwargs)
        index = len(conns)
        conns.append(conn)
        def sql(statement: str) -> None:
            # SQLite-internal FTS5 statements vary by SQLite build.
            if statement.lstrip().startswith('--') or re.search(r"(['\"])(main|temp)\1\s*\.", statement):
                return
            if active:
                events.append(f'sql conn={index} ' + canonical.compact(statement.strip()))
        conn.set_trace_callback(sql)
        return cast(sqlite3.Connection, conn)

    with pytest.MonkeyPatch.context() as patch:
        pin: Any = pin_char_counter
        context: Any = engine_context
        pin(patch)
        install_clock(patch)
        patch.setenv('HERMES_HOME', str(root / 'home'))
        patch.setenv('LCM_HOST_MESSAGE_UID', 'shadow' if case.identity else 'off')
        uid_module = importlib.import_module('hermes_lcm.host_uid_emit')
        patch.setattr(uid_module, '_host_uid_capability', True)
        patch.setattr(sqlite3, 'connect', traced_connect)
        host = ModuleType('agent.context_compressor')
        if case.metadata:
            host.MODEL_ONLY_DISPLAY_METADATA_KEY = 'model_only'  # type: ignore[attr-defined]
        patch.setitem(sys.modules, 'agent.context_compressor', host)
        engine_module._model_only_display_metadata_key.cache_clear()
        patch.setattr(engine_module, '_host_estimate', lambda _rows: None)
        settings = dict(embeddings_enabled=False, temporal_rollups_enabled=False,
                        fresh_tail_count=1, fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                        max_assembly_tokens=case.effective_cap, reserve_tokens_floor=0,
                        proactive_recall_enabled=case.recall, embedding_provider='mock', embedding_model='mock-model',
                        large_output_externalization_enabled=case.external,
                        large_output_externalization_threshold_chars=100,
                        large_output_active_replay_stubbing_enabled=case.external,
                        large_output_active_replay_stub_threshold_tokens=100,
                        large_output_externalization_path=str(root / 'home' / 'payloads'))
        home = root / 'home'
        home.mkdir(exist_ok=True)
        with sqlite3.connect(home / 'state.db') as host_db:
            host_db.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)')
            host_db.execute("INSERT INTO sessions VALUES ('S', NULL, NULL)")
        with context(root, window=case.window, conversation_id='trace-conversation', **settings) as engine:
            system, tail, retained = setup(engine, case, patch)
            inputs = [system, tail, retained, getattr(engine, '_pending_context_anchor_messages', None)]
            events.append('input aliases ' + encoded(canonical.graph(inputs)))
            events.append('input before ' + canonical.compact(inputs))
            initial = snapshot(engine, canonical)
            events.extend(f'before {name} ' + canonical.compact(value) for name, value in initial.items())
            # Code objects are the qualified observer identity, never co_name.
            target = inspect.unwrap(engine_module.LCMEngine._assemble_context).__code__
            store_map = engine_module.LCMEngine._get_store_id_map_for_messages.__code__
            cleanup = engine_module.LCMEngine._sanitize_active_context_messages.__code__
            visibility = engine_module._stamp_generated_model_only.__code__
            counters = {engine_module.count_message_tokens.__code__, engine_module.count_messages_tokens.__code__,
                        engine_module.count_tokens.__code__}
            services: dict[CodeType, str] = {}
            for module_name, class_name in (('store', 'MessageStore'), ('dag', 'SummaryDAG')):
                cls = getattr(importlib.import_module('hermes_lcm.' + module_name), class_name)
                for name, method in vars(cls).items():
                    if not name.startswith('_') and inspect.isfunction(method):
                        services[method.__code__] = module_name + '.' + name
            calls: dict[int, str] = {}
            entries = 0
            def profile(frame: FrameType, event: str, arg: Any) -> None:
                nonlocal entries
                code = frame.f_code
                if code is store_map and event == 'call':
                    for row in frame.f_locals['messages']:
                        canonical.alias(row)
                if code is target and event == 'call':
                    entries += 1
                    events.append('assembly enter rejected=' + canonical.compact(frame.f_locals.get('_reply_rejected')))
                caller = frame.f_back
                if (event == 'call' and code in {cleanup, visibility} and caller is not None
                        and caller.f_code is target and 'existing_folded_lineage' in caller.f_locals
                        and caller.f_locals.get('folded_original_tail') is not None):
                    # Fold recovery currently has no generated rows. Record the
                    # shared provenance list, including when empty: cleanup must
                    # update the same collection used by the final visibility pass.
                    generated = frame.f_locals.get('generated_rows')
                    kind = 'cleanup' if code is cleanup else 'visibility'
                    events.append(f'fold provenance {kind} ' + encoded({
                        'assembly_owned': generated is caller.f_locals['generated_rows'],
                        'rows': len(generated) if generated is not None else None,
                    }))
                label = services.get(code)
                if code in counters and frame.f_back is not None and frame.f_back.f_code.co_filename == target.co_filename:
                    label = 'counter ' + code.co_qualname
                if label is None:
                    return
                if event == 'call':
                    arguments = {k: v for k, v in frame.f_locals.items() if k not in {'self', 'cls'}}
                    calls[id(frame)] = label
                    events.append(label + ' call ' + canonical.compact(arguments))
                elif event == 'return' and id(frame) in calls:
                    events.append(calls.pop(id(frame)) + ' return ' + canonical.compact(arg))
            scandir, glob, iterdir = os.scandir, Path.glob, Path.iterdir
            repository = Path(engine_module.__file__).resolve().parent
            payload_root = (root / 'home' / 'payloads').resolve()
            scans = dict(scandir=0, glob=0, iterdir=0)
            def count_scan(kind: str, path: Any, caller: FrameType) -> None:
                # Count LCM call sites, not version-dependent pathlib internals.
                if (active and Path(caller.f_code.co_filename).is_relative_to(repository)
                        and not isinstance(path, int)
                        and Path(os.fsdecode(path)).resolve().is_relative_to(payload_root)):
                    scans[kind] += 1
            def scan(path: Any = '.') -> Any:
                count_scan('scandir', path, sys._getframe(1))
                return scandir(path)
            def path_glob(path: Path, pattern: str, *args: Any, **kwargs: Any) -> Iterator[Path]:
                count_scan('glob', path, sys._getframe(1))
                return glob(path, pattern, *args, **kwargs)
            def path_iterdir(path: Path) -> Iterator[Path]:
                count_scan('iterdir', path, sys._getframe(1))
                return iterdir(path)
            patch.setattr(os, 'scandir', scan)
            patch.setattr(Path, 'glob', path_glob)
            patch.setattr(Path, 'iterdir', path_iterdir)
            handler = Logs(events, canonical)
            logger = logging.getLogger('hermes_lcm')
            prior_level = logger.level
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            old_profile = sys.getprofile()
            old_thread_profile = threading.getprofile()
            gc_enabled = gc.isenabled()
            # Cyclic finalizers from earlier tests are not assembly side effects.
            # Defer automatic collection until the process-wide observer is off.
            gc.disable()
            active = True
            sys.setprofile(profile)
            threading.setprofile(profile)
            try:
                rows = ([system] if system else []) + ([retained] if retained else []) + tail
                if case.caller == 'committed':
                    result = engine._assemble_committed_compaction_context(rows, rows, case.cap, persist=case.persist)
                elif case.caller == 'overflow':
                    result = engine._assemble_overflow_recovery_context(system, tail, case.cap, retained)
                elif case.caller in {'stub', 'stub-restore'}:
                    result = engine._stub_first_exit(rows, rows, rows, rows, 1000)
                elif case.caller == 'compress':
                    escalation: Any = importlib.import_module('hermes_lcm.escalation')
                    patch.setattr(escalation, '_call_llm_for_summary', lambda *a, **kw: 'Earlier turns.\nExpand for details about: turns')
                    engine._foreground_budget = engine._new_foreground_budget()
                    result = engine._compress_impl(rows, current_tokens=3000, force=True)
                else:
                    result = engine._assemble_context(system, tail, assembly_cap_override=case.cap,
                        include_lcm_note=case.note, retained_user_message=retained,
                        stub_over_cap_tool_results=case.forced, persist=case.persist)
            finally:
                sys.setprofile(old_profile)
                threading.setprofile(old_thread_profile)
                active = False
                logger.removeHandler(handler)
                logger.setLevel(prior_level)
                if gc_enabled:
                    gc.enable()
            events.append(f'assembly entries={entries} payload ' + ' '.join(f'{kind}={count}' for kind, count in scans.items()))
            events.append('result aliases ' + encoded(canonical.graph(result)))
            events.extend(f'row {i} {canonical.compact(row)}' for i, row in enumerate(result or []))
            events.append('input after ' + canonical.compact(inputs))
            events.append('input aliases after ' + encoded(canonical.graph(inputs)))
            final = snapshot(engine, canonical, result)
            events.extend(f'after {name} ' + canonical.compact(value) for name, value in final.items())
            for path in sorted((root / 'home' / 'payloads').glob('**/*')):
                if path.is_file():
                    events.append('payload ' + str(path.relative_to(root)) + ' sha256=' + hashlib.sha256(path.read_bytes()).hexdigest())
        engine_module._model_only_display_metadata_key.cache_clear()
        importlib.import_module('hermes_lcm.retrieval_core')._reset_vector_store_pool()
    return events


def first_difference(expected: Trace, actual: Trace) -> str | None:
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else '<end>'
        right = actual[index] if index < len(actual) else '<end>'
        if left != right:
            return f'event {index}: expected {left!r}, got {right!r}'
    return None


def main() -> None:
    environment = dict(os.environ)
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--repo', type=Path, required=True)
        parser.add_argument('--out', type=Path, required=True)
        args = parser.parse_args()
        for key in product_env_keys(ROOT) | product_env_keys(args.repo.resolve()):
            os.environ.pop(key, None)
        with tempfile.TemporaryDirectory(prefix='assembly-trace-') as directory:
            result = {case.name: observe(case, Path(directory).resolve() / case.name) for case in SCENARIOS}
        args.out.write_text(json.dumps(result, sort_keys=True, indent=0) + '\n')
        print(f'Wrote {len(result)} assembly scenarios')
    finally:
        os.environ.clear()
        os.environ.update(environment)


if __name__ == '__main__':
    main()
