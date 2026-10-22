"""Synthetic paired costs, same fixture/process and alternating execution order."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import hermes_lcm
from hermes_lcm import command
from tests.test_issue_1652_catchup_disclosure import engine as engine, backfill, recall
from tests.test_issue_1659_catchup import seed_uncertain

HERE = Path(__file__).parent


def load_base():
    spec = importlib.util.spec_from_file_location('hermes_lcm.command_base', HERE / 'command-base.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@contextmanager
def active(module):
    previous = sys.modules['hermes_lcm.command']
    sys.modules['hermes_lcm.command'] = module
    hermes_lcm.command = module
    try:
        yield
    finally:
        sys.modules['hermes_lcm.command'] = previous
        hermes_lcm.command = previous


def measure(functions, n):
    samples = {label: [] for label in functions}
    for function in functions.values():
        for _ in range(5):
            function()
    for pair in range(n):
        labels = list(functions) if pair % 2 == 0 else list(reversed(functions))
        for label in labels:
            started = time.perf_counter_ns()
            functions[label]()
            samples[label].append((time.perf_counter_ns() - started) / 1e6)
    medians = {label: statistics.median(values) for label, values in samples.items()}
    return {'samples_ms': samples, 'median_ms': medians, 'n': n}


def test_costs(engine, monkeypatch):
    base = load_base()
    monkeypatch.setattr(base, 'resolve_provider', command.resolve_provider)
    engine._config.ollama_base_url = 'http://192.0.2.1:11434'
    for task in ('summary', 'chunk'):
        backfill(engine, task, limit=100)
    results = {'base_sha': '238d1a9e876dea55b2e2f7a68110afd21c4e779e',
               'method': 'paired interleaved, reversed order each pair, five warmups per arm; same synthetic fixture; fake non-loopback Ollama; steady costs before uncertain seeding'}
    def call(mod, name):
        with active(mod):
            if name == 'status':
                result = mod._status_text(engine)
                assert isinstance(result, str) and result
            else:
                result = recall(engine)
                assert result['degraded'] is False
    for name in ('status', 'recall'):
        result = measure({label: lambda m=mod: call(m, name)
                          for label, mod in [('base', base), ('candidate', command)]}, 200)
        before, after = (result['median_ms'][k] for k in ('base', 'candidate'))
        result['allowed_ms'] = before + max(before * .1, .05)
        result['pass'] = after <= result['allowed_ms']
        results[name] = result
    identity, rows = seed_uncertain(engine)
    monkeypatch.setattr(base, '_chunk_pending_rows', lambda *a, **kw: (0, [], {}))
    monkeypatch.setattr(command, '_chunk_pending_rows', lambda *a, **kw: (0, [], {}))
    def pass_once(mod):
        with active(mod):
            result = mod._chunk_backfill_run(engine._config, engine._store.db_path,
                apply=True, limit=1, retry_uncertain=False, policy='conversational', maintenance_slice=1)
            assert result['error'] is None, result
    results['inflight'] = measure({label: lambda m=mod: pass_once(m)
                          for label, mod in [('base', base), ('candidate', command)]}, 20)
    results['inflight']['uncertain_rows'] = rows
    results['inflight']['sources'] = 60
    results['inflight']['selection_isolation'] = '_chunk_pending_rows returns empty for both arms; full pass including actionable count and record publication'
    (HERE / 'costs.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k: {f: v for f, v in value.items() if f != 'samples_ms'}
                      for k, value in results.items() if isinstance(value, dict)}, indent=2))
    assert results['status']['pass'] and results['recall']['pass'], results
