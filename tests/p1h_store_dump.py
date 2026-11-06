"""Synthetic all-table dump worker for test_p1h_store_identity (no live homes)."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tree', type=Path)
    parser.add_argument('root', type=Path)
    parser.add_argument('host', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree))
    package = ModuleType('hermes_lcm')
    package.__path__ = [str(args.tree)]
    sys.modules['hermes_lcm'] = package
    import tests.conftest  # noqa: F401
    import pytest
    from tests.behaviour.reconcile_trace_harness import install_clock
    from tests.harness import pin_char_counter
    from tests.test_host_uid_shadow import _engine, _m

    def dump(conn):
        tables = {}
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            quoted = '"' + name.replace('"', '""') + '"'
            columns = [r[1] for r in conn.execute('PRAGMA table_info(' + quoted + ')')]
            clocks = {'timestamp', 'ingested_at', 'created_at', 'updated_at', 'registered_at', 'completed_at'}
            indexes = [i for i, column in enumerate(columns) if column not in clocks]
            rows = [[{'bytes': v.hex()} if isinstance(v, bytes) else v
                     for i, v in enumerate(row) if i in indexes]
                    for row in conn.execute('SELECT * FROM ' + quoted)]
            tables[name] = {'columns': [columns[i] for i in indexes],
                            'rows': sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))}
        return tables

    results = {}
    with pytest.MonkeyPatch.context() as patch:
        pin_char_counter(patch)
        install_clock(patch)
        bootstrap = importlib.import_module('hermes_lcm.db_bootstrap')
        patch.setattr(bootstrap, 'uuid', SimpleNamespace(uuid4=lambda: uuid.UUID(int=2293)))
        engine_type = importlib.import_module('hermes_lcm.engine').LCMEngine
        for method in ('_schedule_rollup_maintenance', '_schedule_embedding_maintenance'):
            patch.setattr(engine_type, method, lambda *_args: None)
        for mode in ('off', 'shadow'):
            patch.setenv('LCM_INGEST_LEDGER', mode)
            patch.setenv('LCM_HOST_MESSAGE_UID', 'off')
            patch.setenv('LCM_IDENTITY_ANCHOR', 'on')
            for shape in ('new-2000', 'per-turn', 'cold-cursor', 'equal-drop', 'merge-insert'):
                root = args.root / (mode + '-' + shape)
                root.mkdir(parents=True)
                patch.setenv('HERMES_HOME', str(args.host))
                engine = _engine(root)
                engine._hermes_home = engine._store._hermes_home = str(args.host)
                engine.on_session_start('S', platform='cli', context_length=200_000, conversation_id='conv')
                states = []
                try:
                    if shape == 'new-2000':
                        views = [[_m('user' if i % 2 == 0 else 'assistant', f'row {i}', 10. + i)
                                  for i in range(2000)]]
                    elif shape == 'per-turn':
                        rows = [_m('user' if i % 2 == 0 else 'assistant', f'turn {i}', 10. + i)
                                for i in range(12)]
                        views = [rows[:i] for i in range(2, 13, 2)]
                    elif shape == 'cold-cursor':
                        a, b, c = (_m('user', 'question', 10.), _m('assistant', 'answer', 11.),
                                   _m('user', 'next', 12.))
                        views = [[a, b], [a, b], [a, b, c], [c]]
                    elif shape == 'equal-drop':
                        a, b = _m('user', 'same', 10.), _m('assistant', 'answer', 11.)
                        views = [[a, b], [a], [a, a, b], [a, b]]
                    else:
                        a, b = _m('user', 'left', 10.), _m('assistant', 'right', 20.)
                        views = [[a, b], [a, _m('user', 'middle', 15.), b],
                                 [a, _m('user', 'middle\n\nnew tail', 15.), b]]
                    for index, view in enumerate(views):
                        if shape == 'cold-cursor' and index:
                            engine.on_session_start('S', platform='cli', context_length=200_000, conversation_id='conv')
                        engine._ingest_messages([dict(row) for row in view])
                        states.append(dump(engine._store._conn))
                    results[mode + '/' + shape] = states
                finally:
                    engine.shutdown()
    args.output.write_text(json.dumps(results, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
