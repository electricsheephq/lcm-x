from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench.tools import rowselect


def _git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def package(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.name', 'Fixture')
    _git(repo, 'config', 'user.email', 'fixture@example.invalid')
    files = {
        '__init__.py': 'from . import engine\n',
        'engine.py': 'from . import helper, loader, empty, exempt, hubleaf\n',
        'store.py': 'from . import middle\n',
        'middle.py': 'from . import helper\n',
        'loader.py': 'import importlib\nimportlib.import_module("hermes_lcm.dynamic")\n',
        'helper.py': '', 'dynamic.py': '', 'empty.py': '', 'exempt.py': '',
        'hubleaf.py': '', 'orphan.py': '', 'deleted.py': '', 'old.py': 'unique = 123456789\n',
        'scriptonly.py': '', 'plugin.yaml': '',
        'scripts/tool.py': 'from hermes_lcm import helper, scriptonly\n',
        'docs/readme.md': '', 'unknown.txt': '',
        'absolute.py': 'import hermes_lcm.helper\n',
        'named.py': 'MODULE = "hermes_lcm.helper"\n',
        'relative.py': 'from . import helper\n',
        'nested/__init__.py': '',
        'nested/reader.py': 'from ..helper import value\n',
        'byname.py': 'from importlib import import_module\n'
                     'def load():\n    return import_module(".helper", package="hermes_lcm")\n',
        'builtin.py': '__import__("hermes_lcm.helper")\n',
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    table = json.loads(rowselect.DEFAULT_TABLE.read_text())
    table['entries'] = [{'glob': '*.py', 'classes': ['cli']}]
    table['entries'][:0] = [{'glob': name, 'classes': classes} for name, classes in [
        ('store.py', ['storage']), ('loader.py', ['recall']), ('engine.py', ['host']),
        ('absolute.py', ['delivery']), ('nested/reader.py', ['summariser']),
        ('scripts/**', ['instrument']), ('plugin.yaml', ['host'])]]
    table['exempt'] = {'exempt.py': 'Fixture CLI metadata, no memory behavior.'}
    table_path = tmp_path / 'table.json'
    table_path.write_text(json.dumps(table))
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-qm', 'base')
    base = _git(repo, 'rev-parse', 'HEAD')
    for name in files:
        if name not in {'store.py', 'middle.py', 'loader.py', 'nested/__init__.py'}:
            with (repo / name).open('a') as stream:
                stream.write('\n# changed\n' if name.endswith('.py') else 'changed\n')
    (repo / 'deleted.py').unlink()
    (repo / 'old.py').rename(repo / 'new.py')
    (repo / 'new.py').write_text('unique = 123456789\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-qm', 'head')
    return repo, base, table_path


def test_selection_rules(package):
    repo, base, table = package
    result = rowselect.select(repo, base, 'HEAD', table)
    paths = result['paths']
    memory = rowselect.MEMORY
    def reasons(path):
        return {r['reason'] for r in paths[path]['reasons']}
    assert 'cli' in paths['helper.py']['classes']
    assert 'via middle.py -> store.py' in reasons('helper.py')
    assert {'via named.py', 'via relative.py', 'via byname.py', 'via builtin.py',
            'via absolute.py', 'via nested/reader.py'} <= reasons('helper.py')
    assert {'storage', 'delivery', 'summariser'} <= set(paths['helper.py']['classes'])
    assert not any('engine.py' in r or '__init__.py' in r for r in reasons('helper.py'))
    assert not any('scripts/' in r for r in reasons('helper.py'))
    assert set(paths['unknown.txt']['classes']) == rowselect.CLASSES
    assert paths['docs/readme.md']['classes'] == []
    assert paths['docs/readme.md']['reasons'] == [{'reason': 'non-product', 'classes': []}]
    assert memory <= set(paths['engine.py']['classes'])
    assert 'hub closure' in reasons('empty.py')
    assert 'matched *.py' in reasons('empty.py')
    assert paths['exempt.py']['classes'] == ['cli']
    assert 'recall' in paths['dynamic.py']['classes']
    assert 'via loader.py' in reasons('dynamic.py')
    assert 'not in import graph' in reasons('orphan.py')
    assert memory <= set(paths['orphan.py']['classes'])
    assert 'absent at head' in reasons('deleted.py')
    assert 'absent at head' in reasons('old.py')
    assert 'not in import graph' in reasons('new.py')
    assert paths['scripts/tool.py'] == {
        'classes': ['instrument'],
        'reasons': [{'reason': 'matched scripts/**', 'classes': ['instrument']}]}
    assert 'not in import graph' in reasons('scriptonly.py')
    assert 'hub closure' in reasons('hubleaf.py')
    importers_only = {'absolute.py', 'builtin.py', 'byname.py', 'named.py', 'nested/reader.py',
                      'relative.py'}
    assert set(result['fail_closed']) == importers_only | {
        'deleted.py', 'empty.py', 'hubleaf.py', 'new.py', 'old.py', 'orphan.py', 'scriptonly.py',
        'unknown.txt'}
    assert result['always'] == ['L1', 'R']
    assert len(result['base']) == len(result['head']) == len(base) == 40
    assert len(result['table_sha256']) == 64


def test_rows_union_without_fail_closed(package):
    repo, _, table = package
    head = _git(repo, 'rev-parse', 'HEAD')
    for name in ('middle.py', 'plugin.yaml', 'scripts/tool.py', 'docs/readme.md'):
        with (repo / name).open('a') as stream:
            stream.write('\n# again\n' if name.endswith('.py') else 'again\n')
    _git(repo, 'commit', '-qam', 'small')
    result = rowselect.select(repo, head, 'HEAD', table)
    assert result['fail_closed'] == []
    assert result['paths']['middle.py']['classes'] == ['cli', 'storage']
    assert result['paths']['plugin.yaml']['classes'] == ['host']
    assert result['rows'] == ['d1', 'd4-d8', 'phase-a', 'phase-c-cells']


def test_cli_determinism_text_errors_and_help(package, capsys):
    repo, base, table = package
    args = ['--repo', str(repo), '--base', base, '--head', 'HEAD', '--table', str(table)]
    script = str(Path(rowselect.__file__))
    outputs = [subprocess.run([sys.executable, script, *args], capture_output=True, text=True,
                              check=True, env={**os.environ, 'PYTHONHASHSEED': seed}).stdout
               for seed in ('1', '2')]
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])['head'] == _git(repo, 'rev-parse', 'HEAD')
    assert rowselect.main(args) == 0
    assert capsys.readouterr().out == outputs[0]
    assert rowselect.main([*args, '--text']) == 0
    text = capsys.readouterr().out.splitlines()
    assert text[0] == f'base: {base}'
    assert text[1] == f"head: {_git(repo, 'rev-parse', 'HEAD')}"
    assert text[2].startswith('table_sha256: ') and len(text[2]) == len('table_sha256: ') + 64
    assert 'docs/readme.md\t-\tnon-product (-)' in text
    assert text[-3].startswith('rows: d1, d2-d3, d4-d8, d7, phase-a, phase-c-cells')
    assert text[-2] == 'always: L1, R'
    with pytest.raises(SystemExit) as error:
        rowselect.main([*args, '--head', 'missing-ref'])
    assert error.value.code == 2
    assert capsys.readouterr().err.startswith('rowselect: CalledProcessError: ')
    with pytest.raises(SystemExit) as error:
        rowselect.main([*args, '--table', str(repo / 'missing.json')])
    assert error.value.code == 2
    assert 'missing.json' in capsys.readouterr().err
    completed = subprocess.run([sys.executable, script, '--help'],
                               capture_output=True, text=True, check=True)
    assert completed.stdout.startswith('usage:')


def test_real_table_schema():
    table, _ = rowselect.load_table(rowselect.DEFAULT_TABLE)
    assert table['version'] == 1
    assert set(table['memory_classes']) == rowselect.MEMORY
    assert set(table['rows']) <= rowselect.CLASSES
    assert all(set(rows) <= rowselect.ROWS for rows in table['rows'].values())
    assert all(set(entry['classes']) <= rowselect.CLASSES for entry in table['entries'])
    assert isinstance(table['exempt'], dict)
    assert all(isinstance(glob, str) for glob in table['standalone'])
    repo = Path(__file__).resolve().parents[3]
    for hub in table['hubs']:
        _git(repo, 'cat-file', '-e', f'HEAD:{hub}')
