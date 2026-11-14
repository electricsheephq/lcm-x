"""Real-ingest differential matrix for #2539 (I1/I6); no compare-helper oracle.

Record on 7e67daf16 with LCM_2539_MATRIX_RECORD=<directory>, then combine the
per-cell JSON objects into fixtures/compare_matrix_main_7e67daf16.json.
Every current stored/host pair runs on every route; no current cells sampled.
"""
import copy
import hashlib
from collections import Counter, defaultdict
from functools import lru_cache
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

from tests.invariants.test_p1e_cold_tails import fail_end_read
from tests.test_issue_10_persisted_output_spillover import _marker
from tests.test_issue_1931_delegate_children import host as host
from tests.test_issue_2482_interrupted_child import _setup, call
from tests.test_issue_2521_spillover_upgrade_replay import (
    CONVERSATION, SESSION, SPILL_TEXT, _engine, _host_history, _v0292_store,
)

FIXTURE = Path(__file__).with_name('fixtures') / 'compare_matrix_main_7e67daf16.json'
ROUTES = ('child-live', 'child-counted', 'child-end',
          'offcurrent-end', 'cold-pairwise', 'cold-counted',
          'child-compress-s1', 'child-compress-s2', 'reconcile-replay')
FORMS = ('small', 'inline', 'recovered', 'bare')
HOSTS = ('identical', 'present', 'missing', 'inline', 'different')
CASES = [(r, s, p, loc, h) for r in ROUTES for s in FORMS
         for p in ('plain', 'data-uri') for loc in ('spillover', 'hermes-results') for h in HOSTS]
# Genuine v0.29.2 stores: spillover bare markers and temp-dir recovered results.
CASES += [(r, 'v0292', p, loc, h) for r in (*ROUTES[3:6], 'reconcile-replay')
          for p in ('plain', 'data-uri') for loc in ('spillover', 'hermes-results') for h in HOSTS]
CASES += [(r, 'v0292-crlf', 'plain', 'hermes-results', h)
          for r in (*ROUTES[3:6], 'reconcile-replay') for h in HOSTS]
# Existing keys are occurrence 1; occurrence 2 repeats the same host call id/result.
REPEATED_CASES = [(*case, 'occurrences-2') for case in CASES
                  if case[0] == 'reconcile-replay' or
                  (case[0].startswith(('child-', 'cold-')) and case[2] == 'data-uri')]
CASES += REPEATED_CASES
# Existing keys denote a full host window. Cover shifted partial histories with
# both occurrence counts, and the off-current full-window repeat control.
CASES += [(*case, 'occurrences-2') for case in CASES
          if len(case) == 5 and case[0] == 'offcurrent-end' and case[2] == 'data-uri']
CASES += [(*case[:5], f'occurrences-{n}', 'shifted') for case in CASES if len(case) == 5
          and (case[0] == 'reconcile-replay' or (case[0] in ROUTES[3:6] and case[2] == 'data-uri'))
          for n in (1, 2)]
# S1/S2 replace the synthetic child-compress and child-output-counted routes.
# Hermes 0.21.3 (345cd2b0), agent/context_compressor.py:651 (literal),
# :440 (rewrite), cited by R4-5 Q2; the test hoststub has no compressor module.
_CLEARED_OUTPUT = '[Old tool output cleared to save context space]'
_MARKER = re.compile(
    r'<persisted-output>\n[^\n]*?\(([0-9,]+) characters,[^\n]*\n'
    r'Full output saved to: ([^\n]+)\n.*?'
    r'Preview \(first ([0-9]+) chars\):\n(.*)\n\.\.\.\n</persisted-output>', re.S)
_EXCEPTION_CELLS = set()
_LEGACY_EXCEPTION_CELLS = set()
_GROWTH_EXCEPTION_CELLS = set()
# R4-8 and its amendment: the data checks decide; these names bound them, so a twelfth stored-marker
# or an eighth growth admission fails its own cell, also under -n 8.
_PINNED_LEGACY_CELLS = frozenset(
    f'reconcile-replay/{form}/{host}{window}'
    for form in ('v0292/data-uri/spillover', 'v0292-crlf/plain/hermes-results')
    for host in ('identical', 'present') for window in ('', '/occurrences-1/shifted')) | frozenset(
    f'reconcile-replay/bare/{form}/present/occurrences-1/shifted'
    for form in ('data-uri/spillover', 'data-uri/hermes-results', 'plain/hermes-results'))
_PINNED_GROWTH_CELLS = frozenset(
    f'reconcile-replay/{form}/{host}/occurrences-2/shifted'
    for form in ('v0292/data-uri/spillover', 'v0292-crlf/plain/hermes-results')
    for host in ('identical', 'present')) | frozenset(
    f'reconcile-replay/bare/{form}/present/occurrences-2/shifted'
    for form in ('data-uri/spillover', 'data-uri/hermes-results', 'plain/hermes-results')) | frozenset(
    # #2661 named copy (#2558): without the legacy bare-marker binding the full-window repeat is stored once more at
    # the second replay, as in v0.30.0; the R4-8 data check still decides each cell.
    f'reconcile-replay/v0292/data-uri/spillover/{host}/occurrences-2' for host in ('identical', 'present'))
assert len(_PINNED_LEGACY_CELLS) == 11 and len(_PINNED_GROWTH_CELLS) == 9
_HOST_RECOVERY_CELLS = set()

_EXT = re.compile(r'\[Externalized (?:LCM ingest payload|tool output|payload):[^\]]*?;\s*ref=([^;\]\s]+)\]')


class MainGap(AssertionError):
    """Only a fixture-proven main loss or replay growth can xfail."""


class NamedLimit(AssertionError):
    """R4-8: replay growth main lacks, admitted only by the data check in _repeated_marker_growth."""


def _cell_id(case):
    return '/'.join(case)


def _restore(text, storage):
    """Read the payload JSON named by each placeholder, including substring payloads."""
    if not isinstance(text, str):
        return text
    for _ in range(8):
        def expand(match):
            ref = match[1]
            payload = json.loads((storage / Path(ref).name).read_text())
            assert isinstance(payload['content'], str)
            return payload['content']
        new = _EXT.sub(expand, text)
        if new == text:
            return new
        text = new
    raise AssertionError('recursive payload fixture')


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _marker_fields(text):
    """Independent grammar: never consult product recovery/compare or source metadata."""
    match = _MARKER.fullmatch(text)
    if not match:
        return None
    chars, name, length, preview = match.groups()
    assert len(preview) == int(length), 'malformed synthetic marker preview'
    return {'filename': Path(name).name, 'chars': int(chars.replace(',', '')),
            'preview_length': int(length), 'preview_sha256': _sha(preview)}


def _position(row, positions):
    # A replay copy has the same logical tool-call slot, even if main appended it
    # at another physical store index. The slot comes from the host call order;
    # the exception also requires the exact role and call id, never body alone.
    return positions.get(row.get('tool_call_id') or '', -1)


def _snapshot(rows, storage, home, temp, positions):
    groups, markers, bodies, multiset = defaultdict(set), {}, [], Counter()
    for row in rows:
        text = _restore(row.get('content') or '', storage)
        role, call_id = row['role'], row.get('tool_call_id') or ''
        canonical = text.replace(str(home), '@HOME@').replace(str(temp), '@TEMP@')
        digest = _sha(canonical)
        groups[(role, call_id)].add(digest)
        multiset[(role, call_id, digest)] += 1
        marker = _marker_fields(text)
        if marker:
            markers[f'{role}/{call_id}/{digest}'] = dict(marker, role=role, call_id=call_id, position=_position(row, positions))
        bodies.append((role, call_id, _position(row, positions), text))
    contents = [[r, c, sorted(digests)] for (r, c), digests in sorted(groups.items())]
    return contents, markers, bodies, multiset


def _host_rows(view, body, home, temp):
    """One host row per slot, with canonical marker/body alternatives."""
    rows = []
    for row in view:
        content = row.get('content') or ''
        alternatives = {content}
        if _marker_fields(content) and 'DIFFERENT PREVIEW' not in content:
            alternatives.add(body)
        rows.append({(row['role'], row.get('tool_call_id') or '',
                      _sha(text.replace(str(home), '@HOME@').replace(str(temp), '@TEMP@')))
                     for text in alternatives})
    return rows


def _match_host_rows(multiset, host_rows):
    """Match distinct host slots, including overlapping marker/body alternatives."""
    held = {}

    def match(row, visited):
        for slot, alternatives in enumerate(host_rows):
            if slot in visited or row not in alternatives:
                continue
            visited.add(slot)
            if slot not in held or match(held[slot], visited):
                held[slot] = row
                return True
        return False

    unmatched = [row for row in sorted(multiset.elements()) if not match(row, set())]
    return unmatched, [row for slot, row in enumerate(host_rows) if slot not in held]


def _unmatched_extras(main, candidate, host_rows, produced=frozenset()):
    # Main consumes host slots one-for-one before candidate-only rows can use them.
    _, not_held = _match_host_rows(main, host_rows)
    # Compressor-produced rows that main consumes are not host rows main loses.
    allowed = [alternatives - produced for alternatives in not_held]
    unmatched, _ = _match_host_rows(candidate - main, allowed)
    return unmatched


def _lost_contents(baseline, current, markers, bodies):
    current = {(r, c, d) for r, c, digests in current for d in digests}
    missing, exceptions = [], 0
    for role, call_id, digests in baseline:
        for digest in digests:
            if (role, call_id, digest) in current:
                continue
            marker = markers.get(f'{role}/{call_id}/{digest}')
            if marker and role == 'tool' and marker['call_id'] == call_id and any(
                    (r, c, pos) == (role, call_id, marker['position'])
                    and len(text) == marker['chars']
                    and _sha(text[:marker['preview_length']]) == marker['preview_sha256']
                    for r, c, pos, text in bodies):
                exceptions += 1
            else:
                missing.append((role, call_id, digest))
    return missing, exceptions


@lru_cache(maxsize=1)
def _fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope='module', autouse=True)
def _print_exception_cells(request, tmp_path_factory):
    yield
    # Workers have no terminalreporter. Preserve their counters at the basetemp
    # root for session-end aggregation by the evidence runner (also on xfail).
    (tmp_path_factory.getbasetemp() / 'matrix-exceptions.json').write_text(json.dumps({
        'marker': sorted(_EXCEPTION_CELLS), 'legacy': sorted(_LEGACY_EXCEPTION_CELLS),
        'growth': sorted(_GROWTH_EXCEPTION_CELLS), 'host_recovery': sorted(_HOST_RECOVERY_CELLS)}))
    reporter = request.config.pluginmanager.getplugin('terminalreporter')
    if reporter:
        reporter.write_line(f'#2539 corrected bar (ii): {len(_EXCEPTION_CELLS)} cells used the marker exception')
        reporter.write_line(f'#2539 corrected bar (iii): {len(_HOST_RECOVERY_CELLS)} cells where the candidate keeps rows main loses (#2543)')
        for name, cells in (('bar (ii) stored-marker', _LEGACY_EXCEPTION_CELLS), ('bar (iv) R4-8 growth', _GROWTH_EXCEPTION_CELLS)):
            reporter.write_line(f'#2558 {name} exception: {len(cells)} cells')
            for cell in sorted(cells):
                reporter.write_line(f'  {cell}')


def _legacy_marker_losses(original, rows, handed, main, candidate, storage, home, temp):
    """R4-8 (restating R4-7): a stored bare marker kept byte-identical under its own call id, in place of
    its own file's body only, when the host re-sent that exact marker and the candidate adds no row."""
    if candidate - main:
        return set()  # No candidate-only row in this step, even a host recovery.
    allowed = set()
    for old in original:
        if old['role'] != 'tool' or old not in rows:
            continue  # Full row equality includes session, call id, text and position.
        marker = _restore(old['content'], storage)
        match = _MARKER.fullmatch(marker)
        if not match:
            continue
        if not any(m.get('role') == 'tool' and m.get('tool_call_id') == old['tool_call_id']
                   and m.get('content') == marker for m in handed):
            continue  # The host must have re-sent this exact marker, not the body or other text.
        path = Path(match[2])  # The actual dump's file, never a guessed body.
        if not path.is_file():
            continue
        body = path.read_text(encoding='utf-8')
        fields = _marker_fields(marker)
        if len(body) != fields['chars'] or _sha(body[:fields['preview_length']]) != fields['preview_sha256']:
            continue
        canonical = body.replace(str(home), '@HOME@').replace(str(temp), '@TEMP@')
        identity = ('tool', old['tool_call_id'], _sha(canonical))
        if main[identity] and not candidate[identity]:
            allowed.add(identity)
    return allowed


def _repeated_marker_growth(original, handed, storage, counts, main_counts, missing, third_growth):
    """R4-8 named limit: the host reuses a call id and repeats the identical marker of a stored bare row.

    Admitted only when no host row is missing, a third replay adds nothing and no step is above main."""
    if missing or third_growth or len(counts) != len(main_counts) or any(c > m for c, m in zip(counts, main_counts)):
        return False
    for old in original:
        marker = _restore(old['content'], storage) if old['role'] == 'tool' else ''
        if _MARKER.fullmatch(marker) and sum(
                m.get('role') == 'tool' and m.get('tool_call_id') == old['tool_call_id'] and m.get('content') == marker
                for m in handed) >= 2:
            return True
    return False


def _view(content):
    return [{'role': 'user', 'content': 'Dump the ledger.'}, call(1),
            {'role': 'tool', 'tool_call_id': 'c1', 'content': content}]


def _missing(final_view, rows, storage, body):
    """Host markers may be durable bare text or their exact recovered body."""
    contents = {(r['role'], r.get('tool_call_id') or '', _restore(r.get('content') or '', storage)) for r in rows}
    missing = []
    for m in final_view:
        content = m.get('content') or ''
        alternatives = {content}
        # These synthetic valid markers have one exact body; changed-preview controls do not.
        if content.startswith('<persisted-output>') and 'DIFFERENT PREVIEW' not in content:
            alternatives.add(body)
        if not any((m['role'], m.get('tool_call_id') or '', c) in contents for c in alternatives):
            missing.append([m['role'], m.get('tool_call_id') or '', content])
    return missing


def _v0292_matrix_store(tmp_path, temp, preview, location, crlf=False):
    """Load real-engine dumps; payload files are storage evidence, never re-recovered."""
    if (preview, location, crlf) == ('plain', 'spillover', False):
        _, marker, path = _v0292_store(tmp_path)
        return marker, path, SPILL_TEXT
    name = ('temp_plain_crlf' if crlf else
            'spillover_data_uri' if location == 'spillover' else
            'temp_plain_lf' if preview == 'plain' else 'temp_data_uri')
    fixtures = Path(__file__).parents[1] / 'fixtures'
    home = tmp_path / 'hermes'
    path = (home / 'cache/spillover' if location == 'spillover' else temp / 'hermes-results') / 'call_spill.txt'
    body = ('data:application/octet-stream;base64,' + 'QUJD' * 80 + '\n' if preview == 'data-uri' else '') + SPILL_TEXT
    path.parent.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    path.write_bytes((body.replace('\n', '\r\n') if crlf else body).encode())

    def substitute(text):
        return text.replace('@HERMES_HOME@', str(home)).replace('@HOST_TEMP@', str(temp))

    with sqlite3.connect(home / 'lcm.db') as connection:
        connection.executescript(substitute((fixtures / f'v0292_{name}_store.sql').read_text()))
    with sqlite3.connect(home / 'lcm.db') as connection:
        for table in ('messages_fts', 'messages_fts_stem'):
            connection.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
            connection.execute(f"INSERT INTO {table}({table}) VALUES('integrity-check')")
    storage = home / 'externalized'
    storage.mkdir()
    for payload in (fixtures / f'v0292_{name}_payloads').glob('*.json'):
        (storage / payload.name).write_text(substitute(payload.read_text()))
    return _marker(path, body, preview=body[:400] if preview == 'data-uri' else body[:30]), path, body


@pytest.mark.parametrize('case', CASES, ids=_cell_id)
def test_session_end_compare_matrix(case, host, tmp_path, monkeypatch, request):
    _run_case(case, host, tmp_path, monkeypatch, request)


@pytest.mark.parametrize('case,expected', [
    # unreachable: an orphan produced tool row is dropped by the sanitizer; tracked in #2553
    # B-iii main: [3,4,4,4,6].
    (('child-compress', 'small', 'plain', 'spillover', 'present'), [3, 5, 5, 5, 7]),
    # unreachable: an orphan produced tool row is dropped by the sanitizer; tracked in #2553
    # B-ii main: [3,6,6,6,8].
    (('child-compress', 'recovered', 'plain', 'spillover', 'inline'), [3, 4, 4, 4, 6]),
], ids=['child-compress/small/plain/spillover/present',
        'child-compress/recovered/plain/spillover/inline'])
def test_synthetic_orphan_characterization(case, expected, host, tmp_path, monkeypatch, request):
    _run_case(case, host, tmp_path, monkeypatch, request, characterization=expected)


@pytest.mark.parametrize('preview', ['plain', 'data-uri'], ids=['path-temp-lf', 'path-temp-data-uri'])
def test_recovered_temp_different_path_characterization(preview, host, tmp_path, monkeypatch, request):
    """#2703 resolved the extra pair in #2553; no host row is lost."""
    # #2703 keeps the stored identity when the recovered content is equal, so a changed path stores two fewer copies;
    # no host row missing
    _run_case(('reconcile-replay', 'v0292', preview, 'hermes-results', 'path'),
              host, tmp_path, monkeypatch, request, characterization=[4, 5, 5, 7])


def _run_case(case, host, tmp_path, monkeypatch, request, characterization=None):
    route, stored_form, preview, location, host_form = case[:5]
    repeated = 'occurrences-2' in case
    shifted = 'shifted' in case
    legacy = stored_form.startswith('v0292')
    temp = tmp_path / 'host-temp'
    temp.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(temp))
    child_route = route.startswith('child-')
    if child_route:
        _, _, ctx, parent, child, engine = _setup(host, 'own_engine')
        reader, session = parent, 'child'
    else:
        if legacy:
            original, path, legacy_body = _v0292_matrix_store(
                tmp_path, temp, preview, location, stored_form == 'v0292-crlf')
        engine = _engine(tmp_path)
        reader, session = engine, SESSION if legacy else 'S1'
    home = Path(engine._hermes_home)
    storage = Path(engine._config.large_output_externalization_path or home / 'lcm' / 'externalized')
    # Resolve the actual default directory without calling a compare or payload helper.
    if not engine._config.large_output_externalization_path:
        from hermes_lcm.externalize import DEFAULT_LARGE_OUTPUT_DIRNAME
        storage = home / DEFAULT_LARGE_OUTPUT_DIRNAME
    body = ('small result' if stored_form == 'small' else
            ('data:application/octet-stream;base64,' + 'QUJD' * 80 + '\n' if preview == 'data-uri' else '')
            + 'MATRIX_BODY:\n' + 'line é文 abcdef 0123456789\n' * 700)
    if legacy:
        body = legacy_body
        first = _host_history(original)
    else:
        path = (home / 'cache' / 'spillover' if location == 'spillover' else temp / 'hermes-results') / 'c1.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = _marker(path, body, preview=body[:400] if preview == 'data-uri' else body[:30])
        if stored_form == 'recovered':
            path.write_text(body)
        original = body if stored_form in ('small', 'inline') else marker
        first = _view(original)
        if route == 'child-counted':
            first.append({'role': 'assistant', 'content': 'old tail'})
        if not child_route:
            # More than the eight-row fingerprint window: exercise the store compare.
            first = [{'role': ('user', 'assistant')[i % 2], 'content': f'pad {i}'} for i in range(7)] + first
    if shifted:
        if not legacy:
            first = first[-3:] + [{'role': 'assistant', 'content': 'FINAL'}]
        # Reuse non-tool text to keep the interned oracle below its size cap;
        # the exact reviewer regression separately uses 200 distinct rows.
        padding = [m for m in first if m['role'] in ('user', 'assistant') and not m.get('tool_calls')]
        first += [dict(padding[i % len(padding)], timestamp=1762700001.0 + len(first) + i)
                  for i in range(200)] + copy.deepcopy(first[-4:])
    if route == 'reconcile-replay' or shifted:
        for i, message in enumerate(first):
            if shifted:
                message['timestamp'] = 1762700001.0 + i
            else:
                message.setdefault('timestamp', 1762700001.0 + i)
    counts, absent, steps, host_steps, row_steps, handed_steps = [], [], [], [], [], []
    produced = set()

    def record_step(handed):
        if child_route:
            # The host hook waits at most 2 s for a live-child store (#2482); the
            # store may land later under load, so wait for it before reading.
            aux = ctx.module('aux_session')  # the host's own copy of the plugin, as in the #2482 tests
            warned = set(aux._LIVE_CHILD_WARNED)
            aux.drain_live_child_stores(60.0)  # returns None; a timeout only adds to _LIVE_CHILD_WARNED
            assert aux._LIVE_CHILD_WARNED <= warned, 'a live-child store did not land within 60 s'
        rows = reader._store.get_session_messages(session)
        row_steps.append(rows)
        handed_steps.append(copy.deepcopy(handed))
        counts.append(len(rows))
        multiset = _snapshot(rows, storage, home, temp, {})[3]
        steps.append([[r, c, d, n] for (r, c, d), n in sorted(multiset.items())])
        host_steps.append(_host_rows(handed, body, home, temp))

    try:
        if child_route:
            child.api_call(ctx, first, 1)
        else:
            engine.on_session_start(session, platform='cli', context_length=200_000,
                                    conversation_id=CONVERSATION if legacy else 'conv1')
            if not legacy or shifted:
                if legacy:
                    gone = path.with_suffix('.gone')
                    path.rename(gone)
                engine.ingest(copy.deepcopy(first))
                if legacy:
                    gone.rename(path)
            if route != 'reconcile-replay':
                engine.on_session_start('foreground', platform='cli', context_length=200_000)
        record_step(first)
        if host_form in ('present', 'identical', 'different'):
            if host_form != 'identical':
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
        elif host_form == 'missing' and path.exists():
            path.rename(tmp_path / 'pruned-source')
        marker = _marker(path, body, preview=body[:400] if preview == 'data-uri' else body[:30])
        replay_content = {'identical': original, 'present': marker, 'missing': marker,
                          'inline': body, 'different': _marker(path, body, preview='DIFFERENT PREVIEW'),
                          'path': marker.replace(str(path), str(path.with_name('other.txt')))}[host_form]
        if host_form == 'path':
            path.with_name('other.txt').write_text(body)
        view = copy.deepcopy(first[120:124] + first[-4:] if shifted else first)
        tool = next(m for m in view if m['role'] == 'tool')
        tool['content'] = replay_content
        if route == 'child-counted':
            view = [tool]
        view += [{'role': 'assistant', 'content': 'FINAL'}]
        if characterization is not None and child_route:
            # Deliberately bypass the sanitizer ONLY for the two unreachable controls.
            tool = dict(tool, tool_call_id='compact-result')
            output = [copy.deepcopy(tool), {'role': 'assistant', 'content': 'compression output tail'}]
            produced = set().union(*_host_rows(output, body, home, temp))
            monkeypatch.setattr(engine, '_compress_bypassed', lambda *a: copy.deepcopy(output))
            child.compress(engine, first)
            view = [copy.deepcopy(tool), {'role': 'assistant', 'content': 'FINAL'}]
        if route.startswith('child-compress-'):
            invocations, sanitized, remembered = [], [], []
            summary = {'role': 'user', 'content': '[CONTEXT COMPACTION] ledger summary'}
            compression_input = copy.deepcopy(view[:-1])
            # A fresh assistant row makes this pre-store's input covered even
            # when main must append a changed tool form. Retain it in the
            # compressor result, then drop it at handover so both the stored
            # tail and the compression boundary miss: counted fallback.
            compression_input.append({'role': 'assistant', 'content': 'compression input tail'})

            class ContextCompressor:
                def __init__(self, **kwargs):
                    self.compression_count = 0

                def compress(self, messages, **kwargs):
                    invocations.append(copy.deepcopy(messages))
                    self.compression_count += 1
                    # S1 keeps the original pair byte-for-byte; S2 changes only
                    # the tool content, retaining the existing assistant call/id.
                    pair = copy.deepcopy(messages[1:3])
                    if route.endswith('-s2'):
                        pair[1]['content'] = _CLEARED_OUTPUT
                    return [copy.deepcopy(summary), *pair, copy.deepcopy(messages[-1])]

            compressor_module = types.ModuleType('agent.context_compressor')
            compressor_module.ContextCompressor = ContextCompressor
            monkeypatch.setitem(sys.modules, 'agent.context_compressor', compressor_module)
            sanitize = engine._sanitize_active_context_messages
            remember = engine._remember_delegate_child_outputs

            def observe_sanitize(messages):
                result = sanitize(messages)
                sanitized.append(copy.deepcopy(result))
                return result

            def observe_outputs(sid, messages, result):
                remembered.append(copy.deepcopy(result))
                return remember(sid, messages, result)

            monkeypatch.setattr(engine, '_sanitize_active_context_messages', observe_sanitize)
            monkeypatch.setattr(engine, '_remember_delegate_child_outputs', observe_outputs)
            # The test host supplies a model without a network client; only the
            # injected native compressor runs, through bypass.py:284-311/632-672.
            monkeypatch.setattr(engine, 'model', 'matrix-host-model')
            output = child.compress(engine, compression_input)
            assert len(invocations) == 1 and invocations[0] == compression_input
            assert sanitized and sanitized[-1] == output
            assert remembered == [output], 'covered input must remember the sanitized compressor result'
            expected_tool = dict(tool, content=_CLEARED_OUTPUT) if route.endswith('-s2') else tool
            assert [(m['role'], m.get('content'), m.get('tool_call_id'), m.get('tool_calls')) for m in output] == [
                (m['role'], m.get('content'), m.get('tool_call_id'), m.get('tool_calls'))
                for m in [summary, compression_input[1], expected_tool, compression_input[-1]]]
            # Independent result-minus-input identities, including marker/body
            # alternatives: none may be admitted as a bar-(iii) host recovery.
            before = set().union(*_host_rows(compression_input, body, home, temp))
            produced = set().union(*_host_rows(output, body, home, temp)) - before
            view = [*copy.deepcopy(output[:-1]), {'role': 'assistant', 'content': 'FINAL'}]
            fallbacks = []
            note = engine._note_delegate_child_fallback
            def observe(sid, reason):
                fallbacks.append(reason)
                return note(sid, reason)
            monkeypatch.setattr(engine, '_note_delegate_child_fallback', observe)

        def replay(handed, n):
            nonlocal engine, reader
            if route == 'reconcile-replay':
                # The public cold-process route used by #2521's _replay.
                engine.shutdown()
                engine = reader = _engine(tmp_path)
                engine.on_session_start(session, platform='cli', context_length=200_000,
                                        conversation_id=CONVERSATION if legacy else 'conv1')
                engine.ingest(copy.deepcopy(handed))
            elif route in ('child-live', 'child-counted', 'child-compress') or route.startswith('child-compress-'):
                child.api_call(ctx, handed, n)
            elif route == 'child-end':
                child.end(engine, handed)
            elif route.startswith('cold-'):
                with monkeypatch.context() as patch:
                    # The legacy fixture session id differs from fail_end_read's S1.
                    if session == 'S1':
                        patch.setattr(engine._store, 'get_range', fail_end_read(engine))
                    else:
                        read = engine._store.get_range
                        def busy(sid, *a, **kw):
                            if sid == session:
                                import sqlite3
                                raise sqlite3.OperationalError('database is busy')
                            return read(sid, *a, **kw)
                        patch.setattr(engine._store, 'get_range', busy)
                    engine.on_session_end(session, handed)
                assert engine._pending_session_tails
                tail = engine._pending_session_tails[-1]
                # The default retry budget is 50 ms; under load it can expire
                # before the retry commits. The matrix compares rows, not timing.
                engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
                if route == 'cold-counted':
                    absent.append(engine._cold_tail_absent_count(tail))
            else:
                engine.on_session_end(session, handed)
            record_step(handed)

        if repeated:
            second_call = copy.deepcopy(next(m for m in first if m.get('tool_calls')))
            second_tool = copy.deepcopy(tool)
            # S2 hands back a cleared first result; the second occurrence is the
            # original host result, with the same id and unmodified marker.
            second_tool['content'] = replay_content
            stamp = max(m.get('timestamp', 1762700000.0) for m in view)
            second_call['timestamp'], second_tool['timestamp'] = stamp + 1, stamp + 2
            view += [second_call, second_tool, {'role': 'assistant', 'content': 'second occurrence',
                                              'timestamp': stamp + 3}]
        if route == 'reconcile-replay':
            for i, message in enumerate(view):
                message.setdefault('timestamp', 1762700001.0 + i)
        replay(view, 2)
        if route.startswith('child-compress-'):
            assert any('resuming with counted identities' in reason for reason in fallbacks)
        replay(view, 3)
        if repeated or shifted:
            replay(view, 4)
        if child_route:
            child.end(engine, view)
            record_step(view)
        updated = [*view, {'role': 'user', 'content': 'new question'},
                   {'role': 'assistant', 'content': 'new answer'}]
        if route == 'reconcile-replay':
            for i, message in enumerate(updated):
                message.setdefault('timestamp', 1762700001.0 + i)
        replay(updated, 4)
        rows = reader._store.get_session_messages(session)
        missing = _missing(updated, rows, storage, body)
        # Ordered host call slots are independent of physical replay-copy rows.
        call_ids = list(dict.fromkeys(m.get('tool_call_id') for m in [*first, *updated] if m.get('tool_call_id')))
        positions = {call_id: i for i, call_id in enumerate(call_ids)}
        contents, markers, bodies, _ = _snapshot(rows, storage, home, temp, positions)
        missing_digests = sorted([r, c, _sha(t.replace(str(home), '@HOME@').replace(str(temp), '@TEMP@'))]
                                 for r, c, t in missing)
        result = {'steps': steps, 'counts': counts, 'contents': contents, 'markers': markers,
                  'missing': missing_digests, 'replay_growth': counts[2] - counts[1], 'absent': absent}
        if repeated or shifted:
            result['third_replay_growth'] = counts[3] - counts[2]
            print(f'OCCURRENCES {_cell_id(case)} counts={counts} third_growth={result["third_replay_growth"]}')
        if characterization is not None:
            assert counts == characterization
            if host_form == 'path':
                assert not missing, 'characterized surplus must lose no host row'
            return
        record = os.environ.get('LCM_2539_MATRIX_RECORD')
        if record:
            target = Path(record)
            target.mkdir(parents=True, exist_ok=True)
            (target / ('--'.join(case) + '.json')).write_text(json.dumps({_cell_id(case): result}, sort_keys=True))
            return
        fixture = _fixture()
        baseline = fixture['cells'][_cell_id(case)]
        extras = False
        unmatched = []
        legacy_losses = set()
        for main_index, current, host_rows, step_rows, handed in zip(baseline['steps'], steps, host_steps, row_steps,
                                                                   handed_steps, strict=True):
            main = Counter({(r, c, d): n for r, c, d, n in fixture['multisets'][main_index]})
            candidate = Counter({(r, c, d): n for r, c, d, n in current})
            extras |= bool(candidate - main)
            unmatched.extend(_unmatched_extras(main, candidate, host_rows, produced))
            permitted = _legacy_marker_losses(row_steps[0], step_rows, handed, main, candidate, storage, home, temp)
            # Bar (ii) concerns what main ends with; retain the last step's
            # proof, including its no-candidate-only-row condition.
            legacy_losses = permitted
        assert not unmatched, (
            f'STOP bar iii {_cell_id(case)}: main={baseline["counts"]} candidate={counts} extra={unmatched}')
        if extras:
            _HOST_RECOVERY_CELLS.add(_cell_id(case))
        baseline_contents = fixture['content_sets'][baseline['contents']]
        lost, exceptions = _lost_contents(baseline_contents, contents, fixture['markers'], bodies)
        if set(lost) & legacy_losses:
            assert _cell_id(case) in _PINNED_LEGACY_CELLS, f'STOP bar ii {_cell_id(case)}: a new stored-marker admission'
            _LEGACY_EXCEPTION_CELLS.add(_cell_id(case))
        lost = [identity for identity in lost if identity not in legacy_losses]
        if exceptions:
            _EXCEPTION_CELLS.add(_cell_id(case))
        assert not lost, (f'STOP bar ii {_cell_id(case)}: main={baseline["counts"]}, candidate={counts}; lost={lost}')
        assert all(n <= m for n, m in zip(absent, baseline['absent'], strict=True)), (
            f'STOP counted diagnostic {_cell_id(case)}: main={baseline["absent"]}, candidate={absent}')
        gaps = []
        if missing:
            assert set(map(tuple, result['missing'])) <= set(map(tuple, fixture['missing_sets'][baseline['missing']])), 'new host-row loss'
            missing_ids = set(map(tuple, result['missing']))
            if missing_ids & produced:
                gaps.append('compaction outputs consumed')
            if missing_ids - produced:
                gaps.append('host rows absent (#2543 on off-current routes)' if not child_route else 'host rows absent')
        limit = False
        if result['replay_growth']:
            if baseline['replay_growth'] < result['replay_growth']:
                limit = _repeated_marker_growth(row_steps[0], handed_steps[1], storage, counts, baseline['counts'],
                                                missing, result.get('third_replay_growth', 1))
                assert limit, (
                    f'STOP bar iv {_cell_id(case)}: main={baseline["counts"]} candidate={counts}; new second-replay growth')
                assert _cell_id(case) in _PINNED_GROWTH_CELLS, f'STOP bar iv {_cell_id(case)}: a new named-limit admission'
                _GROWTH_EXCEPTION_CELLS.add(_cell_id(case))
            else:
                gaps.append('second replay adds rows')
        if (repeated or shifted) and result['third_replay_growth']:
            assert baseline['third_replay_growth'] >= result['third_replay_growth'] > 0, (
                f'STOP bar iv {_cell_id(case)}: main={baseline["counts"]} candidate={counts}; new third-replay growth')
            gaps.append('third replay adds rows')
        if limit and not gaps:
            request.node.add_marker(pytest.mark.xfail(strict=True, raises=NamedLimit,
                reason='R4-8 named limit: a repeated marker of a stored bare row in a shifted host window' if shifted
                else '#2661 named copy (#2558): a repeated data-URI marker of a stored bare row in a full host window'))
            raise NamedLimit('second replay adds rows; no host row missing, third replay adds nothing, never above main')
        if gaps:
            request.node.add_marker(pytest.mark.xfail(strict=True, raises=MainGap,
                reason='main 7e67daf16 fixture: ' + ', '.join(gaps)))
            raise MainGap(', '.join(gaps))
    finally:
        if not child_route:
            engine.shutdown()


@pytest.mark.parametrize('wrong', ['position', 'call_id', 'length', 'preview', 'non-marker'])
def test_marker_exception_requires_exact_data(wrong):
    marker = _marker(Path('c1.txt'), 'abcde', preview='abc')
    fields = dict(_marker_fields(marker), role='tool', call_id='c1', position=0)
    body = ['tool', 'c1', 0, 'abcde']
    if wrong == 'position':
        body[2] = 1
    elif wrong == 'call_id':
        body[1] = 'other'
    elif wrong == 'length':
        body[3] += 'f'
    elif wrong == 'preview':
        body[3] = 'xbcde'
    digest = _sha(marker)
    metadata = {} if wrong == 'non-marker' else {f'tool/c1/{digest}': fields}
    baseline = [['tool', 'c1', [digest]]]
    assert _lost_contents(baseline, [], metadata, [body]) == ([('tool', 'c1', digest)], 0)
    assert _lost_contents(baseline, [], {f'tool/c1/{digest}': fields}, [('tool', 'c1', 0, 'abcde')]) == ([], 1)


def test_step_multiset_requires_distinct_host_rows():
    held = ('tool', 'c1', _sha('stored'))
    lost = ('assistant', '', _sha('FINAL'))
    main = Counter({held: 1})
    host_rows = [{held}, {lost}]
    assert _unmatched_extras(main, main + Counter({held: 1}), host_rows) == [held]
    assert _unmatched_extras(main, main + Counter({lost: 1}), host_rows) == []
    assert _unmatched_extras(main, main + Counter({lost: 2}), host_rows) == [lost]
    assert _unmatched_extras(main, main + Counter({lost: 1}), host_rows, {lost}) == [lost]


@pytest.mark.parametrize('wrong', [None, 'changed-row', 'different-file-body', 'call-id', 'extra-row', 'host-sent-body'])
def test_legacy_exception_requires_dump_row_and_its_file(wrong, tmp_path):
    path = tmp_path / 'saved.txt'
    path.write_text('abcde')
    old = {'role': 'tool', 'session_id': 'legacy-session', 'store_id': 3,
           'tool_call_id': 'c1', 'content': _marker(path, 'abcde', preview='abc')}
    row = dict(old)
    missing = ('tool', 'c1', _sha('abcde'))
    held = ('tool', 'c1', _sha(old['content']))
    main, candidate = Counter({missing: 1, held: 1}), Counter({held: 1})
    if wrong == 'changed-row':
        row['session_id'] = 'another-session'
    elif wrong == 'different-file-body':
        path.write_text('abcdf')
    elif wrong == 'call-id':
        main = Counter({('tool', 'other', missing[2]): 1, held: 1})
    elif wrong == 'extra-row':
        candidate[('assistant', '', _sha('new'))] += 1
    handed = [{'role': 'tool', 'tool_call_id': 'c1',
               'content': 'abcde' if wrong == 'host-sent-body' else old['content']}]
    assert _legacy_marker_losses([old], [row], handed, main, candidate, tmp_path, tmp_path, tmp_path) == (
        {missing} if wrong is None else set())


@pytest.mark.parametrize('wrong', [None, 'missing', 'third-replay', 'above-main', 'one-occurrence', 'other-text',
                                   'stored-body'])
def test_repeated_marker_growth_requires_each_condition(wrong, tmp_path):
    path = tmp_path / 'saved.txt'
    marker = _marker(path, 'abcde', preview='abc')
    old = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'abcde' if wrong == 'stored-body' else marker}
    sent = {'role': 'tool', 'tool_call_id': 'c1', 'content': marker}
    handed = [sent, dict(sent, content=marker + ' ') if wrong == 'other-text' else dict(sent)]
    if wrong == 'one-occurrence':
        handed = handed[:1]
    counts, main_counts = [208, 212, 214, 214, 216], [208, 214, 214, 214, 216]
    if wrong == 'above-main':
        counts = [208, 212, 214, 214, 217]
    assert _repeated_marker_growth([old], handed, tmp_path, counts, main_counts,
                                   [('tool', 'c1', 'x')] if wrong == 'missing' else [],
                                   1 if wrong == 'third-replay' else 0) == (wrong is None)


def write_fixture(records, destination):
    """Intern repeated digest lists; each cell resolves to sorted role/call/digest sets."""
    fixture = json.loads(Path(destination).read_text())
    cells, markers = {}, dict(fixture['markers'])
    for path in sorted(Path(records).glob('*.json')):
        for cell, result in json.loads(path.read_text()).items():
            assert cell not in cells
            for digest, marker in result.pop('markers').items():
                assert digest not in markers or markers[digest] == marker
                markers[digest] = marker
            if cell in fixture['cells']:
                old = dict(fixture['cells'][cell])
                old['contents'] = fixture['content_sets'][old['contents']]
                old['missing'] = fixture['missing_sets'][old['missing']]
                old['steps'] = [fixture['multisets'][i] for i in old['steps']]
                assert old == result, f'STOP: existing main fixture changed: {cell}'
            else:
                cells[cell] = result
    new_cells = list(cells.values())
    cells = dict(fixture['cells'], **cells)
    pools = {}
    for field in ('contents', 'missing'):
        serial = sorted({json.dumps(result[field], sort_keys=True) for result in new_cells})
        pool_name = 'content_sets' if field == 'contents' else 'missing_sets'
        serial = [json.dumps(value, sort_keys=True) for value in fixture[pool_name]] + [
            value for value in serial if json.loads(value) not in fixture[pool_name]]
        indexes = {value: i for i, value in enumerate(serial)}
        pools['content_sets' if field == 'contents' else 'missing_sets'] = [json.loads(value) for value in serial]
        for result in new_cells:
            result[field] = indexes[json.dumps(result[field], sort_keys=True)]
    serial = sorted({json.dumps(step) for result in new_cells for step in result['steps']})
    serial = [json.dumps(value) for value in fixture['multisets']] + [
        value for value in serial if json.loads(value) not in fixture['multisets']]
    indexes = {value: i for i, value in enumerate(serial)}
    pools['multisets'] = [json.loads(value) for value in serial]
    for result in new_cells:
        result['steps'] = [indexes[json.dumps(step)] for step in result['steps']]
    fixture = dict(pools, cells=cells, markers=markers, main_sha='7e67daf16478eff0d153b5973648c6d251f13d89')
    encoded = json.dumps(fixture, sort_keys=True, separators=(',', ':')) + '\n'
    assert len(encoded.encode('utf-8')) <= 1_000_000, 'STOP: fixture exceeds 1 MB'
    Path(destination).write_text(encoded)


@pytest.mark.parametrize('preview,location,crlf', [
    ('plain', 'spillover', False), ('data-uri', 'spillover', False),
    ('plain', 'hermes-results', True),
], ids=['spillover-plain', 'spillover-data-uri', 'temp-crlf'])
@pytest.mark.parametrize('incoming', ['same', 'preview', 'path', 'count'])
def test_v0292_keep_and_session_end_agree(preview, location, crlf, incoming, tmp_path, monkeypatch):
    temp = tmp_path / 'host-temp'
    temp.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(temp))
    marker, path, body = _v0292_matrix_store(tmp_path, temp, preview, location, crlf)
    engine = _engine(tmp_path)
    try:
        engine.on_session_start(SESSION, platform='cli', context_length=200_000, conversation_id=CONVERSATION)
        stored = next(row for row in engine._store.get_session_messages(SESSION) if row['role'] == 'tool')
        text = marker
        if incoming == 'preview':
            text = _marker(path, body, preview='DIFFERENT PREVIEW')
        elif incoming == 'path':
            other = path.with_name('other.txt')
            other.write_bytes(body.encode())
            text = marker.replace(str(path), str(other))
        elif incoming == 'count':
            text = _marker(path, body, count=len(body) + 1,
                           preview=body[:400] if preview == 'data-uri' else body[:30])
        host_message = dict(_host_history(text)[2])
        keep_equal = (engine._message_replay_identity(stored, stored_row=True)
                      == engine._message_replay_identity(host_message))
        end_equal = engine._session_end_rows_equal(stored, host_message, session_id=SESSION)
        assert keep_equal == end_equal == (incoming == 'same'), (
            f'keep={keep_equal}, session-end={end_equal}, incoming={incoming}')
    finally:
        engine.shutdown()
