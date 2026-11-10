"""#2547: a seeded comparison of the session-end compare against main.

Main is this tree with ``_without_lcm_marker_lines`` replaced by the identity function: every #2547 change goes
through that helper, so the replacement gives main's compare exactly. For each generated pair, the candidate must equal
main, EXCEPT when both sides are Hermes markers, at least one carries the lines ingest writes at externalization off
(built here with ingest's own writers), and the two texts are equal without those lines. Then the candidate matches
and main does not. A reference written here, apart from the product helper, decides that class.
"""

import json
import os
import random
from collections import Counter

import pytest

from hermes_lcm import prefix_matching
from hermes_lcm.ingest_persisted import (
    _add_inline_persisted_output_generation_metadata,
    _add_inline_persisted_output_identity_metadata,
    _has_lossy_sensitive_redaction,
    _persisted_output_marker_identity_digest,
    _persisted_output_preview_prefix,
)
from hermes_lcm.ingest_policy import _SENSITIVE_PLACEHOLDER_PREFIX
from hermes_lcm.ingest_protection import redact_sensitive_value
from tests.test_issue_10_persisted_output_spillover import _marker
from tests.test_issue_2521_spillover_upgrade_replay import _engine

SECRET = 'api_key=differentialsecret1234567890abcdef'
PASSWORD = 'password=hunter2hunter2'  # its placeholder carries no digest: a lossy redaction
WRONG = 'f' * 64
KIND = None
STAT = {'size': 300, 'mtime_ns': 1762700000000000000, 'ctime_ns': 1762700000000000001}


@pytest.fixture(params=[False, True], ids=['no-redaction', 'redaction'])
def engine(tmp_path, request):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = request.param
    engine._config.sensitive_patterns = ['all'] if request.param else []
    engine.on_session_start('S1', platform='cli', context_length=200_000, conversation_id='c')
    try:
        yield engine
    finally:
        engine.shutdown()


def _main(fn, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(prefix_matching, '_without_lcm_marker_lines', lambda text: text)
        return fn()


def _main_ident(engine, row):
    return engine._session_end_prefix_compare_identity(row, session_id='S1', recover_markers=False)


def _bare(rng, home):
    path = rng.choice([f'/tmp/hermes-results/{rng.choice(["c1", "c2"])}.txt',
                       f'{home}/cache/spillover/{rng.choice(["c1", "c2"])}.txt'])
    body = rng.choice(['x' * 300, 'y' * 300, 'data:image/png;base64,' + 'A' * 280, SECRET + ' ' + 'z' * 260,
                       PASSWORD + ' ' + 'v' * 270,
                       'inner </persisted-output> ' + 'w' * 270])
    text = _marker(path, body, count=rng.choice([None, 999]), preview=rng.choice([None, body[:29], 'other preview']))
    return text.replace('\n', '\r\n') if rng.random() < 0.15 else text


def _form(rng, raw, config):
    """(text, reduced): the stored or host text, and what it denotes for the reference (None: not a clean LCM form)."""
    global KIND
    kind = KIND = rng.choice(['bare', 'ingest', 'ingest', 'generation-only', 'wrong-digest', 'cut', 'after-close',
                       'twice'])
    stored = redact_sensitive_value(raw, config, parse_json_strings=False) if rng.random() < 0.5 else raw
    if kind == 'bare':
        return stored, stored
    if kind in ('ingest', 'generation-only'):
        text = stored
        if kind == 'ingest' and not _has_lossy_sensitive_redaction(text):
            text = _add_inline_persisted_output_identity_metadata(text, _persisted_output_marker_identity_digest(raw))
        if rng.random() < 0.7:
            text = _add_inline_persisted_output_generation_metadata(text, STAT)
        return text, stored
    if kind == 'wrong-digest':
        text = _add_inline_persisted_output_identity_metadata(stored, WRONG)
        if text == stored:  # a lossy redaction gets no identity line
            return stored, stored
        if _SENSITIVE_PLACEHOLDER_PREFIX in (_persisted_output_preview_prefix(
                redact_sensitive_value(stored, config, parse_json_strings=False)) or ''):
            # Decided rule: a redacted preview's digest was taken before redaction and cannot be checked,
            # so the line is read as LCM-X's own; counted apart so the class stays visible.
            KIND = 'wrong-digest-redacted-preview'
            return text, stored
        return text, None
    if kind == 'cut':
        line = '[LCM persisted-output marker identity: preview_sha256=' + 'a' * 40
        return stored.replace('</persisted-output>', line + '\n</persisted-output>', 1), None
    if kind == 'after-close':
        return stored + '\n[LCM persisted-output file generation: size=1; mtime_ns=2; ctime_ns=3]', None
    text = _add_inline_persisted_output_identity_metadata(stored, _persisted_output_marker_identity_digest(raw))
    line = '[LCM persisted-output marker identity:'
    twice = text.replace(line, '[LCM persisted-output file generation: size=1; mtime_ns=2; ctime_ns=3]\n' + line, 1)
    return (twice, None) if twice != stored else (stored, stored)


def _row(text, call='c1'):
    return {'role': 'tool', 'tool_call_id': call, 'content': text}


def test_seeded_marker_lines_comparison(engine, monkeypatch):
    seed, count = int(os.environ.get('LCMX_2547_SEED', '2547')), int(os.environ.get('LCMX_2547_PAIRS', '1500'))
    rng, home, config, tally = random.Random(seed), str(engine._hermes_home), engine._config, Counter()
    # Positive control: the redaction variant really redacts the generator's secret.
    assert (redact_sensitive_value(SECRET, config, parse_json_strings=False) != SECRET) is config.sensitive_patterns_enabled
    for index in range(count):
        raw_a = _bare(rng, home)
        raw_b = raw_a if rng.random() < 0.6 else _bare(rng, home)
        (a, red_a), kind_a = _form(rng, raw_a, config), KIND
        (b, red_b), kind_b = _form(rng, raw_b, config), KIND
        stored, host = _row(a), _row(b, 'c1' if rng.random() < 0.9 else 'c2')
        head = engine._session_end_rows_equal(stored, host, session_id='S1')
        main = _main(lambda: engine._session_end_rows_equal(stored, host, session_id='S1'), monkeypatch)
        ident = engine._session_end_prefix_compare_identity(host, session_id='S1', recover_markers=False)
        ident_main = _main(lambda: engine._session_end_prefix_compare_identity(host, session_id='S1',
                                                                              recover_markers=False), monkeypatch)
        if head == main:
            tally['same'] += 1
        else:
            # The only admitted flip: main unequal, candidate equal, both sides reduce to one host marker.
            assert (main, head) == (False, True), (seed, index, 'flip', main, head)
            assert red_a is not None and red_b is not None, (seed, index, 'flip on a malformed form')
            assert stored['tool_call_id'] == host['tool_call_id'], (seed, index)
            clean = [redact_sensitive_value(x, config, parse_json_strings=False) for x in (red_a, red_b)]
            assert clean[0] == clean[1], (seed, index, 'flip on unequal bare texts')
            tally['flip'] += 1
        if (not head and red_a is not None and red_b is not None and stored['tool_call_id'] == host['tool_call_id']
                and len({redact_sensitive_value(x, config, parse_json_strings=False) for x in (red_a, red_b)}) == 1):
            # Reduced texts equal, still unequal: a remaining loss shape at an off-current end; counted, not hidden.
            tally['still-unequal:' + ('a-redacted' if red_a != raw_a else 'b-redacted' if red_b != raw_b
                                      else 'other') + f':{kind_a}/{kind_b}'] += 1
        # Every consumer row (the note's table) reduces to one of these reads. The host side with recovery
        # (positional and counted routes) and the counted identity against the stored row's identity: a flip only
        # main-unequal to candidate-equal, and only in the class above.
        stored_ident = engine._session_end_prefix_compare_identity(stored, session_id='S1', recover_markers=False)
        for name, read in (
                ('recovered', lambda: engine._session_end_prefix_compare_identity(host, session_id='S1') == stored_ident),
                ('counted', lambda: engine._session_end_counted_identity(host, {stored_ident}, session_id='S1')
                 == stored_ident)):
            cand = read()
            base = _main(lambda: (engine._session_end_counted_identity(
                host, {_main_ident(engine, stored)}, session_id='S1') if name == 'counted'
                else engine._session_end_prefix_compare_identity(host, session_id='S1')) == _main_ident(engine, stored),
                monkeypatch)
            if cand != base:
                assert (base, cand) == (False, True), (seed, index, name, base, cand)
                assert red_a is not None and red_b is not None, (seed, index, name, 'flip on a malformed form')
                tally['flip:' + name] += 1
        # The upgrade replay's legacy keep (reconcile.py) reads the default form, which never strips: main's.
        assert engine._bare_marker_identity(b, session_id='S1') == _main(
            lambda: engine._bare_marker_identity(b, session_id='S1'), monkeypatch), (seed, index, 'legacy keep')
        if red_b is None:
            assert ident == ident_main, (seed, index, 'identity changed on a malformed form')
        elif ident != ident_main:
            tally['identity'] += 1
        tally['kind:' + kind_b] += 1
    print('C2547DIFF', json.dumps(dict(tally), sort_keys=True))
    assert tally['flip'] and tally['same']
