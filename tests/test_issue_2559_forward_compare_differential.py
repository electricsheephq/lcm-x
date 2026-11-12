"""#2559 seeded differential: the session-end compare never takes a changed row for its stored copy.

Rows go through real ingest; each stored row is compared with mutated host copies by `_session_end_rows_equal` and by
the counted identity. The reference form is written here on its own (a small JSON lexer, no `json` module): numeral
tokens verbatim, strings decoded, object pairs stably sorted by key (repeated keys keep their order), whitespace
dropped; argument text that does not lex is compared as text.

Bars:
- loss (hard, every seed): compare-equal implies reference-equal, unless main's own reader already calls the pair
  equal (an inherited case; those are counted and limited to the literal-placeholder mutation).
- copy (the issue's shapes): an unchanged host copy compares equal to its stored row, on every route's two entry
  points. Main fails this bar for the inline shapes; the fix flips it.
"""

import base64
import copy
import json
import random
import re
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SEED = 25590301
BASES = 48  # rows per run; each meets every mutation below


# ---------------------------------------------------------------- reference form (independent of the product code)
_NUM = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_WS = " \t\n\r"
_ESC = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}


class _NotJson(Exception):
    pass


def _lex_value(text, i):
    while i < len(text) and text[i] in _WS:
        i += 1
    if i >= len(text):
        raise _NotJson
    ch = text[i]
    if ch == '{':
        pairs, i = [], i + 1
        while True:
            while i < len(text) and text[i] in _WS:
                i += 1
            if i < len(text) and text[i] == '}' and not pairs:
                return ('obj', tuple(sorted(pairs, key=lambda p: p[0]))), i + 1
            key, i = _lex_value(text, i)
            if key[0] != 'str':
                raise _NotJson
            while i < len(text) and text[i] in _WS:
                i += 1
            if i >= len(text) or text[i] != ':':
                raise _NotJson
            value, i = _lex_value(text, i + 1)
            pairs.append((key[1], value))
            while i < len(text) and text[i] in _WS:
                i += 1
            if i < len(text) and text[i] == ',':
                i += 1
                continue
            if i < len(text) and text[i] == '}':
                return ('obj', tuple(sorted(pairs, key=lambda p: p[0]))), i + 1
            raise _NotJson
    if ch == '[':
        items, i = [], i + 1
        while True:
            while i < len(text) and text[i] in _WS:
                i += 1
            if i < len(text) and text[i] == ']' and not items:
                return ('arr', tuple(items)), i + 1
            value, i = _lex_value(text, i)
            items.append(value)
            while i < len(text) and text[i] in _WS:
                i += 1
            if i < len(text) and text[i] == ',':
                i += 1
                continue
            if i < len(text) and text[i] == ']':
                return ('arr', tuple(items)), i + 1
            raise _NotJson
    if ch == '"':
        out, i = [], i + 1
        while True:
            if i >= len(text):
                raise _NotJson
            c = text[i]
            if c == '"':
                return ('str', ''.join(out)), i + 1
            if c == '\\':
                if i + 1 >= len(text):
                    raise _NotJson
                e = text[i + 1]
                if e in _ESC:
                    out.append(_ESC[e])
                    i += 2
                elif e == 'u' and re.fullmatch(r'[0-9a-fA-F]{4}', text[i + 2:i + 6] or ''):
                    code = int(text[i + 2:i + 6], 16)
                    i += 6
                    if 0xD800 <= code < 0xDC00 and text[i:i + 2] == '\\u' and re.fullmatch(
                            r'[dD][c-fC-F][0-9a-fA-F]{2}', text[i + 2:i + 6] or ''):
                        code = 0x10000 + ((code - 0xD800) << 10) + (int(text[i + 2:i + 6], 16) - 0xDC00)
                        i += 6
                    out.append(chr(code))
                else:
                    raise _NotJson
            elif ord(c) < 0x20:
                raise _NotJson
            else:
                out.append(c)
                i += 1
    for literal in ('true', 'false', 'null', 'NaN', 'Infinity', '-Infinity'):
        if text.startswith(literal, i):
            return ('tok', literal), i + len(literal)
    match = _NUM.match(text, i)
    if match:
        return ('tok', match.group(0)), match.end()
    raise _NotJson


def ref_json_text(text):
    """The reference form of a JSON text, or ('text', text) when it does not lex."""
    try:
        value, end = _lex_value(text, 0)
        if text[end:].strip(_WS):
            raise _NotJson
        return value
    except (_NotJson, RecursionError):
        return ('text', text)


def ref_value(value, key=None):
    if isinstance(value, dict):
        return ('obj', tuple(sorted(((k, ref_value(v, k)) for k, v in value.items()), key=lambda p: p[0])))
    if isinstance(value, list):
        return ('arr', tuple(ref_value(v) for v in value))
    if isinstance(value, str):
        return ref_json_text(value) if key == 'arguments' else ('str', value)
    return ('py', type(value).__name__, repr(value))


def ref_row(row):
    return (str(row.get('role') or ''), ref_value(row.get('content')), str(row.get('tool_call_id') or ''),
            str(row.get('tool_name') or ''), ref_value(row.get('tool_calls') or None))


def test_reference_form_examples():
    assert ref_json_text('{"a": 1, "b": "\\u00e9\\/"}') == ref_json_text('{"b":"é/","a":1}')
    assert ref_json_text('{"n": 1e400}') != ref_json_text('{"n": 2e400}')
    assert ref_json_text('{"n": 0.1}') != ref_json_text('{"n": 0.10000000000000001}')
    assert ref_json_text('{"n": -0}') != ref_json_text('{"n": 0}')
    assert ref_json_text('{"a": 1, "a": 2}') != ref_json_text('{"a": 2, "a": 1}')
    assert ref_json_text('{"x": "{\\"y\\": 1}"}') != ref_json_text('{"x": "{\\"y\\":1}"}')
    assert ref_json_text('{"a": 1') == ('text', '{"a": 1')


# ---------------------------------------------------------------- rows and mutations
def _b64(rng, n):
    return base64.b64encode(bytes(rng.randrange(256) for _ in range(n))).decode('ascii')


def _wrapped(rng):
    long = _b64(rng, 3420)
    return '\n'.join(long[i:i + 76] for i in range(0, len(long), 76))


_NUMERALS = ['7', '1.5', '0.1', '-0', '1E0', '1e400', '12345678901234567890', '2.50']
SHAPES = ('args-wrapped', 'args-datauri-key', 'args-plain', 'list-wrapped', 'list-datauri', 'list-plain')
INLINE_SHAPES = SHAPES[:2] + SHAPES[3:5]


def _base(rng, shape, index):
    if shape.startswith('args'):
        numeral = rng.choice(_NUMERALS)
        if shape == 'args-wrapped':
            body = json.dumps('hdr\n' + _wrapped(rng) + '\nend')
            text = '{"path": "a%d.b64", "n": %s, "content": %s}' % (index, numeral, body)
        elif shape == 'args-datauri-key':
            key = json.dumps('data:image\\/png;base64,' + _b64(rng, 1200))
            text = '{%s: "v", "n": %s, "b": "x%d"}' % (key, numeral, index)
        else:
            text = '{"path": "a%d.txt", "n": %s, "content": "hdr\\nplain body\\nend"}' % (index, numeral)
        return {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': f'c{index}', 'type': 'function', 'function': {'name': 'write_file', 'arguments': text}}]}
    if shape == 'list-wrapped':
        parts = [{'type': 'text', 'text': f'hdr{index}\n' + _wrapped(rng) + '\nend'}]
    elif shape == 'list-datauri':
        parts = [{'type': 'text', 'text': f'see {index}'},
                 {'type': 'image_url', 'image_url': {'url': 'data:image\\/png;base64,' + _b64(rng, 1200)}}]
    else:
        parts = [{'type': 'text', 'text': f'plain {index}'}, {'type': 'text', 'text': 'second'}]
    return {'role': 'user', 'content': parts}


def _args(row):
    return row['tool_calls'][0]['function']['arguments']


def _set_args(row, text):
    row = copy.deepcopy(row)
    row['tool_calls'][0]['function']['arguments'] = text
    return row


def _flip_char(text, rng):
    spots = [m.start() for m in re.finditer(r'[A-Za-z0-9]{8}', text)]
    i = rng.choice(spots) + 3 if spots else 0
    return text[:i] + ('B' if text[i] != 'B' else 'C') + text[i + 1:]


def _mutations(row, stored, rng):
    """(name, host copy) pairs for one base row; `stored` supplies literal placeholder text."""
    out = [('unchanged', copy.deepcopy(row))]
    placeholder = re.search(r'\[Externalized LCM ingest payload:[^\]]*\]', json.dumps(stored, ensure_ascii=False))
    if 'tool_calls' in row:
        text = _args(row)
        number = re.search(r'"n": (\S+?)([,}])', text)
        for other in [n for n in _NUMERALS + ['1.50', '15e-1', '1.5000000000000001', '0.10000000000000001', '0',
                                              '2e400', '12345678901234567891'] if n != number.group(1)][:6]:
            out.append(('numeral', _set_args(row, text[:number.start(1)] + other + text[number.end(1):])))
        out.append(('compact', _set_args(row, re.sub(r'": ', '":', text).replace(', "', ',"'))))
        out.append(('respaced', _set_args(row, text.replace(': ', ' :  '))))
        out.append(('escape-slash', _set_args(row, text.replace('\\/', '/', 1) if '\\/' in text
                                                else text.replace('"a', '"\\u0061', 1))))
        pairs = re.match(r'\{(.*)\}$', text, re.S).group(1)
        first, rest = pairs.split(', ', 1)
        out.append(('key-order', _set_args(row, '{' + rest + ', ' + first + '}')))
        out.append(('duplicate-key', _set_args(row, text[:-1] + ', "n": 3}')))
        out.append(('json-in-value', _set_args(row, text[:-1] + ', "j": "{\\"y\\": 1}"}')))
        out.append(('json-in-value-compact', _set_args(row, text[:-1] + ', "j": "{\\"y\\":1}"}')))
        out.append(('payload-edit', _set_args(row, _flip_char(text, rng))))
        out.append(('not-json', _set_args(row, text[:-1])))
        if placeholder:
            out.append(('literal-placeholder', _set_args(row, re.sub(
                r'"content": "(?:[^"\\]|\\.)*"|"data:(?:[^"\\]|\\.)*"',
                lambda m: json.dumps(placeholder.group(0)) if not m.group(0).startswith('"content"')
                else '"content": ' + json.dumps(placeholder.group(0)), text, count=1))))
        for field, value in (('id', 'other'), ('name', 'read_file')):
            changed = copy.deepcopy(row)
            if field == 'id':
                changed['tool_calls'][0]['id'] += '-x'
            else:
                changed['tool_calls'][0]['function']['name'] = value
            out.append((f'call-{field}', changed))
        extra = copy.deepcopy(row)
        extra['tool_calls'][0]['extra'] = 'k'
        out.append(('call-extra-key', extra))
    else:
        parts = row['content']
        out.append(('part-order', dict(row, content=list(reversed(parts)))))
        edited = copy.deepcopy(row)
        target = edited['content'][-1]
        if target.get('type') == 'image_url':
            target['image_url']['url'] = _flip_char(target['image_url']['url'], rng)
        else:
            target['text'] = _flip_char(target['text'], rng)
        out.append(('payload-edit', edited))
        text_edit = copy.deepcopy(row)
        text_edit['content'][0]['text'] += '!'
        out.append(('text-edit', text_edit))
        out.append(('extra-part', dict(row, content=parts + [{'type': 'text', 'text': 'more'}])))
        if placeholder:
            literal = copy.deepcopy(row)
            if literal['content'][-1].get('type') == 'image_url':
                literal['content'][-1]['image_url']['url'] = placeholder.group(0)
            else:
                literal['content'][-1]['text'] = placeholder.group(0)
            out.append(('literal-placeholder', literal))
    return out


# ---------------------------------------------------------------- the run
@pytest.fixture(scope='module')
def run(tmp_path_factory):
    home = tmp_path_factory.mktemp('diff2559') / 'hermes'
    engine = LCMEngine(config=LCMConfig(database_path=str(home / 'lcm.db'),
                                        large_output_externalization_path=str(home / 'externalized')),
                       hermes_home=str(home))
    try:
        return _drive(engine)
    finally:
        engine.shutdown()


def _drive(engine):
    rng = random.Random(SEED)
    bases = [_base(rng, SHAPES[i % len(SHAPES)], i) for i in range(BASES)]
    rows = []
    for i, row in enumerate(bases):
        rows += [{'role': 'user', 'content': f'turn {i}'}, row]
        if 'tool_calls' in row:
            rows.append({'role': 'tool', 'tool_call_id': row['tool_calls'][0]['id'], 'content': f'ok {i}'})
    for k, row in enumerate(rows):
        row['timestamp'] = float(k + 1)
    engine.on_session_start('S', platform='cli', context_length=200_000)
    engine.ingest(copy.deepcopy(rows))
    stored_rows = engine._store.get_session_messages('S')
    assert len(stored_rows) == len(rows)
    stored_of = {id(r): s for r, s in zip(rows, stored_rows)}
    results = []
    for i, base in enumerate(bases):
        shape = SHAPES[i % len(SHAPES)]
        stored = stored_of[id(next(r for r in rows if r is base))]
        stored_id = engine._session_end_prefix_compare_identity(stored, session_id='S', recover_markers=False)
        for name, host in _mutations(base, stored, rng):
            equal = engine._session_end_rows_equal(stored, host, session_id='S')
            counted = engine._session_end_counted_identity(host, Counter([stored_id]), session_id='S') == stored_id
            mains = stored_id == engine._session_end_prefix_compare_identity(host, session_id='S')
            results.append({'shape': shape, 'mutation': name, 'equal': equal, 'counted': counted, 'mains': mains,
                            'ref_equal': ref_row(host) == ref_row(base)})
    return results


def test_no_changed_row_compares_equal(run):
    """The loss bar: every match is reference-equal, or one main's reader already makes (literal placeholders only)."""
    bad = [r for r in run if (r['equal'] or r['counted']) and not r['ref_equal'] and not r['mains']]
    assert not bad, bad[:5]
    inherited = Counter(r['mutation'] for r in run if (r['equal'] or r['counted']) and not r['ref_equal'])
    assert set(inherited) <= {'literal-placeholder'}, inherited
    assert len(run) > 10 * BASES


def test_plain_controls_compare_equal_unchanged(run):
    plain = [r for r in run if r['mutation'] == 'unchanged' and r['shape'].endswith('plain')]
    assert plain and all(r['equal'] and r['counted'] for r in plain)


@pytest.mark.xfail(strict=True, reason='#2559: an unchanged inline-payload row compares unequal to its stored copy')
def test_unchanged_inline_rows_compare_equal(run):
    """The copy bar for the issue's shapes: host default spacing, unchanged row."""
    inline = [r for r in run if r['mutation'] == 'unchanged' and r['shape'] in INLINE_SHAPES]
    assert inline and all(r['equal'] and r['counted'] for r in inline), Counter(
        (r['shape'], r['equal'], r['counted']) for r in inline)


def test_the_loss_bar_catches_a_compare_that_merges_changed_rows(run):
    """The bar is live: with every pair taken as equal, the changed rows it must reject are found per mutation."""
    merged = [dict(r, equal=True, mains=False) for r in run]
    caught = Counter(r['mutation'] for r in merged if r['equal'] and not r['ref_equal'] and not r['mains'])
    expected = {'numeral', 'escape-slash', 'duplicate-key', 'json-in-value', 'json-in-value-compact', 'payload-edit',
                'not-json', 'call-id', 'call-name', 'call-extra-key', 'part-order', 'text-edit', 'extra-part',
                'literal-placeholder'}
    assert expected <= set(caught), caught
    assert not {'unchanged', 'compact', 'respaced', 'key-order'} & set(caught), caught
