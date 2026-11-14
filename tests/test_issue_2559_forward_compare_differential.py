"""#2559 seeded differential: the session-end compare never takes a changed row for its stored copy.

Rows go through real ingest; each stored row is compared with mutated host copies by `_session_end_rows_equal` and by
the counted identity. The reference form is written here on its own (a small JSON lexer, no `json` module): numeral
tokens verbatim, strings decoded, object pairs stably sorted by key (repeated keys keep their order), whitespace
dropped; argument text that does not lex is compared as text.

Bars:
- loss (hard, every seed): compare-equal implies reference-equal, except (a) pairs main's own reader already calls
  equal (inherited; limited to the literal-placeholder mutation) and (b) the one admitted residual, RM ruling T1-9:
  a quote-escaped, non-JSON argument text and its JSON twin. Ingest already stored the JSON form at the first store, so
  the stored row is content-equal to the twin and not storing it again loses nothing.
- copy (round 2 scope, RM T1-8 and T1-12): an unchanged tool call whose arguments hold an inline payload compares equal
  to its stored row in `_session_end_rows_equal`, when its arguments are numeral-free or hold nonzero integers only.
  Main fails this bar; the fix flips it. List-form content and counted routes are named limits and stay out of it.
- residual pin (T1-12): on ingest's re-dump route, arguments with a decimal or zero numeral keep main's answer.
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


_SAFE_NUMERALS = ['7', '12345678901234567890']
_RESIDUAL_NUMERALS = ['1.5', '0.1', '-0', '0', '1E0', '1e400', '2.50']
_NUMERALS = _SAFE_NUMERALS + _RESIDUAL_NUMERALS
SHAPES = ('args-wrapped', 'args-datauri-key', 'args-plain', 'list-wrapped', 'list-datauri', 'list-plain')
INLINE_ARGS_SHAPES = SHAPES[:2]
# Route-crossing and reviewer-witness bases (round 2): each meets only its own mutations.
SPECIAL_SHAPES = ('args-escaped', 'args-raw-nonjson', 'args-raw-datauri', 'args-nested', 'args-score')


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


def _special_base(rng, shape, index):
    """Bases whose H_old takes a route other than the plain re-dump (route-crossing seeds, reviewer witnesses)."""
    wrapped = _wrapped(rng)
    if shape == 'args-escaped':  # T1-9: quote-escaped, not JSON; ingest's unescaped reading re-dumps it
        text = json.dumps({'path': f'e{index}', 'content': 'hdr\n' + wrapped + '\nend'}).replace('"', '\\"')
    elif shape == 'args-raw-nonjson':  # raw newlines inside a quoted value: not JSON; the raw route keeps the bytes
        text = '{"path": "r%d", "content": "%s"}' % (index, wrapped)
    elif shape == 'args-raw-datauri':  # non-canonical spacing, an unescaped data URI: the raw route
        text = '{"u": "data:image/png;base64,%s", "k": %d}' % (_b64(rng, 1200), index + 1)
    elif shape == 'args-nested':  # DR-1: JSON text in a value, holding a payload and a decimal one level down
        inner = '{"n": 1.00, "img": "d\\u0061ta:image/png;base64,%s"}' % _b64(rng, 1200)
        text = json.dumps({'meta': inner, 'content': 'hdr\n' + wrapped + '\nend'})
    else:  # 'args-score' (DR-2): a float scalar outside the arguments string
        text = json.dumps({'path': f's{index}', 'content': 'hdr\n' + wrapped + '\nend'})
    row = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': f'x{index}', 'type': 'function', 'function': {'name': 'write_file', 'arguments': text}}]}
    if shape == 'args-score':
        row['tool_calls'][0]['score'] = -0.0
    return row


def _special_mutations(shape, row, stored, rng):
    out = [('unchanged', copy.deepcopy(row))]
    text = _args(row)
    if shape == 'args-escaped':
        out.append(('json-twin', _set_args(row, text.replace('\\"', '"'))))  # the admitted residual (T1-9)
    elif shape == 'args-raw-nonjson':
        out.append(('json-twin', _set_args(row, json.dumps(json.loads(text, strict=False)))))
    elif shape == 'args-raw-datauri':
        out.append(('literal-backslash', _set_args(row, text.replace('image/png', 'image\\\\/png', 1))))
    elif shape == 'args-nested':
        out.append(('nested-respell', _set_args(row, text.replace('1.00', '1.0', 1).replace('d\\\\u0061ta', 'data', 1))))
    else:
        zero = copy.deepcopy(row)
        zero['tool_calls'][0]['score'] = 0.0
        out.append(('score-zero', zero))
    return out


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
        if placeholder and '"content": ' in text:  # DR-4: a JSON-escaped literal placeholder (\u005b...)
            escaped = json.dumps(placeholder.group(0)).replace('"[', '"\\u005b', 1)
            out.append(('escaped-literal-placeholder', _set_args(row, re.sub(
                r'"content": "(?:[^"\\]|\\.)*"', lambda m: '"content": ' + escaped, text, count=1))))
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
    shapes = [SHAPES[i % len(SHAPES)] for i in range(BASES)]
    for k, shape in enumerate(SPECIAL_SHAPES * 2):
        bases.append(_special_base(rng, shape, BASES + k))
        shapes.append(shape)
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
    pairs = []
    for base, shape in zip(bases, shapes):
        stored = stored_of[id(base)]
        made = (_special_mutations(shape, base, stored, rng) if shape in SPECIAL_SHAPES
                else _mutations(base, stored, rng))
        pairs += [(shape, name, base, stored, host) for name, host in made]
    pairs += _literal_copy_pairs(engine, bases, shapes, stored_of, rng, len(rows))
    results = []
    for shape, name, base, stored, host in pairs:
        stored_id = engine._session_end_prefix_compare_identity(stored, session_id='S', recover_markers=False)
        equal = engine._session_end_rows_equal(stored, host, session_id='S')
        counted = engine._session_end_counted_identity(host, Counter([stored_id]), session_id='S') == stored_id
        mains = stored_id == engine._session_end_prefix_compare_identity(host, session_id='S')
        number = re.search(r'"n": ([^,}\s]+)', _args(base)) if 'tool_calls' in base else None
        results.append({'shape': shape, 'mutation': name, 'equal': equal, 'counted': counted, 'mains': mains,
                        'ref_equal': ref_row(host) == ref_row(base), 'numeral': number.group(1) if number else None})
    return results


def _literal_copy_pairs(engine, bases, shapes, stored_of, rng, already):
    """Astra R2-1: H_old holds a literal copy of an earlier row's placeholder (same session, same field, the file
    present) next to its own payload; H_new holds that payload's content instead. The pair must compare unequal."""
    from hermes_lcm.ingest_integrity import restore_ingest_payload_placeholders
    out, extra = [], []
    for base, shape in zip(bases, shapes):
        if shape != 'args-wrapped' or len(extra) >= 3:
            continue
        stored_args = _args(stored_of[id(base)])
        found = re.search(r'\[Externalized LCM ingest payload:[^\]]*\]', stored_args)
        payload = restore_ingest_payload_placeholders(found.group(0), config=engine._config,
                                                      hermes_home=engine._hermes_home, session_id='S')
        own = _wrapped(rng)
        def call(content, k=len(extra)):
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': f'lc{k}', 'type': 'function', 'function': {'name': 'write_file', 'arguments': json.dumps(
                    {'path': f'lc{k}', 'content': 'hdr\n' + content + '\nend', 'd': 'hdr\n' + own + '\nend'})}}]}
        extra.append((call(found.group(0)), call(payload)))
    rows = []
    for k, (old, _new) in enumerate(extra):
        rows += [{'role': 'user', 'content': f'copy {k}'}, old,
                 {'role': 'tool', 'tool_call_id': f'lc{k}', 'content': f'ok copy {k}'}]
    for k, row in enumerate(rows):
        row['timestamp'] = float(already + k + 1)
    engine.ingest(copy.deepcopy(rows))
    stored_rows = engine._store.get_session_messages('S')[already:]
    for k, (old, new) in enumerate(extra):
        out.append(('args-literal-copy', 'restored-literal', old, stored_rows[3 * k + 1], new))
    return out


def test_no_changed_row_compares_equal(run):
    """The loss bar: every match is reference-equal, or main's reader already makes it (literal placeholders only),
    or it is the one admitted residual (T1-9: the quote-escaped argument text and its JSON twin)."""
    admitted = {('args-escaped', 'json-twin')}
    bad = [r for r in run if (r['equal'] or r['counted']) and not r['ref_equal'] and not r['mains']
           and (r['shape'], r['mutation']) not in admitted]
    assert not bad, bad[:5]
    inherited = Counter(r['mutation'] for r in run if (r['equal'] or r['counted']) and not r['ref_equal']
                        and (r['shape'], r['mutation']) not in admitted)
    assert set(inherited) <= {'literal-placeholder'}, inherited
    assert len(run) > 10 * BASES
    assert {r['shape'] for r in run} >= set(SHAPES) | set(SPECIAL_SHAPES) | {'args-literal-copy'}


def test_plain_controls_compare_equal_unchanged(run):
    plain = [r for r in run if r['mutation'] == 'unchanged' and r['shape'].endswith('plain')]
    assert plain and all(r['equal'] and r['counted'] for r in plain)


@pytest.mark.xfail(strict=True, reason='#2559: an unchanged inline-payload tool call compares unequal to its stored copy')
def test_unchanged_inline_rows_compare_equal(run):
    """The copy bar (round 2 scope): unchanged tool calls with an inline payload in their arguments, numeral-free or
    nonzero integers only, compare equal in `_session_end_rows_equal`. Counted routes are a named limit."""
    inline = [r for r in run if r['mutation'] == 'unchanged' and r['shape'] in INLINE_ARGS_SHAPES
              and r['numeral'] in _SAFE_NUMERALS]
    assert len(inline) >= 4 and all(r['equal'] for r in inline), Counter((r['shape'], r['equal']) for r in inline)


@pytest.mark.xfail(strict=True, reason='#2559 (T1-9): the admitted residual compares equal only with the fix')
def test_the_admitted_residual_compares_equal(run):
    """T1-9: the JSON twin of a quote-escaped argument text is content-equal to the row ingest stored (the JSON form)."""
    cells = [r for r in run if (r['shape'], r['mutation']) == ('args-escaped', 'json-twin')]
    assert cells and all(r['equal'] for r in cells), cells


def test_redump_route_decimal_or_zero_numerals_keep_mains_copy(run):
    """T1-12 residual pin: on ingest's re-dump route an argument text holding a decimal or zero numeral keeps main's
    answer (an extra copy at the next handover, as on main): no origin evidence exists for a re-dumped numeral."""
    cells = [r for r in run if r['mutation'] == 'unchanged' and r['shape'] in INLINE_ARGS_SHAPES
             and r['numeral'] in _RESIDUAL_NUMERALS]
    assert cells and not any(r['equal'] or r['mains'] for r in cells), Counter((r['numeral'], r['equal']) for r in cells)


def test_the_loss_bar_catches_a_compare_that_merges_changed_rows(run):
    """The bar is live: with every pair taken as equal, the changed rows it must reject are found per mutation."""
    merged = [dict(r, equal=True, mains=False) for r in run]
    caught = Counter(r['mutation'] for r in merged if r['equal'] and not r['ref_equal'] and not r['mains'])
    expected = {'numeral', 'escape-slash', 'duplicate-key', 'json-in-value', 'json-in-value-compact', 'payload-edit',
                'not-json', 'call-id', 'call-name', 'call-extra-key', 'part-order', 'text-edit', 'extra-part',
                'literal-placeholder', 'escaped-literal-placeholder', 'json-twin', 'nested-respell', 'score-zero',
                'restored-literal', 'literal-backslash'}
    assert expected <= set(caught), caught
    assert not {'unchanged', 'compact', 'respaced', 'key-order'} & set(caught), caught
