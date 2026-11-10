"""#2559 lane edit: only a stored string that decodes to a list or dict takes the per-leaf restore.

Every other string with an ingest ref keeps the #2539 path, including its whole-placeholder payload load
and that load's session check.
"""

import json

import pytest

from tests.test_issue_1931_delegate_children import host  # noqa: F401
from tests.test_issue_2521_spillover_upgrade_replay import _engine, _home

_PLACEHOLDER = '[Externalized LCM ingest payload: 4 chars; ref={ref}]'


@pytest.fixture
def engine(tmp_path):
    engine = _engine(tmp_path)
    storage = _home(tmp_path) / 'externalized'
    storage.mkdir(parents=True, exist_ok=True)
    # The whole-placeholder load reads the payload whatever its kind (here the default, tool_result);
    # the textual restore takes only ingest_payload payloads.
    (storage / 'plain.json').write_text(json.dumps({'content': 'BODY', 'session_id': 'S1'}))
    (storage / 'leaf.json').write_text(json.dumps({
        'kind': 'ingest_payload', 'content': 'LEAF', 'session_id': 'S1', 'field_path': 'content[0].text'}))
    try:
        yield engine
    finally:
        engine.shutdown()


def test_a_whole_placeholder_string_keeps_the_payload_load_and_its_session_check(engine):
    value = _PLACEHOLDER.format(ref='plain.json')
    assert engine._session_end_prefix_compare_value(value, session_id='S1') == 'BODY'
    assert engine._session_end_prefix_compare_value(value, session_id='S2') == value


def test_list_content_stored_as_text_is_restored_per_leaf(engine):
    stored = json.dumps([{'type': 'text', 'text': _PLACEHOLDER.format(ref='leaf.json')}])
    restored = engine._session_end_prefix_compare_value(stored, session_id='S1')
    assert json.loads(restored) == [{'type': 'text', 'text': 'LEAF'}]
    assert engine._session_end_prefix_compare_value(stored, session_id='S2') == stored


def _call(arguments):
    return {'role': 'assistant', 'content': '', 'timestamp': 2.0,
            'tool_calls': [{'id': 'k1', 'type': 'function', 'function': {'name': 'f', 'arguments': arguments}}]}


def test_arguments_with_duplicate_keys_compare_as_text(engine):
    # NeonDiff (#2572): parsing would collapse these to one object; ingest keeps duplicate-key JSON as text.
    first, second = '{"image":"A","image":"plain"}', '{"image":"B","image":"plain"}'
    assert not engine._session_end_rows_equal(_call(first), _call(second), session_id='S1')
    assert engine._session_end_rows_equal(_call(first), _call(first), session_id='S1')
    # Text compares stripped, like invalid-JSON arguments (#2572 review: an edge-whitespace re-send stored a row).
    assert engine._session_end_rows_equal(_call(first), _call('  ' + first + '\n'), session_id='S1')


def test_a_payload_held_in_an_argument_key_is_restored_before_the_keys_are_sorted(engine, tmp_path):
    # NeonDiff (#2572): ingest can externalize a payload held in a key; restoring only values left the
    # stored placeholder key sorting before "b" while the host's data URI sorts after it.
    import base64

    key = 'data:image/png;base64,' + base64.b64encode(bytes(range(256)) * 20).decode()
    sent = _call(json.dumps({key: 'v', 'b': 'x'}))
    engine.on_session_start('S1', platform='cli', context_length=200_000, conversation_id='c')
    engine.ingest([{'role': 'user', 'content': 'go', 'timestamp': 1.0}, json.loads(json.dumps(sent)),
                   {'role': 'tool', 'tool_call_id': 'k1', 'content': 'ok', 'timestamp': 3.0},
                   {'role': 'assistant', 'content': 'done', 'timestamp': 4.0}])
    stored = next(r for r in engine._store.get_session_messages('S1') if r['role'] == 'assistant' and r.get('tool_calls'))
    raw = stored['tool_calls'] if isinstance(stored['tool_calls'], str) else json.dumps(stored['tool_calls'])
    assert '[Externalized LCM ingest payload:' in raw and key not in raw  # the key itself was moved out
    assert engine._session_end_rows_equal(stored, sent, session_id='S1')
    changed = _call(json.dumps({key[:-4] + 'AAAA': 'v', 'b': 'x'}))
    assert not engine._session_end_rows_equal(stored, changed, session_id='S1')


def _key_payload(storage, name):
    import base64

    payload = 'data:image/png;base64,' + base64.b64encode(bytes(range(256)) * 20).decode()
    (storage / name).write_text(json.dumps({'kind': 'ingest_payload', 'content': payload, 'session_id': 'S1',
                                            'field_path': 'tool_calls[0].function.arguments.<key>'}))
    return payload, _PLACEHOLDER.format(ref=name)


@pytest.mark.parametrize('layout', ['placeholder-and-payload', 'two-placeholders'])
def test_two_argument_keys_that_restore_to_one_payload_are_never_merged(engine, tmp_path, layout):
    # Astra (#2572 round 2): restoring both keys to one payload merged them, so a changed value compared equal.
    storage = _home(tmp_path) / 'externalized'
    payload, first_key = _key_payload(storage, 'key1.json')
    _, second_key = _key_payload(storage, 'key2.json')
    keys = (first_key, payload if layout == 'placeholder-and-payload' else second_key)
    sent = _call(json.dumps({keys[0]: 'one', keys[1]: 'two'}))
    changed = _call(json.dumps({keys[0]: 'changed', keys[1]: 'two'}))
    assert engine._session_end_rows_equal(sent, sent, session_id='S1')
    assert not engine._session_end_rows_equal(sent, changed, session_id='S1')
    assert not engine._session_end_rows_equal(sent, _call(json.dumps({payload: 'two'})), session_id='S1')


def test_a_changed_value_under_a_placeholder_key_is_stored_for_a_live_child(host, tmp_path):  # noqa: F811
    import base64
    import copy

    from hermes_lcm.externalize import externalize_ingest_payload
    from tests.test_issue_2482_interrupted_child import _setup

    _, _, ctx, reader, child, engine = _setup(host, 'own_engine')
    body = 'data:image/png;base64,' + base64.b64encode(bytes(range(256)) * 20).decode()
    alias = externalize_ingest_payload(body, role='assistant', session_id='child',
                                       field_path='tool_calls[0].function.arguments.<key>', config=reader._config,
                                       hermes_home=reader._hermes_home)['placeholder']
    first = [{'role': 'user', 'content': 'go'}, _call(json.dumps({alias: 'first', body: 'second'})),
             {'role': 'tool', 'tool_call_id': 'k1', 'content': 'ok'}, {'role': 'assistant', 'content': 'done'}]
    for i, row in enumerate(first):
        row['timestamp'] = i + 1.0
    aux = ctx.module('aux_session')
    child.api_call(ctx, copy.deepcopy(first), 1)
    aux.drain_live_child_stores(60)
    view = copy.deepcopy(first)
    view[1]['tool_calls'][0]['function']['arguments'] = json.dumps({alias: 'changed', body: 'second'})
    view.append({'role': 'assistant', 'content': 'final', 'timestamp': 5.0})
    for n in (2, 3):
        child.api_call(ctx, copy.deepcopy(view), n)
        aux.drain_live_child_stores(60)
    child.end(engine, view)
    aux.drain_live_child_stores(60)
    calls = []
    for row in reader._store.get_session_messages('child'):
        if row.get('tool_calls'):
            stored = row['tool_calls']
            calls.extend(json.loads(stored) if isinstance(stored, str) else stored)
    assert sum('changed' in call['function']['arguments'] for call in calls) == 1, len(calls)


def _content(form, first, second, changed):
    pair = {first: changed, second: 'B'}
    if form == 'dict':
        return pair
    if form == 'list':
        return [dict({'type': 'text', 'text': 'x'}, **pair)]
    return [{'type': 'text', 'text': 'x', 'meta': pair}]


@pytest.mark.parametrize('keys', ['placeholder-and-payload', 'two-placeholders'])
@pytest.mark.parametrize('form', ['dict', 'list', 'nested'])
def test_a_changed_value_under_colliding_content_keys_is_stored_for_a_live_child(host, form, keys):  # noqa: F811
    # Codex connector (#2572 round 3): content keys that restore to one payload, then a re-send with only the first
    # value changed. Recorded on main 7e67daf16 first: [2, 4, 4, 4] rows and the changed row stored once.
    import base64
    import copy
    import re
    from pathlib import Path

    from hermes_lcm.externalize import externalize_ingest_payload
    from tests.test_issue_2482_interrupted_child import _setup

    _, _, ctx, reader, child, engine = _setup(host, 'own_engine')
    body = 'data:image/png;base64,' + base64.b64encode(bytes(range(256)) * 20).decode()
    alias = externalize_ingest_payload(body, role='user', session_id='child', field_path='content[0].<key>',
                                       config=reader._config, hermes_home=reader._hermes_home)['placeholder']
    second = body
    if keys == 'two-placeholders':  # a second, distinct ref to the same payload
        name = Path(re.search(r'ref=([^\]]+)\]', alias).group(1)).name
        source = next(Path(reader._hermes_home).rglob(name))
        source.with_name('twin-' + name).write_bytes(source.read_bytes())
        second = alias.replace(name, 'twin-' + name)
    first = [{'role': 'user', 'content': _content(form, alias, second, 'A')}, {'role': 'assistant', 'content': 'seen'}]
    for i, row in enumerate(first):
        row['timestamp'] = i + 1.0
    aux = ctx.module('aux_session')
    child.api_call(ctx, copy.deepcopy(first), 1)
    aux.drain_live_child_stores(60)
    counts = [len(reader._store.get_session_messages('child'))]
    view = copy.deepcopy(first)
    view[0]['content'] = _content(form, alias, second, 'CHANGED')
    view.append({'role': 'assistant', 'content': 'final', 'timestamp': 3.0})
    for n in (2, 3):
        child.api_call(ctx, copy.deepcopy(view), n)
        aux.drain_live_child_stores(60)
        counts.append(len(reader._store.get_session_messages('child')))
    child.end(engine, view)
    aux.drain_live_child_stores(60)
    rows = reader._store.get_session_messages('child')
    counts.append(len(rows))
    users = [r['content'] if isinstance(r['content'], str) else json.dumps(r['content'])
             for r in rows if r['role'] == 'user']
    assert counts == [2, 4, 4, 4] and sum('CHANGED' in text for text in users) == 1, (counts, len(users))


@pytest.mark.parametrize('route', ['live', 'end'])
@pytest.mark.parametrize('note', ['plain', 'café'], ids=['ascii', 'non-ascii'])
@pytest.mark.parametrize('redaction', [False, True], ids=['redaction-off', 'redaction-on'])
def test_an_identical_call_with_a_redacted_argument_is_stored_once(host, redaction, note, route):  # noqa: F811
    # NeonDiff (#2572 round 4): with redaction on, the redactor writes JSON unescaped while the compare's canonical
    # dump escaped non-ASCII, so an identical call compared unequal and was stored again. Recorded on main 7e67daf16
    # first: [4, 5, 5] rows and one call row in all eight cases.
    import copy
    import dataclasses

    from tests.test_issue_2482_interrupted_child import _setup

    _, _, ctx, reader, child, engine = _setup(host, 'own_engine')
    for each in {id(e): e for e in (reader, engine)}.values():
        each._config = dataclasses.replace(each._config, sensitive_patterns_enabled=redaction)
    secret = 'sk-' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0'
    first = [{'role': 'user', 'content': 'go'},
             _call(json.dumps({'api_key': secret, 'note': note}, ensure_ascii=False)),
             {'role': 'tool', 'tool_call_id': 'k1', 'content': 'ok'}, {'role': 'assistant', 'content': 'done'}]
    for i, row in enumerate(first):
        row['timestamp'] = i + 1.0
    aux = ctx.module('aux_session')
    child.api_call(ctx, copy.deepcopy(first), 1)
    aux.drain_live_child_stores(60)
    counts = [len(reader._store.get_session_messages('child'))]
    view = copy.deepcopy(first) + [{'role': 'assistant', 'content': 'final', 'timestamp': 5.0}]
    for n in (2, 3):
        if route == 'live':
            child.api_call(ctx, copy.deepcopy(view), n)
        else:
            child.end(engine, copy.deepcopy(view))
        aux.drain_live_child_stores(60)
        counts.append(len(reader._store.get_session_messages('child')))
    calls = [r for r in reader._store.get_session_messages('child') if r.get('tool_calls')]
    assert counts == [4, 5, 5] and len(calls) == 1, (counts, len(calls))


@pytest.mark.parametrize('route', ['live', 'end'])
@pytest.mark.parametrize('other', ['A', 'Bearer a', 'c'], ids=['sorts-below', 'sorts-between', 'sorts-above'])
@pytest.mark.parametrize('redaction', [False, True], ids=['redaction-off', 'redaction-on'])
def test_an_identical_call_with_a_redacted_argument_key_is_stored_once(host, redaction, other, route):  # noqa: F811
    # Codex connector (#2572 round 5): a key holding a bearer token sorted by its original text on the host side but
    # by its redacted text on the stored side, so a key sorting between the two flipped the order and the identical
    # call was stored again. Recorded on main 7e67daf16 first: [4, 5, 5] rows and one call row in all twelve cases.
    import copy
    import dataclasses

    from tests.test_issue_2482_interrupted_child import _setup

    _, _, ctx, reader, child, engine = _setup(host, 'own_engine')
    for each in {id(e): e for e in (reader, engine)}.values():
        each._config = dataclasses.replace(each._config, sensitive_patterns_enabled=redaction)
    first = [{'role': 'user', 'content': 'go'}, _call(json.dumps({'Bearer ' + 'a' * 20 + '0' * 20: 'v', other: 'x'})),
             {'role': 'tool', 'tool_call_id': 'k1', 'content': 'ok'}, {'role': 'assistant', 'content': 'done'}]
    for i, row in enumerate(first):
        row['timestamp'] = i + 1.0
    aux = ctx.module('aux_session')
    child.api_call(ctx, copy.deepcopy(first), 1)
    aux.drain_live_child_stores(60)
    counts = [len(reader._store.get_session_messages('child'))]
    view = copy.deepcopy(first) + [{'role': 'assistant', 'content': 'final', 'timestamp': 5.0}]
    for n in (2, 3):
        if route == 'live':
            child.api_call(ctx, copy.deepcopy(view), n)
        else:
            child.end(engine, copy.deepcopy(view))
        aux.drain_live_child_stores(60)
        counts.append(len(reader._store.get_session_messages('child')))
    calls = [r for r in reader._store.get_session_messages('child') if r.get('tool_calls')]
    assert counts == [4, 5, 5] and len(calls) == 1, (counts, len(calls))


@pytest.mark.parametrize('route', ['live', 'end'])
@pytest.mark.parametrize('transport', ['direct', 'json'])
@pytest.mark.parametrize('note', ['😀', '\ud800'], ids=['surrogate-pair', 'lone-surrogate'])
@pytest.mark.parametrize('redaction', [False, True], ids=['redaction-off', 'redaction-on'])
def test_an_identical_call_with_surrogate_code_units(host, redaction, note, transport, route, request):  # noqa: F811
    # gpt-6-astra (#2572 round 5): arguments holding raw surrogate code units, sent as Python strings (direct) or
    # through JSON text (json). Recorded on main 7e67daf16 first: [4, 5, 5] rows and one call row in all 16 cases.
    import copy
    import dataclasses
    import os

    from tests.test_issue_2482_interrupted_child import _setup

    _, _, ctx, reader, child, engine = _setup(host, 'own_engine')
    for each in {id(e): e for e in (reader, engine)}.values():
        each._config = dataclasses.replace(each._config, sensitive_patterns_enabled=redaction)
    first = [{'role': 'user', 'content': 'go'},
             _call(json.dumps({'api_key': 'sk-' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0', 'note': note},
                              ensure_ascii=False)),
             {'role': 'tool', 'tool_call_id': 'k1', 'content': 'ok'}, {'role': 'assistant', 'content': 'done'}]
    for i, row in enumerate(first):
        row['timestamp'] = i + 1.0
    if transport == 'json':
        first = json.loads(json.dumps(first, ensure_ascii=True))
    aux = ctx.module('aux_session')
    child.api_call(ctx, copy.deepcopy(first), 1)
    aux.drain_live_child_stores(60)
    counts = [len(reader._store.get_session_messages('child'))]
    view = copy.deepcopy(first) + [{'role': 'assistant', 'content': 'final', 'timestamp': 5.0}]
    for n in (2, 3):
        if route == 'live':
            child.api_call(ctx, copy.deepcopy(view), n)
        else:
            child.end(engine, copy.deepcopy(view))
        aux.drain_live_child_stores(60)
        counts.append(len(reader._store.get_session_messages('child')))
    calls = len([r for r in reader._store.get_session_messages('child') if r.get('tool_calls')])
    record = os.environ.get('SURROGATE_RECORD')
    if record:  # recording never skips the assert below
        with open(record, 'a') as out:
            out.write(json.dumps([request.node.callspec.id, counts, calls]) + '\n')
    assert (counts, calls) == ([4, 5, 5], 1), (counts, calls)
