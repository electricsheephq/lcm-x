"""#2559 stays open at default settings for v0.30.0 (fix planned for v0.30.1 on #2572).

A delegated child's tool call whose arguments hold an inline encoded payload is stored once more at the next handover
when the host re-sends it unchanged: the session-end compare reads the stored side with LCM-X's own escaping in place.
This strict xfail pins today's second copy (the compare matrix's T7 line-wrapped shape, live hook); a fix flips it.
"""

import base64
import copy
import json

import pytest

from tests.test_issue_1931_delegate_children import host  # noqa: F401
from tests.test_issue_2482_interrupted_child import _setup


class SecondCopyStored(AssertionError):
    """Raised only by the row-count check, so a hook failure is never taken as the #2559 xfail."""


def _call(inline=True):
    def b64(n):
        return base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(n))).decode('ascii')
    long = b64(3420)
    wrapped = '\n'.join(long[i:i + 76] for i in range(0, len(long), 76))  # line-wrapped base64
    arguments = json.dumps({'path': 'a.b64', 'content': 'hdr\n' + (wrapped if inline else 'plain body') + '\nend'})
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'm1', 'type': 'function', 'function': {'name': 'write_file', 'arguments': arguments}}]}


@pytest.mark.xfail(strict=True, raises=SecondCopyStored,
                   reason='#2559: an identical inline-payload call is stored once more at the next handover')
def test_an_identical_inline_payload_call_is_stored_once_across_handovers(host):  # noqa: F811
    _, _, ctx, reader, child, _engine = _setup(host, 'own_engine')
    first = [{'role': 'user', 'content': 'Write the attachment file.'}, _call(),
             {'role': 'tool', 'tool_call_id': 'm1', 'content': 'wrote a.b64'},
             {'role': 'assistant', 'content': 'Done: a.b64 written.'}]
    for i, row in enumerate(first):
        row['timestamp'] = float(i + 1)
    aux = ctx.module('aux_session')
    child.api_call(ctx, copy.deepcopy(first), 1)
    aux.drain_live_child_stores(60.0)
    stored = reader._store.get_session_messages('child')
    if '[Externalized LCM ingest payload:' not in json.dumps(stored):  # setup, outside the strict xfail
        pytest.fail('setup: ingest did not externalize the payload')
    view = copy.deepcopy(first) + [{'role': 'assistant', 'content': 'FINAL', 'timestamp': 11.0}]
    child.api_call(ctx, view, 2)
    aux.drain_live_child_stores(60.0)
    rows = reader._store.get_session_messages('child')
    if sum(row.get('content') == 'FINAL' for row in rows) != 1:  # surplus, never loss: outside the xfail
        pytest.fail('setup: the second handover did not store FINAL once')
    calls = [row for row in rows if row.get('tool_calls')]
    if len(calls) != 1:
        raise SecondCopyStored(f'{len(calls)} call rows')


def test_an_identical_plain_call_is_stored_once_across_handovers(host):  # noqa: F811
    """Control: the same handovers with a plain argument store the call once."""
    _, _, ctx, reader, child, _engine = _setup(host, 'own_engine')
    first = [{'role': 'user', 'content': 'Write the file.'}, _call(inline=False),
             {'role': 'tool', 'tool_call_id': 'm1', 'content': 'wrote a.b64'},
             {'role': 'assistant', 'content': 'Done: a.b64 written.'}]
    for i, row in enumerate(first):
        row['timestamp'] = float(i + 1)
    aux = ctx.module('aux_session')
    child.api_call(ctx, copy.deepcopy(first), 1)
    aux.drain_live_child_stores(60.0)
    view = copy.deepcopy(first) + [{'role': 'assistant', 'content': 'FINAL', 'timestamp': 11.0}]
    child.api_call(ctx, view, 2)
    aux.drain_live_child_stores(60.0)
    rows = reader._store.get_session_messages('child')
    assert len([row for row in rows if row.get('tool_calls')]) == 1
    assert sum(row.get('content') == 'FINAL' for row in rows) == 1
