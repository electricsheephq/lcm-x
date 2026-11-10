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


def _call():
    def b64(n):
        return base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(n))).decode('ascii')
    long = b64(3420)
    wrapped = '\n'.join(long[i:i + 76] for i in range(0, len(long), 76))  # line-wrapped base64
    arguments = json.dumps({'path': 'a.b64', 'content': 'hdr\n' + wrapped + '\nend'})
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'm1', 'type': 'function', 'function': {'name': 'write_file', 'arguments': arguments}}]}


@pytest.mark.xfail(strict=True, raises=AssertionError,
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
    assert '[Externalized LCM ingest payload:' in json.dumps(stored)  # ingest really externalized the payload
    view = copy.deepcopy(first) + [{'role': 'assistant', 'content': 'FINAL', 'timestamp': 11.0}]
    child.api_call(ctx, view, 2)
    aux.drain_live_child_stores(60.0)
    calls = [row for row in reader._store.get_session_messages('child') if row.get('tool_calls')]
    assert len(calls) == 1, f'{len(calls)} call rows'
