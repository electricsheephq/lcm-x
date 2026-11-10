"""#2559 lane edit: only a stored string that decodes to a list or dict takes the per-leaf restore.

Every other string with an ingest ref keeps the #2539 path, including its whole-placeholder payload load
and that load's session check.
"""

import json

import pytest

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
