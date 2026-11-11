"""#1188 unit 2: opt-in session breadth changes only the read-side order."""
import json

import pytest

from hermes_lcm import config, recall_delivery, tools
from tests import test_recall_session_anchor, test_lcm_recall

recall_engine = test_lcm_recall.recall_engine
orchard = test_recall_session_anchor.orchard


def test_default_bytes(orchard):
    assert config.LCMConfig().recall_min_sessions == 0
    golden = json.loads(test_recall_session_anchor.GOLDEN.read_text())
    assert test_recall_session_anchor.outputs(orchard) == golden['outputs']
    for detail in ('snippets', 'answer_ready'):
        assert tools.lcm_recall({'query': 'orchard', 'detail': detail}, engine=orchard,
                                min_sessions=0) == golden['outputs'][detail]


def entries(sessions):
    return [dict(hit=dict(session_id=s, store_id=i), score=1 / (i + 1))
            for i, s in enumerate(sessions)]


@pytest.mark.parametrize('sessions,limit,requested,anchor,expected,promoted', [
    (['A', 'A', 'A', 'B', 'C'], 3, 3, 1, [1, 3, 4, 0, 2], 2),
    (['A', 'A', 'B'], 2, 2, 1, [1, 2, 0], 1),
    (['A', None, 'B'], 2, 10, 0, [0, 2, 1], 1),
    (['A', 'B', 'C'], 1, 10, 0, [0, 1, 2], 0),
    (['A', 'A'], 2, 10, 0, [0, 1], 0),
    ([], 2, 2, None, [], 0),
])
def test_floor_preserves_entries_scores_and_anchor(sessions, limit, requested, anchor, expected, promoted):
    ordered = entries(sessions)
    before = json.dumps(ordered)
    protected = ordered[anchor] if anchor is not None else None
    result, info = tools._lcm_recall_session_floor(
        ordered, limit, requested, protected_entry=protected)
    assert [e['hit']['store_id'] for e in result] == expected
    assert sorted(map(id, result)) == sorted(map(id, ordered))
    assert json.dumps(ordered) == before
    if protected is not None:
        assert any(e is protected for e in result[:limit])
    assert info == dict(requested=requested, achieved=min(limit, len(set(sessions) - {None})),
                        promoted=promoted, delivered=None)


@pytest.fixture
def ranked(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.recall_reference_strict = False
    monkeypatch.setattr(tools.time, 'time', lambda: 1700000200)
    ids = [e._store.append(s, {'role': 'user', 'content': f'needle My favorite hobby is pottery. row {i}',
                              'timestamp': 1700000000}) for i, s in enumerate(('A', 'A', 'A', 'B', 'C'))]
    original = tools._lcm_recall_fts_arm
    calls = []
    def arm(*a, **kw):
        calls.append(kw['candidate_limit'])
        hits, error = original(*a, **kw)
        return sorted(hits, key=lambda h: h['store_id']), error
    monkeypatch.setattr(tools, '_lcm_recall_fts_arm', arm)
    return e, ids, calls


@pytest.mark.parametrize('guidance', ['v1', 'v2'])
@pytest.mark.parametrize('arg,requested', [({}, 3), ({'min_sessions': 2}, 2),
    ({'min_sessions': None}, 3), ({'min_sessions': 'junk'}, 3),
    ({'min_sessions': 99}, 10), ({'min_sessions': -4}, 0), ({'min_sessions': 0}, 0)])
def test_override_pool_and_read_only(ranked, guidance, arg, requested):
    e, ids, calls = ranked
    e._config.recall_min_sessions = 3
    e._config.recall_guidance = guidance
    changes = e._store.connection.total_changes
    result = json.loads(tools.lcm_recall({'query': 'needle', 'limit': 3, **arg}, engine=e))
    assert calls == [100 if requested else 50]
    if requested:
        assert result['provenance']['session_floor'] == dict(
            requested=requested, achieved=min(requested, 3), promoted=min(requested, 3) - 1,
            delivered=min(requested, 3))
        assert [h['store_id'] for h in result['hits']] == ([ids[0], ids[1], ids[3]] if requested == 2 else [ids[0], ids[3], ids[4]])
    else:
        assert 'session_floor' not in result['provenance']
        assert [h['store_id'] for h in result['hits']] == ids[:3]
    assert e._store.connection.total_changes == changes


@pytest.mark.parametrize('value,expected', [('4', 4), ('99', 10), ('-1', 0), ('junk', 0)])
def test_config_env(monkeypatch, value, expected):
    monkeypatch.setenv('LCM_RECALL_MIN_SESSIONS', value)
    assert config.LCMConfig.from_env().recall_min_sessions == expected


def test_config_yaml_and_environment_precedence(monkeypatch):
    monkeypatch.setattr(config, '_load_hermes_config_yaml_for_env', lambda env: {
        'lcm': {'recall_min_sessions': 7}})
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_min_sessions == 7
    assert 'recall_min_sessions' not in cfg.ignored_config_yaml_lcm_keys
    monkeypatch.setenv('LCM_RECALL_MIN_SESSIONS', '2')
    assert config.LCMConfig.from_env().recall_min_sessions == 2


@pytest.mark.parametrize('guidance', ['v1', 'v2'])
def test_delivery_count_after_fitting(ranked, monkeypatch, guidance):
    e, _, _ = ranked
    e._config.recall_guidance = guidance
    args = {'query': 'needle', 'detail': 'answer_ready', 'limit': 3, 'min_sessions': 3}
    full = json.loads(tools.lcm_recall(args, engine=e))
    assert full['provenance']['session_floor']['delivered'] == 3
    cap = len(json.dumps(full, ensure_ascii=False)) - len(json.dumps(full['hits'][-1], ensure_ascii=False)) + 40
    monkeypatch.setattr(tools, '_LCM_RECALL_RESPONSE_CHAR_CAP', cap)
    fitted = json.loads(tools.lcm_recall(args, engine=e))
    assert len(fitted['hits']) == 2
    assert fitted['provenance']['session_floor'] == dict(requested=3, achieved=3, promoted=2, delivered=2)


def test_floor_precedes_later_statements_without_lost_or_duplicate_entries(ranked, monkeypatch):
    e, ids, _ = ranked
    later = e._store.append('D', {'role': 'user', 'content': 'My favorite hobby is now painting.', 'timestamp': 1700000100})
    original = recall_delivery.deliver_recall
    captured = []
    def deliver(ordered, **kw):
        captured.extend(ordered)
        return original(ordered, **kw)
    monkeypatch.setattr(recall_delivery, 'deliver_recall', deliver)
    result = json.loads(tools.lcm_recall({'query': 'needle', 'limit': 3, 'min_sessions': 3}, engine=e, later_statements=1))
    assert [x['hit']['store_id'] for x in captured[:3]] == [ids[0], ids[3], ids[4]]
    assert sorted(x['hit']['store_id'] for x in captured) == ids
    delivered = [h['store_id'] for h in result['hits']]
    assert delivered == [ids[0], later, ids[3]]
    assert len(delivered) == len(set(delivered))
    assert result['provenance']['session_floor']['delivered'] == 3


def test_anchor_is_protected_through_floor_and_later_placement(ranked, monkeypatch):
    e, ids, _ = ranked
    def anchor(ordered, *a, **kw):
        return [ordered[1], ordered[0], *ordered[2:]], {'fired': True, 'position': 2}
    monkeypatch.setattr(tools, 'place_fts_anchor', anchor)
    seen = []
    original = tools._lcm_recall_later_statements
    def later(engine, ordered, **kw):
        seen.append(kw['fts_anchor_identity'])
        assert ids[0] in [entry['hit']['store_id'] for entry in ordered[:3]]
        return original(engine, ordered, **kw)
    monkeypatch.setattr(tools, '_lcm_recall_later_statements', later)
    result = json.loads(tools.lcm_recall({'query': 'needle', 'limit': 3, 'min_sessions': 3},
                                       engine=e, later_statements=1))
    assert seen == [('message', ids[0])]
    assert [h['store_id'] for h in result['hits']] == [ids[0], ids[3], ids[4]]


def test_session_delivery_refreshes_floor(ranked):
    e, _, _ = ranked
    e._config.recall_min_sessions = 3
    result = json.loads(tools.lcm_recall({'query': 'needle', 'limit': 3}, engine=e,
                                       delivery_unit='session'))
    sessions = {h['session_id'] for h in result['hits']}
    assert result['provenance']['session_floor']['delivered'] == len(sessions) == 3
