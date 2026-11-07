"""Legacy and current-GA rollback: no unprefixed read/write of a prefixed corpus."""

from pathlib import Path

import pytest

from tests.compat.helpers import PREVIOUS_GA, run_old

BASE = "849da257b52fc9833955928816dd1585184988b1"
# #1942's regression specifically needs the reader that predates prefix isolation.
LEGACY_READER = "v0.28.1"
QUERY = "query: "
DOCUMENT = "document: "

SCENARIO = r'''
import socket, sqlite3, struct
from hermes_lcm import command, tools, embedding_provider as ep, embedding_maintenance as maintenance
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.vector_store import VectorStore
VectorStore._now = staticmethod(lambda: '2000-01-01T00:00:00+00:00')
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden'))
config = LCMConfig(**dict(options, embeddings_enabled=True, embedding_provider='ollama',
                         embedding_model='prefix-test', embedding_binary_prescreen=False))
config.embedding_query_prefix = 'query: ' if data == 'prefix' else ''
config.embedding_document_prefix = 'document: ' if data == 'prefix' else ''
calls = []
def transport(**kwargs):
    texts = kwargs['payload']['input']
    calls.extend(texts)
    vectors = [[1.0, 0.0] if text.startswith(('document: ', 'query: ')) else [0.0, 1.0]
               for text in texts]
    return ep.HttpResponse(200, {}, json.dumps({'embeddings': vectors}).encode())
ep._default_http_transport = transport
ep.probe_provider_availability = lambda config: {'available': True, 'detail': 'synthetic'}
def seed(text):
    with __import__('contextlib').closing(SummaryDAG(db)) as dag:
        return dag.add_node(SummaryNode(session_id='fixture', summary=text, depth=0,
                            source_token_count=10, created_at=1.0, earliest_at=1.0, latest_at=1.0))
if mode == 'initial':
    seed('first summary')
elif mode == 'rollback':
    seed('new summary')
resolution = {}
if mode != 'rollback':
    registration = command._embedding_register_profiles(config, db, allow_download=False)
    assert isinstance(registration, dict), registration
    result = command._embedding_backfill_summary_run(config, db, apply=True, limit=10, retry_uncertain=False)
    assert isinstance(result, dict) and not result['failed'], result
    outcome = result['status']
else:
    store = VectorStore(db, config=config)
    def resolved(name, row):
        resolution[name] = row['identity_hash'] if row is not None else None
    for provider in (None, 'ollama'):
        resolved('summary-' + str(provider), store._resolve_profile('prefix-test', provider=provider))
        resolved('chunk-' + str(provider), store._resolve_chunk_profile('prefix-test', provider=provider))
    resolved('current-summary', store._current_profile())
    resolved('current-chunk', store._current_chunk_profile())
    resolved('backfill-summary', command._embedding_current_profile(store._conn))
    resolved('backfill-chunk', command._chunk_current_profile(store._conn))
    for task in ('summary', 'chunk'):
        resolution['revision-' + task] = tools._lcm_embedding_profile_revision(
            store._conn, provider='ollama', model='prefix-test', task=task)
    from types import SimpleNamespace
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=db),
                             _dag=SimpleNamespace(connection=store._conn))
    resolution['active-report'] = tools._active_embedding_profile_exists(engine)
    resolution['usable-report'] = tools._lcm_recall_has_usable_vector_corpus(
        engine, run_summary=True, run_chunk=True, provider_name='ollama', model_name='prefix-test',
        excluded_session_ids=set(), deadline=__import__('time').monotonic() + 5)
    provider = ep.resolve_provider(config)
    query = provider.embed_query('find summary')
    # Prefix-aware retrieval passes the provider's identity suffix (retrieval_core).
    # The v0.28.1 API predates this argument and must keep its original probe.
    knn_options = {'revision_suffix': provider.identity_revision_suffix} if data == 'prefix-aware-reader' else {}
    resolution['knn'] = store.knn(query, model='prefix-test', provider='ollama', **knn_options).coverage
    resolution['knn-chunks'] = store.knn_chunks(query, model='prefix-test', provider='ollama', **knn_options).coverage
    store.close()
    outcome = maintenance.run_incremental_embedding_pass(db, config)
with sqlite3.connect(db) as conn:
    conn.row_factory = sqlite3.Row
    profiles = [dict(row) for row in conn.execute('SELECT * FROM lcm_embedding_profile ORDER BY identity_hash')]
    vectors = [dict(row) for row in conn.execute(
        'SELECT embedded_id, identity_hash, hex(vec) AS vec FROM lcm_embedding_vectors ORDER BY identity_hash, embedded_id')]
    meta = [dict(row) for row in conn.execute('SELECT * FROM lcm_embedding_meta ORDER BY identity_hash, embedded_id')]
    summaries = [dict(row) for row in conn.execute('SELECT * FROM summary_nodes ORDER BY node_id')]
print(json.dumps(dict(calls=calls, profiles=profiles, vectors=vectors, meta=meta,
                     summaries=summaries, resolution=resolution, outcome=outcome)))
'''


def cycle(tree, old, directory, host_root, prefix, *, prefix_aware_reader=False):
    directory.mkdir()
    initial = run_old(tree, directory, host_root, SCENARIO, 'initial', prefix)
    rollback = run_old(old, directory, host_root, SCENARIO, 'rollback',
                       'prefix-aware-reader' if prefix_aware_reader else '')
    upgraded = run_old(tree, directory, host_root, SCENARIO, 'upgrade', prefix)
    return initial, rollback, upgraded


def test_prefixed_identity_survives_legacy_ga_rollback(tmp_path, old_tree, host_root):
    tree = Path(__file__).resolve().parents[2]
    first, old, final = cycle(tree, old_tree(LEGACY_READER), tmp_path / 'prefix', host_root, 'prefix')
    prefixed = {row['identity_hash'] for row in first['profiles']}
    assert first['calls'] == [QUERY + 'warmup', DOCUMENT + 'first summary']
    # Check all old resolvers before maintenance, including provider-less and reporting paths.
    assert all(value is None for key, value in old['resolution'].items()
               if key not in {'knn', 'knn-chunks', 'active-report', 'usable-report'}), old
    assert old['resolution']['knn'] == old['resolution']['knn-chunks'] == 'none'
    assert old['resolution']['active-report'] is old['resolution']['usable-report'] is False
    assert old['outcome'] == 'no_profile'
    assert old['calls'] == ['find summary']
    assert old['vectors'] == first['vectors'], 'old code must never write under a prefixed hash'
    assert final['calls'] == [QUERY + 'warmup', DOCUMENT + 'new summary']
    assert {row['identity_hash'] for row in final['profiles']} == prefixed
    assert len(final['vectors']) == 2
    assert all(row['identity_hash'] in prefixed and row['vec'] == '0000803F00000000'
               for row in final['vectors']), 'every prefixed vector must encode prefixed text'


def test_prefixed_identity_survives_previous_ga_rollback(tmp_path, old_tree, host_root):
    tree = Path(__file__).resolve().parents[2]
    first, old, final = cycle(tree, old_tree(PREVIOUS_GA), tmp_path / 'prefix', host_root, 'prefix',
                             prefix_aware_reader=True)
    # Prefix-aware readers can inspect the profile; the unprefixed query and
    # maintenance paths must still refuse to use or contaminate its vectors.
    assert first['calls'] == [QUERY + 'warmup', DOCUMENT + 'first summary']
    assert old['calls'] == ['find summary']
    assert old['resolution']['knn'] == old['resolution']['knn-chunks'] == 'none'
    assert old['resolution']['usable-report'] is False
    assert old['vectors'] == first['vectors']
    assert old['meta'] == first['meta']
    assert old['summaries'][:len(first['summaries'])] == first['summaries']
    assert len(old['summaries']) == len(first['summaries']) + 1
    assert final['summaries'] == old['summaries']
    assert final['calls'] == [QUERY + 'warmup', DOCUMENT + 'new summary']
    prefixed = {row['identity_hash'] for row in first['profiles']}
    assert {row['identity_hash'] for row in final['profiles']} == prefixed
    assert len(final['vectors']) == 2
    assert all(row['identity_hash'] in prefixed and row['vec'] == '0000803F00000000'
               for row in final['vectors'])


def test_previous_ga_base_only_control(tmp_path, old_tree, host_root):
    old = old_tree(PREVIOUS_GA)
    first, rollback, final = cycle(old, old, tmp_path / 'base', host_root, '')
    assert first['calls'] == ['warmup', 'first summary']
    assert rollback['calls'] == ['find summary', 'new summary']
    assert rollback['resolution']['knn'] == 'full'
    assert 'summaries=complete:1' in rollback['outcome']
    assert final['calls'] == ['warmup']
    assert len(final['vectors']) == 2


def test_prefix_unset_rows_and_hashes_identical_to_main(tmp_path, old_tree, host_root):
    tree = Path(__file__).resolve().parents[2]
    baseline = old_tree('main-at-dispatch', {'main-at-dispatch': BASE})
    old = old_tree(PREVIOUS_GA)
    actual = cycle(tree, old, tmp_path / 'current', host_root, '')
    expected = cycle(baseline, old, tmp_path / 'main', host_root, '')
    assert actual == expected


def test_rc1_mixed_profile_is_quarantined_and_reembedded(tmp_path, old_tree, host_root):
    baseline = old_tree('rc1-encoding', {'rc1-encoding': BASE})
    old = old_tree(LEGACY_READER)
    first = run_old(baseline, tmp_path, host_root, SCENARIO, 'initial', 'prefix')
    mixed = run_old(old, tmp_path, host_root, SCENARIO, 'rollback', '')
    assert mixed['calls'] == ['find summary', 'new summary']
    assert {row['vec'] for row in mixed['vectors']} == {'0000803F00000000', '000000000000803F'}
    tree = Path(__file__).resolve().parents[2]
    repaired = run_old(tree, tmp_path, host_root, SCENARIO, 'upgrade', 'prefix')
    assert repaired['calls'][0] == QUERY + 'warmup'
    assert sorted(repaired['calls'][1:]) == sorted([DOCUMENT + 'first summary', DOCUMENT + 'new summary'])
    assert repaired['summaries'] == mixed['summaries']
    retired = [row for row in repaired['profiles'] if row['task'].startswith('retired:')]
    assert len(retired) == 2 and all(row['active'] == 0 and row['archived_at'] for row in retired)
    previous_hashes = {row['identity_hash'] for row in first['profiles']}
    assert not previous_hashes & {row['identity_hash'] for row in repaired['profiles']}
    assert [row for row in repaired['vectors'] if row['identity_hash'] in previous_hashes] == mixed['vectors']
    new_vectors = [row for row in repaired['vectors'] if row['identity_hash'] not in previous_hashes]
    assert len(new_vectors) == 2 and all(row['vec'] == '0000803F00000000' for row in new_vectors)
    again = run_old(tree, tmp_path, host_root, SCENARIO, 'upgrade', 'prefix')
    assert again['calls'] == [QUERY + 'warmup']
    assert again['profiles'] == repaired['profiles'] and again['vectors'] == repaired['vectors']
    rollback = run_old(old, tmp_path, host_root, SCENARIO, 'rollback', '')
    assert rollback['outcome'] == 'no_profile' and rollback['vectors'] == repaired['vectors']


@pytest.mark.parametrize('provider', ('ollama', 'openai-compatible'))
def test_old_resolvers_miss_nonempty_prefix_revision(provider, tmp_path, old_tree, host_root):
    # Exercise the other provider's profile shape without any external endpoint.
    script = SCENARIO.split("if mode == 'initial':")[0] + r'''
store = VectorStore(db, config=config)
for task in ('summary', 'chunk'):
    store.register_profile('prefix-test', mode, 2, task=task,
                           revision='privacy:off' if mode == 'openai-compatible' else '',
                           revision_suffix='+prefix:0123456789ab')
store.close()
print(json.dumps(True))
'''
    tree = Path(__file__).resolve().parents[2]
    assert run_old(tree, tmp_path, host_root, script, provider)
    script = r'''
from hermes_lcm import command, tools
from hermes_lcm.vector_store import VectorStore
store = VectorStore(db, config=LCMConfig(embeddings_enabled=False))
assert store._resolve_profile('prefix-test', provider=mode) is None
assert store._resolve_profile('prefix-test') is None
assert store._resolve_chunk_profile('prefix-test', provider=mode) is None
assert store._resolve_chunk_profile('prefix-test') is None
assert store._current_profile() is store._current_chunk_profile() is None
assert command._embedding_current_profile(store._conn) is command._chunk_current_profile(store._conn) is None
for task in ('summary', 'chunk'):
    assert tools._lcm_embedding_profile_revision(store._conn, provider=mode, model='prefix-test', task=task) is None
store.close()
print(json.dumps(True))
'''
    assert run_old(old_tree(LEGACY_READER), tmp_path, host_root, script, provider)
