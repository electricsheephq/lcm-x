"""T1-24: pin every non-test callback occurrence, without a profiler."""
import ast
import os
from collections import Counter
from pathlib import Path

from tests.message_store_callback_audit import METHODS, ROOT

# Each occurrence is pinned, including callback clearing and getattr strings.
ALLOWLIST = (
    ('bench/fmp/tests/test_switches.py', 'make_store.frozen_connect', 'create_function', 'instrument, not product: connection is the instrumented connection'),
    ('bench/instruments/busy_bar/holds.py', 'probe.traced_connect', 'set_trace_callback', 'instrument, not product: conn is the instrumented connection'),
    ('bench/instruments/local_catchup/tests/test_1653.py', 'test_08_audit_reads_one_snapshot.concurrent', 'set_trace_callback', 'instrument, not product: c is the instrumented connection'),
    ('bench/instruments/local_catchup/tests/test_i2_fix1.py', 'test_cheap_counts_ignore_new_trace_and_foreign_identity.connect', 'set_trace_callback', 'instrument, not product: c is the instrumented connection'),
    ('bench/instruments/reliability/test_reply_held_fault.py', 'test_kill_exit_checks_rollback_order', 'set_trace_callback', 'instrument, not product: held.reservation is the instrumented connection'),
    ('command.py', '_register_summary_has_text', 'create_function', 'VectorStore.connection or _embedding_read_connection; each opens a connection independently of MessageStore'),
    ('docs/process/evidence/1880-persisted-hold/cost/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-final-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-final-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-final-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-final-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-final-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-final-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2-failed-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2-failed-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2-failed-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2-failed-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2-failed-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-h2-failed-hosted/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-integration-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-integration-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-integration-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-integration-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-integration-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-integration-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-review-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-review-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-review-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-review-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe under docs/process/evidence; sets a trace callback on the MessageStore connection by design, to measure its statements'),
    ('docs/process/evidence/1880-persisted-hold/cost-review-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._dag._conn, the separate DAG connection'),
    ('docs/process/evidence/1880-persisted-hold/cost-review-delta/cost_probe.py', 'main', 'set_trace_callback', 'instrument, not product: archived cost probe on held._lifecycle._conn, the separate lifecycle connection'),
    ('memory_tools.py', '_assertions', 'create_function', 'AssertionStore.connection; AssertionStore._open_connection opens it independently of MessageStore'),
    ('recall_session.py', 'deliver_session', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('retrieval_core.py', '_run_pooled_knn', 'set_progress_handler', 'VectorStore/ReadOnlyVectorStore._conn; each constructor opens its own connection, separate from MessageStore'),
    ('retrieval_core.py', '_run_pooled_knn', 'set_progress_handler', 'VectorStore/ReadOnlyVectorStore._conn; each constructor opens its own connection, separate from MessageStore'),
    ('retrieval_core.py', 'hydrate_chunk_hits', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('retrieval_core.py', 'hydrate_semantic_nodes', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('retrieval_core.py', 'duplicate_identity_rows', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('tools.py', '_lcm_recall_event_time_rows', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('tools.py', '_lcm_recall_summary_event_time_rows', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('tools.py', '_lcm_recall_snippet_windows', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('tools.py', '_lcm_recall_summary_source_hits', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('tools.py', '_lcm_recall_has_usable_vector_corpus', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('tools.py', '_lcm_recall_adjacent_reply', 'set_progress_handler', 'local conn from sqlite3.connect(mode=ro) in this function; separate from MessageStore._conn'),
    ('vector_store.py', 'VectorStore._load_binary_with_deadline', 'set_progress_handler', 'VectorStore._conn; _init_db opens it independently of MessageStore'),
    ('vector_store.py', 'VectorStore._load_binary_with_deadline', 'set_progress_handler', 'VectorStore._conn; _init_db opens it independently of MessageStore'),
)


class _Sites(ast.NodeVisitor):
    def __init__(self, path, sites):
        self.path, self.sites, self.names = path, sites, []

    def scoped(self, node):
        self.names.append(node.name)
        self.generic_visit(node)
        self.names.pop()

    visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = scoped

    def add(self, method):
        self.sites[(self.path, '.'.join(self.names), method)] += 1

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr in METHODS:
            self.add(node.func.attr)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in METHODS:
            self.add(node.value)


def test_callback_sites_match_allowlist():
    actual = Counter()
    for folder, directories, files in os.walk(ROOT):
        directories[:] = [name for name in directories if not name.startswith('.') and name != '__pycache__'
                          and not (str(ROOT) == folder and name == 'tests')]
        for filename in sorted(files):
            if filename.endswith('.py'):
                path = Path(folder) / filename
                _Sites(path.relative_to(ROOT).as_posix(), actual).visit(ast.parse(path.read_text()))
    expected = Counter(entry[:3] for entry in ALLOWLIST)
    assert actual == expected, f'T1-24 callback sites changed: added={actual - expected}; removed={expected - actual}'
