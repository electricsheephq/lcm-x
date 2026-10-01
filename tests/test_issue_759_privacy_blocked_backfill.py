"""#759: withhold refused provider copies without starving the rest of a window."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.ingest_protection as privacy_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.embedding_provider import EmbeddedDocumentBatch
from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError, embedding_privacy_revision
from hermes_lcm.vector_store import VectorStore

# Synthetic key-body shape from the verified #759 reproduction, not a credential.
REFUSED_TEXT = (
    "Pasted from the deploy console:\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n"
    "MHcCAQEEIQD1eJ7yhkG0987xyzABCDEFghijkLMNOPqrstuvwxyz0987654321pq\n"
    "end of paste."
)
SYNTHETIC_SECRET = "ISSUE759SYNTHETICAPIKEY"


class CaptureProvider:
    provider_id = "voyage"

    def __init__(self, model):
        self.model_id = model
        self.documents = []
        self.groups = []
        self.last_skipped_documents = []
        self.supports_contextualized_grouping = model == "voyage-context-3"

    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [[1.0, float(i)] for i, _ in enumerate(texts)]

    def embed_chunk_group_batches(self, groups, *, before_dispatch):
        self.groups.append(groups)
        items = [item for group in groups for item in group]
        indexes = tuple(index for index, _text in items)
        before_dispatch(indexes)
        self.documents.append([text for _index, text in items])
        yield EmbeddedDocumentBatch(
            indexes=indexes, vectors=tuple((1.0, float(index)) for index in indexes)
        )


def _setup(tmp_path, monkeypatch, corpus, *, refused=(5,), contextualized=False):
    """Reuse the repro's direct rows and registered cloud identity; no network."""
    db = tmp_path / "backfill.db"
    model = "voyage-context-3" if contextualized else "voyage-4-large"
    config = LCMConfig(
        database_path=str(db), embeddings_enabled=True,
        embedding_provider="voyage", embedding_model=model,
    )
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=db))
    texts = [
        f"Document {i}: " + "Chunk safety sentence: please review the release checklist. " * 6
        + f"api_key={SYNTHETIC_SECRET}"
        for i in range(12)
    ]
    for index in refused:
        texts[index] = REFUSED_TEXT
    if corpus == "summary":
        dag = SummaryDAG(db)
        try:
            ids = [str(dag.add_node(SummaryNode(
                session_id="test-session", depth=0, summary=text,
                created_at=float(i), latest_at=float(i),
            ))) for i, text in enumerate(texts, start=1)]
        finally:
            dag.close()
    else:
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, "
                "source TEXT DEFAULT '', role TEXT NOT NULL, content TEXT, timestamp REAL NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO messages VALUES(?,?,?,?,?,?)",
                [(i, "test-session", "history", "user", text, float(i))
                 for i, text in enumerate(texts, start=1)],
            )
        ids = [f"{i}:0" for i in range(1, 13)]
    store = VectorStore(db, config=config)
    try:
        store.register_profile(
            model, "voyage", 2, revision=embedding_privacy_revision(config),
            task="chunk" if corpus == "chunks" else "summary",
        )
    finally:
        store.close()
    provider = CaptureProvider(model)
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kwargs: provider)
    args = "embed backfill" + (" --corpus chunks --confirm-raw-text" if corpus == "chunks" else "")
    return engine, ids, provider, args


def _embedded(engine, corpus):
    with sqlite3.connect(engine._store.db_path) as conn:
        if corpus == "chunks":
            return {str(row[0]) for row in conn.execute("SELECT chunk_id FROM lcm_chunk_meta")}
        return {str(row[0]) for row in conn.execute(
            "SELECT embedded_id FROM lcm_embedding_meta WHERE embedded_kind=?",
            ("summary",),
        )}


def _field(report, name):
    return next((line.split(": ", 1)[1] for line in report.splitlines()
                 if line.startswith(f"{name}: ")), None)


def _assert_withheld(report, count, *, selected):
    assert _field(report, "status") == "partial"
    assert _field(report, "privacy_blocked") == str(count)
    assert _field(report, "selected") == str(selected)
    assert "privacy_refused" not in report
    assert _field(report, "error") is None
    line = _field(report, "privacy_withheld")
    assert line is not None and line.startswith(f"{count} document(s)") and "stay pending" in line


def _assert_dispatch(engine, ids, provider, corpus):
    assert _embedded(engine, corpus) == set(ids) - {ids[5]}
    outbound = [text for request in provider.documents for text in request]
    assert len(outbound) == 11
    assert {text.split(":", 1)[0] for text in outbound} == {f"Document {i}" for i in range(12) if i != 5}
    assert all(REFUSED_TEXT not in text and SYNTHETIC_SECRET not in text for text in outbound)
    assert all("[LCM embedding privacy:" in text for text in outbound)
    with sqlite3.connect(engine._store.db_path) as conn:
        inflight_ids = {row[0] for row in conn.execute("SELECT embedded_id FROM lcm_embedding_backfill_inflight")}
    assert ids[5] not in inflight_ids


def test_summary_backfill_dispatches_remainder_when_one_document_is_refused(tmp_path, monkeypatch):
    engine, ids, provider, args = _setup(tmp_path, monkeypatch, "summary")
    report = handle_lcm_command(args + " --apply", engine)
    _assert_dispatch(engine, ids, provider, "summary")
    _assert_withheld(report, 1, selected=12)


def test_chunk_backfill_dispatches_remainder_when_one_message_is_refused(tmp_path, monkeypatch):
    engine, ids, provider, args = _setup(tmp_path, monkeypatch, "chunks")
    report = handle_lcm_command(args + " --apply", engine)
    _assert_dispatch(engine, ids, provider, "chunks")
    _assert_withheld(report, 1, selected=12)


@pytest.mark.parametrize("corpus", ["summary", "chunks"])
def test_refused_row_stays_pending_and_is_reported_again(tmp_path, monkeypatch, corpus):
    engine, ids, provider, args = _setup(tmp_path, monkeypatch, corpus)
    handle_lcm_command(args + " --apply", engine)
    calls = len(provider.documents)
    report = handle_lcm_command(args + " --apply", engine)
    _assert_withheld(report, 1, selected=1)
    assert _field(report, "embedded") == "0"
    assert _field(report, "remaining") == "1"
    assert len(provider.documents) == calls
    assert _embedded(engine, corpus) == set(ids) - {ids[5]}


@pytest.mark.parametrize("corpus", ["summary", "chunks"])
@pytest.mark.parametrize("policy_error", ["mismatch", "unknown", "empty", "no_revision", "raised"])
def test_policy_level_error_still_refuses_whole_run(tmp_path, monkeypatch, corpus, policy_error):
    engine, _ids, provider, args = _setup(tmp_path, monkeypatch, corpus)
    if policy_error == "mismatch":
        engine._config.sensitive_patterns = ["api_key"]
    elif policy_error == "unknown":
        engine._config.sensitive_patterns = ["api_key", "unknown-test-pattern"]
    elif policy_error == "empty":
        engine._config.sensitive_patterns = []
    elif policy_error == "no_revision":
        monkeypatch.setattr(command_mod, "embedding_privacy_revision", lambda _config: None)
    else:
        def raised(_config):
            raise EmbeddingPrivacyPolicyError("cloud embedding privacy test policy failure")
        monkeypatch.setattr(command_mod, "embedding_privacy_revision", raised)
    dry = handle_lcm_command(args, engine)
    assert _field(dry, "status") == "refused"
    report = handle_lcm_command(args + " --apply", engine)
    assert _field(report, "status") == "error"
    assert _field(report, "stop_reason") == "privacy_refused"
    assert _field(report, "privacy_withheld") is None
    assert provider.documents == []
    assert _embedded(engine, corpus) == set()


@pytest.mark.parametrize("corpus", ["summary", "chunks"])
def test_dry_run_reports_withheld_count_not_refused(tmp_path, monkeypatch, corpus):
    engine, _ids, provider, args = _setup(tmp_path, monkeypatch, corpus)
    report = handle_lcm_command(args, engine)
    assert _field(report, "status") == "dry-run"
    assert _field(report, "privacy_blocked") == "1"
    assert _field(report, "selected") == "12"
    assert "1 document(s)" in _field(report, "privacy_withheld")
    assert "stay pending" in _field(report, "privacy_withheld")
    assert _field(report, "error") is None
    assert provider.documents == []


@pytest.mark.parametrize("corpus", ["summary", "chunks"])
def test_all_selected_refused_reports_partial_not_complete(tmp_path, monkeypatch, corpus):
    engine, _ids, provider, args = _setup(tmp_path, monkeypatch, corpus, refused=range(12))
    def unexpected_resolution(*_args, **_kwargs):
        pytest.fail("an all-withheld selection must not resolve a provider")
    monkeypatch.setattr(command_mod, "resolve_provider", unexpected_resolution)
    report = handle_lcm_command(args + " --apply", engine)
    _assert_withheld(report, 12, selected=12)
    assert _field(report, "embedded") == "0"
    assert provider.documents == []


@pytest.mark.parametrize("corpus,contextualized", [("summary", False), ("chunks", False), ("chunks", True)])
def test_marked_rows_are_exactly_the_dispatched_ones(tmp_path, monkeypatch, corpus, contextualized):
    engine, ids, provider, args = _setup(tmp_path, monkeypatch, corpus, contextualized=contextualized)
    engine._config.embedding_max_batch_items = 4
    report = handle_lcm_command(args + " --apply", engine)
    _assert_dispatch(engine, ids, provider, corpus)
    _assert_withheld(report, 1, selected=12)
    if contextualized:
        assert provider.groups
        assert all(len(group) == 1 for groups in provider.groups for group in groups)


def _gauntlet_chunk_check(report, outbound, config):
    """Execute the instrument's actual chunk outcome + leak sweep, without its live matrix."""
    path = Path(__file__).resolve().parents[1] / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    spec = importlib.util.spec_from_file_location("issue759_phase_a", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(path.read_text())
    battery = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_planted_secret")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, node in enumerate(battery.body) if assigns(node, "chunk_start"))
    end = next(i for i, node in enumerate(battery.body) if assigns(node, "recall_start"))
    namespace = dict(vars(module))
    def command(_args, _engine):
        namespace["outbound"].extend(outbound)
        return report
    namespace.update(
        mod={"command": SimpleNamespace(handle_lcm_command=command), "ingest_protection": privacy_mod},
        engine=SimpleNamespace(_config=config), outbound=[], known=[],
        fixtures=[{"kind": "standard", "secrets": [SYNTHETIC_SECRET]}],
    )
    exec(compile(ast.Module(body=battery.body[start:end], type_ignores=[]), str(path), "exec"), namespace)


def test_gauntlet_chunk_battery_accepts_partial(tmp_path, monkeypatch):
    engine, _ids, provider, args = _setup(tmp_path, monkeypatch, "chunks")
    report = handle_lcm_command(args + " --apply", engine)
    assert _field(report, "status") == "partial"
    _gauntlet_chunk_check(report, [text for batch in provider.documents for text in batch], engine._config)


@pytest.mark.parametrize("failure", ["error", "no_blocked", "no_selected", "no_dispatch", "leak"])
def test_gauntlet_chunk_battery_rejects_invalid_partial(tmp_path, failure):
    config = LCMConfig()
    report = "status: partial\nprivacy_blocked: 1\nselected: 12"
    outbound = ["Chunk safety sentence [LCM embedding privacy: api_key]"]
    if failure == "error":
        report = report.replace("partial", "error") + "\nstop_reason: privacy_refused"
    elif failure == "no_blocked":
        report = report.replace("privacy_blocked: 1", "privacy_blocked: 0")
    elif failure == "no_selected":
        report = report.replace("selected: 12", "selected: 0")
    elif failure == "no_dispatch":
        outbound = []
    else:
        outbound.append(SYNTHETIC_SECRET)
    with pytest.raises(AssertionError):
        _gauntlet_chunk_check(report, outbound, config)
