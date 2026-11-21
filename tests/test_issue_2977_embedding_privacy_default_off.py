"""#2977: cloud embedding privacy requires explicit opt-in."""

from contextlib import closing, contextmanager
from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_lcm import command, embedding_provider, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.embedding_maintenance import (
    incremental_embedding_state,
    run_incremental_embedding_pass,
)
from hermes_lcm.ingest_policy import sensitive_pattern_status
from hermes_lcm.store import MessageStore
from tests.test_embedding_privacy import CaptureProvider


RAW = "Zebrawood orchard password=fixture-passphrase"
FILTERED = "Zebrawood orchard password=[LCM embedding privacy: name=password_assignment]"


@pytest.fixture
def providers(monkeypatch):
    by_model = {}

    def resolve(config, **_kwargs):
        if config.embedding_model not in by_model:
            provider = CaptureProvider()
            provider.model_id = config.embedding_model
            by_model[provider.model_id] = provider
        return by_model[config.embedding_model]

    monkeypatch.setattr(command, "resolve_provider", resolve)
    monkeypatch.setattr(tools, "resolve_provider", resolve)

    def probe(config):
        return {
            "available": True, "detail": "offline fake",
            "provider": "voyage", "model": config.embedding_model,
        }

    monkeypatch.setattr(embedding_provider, "probe_provider_availability", probe)
    monkeypatch.setattr(tools, "probe_provider_availability", probe)
    return by_model


@contextmanager
def _open_store(config, tmp_path):
    with (
        closing(MessageStore(config.database_path, ingest_protection_config=config)) as store,
        closing(SummaryDAG(config.database_path)) as dag,
    ):
        yield SimpleNamespace(
            _store=store, _dag=dag, _config=config,
            _hermes_home=str(tmp_path), current_session_id="orchard",
        )


def _seed(engine, text=RAW, timestamp=1):
    store_id = engine._store.append("orchard", {"role": "user", "content": text})
    engine._dag.add_node(SummaryNode(
        session_id="orchard", depth=0, summary=text, source_ids=[store_id],
        source_type="messages", created_at=timestamp, latest_at=timestamp,
    ))


def _recall(engine, include="summaries"):
    response = json.loads(tools.lcm_recall({"query": "Zebrawood orchard", "include": include}, engine=engine))
    assert "error" not in response
    assert response["hits"]
    return response


@pytest.mark.issue(2977)
@pytest.mark.parametrize("setting", ["unset", "none", "true"])
def test_cloud_embedding_documents_require_explicit_true(tmp_path, monkeypatch, providers, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "voyage-4-large")
    if setting == "true":
        monkeypatch.setenv("LCM_EMBEDDING_PRIVACY_ENABLED", "true")
    else:
        monkeypatch.delenv("LCM_EMBEDDING_PRIVACY_ENABLED", raising=False)
    config = LCMConfig.from_env()
    config.database_path = str(tmp_path / "embeddings.db")
    if setting == "none":
        config = replace(config, embedding_privacy_enabled=None)
    enabled = setting == "true"

    with _open_store(config, tmp_path) as engine:
        _seed(engine)
        assert "status: ready" in command.handle_lcm_command("embed warmup", engine)
        assert "status: complete" in command.handle_lcm_command("embed backfill --apply", engine)
    assert providers[config.embedding_model].documents == [[FILTERED if enabled else RAW]]
    status = sensitive_pattern_status(config)
    assert status["embedding_privacy_enabled"] is enabled
    assert status["embedding_privacy_setting"] == ("on" if enabled else "off")


def _durable_rows(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.text_factory = bytes
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("messages", "messages_fts", "summary_nodes", "summary_node_provenance")
        }


def _vector_rows(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        return {
            task: conn.execute(
                f"SELECT p.revision, p.active, v.* FROM {table} AS v "
                "JOIN lcm_embedding_profile AS p USING (identity_hash) ORDER BY p.active, v.rowid"
            ).fetchall()
            for task, table in (("summary", "lcm_embedding_vectors"), ("chunk", "lcm_chunk_vectors"))
        }


@pytest.mark.issue(2977)
@pytest.mark.parametrize("spelling", ["voyage", "VOYAGE", "voyageai", " VOYAGEAI "])
def test_summary_backfill_migrates_provider_spellings(tmp_path, monkeypatch, providers, spelling):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", spelling)
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "voyage-4-large")
    config = LCMConfig.from_env()
    config.database_path = str(tmp_path / "embeddings.db")
    assert embedding_provider.resolve_provider(config, for_backfill=True).provider_id == "voyage"
    protected = replace(config, embedding_provider="voyage", embedding_privacy_enabled=True)
    with _open_store(protected, tmp_path) as engine:
        _seed(engine)
        assert "status: ready" in command.handle_lcm_command("embed warmup", engine)

    result = command._embedding_backfill_summary_run(
        config, config.database_path, apply=True, limit=10, retry_uncertain=False,
    )
    assert result["status"] == "complete", result
    assert result["privacy_revision"] == "privacy:off"
    assert result["embedded"] == 1
    assert providers[config.embedding_model].documents == [[RAW]]


@pytest.mark.issue(2977)
def test_summary_backfill_previews_migration_without_writes(tmp_path, monkeypatch, providers):
    db_path = tmp_path / "embeddings.db"
    config = LCMConfig(
        database_path=str(db_path), embeddings_enabled=True,
        embedding_provider="voyage", embedding_model="voyage-4-large",
    )
    protected = replace(config, embedding_privacy_enabled=True)
    with _open_store(protected, tmp_path) as engine:
        _seed(engine)
        assert "status: ready" in command.handle_lcm_command("embed warmup", engine)
        assert "status: complete" in command.handle_lcm_command("embed backfill --apply", engine)
        _seed(engine, "The second orchard has several young trees", timestamp=2)
        _seed(engine, "The newest summary is also pending", timestamp=3)

    selections = []
    pending_rows = command._embedding_pending_rows

    def observe_pending(conn, identity, limit):
        pending, rows = pending_rows(conn, identity, limit)
        selections.append((identity, [row["node_id"] for row in rows]))
        return pending, rows

    monkeypatch.setattr(command, "_embedding_pending_rows", observe_pending)
    before = db_path.read_bytes()
    options = {"limit": 10, "retry_uncertain": False}
    preview = command._embedding_backfill_summary_run(config, db_path, apply=False, **options)
    assert db_path.read_bytes() == before
    assert providers[config.embedding_model].documents == [[FILTERED]]
    applied = command._embedding_backfill_summary_run(config, db_path, apply=True, **options)

    assert preview["status"] == "dry-run", preview
    assert applied["status"] == "complete", applied
    preview_selection, apply_selection = selections[:2]
    assert preview_selection == apply_selection
    assert preview["pending"] == applied["pending"] == 3
    assert applied["embedded"] == 3
    for field in ("selected", "estimated_tokens", "estimated_cost_tokens", "estimated_batches",
                  "privacy_revision", "error"):
        assert preview[field] == applied[field], field
    assert preview["estimated_cost_tokens"] > 0


@pytest.mark.issue(2977)
def test_reopened_cloud_store_recalls_and_reembeds_under_default(tmp_path, providers):
    default = LCMConfig(
        database_path=str(tmp_path / "embeddings.db"), embeddings_enabled=True,
        embedding_provider="voyage", embedding_model="voyage-4-large",
    )
    protected = replace(default, embedding_privacy_enabled=True)
    with _open_store(protected, tmp_path) as engine:
        _seed(engine)
        assert "status: ready" in command.handle_lcm_command("embed warmup", engine)
        report = command.handle_lcm_command("embed backfill --corpus both --apply --confirm-raw-text", engine)
        assert report.count("status: complete") == 2
    previous_rows = _durable_rows(default.database_path)
    previous_vectors = _vector_rows(default.database_path)
    assert set(len(rows) for rows in previous_vectors.values()) == {1}
    assert all(rows[0][0].startswith("privacy:v3:") for rows in previous_vectors.values())

    with _open_store(default, tmp_path) as engine:
        response = _recall(engine, "all")
        assert response["degraded"] is True
        assert "embedding_identity_stale:" in response["degraded_reason"]
        assert response["provenance"]["coverage"]["fts"] == "ok"
        assert tools._embedding_provider_health_check(engine)["detail"]["stale_tasks"] == ["summary", "chunk"]
        assert run_incremental_embedding_pass(default.database_path, default) == "summaries=complete:1"
        assert incremental_embedding_state(default.database_path)[1] == 0
        assert _recall(engine)["provenance"]["coverage"]["summary"] == "full"
        assert tools._embedding_provider_health_check(engine)["detail"]["stale_tasks"] == ["chunk"]
        assert "status: ready" in command.handle_lcm_command("embed warmup", engine)
        report = command.handle_lcm_command("embed backfill --corpus chunks --apply --confirm-raw-text", engine)
        assert "status: complete" in report
        assert _recall(engine, "all")["provenance"]["coverage"]["chunk"] == "full"
        assert tools._embedding_provider_health_check(engine)["status"] == "pass"

    assert _durable_rows(default.database_path) == previous_rows
    for task, rows in _vector_rows(default.database_path).items():
        assert len(rows) == 2
        old, new = rows
        revision, _, *vector = previous_vectors[task][0]
        assert old == (revision, 0, *vector)
        assert new[:2] == ("privacy:off", 1)
    for provider in providers.values():
        assert provider.documents == [[FILTERED], [RAW]]
