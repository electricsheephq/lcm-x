"""Review regressions: revision repair, cloud coverage and low-parent fences."""
import importlib
import json
from types import SimpleNamespace

import pytest

from tests.test_issue_1536_chunk_revision import (
    LONG, SHORT, add_rows, command, counts, metadata, store as store,
)
from tests.test_issue_1543_coverage_marker import certify, marker, older_walk, run


def assert_repaired_or_visible(store, missing):
    """A repair may finish or retain backlog, but cannot hide an unembedded row."""
    conn, identity = store[1].connection, store[2].identity_hash
    absent = missing - counts(store).keys()
    if absent:
        coverage = command._chunk_coverage(conn, identity, command._chunk_cursor(conn, identity)[0])
        if command._chunk_policy_revision(conn, identity) >= command.CHUNK_POLICY_REVISION:
            assert not coverage or all(not coverage[0] <= sid <= coverage[1] for sid in absent)
        assert command._chunk_backlog_estimate(conn, identity, "conversational") != "0"


@pytest.mark.parametrize("arrival", [False, True])
def test_v0281_midrepair_extension_does_not_hide_gap(store, arrival):
    add_rows(store, [(i, "user", LONG) for i in range(1, 8)] +
             [(8, "user", SHORT), (9, "assistant", LONG)], {1, 3, 4, 5, 6, 7, 9})
    metadata(store, "chunk_policy_revision_through", 7)
    store[1].connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)", (
        f"chunk_gapfill_scan:{store[2].identity_hash}",
        json.dumps({"revision": 2, "policy": "conversational", "high": 9, "pending": [8]})))
    assert command._chunk_policy_revision(store[1].connection, store[2].identity_hash) == 1
    if arrival:
        add_rows(store, [(10, "assistant", LONG)], set())
    run(store, limit=1)
    for _ in range(4):
        run(store)
    assert_repaired_or_visible(store, {2, 8})
    assert 2 in counts(store)


@pytest.mark.parametrize("shape", ["arrival", "selection_found_gap", "current_control"])
def test_revision_boundary_extension(store, monkeypatch, shape):
    embedded = {1, 4, 5, 6, 7} if shape == "selection_found_gap" else {1, 3, 4, 5, 6, 7}
    add_rows(store, [(i, "user", LONG) for i in range(1, 8)], embedded)
    metadata(store, "chunk_policy_revision", 2)
    bounds = (3, 7) if shape == "current_control" else (1, 6 if shape == "selection_found_gap" else 7)
    command._record_chunk_policy_revision(store[1].connection, store[2].identity_hash, bounds, "conversational")
    if shape != "current_control":
        monkeypatch.setattr(command, "CHUNK_POLICY_REVISION", 3)
    if shape != "selection_found_gap":
        add_rows(store, [(8, "assistant", LONG)], set())
    run(store, limit=1)
    for _ in range(3):
        run(store)
    assert_repaired_or_visible(store, {2, 3})
    assert 2 in counts(store)


def test_cloud_legacy_rewrite_revalidates_wider_scan(store, monkeypatch):
    add_rows(store, [(i, "user", LONG) for i in range(1, 8)], {5, 6, 7})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (5, 7))
    older_walk(store, {1, 2, 4})
    metadata(store, "chunk_policy_revision_through", 2)
    with monkeypatch.context() as cloud_patch:
        cloud_patch.setattr(command, "chunk_message", lambda *a, **k: pytest.fail("cloud chunked"))
        cloud_patch.setattr(command, "resolve_provider", lambda *a, **k: pytest.fail("cloud dispatched"))
        command._validate_cloud_chunk_metadata(store[0], store[0].database_path)
    assert_repaired_or_visible(store, {3})
    assert command._chunk_backlog_estimate(store[1].connection, store[2].identity_hash, "conversational") == "1 (gap-fill)"


@pytest.mark.parametrize("mode", ["skip", "withheld"])
def test_cloud_manual_extension_keeps_withheld_low_row_visible(store, monkeypatch, mode):
    config, vectors, _, texts = store
    ingest = importlib.import_module("hermes_lcm.ingest_protection")
    config.embedding_provider = "voyage"
    revision = ingest.embedding_privacy_revision(config)
    vectors.register_profile("model-a", "voyage", 2, task="chunk", revision=revision)
    identity = command.EmbeddingIdentity.canonical("voyage", "model-a", revision, 2, "float32", "little", "chunk")
    cloud = config, vectors, identity, texts
    add_rows(cloud, [(i, "user", LONG + f" row {i}.") for i in range(1, 13)], set())
    provider = SimpleNamespace(provider_id="voyage", model_id="model-a", last_skipped_documents=[])
    target = False

    def embed(documents):
        provider.last_skipped_documents = [i for i, d in enumerate(documents)
                                           if target and mode == "skip" and "row 4." in d]
        return [[1.0, 0.0] for i, _ in enumerate(documents) if i not in provider.last_skipped_documents]

    protect = command.protect_embedding_text

    def protected(text, cfg, **kwargs):
        if target and mode == "withheld" and "row 4." in text:
            raise ingest.EmbeddingPrivacyPolicyError("synthetic residual privacy guard")
        return protect(text, cfg, **kwargs)

    provider.embed_documents = embed
    monkeypatch.setattr(command, "resolve_provider", lambda *a, **k: provider)
    monkeypatch.setattr(command, "protect_embedding_text", protected)
    monkeypatch.setenv("LCM_EMBEDDING_BACKFILL_BATCH_SIZE", "1")
    for limit in (5, 6):
        result = command._chunk_backfill_run(config, config.database_path, apply=True, limit=limit,
                    retry_uncertain=False, policy="conversational", confirm_raw_text=True)
        assert result["error"] is None and not result["failed"], result
        command._validate_cloud_chunk_metadata(config, config.database_path)
        target = True
    assert 4 not in counts(cloud)
    proof = marker(cloud)
    assert not proof["from"] <= 4 <= proof["through"]
    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "conversational") == "2 (gap-fill)"


def test_low_walk_without_captured_parent_is_unproven(store):
    add_rows(store, [(i, "user", LONG) for i in range(1, 11)], {1, 2, 3, 4, 6, 7, 8, 9, 10})
    metadata(store, "chunk_policy_revision", 2)
    store[1].connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)", (
        f"chunk_policy_revision_through:{store[2].identity_hash}",
        json.dumps({"from": 6, "through": 10, "policy": "conversational"})))
    conn, identity = store[1].connection, store[2].identity_hash
    parent_key = f"chunk_gapfill_scan:{identity}"
    fired = []

    class DropParentAfterEpochWrite:
        def execute(self, sql, parameters=()):
            cursor = conn.execute(sql, parameters)
            if (not fired and sql.startswith("INSERT INTO metadata(key,value) SELECT")
                    and parameters[0] == parent_key):
                fired.append(1)
                conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id=10")
            return cursor

        def __getattr__(self, name):
            return getattr(conn, name)

    state = {"coverage": (6, 10, "conversational")}
    rows = list(command._chunk_gapfill_rows(DropParentAfterEpochWrite(), identity, (1, 5),
                "conversational", 32, record_progress=True, gap_scan_state=state))
    assert fired == [1] and state["parent_value"] is None
    assert rows == [] and state["unproven"]
    assert metadata(store, "chunk_policy_revision_through") is None


def test_cloud_validation_exception_still_runs_summary_maintenance(store, monkeypatch, caplog):
    config, vectors, _, _ = store
    config.embedding_provider = "voyage"
    vectors.register_profile("model-a", "voyage", 2)
    maintenance = importlib.import_module("hermes_lcm.embedding_maintenance")
    providers = importlib.import_module("hermes_lcm.embedding_provider")
    monkeypatch.setattr(providers, "probe_provider_availability", lambda _: {"available": True})
    calls = []

    def summary(*args, **kwargs):
        calls.append("summary")
        return {"status": "complete", "embedded": 1, "selected": 1, "privacy_withheld": 0,
                "skipped": [], "failed": [], "stop_reason": None, "error": None}

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic validation fault")

    monkeypatch.setattr(command, "_validate_cloud_chunk_metadata", broken)
    monkeypatch.setattr(command, "_embedding_backfill_summary_run", summary)
    with caplog.at_level("DEBUG", logger=maintenance.__name__):
        outcome = maintenance.run_incremental_embedding_pass(config.database_path, config)
    assert calls == ["summary"]
    assert outcome != "error"
    assert "synthetic validation fault" in caplog.text
    assert metadata(store, "chunk_policy_revision_through") is None
