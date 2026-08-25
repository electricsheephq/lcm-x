from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.tools as tools_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.ingest_protection import (
    EmbeddingPrivacyPolicyError,
    embedding_privacy_revision,
    protect_embedding_text,
    redact_sensitive_text,
    validate_embedding_privacy_dispatch,
)
from hermes_lcm.vector_store import VectorStore


class CaptureProvider:
    provider_id = "voyage"
    model_id = "voyage-4-large"

    def __init__(self):
        self.queries: list[str] = []
        self.documents: list[list[str]] = []

    def embed_query(self, text):
        self.queries.append(str(text))
        return [1.0, 0.0]

    def embed_documents(self, texts):
        current = [str(text) for text in texts]
        self.documents.append(current)
        return [[1.0, 0.0] for _ in current]


def _config(tmp_path, *, enabled=True, patterns=None):
    return LCMConfig(
        database_path=str(tmp_path / "privacy.db"),
        embeddings_enabled=True,
        embedding_provider="voyage",
        embedding_model="voyage-4-large",
        sensitive_patterns_enabled=enabled,
        sensitive_patterns=(
            ["api_key", "bearer_token", "password_assignment", "private_key"]
            if patterns is None
            else patterns
        ),
    )


def _engine_with_summary(tmp_path, text: str):
    config = _config(tmp_path)
    engine = SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=tmp_path / "privacy.db"),
    )
    dag = SummaryDAG(engine._store.db_path)
    try:
        dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=0,
                summary=text,
                created_at=1.0,
                latest_at=1.0,
            )
        )
    finally:
        dag.close()
    revision = embedding_privacy_revision(config)
    store = VectorStore(engine._store.db_path, config=config)
    try:
        store.register_profile(
            "voyage-4-large",
            "voyage",
            2,
            revision=revision,
        )
    finally:
        store.close()
    return engine


def test_cloud_privacy_policy_requires_enabled_nonempty_known_patterns(tmp_path):
    with pytest.raises(EmbeddingPrivacyPolicyError, match="enabled"):
        embedding_privacy_revision(_config(tmp_path, enabled=False))
    with pytest.raises(EmbeddingPrivacyPolicyError, match="nonempty"):
        embedding_privacy_revision(_config(tmp_path, patterns=[]))
    with pytest.raises(EmbeddingPrivacyPolicyError, match="unknown"):
        embedding_privacy_revision(_config(tmp_path, patterns=["api_key", "future_pattern"]))


def test_provider_transform_removes_secret_metadata_and_canonicalizes_placeholders(tmp_path):
    config = _config(tmp_path)
    raw = (
        "api_key=abcdefghijklmnop "
        "password=correct-horse-battery "
        "Bearer abcdefghijklmnop "
        "api_key=[LCM sensitive redaction: name=api_key; chars=16; bytes=16; sha256=deadbeefdeadbeef]"
    )

    protected, revision, changed = protect_embedding_text(raw, config)

    assert changed is True
    assert revision.startswith("privacy:v1:")
    assert "abcdefghijklmnop" not in protected
    assert "correct-horse-battery" not in protected
    assert "deadbeefdeadbeef" not in protected
    assert "chars=" not in protected
    assert "bytes=" not in protected
    assert "sha256=" not in protected
    assert protected.count("[LCM embedding privacy: name=api_key]") == 2
    validate_embedding_privacy_dispatch([protected], config, expected_revision=revision)


def test_truncated_private_key_is_redacted_before_semantic_cloud_call(tmp_path):
    config = _config(tmp_path)
    provider = CaptureProvider()
    engine = SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=tmp_path / "privacy.db"),
    )
    revision = embedding_privacy_revision(config)
    store = VectorStore(engine._store.db_path, config=config)
    try:
        store.register_profile(
            provider.model_id,
            provider.provider_id,
            2,
            revision=revision,
        )
    finally:
        store.close()

    raw_query = "context before\n-----BEGIN PRIVATE KEY-----\ntruncated material"
    result = tools_mod._lcm_grep_embed_query(
        provider,
        raw_query,
        engine=engine,
        task="summary",
        remaining_s=1.0,
    )

    assert result == [1.0, 0.0]
    assert provider.queries == [
        "context before\n[LCM embedding privacy: name=private_key]"
    ]
    assert "truncated material" not in provider.queries[0]


def test_semantic_query_makes_zero_cloud_calls_when_policy_disabled(tmp_path):
    config = _config(tmp_path, enabled=False)
    provider = CaptureProvider()
    engine = SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=tmp_path / "privacy.db"),
    )

    with pytest.raises(EmbeddingPrivacyPolicyError, match="enabled"):
        tools_mod._lcm_grep_embed_query(
            provider,
            "api_key=abcdefghijklmnop",
            engine=engine,
            task="summary",
            remaining_s=1.0,
        )

    assert provider.queries == []


def test_semantic_query_requires_registered_policy_revision_and_redacts(tmp_path):
    config = _config(tmp_path)
    provider = CaptureProvider()
    engine = SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=tmp_path / "privacy.db"),
    )
    revision = embedding_privacy_revision(config)
    store = VectorStore(engine._store.db_path, config=config)
    try:
        store.register_profile(
            provider.model_id,
            provider.provider_id,
            2,
            revision=revision,
        )
    finally:
        store.close()

    result = tools_mod._lcm_grep_embed_query(
        provider,
        "api_key=abcdefghijklmnop",
        engine=engine,
        task="summary",
        remaining_s=1.0,
    )

    assert result == [1.0, 0.0]
    assert len(provider.queries) == 1
    assert "abcdefghijklmnop" not in provider.queries[0]
    assert provider.queries[0].endswith("[LCM embedding privacy: name=api_key]")


def test_warmup_refuses_cloud_before_probe_when_policy_disabled(monkeypatch, tmp_path):
    config = _config(tmp_path, enabled=False)
    provider = CaptureProvider()
    engine = SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=tmp_path / "privacy.db"),
    )
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config: provider)

    result = command_mod._embedding_warmup_text(engine)

    assert "status: error" in result
    assert "sensitive" in result.lower()
    assert provider.queries == []


def test_summary_backfill_transforms_provider_input_and_reports_aggregate_privacy(
    monkeypatch, tmp_path
):
    secret = "abcdefghijklmnop"
    engine = _engine_with_summary(tmp_path, f"api_key={secret}")
    provider = CaptureProvider()
    monkeypatch.setattr(
        command_mod,
        "resolve_provider",
        lambda _config, **_kwargs: provider,
    )

    result = command_mod._embedding_backfill_summary_text(
        engine,
        apply=True,
        limit=10,
        retry_uncertain=False,
    )

    assert "status: complete" in result
    assert "privacy_transformed: 1" in result
    assert "privacy_blocked: 0" in result
    assert len(provider.documents) == 1
    assert secret not in provider.documents[0][0]
    assert provider.documents[0][0].endswith(
        "[LCM embedding privacy: name=api_key]"
    )


def test_summary_backfill_revalidates_policy_before_each_cloud_subdispatch(
    monkeypatch, tmp_path
):
    engine = _engine_with_summary(tmp_path, "benign summary")

    class DriftProvider(CaptureProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            engine._config.sensitive_patterns = ["password_assignment"]
            before_dispatch((0,))
            self.documents.append(list(texts))
            raise AssertionError("cloud dispatch must not occur after policy drift")
            yield  # pragma: no cover

    provider = DriftProvider()
    monkeypatch.setattr(
        command_mod,
        "resolve_provider",
        lambda _config, **_kwargs: provider,
    )

    result = command_mod._embedding_backfill_summary_text(
        engine,
        apply=True,
        limit=10,
        retry_uncertain=False,
    )

    assert "status: error" in result
    assert "policy changed before provider dispatch" in result
    assert provider.documents == []


# --- #365: pattern-ordering bypass (password/passphrase prefix eats the PEM begin marker) ---

_KEY_BODY = "MIIEvQIBADANBgkqSUPERSECRETKEYMATERIAL"


@pytest.mark.parametrize(
    "prefixed",
    [
        "password: -----BEGIN PRIVATE KEY-----\n%s\n-----END PRIVATE KEY-----",
        "passphrase=-----BEGIN OPENSSH PRIVATE KEY-----\n%s\n-----END OPENSSH PRIVATE KEY-----",
        "pwd=-----BEGIN EC PRIVATE KEY-----\n%s\n-----END EC PRIVATE KEY-----",
    ],
)
def test_embedding_privacy_redacts_pem_after_password_assignment(tmp_path, prefixed):
    """#365: an assignment pattern must not consume the PEM begin marker and leak the body."""
    cfg = _config(tmp_path)
    text = prefixed % _KEY_BODY
    protected, _revision, changed = protect_embedding_text(text, cfg)
    assert _KEY_BODY not in protected
    assert changed
    # dispatch validation must also pass on the protected text (no residual)
    revision = embedding_privacy_revision(cfg)
    validate_embedding_privacy_dispatch([protected], cfg, expected_revision=revision)


def test_embedding_privacy_residual_flags_orphaned_end_marker(tmp_path):
    """Transform-independent residual check: a surviving PEM END marker fails closed."""
    cfg = _config(tmp_path)
    # Simulate a begin-marker already consumed by an upstream transform.
    orphaned = "password: [prior] PRIVATE KEY-----\n%s\n-----END PRIVATE KEY-----" % _KEY_BODY
    revision = embedding_privacy_revision(cfg)
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch([orphaned], cfg, expected_revision=revision)


def test_embedding_privacy_control_and_plain_secrets_still_redact(tmp_path):
    cfg = _config(tmp_path)
    own_line = "note\n-----BEGIN PRIVATE KEY-----\n%s\n-----END PRIVATE KEY-----" % _KEY_BODY
    protected, _r, _c = protect_embedding_text(own_line, cfg)
    assert _KEY_BODY not in protected
    plain, _r2, _c2 = protect_embedding_text("password: hunter2secretvalue", cfg)
    assert "hunter2secretvalue" not in plain


def test_durable_redaction_pem_after_password_survives_ordering(tmp_path):
    """#365 durable path: private_key runs before password_assignment; key body never stored."""
    cfg = _config(tmp_path)
    text = "password: -----BEGIN PRIVATE KEY-----\n%s\n-----END PRIVATE KEY-----" % _KEY_BODY
    out = redact_sensitive_text(text, cfg)
    assert _KEY_BODY not in out
    assert "private_key" in out


def test_durable_redacts_truncated_pem_after_password(tmp_path):
    """#365 review finding 1: truncated key (BEGIN, no END) must not leak to durable storage."""
    cfg = _config(tmp_path)
    text = "password: -----BEGIN PRIVATE KEY-----\n%s\n<no end marker>" % _KEY_BODY
    out = redact_sensitive_text(text, cfg)
    assert _KEY_BODY not in out
    # embedding path parity
    protected, _r, _c = protect_embedding_text(text, cfg)
    assert _KEY_BODY not in protected


def test_residual_allows_prose_with_bare_end_marker(tmp_path):
    """#365 review finding 2: prose containing only an END phrase must NOT fail closed."""
    cfg = _config(tmp_path)
    prose = "The key file terminates with the line -----END PRIVATE KEY----- and nothing else."
    revision = embedding_privacy_revision(cfg)
    # must not raise
    validate_embedding_privacy_dispatch([prose], cfg, expected_revision=revision)
    protected, _r, _c = protect_embedding_text(prose, cfg)
    assert "-----END PRIVATE KEY-----" in protected  # left intact, no false redaction
