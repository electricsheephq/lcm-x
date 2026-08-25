from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.ingest_protection as ingest_protection_mod
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
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore


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
    assert revision.startswith("privacy:v3:")
    assert not revision.startswith("privacy:v2:")
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

    key_body = "MIIEvQIBADANBgkqSUPERSECRETKEYMATERIAL"
    raw_query = "context before\n-----BEGIN PRIVATE KEY-----\n" + key_body
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
    assert key_body not in provider.queries[0]


def test_privacy_v3_revision_makes_prior_vectors_pending(tmp_path):
    config = _config(tmp_path)
    dag = SummaryDAG(config.database_path)
    try:
        node_id = dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=0,
                summary="benign summary",
                created_at=1.0,
                latest_at=1.0,
            )
        )
    finally:
        dag.close()

    current_revision = embedding_privacy_revision(config)
    old_revision = current_revision.replace("privacy:v3:", "privacy:v2:", 1)
    store = VectorStore(config.database_path, config=config)
    try:
        old_identity_hash = store.register_profile(
            "voyage-4-large", "voyage", 2, revision=old_revision
        )
        old_identity = EmbeddingIdentity.canonical(
            "voyage",
            "voyage-4-large",
            old_revision,
            2,
            "float32",
            "little",
            "summary",
        )
        store.record_embedding(
            str(node_id),
            "summary",
            "voyage-4-large",
            [1.0, 0.0],
            identity=old_identity,
        )
        current_identity_hash = store.register_profile(
            "voyage-4-large", "voyage", 2, revision=current_revision
        )
    finally:
        store.close()

    assert current_identity_hash != old_identity_hash
    conn = sqlite3.connect(config.database_path)
    conn.row_factory = sqlite3.Row
    try:
        pending, rows = command_mod._embedding_pending_rows(
            conn, current_identity_hash, 10
        )
    finally:
        conn.close()
    assert pending == 1
    assert [int(row["node_id"]) for row in rows] == [node_id]


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


def test_truncated_private_key_redaction_preserves_trailing_prose(tmp_path):
    cfg = _config(tmp_path)
    prose = "This trailing prose must remain visible."
    padded_body = _KEY_BODY + "=="
    text = "prefix\n-----BEGIN PRIVATE KEY-----\n%s\n%s" % (padded_body, prose)

    durable = redact_sensitive_text(text, cfg)
    protected, _revision, changed = protect_embedding_text(text, cfg)

    assert padded_body not in durable
    assert padded_body not in protected
    assert prose in durable
    assert prose in protected
    assert changed is True


def test_bare_private_key_begin_followed_by_prose_is_left_intact(tmp_path):
    cfg = _config(tmp_path)
    text = "-----BEGIN PRIVATE KEY-----\nThis is prose, not a base64 key body."

    durable = redact_sensitive_text(text, cfg)
    protected, _revision, changed = protect_embedding_text(text, cfg)

    assert durable == text
    assert protected == text
    assert changed is False


def test_durable_truncated_private_key_uses_linear_scanner_with_regex_installed(
    monkeypatch, tmp_path
):
    cfg = _config(tmp_path)
    text = "-----BEGIN PRIVATE KEY-----\n%s" % _KEY_BODY
    monkeypatch.setattr(ingest_protection_mod, "_regex_engine", object())

    def unexpected_regex_lookup(_name):
        raise AssertionError("private_key must bypass the optional regex engine")

    monkeypatch.setattr(
        ingest_protection_mod, "_regex_pattern_for", unexpected_regex_lookup
    )

    durable = redact_sensitive_text(text, cfg)

    assert _KEY_BODY not in durable
    assert "private_key" in durable


def test_durable_linear_scanner_redacts_large_truncated_body_without_regex(
    monkeypatch, tmp_path
):
    cfg = _config(tmp_path)
    monkeypatch.setattr(ingest_protection_mod, "_regex_engine", None)
    ingest_protection_mod._SENSITIVE_REGEX_CATALOG.clear()
    body = "A" * (ingest_protection_mod._SENSITIVE_STDLIB_MAX_CHARS + 10)
    text = "-----BEGIN PRIVATE KEY-----\n" + body

    durable = redact_sensitive_text(text, cfg)

    assert body not in durable
    assert "private_key" in durable


def test_residual_allows_prose_with_bare_end_marker(tmp_path):
    """#365 review finding 2: prose containing only an END phrase must NOT fail closed."""
    cfg = _config(tmp_path)
    prose = "This is an illustrative terminator -----END PRIVATE KEY----- in prose."
    revision = embedding_privacy_revision(cfg)
    # must not raise
    validate_embedding_privacy_dispatch([prose], cfg, expected_revision=revision)
    protected, _r, _c = protect_embedding_text(prose, cfg)
    assert "-----END PRIVATE KEY-----" in protected  # left intact, no false redaction


def test_residual_private_key_detector_is_linear_on_long_single_line(tmp_path):
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)

    started = time.perf_counter()
    validate_embedding_privacy_dispatch(
        ["A" * 8000], cfg, expected_revision=revision
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1


def test_truncated_key_short_terminal_line_is_redacted_both_paths(tmp_path):
    # Round-3 finding 1: a valid short final base64 line (real PEM tails are
    # often shorter than a wrapped line) must be inside the redaction bound.
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    text = "-----BEGIN PRIVATE KEY-----\n" + "A" * 64 + "\nCDEF\ntrailing prose stays"
    for redacted in (
        redact_sensitive_text(text, cfg),
        protect_embedding_text(text, cfg)[0],
    ):
        assert "AAAA" not in redacted
        assert "CDEF" not in redacted
        assert "trailing prose stays" in redacted
    protected, _r, _c = protect_embedding_text(text, cfg)
    validate_embedding_privacy_dispatch([protected], cfg, expected_revision=revision)


def test_stripped_key_with_short_terminal_line_fails_residual_validation(tmp_path):
    # Round-3 finding 1 (residual half): body + short tail + END, BEGIN absent.
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    stripped = "A" * 64 + "\nCDEF\n-----END PRIVATE KEY-----"
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch([stripped], cfg, expected_revision=revision)


def test_truncated_key_never_consumes_prose_before_a_complete_key(tmp_path):
    # Round-3 finding 2: END pairing must be body-adjacent; a truncated key
    # followed by prose and a complete key redacts both keys, keeps the prose.
    cfg = _config(tmp_path)
    body = "B" * 64
    text = (
        "-----BEGIN PRIVATE KEY-----\n" + "A" * 64 + "\n"
        "meeting notes about the incident\n"
        "-----BEGIN PRIVATE KEY-----\n" + body + "\n-----END PRIVATE KEY-----\n"
        "closing prose"
    )
    for redacted in (
        redact_sensitive_text(text, cfg),
        protect_embedding_text(text, cfg)[0],
    ):
        assert "AAAA" not in redacted
        assert body not in redacted
        assert "meeting notes about the incident" in redacted
        assert "closing prose" in redacted
        assert "-----END PRIVATE KEY-----" not in redacted


def test_many_unmatched_begin_markers_scan_linearly_both_paths(tmp_path):
    # Round-3 finding 3: per-BEGIN END searches to EOF were quadratic (CI
    # measured 13.9s on the pathological ingest input); the single-pass
    # scanner must stay far under the CI 3s bound on both paths.
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    pathological = ("-----BEGIN PRIVATE KEY-----\n" + "A" * 64 + "\n") * 20000
    started = time.perf_counter()
    redacted = redact_sensitive_text(pathological, cfg)
    protected, _r, _c = protect_embedding_text(pathological, cfg)
    validate_embedding_privacy_dispatch([protected], cfg, expected_revision=revision)
    elapsed = time.perf_counter() - started
    assert "AAAA" not in redacted
    assert "AAAA" not in protected
    assert elapsed < 3.0


def test_inline_single_line_private_key_still_redacted(tmp_path):
    # private_key now always routes through the linear scanner, so the scanner
    # must also cover the whitespace-separated single-line PEM form the old
    # DOTALL regex handled.
    cfg = _config(tmp_path)
    text = "before -----BEGIN PRIVATE KEY----- MIIEvQIBADANBgkqhkiG9w0BAQEFAASC -----END PRIVATE KEY----- after"
    for redacted in (
        redact_sensitive_text(text, cfg),
        protect_embedding_text(text, cfg)[0],
    ):
        assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC" not in redacted
        assert "before" in redacted and "after" in redacted


def test_complete_key_with_only_short_body_lines_is_redacted(tmp_path):
    # Complete-block adjacency must be lenient about body-line width
    # (mirrors the redact-path coverage in test_ingest_protection).
    cfg = _config(tmp_path)
    text = "-----BEGIN RSA PRIVATE KEY-----\nabcdef\n-----END RSA PRIVATE KEY-----"
    for redacted in (
        redact_sensitive_text(text, cfg),
        protect_embedding_text(text, cfg)[0],
    ):
        assert "abcdef" not in redacted
        assert "BEGIN RSA PRIVATE KEY" not in redacted


def test_line_shape_variants_cannot_bypass_key_redaction(tmp_path):
    # Round-5 (review-probe shapes): CR-only endings, indented/prefixed
    # markers, multiple short tails, and short-line re-wraps are all key
    # material; redaction must not be defeatable by line-shape alone.
    cfg = _config(tmp_path)
    body = "Q" * 64
    variants = {
        "cr_only": "-----BEGIN PRIVATE KEY-----\r" + body + "\r-----END PRIVATE KEY-----\r",
        "indented": "  -----BEGIN PRIVATE KEY-----\n  " + body + "\n  -----END PRIVATE KEY-----",
        "prefixed": "key: -----BEGIN PRIVATE KEY-----\n" + body + "\n-----END PRIVATE KEY-----",
        "two_short_tails": "-----BEGIN PRIVATE KEY-----\n" + body + "\nBBBBBBBB\nCCCCCCCC\nprose stays",
        "all_short_rewrap": "-----BEGIN PRIVATE KEY-----\n" + "\n".join(["AbCd1234"] * 12),
    }
    for name, text in variants.items():
        for redacted in (
            redact_sensitive_text(text, cfg),
            protect_embedding_text(text, cfg)[0],
        ):
            assert body not in redacted, name
            assert "BBBBBBBB" not in redacted, name
            assert "AbCd1234" not in redacted, name
    assert "prose stays" in redact_sensitive_text(variants["two_short_tails"], cfg)
    assert redact_sensitive_text(variants["prefixed"], cfg).startswith("key: ")


def test_decoy_begin_does_not_shield_orphan_body(tmp_path):
    # Round-5: a bare BEGIN plus prose earlier in the text must not suppress
    # redaction or validation of a stripped body+END later (the old detector's
    # seen-BEGIN suppression was global and provably wrong).
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    body = "Q" * 64
    decoy = (
        "-----BEGIN PRIVATE KEY-----\nThis is prose, not a key body.\n"
        + body + "\n-----END PRIVATE KEY-----"
    )
    for redacted in (
        redact_sensitive_text(decoy, cfg),
        protect_embedding_text(decoy, cfg)[0],
    ):
        assert body not in redacted
        assert "-----END PRIVATE KEY-----" not in redacted
        assert "This is prose, not a key body." in redacted
    protected, _r, _c = protect_embedding_text(decoy, cfg)
    validate_embedding_privacy_dispatch([protected], cfg, expected_revision=revision)
    # The raw decoy text (unredacted) must FAIL validation despite the BEGIN.
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch([decoy], cfg, expected_revision=revision)


def test_round6_completeness_and_precision_dispositions(tmp_path):
    # Round-6 review shapes: decorated END closes structure (F1); a decoy
    # inline BEGIN cannot blind a later valid inline key (F2); orphan runs
    # include leading short lines (F4); assignment prose after a bare BEGIN
    # is NOT consumed (F5); inline prose naming both markers is NOT consumed
    # (F6); a space-riddled body is out of redactor scope but the validator
    # pair-guard blocks its dispatch (F3, fail-closed on the cloud path).
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    body = "Q" * 64

    # F1: decorated orphan END — body redacted on both paths, raw text flagged.
    f1 = body + "\nlabel: -----END PRIVATE KEY-----\ntrailing prose"
    for redacted in (redact_sensitive_text(f1, cfg), protect_embedding_text(f1, cfg)[0]):
        assert body not in redacted
        assert "trailing prose" in redacted
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch([f1], cfg, expected_revision=revision)

    # F2: decoy inline BEGIN, then a valid inline key on the same line.
    f2 = (
        "note -----BEGIN PRIVATE KEY----- prose then -----BEGIN PRIVATE KEY----- "
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC -----END PRIVATE KEY----- tail"
    )
    for redacted in (redact_sensitive_text(f2, cfg), protect_embedding_text(f2, cfg)[0]):
        assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC" not in redacted
        assert redacted.startswith("note ")
        assert redacted.endswith(" tail")

    # F4: orphan run with leading short lines — the whole run goes.
    f4 = "TAIL\n" + body + "\n-----END PRIVATE KEY-----\nprose stays"
    for redacted in (redact_sensitive_text(f4, cfg), protect_embedding_text(f4, cfg)[0]):
        assert "TAIL" not in redacted
        assert body not in redacted
        assert "prose stays" in redacted

    # F5: assignment prose after a bare BEGIN is preserved and not flagged.
    f5 = "-----BEGIN PRIVATE KEY-----\nenvironment=prod\nregion=eu"
    for redacted in (redact_sensitive_text(f5, cfg), protect_embedding_text(f5, cfg)[0]):
        assert "environment=prod" in redacted
        assert "region=eu" in redacted
    protected, _r, _c = protect_embedding_text(f5, cfg)
    validate_embedding_privacy_dispatch([protected], cfg, expected_revision=revision)

    # F6: prose naming both markers inline is preserved and dispatchable.
    f6 = "The docs mention -----BEGIN PRIVATE KEY----- and -----END PRIVATE KEY----- inline."
    for redacted in (redact_sensitive_text(f6, cfg), protect_embedding_text(f6, cfg)[0]):
        assert redacted == f6
    validate_embedding_privacy_dispatch([f6], cfg, expected_revision=revision)

    # F3: space-riddled body — redactor scope limitation (declared), but the
    # surviving exact BEGIN/END pair blocks the embedding dispatch.
    f3 = "-----BEGIN PRIVATE KEY-----\nSYNTH ETIC BODY 1234 5678 90AB\n-----END PRIVATE KEY-----"
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch([f3], cfg, expected_revision=revision)
