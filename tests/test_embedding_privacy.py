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


def test_round7_realistic_shapes_encrypted_json_logs(tmp_path):
    # Round-7 (thread sweep + R6, all in declared scope): encrypted PEM armor
    # headers, JSON-escaped serialization, log-collector prefixes, all-short
    # orphan runs — and the precision guards those fixes must not break.
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    body = "Q" * 64

    # RFC 1421 encrypted key: headers + blank line, complete AND truncated.
    enc_complete = (
        "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\n"
        "DEK-Info: AES-128-CBC,ABCDEF0123456789\n\n" + body + "\n"
        "-----END RSA PRIVATE KEY-----"
    )
    enc_truncated = (
        "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\n"
        "DEK-Info: AES-128-CBC,ABCDEF0123456789\n\n" + body + "\nprose stays"
    )
    for text in (enc_complete, enc_truncated):
        for redacted in (
            redact_sensitive_text(text, cfg),
            protect_embedding_text(text, cfg)[0],
        ):
            assert body not in redacted, text[:40]
    assert "prose stays" in redact_sensitive_text(enc_truncated, cfg)

    # JSON-escaped one-physical-line key (serialized logs).
    esc = (
        '{"log": "-----BEGIN PRIVATE KEY-----\\\\n' + "M" * 40
        + '\\\\n-----END PRIVATE KEY-----"}'
    )
    for redacted in (
        redact_sensitive_text(esc, cfg),
        protect_embedding_text(esc, cfg)[0],
    ):
        assert "M" * 40 not in redacted

    # Log-collector prefixes on every line.
    logged = (
        "INFO -----BEGIN PRIVATE KEY-----\nINFO " + body + "\n"
        "INFO -----END PRIVATE KEY-----"
    )
    for redacted in (
        redact_sensitive_text(logged, cfg),
        protect_embedding_text(logged, cfg)[0],
    ):
        assert body not in redacted

    # All-short orphan run (stripped, re-wrapped) directly before END.
    orphan_short = "AbCd0011\nEfGh2233\nIjKl4455\n-----END PRIVATE KEY-----"
    for redacted in (
        redact_sensitive_text(orphan_short, cfg),
        protect_embedding_text(orphan_short, cfg)[0],
    ):
        assert "AbCd0011" not in redacted

    # PRECISION: a lone token before prose that merely mentions a decorated
    # END marker is ordinary content — kept, and dispatchable.
    fp = (
        "abcdef0123456789AB\n"
        "see label: -----END PRIVATE KEY----- for format details"
    )
    for redacted in (
        redact_sensitive_text(fp, cfg),
        protect_embedding_text(fp, cfg)[0],
    ):
        assert redacted == fp
    validate_embedding_privacy_dispatch([fp], cfg, expected_revision=revision)

    # Pair guard covers decorated END: space-riddled body between an exact
    # BEGIN line and a labeled END line still blocks the dispatch.
    pair = (
        "-----BEGIN PRIVATE KEY-----\nSYNTH ETIC BODY 1234 5678 90AB\n"
        "label: -----END PRIVATE KEY-----"
    )
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch([pair], cfg, expected_revision=revision)


def test_round8_serialized_and_prefixed_compositions(tmp_path):
    # Round-8: virtual-line splitting makes serialized keys first-class model
    # input (json.dumps at any nesting depth), and BEGIN-anchored scans
    # reclassify log-prefixed lines — so compositions of the accidental shapes
    # (serialization x truncation x armor x prefixes) are covered by
    # construction, with English-word tails protected.
    import json as _json

    cfg = _config(tmp_path)
    body = "Q" * 64

    serialized_truncated = _json.dumps(
        {"log": "-----BEGIN PRIVATE KEY-----\n" + body + "\nCDEF"}
    )
    serialized_encrypted = _json.dumps(
        {
            "log": "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\n"
            "DEK-Info: AES-128-CBC,AABB\n\n" + body + "\n-----END RSA PRIVATE KEY-----"
        }
    )
    doubly_serialized = _json.dumps({"outer": serialized_truncated})
    prefixed_armor_truncated = (
        "INFO -----BEGIN RSA PRIVATE KEY-----\nINFO Proc-Type: 4,ENCRYPTED\n"
        "INFO DEK-Info: AES-128-CBC,AABB\nINFO\nINFO " + body + "\nplain prose stays"
    )
    prefixed_short_tail = "-----BEGIN PRIVATE KEY-----\n" + body + "\nINFO CDEF\nprose stays"
    for text in (
        serialized_truncated,
        serialized_encrypted,
        doubly_serialized,
        prefixed_armor_truncated,
        prefixed_short_tail,
    ):
        for redacted in (
            redact_sensitive_text(text, cfg),
            protect_embedding_text(text, cfg)[0],
        ):
            assert body not in redacted, text[:50]
    assert "plain prose stays" in redact_sensitive_text(prefixed_armor_truncated, cfg)
    assert "prose stays" in redact_sensitive_text(prefixed_short_tail, cfg)

    # Precision: English-word lines after a bare BEGIN are prose, not body.
    prose = "-----BEGIN PRIVATE KEY-----\nservice productionservice\nMeeting Notes About Keys"
    for redacted in (
        redact_sensitive_text(prose, cfg),
        protect_embedding_text(prose, cfg)[0],
    ):
        assert redacted == prose

    # Perf bound: same-line marker storms do bounded work per line.
    import time as _time

    storm = "-----BEGIN PRIVATE KEY----- x " * 800
    started = _time.perf_counter()
    redact_sensitive_text(storm, cfg)
    assert _time.perf_counter() - started < 1.0


def test_round9_escape_artifact_normalization(tmp_path):
    # Round-9 (R8 review): serialization/prefix artifacts are normalized, not
    # special-cased — trailing escape backslashes and escaped-tab indentation
    # trim before classification; colon-ended log prefixes reclassify inside
    # BEGIN scans; the armor/body separator accepts a prefixed blank remnant.
    import json as _json

    cfg = _config(tmp_path)
    body = "Q" * 64
    cases = [
        _json.dumps({"outer": _json.dumps({"log": "-----BEGIN PRIVATE KEY-----\n" + body + "\nCDEF"})}),
        (
            "host app -----BEGIN RSA PRIVATE KEY-----\nhost app Proc-Type: 4,ENCRYPTED\n"
            "host app DEK-Info: AES-128-CBC,AABB\nhost app\nhost app " + body + "\nprose stays"
        ),
        "INFO: -----BEGIN PRIVATE KEY-----\nINFO: " + body + "\nINFO: CDEF",
        _json.dumps({"log": "\t-----BEGIN PRIVATE KEY-----\n\t" + body + "\n\t-----END PRIVATE KEY-----"}),
        _json.dumps({"o": _json.dumps({"log": "app -----BEGIN RSA PRIVATE KEY-----\napp Proc-Type: 4,ENCRYPTED\napp\napp " + body})}),
    ]
    for text in cases:
        for redacted in (
            redact_sensitive_text(text, cfg),
            protect_embedding_text(text, cfg)[0],
        ):
            assert body not in redacted, text[:60]
            assert "CDEF" not in redacted, text[:60]
    assert "prose stays" in redact_sensitive_text(cases[1], cfg)
    nk = _json.dumps({"msg": "no keys here, just a normal log line with data"})
    assert redact_sensitive_text(nk, cfg) == nk

    # R9 review: escaped-tab indentation AFTER a log prefix, serialized —
    # the composition of all five factors (serialization, prefix, escaped
    # tab, armor, truncation) must redact.
    five = _json.dumps(
        {"log": "app \t-----BEGIN RSA PRIVATE KEY-----\napp \tProc-Type: 4,ENCRYPTED\napp \t\napp \t" + body}
    )
    for redacted in (
        redact_sensitive_text(five, cfg),
        protect_embedding_text(five, cfg)[0],
    ):
        assert body not in redacted


def test_round11_thread_sweep_shapes(tmp_path):
    # Round-11 (PR-thread sweep): CodeQL ReDoS on the trim regexes (now
    # linear hand-rolled scans + linear segment splitter), missing escaped
    # separator classes, body-on-BEGIN-line, END-before-BEGIN ordering,
    # one-line orphan bodies (v1-era migration rows), and bare English
    # tokens as body evidence.
    import time as _time

    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    body = "Q" * 64
    mii = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC"

    # ReDoS: long backslash runs are linear (was 34s at 100k pre-fix).
    started = _time.perf_counter()
    redact_sensitive_text(
        "-----BEGIN PRIVATE KEY-----\\n" + "\\" * 100000 + "x", cfg
    )
    assert _time.perf_counter() - started < 1.0

    # Escaped separator classes json.dumps emits beyond \n (\f is the
    # shorthand json.dumps uses for U+000C).
    for sep in ("\\r", "\\u2028", "\\u000b", "\\f"):
        text = '{"log": "-----BEGIN PRIVATE KEY-----' + sep + body + sep + 'tail"}'
        assert body not in redact_sensitive_text(text, cfg)

    # Newline-normalized: payload ON the BEGIN line.
    joined = "-----BEGIN PRIVATE KEY----- " + mii + "\n-----END PRIVATE KEY-----"
    assert mii not in redact_sensitive_text(joined, cfg)
    joined_trunc = "-----BEGIN PRIVATE KEY----- " + mii + "\nprose stays"
    out = redact_sensitive_text(joined_trunc, cfg)
    assert mii not in out and "prose stays" in out

    # Compacted "END ... BEGIN" line closes the preceding orphan run.
    compact = (
        body + "\n-----END PRIVATE KEY----- -----BEGIN PRIVATE KEY-----\n"
        + "R" * 64 + "\n-----END PRIVATE KEY-----"
    )
    out = redact_sensitive_text(compact, cfg)
    assert body not in out and "R" * 64 not in out

    # One-line orphan body (upgraded v1-era row): redacted AND dispatch-blocked.
    orphan_line = "[old placeholder] " + mii + " -----END PRIVATE KEY-----"
    assert mii not in redact_sensitive_text(orphan_line, cfg)
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch([orphan_line], cfg, expected_revision=revision)

    # Bare English tokens are never truncated-redaction evidence...
    prose = "-----BEGIN PRIVATE KEY-----\nThisIsNotAKeyLine\nfoo bar"
    assert redact_sensitive_text(prose, cfg) == prose
    # ...but a complete block with a short English body still redacts.
    complete = "-----BEGIN PRIVATE KEY-----\nexample\n-----END PRIVATE KEY-----"
    assert "example" not in redact_sensitive_text(complete, cfg)


def test_every_splitlines_separator_survives_serialization(tmp_path):
    # Closed-form: derive the separator set from str.splitlines itself, so
    # this cannot go stale — any char Python treats as a line break must also
    # split virtual lines in its json-serialized (escaped) form.
    import json as _json

    cfg = _config(tmp_path)
    body = "Q" * 64
    separators = [
        chr(c) for c in range(0x110000 if False else 0x3000)
        if len(("a" + chr(c) + "b").splitlines()) > 1
    ]
    assert len(separators) >= 10  # \n \r \v \f FS GS RS NEL LS PS
    for sep in separators:
        text = _json.dumps(
            {"log": "-----BEGIN PRIVATE KEY-----" + sep + body + sep + "tail"}
        )
        redacted = redact_sensitive_text(text, cfg)
        assert body not in redacted, f"leak with separator U+{ord(sep):04X}"


def test_round14_serialization_and_prefix_variants(tmp_path):
    # Round-14 thread sweep: single-quoted/repr logs, JSON solidus escapes,
    # prefixed orphan bodies (END-anchored), whitespace-chunked truncated
    # bodies, no-space colon prefixes, attached Markdown markers — with the
    # precision guards each demands.
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    body = "Q" * 64

    single_quoted = "{'log': '-----BEGIN PRIVATE KEY-----\\n" + "M" * 40 + "'}"
    solidus = (
        '{"log": "-----BEGIN PRIVATE KEY-----\\n'
        + "QQQQabcd" * 6
        + '\\/QQQQ\\nprose"}'
    )
    prefixed_orphan = "[ph]\nINFO " + body + "\nINFO -----END PRIVATE KEY-----\nprose stays"
    chunked = "-----BEGIN PRIVATE KEY-----\nSYNTHET1C B0DY 1234 5678 90AB CDEF\nprose stays"
    colon = "INFO:-----BEGIN PRIVATE KEY-----\nINFO:" + body + "\nINFO:CDEF"
    blockquote = ">-----BEGIN PRIVATE KEY-----\n>" + body + "\n>prose text here"
    for text, leak in (
        (single_quoted, "M" * 40),
        (solidus, "QQQQabcd"),
        (prefixed_orphan, body),
        (chunked, "SYNTHET1C"),
        (colon, body),
        (blockquote, body),
    ):
        for redacted in (
            redact_sensitive_text(text, cfg),
            protect_embedding_text(text, cfg)[0],
        ):
            assert leak not in redacted, text[:50]
    assert "prose stays" in redact_sensitive_text(prefixed_orphan, cfg)
    assert "prose text here" in redact_sensitive_text(blockquote, cfg)
    with pytest.raises(EmbeddingPrivacyPolicyError):
        validate_embedding_privacy_dispatch(
            ["INFO " + body + "\nINFO -----END PRIVATE KEY-----"],
            cfg,
            expected_revision=revision,
        )
    # Attached markers composed with armor (R14 review's cross-product case).
    attached_armor = (
        ">-----BEGIN RSA PRIVATE KEY-----\n>Proc-Type: 4,ENCRYPTED\n"
        ">DEK-Info: AES-128-CBC,AABB\n>\n>" + body + "\nprose stays"
    )
    out = redact_sensitive_text(attached_armor, cfg)
    assert body not in out and "prose stays" in out

    # Precision: prefixed hash dumps with no marker, prose after BEGIN.
    dump = "INFO " + "a1b2c3d4" * 8 + "\nINFO " + "e5f6a7b8" * 8 + "\nno markers anywhere"
    assert redact_sensitive_text(dump, cfg) == dump
    prose = "-----BEGIN PRIVATE KEY-----\nmeeting notes about the incident\nmore prose"
    assert redact_sensitive_text(prose, cfg) == prose


def test_round16_orphan_symmetry_and_escape_spellings(tmp_path):
    # Round-16: the END-anchored backward pass mirrors the BEGIN-anchored
    # forward scan (prefixed/colon/timestamped orphan bodies), END tightness
    # is suffix-based (a marker ending its line closes orphans regardless of
    # prefix length), and \\u002f joins the solidus spellings.
    cfg = _config(tmp_path)
    revision = embedding_privacy_revision(cfg)
    body = "Q" * 64

    u002f = (
        '{"log": "-----BEGIN PRIVATE KEY-----\\n'
        + "QQQQabcd" * 4
        + "\\u002f"
        + "QQQQabcd" * 2
        + '\\nprose"}'
    )
    colon_orphan = "INFO:" + body + "\nINFO:-----END PRIVATE KEY-----\nprose stays"
    stamped_orphan = (
        "2026-08-26T12:00:00Z INFO " + body + "\n"
        "2026-08-26T12:00:00Z INFO -----END PRIVATE KEY-----\ntail prose"
    )
    for text, leak in (
        (u002f, "QQQQabcd"),
        (colon_orphan, body),
        (stamped_orphan, body),
    ):
        for redacted in (
            redact_sensitive_text(text, cfg),
            protect_embedding_text(text, cfg)[0],
        ):
            assert leak not in redacted, text[:50]
    assert "prose stays" in redact_sensitive_text(colon_orphan, cfg)
    assert "tail prose" in redact_sensitive_text(stamped_orphan, cfg)
    protected, _r, _c = protect_embedding_text(colon_orphan, cfg)
    validate_embedding_privacy_dispatch([protected], cfg, expected_revision=revision)

    # R16: a long (>200-line) prefixed orphan run redacts COMPLETELY — a
    # capped walk that partially redacts also destroys the END evidence
    # validation needs (fail-open); the watermark walk is uncapped + linear.
    lines = ["INFO:" + ("A" * 63) + chr(66 + (i % 20)) for i in range(201)]
    long_run = "\n".join(lines) + "\nINFO:-----END PRIVATE KEY-----"
    out = redact_sensitive_text(long_run, cfg)
    assert lines[0] not in out and lines[-1] not in out

    # R16: prefix composed with escaped solidus normalizes on the prefix path.
    pfx_solidus = (
        '{"log": "INFO -----BEGIN PRIVATE KEY-----\\nINFO '
        + "QQQQabcd" * 4
        + '\\/QQQQ\\nprose"}'
    )
    assert "QQQQabcd" not in redact_sensitive_text(pfx_solidus, cfg)

    # R17: solidus escapes nest with serialization depth — derive depths 1-3
    # from json.dumps itself so the spelling can never drift.
    import json as _json16

    inner = {"log": "INFO -----BEGIN PRIVATE KEY-----\nINFO " + "QQQQabcd" * 4 + "/QQQQ\nprose"}
    d1 = _json16.dumps(inner)
    d2 = _json16.dumps({"o": d1})
    d3 = _json16.dumps({"oo": d2})
    for depth_text in (d1, d2, d3):
        for redacted in (
            redact_sensitive_text(depth_text, cfg),
            protect_embedding_text(depth_text, cfg)[0],
        ):
            assert "QQQQabcd" not in redacted

    # R19: doc headings after a bare BEGIN stay; PEM tails after body still
    # redact; blank separators after unarmored BEGINs; 5-token syslog prefixes.
    heading = "-----BEGIN PRIVATE KEY-----\nIMPORTANT\nthis heading explains keys"
    assert redact_sensitive_text(heading, cfg) == heading
    tail = "-----BEGIN PRIVATE KEY-----\n" + body + "\nCDEF\nprose"
    assert "CDEF" not in redact_sensitive_text(tail, cfg)
    blank_sep = "-----BEGIN PRIVATE KEY-----\n\n" + body + "\nprose stays"
    out = redact_sensitive_text(blank_sep, cfg)
    assert body not in out and "prose stays" in out
    syslog = (
        "Aug 26 12:34:56 host app[123]: -----BEGIN PRIVATE KEY-----\n"
        "Aug 26 12:34:56 host app[123]: " + body + "\n"
        "Aug 26 12:34:56 host app[123]: CDEF\nplain tail"
    )
    out = redact_sensitive_text(syslog, cfg)
    assert body not in out and "CDEF" not in out and "plain tail" in out

    # Precision: suffixed marker mentions and hash dumps stay untouched.
    for keep in (
        "we discussed the -----END PRIVATE KEY----- marker in prose",
        "abcdef0123456789AB\nsee label: -----END PRIVATE KEY----- for format details",
        "INFO " + "a1b2c3d4" * 8 + "\nINFO " + "e5f6a7b8" * 8 + "\nno markers anywhere",
    ):
        assert redact_sensitive_text(keep, cfg) == keep
