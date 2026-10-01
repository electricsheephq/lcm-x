"""#394 tail-width precision and #389 pure-hex exclusion mutation pins."""

import pytest

import hermes_lcm.command as command
import hermes_lcm.ingest_protection as ip
from hermes_lcm.config import LCMConfig


_PROSE = (
    "We met at the station after the long meeting.\n"
    "Please file the expense report under REF2026QX7781ABCD\n"
    "Then forward the same report to ACCT4410ZZ8812WXYZ\n"
    "Thanks again."
)
_PEM_B0 = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj"
_PEM_B1 = "MHcCAQEEIQD1eJ7yhkG0987xyzABCDEFghijkLMNOPqrstuvwxyz0987654321pq"


def _config(tmp_path):
    return LCMConfig(
        database_path=str(tmp_path / "privacy.db"),
        embeddings_enabled=True,
        embedding_provider="voyage",
        embedding_model="voyage-4-large",
        sensitive_patterns_enabled=False,
        embedding_privacy_enabled=None,
        sensitive_patterns=[
            "api_key", "bearer_token", "password_assignment", "private_key"
        ],
    )


def test_prose_lines_ending_in_long_token_are_not_refused(tmp_path):
    cfg = _config(tmp_path)
    assert ip._has_orphan_full_width_base64_run(_PROSE) is False
    protected, revision, changed = ip.protect_embedding_text(_PROSE, cfg)
    assert protected == _PROSE
    assert changed is False
    ip.validate_embedding_privacy_dispatch(
        [protected], cfg, expected_revision=revision
    )


def test_numbered_list_with_sub_40_tail_is_not_refused(tmp_path):
    text = "Items:\n1. " + "Ab3" * 13 + "\n2. " + "Qz9x" * 19 + "Q9\nend"
    cfg = _config(tmp_path)
    assert ip._has_orphan_full_width_base64_run(text) is False
    protected, revision, changed = ip.protect_embedding_text(text, cfg)
    assert protected == text
    assert changed is False
    ip.validate_embedding_privacy_dispatch(
        [protected], cfg, expected_revision=revision
    )


def test_backfill_prepare_does_not_refuse_prose(tmp_path):
    cfg = _config(tmp_path)
    documents, revision, transformed, blocked, error = (
        command._prepare_embedding_provider_documents(
            [("n1", _PROSE, 10)],
            config=cfg,
            provider_name="voyage",
            expected_revision=ip.embedding_privacy_revision(cfg),
        )
    )
    assert blocked == 0
    assert error is None
    assert transformed == 0
    assert revision == ip.embedding_privacy_revision(cfg)
    assert len(documents) == 1
    assert documents[0][:2] == ("n1", _PROSE)


def test_prefixed_full_width_body_lines_still_block(tmp_path):
    text = f"INFO {_PEM_B0}\nINFO {_PEM_B1}"
    cfg = _config(tmp_path)
    assert ip._has_orphan_full_width_base64_run(text) is True
    try:
        protected, revision, _changed = ip.protect_embedding_text(text, cfg)
    except ip.EmbeddingPrivacyPolicyError:
        return
    with pytest.raises(ip.EmbeddingPrivacyPolicyError):
        ip.validate_embedding_privacy_dispatch(
            [protected], cfg, expected_revision=revision
        )


def test_unprefixed_full_width_body_lines_still_block():
    assert ip._has_orphan_full_width_base64_run(f"{_PEM_B0}\n{_PEM_B1}") is True


def test_long_token_numbered_list_stays_blocked_boundary():
    # Long prefixed IDs remain indistinguishable from a key body: fail closed.
    text = "\n".join(f"{i}. " + "Xy7" * 20 for i in range(1, 18))
    assert ip._has_orphan_full_width_base64_run(text) is True


def test_389_pure_hex_31_run_near_private_key_placeholder_blocks():
    ph = ip._embedding_privacy_placeholder("private_key")
    assert ip._pem_fragment_near_private_key_placeholder(
        ph + " 0123456789abcdef0123456789abcde"
    ) is True


def test_389_mixed_run_with_embedded_40_hex_blocks():
    ph = ip._embedding_privacy_placeholder("private_key")
    assert ip._pem_fragment_near_private_key_placeholder(
        ph + " Zq" + "0123456789abcdef" * 2 + "01234567Xy"
    ) is True


def test_389_pure_hex_40_run_dispatches():
    ph = ip._embedding_privacy_placeholder("private_key")
    assert ip._pem_fragment_near_private_key_placeholder(
        ph + " " + "0123456789abcdef0123456789abcdef01234567"
    ) is False
