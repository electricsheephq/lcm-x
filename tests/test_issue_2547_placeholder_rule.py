"""#2547: the identity line under a redacted preview (release-manager ruling of 11-10, condition b).

Ingest writes the identity line from the marker's raw preview, before redaction, so when the stored preview holds a
redaction placeholder the digest cannot be checked; the compare then reads the line as LCM-X's own. Without a
placeholder, an identity-shaped line whose digest is not the marker's own stays part of the text.
"""

import pytest

from hermes_lcm.ingest_persisted import (
    _add_inline_persisted_output_identity_metadata,
    _persisted_output_marker_identity_digest,
)
from hermes_lcm.ingest_protection import redact_sensitive_value
from tests.test_issue_10_persisted_output_spillover import _marker
from tests.test_issue_2521_spillover_upgrade_replay import _engine

SECRET_BODY = "api_key=placeholderrule1234567890abcdef " + "z" * 260


@pytest.fixture
def engine(tmp_path):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["all"]
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="c")
    try:
        yield engine
    finally:
        engine.shutdown()


def _tool(content):
    return {"role": "tool", "tool_call_id": "c1", "content": content}


def test_a_redacted_preview_with_its_ingest_identity_line_equals_the_host_marker(engine):
    raw = _marker("/tmp/hermes-results/c1.txt", SECRET_BODY)
    redacted = redact_sensitive_value(raw, engine._config, parse_json_strings=False)
    if redacted == raw:
        pytest.fail("setup: redaction did not change the preview")
    stored = _add_inline_persisted_output_identity_metadata(redacted, _persisted_output_marker_identity_digest(raw))
    assert stored != redacted
    assert engine._session_end_rows_equal(_tool(stored), _tool(raw), session_id="S1")


def test_an_identity_line_with_a_wrong_digest_and_no_placeholder_stays_unequal(engine):
    raw = _marker("/tmp/hermes-results/c1.txt", "x" * 300)
    forged = _add_inline_persisted_output_identity_metadata(raw, "f" * 64)
    assert forged != raw
    assert not engine._session_end_rows_equal(_tool(forged), _tool(raw), session_id="S1")
    assert not engine._session_end_rows_equal(_tool(raw), _tool(forged), session_id="S1")
    ident = engine._session_end_prefix_compare_identity(_tool(forged), session_id="S1", recover_markers=False)
    assert ident != engine._session_end_prefix_compare_identity(_tool(raw), session_id="S1", recover_markers=False)
