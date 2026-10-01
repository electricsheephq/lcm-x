"""#491: redact env-style assignments without widening values or replay identity.

All credential-shaped values are explicitly synthetic; no provider is called.
Boolean comparisons keep synthetic credential text out of failure output.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from hermes_lcm import ingest_protection as ip
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


SYNTHETIC = "SYNTHETIC_NOT_A_LIVE_KEY_" + "x" * 32
CONFIG = SimpleNamespace(sensitive_patterns_enabled=True, sensitive_patterns=["api_key"])
KEY_NAMES = ("api_key", "api_token", "access_token", "secret_key", "client_secret")


def _case(prefix, secret=SYNTHETIC, suffix=""):
    return prefix + secret + suffix, prefix + ip._sensitive_placeholder("api_key", secret) + suffix


CASES = [
    pytest.param(*_case("OPENROUTER_API_KEY=", "sk-or-v1-" + SYNTHETIC), id="openrouter-env"),
    pytest.param(*_case('export OPENAI_API_KEY="', "sk-proj-" + SYNTHETIC, '"'), id="openai-export"),
    pytest.param(*_case("MY_SERVICE_ACCESS_TOKEN: "), id="access-token-yaml"),
    pytest.param(*_case("CLIENT_SECRET="), id="client-secret"),
    pytest.param(*_case('{"OPENROUTER_API_KEY": "', suffix='"}'), id="json-env-key"),
    pytest.param(*_case("MY_SERVICE_SECRET_KEY: '", suffix="'"), id="quoted-yaml-env-key"),
    *[
        pytest.param(*_case("SERVICE42_" + name.replace("_", separator).upper() + "="),
                     id=f"env-{name}-{separator or 'compact'}")
        for name in KEY_NAMES
        for separator in ("_", "-", "")
    ],
    *[
        pytest.param(text, text, id=case_id)
        for case_id, text in (
            ("letter-before-name", "xapi_key=" + SYNTHETIC),
            ("digit-before-name", "7api_key=" + SYNTHETIC),
            ("name-suffix", "API_KEY_ID=" + SYNTHETIC),
            ("short-value", "OPENAI_API_KEY=short123456"),
            ("prose", "the api key is rotated weekly"),
        )
    ],
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_env_names_preserve_prefix_and_redact_only_value(text, expected):
    matches = ip.redact_sensitive_text(text, CONFIG) == expected
    assert matches


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_timeout_engine_matches_stdlib(text, expected, monkeypatch):
    if ip._regex_engine is None:
        pytest.skip("the optional regex module is not installed; the stdlib pattern is the only engine")
    # Compile through the production mirror rather than a test-only regex.
    monkeypatch.setattr(ip, "_SENSITIVE_REGEX_CATALOG", {})
    timeout_pattern = ip._regex_pattern_for("api_key")
    def replace(match):
        return ip._redact_match("api_key", match)

    timed = timeout_pattern.sub(replace, text, timeout=ip._SENSITIVE_MATCH_TIMEOUT_SECONDS)
    stdlib = ip._SENSITIVE_PATTERN_CATALOG["api_key"].sub(replace, text)
    engines_agree = timed == stdlib
    matches = timed == expected
    assert engines_agree
    assert matches


def _engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key"],
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("issue-491", platform="cli", conversation_id="issue-491",
                            context_length=200_000)
    return engine


def test_real_catalog_widening_replays_old_store_without_restoring(tmp_path, monkeypatch):
    # Use the #758 widening fixture's 20-row host tail, but restore the real
    # shipped catalog for restart instead of simulating a widened candidate.
    candidate = ip._SENSITIVE_PATTERN_CATALOG["api_key"]
    old_pattern = re.compile(
        candidate.pattern.replace(r"(?<![A-Za-z0-9])(?:api", r"\b(?:api", 1),
        candidate.flags,
    )
    line, expected = _case('export OPENAI_API_KEY="', "sk-proj-" + SYNTHETIC, '"')
    host = []
    for turn in range(10):
        host.append({"role": "user", "content": f"user turn {turn}" + ("\n" + line if turn == 3 else "")})
        host.append({"role": "assistant", "content": f"assistant reply {turn}"})
    with monkeypatch.context() as old:
        old.setitem(ip._SENSITIVE_PATTERN_CATALOG, "api_key", old_pattern)
        old_misses_env = ip.redact_sensitive_text(line, CONFIG) == line
        assert old_misses_env
        first = _engine(tmp_path)
        try:
            first.ingest(list(host))
            original_rows = first._store.get_session_messages("issue-491")
            assert first._store.get_session_count("issue-491") == 20
            assert any(line in row["content"] for row in original_rows)
        finally:
            first.shutdown()
    candidate_redacts_env = ip.redact_sensitive_text(line, CONFIG) == expected
    assert candidate_redacts_env
    for restart in range(2):
        host.append({"role": "user", "content": f"new user row after restart {restart}"})
        engine = _engine(tmp_path)
        try:
            engine.ingest(list(host))
            assert engine._store.get_session_count("issue-491") == 21 + restart
            assert engine._last_ingest_reconciliation["reason"] == (
                "replayed durable tail across a redaction policy change"
            )
            # Forward-only: replay does not rewrite the old durable rows.
            rows = engine._store.get_session_messages("issue-491")
            old_rows_unchanged = rows[:20] == original_rows
            assert old_rows_unchanged
        finally:
            engine.shutdown()
