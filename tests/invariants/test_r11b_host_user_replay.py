"""T1-613 regressions for keyed redaction and host replay."""

import json
import re

import pytest

from hermes_lcm.ingest_protection import redact_sensitive_value
from tests.harness.engine import stored_rows
from tests.invariants.test_p0_host_user_rows import (
    _assistant, _stamp, _user, open_engine as open_engine,
)
from tests.invariants.test_r8_host_user_replay import _cleanup
from tests.invariants.test_r10_host_user_replay import LITERAL_MARKER

PATTERNS = dict(
    sensitive_patterns_enabled=True,
    sensitive_patterns=["api_key", "bearer_token", "password_assignment", "private_key"],
)
SECRET = "sk-synthetic-first-123456789"
OTHER_SECRET = "sk-synthetic-second-987654321"
PASSWORD = "synthetic café phrase"
BEARER_MARKER = r"Bearer \[LCM sensitive redaction: name=bearer_token;[^\]]*\]"


def _tool_calls(header):
    return [{
        "id": "call_synthetic",
        "type": "function",
        "function": {
            "name": "fixture",
            "arguments": json.dumps({"headers": {"Authorization": header}}),
        },
    }]


def _header(tool_calls):
    return json.loads(tool_calls[0]["function"]["arguments"])["headers"]["Authorization"]


def _tool_history(content):
    answer = _assistant(content)
    answer["tool_calls"] = _tool_calls(f"Bearer {SECRET}")
    result = {"role": "tool", "tool_call_id": "call_synthetic", "content": "Synthetic result."}
    return [_user("Synthetic question."), answer, result]


@pytest.mark.parametrize("shape", ["Authorization", "token", "JSON string", "tool arguments"])
def test_a_key_and_delivery_shapes(open_engine, shape):
    raw = f"Bearer {SECRET}"
    value = {
        "Authorization": {"Authorization": raw},
        "token": {"token": raw},
        "JSON string": json.dumps({"Authorization": raw, "token": raw}),
        "tool arguments": _tool_calls(raw),
    }[shape]
    with open_engine(**PATTERNS) as engine:
        first = redact_sensitive_value(value, engine._config, parse_json_strings=True)
        assert SECRET not in repr(first)
        assert "Bearer " in repr(first)
        assert redact_sensitive_value(first, engine._config, parse_json_strings=True) == first


def test_b_keyed_value_shapes_are_fixed_points(open_engine):
    with open_engine(**PATTERNS) as engine:
        def redact(value):
            return redact_sensitive_value(value, engine._config, parse_json_strings=True)

        placeholder = redact({"api_key": SECRET})["api_key"]
        shapes = {
            "bare API key": {"api_key": SECRET},
            "client secret": {"client_secret": OTHER_SECRET},
            "bare password": {"password": PASSWORD},
            "quoted password": {"password": f'password="{PASSWORD}"'},
            "quoted API assignment": {"api_key": f'api_key="{SECRET}"'},
            "list parts": [{"type": "text", "text": f"api_key={SECRET}"}],
            "nested keyed list": {"parts": [{"api_key": SECRET}]},
            "duplicate keys": f'{{"client_secret":"{SECRET}","client_secret":"{OTHER_SECRET}"}}',
            "escaped duplicate keys": json.dumps(
                f'{{"client_secret":"{SECRET}","client_secret":"{OTHER_SECRET}"}}'
            ),
            "nested JSON": json.dumps({"payload": json.dumps({"client_secret": SECRET})}),
            "bare token": {"token": SECRET},
            "bearer header": {"Authorization": f"Bearer {SECRET}"},
            "multiple placeholders": {"api_key": f"api_key={SECRET}\tBearer {OTHER_SECRET}"},
            "surrounding keyed text": {"api_key": f"Synthetic note\tapi_key={OTHER_SECRET}"},
            "mixed marker forms": {"api_key": f"{SECRET}\t{placeholder}  {LITERAL_MARKER}\n{SECRET}"},
            "private key within keyed text": {
                "api_key": "-----BEGIN PRIVATE KEY-----\nQUJDREVGR0hJSktMTU5PUA==\n-----END PRIVATE KEY-----"
            },
        }
        changed_on_second_pass = []
        for shape, value in shapes.items():
            first = redact(value)
            assert first != value, shape
            if redact(first) != first:
                changed_on_second_pass.append(shape)
        assert changed_on_second_pass == []


def test_c_no_progress_restores_host_reasoning(open_engine):
    host = [*_tool_history("<think>Synthetic host reasoning.</think>Visible answer."),
            _user("Another question.")]
    _stamp(host)
    answer = host[1]
    with open_engine(**PATTERNS) as engine:
        engine.ingest(host)
        protected = redact_sensitive_value(answer["tool_calls"], engine._config, parse_json_strings=True)
        assert re.fullmatch(BEARER_MARKER, _header(protected))
        result = _cleanup(engine, host)
        returned, = [row for row in result if row.get("message_uid") == answer["message_uid"]]
        assert returned == answer | {"tool_calls": protected}


@pytest.mark.parametrize("reopen", [False, True], ids=["warm", "cold"])
def test_d_session_end_matches_stored_header(open_engine, monkeypatch, reopen):
    host = _tool_history("Synthetic tool use.")
    _stamp(host)
    answer = host[1]

    def check(engine, before, ending):
        engine.ingest(ending)
        assert stored_rows(engine) == before
        assert engine._session_end_store_prefix_count("S", ending) == len(before)
        engine.on_session_end("S", ending)
        assert stored_rows(engine, "S") == before

    with open_engine(**PATTERNS) as engine:
        protected = redact_sensitive_value(answer["tool_calls"], engine._config, parse_json_strings=True)
        assert re.fullmatch(BEARER_MARKER, _header(protected))
        ending = [host[0], answer | {"tool_calls": protected}, host[2]]
        # Seed a host row that already carries the first-pass bytes.
        with monkeypatch.context() as seed:
            seed.setattr(engine._config, "sensitive_patterns_enabled", False)
            engine.ingest(ending)
        before = stored_rows(engine)
        stored, = [row for row in before if row["role"] == "assistant"]
        assert len(before) == len(ending)
        assert _header(stored["tool_calls"]) == _header(protected)
        if not reopen:
            check(engine, before, ending)
            return

    with open_engine(**PATTERNS) as engine:
        check(engine, before, ending)


@pytest.mark.parametrize("malformed", [
    "example",
    "name=example; chars=1; bytes=1; sha256=0123456789abcdef",
    "name=api_key; chars=1; sha256=0123456789abcdef",
    "name=api_key; chars=01; bytes=1; sha256=0123456789abcdef",
    "name=api_key; chars=1; bytes=1",
    "name=api_key; chars=1; bytes=1; sha256=0123456789abcdEF",
    "name=password_assignment; chars=1; bytes=1; sha256=0123456789abcdef",
], ids=["literal", "unknown-name", "missing-bytes", "leading-zero", "missing-digest",
        "uppercase-digest", "password-with-digest"])
def test_malformed_marker_keeps_keyed_redaction(open_engine, malformed):
    marker = f"[LCM sensitive redaction: {malformed}]"
    with open_engine(**PATTERNS) as engine:
        config = engine._config
        placeholder = redact_sensitive_value({"api_key": SECRET}, config)["api_key"]
        value = {"api_key": f"{SECRET} {placeholder} {marker}"}
        expected = {"api_key": f"{placeholder} {placeholder} {marker}"}
        assert redact_sensitive_value(value, config) == expected
        assert redact_sensitive_value(expected, config) == expected


def test_emitted_password_placeholder_is_byte_identical(open_engine):
    with open_engine(**PATTERNS) as engine:
        config = engine._config
        first = redact_sensitive_value({"password": f'password="{PASSWORD}"'}, config)
        placeholder = first["password"]
        assert re.fullmatch(
            r'password="\[LCM sensitive redaction: name=password_assignment; chars=\d+; bytes=\d+\]"',
            placeholder,
        )
        assert redact_sensitive_value(first, config)["password"].encode() == placeholder.encode()


@pytest.mark.xfail(strict=True, reason="T1-613: credential beside a genuine placeholder is an rc1 known issue")
def test_credential_beside_genuine_placeholder(open_engine):
    with open_engine(**PATTERNS) as engine:
        config = engine._config
        placeholder = redact_sensitive_value({"api_key": OTHER_SECRET}, config)["api_key"]
        value = {"api_key": f"{SECRET} {placeholder}"}
        assert SECRET not in redact_sensitive_value(value, config)["api_key"]
