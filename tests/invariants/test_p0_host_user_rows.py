"""Public compress() must retain host-owned user occurrences and their API text."""

import logging
from copy import deepcopy

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit, survival_fit, tokens
from hermes_lcm.compaction import CompactionMixin
from tests.harness.engine import engine_context
from tests.test_host_uid_shadow import _state_db
from tests.test_ingest_protection import _PrefixPattern, _broken_assistant_output
from tests.test_issue_671_stub_first_exit import tool_pair

X = "X" * 1297
Y = "Y" * 153
EMPTY_FORMS = ({"content": ""}, {"content": None}, {})


@pytest.fixture(params=[None, "on", "off"], ids=["uid-unset", "uid-on", "uid-off"])
def open_engine(request, monkeypatch, tmp_path):
    if request.param is not None:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", request.param)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", request.param == "on")
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda _: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: len(text) // 4 if text else 0)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Prior work.", 1))
    home = tmp_path / "home"
    home.mkdir()
    _state_db(home, [("S", None, None)])
    defaults = dict(fresh_tail_count=4, leaf_chunk_tokens=100, threshold_full_sweep_enabled=False,
                    fresh_tail_pressure_yield_enabled=False, embeddings_enabled=False,
                    temporal_rollups_enabled=False, survival_fit=False, incremental_max_depth=0)
    return lambda **overrides: engine_context(
        tmp_path, window=32_768, conversation_id="conv", **(defaults | overrides),
    )


def _user(content, **fields):
    return dict(role="user", content=content, **fields)


def _assistant(content):
    return {"role": "assistant", "content": content}


def _stamp(rows):
    for index, row in enumerate(rows, 1):
        row.setdefault("message_uid", f"{index:032x}")


def _compress(engine, rows, **kwargs):
    _stamp(rows)
    snapshot = deepcopy(rows)
    kwargs.setdefault("current_tokens", tokens.count_messages_tokens(rows))
    result = engine.compress(rows, **kwargs)
    assert rows == snapshot
    return result


def _stop_line(caplog):
    return [line for line in caplog.messages if line.startswith("LCM compaction stop:")][-1]


def test_8300_wrapper_returns_host_user(open_engine, monkeypatch):
    rows = [_user(X, api_content=Y)]
    monkeypatch.setattr(CompactionMixin, "compress", lambda self, messages, **_: [dict(messages[0], content="X'")])
    with open_engine() as engine:
        returned, = _compress(engine, rows)
    assert returned is rows[0]
    assert returned["api_content"] == Y


def _cached_compress(engine, rows, cached_content):
    source = rows[0]
    _stamp(rows)
    engine.ingest(rows)
    cached = engine._last_active_replay_messages[0]
    assert cached is not source
    cached["content"] = cached_content
    engine._preflight_automatic_request = True
    return _compress(engine, rows)[0]


def test_control_8300_equal_cache_keeps_sidecar(open_engine):
    rows = [_user(X, api_content=Y)]
    with open_engine() as engine:
        returned = _cached_compress(engine, rows, X)
    assert returned["api_content"] == Y


def test_8300_different_cache_returns_host_user(open_engine, monkeypatch):
    rows = [_user(X, api_content=Y)]
    with open_engine() as engine:
        observed = []
        keep = engine._keep_host_held_stubs

        def replay(*args):
            result = keep(*args)
            observed.append(result[0])
            return result

        monkeypatch.setattr(engine, "_keep_host_held_stubs", replay)
        returned = _cached_compress(engine, rows, "X'")
    assert observed and observed[-1] is rows[0]
    assert returned is rows[0]
    assert returned["api_content"] == Y


@pytest.mark.parametrize("form", range(3), ids=["empty-string", "none", "absent"])
def test_sidecar_only_keeps_original_empty_form(open_engine, monkeypatch, form):
    source = dict(role="user", api_content=Y, **EMPTY_FORMS[form])

    def rewritten(self, messages, **kwargs):
        return [dict(role="user", message_uid=messages[0]["message_uid"], api_content=Y,
                     **EMPTY_FORMS[(form + 1) % len(EMPTY_FORMS)])]

    monkeypatch.setattr(CompactionMixin, "compress", rewritten)
    with open_engine() as engine:
        returned, = _compress(engine, [source])
    assert returned is source
    assert returned["api_content"] == Y


@pytest.mark.parametrize("held", [False, True], ids=["below-threshold", "hold"])
def test_no_progress_returns_host_list(open_engine, monkeypatch, caplog, held):
    rows = [_user("Question"), _assistant("<think>Internal work</think>Visible reply.")]
    with open_engine() as engine, caplog.at_level(logging.INFO, logger="hermes_lcm"):
        if held:
            _stamp(rows)
            engine.ingest(rows)
            engine._last_active_replay_messages[1]["content"] = "Visible reply."
            engine._start_no_progress_hold("no_progress")
            assert engine._no_progress_hold_blocks(tokens.count_messages_tokens(rows))
        else:
            engine._preflight_automatic_request = True
        seen = []
        ingest_view = engine._ingest_compression_view

        def observe(*args, **kwargs):
            action, output = ingest_view(*args, **kwargs)
            seen.append((action, output))
            return action, output

        monkeypatch.setattr(engine, "_ingest_compression_view", observe)
        returned = _compress(engine, rows)
        assert len(seen) == 1
        action, working = seen[0]
        assert action == "return"
        assert working[0]["content"] == rows[0]["content"]
        assert working[1]["content"] == "Visible reply."
        assert engine.last_compression_status == "sanitized"
        assert "leaves=0 progress=none" in _stop_line(caplog)
        assert engine._ingest_cursor == len(rows)
    assert returned is rows


@pytest.mark.parametrize("cached_content", ["", "A"], ids=["empty", "first-part"])
def test_host_merged_user_keeps_both_parts(open_engine, cached_content):
    rows = [_user("A\n\nB")]
    with open_engine() as engine:
        returned = _cached_compress(engine, rows, cached_content)
    assert returned is rows[0]
    assert returned["content"] == "A\n\nB"


def _history():
    return [row for i in range(12) for row in (
        _user(f"Question {i}: " + "alpha beta gamma " * 30),
        _assistant(f"Answer {i}: " + "delta epsilon " * 30),
    )]


def test_summary_carrier_keeps_unrelated_tail_sidecar(open_engine, caplog):
    rows = _history()
    users = {row["content"]: row for row in rows if row["role"] == "user"}
    for row in users.values():
        row["api_content"] = Y
    with open_engine() as engine, caplog.at_level(logging.INFO, logger="hermes_lcm"):
        returned = _compress(engine, rows, force=True)
        assert "leaves=1 progress=leaf" in _stop_line(caplog)
        assert engine._compress_host_rows_consumed > 0
        assert engine._durable_commit_proof_payload()["emissions"]
        carrier = next((row for row in returned if engine._generated_context_carrier_remainder(row)), None)
        assert carrier is None, "the durable proof describes a carrier where the host needs a standalone user row"
        assert any(row is rows[-4] for row in returned)
        for row in returned:
            if row.get("content") in users:
                assert row is users[row["content"]]
                assert row["api_content"] == Y


@pytest.mark.parametrize("kind", ["redaction", "ignore", "quarantine", "externalized-tool"])
def test_control_no_progress_adopts_placeholder_without_sidecar(open_engine, caplog, kind):
    rows = [_user("Question")]
    if kind == "redaction":
        rows = [_user("Please use api_key=sk-syntheticexample1234567890", api_content=Y)]
        settings = dict(sensitive_patterns_enabled=True, sensitive_patterns=["api_key"])
        index, marker = 0, "[LCM sensitive redaction:"
    elif kind == "ignore":
        rows = [_user("Cronjob Response: synthetic heartbeat", api_content=Y), _user("Question")]
        settings = {}
        index, marker = 0, "[LCM active replay placeholder: message ignored;"
    elif kind == "quarantine":
        rows.append(dict(_assistant(_broken_assistant_output()), api_content=Y))
        settings = dict(large_output_externalization_enabled=False)
        index, marker = 1, "assistant output quarantined;"
    else:
        rows.extend(tool_pair("payload-1", "synthetic output " * 600))
        rows[2]["api_content"] = Y
        settings = dict(fresh_tail_count=1, large_output_externalization_enabled=True,
                        large_output_externalization_threshold_chars=1000,
                        large_output_active_replay_stubbing_enabled=True,
                        large_output_active_replay_stub_threshold_tokens=5)
        rows.append(_user("Follow-up"))
        index, marker = 2, "[Externalized tool output:"
    with open_engine(**settings) as engine, caplog.at_level(logging.INFO, logger="hermes_lcm"):
        if kind == "ignore":
            engine._compiled_ignore_message_patterns = [_PrefixPattern()]
        engine._preflight_automatic_request = True
        returned = _compress(engine, rows)
        assert "leaves=0 progress=none" in _stop_line(caplog)
        assert engine._ingest_cursor == len(returned)
    assert marker in returned[index]["content"]
    assert "api_content" not in returned[index]
