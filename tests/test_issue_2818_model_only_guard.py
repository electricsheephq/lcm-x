"""#2818: replay projection should preserve the identity of a no-progress compress."""

import sys
from copy import deepcopy

import pytest

from hermes_lcm.compaction import CompactionMixin
from hermes_lcm.engine import _model_only_display_metadata_keys
from tests.behaviour.test_child_isolation_lifecycle import native as native
from tests.harness import engine_context
from tests.test_issue_1201_summary_model_only import _compress, _history, _host, _set_replay_path, _summary_rows
from tests.test_issue_1201_summary_model_only import engine as engine
from tests.test_issue_1201_summary_model_only import host as host


@pytest.mark.parametrize(("replay_path", "project"), [
    pytest.param("shortcut", True, id="shortcut-projected"),
    pytest.param("shortcut", False, id="shortcut-unprojected"),
    pytest.param("shortcut-off", True, id="shortcut-off-projected"),
    pytest.param("shortcut-off", False, id="shortcut-off-unprojected"),
    pytest.param("uid-on", True, id="uid-on-projected", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="UID replay of the projected host list fails to restore host_model_only on the inner summary",
    )),
    pytest.param("uid-on", False, id="uid-on-unprojected"),
])
def test_second_compress_preserves_input_identity(engine, monkeypatch, project, replay_path):
    _host(monkeypatch, "host_model_only")
    history = _history()
    # Hermes keeps the summary separate from a user row with structured content.
    history[-4]["content"] = [{"type": "text", "text": history[-4]["content"]}]
    history[-3]["tool_calls"] = [{
        "id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"},
    }]
    history[-2] = {"role": "tool", "tool_call_id": "call-1", "content": "Synthetic lookup result."}
    for i, row in enumerate(history[-3:]):
        row["timestamp"] = 1_760_000_000 + i
    _set_replay_path(engine, monkeypatch, history, replay_path)
    output = _compress(engine, history)
    [summary] = _summary_rows(output)
    assert summary["display_metadata"] == {"host_model_only": True}
    assert {row["role"] for row in output if "timestamp" in row} >= {"assistant", "tool"}

    if project:
        output = deepcopy(output)
        hide_keys = _model_only_display_metadata_keys()
        for row in output:
            if row["role"] == "user" and "display_metadata" in row:
                metadata = row["display_metadata"]
                for key in hide_keys:
                    metadata.pop(key, None)
                if not metadata:
                    del row["display_metadata"]
            if row["role"] in {"assistant", "tool"}:
                row.pop("timestamp", None)

    before = deepcopy(output)
    candidates = []
    recognition = []
    compress_impl = engine._compress_impl

    def capture(messages, *args, **kwargs):
        recognition.append(engine._uid_recognition_enabled() and any(row.get("message_uid") for row in messages))
        candidate = compress_impl(messages, *args, **kwargs)
        candidates.append(candidate)
        return candidate

    monkeypatch.setattr(engine, "_compress_impl", capture)
    second = _compress(engine, output)
    if recognition != [replay_path == "uid-on"]:
        pytest.fail(f"unexpected UID recognition at compress: {recognition}")
    [candidate] = candidates
    assert second is output
    assert output == before
    if replay_path == "shortcut":
        assert candidate is output
    else:
        assert candidate is not output
        candidate_public = [engine._public_compression_row(row) for row in candidate]
        [rebuilt_summary] = _summary_rows(candidate_public)
        assert rebuilt_summary.get("display_metadata") == {"host_model_only": True}
        if project:
            del rebuilt_summary["display_metadata"]
        assert candidate_public == [engine._public_compression_row(row) for row in before]


@pytest.mark.parametrize(
    ("result_fields", "input_fields", "equal"),
    [
        pytest.param({"display_metadata": {"model_only": True}}, {}, True, id="added-marker"),
        pytest.param({"display_metadata": {"model_only": True, "host_model_only": True, "origin": "lcm"}},
                     {"display_metadata": {"origin": "lcm"}}, True, id="both-markers-added"),
        pytest.param({"display_metadata": {}}, {}, True, id="result-empty"),
        pytest.param({}, {"display_metadata": {}}, True, id="input-empty"),
        pytest.param({}, {"display_metadata": {"host_model_only": True}}, False, id="marker-removed"),
        pytest.param({"display_metadata": {"host_model_only": True}},
                     {"display_metadata": {"host_model_only": False}}, False, id="marker-updated"),
        pytest.param({"display_metadata": {"host_model_only": False}}, {}, False, id="false-marker-added"),
        pytest.param({"display_metadata": {"host_model_only": 1}}, {}, False, id="integer-marker-added"),
        pytest.param({"display_metadata": {"model_only": True, "origin": "lcm"}}, {}, False, id="other-metadata"),
        pytest.param({"display_kind": "steer"}, {}, False, id="display-kind"),
        pytest.param({"content": "Changed text."}, {}, False, id="content"),
        pytest.param({"role": "assistant"}, {}, False, id="role"),
        pytest.param({"message_uid": "changed-uid"}, {}, False, id="message-uid"),
        pytest.param({"tool_calls": [{"id": "call-1"}]}, {}, False, id="tool-calls"),
    ],
)
def test_compression_row_comparison(monkeypatch, result_fields, input_fields, equal):
    _host(monkeypatch, "host_model_only")
    common = {"role": "user", "content": "Synthetic text."}
    result = {**common, **result_fields}
    original = {**common, **input_fields}
    before = deepcopy((result, original))

    assert CompactionMixin()._compression_rows_equal(result, original) is equal
    assert (result, original) == before


def test_bypassed_compress_preserves_input_identity_for_marker(tmp_path, monkeypatch, native):
    compressor = sys.modules["agent.context_compressor"].ContextCompressor
    original = [{"role": "user", "content": "Synthetic summary."}]
    marked = [{**original[0], "display_metadata": {"model_only": True}}]
    created, _, _ = native

    with engine_context(tmp_path, window=200_000, session_id="ignored-2818",
                        ignore_session_patterns=["ignored-*"]) as bypassed:
        bypassed.model = "test-model"
        monkeypatch.setattr(compressor, "compress", lambda *args, **kwargs: marked)
        assert bypassed.compress(original, force=True) is original
        assert len(created) == 1

        longer = marked + [{"role": "assistant", "content": "New row."}]
        monkeypatch.setattr(compressor, "compress", lambda *args, **kwargs: longer)
        assert bypassed.compress(original, force=True) == longer
