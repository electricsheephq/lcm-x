"""#1201: display-only summaries, visible carriers, and auxiliary prompt isolation."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm import assertion_extraction, escalation, extraction, host_evidence, selective_compiler
from hermes_lcm import expand_query_tools
from tests.harness import make_engine
from hermes_lcm import engine as engine_module
from hermes_lcm.assertion_store import SourceSnapshot
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_analysis import _merge_adjacent_assistant_messages
from hermes_lcm.reconcile import _emission_identity, _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from hermes_lcm.tokens import count_messages_tokens
from tests.test_overflow_reserves_sanitizer_stubs import (
    OBJECTIVE, add_summaries, assemble, assistant_calls, full_cap, result,
)


def _host(monkeypatch, key="model_only"):
    module = ModuleType("agent.context_compressor")
    if key is not None:
        module.MODEL_ONLY_DISPLAY_METADATA_KEY = key
    monkeypatch.setitem(sys.modules, "agent.context_compressor", module)
    resolver = getattr(engine_module, "_model_only_display_metadata_key", None)
    if resolver is not None:
        resolver.cache_clear()


@pytest.fixture(autouse=True)
def host(monkeypatch):
    _host(monkeypatch)
    yield
    resolver = getattr(engine_module, "_model_only_display_metadata_key", None)
    if resolver is not None:
        resolver.cache_clear()


@pytest.fixture
def engine(tmp_path, monkeypatch):
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "l.db"), fresh_tail_count=4,
        leaf_chunk_tokens=100, extraction_enabled=False,
        temporal_rollups_enabled=False,
    ), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("issue-1201", platform="cli", context_length=200_000)
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda *a, **k: ("Earlier work completed.\nExpand for details about: work", 1))
    yield instance
    instance.shutdown()


@pytest.fixture
def expansion_engine(tmp_path, monkeypatch):
    instance = make_engine(
        tmp_path, window=200_000, session_id="issue-1201", fresh_tail_count=4,
        leaf_chunk_tokens=100, extraction_enabled=False, temporal_rollups_enabled=False,
    )
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda *a, **k: ("Earlier work completed.\nExpand for details about: work", 1))
    try:
        yield instance
    finally:
        instance.shutdown()


def _seed(engine):
    engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, depth=0, summary="Earlier work completed.",
        token_count=10, source_token_count=100, source_ids=[], source_type="messages",
        created_at=1, earliest_at=1, latest_at=1, expand_hint="work",
    ))


def _history():
    return [{"role": "system", "content": "System instructions."}] + [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"Human turn {i}. " + "detail " * 60}
        for i in range(20)
    ]


def _compress(engine, messages):
    return engine.compress(messages, current_tokens=count_messages_tokens(messages), force=True)


def _summary_rows(output):
    return [row for row in output if " Summary (d" in str(row.get("content"))
            and "System instructions." not in str(row.get("content"))]


_REAL_MERGE_PROBE = """
import importlib.util, json, os, sys
sys.path.insert(0, os.environ["LCM_REAL_HERMES_SRC"])
from agent.agent_runtime_helpers import _merge_consecutive_users
from agent.context_compressor import is_compaction_summary_message, split_user_originated_turn
root = sys.argv[1]
spec = importlib.util.spec_from_file_location("lcm_merge_probe", root + "/__init__.py",
                                            submodule_search_locations=[root])
sys.modules["lcm_merge_probe"] = importlib.util.module_from_spec(spec)
from lcm_merge_probe.engine import _host_will_merge_user_rows
results = []
for rows in json.load(sys.stdin):
    previous = rows[0]
    summary = is_compaction_summary_message(previous)
    carrier = split_user_originated_turn(previous)[0] is not None
    predicted = _host_will_merge_user_rows(*rows)
    merged, repairs = _merge_consecutive_users(rows)
    results.append(dict(summary=summary, carrier=carrier, predicted=predicted,
                        merged=merged, repairs=repairs))
print(json.dumps(results))
"""


def _real_merge(rows, tmp_path):
    if not os.environ.get("LCM_REAL_HERMES_SRC"):
        pytest.skip("no real Hermes source supplied")
    home = tmp_path / "merge-home"
    home.mkdir(exist_ok=True)
    probe = """
import json, os, sys
sys.path.insert(0, os.environ["LCM_REAL_HERMES_SRC"])
from agent.agent_runtime_helpers import _merge_consecutive_users
merged, repairs = _merge_consecutive_users(json.load(sys.stdin))
print(json.dumps(dict(merged=merged, repairs=repairs)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe], input=json.dumps(rows), capture_output=True,
        text=True, timeout=60, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HERMES_HOME": str(home),
             "PYTHONPATH": os.environ["LCM_REAL_HERMES_SRC"],
             "LCM_REAL_HERMES_SRC": os.environ["LCM_REAL_HERMES_SRC"],
             "PYTHONDONTWRITEBYTECODE": "1", "HERMES_DISABLE_LAZY_INSTALLS": "1"},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("key", ["model_only", "host_model_only"])
@pytest.mark.parametrize("sentinel_row", ["generated", "human"])
def test_r5_pinned_host_sentinel_merge_keeps_human_visible(monkeypatch, key, sentinel_row):
    _host(monkeypatch, key)

    def split_user_originated_turn(row):
        return None, None

    monkeypatch.setattr(sys.modules["agent.context_compressor"], "split_user_originated_turn",
                        split_user_originated_turn, raising=False)
    STEER_DISPLAY_KIND = "steer"

    def pinned_merge(rows):
        # Hermes f97608f: no JSON-sentinel exemption; the first row keeps its metadata.
        merged = []
        for msg in deepcopy(rows):
            prev = merged[-1] if merged and isinstance(merged[-1], dict) else None
            if (
                prev is not None and prev.get("role") == "user"
                and isinstance(msg, dict) and msg.get("role") == "user"
                and split_user_originated_turn(prev)[0] is None
                and prev.get("display_kind") != STEER_DISPLAY_KIND
                and isinstance(prev.get("content", ""), str)
                and isinstance(msg.get("content", ""), str)
            ):
                before, after = prev.get("content", ""), msg.get("content", "")
                prev["content"] = before + "\n\n" + after if before and after else before or after
            else:
                merged.append(msg)
        return merged

    generated = {"role": "user", "content": "Generated context."}
    human = {"role": "user", "content": "Human request must stay visible."}
    sentinel = generated if sentinel_row == "generated" else human
    sentinel["content"] = "\x00json:" + sentinel["content"]
    rows = [generated, human]
    engine_module._stamp_generated_model_only(rows, [generated])
    merged = pinned_merge(rows)
    assert len(merged) == 1 and human["content"] in merged[0]["content"]
    assert not merged[0].get("display_metadata", {}).get(key), "merged human text was hidden"


def _r3_generated_merge_group_display(engine, monkeypatch, human):
    _seed(engine)
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
        "role": "user", "content": "<relevant-memories>Retrieved context.</relevant-memories>",
        "display_metadata": {"origin": "recall"},
    })
    tail = [{"role": "assistant", "content": "A human-visible reply."},
            {"role": "user", "content": "Newest human request."}]
    if human:
        tail.insert(0, {"role": "user", "content": "Human prompt absorbed by the group."})
    before = deepcopy(tail)
    output = engine._assemble_context(None, tail, persist=True)
    assert _summary_rows(output) == [output[0]]
    assert "relevant-memories" in output[1]["content"]
    assert all(bool(row.get("display_metadata", {}).get("model_only")) is not human
               for row in output[:2])
    assert tail == before
    return output


@pytest.mark.parametrize("human", [False, True], ids=["generated-only", "absorbs-human"])
def test_r3_generated_merge_group_display(engine, monkeypatch, human):
    _r3_generated_merge_group_display(engine, monkeypatch, human)


@pytest.mark.skipif(not os.environ.get("LCM_REAL_HERMES_SRC"), reason="no real Hermes source supplied")
@pytest.mark.parametrize("human", [False, True], ids=["generated-only", "absorbs-human"])
def test_r3_generated_merge_group_display_real_host(engine, monkeypatch, tmp_path, human):
    output = _r3_generated_merge_group_display(engine, monkeypatch, human)
    result = _real_merge(output, tmp_path)
    merged = result["merged"]
    assert result["repairs"] == (2 if human else 1)
    assert bool(merged[0].get("display_metadata", {}).get("model_only")) is not human
    visible = [row for row in merged if not row.get("display_metadata", {}).get("model_only")]
    assert any("Human prompt absorbed" in row["content"] for row in visible) is human


@pytest.mark.skipif(not os.environ.get("LCM_REAL_HERMES_SRC"), reason="no real Hermes source supplied")
@pytest.mark.parametrize("key", [None, "model_only", "host_model_only"])
def test_issue_1370_real_host_merge_keeps_current_user_prompt_visible(engine, monkeypatch, tmp_path, key):
    _host(monkeypatch, key)
    _seed(engine)
    tail = [{"role": "user", "content": "The current human request."}]
    before = deepcopy(tail)
    output = engine._assemble_context(None, tail, persist=False)
    assert len(output) == 2 and all(row["role"] == "user" for row in output)
    assert _summary_rows(output) == [output[0]]
    assert output[1] == tail[0] and tail == before
    marked = deepcopy(output[0])
    marked["display_metadata"] = {"model_only": True}
    # Native carriers are not LCM's [Recent Summary ...] shape. Exercise both
    # recognition routes and every other exemption with the real host predicate.
    exempt = [
        {**marked, "_compressed_summary": True},
        {**marked, "content": "[CONTEXT SUMMARY]: Earlier work."},
        {**marked, "display_kind": "steer"},
        {**marked, "content": "\x00json:undecodable"},
        {**marked, "content": [{"type": "text", "text": "Earlier work."}]},
    ]
    cases = [output, [marked, tail[0]], *[[row, tail[0]] for row in exempt],
             [marked, {"role": "user", "content": "\x00json:undecodable"}],
             [marked, {"role": "user", "content": [{"type": "text", "text": "Human request."}]}],
             [marked, {"role": "assistant", "content": "Reply."}],
             [{"role": "user"}, tail[0]]]
    home = tmp_path / "real-hermes-home"
    home.mkdir()
    completed = subprocess.run(
        [sys.executable, "-c", _REAL_MERGE_PROBE, str(Path(__file__).resolve().parents[1])],
        input=json.dumps(cases), capture_output=True, text=True, timeout=60, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HERMES_HOME": str(home),
             "LCM_REAL_HERMES_SRC": os.environ["LCM_REAL_HERMES_SRC"],
             "PYTHONDONTWRITEBYTECODE": "1", "HERMES_DISABLE_LAZY_INSTALLS": "1"},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    assert all(row["predicted"] or not row["repairs"] for row in results), results
    # Newer hosts may exempt sentinel strings; conservatively predict their merge.
    assert [row["predicted"] for row in results] == [
        True, True, False, False, False, True, False, True, False, False, True,
    ]
    assert all(not row["summary"] and not row["carrier"] for row in results[:2])
    assert [row["summary"] for row in results[2:7]] == [True, True, False, False, False]
    sentinel_repairs = results[5]["repairs"]  # pinned host merges; newer hosts may exempt
    assert sentinel_repairs in (0, 1)
    assert [row["repairs"] for row in results] == [
        1, 1, 0, 0, 0, sentinel_repairs, 0, sentinel_repairs, 0, 0, 1,
    ]
    merged = results[0]["merged"]
    assert len(merged) == 1 and results[0]["repairs"] == 1
    assert merged[0]["content"] == output[0]["content"] + "\n\n" + tail[0]["content"]
    visible = [row for row in merged if not row.get("display_metadata", {}).get(key)]
    assert visible == merged, "the host merge must not hide the current user request"
    assert results[1]["merged"][0]["display_metadata"]["model_only"] is True
    assert all(row["merged"][0]["display_metadata"]["model_only"] is True for row in results[2:7])


@pytest.mark.parametrize("key", ["model_only", "host_model_only"])
@pytest.mark.parametrize("following", [
    {"role": "assistant", "content": "A human-visible reply."},
    {"role": "user", "content": [{"type": "text", "text": "A structured request."}]},
    {"role": "user", "content": {"type": "text", "text": "A structured request."}},
    {"role": "user", "content": "\x00json:undecodable content"},
], ids=["assistant", "user-list", "user-dict", "user-json-sentinel"])
def test_issue_1370_follower_controls_summary_stamp(engine, monkeypatch, key, following):
    _host(monkeypatch, key)
    _seed(engine)
    output = engine._assemble_context(None, [deepcopy(following)], persist=False)
    assert len(output) == 2 and output[0]["role"] == "user"
    assert _summary_rows(output) == [output[0]]
    assert output[1] == following
    sentinel = isinstance(following["content"], str) and following["content"].startswith("\x00json:")
    assert output[0].get("display_metadata") == (None if sentinel else {key: True})
    assert key not in output[1].get("display_metadata", {})


def _r2_survival_fit_standalone_summary(engine, monkeypatch, key, path):
    _host(monkeypatch, key)
    _seed(engine)
    tail = [{"role": "user", "content": "Older request. " + "detail " * 200},
            {"role": "assistant", "content": "Older reply. " + "detail " * 200},
            {"role": "user", "content": "Newest human request."},
            {"role": "assistant", "content": "Newest human reply."}]
    engine.ingest(tail)
    incoming = engine._assemble_context(None, tail)
    remainder = engine._generated_context_carrier_remainder(incoming[0])
    assert remainder == tail[0]["content"]
    summary = incoming[0]["content"][:engine._verified_lcm_summary_prefix_end(incoming[0]["content"])]
    budget = count_messages_tokens([{"role": "user", "content": summary}, *tail[-2:]])
    engine._ingest_cursor = len(incoming)
    engine._ingest_cursor_needs_reconcile = False
    if path == "direct":
        fitted = engine._survival_fit(incoming, incoming, 0, "recovery_attempt:test", request_cap=budget)
    else:
        monkeypatch.setattr(engine, "_survival_fit_args", lambda *a, **k: {
            "reason": "recovery_attempt:test", "request_cap": budget})
        def compress_impl(*a, **k):
            if path == "exception-compress":
                raise RuntimeError("synthetic compaction failure")
            return incoming
        monkeypatch.setattr(engine, "_compress_impl", compress_impl)
        fitted = engine.compress(incoming, current_tokens=0, force=True)
    assert fitted is not incoming and fitted[0]["content"] == summary
    assert fitted[1:] == tail[-2:]
    assert key not in fitted[0].get("display_metadata", {})
    assert all(key not in row.get("display_metadata", {}) for row in fitted[1:])
    assert "display_metadata" not in incoming[0]  # a carrier holds human text
    metadata = {"origin": "survival", key: False}
    row = {"role": "user", "content": summary, "display_metadata": metadata}
    engine._survival_summary_identity(row, summary, None, "survival_summary")
    # Identity formation no longer decides display before the final group exists.
    assert row["display_metadata"] == metadata
    assert metadata == {"origin": "survival", key: False}
    carrier = {"role": "user", "content": summary + "\n\nHuman request."}
    engine._survival_summary_identity(carrier, summary, None, "carrier")
    assert "display_metadata" not in carrier
    _host(monkeypatch, None)
    unsupported = {"role": "user", "content": summary}
    engine._survival_summary_identity(unsupported, summary, None, "survival_summary")
    assert "display_metadata" not in unsupported
    return fitted


@pytest.mark.parametrize("key", ["model_only", "host_model_only"])
@pytest.mark.parametrize("path", ["direct", "normal-compress", "exception-compress"])
def test_r2_survival_fit_stamps_standalone_summary(engine, monkeypatch, key, path):
    _r2_survival_fit_standalone_summary(engine, monkeypatch, key, path)


@pytest.mark.skipif(not os.environ.get("LCM_REAL_HERMES_SRC"), reason="no real Hermes source supplied")
@pytest.mark.parametrize("key", ["model_only", "host_model_only"])
@pytest.mark.parametrize("path", ["direct", "normal-compress", "exception-compress"])
def test_r2_survival_fit_standalone_summary_real_host(engine, monkeypatch, tmp_path, key, path):
    fitted = _r2_survival_fit_standalone_summary(engine, monkeypatch, key, path)
    merged = _real_merge(fitted, tmp_path)
    assert merged["repairs"] == 1
    visible = [row for row in merged["merged"] if not row.get("display_metadata", {}).get(key)]
    assert any("Newest human request." in row["content"] for row in visible)


@pytest.mark.parametrize("key", ["model_only", "host_model_only"])
@pytest.mark.parametrize("origin", [False, True], ids=["empty-metadata", "keep-other-metadata"])
@pytest.mark.parametrize("path", ["assistant-merge", "prepend-text", "prepend-list", "prepend-dict"])
def test_r2_human_text_merge_removes_stale_marker(monkeypatch, key, origin, path):
    _host(monkeypatch, key)  # the host-advertised key is cleared too, not only the literal one
    metadata = {key: True, **({"origin": "host"} if origin else {})}
    row = {"role": "assistant", "content": "Earlier generated context.", "display_metadata": metadata}
    if path == "assistant-merge":
        human = {"role": "assistant", "content": "Human-visible reply."}
        generated = [row]
        merged = _merge_adjacent_assistant_messages([row, human], generated_rows=generated)[0]
        assert generated == []
    else:
        row["content"] = ("Human-visible reply." if path == "prepend-text" else
                          [{"type": "text", "text": "Human-visible reply."}] if path == "prepend-list" else
                          {"type": "text", "text": "Human-visible reply."})
        merged = LCMEngine._prepend_generated_context_to_message(row, "Generated context.")
    assert "Human-visible reply." in str(merged["content"])
    assert key not in merged.get("display_metadata", {})
    assert merged.get("display_metadata") == ({"origin": "host"} if origin else None)
    assert row["display_metadata"] is metadata and metadata[key] is True
    if path == "assistant-merge":
        other = {"role": "assistant", "content": "More generated context."}
        generated = [row, other]
        merged = _merge_adjacent_assistant_messages([row, other], generated_rows=generated)[0]
        assert generated == [merged] and merged["display_metadata"][key] is True


def test_r2_over_cap_trim_tracks_generated_summary(engine, monkeypatch, caplog):
    nodes = add_summaries(engine)
    tail = [assistant_calls("call-0", "call-1"), result("call-0"), result("call-1")]
    cap = full_cap(engine, nodes, tail)
    original = LCMEngine._sanitize_active_context_messages

    def inflating(self, messages, **options):
        cleaned = original(self, messages, **options)
        if options.get("insert_missing_tool_stubs", True) and any(
            _PRESERVED_OBJECTIVE_CONTEXT_PREFIX in str(row.get("content", "")) for row in cleaned
        ):
            cleaned = [*cleaned, {"role": "user", "content": "unbudgeted " * 400}]
        return cleaned

    monkeypatch.setattr(LCMEngine, "_sanitize_active_context_messages", inflating)
    output = assemble(engine, tail, cap)
    assert OBJECTIVE not in "\n".join(str(row.get("content")) for row in output)
    assert "dropping the preserved user objective" in caplog.text
    summaries = [row for row in output if "orchard" in str(row.get("content"))]
    assert summaries and all(row.get("display_metadata") == {"model_only": True} for row in summaries)


def test_t1_compaction_leaves_mergeable_summary_visible(engine):
    messages = _history()
    before = deepcopy(messages)
    output = _compress(engine, messages)
    summaries = _summary_rows(output)
    assert summaries
    assert all("display_metadata" not in row for row in summaries)
    assert all("display_metadata" not in row for row in output if row not in summaries)
    assert messages == before


@pytest.mark.parametrize("missing", ["attribute", "module"])
def test_t2_unsupported_host_full_output_unchanged(engine, monkeypatch, missing):
    _seed(engine)
    tail = [{"role": "user", "content": "A real request."}]
    _host(monkeypatch, None)
    if missing == "module":
        monkeypatch.setitem(sys.modules, "agent.context_compressor", None)
    # This is the complete base assembly, including role and summary formatting.
    output = engine._assemble_context(None, tail, persist=False)
    assert output == [
        {"role": "user", "content": "[Recent Summary (d0, node 1)]\nEarlier work completed.\n[Expand for details: work]"},
        *tail,
    ]


@pytest.mark.parametrize("shape", ["retained-fold", "retained-structured", "user-carrier"])
def test_t3_human_carriers_stay_visible(engine, shape):
    _seed(engine)
    retained = None
    if shape.startswith("retained"):
        retained = {"role": "user", "content": "The sole user objective."}
        tail = [{"role": "assistant", "content": "A real assistant reply."}]
        if shape == "retained-structured":
            tail[0]["content"] = [{"type": "text", "text": tail[0]["content"]}]
        engine.ingest([retained, *tail])
    else:
        tail = [{"role": "user", "content": "A real earlier request."},
                {"role": "assistant", "content": "A real assistant reply."},
                {"role": "user", "content": "A real latest request."}]
        engine.ingest(tail)
    output = engine._assemble_context(None, tail, retained_user_message=retained)
    assert any(" Summary (d" in str(row["content"]) and "A real" in str(row["content"])
               for row in output)
    assert all(not row.get("display_metadata", {}).get("model_only") for row in output)


@pytest.mark.parametrize("site", ["recall", "generated-context", "anchor", "assistant-merge"])
@pytest.mark.parametrize("cap", [None, 160], ids=["uncapped", "fit-candidates"])
def test_t1_other_standalone_consumers(engine, monkeypatch, site, cap):
    tail = [{"role": "user", "content": "A real request."}]
    retained = None
    if site in {"recall", "assistant-merge"}:
        _seed(engine)
        monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
            "role": "user", "content": "<relevant-memories>Retrieved context.</relevant-memories>",
            "display_metadata": {"origin": "recall"},
        })
    elif site == "generated-context":
        _seed(engine)
        retained = {"role": "user", "content": "The sole user objective."}
        tail = [{"role": "user", "content": "Another real request."}]
    else:
        monkeypatch.setattr(engine, "_latest_user_context_anchor_candidates",
                            lambda *a: [f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nAn omitted objective."])
    if site == "assistant-merge":
        tail = [{"role": "assistant", "content": "A real assistant reply."}]
        monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
            "role": "assistant", "content": "Retrieved context."})
    measured = []
    count = engine_module.count_message_tokens
    def measure(row):
        measured.append(deepcopy(row))
        return count(row)
    monkeypatch.setattr(engine_module, "count_message_tokens", measure)
    output = engine._assemble_context(None, tail, retained_user_message=retained, assembly_cap_override=cap)
    generated = [row for row in output if not any(
        text in str(row["content"]) for text in ("A real", "Another real", "sole user"))]
    assert generated
    # Recall/anchor user rows will absorb the plain user tail in the host.
    expected_stamp = site not in {"recall", "anchor"}
    assert all(bool(row.get("display_metadata", {}).get("model_only")) == expected_stamp for row in generated)
    assert all(not row.get("display_metadata", {}).get("model_only")
               for row in output if "A real" in str(row["content"]))
    assert all(not row.get("display_metadata", {}).get("model_only") for row in measured)


def test_t1_cleanup_copy_of_standalone_recall_is_marked(engine, monkeypatch):
    _seed(engine)
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
        "role": "assistant", "content": "<think>Internal reasoning.</think>Retrieved context."})
    output = engine._assemble_context(None, [{"role": "user", "content": "A real request."}])
    recall = next(row for row in output if row["content"] == "Retrieved context.")
    assert recall["display_metadata"] == {"model_only": True}


def test_t3_same_text_host_cleanup_copy_stays_visible(engine, monkeypatch):
    _seed(engine)
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
        "role": "assistant", "content": "Retrieved context."})
    tail = [{"role": "assistant", "content": "<think>Internal reasoning.</think>Retrieved context.",
             "finish_reason": "incomplete"}, {"role": "user", "content": "A real request."}]
    output = engine._assemble_context(None, tail)
    human = next(row for row in output if row.get("finish_reason") == "incomplete")
    recall = next(row for row in output if row["content"] == "Retrieved context." and row is not human)
    assert "display_metadata" not in human
    assert recall["display_metadata"] == {"model_only": True}


def test_t4_host_display_projection_keeps_each_human_turn_once(engine):
    originals = _history()
    output = _compress(engine, originals)
    output_contents = {str(row["content"]) for row in output}
    compacted = [row for row in originals if str(row["content"]) not in output_contents]
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE messages(content TEXT, active INT, compacted INT, display_metadata TEXT)")
        for active, rows in [(0, compacted), (1, output)]:
            db.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", [
                (str(row["content"]), active, 1 - active, json.dumps(row.get("display_metadata")))
                for row in rows])
        visible = [row[0] for row in db.execute(
            "SELECT content FROM messages WHERE (active=1 OR compacted=1) "
            "AND COALESCE(json_extract(display_metadata, '$.model_only'),0)=0")]
    humans = [row["content"] for row in originals if row["role"] != "system"]
    assert all(Counter(visible)[content] == 1 for content in humans)
    # A summary that will absorb the real user row must remain display-visible.
    assert all(row["content"] in visible for row in _summary_rows(output))


def test_t5_metadata_does_not_change_identity_or_reconciliation(engine, tmp_path, monkeypatch):
    originals = _history()
    marked = _compress(engine, originals)
    # Exercise both metadata variants even when #1370 leaves this summary visible.
    _summary_rows(marked)[0]["display_metadata"] = {"model_only": True}
    assert _summary_rows(marked)[0].get("display_metadata") == {"model_only": True}
    unmarked = deepcopy(marked)
    for row in unmarked:
        row.pop("display_metadata", None)
    assert [_emission_identity(row) for row in marked] == [_emission_identity(row) for row in unmarked]
    # Two cold engines consume the same durable first-compaction proof independently.
    engine.shutdown()
    results = []
    for index, messages in enumerate([marked, unmarked]):
        path = tmp_path / f"replay-{index}.db"
        with sqlite3.connect(engine._config.database_path) as source, sqlite3.connect(path) as target:
            source.backup(target)
        replay = LCMEngine(config=LCMConfig(database_path=str(path), fresh_tail_count=4,
                           leaf_chunk_tokens=100, temporal_rollups_enabled=False),
                           hermes_home=str(tmp_path / f"home-{index}"))
        replay.on_session_start("issue-1201", platform="cli", context_length=200_000)
        try:
            output = _compress(replay, deepcopy(messages))
            results.append((replay._last_ingest_reconciliation, replay._store.get_session_count("issue-1201"),
                            [_emission_identity(row) for row in output]))
        finally:
            replay.shutdown()
    assert results[0] == results[1]


def test_t6_existing_metadata_and_host_constant(engine, monkeypatch):
    _host(monkeypatch, "host_model_only")
    _seed(engine)
    metadata = {"origin": "recall", "host_model_only": False}
    recall = {"role": "user", "content": "Retrieved context.", "display_metadata": metadata}
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: recall.copy())
    tail = [{"role": "user", "content": [{"type": "text", "text": "A real request."}],
             "display_metadata": {"origin": "human"}}]
    output = engine._assemble_context(None, tail)
    assert next(row for row in output if row["content"] == recall["content"])["display_metadata"] == {
        "origin": "recall", "host_model_only": True}
    assert output[-1]["display_metadata"] == tail[0]["display_metadata"]
    assert metadata == {"origin": "recall", "host_model_only": False}


@pytest.mark.parametrize("builder", ["summary-l1", "summary-l2", "extraction", "assertion",
                                    "host-evidence", "selective", "expansion"])
def test_t7_auxiliary_prompt_builders_never_forward_live_rows(expansion_engine, monkeypatch, tmp_path, builder):
    engine = expansion_engine
    rows = [{"role": "user", "content": "A synthetic decision to preserve.",
             "display_metadata": {"model_only": True}, "display_kind": "summary"}]
    before = deepcopy(rows)
    calls = []
    module = ModuleType("agent.auxiliary_client")
    def call_llm(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    serialized = engine._serialize_messages(rows)
    # The public serializers take rows; the downstream auxiliary seams take text/envelopes.
    if builder.startswith("summary"):
        prompt = (escalation._build_l1_prompt(serialized, 100, 0) if builder == "summary-l1"
                  else escalation._build_l2_prompt(serialized, 100))
        escalation._call_llm_for_summary(prompt, 100)
    elif builder == "extraction":
        extraction.extract_before_compaction(serialized, str(tmp_path / "notes"))
    elif builder == "assertion":
        content = rows[0]["content"]
        snapshot = SourceSnapshot(1, "issue-1201", "cli", "user", content, 1,
                                  hashlib.sha256(content.encode()).hexdigest())
        prompt = assertion_extraction.build_structured_assertion_prompt(snapshot, SimpleNamespace(query_assertions=lambda **k: []))
        assertion_extraction._call_structured_assertion_llm(prompt, "", 1)
    elif builder == "host-evidence":
        host_evidence.call_auxiliary_selector({"question": "What was decided?", "baseline_evidence": rows})
    elif builder == "selective":
        selective_compiler.call_selective_auxiliary_selector({
            "status": "selector_required", "prompt": selective_compiler._prompt(
                {"operation": "latest_state", "facet": "decision"},
                [{**rows[0], "handle": "h1", "quote": rows[0]["content"], "date": None}])})
    else:
        expand_query_tools._synthesize_expansion_answer(prompt="What was decided?", context_blocks=rows,
                                          model="", max_tokens=100, timeout=1)
    assert len(calls) == 1
    for outgoing in calls[0]:
        assert set(outgoing) == {"role", "content"}
        assert all(outgoing is not source and outgoing != source for source in rows)
    assert rows == before


def test_t8_second_compress_uses_public_row_noop_shortcut(engine, monkeypatch):
    output = _compress(engine, _history())
    assert "display_metadata" not in _summary_rows(output)[0]
    compared = []
    candidates = []
    compress_impl = engine._compress_impl
    def capture(*args, **kwargs):
        candidate = compress_impl(*args, **kwargs)
        candidates.append(candidate)
        return candidate
    monkeypatch.setattr(engine, "_compress_impl", capture)
    public_row = engine._public_compression_row
    def record(row):
        compared.append(row)
        return public_row(row)
    monkeypatch.setattr(engine, "_public_compression_row", record)
    second = _compress(engine, output)
    assert second is output
    assert len(candidates) == 1 and candidates[0] is not output
    assert len(compared) == 2 * len(output)
    assert all(public_row(left) == public_row(right) for left, right in zip(compared[::2], compared[1::2]))
    assert "display_metadata" not in _summary_rows(second)[0]
