"""Signed E2 delivery contract, using synthetic rows and real read connections."""
from copy import deepcopy
import importlib
import json
import re
import sqlite3
import time

import pytest

from hermes_lcm import tools
from hermes_lcm.config import LCMConfig
from test_recall_delivery_contract import set_arms
from test_recall_ranking_contract import NOW, corpus as corpus, recall
from tests.test_proactive_recall_injection import LCMEngine


def detector():
    return importlib.import_module("hermes_lcm.retraction")


def add(corpus, text, role="user", session="old", observed=None):
    store = corpus[0]._store
    sid = store.append(session, {"role": role, "content": text})
    store.connection.execute("UPDATE messages SET observed_at=? WHERE store_id=?", (observed, sid))
    store.connection.commit()
    return dict(kind="message_excerpt", store_id=sid, session_id=session, role=role,
                timestamp=NOW, snippet=text, content_offset=0, from_current_session=False,
                expand_hint=f"lcm_expand(store_id={sid})")


def entries(hits):
    return [{"hit": h} for h in hits]


def links(corpus, hits, excluded=()):
    return detector().find_retractions(corpus[0], entries(hits),
                                      excluded_session_ids=list(excluded), deadline=time.monotonic() + 5)


def deliver(corpus, monkeypatch, hits, mode="suppress", **kwargs):
    set_arms(monkeypatch, fts=hits, chunk=hits)
    corpus[0]._config.retraction_delivery = mode
    return recall(corpus, scope_bias=0, **kwargs)


def trace(corpus, monkeypatch, call):
    statements = []
    original = sqlite3.connect

    def connect(*args, **kwargs):
        conn = original(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    corpus[0]._store.connection.set_trace_callback(statements.append)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        result = call()
    corpus[0]._store.connection.set_trace_callback(None)
    filtered = [s for s in statements if not s.lstrip().startswith("--")
                and not re.search(r"(['\"])(main|temp)\1\s*\.", s)]
    return result, filtered


@pytest.mark.parametrize("mode", ["codeliver", "suppress"])
@pytest.mark.parametrize("later_session", [False, True])
def test_seeded_arms(corpus, monkeypatch, mode, later_session):
    old = add(corpus, "My favorite drink is saffron tea.")
    assistant = add(corpus, "I will offer saffron tea as your favorite drink.", "assistant")
    request = add(corpus, "Please forget my favorite drink saffron tea.",
                  session="later" if later_session else "old")
    baseline = json.loads(deliver(corpus, monkeypatch, [old, assistant], "off"))
    payload = json.loads(deliver(corpus, monkeypatch, [old, assistant], mode))
    if mode == "codeliver":
        assert len(payload["hits"]) == 2
        assert "annotations" not in payload["hits"][0]
        assert payload["hits"][1]["annotations"] == [f'Later, the user said: "{request["snippet"]}"']
    else:
        assert payload["total_results"] == 1
        hit = payload["hits"][0]
        assert hit["kind"] == "retraction"
        assert hit["withdrawn_store_id"] == assistant["store_id"]
        assert hit["cites"] == [request["store_id"]]
        assert hit["snippet"] == ('[Withdrawn: the user later asked to forget an earlier statement here. '
                                  f'The user said: "{request["snippet"]}"]')
        assert hit["session_id"] == request["session_id"]
        assert hit["score"] == baseline["hits"][1]["score"]
        assert hit["from_current_session"] is False
        assert hit["timestamp"]
        assert not {"store_id", "node_id", "expand_hint", "exact_ref", "event_time"} & hit.keys()


@pytest.mark.parametrize("mode", ["codeliver", "suppress"])
def test_deictic_exact_predecessor_and_intervening_assistant(corpus, monkeypatch, mode):
    unrelated = add(corpus, "An earlier unrelated conversation.")
    old = add(corpus, "The launch color is violet.")
    assistant = add(corpus, "I will use violet.", "assistant")
    add(corpus, "Forget what I just said.")
    result, reason = links(corpus, [unrelated, old, assistant])
    assert reason is None
    assert set(result[0][1]) == {1, 2}
    payload = json.loads(deliver(corpus, monkeypatch, [unrelated, old, assistant], mode))
    assert payload["hits"][0]["store_id"] == unrelated["store_id"]
    assert payload["total_results"] == (3 if mode == "codeliver" else 2)


@pytest.mark.parametrize("text", [
    "Don't forget to book saffron tea.", "I forgot my keys.",
    "Forget the weather, tell me about music.", "I no longer like saffron tea.",
    "If I ever ask you to forget saffron tea, ask why.",
    "She told her assistant to forget saffron tea.",
    'The note says "please forget saffron tea".',
    '"Please forget saffron tea."', "I am forgetting saffron tea lately.",
])
def test_classifier_rejects(text):
    assert detector().request_object(text) is None


@pytest.mark.parametrize("text,object_text", [
    ("Please forget my favorite drink saffron tea.", "my favorite drink saffron tea"),
    ("Disregard what I said about saffron tea.", "saffron tea"),
    ("Stop using my earlier statement about saffron tea.", "saffron tea"),
    ("Forget that.", ""), ("Forget what I just said.", ""),
    ("Would you mind forgetting my saffron tea preference?", "my saffron tea preference"),
    ("Would you mind disregarding my saffron tea preference?", "my saffron tea preference"),
    ("I want my saffron tea preference forgotten.", "my saffron tea preference"),
])
def test_classifier_requests(text, object_text):
    assert detector().request_object(text) == object_text


@pytest.mark.parametrize("text,object_text", [
    ("Forget what I told you about saffron tea.", "saffron tea"),
    ("Please forget what I told you.", ""),
    ("Forget what I said earlier.", ""),
    ("Forget that please.", ""),
    ("Forget earlier.", ""),
    ("Forget it now.", ""),
    ("Forget all that.", ""),
    ("Forget the deadline, my boss told me.", None),
])
def test_fix1_canonical_objects(text, object_text):
    assert detector().request_object(text) == object_text


@pytest.mark.parametrize("text", [
    "Don't forget to book X", "I forgot my keys", "Forget the weather, tell me about Y",
    "I no longer like X", "If I ever ask you to forget…", "She told her assistant to forget…",
])
def test_fix1_packet_probe10_has_no_effect(corpus, monkeypatch, text):
    old = add(corpus, "The weather and X are important.")
    add(corpus, text)
    assert links(corpus, [old]) == ([], None)
    assert deliver(corpus, monkeypatch, [old], "suppress") == deliver(corpus, monkeypatch, [old], "off")


def test_duplicate_copy_after_request_prevents_link(corpus):
    old = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Please forget my favorite drink saffron tea.")
    restated = add(corpus, "Saffron tea is my favorite drink.")
    old["duplicate_store_ids"] = [restated["store_id"]]
    assert links(corpus, [old]) == ([], None)


def test_duplicate_event_time_newest_copy_is_guard(corpus):
    old = add(corpus, "Saffron tea is my favorite drink.", observed=NOW - 100)
    copy = add(corpus, "Saffron tea is my favorite drink.", observed=NOW + 100)
    old["duplicate_store_ids"] = [copy["store_id"]]
    add(corpus, "Please forget my favorite drink saffron tea.", observed=NOW)
    assert links(corpus, [old]) == ([], None)


def test_observed_at_guard_and_normalized_time(corpus, monkeypatch):
    old = add(corpus, "Saffron tea is my favorite drink.", observed=NOW + 100)
    request = add(corpus, "Please forget my favorite drink saffron tea.", observed=NOW)
    assert links(corpus, [old]) == ([], None)
    corpus[0]._store.connection.execute("UPDATE messages SET observed_at=? WHERE store_id=?",
                                       ("2026-01-02T04:04:05+01:00", request["store_id"]))
    corpus[0]._store.connection.commit()
    hit = json.loads(deliver(corpus, monkeypatch, [old]))["hits"][0]
    assert hit["event_time"] == "2026-01-02T03:04:05Z"


def test_newest_request_wins_and_three_target_limit(corpus, monkeypatch):
    hits = [add(corpus, f"Saffron tea is my favorite drink, version {i}.") for i in range(4)]
    older = add(corpus, "Please forget my favorite drink saffron tea.")
    newest = add(corpus, "Please disregard my favorite drink saffron tea.")
    found, reason = links(corpus, hits)
    assert reason is None
    assert [(r["store_id"], indices) for r, indices in found] == [
        (newest["store_id"], [3, 2, 1]), (older["store_id"], [0]),
    ]
    payload = json.loads(deliver(corpus, monkeypatch, hits, limit=4))
    assert payload["total_results"] == 2
    assert payload["hits"][0]["cites"] == [older["store_id"]]
    assert payload["hits"][1]["cites"] == [newest["store_id"]]


@pytest.mark.parametrize("duplicate", [False, True])
def test_request_already_delivered_and_never_an_original(corpus, monkeypatch, duplicate):
    old = add(corpus, "Saffron tea is my favorite drink.")
    request = add(corpus, "Please forget my favorite drink saffron tea.")
    delivered = deepcopy(request)
    if duplicate:
        delivered["store_id"] = old["store_id"]
        delivered["duplicate_store_ids"] = [request["store_id"]]
    hits = [old, delivered]
    # Direct application isolates duplicate-delivery recognition from retrieval's deduplication.
    raw_hits = [{**h, "score": .03} for h in hits]
    result, truncated = detector().apply_retractions(raw_hits, [(dict(request, content=request["snippet"]), [0])],
                                                    mode="codeliver", response_chars=0, response_char_cap=64000)
    assert result == raw_hits and not truncated
    result, _ = detector().apply_retractions(raw_hits, [(dict(request, content=request["snippet"]), [0])],
                                            mode="suppress", response_chars=0, response_char_cap=64000)
    assert result[0]["snippet"] == "[Withdrawn: the user later asked to forget an earlier statement here.]"
    later = add(corpus, "Forget that.")
    assert later["store_id"] > request["store_id"]
    assert links(corpus, [request]) == ([], None)


def test_explicit_object_requires_overlap_not_adjacency(corpus):
    old = add(corpus, "The launch color is violet.")
    add(corpus, "Please forget my favorite drink saffron tea.")
    assert links(corpus, [old]) == ([], None)


def test_first_user_deictic_request_has_no_predecessor(corpus):
    assistant = add(corpus, "A standalone assistant message.", "assistant", session="new")
    add(corpus, "Forget that.", session="new")
    assert links(corpus, [assistant]) == ([], None)


@pytest.mark.parametrize("request_text", ["Would you mind forgetting my saffron tea preference?",
                                     "I want my saffron tea preference forgotten."])
def test_morphological_fts_variants(corpus, request_text):
    old = add(corpus, "My saffron tea preference is important.")
    add(corpus, request_text)
    found, reason = links(corpus, [old])
    assert reason is None and found[0][1] == [0]


def test_excluded_session_is_byte_identical(corpus, monkeypatch):
    old = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Please forget my favorite drink saffron tea.", session="excluded")
    args = {"exclude_session_ids": ["excluded"]}
    assert deliver(corpus, monkeypatch, [old], "suppress", **args) == deliver(corpus, monkeypatch, [old], "off", **args)


@pytest.mark.parametrize("summary_only", [False, True])
def test_no_message_hits_zero_added_sql(corpus, monkeypatch, summary_only):
    summary = dict(kind="summary", node_id="n1", store_id=1, session_id="old", snippet="tea", timestamp=NOW)
    set_arms(monkeypatch, summary=[summary] if summary_only else [])
    args = {"include": "summaries"}
    engine = corpus[0]
    engine._config.retraction_delivery = "off"
    off, off_sql = trace(corpus, monkeypatch, lambda: recall(corpus, **args))
    engine._config.retraction_delivery = "suppress"
    on, on_sql = trace(corpus, monkeypatch, lambda: recall(corpus, **args))
    assert on == off
    assert len(on_sql) - len(off_sql) == 0


@pytest.mark.parametrize("args", [
    {"detail": "answer_ready", "include_occurrence_time": True},
    {"detail": "answer_ready", "seen_refs": []},
    {"delivery_unit": "session"},
])
def test_unsupported_scopes_unchanged(corpus, monkeypatch, args):
    old = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Please forget my favorite drink saffron tea.")
    set_arms(monkeypatch, fts=[old], chunk=[old])
    tool_kwargs = {"delivery_unit": args["delivery_unit"]} if "delivery_unit" in args else {}
    call_args = {k: v for k, v in args.items() if k != "delivery_unit"}
    def call():
        return tools.lcm_recall({"query": "saffron", **call_args}, engine=corpus[0], **tool_kwargs)
    corpus[0]._config.retraction_delivery = "off"
    off, off_sql = trace(corpus, monkeypatch, call)
    corpus[0]._config.retraction_delivery = "suppress"
    on, on_sql = trace(corpus, monkeypatch, call)
    assert on == off
    assert len(on_sql) - len(off_sql) == 0


def turn(corpus):
    engine = corpus[0]
    engine._config.proactive_recall_enabled = True
    engine._config.proactive_recall_budget_tokens = 2000
    engine._newest_user_message_text = LCMEngine._newest_user_message_text
    engine._wrap_relevant_memories = LCMEngine._wrap_relevant_memories
    for counter in ("skipped", "injected", "timeout"):
        setattr(engine, f"_proactive_recall_{counter}_count", 0)
    return LCMEngine._build_proactive_recall_message(engine, [{"role": "user", "content": "saffron tea"}], "user", set())


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("path", ["tool", "turn"])
@pytest.mark.parametrize("mode", ["codeliver", "suppress"])
def test_sql_budget_and_turn_render(corpus, monkeypatch, strict, path, mode):
    old = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Please forget my favorite drink saffron tea.", observed="2026-01-02T03:04:05Z")
    set_arms(monkeypatch, fts=[old], chunk=[old])
    corpus[0]._config.recall_reference_strict = strict
    call = (lambda: recall(corpus)) if path == "tool" else (lambda: turn(corpus))
    corpus[0]._config.retraction_delivery = "off"
    off, off_sql = trace(corpus, monkeypatch, call)
    corpus[0]._config.retraction_delivery = mode
    on, on_sql = trace(corpus, monkeypatch, call)
    assert on != off
    assert len(on_sql) - len(off_sql) == 2
    if path == "turn":
        assert on is not None
        expected = 'Later, on 2026-01-02T03:04:05Z, the user said:' if mode == "codeliver" else '- [2026-01-02 03:04 UTC] [Withdrawn:'
        assert expected in on["content"]
        assert ('(expand:' in on["content"]) is (mode == "codeliver")


@pytest.mark.parametrize("reason", ["deadline", "read_error"])
def test_lookup_failure_preserves_hits_and_does_not_timeout(corpus, monkeypatch, reason):
    old = add(corpus, "Saffron tea is my favorite drink.")
    off = json.loads(deliver(corpus, monkeypatch, [old], "off"))
    original = detector().find_retractions
    def unavailable(engine, candidates, **kwargs):
        with monkeypatch.context() as patch:
            if reason == "deadline":
                patch.setattr(detector(), "time", type("Clock", (), {"monotonic": staticmethod(lambda: kwargs["deadline"] + 1)}))
            else:
                def fail(*a, **kw):
                    raise sqlite3.OperationalError("synthetic read failure")
                patch.setattr(detector(), "sqlite3", type("SQLite", (), {"connect": staticmethod(fail), "Error": sqlite3.Error}))
            return original(engine, candidates, **kwargs)
    monkeypatch.setattr(detector(), "find_retractions", unavailable)
    on = json.loads(deliver(corpus, monkeypatch, [old]))
    assert on.pop("retraction_unavailable") == reason
    assert on == off and "timeout" not in on


@pytest.mark.parametrize("mode", ["codeliver", "suppress"])
def test_size_bound_and_truncation(corpus, monkeypatch, mode):
    old = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Please forget my favorite drink saffron tea.")
    off = json.loads(deliver(corpus, monkeypatch, [old], "off"))
    cap = sum(len(json.dumps(h, ensure_ascii=False)) for h in off["hits"]) + 5
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    on = json.loads(deliver(corpus, monkeypatch, [old], mode))
    assert on["hits"] == off["hits"] and on["retraction_truncated"]
    r = dict(store_id=999, session_id="later", content="Please forget saffron tea. " + "記憶 " * 1000, timestamp=NOW)
    hits = [{**old, "store_id": i, "score": .02} for i in range(1, 8)]
    found = [(dict(r, store_id=999 + i), [i]) for i in range(7)]
    changed, truncated = detector().apply_retractions(hits, found, mode=mode, response_chars=0, response_char_cap=64000)
    added = [h.get("annotations") or h for h in changed if h != hits[changed.index(h)]]
    assert sum(len(json.dumps(a, ensure_ascii=False)) for a in added) <= 2000
    assert truncated
    assert all(len((h.get("annotations") or [h.get("snippet", "")])[0]) < 900 for h in changed)


def test_bounded_candidate_scan(corpus):
    old = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Please forget my favorite drink saffron tea.")
    for _ in range(201):
        add(corpus, "Don't forget to book a meeting.")
    assert links(corpus, [old]) == ([], None)


def test_suppression_recomputes_fts_anchor(corpus, monkeypatch):
    old = add(corpus, "Saffron tea is my favorite drink.")
    unrelated = [add(corpus, f"Unrelated design number {i}.", session=f"other-{i}") for i in range(6)]
    add(corpus, "Please forget my favorite drink saffron tea.")
    set_arms(monkeypatch, fts=[old], chunk=unrelated)
    corpus[0]._config.retraction_delivery = "suppress"
    payload = json.loads(recall(corpus, limit=4, scope_bias=0))
    assert payload["provenance"]["fts_anchor"]["fired"]
    assert payload["provenance"]["fts_anchor"]["delivered"] is False
    assert any(h["kind"] == "retraction" for h in payload["hits"])


@pytest.mark.parametrize("observed", [None, "malformed", "2026-01-02T03:04:05", 0])
def test_invalid_event_time_uses_exact_no_time_string(corpus, monkeypatch, observed):
    old = add(corpus, "Saffron tea is my favorite drink.")
    request = add(corpus, "Please forget my favorite drink saffron tea.", observed=observed)
    hit = json.loads(deliver(corpus, monkeypatch, [old], "codeliver"))["hits"][0]
    assert hit["annotations"] == [f'Later, the user said: "{request["snippet"]}"']
    hit = json.loads(deliver(corpus, monkeypatch, [old], "suppress"))["hits"][0]
    assert "event_time" not in hit


@pytest.mark.parametrize("mode", ["mark", "garbage", "off"])
def test_invalid_direct_config_is_off(corpus, monkeypatch, mode):
    old = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Please forget my favorite drink saffron tea.")
    assert deliver(corpus, monkeypatch, [old], mode) == deliver(corpus, monkeypatch, [old], "off")


@pytest.mark.parametrize("raw,expected", [("codeliver", "codeliver"), ("suppress", "suppress"),
                                           (" CoDeliver ", "codeliver"), ("mark", "off"), ("garbage", "off")])
def test_config_modes(monkeypatch, raw, expected):
    monkeypatch.setenv("LCM_RETRACTION_DELIVERY", raw)
    assert LCMConfig.from_env().retraction_delivery == expected
