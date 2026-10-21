"""C2: default-off, occurrence-bound verbatim history, never new durable turns."""
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.test_issue_488_emission_descriptors import _record
from tests.test_issue_488_emission_proof import _real_hermes_merge

HEADER = "[Earlier user messages, verbatim reference; not new turns."


@pytest.fixture
def factory(tmp_path, monkeypatch):
    _stub_summary(monkeypatch, lambda **kw: ("Deterministic summary omitting user wording.", 1))
    engines = []

    def build(name="case", budget=512, sid="S", conversation="conv"):
        config = LCMConfig(database_path=str(tmp_path / name / "lcm.db"),
                           embeddings_enabled=False, temporal_rollups_enabled=False,
                           empty_lifecycle_gc_enabled=False)
        # Deliberately works on base: missing behavior must fail, not config construction.
        config.user_carry_tokens = budget
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / name / "home"))
        engine.on_session_start(sid, platform="cli", conversation_id=conversation, context_length=128_000)
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        if not engine._stable_use_closed:
            engine.shutdown()


def _stub_summary(monkeypatch, summarize):
    # Host-contract tests can replace package module aliases; patch the executing globals.
    scope = LCMEngine._summarize_leaf_chunk_with_rescue.__globals__
    monkeypatch.setitem(scope['summarize_leaf_chunk_with_rescue'].__globals__,
                        'summarize_with_escalation', summarize)
    monkeypatch.setitem(scope, 'summarize_with_escalation', summarize)


def _seed(engine, users=("Origin: preserve the exact initial objective.", "Older constraint.", "Latest live request.")):
    rows = []
    for i, text in enumerate(users):
        rows.extend([{"role": "user", "content": text, "timestamp": 100.0 + i * 2},
                     {"role": "assistant", "content": f"Reply {i}", "timestamp": 101.0 + i * 2}])
    engine.ingest(rows)
    return rows


def _node(engine, depth=1):
    return engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, depth=depth, summary="Condensation omits the user's wording.",
        token_count=10, source_token_count=1000, source_ids=[], source_type="messages", expand_hint="details"))


def _text(view):
    return "\n".join(row.get("content", "") for row in view)


def _emit(engine, rows, *, system=False, tail=None, persist=True, cap=None):
    return engine._assemble_context({"role": "system", "content": "System"} if system else None,
                                    rows[-4:] if tail is None else tail,
                                    include_lcm_note=False, persist=persist, assembly_cap_override=cap)


def _stored(engine):
    return [(r["store_id"], r["role"], r["content"]) for r in engine._store.get_session_messages(engine._session_id)]


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("carry_only", [False, True])
def test_carry_reingest_stores_no_generated_user(factory, cold, system, carry_only):
    engine = factory()
    rows = _seed(engine)
    if not carry_only:
        _node(engine)
    emitted = _emit(engine, rows, system=system)
    assert HEADER in _text(emitted), "enabled setup must emit a nonempty packet"
    if carry_only:
        generated = next(row["content"] for row in emitted if HEADER in row.get("content", ""))
        assert engine._verified_lcm_summary_prefix(generated)[1] == []
    _record(engine, rows, emitted)
    before = _stored(engine)
    host = _real_hermes_merge(deepcopy(emitted))
    new = {"role": "user", "content": "Genuine new turn.", "timestamp": 200.0}
    if cold:
        config, home = engine._config, engine._hermes_home
        engine.shutdown()
        engine = LCMEngine(config=config, hermes_home=home)
        engine.on_session_start("S", platform="cli", conversation_id="conv", context_length=128_000)
    try:
        engine.ingest(host + [new])
        after = _stored(engine)
        assert after[:-1] == before
        assert after[-1][1:] == ("user", new["content"])
        assert all(HEADER not in text for _, _, text in after)
    finally:
        if cold:
            engine.shutdown()


def _packet(view):
    text = _text(view)
    assert HEADER in text, "enabled setup must emit a nonempty packet"
    return text[text.index(HEADER):]


def test_carry_origin_survives_d1_d2(factory):
    engine = factory()
    rows = _seed(engine)
    for depth in (1, 2):
        _node(engine, depth)
        text = _packet(_emit(engine, rows))
        assert text.count(rows[0]["content"]) == 1


def test_carry_history_priority_and_chronology(factory):
    engine = factory(budget=2048)
    rows = _seed(engine, ("Origin", "Equal text", "Middle history", "Equal text", "Latest"))
    text = _packet(_emit(engine, rows, tail=rows[-2:]))
    assert re.findall(r"\[Earlier user message, store (\d+)\]", text) == ["1", "3", "5", "7"]
    assert text.count("Equal text") == 2  # equal words are distinct occurrences
    engine._config.user_carry_tokens = 145
    text = _packet(_emit(engine, rows, tail=rows[-2:]))
    assert "[Earlier user message, store 1]" in text
    assert text.index("store 1]") < text.index("store 7]")


def test_carry_later_instruction_precedence(factory):
    engine = factory()
    rows = _seed(engine, ("Keep outputs blue.", "Correction: use green.", "Continue."))
    text = _packet(_emit(engine, rows, tail=rows[-2:]))
    assert text.index("Keep outputs blue.") < text.index("Correction: use green.")
    assert "A later user instruction overrides a conflicting earlier one" in text
    assert "the latest live user instruction takes precedence" in text


@pytest.mark.parametrize("cap", [512, 2048])
@pytest.mark.parametrize("unit", [" abc xyz ", " 漢字かな\n\t"])
def test_carry_budget_one_excerpt_and_expand(factory, cap, unit):
    engine = factory(budget=cap)
    original = " \n" + unit * 2500 + "\n  "
    rows = _seed(engine, (original, "Small older row.", "Live."))
    packet = _packet(_emit(engine, rows, tail=rows[-2:]))
    end = engine._verified_user_carry_end(packet)
    packet = packet[:end]
    assert count_message_tokens({"role": "user", "content": packet}) <= cap
    match = re.search(r"chars (\d+)-(\d+) of (\d+) omitted; recover them with lcm_expand store_id=(\d+) content_offset=(\d+)", packet)
    assert match and packet.count("[... excerpt:") == 1
    a, b, size, store_id, offset = map(int, match.groups())
    assert store_id == 1 and size == len(original) and offset == a
    assert packet.split("store 1]\n", 1)[1].startswith(original[:a])
    assert packet.endswith(original[b:] + "\n\n[End earlier user messages]")
    recovered = json.loads(engine.handle_tool_call("lcm_expand", {"store_id": 1, "content_offset": a, "max_tokens": 100000}))
    assert recovered["content"].startswith(original[a:b])
    for tiny in (1, 40):
        engine._config.user_carry_tokens = tiny
        assert HEADER not in _text(_emit(engine, rows, tail=rows[-2:]))


def _reader_tree(tmp_path, revision):
    root = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", revision], capture_output=True).returncode:
        return None  # optional historical reader; current-code assertions still run
    names = subprocess.check_output(["git", "-C", str(root), "ls-tree", "--name-only", revision], text=True).splitlines()
    names = [n for n in names if n.endswith(".py") or n in ("access_context", "access_policy", "plugin.yaml")]
    archive = subprocess.check_output(["git", "-C", str(root), "archive", revision, "--", *names])
    old = tmp_path / "reader"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter="data")
    return old


def _run_reader(old, code, *args):
    bootstrap = ("import importlib.util,sys; p=sys.argv[1]; "
                 "s=importlib.util.spec_from_file_location('hermes_lcm',p+'/__init__.py',submodule_search_locations=[p]); "
                 "sys.modules['hermes_lcm']=importlib.util.module_from_spec(s); ")
    completed = subprocess.run([sys.executable, "-B", "-c", bootstrap + code, str(old), *map(str, args)],
                               text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout)


def test_carry_zero_is_byte_identical(factory, tmp_path, monkeypatch):
    old = _reader_tree(tmp_path, "8b1626f31d7db59af390792d83ca6a21e15d4bdd")
    code = """
import json
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
e=LCMEngine(config=LCMConfig(database_path=sys.argv[2]), hermes_home=sys.argv[3])
e.on_session_start('S',platform='cli',conversation_id='conv',context_length=128000)
rows=json.loads(sys.argv[4]); e.ingest(rows)
views=[e._assemble_context(None,rows[-4:],include_lcm_note=False,persist=write,assembly_cap_override=cap,
       stub_over_cap_tool_results=overflow) for cap,write,overflow in ((None,True,False),(40,True,False),
       (40,True,True),(None,False,False))]
views.append(e._assemble_context({'role':'system','content':'System'},rows[1:2],retained_user_message=rows[0],
                               include_lcm_note=False,persist=False))
print(json.dumps(views,sort_keys=True)); e.shutdown()
"""
    engine = factory(budget=0)
    rows = _seed(engine)
    expected = json.loads((Path(__file__).parent / "fixtures" / "user_carry_off.json").read_text())
    if old is not None:
        assert _run_reader(old, code, tmp_path / "base.db", tmp_path / "base-home", json.dumps(rows)) == expected
    monkeypatch.setattr(engine._store, "load_lineage_user_rows", lambda *a, **kw: pytest.fail("carry SQL at zero"), raising=False)
    views = [_emit(engine, rows, persist=write, cap=cap) if not overflow else engine._assemble_context(
        None, rows[-4:], include_lcm_note=False, assembly_cap_override=cap, stub_over_cap_tool_results=True)
        for cap, write, overflow in ((None, True, False), (40, True, False), (40, True, True), (None, False, False))]
    views.append(engine._assemble_context({"role": "system", "content": "System"}, rows[1:2],
                 retained_user_message=rows[0], include_lcm_note=False, persist=False))
    assert json.dumps(views, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert "user_carry" not in engine.get_status()
    for value in (None, "0"):
        monkeypatch.delenv("LCM_USER_CARRY_TOKENS", raising=False)
        if value:
            monkeypatch.setenv("LCM_USER_CARRY_TOKENS", value)
        assert getattr(LCMConfig.from_env(), "user_carry_tokens", 0) == 0


def test_carry_tail_and_retained_anchor_not_duplicated(factory):
    engine = factory()
    rows = _seed(engine)
    text = _packet(_emit(engine, rows))
    assert text.count(rows[0]["content"]) == 1 and text.count(rows[2]["content"]) == 1
    sole = factory("sole")
    initial = _seed(sole, ("Only real user.",))
    sole._prepare_retained_user_anchor([{"role": "system", "content": "System"}, *initial])
    assert HEADER not in _text(sole._assemble_context({"role": "system", "content": "System"}, initial[1:],
                              retained_user_message=initial[0], include_lcm_note=False))
    second = [*initial, {"role": "user", "content": "Second genuine user.", "timestamp": 200.0}]
    sole.ingest(second)
    assert "Only real user." in _packet(_emit(sole, second, tail=second[-1:]))
    engine._pending_context_anchor_messages = rows
    objective_view = _emit(engine, rows, tail=rows[-1:])
    _packet(objective_view)
    candidates = engine._latest_user_context_anchor_candidates(objective_view, objective_view[-1:])
    assert candidates and HEADER not in candidates[0]


def test_carry_replay_rotation_disable_and_paste(factory, tmp_path, monkeypatch):
    engine = factory()
    rows = _seed(engine)
    emitted = _emit(engine, rows)
    _packet(emitted)
    _record(engine, rows, emitted)
    before = _stored(engine)
    # The actual previous GA reader consumes the same v4 descriptor with no carry-aware parser.
    old = _reader_tree(tmp_path, "fafc21b977731da04bdae374cee36288c104adeb")
    engine.shutdown()
    code = """
import json
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
e=LCMEngine(config=LCMConfig(database_path=sys.argv[2]),hermes_home=sys.argv[3])
e.on_session_start('S',platform='cli',conversation_id='conv',context_length=128000)
e.ingest(json.loads(sys.argv[4])); e.ingest(json.loads(sys.argv[4]))
print(json.dumps([(r['store_id'],r['role'],r['content']) for r in e._store.get_session_messages('S')]))
e.shutdown()
"""
    if old is not None:
        assert _run_reader(old, code, engine._config.database_path, engine._hermes_home, json.dumps(emitted)) == [list(r) for r in before]
    reopened = factory(budget=0)
    reopened.ingest(deepcopy(emitted))
    assert _stored(reopened) == before
    paste = {"role": "user", "content": emitted[0]["content"], "timestamp": 300.0}
    reopened.ingest([*deepcopy(emitted), paste])
    assert _stored(reopened)[:-1] == before and _stored(reopened)[-1][2] == paste["content"]
    assert reopened._raw_message_tokens(paste) == count_message_tokens(paste)
    leaf, _, _ = reopened._leaf_summary_input([(paste, [_stored(reopened)[-1][0]])], [paste])
    assert leaf[0]["content"] == paste["content"]
    reopened.on_session_start("unrelated", platform="cli", conversation_id="other", context_length=128000)
    reopened.ingest([paste])
    assert _stored(reopened)[-1][2] == paste["content"]  # wrong-session proof has no stripping authority
    rotated = factory("rotated")
    source = _seed(rotated)
    view = _emit(rotated, source)
    _record(rotated, source, view)
    for i in range(35):
        parent = rotated._session_id
        rotated.on_session_end(parent, source)
        rotated.on_session_start(f"child-{i}", platform="cli", conversation_id="conv", context_length=128000,
                                 boundary_reason="compression", old_session_id=parent)
        rotated.ingest(deepcopy(view))
        source = [*view, {"role": "user", "content": f"Turn after rotation {i}", "timestamp": 400.0 + 2 * i},
                  {"role": "assistant", "content": f"Reply after rotation {i}", "timestamp": 401.0 + 2 * i}]
        rotated.ingest(source)
        view = _emit(rotated, source)
        _record(rotated, source, view)
    assert rows[0]["content"] in _packet(view)
    assert len(rotated._user_carry_lineage()) == 36
    assert rotated._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == len(rows) + 70
    for budget in (0, 512):
        authored = factory(f"authored-{budget}")
        original = _seed(authored)
        quote = _emit(authored, original)[0]["content"]
        _packet([{"content": quote}])
        authored._config.user_carry_tokens = budget
        message = {"role": "user", "content": quote, "timestamp": 900.0}
        assert authored._message_replay_identity(message) == authored._message_replay_identity(message, strip_carrier=False)
        assert authored._carry_tokens_in_view([message]) == 0
        authored.ingest([*original, message, {"role": "assistant", "content": "Quoted packet reply"}])
        assert _stored(authored)[-2][2] == quote
        assert authored._raw_message_tokens(message) == count_message_tokens(message)
        seen = []
        def summarize(**kwargs):
            seen.append(kwargs["text"])
            return "Bounded summary preserving raw-source admission", 1
        monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
        authored._config.fresh_tail_count, authored._config.fresh_tail_max_tokens = 2, 0
        authored._config.leaf_chunk_tokens = count_messages_tokens([*original, message])
        authored.compress([*original, message, {"role": "assistant", "content": "Quoted packet reply"},
                          {"role": "user", "content": "Newest request"}, {"role": "assistant", "content": "Reply"}], force=True)
        assert any(quote in text for text in seen), ("authored no-proof packet must reach the summarizer verbatim", budget, seen, authored.last_compression_noop_reason, authored._config.leaf_chunk_tokens)


def test_carry_scaffold_and_scope_filter(factory):
    engine = factory(budget=2048)
    rows = _seed(engine, ("Owned origin", "[Current user objective preserved from compacted history]\nAuthored prose", "Latest"))
    text = _packet(_emit(engine, rows, tail=rows[-2:]))
    assert "Owned origin" in text and "Authored prose" in text
    engine._config.ignore_message_patterns = ["^Ignored"]
    engine._compiled_ignore_message_patterns = [SimpleNamespace(search=lambda text, timeout: re.search("^Ignored", text))]
    for content, conv in (("Ignored synthetic", "conv"), (" ", "conv"), ("Foreign secret-free fixture", "other"),
                          ("[Current user objective preserved from compacted history]\nSynthetic", "conv")):
        engine._store.append("S", {"role": "user", "content": content}, conversation_id=conv)
    later = [{"role": "user", "content": "New live", "timestamp": 200.0},
             {"role": "assistant", "content": "New reply", "timestamp": 201.0}]
    engine.ingest([*rows, *later])
    packet = _packet(_emit(engine, later, tail=later))
    assert all(word not in packet for word in ("Ignored", "Foreign", "Synthetic"))
    for sid, conversation in (("fork", "fork-conv"), ("reset", "reset-conv")):
        engine.on_session_start(sid, platform="cli", conversation_id=conversation, context_length=128000)
        new = _seed(engine, ("New scope only",))
        assert HEADER not in _text(_emit(engine, new, tail=new))
    paged = factory("paged")
    paged._store.append_batch("S", [{"role": "user", "content": " "} for _ in range(2100)], conversation_id="conv")
    source = _seed(paged)
    pre_reset = _packet(_emit(paged, source))
    assert source[0]["content"] in pre_reset
    state = getattr(paged, "_last_user_carry", None)
    paged._mark_thread_context_stateless("child")
    try:
        assert paged._plan_user_carry(source[-2:]) == ""
    finally:
        paged._clear_thread_context_stateless("child")
    assert getattr(paged, "_last_user_carry", None) == state
    paged.on_session_reset()
    assert paged._user_carry_status()["entries"] == 0
    paged.on_session_start("S", platform="cli", conversation_id="conv", context_length=128000)
    fresh = [{"role": "user", "content": "First post-reset user", "timestamp": 900.0}]
    paged.ingest(fresh)
    assert HEADER not in _text(_emit(paged, fresh, tail=fresh))
    assert paged._verified_user_carry_end(pre_reset) is None


@pytest.mark.parametrize("window", [32000, 128000])
def test_carry_prefix_budget_and_loop_accounting(factory, monkeypatch, window):
    import hermes_lcm.engine as module
    engine = factory()
    engine._config.fresh_tail_count, engine._config.fresh_tail_max_tokens = 4, 0
    engine._config.leaf_chunk_tokens = 300
    engine._set_context_length(window, source="test")
    inputs = []
    def summarize(**kwargs):
        inputs.append(kwargs["text"])
        return "Bounded summary with omitted user words. Expand for details: work", 1
    monkeypatch.setattr(module, "summarize_with_escalation", summarize)
    rows = _seed(engine, tuple(f"User {i}: " + "work item " * 150 for i in range(8)))
    emitted = engine.compress(rows, current_tokens=count_messages_tokens(rows), force=True)
    _packet(emitted)
    assert inputs and all(HEADER not in text for text in inputs)
    before = _stored(engine)
    for i in range(2):
        host = _real_hermes_merge(deepcopy(emitted))
        call = f"call-{i}"
        host.extend([{"role": "user", "content": f"Next {i}: " + "task " * 400, "timestamp": 300.0 + i * 4},
                     {"role": "assistant", "content": "Tool step", "tool_calls": [
                         {"id": call, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": call, "content": "Tool result " * 30},
                     {"role": "assistant", "content": "Latest reply", "timestamp": 303.0 + i * 4}])
        engine.ingest(host)
        emitted = engine.compress(host, current_tokens=count_messages_tokens(host), force=True)
        assert all(HEADER not in text for text in inputs)
        assert all(HEADER not in text for _, _, text in _stored(engine))
        tool = next(j for j, row in enumerate(emitted) if row.get("tool_call_id") == call)
        assert any(c["id"] == call for c in emitted[tool - 1]["tool_calls"])
    assert _stored(engine)[:len(before)] == before
    occurrences, _ = engine._replay_occurrences(emitted)
    raw = sum(engine._raw_message_tokens(m, o) for m, o in zip(emitted, occurrences))
    assert raw < count_messages_tokens(emitted) and raw > 0
    emitted = [*emitted, *[{"role": role, "content": f"Automatic debt {i}: " + "pending task " * 300}
                             for i in range(4) for role in ("user", "assistant")]]
    engine.threshold_tokens = count_messages_tokens(emitted) - 1
    calls, frontier = len(inputs), engine._last_compacted_store_id
    assert engine.should_compress_preflight(emitted)
    engine.compress(emitted, current_tokens=count_messages_tokens(emitted))
    assert len(inputs) > calls and engine._last_compacted_store_id > frontier
    for cap in (250, 600):
        bounded = _emit(engine, host, tail=host[-4:], cap=cap)
        assert count_messages_tokens(bounded) <= cap
    engine._survival_host_overhead(host, count_messages_tokens(host) + window // 3)
    pressure = _emit(engine, host, tail=host[-4:])
    assert count_messages_tokens(pressure) + window // 3 <= engine._survival_ceiling()
    missing = [{"role": "assistant", "content": "Pending", "tool_calls": [
        {"id": "missing", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}]
    with_stub = _emit(engine, host, tail=host[-2:] + missing, cap=250)
    assert count_messages_tokens(with_stub) <= 250
    assert any(row.get("tool_call_id") == "missing" for row in with_stub)


def test_carry_trial_and_proof_failure(factory, monkeypatch):
    engine = factory()
    rows = _seed(engine)
    before = list(engine._store._conn.iterdump())
    trial = _emit(engine, rows, persist=False)
    _packet(trial)
    assert list(engine._store._conn.iterdump()) == before
    assert getattr(engine, "_last_user_carry", None) is None
    final = _emit(engine, rows)
    assert trial == final
    fitted = engine._survival_fit(rows, final, 0, "carry-pressure", request_cap=count_messages_tokens(rows[-4:]))
    assert HEADER not in _text(fitted) and rows[-2]["content"] in _text(fitted)
    def assembled(_messages, **kwargs):
        result = _emit(engine, rows)
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(result), False
        engine._last_compression_status = "compacted"
        return result
    monkeypatch.setattr(engine, "_compress_impl", assembled)
    monkeypatch.setattr(engine, "_persist_compress_commit_proof", lambda *a, **kw: None)
    saved_rows = _stored(engine)
    fallback = engine.compress(rows, force=True)
    assert HEADER not in _text(fallback) and rows[-2]["content"] in _text(fallback)
    assert _stored(engine) == saved_rows
    folded = factory("fold")
    history = _seed(folded, ("Retained", "Eligible history", "Live"))
    args = ({"role": "system", "content": "System"}, history[-3:])
    options = dict(retained_user_message=history[2], include_lcm_note=False)
    _node(folded)
    _packet(folded._assemble_context(*args, **options))
    monkeypatch.setattr(folded, "_write_folded_tail_lineage", lambda *a: False)
    assert HEADER not in _text(folded._assemble_context(*args, **options))


def test_carry_status_matches_final_emission(factory):
    engine = factory()
    rows = _seed(engine)
    emitted = _emit(engine, rows)
    packet = _packet(emitted)
    packet = packet[:engine._verified_user_carry_end(packet)]
    status = engine.get_status()["user_carry"]
    assert status["delivered_tokens"] == count_message_tokens({"role": "user", "content": packet})
    assert status["entries"] == 1 and status["excerpt_store_id"] is None
    assert json.loads(engine.handle_tool_call("lcm_status", {}))["user_carry"] == status
    paste = {"role": "user", "content": packet, "timestamp": 900.0}
    engine._record_user_carry_status([paste])
    assert engine.get_status()["user_carry"]["entries"] == 0
    _emit(engine, rows, cap=20)
    assert engine.get_status()["user_carry"]["delivered_tokens"] == 0
    engine.on_session_start("other", platform="cli", conversation_id="other-conv", context_length=128000)
    assert engine.get_status()["user_carry"]["entries"] == 0
