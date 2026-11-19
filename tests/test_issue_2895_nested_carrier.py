"""#2895: a user row carried behind the summary head never leaves the returned list uncounted.

A pass whose leaf takes only stored rows the list does not carry keeps the incoming carrier ("summary\\n\\nrow")
as the first tail row. Assembly wrapped it as the user row of a NEW carrier ("new summary\\n\\nold summary\\n\\n
row"); when the survival fit then cut that head it peeled one level, read the remainder (still a carrier) as
generated and dropped the row without counting or naming it. Synthetic data only: a host loop with the #904
stand-ins (count_tokens = len // 4, no host estimator, a content-keyed summary) on small windows.
Assembly is byte-identical to main off the carrier path.
"""

import hashlib
import json
import logging
import os
import random
import re
import sqlite3

import pytest

import hermes_lcm.config as lcm_config
import hermes_lcm.engine as engine_module
import hermes_lcm.host_uid_emit as host_uid_emit
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.emission_projection import _emission_identity, _finalize_emission_descriptors
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens

_HEADER = re.compile(r"Summary \(d(\d+), node (\d+)\)")
_WORDS = ("amber", "birch", "cedar", "delta", "elm", "fjord")
# The two minimal sessions (12K window; 32K window with 15K host overhead), as (kind, ...) events.
_CASE_12K = dict(ctx=12000, seed=1, overhead=0, events=[
    ["turn", [[2, 80], [2, 80]], 20, 300, False], ["turn", [], 60, 80, False],
    ["turn", [[2, 300], [1, 900]], 600, 300, False], ["turn", [], 600, 300, False],
    ["turn", [], 600, 20, True], ["turn", [[2, 300]], 200, 800, False], ["turn", [], 60, 800, False],
    ["turn", [], 200, 80, True], ["turn", [], 60, 80, False], ["turn", [], 200, 80, False],
    ["turn", [], 20, 800, False], ["turn", [[1, 900], [3, 300]], 600, 80, False]])
_CASE_32K = dict(ctx=32000, seed=6, overhead=15000, events=[
    ["k4", "user", 20], ["turn", [[1, 80]], 60, 20, False], ["turn", [], 200, 300, False],
    ["turn", [[1, 80]], 600, 800, True], ["turn", [], 600, 800, False], ["turn", [[2, 300]], 600, 800, False],
    ["turn", [], 600, 800, True], ["turn", [[2, 300]], 60, 800, False], ["turn", [], 600, 80, False],
    ["turn", [[1, 80]], 20, 20, False], ["turn", [], 600, 20, False], ["turn", [[3, 900]], 20, 800, False],
    ["turn", [], 60, 300, True], ["turn", [[1, 900]], 600, 20, False], ["turn", [[2, 300]], 600, 800, False],
    ["turn", [[3, 900]], 600, 800, False], ["turn", [], 200, 300, False]])


@pytest.fixture
def stand_ins(monkeypatch):
    """The #904 stand-ins and a record of every survival-fit plan; returns the plan record."""
    for key in list(os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: len(text) // 4 if text else 0)

    def summary(text, source_tokens=0, token_budget=0, depth=0, *args, **kwargs):
        digest = hashlib.sha256(str(text).encode()).hexdigest()[:12]
        return f"Earlier turns {digest} depth {depth}.\nExpand for details about: turns", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summary)
    plans = []
    apply_fit = LCMEngine._apply_survival_fit

    def record(self, result, plan, *args, **kwargs):
        fitted = apply_fit(self, result, plan, *args, **kwargs)
        if fitted is plan.fitted:
            plans.append((plan.count, set(int(x) for x in plan.ids or ())))
        return fitted

    monkeypatch.setattr(LCMEngine, "_apply_survival_fit", record)
    return plans


def _set_mode(monkeypatch, mode):
    """``unset``: no LCM_HOST_MESSAGE_UID, a host that is not uid-capable (the CI shape); ``on``: carry mode on a
    uid-capable host."""
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    if mode == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", mode == "on")
    monkeypatch.setattr(lcm_config, "_host_message_uid_mode_warned", False, raising=False)


def _real_row_text(engine, message):
    """The text behind a verified carrier at any nesting depth, else None (the oracle's own peel)."""
    remainder = engine._generated_context_carrier_remainder(message)
    while remainder is not None:
        inner = engine._generated_context_carrier_remainder({"role": "user", "content": remainder})
        if inner is None:
            return remainder
        remainder = inner
    return None


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _rendered_nodes(engine, messages):
    """Match the DAG's rendered parts byte for byte, including parts after an objective anchor."""
    rendered = set()
    for message in messages:
        for key in ("content", "api_content"):
            content = message.get(key)
            if not isinstance(content, str):
                continue
            for _, node_id in _HEADER.findall(content):
                node = engine._dag.get_node(int(node_id))
                if node is not None and engine_module._summary_part_text(node) in content:
                    rendered.add(node.node_id)
    return rendered


class Session:
    """A host loop on the real engine: the threshold gate, compress, and the returned list handed back."""

    def __init__(self, tmp_path, monkeypatch, plans, *, ctx, seed, mode="unset", overhead=0, state_db=False,
                 system=False, tag="case", host="simple", assembly_cap=0):
        _set_mode(monkeypatch, mode)
        home = tmp_path / tag
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        if state_db:  # the host state.db beside lcm.db: a lineage root, so host uids bind (test_host_uid_shadow)
            conn = sqlite3.connect(home / "state.db")
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT, "
                         "source TEXT, model_config TEXT)")
            conn.execute("INSERT INTO sessions VALUES (?, NULL, NULL, NULL, NULL)", (tag,))
            conn.commit()
            conn.close()
        self.engine = LCMEngine(config=LCMConfig(
            database_path=str(home / "lcm.db"), embeddings_enabled=False, temporal_rollups_enabled=False,
            max_assembly_tokens=assembly_cap,
        ), hermes_home=str(home))
        self.engine.on_session_start(tag, conversation_id="conv-" + tag, context_length=ctx)
        self.plans, self.rng, self.tag, self.overhead = plans, random.Random(seed), tag, overhead
        self.stamp, self.host = mode == "on", host
        self.msgs = [{"role": "system", "content": "You are a helpful assistant."}] if system else []
        self.turn_no = self.uid_n = 0
        self.passes = []  # (before, result, applied fit plans)
        self.on_pass = None

    def text(self, kind, n):
        words = " ".join(self.rng.choice(_WORDS) for _ in range(n))
        return f"[{kind}-{self.turn_no:03d}-{self.rng.randrange(10**6):06d}] {words}"

    def row(self, **row):
        if self.stamp:  # a uid-capable host stamps every row it persists
            self.uid_n += 1
            row["message_uid"] = hashlib.sha256(f"{self.tag}-{self.uid_n}".encode()).hexdigest()[:32]
        return row

    def host_call(self):
        engine = self.engine
        if self.host == "simple":  # the minimal sessions' loop: the threshold gate only
            sent = count_messages_tokens(self.msgs) + self.overhead
            engine.update_from_response({"prompt_tokens": sent, "completion_tokens": 10, "total_tokens": sent + 10})
            if engine.should_compress(sent):
                self.one_pass(sent)
            return
        # tests/harness host_call: ingest, the threshold then the preflight gate, the breaker gate, one overflow retry
        engine.ingest(self.msgs)
        sent = count_messages_tokens(self.msgs) + self.overhead
        compressed = engine.should_compress(sent)
        if not compressed:
            info = getattr(engine, "should_compress_info", None)
            compressed = bool(not (callable(info) and info(sent)[1]) and engine.should_compress_preflight(self.msgs))
        if compressed and type(engine)._automatic_compression_blocked(engine):
            compressed = False
        if compressed:
            self.one_pass(sent)
            sent = count_messages_tokens(self.msgs) + self.overhead
        if engine.context_length > 0 and sent > engine.context_length:
            self.one_pass(sent, bypass_cooldown=True)

    def one_pass(self, sent, **kwargs):
        before = self.msgs
        del self.plans[:]
        result = self.engine.compress(before, current_tokens=sent, **kwargs)
        self.passes.append((before, result, list(self.plans)))
        if self.on_pass is not None:
            self.on_pass(self, before, result, list(self.plans))
        self.msgs = result

    def turn(self, steps=(), user_words=60, reply_words=60, no_reply=False):
        self.turn_no += 1
        self.msgs.append(self.row(role="user", content=self.text("U", user_words)))
        self.host_call()
        for k, (n, words) in enumerate(steps):
            ids = [f"c{self.turn_no}s{k}x{j}" for j in range(n)]
            self.msgs.append(self.row(role="assistant", content="", tool_calls=[{
                "id": i, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": i})},
            } for i in ids]))
            for i in ids:
                self.msgs.append(self.row(role="tool", tool_call_id=i, name="lookup", content=self.text("R" + i, words)))
            self.host_call()
        if not no_reply:
            self.msgs.append(self.row(role="assistant", content=self.text("A", reply_words)))
        self.engine.ingest(self.msgs)
        self.engine.note_turn_complete()

    def replay(self, events, max_passes=None):
        for event in events:
            if max_passes is not None and len(self.passes) >= max_passes:
                return
            kind = event[0]
            if kind == "turn":
                self.turn([tuple(step) for step in event[1]], *event[2:])
            elif kind == "k1":  # a host row ingested, then absent from the next list
                self.msgs.append(self.row(role="user", content=self.text("GONE", event[1])))
                self.engine.ingest(self.msgs)
                self.msgs = self.msgs[:-1]
            elif kind == "k2":  # a delivery row the engine ingests at turn end that the list never carries
                self.engine.ingest(self.msgs + [{"role": "assistant", "content": self.text("DELIVERY", event[1])}])
                self.engine.note_turn_complete()
            elif kind == "k4":  # a row another writer appends to the store
                self.engine._store.append(self.engine._session_id, {"role": event[1], "content": self.text(
                    "OOB", event[2])}, conversation_id=self.engine._conversation_id)

    def close(self):
        self.engine.shutdown()


def _carried_row_fate(session, before, result, plans, caplog_lines):
    """``(carried text, fate)`` for the row the incoming head carried, else None. ``kept``: in a returned row (as a
    row or as carried text), or a stored copy under a node whose text the list renders; ``named``: an applied
    fit named the exact stored row; ``uncounted``: neither. A count alone cannot identify the row."""
    engine = session.engine
    head = next((m for m in before if m.get("role") != "system"), {})
    carried = _real_row_text(engine, head)
    if carried is None:
        return None
    for message in result:
        if message.get("role") != "user":
            continue
        for key in ("content", "api_content"):
            field = message.get(key)
            if field == carried or _real_row_text(engine, {"role": "user", "content": field}) == carried:
                return carried, "kept"
    stored = engine._store.get_session_messages(engine._session_id, limit=1_000_000)
    copies = {int(r["store_id"]) for r in stored if _text(r.get("content")) == carried}
    if not copies:
        return carried, "unstored"  # carried before any ingest stored it (a host that compresses before ingest)
    nodes = {int(n.node_id): n for n in engine._dag.get_session_nodes(engine._session_id)}

    def cover(node_id):
        node = nodes.get(node_id)
        if node is None:
            return set()
        if node.source_type == "messages":
            return {int(x) for x in node.source_ids}
        return set().union(*[cover(int(child)) for child in node.source_ids] or [set()])

    rendered = _rendered_nodes(engine, result)
    if copies & set().union(*[cover(n) for n in rendered] or [set()]):
        return carried, "kept"
    applied = any("survival fit applied" in line for line in caplog_lines)
    if applied and any(copies & ids for _, ids in plans):
        return carried, "named"
    return carried, "uncounted"


def _nested_head(engine, result):
    head = next((m for m in result if m.get("role") != "system"), {})
    remainder = engine._generated_context_carrier_remainder(head)
    return remainder is not None and engine._generated_context_carrier_remainder(
        {"role": "user", "content": remainder}) is not None


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def lines():
    handler = _Lines()
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def _checked(session, lines, failures, stats, allowed=("kept", "named")):
    """Install the per-pass checks: the property (A, F) and a returned head that is never nested (B)."""
    def check(s, before, result, plans):
        engine = s.engine
        fate = _carried_row_fate(s, before, result, plans, lines.lines)
        if fate is not None:
            stats["carrier_in"] += 1
            stats[fate[1]] = stats.get(fate[1], 0) + 1
            if fate[1] not in allowed:
                failures.append(f"pass={len(s.passes)} carried row {fate[1]}: {fate[0][:40]!r}")
        if _nested_head(engine, result):
            failures.append(f"pass={len(s.passes)} nested head returned")
        rows = [m for m in result if m.get("role") != "system"]  # a carrier after the head is generated to the fit
        stats["late_carriers"] += sum(engine._generated_context_carrier_remainder(m) is not None for m in rows[1:])
        del lines.lines[:]
    session.on_pass = check


# -- A: the two minimal sessions -------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["unset", "on"])
@pytest.mark.parametrize("case", [_CASE_12K, _CASE_32K], ids=["12k", "32k-o15000"])
def test_minimal_session_keeps_every_carried_row(tmp_path, monkeypatch, stand_ins, lines, case, mode):
    session = Session(tmp_path, monkeypatch, stand_ins, ctx=case["ctx"], seed=case["seed"], mode=mode,
                      overhead=case["overhead"])
    failures, stats = [], {"carrier_in": 0, "late_carriers": 0}
    _checked(session, lines, failures, stats)
    try:
        session.replay(case["events"])
    finally:
        session.close()
    assert stats["carrier_in"] > 0  # the regime was reached
    assert not failures, "\n".join(failures)


# -- B: the head is never nested ---------------------------------------------------------------------------------

@pytest.mark.parametrize("case", [_CASE_12K, _CASE_32K], ids=["12k", "32k-o15000"])
def test_head_holds_one_summary_prefix_and_one_row(tmp_path, monkeypatch, stand_ins, case):
    assembled = []
    emit = LCMEngine._emit_summary_carrier

    def record(self, summary_message, leading_msg, result, *args, **kwargs):
        carried_tail, tail_selected = emit(self, summary_message, leading_msg, result, *args, **kwargs)
        # the head as assembly forms it, before the survival fit can split it again
        assembled.append([dict(result[-1])] if result else [])
        assembled[-1].extend(dict(m) for m in tail_selected[:1])
        return carried_tail, tail_selected

    monkeypatch.setattr(LCMEngine, "_emit_summary_carrier", record)
    session = Session(tmp_path, monkeypatch, stand_ins, ctx=case["ctx"], seed=case["seed"],
                      overhead=case["overhead"])
    try:
        session.replay(case["events"])
        engine = session.engine
        carried = 0
        for number, (before, result, _plans) in enumerate(session.passes, start=1):
            carried += engine._generated_context_carrier_remainder(before[0]) is not None
            assert not _nested_head(engine, result), f"pass {number} nests the head"
            for message in result:  # no returned row holds a summary prefix behind another
                assert not (engine._generated_context_carrier_remainder(message) is not None
                            and _real_row_text(engine, message) != engine._generated_context_carrier_remainder(
                                message)), f"pass {number} returns a nested carrier"
        assert carried > 0
        assert assembled
        for number, rows in enumerate(assembled, start=1):
            for message in rows:
                assert _real_row_text(engine, message) == engine._generated_context_carrier_remainder(message), (
                    f"assembly {number} nests a carrier")
    finally:
        session.close()


# -- C: identity -------------------------------------------------------------------------------------------------

def _host_keys(message):
    return {k: message[k] for k in ("role", "content", "tool_calls", "tool_call_id", "name") if k in message}


@pytest.mark.parametrize("case", [_CASE_12K, _CASE_32K], ids=["12k", "32k-o15000"])
def test_unwrapped_row_keeps_its_uid_and_store_row(tmp_path, monkeypatch, stand_ins, case):
    """With real uid bindings: the unwrapped row gets back the carrier's single proven host uid, the new head
    absorbs it with one head uid of its own, and a reload (host keys only) maps the row to the stored row the
    incoming carrier mapped to. In these sessions the hidden-only guard then returns the incoming list unchanged,
    so the head is checked as assembly forms it."""
    unwrapped, checked = [], []
    unwrap = getattr(LCMEngine, "_unwrap_incoming_carrier", None)
    emit = LCMEngine._emit_summary_carrier

    def record_unwrap(self, row):
        out = unwrap(self, row)
        if out is not row:
            mapped = (self._get_store_ids_for_messages([row]), self._get_store_ids_for_messages([_host_keys(out)]))
            unwrapped.append((row, out, mapped))
        return out

    def record_emit(self, summary_message, leading_msg, result, *args, **kwargs):
        carried_tail, tail_selected = emit(self, summary_message, leading_msg, result, *args, **kwargs)
        for row, out, mapped in unwrapped:
            if carried_tail is not out:
                continue
            # the carrier's single bound host uid, proven as the survival fit's split proves it
            uids = [u for u in [row.get("message_uid"), *(row.get("_absorbed_message_uids") or [])]
                    if isinstance(u, str)]
            hosts = self._host_uid_host_uids(uids) - self._host_uid_engine_uids(uids)
            assert "_absorbed_message_uids" not in out and "api_content" not in out
            if len(hosts) != 1:
                assert "message_uid" not in out  # no proven host uid: the row carries none of the carrier's
                continue
            host_uid = next(iter(hosts))
            assert out.get("message_uid") == host_uid
            head = result[-1]
            assert self._generated_context_carrier_remainder(head) == out["content"]
            checked.append((head, host_uid, mapped))
        return carried_tail, tail_selected

    if unwrap is not None:
        monkeypatch.setattr(LCMEngine, "_unwrap_incoming_carrier", record_unwrap)
    monkeypatch.setattr(LCMEngine, "_emit_summary_carrier", record_emit)
    session = Session(tmp_path, monkeypatch, stand_ins, ctx=case["ctx"], seed=case["seed"], mode="on",
                      overhead=case["overhead"], state_db=True)
    try:
        session.replay(case["events"])
        engine = session.engine
        stored = engine._store.get_session_messages(engine._session_id, limit=1_000_000)
        assert checked, "no pass unwrapped an incoming carrier with a bound host uid into a new head"
        for head, host_uid, (mapped_in, mapped_out) in checked:
            # the head's uids were minted after assembly formed it: its own uid, plus the row's absorbed
            assert isinstance(head.get("message_uid"), str) and head["message_uid"] != host_uid
            assert head.get("_absorbed_message_uids") == [host_uid]
            row = engine._generated_context_carrier_remainder(head)
            rows = [int(r["store_id"]) for r in stored if r.get("content") == row]
            assert mapped_out and mapped_out == mapped_in and mapped_out[0] in rows
    finally:
        session.close()


# -- D: the survival-fit backstop --------------------------------------------------------------------------------

@pytest.mark.parametrize("mapped", [True, False])
def test_survival_cut_counts_row_behind_nested_carrier(tmp_path, monkeypatch, stand_ins, mapped):
    case = _CASE_12K
    session = Session(tmp_path, monkeypatch, stand_ins, ctx=case["ctx"], seed=case["seed"])
    try:
        session.replay(case["events"])
        engine = session.engine
        nodes = engine._dag.get_session_nodes(engine._session_id)
        assert len(nodes) >= 2
        outer, inner = (engine_module._summary_part_text(node) for node in nodes[-2:])
        rows = [{"role": "user", "content": "[U-900] the carried row"},
                {"role": "assistant", "content": "[A-900] " + "elm " * 200},
                {"role": "user", "content": "[U-901] the next row"},
                {"role": "assistant", "content": "[A-901] short reply"}]
        for row in rows:
            engine._store.append(engine._session_id, row, conversation_id=engine._conversation_id)
        stored = {r["content"]: int(r["store_id"]) for r in
                  engine._store.get_session_messages(engine._session_id, limit=1_000_000)}
        nested = {"role": "user", "content": f"{outer}\n\n{inner}\n\n{rows[0]['content']}"}
        assert _real_row_text(engine, nested) == rows[0]["content"]
        result = [nested, *(dict(row) for row in rows[1:])]
        store_ids = {id(m): stored[m["content"]] for m in result[1:]}
        if mapped:
            store_ids[id(nested)] = stored[rows[0]["content"]]
        budget = engine._survival_measure(result) - 100
        cut = engine._survival_cut(result, 1, budget, True, "test", store_ids, True)
        assert cut is not None
        fitted, count, ids, _projected, notice = cut
        assert not any(rows[0]["content"] in _text(m.get("content")) for m in fitted)
        assert count == 2  # the carried row and its reply
        assert (stored[rows[0]["content"]] in ids) is mapped
        assert ids == sorted(stored[c] for c in (rows[0]["content"], rows[1]["content"]) if mapped or c != rows[0][
            "content"])
    finally:
        session.close()


# -- E: no change when the first tail row is not a carrier -------------------------------------------------------

def _neutralised(monkeypatch):
    """Main's assembly and fit: no unwrap of an incoming carrier, a split that peels one level."""
    monkeypatch.setattr(LCMEngine, "_unwrap_incoming_carrier", lambda self, row: row, raising=False)
    monkeypatch.setattr(LCMEngine, "_carried_user_text", LCMEngine._generated_context_carrier_remainder,
                        raising=False)


def _five_passes(tmp_path, monkeypatch, plans, seed, system, tag):
    session = Session(tmp_path, monkeypatch, plans, ctx=12000, seed=seed, system=system, tag=tag)
    try:
        session.replay(_random_events(random.Random(seed), 60), max_passes=5)
        return session, [(json.dumps(b, sort_keys=True), json.dumps(r, sort_keys=True)) for b, r, _ in session.passes]
    finally:
        session.close()


@pytest.mark.parametrize("system", [True, False], ids=["system-led", "no-system"])
def test_lists_unchanged_without_an_incoming_carrier(tmp_path, monkeypatch, stand_ins, system):
    """Assembly is byte-identical to main off the carrier path, with or without a leading system row."""
    changed = []
    for seed in range(20):
        fixed_session, fixed = _five_passes(tmp_path, monkeypatch, stand_ins, seed, system, f"fix{seed}")
        with monkeypatch.context() as patch:
            _neutralised(patch)
            main_session, main = _five_passes(tmp_path, monkeypatch, stand_ins, seed, system, f"main{seed}")
        assert len(main) >= 1
        for number, ((before_main, result_main), (before_fix, result_fix)) in enumerate(zip(main, fixed), 1):
            assert before_main == before_fix  # compared up to the first divergence only: the sessions fork there
            if result_main != result_fix:
                engine, before = main_session.engine, json.loads(before_main)
                head = next((m for m in before if m.get("role") != "system"), {})
                assert engine._generated_context_carrier_remainder(head) is not None, (seed, number)
                changed.append((seed, number))
                break
    # the passes that differ all started from an incoming carrier; on main they nest the head (or keep it whole)
    print(f"changed passes (seed, pass): {changed}")


# -- F: a randomised walk ----------------------------------------------------------------------------------------

def _random_events(rng, turns):
    """walk.random_events of the verifier's probe, without /steer rows or no-reply turns (no two plain user rows
    in a row, which a host would merge)."""
    events = []
    for _ in range(turns):
        r = rng.random()
        if r < 0.07:
            events.append(["k1", rng.choice((20, 60, 200))])
        elif r < 0.14:
            events.append(["k2", rng.choice((20, 60, 200))])
        elif r < 0.19:
            pass  # a /steer row in the probe
        elif r < 0.26:
            events.append(["k4", rng.choice(("user", "user", "assistant")), rng.choice((20, 60, 200))])
        steps = []
        if rng.random() < 0.45:
            steps = [[rng.choice((1, 2, 2, 3)), rng.choice((20, 80, 300, 900))] for _ in range(rng.choice((1, 1, 2)))]
        events.append(["turn", steps, rng.choice((20, 60, 200, 600)), rng.choice((20, 80, 300, 800)), False])
        rng.random()  # the probe's no-reply draw
    return events


_WALKS = [(12000, 0), (16000, 0), (32000, 15000), (24000, 0)]


@pytest.mark.parametrize("chunk", range(10))
def test_random_walks_leave_no_carried_row_uncounted(tmp_path, monkeypatch, stand_ins, lines, chunk):
    failures, stats = [], {"carrier_in": 0, "late_carriers": 0}
    for offset in range(20):
        seed = 1000 + chunk * 20 + offset
        ctx, overhead = _WALKS[seed % len(_WALKS)]
        session = Session(tmp_path, monkeypatch, stand_ins, ctx=ctx, seed=seed, overhead=overhead,
                          mode="on" if seed % 2 else "unset", tag=f"walk{seed}", host="harness")
        # A row no ingest stored is outside this issue's storage coverage.
        _checked(session, lines, failures, stats, allowed=("kept", "named", "unstored"))
        try:
            session.replay(_random_events(random.Random(seed), 30))
        finally:
            session.close()
        failures[:] = [f if f.startswith("seed=") else f"seed={seed} {f}" for f in failures]
    print(f"chunk {chunk}: {stats}")
    assert stats["carrier_in"] > 0
    assert not failures, "\n".join(failures[:5])


@pytest.fixture
def carrier_session(tmp_path, monkeypatch, stand_ins, request):
    mode = getattr(request, "param", "unset")
    session = Session(tmp_path, monkeypatch, stand_ins, ctx=12000, seed=1, mode=mode, state_db=mode == "on")
    try:
        yield session
    finally:
        session.close()


def _leaf(engine, label, source_ids=()):
    node_id = engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, summary=label.ljust(156, "."), token_count=39,
        source_ids=list(source_ids), expand_hint="turns",
    ))
    return engine._dag.get_node(node_id)


def _summary(nodes):
    return "\n\n---\n\n".join(engine_module._summary_part_text(node) for node in nodes)


def _tail(session):
    return [session.row(role=role, content=label + (" elm" * 30)) for role, label in (
        ("user", "carried question"), ("assistant", "answer"),
        ("user", "next question"), ("assistant", "next answer"),
    )]


@pytest.mark.parametrize("cap", [*range(400, 1501, 50), None], ids=lambda cap: "survival" if cap is None else str(cap))
def test_carrier_unwrap_precedes_prefix_budget(carrier_session, cap):
    """Four real tail rows leave the same summary room with or without the incoming prefix."""
    session = carrier_session
    engine = session.engine
    rows = [{"role": role, "content": content} for role, content in (
        ("user", "[U-1] the carried user row " + "amber " * 10),
        ("assistant", "[A-1] " + "birch " * 120), ("user", "[U-2] next"), ("assistant", "[A-2] ok"),
    )]
    nodes = [engine._dag.get_node(engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, summary=f"Node{i} " + "qz " * 40, token_count=40,
        expand_hint=f"Expand for details about: topic{i}", created_at=1 + i,
    ))) for i in range(8)]
    assert all(count_message_tokens({"role": "user", "content": _summary([node])}) == 56 for node in nodes)
    incoming = {**rows[0], "content": f"{_summary(nodes)}\n\n{rows[0]['content']}"}
    if cap is None:
        engine.context_length = 600
        assert engine._effective_assembly_token_cap() is None
    budget = cap if cap is not None else engine._survival_ceiling()

    plain = engine._assemble_context(None, rows, assembly_cap_override=cap, persist=False)
    result = engine._assemble_context(None, [incoming, *rows[1:]], assembly_cap_override=cap, persist=False)
    expected, rendered = _rendered_nodes(engine, plain[:1]), _rendered_nodes(engine, result[:1])
    incoming_ids = {node.node_id for node in nodes}
    print(f"R3 cap={cap} budget={budget} rendered={sorted(rendered)} plain={sorted(expected)} "
          f"dropped={sorted(incoming_ids - rendered)}")
    assert expected and len(plain) == 4
    assert engine._generated_context_carrier_remainder(result[0]) == rows[0]["content"]
    assert rendered == expected
    assert count_messages_tokens(result) <= budget
    selected = [node for node in nodes if node.node_id in rendered]
    for node in nodes:
        if node.node_id not in rendered:
            assert count_message_tokens({"role": "user", "content": _summary([*selected, node])}) > (
                budget - count_messages_tokens(rows))
            assert engine._dag.get_node(node.node_id) == node


def test_carrier_keeps_original_prefix_when_no_replacement_fits(carrier_session):
    engine = carrier_session.engine
    rows = _tail(carrier_session)
    child = _leaf(engine, "old leaf")
    incoming = {**rows[0], "content": f"{_summary([child])}\n\n{rows[0]['content']}"}
    parent_id = engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, depth=1, summary="longer parent " * 100,
        token_count=350, source_ids=[child.node_id], source_type="nodes", expand_hint="turns",
    ))
    tail = [incoming, *rows[1:]]
    tight = engine._assemble_context(None, tail, assembly_cap_override=300, persist=False)
    assert tight[0]["content"] == incoming["content"]
    wide = engine._assemble_context(None, tail, assembly_cap_override=600, persist=False)
    assert _rendered_nodes(engine, wide[:1]) == {parent_id}
    assert engine._dag.get_node(parent_id).source_ids == [child.node_id]
    assert engine._generated_context_carrier_remainder(wide[0]) == rows[0]["content"]


def test_capped_walk_preserves_carrier_nodes_at_pass_29(tmp_path, monkeypatch, stand_ins):
    """The reviewer's seed 8014 at 32K with a 3000-token assembly cap."""
    session = Session(tmp_path, monkeypatch, stand_ins, ctx=32000, seed=8014, assembly_cap=3000)
    engine, rng = session.engine, session.rng
    captured = []

    def capture(_, before, result, plans):
        if len(session.passes) == 29:
            captured.append((list(before), list(result)))

    session.on_pass = capture
    unwrap = getattr(engine, "_unwrap_incoming_carrier", None)
    if unwrap is not None:
        # Replay the reviewed late unwrap through pass 28 so every candidate sees the same input at pass 29.
        emit = engine._emit_summary_carrier

        def unwrap_at_pass_29(row):
            return unwrap(row) if len(session.passes) >= 28 else row

        def late_unwrap(summary_message, leading_msg, result, tail_selected, *args):
            if len(session.passes) < 28 and summary_message is not None and tail_selected:
                tail_selected = [unwrap(tail_selected[0]), *tail_selected[1:]]
            return emit(summary_message, leading_msg, result, tail_selected, *args)

        monkeypatch.setattr(engine, "_unwrap_incoming_carrier", unwrap_at_pass_29)
        monkeypatch.setattr(engine, "_emit_summary_carrier", late_unwrap)

    def summary(text, source_tokens=0, token_budget=0, depth=0, *args, **kwargs):
        digest = hashlib.sha256(str(text).encode()).hexdigest()[:10]
        return f"Synopsis {digest} at depth {depth}: " + "notes " * 12 + "\nExpand for details about: turns", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summary)

    def row(role, kind, size, **fields):
        marker = f"<{kind}-{session.turn_no:03d}-{rng.randrange(10**7):07d}> "
        words = " ".join(rng.choice(("oak", "pine", "rowan", "yew", "ash", "larch", "hazel")) for _ in range(size))
        return {"role": role, "content": marker + words, **fields}

    def model_call():
        engine.ingest(session.msgs)
        sent = count_messages_tokens(session.msgs)
        engine.update_from_response({"prompt_tokens": sent, "completion_tokens": 10, "total_tokens": sent + 10})
        if engine.should_compress(sent):
            session.one_pass(sent)
            sent = count_messages_tokens(session.msgs)
        if sent > engine.context_length:
            session.one_pass(sent, bypass_cooldown=True)

    try:
        for session.turn_no in range(1, 31):
            draw = rng.random()
            if draw < 0.08:
                engine.ingest([*session.msgs, row("user", "GONE", rng.choice((20, 80, 300)))])
            elif draw < 0.16:
                engine.ingest([*session.msgs, row("assistant", "DLV", rng.choice((20, 80, 300)))])
                engine.note_turn_complete()
            elif draw < 0.24:
                engine._store.append(engine._session_id, row(
                    rng.choice(("user", "assistant")), "OOB", rng.choice((20, 80, 300))),
                    conversation_id=engine._conversation_id)
            session.msgs.append(row("user", "U", rng.choice((15, 50, 150, 500))))
            model_call()
            if rng.random() < 0.45:
                for step in range(rng.choice((1, 1, 2))):
                    ids = [f"t{session.turn_no}x{step}y{i}" for i in range(rng.choice((1, 2, 3)))]
                    session.msgs.append({"role": "assistant", "content": "", "tool_calls": [
                        {"id": call_id, "type": "function",
                         "function": {"name": "probe", "arguments": json.dumps({"q": call_id})}}
                        for call_id in ids]})
                    for call_id in ids:
                        session.msgs.append(row("tool", "R", rng.choice((20, 80, 300, 900)),
                                                tool_call_id=call_id, name="probe"))
                    model_call()
            session.msgs.append(row("assistant", "A", rng.choice((15, 60, 200, 700))))
            engine.ingest(session.msgs)
            engine.note_turn_complete()
            if len(session.passes) >= 29:
                break

        (before, result), = captured
        _, incoming = engine._verified_lcm_summary_prefix(before[0]["content"])
        rendered = _rendered_nodes(engine, result)
        covered = set(rendered)
        pending = list(rendered)
        while pending:
            node = engine._dag.get_node(pending.pop())
            if node.source_type == "nodes":
                children = set(node.source_ids) - covered
                covered.update(children)
                pending.extend(children)
        missing = set(incoming) - covered
        room = 3000 - count_messages_tokens(result)
        print(f"R3 walk seed=8014 pass=29 incoming={incoming} rendered={sorted(rendered)} "
              f"missing={sorted(missing)} room={room}")
        assert {26, 32} <= set(incoming)
        for node_id in missing:
            node = engine._dag.get_node(node_id)
            assert node is not None
            assert room < count_message_tokens({"role": "user", "content": _summary([node])}), node_id
    finally:
        session.close()


def test_system_led_merging_host_does_not_nest(carrier_session):
    """A system-led host that merges adjacent user rows exercises the carrier path."""
    session = carrier_session
    engine = session.engine
    system = {"role": "system", "content": "Keep the system instructions."}
    rows = _tail(session)

    def host_merge(messages):
        merged = []
        for row in messages:
            if merged and row["role"] == merged[-1]["role"] == "user":
                merged[-1] = {**merged[-1], "content": f"{merged[-1]['content']}\n\n{row['content']}"}
            else:
                merged.append(row)
        return merged

    old = _leaf(engine, "first leaf")
    incoming = host_merge(engine._assemble_context(system, rows, include_lcm_note=False, persist=False))
    assert incoming[0] == system
    assert engine._generated_context_carrier_remainder(incoming[1]) == rows[0]["content"]
    new = _leaf(engine, "hidden row leaf")
    result = host_merge(engine._assemble_context(system, incoming[1:], include_lcm_note=False, persist=False))
    assert result[0] == system
    assert _rendered_nodes(engine, result[1:2]) == {old.node_id, new.node_id}
    assert not _nested_head(engine, result)
    assert engine._generated_context_carrier_remainder(result[1]) == rows[0]["content"]


@pytest.mark.parametrize("carrier_session", ["unset", "on"], indirect=True, ids=["emit-off", "emit-on"])
@pytest.mark.parametrize("depth", [1, 2], ids=["single", "double"])
def test_ingest_after_unwrap_keeps_exact_store_row(carrier_session, monkeypatch, depth):
    session = carrier_session
    engine = session.engine
    rows = _tail(session)
    engine.ingest(rows)
    before = engine._get_store_ids_for_messages(rows)
    assert len(before) == len(rows)
    nodes = [_leaf(engine, f"prior leaf {i}") for i in range(depth)]
    span = "".join(f"{_summary([node])}\n\n" for node in nodes)
    carrier = {"role": "user", "content": span + rows[0]["content"]}
    if session.stamp:
        carrier["_absorbed_message_uids"] = [rows[0]["message_uid"]]
    incoming = [carrier, *rows[1:]]
    # A previous round's proof can bind the entire nested span to the real occurrence.
    binding = engine._emission_binding()
    emissions = _finalize_emission_descriptors(incoming, [{
        "row": carrier, "kind": "carrier", "span": span,
        "retained_source": {"store_id": before[0]}, "full_identity": _emission_identity(carrier),
    }], binding)
    engine._last_emission_descriptors = {"version": 4, **binding, "emissions": emissions}
    assert engine._get_store_ids_for_messages(incoming) == before
    _leaf(engine, "new hidden leaf")

    carried = []
    emit = engine._emit_summary_carrier

    def record(*args, **kwargs):
        row, tail = emit(*args, **kwargs)
        carried.append(row)
        return row, tail

    monkeypatch.setattr(engine, "_emit_summary_carrier", record)
    result = engine._assemble_context(None, incoming)
    unwrapped, = carried
    assert unwrapped["content"] == rows[0]["content"]
    assert unwrapped.get("message_uid") == rows[0].get("message_uid")
    assert engine._get_store_ids_for_messages([unwrapped]) == before[:1]

    # Publish the returned list's occurrence proof as compress() does, then force reconciliation on replay.
    engine._frontier_holder().adopt(len(result))
    returned, emissions, reason = engine._record_compress_commit_proof(incoming, result)
    assert reason is None and emissions[0]["retained_source"] == {"store_id": before[0]}
    assert engine._get_store_ids_for_messages(returned) == before
    stored = engine._store.get_session_messages(engine._session_id, limit=100)
    engine._frontier_holder().invalidate()
    engine.ingest(returned)
    assert engine._store.get_session_messages(engine._session_id, limit=100) == stored


def test_carrier_preserves_pasted_summary_identity(carrier_session):
    session = carrier_session
    engine = session.engine
    outer, inner = [_leaf(engine, label) for label in ("outer leaf", "pasted leaf")]
    pasted = f"{_summary([inner])}\n\nOK"
    rows = [{"role": "user", "content": pasted}, *_tail(session)[1:]]
    other_id = engine._store.append(engine._session_id, {"role": "user", "content": "OK"},
                                    conversation_id=engine._conversation_id)
    pasted_id = None
    for row in rows:
        store_id = engine._store.append(engine._session_id, row, conversation_id=engine._conversation_id)
        if row is rows[0]:
            pasted_id = store_id
    assert pasted_id != other_id
    incoming = {"role": "user", "content": f"{_summary([outer])}\n\n{pasted}"}
    assert engine._get_store_ids_for_messages([incoming]) == [pasted_id]

    result = engine._assemble_context(None, [incoming, *rows[1:]])
    candidate = engine._pending_emission_candidates[0]
    remainder = engine._generated_context_carrier_remainder(result[0])
    print(f"pasted store_id={pasted_id}, OK store_id={other_id}, emitted source={candidate['retained_source']}, "
          f"remainder={'pasted' if remainder == pasted else 'OK' if remainder == 'OK' else 'nested'}")
    assert candidate["retained_source"] == {"store_id": pasted_id}
    assert remainder == pasted


def test_carried_row_oracle_requires_exact_evidence(carrier_session):
    session = carrier_session
    engine = session.engine
    content = "Exact carried question."
    user = {"role": "user", "content": content}
    store_id = engine._store.append(engine._session_id, user, conversation_id=engine._conversation_id)
    node = _leaf(engine, "unrelated")
    covering = _leaf(engine, "covers carried question", [store_id])
    carrier = {"role": "user", "content": f"{_summary([node])}\n\n{content}"}
    decoy = {"role": "user", "content": f"Quoting {content} {_summary([covering])[:-1]}"}
    applied = ["LCM survival fit applied"]
    assert _carried_row_fate(session, [carrier], [decoy], [], []) == (content, "uncounted")
    assert _carried_row_fate(session, [carrier], [], [(2, {store_id + 1})], applied) == (content, "uncounted")
    for row in (user, carrier, {"role": "user", "content": _summary([covering])}):
        assert _carried_row_fate(session, [carrier], [row], [], []) == (content, "kept")
    assert _carried_row_fate(session, [carrier], [], [(1, {store_id})], applied) == (content, "named")
