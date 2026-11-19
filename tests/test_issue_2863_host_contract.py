"""#2863/#2872: the host contract a long tool turn on a small window depends on.

A MODEL of Hermes 3b421578's per-turn tool loop around a real LCMEngine (cut from the issue's probe v3). Every rule
cites a host line (`agent/<file>:<line>` at 3b421578); the host is not imported. Provider counts are SIMULATED:
trace rows at 0.44 tokens/char (the reporter's provider), other text at 0.30. Synthetic data only.

H1 one attempt counter for both gates, cap 3 (agent_init.py:1379-1383; conversation_loop.py:1486-1489,1742).
H2 a committed compaction (conversation_compression.py:3361-3367, 3411-3417) sets last_prompt_tokens=-1 and
   awaiting_real_usage_after_compression=True on the engine, and `_verify_compaction_cleared_threshold=True` only when
   the engine's `_last_compression_made_progress` is true (3420-3427, read at 4115-4117).
H3 the post-tool gate returns early while awaiting (turn_preflight.py:274-275); its pressure is the usage anchor
   (usage_anchor.py:108-122), else last_prompt_tokens > 0, else 0 (turn_preflight.py:283-296).
H4 the pre-API gate (turn_preflight_gate.py:60-101; turn_preflight.py:85-192): latch when a previous pre-API figure
   exists and pressure did not fall 5% (turn_context.py:498-503); a commit and a no-progress abort both re-gate.
H5 rearm (turn_usage.py:128-162): verify is read BEFORE update_from_response; attempts, latch and the last pre-API
   figure reset when attempts > 0, verify was set and 0 < prompt < threshold.
H6 verdict (conversation_compression.py:3442-3475): output == input -> no-progress abort, else committed.
"""

import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import compaction, tokens as T
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine, _LCM_SUMMARY_PART_HEADER_RE
from hermes_lcm.externalize import (extract_externalized_refs, get_large_output_storage_dir, load_externalized_payload,
                                    maybe_externalize_tool_output)
from hermes_lcm.tokens import count_messages_tokens

pytestmark = pytest.mark.issue(2863)

WINDOW = 64_000
HEX_TOK_PER_CHAR, TEXT_TOK_PER_CHAR = 0.44, 0.30
# Run 2's shape: steps after each request; S = one 1,000-char result, M = one 19,300-char trace result,
# L2 = two 26,800-char trace results (+8.5K and +23.6K provider tokens in the report).
R2_STEPS = ["S"] * 7 + ["M", "L2", "S"] + ["S", "S", "M", "L2", "S"] + ["M", "L2", "S"] * 4 + ["S"]
R2_REQUESTS = len(R2_STEPS) + 1
SIZES = {"S": (1_000,), "M": (19_300,), "L2": (26_800, 26_800), "L": (26_171, 23_562, 23_577)}
USER = {"role": "user", "content": "Fix the 6 failing unit tests in this repo. Run the suite after each change."}
SUMMARY_CHARS = 6_348


# ---------------------------------------------------------------- synthetic rows and simulated provider counts
def trace_rows(seed, chars):
    rnd = random.Random(seed)
    rows, n, total = [], 0, 0
    while total < chars:
        row = (f"trace {seed % 100:02d}.{n:04d} op=frame ts=17000{n:05d} payload="
               + "".join(rnd.choice("0123456789abcdef") for _ in range(96)))
        rows.append(row)
        total += len(row) + 1
        n += 1
    return "\n".join(rows)[:chars]


def small_text(seed, chars=1_000):
    lines, n = [], 0
    while sum(len(x) + 1 for x in lines) < chars:
        lines.append(f"FAILED tests/test_parser.py::test_case_{(seed + n) % 6} - AssertionError: expected {n} got {n + 1}")
        n += 1
    return "\n".join(lines)[:chars]


def step_rows(r, kind):
    maker = small_text if kind == "S" else trace_rows
    calls, results = [], []
    for j, n in enumerate(SIZES[kind]):
        cid = f"call_{r:03d}_{j}"
        calls.append({"id": cid, "type": "function",
                      "function": {"name": "terminal", "arguments": json.dumps({"command": f"pytest -q -k case{j}"})}})
        results.append({"role": "tool", "tool_call_id": cid, "content": maker(r * 10 + j, n)})
    return [{"role": "assistant", "content": f"Running the suite (step {r})." if kind == "S" else "",
             "tool_calls": calls}] + results


def real_msg_tokens(m):
    content = m.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content)
    hexc = other = 0
    for line in content.split("\n"):
        if line.startswith("trace ") and "payload=" in line:
            hexc += len(line) + 1
        else:
            other += len(line) + 1
    if m.get("tool_calls"):
        other += len(json.dumps(m["tool_calls"]))
    return 4 + HEX_TOK_PER_CHAR * hexc + TEXT_TOK_PER_CHAR * other


def host_rough(msgs):
    """agent/model_metadata.py:2349-2366,2567-2569 over the wire row (ASCII only here: (len+3)//4)."""
    total = 0
    for m in msgs:
        text = str({k: v for k, v in m.items() if not str(k).startswith("_") and k != "timestamp"})
        total += (len(text) + 3) // 4
    return total


def msgs_at(r, steps):
    msgs = [dict(USER)]
    for k in range(1, r):
        msgs += step_rows(k, steps[k - 1])
    return msgs


# Calibrated once from run 1 (the reporter's numbers): request 8 prompt 47,731, its post-tool pressure 49,230.
R1_STEPS = ["S"] * 5 + ["L"] + ["S", "S", "S"] + ["L"] + ["S", "L"] * 5 + ["S", "S"]
OVERHEAD = int(47_731 - sum(real_msg_tokens(m) for m in msgs_at(8, R1_STEPS)))
_P7 = int(OVERHEAD + sum(real_msg_tokens(m) for m in msgs_at(7, R1_STEPS)))
COMPLETION = int(49_230 - _P7 - host_rough([m for m in step_rows(7, R1_STEPS[6]) if m["role"] == "tool"]))
ROUGH_OVERHEAD = int(OVERHEAD / 1.2)


# ---------------------------------------------------------------- engine
def force_char_estimate(monkeypatch):
    """The install has no tokenizer: the loader finished without an encoder."""
    monkeypatch.setattr(T, "_encoder_ready", True)
    monkeypatch.setattr(T, "_encoder", None)
    monkeypatch.setattr(T, "_encoder_generation", T._encoder_generation + 1000)
    T._count_tokens_cached.cache_clear()
    assert T.token_counter_status()["counter"] == "char_estimate"


class ModelClock:
    """The engine's holds are wall-clock; a simulated request advances the clock 30 model seconds."""

    def __init__(self, monkeypatch, dt=30.0):
        self.dt, self.offset = dt, 0.0
        real = time.monotonic
        monkeypatch.setattr(time, "monotonic", lambda: real() + self.offset)

    def tick(self):
        self.offset += self.dt


def stub_summarizer(chunk, focus_topic=None, **_kw):
    body = "The agent ran the suite, read the failing parser cases and edited the timestamp check. " * 80
    text = (f"[summary of {len(chunk)} messages: tests failing in parser] " + body)[:SUMMARY_CHARS]
    return chunk, count_messages_tokens(chunk), text, 1, 1


@pytest.fixture
def engine(tmp_path, monkeypatch):
    force_char_estimate(monkeypatch)
    home = tmp_path / "hermes-home"
    home.mkdir()
    eng = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(home))
    assert str(get_large_output_storage_dir(eng._config, hermes_home=str(home), create=False)).startswith(str(tmp_path))
    eng.on_session_start("s2863", platform="cli", model="probe-model", context_length=WINDOW)
    eng.update_model("probe-model", WINDOW, provider="custom")
    monkeypatch.setattr(eng, "_summarize_leaf_chunk_with_rescue", stub_summarizer)
    try:
        yield eng
    finally:
        eng.shutdown()


# ---------------------------------------------------------------- checks after every compress (A3)
def _text(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c) if c is not None else ""


def _call_key(t):
    fn = t.get("function") if isinstance(t.get("function"), dict) else {}
    args = fn.get("arguments")
    return str(t.get("id")), str(fn.get("name")), args if isinstance(args, str) else json.dumps(args, sort_keys=True)


def _row_key(m):
    calls = tuple(sorted(_call_key(t) for t in (m.get("tool_calls") or []) if isinstance(t, dict)))
    return m.get("role"), _text(m), m.get("tool_call_id") or calls or None


def is_stub(m):
    return m.get("role") == "tool" and bool(extract_externalized_refs(m.get("content") or ""))


def _digest(m):
    return hashlib.sha256(json.dumps(_row_key(m), default=str).encode()).hexdigest()


def pairing_problems(out, newest_id):
    """Every tool result consumes exactly one matching call id of the assistant row it answers."""
    open_ids, answered, problems = Counter(), set(), []
    for i, m in enumerate(out):
        if m.get("role") == "assistant":
            open_ids, answered = Counter(tc.get("id") for tc in (m.get("tool_calls") or [])), set()
        elif m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if open_ids[cid] > 0:
                open_ids[cid] -= 1
                answered.add(cid)
            else:
                problems.append(f"{'duplicate' if cid in answered else 'orphan'} tool row {i} {cid}")
        else:
            open_ids, answered = Counter(), set()
    if newest_id and not any(m.get("role") == "tool" and m.get("tool_call_id") == newest_id for m in out):
        problems.append(f"newest result {newest_id} missing")
    return problems


def _occurrence_digest(eng, m):
    """One key for both forms of a tool result: a pointer row counts as the payload it names."""
    refs = extract_externalized_refs(m.get("content") or "") if is_stub(m) else []
    if len(refs) == 1:
        payload = (load_externalized_payload(refs[0], config=eng._config, hermes_home=eng._hermes_home) or {})
        if payload.get("content") is not None:
            return _digest(dict(m, content=payload["content"]))
    return _digest(m)


def conservation_violations(eng, out):
    """By content, with copies (copies count). A stored row is covered when any stored copy of it is under a node whose
    summary is in `out`, or a copy of it is in `out`. A duplicate: `out` holds more copies of a row than the store has
    outside those nodes. A tool result is one occurrence in either form, payload or pointer."""
    sid = eng._session_id
    texts = [_text(m) for m in out]
    joined = "\n".join(texts)
    header_ids = {int(m.group(2)) for t in texts for m in _LCM_SUMMARY_PART_HEADER_RE.finditer(t)}
    covered = set()
    for node in eng._dag.get_session_nodes(sid, limit=100_000):
        if node.node_id in header_ids or (node.summary and node.summary in joined):
            covered.update(eng._dag.source_message_ids(node.node_id, limit=1_000_000))
    rows = [(r, _occurrence_digest(eng, r)) for r in eng._store.get_session_messages(sid, limit=100_000)]
    summarised = Counter(d for r, d in rows if r["store_id"] in covered)
    outside = Counter(d for r, d in rows if r["store_id"] not in covered)
    keys = [_occurrence_digest(eng, m) for m in out]
    listed = Counter(keys)
    violations = [("missing", r["store_id"], r.get("role"), r.get("tool_call_id")) for r, d in rows
                  if r["store_id"] not in covered and d not in summarised and d not in listed]
    for digest, n in listed.items():
        if (digest in outside or digest in summarised) and n > outside[digest]:
            m = out[keys.index(digest)]
            violations.append(("duplicate", n, outside[digest], m.get("role"), m.get("tool_call_id")))
    return violations


# ---------------------------------------------------------------- the host model
class Anchor:
    def __init__(self, prompt, completion, msgs):
        self.prompt, self.completion, self.base = prompt, completion, len(msgs)
        self.prefix = copy.deepcopy(msgs)

    def tokens(self, msgs):
        """usage_anchor.py:108-122; None when stale (prefix mismatch, 87-105)."""
        if len(msgs) < self.base or msgs[:self.base] != self.prefix:
            return None
        delta = msgs[self.base:]
        if delta and delta[0].get("role") == "assistant":
            delta = delta[1:]
        return self.prompt + self.completion + (host_rough(delta) if delta else 0)


class Host:
    def __init__(self, eng, steps, clock, cap=3):
        self.eng, self.steps, self.clock, self.cap = eng, steps, clock, cap
        self.thr = int(eng.threshold_tokens)
        self.attempts, self.blocked, self.last_pre, self.anchor = 0, False, None, None
        self.verdicts, self.requests, self.passes = [], [], []

    def compress(self, msgs, pressure, req):  # H6 + H2
        eng = self.eng
        before = copy.deepcopy(msgs)
        newest = next((m.get("tool_call_id") for m in reversed(msgs) if m.get("role") == "tool"), None)
        out = eng.compress(list(msgs), current_tokens=pressure)  # conversation_compression.py:2904-2907
        made_progress = bool(getattr(eng, "_last_compression_made_progress", False))
        if getattr(eng, "_last_compress_aborted", False):
            verdict, result = "aborted_compressor", before
        elif out == before or not out:
            verdict, result = "aborted_no_progress", before
        else:
            verdict, result = "committed", list(out)
            eng.on_session_start(eng._session_id, boundary_reason="compression",
                                 old_session_id=eng._session_id, platform="cli")
            eng.last_prompt_tokens = -1
            eng.last_completion_tokens = 0
            eng.awaiting_real_usage_after_compression = True
            self.anchor = None
            if made_progress:
                eng._verify_compaction_cleared_threshold = True
        self.verdicts.append((req, verdict, made_progress))
        self.passes.append({"req": req, "pairing": pairing_problems(result, newest),
                            "conservation": conservation_violations(eng, result)})
        return result

    def pre_pressure(self, msgs):
        a = self.anchor.tokens(msgs) if self.anchor else None
        if a is not None:
            return a, True
        return host_rough(msgs) + ROUGH_OVERHEAD, False

    def run(self):
        eng, msgs = self.eng, [dict(USER)]
        eng.should_compress_preflight(list(msgs))  # turn prologue (turn_context_compaction.py:436-460)
        r = 0
        for _guard in range(500):
            if r >= len(self.steps) + 1:
                break
            P, anchored = self.pre_pressure(msgs)  # H4
            prev, self.last_pre = self.last_pre, None
            if prev is not None and P >= self.thr and not (prev > 0 and P < prev * 0.95):
                self.blocked = True
            defer = False if anchored else bool(getattr(eng, "should_defer_preflight_to_real_usage", lambda _t: False)(P))
            if len(msgs) > 1 and self.attempts < self.cap and not self.blocked and not defer and eng.should_compress(P):
                self.attempts += 1
                self.last_pre = P
                msgs = self.compress(msgs, P, r + 1)
                continue
            r += 1  # send (H5)
            self.clock.tick()
            prompt = int(OVERHEAD + sum(real_msg_tokens(m) for m in msgs))
            verify_before = bool(getattr(eng, "_verify_compaction_cleared_threshold", False))
            eng.update_from_response({"prompt_tokens": prompt, "completion_tokens": COMPLETION,
                                      "total_tokens": prompt + COMPLETION})
            self.anchor = Anchor(prompt, COMPLETION, msgs)
            if self.attempts and verify_before and 0 < prompt < self.thr:
                self.attempts, self.blocked, self.last_pre = 0, False, None
            newest = next((m for m in reversed(msgs) if m.get("role") == "tool"), None)
            self.requests.append({"req": r, "provider": prompt, "newest_stub": newest if newest and is_stub(newest)
                                  else None})
            if r > len(self.steps):
                break
            new_rows = step_rows(r, self.steps[r - 1])  # tool round + post-tool gate (H3)
            msgs = msgs + new_rows
            if getattr(eng, "awaiting_real_usage_after_compression", False):
                continue
            P2 = self.anchor.tokens(msgs)
            if P2 is None:
                P2 = max(int(eng.last_prompt_tokens or 0), 0)
            if self.attempts < self.cap and eng.should_compress(P2):
                self.attempts += 1
                msgs = self.compress(msgs, P2, r + 1)
        return self.requests


@pytest.fixture
def run2(engine, monkeypatch):
    host = Host(engine, R2_STEPS, ModelClock(monkeypatch))
    host.run()
    return engine, host


def test_run2_keeps_compacting_past_the_attempt_cap_and_stays_under_the_window(run2):
    _eng, host = run2
    assert len(host.requests) == R2_REQUESTS
    commits = [v for v in host.verdicts if v[1] == "committed"]
    # The cap is 3 per turn: a fourth commit exists only because a real response under the threshold re-armed it.
    assert len(commits) > 3, host.verdicts
    assert all(made for _req, _verdict, made in commits), host.verdicts
    assert max(x["provider"] for x in host.requests) < WINDOW, [x["provider"] for x in host.requests]


def test_run2_every_pass_conserves_rows_and_tool_pairs(run2):
    _eng, host = run2
    assert host.passes
    for p in host.passes:
        assert p["pairing"] == [], p
        assert p["conservation"] == [], p


def test_run2_pointer_names_tool_size_and_expand_call_and_expands_to_the_original(run2):
    eng, host = run2
    first = next(x for x in host.requests if x["newest_stub"])
    stub = first["newest_stub"]
    text, cid = stub["content"], stub["tool_call_id"]
    refs = extract_externalized_refs(text)
    assert len(refs) == 1
    original = next(m["content"] for r, k in enumerate(R2_STEPS, start=1) for m in step_rows(r, k)
                    if m.get("tool_call_id") == cid)
    assert "tool=terminal;" in text, text
    assert f"chars={len(original)};" in text, text
    assert f'lcm_expand(externalized_ref="{refs[0]}")' in text, text
    parts, offset = [], None
    for _ in range(50):
        args = {"externalized_ref": refs[0]}
        if offset is not None:
            args["content_offset"] = offset
        out = json.loads(eng.handle_tool_call("lcm_expand", args))
        assert "error" not in out, out
        parts.append(out.get("content", ""))
        if not out.get("has_more"):
            break
        offset = out.get("next_content_offset")
    got = "".join(parts)
    assert hashlib.sha256(got.encode()).hexdigest() == hashlib.sha256(original.encode()).hexdigest()


# ---------------------------------------------------------------- the engine's side of the contract
def test_compress_reports_progress_on_every_exit(engine, monkeypatch):
    msgs = msgs_at(12, R2_STEPS)
    out = engine.compress(list(msgs), current_tokens=60_000)
    assert out != msgs and count_messages_tokens(out) < count_messages_tokens(msgs)
    assert engine._last_compression_made_progress is True

    tiny = [dict(USER)]
    noop = engine.compress(list(tiny), current_tokens=10)
    assert noop == tiny
    assert engine._last_compression_made_progress is False

    engine.compress(list(msgs_at(12, R2_STEPS)), current_tokens=60_000)
    assert engine._last_compression_made_progress is True

    def boom(*_a, **_k):
        raise RuntimeError("inner step failed")
    monkeypatch.setattr(engine, "_compress_impl", boom)
    with pytest.raises(RuntimeError):
        engine.compress(list(tiny), current_tokens=10)
    assert engine._last_compression_made_progress is False


def _record_progress_writes(engine, monkeypatch):
    """Every write of the progress flag, with the writing thread."""
    state, writes = {"value": engine.__dict__.pop("_last_compression_made_progress", False)}, []

    def write(_self, value):
        writes.append((threading.get_ident(), value))
        state["value"] = value
    monkeypatch.setattr(type(engine), "_last_compression_made_progress",
                        property(lambda _self: state["value"], write), raising=False)
    return writes


@pytest.mark.parametrize("prior", [True, False])
def test_a_refused_compress_leaves_the_progress_flag_as_it_was(engine, monkeypatch, prior):
    # Another thread owns the engine (the one-owner guard's state): this call is refused and returns its own list.
    engine._last_compression_made_progress = prior
    writes = _record_progress_writes(engine, monkeypatch)
    refusals, listed = engine._compress_reentry_refusals, list(msgs_at(12, R2_STEPS))
    engine._compress_owner, engine._compress_owner_depth = -1, 1
    try:
        assert engine.compress(listed, current_tokens=60_000) is listed
    finally:
        engine._compress_owner, engine._compress_owner_depth = None, 0
    assert engine._compress_reentry_refusals == refusals + 1
    assert engine._last_compression_made_progress is prior and writes == []


@pytest.mark.parametrize("order", ["refused call ends inside the pass", "refused call ends after the owner's report"])
def test_a_refused_concurrent_compress_never_clears_the_owners_progress(engine, monkeypatch, order):
    # Two threads, one engine: the owner's pass makes the list smaller; a second call is refused during it.
    # The owner's host reads the flag after both calls have returned, so whichever order their exits take, it
    # must read True, and the refused call must never have written the flag.
    in_pass, release, reported, owner = threading.Event(), threading.Event(), threading.Event(), {}
    impl, made = engine._compress_impl, compaction._made_progress

    def owned_pass(*args, **kwargs):
        owner["ident"] = threading.get_ident()
        in_pass.set()
        assert release.wait(10)
        return impl(*args, **kwargs)

    def late_report(before, returned):  # a refused call that reports anything does so after the owner's report
        if threading.get_ident() != owner.get("ident"):
            assert reported.wait(10)
        return made(before, returned)
    monkeypatch.setattr(engine, "_compress_impl", owned_pass)
    if order == "refused call ends after the owner's report":
        monkeypatch.setattr(compaction, "_made_progress", late_report)
    writes = _record_progress_writes(engine, monkeypatch)
    msgs, listed = msgs_at(12, R2_STEPS), list(msgs_at(12, R2_STEPS))
    with ThreadPoolExecutor(max_workers=2) as workers:
        primary = workers.submit(engine.compress, list(msgs), 60_000)
        assert in_pass.wait(10)
        second = workers.submit(engine.compress, listed, 60_000)
        if order == "refused call ends inside the pass":
            assert second.result(timeout=10) is listed
        deadline = time.monotonic() + 10
        while engine._compress_reentry_refusals < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        out = primary.result(timeout=60)
        reported.set()
        assert second.result(timeout=10) is listed
    assert engine._compress_reentry_refusals == 1
    assert out != msgs and count_messages_tokens(out) < count_messages_tokens(msgs)
    assert engine._last_compression_made_progress is True  # the owner's host reads after both calls returned
    assert {ident for ident, _value in writes} == {owner["ident"]}, writes


def test_new_engine_reports_no_progress(engine):
    assert engine._last_compression_made_progress is False


def _arm_host_flags(eng):
    eng._verify_compaction_cleared_threshold = True
    eng.awaiting_real_usage_after_compression = True


def _host_flags(eng):
    return eng._verify_compaction_cleared_threshold, eng.awaiting_real_usage_after_compression


@pytest.mark.parametrize("usage", [
    {},
    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    {"prompt_tokens": 30_000, "completion_tokens": 100, "total_tokens": 30_100},
], ids=["usage-less", "zero-count", "real-count"])
def test_any_foreground_response_clears_both_host_flags(engine, usage):
    # context_compressor.py:2899-2901 clears both with or without a count; turn_usage.py:87-91 sends {} when the
    # provider omits usage, so a count-gated clear would leave the post-tool gate off for the rest of the turn.
    _arm_host_flags(engine)
    engine.update_from_response(usage)
    assert _host_flags(engine) == (False, False)


def test_auxiliary_response_leaves_both_host_flags(engine, monkeypatch):
    aux = ["aux-2863"]
    monkeypatch.setattr(engine, "_thread_context_session_id", lambda: aux[0])
    assert engine._thread_context_stateless()
    _arm_host_flags(engine)
    engine.update_from_response({"prompt_tokens": 900, "completion_tokens": 50, "total_tokens": 950})
    engine.update_from_response({})
    assert _host_flags(engine) == (True, True)
    aux[0] = ""
    engine.update_from_response({})
    assert _host_flags(engine) == (False, False)


def _side_engine(tmp_path, monkeypatch, kind):
    force_char_estimate(monkeypatch)
    home = tmp_path / "hermes-home"
    home.mkdir()
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), **{f"{kind}_session_patterns": ["side-*"]})
    return LCMEngine(config=config, hermes_home=str(home))


@pytest.mark.parametrize("kind", ["stateless", "ignore"])
def test_known_limit_a_side_channel_response_on_a_shared_engine_clears_the_foregrounds_host_flags(
        tmp_path, monkeypatch, kind):
    # Hermes 3b421578 gives each agent its own engine copy (agent_init.py:1779-1785), so this shape needs a host that
    # shares one engine across agents. There the side channel's response ends the foreground's post-compaction wait.
    eng = _side_engine(tmp_path, monkeypatch, kind)
    try:
        eng.on_session_start("s2863", platform="cli", model="probe-model", context_length=WINDOW)
        eng.update_from_response({"prompt_tokens": 47_000, "completion_tokens": 100, "total_tokens": 47_100})
        _arm_host_flags(eng)  # the host's post-commit state (conversation_compression.py:3411-3427)
        eng.on_session_start("side-1", platform="cron", model="probe-model", context_length=WINDOW)
        assert eng.side_channel_active and eng._bypasses_lcm_context_management()
        eng.update_from_response({"prompt_tokens": 900, "completion_tokens": 50, "total_tokens": 950})
        assert _host_flags(eng) == (False, False)
        _arm_host_flags(eng)
        eng.update_from_response({})
        assert _host_flags(eng) == (False, False)
    finally:
        eng.shutdown()


@pytest.mark.parametrize("kind", ["stateless", "ignore"])
def test_a_rebound_ignored_or_stateless_sessions_own_responses_clear_its_host_flags(tmp_path, monkeypatch, kind):
    # One agent's engine copy moves from a normal session to an ignored or stateless one, compacts through the host
    # fallback and waits for usage (conversation_compression.py:3414): its own responses end that wait.
    eng = _side_engine(tmp_path, monkeypatch, kind)
    try:
        eng.on_session_start("s2863", platform="cli", model="probe-model", context_length=WINDOW)
        eng.update_from_response({"prompt_tokens": 47_000, "completion_tokens": 100, "total_tokens": 47_100})
        eng.on_session_start("side-1", platform="cron", model="probe-model", context_length=WINDOW)
        assert eng.side_channel_active and eng._bypasses_lcm_context_management()
        for usage in ({"prompt_tokens": 900, "completion_tokens": 5, "total_tokens": 905}, {},
                      {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}):
            _arm_host_flags(eng)
            eng.update_from_response(usage)
            assert _host_flags(eng) == (False, False), usage
    finally:
        eng.shutdown()


@pytest.mark.parametrize("kind", ["stateless", "ignore"])
def test_standalone_ignored_or_stateless_session_response_clears_both_host_flags(tmp_path, monkeypatch, kind):
    # A process whose only binding is an ignored or stateless session (a cron process of its own): no foreground
    # binding exists, so it is not a side channel and its own responses end the host's post-compaction wait.
    eng = _side_engine(tmp_path, monkeypatch, kind)
    try:
        eng.on_session_start("side-1", platform="cron", model="probe-model", context_length=WINDOW)
        assert not eng.side_channel_active and eng._bypasses_lcm_context_management()
        _arm_host_flags(eng)
        eng.update_from_response({"prompt_tokens": 900, "completion_tokens": 5, "total_tokens": 905})
        assert _host_flags(eng) == (False, False)
        _arm_host_flags(eng)
        eng.update_from_response({})
        assert _host_flags(eng) == (False, False)
    finally:
        eng.shutdown()


# ---------------------------------------------------------------- the checks themselves flag what they guard
def _calls(*ids, arguments="{}"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "terminal", "arguments": arguments}} for cid in ids]}


def _result(cid, text="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": f"{text} {cid}"}


def test_pairing_problems_flags_a_result_that_reuses_a_call_id():
    assert pairing_problems([dict(USER), _calls("a", "b"), _result("a"), _result("b")], "b") == []
    problems = pairing_problems([dict(USER), _calls("a", "b"), _result("b"), _result("b")], "b")
    assert problems == ["duplicate tool row 3 b"], problems
    assert pairing_problems([dict(USER), _calls("a"), _result("a"), _result("c")], None) == ["orphan tool row 3 c"]


def test_conservation_violations_flags_a_duplicated_row_and_counts_stored_copies(engine):
    msgs = msgs_at(4, R2_STEPS)
    engine.ingest(list(msgs))
    assert conservation_violations(engine, msgs) == []
    tool_row = next(m for m in msgs if m.get("role") == "tool")
    found = conservation_violations(engine, msgs + [dict(tool_row)])
    assert [v[0] for v in found] == ["duplicate"], found
    found = conservation_violations(engine, msgs + [dict(USER)])
    assert [v[0] for v in found] == ["duplicate"], found
    # A row the store legitimately holds twice may appear twice; a third copy is a duplicate.
    again = msgs + [dict(USER)]
    engine.ingest(list(again))
    assert sum(_digest(r) == _digest(USER) for r in engine._store.get_session_messages(engine._session_id)) == 2
    assert conservation_violations(engine, again) == []
    found = conservation_violations(engine, again + [dict(USER)])
    assert [v[0] for v in found] == ["duplicate"], found
    assert [v[0] for v in conservation_violations(engine, again[:-2] + again[-1:])] == ["missing"]
    # Copies count: a row stored twice (a restart stores it again) and listed once is covered.
    assert conservation_violations(engine, msgs) == []


def _fake_engine(rows, nodes):
    """A hand-built store: `nodes` maps a node id to (its summary, the store ids it covers)."""
    store = SimpleNamespace(get_session_messages=lambda sid, limit=0: [dict(r) for r in rows])
    dag = SimpleNamespace(
        get_session_nodes=lambda sid, limit=0: [SimpleNamespace(node_id=n, summary=s) for n, (s, _) in nodes.items()],
        source_message_ids=lambda nid, limit=0: list(nodes[nid][1]))
    return SimpleNamespace(_session_id="s", _store=store, _dag=dag, _config=None, _hermes_home=None)


def test_conservation_violations_counts_a_restart_copy_as_covered():
    answer = {"role": "assistant", "content": "answer"}
    rows = [dict(USER, store_id=1), dict(answer, store_id=2), dict(USER, store_id=3)]
    # (a) the row is summarised as store id 1 and stored again as id 3 outside every node; the list holds the summary.
    summary = {"role": "user", "content": "SUMMARY-SEVEN"}
    assert conservation_violations(_fake_engine(rows, {7: ("SUMMARY-SEVEN", [1])}), [summary, dict(answer)]) == []
    # ... but the summary and a raw copy of the summarised row would hold one copy more than the store has outside.
    found = conservation_violations(_fake_engine(rows, {7: ("SUMMARY-SEVEN", [1])}), [summary, dict(USER), dict(USER)])
    assert [v[0] for v in found] == ["missing", "duplicate"], found
    # (b) the row is stored twice and listed once.
    assert conservation_violations(_fake_engine(rows, {}), [dict(USER), dict(answer)]) == []


def _pointer(eng, row):
    made = maybe_externalize_tool_output(row["content"], tool_call_id=row["tool_call_id"], session_id=eng._session_id,
                                         config=eng._config, hermes_home=eng._hermes_home, force=True,
                                         tool_name="terminal")
    return dict(row, content=made["placeholder"])


def test_conservation_violations_counts_a_tool_result_once_in_either_form(engine):
    result = _result("a", "the full output")
    msgs = [dict(USER), _calls("a"), result]
    engine.ingest(list(msgs))
    pointer = _pointer(engine, result)
    assert is_stub(pointer) and not is_stub(result)
    assert conservation_violations(engine, msgs[:2] + [pointer]) == []
    # One stored payload returned as the payload AND as its pointer: one copy too many.
    found = conservation_violations(engine, msgs + [pointer])
    assert [v[0] for v in found] == ["duplicate"] and found[0][1:3] == (2, 1), found


def test_conservation_violations_matches_two_pointers_to_a_payload_and_a_pointer(engine):
    # The same result stored twice, once as its payload and once (after a restart) as its pointer.
    result = _result("a", "the full output")
    engine.ingest([dict(USER), _calls("a"), result])
    pointer = _pointer(engine, result)
    engine.ingest([dict(USER), _calls("a"), result, pointer])
    stored = [r for r in engine._store.get_session_messages(engine._session_id) if r.get("role") == "tool"]
    assert sorted(is_stub(r) for r in stored) == [False, True], stored
    assert conservation_violations(engine, [dict(USER), _calls("a"), pointer, dict(pointer)]) == []
    found = conservation_violations(engine, [dict(USER), _calls("a"), pointer, dict(pointer), dict(result)])
    assert [v[0] for v in found] == ["duplicate"], found


def test_conservation_violations_tells_calls_apart_by_name_and_arguments(engine):
    # The same call id with different arguments, the first row returned twice: the second call is missing.
    first, second = _calls("a", arguments='{"command": "ls"}'), _calls("a", arguments='{"command": "make clean"}')
    user2 = {"role": "user", "content": "Now clean the build folder."}
    msgs = [dict(USER), first, _result("a"), user2, second, _result("a", "done")]
    engine.ingest(list(msgs))
    assert conservation_violations(engine, msgs) == []
    found = conservation_violations(engine, msgs[:4] + [dict(first)] + msgs[5:])
    assert sorted(v[0] for v in found) == ["duplicate", "missing"], found
    renamed = copy.deepcopy(second)
    renamed["tool_calls"][0]["function"]["name"] = "read_file"
    found = conservation_violations(engine, msgs[:4] + [renamed] + msgs[5:])
    assert [v[0] for v in found] == ["missing"], found


def test_summary_target_ratio_is_the_host_default():
    # agent/context_compressor.py:2801; read by idle compaction (turn_context_compaction.py:171).
    assert LCMEngine.summary_target_ratio == 0.20
