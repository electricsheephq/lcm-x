"""The same reader instructions across three evidence delivery modes."""
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .models import READER_MAX_TOKENS, harness, input_bound

# Never offered: lcm_expand_query calls another model; the rest are operator tools.
EXCLUDED_TOOLS = ("lcm_expand_query", "lcm_status", "lcm_inspect", "lcm_doctor")
INSTRUCTIONS = ("The current date is {date}. Answer concisely. Say you don't know when the history "
                "does not contain the answer. ")
EVIDENCE = "Use the supplied history evidence to answer the question."
AGENTIC = ("The conversation history is NOT in this prompt. Retrieve it with the provided tools "
           "before answering.")
# D3: token estimate for request growth is ceil(JSON characters / 4), added to the
# provider-reported prompt_tokens of the previous call. A tool round is offered only
# if a forced final after it still fits, reserving the largest tool result seen
# (at least TOOL_RESULT_RESERVE) plus the assistant's tool-call turn.
TOOL_RESULT_RESERVE, TURN_RESERVE = 8_000, 4_000
# As the Hermes host does at its iteration limit (MAX_ITERATIONS_SUMMARY_REQUEST), the forced
# final carries a user nudge; without it Opus ends a tool_choice "none" turn with reasoning only.
FINAL_NUDGE = ("The tool budget for this question is exhausted. Give your final answer to the question "
               "now from what you have retrieved, without calling any more tools.")
NUDGE = {"role": "user", "content": FINAL_NUDGE}


def tokens(value):
    return -(-len(json.dumps(value, ensure_ascii=False)) // 4)


def schemas(engine):
    # D10: the production registration under this engine's effective config.
    from hermes_lcm.engine import LCMEngine
    shim = SimpleNamespace(_config=engine._config, _disabled_tool_names=LCMEngine._disabled_tool_names)
    shim._omitted_tool_names = lambda: LCMEngine._omitted_tool_names(shim)
    return [{"type": "function", "function": s} for s in LCMEngine.get_tool_schemas(shim)
            if s["name"] not in EXCLUDED_TOOLS]


class Clock:
    """`time` proxy whose time() is this thread's question date while one is set (D12)."""
    def __init__(self):
        self.local = threading.local()

    def __getattr__(self, name):
        return getattr(time, name)

    def time(self):
        now = getattr(self.local, "now", None)
        return time.time() if now is None else now


class ClockThreading:
    """`threading` proxy: production deadline workers inherit the creating thread's clock."""
    def __getattr__(self, name):
        return getattr(threading, name)

    class Thread(threading.Thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fmp_now = getattr(CLOCK.local, "now", None)

        def run(self):
            CLOCK.local.now = self.fmp_now
            super().run()


CLOCK, INSTALL = Clock(), threading.Lock()


@contextmanager
def question_clock(date):
    # Product code is not changed: its module-level time/threading names are swapped
    # for proxies that fall back to the real clock on threads without a question.
    from hermes_lcm import store, tools
    with INSTALL:
        tools.time = store.time = CLOCK
        tools.threading = ClockThreading()
    CLOCK.local.now = datetime.fromisoformat(date).timestamp() if date else None
    try:
        yield
    finally:
        CLOCK.local.now = None


def dispatch(name, arguments, engine, offered=None):
    # D5: as the Hermes host does, failures become the tool result and the question continues.
    from hermes_lcm import tools as production
    if name not in {s["function"]["name"] for s in offered or schemas(engine)}:
        return json.dumps({"error": f"Unknown LCM tool: {name}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as exc:
        return json.dumps({"error": f"Invalid JSON arguments for {name}: {exc}"}, ensure_ascii=False)
    try:
        return getattr(production, name)(args, engine=engine)
    except Exception as exc:
        return json.dumps({"error": f"Error executing {name}: {exc}"}, ensure_ascii=False)


def single(q, engine, speakers=None, hit_label="speaker", store_text="speaker-prefix"):
    raw = dispatch("lcm_recall", {"query": q.question, "limit": 25}, engine)
    payload = json.loads(raw)
    if "error" in payload:
        raise RuntimeError("recall_failed")
    session = payload.get("provenance", {}).get("delivery", {}).get("unit") == "session"
    rows = ({h["store_id"]: {**h, "observed_at": None} for h in payload.get("hits", [])} if session else
            engine._store.get_batch([h["store_id"] for h in payload.get("hits", [])
                                    if isinstance(h.get("store_id"), int)]))
    lines = []
    for hit in payload.get("hits", []):
        row = rows.get(hit.get("store_id"))
        if row is not None:
            stamp = row.get("observed_at")
            date = datetime.fromtimestamp(stamp, timezone.utc).isoformat() if stamp else "unknown"
            if session:
                date = row.get("event_time", "unknown")
            speaker = (speakers or {}).get(str(row["store_id"])) if hit_label == "speaker" else None
            role = speaker or row["role"]
            content = row["content"]
            prefix = (speaker + (f" ({date[:10]})" if store_text == "name-date" else "") + ": ") if speaker else ""
            if store_text != "plain":
                if prefix and content.startswith(prefix):
                    content = content[len(prefix):]
                elif hit_label == "role" and getattr(q, "dataset", "") == "locomo":
                    content = content.partition(": ")[2]
            lines.append(f"[{date}] {role}: {content}")
        else:
            lines.append(f"[{hit.get('event_time', 'unknown')}] summary: {hit.get('snippet', '')}")
    return "\n".join(lines), raw


def full_context(q):
    return "\n".join(f"[{s['date']}] {t.get('speaker', t['role'])}: {t['content']}"
                     for s in sorted(q.sessions, key=lambda s: s["date"]) for t in s["turns"])


def context_length(client, reader):
    return client.context_length(reader) if hasattr(client, "context_length") else \
        client.metadata[reader]["context_length"]


def answer(q, mode, client, reader, engine=None, speakers=None, tool_budget=12, input_budget=200_000, records=None,
           server=None, hit_label="speaker", store_text="speaker-prefix"):
    records = records if records is not None else []
    if mode == "agentic" and harness(reader):
        result = harness_agentic(q, client, reader, server, records)
    else:
        with question_clock(q.question_date):
            result = _answer(q, mode, client, reader, engine, speakers, tool_budget, input_budget, records, hit_label, store_text)
    return {**result, **date_exposure(q)}


def date_exposure(q):
    return {"wall_date_visible": True, "temporal_question": q.qtype in ("temporal-reasoning", "2")}


def harness_agentic(q, client, reader, server, records):
    # The harness (Codex or Claude Code) runs its own loop over the MCP reader; its sufficiency stop
    # is its own, capped at models.AGENTIC_TOOL_CAP counted calls by the server. The trace is metadata only.
    system = INSTRUCTIONS.format(date=q.question_date) + AGENTIC
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "Question: " + q.question}]
    result = {"qid": q.qid, "qtype": q.qtype, "status": "ok", "hypothesis": "", "calls": records,
              "trace": [], "tool_calls": 0, "forced_final": None, "finish_reason": None}
    try:
        message = client.agentic(reader, messages, records, server)
    finally:
        logs = [Path(c["mcp_log"]) for c in records if c.get("mcp_log")]
        if logs and logs[-1].exists():
            result["trace"] = [json.loads(line) for line in logs[-1].read_text().splitlines()]
        result["tool_calls"] = sum(bool(t.get("counted")) for t in result["trace"])
        result["forced_final"] = "tool_calls" if any(t.get("counted") is False for t in result["trace"]) else None
    result["finish_reason"] = records[-1].get("finish_reason") if records else None
    result["hypothesis"] = message["content"]
    return result


def _answer(q, mode, client, reader, engine, speakers, tool_budget, input_budget, records, hit_label, store_text):
    trace = []
    result = {"qid": q.qid, "qtype": q.qtype, "status": "ok", "hypothesis": "", "calls": records,
              "trace": trace, "tool_calls": 0, "forced_final": None, "finish_reason": None}
    system = INSTRUCTIONS.format(date=q.question_date) + (AGENTIC if mode == "agentic" else EVIDENCE)
    user = "Question: " + q.question
    if mode == "single":
        evidence, raw = single(q, engine, speakers, hit_label, store_text)
        trace.append({"tool": "lcm_recall", "result": raw})
        user = "History evidence:\n" + evidence + "\n" + user
    elif mode == "full_context":
        user = "History evidence:\n" + full_context(q) + "\n" + user
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if mode == "full_context" and input_bound(messages) > context_length(client, reader) - READER_MAX_TOKENS:
        return {**result, "status": "skipped", "reason": "does not fit"}
    offered = schemas(engine) if mode == "agentic" else None
    used, last, sent, reserve = 0, 0, 0, TOOL_RESULT_RESERVE
    if getattr(getattr(engine, "_config", None), "recall_delivery_unit", "hits") == "session":
        reserve = 16_000

    def pending():  # estimated prompt tokens of the next call
        return last + tokens(messages[sent:]) if sent else tokens([messages, offered])

    while True:
        estimate = pending()
        tools_on = bool(offered) and result["tool_calls"] < tool_budget and \
            used + 2 * estimate + reserve + TURN_RESERVE <= input_budget
        if offered and not tools_on:
            result["forced_final"] = "tool_calls" if result["tool_calls"] >= tool_budget else "input_tokens"
            messages.append(dict(NUDGE))
            estimate = pending()
        if used + estimate > input_budget:
            raise RuntimeError("input_budget_exceeded")
        # D6: the forced final keeps the tool definitions and disables them with tool_choice.
        message = client.call(reader, messages, records, tools=offered, max_tokens=READER_MAX_TOKENS,
                              tool_choice="auto" if tools_on else "none")
        ok = [c for c in records if c.get("status") == "ok"]
        used = sum(c["input_tokens"] if c["input_tokens"] is not None else input_bound(messages, offered) for c in ok)
        last, sent = ok[-1]["input_tokens"] if ok[-1]["input_tokens"] is not None else input_bound(messages, offered), len(messages)
        result["finish_reason"] = ok[-1].get("finish_reason")
        if used > input_budget:  # admission is an estimate; the reported usage is the budget
            return {**result, "status": "budget_exceeded", "reason": "reported input tokens over budget",
                    "input_tokens": used}
        calls = message.get("tool_calls") or []
        if not calls:
            if not message.get("content"):
                raise RuntimeError("empty_reader_answer")
            result["hypothesis"] = message["content"]
            return result
        if not tools_on:
            raise RuntimeError("reader_violated_tool_budget")
        messages.append(message)  # Preserve reasoning_details for reasoning providers.
        for call in calls:
            fn = call["function"]
            allowed = result["tool_calls"] < tool_budget
            raw = (dispatch(fn["name"], fn["arguments"], engine, offered) if allowed
                   else json.dumps({"error": "tool budget exhausted; answer now"}))
            reply = {"role": "tool", "tool_call_id": call["id"], "content": raw}
            if allowed and used + last + tokens(messages[sent:] + [reply, NUDGE]) > input_budget:
                allowed, raw = False, json.dumps({"error": "input token budget exhausted; answer now"})
                reply["content"] = raw
            # A discarded result releases its reservation: only what the reader receives is reserved.
            reserve = max(reserve, tokens(raw))
            trace.append({"tool": fn["name"], "arguments": fn["arguments"], "result": raw})
            messages.append(reply)
            result["tool_calls"] += int(allowed)
