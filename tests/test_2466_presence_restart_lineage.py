"""#2466 review (#2463 threads): a restart with incomplete lineage. A tool pair is stored in the parent; the child's
predecessor record was never written and there is no state.db; after a restart the child carries the parent's rows
and then sees a prefix edit (class A V1). The carried pair is stored exactly as often as on main (38e9cc19 and
78a9d065 both store it twice: main's own cold-bind copy, #2463/#1674), so the presence check adds no copy; the only
difference from main is the moved call, stored once (main loses it)."""
import json
from collections import Counter
import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 40
def _u(t, ts): return {"role": "user", "content": t + PAD, "timestamp": float(ts)}
def _a(t, ts): return {"role": "assistant", "content": t + PAD, "timestamp": float(ts)}
def _call(i, ts): return {"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [{"id": i, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x"}'}}]}
def _res(i, t, ts): return {"role": "tool", "content": t, "tool_call_id": i, "tool_name": "read_file", "timestamp": float(ts)}


def _engine(tmp_path, session):
    e = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path / "home"))
    e.on_session_start(session, platform="cli", context_length=1_000_000)
    return e


def _all(e):
    rows = Counter()
    for sid in ("P", "C"):
        for r in e._store.get_session_messages(sid):
            tc = r.get("tool_calls") or []
            tc = json.loads(tc) if isinstance(tc, str) else tc
            rows[(sid, r.get("role"), str(r.get("content") or "")[:24], r.get("tool_call_id") or "", ",".join(sorted(str(c.get("id")) for c in tc)))] += 1
    return rows


@pytest.mark.parametrize("own_first", [True, False])
@pytest.mark.parametrize("pad_rows", [0, 80])
def test_restart_incomplete_lineage(tmp_path, own_first, pad_rows):
    parent = [_u("P1 first", 100), _a("A1 reply", 101), _call("call_old", 102), _res("call_old", "old out", 103), _a("A2 after", 104)]
    pad = [r for i in range(pad_rows) for r in (_u(f"E{i} more", 105 + i), _a(f"E{i} reply", 105.5 + i))]
    e = _engine(tmp_path, "P")
    e.ingest(parent + pad)
    e.shutdown()
    e = _engine(tmp_path, "C")  # restart; the C -> P predecessor record was never written
    try:
        carried = parent + pad
        hist = [*carried, _u("P14 in-flight", 900), _a("A14 reply", 901), _call("call_c", 902), _res("call_c", "c out", 903), _a("A15 after", 904)]
        e.ingest(hist)
        slot = len(carried)
        a, b = (1000, 900) if own_first else (900, 1000)
        start = [*hist, _u("P60 new prompt", a)]
        end = list(hist)
        end[slot] = _u("P60 new prompt", b)
        end += [_call("call_new", 1001), _res("call_new", "new out", 1002), _a("A60 after", 1003)]
        for v in (start, end, end):
            e.ingest(v)
        rows = _all(e)
        old = (sum(c for k, c in rows.items() if k[4] == "call_old"), sum(c for k, c in rows.items() if k[3] == "call_old"))
        new = (sum(c for k, c in rows.items() if k[4] == "call_new"), sum(c for k, c in rows.items() if k[3] == "call_new"))
        assert old == (2, 2), old  # main's count on 38e9cc19 and 78a9d065 (its cold-bind copy, #2463)
        assert new == (1, 1), new  # class A: the moved call and its result, once each
    finally:
        e.shutdown()
