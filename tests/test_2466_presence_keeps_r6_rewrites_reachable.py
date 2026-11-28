"""loss_vs_main probe: an R6 in-place rewrite of an unstamped user row that main stores (version rewind) while
the PR's earlier cursor turns it into a positional replay. Self-contained helpers (main lacks the PR test file)."""
from __future__ import annotations


import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-probe-r6"
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path, **overrides):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False, **overrides)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _u(text, ts=None):
    row = {"role": "user", "content": text + PAD}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _a(text):
    return {"role": "assistant", "content": text + PAD}


def _count(engine, role, content):
    return sum(1 for r in engine._store.get_session_messages(SESSION)
               if r.get("role") == role and (r.get("content") or "") == content)


@pytest.mark.parametrize("presence", ["predecessor", "none", "summary"])
@pytest.mark.parametrize("replace_reply", [True, False])
def test_r6_rewrite_between_pr_cursor_and_main_cursor_is_stored(tmp_path, presence, replace_reply):
    engine = _engine(tmp_path)
    try:
        target = "continue with the plan" + PAD
        if presence == "predecessor":
            engine._store.append_batch("P", [_u("continue with the plan", 50)])
            engine._record_rotation_predecessor(SESSION, "P")
        view = [_u("U0 first", 1000), _a("R0 reply"), _u("U1 second", 1001), _a("R1 reply"),
                _u("X steer note"), _a("RX reply"), _u("U2 third", 1002), _a("R2 reply")]
        engine.ingest(view)
        x = view[4]
        # Host turn end: an earlier unstamped reply is replaced (PR-unaccounted, before X), X (the same host
        # object LCM stored) is rewritten in place (R6), and a later stamped user is replaced (main-missed).
        if replace_reply:
            view[1] = _a("R0 a replaced reply")
        x["content"] = (target if presence != "summary" else "CONTEXT SUMMARY: edited note" + PAD)
        view[6] = _u("U2 a replaced third prompt", 1003)
        engine.ingest(view)
        out = {"x_new": _count(engine, "user", x["content"]),
               "u2_new": _count(engine, "user", view[6]["content"]),
               "r0_new": _count(engine, "assistant", view[1]["content"])}
        print("PROBE", presence, replace_reply, out, "cursor", engine._ingest_cursor)
        assert out["u2_new"] == 1
        assert out["x_new"] == 1, out
    finally:
        engine.shutdown()
