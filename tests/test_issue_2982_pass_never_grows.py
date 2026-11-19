"""#2982 reproduction: a compaction pass whose leaves are mostly hidden stored rows (or copies) returns a list LARGER
than its input.

Shape (the reference agent's): every tool output in the host list is already the host's salvage placeholder
("[Old tool output cleared to save context space]"), while the store holds it in full. Older turns the host no longer
shows (hidden stored rows) are interleaved with shown turns. A leaf takes its hidden rows from the store at full size
(store-complete, store_complete.py), and they fill the leaf budget (leaf_chunk_tokens), so it consumes only a few shown
rows: short user asks, tool calls and placeholders that free few list tokens. Its summary is sized to the leaf target
(leaf_target_min_tokens = leaf_target_max_tokens = 2,000 by default), whatever the leaf frees. In the copy-heavy shape
the store also holds a re-ingest copy (byte-equal) of every shown tool call and result; the copies are read but free
no list row.

Size by the HOST's estimator: agent.model_metadata.estimate_messages_tokens_rough when the host stub on the path
carries it (the bench host), else a vendored copy of its ASCII formula (ceil(len(str(wire shadow)) / 4) a row, with
the persistence-only fields dropped). The plugin's count (tokens.count_messages_tokens) is reported too.
Bar (#2982): by the host's estimator, a pass never returns a list larger than its input. Red on main until fixed.
"""

from __future__ import annotations

import pytest

import hermes_lcm.escalation as escalation
from hermes_lcm.compaction import CompactionMixin
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens, count_tokens

pytestmark = pytest.mark.issue(2982)

CLEARED = "[Old tool output cleared to save context space]"
QUIET = {"embeddings_enabled": False, "temporal_rollups_enabled": False, "large_output_externalization_enabled": False,
         "large_output_active_replay_stubbing_enabled": False}
PERSISTENCE_ONLY = {"timestamp", "display_kind", "display_metadata", "_row_id", "_submit_row_session_id",
                    "message_uid", "_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid"}

try:
    from agent.model_metadata import estimate_messages_tokens_rough as _host_rough
    HOST_ESTIMATOR = "agent.model_metadata.estimate_messages_tokens_rough"
except ImportError:  # the repo's host stub carries only the ContextEngine contract
    _host_rough = None
    HOST_ESTIMATOR = "vendored estimate_messages_tokens_rough (ASCII formula)"


def host_estimate(messages) -> int:
    if _host_rough is not None:
        return int(_host_rough(messages))
    return sum((len(str({k: v for k, v in m.items() if k not in PERSISTENCE_ONLY})) + 3) // 4 for m in messages)


def _words(tokens: int, word: str) -> str:
    return " ".join(f"{word}{i % 97}" for i in range(int(tokens * 0.8)))


def _turn(i: int, user_tokens: int, output_tokens: int) -> list[dict]:
    call = {"id": f"call-{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    return [{"role": "user", "content": f"[U{i}] " + _words(user_tokens, f"ask{i}x")},
            {"role": "assistant", "content": f"reading {i}", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"call-{i}", "tool_name": "read_file",
             "content": f"[OUT{i}] " + _words(output_tokens, f"out{i}y")}]


SUMMARY_BUDGETS: list[int] = []


def _summary(max_tokens: int) -> str:
    """A stand-in summary at 90% of the target the plugin asks for (its output limit is twice the target: 4,000 for
    the default 2,000-token leaf target), as a real summariser of a full stored source writes."""
    SUMMARY_BUDGETS.append(max_tokens)
    return "Earlier work.\n" + _words(int(max_tokens // 2 * 0.9), "fact") + "\nExpand for details about: work"


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    """An engine on a 1M window (the reference agent's). ``context_threshold`` lowers the compaction threshold so the
    pressure the host reports can match the list's own size."""
    SUMMARY_BUDGETS.clear()
    monkeypatch.setattr(escalation, "_invoke_summary_llm",
                        lambda prompt, max_tokens, model="", timeout=None, reasoning_effort="": _summary(max_tokens))
    made = []

    def make(**overrides):
        instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=4, **QUIET,
                                              **overrides), hermes_home=str(tmp_path))
        instance.on_session_start("S", platform="cli", context_length=1_000_000, conversation_id="conv")
        made.append(instance)
        return instance

    yield make
    for instance in made:
        instance.shutdown()


@pytest.fixture
def leaves(monkeypatch):
    """Per published leaf: the shown list rows it consumed (their host estimate and plugin count), the stored rows it
    read that the list does not show, and its summary's host estimate (as a user row) and plugin count."""
    class Found(list):
        pre_condensation_host: list

    found, advance = Found(), CompactionMixin._advance_after_leaf

    def spy(self, state, leaf, *args, **kwargs):
        start = leaf.leading_anchor_count
        shown = state.pressure_messages[start:start + leaf.selected_raw_len]
        head = {"role": "user", "content": leaf.summary_text}
        found.append({"shown_rows": len(shown), "hidden_rows": len(leaf.compacted_chunk) - len(leaf.compacted_positions),
                      "copies_excluded": len(self._store_complete_copy_exclusion_proofs or {}),
                      "covered_store_ids": len(leaf.consumed_store_ids),
                      "freed_host": host_estimate(shown), "freed_plugin": count_messages_tokens(shown),
                      "summary_host": host_estimate([head]), "summary_plugin": count_tokens(leaf.summary_text)})
        return advance(self, state, leaf, *args, **kwargs)

    def condense_spy(self, state, *args, **kwargs):
        out = condense(self, state, *args, **kwargs)
        found_context.append(host_estimate(state.pre_condensation_context))  # the list after the leaves, before condensing
        return out

    found_context, condense = [], CompactionMixin._condense_after_leaves
    monkeypatch.setattr(CompactionMixin, "_advance_after_leaf", spy)
    monkeypatch.setattr(CompactionMixin, "_condense_after_leaves", condense_spy)
    found.pre_condensation_host = found_context  # type: ignore[attr-defined]
    return found


def _pass(engine, *, hidden: int, copies: bool, matched: bool = False, turns: int = 40, old: int = 32,
          user_tokens: int = 15, output_tokens: int = 4_000):
    """Store every turn in full, then hand the host list. Of the oldest ``old`` turns, ``hidden`` of every
    ``hidden + 1`` are not in the list (hidden stored rows). ``copies``: after every shown old turn the store also holds
    a byte-equal re-ingest copy of its tool call and result, which the list does not show. Every tool output in the list
    but the newest two is the host's salvage placeholder. ``matched``: the host reports the list's own size (by its
    estimator) as the pressure, instead of the threshold plus 20,000."""
    def is_hidden(i):
        return hidden and i < old and i % (hidden + 1) != hidden

    stored_rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows = _turn(i, user_tokens, output_tokens)
        stored_rows += rows
        if copies and i < old and not is_hidden(i):
            stored_rows += [dict(rows[1]), dict(rows[2])]
    engine.ingest(stored_rows)
    stored = len(engine._store.get_session_messages("S", limit=100_000))
    shown = [row for i in range(turns) if not is_hidden(i) for row in _turn(i, user_tokens, output_tokens)]
    tools = [index for index, row in enumerate(shown) if row["role"] == "tool"]
    for index in tools[:-2]:
        shown[index] = {**shown[index], "content": CLEARED}
    view = [stored_rows[0]] + shown + [{"role": "user", "content": "latest ask"}]
    pressure = host_estimate(view) if matched else engine.threshold_tokens + 20_000
    result = engine.compress(view, current_tokens=pressure)
    after = len(engine._store.get_session_messages("S", limit=100_000))
    return view, result, stored, after


def _report(engine, view, result, stored, after, found):
    status = engine.get_status()
    in_host = host_estimate(view)
    return {"leaf_only_host": found.pre_condensation_host[-1] if found.pre_condensation_host else None,
            "reconciled_host": in_host - sum(leaf["freed_host"] for leaf in found)
            + sum(leaf["summary_host"] for leaf in found),
            "host_estimator": HOST_ESTIMATOR, "in_host": host_estimate(view), "out_host": host_estimate(result),
            "in_plugin": count_messages_tokens(view), "out_plugin": count_messages_tokens(result),
            "rows_in": len(view), "rows_out": len(result), "list_rows_freed": sum(leaf["shown_rows"] for leaf in found),
            "stored_before": stored, "stored_after": after, "summary_budgets": sorted(set(SUMMARY_BUDGETS)),
            "leaves": found, "sweep": status.get("threshold_full_sweep"),
            "progress_reported": getattr(engine, "_last_compression_made_progress", "n/a (no #2863)")}


@pytest.mark.parametrize(("hidden", "copies", "matched"), [
    (2, False, False), (3, False, False), (1, True, False), (0, True, False), (2, False, True), (1, True, True)],
    ids=["two-in-three-hidden", "three-in-four-hidden", "half-hidden-with-copies", "copy-heavy",
         "two-in-three-hidden-matched-pressure", "half-hidden-with-copies-matched-pressure"])
def test_hidden_or_copy_heavy_leaves_never_grow_the_list(make_engine, leaves, hidden, copies, matched):
    """copy-heavy: every unshown row is a proven re-ingest copy (the twin proof excludes it: no budget, no summary).
    matched: the compaction threshold is 1.2% of the 1M window (12,000), so the list itself is over it; the summary
    prefix target stays 24,000, the 1M window's default (one leaf budget), not a quarter of that small threshold."""
    engine = make_engine(**({"context_threshold": 0.012, "summary_prefix_target_tokens": 24_000} if matched else {}))
    view, result, stored, after = _pass(engine, hidden=hidden, copies=copies, matched=matched)
    report = _report(engine, view, result, stored, after, leaves)
    print("REPORT", report)
    assert after == stored + 1, report  # premise: only the new user row is stored (placeholders map to their rows)
    assert leaves and any(leaf["hidden_rows"] or leaf["copies_excluded"] for leaf in leaves), report  # unshown rows read
    assert report["out_host"] <= report["in_host"], (
        f"the pass grew the list by the host's estimator: {report['in_host']} -> {report['out_host']}; {report}")


def test_control_no_hidden_rows_shrinks(make_engine, leaves):
    """The same shape with every turn shown and no copies: the leaves consume shown rows only."""
    engine = make_engine()
    view, result, stored, after = _pass(engine, hidden=0, copies=False)
    report = _report(engine, view, result, stored, after, leaves)
    print("REPORT", report)
    assert leaves and report["out_host"] <= report["in_host"], report
