"""#457 committed-prefix resume: negative controls, remainders and restarts.

A retry after a cancelled-but-committed compaction replays the rows the DAG
already covers. The first leaf pass drops that run only when it is exactly the
session's durable rows ending at the lifecycle frontier; every other shape must
stay byte-identical to the pre-fix fail-open path.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import is_externalized_placeholder
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError, LifecycleStateStore
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "issue-457", "issue-457-conv"
HEADER = "Summary (d0, node "


class _HostCancel(BaseException):
    """Stand-in for the host's hard cancel (a BaseException, like AuxiliaryExplicitCancellation)."""


class _Provider:
    def __init__(self, cancel_on=()):
        self.calls, self.cancel_on = 0, set(cancel_on)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls in self.cancel_on:
            raise _HostCancel()
        return f"Stub summary {self.calls} of ordered work.\nExpand for details about: stub", 1


def _engine(tmp_path, **overrides):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SID, platform="cli", conversation_id=CID, context_length=200_000)
    if engine._config.threshold_full_sweep_enabled:
        engine.threshold_tokens = 1
    return engine


def _transcript(prefix_tokens=24_000):
    plain, i = [], 0
    while len(plain) < 40 or count_messages_tokens(plain[:-30]) < prefix_tokens:
        plain.append({"role": "user" if i % 2 == 0 else "assistant",
                      "content": f"owned turn {i} " + " ".join(f"w{i}x{j}" for j in range(120))})
        i += 1
    return plain + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
         "function": {"name": "synthetic", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "synthetic result"},
    ]


def _compress(engine, messages):
    return engine.compress(messages, current_tokens=count_messages_tokens(messages), force=True)


def _frontier(engine):
    return int(engine._lifecycle.get_by_conversation(CID).current_frontier_store_id)


def _leaves(engine):
    return [n for n in engine._dag.get_session_nodes(SID) if n.source_type == "messages"]


RESUME_HELPER = "_committed_replay_drops"


def _NO_RESUME(self, working, start):
    return [], None, 0, 0


def _spy_resumes(monkeypatch):
    """Record how many replayed rows each first-pass resume check consumed."""
    found = []
    real = getattr(LCMEngine, RESUME_HELPER)

    def spy(self, working, start):
        result = real(self, working, start)
        found.append(len(result[0]))
        return result

    monkeypatch.setattr(LCMEngine, RESUME_HELPER, spy)
    return found


def _cancelled_commit(tmp_path, monkeypatch, messages, *, cancel_on=(), **overrides):
    """Ingest, run one compaction whose result the host discards, return the engine."""
    provider = _Provider(cancel_on)
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    engine = _engine(tmp_path, **overrides)
    engine.ingest(messages)
    committed = None
    try:
        committed = deepcopy(_compress(engine, messages))
    except _HostCancel:
        pass
    return engine, provider, committed


def _snapshot(engine, provider, out):
    rows = engine._store._conn.execute(
        "SELECT session_id, role, content FROM messages ORDER BY store_id"
    ).fetchall()
    return {
        "out": json.dumps(out, sort_keys=True, default=str),
        "status": engine._last_compression_status,
        "noop": engine._last_compression_noop_reason,
        "calls": provider.calls,
        "rows": len(rows),
        "duplicates": len(rows) - len(set(rows)),
        "frontier": _frontier(engine),
        "leaves": len(_leaves(engine)),
    }


# -- negative controls: every failed check is byte-identical to the pre-fix path ---------------

def _edit_covered(engine, replay):
    replay[5] = {**replay[5], "content": replay[5]["content"] + " edited"}


def _hole(engine, replay):
    del replay[50]


def _injected_before_next(engine, replay):
    replay.insert(110, {"role": "user", "content": "injected row before the frontier successor"})


def _rotated_binding(engine, replay):
    other = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
    other.on_session_start("issue-457-other", platform="cli", conversation_id=CID, context_length=200_000)
    other.shutdown()


def _parent_session_prefix(engine, replay):
    engine.on_session_start("issue-457-child", boundary_reason="compression", old_session_id=SID,
                            platform="cli", conversation_id=CID, context_length=200_000)


CONTROLS = {
    "edited-covered-row": _edit_covered,
    "hole-in-run": _hole,
    "row-injected-before-successor": _injected_before_next,
    "rotated-binding": _rotated_binding,
    "parent-session-prefix": _parent_session_prefix,
}


def _control_run(tmp_path, monkeypatch, mutate, *, resume):
    messages = _transcript()
    engine, provider, _ = _cancelled_commit(tmp_path, monkeypatch, messages)
    try:
        replay = deepcopy(messages)
        mutate(engine, replay)
        with monkeypatch.context() as patch:
            if resume:
                found = _spy_resumes(patch)
            else:
                found = []
                patch.setattr(LCMEngine, RESUME_HELPER, _NO_RESUME)
            out = _compress(engine, replay)
        return {**_snapshot(engine, provider, out), "resumed": [k for k in found if k]}
    finally:
        engine.shutdown()


@pytest.mark.parametrize("control", sorted(CONTROLS))
def test_negative_controls_keep_the_prefix_fail_open_output(tmp_path, monkeypatch, control):
    fixed = _control_run(tmp_path / "fixed", monkeypatch, CONTROLS[control], resume=True)
    today = _control_run(tmp_path / "today", monkeypatch, CONTROLS[control], resume=False)
    assert fixed["resumed"] == []
    assert fixed == {**today, "resumed": []}


@pytest.mark.parametrize("restart", [False, True], ids=["same-engine", "cold-engine"])
def test_operator_rotate_frontier_without_summary_is_not_resumed(tmp_path, monkeypatch, restart):
    """rotate_active_session advances the lifecycle frontier with NO summary and leaves the
    in-process marker behind; a cold restart re-binds the marker to that frontier. Either
    way those rows have no committed lineage and must never be dropped from the active view."""
    def run(resume):
        messages = _transcript()
        provider = _Provider()
        with monkeypatch.context() as patch:
            patch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
            engine = _engine(tmp_path / str(resume))
            try:
                engine.ingest(messages)
                assert engine.rotate_active_session(apply=True)["ok"]
                assert _frontier(engine) > int(engine._last_compacted_store_id or 0)
                if restart:
                    engine = _restart(engine, tmp_path / str(resume))
                    assert _frontier(engine) == int(engine._last_compacted_store_id or 0)
                if resume:
                    found = _spy_resumes(patch)
                else:
                    found = []
                    patch.setattr(LCMEngine, RESUME_HELPER, _NO_RESUME)
                out = _compress(engine, messages)
                return {**_snapshot(engine, provider, out), "resumed": [k for k in found if k]}
            finally:
                engine.shutdown()

    fixed, today = run(True), run(False)
    assert fixed["resumed"] == [] and fixed == today


def test_positive_control_same_harness_resumes(tmp_path, monkeypatch):
    """The control harness itself can see a resume: the unmodified replay resumes."""
    fixed = _control_run(tmp_path / "fixed", monkeypatch, lambda engine, replay: None, resume=True)
    today = _control_run(tmp_path / "today", monkeypatch, lambda engine, replay: None, resume=False)
    assert fixed["resumed"] == [110]
    assert (fixed["status"], fixed["calls"], fixed["rows"], fixed["duplicates"]) == ("compacted", 1, 142, 0)
    assert (today["status"], today["calls"], today["rows"]) == ("error", 2, 142)


# -- a cancel between leaf passes: the retry summarizes only the remainder ---------------------

SWEEP = {"threshold_full_sweep_enabled": True, "leaf_chunk_tokens": 12_500,
         "summary_prefix_target_tokens": 1_000_000}


def _restart(engine, tmp_path, **overrides):
    engine.shutdown()
    return _engine(tmp_path, **overrides)


@pytest.mark.parametrize("restart", [False, True], ids=["same-engine", "cold-engine"])
def test_mid_sweep_cancel_retry_summarizes_only_the_remainder(tmp_path, monkeypatch, restart):
    messages = _transcript()
    original = deepcopy(messages)
    engine, provider, committed = _cancelled_commit(tmp_path, monkeypatch, messages, cancel_on={2}, **SWEEP)
    try:
        assert committed is None and provider.calls == 2
        (leaf1,) = _leaves(engine)
        first_frontier = _frontier(engine)
        assert first_frontier == max(leaf1.source_ids) and 0 < first_frontier < 110
        if restart:
            engine = _restart(engine, tmp_path, **SWEEP)
        found = _spy_resumes(monkeypatch)
        out = _compress(engine, messages)
        leaves = _leaves(engine)
        assert found[:1] == [first_frontier] and engine._last_compression_status == "compacted"
        assert provider.calls == 3, "exactly one call over the remainder"
        assert [n.node_id for n in leaves][:1] == [leaf1.node_id] and len(leaves) == 2
        leaf2 = leaves[1]
        assert not set(leaf2.source_ids) & set(leaf1.source_ids)
        assert leaf2.source_ids == list(range(first_frontier + 1, max(leaf2.source_ids) + 1))
        assert _frontier(engine) == max(leaf2.source_ids) == 110
        assert not [n for n in engine._dag.get_session_nodes(SID) if n.depth > 0], "no condensation"
        assert engine._store.get_session_count(SID) == 142 and messages == original
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert [text.count(f"{HEADER}{n.node_id})") for n in leaves] == [1, 1]
        assert all(str(m["content"]) in text for m in messages[110:] if m.get("content"))
    finally:
        engine.shutdown()


def test_publication_failure_after_resume_keeps_summary_and_the_rest(tmp_path, monkeypatch):
    messages = _transcript()
    engine, provider, _ = _cancelled_commit(tmp_path, monkeypatch, messages, cancel_on={2}, **SWEEP)
    try:
        (leaf1,) = _leaves(engine)
        first_frontier = _frontier(engine)

        def conflict(self, *args, **kwargs):
            raise LifecyclePublicationConflictError("forced remainder conflict")

        monkeypatch.setattr(LifecycleStateStore, "stage_compaction_publication", conflict)
        found = _spy_resumes(monkeypatch)
        out = _compress(engine, messages)
        assert found[:1] == [first_frontier] and provider.calls == 3
        assert engine._last_compression_status == "error"
        assert _leaves(engine) == [leaf1] and _frontier(engine) == first_frontier
        text = "\n".join(str(m.get("content") or "") for m in out)
        # Covered rows leave the view only together with their committed summary.
        assert text.count(f"{HEADER}{leaf1.node_id})") == 1
        covered = len(leaf1.source_ids)
        assert all(str(m["content"]) in text for m in messages[covered:] if m.get("content"))
        assert not any(str(m["content"]) in text for m in messages[:covered])
    finally:
        engine.shutdown()


def _anchored_tool_transcript():
    messages = [{"role": "system", "content": "stable system prompt"},
                {"role": "user", "content": "the only real user prompt"}]
    for index in range(10):
        call_id = f"issue_457_call_{index}"
        messages += [
            {"role": "assistant", "content": f"working step {index} " + "a" * 160,
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": f"result {index} " + " ".join(f"r{index}x{j}" for j in range(400))},
        ]
    return messages


ANCHORED = {"fresh_tail_count": 4, "leaf_chunk_tokens": 1,
            "large_output_externalization_enabled": True,
            "large_output_externalization_threshold_chars": 500}


def test_retained_anchor_tool_pairs_and_externalized_payloads_resume(tmp_path, monkeypatch):
    """The anchor-status flip of this same fixture is
    test_retained_anchor_that_loses_anchor_status_is_preserved."""
    messages = _anchored_tool_transcript()
    engine, provider, committed = _cancelled_commit(tmp_path, monkeypatch, messages, **ANCHORED)
    try:
        assert engine._last_compression_status == "compacted" and provider.calls == 1
        assert committed[1] == messages[1], "the sole real user stays the retained anchor"
        externalized = sum(
            is_externalized_placeholder(str(content or ""))
            for (content,) in engine._store._conn.execute(
                "SELECT content FROM messages WHERE role = 'tool' AND store_id <= ?", (_frontier(engine),)
            )
        )
        frontier = _frontier(engine)
        found = _spy_resumes(monkeypatch)
        leading = []
        real_leading = LCMEngine._leading_anchor_count
        monkeypatch.setattr(LCMEngine, "_leading_anchor_count",
                            lambda self, m: leading.append(real_leading(self, m)) or leading[-1])
        out = _compress(engine, deepcopy(messages))
        assert leading[:1] == [2] and found[:1] and found[0] > 0, "the resume must fire"
        assert externalized > 0, "the covered run carries externalized payloads"
        assert provider.calls == 1 and engine._last_compression_status == "compacted"
        # The one-time LCM system note is appended only while compression_count == 0, and the
        # cancelled attempt already counted in this process: the host's own system row returns.
        assert out[0] == messages[0] and out[1:] == committed[1:]
        assert _frontier(engine) == frontier and len(_leaves(engine)) == 1
    finally:
        engine.shutdown()


# -- after the host adopts the resumed output ----------------------------------------------------

def _resumed(tmp_path, monkeypatch, messages, **overrides):
    engine, provider, committed = _cancelled_commit(tmp_path, monkeypatch, messages, **overrides)
    retried_at = time.time()
    out = _compress(engine, messages)
    assert engine._last_compression_status == "compacted" and provider.calls == 1
    return engine, provider, committed, out, retried_at


def test_cold_restart_after_adopting_resumed_output_stores_only_new_rows(tmp_path, monkeypatch):
    messages = _transcript()
    engine, provider, committed, out, retried_at = _resumed(tmp_path, monkeypatch, messages)
    try:
        assert out == committed
        engine.on_session_end(SID, messages)
        engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, platform="cli",
                                conversation_id=CID, context_length=200_000)
        engine = _restart(engine, tmp_path)
        live = deepcopy(out) + [{"role": "user", "content": "NEW-457 user row after the resume"},
                                {"role": "assistant", "content": "NEW-457 reply after the resume"}]
        engine.ingest(live)
        rows = engine._store._conn.execute(
            "SELECT session_id, role, content FROM messages ORDER BY store_id"
        ).fetchall()
        assert len(rows) == 144 and len(set(rows)) == 144, "0 re-store, 0 duplicates"
        assert [r[2] for r in rows[-2:]] == [m["content"] for m in live[-2:]]
        # The restart binds the resumed attempt's durable proof, and the summary projects
        # as a proven emission through its v4 descriptors.
        payload = engine._durable_commit_proof_payload()
        assert payload["descriptor_version"] == 4 and payload["created_at"] >= retried_at
        assert payload["last_store_id"] == 142 and payload["emissions"]
        assert engine._cursor_from_durable_commit_proof(live) == len(live), "the new rows follow the proof"
        projection, _occurrences = engine._occurrence_replay_identities(live, payload)
        assert HEADER in str(projection.entries[0].generated_span or "")
    finally:
        engine.shutdown()


def test_steady_state_after_resume_consumes_nothing_extra(tmp_path, monkeypatch):
    """[summary] + tail + new rows is not a committed replay: output as without the resume."""
    def run(resume):
        messages = _transcript()
        engine, provider, _committed, out, _t = _resumed(tmp_path / str(resume), monkeypatch, messages)
        try:
            live = deepcopy(out) + [{"role": "user", "content": "NEW-457 steady user row"},
                                    {"role": "assistant", "content": "NEW-457 steady reply"}]
            with monkeypatch.context() as patch:
                if resume:
                    found = _spy_resumes(patch)
                else:
                    found = []
                    patch.setattr(LCMEngine, RESUME_HELPER, _NO_RESUME)
                result = _compress(engine, live)
            return {**_snapshot(engine, provider, result), "resumed": [k for k in found if k]}
        finally:
            engine.shutdown()

    fixed, today = run(True), run(False)
    assert fixed["resumed"] == [] and fixed == today
    assert fixed["rows"] == 144 and fixed["duplicates"] == 0


def test_covered_row_duplicating_a_tail_row_loses_nothing(tmp_path, monkeypatch):
    messages = _transcript()
    messages[130] = {**messages[130], "content": messages[2]["content"]}
    engine, provider, committed, out, _t = _resumed(tmp_path, monkeypatch, messages)
    try:
        assert out == committed and engine._store.get_session_count(SID) == 142
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert text.count(messages[2]["content"]) == 1, "the tail copy stays; the covered copy is summarized"
        assert all(str(m["content"]) in text for m in messages[110:] if m.get("content"))
    finally:
        engine.shutdown()


def test_resume_does_not_engage_the_fresh_tail_pressure_yield(tmp_path, monkeypatch):
    messages = _transcript()
    engine, provider, committed = _cancelled_commit(tmp_path, monkeypatch, messages)
    try:
        yields = []
        real_yield = LCMEngine._maybe_engage_fresh_tail_pressure_yield
        monkeypatch.setattr(LCMEngine, "_maybe_engage_fresh_tail_pressure_yield",
                            lambda self, *a, **k: yields.append(real_yield(self, *a, **k)) or yields[-1])
        found = _spy_resumes(monkeypatch)
        out = _compress(engine, messages)
        assert found == [110] and yields == [] and out == committed and provider.calls == 1
    finally:
        engine.shutdown()


# -- round 2: consumed rows must be accounted for by committed lineage ---------------------------

def _texts(out):
    return "\n".join(str(m.get("content") or "") for m in out)


def test_retained_anchor_that_loses_anchor_status_is_preserved(tmp_path, monkeypatch):
    """N1: attempt 1 keeps user row 2 as the retained anchor and summarizes 3..18. A new user
    turn before the retry ends row 2's anchor status; it was never summarized, so the retry
    must keep it (raw or under a NEW leaf) while rows 3..18 are not summarized again."""
    messages = _anchored_tool_transcript()
    engine, provider, committed = _cancelled_commit(tmp_path, monkeypatch, messages, **ANCHORED)
    try:
        (leaf1,) = _leaves(engine)
        assert committed[1] == messages[1] and leaf1.source_ids == list(range(3, 19))
        assert _frontier(engine) == 18 and provider.calls == 1
        retry = deepcopy(messages) + [{"role": "user", "content": "a NEW user turn before the retry"}]
        conflicts = []
        real_stage = LifecycleStateStore.stage_compaction_publication

        def stage(self, *args, **kwargs):
            try:
                return real_stage(self, *args, **kwargs)
            except LifecyclePublicationConflictError as exc:
                conflicts.append(str(exc))
                raise

        monkeypatch.setattr(LifecycleStateStore, "stage_compaction_publication", stage)
        engine.ingest(retry)
        out = _compress(engine, retry)
        leaves = _leaves(engine)
        new_leaves = [n for n in leaves if n.node_id != leaf1.node_id]
        assert messages[1]["content"] in _texts(out) or any(2 in n.source_ids for n in new_leaves), \
            "the original prompt stays in the active context"
        assert not any(set(n.source_ids) & set(leaf1.source_ids) for n in new_leaves), "3..18 not re-summarized"
        assert provider.calls <= 2 and conflicts == []
        assert engine._last_compression_status == "compacted"
        # Kept raw right after the committed summary; the retry makes no provider call.
        assert provider.calls == 1 and new_leaves == []
        assert [m["content"] for m in out[2:]] == [m["content"] for m in retry[1:2] + retry[18:]]
        assert f"{HEADER}{leaf1.node_id})" in str(out[1]["content"]) and out[0] == messages[0]
        rows = engine._store._conn.execute("SELECT session_id, role, content FROM messages").fetchall()
        assert len(rows) == len(retry) and len(set(rows)) == len(rows), "0 duplicate rows"
    finally:
        engine.shutdown()


def test_cold_restart_after_operator_rotation_drops_nothing(tmp_path, monkeypatch):
    """N2: rotate_active_session advances the frontier with no summary; a cold restart binds
    the in-process marker to it. The replay still has no committed lineage: nothing is dropped."""
    messages = _transcript()
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    engine = _engine(tmp_path)
    try:
        engine.ingest(messages)
        assert engine.rotate_active_session(apply=True)["ok"] and _frontier(engine) == 110
        engine = _restart(engine, tmp_path)
        assert int(engine._last_compacted_store_id or 0) == _frontier(engine) == 110
        out = _compress(engine, deepcopy(messages))
        text = _texts(out)
        assert all(str(m["content"]) in text for m in messages if m.get("content")), "all 142 rows kept"
        # The resume-disabled control makes the same one provider call and keeps the same rows.
        assert engine._dag.get_session_nodes(SID) == [] and engine._store.get_session_count(SID) == 142
    finally:
        engine.shutdown()


def test_frontier_on_a_user_row_outside_lineage_is_not_resumed(tmp_path, monkeypatch):
    """N3: a cancelled compaction commits 1..110, then an operator rotation moves the frontier
    to a later USER row with no summary; after a cold restart the run's last row F is a user
    row outside committed lineage, so the helper consumes nothing."""
    def run(resume):
        messages = _transcript()
        for index in range(13):
            messages.append({"role": "user" if index % 2 == 0 else "assistant",
                             "content": f"extra turn {index} " + " ".join(f"e{index}x{j}" for j in range(60))})
        engine, provider, _ = _cancelled_commit(tmp_path / str(resume), monkeypatch, messages[:142])
        try:
            engine.ingest(messages)
            assert engine.rotate_active_session(apply=True)["ok"]
            frontier = _frontier(engine)
            engine = _restart(engine, tmp_path / str(resume))
            role = engine._store._conn.execute(
                "SELECT role FROM messages WHERE store_id = ?", (frontier,)).fetchone()[0]
            with monkeypatch.context() as patch:
                if resume:
                    found = _spy_resumes(patch)
                else:
                    found = []
                    patch.setattr(LCMEngine, RESUME_HELPER, _NO_RESUME)
                out = _compress(engine, deepcopy(messages))
            return {**_snapshot(engine, provider, out), "resumed": [k for k in found if k],
                    "f_role": role, "frontier_before": frontier}
        finally:
            engine.shutdown()

    fixed, today = run(True), run(False)
    assert fixed["f_role"] == "user" and fixed["frontier_before"] > 110
    assert fixed["resumed"] == [] and fixed == {**today, "resumed": []}


def test_resumed_rows_are_discounted_from_the_token_estimate(tmp_path, monkeypatch):
    """T2: with dynamic leaf chunking the retry stops where an uninterrupted run stops: the
    committed summary, not the dropped rows, counts toward the continuation estimate."""
    dynamic = {"dynamic_leaf_chunk_enabled": True, "leaf_chunk_tokens": 4_000}
    messages = _transcript(80_000)
    tokens = count_messages_tokens(messages)

    with monkeypatch.context() as patch:
        control = _Provider()
        patch.setattr(lcm_engine_module, "summarize_with_escalation", control)
        engine = _engine(tmp_path / "control", **dynamic)
        try:
            engine.threshold_tokens = 40_000
            engine.ingest(messages)
            engine.compress(messages, current_tokens=tokens)
            control_leaves = [len(n.source_ids) for n in _leaves(engine)]
            raw_left = engine._store.get_session_count(SID) - _frontier(engine) - 32
        finally:
            engine.shutdown()
    assert control.calls == len(control_leaves) == 2 and raw_left > 0, "stopped by the threshold"

    provider = _Provider(cancel_on={2})
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    engine = _engine(tmp_path / "retry", **dynamic)
    try:
        engine.threshold_tokens = 40_000
        engine.ingest(messages)
        with pytest.raises(_HostCancel):
            engine.compress(messages, current_tokens=tokens)
        assert [len(n.source_ids) for n in _leaves(engine)] == control_leaves[:1]
        found = _spy_resumes(monkeypatch)
        engine.compress(deepcopy(messages), current_tokens=tokens)
        assert found[:1] == [control_leaves[0]]
        assert provider.calls - 2 == control.calls - 1, "the retry runs only the control's remaining pass"
        assert len(_leaves(engine)) == len(control_leaves)
    finally:
        engine.shutdown()
