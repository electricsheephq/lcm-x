"""R-STABLE: the host's replay is byte-stable between compaction swaps (I1, I2, I11).

The fixture sets database_path and hermes_home under tmp_path to isolate state.
embeddings_enabled=False and temporal_rollups_enabled=False keep background work out.
leaf_chunk_tokens=1_000 sets a small nominal leaf target. fresh_tail_count=2 keeps the latest pair
and rises to 64 for the fresh-only probe. fresh_tail_pressure_yield_enabled=False
keeps that tail fixed. context_threshold=0.8 sets the ordinary 960_000-token budget
in a 1_200_000-token window. The fixture sets threshold_tokens=12_000 to trigger
leaf and fresh-only attempts, and replay tokens minus 500 to trigger stub-only exits.
threshold_full_sweep_enabled=False isolates each threshold pass. l3_truncate_tokens=1
keeps the leaf sources above the verbatim shortcut so the summary stand-in runs.
incremental_max_depth=0 (1 for condensation) and
condensation_fanin=2 limit condensation to its own cells with a small seed.
large_output_externalization_enabled=True and large_output_active_replay_stubbing_enabled=True
exercise the stub path. large_output_externalization_threshold_chars=1_000_000 keeps
the seed and corpus inline while probing aged replay stubs. LCM_HOST_MESSAGE_UID=off/on
selects identity modes. summary_timeout_ms=23_000 is loaded for the config-reload event.

Publication rule (rulings T1-555, T1-557, T1-581): a summary node may first become visible only in the return of a
call that itself commits (it stores a leaf or a condensation, takes the stub-only exit, or is an overflow commit),
not necessarily the call that stored it, so a hidden-only pass (#904) followed by a later commit that shows both
summaries passes. The one exception is a fault between a leaf's commit and its call's return: that call fails open
and its own leaves may first show exactly one call late ("fault-late publication", F6). Any other first showing in
a non-commit return fails as "deferred publication". A host-merged, non-canonical head re-rendered in a stub-only
exit is the allowed "merged-head repair"; a canonical head rewritten there fails as "needless head rewrite".
"""

import hashlib
import json
import logging
import os
import socket
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from itertools import count, zip_longest

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit
from hermes_lcm.config import LCMConfig, host_message_uid_mode
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from hermes_lcm.tokens import count_messages_tokens
from tests.behaviour.test_ingest_trace import reopen
from tests.harness import pin_char_counter
from tests.invariants.test_issue_1540_prefix_proof import CLEARED
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_issue_671_stub_first_exit import STUB, payload_of, run, stop_line, text_view, tool_pair, tool_view

WINDOW = 1_200_000
ORDINARY_THRESHOLD = 960_000
SID = "r-stable"
BINDING = dict(platform="cli", conversation_id=SID, context_length=WINDOW)
SUMMARY_MARKER = "R-STABLE source "
EVENTS = ("none", "cold-bind", "session-switch", "config-reload", "inbox")
PASS_KINDS = ("leaf", "condensation", "stub-only", "stub-only-merged-head", "refused")
KNOWN_FAILS = {
    "refused/none/off": frozenset({"non-commit return"}),
    "refused/none/on": frozenset({"non-commit return"}),
    "refused/cold-bind/off": frozenset({"non-commit return"}),
    "refused/cold-bind/on": frozenset({"non-commit return"}),
    "refused/session-switch/off": frozenset({"non-commit return"}),
    "refused/session-switch/on": frozenset({"non-commit return"}),
    "refused/config-reload/off": frozenset({"non-commit return"}),
    "refused/config-reload/on": frozenset({"non-commit return"}),
    "refused/inbox/off": frozenset({"non-commit return"}),
    "refused/inbox/on": frozenset({"non-commit return"}),
}
GUARDRAIL_LOG = "LCM assembly guardrail recovery:"
FIT_LOG = "LCM survival fit applied ("


class KnownSwapRuleFailure(AssertionError):
    """The cell reproduced exactly its pinned failure names."""


def swap_cell(uid_mode, event, pass_kind):
    baseline = KNOWN_FAILS.get(f"{pass_kind}/{event}/{uid_mode}")
    marks = []
    if baseline:
        names = ", ".join(sorted(baseline))
        marks.append(pytest.mark.xfail(
            strict=True, raises=KnownSwapRuleFailure,
            reason=f"R-STABLE baseline on f77d6ed9: {names}; ruling T1-555",
        ))
    return pytest.param(uid_mode, event, pass_kind, id=f"{uid_mode}-{event}-{pass_kind}", marks=marks)


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Row:
    data: bytes
    digest: str
    kind: str

    @classmethod
    def capture(cls, index, message):
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        kind = message["role"]
        if kind == "tool" and (text == CLEARED or text.startswith(STUB)):
            kind = "stub"
        elif index == 0 and text.startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX):
            kind = "carrier"
        elif SUMMARY_MARKER in text and ("Summary (" in text or "Summary[" in text):
            kind = "summary"
        elif index == 0:
            kind = "head"
        data = encode(message)
        return cls(data, sha(data), kind)


def capture(messages):
    return [Row.capture(index, message) for index, message in enumerate(messages)]


def first_diff(before, after, start=0):
    if start:
        before, after = before[start:], after[start:]
    for index, (left, right) in enumerate(zip_longest(before, after), start):
        if left is None or right is None or left.data != right.data:
            return index, left, right
    return None


def summary_text(node):
    return engine_module._summary_part_text(node)


def visible_nodes(nodes, messages):
    """Locate rendered DAG spans in the provider head, including a folded carrier."""
    visible = {}
    for index, row in enumerate(messages):
        content = row.get("content")
        if row["role"] == "system":
            continue
        if isinstance(content, str) and SUMMARY_MARKER in content:
            matches = {node_id: index for node_id, node in nodes.items() if summary_text(node) in content}
            if matches:
                visible.update(matches)
                continue
        if index != 0 or row["role"] != "user" or visible:
            break  # a retained user anchor can sit ahead of the generated summary
    return visible


def source_closure(nodes, ids):
    """Find the leaves a rendered condensation publishes through its source links."""
    covered = set(ids)
    pending = list(ids)
    while pending:
        node = nodes[pending.pop()]
        if node.source_type == "nodes":
            added = set(node.source_ids) - covered
            covered.update(added)
            pending.extend(added)
    return covered


def changed_summary_start(nodes, new_ids, old_visible, new_visible):
    if not new_ids:
        return []
    # An existing head may change when this pass adds to its summary or
    # replaces a node in it. An independent leading anchor is still stable.
    replaced = source_closure(nodes, new_ids) - new_ids
    return [*(index for node, index in old_visible.items() if node in replaced),
            *(index for node, index in new_visible.items() if node in new_ids)]


def stored_nodes(engine):
    return {node.node_id: node for node in engine._dag.get_session_nodes(SID)}


def corpus(engine, stamp):
    """Build the reference replay and the durable originals behind its placeholders."""
    pairs, stored = [], []
    for index in range(673):
        cid = f"history-{index}"
        if index < 214:
            for occurrence in range(18 if index < 88 else 17):
                stored.extend(tool_pair(cid, f"Historical result {index}/{occurrence}."))
                stored[-2]["content"] = f"Historical read {index}/{occurrence}."
            payload = CLEARED
        elif index < 670:
            payload = json.dumps({"status": "ok", "document": index, "lines": [index, index + 1]})
        else:
            payload = [{"type": "text", "text": f"Listing {index}.\n" + "Synthetic listing.\n" * 50_000}]
        pairs.append(tool_pair(cid, payload))
    # 673 results in 218 complete tool groups, for 891 supplied rows.
    active = []
    start = 0
    for size in (4,) * 19 + (3,) * 199:
        group = pairs[start:start + size]
        active.append({**group[0][0], "tool_calls": [call["tool_calls"][0] for call, _ in group]})
        active.extend(result for _, result in group)
        start += size
    engine._store.append_batch(SID, stamp(stored), conversation_id=SID)
    return active


def followups(inbox=False):
    ordinary = {"role": "user", "content": "Please queue another comparison."}
    if inbox:
        ordinary["content"] = "Inbox admission from an external sender: please compare the queued accounts."
    yield [ordinary]
    yield tool_pair("new-small", "inline result")
    yield tool_pair("new-large", payload_of(11_000, "survey"))
    yield tool_pair("new-json", json.dumps({"new_result": [1, 2, 3]}))
    yield None  # Paste the summary current at this call.


@dataclass
class Host:
    cell: str
    caplog: object
    engine: object
    stamp: object
    messages: list = field(default_factory=list, repr=False)
    snapshots: list = field(default_factory=list, repr=False)
    calls: list = field(default_factory=list, repr=False)
    failures: list = field(default_factory=list)
    node_calls: dict = field(default_factory=dict)
    published: set = field(default_factory=set)

    def fail(self, name, diff):
        index, before, after = diff
        left = json.loads(before.data) if before else {}
        right = json.loads(after.data) if after else {}
        self.failures.append(dict(
            cell=self.cell, name=name, call=len(self.calls) + 1, row=index,
            kind=(before or after).kind,
            before=before.digest if before else "missing",
            after=after.digest if after else "missing",
            changed_keys=sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key)),
        ))

    def call(self, appended, overhead=0, *, merge_head=False):
        engine = self.engine
        prior = self.snapshots[-1] if self.snapshots else []
        appended = self.stamp(appended)
        self.messages.extend(appended)
        supplied = capture(self.messages)
        if prior and (diff := first_diff(prior, supplied[:len(prior)])):
            self.fail("rule-1 change", diff)
        if merge_head:
            # Mirror the host coalescing consecutive user rows in _build_host_twin.
            head, next_user = self.messages[:2]
            assert head["role"] == next_user["role"] == "user" and SUMMARY_MARKER in head["content"]
            merged = {**head, "content": head["content"] + "\n\n---\n\n" + next_user["content"]}
            self.messages[:2] = [merged]
            supplied = [Row.capture(0, merged), *supplied[2:]]
        supplied_tools = {row["tool_call_id"]: (index, row["content"]) for index, row in enumerate(self.messages)
                          if row["role"] == "tool"}
        self.caplog.clear()
        engine.ingest(self.messages)
        engine.note_turn_complete()
        observed = count_messages_tokens(self.messages) + overhead
        preflight = engine.should_compress_preflight(self.messages)
        threshold = engine.should_compress(observed) if not preflight else False
        ran = preflight or threshold
        if ran and (diff := first_diff(supplied, capture(self.messages))):
            self.fail("rule-1 change", diff)
        before_nodes = stored_nodes(engine)
        old_visible = visible_nodes(before_nodes, self.messages)
        stop, status, leaves, overflow = "", "", 0, False
        if ran:
            returned = run(engine, self.messages, self.caplog, current_tokens=observed)
            assert isinstance(returned, list), "compress did not return a host list"
            self.messages = returned
            record = stop_line(self.caplog)
            stop, leaves = record.group(1), int(record.group(2))
            status = engine.last_compression_status
        after_nodes = stored_nodes(engine)
        new = after_nodes.keys() - before_nodes.keys()
        condensed = sum(after_nodes[node_id].source_type == "nodes" for node_id in new)
        returned_rows = capture(self.messages)
        diff = first_diff(supplied, returned_rows)
        if ran and diff:
            cap = engine._effective_assembly_token_cap()
            guardrail = status == "overflow_recovery" or any(
                line.startswith((GUARDRAIL_LOG, FIT_LOG)) or line.endswith(", forced overflow recovery")
                for line in self.caplog.messages
            )
            overflow = guardrail and (observed > WINDOW or cap is not None and observed >= cap)
        kinds = [
            label for took, label in (
                (leaves, "leaf"), (condensed, "condensation"), (stop == "stub_first_exit", "stub-only"),
                (overflow, "overflow commit"),
            ) if took
        ]
        call_number = len(self.calls) + 1
        for node_id in new:
            self.node_calls[node_id] = call_number
        visible = visible_nodes(after_nodes, self.messages)
        covered = source_closure(after_nodes, visible)
        # T1-557: a summary node may first become visible only in the return of a call that itself commits,
        # not necessarily the call that stored it (a hidden-only pass, #904, then a later commit is a PASS).
        # T1-581: a fault between a leaf's commit and its call's return fails open; that call's own leaves may
        # first show exactly one call late. Any other first showing in a non-commit return is a FAIL.
        newly_visible = sorted(covered - self.published)
        previous = self.calls[-1] if self.calls else None
        late = sorted(
            node_id for node_id in newly_visible
            if previous is not None and previous["status"] == "error"
            and self.node_calls.get(node_id) == call_number - 1
        ) if newly_visible and not kinds else []
        deferred = [] if kinds else [node_id for node_id in newly_visible if node_id not in late]
        if newly_visible and not kinds:
            assert diff, f"newly visible summary nodes without a change: {newly_visible}"
        if late and not deferred:
            kinds.append("fault-late publication")
            new = set(new) | set(late)
        if deferred:
            self.fail("deferred publication", diff)
        elif diff and not kinds:
            self.fail("non-commit return" if ran else "rule-1 change", diff)
        repair = None
        if diff and kinds and not overflow:
            cuts = changed_summary_start(after_nodes, new, old_visible, visible)
            if stop == "stub_first_exit":
                cuts.extend(
                    supplied_tools[row["tool_call_id"]][0] for index, row in enumerate(self.messages)
                    if row["role"] == "tool" and row["tool_call_id"] in supplied_tools
                    and returned_rows[index].kind == "stub"
                    and row["content"] != supplied_tools[row["tool_call_id"]][1]
                )
                if merge_head and diff[0] == 0:
                    repair = dict(name="merged-head repair", row=0, kind=supplied[0].kind,
                                  before=supplied[0].digest, after=returned_rows[0].digest)
                    diff = first_diff(supplied, returned_rows, start=1)
            if diff and diff[0] < min(cuts, default=len(supplied)):
                name = "needless head rewrite" if diff[1].kind in {"head", "summary", "carrier"} else "rule-1 change"
                self.fail(name, diff)
        self.published.update(covered)
        self.snapshots.append(returned_rows)
        refused = any("reason=publication_invariant_conflict" in line for line in self.caplog.messages) if ran else False
        result = dict(call=call_number, ran=bool(ran), preflight=bool(preflight), status=status, refused=refused,
                      stop=stop, commit=kinds, merged_head_repair=repair, leaves=leaves,
                      condensations=condensed, tokens=observed,
                      new_nodes=sorted(new), visible_nodes=visible, deferred_nodes=deferred,
                      pending_nodes=sorted(self.node_calls.keys() - self.published),
                      rows=[[row.kind, row.digest] for row in returned_rows])
        self.calls.append(result)
        return result


def session_switch(engine, messages):
    engine.on_session_end(SID, messages)
    engine.on_session_start("r-stable-side", platform="cli", conversation_id="r-stable-side",
                            context_length=WINDOW)
    side = [{"role": "user", "content": "Side session only."}]
    engine.ingest(side)
    engine.on_session_end("r-stable-side", side)
    engine.on_session_start(SID, **BINDING)


def refuse_leaf(*_args, **_kwargs):
    # The real DAG insertion rolls back when its pre-commit lifecycle fence refuses.
    raise LifecyclePublicationConflictError("synthetic R-STABLE publication conflict")


@pytest.fixture
def uid_mode(request, monkeypatch, tmp_path):
    for key in tuple(os.environ):
        if key.startswith(("LCM_", "HERMES_")):
            monkeypatch.delenv(key)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("HERMES_HOME", str(empty))
    # The shared hotfix fixture's unset arm is shadow on current main. For
    # this matrix exercise actual off and on with the same capability seam.
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", request.param)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    assert host_message_uid_mode() == request.param
    assert host_uid_emit.identity_emit_enabled() == (request.param == "on")
    return request.param


@pytest.fixture
def replay(uid_mode, monkeypatch, tmp_path, caplog):
    def summarize(*, text, timeout, **_kwargs):
        deadline = time.perf_counter() + timeout
        summary = f"{SUMMARY_MARKER}{sha(text.encode())[:20]}.\nExpand for details about: turns"
        assert time.perf_counter() < deadline, "summary stand-in exceeded its timeout"
        return summary, 1

    def deny_network(*_args, **_kwargs):
        pytest.fail("R-STABLE attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    serial = count(1)
    flush = _host_flush()

    def stamp(rows):
        stamped = flush(rows)
        if uid_mode == "on":
            for row in stamped:
                row.setdefault("message_uid", f"{next(serial):032x}")
        return stamped

    with ExitStack() as engines:
        def build(kind, event):
            config = LCMConfig(
                database_path=str(tmp_path / "lcm.db"), embeddings_enabled=False, temporal_rollups_enabled=False,
                leaf_chunk_tokens=1_000, fresh_tail_count=2, fresh_tail_pressure_yield_enabled=False,
                context_threshold=0.8, threshold_full_sweep_enabled=False, l3_truncate_tokens=1,
                incremental_max_depth=int(kind == "condensation"), condensation_fanin=2,
                large_output_externalization_enabled=True, large_output_externalization_threshold_chars=1_000_000,
                large_output_active_replay_stubbing_enabled=True,
            )
            engine = engine_module.LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
            engines.callback(engine.shutdown)
            engine.on_session_start(SID, **BINDING)
            return Host(f"{kind}/{event}/{uid_mode}", caplog, engine, stamp), engines

        yield build


@pytest.fixture
def exercise(replay, monkeypatch, tmp_path, pass_kind, event):
    host, engines = replay(pass_kind, event)
    started = time.perf_counter()
    error = ""
    try:
        engine = host.engine
        engine.threshold_tokens = 12_000
        seed = host.call(text_view()[1:])
        assert seed["commit"] == ["leaf"], f"seed pass did not store a leaf: {seed['stop']}"
        assert SUMMARY_MARKER in host.messages[0]["content"], "seed summary was not in row 0"
        engine.threshold_tokens = ORDINARY_THRESHOLD
        for turn in range(4):
            ordinary = host.call([{"role": "user", "content": f"Seed follow-up {turn}."}])
            assert not ordinary["commit"], f"seed follow-up {ordinary['call']} committed: {ordinary['commit']}"
        # Reported provider tokens include room for the system and tool schemas.
        # A fresh-only threshold attempt still goes through the production gate.
        engine.threshold_tokens = 12_000
        with monkeypatch.context() as patch:
            patch.setattr(engine._config, "fresh_tail_count", 64)
            fresh = host.call([{"role": "user", "content": "Seed follow-up 4."}], overhead=12_000)
        assert fresh["ran"] and not fresh["commit"], f"fresh-only attempt: {fresh['stop']}"

        # The artificial seed pause represents waiting out the no-progress hold
        # before arranging the cell's next independent threshold pass.
        monotonic, wall = time.monotonic, time.time
        elapsed = engine_module._SWEEP_BUDGET_HOLD_SECONDS + 1
        monkeypatch.setattr(time, "monotonic", lambda: monotonic() + elapsed)
        monkeypatch.setattr(time, "time", lambda: wall() + elapsed)

        history = corpus(engine, host.stamp)
        merged_head = pass_kind == "stub-only-merged-head"
        target_kind = "stub-only" if merged_head else pass_kind
        probe = tool_view()[1:] if target_kind == "stub-only" else text_view(4)[1:]
        appended = [*probe, *history]
        engine.threshold_tokens = (count_messages_tokens(host.messages + appended) - 500
                                   if target_kind == "stub-only" else 12_000)
        rejection = monkeypatch.context() if pass_kind == "refused" else nullcontext()
        with rejection as patch:
            if patch:
                patch.setattr(engine._lifecycle, "stage_compaction_publication", refuse_leaf)
            target = host.call(appended, merge_head=merged_head)

        reached = (
            target_kind in target["commit"] if target_kind != "refused"
            else target["refused"] and target["status"] == "error" and not target["commit"]
        )
        assert reached, (
            f"{host.cell}: target pass not reached at call {target['call']}, "
            f"commit={target['commit']}, stop={target['stop']}, status={target['status']}"
        )
        engine.threshold_tokens = ORDINARY_THRESHOLD
        for index, appended in enumerate(followups(inbox=event == "inbox")):
            if index == (0 if event == "inbox" else 2):
                if event == "cold-bind":
                    host.engine = reopen(engine, tmp_path, engines)
                elif event == "session-switch":
                    session_switch(engine, host.messages)
                elif event == "config-reload":
                    # Config is loaded at engine construction in the plugin.
                    # Reload it from the isolated environment and bind the same session/store.
                    with monkeypatch.context() as patch:
                        patch.setenv("LCM_SUMMARY_TIMEOUT_MS", "23000")
                        loaded = LCMConfig.from_env()
                    assert loaded.summary_timeout_ms == 23_000
                    engine._config = replace(engine._config, summary_timeout_ms=loaded.summary_timeout_ms)
                    host.engine = reopen(engine, tmp_path, engines)
            if index == 4:
                head = next(row["content"] for row in host.messages if isinstance(row["content"], str)
                            and SUMMARY_MARKER in row["content"])
                appended = [{"role": "user", "content": head}]
            ordinary = host.call(appended)
            assert set(ordinary["commit"]) <= {"overflow commit", "fault-late publication"}, (
                f"ordinary call {ordinary['call']} committed: {ordinary['commit']}"
            )
        ordinary_streak = 0
        for call in host.calls[-5:]:
            ordinary_streak = 0 if call["commit"] else ordinary_streak + 1
        for index in range(10):
            if ordinary_streak == 5:
                break
            ordinary = host.call([{"role": "user", "content": f"Overflow follow-up {index}."}])
            ordinary_streak = 0 if ordinary["commit"] else ordinary_streak + 1
        assert ordinary_streak == 5, "could not observe five ordinary calls after the last commit"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence = dict(cell=host.cell, seconds=round(time.perf_counter() - started, 3), error=error,
                        failures=host.failures, calls=host.calls,
                        merged_head_repairs=sum(call["merged_head_repair"] is not None for call in host.calls),
                        overflow_commits=sum("overflow commit" in call["commit"] for call in host.calls))
        (tmp_path.parent / f"rstable-{host.cell.replace('/', '-')}.json").write_bytes(encode(evidence) + b"\n")
    return host


@pytest.mark.parametrize(
    ("uid_mode", "event", "pass_kind"),
    [swap_cell(uid_mode, event, kind) for uid_mode in ("off", "on") for event in EVENTS for kind in PASS_KINDS],
    indirect=["uid_mode"],
)
def test_swap_only_rewrites(exercise):
    details = "\n".join(
        "{cell}: {name}, call {call}, row {row} ({kind}), {before} -> {after}".format(**failure)
        for failure in exercise.failures
    )
    baseline = KNOWN_FAILS.get(exercise.cell)
    if baseline is None:
        assert not exercise.failures, details
        return
    actual = frozenset(failure["name"] for failure in exercise.failures)
    if actual == baseline:
        raise KnownSwapRuleFailure(details)
    if actual:
        pytest.fail(f"{exercise.cell}: expected {sorted(baseline)}, got {sorted(actual)}\n{details}")
    # Return normally when the failure is fixed so the strict marker reports XPASS.
