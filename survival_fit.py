"""#582 survival fit: a compaction that cannot bring the list under the model window never costs the session.

When compress() is about to return a list that is still over the window (a publication failure, a sweep
deadline, a no-op, a lock after commit, an exception), the list is fitted on the way out:
- budget = effective window x (1 - LCM_SURVIVAL_RESERVE) minus the host's observed-minus-counted overhead;
- the FINAL list is measured with the host's own request estimator (and LCM's count, whichever is larger)
  and the oldest whole user turns are dropped until it fits; a cut falls only on a user row, so no
  tool result is orphaned;
- when the newest user turn alone is over budget, its largest stored rows are projected (tool outputs
  first): head/tail of the stored text around a mark naming the store id and the projection parameters,
  tool-call arguments over the head size replaced by the mark. A projection is a pure function of the
  stored row and its mark, so a re-ingest (same process or cold resume) recomputes it from that row and
  recognises the copy only on an exact match: it is never stored again, and no new message can take an
  old row's identity unless byte-identical to its projection. The raw rows stay in the store;
- a row that is not durably stored is never omitted (a row the ingest cursor cannot prove stored counts
  as durable only when it is DAG-verified LCM scaffold), and the list is never empty (#91).
- LCM's summary prefix (scaffold or verified carrier rows right after the system slot) stays whenever whole
  oldest turns can leave instead and the final list fits; otherwise the old rule applies and a WARNING says
  so (#650).
It writes no message, node or lifecycle row: only the metadata counter /lcm doctor reads. The global
assembly cap is never set (that would force overflow and trim every good compaction). The notice goes in
the system-prefix slot when the list has one, never as a conversation row; the one-shot user warning goes
through the host's automatic-compaction status hook. LCM_SURVIVAL_FIT=false turns it off.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .host_uid import _valid_uid
from .host_uid_emit import ADDRESS_KEYS, IDENTITY_KEYS, identity_emit_enabled, record_absorbed_message
from .message_analysis import _is_codex_interim
from .message_content import normalize_content_value
from .emission_projection import _NOTICE_BLOCK_RE, survival_notice_block
from .reconcile import ReconcileMixin, _emission_identity
from .store import _normalize_observed_at
from .survival_codec import (
    _HEAD, _TAIL,
    _PROJECTED as _PROJECTED,
    _PROJECTED_RE as _PROJECTED_RE,
    _PROJECTED_PREFIX as _PROJECTED_PREFIX,
    bounded_call, projected_fields, project_message, project_message_legacy, projection_mark_candidates,
)
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

SURVIVAL_FIT_COUNTER_KEY = "survival_fit:counter"
_NOTICE = ("[LCM survival fit: {n} earlier messages (store ids {first}..{last}) are stored verbatim but not in "
           "live context; lcm_grep / lcm_load_session reach them.]")
_WARNING_SUMMARY_FAILED = ("LCM could not summarise part of this conversation in time. To keep the session alive, {n} older "
                           "messages left live context; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
                           "/lcm doctor reports it.")
_WARNING = _WARNING_SUMMARY_FAILED
_WARNING_FITTED = ("To keep this conversation within the model's context window, {n} older messages left live context "
                   "during LCM context fitting; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
                   "/lcm doctor reports why.")
_NOTICE_RANGES = ("[LCM survival fit: {n} earlier messages (store ids {first}..{last}) are stored verbatim but not in "
                  "live context; read with lcm_load_session, next_cursor, and lcm_expand(store_id) if truncated: "
                  "{calls}.]")
_NOTICE_REST = "; and {count} earlier ranges (store ids {first}..{last}); lcm_grep reaches them"
_NOTICE_MAX_RANGES = 8  # #2877: the newest ranges out of view are named with their calls; older ones are counted
_NOTICE_MAX_TOKENS = 300  # the bound for 12-digit store ids and two sessions with ids up to 128 characters
# the notice ends at its own closing '.]' (then a blank line or the end), so a ']' in a session id cannot split it
_NOTICE_RE = re.compile(r"(?:\n\n)?\[LCM survival fit: \d+ earlier messages \(store ids [^()\]]*\) are stored verbatim "
                        r"but not in live context(?:; lcm_grep / lcm_load_session reach them\.\]|; read with "
                        r"lcm_load_session, [^\n]*?\.\](?=\n\n|$))")
# #2896: on a list with no leading system row the notice is one marked block right after the last verified summary
# part of the head (then a separator or the end of the text); the summary verifier admits it there and only there


def _survival_block(parts: str, node_ids, notice: str) -> str:
    """#2896: the marked block for ``notice`` after ``parts``; over the token cap (the wrapper counted) the counted
    notice, which has no call, takes its place."""
    block = survival_notice_block(parts, node_ids, notice)
    if count_tokens(block) > _NOTICE_MAX_TOKENS:
        head = re.match(r"\[LCM survival fit: (\d+) earlier messages \(store ids ([^()]*?)\.\.([^()]*)\)", notice)
        block = survival_notice_block(parts, node_ids, _NOTICE.format(n=head[1], first=head[2], last=head[3]))
    return block


def survival_notice_block_start(content: str, prefix_end: int) -> Optional[int]:
    """#2896: where the marked notice block that ends the verified summary prefix ``content[:prefix_end]`` starts, else
    None (the prefix is its summary parts alone)."""
    start = content.rfind("\n\n<lcm-survival-notice digest=", 0, prefix_end)
    block = _NOTICE_BLOCK_RE.match(content, start) if start >= 0 else None
    return start if block is not None and block.end() == prefix_end else None


_WALK_SQL = (
    "WITH RECURSIVE walk(root, node_id) AS (SELECT CAST(value AS INTEGER), CAST(value AS INTEGER) FROM json_each(?) "
    "UNION SELECT walk.root, CAST(source.value AS INTEGER) FROM walk JOIN summary_nodes AS node ON node.node_id = "
    "walk.node_id, json_each(node.source_ids) AS source WHERE node.source_type != 'messages'), "
    "covered(store_id, root) AS (SELECT CAST(source.value AS INTEGER), walk.root FROM walk JOIN summary_nodes AS node "
    "ON node.node_id = walk.node_id, json_each(node.source_ids) AS source WHERE node.source_type = 'messages') ")
_OUT_OF_VIEW_SQL = _WALK_SQL + (
    "SELECT message.store_id, message.role, length(message.content), group_concat(covered.root) FROM messages AS "
    "message LEFT JOIN covered ON covered.store_id = message.store_id WHERE message.session_id = ? "
    "GROUP BY message.store_id ORDER BY message.store_id")
_COVERED_SQL = _WALK_SQL + (
    "SELECT message.store_id, message.role, length(message.content), group_concat(covered.root) FROM messages AS message "
    "LEFT JOIN covered ON covered.store_id = message.store_id WHERE message.store_id IN (SELECT value FROM json_each(?)) "
    "GROUP BY message.store_id")
# a carry range is half-open (start < store id <= end), read over the source session's own rows (idx_msg_session)
_CARRIED_SQL = "SELECT store_id FROM messages WHERE session_id = ? AND store_id > ? AND store_id <= ? ORDER BY store_id"
_RECOVERY_THRESHOLD_SHARE = 0.95  # #608: a recovery attempt's request fits under this share of the threshold


def _copy_key(role, content, tool_calls, tool_call_id) -> str:
    """#2877: a row's identity for the copy check: role, content, tool calls (replay identity's stable form) and call
    id (not a tool row's name, which the store may not hold; replay identity treats a missing name as compatible)."""
    key = [role, content or "", ReconcileMixin._stable_tool_calls_identity(tool_calls), tool_call_id or ""]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


OutOfViewRead = namedtuple("OutOfViewRead", "roots_read kept order position roots uncovered by_root keys key_roots sessions")


def read_out_of_view(store: Any, session_id: str, node_ids, kept=None, also=()) -> OutOfViewRead:
    """This session's non-system rows in order with the read roots covering each (the statement above), then one bounded
    read of the digests the copy check needs (and the session of each ``also`` id outside this session's rows, with
    the read roots covering it); it answers any shown set from ``kept`` (default all) to ``node_ids`` alike."""
    roots_read = frozenset(int(node) for node in node_ids)
    kept = roots_read if kept is None else frozenset(int(node) for node in kept)
    order, roots, by_root, uncovered, sizes = [], {}, {}, [], {}
    for store_id, role, size, covering in store.connection.execute(
            _OUT_OF_VIEW_SQL, (json.dumps(sorted(roots_read)), session_id)).fetchall():
        if role == "system":
            continue  # the system slot is always carried
        order.append(store_id)
        sizes[store_id] = (role, size)
        if not covering:
            uncovered.append(store_id)
            continue
        roots[store_id] = {int(node) for node in str(covering).split(",")}
        for node in roots[store_id]:
            by_root.setdefault(node, []).append(store_id)
    sessions = dict.fromkeys(int(store_id) for store_id in also if int(store_id) not in sizes)
    if sessions and roots_read:  # another session's rows (a rotation's carried parent rows): shapes and read roots
        for store_id, role, size, covering in store.connection.execute(
                _COVERED_SQL, (json.dumps(sorted(roots_read)), json.dumps(list(sessions)))).fetchall():
            sizes[store_id] = (role, size)
            if covering:
                roots[store_id] = {int(node) for node in str(covering).split(",")}
    # the shapes of the rows that may be named, so a covered row that may be their copy is read
    wanted = {sizes[store_id] for store_id in [*order, *sessions] if store_id in sizes
              and not roots.get(store_id, set()) & kept}
    ids = [store_id for store_id in order if store_id not in roots or sizes[store_id] in wanted] + list(sessions)
    keys, key_roots = {}, {}
    for store_id, session, role, *fields in store.connection.execute(
            "SELECT store_id, session_id, role, content, tool_calls, tool_call_id FROM messages WHERE store_id IN "
            "(SELECT value FROM json_each(?))", (json.dumps(ids),)).fetchall():
        if store_id in sessions:
            sessions[store_id] = session
        content, calls, call_id = fields
        try:  # the column's own JSON layer; the value inside is what the host passed
            calls = json.loads(calls) if calls else None
        except ValueError:
            pass
        keys[store_id] = key = _copy_key(role, content, calls, call_id)
        for node in roots.get(store_id, ()):
            key_roots.setdefault(key, set()).add(node)
    position = {store_id: index for index, store_id in enumerate(order)}
    return OutOfViewRead(roots_read, kept, order, position, roots, uncovered, by_root, keys, key_roots, sessions)


def survival_notice_text(count: int, ranges, rest=None) -> str:
    """The notice for ``ranges`` [(session id, first, last), ...] in store-id order, each with the after_store_id that
    loads it, and ``rest`` (count, first, last), the older ranges counted in one clause. The head names their span."""
    groups = []
    for session, first, last in ranges:
        if not groups or groups[-1][0] != session:
            groups.append((session, []))
        # where each range starts and stops, and a limit that reads it in one page up to lcm_load_session's cap
        groups[-1][1].append(f"after_store_id={first - 1} limit={min(200, last - first + 1)} to {last}")
    # a session id ends at a space and groups at ';', characters ids never hold (their ':' are common)
    calls = "; ".join(f"session_id={session} " + ", ".join(items) for session, items in groups)
    calls += _NOTICE_REST.format(count=rest[0], first=rest[1], last=rest[2]) if rest else ""
    bounds = [x for _, first, last in ranges for x in (first, last)] + (list(rest[1:]) if rest else [])
    return _NOTICE_RANGES.format(n=count, first=min(bounds), last=max(bounds), calls=calls)


def survival_fit_notice(engine: Any, messages, shown_node_ids, *, left=(), carried=(), store_ids=None,
                        read: Optional[OutOfViewRead] = None) -> Optional[str]:
    """#2877: for a list about to be returned whose summary heads show ``shown_node_ids``, every range (in session
    order) of this session's rows out of view, each with the call that loads it, else None. A row is in view when it
    or a byte-equal stored copy is in the list or under a shown node: a row with a visible byte-equal copy is not
    named, except rows this fit dropped. ``left``: store ids that left the list, named in any case if stored (another
    session's under its own id). ``carried``: another session's stored rows (a rotation's carried parent rows), named
    while out of view under their own session id. ``store_ids`` and ``read`` only save work."""
    session, shown = str(engine._session_id or ""), {int(node) for node in shown_node_ids}
    left = sorted({int(store_id) for store_id in left})
    carried = sorted({int(store_id) for store_id in carried} - set(left))
    if store_ids is None:
        store_ids = engine._get_store_id_map_for_messages([m for m in messages if m.get("role") != "system"])
    listed = {store_ids[id(m)] for m in messages if id(m) in store_ids}
    if (read is None or not read.kept <= shown <= read.roots_read
            or any(x not in read.position and x not in read.sessions for x in (*left, *carried))):
        read = read_out_of_view(engine._store, session, shown, also=[*left, *carried, *listed])
    # a listed row with no store id still shows its content (Claude F8)
    in_view = {read.keys[store_id] for store_id in listed if store_id in read.keys} | {
        _copy_key(m.get("role"), normalize_content_value(m.get("content")), m.get("tool_calls"), m.get("tool_call_id"))
        for m in messages if id(m) not in store_ids and m.get("role") != "system"}

    def seen(store_id) -> bool:
        key = read.keys.get(store_id)
        return bool(store_id in listed or read.roots.get(store_id, set()) & shown
                    or key is not None and (key in in_view or read.key_roots.get(key, set()) & shown))

    candidates = set(read.uncovered).union(*(read.by_root.get(node, ()) for node in read.roots_read - shown))
    out = sorted({read.position[x] for x in candidates if not seen(x)} | {read.position[x] for x in left
                                                                           if x in read.position})
    # this session's rows merge in its row order (owner None); another session's (a rotation's parent) by adjacent ids
    other = sorted({*left, *(x for x in carried if not seen(x))})
    rows = [(None, index) for index in out] + [(read.sessions[x], x) for x in other  # a row the store lacks: no call
                                               if x not in read.position and read.sessions.get(x)]
    ranges = []
    for owner, index in rows:
        if ranges and ranges[-1][0] == owner and ranges[-1][2] == index - 1:
            ranges[-1][2] = index
        else:
            ranges.append([owner, index, index])
    if not ranges:
        return None
    ranges = sorted(([session, read.order[a], read.order[b]] if owner is None else [owner, a, b]
                     for owner, a, b in ranges), key=lambda r: r[1])
    named = ranges[-_NOTICE_MAX_RANGES:]
    while True:  # the cap, by the counter the fit's budget uses: the oldest named ranges fold into the counted clause
        rest = ranges[:len(ranges) - len(named)]
        text = survival_notice_text(len(rows), named, (len(rest), min(r[1] for r in rest), max(r[2] for r in rest))
                                    if rest else None)
        if count_tokens(text) <= _NOTICE_MAX_TOKENS:
            return text
        if len(named) == 1:  # one range whose session id alone passes the cap: the counted form, which has no call
            return _NOTICE.format(n=len(rows), first=ranges[0][1], last=max(r[2] for r in ranges))
        named = named[1:]


def survival_counter_store_key(store: Any) -> str:
    """#719: one lost-write marker per database file: the identity the store fixed when it opened the file."""
    return str(getattr(store, "db_identity", None) or store.db_path)


def _carries_survival_notice(raw: Any, text: str) -> bool:
    """A system row the fit put its notice in: appended to string content, or as the last text part of
    list content (whose normalized form is JSON, where the separator is escaped)."""
    if isinstance(raw, list):
        last = raw[-1] if raw else None
        return isinstance(last, dict) and last.get("type") == "text" and str(last.get("text") or "").startswith(
            "[LCM survival fit: ")
    return "\n\n[LCM survival fit: " in text


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _host_estimate(messages) -> Optional[int]:
    """The host's own request estimator (Hermes agent.model_metadata), when the host provides one."""
    try:
        from agent.model_metadata import estimate_messages_tokens_rough
    except Exception:
        return None
    try:
        return int(estimate_messages_tokens_rough(messages))
    except Exception:
        return None


@dataclass(frozen=True)
class FitPlan:
    """A selected fit and the original prefix rows needed by its caller's apply step."""

    fitted: List[Dict[str, Any]]
    count: int
    ids: List[int]
    projected: bool
    notice: str
    emergency: bool
    emergency_span: List[Dict[str, Any]]


def survival_reserve_fraction(config: Any) -> float:
    """Resolve the survival reserve with the existing default, coercion and clamp."""
    return min(0.9, max(0.0, float(getattr(config, "survival_reserve", 0.15) or 0.0)))


class SurvivalFitMixin:
    """Mixed into LCMEngine; reads ``self._store``, ``self._config`` and ``self.context_length``."""

    def _survival_measure(self, messages) -> int:
        host = _host_estimate(messages)
        counted = count_messages_tokens(messages)
        return counted if host is None else max(host, counted)

    def _survival_fit_budget(self, messages, observed_tokens, window_cap: Optional[int] = None,
                             request_cap: Optional[int] = None) -> Optional[int]:
        window = int(getattr(self, "context_length", 0) or 0)
        if window <= 0 or not getattr(self._config, "survival_fit", True):
            return None
        if _positive_int(window_cap):  # #608: the size of a request the provider rejected
            window = min(window, window_cap)
        reserve = survival_reserve_fraction(self._config)
        # system prompt + tools the host adds; more than half the window is a stale or synthetic observation
        overhead = min(window // 2, self._survival_host_overhead(messages, observed_tokens))
        ceiling = int(window * (1 - reserve))
        if _positive_int(request_cap):  # #608: a recovery attempt's request under the compaction threshold
            ceiling = min(ceiling, request_cap)
        return max(1, ceiling - overhead)

    def _survival_host_overhead(self, messages, observed_tokens) -> int:
        """Largest observed overhead for this binding, capped at half the window (#1012 F4, #671)."""
        counted = _host_estimate(messages)
        counted = count_messages_tokens(messages) if counted is None else counted
        overhead = max(0, int(observed_tokens or 0) - counted)
        key = str(getattr(self, "_conversation_id", "") or getattr(self, "_session_id", "") or "")
        previous = getattr(self, "_survival_overhead_observation", None)
        if key:  # the stored peak stays uncapped: a later, larger window must not under-reserve
            overhead = max(overhead, previous[1] if previous and previous[0] == key else 0)
            self._survival_overhead_observation = (key, overhead)
        window = int(getattr(self, "context_length", 0) or 0)
        return min(window // 2, overhead) if window > 0 else overhead

    def _survival_assembly_overhead(self, messages, observed_tokens) -> int:
        """The binding's peak when this pass recorded the host's figure for its own list; otherwise measure
        ``messages`` and record it, as before. Assembly sees the ingest view, where a first-sight stub is far
        smaller than the row the host counted (#1107)."""
        key = str(getattr(self, "_conversation_id", "") or getattr(self, "_session_id", "") or "")
        previous = getattr(self, "_survival_overhead_observation", None)
        if not (getattr(self, "_survival_pass_observed", False) and key and previous and previous[0] == key):
            return self._survival_host_overhead(messages, observed_tokens)
        window = int(getattr(self, "context_length", 0) or 0)
        return min(window // 2, previous[1]) if window > 0 else previous[1]

    def _survival_fit_args(self, messages, observed_tokens, reason: str, recovery: bool, *,
                           automatic: bool = False) -> Dict[str, Any]:
        """#608: the fit's reason and caps. A recovery attempt (the host's ``bypass_cooldown``) fits under the
        rejected request's size (``observed_tokens``, else the input's measure) and under the threshold."""
        threshold = int(getattr(self, "threshold_tokens", 0) or 0)
        if not recovery:
            # #668: normal threshold exits leave headroom; exceptions and below-threshold cleanup do not.
            if automatic and threshold > 0 and (observed_tokens or 0) >= threshold and self._config.survival_fit:
                return {"reason": f"exit_fit:{reason}", "request_cap": int(threshold * _RECOVERY_THRESHOLD_SHARE)}
            return {"reason": reason}
        return {"reason": f"recovery_attempt:{reason}",
                "window_cap": observed_tokens if _positive_int(observed_tokens) else self._survival_measure(messages),
                "request_cap": int(threshold * _RECOVERY_THRESHOLD_SHARE) if threshold > 0 else None}

    def _survival_generated(self, message) -> bool:
        """LCM's own regenerated context (summaries, carriers): derived from stored rows, never a row."""
        return (self._is_replayed_context_scaffold_message(message)
                or self._generated_context_carrier_remainder(message) is not None
                or self._is_context_summary_content(message.get("content")))  # a host summary of stored rows

    def _survival_completed_reply(self, message) -> bool:
        """#798: a completed textual reply needs a host stamp for replay on cold resume."""
        # A Codex Responses interim row replays only with its chain of continuation state.
        return (message.get("role") == "assistant" and not message.get("tool_calls")
                and not _is_codex_interim(message)
                and bool((normalize_content_value(message.get("content")) or "").strip())
                and _normalize_observed_at(message.get("timestamp")) is not None
                and not self._survival_generated(message))

    def _survival_fit(self, messages, result, observed_tokens, reason: str, *, after_exception: bool = False,
                      window_cap: Optional[int] = None, request_cap: Optional[int] = None,
                      summary_failure: str = "", fit_cause: str = ""):
        """``result``, or the fitted list when ``result`` is over the survival budget."""
        if not self._session_id or self._bypasses_lcm_context_management():
            return result
        self._survival_view_read = None  # #2877: one read of the session per fit (a re-planned fit reads again)
        self._survival_notice_unplaced = False
        exit_fit = reason.startswith("exit_fit:")
        # #668: an exit fit's budget without the exit cap; a list over it is a session at risk, not headroom
        window_budget = self._survival_fit_budget(messages, observed_tokens) if exit_fit else None
        budget = self._survival_fit_budget(messages, observed_tokens, window_cap, request_cap)
        if budget is None or not isinstance(result, list) or not result:
            return result
        before = self._survival_measure(result)
        if before <= budget:
            return result
        if exit_fit and window_budget is not None and before > window_budget:
            # #2039: keep exit headroom when whole turns suffice, with the plain at-risk semantics.
            reason = reason[len("exit_fit:"):]
            # No whole-turn cut: the plain fit below applies and logs its own fit, so no "skipped" line here.
            plan, persisted = self._plan_survival_fit(
                messages, result, observed_tokens, before, budget, reason, exit_fit=True,
                after_exception=after_exception, window_cap=window_cap, request_cap=request_cap, log_skip=False)
            if plan is None:
                return self._survival_fit(messages, result, observed_tokens, reason,
                                          after_exception=after_exception, summary_failure=summary_failure, fit_cause=fit_cause)
            exit_fit = False
        else:
            plan, persisted = self._plan_survival_fit(
                messages, result, observed_tokens, before, budget, reason, exit_fit=exit_fit,
                after_exception=after_exception, window_cap=window_cap, request_cap=request_cap)
        if plan is None:
            return result
        fitted = self._apply_survival_fit(result, plan, before, budget, reason,
                                          after_exception=after_exception, persisted=persisted, exit_fit=exit_fit,
                                          summary_failure=summary_failure, fit_cause=fit_cause)
        if fitted is not result:
            from .engine import _stamp_generated_model_only
            # Pure DAG-verified scaffold only: carriers include human text. Keep
            # row/list identities unless this view needs different display metadata.
            generated_rows = [row for row in fitted if row.get("role") != "system"
                              and self._is_verified_replay_scaffold_message(row)]
            _stamp_generated_model_only(fitted, generated_rows, copy_rows=True)
        return fitted

    def _plan_survival_fit(self, messages, result, observed_tokens, before: int, budget: int, reason: str, *,
                           exit_fit: bool, after_exception: bool = False, window_cap: Optional[int] = None,
                           request_cap: Optional[int] = None, log_skip: bool = True) -> tuple[Optional[FitPlan], bool]:
        """Select a fit and preserve the pre-cut durability proof for the caller's apply step."""
        system = 0
        while system < len(result) and isinstance(result[system], dict) and result[system].get("role") == "system":
            system += 1
        # #650: LCM's summary prefix right after the system slot (on Hermes the first row, role user) stays
        # when whole oldest turns can leave instead and the final list fits the budget; else the old rule.
        # Only rows of proven provenance are the prefix, never a phrase match: a summary row whose DAG node
        # exists in this session at the same depth with equal bytes, a preserved objective/todo row, or a
        # verified carrier (#678). A genuine summary that fails verification falls back to the v0.24.6 rule.
        prefix = system
        while prefix < len(result) and isinstance(result[prefix], dict) and (
                self._is_verified_replay_scaffold_message(result[prefix])
                or self._generated_context_carrier_remainder(result[prefix]) is not None):
            prefix += 1
        # The ingest cursor indexes this list with nothing to reconcile: every row of it is persisted,
        # including rows the identity mapper cannot pin to one stored copy (duplicates, stubbed tools).
        # After an exception the cursor proves nothing (this call's writes may have failed or been rolled
        # back): durability is then the store-id map and DAG-verified scaffold only.
        persisted = (not after_exception and not self._ingest_cursor_needs_reconcile
                     and self._ingest_cursor == len(result))
        # one map of the whole conversation part: its occurrence and order evidence needs every row (#650)
        store_ids = self._get_store_id_map_for_messages(result[system:])

        def whole_turn_cut(**kwargs):  # #2896 F2: a cut the block tips over budget is retried without it (main's list)
            return (self._survival_cut(result, prefix, budget, persisted, reason, store_ids, True, **kwargs)
                    or (None if system else self._survival_cut(result, prefix, budget, persisted, reason, store_ids,
                                                               True, notice_block=False, **kwargs)))

        if exit_fit:
            # #668: headroom drops only whole turns between the summary prefix and the fresh tail: never the
            # prefix, a protected fresh-tail row or a projection. The window budget still holds the list.
            cut = whole_turn_cut(keep_from=self._fresh_tail_start(result))
            if cut is None:
                if log_skip:
                    logger.info("LCM exit fit skipped: no whole-turn cut between the summary prefix and the fresh tail "
                                "reaches the exit cap (tokens=%d, budget=%d)", before, budget)
                return None, persisted
            emergency = False
        else:
            cut = whole_turn_cut() if prefix > system else None
            emergency = prefix > system and cut is None
            # #1281: preserve user text when marks cannot fit and the list already fits the window.
            # Recovery caps keep the existing last-resort marks unconditionally (#608).
            hard_cap = (int(self.context_length) - self._survival_host_overhead(messages, observed_tokens)
                        if window_cap is None and request_cap is None else None)
            cut = cut or self._survival_cut(result, system, budget, persisted, reason, store_ids, False,
                                            hard_cap=hard_cap)
        if cut is None:
            return None, persisted
        fitted, count, ids, projected, notice = cut
        return FitPlan(fitted, count, ids, projected, notice, emergency, result[system:prefix]), persisted

    def _survival_summary_identity(self, row: dict, summary: str, uid, proof_kind: str, absorbed_from=None, taken=()) -> dict:
        """The survival summary part (B1's gate): the carrier's engine uid, else a minted ``survival_summary``
        one; a re-formed carrier absorbs its user row's identity as the host's consecutive-user merge would."""
        if identity_emit_enabled():
            if isinstance(uid, str) and uid:
                row["message_uid"] = uid
            else:
                basis = hashlib.sha256(summary.encode("utf-8")).hexdigest()
                self._mint_engine_uids([(row, "survival_summary", basis, proof_kind)],
                                       taken=(message.get("message_uid") for message in taken))
            if absorbed_from is not None:
                record_absorbed_message(row, absorbed_from)
        return row

    def _survival_cut(self, result, lead: int, budget: int, persisted: bool, reason: str, store_ids,
                      whole_turns: bool, keep_from: Optional[int] = None, hard_cap: Optional[int] = None,
                      notice_block: bool = True):
        """``(fitted, count, ids, projected, notice)`` with ``result[:lead]`` kept, else None. The cut is the
        oldest whole-turn one whose FINAL list (notice included) fits; with ``whole_turns`` there is no
        projection. A carrier ending the kept prefix is split: its user row leaves with its turn, and the
        summary is re-formed around the first kept user row as assembly forms it. ``keep_from``: no cut drops
        ``result[keep_from:]`` (#668: an exit fit keeps the fresh tail). ``_survival_cut_carrier``: the
        returned list's re-formed carrier, its summary span and its user row's store id, else None (#1000).
        ``notice_block`` False: no #2896 notice block (the list main forms)."""
        self._survival_cut_carrier = None
        head, body = list(result[:lead]), list(result[lead:])
        summary = summary_uid = formed = head_carrier = summary_nodes = None
        unplaced = False
        remainder = self._generated_context_carrier_remainder(head[-1]) if whole_turns and head else None
        if remainder is not None:
            carrier = head.pop()
            end = self._verified_lcm_summary_prefix_end(carrier["content"])
            start = survival_notice_block_start(carrier["content"], end)  # #2896: an earlier fit's block is replaced
            summary = carrier["content"][:end if start is None else start]
            summary_nodes = self._verified_lcm_summary_prefix(summary)[1]  # #2896: the block's digest binds to them
            body.insert(0, {**carrier, "content": remainder})
            if identity_emit_enabled():  # site 18 (R3-5): the summary part keeps the engine uid; the user-only
                # remainder copies no identity or address, and gets back a single proven host uid
                absorbed = carrier.get("_absorbed_message_uids")
                absorbed = absorbed if isinstance(absorbed, list) else []
                candidates = list(dict.fromkeys(u for u in [carrier.get("message_uid"), *absorbed] if _valid_uid(u)))
                engine = self._host_uid_engine_uids(candidates)
                summary_uid = next((u for u in candidates if u in engine), None)
                for key in IDENTITY_KEYS + ADDRESS_KEYS:
                    body[0].pop(key, None)
                hosts = self._host_uid_host_uids(candidates) - engine  # never an engine uid (F3)
                if len(hosts) == 1:
                    body[0]["message_uid"] = next(iter(hosts))
            if id(carrier) in store_ids:
                store_ids = {**store_ids, id(body[0]): store_ids[id(carrier)]}

        body_ids = {id(m) for m in body}
        # #2877: nodes the kept head shows, nodes each body row shows (an emergency cut drops some), one read per fit
        kept_head = [*head, *([carrier] if remainder is not None else [])]  # the carrier's summary stays
        shows = {id(row): set(self._verified_lcm_summary_prefix(row["content"])[1]) for row in [*kept_head, *body]
                 if row.get("role") == "user" and isinstance(row.get("content"), str)}
        always = set().union(*(shows.get(id(row), ()) for row in kept_head))
        optional = {id(row): shows[id(row)] for row in body if shows.get(id(row))}
        # a rotation's carried parent rows (Codex (b)): named on every fit while they are out of view, not only when cut
        read = getattr(self, "_survival_view_read", None)
        if read is None or read[0] is not result or read[1] is not None and not read[1].kept <= always:
            try:  # one indexed statement per carry range, over the source session's own rows, once per list
                carried = read[2] if read is not None and read[0] is result else [
                    x for source, first, last in self._load_compression_carry_ranges() if source != self._session_id
                    for (x,) in self._store.connection.execute(_CARRIED_SQL, (source, int(first), int(last)))]
                read = (result, read_out_of_view(self._store, self._session_id, always.union(*optional.values()), always,
                                                 also=[*store_ids.values(), *carried]), carried)
            except Exception:  # the notice then names this cut's rows only, as before
                logger.debug("LCM survival fit: the out-of-view read failed", exc_info=True)
                read = (result, None, [])
            self._survival_view_read = read

        def durable(message) -> bool:  # unproven rows: only DAG-verified scaffold, never a phrase match
            return persisted or id(message) in store_ids or self._is_verified_replay_scaffold_message(message)

        def build(cut: int, kept, reply=None):
            nonlocal formed, unplaced, head_carrier
            formed = head_carrier = None
            unplaced = False
            kept_ids = {id(row) for row in kept}  # live rows: identity, one pass (not kept x dropped)
            dropped = [m for m in body[:cut] if m is not reply and id(m) not in kept_ids]
            mapped = sorted(store_ids[id(message)] for message in dropped if id(message) in store_ids)
            # coverage reads the counted rows only: a generated carrier's id never stands in for an unmapped row
            ids = sorted(store_ids[id(message)] for message in dropped
                         if id(message) in store_ids and not self._survival_generated(message))
            count = sum(1 for message in dropped if not self._survival_generated(message))
            # #2877: every row of the session out of view after this cut, with its call; else (or no read) today's
            gone, notice = {id(m) for m in dropped}, None
            try:
                notice = read[1] and survival_fit_notice(
                    self, [*head, *(m for m in body if id(m) not in gone)],
                    always.union(*(nodes for key, nodes in optional.items() if key not in gone)),
                    left=mapped, carried=read[2], store_ids=store_ids, read=read[1])
            except Exception:
                logger.debug("LCM survival fit: the cumulative notice failed", exc_info=True)
            notice = notice or _NOTICE.format(n=count, first=mapped[0] if mapped else "-",
                                              last=mapped[-1] if mapped else "-")
            out = list(head)
            block = ""
            if out and out[0].get("role") == "system":  # the notice never edits a generated summary row
                out[0] = {**out[0], "content": self._survival_with_notice(out[0].get("content"), notice)}
            elif not notice_block:  # #2896 F2: main's list, the notice unplaced
                unplaced = True
            elif summary is not None and summary_nodes:  # #2896: no system row: one marked block in the summary head
                block = _survival_block(summary, summary_nodes, notice)
            elif summary is None:  # the head's last verified glued carrier takes the block (never a standalone row)
                unplaced = True
                for i in reversed(range(len(out))):
                    row = out[i]
                    if id(row) not in store_ids or self._generated_context_carrier_remainder(row) is None:
                        continue
                    end = self._verified_lcm_summary_prefix_end(row["content"])
                    start = survival_notice_block_start(row["content"], end)
                    parts, rest = row["content"][:end if start is None else start], row["content"][end:]
                    block = _survival_block(parts, self._verified_lcm_summary_prefix(parts)[1], notice)
                    out[i] = {**row, "content": parts + block + rest}
                    separator = "\n\n---\n\n" if rest.startswith("\n\n---\n\n") else "\n\n"
                    head_carrier, unplaced = (out[i], parts + block + separator, store_ids[id(row)]), False
                    break
            else:
                unplaced = True
            if summary is not None:
                merged = {"role": "user", "content": f"{summary}{block}\n\n{kept[0].get('content')}"} if kept else None
                if (merged and kept[0].get("role") == "user" and isinstance(kept[0].get("content"), str)
                        and any(message.get("role") == "user" for message in kept[1:])
                        and self._generated_context_carrier_remainder(merged) == kept[0]["content"]):
                    self._survival_summary_identity(merged, summary, summary_uid, "carrier", kept[0],
                                                    taken=out + kept[1:])  # site 19
                    formed = (merged, kept[0], block)
                    kept = [merged, *kept[1:]]
                else:  # a standalone summary row takes no block (#2896: main's shape)
                    out.append(self._survival_summary_identity({"role": "user", "content": summary}, summary,
                                                               summary_uid, "survival_summary", taken=out + kept))
                    unplaced = unplaced or bool(block)
            return out + kept or result[-1:], count, ids, notice

        def accepted(fitted, count, ids, kept, notice):
            """The returned cut: its carrier is recorded only when its user row is a mapped stored row."""
            self._survival_notice_unplaced = unplaced  # #2896: warned only when the fit is applied
            if formed is not None and id(formed[1]) in store_ids:
                self._survival_cut_carrier = (formed[0], f"{summary}{formed[2]}\n\n", store_ids[id(formed[1])])
            elif head_carrier is not None:  # #2896: the head carrier holding the block is the fit's emission
                self._survival_cut_carrier = head_carrier
            return fitted, count, ids, any(id(m) not in body_ids for m in kept), notice

        def keep_reply(cut: int, kept):
            reply = body[cut - 1] if cut else None
            if reply is not None and self._survival_completed_reply(reply):
                candidate = kept if any(row is reply for row in kept) else [reply, *kept]
                if self._survival_measure(build(cut, candidate, reply)[0]) <= budget:
                    return candidate, reply
                # v0.26.x recognises head/tail, but not #917's middle-band head+mark reply. Whole-turn cuts
                # (exit fits) promise no projection: the reply stays verbatim or goes.
                if not whole_turns and len(normalize_content_value(reply.get("content")) or "") > 2 * _HEAD:
                    row = self._store.get(store_ids[id(reply)]) if id(reply) in store_ids else None
                    if row and row.get("role") == "assistant":
                        fields = self._survival_projected_fields(row, count_message_tokens(reply), _HEAD, _TAIL)
                        view = {k: v for k, v in {**reply, **fields}.items() if k != "tool_calls" or v}
                        candidate = [view, *kept]
                        if self._survival_measure(build(cut, candidate, reply)[0]) <= budget:
                            return candidate, reply
            return kept, None

        users = [i for i, message in enumerate(body) if isinstance(message, dict) and message.get("role") == "user"]
        covered = None
        if reason.startswith("exit_fit:"):  # #738: an exit fit drops only rows a summary node covers
            droppable = body if keep_from is None else body[:max(0, keep_from - len(head))]
            try:
                covered = self._store_complete_node_covered([store_ids[id(m)] for m in droppable if id(m) in store_ids])
            except Exception:  # unknown coverage drops nothing
                logger.debug("LCM exit fit: the coverage read failed; no cut", exc_info=True)
                covered = set()
        for index in users:
            if keep_from is not None and len(head) + index > keep_from:
                break  # the cut would drop a protected row
            kept, reply = keep_reply(index, body[index:]) if index == users[-1] else (body[index:], None)
            dropped = [m for m in body[:index] if m is not reply]
            if index and not all(durable(message) for message in dropped):
                break  # never omit a row that is not durably stored
            if index and covered is not None:  # #738: every dropped row is a covered stored row or verified scaffold
                if not all(store_ids.get(id(m)) in covered or self._is_verified_replay_scaffold_message(m)
                           for m in dropped):
                    break  # an unmapped, merged, stubbed or uncovered row stays, and so does every longer region
                if not any(id(m) in store_ids for m in dropped):
                    continue  # scaffold alone is not cut; a longer region may add covered stored rows
            if index:
                fitted, count, ids, notice = build(index, kept, reply)
                if self._survival_measure(fitted) <= budget:
                    return accepted(fitted, count, ids, kept, notice)
        if whole_turns:
            return None
        # the newest user turn alone is over budget: a bounded projection of it
        cut = users[-1] if users else 0
        if not all(durable(message) for message in body[:cut]):
            logger.warning("LCM survival fit skipped: an over-budget list holds rows not yet stored (reason=%s)", reason)
            return None
        noticed = build(cut, body[cut:])[0][:len(head)]  # the head as returned: its notice counts too
        head_tokens = self._survival_measure(noticed)
        kept = self._survival_projection(body[cut:], store_ids, budget - head_tokens,
                                         hard_cap=None if hard_cap is None else hard_cap - head_tokens)
        kept, reply = keep_reply(cut, kept)
        fitted, count, ids, notice = build(cut, kept, reply)
        return accepted(fitted, count, ids, kept, notice)

    @staticmethod
    def _survival_with_notice(content: Any, notice: str) -> Any:
        """The system slot with ``notice``, replacing an earlier fit's notice (one notice, the newest)."""
        if isinstance(content, list):
            kept = [part for part in content if not (isinstance(part, dict) and part.get("type") == "text"
                                                     and _NOTICE_RE.fullmatch(str(part.get("text") or "")))]
            return kept + [{"type": "text", "text": notice}]
        text = _NOTICE_RE.sub("", "\n\n" + (normalize_content_value(content) or ""))[2:] \
            if content is not None else ""
        return f"{text}\n\n{notice}"

    def _survival_projection(self, turn: List[Dict[str, Any]], store_ids, limit: int,
                             hard_cap: Optional[int] = None) -> List[Dict[str, Any]]:
        """The newest turn, its largest stored rows projected until it fits (tool outputs
        first, then others; user rows last, and only to fit); never an unstored row,
        never an empty list. A projection is a pure function of the stored row and
        its mark's parameters (store id, tokens, head/tail), so a re-ingest recognises
        it by recomputing it from that row (``_survival_projection_source``).
        With a hard cap, discard unsuccessful user marks if the pre-user list
        already fits it; no hard cap retains the legacy/recovery behaviour.
        """
        out = list(turn)
        order = sorted(range(len(out)), key=lambda i: (out[i].get("role") != "tool", -count_message_tokens(out[i])))
        for index in order:
            if self._survival_measure(out) <= limit:
                break
            message, source = out[index], turn[index]
            tokens = count_message_tokens(message)
            if id(source) not in store_ids or tokens < 256:
                continue
            row = self._store.get(store_ids[id(source)])
            if not row or str(row.get("role") or "") != str(message.get("role") or ""):
                continue
            projected = project_message(message, row, tokens)
            if count_message_tokens(projected) < tokens:
                out[index] = projected
        for roles in (("tool", "assistant"), ("user",)):
            if self._survival_measure(out) <= limit:
                break
            before_user = (list(out) if roles == ("user",) and hard_cap is not None
                           and self._survival_measure(out) <= hard_cap else None)
            # #1220: exhaust head forms before short-row marks; user rows last, and only to fit.
            for index in order:
                source = turn[index]
                if source.get("role") not in roles:  # skip before measuring: the user pass passes every other row
                    continue
                if self._survival_measure(out) <= limit:
                    break
                tokens = count_message_tokens(source)
                if id(source) not in store_ids or tokens < 256:
                    continue
                row = self._store.get(store_ids[id(source)])
                if (not row or row.get("role") != source.get("role")
                        or len(normalize_content_value(row.get("content")) or "") > 2 * _HEAD):
                    continue
                projected = project_message_legacy(out[index], row, tokens)
                if count_message_tokens(projected) < count_message_tokens(out[index]):
                    out[index] = projected
            if before_user is not None and self._survival_measure(out) > limit:
                out = before_user
        return out

    _survival_projected_fields = staticmethod(projected_fields)
    _survival_bounded_call = staticmethod(bounded_call)

    def _survival_projection_source(self, message: Dict[str, Any], role: str, content: str, *, store=None) -> Optional[Dict[str, Any]]:
        """The stored row ``message`` is a projection of, else None. Self-verifying: each mark names a store
        id and the projection parameters; the row is loaded and its projection recomputed from its stored
        bytes, and the message must agree with it on every key the projection defines or the store keeps:
        role, content, tool_call_id, tool_calls (names and arguments), the tool name (tool rows), and the
        host timestamp. A copy of a projection keeps the source's stamp (the host copies keep timestamps):
        a message with a stamp other than the row's observed_at or one of its recorded alias stamps is a
        new occurrence (R5-1). An unstamped message is judged on the rest. Not compared: ``name`` on other
        roles (a participant label the store does not keep) and host-private or reasoning keys, which the
        projection copies from the host message unchanged and the store never holds.
        A genuinely new message can match only by being byte-identical to a projection of a stored row
        under that row's own stamp."""
        calls = message.get("tool_calls")
        matches = projection_mark_candidates(content, calls)
        target = {} if store is None else dict(store=store)
        store = getattr(self, "_store", None) if store is None else store
        if store is None:
            return None
        for match in matches:
            mark_role, tokens, store_id, head, tail = match.groups()
            if mark_role != role:
                continue
            try:
                row = store.get(int(store_id))
            except Exception:
                row = None
            if not row or str(row.get("role") or "") != role:
                continue
            if not self._survival_stamp_matches(message, row, **target):
                continue
            fields = self._survival_projected_fields(row, int(tokens), int(head), int(tail))
            legacy_fields = self._survival_projected_fields(row, int(tokens), int(head), int(tail), legacy=True)
            # The marked form exists only for rows with stored calls (a storage-only call projection).
            marked_fields = self._survival_projected_fields(row, int(tokens), int(head), int(tail), mark_calls=True) \
                if fields["tool_calls"] else fields
            if (content in (fields["content"], legacy_fields["content"], marked_fields["content"])
                    and str(message.get("tool_call_id") or "") == str(row.get("tool_call_id") or "")
                    and (calls or None) == fields["tool_calls"]
                    and (role != "tool" or str(message.get("tool_name") or message.get("name") or "")
                         == str(row.get("tool_name") or ""))):
                return row
        return None

    def _survival_projection_followers(self, messages, idx: int, row: Dict[str, Any], stamps) -> list:
        """B-ROLL-1 (rc2): ``[(index, stored_row), ...]`` for the unstamped non-user rows the view carries
        right after ``messages[idx]``, a replayed projection of ``row``, that are exactly the rows stored
        right after ``row``, in order; the first row that differs, is stamped or is a user/system row ends
        it. A projected newest user row has no positional replay proof, so without this its replies are
        stored again on a cold resume."""
        role = str(messages[idx].get("role") or "")
        source = self._survival_projection_source(messages[idx], role, normalize_content_value(
            messages[idx].get("content")) or "")
        if source is None or int(source["store_id"]) != int(row["store_id"]):
            return []  # the replies follow the row the projection names, never an identical earlier one
        if not str(self._conversation_id or "").strip():
            return []  # no active conversation to scope the read by: a blank scope would read every one
        stored = self._store.get_range(str(row["session_id"]), start_id=int(row["store_id"]) + 1,
                                       limit=max(len(messages) - idx - 1, 1),  # the active conversation only
                                       conversation_id=self._conversation_id, include_blank_conversation=True)
        out = []
        for k, stored_row in zip(range(idx + 1, len(messages)), stored):
            if (k in stamps or str(messages[k].get("role") or "") in ("user", "system")
                    or self._message_replay_identity(messages[k], strip_carrier=False)
                    != self._message_replay_identity(stored_row, stored_row=True)):
                break
            out.append((k, stored_row))
        return out

    def _survival_stamp_matches(self, message: Dict[str, Any], row: Dict[str, Any], *, store=None) -> bool:
        """No host stamp; or the source row's own: its observed_at or a recorded alias stamp (normalized as
        the identity anchor normalizes them); or any stamp when the source was stored unstamped (R6-1: a
        host that re-inserts the copy with a fresh stamp, Hermes 0.21.2; the byte rule still decides)."""
        stamp = _normalize_observed_at(message.get("timestamp"))
        observed = _normalize_observed_at(row.get("observed_at"))
        if stamp is None or observed is None or observed == stamp:
            return True
        try:
            aliases = (self._store if store is None else store).get_message_relations([int(row["store_id"])], "alt_stamp")
        except Exception:
            return False
        return any(_normalize_observed_at(rel.get("observed_at")) == stamp for rel in aliases)

    def _survival_record(self, reason, count, ids, before, after, budget, projected, notice, *, shortened=True,
                         warn_user=True, summary_failure: str = "", fit_cause: str = "") -> None:
        """Log the fit, update the doctor counter and warn the user once per conversation."""
        # Evidence travels with this pass, independent of its final status or the engine-wide budget.
        summary_failed = bool(summary_failure)
        cause = summary_failure or fit_cause
        uncovered = 0
        if shortened:
            try:  # a diagnostic: its failure never fails the fit
                uncovered = None if len(ids) < count else len(set(ids) - self._store_complete_node_covered(ids))
            except Exception:
                logger.debug("LCM survival fit: the uncovered-row count failed", exc_info=True)
                uncovered = None
        if shortened:
            logger.log(
                logging.INFO if reason.startswith("exit_fit:") else logging.WARNING,
                "LCM survival fit applied (reason=%s, conversation=%s, dropped_rows=%d, store_ids=%s..%s, "
                "uncovered_rows=%s, projected=%s, tokens=%d->%d, budget=%d)",
                reason, self._conversation_id or self._session_id, count, ids[0] if ids else "-", ids[-1] if ids else "-",
                "unknown" if uncovered is None else uncovered, projected, before, after, budget,
            )
            if getattr(self, "_survival_notice_unplaced", False):  # #2896 (c): those rows stay with no pointer
                logger.warning("LCM survival fit: no system row and no glued carrier for the notice; it is not placed")
        self._last_survival_fit = {"reason": reason, "dropped_rows": count, "notice": notice, "at": time.time(),
                                   **({"cause": cause} if cause else {}),
                                   "uncovered_rows": uncovered, "reached_budget": after <= budget}
        # Per store: absent means no lost shortened write; False: lost drop-only fit; True: lost projection.
        lost_by_store = getattr(self, "_survival_counter_write_lost", {})
        self._survival_counter_write_lost = lost_by_store
        store_key = survival_counter_store_key(self._store)
        lost = lost_by_store.get(store_key)

        def counted(record):  # runs inside the store's write transaction: concurrent engines add, never overwrite
            record = record if isinstance(record, dict) else {}
            try:
                count = int(record.get("count") or 0)
            except (TypeError, ValueError, OverflowError):  # #618 item 14: a damaged count restarts the record
                return {**counted({}), "count_lost": True}
            return {"count": count + 1, "last_reason": reason, "last_at": time.time(),
                    **({"last_cause": cause} if cause else {}),
                    "last_conversation": str(self._conversation_id or self._session_id or ""),
                    "last_reached_budget": after <= budget,
                    "last_uncovered_rows": uncovered,
                    "last_shortened": shortened,
                    "ever_shortened": shortened or lost is not None or (record.get("ever_shortened", True) if record else False),
                    "unreached_budget_count": int(record.get("unreached_budget_count") or 0) + (after > budget),
                    # fits that projected a row (#601); a record from before the key stays unknown (no key)
                    **({"projected_count": int(record.get("projected_count") or 0) + bool(projected)}
                       if lost is not True and ("projected_count" in record or not record.get("count")) else {}),
                    **({"uncovered_fit_count": int(record.get("uncovered_fit_count") or 0)
                       + bool(shortened and uncovered is not None and uncovered > 0)}
                       if "uncovered_fit_count" in record or not record.get("count") else {}),
                    # shortened fits whose coverage could not be read: uncovered_fit_count is a lower bound while > 0
                    **({"unknown_coverage_fit_count": int(record.get("unknown_coverage_fit_count") or 0)
                       + bool(shortened and uncovered is None)}
                       if "unknown_coverage_fit_count" in record or not record.get("count") else {}),
                    **({"count_lost": True} if record.get("count_lost") else {})}  # #618 item 14: kept

        try:
            if self._store.update_metadata_json(SURVIVAL_FIT_COUNTER_KEY, counted) is None:
                raise RuntimeError("survival-fit counter was not written")
        except Exception:
            if shortened:
                lost_by_store[store_key] = bool(projected) or lost is True
            logger.warning("LCM survival-fit counter write failed (projected=%s); /lcm doctor under-counts survival fits "
                           "for this store", projected, exc_info=True)
        else:
            lost_by_store.pop(store_key, None)
        key = str(self._conversation_id or self._session_id or "")
        if shortened and warn_user and key not in self._survival_fit_warned:
            self._survival_fit_warned.add(key)
            warning = _WARNING_SUMMARY_FAILED if summary_failed else _WARNING_FITTED
            self._survival_fit_pending_warning = (key, warning.format(n=count))  # R6-4: owned by its conversation
            self.emit_automatic_compaction_status = True  # the host asks the hook below once more

    def get_automatic_compaction_status_message(self, *, phase: str, default_message: str, **context: Any):
        """LCM keeps automatic compaction silent; a pending survival-fit warning is shown once."""
        pending = getattr(self, "_survival_fit_pending_warning", None)
        self._survival_fit_pending_warning = None
        self.emit_automatic_compaction_status = False
        if not pending or pending[0] != str(self._conversation_id or self._session_id or ""):
            return None  # R6-4: another conversation's warning is never delivered here
        return pending[1]

    def _apply_survival_fit(self, result, plan: FitPlan, before: int, budget: int, reason: str, *,
                            after_exception: bool, persisted: bool, exit_fit: bool,
                            summary_failure: str = "", fit_cause: str = ""):
        """Apply the selected fit's diagnostics, cursor updates and carrier registration."""
        fitted, count, ids, projected, notice = plan.fitted, plan.count, plan.ids, plan.projected, plan.notice
        emergency = plan.emergency
        evidence = {"summary_failure": summary_failure, "fit_cause": fit_cause} if summary_failure or fit_cause else {}
        after = self._survival_measure(fitted)
        if after >= before:  # nothing stored could leave: the list is already as small as it gets
            logger.warning("LCM survival fit could not shorten the list (before=%d, budget=%d, reason=%s)",
                           before, budget, reason)
            self._survival_record(reason, 0, [], before, before, budget, False, "", shortened=False, **evidence)
            return result
        if emergency and any(all(m is not row for m in fitted) for row in plan.emergency_span):
            logger.warning("LCM survival fit dropped the summary prefix (emergency: prefix=%d tokens, budget=%d)",
                           self._survival_measure(plan.emergency_span), budget)
        if after_exception or not persisted:
            self._frontier_holder().invalidate()
        else:
            self._frontier_holder().adopt(len(fitted))
        if after > budget:  # still the best list available: returned, but never reported as within budget
            logger.warning("LCM survival fit could not reach budget (after=%d, budget=%d, reason=%s)", after, budget, reason)
        self._survival_record(reason, count, ids, before, after, budget, projected, notice, warn_user=not exit_fit, **evidence)
        formed = getattr(self, "_survival_cut_carrier", None)
        if formed is not None:  # #1000: the re-formed carrier is LCM's emission, as assembly registers its own;
            merged, span, store_id = formed  # the proof finalizes it apart from the pending candidates (output order)
            self._survival_fit_emission = {"kind": "carrier", "span": span, "retained_source": {"store_id": store_id},
                                           "full_identity": _emission_identity(merged), "row": merged}
        return fitted
