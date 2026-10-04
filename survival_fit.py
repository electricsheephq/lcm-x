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
from typing import Any, Dict, List, Optional

from .host_uid import _valid_uid
from .host_uid_emit import ADDRESS_KEYS, IDENTITY_KEYS, identity_emit_enabled, record_absorbed_message
from .message_content import normalize_content_value
from .store import _normalize_observed_at
from .tokens import count_message_tokens, count_messages_tokens

logger = logging.getLogger(__name__)

SURVIVAL_FIT_COUNTER_KEY = "survival_fit:counter"
_NOTICE = ("[LCM survival fit: {n} earlier messages (store ids {first}..{last}) are stored verbatim but not in "
           "live context; lcm_grep / lcm_load_session reach them.]")
_WARNING = ("LCM could not summarise part of this conversation in time. To keep the session alive, {n} older "
            "messages left live context; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
            "/lcm doctor reports it.")
_NOTICE_RE = re.compile(r"(?:\n\n)?\[LCM survival fit: \d+ earlier messages \(store ids [^()\]]*\) are stored verbatim "
                        r"but not in live context; lcm_grep / lcm_load_session reach them\.\]")
_PROJECTED = ("[LCM survival fit: this {role} message ({tokens} tokens) is stored verbatim as store id {store_id}; "
              "lcm_expand / lcm_grep reach the full text. (projection {head}/{tail})]")
_PROJECTED_RE = re.compile(r"\[LCM survival fit: this ([a-z_]+) message \((\d+) tokens\) is stored verbatim as store id "
                           r"(\d+); lcm_expand / lcm_grep reach the full text\. \(projection (\d+)/(\d+)\)\]")
_PROJECTED_PREFIX = "[LCM survival fit: this "
_HEAD, _TAIL = 1200, 600  # a projected text keeps its first _HEAD and last _TAIL characters
_RECOVERY_THRESHOLD_SHARE = 0.95  # #608: a recovery attempt's request fits under this share of the threshold


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
        reserve = min(0.9, max(0.0, float(getattr(self._config, "survival_reserve", 0.15) or 0.0)))
        counted = _host_estimate(messages)
        counted = count_messages_tokens(messages) if counted is None else counted
        # system prompt + tools the host adds; more than half the window is a stale or synthetic observation
        overhead = min(window // 2, max(0, int(observed_tokens or 0) - counted))
        ceiling = int(window * (1 - reserve))
        if _positive_int(request_cap):  # #608: a recovery attempt's request under the compaction threshold
            ceiling = min(ceiling, request_cap)
        return max(1, ceiling - overhead)

    def _survival_host_overhead(self, messages, observed_tokens) -> int:
        """The host overhead ``_survival_fit_budget`` subtracts: the host's observed count less the list's own
        measure, at most half the window (#671: the stub-first exit measures with it)."""
        counted = _host_estimate(messages)
        counted = count_messages_tokens(messages) if counted is None else counted
        overhead = max(0, int(observed_tokens or 0) - counted)
        window = int(getattr(self, "context_length", 0) or 0)
        return min(window // 2, overhead) if window > 0 else overhead

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

    def _survival_fit(self, messages, result, observed_tokens, reason: str, *, after_exception: bool = False,
                      window_cap: Optional[int] = None, request_cap: Optional[int] = None):
        """``result``, or the fitted list when ``result`` is over the survival budget."""
        exit_fit = reason.startswith("exit_fit:")
        # #668: an exit fit's budget without the exit cap; a list over it is a session at risk, not headroom
        window_budget = self._survival_fit_budget(messages, observed_tokens) if exit_fit else None
        budget = self._survival_fit_budget(messages, observed_tokens, window_cap, request_cap)
        if budget is None or not isinstance(result, list) or not result or not self._session_id or \
                self._bypasses_lcm_context_management():
            return result
        before = self._survival_measure(result)
        if before <= budget:
            return result
        if exit_fit and window_budget is not None and before > window_budget:
            return self._survival_fit(messages, result, observed_tokens, reason[len("exit_fit:"):],
                                      after_exception=after_exception)
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
        if exit_fit:
            # #668: headroom drops only whole turns between the summary prefix and the fresh tail: never the
            # prefix, a protected fresh-tail row or a projection. The window budget still holds the list.
            cut = self._survival_cut(result, prefix, budget, persisted, reason, store_ids, True,
                                     keep_from=self._fresh_tail_start(result))
            if cut is None:
                logger.info("LCM exit fit skipped: no whole-turn cut between the summary prefix and the fresh tail "
                            "reaches the exit cap (tokens=%d, budget=%d)", before, budget)
                return result
            emergency = False
        else:
            cut = self._survival_cut(result, prefix, budget, persisted, reason, store_ids, True) if prefix > system else None
            emergency = prefix > system and cut is None
            cut = cut or self._survival_cut(result, system, budget, persisted, reason, store_ids, False)
        if cut is None:
            return result
        fitted, count, ids, projected, notice = cut
        after = self._survival_measure(fitted)
        if after >= before:  # nothing stored could leave: the list is already as small as it gets
            logger.warning("LCM survival fit could not shorten the list (before=%d, budget=%d, reason=%s)",
                           before, budget, reason)
            self._survival_record(reason, 0, [], before, before, budget, False, "", shortened=False)
            return result
        if emergency and any(all(m is not row for m in fitted) for row in result[system:prefix]):
            logger.warning("LCM survival fit dropped the summary prefix (emergency: prefix=%d tokens, budget=%d)",
                           self._survival_measure(result[system:prefix]), budget)
        if after_exception or not persisted:
            self._ingest_cursor, self._ingest_cursor_needs_reconcile = 0, True
        else:
            self._ingest_cursor = len(fitted)
        if after > budget:  # still the best list available: returned, but never reported as within budget
            logger.warning("LCM survival fit could not reach budget (after=%d, budget=%d, reason=%s)", after, budget, reason)
        self._survival_record(reason, count, ids, before, after, budget, projected, notice, warn_user=not exit_fit)
        return fitted

    def _survival_summary_identity(self, row: dict, summary: str, uid, proof_kind: str, absorbed_from=None) -> dict:
        """The survival summary part (B1's gate): the carrier's engine uid, else a minted ``survival_summary``
        one; a re-formed carrier absorbs its user row's identity as the host's consecutive-user merge would."""
        if identity_emit_enabled():
            if isinstance(uid, str) and uid:
                row["message_uid"] = uid
            else:
                basis = hashlib.sha256(summary.encode("utf-8")).hexdigest()
                self._mint_engine_uids([(row, "survival_summary", basis, proof_kind)])
            if absorbed_from is not None:
                record_absorbed_message(row, absorbed_from)
        return row

    def _survival_cut(self, result, lead: int, budget: int, persisted: bool, reason: str, store_ids,
                      whole_turns: bool, keep_from: Optional[int] = None):
        """``(fitted, count, ids, projected, notice)`` with ``result[:lead]`` kept, else None. The cut is the
        oldest whole-turn one whose FINAL list (notice included) fits; with ``whole_turns`` there is no
        projection. A carrier ending the kept prefix is split: its user row leaves with its turn, and the
        summary is re-formed around the first kept user row as assembly forms it. ``keep_from``: no cut drops
        ``result[keep_from:]`` (#668: an exit fit keeps the fresh tail)."""
        head, body = list(result[:lead]), list(result[lead:])
        summary = summary_uid = None
        remainder = self._generated_context_carrier_remainder(head[-1]) if whole_turns and head else None
        if remainder is not None:
            carrier = head.pop()
            summary = carrier["content"][:self._verified_lcm_summary_prefix_end(carrier["content"])]
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
                hosts = self._host_uid_host_uids(candidates)
                if len(hosts) == 1:
                    body[0]["message_uid"] = next(iter(hosts))
            if id(carrier) in store_ids:
                store_ids = {**store_ids, id(body[0]): store_ids[id(carrier)]}

        def durable(message) -> bool:  # unproven rows: only DAG-verified scaffold, never a phrase match
            return persisted or id(message) in store_ids or self._is_verified_replay_scaffold_message(message)

        def build(cut: int, kept):
            dropped = body[:cut]
            ids = sorted(store_ids[id(message)] for message in dropped if id(message) in store_ids)
            count = sum(1 for message in dropped if not self._survival_generated(message))
            notice = _NOTICE.format(n=count, first=ids[0] if ids else "-", last=ids[-1] if ids else "-")
            out = list(head)
            if out and out[0].get("role") == "system":  # the notice never edits a generated summary row
                out[0] = {**out[0], "content": self._survival_with_notice(out[0].get("content"), notice)}
            if summary is not None:
                merged = {"role": "user", "content": f"{summary}\n\n{kept[0].get('content')}"} if kept else None
                if (merged and kept[0].get("role") == "user" and isinstance(kept[0].get("content"), str)
                        and any(message.get("role") == "user" for message in kept[1:])
                        and self._generated_context_carrier_remainder(merged) == kept[0]["content"]):
                    self._survival_summary_identity(merged, summary, summary_uid, "carrier", kept[0])  # site 19
                    kept = [merged, *kept[1:]]
                else:
                    out.append(self._survival_summary_identity({"role": "user", "content": summary}, summary,
                                                               summary_uid, "survival_summary"))
            return out + kept or result[-1:], count, ids, notice

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
            if index and not all(durable(message) for message in body[:index]):
                break  # never omit a row that is not durably stored
            if index and covered is not None:  # #738: every dropped row is a covered stored row or verified scaffold
                if not all(store_ids.get(id(m)) in covered or self._is_verified_replay_scaffold_message(m)
                           for m in body[:index]):
                    break  # an unmapped, merged, stubbed or uncovered row stays, and so does every longer region
                if not any(id(m) in store_ids for m in body[:index]):
                    continue  # scaffold alone is not cut; a longer region may add covered stored rows
            if index:
                fitted, count, ids, notice = build(index, body[index:])
                if self._survival_measure(fitted) <= budget:
                    return fitted, count, ids, False, notice
        if whole_turns:
            return None
        # the newest user turn alone is over budget: a bounded projection of it
        cut = users[-1] if users else 0
        if not all(durable(message) for message in body[:cut]):
            logger.warning("LCM survival fit skipped: an over-budget list holds rows not yet stored (reason=%s)", reason)
            return None
        noticed = build(cut, body[cut:])[0][:len(head)]  # the head as returned: its notice counts too
        kept = self._survival_projection(body[cut:], store_ids, budget - self._survival_measure(noticed))
        fitted, count, ids, notice = build(cut, kept)
        return fitted, count, ids, any(new is not old for new, old in zip(kept, body[cut:])), notice

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

    def _survival_projection(self, turn: List[Dict[str, Any]], store_ids, limit: int) -> List[Dict[str, Any]]:
        """The newest turn, its largest stored rows projected until it fits (tool outputs first, then
        others); never a row that is not stored, never an empty list. A projection is a pure function of
        the stored row and the parameters its mark carries (store id, token count, head/tail), so a
        re-ingest recognises it by recomputing it from that row (``_survival_projection_source``)."""
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
            fields = self._survival_projected_fields(row, tokens, _HEAD, _TAIL)
            out[index] = {key: value for key, value in {**message, **fields}.items()
                          if key != "tool_calls" or value}
        return out

    @staticmethod
    def _survival_projected_fields(row: Dict[str, Any], tokens: int, head: int, tail: int) -> Dict[str, Any]:
        """The projection of a stored row: its content as head/tail around the mark (the mark alone when
        short), its tool calls with arguments over ``head`` characters replaced by the mark, and its
        stored tool linkage. Deterministic in (row bytes, tokens, head, tail)."""
        store_id = int(row["store_id"])
        marker = _PROJECTED.format(role=row.get("role"), tokens=tokens, store_id=store_id, head=head, tail=tail)
        text = normalize_content_value(row.get("content")) or ""
        if text:
            text = f"{text[:head]}\n...\n{marker}\n...\n{text[-tail:]}" if len(text) > 2 * head else marker
        calls = row.get("tool_calls")
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except ValueError:
                calls = None
        fields: Dict[str, Any] = {"content": text,
                                  "tool_calls": [SurvivalFitMixin._survival_bounded_call(call, marker, head)
                                                 for call in calls] if isinstance(calls, list) and calls else None}
        if row.get("tool_call_id"):
            fields["tool_call_id"] = row["tool_call_id"]
        return fields

    @staticmethod
    def _survival_bounded_call(call: Any, marker: str, head: int = _HEAD) -> Any:
        """A view copy of a tool call whose arguments are over ``head`` characters: valid JSON carrying the
        provenance notice. The stored row keeps the verbatim arguments; the call stays data."""
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(arguments, str) or len(arguments) <= head:
            return call
        notice = f"{marker} Its tool-call arguments ({len(arguments)} characters) are not in live context."
        return {**call, "function": {**function, "arguments": json.dumps({"lcm_survival_fit": notice})}}

    def _survival_projection_source(self, message: Dict[str, Any], role: str, content: str) -> Optional[Dict[str, Any]]:
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
        haystack = content if _PROJECTED_PREFIX in content else ""
        if isinstance(calls, list):
            haystack += "".join(str((call.get("function") or {}).get("arguments") or "") for call in calls
                                if isinstance(call, dict) and isinstance(call.get("function"), dict)
                                and "lcm_survival_fit" in str(call["function"].get("arguments") or ""))
        store = getattr(self, "_store", None)
        if _PROJECTED_PREFIX not in haystack or store is None:
            return None
        for match in list(_PROJECTED_RE.finditer(haystack))[:4]:
            mark_role, tokens, store_id, head, tail = match.groups()
            if mark_role != role:
                continue
            try:
                row = store.get(int(store_id))
            except Exception:
                row = None
            if not row or str(row.get("role") or "") != role:
                continue
            if not self._survival_stamp_matches(message, row):
                continue
            fields = self._survival_projected_fields(row, int(tokens), int(head), int(tail))
            if (content == fields["content"]
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

    def _survival_stamp_matches(self, message: Dict[str, Any], row: Dict[str, Any]) -> bool:
        """No host stamp; or the source row's own: its observed_at or a recorded alias stamp (normalized as
        the identity anchor normalizes them); or any stamp when the source was stored unstamped (R6-1: a
        host that re-inserts the copy with a fresh stamp, Hermes 0.21.2; the byte rule still decides)."""
        stamp = _normalize_observed_at(message.get("timestamp"))
        observed = _normalize_observed_at(row.get("observed_at"))
        if stamp is None or observed is None or observed == stamp:
            return True
        try:
            aliases = self._store.get_message_relations([int(row["store_id"])], "alt_stamp")
        except Exception:
            return False
        return any(_normalize_observed_at(rel.get("observed_at")) == stamp for rel in aliases)

    def _survival_record(self, reason, count, ids, before, after, budget, projected, notice, *, shortened=True,
                         warn_user=True) -> None:
        """Log the fit, update the doctor counter and warn the user once per conversation."""
        uncovered = 0
        if reason.startswith("exit_fit:"):
            try:  # a diagnostic: its failure never fails the fit
                uncovered = None if len(ids) < count else len(set(ids) - self._store_complete_node_covered(ids))
            except Exception:
                logger.debug("LCM exit fit: the uncovered-row count failed", exc_info=True)
                uncovered = None
        if shortened:
            logger.log(
                logging.INFO if reason.startswith("exit_fit:") else logging.WARNING,
                "LCM survival fit applied (reason=%s, conversation=%s, dropped_rows=%d, store_ids=%s..%s, "
                "uncovered_rows=%s, projected=%s, tokens=%d->%d, budget=%d)",
                reason, self._conversation_id or self._session_id, count, ids[0] if ids else "-", ids[-1] if ids else "-",
                "unknown" if uncovered is None else uncovered, projected, before, after, budget,
            )
        self._last_survival_fit = {"reason": reason, "dropped_rows": count, "notice": notice, "at": time.time(),
                                   "uncovered_rows": uncovered, "reached_budget": after <= budget}

        def counted(record):  # runs inside the store's write transaction: concurrent engines add, never overwrite
            record = record if isinstance(record, dict) else {}
            try:
                count = int(record.get("count") or 0)
            except (TypeError, ValueError, OverflowError):  # #618 item 14: a damaged count restarts the record
                return {**counted({}), "count_lost": True}
            return {"count": count + 1, "last_reason": reason, "last_at": time.time(),
                    "last_conversation": str(self._conversation_id or self._session_id or ""),
                    "last_reached_budget": after <= budget,
                    "last_shortened": shortened,
                    "ever_shortened": shortened or (record.get("ever_shortened", True) if record else False),
                    "unreached_budget_count": int(record.get("unreached_budget_count") or 0) + (after > budget),
                    # fits that projected a row (#601); a record from before the key stays unknown (no key)
                    **({"projected_count": int(record.get("projected_count") or 0) + bool(projected)}
                       if "projected_count" in record or not record.get("count") else {}),
                    **({"count_lost": True} if record.get("count_lost") else {})}  # #618 item 14: kept

        try:
            self._store.update_metadata_json(SURVIVAL_FIT_COUNTER_KEY, counted)
        except Exception:
            logger.warning("LCM survival-fit counter write failed (projected=%s); /lcm doctor under-counts survival fits "
                           "for this store", projected, exc_info=True)
        key = str(self._conversation_id or self._session_id or "")
        if shortened and warn_user and key not in self._survival_fit_warned:
            self._survival_fit_warned.add(key)
            self._survival_fit_pending_warning = (key, _WARNING.format(n=count))  # R6-4: owned by its conversation
            self.emit_automatic_compaction_status = True  # the host asks the hook below once more

    def get_automatic_compaction_status_message(self, *, phase: str, default_message: str, **context: Any):
        """LCM keeps automatic compaction silent; a pending survival-fit warning is shown once."""
        pending = getattr(self, "_survival_fit_pending_warning", None)
        self._survival_fit_pending_warning = None
        self.emit_automatic_compaction_status = False
        if not pending or pending[0] != str(self._conversation_id or self._session_id or ""):
            return None  # R6-4: another conversation's warning is never delivered here
        return pending[1]
