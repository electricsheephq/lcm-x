"""Bounded decomposition and occurrence matching for identity recognition."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from .store import _normalize_observed_at

# Keep the established diagnostics and caplog namespace across the extraction.
logger = logging.getLogger("hermes_lcm.identity_anchor")

Row = Mapping[str, Any]
KeysOf = Callable[[Row], Iterable[Any]]

_MAX_DECOMPOSITIONS = 3
_DECOMPOSE_BUDGET = 2048  # T3: prefixes visited per decomposition (ingest thread)
_DECOMPOSE_MAX_PARTS = 64
_MATCH_WORK_PER_ITEM, _MATCH_WORK_FLOOR = 32, 4096  # matching search budget per row + occurrence + key (see below)


def _decompositions(content: str, texts: set[str], *, partial: bool) -> list[tuple[list[str], str]] | None:
    """Ordered splits of ``content`` into ``"\\n\\n"``-joined whole ``texts`` (at least two parts).
    ``partial`` also returns ``(parts, remainder)`` for a held prefix followed by a new remainder.
    Stops after ``_MAX_DECOMPOSITIONS`` (the caller only needs to know whether one is unique).
    Iterative and bounded (T3): more than ``_DECOMPOSE_BUDGET`` visited prefixes or
    ``_DECOMPOSE_MAX_PARTS`` parts returns None, "ambiguous": the row is stored whole."""
    out: list[tuple[list[str], str]] = []
    by_head: dict[str, list[str]] = defaultdict(list)
    text: str | None
    short = [text for text in texts if 0 < len(text) < 16]
    for text in texts:
        if len(text) >= 16:
            by_head[text[:16]].append(text)
    stack: list[tuple[int, list[str], Iterator[str]]] = []  # (pos, parts, texts left to try): the recursive walk, iteratively (T3)
    visited = 0

    def enter(pos: int, parts: list[str]) -> bool:
        """One walk() call: False once the budget is spent."""
        nonlocal visited
        if len(out) >= _MAX_DECOMPOSITIONS:
            return True
        visited += 1
        if visited > _DECOMPOSE_BUDGET or len(stack) >= _DECOMPOSE_MAX_PARTS:
            return False
        if partial and parts and pos < len(content):
            out.append((list(parts), content[pos:]))
        stack.append((pos, parts, iter(by_head.get(content[pos:pos + 16], []) + short)))
        return True

    if not enter(0, []):
        return None
    while stack:
        pos, parts, pending = stack[-1]
        text = next(pending, None)
        if text is None:
            stack.pop()
            continue
        end = pos + len(text)
        if not content.startswith(text, pos):
            continue
        if end == len(content) and len(parts) >= 1:
            out.append((parts + [text], ""))
        elif content.startswith("\n\n", end) and end + 2 < len(content) and not enter(end + 2, parts + [text]):
            return None
    return out


def _match_occurrences(rows: Iterable[Row], keys_of: KeysOf, occurrences: Sequence[tuple[Any, Any]]) -> dict[int, Any]:
    """``{store_id: occurrence}``: a maximum matching of stored ``rows`` (store order; a row listed twice, e.g.
    under an alternate stamp, is one row) to view ``occurrences`` ``[(occurrence, key)]`` (view order) by any key
    of ``keys_of(row)``, each side once. A row takes its first key with a free occurrence, else an augmenting
    path frees one; each key's rows then take its occurrences in store order <-> view order, so rows with one
    admissible key get what an in-order walk gave them. Deterministic; O(rows + occurrences + keys) memory.
    Rows with the same keys are one type and a search walks types, not rows. The searches of one call share a
    work budget (key and type visits); once it is spent the remaining rows take only a free key of their own, the
    result stays a valid deterministic matching that may fall short of maximum, and one WARNING (counts) is logged."""
    slots: dict[Any, list[Any]] = defaultdict(list)
    for occurrence, key in occurrences:
        slots[key].append(occurrence)
    merged: dict[int, set[Any]] = {}
    for row in rows:
        merged.setdefault(int(row["store_id"]), set()).update(keys_of(row))
    kinds = {sid: tuple(sorted((k for k in ks if k in slots), key=repr)) for sid, ks in merged.items()}
    load: Counter[Any] = Counter()
    units: dict[tuple[Any, ...], Counter[Any]] = defaultdict(Counter)  # type -> key -> its rows on that key
    matched: set[int] = set()
    movable: dict[Any, dict[tuple[Any, ...], None]] = defaultdict(dict)  # key -> multi-key types with a unit on it (ordered set); nothing else moves
    dead: set[Any] = set()  # never an endpoint or a waypoint again (load never falls; Kuhn's lemma for failed searches)
    budget, work, spent = _MATCH_WORK_PER_ITEM * (len(kinds) + len(occurrences) + len(slots)) + _MATCH_WORK_FLOOR, 0, False
    for sid, own in kinds.items():
        found = next((k for k in own if load[k] < len(slots[k])), None)
        parent: dict[Any, Any] = {}
        if found is None and own and not spent:
            queue: list[Any] = []
            for k in own:
                if k not in dead:
                    parent[k] = (None, None)
                    (queue.append if movable[k] else dead.add)(k)  # full with nothing that can leave: dead
            seen: set[tuple[Any, ...]] = set()
            for k in queue:  # BFS over keys; the queue grows while it is walked
                gone = []
                for kind in movable[k]:
                    spent = work == budget
                    if spent:
                        break
                    work += 1
                    others = [other for other in kind if other != k and other not in dead]
                    if not others:
                        gone.append(kind)  # stuck on k for good
                    elif kind not in seen:
                        seen.add(kind)
                        for other in others:
                            if other in parent:
                                continue
                            spent = work == budget
                            if spent:
                                break
                            work += 1
                            parent[other] = (kind, k)
                            if load[other] < len(slots[other]):
                                found = other
                                break
                            (queue.append if movable[other] else dead.add)(other)
                    if found is not None or spent:
                        break
                for kind in gone:
                    del movable[k][kind]
                if found is not None or spent:
                    break
            if found is None and not spent:
                dead.update(parent)
        if found is None:
            continue
        load[found] += 1
        kind, previous = parent.get(found, (None, None))
        while previous is not None:  # one unit of each type on the path moves one key along
            units[kind][previous] -= 1
            if not units[kind][previous]:
                movable[previous].pop(kind, None)
            units[kind][found] += 1
            movable[found][kind] = None
            found = previous
            kind, previous = parent[found]
        matched.add(sid)
        units[own][found] += 1
        if len(own) > 1:
            movable[found][own] = None
    given: dict[tuple[Any, ...], Counter[Any]] = defaultdict(Counter)  # type -> key -> its rows given that key so far
    taken: Counter[Any] = Counter()
    result = {}
    for sid, kind in kinds.items():  # store order: a type's rows fill its keys by units, keys in order
        if sid in matched:
            k = next(k for k in kind if given[kind][k] < units[kind][k])
            given[kind][k] += 1
            result[sid] = slots[k][taken[k]]
            taken[k] += 1
    if spent:
        logger.warning("LCM identity-anchor matching spent its work budget %d: rows=%d occurrences=%d keys=%d matched=%d",
                       budget, len(kinds), len(occurrences), len(slots), len(result))
    return result


def _reserve_shown(pool: Sequence[Row], forms_of: KeysOf, occurrences: Sequence[tuple[Any, Any]], aliases: Mapping[int, Iterable[Any]] | None = None) -> set[int]:
    """B-ID-1: store ids of ``pool`` rows (store order) the host view shows as their own occurrences
    ``[(occurrence, (stamp, form))]``. A stamped row answers only its own stamp and its recorded alias stamps
    (``aliases``: store id -> stamps, B-ID-3), a NULL-stamped (legacy) row an unstamped occurrence. #583: then a
    NULL row LCM stored before the host stamped it answers a stamped occurrence of its form still unmatched,
    keyed by form alone (a row's keys are its forms, never its stamps: linear), in store order."""
    aliases = aliases or {}

    def keys_of(row: Row) -> set[Any]:
        stamps = {_normalize_observed_at(row.get("observed_at")), *aliases.get(int(row["store_id"]), ())}
        return {(stamp, form) for stamp in stamps for form in forms_of(row)}

    first = _match_occurrences(pool, keys_of, occurrences)
    null = [row for row in pool if _normalize_observed_at(row.get("observed_at")) is None and int(row["store_id"]) not in first]
    taken = set(first.values())
    left = [(occurrence, ("null", key[1])) for occurrence, key in occurrences if key[0] is not None and occurrence not in taken]
    reserved = set(first)
    if null and left:  # over-reservation only stores a composite whole: duplication, never loss
        second = _match_occurrences(null, lambda row: {("null", form) for form in forms_of(row)}, left)
        reserved.update(second)
        rest = [row for row in null if int(row["store_id"]) not in second]
        if rest:
            reserved |= _reroute_for_null(pool, rest, first, taken | set(second.values()), dict(occurrences), keys_of, forms_of)
    return reserved


def _reroute_for_null(pool: Sequence[Row], rest: Sequence[Row], first: Mapping[int, Any], taken: set[Any], key_of: Mapping[Any, Any], keys_of: KeysOf, forms_of: KeysOf) -> set[int]:
    """PR #590: augmentation from pass one for the NULL rows ``rest`` still unreserved. One takes an occurrence of
    its form a stamped row holds only when that row can move to a free occurrence of another of its OWN keys, so a
    stamped row is never unmatched. Free occurrences only fall here: a holder that cannot move never can, and
    leaves the index after one look. Bounded like ``_match_occurrences``; a spent budget keeps what was found."""
    free = Counter(key for occurrence, key in key_of.items() if key[0] is not None and occurrence not in taken)
    current = {int(row["store_id"]): key_of[first[int(row["store_id"])]] for row in pool
               if int(row["store_id"]) in first and key_of[first[int(row["store_id"])]][0] is not None}
    holders: dict[Any, deque[int]] = defaultdict(deque)  # form -> stamped rows holding an occurrence of it, store order
    rows = {int(row["store_id"]): row for row in pool}
    for sid, key in current.items():
        holders[key[1]].append(sid)
    out: set[int] = set()
    budget, work = _MATCH_WORK_PER_ITEM * (len(pool) + len(key_of)) + _MATCH_WORK_FLOOR, 0
    for row in rest:
        for form in sorted(forms_of(row), key=repr):
            queue = holders.get(form)
            while queue and int(row["store_id"]) not in out and work < budget:
                sid = queue.popleft()
                work += 1
                target = next((k for k in sorted(keys_of(rows[sid]), key=repr) if k != current[sid] and free[k] > 0), None)
                if target is not None:  # the holder moves; this NULL row takes the occurrence it left
                    free[target] -= 1
                    current[sid] = target
                    holders[target[1]].append(sid)
                    out.add(int(row["store_id"]))
            if int(row["store_id"]) in out:
                break
        if work >= budget:
            logger.warning("LCM identity-anchor NULL re-route spent its work budget %d: rows=%d occurrences=%d reserved=%d",
                           budget, len(pool), len(key_of), len(out))
            break
    return out
