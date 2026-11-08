"""multiset-v1 lossless bar, ported from the v0.24.3-rc1 gauntlet ``lossless_bar_multiset.py``.

Key = (role, sha256(edge-stripped content)) over non-empty B2 user/assistant rows; internal whitespace is exact.
Phase C alone keeps assistant bytes exact; Hermes strips stored content, so its drivers must strip ``raw_answer``.
No NFC or CRLF
normalisation is applied: neither host nor plugin performs one on message content. Per key the
stored row count must equal the expected (transcript) count: fewer = loss (deficit), more = duplicates
(surplus). Repeated identical items are fine as long as the multiplicity matches. A stored-only key is
surplus and fails; a split assistant answer (one reply stored as two adjacent rows) is reported apart, as in
the source, and fails too (its fragments are stored-only keys). ``host`` (D-A, scorers/host_parity.py) licenses a
user-row surplus only up to what the host's own state.db holds in the lineage; ``None`` (no host evidence) licenses
nothing.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import importlib.util
import json
import re
import sys
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path


def stripped_replay_content(content):
    """The product's replay transform (escalation.stripped_replay_content), loaded on first use.

    Bench-only runs do not register the plugin package, so register it here the way tests/conftest.py does
    (without executing its plugin entry point); the scorer never re-implements the transform."""
    module = sys.modules.get("hermes_lcm.escalation")
    if module is None:
        if "hermes_lcm" not in sys.modules:
            root = Path(__file__).resolve().parents[4]
            spec = importlib.util.spec_from_file_location(
                "hermes_lcm", root / "__init__.py", submodule_search_locations=[str(root)])
            package = importlib.util.module_from_spec(spec)
            package.__path__ = [str(root)]
            sys.modules["hermes_lcm"] = package
        module = importlib.import_module("hermes_lcm.escalation")
    return module.stripped_replay_content(content)


def norm(text: str) -> str:
    return (text or "").strip()


def h(text: str) -> str:
    return hashlib.sha256(norm(text).encode()).hexdigest()


def row_key(role: str, text: str) -> tuple:
    return role, h(text) if role == "user" else hashlib.sha256((text or "").encode()).hexdigest()


def licence(key: tuple, surplus: int, expected: int, host: dict | None, store_ids: list, spent: int = 0) -> dict | None:
    """D-A: licensed = min(surplus, host_count(k) - expected(k) - spent(k)), floored at 0; user rows only, never a
    deficit. ``spent``: host occurrences of k already used as a paired merge record's constituents (#823)."""
    held = (host or {}).get(key) if key[0] == "user" else None
    n = min(surplus, held["n"] - expected - spent) if held else 0
    return {"role": key[0], "sha256": key[1], "expected": expected, "stored": expected + surplus, "host": held["n"],
            "licensed": n, "store_ids": sorted(store_ids)[:10], "host_row_ids": sorted(held["ids"])[:10],
            "tags": sorted(held.get("tags", []))} \
        if n > 0 else None


COVER_STEPS = 20000  # #804: expansion budget for one composite; real composites take a few dozen
COVER_PARTS = 64  # #804: most parts in one cover (bounds the recursion); real composites have 2-5


def licensed_parts(text: str, licences: dict, times: int) -> dict | None:
    """#804: split a held user composite at its "\n\n" joins into >= 2 parts that are each a host-licensed user key in
    this lineage, with ``times`` licences per use still available; returns {key: uses} or None. The host merges
    consecutive user turns as R + "\n\n" + U, so only those joins are cut points; each part uses the same edge-strip
    key rule. Capacity is checked while searching, so a cover that over-spends a licence never hides a valid one.
    Overlapping cuts in a longer newline run are kept: a raw part may end or start with whitespace, which the key
    rule strips. The search is fail-closed: past ``COVER_STEPS`` expansions or ``COVER_PARTS`` parts it gives up
    and the composite stays a deficit."""
    starts = [0] + [i + 2 for i in range(len(text) - 1) if text.startswith("\n\n", i)]
    ends = [s - 2 for s in starts[1:]] + [len(text)]
    keys: dict[tuple, tuple] = {}
    failed: set = set()
    steps = [COVER_STEPS]

    def key_of(k: int, j: int) -> tuple:
        if (k, j) not in keys:
            keys[(k, j)] = ("user", h(text[starts[k]:ends[j]]))
        return keys[(k, j)]

    def cover(k: int, used: Counter) -> list | None:  # parts covering text[starts[k]:] within the licences left
        state = (k, tuple(sorted((key, n) for key, n in used.items() if n)))
        if state in failed or steps[0] <= 0 or sum(used.values()) >= COVER_PARTS:  # depth = parts used so far
            return None
        steps[0] -= 1
        for j in range(k, len(ends)):
            key = key_of(k, j)
            if key not in licences or licences[key]["licensed"] < (used[key] + 1) * times:
                continue
            if j == len(ends) - 1:
                return [key]
            used[key] += 1
            rest = cover(j + 1, used)
            used[key] -= 1
            if rest is not None:
                return [key, *rest]
        failed.add(state)
        return None

    parts = cover(0, Counter())
    return dict(Counter(parts)) if parts and len(parts) >= 2 else None


def merge_witnesses(host, events, group):
    """Attach observed merges independently of surplus licences (generated carriers license no surplus)."""
    histories = defaultdict(lambda: defaultdict(list))
    for index, event in enumerate(events):
        if event.get("event") in ("host_merge", "engine_carrier") and event.get("session"):
            witness = {"witness": {"event_index": index, "phase": event.get("phase"), "turn": event.get("turn")},
                       "spans": event.get("spans", []), "bytes": event.get("bytes"), "kind": event["event"]}
            histories[group(event["session"])][event["sha256"]].append(witness)
    if host is None:  # no host DB / lineage evidence
        return
    for g, history in histories.items():
        keys = host.setdefault(g, {"keys": {}, "tags": {}})["keys"]
        for digest, witnesses in history.items():
            witnessed_merges = [w for w in witnesses if w["kind"] == "host_merge"]
            if not witnessed_merges:
                continue  # a carrier verifier alone grants no host-merge credit
            held = keys.setdefault(("user", digest), {"n": 0, "ids": []})
            held.update(span_witnesses=witnessed_merges, merge_history=history, span_capacity=max(1, held["n"]))
            # An exact observed newline join can also prove the existing split-parts case. Replays
            # of the same join are one witness, not independent occurrence capacity.
            if not held.get("merges"):
                for witness in witnessed_merges:
                    offset, parts = 0, []
                    for span in witness["spans"]:
                        if span.get("byte_offset") != offset:
                            break
                        parts.append(("user", span["sha256"]))
                        offset += span["bytes"] + 2
                    else:
                        if len(parts) >= 2 and offset - 2 == witness["bytes"]:
                            held["merges"] = [{"parts": parts, "witness": witness["witness"]}]
                            break


def merged_boundaries(raw, witness, history):
    """Expand only earlier observed input joins, checking every nested slice hash before using its boundaries."""
    pending, boundaries = [(0, witness)], set()
    scheduled = {(0, witness["witness"]["event_index"])}
    steps = 0
    while pending and steps < COVER_STEPS:
        base, current = pending.pop()
        steps += 1
        for span in current.get("spans", []):
            offset, size = span.get("byte_offset"), span.get("bytes")
            if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0:
                continue
            start, end = base + offset, base + offset + size
            if end > len(raw) or hashlib.sha256(raw[start:end]).hexdigest() != span.get("sha256"):
                continue
            boundaries.add((start, end))
            if len(boundaries) > COVER_PARTS:
                return []  # fail closed on oversized or ambiguous evidence
            for child in history.get(span["sha256"], []):
                child_index = child["witness"]["event_index"]
                state = (start, child_index)
                if child_index < current["witness"]["event_index"] and state not in scheduled:
                    if len(scheduled) >= COVER_STEPS:
                        return []
                    scheduled.add(state)
                    pending.append((start, child))
    return sorted(boundaries) if not pending else []


def embedded_credits(texts, want, stored_rows, stored, paired, host):
    """Reserve nonoverlapping byte spans proved by the host's actual merge pass, within one lineage.

    Exact rows are reserved first. Host copies do not create unbounded occurrences: candidate rows are
    capped by that key's host multiplicity. A hash co-presence or substring alone proves no merge.
    """
    used, choices, counts = defaultdict(list), defaultdict(dict), Counter()
    direct = Counter()
    slots = Counter()
    for sid, _session, role, content in sorted(stored_rows):
        key = (role, h(content))
        if role != "user" or sid not in stored.get(key, []):
            continue
        raw = norm(content).encode()
        if direct[key] < want[key]:
            direct[key] += 1
            used[sid].append((0, len(raw)))
        held = (host or {}).get(key, {})
        slots[key] += 1
        if slots[key] > held.get("span_capacity", held.get("n", 0)):
            continue
        witnesses = held.get("span_witnesses", [])
        for witness in witnesses:
            boundaries = merged_boundaries(raw, witness, held.get("merge_history", {}))
            starts, ends = sorted({a for a, _b in boundaries}), {b for _a, b in boundaries}
            for part in sorted(want):
                if len(stored.get(part, [])) + paired[part] >= want[part]:
                    continue
                if part[0] != "user":
                    continue
                target = norm(texts[part]).encode()
                for offset in starts:
                    end = offset + len(target)
                    if end not in ends or raw[offset:end] != target or any(
                            offset < b and a < end for a, b in used[sid]):
                        continue
                    choices[part].setdefault((sid, offset, end), {"role": "user", "sha256": part[1], "store_id": sid,
                                    "byte_offset": offset, "bytes": len(target), "composite_sha256": key[1],
                                    "provenance": "host_merge_span", "witness": witness["witness"]})
    # Canonical constrained-first search: at most COVER_PARTS levels and COVER_STEPS states.
    # Keep the best witnessed assignment if the budget expires; never reuse overlapping bytes.
    units = [part for part in sorted(choices, key=lambda p: (len(choices[p]), -len(norm(texts[p]).encode()), p))
             for _ in range(min(want[part] - len(stored.get(part, [])) - paired[part], len(choices[part])))]
    units = units[:COVER_PARTS]
    best, best_score, steps = [], -1, 0
    needed = sum(max(0, n - len(stored.get(p, [])) - paired[p]) for p, n in want.items())

    def coverage(selected):
        """Rank spans with the same canonical, disjoint parts fallback score() will actually use."""
        selected_counts = Counter((c["role"], c["sha256"]) for c in selected)
        reserved_ids = {c["store_id"] for c in selected}
        spare = Counter({p: sum(sid not in reserved_ids for sid in ids[want[p]:]) for p, ids in stored.items()})
        matched = len(selected)
        for part in sorted(want):
            if part[0] != "user":
                continue
            remaining = want[part] - len(stored.get(part, [])) - paired[part] - selected_counts[part]
            for merge in ((host or {}).get(part) or {}).get("merges", []):
                if remaining <= 0:
                    break
                uses = Counter(merge["parts"])
                if all(spare[p] >= n for p, n in uses.items()):
                    spare.subtract(uses)
                    remaining -= 1
                    matched += 1
        return matched

    def assign(index, selected):
        nonlocal best, best_score, steps
        if steps >= COVER_STEPS:
            return
        steps += 1
        merit = coverage(selected)
        if merit > best_score:
            best, best_score = selected[:], merit
        if index == len(units) or best_score == needed:
            return
        part = units[index]
        for (sid, start, end), credit in sorted(choices[part].items()):
            if any(start < b and a < end for a, b in used[sid]):
                continue
            used[sid].append((start, end))
            assign(index + 1, selected + [credit])
            used[sid].pop()
            if best_score == needed or steps >= COVER_STEPS:
                break
        assign(index + 1, selected)

    assign(0, [])
    counts.update((c["role"], c["sha256"]) for c in best)
    return counts, best


def score(expected: list[tuple[str, str]], stored_rows: list[tuple], host: dict | None = None,
          *, _key=None, surplus_keys=None) -> dict:
    """``expected``: (role, text) items; ``stored_rows``: (store_id, session_id, role, content); ``host``: the
    lineage's host occurrences (key -> {"n", "ids"}), or None."""
    stored_rows = sorted(stored_rows, key=lambda r: r[0])
    replay_rows = {sid: (session, role, content) for sid, session, role, content in stored_rows}
    key_of = _key or (lambda role, text: (role, h(text)))
    stored, by_session = defaultdict(list), defaultdict(list)
    for sid, session, role, content in stored_rows:
        if role not in ("user", "assistant") or not norm(content or ""):
            continue
        stored[key_of(role, content)].append(sid)
        if role == "assistant":
            by_session[session].append((sid, content or ""))
    want = Counter(key_of(role, text) for role, text in expected if norm(text))
    texts = {key_of(role, text): text for role, text in expected}
    composites, paired, spent = [], Counter(), Counter()
    consumed = defaultdict(list)  # copies a host merge record took as a composite's parts: kept copies of their key
    embedded, span_credits = embedded_credits(texts, want, stored_rows, stored, paired, host)
    embedded_ids = {c["store_id"] for c in span_credits}
    # #823: each active host merge record covers one missing composite. Reserve transcript rows first;
    # consume its parts before surplus licensing so no stored row can serve both purposes.
    for key, n in sorted(want.items()):
        if key[0] != "user":
            continue
        for merge in ((host or {}).get(key) or {}).get("merges", []):
            if len(stored.get(key, [])) + paired[key] + embedded[key] >= n:
                break
            uses = Counter(merge["parts"])
            available = {k: [sid for sid in stored.get(k, [])[want[k]:] if sid not in embedded_ids] for k in uses}
            if any(len(available[k]) < count for k, count in uses.items()):
                continue
            parts = []
            for k, count in uses.items():
                ids = available[k][:count]
                stored[k] = [sid for sid in stored[k] if sid not in ids]
                consumed[k] += ids
                parts.append({"sha256": k[1], "uses": count, "store_ids": ids})
            paired[key] += 1
            spent.update(uses)  # the host rows that prove the record license no surplus too
            composites.append({"role": "user", "expected": n, "as_parts": 1, "parts": parts,
                               **({"provenance": "host_merge_event", "witness": merge["witness"]} if "witness" in merge
                                  else {"provenance": "host_merge_record", "host_row_id": merge["host_row_id"]})})
    missing, duplicated, split, licensed = [], [], [], []
    for key, n in want.items():
        n -= paired[key] + embedded[key]
        have = len(stored.get(key, []))
        entry = {"role": key[0], "expected": n, "stored": have, "store_ids": stored.get(key, [])[:20],
                 "preview": norm(texts[key])[:80]}
        if have < n:
            if key[0] == "assistant" and have == 0:
                target = texts[key]
                for rows in by_session.values():
                    for k in range(len(rows) - 1):
                        if norm(target) == norm(rows[k][1] + rows[k + 1][1]):  # B2's edge-stripped split diagnostic
                            entry["split_match"] = [rows[k][0], rows[k + 1][0]]
                if "split_match" in entry:
                    split.append(entry)
                    continue
            missing.append({**entry, "sha256": key[1]})
        elif have > n:
            if lic := licence(key, have - n, n, host, stored[key], spent[key]):
                licensed.append(lic)
                entry["licensed"] = lic["licensed"]
            if have - n > entry.get("licensed", 0):
                duplicated.append(entry)
                if surplus_keys is not None:
                    surplus_keys.append((key, have - n - entry.get("licensed", 0), stored[key] + consumed[key],
                                         consumed[key] + [sid for sid in stored[key] if sid in embedded_ids], replay_rows))
    extra = []
    for k, v in stored.items():
        if k not in want:
            lic = licence(k, len(v), 0, host, v, spent[k])
            licensed += [lic] if lic else []
            if len(v) > (lic or {}).get("licensed", 0):
                if surplus_keys is not None:
                    surplus_keys.append((k, len(v) - (lic or {}).get("licensed", 0), v + consumed[k],
                                         consumed[k] + [sid for sid in v if sid in embedded_ids], replay_rows))
                extra.append({"role": k[0], "copies": len(v) - (lic or {}).get("licensed", 0), "store_ids": v[:6]})
    # #804: a held user composite whose parts the host durably stored apart is stored here as those parts, each
    # licensed by host parity; the composite key is then not a deficit. Only the occurrences the host did not store
    # as the composite itself pair (those LCM should have stored whole): the whole deficit must fit within them. Same
    # lineage only; reported, never silent.
    licence_records = {(r["role"], r["sha256"]): r for r in licensed}
    by_key = {}
    for key, record in licence_records.items():
        reserved = want[key] - paired[key] - embedded[key]
        ids = [sid for sid in stored[key][reserved:] if sid not in embedded_ids]
        by_key[key] = dict(record, licensed=min(record["licensed"], len(ids)), store_ids=ids)
    for entry in list(missing):
        deficit = entry["expected"] - entry["stored"]
        held = ((host or {}).get(("user", entry["sha256"])) or {}).get("n", 0)
        if entry["role"] == "user" and deficit <= entry["expected"] - held and \
                (uses := licensed_parts(norm(texts[("user", entry["sha256"])]), by_key, deficit)):
            parts = []
            for key, n in uses.items():
                count = n * deficit
                by_key[key]["licensed"] -= count
                licence_records[key]["licensed"] -= count
                ids, by_key[key]["store_ids"] = by_key[key]["store_ids"][:count], by_key[key]["store_ids"][count:]
                parts.append({"sha256": key[1], "uses": n, "store_ids": ids})
            missing.remove(entry)
            composites.append({"role": "user", "expected": entry["expected"], "as_parts": deficit, "preview": entry["preview"],
                               "parts": parts})
    licensed = [r for r in licensed if r["licensed"] > 0]
    # Every stored-only key and every split reply is surplus: no host transform licenses them.
    return {
        "instrument": "multiset-v1",
        "verdict": "PASS" if not (missing or duplicated or extra or split) else "FAIL",
        "expected_items": sum(want.values()),
        "distinct_keys": len(want),
        "missing_keys": len(missing),
        "deficit_rows": sum(e["expected"] - e["stored"] for e in missing),
        "duplicated_keys": len(duplicated),
        "surplus_rows": sum(e["stored"] - e["expected"] - e.get("licensed", 0) for e in duplicated) + sum(e["copies"] for e in extra),
        "host_parity_licensed": sorted(licensed, key=lambda r: json.dumps(r, sort_keys=True)),
        "held_composites_as_parts": sorted(composites, key=lambda r: json.dumps(r, sort_keys=True)),
        "embedded_span_credits": sorted(span_credits, key=lambda r: json.dumps(r, sort_keys=True)),
        "missing": sorted(missing, key=lambda r: json.dumps(r, sort_keys=True)),
        "duplicated": sorted(duplicated, key=lambda r: json.dumps(r, sort_keys=True)),
        "split_assistant_turns": sorted(split, key=lambda r: json.dumps(r, sort_keys=True)),
        "split_keys": len(split),
        "stored_rows_not_expected": len(extra),
        "extra": sorted(extra, key=lambda r: json.dumps(r, sort_keys=True)),
    }


def _nonspace_nfd_len(text):
    # Exact: NFD counts add across joins and vkey preserves them; isspace() agrees with regex \s.
    return sum(not ch.isspace() for ch in unicodedata.normalize("NFD", text))


def phase_c_score(expected: list[tuple[str, str]], rows: list[tuple]) -> dict:
    """Phase C adopts the release multiset-v2 split predicate (r2, sha256 8943a6a7…) without its own-turn
    prompt rule; unlike r2, user boundaries span the owning conversation. Store order guards own turns.
    Runs are bounded by the turn (user rows), departing from r2's 8-fragment cap (#2214).
    Rows: (store_id, session_id, role, content, conversation_id). v2 normalization is split-only."""
    rows = sorted(rows, key=lambda r: r[0])
    items = [(i, r, t) for i, (r, t) in enumerate(expected) if norm(t)]
    owners = {c for _, _, r, t, c in rows if items and row_key(r, t) == row_key(*items[0][1:])}
    if len(owners) != 1 or next(iter(owners)) in (None, ""):  # an unset id names no conversation
        return {"verdict": "INCONCLUSIVE", "reason": "first transcript item has no unique owning conversation"}
    owner = owners.pop()
    foreign = dict(Counter(c for *_, c in rows if c != owner))
    owned = [r[:4] for r in rows if r[4] == owner]
    def vkey(role, text):  # inherited v2 NFC/CRLF/whitespace normalization; never used by whole-row keys
        return role, re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").replace("\r\n", "\n")).strip()
    want = Counter(vkey(r, t) for r, t in expected)
    held = Counter(vkey(r, t) for _, _, r, t in owned if norm(t))
    assistants, user_ids = defaultdict(list), []
    for sid, session, role, text in owned:
        if role == "user":
            user_ids.append(sid)  # empty users remain turn boundaries (v2 MB1)
        if role == "assistant" and norm(text):
            assistants[session].append((sid, text))
    user_ids.sort()
    used, replacements, splits = set(), [], []
    for index, role, text in items:
        vk = vkey(role, text)
        if role != "assistant" or held[vk] or want[vk] != 1:
            continue
        hits = []
        limit = _nonspace_nfd_len(text)
        for session, arows in assistants.items():
            for k in range(len(arows)):
                ids, parts = [arows[k][0]], [arows[k][1]]
                next_user = bisect_right(user_ids, ids[0])
                boundary = user_ids[next_user] if next_user < len(user_ids) else None
                total = _nonspace_nfd_len(parts[0])
                for j in range(k + 1, len(arows)):
                    sid, content = arows[j]
                    if boundary is not None and sid > boundary:
                        break
                    ids.append(sid)
                    parts.append(content)
                    total += _nonspace_nfd_len(content)
                    if total > limit:
                        break
                    if total < limit:
                        continue
                    if vk in (vkey(role, " ".join(parts)), vkey(role, "".join(parts))):
                        if all(vkey(role, c) not in want for c in parts):
                            hits.append((ids, session))
                        break
        if len(hits) == 1 and not used.intersection(hits[0][0]):  # uniqueness before use (v2 MB2)
            ids, session = hits[0]
            used.update(ids)
            replacements.append((ids[0], session, role, text))
            splits.append({"transcript_index": index, "role": role, "split_match": ids})
    matched = sorted([r for r in owned if r[0] not in used] + replacements, key=lambda r: r[0])
    out = score(expected, matched, _key=row_key)  # unused fragment copies remain surplus; no row is borrowed twice
    keys = {row_key(r, t) for r, t in expected if norm(t)}
    foreign_transcript_rows = sum(c != owner and r in ("user", "assistant") and bool(norm(t)) and
                                  row_key(r, t) in keys for _, _, r, t, c in rows)
    if foreign_transcript_rows:
        out.update(verdict="FAIL", surplus_rows=out["surplus_rows"] + foreign_transcript_rows)
    out.update(owning_conversation=owner, foreign_conversations=foreign, phase_c_split_matches=splits,
               foreign_transcript_rows=foreign_transcript_rows,
               instrument="phase-c-multiset-v2", accepted_split_keys=len(splits))
    sequence = [row_key(r, t) for _, _, r, t in matched if r in ("user", "assistant") and norm(t)]
    cursor = 0
    for index, role, text in items:
        target = row_key(role, text)
        pos = next((j for j in range(cursor, len(sequence)) if sequence[j] == target), None)
        if pos is not None:
            cursor = pos + 1
        elif target in sequence:
            out.update(verdict="FAIL", out_of_order={"transcript_index": index, "role": role})
            break
    return out


@functools.cache
def todo_header() -> str:
    """The todo injection header: the product's ``_PRESERVED_TODO_CONTEXT_PREFIX`` (engine.py), read, never copied."""
    tree = ast.parse((Path(__file__).resolve().parents[4] / "engine.py").read_text())
    return next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and [getattr(t, "id", None) for t in n.targets] == ["_PRESERVED_TODO_CONTEXT_PREFIX"])


def link_pending_ids(relations):
    """#2377: rows the ingest marked ``link_pending`` (related_store_id NULL). It marks rows only when computing their
    links raised, and then over-marks (a new row of that ingest is marked too), so a surplus row that carries one is a
    visible copy; the mark on any row is a defect signal."""
    return {sid for sid, kind, other, *_ordinal in relations if kind == "link_pending" and other is None}


def composite_witness(relations, rows):
    """#2212 ``composite_of`` witness (provisional format): the stored user rows C it credits. C holds one relation
    per segment, ordinals 0..n-1 contiguous with n >= 2, each naming an earlier stored user row of C's session, or NULL
    for the host scaffold, at most once and only at ordinal 0 (the product renders the todo block first). With one
    NULL that leftover span starts with the todo injection header; with
    none, ordinal 0 names a stored user row that starts with it. C's content is the segments joined by "\n\n" in
    ordinal order, byte for byte. Valid composites consume their non-NULL parts once, in ascending C store id:
    a composite sharing any already-claimed part receives no credit; NULL claims nothing. Only C is credited:
    the contained rows are scored as ordinary rows. A replayed composite needs its own possible_replay_of link."""
    header, segments, valid = todo_header(), defaultdict(list), {}
    for sid, kind, other, *ordinal in relations:
        if kind == "composite_of":
            segments[sid].append((ordinal[0] if ordinal else None, other))
    for c, parts in segments.items():
        session, role, content = rows.get(c, (None, None, None))
        ordinals = [o for o, _r in parts]
        if role != "user" or len(parts) < 2 or not all(type(o) is int for o in ordinals) \
                or sorted(ordinals) != list(range(len(parts))):
            continue
        related = [r for _o, r in sorted(parts)]
        nulls = [i for i, r in enumerate(related) if r is None]
        if nulls not in ([], [0]) or any(r not in rows or rows[r][0] != session for r in related if r is not None):
            continue
        if any(r >= c or rows[r][1] != "user" for r in related if r is not None):
            continue
        texts, content = [None if r is None else rows[r][2] or "" for r in related], content or ""
        if nulls:  # the scaffold is the one span the contained rows leave
            i = nulls[0]
            head = "".join(t + "\n\n" for t in texts[:i])
            tail = "".join("\n\n" + t for t in texts[i + 1:])
            if len(content) < len(head) + len(tail) or not (content.startswith(head) and content.endswith(tail)):
                continue
            texts[i] = content[len(head):len(content) - len(tail)]
            first_ok = texts[i].startswith(header)
        else:
            first_ok = rows[related[0]][1] == "user" and texts[0].startswith(header)
        if first_ok and "\n\n".join(texts) == content:
            valid[c] = {r for r in related if r is not None}
    out, claimed = set(), set()
    for c, parts in sorted(valid.items()):
        if not parts & claimed:
            out.add(c)
            claimed.update(parts)
    return out


def attribution_forms(relations, rows, logged=None):
    """The witnesses that credit a surplus row as a visible copy beyond the relation witnesses: store id sets keyed
    by their output count. ``logged``: the store ids the LCM-X log names as a failed link's source (the gate audit's
    log input); None leaves the form out."""
    forms = {"link_pending_rows": link_pending_ids(relations), "composite_rows": composite_witness(relations, rows)}
    if logged is not None:
        forms["logged_link_rows"] = set(logged)
    return forms


def split_surplus(groups, relations, rows=None, forms=None, creditable=None):
    """Maximum matching of relation witnesses to rows, with each key capped at its unlicensed surplus.

    Flow: key quota -> unique self/pair witness -> stored row. Witness and row nodes belong to one key, so a row
    explains at most one copy of each key it holds: an assistant row whose content and tool entries were stored
    twice explains one copy of each. Full ID lists include expected copies, and they count copies, not rows: a row
    whose ``tool_calls`` holds one call entry twice appears twice and still explains at most one of them.
    Store order identifies a key's expected copy only when the key keeps at most one copy (expected, licensed or
    loose-allowed): that copy is the earliest (as for licences and silent examples), so only the last ``surplus``
    copies can be explained. A self relation counts only on such a row, and a pair explains only such an endpoint,
    so a relation on the expected copy (an edited revision's ``supersedes``, say) never explains a later unmarked
    copy. A key that keeps two or more copies counts its worst case, because a silent re-store stored before a
    related later copy cannot be told apart: its surplus fills rows without a self relation first, then the
    self-related rows that hold the most copies, and each self-related row it reaches explains one copy; pairs
    explain nothing there. A copy that a host merge record consumed as a composite's part is a kept copy of its key
    and stays in the key's ID list. A group's optional fourth item lists the copies that a host transform's
    accounting chose out of store order: those a #823 merge record consumed and those an embedded credit reserved.
    A key with any takes the worst case whatever it keeps, because #823 merges and #804 licensed parts pick their
    copies around reserved rows, so store order no longer identifies the kept copy.
    Each ``attribution_forms`` set (``forms``, computed from the relations when None) is a self witness too; a form's
    count is the visible surplus it adds to the relation witnesses alone, so two forms on one row both count it.
    ``creditable``: a set, when given, receives each key's copies that the matching can credit (its ``later`` set
    in the pass with every form), so a caller can report credited rows without counting a key's expected copy.
    """
    # Stored text only verifies an existing directed witness; it never creates one.
    # ``rows`` (store_id -> (session, role, content)) covers every stored row, tool results included; without it,
    # the evidence maps attached to surplus groups are merged once each (score() shares one map across groups).
    if rows is None:
        rows, seen = {}, set()
        for group in groups:
            if len(group) > 4 and id(group[4]) not in seen:
                seen.add(id(group[4]))
                rows.update(group[4])
    forms = attribution_forms(relations, rows) if forms is None else forms
    held = {sid for group in groups for sid in group[2]}
    live = {name: ids & held for name, ids in forms.items()}
    # Passes: one with every touched form, one without, one per touched form (up to 5 in the audit, 4 in B2).
    out = _matched(groups, relations, rows, set().union(*live.values()), creditable)
    base = _matched(groups, relations, rows, set())["visible_surplus_rows"] if any(live.values()) else 0
    for name, ids in live.items():
        out[name] = _matched(groups, relations, rows, ids)["visible_surplus_rows"] - base if ids else 0
    out["link_pending_rows_total"] = len(forms["link_pending_rows"])
    return out


def _matched(groups, relations, rows, credited, creditable=None):
    graph, source, sink = defaultdict(dict), ("source",), ("sink",)

    def edge(a, b, capacity=1):
        if b not in graph[a]:
            graph[a][b], graph[b][a] = capacity, 0

    self_ids = {sid for sid, kind, *_rest in relations if kind in ("supersedes", "generated_unproven")} | credited
    pairs = sorted({tuple(sorted((sid, other))) for sid, kind, other, *_rest in relations
                    if kind == "possible_replay_of" and other is not None and sid != other})
    replay_rows = rows
    stripped_pairs = [(other, sid) for sid, kind, other, *_rest in relations
                      if kind == "possible_replay_of" and other in replay_rows and sid in replay_rows and other < sid
                      and replay_rows[sid][:2] == replay_rows[other][:2]
                      and stripped_replay_content(replay_rows[sid][2]) is not None
                      and stripped_replay_content(replay_rows[sid][2]) == stripped_replay_content(replay_rows[other][2])]
    caps = []
    for index, (_key, surplus, ids, *merged) in enumerate(groups):
        quota, held, count = ("key", index), set(ids), Counter(ids)
        copies = sorted(ids)  # one row can hold a key twice (a repeated tool-call entry): count copies, not rows
        kept = len(copies) - surplus  # the key's expected, licensed or loose-allowed copies
        if kept <= 1 and not (merged and merged[0]):
            # Store order identifies the expected copy: only the key's last `surplus` copies can be explained.
            cap, later = surplus, set(copies[max(0, kept):]) if surplus > 0 else set()
            witnesses = [("self", sid) for sid in sorted(later & self_ids)]
            witnesses += [("pair", a, b) for a, b in pairs if a in held and b in held and (a in later or b in later)]
            witnesses += [("pair", a, b) for a, b in stripped_pairs if b in later and a not in held]
        else:
            # Worst case: the surplus fills rows without a self relation first, then the self-related rows that hold
            # the most copies; each self-related row it reaches explains one copy, and pairs explain nothing.
            later, cap = held & self_ids, 0
            left = surplus - sum(n for sid, n in count.items() if sid not in later)
            for n in sorted((count[sid] for sid in later), reverse=True):
                if left <= 0:
                    break
                cap, left = cap + 1, left - n
            witnesses = [("self", sid) for sid in sorted(later)]
        if creditable is not None:
            creditable.update(later)
        caps.append(cap)
        edge(source, quota, cap)
        for witness in witnesses:
            before, after = ("in", index, witness), ("out", index, witness)
            edge(quota, before)
            edge(before, after)
            for sid in witness[1:]:
                if sid not in later:
                    continue
                row = ("row", index, sid)
                edge(after, row)
                edge(row, sink)
    visible = 0
    while True:
        parent, queue = {source: None}, [source]
        for node in queue:
            for nxt, capacity in graph[node].items():
                if capacity > 0 and nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
            if sink in parent:
                break
        if sink not in parent:
            break
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            graph[previous][node] -= 1
            graph[node][previous] += 1
            node = previous
        visible += 1
    examples = []
    for index, (key, surplus, ids, *_merged) in enumerate(groups):
        silent = surplus - (caps[index] - graph[source][("key", index)])  # surplus minus what the matching explained
        if not silent:
            continue
        # Best-effort representatives: the highest-numbered rows that still hold a copy the matching left unexplained
        # (a row explains at most one copy), usually the later copies rather than the original. Rows without a self
        # relation sort last, so they are taken first: in a worst-case key the likely silent copy is the unmarked one.
        copies = sorted(ids)
        kept = len(copies) - surplus
        # With one kept copy and no reserved merge, store order identifies the surplus.
        candidates = copies[max(0, kept):] if kept <= 1 and not (_merged and _merged[0]) else copies
        count = Counter(candidates)
        unexplained = sorted((sid for sid, n in count.items() if n > 1 - graph[("row", index, sid)].get(sink, 1)),
                             key=lambda sid: (sid not in self_ids, sid))
        examples += [{"store_id": sid, "role": key[0], "sha256": key[1]} for sid in unexplained[-silent:]]
    return {"visible_surplus_rows": visible, "silent_surplus_rows": sum(g[1] for g in groups) - visible,
            "silent_surplus_examples": sorted(examples, key=lambda e: (e["store_id"], e["role"], e["sha256"]))[:10]}
