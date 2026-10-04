"""multiset-v1 lossless bar, ported from the v0.24.3-rc1 gauntlet ``lossless_bar_multiset.py``.

Key = (role, sha256(content; user edges stripped, assistant bytes exact)) over non-empty user/assistant rows:
internal whitespace is compared EXACTLY (a collapsed "\n\n" separator is a different row). The only licensed
transform in whole-row keys is the user edge strip, which the host performs on ACP prompts (acp_adapter/server.py
``user_text = _extract_text(prompt).strip()``: eva/rs34 :818, customer :786, upstream :820). Hermes also strips
assistant content on store (``_assistant_content_for_storage``), so exact assistant keys rely on the drivers
stripping ``raw_answer`` too. No NFC or CRLF
normalisation is applied: neither host nor plugin performs one on message content. Per key the
stored row count must equal the expected (transcript) count: fewer = loss (deficit), more = duplicates
(surplus). Repeated identical items are fine as long as the multiplicity matches. A stored-only key is
surplus and fails; a split assistant answer (one reply stored as two adjacent rows) is reported apart, as in
the source, and fails too (its fragments are stored-only keys). ``host`` (D-A, scorers/host_parity.py) licenses a
user-row surplus only up to what the host's own state.db holds in the lineage; ``None`` (no host evidence) licenses
nothing.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict


def norm(text: str) -> str:
    return (text or "").strip()


def h(text: str) -> str:
    return hashlib.sha256(norm(text).encode()).hexdigest()


def row_key(role: str, text: str) -> tuple:
    return role, h(text) if role == "user" else hashlib.sha256((text or "").encode()).hexdigest()


def licence(key: tuple, surplus: int, expected: int, host: dict | None, store_ids: list) -> dict | None:
    """D-A: licensed = min(surplus, host_count(k) - expected(k)), floored at 0; user rows only, never a deficit."""
    held = (host or {}).get(key) if key[0] == "user" else None
    n = min(surplus, held["n"] - expected) if held else 0
    return {"role": key[0], "sha256": key[1], "expected": expected, "stored": expected + surplus, "host": held["n"],
            "licensed": n, "store_ids": store_ids[:10], "host_row_ids": held["ids"][:10], "tags": held.get("tags", [])} \
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


def score(expected: list[tuple[str, str]], stored_rows: list[tuple], host: dict | None = None) -> dict:
    """``expected``: (role, text) items; ``stored_rows``: (store_id, session_id, role, content); ``host``: the
    lineage's host occurrences (key -> {"n", "ids"}), or None."""
    stored, by_session = defaultdict(list), defaultdict(list)
    for sid, session, role, content in stored_rows:
        if role not in ("user", "assistant") or not norm(content or ""):
            continue
        stored[row_key(role, content)].append(sid)
        if role == "assistant":
            by_session[session].append((sid, content or ""))
    want = Counter(row_key(role, text) for role, text in expected if norm(text))
    texts = {row_key(role, text): text for role, text in expected}
    composites, paired = [], Counter()
    # #823: each active host merge record covers one missing composite. Reserve transcript rows first;
    # consume its parts before surplus licensing so no stored row can serve both purposes.
    for key, n in want.items():
        if key[0] != "user":
            continue
        for merge in ((host or {}).get(key) or {}).get("merges", []):
            if len(stored.get(key, [])) + paired[key] >= n:
                break
            uses = Counter(merge["parts"])
            if any(len(stored.get(k, [])) - want[k] < count for k, count in uses.items()):
                continue
            parts = []
            for k, count in uses.items():
                ids = stored[k][want[k]:want[k] + count]
                del stored[k][want[k]:want[k] + count]
                parts.append({"sha256": k[1], "uses": count, "store_ids": ids})
            paired[key] += 1
            composites.append({"role": "user", "expected": n, "as_parts": 1, "parts": parts,
                               "provenance": "host_merge_record", "host_row_id": merge["host_row_id"]})
    missing, duplicated, split, licensed = [], [], [], []
    for key, n in want.items():
        n -= paired[key]
        have = len(stored.get(key, []))
        entry = {"role": key[0], "expected": n, "stored": have, "store_ids": stored.get(key, [])[:20],
                 "preview": norm(texts[key])[:80]}
        if have < n:
            if key[0] == "assistant" and have == 0:
                target = texts[key]
                for rows in by_session.values():
                    for k in range(len(rows) - 1):
                        if target == rows[k][1] + rows[k + 1][1]:  # exact assistant bytes, as in the keys
                            entry["split_match"] = [rows[k][0], rows[k + 1][0]]
                if "split_match" in entry:
                    split.append(entry)
                    continue
            missing.append({**entry, "sha256": key[1]})
        elif have > n:
            if lic := licence(key, have - n, n, host, stored[key]):
                licensed.append(lic)
                entry["licensed"] = lic["licensed"]
            if have - n > entry.get("licensed", 0):
                duplicated.append(entry)
    extra = []
    for k, v in stored.items():
        if k not in want:
            lic = licence(k, len(v), 0, host, v)
            licensed += [lic] if lic else []
            if len(v) > (lic or {}).get("licensed", 0):
                extra.append({"role": k[0], "copies": len(v) - (lic or {}).get("licensed", 0), "store_ids": v[:6]})
    # #804: a held user composite whose parts the host durably stored apart is stored here as those parts, each
    # licensed by host parity; the composite key is then not a deficit. Only the occurrences the host did not store
    # as the composite itself pair (those LCM should have stored whole): the whole deficit must fit within them. Same
    # lineage only; reported, never silent.
    by_key = {(r["role"], r["sha256"]): r for r in licensed}
    for entry in list(missing):
        deficit = entry["expected"] - entry["stored"]
        held = ((host or {}).get(("user", entry["sha256"])) or {}).get("n", 0)
        if entry["role"] == "user" and deficit <= entry["expected"] - held and \
                (uses := licensed_parts(norm(texts[("user", entry["sha256"])]), by_key, deficit)):
            for key, n in uses.items():
                by_key[key]["licensed"] -= n * deficit
            missing.remove(entry)
            composites.append({"role": "user", "expected": entry["expected"], "as_parts": deficit, "preview": entry["preview"],
                               "parts": [{"sha256": key[1], "uses": n, "store_ids": by_key[key]["store_ids"]}
                                         for key, n in uses.items()]})
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
        "host_parity_licensed": licensed,
        "held_composites_as_parts": composites,
        "missing": missing[:40],
        "duplicated": duplicated[:40],
        "split_assistant_turns": split,
        "split_keys": len(split),
        "stored_rows_not_expected": len(extra),
        "extra": extra[:20],
    }


def phase_c_score(expected: list[tuple[str, str]], rows: list[tuple]) -> dict:
    """Phase C adopts the release multiset-v2 split predicate (r2, sha256 8943a6a7…) without its own-turn
    prompt rule; the store-order check below fails the borrowed-turn case that rule guarded.
    Rows: (store_id, session_id, role, content, conversation_id). v2 normalization is split-only."""
    rows = sorted(rows, key=lambda r: r[0])
    items = [(i, r, t) for i, (r, t) in enumerate(expected) if norm(t)]
    owners = {c for _, _, r, t, c in rows if items and row_key(r, t) == row_key(*items[0][1:])}
    if len(owners) != 1:
        return {"verdict": "INCONCLUSIVE", "reason": "first transcript item has no unique owning conversation"}
    owner = owners.pop()
    foreign = dict(Counter(c for *_, c in rows if c != owner))
    owned = [r[:4] for r in rows if r[4] == owner]
    def vkey(role, text):  # inherited v2 NFC/CRLF/whitespace normalization; never used by whole-row keys
        return role, re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").replace("\r\n", "\n")).strip()
    want = Counter(vkey(r, t) for r, t in expected)
    held = Counter(vkey(r, t) for _, _, r, t in owned if norm(t))
    assistants, users = defaultdict(list), defaultdict(list)
    for sid, session, role, text in owned:
        if role == "user":
            users[session].append(sid)  # empty users remain turn boundaries (v2 MB1)
        if role == "assistant" and norm(text):
            assistants[session].append((sid, text))
    used, replacements, splits = set(), [], []
    for index, role, text in items:
        vk = vkey(role, text)
        if role != "assistant" or held[vk] or want[vk] != 1:
            continue
        hits = []
        for session, arows in assistants.items():
            for k in range(len(arows)):
                for n in range(2, 9):  # v2 MAX_FRAGMENTS = 8; consecutive non-empty assistant rows
                    run = arows[k:k + n]
                    if len(run) < n:
                        break
                    if vk in (vkey(role, " ".join(c for _, c in run)), vkey(role, "".join(c for _, c in run))):
                        ids = [s for s, _ in run]
                        if not any(ids[0] < u < ids[-1] for u in users[session]) and all(
                                vkey(role, c) not in want for _, c in run):
                            hits.append((ids, session))
                        break
        if len(hits) == 1 and not used.intersection(hits[0][0]):  # uniqueness before use (v2 MB2)
            ids, session = hits[0]
            used.update(ids)
            replacements.append((ids[0], session, role, text))
            splits.append({"transcript_index": index, "role": role, "split_match": ids})
    matched = sorted([r for r in owned if r[0] not in used] + replacements, key=lambda r: r[0])
    out = score(expected, matched)  # unused fragment copies remain surplus; no stored row is borrowed twice
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
