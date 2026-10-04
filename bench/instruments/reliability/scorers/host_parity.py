"""D-A host-parity licence for B1/B2 (DESIGN-436-logical-identity.md REVISION 2, principle P-HOST).

LCM's lossless contract is fidelity to the host's durable conversation, so a stored user-row surplus of key k in
lineage L is licensed only up to what the host's own state.db holds in L:
licensed = min(surplus, host_count_L(k) - expected_L(k)), floored at 0 (multiset.licence). Deficits, assistant rows
and tool rows are never licensed. B2 keys host rows exactly as it keys stored rows (multiset.h: the edge strip only);
B1 counts ``[Tnn] user turn`` tags in the same host occurrences.

host_count counts what one host view holds, never physical copies. Hermes re-issues every durable row on every
rotation copy (a child session) and in-place generation (the old rows go inactive), H1 even with a fresh timestamp
per generation (436-host-identity-map.md), so neither rows across sessions nor distinct timestamps add up. A key's
count is the largest number of ACTIVE rows one session of the lineage holds at once (a durable double persist: H2
rows 105/106, PROBE.md group 1), and 1 when the lineage holds it only on inactive rows (a compacted generation).
A key whose content begins with one of the scored plugin tree's own generated-carrier markers
(plugin_tree.carrier_markers) is never licensed, nor counted toward a B1 tag: LCM's carriers are LCM's responsibility
(R8 / D-D), so a host echo of one licenses nothing. Fails closed: a missing or unreadable state.db, or a plugin tree
whose carrier markers cannot be read, grants no licence, and the reason is recorded. The state.db is the cell's
sqlite backup copy, opened read-only.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .. import plugin_tree
from . import multiset

USER_TAG = r"\[([A-Z]\d{2,3})\] user turn"


def load(state: Path, group, tree: str | None = None) -> tuple[dict | None, str | None]:
    """lineage -> {"keys": {key: {"n", "ids"}}, "tags": {tag: {"n", "ids"}}} from the host's user rows, or (None, why)."""
    if not state.exists():
        return None, "no host state.db in the cell"
    if not tree:
        return None, "no plugin tree recorded for the cell, so no carrier markers"
    try:
        header, prefixes = plugin_tree.carrier_markers(Path(tree))
    except (OSError, SyntaxError, ValueError) as exc:
        return None, f"plugin carrier markers unreadable ({tree}): {exc!r}"[:200]
    try:
        con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
        try:
            columns = {r[1] for r in con.execute("pragma table_info(messages)")}
            provenance = {"message_uid", "absorbed_message_uids"} <= columns
            extra = ", message_uid, absorbed_message_uids" if provenance else ", NULL, NULL"
            rows = con.execute("select id, session_id, content, active" + extra + " from messages"
                               " where role = 'user' order by id").fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return None, f"host state.db unreadable: {exc!r}"[:200]
    held, texts = defaultdict(list), {}  # (lineage, key) -> [(row id, session, active)]
    standalone, merges = defaultdict(dict), {}
    for rid, sid, content, active, uid, raw_absorbed in rows:
        head = (content or "").lstrip()
        if multiset.norm(content or "") and not (header.match(head) or head.startswith(prefixes)):
            key = ("user", multiset.h(content))
            g = group(sid)
            held[(g, key)].append((rid, sid, active))
            texts[key] = content
            try:
                absorbed = json.loads(raw_absorbed or "[]")
            except (ValueError, TypeError):
                continue
            if not uid or not isinstance(absorbed, list) or not all(isinstance(u, str) and u for u in absorbed):
                continue
            if not absorbed:
                standalone[(g, uid)][key] = content
            elif active == 1 and len(absorbed) + 1 <= multiset.COVER_PARTS:
                uids = (uid, *absorbed)
                if len(set(uids)) == len(uids):
                    merges.setdefault((g, key, uids), rid)
    out = defaultdict(lambda: {"keys": {}, "tags": defaultdict(lambda: {"n": 0, "ids": []})})
    for (g, key), rs in held.items():
        n = max([1, *Counter(s for _r, s, a in rs if a == 1).values()])
        ids = [r for r, _s, _a in rs]
        tags = sorted(set(re.findall(USER_TAG, texts[key])))
        out[g]["keys"][key] = {"n": n, "ids": ids, "tags": tags}  # scripted-prompt tags, never the text
        for tag in tags:
            out[g]["tags"][tag]["n"] += n
            out[g]["tags"][tag]["ids"] = sorted(out[g]["tags"][tag]["ids"] + ids)
    for (g, key, uids), rid in merges.items():
        parts = [standalone.get((g, uid), {}) for uid in uids]
        if all(len(p) == 1 for p in parts):
            resolved = [next(iter(p.items())) for p in parts]
            if multiset.norm("\n\n".join(text for _k, text in resolved)) == multiset.norm(texts[key]):
                out[g]["keys"][key].setdefault("merges", []).append(
                    {"host_row_id": rid, "parts": [k for k, _text in resolved]})  # no payload or full uids
    return out, None


def b1_licence(tag: str, want: int, have: int, host: dict | None, store_ids: list) -> dict | None:
    """The B1 duplicate-tag count under the same rule (multiset.licence) as B2."""
    held = (host or {}).get("tags", {}).get(tag)
    n = min(have - want, held["n"] - want) if held and have > want else 0
    return {"tag": tag, "expected": want, "stored": have, "host": held["n"], "licensed": n,
            "store_ids": store_ids[:10], "host_row_ids": held["ids"][:10]} if n > 0 else None


def summary(records: list[dict], why: str | None) -> dict:
    """``host_parity_licensed``: the licensed row count and up to 10 row records (never silent)."""
    return {"rows": sum(r["licensed"] for r in records), "records": records[:10], **({"unavailable": why} if why else {})}
