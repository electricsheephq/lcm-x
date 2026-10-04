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
(plugin_tree.carrier_markers) is never licensed, nor counted toward a B1 tag. Recovery prefixes anywhere in a
merged row also exclude it: LCM's carriers are LCM's responsibility
(R8 / D-D), so a host echo of one licenses nothing. Fails closed: a missing or unreadable state.db, or a plugin tree
whose carrier markers cannot be read, grants no licence, and the reason is recorded. The state.db is the cell's
sqlite backup copy, opened read-only.
"""
from __future__ import annotations

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
        recovery = plugin_tree.recovery_prefixes(Path(tree))
    except (OSError, SyntaxError, ValueError) as exc:
        return None, f"plugin carrier markers unreadable ({tree}): {exc!r}"[:200]
    try:
        con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
        try:
            rows = con.execute("select id, session_id, content, active from messages"
                               " where role = 'user' order by id").fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return None, f"host state.db unreadable: {exc!r}"[:200]
    held, texts = defaultdict(list), {}  # (lineage, key) -> [(row id, session, active)]
    for rid, sid, content, active in rows:
        head = (content or "").lstrip()
        if multiset.norm(content or "") and not (header.match(head) or head.startswith(prefixes)
                                               or any(p in head for p in recovery)):
            key = ("user", multiset.h(content))
            held[(group(sid), key)].append((rid, sid, active))
            texts[key] = content
    out = defaultdict(lambda: {"keys": {}, "tags": defaultdict(lambda: {"n": 0, "ids": []})})
    for (g, key), rs in held.items():
        n = max([1, *Counter(s for _r, s, a in rs if a == 1).values()])
        ids = [r for r, _s, _a in rs]
        tags = sorted(set(re.findall(USER_TAG, texts[key])))
        out[g]["keys"][key] = {"n": n, "ids": ids, "tags": tags}  # scripted-prompt tags, never the text
        for tag in tags:
            out[g]["tags"][tag]["n"] += n
            out[g]["tags"][tag]["ids"] = sorted(out[g]["tags"][tag]["ids"] + ids)
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
