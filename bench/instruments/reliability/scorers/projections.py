"""Verify survival views with the scored plugin's pure codec and an original occurrence witness."""
from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path


def codec(tree):
    path = Path(tree).resolve()
    name = "_reliability_codec_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return importlib.import_module(name + ".survival_codec")


def verify(atts, rows, events, group, tree):
    """Only map an exactly regenerated view to an original witnessed in this or an earlier attempt.

    The full original hash is checked too: head/tail equality alone cannot detect an altered middle.
    An unavailable codec or unverified lookalike leaves the held bytes as an ordinary B2 expectation.
    """
    from .bars import attempt_group

    receipts = []
    if not tree:
        return receipts
    try:
        renderer = codec(tree)
    except (ImportError, OSError, SyntaxError):
        return receipts
    ingress = {}
    attempt_index = -1
    for e in events:
        if e.get("event") in ("user_sent", "retry"):
            attempt_index += 1
        elif e.get("event") == "engine_ingress" and e.get("session"):
            ingress.setdefault(max(0, attempt_index), []).append(e)
    originals = {}
    by_id = {r[0]: r for r in rows}
    for i, a in enumerate(atts):
        g = attempt_group(a, group)
        originals.setdefault(g, set()).add(hashlib.sha256(a["persist"].encode()).hexdigest())
        for e in ingress.get(i, []):
            originals.setdefault(group(e["session"]), set()).update(
                r["sha256"] for r in e.get("occurrences", []) if r.get("role") == "user")
        held = a.get("held")
        if not isinstance(held, str):
            continue
        for match in renderer.projection_mark_candidates(held, None):
            role, tokens, sid, head, tail = match.groups()
            row = by_id.get(int(sid))
            g = attempt_group(a, group)
            if role != "user" or row is None or row[2] != role or group(row[1]) != g:
                continue
            digest = hashlib.sha256(row[3].encode()).hexdigest()
            if digest not in originals.get(g, set()):
                continue
            source = dict(zip(("store_id", "session_id", "role", "content", "tool_calls", "tool_call_id"), row))
            if any(renderer.projected_fields(source, int(tokens), int(head), int(tail), legacy=legacy)["content"]
                   == held for legacy in (False, True)):
                a["held"] = row[3]
                receipts.append({"tag": a["tag"], "session": g, "store_id": row[0],
                                 "projection_sha256": hashlib.sha256(held.encode()).hexdigest(),
                                 "original_sha256": digest})
                break
    return receipts
