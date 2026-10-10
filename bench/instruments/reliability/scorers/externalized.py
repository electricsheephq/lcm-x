"""Resolve B1/B2 rows with the product's placeholder predicate and payload reader."""
from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import externalize
from config import LCMConfig

# ingest_protection's placeholder pattern (a test pins the two equal); that module only imports inside the package.
INGEST_PREFIX = "[Externalized LCM ingest payload:"
INGEST_RE = re.compile(r"\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]")
PAYLOAD_DIRNAME = externalize.DEFAULT_LARGE_OUTPUT_DIRNAME  # run_matrix keeps it beside kept DB copies
_ENV_LOCK = threading.Lock()


@contextmanager
def _cell_env():
    """The cell ran with a clean env: resolve its payloads without the operator shell's LCM_HERMES_BASE_DIR."""
    with _ENV_LOCK:  # run_matrix scores cells on threads
        saved = os.environ.pop("LCM_HERMES_BASE_DIR", None)
        try:
            yield
        finally:
            if saved is not None:
                os.environ["LCM_HERMES_BASE_DIR"] = saved


def resolve(rows: list[tuple], home: Path) -> tuple[list[tuple], dict, set]:
    with _cell_env():
        return _resolve(rows, home)


def _restore_ingest(row: tuple, config, home: Path) -> tuple[tuple, bool]:
    """Ingest-protection placeholders (inline base64/media), expanded as ingest_protection.restore_ingest_payload_
    placeholders does. Only an unreadable payload is a loss; one the product also keeps as a placeholder (another
    kind, another session) stays as stored, for ordinary B2 accounting."""
    if not isinstance(row[3], str) or INGEST_PREFIX not in row[3]:
        return row, True
    lost = []

    def expand(match: re.Match) -> str:
        payload = externalize.load_externalized_payload(match.group(1).strip(), config=config, hermes_home=str(home))
        if payload is None:
            lost.append(match.group(1))
            return match.group(0)
        owner = payload.get("session_id") or ""
        if payload.get("kind") != "ingest_payload" or (row[1] and owner and owner != row[1]) \
                or not isinstance(payload.get("content"), str):
            return match.group(0)
        return payload["content"]

    text = INGEST_RE.sub(expand, row[3])
    return (*row[:3], text, *row[4:]), not lost


def _resolve(rows: list[tuple], home: Path) -> tuple[list[tuple], dict, set]:
    config = LCMConfig()  # The harness forbids path-valued overrides; never read the operator's env/home.
    resolved, ids, losses = [], set(), []
    for row in rows:
        if not externalize.is_externalized_placeholder(row[3]):
            restored, ok = _restore_ingest(row, config, home)
            if ok:
                resolved.append(restored)
                if restored is not row:
                    ids.add(row[0])
            else:
                losses.append({"store_id": row[0], "reason": "missing_ingest_payload"})
            continue
        ref = externalize.extract_externalized_ref(row[3])
        reason = None
        try:
            path = externalize.get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref
            # The product reader defaults an absent content field to ''. Check presence separately.
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or "content" not in raw:
                reason = "no_content_field"
            elif not isinstance(raw["content"], str):
                reason = "invalid_content"
            else:
                payload = externalize.load_externalized_payload(ref, config=config, hermes_home=str(home))
                if payload is None:
                    reason = "unreadable_payload"
        except FileNotFoundError:
            reason = "missing_payload_file"
        except (ValueError, OSError, TypeError, AttributeError):
            reason = "unreadable_payload"
        if not reason:
            row, ok = _restore_ingest((*row[:3], payload["content"], *row[4:]), config, home)
            reason = None if ok else "missing_ingest_payload"
        if reason:
            losses.append({"store_id": row[0], "reason": reason})
        else:
            resolved.append(row)
            ids.add(row[0])
    numbers = {"resolved": len(ids), "unresolved": len(losses), "deficit_rows": len(losses),
               "reasons": dict(Counter(e["reason"] for e in losses)),
               "store_ids": [e["store_id"] for e in losses][:10], "deficits": losses[:10],
               "content_mismatch_rows": 0, "content_mismatch": []}
    return resolved, numbers, ids


def mismatches(numbers: dict, rows: list[tuple], ids: set, per: dict, group, expected_tools, loose, host) -> None:
    """Diagnostic only: name the resolved rows whose bytes match nothing expected. Ordinary B2 accounting already
    fails a wrong resolved row (a missing key plus a surplus one), so this never adds to ``deficit_rows``. It looks
    only at the rows B2 scores: user/assistant/tool rows of an expected group."""
    from . import multiset, tool_calls

    want = {g: {(r, multiset.h(t)) for r, t in items if multiset.norm(t)} for g, (items, _) in per.items()}
    for row in rows:
        sid, session, role, content = row[:4]
        g = group(session)
        if sid not in ids or g not in want or role not in ("user", "assistant", "tool"):
            continue
        if role == "tool":
            keys = tool_calls.stored_keys([row], group)
            bad = any(k not in expected_tools and k not in loose for k in keys)
        else:
            key = (role, multiset.h(content))
            licensed = role == "user" and (host or {}).get(g, {}).get("keys", {}).get(key, {}).get("n", 0)
            bad = bool(multiset.norm(content)) and key not in want[g] and not licensed
        if bad:
            numbers["content_mismatch_rows"] += 1
            if len(numbers["content_mismatch"]) < 10:
                numbers["content_mismatch"].append({"store_id": sid, "sha256": multiset.h(content)})
