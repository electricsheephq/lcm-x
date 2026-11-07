"""Pure, fail-closed recognition of Hermes' two host rewrite shapes (I1)."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HostRewriteMatch:
    arm: str
    consumed: tuple[int, ...]
    k: int


def host_rewrite_match(row, candidates, *, header, todo_span, prompts_stamped: bool = False) -> Optional[HostRewriteMatch]:
    """Candidates belong to one session; consumed parts remain present with ``consumed=True``.

    ``previous`` optionally names the preceding non-user store row, including rows outside the
    stamp scan. Without it, store order is checked against the supplied complete prefix.
    A full 64-row M scan is conservatively treated as exhausted.
    """
    text, stamp = row.get("content"), row.get("stamp")
    if not isinstance(text, str) or row.get("tool_calls"):
        return None
    if (prompts_stamped and row.get("role") == "user" and stamp is None
            and text.startswith(header) and todo_span == len(text)):
        if any(c.get("role") == "user" and isinstance(c.get("content"), str)
               and (c["content"] == text or c["content"].startswith(text + "\n\n")) for c in candidates):
            return HostRewriteMatch("S", (), 1)
    if row.get("role") != "assistant" or stamp is None or "\n" not in text or len(candidates) >= 64:
        return None
    ordered = sorted((c for c in candidates if c.get("observed_at") is not None
                      and c["observed_at"] >= stamp), key=lambda c: (c["observed_at"], c["store_id"]))
    heads = [c for c in ordered if c["observed_at"] == stamp and c.get("role") == "assistant"
             and not c.get("consumed")]
    fits = []
    for head in heads:
        parts, joined = [], ""
        for c in ordered[ordered.index(head):]:
            if c.get("role") == "user":
                continue
            if c.get("role") != "assistant" or c.get("tool_calls") or c.get("consumed"):
                break
            if parts:
                previous = c.get("previous", max((r["store_id"] for r in candidates
                               if r["store_id"] < c["store_id"] and r.get("role") != "user"), default=None))
                if previous != parts[-1]:
                    break
            part = c.get("content")
            if not isinstance(part, str) or not part.strip():
                break
            joined += ("\n" if parts else "") + part.strip()
            parts.append(c["store_id"])
            if joined == text and len(parts) >= 2:
                fits.append(HostRewriteMatch("M", tuple(parts), len(parts)))
                break
            if not text.startswith(joined + "\n"):
                break
    return fits[0] if len(fits) == 1 else None
