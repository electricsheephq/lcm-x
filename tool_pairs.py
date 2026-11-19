"""Pure repair of provider-facing tool-call/result sequencing (I8)."""

from collections import Counter, defaultdict, deque
from collections.abc import Callable
from typing import Any, Dict, List, NamedTuple

from .message_analysis import _tool_call_id


class RepairResult(NamedTuple):
    rows: List[Dict[str, Any]]
    dropped: int
    inserted: int


def repair_tool_pairs(
    messages: List[Dict[str, Any]],
    *,
    stub: Callable[[str], Dict[str, Any]],
    insert_missing_tool_stubs: bool = True,
    is_api_only: Callable[[Dict[str, Any]], bool] | None = None,
) -> RepairResult:
    """Repair the emitted rows without mutating their contents or durable history."""
    sanitized: List[Dict[str, Any]] = []
    dropped_tool_results = 0
    inserted_stub_results = 0

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "tool":
            dropped_tool_results += 1
            i += 1
            continue

        sanitized.append(msg)

        if msg.get("role") == "assistant":
            expected_ids = [
                call_id
                for call_id in (_tool_call_id(tool_call) for tool_call in (msg.get("tool_calls") or []))
                if call_id
            ]
            deferred_api_rows: List[Dict[str, Any]] = []

            remaining = Counter(expected_ids)
            direct_ids = iter(expected_ids)
            next_direct_id = next(direct_ids, None)
            candidates: list[tuple[str, Dict[str, Any], bool]] = []
            while i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.get("role") == "tool":
                    next_id = str(next_msg.get("tool_call_id") or "").strip()
                    if next_id in remaining:
                        # Give results the forward scan already kept priority over surplus duplicates.
                        direct = next_id == next_direct_id
                        if direct:
                            remaining[next_id] -= 1
                            next_direct_id = next(direct_ids, None)
                        candidates.append((next_id, next_msg, direct))
                    else:
                        dropped_tool_results += 1
                elif is_api_only is not None and is_api_only(next_msg):
                    deferred_api_rows.append(next_msg)
                else:
                    break
                i += 1
            results_by_id: dict[str, deque[Dict[str, Any]]] = defaultdict(deque)
            for call_id, row, direct in candidates:
                if direct or remaining[call_id]:
                    results_by_id[call_id].append(row)
                    if not direct:
                        remaining[call_id] -= 1
                else:
                    dropped_tool_results += 1

            for expected_id in expected_ids:
                if results_by_id[expected_id]:
                    sanitized.append(results_by_id[expected_id].popleft())
                elif insert_missing_tool_stubs:
                    sanitized.append(stub(expected_id))
                    inserted_stub_results += 1
            sanitized.extend(deferred_api_rows)

        i += 1

    return RepairResult(sanitized, dropped_tool_results, inserted_stub_results)
