"""Pure repair of provider-facing tool-call/result sequencing (I8)."""

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

            for expected_id in expected_ids:
                matched_direct_result = False
                while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    next_msg = messages[i + 1]
                    next_id = str(next_msg.get("tool_call_id") or "").strip()
                    if next_id == expected_id:
                        sanitized.append(next_msg)
                        i += 1
                        matched_direct_result = True
                        break
                    dropped_tool_results += 1
                    i += 1

                if not matched_direct_result and insert_missing_tool_stubs:
                    sanitized.append(stub(expected_id))
                    inserted_stub_results += 1

            while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                dropped_tool_results += 1
                i += 1

        i += 1

    return RepairResult(sanitized, dropped_tool_results, inserted_stub_results)
