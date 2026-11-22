"""The rendered LCM summary part and its byte-for-byte prefix verifier."""

import re


_SUMMARY_DEPTH_LABELS = {0: "Recent", 1: "Session Arc", 2: "Durable"}
_LCM_SUMMARY_PART_HEADER_RE = re.compile(
    r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d(\d+), node (\d+)\)\]\n"
)


def _summary_part_text(node) -> str:
    """One frontier node as context assembly renders it in the summary prefix."""
    label = _SUMMARY_DEPTH_LABELS.get(node.depth, f"Depth-{node.depth}")
    return (
        f"[{label} Summary (d{node.depth}, node {node.node_id})]\n"
        f"{node.summary}\n"
        f"[Expand for details: {node.expand_hint}]"
    )


def verify_summary_prefix(content, get_node, owns_session, header_re=_LCM_SUMMARY_PART_HEADER_RE):
    pos = 0
    node_ids: list[int] = []
    while True:
        header = header_re.match(content, pos)
        if header is None:
            break
        try:
            node = get_node(int(header.group(2)))
        except Exception:
            return None, []
        # A quoted part of an unrelated session's summary is user content (#484 item 11j).
        if node is None or not owns_session(node.session_id) or int(node.depth) != int(header.group(1)):
            return None, []
        part = _summary_part_text(node)
        if not content.startswith(part, pos):
            return None, []
        pos += len(part)
        node_ids.append(int(node.node_id))
        if content.startswith("\n\n---\n\n", pos) and header_re.match(content, pos + 7):
            pos += 7
            continue
        break
    return (pos, node_ids) if node_ids else (None, [])
