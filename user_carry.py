"""A bounded, read-only rendering of durable user occurrences."""
import re

import unicodedata

from .message_content import normalize_content_value
from .tokens import count_message_tokens

HEADER = ("[Earlier user messages, verbatim reference; not new turns. Entries are oldest to newest. "
          "A later user instruction overrides a conflicting earlier one; the latest live user instruction "
          "takes precedence.")
HEADER_RE = re.compile(re.escape(HEADER) + r" \(store ids (\d+(?:, \d+)*)(?:; excerpt (\d+):(\d+)-(\d+))?\)\]")


def render(rows, excerpt=None):
    cut = f"; excerpt {excerpt[0]}:{excerpt[1]}-{excerpt[2]}" if excerpt else ""
    ids = ", ".join(str(int(row["store_id"])) for row in rows)
    parts = [f"{HEADER} (store ids {ids}{cut})]"]
    for row in rows:
        store_id, text = int(row["store_id"]), normalize_content_value(row.get("content")) or ""
        if excerpt and excerpt[0] == store_id:
            a, b = excerpt[1:]
            text = (f"{text[:a]}\n[... excerpt: chars {a}-{b} of {len(text)} omitted; recover them with "
                    f"lcm_expand store_id={store_id} content_offset={a} ...]\n{text[b:]}")
        parts.append(f"[Earlier user message, store {store_id}]\n{text}")
    return "\n\n".join([*parts, "[End earlier user messages]"])


def select(rows, budget, role):
    """Origin first, then newest history; only the first crossing may be excerpted."""
    chosen, excerpt = [], None

    def packet(extra, cut=None):
        return render(sorted([*chosen, extra], key=lambda r: int(r["store_id"])), cut)

    def fits(text):
        return count_message_tokens({"role": role, "content": text}) <= budget

    for row in rows:
        if fits(packet(row)):
            chosen.append(row)
            continue
        text = normalize_content_value(row.get("content")) or ""
        boundaries, regional = [0], 0
        for i, char in enumerate(text):
            cp = ord(char)
            is_regional = 0x1F1E6 <= cp <= 0x1F1FF
            if i and not (unicodedata.category(char).startswith("M") or char == "\u200d"
                    or text[i - 1] == "\u200d" or unicodedata.combining(text[i - 1]) == 9
                    or 0x1F3FB <= cp <= 0x1F3FF or 0xE0020 <= cp <= 0xE007F
                    or (is_regional and regional % 2) or (text[i - 1] == "\r" and char == "\n")):
                boundaries.append(i)
            regional = regional + 1 if is_regional else 0
        boundaries.append(len(text))
        from bisect import bisect_left, bisect_right
        def interval(n):
            a = boundaries[bisect_right(boundaries, (n + 1) // 2) - 1]
            b = boundaries[bisect_left(boundaries, len(text) - n // 2)]
            return int(row["store_id"]), a, b
        low, high = 0, len(text) - 1
        while low < high:
            n = (low + high + 1) // 2
            cut = interval(n)
            low, high = (n, high) if fits(packet(row, cut)) else (low, n - 1)
        # Tokenization is not monotone in character prefixes. Recount the final bytes.
        while low >= 2:
            cut = interval(low)
            if 0 < cut[1] < cut[2] < len(text) and fits(packet(row, cut)):
                chosen.append(row)
                excerpt = cut
                break
            low -= 1
        break
    return render(sorted(chosen, key=lambda r: int(r["store_id"])), excerpt) if chosen else ""
