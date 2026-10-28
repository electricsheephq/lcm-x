"""PEM block scanning for storage-boundary ingest protection."""

from __future__ import annotations

import re
from typing import Callable as Callable
from .ingest_pem_model import (
    _HEX_DIGEST_RE,
    _PEM_LINE_KIND_ARMOR,
    _PEM_LINE_KIND_BEGIN,
    _PEM_LINE_KIND_BLANK,
    _PEM_LINE_KIND_END,
    _PEM_LINE_KIND_END_INLINE,
    _PEM_LINE_KIND_END_LOOSE,
    _PEM_LINE_KIND_LENIENT_B64,
    _PEM_LINE_KIND_OTHER,
    _PEM_LINE_KIND_PREFIXED_B64,
    _PEM_LINE_KIND_SHORT_B64,
    _PEM_LINE_KIND_STRICT_B64,
    _PRIVATE_KEY_ARMOR_HEADER_RE,
    _PRIVATE_KEY_BEGIN_RE,
    _PRIVATE_KEY_BODY_CHARS_RE,
    _PRIVATE_KEY_END_RE,
    _PRIVATE_KEY_ESCAPED_SEPARATOR_HINT_RE,
    _PRIVATE_KEY_INLINE_RUN_RE,
    _PRIVATE_KEY_INLINE_SPAN_RE,
    _PRIVATE_KEY_MAX_ARMOR_HEADERS,
    _PRIVATE_KEY_STRICT_MIN_CHARS,
    _looks_like_english_token,
    _normalize_escaped_solidus,
    _pem_line_model,
    _split_serialized_pem_line,
    _trim_pem_segment,
)

def _redact_private_key_blocks_with(text: str, placeholder: Callable[[str], str]) -> str:
    """Redact PEM private keys in one linear pass over a normalized line model.

    Handles, fail-closed toward redaction (#365, rounds 3-6):
    - complete blocks: BEGIN line, any contiguous base64 payload lines (any
      width), an adjacent END line — redacted marker-to-marker; a decorated
      END (``label: -----END PRIVATE KEY-----``) closes a block through its
      marker end;
    - truncated blocks: BEGIN plus a contiguous base64 run with no adjacent
      END — redacted through the whole run (short re-wrapped lines included),
      never past it (unrelated prose and following keys survive);
    - orphaned bodies: a contiguous base64 run containing a full-width line —
      leading short lines included — directly followed by an END or decorated
      END line, with no BEGIN: a decoy BEGIN elsewhere must not shield a
      stripped key;
    - inline one-line ``BEGIN ... END`` forms, requiring a real base64 run
      between the markers (prose mentioning both markers is never consumed);
      every BEGIN candidate on the line is tried, so a decoy BEGIN cannot
      blind a later valid inline key;
    - CR-only / unicode line separators and indented or prefixed markers.
    A bare BEGIN marker followed by non-base64 prose is left in place. See the
    THREAT MODEL comment above the predicates for the declared scope.
    """
    if "private key-----" not in text.lower():
        return text
    model = _pem_line_model(text)
    n = len(model)
    parts: list[str] = []
    cursor = 0
    changed = False

    def emit(redact_from: int, redact_to: int) -> None:
        nonlocal cursor, changed
        parts.append(text[cursor:redact_from])
        parts.append(placeholder(text[redact_from:redact_to]))
        cursor = redact_to
        changed = True

    def effective_kind(idx: int) -> int:
        # Inside a BEGIN-anchored scan only: log collectors prefix every line
        # ("INFO Proc-Type: …", "INFO CDEF"), so a line the context-free model
        # calls OTHER gets one more chance with up to 4 leading tokens
        # stripped. English-word tails (pure lowercase, or one capitalized
        # word) never count as body — "service productionservice" is prose.
        kind = model[idx][0]
        if kind not in (_PEM_LINE_KIND_OTHER, _PEM_LINE_KIND_ARMOR):
            return kind
        # An ARMOR-shaped line can be a colon-ended log prefix ("INFO: MII…")
        # in front of real body — prefix-strip before trusting the armor
        # shape; genuine armor values (commas, spaces) never classify as body.
        rest = _normalize_escaped_solidus(text[model[idx][1]:model[idx][2]])
        # Attached Markdown/quote/diff markers (">MII…", "|MII…", "-MII…")
        # carry no whitespace; strip them before any classification. '-' is
        # not in the base64 charset, so stripping it never eats body.
        rest = rest.lstrip(">|-").strip(" \t")
        head, colon_sep, colon_after = rest.partition(":")
        colon_after = colon_after.strip(" \t")
        if (
            colon_sep
            and 2 <= len(head) <= 32
            and colon_after
            and _PRIVATE_KEY_BODY_CHARS_RE.fullmatch(colon_after) is not None
            and not _looks_like_english_token(colon_after)
        ):
            # Colon-prefixed BODY ("INFO: MII…" / "INFO:MII…") — must win
            # over the armor shape, which it also matches. Genuine armor
            # values ("4,ENCRYPTED", "AES-128-CBC,…") contain commas and
            # never classify as base64 body.
            if len(colon_after) >= _PRIVATE_KEY_STRICT_MIN_CHARS:
                return _PEM_LINE_KIND_STRICT_B64
            return _PEM_LINE_KIND_SHORT_B64
        if _PRIVATE_KEY_BODY_CHARS_RE.fullmatch(rest) is not None:
            if not _looks_like_english_token(rest):
                if len(rest) >= _PRIVATE_KEY_STRICT_MIN_CHARS:
                    return _PEM_LINE_KIND_STRICT_B64
                return _PEM_LINE_KIND_SHORT_B64
            # English-shaped single tokens ("abcdef", "example") count inside
            # a BEGIN-anchored scan as ADJACENCY only: they keep a complete
            # block's END reachable but never justify truncated redaction on
            # their own (N1zc precision).
            return _PEM_LINE_KIND_LENIENT_B64
        # Whitespace-chunked body ("SYNTH ETIC 1234 90AB …"): every token
        # base64-charset, none English-shaped, at least two tokens with one
        # of length >= 8 — prose fails the English test per token.
        tokens = rest.split()
        if (
            len(tokens) >= 3
            and all(
                _PRIVATE_KEY_BODY_CHARS_RE.fullmatch(tok) is not None
                and not _looks_like_english_token(tok)
                for tok in tokens
            )
            and max(len(tok) for tok in tokens) >= 8
            and any(any(c.isdigit() or c in "+/=" for c in tok) for tok in tokens)
        ):
            # Digit/symbol evidence required: real chunked base64 carries
            # digits; "INFO IMPORTANT" (all-caps word pairs) does not.
            return _PEM_LINE_KIND_STRICT_B64
        armor_shape = _PRIVATE_KEY_ARMOR_HEADER_RE.fullmatch(rest) is not None
        for _ in range(6):
            split = rest.split(None, 1)
            if len(split) < 2:
                # No further tokens: an armor-shaped line with no hidden body
                # (">Proc-Type: 4,ENCRYPTED", "Proc-Type: 4,ENCRYPTED") is
                # armor; anything else keeps its context-free kind. Checked
                # AFTER token-stripping so a timestamped/colon log prefix
                # hiding real body ("2026-…Z INFO MII…") classifies as body.
                if armor_shape:
                    return _PEM_LINE_KIND_ARMOR
                return kind
            rest = split[1]
            # Serialized indentation can sit AFTER the prefix ("INFO \tMII…"):
            # normalize escaped horizontal whitespace at each strip step.
            _off, rest = _trim_pem_segment(rest)
            if not rest:
                return _PEM_LINE_KIND_ARMOR if armor_shape else kind
            if _PRIVATE_KEY_ARMOR_HEADER_RE.fullmatch(rest) is not None:
                # Time tokens ('12:34:56 …') masquerade as armor mid-strip:
                # note the shape but KEEP STRIPPING — body evidence deeper in
                # the line dominates armor shapes (round-15 principle).
                armor_shape = True
                continue
            if (
                _PRIVATE_KEY_BODY_CHARS_RE.fullmatch(rest) is not None
                and not _looks_like_english_token(rest)
            ):
                if len(rest) >= _PRIVATE_KEY_STRICT_MIN_CHARS:
                    return _PEM_LINE_KIND_STRICT_B64
                return _PEM_LINE_KIND_SHORT_B64
        return _PEM_LINE_KIND_ARMOR if armor_shape else kind

    i = 0
    back_watermark = -1
    while i < n:
        kind, redact_start, content_end, line_start, _marker_end = model[i]
        if kind == _PEM_LINE_KIND_BEGIN:
            j = i + 1
            saw_b64 = False
            # RFC 1421 encrypted-key armor: header lines then one blank line
            # sit between BEGIN and the base64 body; consume them so a
            # truncated encrypted key still redacts through its body.
            armor = 0
            while (
                j < n
                and effective_kind(j) == _PEM_LINE_KIND_ARMOR
                and armor < _PRIVATE_KEY_MAX_ARMOR_HEADERS
            ):
                armor += 1
                j += 1
            while j < n and model[j][0] == _PEM_LINE_KIND_BLANK:
                j += 1
            if j < n and (
                model[j][0] == _PEM_LINE_KIND_OTHER
                and 0
                < len(tokens := text[model[j][1]:model[j][2]].split())
                <= 6
                and all(len(tok) <= 12 for tok in tokens)
                and effective_kind(j) == _PEM_LINE_KIND_OTHER
            ):
                # A log-prefixed blank remnant (lone short prefix tokens).
                j += 1
                while j < n and model[j][0] == _PEM_LINE_KIND_BLANK:
                    j += 1
            if armor and (j >= n or effective_kind(j) not in (
                _PEM_LINE_KIND_STRICT_B64,
                _PEM_LINE_KIND_SHORT_B64,
                _PEM_LINE_KIND_PREFIXED_B64,
                _PEM_LINE_KIND_LENIENT_B64,
                _PEM_LINE_KIND_END,
                _PEM_LINE_KIND_END_INLINE,
                _PEM_LINE_KIND_END_LOOSE,
            )):
                # Headers without a following body or END are not key
                # structure (e.g. a bare BEGIN quoted above config lines).
                i += 1
                continue
            last_evidence = -1
            saw_strict_run = False
            pending_shorts = []
            while j < n:
                run_kind = effective_kind(j)
                if run_kind in (
                    _PEM_LINE_KIND_STRICT_B64,
                    _PEM_LINE_KIND_PREFIXED_B64,
                ):
                    saw_b64 = True
                    saw_strict_run = True
                    last_evidence = j
                elif run_kind == _PEM_LINE_KIND_SHORT_B64:
                    # A short token after body is a real PEM tail; after a
                    # bare BEGIN it is a doc label ("IMPORTANT", "PKCS8") —
                    # never sole evidence. A short-ONLY run counts only as a
                    # re-wrap: >=2 contiguous shorts, each base64-aligned.
                    if saw_strict_run:
                        saw_b64 = True
                        last_evidence = j
                    else:
                        seg = text[model[j][1]:model[j][2]].strip(" \t")
                        tok = seg.split()[-1] if seg.split() else seg
                        pending_shorts.append((j, len(tok.lstrip(">|-")) % 4 == 0))
                elif run_kind == _PEM_LINE_KIND_LENIENT_B64:
                    # An English-shaped line breaks short-run contiguity: a
                    # real re-wrap is contiguous; identifier tokens separated
                    # by words are documentation.
                    if not saw_strict_run:
                        pending_shorts = []
                else:
                    break
                j += 1
            if (
                not saw_strict_run
                and len(pending_shorts) >= 2
                and all(aligned for _idx, aligned in pending_shorts)
            ):
                saw_b64 = True
                last_evidence = pending_shorts[-1][0]
            if j < n and model[j][0] == _PEM_LINE_KIND_END:
                emit(redact_start, model[j][2])
                i = j + 1
                continue
            if j < n and model[j][0] in (
                _PEM_LINE_KIND_END_INLINE,
                _PEM_LINE_KIND_END_LOOSE,
            ):
                emit(redact_start, model[j][4])
                i = j + 1
                continue
            if saw_b64:
                # Truncated bound: through the last EVIDENCE line only —
                # trailing English-shaped lenient lines keep END adjacency
                # alive but are prose when no END follows.
                emit(redact_start, model[last_evidence][2])
                i = last_evidence + 1
                continue
            begin_m = _PRIVATE_KEY_BEGIN_RE.match(text, redact_start)
            if (
                begin_m is not None
                and begin_m.end() < content_end
                and _PRIVATE_KEY_INLINE_RUN_RE.search(
                    text, begin_m.end(), content_end
                )
            ):
                # Newline-normalized truncated form: the base64 payload sits
                # ON the BEGIN line with nothing usable after — redact the
                # marker plus its same-line body.
                emit(redact_start, content_end)
                i += 1
                continue
            i += 1
            continue
        if kind in (
            _PEM_LINE_KIND_STRICT_B64,
            _PEM_LINE_KIND_SHORT_B64,
            _PEM_LINE_KIND_PREFIXED_B64,
        ):
            j = i
            has_strict = kind in (
                _PEM_LINE_KIND_STRICT_B64,
                _PEM_LINE_KIND_PREFIXED_B64,
            )
            while j + 1 < n and model[j + 1][0] in (
                _PEM_LINE_KIND_STRICT_B64,
                _PEM_LINE_KIND_SHORT_B64,
                _PEM_LINE_KIND_PREFIXED_B64,
            ):
                j += 1
                has_strict = has_strict or model[j][0] in (
                    _PEM_LINE_KIND_STRICT_B64,
                    _PEM_LINE_KIND_PREFIXED_B64,
                )
            if (has_strict or j > i) and j + 1 < n and model[j + 1][0] in (
                _PEM_LINE_KIND_END,
                _PEM_LINE_KIND_END_INLINE,
            ):
                end_kind, _rs, end_content_end, _ls, end_marker_end = model[j + 1]
                emit(
                    redact_start,
                    end_content_end
                    if end_kind == _PEM_LINE_KIND_END
                    else end_marker_end,
                )
                i = j + 2
                continue
            if (
                (has_strict or j > i)
                and j + 1 < n
                and model[j + 1][0] == _PEM_LINE_KIND_BEGIN
                and model[j + 1][4] >= 0
            ):
                # Compacted "-----END ...----- -----BEGIN ...-----" line: its
                # leading END closes this orphan run; the BEGIN half is then
                # processed as its own block.
                emit(redact_start, model[j + 1][4])
                i = j + 1
                continue
            i = j + 1
            continue
        if kind in (_PEM_LINE_KIND_END, _PEM_LINE_KIND_END_INLINE):
            # END-anchored backward pass (symmetric to the BEGIN-anchored
            # forward pass): prefixed/colon orphan bodies classify OTHER or
            # ARMOR context-free, so a preceding run is only visible with
            # prefix-aware reclassification. Bounded lookback; evidence
            # rules identical to the forward scan.
            back = i - 1
            first_evidence = -1
            saw_back_evidence = False
            while back >= 0 and back > back_watermark:
                bk = effective_kind(back)
                if bk in (
                    _PEM_LINE_KIND_STRICT_B64,
                    _PEM_LINE_KIND_SHORT_B64,
                    _PEM_LINE_KIND_PREFIXED_B64,
                ):
                    saw_back_evidence = saw_back_evidence or bk != _PEM_LINE_KIND_SHORT_B64
                    first_evidence = back
                    back -= 1
                    continue
                if bk == _PEM_LINE_KIND_LENIENT_B64:
                    back -= 1
                    continue
                break
            back_watermark = max(back_watermark, i)
            if saw_back_evidence and first_evidence >= 0:
                end_bound = content_end if kind == _PEM_LINE_KIND_END else _marker_end
                # Never cross content already emitted (an earlier redaction
                # inside the lookback window): clamp to the cursor.
                emit(max(model[first_evidence][1], cursor), end_bound)
                i += 1
                continue
        if kind in (_PEM_LINE_KIND_END_INLINE, _PEM_LINE_KIND_END_LOOSE):
            # Orphaned ONE-LINE body: an upgraded v1-era row can hold
            # "<placeholder> MII… -----END PRIVATE KEY-----" on one physical
            # line. A substantial contiguous base64 run directly before the
            # marker (only base64/spacing between) is key material.
            end_m = _PRIVATE_KEY_END_RE.search(text, redact_start, content_end)
            if end_m is not None:
                span_start = redact_start
                probe = text[span_start:end_m.start()]
                run = None
                for m in _PRIVATE_KEY_INLINE_RUN_RE.finditer(probe):
                    run = m
                if run is not None and _PRIVATE_KEY_INLINE_SPAN_RE.fullmatch(
                    probe, run.end()
                ):
                    emit(span_start + run.start(), end_m.end())
                    i += 1
                    continue
            i += 1
            continue
        if kind == _PEM_LINE_KIND_OTHER and "PRIVATE KEY-----" in text[line_start:content_end].upper():
            scan = line_start
            candidates = 0
            while scan < content_end:
                candidates += 1
                if candidates > 8:
                    # Bounded work per line: >8 unmatched BEGIN candidates on
                    # one line is a marker storm (adversarial construction,
                    # out of declared scope) — stop rather than go quadratic.
                    break
                begin = _PRIVATE_KEY_BEGIN_RE.search(text, scan, content_end)
                if begin is None:
                    break
                inline_end = _PRIVATE_KEY_END_RE.search(text, begin.end(), content_end)
                if (
                    inline_end is not None
                    and _PRIVATE_KEY_INLINE_SPAN_RE.fullmatch(
                        text, begin.end(), inline_end.start()
                    )
                    and (
                        begin.end() == inline_end.start()
                        or _PRIVATE_KEY_INLINE_RUN_RE.search(
                            text, begin.end(), inline_end.start()
                        )
                    )
                ):
                    emit(begin.start(), inline_end.end())
                    scan = inline_end.end()
                    continue
                scan = begin.end()
        i += 1
    parts.append(text[cursor:])
    return "".join(parts) if changed else text


def _has_orphaned_private_key_body(text: str) -> bool:
    """True when line-structured private-key material survives in ``text``.

    Structural, transform-independent check on the normalized line model
    (round 5): flags a BEGIN line followed by any base64-alphabet line, and a
    base64 run containing a full-width line directly followed by an END line.
    A correct redaction pass leaves neither shape, and there is deliberately
    NO earlier-BEGIN suppression — a decoy BEGIN elsewhere in the text must
    not shield a stripped key from validation. Prose that merely mentions a
    marker inline (not as the whole line) never matches.
    """
    if "private key-----" not in text.lower():
        return False
    model = _pem_line_model(text)
    run_has_strict = False
    run_len = 0
    prev_was_begin = False
    seen_begin_line = False
    for kind, _redact_start, _content_end, _line_start, _marker_end in model:
        if kind == _PEM_LINE_KIND_BEGIN:
            prev_was_begin = True
            seen_begin_line = True
            run_has_strict = False
            run_len = 0
            continue
        if kind in (
            _PEM_LINE_KIND_STRICT_B64,
            _PEM_LINE_KIND_SHORT_B64,
            _PEM_LINE_KIND_PREFIXED_B64,
        ):
            if prev_was_begin:
                return True
            if kind in (_PEM_LINE_KIND_STRICT_B64, _PEM_LINE_KIND_PREFIXED_B64):
                run_has_strict = True
            run_len += 1
            continue
        if kind in (_PEM_LINE_KIND_END, _PEM_LINE_KIND_END_INLINE) and (
            run_has_strict or run_len >= 2
        ):
            return True
        if kind in (_PEM_LINE_KIND_END_INLINE, _PEM_LINE_KIND_END_LOOSE):
            end_m = _PRIVATE_KEY_END_RE.search(
                text, _redact_start, _content_end
            )
            if end_m is not None:
                probe = text[_redact_start:end_m.start()]
                run = None
                for m in _PRIVATE_KEY_INLINE_RUN_RE.finditer(probe):
                    run = m
                if run is not None and _PRIVATE_KEY_INLINE_SPAN_RE.fullmatch(
                    probe, run.end()
                ):
                    return True
        if kind in (_PEM_LINE_KIND_END, _PEM_LINE_KIND_END_INLINE) and seen_begin_line:
            # Fail-closed pair guard (cloud path): a surviving exact BEGIN
            # line before a surviving exact END line is a structure the
            # redactor could not parse (e.g. a body with internal whitespace
            # on every line, which is indistinguishable from prose at the
            # redactor level). Block the dispatch rather than risk shipping
            # key material the line classifier cannot recognize.
            return True
        prev_was_begin = False
        run_has_strict = False
        run_len = 0
    return False


def _pem_marker_with_inline_body(text: str) -> bool:
    """True when a surviving private-key marker shares a line with a base64 run.

    Marker-independent backstop (#383 rounds 4-5): prose that merely MENTIONS a
    marker ("see -----BEGIN PRIVATE KEY----- for format") carries English
    tokens, no 16+ base64 run, and passes; a marker the transform failed to
    parse as structure but which sits beside real body material is blocked
    fail-closed instead of shipping.
    """
    lower = text.lower()
    if "private key-----" not in lower:
        return False
    for raw in text.splitlines() or [text]:
        for piece in _split_serialized_pem_line(raw) if _PRIVATE_KEY_ESCAPED_SEPARATOR_HINT_RE.search(raw) else [(0, raw)]:
            line = piece[1] if isinstance(piece, tuple) else piece
            if "private key-----" not in line.lower():
                continue
            for m in _PRIVATE_KEY_INLINE_RUN_RE.finditer(line):
                run = m.group(0)
                if not _looks_like_english_token(run):
                    return True
    return False


def _has_orphan_full_width_base64_run(text: str) -> bool:
    """True when >=2 contiguous full-width base64 body lines survive, markerless.

    Transform- AND marker-INDEPENDENT backstop (#383 round 6): a truncated key
    whose body is split by a long non-base64 line leaves body lines with no
    surviving BEGIN/END marker and no placeholder in range — invisible to every
    marker- or placeholder-keyed check. Two or more contiguous strict base64
    lines of full PEM wrap width (>=40 chars) are the private-key body
    signature; ordinary prose/config never produces them. Lines already inside
    a redaction placeholder do not count. Fail-closed: a genuine multi-line
    base64 attachment on the cloud path blocks rather than ships (the durable
    store is untouched; a base64 blob has no embedding value anyway).
    """
    run = 0
    for kind, redact_start, content_end, _line_start, _marker_end in _pem_line_model(text):
        if kind in (_PEM_LINE_KIND_STRICT_B64, _PEM_LINE_KIND_PREFIXED_B64) and (
            content_end - redact_start
        ) >= 40:
            segment = text[redact_start:content_end]
            if "[LCM embedding privacy:" in segment or "[LCM sensitive redaction:" in segment:
                run = 0
                continue
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


_PRIVATE_KEY_PLACEHOLDER_RE = re.compile(
    r"\[LCM (?:sensitive redaction|embedding privacy):\s*name=private_key[^\]]*\]",
    re.IGNORECASE,
)


def _pem_fragment_near_private_key_placeholder(text: str) -> bool:
    """True when a 16+ base64-charset token sits near a PRIVATE_KEY placeholder.

    Marker-independent backstop (#383 rounds 4-6, RESTORED for #391): a
    private_key placeholder proves a private key was redacted at that spot, so
    any surviving unbroken 16+ base64-alphabet token in its adjacency window is
    an orphaned key-body fragment and blocks the dispatch fail-closed. Scoped
    to the private_key placeholder name deliberately: a 16+ base64 token beside
    a NON-key placeholder (e.g. a git SHA next to a redacted `api_key`) is
    ambiguous and must NOT over-block — marker-independent MULTI-LINE key
    bodies are covered instead by `_has_orphan_full_width_base64_run`.

    #391 history (owner-decided 2026-08-27): three successive line-model
    rewrites of this backstop each re-opened a leak shape main blocked
    (splitlines escape-blindness; whole-token fullmatch vs glued separators;
    placeholder glued to a following BEGIN marker hiding it from the modeled
    segment). Root cause: reconstructing adjacency through the line model
    keeps introducing blind spots the RAW-TEXT scan never had — placeholders
    and runs are found here by direct finditer over the text, immune to line
    classification. The ONLY delta vs the original is the pure-hex exclusion
    below, which surgically removes the one confirmed realistic over-block
    (#389: a redacted key beside a nearby git SHA / hash digest refused to
    embed). Hex is a base64-charset subset, and a real PEM body run
    effectively never renders as pure hex (DER-in-base64 carries mixed-case
    letters beyond a-f and/or `+` `/`), so excluding pure-hex runs keeps
    key+SHA prose embedding while every key-body fragment still blocks.
    Sub-16 fragments remain the documented precision boundary.
    """
    window = 160
    for pm in _PRIVATE_KEY_PLACEHOLDER_RE.finditer(text):
        lo = max(0, pm.start() - window)
        # Bound the scan to the adjacency window plus 16 chars of headroom: a
        # 16+ run STARTING anywhere inside the window still yields a >=16-char
        # match within [lo, hi] (its first 16 chars land before hi), while an
        # explicit endpos keeps each placeholder's scan O(window) instead of
        # O(remaining text) — the latter is O(placeholders * text) on inputs
        # with many placeholders and no nearby base64 (#383 round-6 perf fix).
        hi = min(len(text), pm.end() + window + _PRIVATE_KEY_STRICT_MIN_CHARS)
        for m in _PRIVATE_KEY_INLINE_RUN_RE.finditer(text, lo, hi):
            run = m.group(0)
            if _HEX_DIGEST_RE.fullmatch(run) is not None:
                # Pure-hex run = git SHA / hash digest, not key material (#389).
                continue
            if len(run) >= _PRIVATE_KEY_STRICT_MIN_CHARS and run not in pm.group(0):
                return True
    return False
