"""PEM line model for storage-boundary ingest protection."""

from __future__ import annotations

import re
from typing import Iterator

# THREAT MODEL (private-key redaction, declared after adversarial rounds 3-6
# of #365/#366): the scanner prevents ACCIDENTAL leakage of PEM-structured
# key material in realistic paste/log/transcript shapes (line-separator and
# whitespace variants, indentation and label prefixes, truncation, stripped
# markers, re-wrapped widths, inline one-line forms). It is NOT proof against
# an adversary who re-encodes content — no content-based redactor can be,
# since key material can be hex/base64-re-encoded, chunk-reordered, or
# otherwise transformed arbitrarily. That out-of-scope class explicitly
# INCLUDES a body deliberately de-contiguated — e.g. a >=159-char non-base64
# line interleaved between every single body line so no two full-width lines
# are adjacent and none sits within a placeholder's proximity window (#383
# round-6 finding B): such an author already owns the key and has restructured
# it beyond any realistic accidental paste/log/transcript shape. The dispatch
# validator is the fail-closed second layer for the ACCIDENTAL-shape class: a
# residual it recognizes as key structure (a marker beside a base64 run, a
# 16+ base64 run in a private_key placeholder's raw-text adjacency window —
# EXCLUDING pure-hex runs, i.e. git SHAs / hash digests, the one legitimate
# common neighbor of a redacted key (#389/#391 owner decision) — or >=2
# contiguous full-width body lines) blocks the embedding dispatch instead of
# leaking. Sub-16 fragments are the accepted precision boundary. It does not claim to
# block every conceivable unparseable arrangement — the deliberately
# de-contiguated body above is out of declared scope, not a guaranteed block.
# Base64 payload line: padding only ever trails (=/== suffix), so assignment
# prose like `environment=prod` never classifies as key body. Strict (>=16
# chars) lines anchor orphan-body detection; shorter lines join a run but
# never start structure on their own.
_PRIVATE_KEY_BODY_CHARS_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
# A pure-hex token (git SHA / hash digest) is a subset of the base64 charset,
# which is exactly why the removed token backstop over-blocked "key + git SHA"
# (#389). A real PEM base64 body effectively never renders as pure hex (it
# carries mixed-case g-z/G-Z letters and/or `+` `/`), so excluding pure-hex
# tokens keeps the #389 controls dispatching while still catching body lines.
_HEX_DIGEST_RE = re.compile(r"[0-9a-fA-F]{32,128}")
_PRIVATE_KEY_STRICT_MIN_CHARS = 16
# A real inline key has a substantial contiguous base64 run between markers;
# prose that merely names a BEGIN/END PRIVATE KEY marker pair with no key body
# between them does not and must not be consumed.
_PRIVATE_KEY_INLINE_RUN_RE = re.compile(r"[A-Za-z0-9+/]{16,}")
# One-or-more backslashes covers every nesting depth of serialization
# ("\\n" from json.dumps, "\\\\n" from json-of-json / logged JSON).
def _split_serialized_pem_line(content: str) -> Iterator[tuple[int, str]]:
    """Yield (offset, piece) splitting on escaped separators and bare quotes.

    Single linear pass. A regex here is quadratic on long backslash runs
    (every failed start re-consumes the run — measured 34s at 100k), so the
    scan is hand-rolled: a backslash run followed by an escaped line
    separator (n, r, r\\n at any depth, or u2028/u2029/u000b/u000c/u0085)
    or a bare quote splits the line; anything else is content.
    """
    n = len(content)
    seg_start = 0
    i = 0
    while i < n:
        ch = content[i]
        if ch == '"' or ch == "'":
            # Double quotes delimit JSON strings; single quotes delimit
            # Python repr()/logged dict values — both end a serialized value.
            yield seg_start, content[seg_start:i]
            i += 1
            seg_start = i
            continue
        if ch != "\\":
            i += 1
            continue
        run = i
        while run < n and content[run] == "\\":
            run += 1
        end = None
        if run < n:
            nxt = content[run]
            if nxt in "nf":
                # json.dumps shorthands: \n, and \f for U+000C (form feed,
                # a str.splitlines separator like the rest).
                end = run + 1
            elif nxt == "r":
                end = run + 1
                run2 = end
                while run2 < n and content[run2] == "\\":
                    run2 += 1
                if run2 > end and run2 < n and content[run2] == "n":
                    end = run2 + 1
            elif nxt == "u" and content[run + 1:run + 5] in (
                # The COMPLETE str.splitlines separator set in \uXXXX form:
                # VT, FF, FS, GS, RS, NEL, LS, PS. With \n/\r/\f handled as
                # shorthands above, every separator splitlines honors is
                # recognized when serialized — the enumeration is closed
                # (test derives it from splitlines itself).
                "000b", "000c", "001c", "001d", "001e",
                "0085", "2028", "2029",
            ):
                end = run + 5
        if end is None:
            # Not a separator: resume AT the char after the run so an escaped
            # quote (backslash + quote) still splits on its quote.
            i = run
            continue
        yield seg_start, content[seg_start:i]
        i = end
        seg_start = end
    yield seg_start, content[seg_start:]


_PRIVATE_KEY_ESCAPED_SEPARATOR_HINT_RE = re.compile(
    r"\\[rnfu]|[\"']"
)


def _normalize_escaped_solidus(text: str) -> str:
    """Linear any-depth solidus unescape for CLASSIFICATION only.

    A backslash run followed by "/" or "u002f"/"u002F" collapses to "/" —
    json.dumps nesting doubles the backslashes per depth, and a regex here
    would be quadratic on backslash storms (same class as the separator
    scan), so the pass is hand-rolled. Never applied to redaction offsets:
    callers classify on the normalized text and redact the raw slice.
    """
    if "\\" not in text:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        run = i
        while run < n and text[run] == "\\":
            run += 1
        if run < n and text[run] == "/":
            out.append("/")
            i = run + 1
            continue
        if run + 4 < n and text[run] == "u" and text[run + 1:run + 5] in ("002f", "002F"):
            out.append("/")
            i = run + 5
            continue
        out.append(text[i:run])
        i = run
    return "".join(out)


def _trim_pem_segment(piece: str) -> tuple[int, str]:
    """Linear trim of escape artifacts around a virtual PEM segment.

    Removes leading/trailing horizontal whitespace and escaped-tab (\t at any
    escaping depth) plus trailing escape backslashes (the next separator's or
    closing quote's escapes). Hand-rolled scans — the earlier regexes had
    nested quantifiers with exponential backtracking on backslash runs
    (CodeQL js/redos-class finding on this PR).
    """
    start = 0
    n = len(piece)
    while start < n:
        ch = piece[start]
        if ch in " \t":
            start += 1
            continue
        if ch == "\\":
            run = start
            while run < n and piece[run] == "\\":
                run += 1
            if run < n and piece[run] == "t":
                start = run + 1
                continue
        break
    end = n
    while end > start:
        ch = piece[end - 1]
        if ch in " \t":
            end -= 1
            continue
        if ch == "\\":
            end -= 1
            continue
        if ch == "t":
            run = end - 1
            while run > start and piece[run - 1] == "\\":
                run -= 1
            if run < end - 1:
                end = run
                continue
        break
    return start, piece[start:end]
# Real base64 key material is high-entropy; an English word/identifier shape
# (pure lowercase, or a single capitalized word) is overwhelmingly prose
# (p(base64) ~1e-6 for 16+ pure-lowercase chars), so prefixed-tail
# classification rejects it — "service productionservice" is config prose,
# not a key line. Applies only to prefix-stripped/prefixed tails; anchored
# body lines keep the plain alphabet rule.
def _looks_like_english_token(token: str) -> bool:
    """English word/identifier shape vs base64-ish key material.

    Letters-only tokens: pure lowercase ("example") and Camel/Title case
    ("Report", "ThisIsNotAKeyLine") read as prose. Base64 of random bytes is
    distinguished by uppercase RUNS: a mixed-case letters-only token with 3+
    consecutive capitals ("MIIEvQIBADANBgkq…") — or an all-caps token — stays
    body-eligible. Anything with a digit, +, /, or = is never English-shaped.
    """
    if not token.isalpha() or not token.isascii():
        return False
    if token.islower():
        return True
    if token.isupper():
        return False
    run = 0
    for ch in token:
        if ch.isupper():
            run += 1
            if run >= 3:
                return False
        else:
            run = 0
    return True
# No ^$ anchors: used via fullmatch(text, pos, endpos), where "^" would only
# match at the real string start, not at pos.
# Backslash admits JSON/log-serialized keys whose newlines are literal \n
# two-char sequences — a one-physical-line PEM in a serialized log is a common
# accidental paste shape.
_PRIVATE_KEY_INLINE_SPAN_RE = re.compile(r"[A-Za-z0-9+/=\t\\ ]*")
# RFC 1421 encapsulated-header line (encrypted traditional PEM):
# "Proc-Type: 4,ENCRYPTED" / "DEK-Info: AES-128-CBC,..." — appears between
# BEGIN and the base64 body, followed by one blank line.
_PRIVATE_KEY_ARMOR_HEADER_RE = re.compile(r"^[A-Za-z0-9-]{2,32}:\s?\S.*$")
_PRIVATE_KEY_MAX_ARMOR_HEADERS = 5
_PRIVATE_KEY_BEGIN_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
_PRIVATE_KEY_END_RE = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
# ANY PEM armor marker (private key, certificate, public key, certificate
# request, …): a truncated private-key body can share a line with a following
# marker of any armor type (#383 finding #3), so the leading-body-run split
# keys off the generic marker. Hyphenated labels (CERTIFICATE-REQUEST,
# X509-CRL) are legal armor labels and must match too.
_PEM_ANY_MARKER_RE = re.compile(r"-----(?:BEGIN|END) [A-Z0-9 _-]*[A-Z0-9]-----", re.IGNORECASE)


_PEM_LINE_KIND_OTHER = 0
_PEM_LINE_KIND_BEGIN = 1
_PEM_LINE_KIND_END = 2
_PEM_LINE_KIND_STRICT_B64 = 3
_PEM_LINE_KIND_SHORT_B64 = 4
_PEM_LINE_KIND_END_INLINE = 5
_PEM_LINE_KIND_ARMOR = 6
_PEM_LINE_KIND_BLANK = 7
_PEM_LINE_KIND_END_LOOSE = 8
_PEM_LINE_KIND_PREFIXED_B64 = 9
_PEM_LINE_KIND_LENIENT_B64 = 10


def _pem_leading_body_run(prefix: str) -> tuple[bool, bool, int]:
    """Scan the leading run of base64-body tokens in ``prefix``.

    Token separators are physical spaces/tabs AND serialized escaped tabs (a
    backslash run followed by ``t``, any nesting depth — the round-18 solidus
    lesson: hand-rolled run-jumps, never a regex over backslash runs). Returns
    ``(saw_body, have_evidence, run_end)`` where ``run_end`` is the raw offset
    just past the last body token; evidence = any digit/base64-symbol in the
    run (keeps all-caps doc headings out — round-21 precision).
    """
    n = len(prefix)
    scan = 0
    run_end = 0
    saw_body = False
    have_evidence = False
    while scan < n:
        # skip separators (physical whitespace / escaped tabs)
        while scan < n:
            if prefix[scan] in " \t":
                scan += 1
                continue
            if prefix[scan] == "\\":
                j = scan
                while j < n and prefix[j] == "\\":
                    j += 1
                if j < n and prefix[j] == "t":
                    scan = j + 1
                    continue
            break
        tok_start = scan
        while scan < n:
            if prefix[scan] in " \t":
                break
            if prefix[scan] == "\\":
                j = scan
                while j < n and prefix[j] == "\\":
                    j += 1
                if j < n and prefix[j] == "t":
                    break
                scan = j
                continue
            scan += 1
        if scan == tok_start:
            break
        ctok = _normalize_escaped_solidus(prefix[tok_start:scan])
        if (
            ctok
            and _PRIVATE_KEY_BODY_CHARS_RE.fullmatch(ctok) is not None
            and not _looks_like_english_token(ctok)
        ):
            saw_body = True
            run_end = scan
            if any(ch.isdigit() or ch in "+/=" for ch in ctok):
                have_evidence = True
        else:
            break
    return saw_body, have_evidence, run_end


def _pem_line_model(text: str) -> list[tuple[int, int, int, int, int]]:
    """Classify every line of ``text`` once for the private-key scanner.

    Returns ``(kind, redact_start, content_end, line_start, marker_end)`` per
    line. Classification runs on the horizontally-stripped content, and
    ``str.splitlines`` handles every line separator (\\n, \\r\\n, bare \\r,
    unicode line breaks) — CR-only or indented key material must not slip past
    a \\n-keyed scan. ``redact_start`` is where a redaction of this line should
    begin (the marker start for a BEGIN line so unrelated prefixes like
    ``key: `` survive; the stripped content start otherwise); ``content_end``
    excludes the line terminator so surrounding newlines are preserved;
    ``marker_end`` is the END marker's absolute end for END/END_INLINE lines
    (-1 otherwise). END_INLINE — a line that contains an END marker amid other
    content — terminates a key body only when a base64 run precedes it
    (``label: -----END PRIVATE KEY-----`` after a body is key structure; prose
    merely mentioning the marker, with no body above, never matches).
    """
    model: list[tuple[int, int, int, int, int]] = []
    offset = 0
    segments: list[tuple[int, str]] = []
    for raw in text.splitlines(keepends=True):
        line_start = offset
        offset += len(raw)
        content = raw.splitlines()[0] if raw else ""
        if _PRIVATE_KEY_ESCAPED_SEPARATOR_HINT_RE.search(content) is not None:
            # JSON/log-serialized key: literal \n (optionally \r\n) escape
            # sequences at any serialization depth are the key's real line
            # separators, and bare quotes delimit the serialized string — both
            # split into virtual lines so armor headers, truncated bodies, and
            # short tails inside serialized text classify exactly like
            # unserialized ones (a tail glued to the JSON closer, "CDEF\"}",
            # still classifies). Offsets stay absolute; spans stay exact.
            for seg_start, piece in _split_serialized_pem_line(content):
                # Serialization artifacts are not content: escaped-tab
                # indentation (\t at any depth) leads a segment, and the
                # escape backslashes of the following separator/closing quote
                # trail it — trim both so the payload classifies.
                trim_off, piece = _trim_pem_segment(piece)
                seg_start += trim_off
                segments.append((line_start + seg_start, piece))
        else:
            segments.append((line_start, content))
    for line_start, content in segments:
        stripped = content.strip(" \t")
        # JSON's optional solidus escape (\/) is legal inside base64 bodies;
        # classify on the unescaped form, keep offsets on the raw text.
        class_stripped = _normalize_escaped_solidus(stripped)
        lead = content.find(stripped[:1]) if stripped else 0
        c_start = line_start + (lead if stripped else 0)
        c_end = c_start + len(stripped)
        kind = _PEM_LINE_KIND_OTHER
        redact_start = c_start
        marker_end = -1
        if "PRIVATE KEY-----" not in stripped.upper():
            # A truncated private-key body can be glued to a following NON-
            # private-key PEM marker (a CERTIFICATE / PUBLIC KEY the key was
            # concatenated before) — this line has no "PRIVATE KEY-----" so the
            # branch below never sees it (#383 adversarial-review finding #3).
            # Split the leading base64 body run off as a forced STRICT_B64 entry
            # so an open truncated block consumes and redacts it; the remainder
            # is OTHER (it opens nothing). Digit/base64-symbol evidence required
            # so all-caps doc headings before a marker are not swept.
            gmarker = _PEM_ANY_MARKER_RE.search(stripped)
            if gmarker is not None and gmarker.start() > 0:
                prefix_seg = stripped[: gmarker.start()]
                # Strip a leading markdown blockquote / diff prefix ("> ", "+ ",
                # "- ") so a prefixed truncated tail before a non-private marker
                # still scans as body (#383 round-7, ckrYc). Offset the emitted
                # STRICT_B64 span past the stripped prefix so raw bounds stay exact.
                pfx = 0
                while pfx < len(prefix_seg) and prefix_seg[pfx] in ">+- \t":
                    pfx += 1
                saw_body, have_evidence, run_end = _pem_leading_body_run(
                    prefix_seg[pfx:]
                )
                if saw_body and have_evidence:
                    body_start = c_start + pfx
                    model.append(
                        (_PEM_LINE_KIND_STRICT_B64, body_start, body_start + run_end, body_start, -1)
                    )
                    rem_start = body_start + run_end
                    model.append(
                        (_PEM_LINE_KIND_OTHER, rem_start, c_end, rem_start, -1)
                    )
                    continue
        if "PRIVATE KEY-----" in stripped.upper():
            begin = _PRIVATE_KEY_BEGIN_RE.search(stripped)
            end = _PRIVATE_KEY_END_RE.search(stripped)
            if begin is not None and begin.end() == len(stripped):
                # A truncated block's body run can share one physical/serialized
                # line with the BEGIN of the next block. Split off the LEADING
                # base64-body run (not just the first token, and short tokens
                # count — the adjacent truncated BEGIN is the structural
                # evidence) as a virtual body line so the open block consumes
                # and redacts its tail; the BEGIN marker becomes its own virtual
                # line. Otherwise the whole-line BEGIN classification orphans the
                # run before the next block's redact_start (#383). Require
                # digit/base64-symbol evidence in the run so all-caps doc
                # headings ("IMPORTANT NOTICE") before a quoted BEGIN are not
                # swept (round-21 precision). Classify normalized, redact raw.
                saw_body = False
                have_evidence = False
                run_end = 0
                if end is None:
                    saw_body, have_evidence, run_end = _pem_leading_body_run(
                        stripped[: begin.start()]
                    )
                if saw_body and have_evidence:
                    marker_start = c_start + begin.start()
                    model.append(
                        (
                            _PEM_LINE_KIND_STRICT_B64,
                            c_start,
                            c_start + run_end,
                            c_start,
                            -1,
                        )
                    )
                    model.append(
                        (
                            _PEM_LINE_KIND_BEGIN,
                            marker_start,
                            c_end,
                            marker_start,
                            -1,
                        )
                    )
                    continue
                kind = _PEM_LINE_KIND_BEGIN
                redact_start = c_start + begin.start()
                if end is not None and end.end() <= begin.start():
                    # A compacted "END ... BEGIN" line: the leading END can
                    # close a preceding orphan run before the BEGIN opens the
                    # next block (marker_end carries the END bound).
                    marker_end = c_start + end.end()
            elif (
                begin is not None
                and (end is None or end.start() < begin.start())
                and _PRIVATE_KEY_INLINE_SPAN_RE.fullmatch(
                    stripped, begin.end()
                )
                and _PRIVATE_KEY_INLINE_RUN_RE.search(stripped, begin.end())
            ):
                # Newline-normalized form: the base64 payload sits ON the
                # BEGIN line ("-----BEGIN ...----- MIIE…"), END on a later
                # line. Treat as a BEGIN whose block span already includes
                # the same-line body.
                kind = _PEM_LINE_KIND_BEGIN
                redact_start = c_start + begin.start()
            elif end is not None and begin is None:
                marker_end = c_start + end.end()
                if end.start() == 0 and end.end() == len(stripped):
                    kind = _PEM_LINE_KIND_END
                elif not stripped[end.end():].strip("\"'),.;:]}"):
                    # Prefix of any length (labels, timestamped log lines)
                    # with nothing but closing punctuation after the marker
                    # (serialization/prose terminators): key structure.
                    kind = _PEM_LINE_KIND_END_INLINE
                else:
                    # A SUFFIX after the marker ("see -----END ... ----- for
                    # format"): prose context — terminates a BEGIN-anchored
                    # block only, never an orphan run.
                    kind = _PEM_LINE_KIND_END_LOOSE
        elif (
            class_stripped
            and _PRIVATE_KEY_BODY_CHARS_RE.fullmatch(class_stripped) is not None
            and not _looks_like_english_token(class_stripped)
        ):
            if len(class_stripped) >= _PRIVATE_KEY_STRICT_MIN_CHARS:
                kind = _PEM_LINE_KIND_STRICT_B64
            else:
                kind = _PEM_LINE_KIND_SHORT_B64
        elif not stripped:
            kind = _PEM_LINE_KIND_BLANK
        elif _PRIVATE_KEY_ARMOR_HEADER_RE.fullmatch(stripped) is not None:
            kind = _PEM_LINE_KIND_ARMOR
        else:
            # Log-collector shape: every body line carries a prefix
            # ("INFO MII…", "2026-… DEBUG MII…"). A line whose final
            # whitespace-separated token is a full-width base64 token, with at
            # most 4 prefix tokens, joins a BEGIN-anchored body run (only —
            # never orphan structure, so hash dumps near prose stay intact).
            parts = stripped.rsplit(None, 1)
            if (
                len(parts) == 2
                and len(parts[0].split()) <= 6
                and len(parts[1]) >= _PRIVATE_KEY_STRICT_MIN_CHARS
                and _PRIVATE_KEY_BODY_CHARS_RE.fullmatch(parts[1]) is not None
                and not _looks_like_english_token(parts[1])
            ):
                kind = _PEM_LINE_KIND_PREFIXED_B64
        model.append((kind, redact_start, c_end, line_start, marker_end))
    return model
