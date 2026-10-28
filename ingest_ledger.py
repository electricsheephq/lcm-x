"""Connection-level ingest ledger API; not yet called by ingest or compress.

Placements are in view order with non-decreasing bounds. Natural gaps are
clamped to the preceding gap, so a later supersedes joins it after earlier NEWs.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import struct
from typing import NamedTuple

from .message_content import normalize_content_value
from .reconcile import ReconcileMixin, _strip_volatile_host_scaffolding
from .sqlite_session import TransactionOwner
from .store import ledger_digest

ORDINAL_STEP = 1 << 20
MIN_GAP = 1 << 10
STAMP_WRITTEN, STAMP_EXISTS, STAMP_CAPPED = "written", "exists", "capped"
_MAX_ORDINAL = (1 << 63) - 1
_ENTRY = struct.Struct(">16sB16s16sq")


class LedgerError(ValueError):
    """Invalid ledger input or state."""


class LedgerFormatError(LedgerError):
    """Unknown or malformed vector format."""


class LedgerTransactionError(LedgerError):
    """A writer was called without the store's transaction owner."""


class ViewEntry(NamedTuple):
    identity_digest: bytes
    stamp_digest: bytes | None
    uidset_digest: bytes | None
    ref: int | None
    is_descriptor: bool = False


class Placement(NamedTuple):
    left: int | None
    right: int | None
    supersedes: int | None = None


def identity_digest(salt, message):
    """Digest PRE-redaction fields: callers must pass the message before ingest protection.

    Canonicalize fields, stripping the todo annotation span then model-switch
    prefix. Classification owns survival projection, host-pruned tool aliases,
    carrier stripping, host rewrites, store-dependent out-of-band stripping and
    persisted-output recovery; none of those run here.
    """
    role = str(message.get("role") or "unknown")
    content = _strip_volatile_host_scaffolding(normalize_content_value(message.get("content")) or "")
    return ledger_digest(salt, role, content,
                         str(message.get("tool_call_id") or ""),
                         ReconcileMixin._stable_tool_calls_identity(message.get("tool_calls")),
                         str(message.get("tool_name") or message.get("name") or "") if role == "tool" else "")


def stamp_digest(salt, stamp):
    return None if stamp is None else ledger_digest(salt, stamp)


def uidset_digest(salt, uid, absorbed):
    absorbed = tuple(absorbed or ())
    if (uid is not None and not isinstance(uid, str)) or any(not isinstance(value, str) for value in absorbed):
        raise LedgerError("uids must be strings")
    return None if not uid else ledger_digest(salt, uid, *sorted(set(absorbed)))


def encode_entries(entries):
    result = bytearray(b"\x01")
    for entry in entries:
        identity, stamp, uidset, ref, descriptor = ViewEntry(*entry)
        descriptor = bool(descriptor)
        if any(value is not None and (not isinstance(value, bytes) or len(value) != 16)
               for value in (identity, stamp, uidset)) or identity is None:
            raise LedgerFormatError("entry digests must be 16 bytes")
        if ref is None and descriptor:
            raise LedgerFormatError("unbound entries cannot be descriptors")
        flags = int(stamp is not None) | (int(uidset is not None) << 1) | (int(descriptor) << 2) | (int(ref is None) << 3)
        try:
            result.extend(_ENTRY.pack(identity, flags, stamp or bytes(16), uidset or bytes(16), 0 if ref is None else ref))
        except (struct.error, OverflowError) as exc:
            raise LedgerFormatError("invalid entry reference or flags") from exc
    return bytes(result)


def decode_entries(blob):
    if not blob or blob[0] != 1 or (len(blob) - 1) % _ENTRY.size:
        raise LedgerFormatError("unknown or malformed entry format")
    entries = []
    for identity, flags, stamp, uidset, ref in _ENTRY.iter_unpack(blob[1:]):
        if (flags & ~15 or (flags & 8 and (ref != 0 or flags & 4))
                or (not flags & 1 and stamp != bytes(16)) or (not flags & 2 and uidset != bytes(16))):
            raise LedgerFormatError("invalid entry flags or absent digest")
        entries.append(ViewEntry(identity, stamp if flags & 1 else None,
                                 uidset if flags & 2 else None, None if flags & 8 else ref, bool(flags & 4)))
    return entries


def prefix_chain(salt, entries):
    chain, previous = bytearray(), "chain"
    for entry in entries:
        previous = ledger_digest(salt, previous, entry[0], entry[1] or b"", entry[2] or b"")
        chain.extend(previous)
    return bytes(chain)


def common_prefix_len(chain_a, chain_b):
    if len(chain_a) % 16 or len(chain_b) % 16:
        raise LedgerFormatError("prefix chains must contain complete digests")
    for offset in range(0, min(len(chain_a), len(chain_b)), 16):
        if chain_a[offset:offset + 16] != chain_b[offset:offset + 16]:
            return offset // 16
    return min(len(chain_a), len(chain_b)) // 16


def plan_ordinals(existing, placements):
    """Return (new ordinals in placement order, changed old-to-new ordinals).

    A supersedes target determines its natural gap, subject to view-order clamping.
    Tied minimal windows choose the leftmost. Duplicate existing labels (part
    witnesses) share one label and therefore one re-spacing map entry.
    """
    original = sorted(set(existing))
    if any(not isinstance(value, int) or not 0 < value <= _MAX_ORDINAL for value in original):
        raise LedgerError("existing ordinals must be positive signed 64-bit integers")
    positions, groups = original.copy(), defaultdict(list)
    previous_left, previous_right, previous_gap = 0, 0, 0
    for index, placement in enumerate(placements):
        left, right, supersedes = Placement(*placement)
        if any(value is not None and value not in original for value in (left, right, supersedes)):
            raise LedgerError("placement bound is not an existing ordinal")
        left_bound, right_bound = left or 0, right if right is not None else _MAX_ORDINAL + 1
        if left_bound >= right_bound or left_bound < previous_left or right_bound < previous_right:
            raise LedgerError("placement bounds must be non-decreasing in view order")
        previous_left, previous_right = left_bound, right_bound
        if supersedes is not None:
            gap = bisect_left(original, supersedes) + 1
        else:
            if left is not None and right is not None and left >= right:
                raise LedgerError("placement bounds are out of order")
            gap = len(original) if right is None else bisect_left(original, right)
        previous_gap = max(gap, previous_gap)
        groups[previous_gap].append(index)
    result = [None] * sum(map(len, groups.values()))
    for gap, indexes in sorted(groups.items()):
        if result[indexes[0]] is not None:
            continue  # A preceding re-spacing already placed this whole gap.
        lower = positions[gap - 1] if gap else 0
        upper = positions[gap] if gap < len(positions) else None
        step = ORDINAL_STEP if upper is None else (upper - lower) // (len(indexes) + 1)
        if step > 0 and lower + step * len(indexes) <= _MAX_ORDINAL:
            for offset, index in enumerate(indexes, 1):
                result[index] = (lower + step * offset if upper is None
                                 else lower + (upper - lower) * offset // (len(indexes) + 1))
            continue
        # Width counts movable EXISTING labels; outer endpoints remain fixed.
        window = None
        for width in range(len(positions) + 1):
            for start in range(max(0, gap - width), min(gap, len(positions) - width) + 1):
                end = start + width
                lower = positions[start - 1] if start else 0
                count = width + sum(len(groups[g]) for g in range(start, end + 1))
                step = ((positions[end] - lower) // (count + 1) if end < len(positions)
                        else min(ORDINAL_STEP, (_MAX_ORDINAL - lower) // count))
                if step >= MIN_GAP:
                    window = start, end, lower, step
                    break
            if window is not None:
                break
        if window is None:
            raise LedgerError("signed 64-bit ordinal space exhausted")
        start, end, value, step = window
        for g in range(start, end + 1):
            for index in groups[g]:
                value += step
                result[index] = value
            if g < end:
                value += step
                positions[g] = value
    return result, {old: new for old, new in zip(original, positions) if old != new}


def lineage_key(host_uid_key: str | None, session_id: str) -> str:
    """Use the resolved key (the host lookup's first value), never cache it here."""
    if host_uid_key and host_uid_key.startswith("provisional:"):
        raise LedgerError("resolved lineage uses the provisional namespace")
    return host_uid_key if host_uid_key else f"provisional:{session_id}"


def _require_owner(conn):
    if not TransactionOwner.owns_connection(conn):
        raise LedgerTransactionError("ledger write requires store.atomic")


def _rows(cursor):
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def next_view_seq(conn):
    _require_owner(conn)
    conn.execute("UPDATE ledger_meta SET value=CAST(value AS INTEGER)+1 WHERE key='view_seq'")
    row = conn.execute("SELECT value FROM ledger_meta WHERE key='view_seq'").fetchone()
    if row is None:
        raise LedgerError("ledger metadata is not initialized")
    return int(row[0])


def insert_bindings(conn, lineage, rows):
    _require_owner(conn)
    columns = ("ordinal", "store_id", "identity_digest", "stamp", "disposition", "tag", "key_rank", "view_seq")
    return [conn.execute(
        "INSERT INTO occurrence_bindings(lineage, " + ",".join(columns) + ") VALUES (?,?,?,?,?,?,?,?,?)",
        (lineage, *(row.get(column) for column in columns)),
    ).lastrowid for row in rows]


def apply_respace(conn, lineage, mapping):
    _require_owner(conn)
    occupied = {row[0] for row in conn.execute(
        "SELECT ordinal FROM occurrence_bindings WHERE lineage=? AND ordinal IS NOT NULL", (lineage,))}
    if (any(value <= 0 or value > _MAX_ORDINAL for value in mapping.values())
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) & (occupied - mapping.keys())):
        raise LedgerError("invalid re-spacing map")
    temporary, count, staged = min(occupied | {0}), 0, []
    for old, new in mapping.items():
        temporary -= 1
        if temporary < -(1 << 63):
            raise LedgerError("temporary ordinal space exhausted")
        cursor = conn.execute("UPDATE occurrence_bindings SET ordinal=? WHERE lineage=? AND ordinal=?",
                              (temporary, lineage, old))
        count += cursor.rowcount
        staged.append((temporary, new))
    for temporary, new in staged:
        conn.execute("UPDATE occurrence_bindings SET ordinal=? WHERE lineage=? AND ordinal=?", (new, lineage, temporary))
    return count


def write_view(conn, salt, lineage, kind, state, entries):
    _require_owner(conn)
    entries = list(entries)
    encoded, chain = encode_entries(entries), prefix_chain(salt, entries)
    seq = next_view_seq(conn)
    conn.execute("INSERT INTO ingest_views VALUES (?,?,?,?,?,?)", (lineage, seq, kind, state, encoded, chain))
    conn.execute("DELETE FROM ingest_views WHERE lineage=? AND kind=? AND view_seq NOT IN "
                 "(SELECT view_seq FROM ingest_views WHERE lineage=? AND kind=? ORDER BY view_seq DESC LIMIT 2)",
                 (lineage, kind, lineage, kind))
    return seq


def add_stamp_observation(conn, lineage, occ_id, stamp, kind, view_seq, source="ledger"):
    _require_owner(conn)
    if stamp is None:
        raise LedgerError("stamp observation requires a stamp")
    if not conn.execute("SELECT 1 FROM occurrence_bindings WHERE lineage=? AND occ_id=?", (lineage, occ_id)).fetchone():
        raise LedgerError("stamp observation does not belong to lineage")
    if conn.execute("SELECT 1 FROM stamp_observations WHERE occ_id=? AND kind=? AND stamp=?",
                    (occ_id, kind, stamp)).fetchone():
        return STAMP_EXISTS
    if conn.execute("SELECT count(*) FROM stamp_observations WHERE occ_id=? AND kind=?", (occ_id, kind)).fetchone()[0] >= 4:
        return STAMP_CAPPED
    conn.execute("INSERT INTO stamp_observations VALUES (?,?,?,?,?,?)", (lineage, occ_id, stamp, kind, view_seq, source))
    capped = conn.execute("SELECT count(*) FROM (SELECT occ_id FROM stamp_observations "
                          "WHERE lineage=? AND kind=? GROUP BY occ_id HAVING count(*)>=4)", (lineage, kind)).fetchone()[0]
    if capped >= 3:
        _update_mark(conn, lineage, "stamp_policy", "stamp_unreliable")
    return STAMP_WRITTEN


def _update_mark(conn, lineage, column, value):
    _require_owner(conn)
    cursor = conn.execute(f"INSERT INTO lineage_marks(lineage,{column}) VALUES (?,?) "
                          f"ON CONFLICT(lineage) DO UPDATE SET {column}=excluded.{column} "
                          f"WHERE lineage_marks.{column} IS NOT excluded.{column}", (lineage, value))
    return bool(cursor.rowcount)


def update_store_mark(conn, lineage, max_store_id):
    return _update_mark(conn, lineage, "max_store_id", max_store_id)


def update_leaf_mark(conn, lineage, max_leaf_id):
    return _update_mark(conn, lineage, "max_leaf_id", max_leaf_id)


def update_legacy_proof_digest(conn, lineage, digest):
    return _update_mark(conn, lineage, "legacy_proof_digest", digest)


def newest_view(conn, lineage, kind):
    rows = _rows(conn.execute("SELECT * FROM ingest_views WHERE lineage=? AND kind=? ORDER BY view_seq DESC LIMIT 1",
                              (lineage, kind)))
    return rows[0] if rows else None


def lineage_ordinals(conn, lineage):
    return [row[0] for row in conn.execute("SELECT DISTINCT ordinal FROM occurrence_bindings "
                                          "WHERE lineage=? AND ordinal IS NOT NULL ORDER BY ordinal", (lineage,))]


def bindings_by_digest(conn, lineage, identity_digest):
    return _rows(conn.execute("SELECT * FROM occurrence_bindings WHERE lineage=? AND identity_digest=? ORDER BY ordinal,occ_id",
                              (lineage, identity_digest)))


def marks(conn, lineage):
    rows = _rows(conn.execute("SELECT * FROM lineage_marks WHERE lineage=?", (lineage,)))
    return rows[0] if rows else None
