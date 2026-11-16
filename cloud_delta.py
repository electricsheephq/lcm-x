"""Page-delta codec for staged cloud copies; no live-store or transport access."""
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
from typing import NamedTuple
import zlib

MAX_OBJECT_BYTES = 2**25
SIZE_FACTOR = 2
MAX_CHAIN_GENERATIONS = 1024
SKIP_CEILING_S = 86400
_MAGIC = b"LCMXPD1\n"
_HEX = r"[0-9a-f]{64}"
_COPY = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


class DeltaRefused(Exception):
    def __init__(self, code, **detail):
        self.code, self.detail = code, detail
        super().__init__(code, detail)


class PageList(NamedTuple):
    page_size: int
    pages: int
    digests: bytes
    file_bytes: int
    file_sha256: str
    page_list_sha256: str


class DeltaObject(NamedTuple):
    path: Path
    sha256: str
    bytes: int
    records: int


def _page_size(p):
    return type(p) is int and 512 <= p <= 65536 and p & (p-1) == 0


def _name(value, grammar):
    if not isinstance(value, str) or re.fullmatch(grammar, value) is None:
        raise DeltaRefused("bad_name")
    return value


def hash_pages(path, *, expected_page_size=None):
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if not size:
            raise DeltaRefused("file_not_whole_pages")
        header = handle.read(18)
        p = int.from_bytes(header[16:18], "big") if len(header) == 18 else 0
        p = 65536 if p == 1 else p
        if not _page_size(p) or (expected_page_size is not None and p != expected_page_size):
            raise DeltaRefused("page_size")
        whole, digests, count = hashlib.sha256(), bytearray(), 0
        page = header + handle.read(p-18)
        while page:
            if len(page) != p:
                raise DeltaRefused("file_not_whole_pages")
            whole.update(page)
            digests.extend(hashlib.sha256(page).digest())
            count += len(page)
            page = handle.read(p)
        if count != size:
            raise DeltaRefused("file_not_whole_pages")
    return PageList(p, count//p, bytes(digests), count, whole.hexdigest(), hashlib.sha256(digests).hexdigest())


def load_head_list(pages_dir, copy_id, page_list_sha256, pages):
    _name(copy_id, _COPY)
    for suffix in ("last", "pending"):
        path = Path(pages_dir) / f"{copy_id}.{suffix}"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                meta = os.fstat(handle.fileno())
                if not stat.S_ISREG(meta.st_mode) or meta.st_size != pages*32:
                    continue
                data = handle.read(pages*32+1)
            if len(data) != pages*32 or hashlib.sha256(data).hexdigest() != page_list_sha256:
                continue
            if suffix == "pending":
                os.replace(path, path.with_suffix(".last"))
            return data
        except OSError:
            continue
    return None


def save_pending_list(pages_dir, copy_id, digests):
    _name(copy_id, _COPY)
    folder = Path(pages_dir)
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = folder / f"{copy_id}.{os.urandom(8).hex()}.tmp"
    with os.fdopen(os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
        handle.write(digests)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, folder / f"{copy_id}.pending")


def promote_pending_list(pages_dir, copy_id):
    _name(copy_id, _COPY)
    try:
        os.replace(Path(pages_dir) / f"{copy_id}.pending", Path(pages_dir) / f"{copy_id}.last")
    except FileNotFoundError:
        pass


def changed_pages(prev_digests, new_digests):
    if len(prev_digests) % 32 or len(new_digests) % 32:
        raise DeltaRefused("bad_page_list")
    return [i//32 for i in range(0, len(new_digests), 32)
            if new_digests[i:i+32] != prev_digests[i:i+32]]


def max_records(page_size):
    return (2**25 - 2**16) // (4 + page_size)


def read_object(data, page_size, records=None):
    """Consume to exhaustion before trusting records; even the final step may refuse."""
    if len(data) < 16 or data[:8] != _MAGIC:
        raise DeltaRefused("delta_magic")
    p, count = struct.unpack(">II", data[8:16])
    if p != page_size:
        raise DeltaRefused("delta_page_size")
    if records is not None and records != count:
        raise DeltaRefused("delta_count")
    stream, buf, seen, last = zlib.decompressobj(), b"", 0, -1
    try:
        for offset in range(16, len(data), 65536):
            chunk = data[offset:offset+65536]
            while chunk:
                buf += stream.decompress(chunk, max(1, 4+p-len(buf)))
                chunk = stream.unconsumed_tail
                if seen == count and buf:
                    raise DeltaRefused("delta_stream")
                if len(buf) == 4+p:
                    index = int.from_bytes(buf[:4], "big")
                    if index <= last:
                        raise DeltaRefused("delta_order", index=index)
                    seen, last = seen+1, index
                    yield index, buf[4:]
                    buf = b""
                if stream.eof:
                    if stream.unused_data or offset+65536 < len(data):
                        raise DeltaRefused("delta_trailing")
                    break
        if not stream.eof or buf or seen != count:
            raise DeltaRefused("delta_stream")
    except zlib.error:
        raise DeltaRefused("delta_stream") from None


def verify_delta(paths, prev_digests, new):
    indexes = []
    for path in paths:
        for index, page in read_object(Path(path).read_bytes(), new.page_size):
            if index >= new.pages or hashlib.sha256(page).digest() != new.digests[index*32:(index+1)*32]:
                raise DeltaRefused("delta_selfcheck", index=index)
            indexes.append(index)
    if indexes != changed_pages(prev_digests, new.digests):
        raise DeltaRefused("delta_selfcheck")


def encode_delta(db_path, prev_digests, new, out_dir, *, records_per_object=None):
    cap = max_records(new.page_size) if records_per_object is None else records_per_object
    if not 1 <= cap <= max_records(new.page_size):
        raise ValueError("records_per_object outside writer limits")
    indexes, result = changed_pages(prev_digests, new.digests), []
    with open(db_path, "rb") as handle:
        if os.fstat(handle.fileno()).st_size != new.file_bytes:
            raise DeltaRefused("delta_selfcheck")
        for start in range(0, len(indexes), cap):
            group = indexes[start:start+cap]
            data = bytearray(_MAGIC + struct.pack(">II", new.page_size, len(group)))
            stream = zlib.compressobj(6)
            for index in group:
                handle.seek(index*new.page_size)
                page = handle.read(new.page_size)
                if len(page) != new.page_size:
                    raise DeltaRefused("delta_selfcheck", index=index)
                data.extend(stream.compress(struct.pack(">I", index)+page))
            data.extend(stream.flush())
            if len(data) > MAX_OBJECT_BYTES:
                raise DeltaRefused("delta_object_too_large", bytes=len(data))
            digest = hashlib.sha256(data).hexdigest()
            folder = Path(out_dir)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / digest
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600), "wb") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(data)
            result.append(DeltaObject(path, digest, len(data), len(group)))
        if os.fstat(handle.fileno()).st_size != new.file_bytes:
            raise DeltaRefused("delta_selfcheck")
    verify_delta([item.path for item in result], prev_digests, new)
    return result


def _fields(value, **fields):
    if type(value) is not dict:
        raise DeltaRefused("bad_database_block")
    for key, kind in fields.items():
        field = value.get(key)
        if type(field) is not kind or (kind is int and field < 0):
            raise DeltaRefused("bad_database_block")
        if key.endswith("sha256") and re.fullmatch(_HEX, field) is None:
            raise DeltaRefused("bad_database_block")


def _shape(db, generation):
    _fields(db, page_size=int, file_bytes=int, file_sha256=str, page_list_sha256=str, base=dict, deltas=list)
    base, entries, p = db["base"], db["deltas"], db["page_size"]
    _fields(base, generation=int, file_bytes=int, file_sha256=str, parts=list)
    if not _page_size(p) or type(generation) is not int or generation < 0:
        raise DeltaRefused("bad_database_block")
    for ref in base["parts"]:
        _fields(ref, key=str, bytes=int)
        _name(ref["key"], rf"parts/{_HEX}\.[0-9a-f]{{32}}")
    if not base["file_bytes"] or base["file_bytes"] % p or sum(r["bytes"] for r in base["parts"]) != base["file_bytes"]:
        raise DeltaRefused("bad_database_block")
    for k, entry in enumerate(entries):
        _fields(entry, generation=int, file_pages=int, objects=list)
        if entry["file_pages"] < 1:
            raise DeltaRefused("bad_database_block")
        for ref in entry["objects"]:
            _fields(ref, key=str, bytes=int, records=int)
            _name(ref["key"], rf"deltas/{_HEX}\.[0-9a-f]{{32}}")
        if entry["generation"] != base["generation"]+1+k:
            raise DeltaRefused("chain_generation")
    last_generation = entries[-1]["generation"] if entries else base["generation"]
    if last_generation != generation:
        raise DeltaRefused("chain_generation")
    if db["file_bytes"] != (entries[-1]["file_pages"]*p if entries else base["file_bytes"]):
        raise DeltaRefused("bad_database_block")
    if not entries and db["file_sha256"] != base["file_sha256"]:
        raise DeltaRefused("bad_database_block")


def _fetch(ref, fetch):
    key = ref["key"]
    try:
        data = fetch(key)
    except (KeyError, FileNotFoundError):
        raise DeltaRefused("object_missing", key=key) from None
    if len(data) != ref["bytes"]:
        raise DeltaRefused("object_bytes", key=key)
    if hashlib.sha256(data).hexdigest() != key.split("/")[1][:64]:
        raise DeltaRefused("object_sha256", key=key)
    return data


def rebuild(path, database, generation, fetch, *, have=None):
    """Rebuild a checked copy. After any refusal the caller must discard the file."""
    _shape(database, generation)
    base, entries, p = database["base"], database["deltas"], database["page_size"]
    if have is not None and (type(have) is not int or not 0 <= have <= len(entries)):
        raise DeltaRefused("start_mismatch")
    try:
        handle = open(path, "x+b" if have is None else "r+b")
    except FileExistsError:
        raise DeltaRefused("output_exists") from None
    with handle:
        previous = base["file_bytes"]//p if not have else entries[have-1]["file_pages"]
        if have is None:
            digest = hashlib.sha256()
            for ref in base["parts"]:
                data = _fetch(ref, fetch)
                handle.write(data)
                digest.update(data)
            if handle.tell() != base["file_bytes"] or digest.hexdigest() != base["file_sha256"]:
                raise DeltaRefused("base_digest_mismatch")
        elif os.fstat(handle.fileno()).st_size != previous*p:
            raise DeltaRefused("start_mismatch")
        for entry in entries[have or 0:]:
            last, next_new, target = -1, previous, entry["file_pages"]
            for ref in entry["objects"]:
                for index, page in read_object(_fetch(ref, fetch), p, ref["records"]):
                    if index <= last:
                        raise DeltaRefused("delta_order", index=index)
                    if index >= target:
                        raise DeltaRefused("delta_index", index=index)
                    if index >= previous:
                        if index != next_new:
                            raise DeltaRefused("delta_gap", index=next_new)
                        next_new += 1
                    handle.seek(index*p)
                    handle.write(page)
                    last = index
            if target > previous and next_new != target:
                raise DeltaRefused("delta_gap", index=next_new)
            handle.truncate(target*p)
            previous = target
        handle.flush()
        try:
            result = hash_pages(path, expected_page_size=p)
        except DeltaRefused:
            raise DeltaRefused("rebuild_digest_mismatch", generation=generation) from None
        if any(getattr(result, key) != database[key] for key in ("file_bytes", "file_sha256", "page_list_sha256")):
            raise DeltaRefused("rebuild_digest_mismatch", generation=generation)
    return result


def base_reason(head, *, page_size, file_bytes, list_matches, forget, object_missing,
                delta_bytes, size_factor=SIZE_FACTOR, max_generations=MAX_CHAIN_GENERATIONS):
    if head is None:
        return "first"
    _fields(head, page_size=int, deltas=list)
    stored = 0
    for entry in head["deltas"]:
        _fields(entry, objects=list)
        for ref in entry["objects"]:
            _fields(ref, bytes=int)
            stored += ref["bytes"]
    for holds, reason in ((head["page_size"] != page_size, "page_size"), (not list_matches, "page_list"),
                          (forget, "forget"), (object_missing, "missing_object"),
                          (stored+delta_bytes >= size_factor*file_bytes, "size"),
                          (len(head["deltas"])+2 > max_generations, "length")):
        if holds:
            return reason
    return None


def store_signature(home):
    signature = []
    for name in ("lcm.db", "lcm.db-wal"):
        try:
            meta = os.stat(Path(home) / name)
            signature.append([meta.st_size, meta.st_mtime_ns])
        except FileNotFoundError:
            signature.append(None)
    return signature


def _sequence(value):
    return tuple(_sequence(v) for v in value) if isinstance(value, (list, tuple)) else value


def skip_blocker(*, manual, signature, recorded_signature, decision_pending, last_pair,
                 pending_pair, retention_done_generation, head_generation, last_publish_s,
                 now_s, ceiling_s=SKIP_CEILING_S):
    for holds, reason in (
        (manual, "manual"),
        (recorded_signature is None or not signature or signature[0] is None
         or _sequence(signature) != _sequence(recorded_signature), "changed"),
        (decision_pending, "decision_pending"),
        (last_pair is None or pending_pair is None or _sequence(last_pair) != _sequence(pending_pair), "attempt_pending"),
        (retention_done_generation is None or head_generation is None
         or retention_done_generation != head_generation, "retention_pending"),
        (last_publish_s is None or now_s < last_publish_s or now_s-last_publish_s >= ceiling_s, "ceiling"),
    ):
        if holds:
            return reason
    return None
