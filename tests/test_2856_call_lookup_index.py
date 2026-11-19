"""#2856: the tool-call payload lookup reads its candidate files only, and still equals the sorted scan.

The oracle below is the lookup as it stood on 0d01c2acd (main), scan included; every case compares
``find_externalized_tool_result_content_for_call`` with it on synthetic folders. The index keys each
file by the call id stored in it, so names and contents may disagree freely here.
"""

import errno
import hashlib
import json
import os
import pathlib
import random
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pytest

from hermes_lcm import externalize
from hermes_lcm.config import LCMConfig
from hermes_lcm.externalize import (
    _marker_file_not_newer_than_payload,
    _persisted_output_marker_entries,
    _persisted_output_marker_matches_preview_digest,
    _tool_call_stub,
    find_externalized_tool_result_content_for_call,
    get_large_output_storage_dir,
)


def oracle_lookup(
    *,
    tool_call_id,
    session_id="",
    expected_chars=None,
    persisted_output_source_path=None,
    persisted_output_preview_sha256=None,
    require_persisted_output_file_not_newer=False,
    allow_redacted_preview_match=True,
    config,
    hermes_home="",
):
    """The pre-#2856 lookup, verbatim apart from its name."""
    if not tool_call_id:
        return None
    if (expected_chars is not None or persisted_output_source_path) and not persisted_output_preview_sha256:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None
    for path in sorted(storage_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("kind", "tool_result") != "tool_result":
            continue
        if (payload.get("tool_call_id") or "") != tool_call_id:
            continue
        if session_id and (payload.get("session_id") or "") != session_id:
            continue
        payload_role = payload.get("role") or ""
        if payload_role and payload_role != "tool":
            continue
        content = payload.get("content")
        if not isinstance(content, str):
            continue
        if (
            expected_chars is not None
            or persisted_output_source_path
            or persisted_output_preview_sha256
            or require_persisted_output_file_not_newer
        ):
            marker_matches = False
            for marker in _persisted_output_marker_entries(payload, include_legacy_preview_prefix=True):
                if expected_chars is not None and marker.get("expected_chars") != expected_chars:
                    continue
                if persisted_output_source_path and marker.get("source_path") != persisted_output_source_path:
                    continue
                if (
                    persisted_output_preview_sha256
                    and not _persisted_output_marker_matches_preview_digest(
                        marker,
                        persisted_output_preview_sha256,
                        config=config,
                        allow_redacted_preview_match=allow_redacted_preview_match,
                    )
                ):
                    continue
                if require_persisted_output_file_not_newer and not _marker_file_not_newer_than_payload(marker, payload):
                    continue
                marker_matches = True
                break
            if not marker_matches:
                continue
        return content
    return None


def _outcome(fn, **kwargs):
    try:
        return ("ok", fn(**kwargs))
    except Exception as exc:  # the oracle raises on some undecodable candidates; both sides must agree
        return ("raise", type(exc).__name__)


IDS = [
    "call_1",
    "call_2",
    "c1",  # the issue-2703 shape: a file named for call_spill holds c1
    "call_spill",
    "toolu_01ABCdef",
    "call_abc_def",
    "a/b:c",  # same name stub as the next id
    "a-b-c",
    "x" * 48 + "1",  # same 48-character name stub as the next id
    "x" * 48 + "2",
    "call_0123456789ab_1f",
    "ingest_payload_field",
    "historical-backfill",
    "tool-result",
    "café-1",
]
NO_ID = [_MISSING := object(), None, "", 5]  # payloads with no usable id
SESSIONS = ["s1", "s2", ""]
RESIDUAL_NAMES = [
    "000.json",
    "payload.json",
    "zzz.json",
    "20261118_120000_bad.json",
    "20261118_120000_a:b_0123456789ab_1.json",
    "20261118_120000_" + "y" * 60 + "_0123456789ab_1.json",
    "20261118_120000__0123456789ab_1.json",
]


class Folder:
    def __init__(self, root: Path, rng: random.Random):
        self.rng = rng
        self.home = root / "home"
        self.home.mkdir()
        self.config = LCMConfig(database_path=str(root / "lcm.db"))
        self.dir = get_large_output_storage_dir(self.config, hermes_home=str(self.home), create=True)
        self.live = root / "live"
        self.live.mkdir()
        self.parked = root / "parked"
        self.markers = []  # (source_path, expected_chars, preview_sha256)
        self.serial = 0

    def _name(self, stub):
        self.serial += 1
        stamp = f"2026111{self.rng.randint(0, 9)}_{self.rng.randint(0, 235959):06d}"
        digest = hashlib.sha256(str(self.serial).encode()).hexdigest()[:12]
        return f"{stamp}_{stub}_{digest}_{self.rng.randint(1, 1 << 64):x}.json"

    def _marker_fields(self, payload):
        rng = self.rng
        if rng.random() < 0.4 or not self.markers:
            source = self.live / f"out-{self.serial}-{rng.random()}.txt"
            source.write_text("x" * rng.randint(1, 50))
            stat = source.stat()
            marker = (str(source), rng.choice([10, 20, 30]), hashlib.sha256(str(self.serial).encode()).hexdigest())
            self.markers.append(marker)
            fresh = {"file_size": stat.st_size, "file_mtime_ns": stat.st_mtime_ns, "file_ctime_ns": stat.st_ctime_ns}
        else:
            marker = rng.choice(self.markers)
            fresh = {}
        source, chars, preview = marker
        if rng.random() < 0.5:
            payload.update(
                persisted_output_source_path=source,
                persisted_output_expected_chars=chars,
                persisted_output_preview_sha256=preview,
                **{f"persisted_output_{k}": v for k, v in fresh.items() if rng.random() < 0.8},
            )
        else:
            entry = {"source_path": source, "expected_chars": chars, "preview_sha256": preview, **fresh}
            if rng.random() < 0.3:
                entry["redacted_preview_sha256"] = hashlib.sha256(b"r" + preview.encode()).hexdigest()
            payload["persisted_output_markers"] = [entry]

    def payload(self, tool_call_id):
        rng = self.rng
        self.serial += 1
        payload = {
            "kind": rng.choice(["tool_result"] * 6 + ["ingest_payload", "raw_payload", None]),
            "tool_call_id": tool_call_id if rng.random() < 0.9 else rng.choice(NO_ID),
            "role": rng.choice(["tool"] * 4 + ["", "assistant"]),
            "session_id": rng.choice(SESSIONS),
            "content": rng.choice([f"content-{self.serial}-{rng.random()}"] * 5 + [None, 7]),
            "created_at": time.time(),
        }
        for key in ("kind", "tool_call_id"):
            if payload[key] is None and rng.random() < 0.5 or payload[key] is _MISSING:
                del payload[key]
        if rng.random() < 0.6:
            self._marker_fields(payload)
        return payload

    def random_name(self, call_id):
        rng = self.rng
        shape = rng.random()
        if shape < 0.45:
            return self._name(_tool_call_stub(call_id))  # name and content agree
        if shape < 0.6:
            return self._name(_tool_call_stub(rng.choice(IDS)))  # name for one id, content for another
        if shape < 0.7:
            return self._name(f"ingest_payload_{rng.choice(['field', 'messages.0.content'])}")
        if shape < 0.75:
            return self._name("historical-backfill")
        name = rng.choice(RESIDUAL_NAMES)
        return name if not (self.dir / name).exists() else f"{rng.randint(0, 99)}{name}"

    def add(self):
        rng = self.rng
        call_id = rng.choice(IDS)
        shape = rng.random()
        if shape < 0.04:
            path = self.dir / self.random_name(call_id)
            if not path.exists():
                path.mkdir()
        elif shape < 0.08:
            (self.dir / f"{self._name(_tool_call_stub(call_id))}.{rng.randint(0, 99)}.tmp").write_text(
                json.dumps(self.payload(call_id))
            )
        elif shape < 0.13:
            (self.dir / self.random_name(call_id)).write_text(rng.choice(["{not json", "", "{\"kind\": 1"]))
        else:
            path = self.dir / self.random_name(call_id)
            if not path.is_dir():
                path.write_text(json.dumps(self.payload(call_id)))

    def files(self):
        return sorted(p for p in self.dir.iterdir() if p.is_file() and p.name.endswith(".json"))

    def _write_in_place(self, path, payload):
        with open(path, "r+", encoding="utf-8") as handle:  # same inode; the folder's times do not move
            handle.seek(0)
            handle.write(json.dumps(payload))
            handle.truncate()

    def rewrite_in_place(self, to_id=None, from_id=None):
        """Rewrite one file's content in place, optionally moving its stored id to or from an id."""
        files = self.files()
        if from_id is not None:
            files = [p for p in files if _stored_id(p) == from_id] or files
        if not files:
            return
        path = self.rng.choice(files)
        target = to_id if to_id is not None else self.rng.choice(IDS)
        payload = self.payload(target)
        if to_id is not None:
            payload.update(kind="tool_result", tool_call_id=to_id)
        self._write_in_place(path, payload)

    def replace_same_name(self):
        files = self.files()
        if not files:
            return
        path = self.rng.choice(files)
        staged = path.with_name(f"{path.name}.{self.rng.randint(0, 999)}.tmp")
        staged.write_text(json.dumps(self.payload(self.rng.choice(IDS))))
        os.replace(staged, path)  # a new inode under the same name

    def add_raising(self):
        """Content the old scan raises on, under any name; it stays for one or more batches."""
        path = self.dir / self.random_name(self.rng.choice(IDS))
        if path.exists():
            return None
        if self.rng.random() < 0.5:
            path.write_bytes(b"\xff\xfe\x00bad")
        else:
            path.write_text(self.rng.choice(["[1, 2]", "7", "\"text\""]))
        return path

    def remove(self):
        files = self.files()
        if files:
            self.rng.choice(files).unlink()

    def age(self):
        past = time.time_ns() - self.rng.randint(5, 10_000) * 1_000_000_000
        os.utime(self.dir, ns=(past, past))

    def park(self):
        self.dir.rename(self.parked)

    def unpark(self):
        self.parked.rename(self.dir)

    def lookup_kwargs(self):
        rng = self.rng
        kwargs = {
            "tool_call_id": rng.choice(IDS + ["call_missing", ""]),
            "session_id": rng.choice(SESSIONS),
            "config": self.config,
            "hermes_home": str(self.home),
        }
        if self.markers and rng.random() < 0.6:
            source, chars, preview = rng.choice(self.markers)
            if rng.random() < 0.8:
                kwargs["expected_chars"] = rng.choice([chars, chars, 99])
            if rng.random() < 0.8:
                kwargs["persisted_output_source_path"] = rng.choice([source, source, "/nowhere"])
            if rng.random() < 0.9:
                kwargs["persisted_output_preview_sha256"] = rng.choice(
                    [preview, preview, hashlib.sha256(b"r" + preview.encode()).hexdigest(), "0" * 64]
                )
        if rng.random() < 0.3:
            kwargs["require_persisted_output_file_not_newer"] = True
        if rng.random() < 0.3:
            kwargs["allow_redacted_preview_match"] = False
        return kwargs


def _stored_id(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload.get("tool_call_id") if isinstance(payload, dict) else None


_OPENED: list | None = None  # payload files read, in order, while a check records them


def _hook_payload_reads(monkeypatch, record):
    """Call ``record(path)`` once per file read, whether through Path.read_text (the old scan) or Path.open
    (the index, which needs the descriptor); read_text's own inner open is not counted twice."""
    real_read, real_open = pathlib.Path.read_text, pathlib.Path.open
    inside = threading.local()

    def read_text(self, *a, **k):
        record(self)
        inside.depth = getattr(inside, "depth", 0) + 1
        try:
            return real_read(self, *a, **k)
        finally:
            inside.depth -= 1

    def open_(self, mode="r", *a, **k):
        if "r" in mode and not getattr(inside, "depth", 0):
            record(self)
        return real_open(self, mode, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    monkeypatch.setattr(pathlib.Path, "open", open_)


def _record_reads(monkeypatch):
    def record(path):
        if _OPENED is not None and path.suffix == ".json":
            _OPENED.append(path.name)

    _hook_payload_reads(monkeypatch, record)


def _is_subsequence(part, whole):
    rest = iter(whole)
    return all(item in rest for item in part)


def _check(folder, kwargs, tally):
    """Same outcome as the old scan, and the files opened are a subsequence of the old scan's."""
    global _OPENED
    _OPENED = []
    expected = _outcome(oracle_lookup, **kwargs)
    old_opened, _OPENED = _OPENED, []
    actual = _outcome(find_externalized_tool_result_content_for_call, **kwargs)
    new_opened, _OPENED = _OPENED, None
    assert actual == expected, kwargs
    assert _is_subsequence(new_opened, old_opened), (new_opened, old_opened)
    tally[expected[0] if expected[0] == "raise" or expected[1] is None else "found"] += 1
    tally["opened_old"] = tally.get("opened_old", 0) + len(old_opened)
    tally["opened_new"] = tally.get("opened_new", 0) + len(new_opened)


@pytest.mark.parametrize("seed", range(16))
def test_index_lookup_equals_sorted_scan(tmp_path, monkeypatch, seed):
    rng = random.Random(2856_200 + seed)
    if seed % 4:
        # A short racy window so quiet steps reach trusted entries; still above the mtime tick of
        # APFS (ns) and of a Linux coarse clock (ms). Seeds 0, 4, 8, 12 keep the product's window.
        monkeypatch.setattr(externalize, "_CALL_INDEX_RACY_NS", 30_000_000, raising=False)  # absent on the base
    _record_reads(monkeypatch)
    folder = Folder(tmp_path, rng)
    tally = {"found": 0, "ok": 0, "raise": 0}
    steps = dict.fromkeys(["add", "remove", "rewrite", "to_id", "from_id", "replace", "raising", "quiet",
                           "age", "missing"], 0)
    for _ in range(rng.randint(0, 3)):
        _check(folder, folder.lookup_kwargs(), tally)  # an empty folder
    for _ in range(rng.randint(20, 60)):
        folder.add()
    parked = False
    raising = []
    for _ in range(170):
        step = rng.random()
        kwargs = folder.lookup_kwargs()
        if parked:
            if step < 0.3:
                folder.unpark()
                parked = False
        elif step < 0.18:
            steps["add"] += 1
            for _ in range(rng.randint(1, 4)):  # back to back, no sleep
                folder.add()
        elif step < 0.24:
            steps["remove"] += 1
            folder.remove()
        elif step < 0.32:
            steps["rewrite"] += 1
            folder.rewrite_in_place()
        elif step < 0.42:
            steps["to_id"] += 1
            folder.rewrite_in_place(to_id=kwargs["tool_call_id"] or "call_1")
        elif step < 0.50:
            steps["from_id"] += 1
            folder.rewrite_in_place(from_id=kwargs["tool_call_id"] or "call_1")
        elif step < 0.56:
            steps["replace"] += 1
            folder.replace_same_name()
        elif step < 0.61:
            steps["raising"] += 1
            path = folder.add_raising()
            if path is not None:
                raising.append(path)
        elif step < 0.71:
            steps["quiet"] += 1
            time.sleep(0.035)  # entries become trusted (under the short window)
        elif step < 0.76:
            steps["age"] += 1
            folder.age()
        elif step < 0.78:
            steps["missing"] += 1
            folder.park()
            parked = True
        _check(folder, kwargs, tally)
        for _ in range(rng.randint(0, 3)):
            _check(folder, folder.lookup_kwargs(), tally)
        if raising and not parked and rng.random() < 0.5:
            raising.pop(rng.randrange(len(raising))).unlink()
    if parked:
        folder.unpark()
    tally["none"] = tally.pop("ok")
    print(f"seed={seed} files={len(folder.files())} {tally} {steps}")
    assert tally["found"] > 0


def _frozen_stat(monkeypatch, target, frozen):
    """Report ``frozen`` for ``target``: a change inside one timestamp tick leaves the stat unchanged."""
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if not isinstance(path, int) and os.fspath(path) == os.fspath(target):
            return frozen
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(externalize.os, "stat", stat)


class _Recent:
    def __init__(self, base):
        self._base = base
        self.st_mtime_ns = self.st_ctime_ns = time.time_ns()

    def __getattr__(self, name):
        return getattr(self._base, name)


def test_racy_folder_mtime_still_sees_new_files(tmp_path, monkeypatch):
    """A file added inside the folder's timestamp tick leaves the folder stat unchanged; it is still found."""
    folder = Folder(tmp_path, random.Random(2856))
    _frozen_stat(monkeypatch, folder.dir, _Recent(os.stat(folder.dir)))
    kwargs = {"tool_call_id": "call_1", "config": folder.config, "hermes_home": str(folder.home)}
    assert find_externalized_tool_result_content_for_call(**kwargs) is None
    name = folder._name(_tool_call_stub("call_1"))
    (folder.dir / name).write_text(json.dumps({"kind": "tool_result", "tool_call_id": "call_1", "content": "new"}))
    assert find_externalized_tool_result_content_for_call(**kwargs) == "new"


def test_same_tick_rewrite_of_the_stored_id_is_seen(tmp_path, monkeypatch):
    """The smallest rewrite a stat cannot see: same size, same inode, same timestamp tick."""
    folder = Folder(tmp_path, random.Random(2857))
    path = folder.dir / folder._name(_tool_call_stub("call_1"))  # named for call_1, holds call_2
    path.write_text(json.dumps({"kind": "tool_result", "tool_call_id": "call_2", "content": "same-size"}))
    kwargs = {"tool_call_id": "call_1", "config": folder.config, "hermes_home": str(folder.home)}
    _frozen_stat(monkeypatch, path, _Recent(os.stat(path)))
    assert find_externalized_tool_result_content_for_call(**kwargs) is None
    folder._write_in_place(path, {"kind": "tool_result", "tool_call_id": "call_1", "content": "same-size"})
    assert find_externalized_tool_result_content_for_call(**kwargs) == "same-size"


def test_same_size_rewrite_with_mtime_restored_is_seen(tmp_path, monkeypatch):
    """A same-size in-place rewrite whose mtime is put back: only the change time moves, and it is seen."""
    monkeypatch.setattr(externalize, "_CALL_INDEX_RACY_NS", 30_000_000, raising=False)
    folder = Folder(tmp_path, random.Random(2859))
    path = folder.dir / folder._name(_tool_call_stub("call_1"))
    path.write_text(json.dumps({"kind": "tool_result", "tool_call_id": "call_2", "content": "same-size"}))
    kwargs = {"tool_call_id": "call_1", "config": folder.config, "hermes_home": str(folder.home)}
    time.sleep(0.05)
    assert find_externalized_tool_result_content_for_call(**kwargs) is None  # cached: holds call_2
    stamp = os.stat(path)
    folder._write_in_place(path, {"kind": "tool_result", "tool_call_id": "call_1", "content": "same-size"})
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (stamp.st_size, stamp.st_mtime_ns, stamp.st_ino)
    time.sleep(0.05)
    assert find_externalized_tool_result_content_for_call(**kwargs) == "same-size"


def test_issue_2703_shape_name_for_one_call_content_for_another(tmp_path):
    folder = Folder(tmp_path, random.Random(2858))
    name = "20251109_145500_call_spill_34e7b012bd2d_18765e66da0ba800.json"
    (folder.dir / name).write_text(json.dumps({"kind": "tool_result", "tool_call_id": "c1", "content": "body"}))
    for call_id in ("c1", "call_spill"):
        kwargs = {"tool_call_id": call_id, "config": folder.config, "hermes_home": str(folder.home)}
        assert find_externalized_tool_result_content_for_call(**kwargs) == oracle_lookup(**kwargs)
    assert find_externalized_tool_result_content_for_call(
        tool_call_id="c1", config=folder.config, hermes_home=str(folder.home)) == "body"


class _Counts:
    def __init__(self, folder_dir: Path, monkeypatch):
        self.target = os.fspath(folder_dir)
        self.listings = {"glob": 0, "iterdir": 0, "scandir": 0, "listdir": 0}
        self.reads = []
        o_glob, o_iterdir = pathlib.Path.glob, pathlib.Path.iterdir
        o_scandir, o_listdir = os.scandir, os.listdir
        counts = self

        def record(path):
            if os.fspath(path.parent) == counts.target:
                counts.reads.append(path.name)

        _hook_payload_reads(monkeypatch, record)

        def glob(self, *a, **k):
            if os.fspath(self) == counts.target:
                counts.listings["glob"] += 1
            return o_glob(self, *a, **k)

        def iterdir(self, *a, **k):
            if os.fspath(self) == counts.target:
                counts.listings["iterdir"] += 1
            return o_iterdir(self, *a, **k)

        def scandir(path=".", *a, **k):
            if not isinstance(path, int) and os.fspath(path) == counts.target:
                counts.listings["scandir"] += 1
            return o_scandir(path, *a, **k)

        def listdir(path=".", *a, **k):
            if not isinstance(path, int) and os.fspath(path) == counts.target:
                counts.listings["listdir"] += 1
            return o_listdir(path, *a, **k)

        monkeypatch.setattr(pathlib.Path, "glob", glob)
        monkeypatch.setattr(pathlib.Path, "iterdir", iterdir)
        monkeypatch.setattr(os, "scandir", scandir)
        monkeypatch.setattr(os, "listdir", listdir)

    def listed(self):
        return max(self.listings["glob"] + self.listings["iterdir"], self.listings["scandir"] + self.listings["listdir"])


def test_unchanged_files_are_parsed_once_and_lookups_read_only_candidates(tmp_path, monkeypatch):
    """N lookups on a quiet folder: one listing; a file is parsed the first time a lookup reaches it, and
    afterwards only files holding the looked-up id are read again, up to the old scan's stopping point.
    Files written within the racy window are read again whenever a lookup reaches them."""
    folder = Folder(tmp_path, random.Random(28560))
    by_id = {}

    def write(index, name_id, call_id):
        name = folder._name(_tool_call_stub(name_id))
        payload = {"kind": "tool_result", "tool_call_id": call_id, "role": "tool",
                   "session_id": f"s{index % 3}", "content": f"c{index}"}
        (folder.dir / name).write_text(json.dumps(payload))
        by_id.setdefault(call_id, []).append(name)

    for index in range(60):
        call_id = f"call_{index % 12}"
        write(index, call_id if index % 5 else "call_0", call_id)  # every fifth name is for another call
    for index in range(8):  # other kinds
        (folder.dir / folder._name("ingest_payload_field")).write_text(
            json.dumps({"kind": "ingest_payload", "tool_call_id": "call_1", "content": f"i{index}"}))
    (folder.dir / "payload.json").write_text(json.dumps({"kind": "tool_result", "tool_call_id": "call_3", "content": "r"}))
    by_id["call_3"].append("payload.json")
    time.sleep(getattr(externalize, "_CALL_INDEX_RACY_NS", 0) / 1e9 + 0.1)  # let every file and the folder settle

    looked_up = ["call_1", "call_3", "call_7", "call_missing", "call_99"]
    lookups = 25

    def kwargs_for(call_id):
        return {"tool_call_id": call_id, "session_id": "s1", "config": folder.config, "hermes_home": str(folder.home)}

    def run(expected):
        for index in range(lookups):
            call_id = looked_up[index % len(looked_up)]
            assert find_externalized_tool_result_content_for_call(**kwargs_for(call_id)) == expected[call_id]

    expected = {call_id: oracle_lookup(**kwargs_for(call_id)) for call_id in looked_up}
    assert expected["call_1"] is not None and expected["call_7"] is not None
    total = 69
    counts = _Counts(folder.dir, monkeypatch)
    run(expected)
    candidate_reads = sum(len(by_id.get(looked_up[i % len(looked_up)], ())) for i in range(lookups))
    print(f"quiet: lookups={lookups} files={total} listings={counts.listings} reads={len(counts.reads)}")
    assert counts.listed() <= 1
    assert len(counts.reads) <= total + candidate_reads

    for index in range(60, 65):  # just written: inside the racy window
        write(index, "call_8", "call_8")
    monkeypatch.undo()
    expected = {call_id: oracle_lookup(**kwargs_for(call_id)) for call_id in looked_up}
    counts = _Counts(folder.dir, monkeypatch)
    run(expected)
    print(f"just written: lookups={lookups} files={total + 5} listings={counts.listings} reads={len(counts.reads)}")
    assert counts.listed() <= lookups
    assert len(counts.reads) <= lookups * 5 + candidate_reads


def test_lookup_and_index_stay_inside_the_given_home(tmp_path, monkeypatch):
    """#2873 guard: given a home, the lookup and its index list, stat and read nothing outside it,
    even with HOME and HERMES_HOME pointing at a decoy folder that holds a matching payload."""
    decoy = tmp_path / "decoy-home"
    decoy_outputs = decoy / ".hermes" / "lcm-large-outputs"
    decoy_outputs.mkdir(parents=True)
    (decoy_outputs / "20261118_120000_call_1_0123456789ab_1.json").write_text(
        json.dumps({"kind": "tool_result", "tool_call_id": "call_1", "content": "decoy"}))
    monkeypatch.setenv("HOME", str(decoy))
    monkeypatch.setenv("HERMES_HOME", str(decoy / ".hermes"))
    given = tmp_path / "given"
    given.mkdir()
    folder = Folder(given, random.Random(2873))
    (folder.dir / folder._name(_tool_call_stub("call_1"))).write_text(
        json.dumps({"kind": "tool_result", "tool_call_id": "call_1", "content": "given"}))

    touched = []
    o_stat, o_scandir, o_listdir = os.stat, os.scandir, os.listdir
    o_glob, o_iterdir = pathlib.Path.glob, pathlib.Path.iterdir

    def record(path):
        if not isinstance(path, int):
            touched.append(os.path.realpath(os.fspath(path)))

    def stat(path, *a, **k):
        record(path)
        return o_stat(path, *a, **k)

    def scandir(path=".", *a, **k):
        record(path)
        return o_scandir(path, *a, **k)

    def listdir(path=".", *a, **k):
        record(path)
        return o_listdir(path, *a, **k)

    def glob(self, *a, **k):
        record(self)
        return o_glob(self, *a, **k)

    def iterdir(self, *a, **k):
        record(self)
        return o_iterdir(self, *a, **k)

    for owner, name, fn in ((os, "stat", stat), (os, "scandir", scandir), (os, "listdir", listdir),
                            (pathlib.Path, "glob", glob), (pathlib.Path, "iterdir", iterdir)):
        monkeypatch.setattr(owner, name, fn)
    _hook_payload_reads(monkeypatch, record)
    for _ in range(3):  # a cold and two warm lookups
        assert find_externalized_tool_result_content_for_call(
            tool_call_id="call_1", config=folder.config, hermes_home=str(folder.home)) == "given"
    monkeypatch.undo()
    home = os.path.realpath(folder.home)
    outside = [path for path in touched if path != home and not path.startswith(home + os.sep)]
    assert touched and not outside, outside
    index = getattr(externalize, "_call_index", {})
    assert not any(os.path.realpath(key).startswith(os.path.realpath(decoy)) for key in index)


# Round 4 (#2856 review): a listing error is never cached, an entry needs the descriptor actually read to
# match the path, the cap holds after every lookup, and deleted names leave the index on a relist.

def _fresh_index(monkeypatch, window_ns=30_000_000):
    monkeypatch.setattr(externalize, "_CALL_INDEX_RACY_NS", window_ns, raising=False)
    monkeypatch.setattr(externalize, "_call_index", OrderedDict(), raising=False)


def _settle(window_ns=30_000_000):
    time.sleep(window_ns / 1e9 + 0.03)


def _kw(folder, call_id):
    return {"tool_call_id": call_id, "config": folder.config, "hermes_home": str(folder.home)}


def _payload_bytes(call_id, content, size=None):
    data = json.dumps({"kind": "tool_result", "tool_call_id": call_id, "content": content})
    return data + " " * (size - len(data)) if size else data


def test_listing_error_is_never_cached(tmp_path, monkeypatch):
    """Python 3.14's Path.glob hides an OSError from scandir (EMFILE, say) and yields nothing; that empty
    listing must not be kept under the folder's unchanged signature."""
    _fresh_index(monkeypatch)
    folder = Folder(tmp_path, random.Random(2860))
    (folder.dir / folder._name("call_1")).write_text(_payload_bytes("call_1", "kept"))
    _settle()
    failing = [True]
    real_scandir, real_glob = os.scandir, pathlib.Path.glob

    def scandir(path=".", *a, **k):
        if failing[0] and not isinstance(path, int) and os.fspath(path) == os.fspath(folder.dir):
            raise OSError(errno.EMFILE, "Too many open files")
        return real_scandir(path, *a, **k)

    def glob(self, *a, **k):  # Python 3.14: the error is swallowed and the listing comes back empty
        try:
            return iter(list(real_glob(self, *a, **k)))
        except OSError:
            return iter([])

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(pathlib.Path, "glob", glob)
    assert find_externalized_tool_result_content_for_call(**_kw(folder, "call_1")) == oracle_lookup(
        **_kw(folder, "call_1"))
    failing[0] = False
    assert find_externalized_tool_result_content_for_call(**_kw(folder, "call_1")) == "kept"


def test_entry_needs_the_descriptor_read_to_match_the_path(tmp_path, monkeypatch):
    """Both path stats see file A while the open reads file B (a symlink swapped and restored): B's id must
    not be cached under A's signature."""
    _fresh_index(monkeypatch)
    folder = Folder(tmp_path, random.Random(2861))
    real = folder.dir / folder._name("call_1")
    real.write_text(_payload_bytes("call_1", "real"))
    imposter = tmp_path / "imposter.json"
    imposter.write_text(_payload_bytes("call_2", "imposter"))
    _settle()
    swapped = [True]
    real_open = pathlib.Path.open

    def open_(self, *a, **k):
        if swapped[0] and self == real:
            return real_open(imposter, *a, **k)
        return real_open(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", open_)
    find_externalized_tool_result_content_for_call(**_kw(folder, "call_1"))  # races: its answer is not compared
    swapped[0] = False
    entries = externalize._call_index[folder.dir][2]
    assert real.name not in entries
    assert find_externalized_tool_result_content_for_call(**_kw(folder, "call_1")) == oracle_lookup(
        **_kw(folder, "call_1")) == "real"


def _cache_items():
    return sum(len(f[1]) + len(f[2]) for f in externalize._call_index.values())


def test_cap_holds_after_every_lookup(tmp_path, monkeypatch):
    _fresh_index(monkeypatch)
    monkeypatch.setattr(externalize, "_CALL_INDEX_MAX_ITEMS", 30, raising=False)
    monkeypatch.setattr(externalize, "_CALL_INDEX_MAX_ENTRIES", 30, raising=False)  # round 3's name
    folders = []
    for index in range(2):
        root = tmp_path / f"home{index}"
        root.mkdir()
        folder = Folder(root, random.Random(2862 + index))
        for n in range(20):
            (folder.dir / folder._name(f"call_{n}")).write_text(_payload_bytes(f"call_{n}", f"c{n}"))
        folders.append(folder)
    _settle()
    for folder in folders * 2:
        for call_id in ("call_missing", "call_5"):
            assert find_externalized_tool_result_content_for_call(**_kw(folder, call_id)) == oracle_lookup(
                **_kw(folder, call_id))
            assert _cache_items() <= 30, _cache_items()


def test_deleted_name_leaves_the_index_on_relist(tmp_path, monkeypatch):
    _fresh_index(monkeypatch)
    folder = Folder(tmp_path, random.Random(2863))
    names = []
    for n in range(5):
        names.append(folder._name(f"call_{n}"))
        (folder.dir / names[-1]).write_text(_payload_bytes(f"call_{n}", f"c{n}"))
    _settle()
    assert find_externalized_tool_result_content_for_call(**_kw(folder, "call_missing")) is None
    assert names[2] in externalize._call_index[folder.dir][2]
    (folder.dir / names[2]).rename(tmp_path / "moved-out.json")
    _settle()
    assert find_externalized_tool_result_content_for_call(**_kw(folder, "call_missing")) is None
    assert names[2] not in externalize._call_index[folder.dir][2]


class _AfterRead:
    """A file object that runs ``after`` once its content has been read."""

    def __init__(self, handle, after):
        self._handle, self._after = handle, after

    def read(self, *a):
        try:
            return self._handle.read(*a)
        finally:
            self._after()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _audit_index():
    """Entries whose signature still matches the file must hold the call id the file holds now."""
    wrong = []
    for entries in (f[2] for f in externalize._call_index.values()):
        for name, (signature, key) in list(entries.items()):
            for folder_path in externalize._call_index:
                path = folder_path / name
                try:
                    if externalize._stat_signature(os.stat(path)) != signature:
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(payload, dict) and key is not externalize._EVERY_CALL and key != payload.get("tool_call_id"):
                    wrong.append((name, key, payload.get("tool_call_id")))
    return wrong


@pytest.mark.parametrize("scenario", ["aba_rename_swap", "hardlink_swap"])
@pytest.mark.parametrize("window_ns", [0, 30_000_000])
def test_file_swapped_during_the_read_is_not_cached(tmp_path, monkeypatch, scenario, window_ns):
    """From the round-3 reviewer's race probes: another file sits under the name only while it is read."""
    _fresh_index(monkeypatch, window_ns)
    folder = Folder(tmp_path, random.Random(2864))
    aside = tmp_path / "aside"
    aside.mkdir()
    (folder.dir / "m100.json").write_text(_payload_bytes("call_x", "x"))
    target, other = folder.dir / "m500.json", folder.dir / "m900.json"
    target.write_text(_payload_bytes("call_A", "AAAA", size=200))
    other.write_text(_payload_bytes("call_B", "BBBB", size=200))
    _settle(window_ns)
    fired = []
    real_open = pathlib.Path.open

    def open_(self, *a, **k):
        if self.name != "m500.json" or fired:
            return real_open(self, *a, **k)
        fired.append(1)
        os.rename(target, aside / "a")
        if scenario == "aba_rename_swap":
            os.rename(other, target)
        else:
            os.link(other, target)
        handle = real_open(self, *a, **k)

        def restore():
            os.rename(target, other if scenario == "aba_rename_swap" else aside / "link")
            os.rename(aside / "a", target)

        return _AfterRead(handle, restore)

    monkeypatch.setattr(pathlib.Path, "open", open_)
    _outcome(find_externalized_tool_result_content_for_call, **_kw(folder, "call_A"))  # races
    assert fired
    monkeypatch.setattr(pathlib.Path, "open", real_open)
    for _ in range(3):
        _settle(window_ns)
        for call_id in ("call_A", "call_B", "call_x", "nope"):
            expected = _outcome(oracle_lookup, **_kw(folder, call_id))
            assert _outcome(find_externalized_tool_result_content_for_call, **_kw(folder, call_id)) == expected
        assert not _audit_index()
