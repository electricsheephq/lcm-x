"""Read an LCM recorder process directory (#2243): ordered events, rebuilt views, the replayable prefix.

Imports nothing from the plugin, so the same file reads recordings made on any tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


class Ref:
    """A recorded reference: ``kind`` is engine, set, unsupported, uncapturable, absent or read_error."""
    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    def __eq__(self, other):
        return isinstance(other, Ref) and (self.kind, self.value) == (other.kind, other.value)

    def __repr__(self):
        return f"Ref({self.kind}={self.value!r})"

    __hash__ = None


class ViewValue:
    """A recorded list of message rows: ``ops`` as recorded, ``messages`` rebuilt in full."""
    __slots__ = ("ops", "messages")

    def __init__(self, ops, messages):
        self.ops, self.messages = ops, messages


class RecordingError(ValueError):
    pass


def _blob(root: Path, sha: str, verify: bool):
    try:
        data = (root / "blobs" / sha[:2] / sha).read_bytes()
    except OSError as exc:
        raise RecordingError("blob_missing") from exc
    if verify and hashlib.sha256(data).hexdigest() != sha:
        raise RecordingError("blob_mismatch")
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise RecordingError("blob_torn") from exc



class _Decoder:
    def __init__(self, root: Path, verify: bool):
        self.root, self.verify, self.rows, self.cache = root, verify, {}, {}

    def blob(self, ref):
        sha = ref["$b"]
        if sha not in self.cache:
            self.cache[sha] = self.value(_blob(self.root, sha, self.verify))
        return copy.deepcopy(self.cache[sha])

    def value(self, raw):
        if isinstance(raw, list):
            return [self.value(v) for v in raw]
        if not isinstance(raw, dict):
            return raw
        if len(raw) == 1:
            (key, inner), = raw.items()
            if key == "$t":
                return tuple(self.value(v) for v in inner)
            if key == "$d":
                return {self.value(k): self.value(v) for k, v in inner}
            if key == "$f":
                return float(inner)
            if key == "$b":
                return self.blob(raw)
            if key == "$v":
                return self.view(inner)
            if key == "$set_items":
                return set(self.value(inner))
            if key.startswith("$") and key not in ("$row",):
                return Ref(key[1:], self.value(inner))
        return {k: self.value(v) for k, v in raw.items()}

    def view(self, raw_ops):
        ops, messages = [], []
        for raw in raw_ops:
            op, token = raw[0], raw[1]
            if op in ("e", "n"):
                self.rows[token] = self.blob(raw[2])
                ops.append((op, token, None))
            elif op == "c":
                if raw[2] not in self.rows:
                    raise RecordingError(f"clone of unknown row token {raw[2]}")
                self.rows[token] = copy.deepcopy(self.rows[raw[2]])
                ops.append((op, token, raw[2]))
            else:
                if token not in self.rows:
                    raise RecordingError(f"unknown row token {token}")
                ops.append((op, token, None))
            messages.append(copy.deepcopy(self.rows[token]))
        return ViewValue(ops, messages)


def _seqs(event):
    return (event["seq"],) if "seq" in event else (event["seq_enter"], event["seq_exit"])


def load_events(proc: Path) -> list:
    """Every complete event line; a torn last line of a segment is not an event."""
    events = []
    for segment in sorted((proc / "events").glob("*.jsonl")):
        data = segment.read_bytes()
        for line in data.split(b"\n")[:-1] if not data.endswith(b"\n") else data.split(b"\n"):
            if line.strip():
                events.append(json.loads(line))
    return events


def read(proc, *, checkpoint=None, verify=True) -> dict:
    """The recording of one process directory, cut to its replayable prefix (or to a sealed checkpoint)."""
    proc = Path(proc)
    root = proc.parent.parent
    raw = load_events(proc)
    status = json.loads((proc / "STATUS.json").read_text()) if (proc / "STATUS.json").exists() else {}
    if status.get("replayable") is False:
        raise RecordingError("STATUS: recording is not replayable")
    seqs = {s for event in raw for s in _seqs(event)}
    end, cause = 1, "end"
    while end in seqs:
        end += 1
    if seqs and end <= max(seqs):
        # Events refused at a full queue leave a hole just before the gap marker.
        cause = "gap" if any(e.get("kind") == "gap" and e["seq"] >= end for e in raw) else "seq_hole"
    ordered = sorted(raw, key=lambda e: min(_seqs(e)))
    lost = 0
    for event in ordered:
        first = min(_seqs(event))
        found = None
        if event.get("kind") in ("gap", "cap"):
            found = event["kind"]
        elif event.get("lost_observation") or event.get("status") == "stack_mismatch":
            found = event.get("lost_observation") or "stack_mismatch"
        elif event.get("lost", 0) > lost:
            found = "lost"
        lost = max(lost, event.get("lost", 0))
        if found and first < end:
            end, cause = first, found
    sealed = None
    if checkpoint is not None:
        sealed = next((e for e in raw if e.get("kind") == "checkpoint_sealed" and e["hours"] == checkpoint), None)
        if sealed is None:
            raise RecordingError(f"no sealed checkpoint +{checkpoint}h")
        if sealed["through"] >= end:
            raise RecordingError(f"checkpoint crosses {cause} at seq {end}")
        if sealed["through"] + 1 < end:
            end, cause = sealed["through"] + 1, "checkpoint"
    prefix = [e for e in raw if max(_seqs(e)) < end]
    decoder = _Decoder(root, verify)
    slots = sorted([(e.get("seq", e.get("seq_enter")), 0, e) for e in prefix]
                   + [(e["seq_exit"], 1, e) for e in prefix if "seq_exit" in e], key=lambda s: (s[0], s[1]))
    for _seq, phase, event in slots:
        names = ("result", "exception") if phase else [k for k in event if k not in ("result", "exception")]
        try:
            for name in names:
                if name in event:
                    event[name] = decoder.value(event[name])
        except RecordingError as exc:
            end, cause = min(_seqs(event)), str(exc)
            if sealed is not None and sealed["through"] >= end:
                raise RecordingError(f"checkpoint crosses {cause} at seq {end}") from exc
            break
    prefix = [e for e in prefix if max(_seqs(e)) < end]
    by_seq = {e.get("seq_enter", e.get("seq")): e for e in prefix}
    events = sorted(prefix, key=lambda e: min(_seqs(e)))
    for event in events:
        event["children"] = []
    for event in events:
        parent = by_seq.get(event.get("parent_seq"))
        if parent is not None:
            parent["children"].append(event)
    marks = status.get("marks") or {}
    has_exit = any(e.get("kind") == "exit_begin" for e in raw)
    snapshot = proc / "snapshot" / "lcm.db"
    manifest = proc / "snapshot" / "large-outputs.manifest.json"
    return {
        "proc": proc, "snapshot": snapshot if snapshot.exists() else None,
        "payloads": proc / "snapshot" / "large-outputs",
        "payload_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        "events": events, "roots": [e for e in events if e.get("parent_seq") is None and "seq_enter" in e],
        "prefix_end": end - 1, "cause": cause, "status": status, "marks": marks,
        "end_state": "exit" if has_exit else "ended without exit", "sealed": sealed,
        "process_start": next((e for e in events if e.get("kind") == "process_start"), None),
    }


def overlaps(root) -> list:
    """Process directories on one db (same ``db8``) whose wall-time ranges, first to last event, overlap."""
    spans = []
    for proc in sorted(Path(root, "proc").glob("*")):
        times = [e.get("t") or (e.get("t_enter") or [None])[0] for e in load_events(proc)]
        times = [t for t in times if t is not None]
        if times:
            spans.append((proc.name.rsplit("-", 1)[-1], min(times), max(times), proc.name))
    flagged = []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]:
                flagged.append((a[3], b[3]))
    return flagged
