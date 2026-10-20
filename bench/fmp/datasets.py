"""Normalize public evaluation files without logging their contents."""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .common import sha


@dataclass
class Question:
    qid: str
    dataset: str
    qtype: str
    question: str
    gold: str
    question_date: str
    is_abstention: bool
    sessions: list
    store_id: str = ""
    session_map: dict = field(default_factory=dict)  # opaque id -> dataset id, scoring only


def opaque(sessions):
    # D1: dataset ids leak labels (LME evidence ids start "answer_"); readers see
    # only s0001.. in session-date order (ties keep file order).
    ordered = sorted(enumerate(sessions), key=lambda pair: (pair[1]["date"], pair[0]))
    renamed = [{**s, "session_id": f"s{n:04d}"} for n, (_, s) in enumerate(ordered, 1)]
    return renamed, {r["session_id"]: s["session_id"] for r, (_, s) in zip(renamed, ordered)}


def iso(value):
    if not value:
        return ""
    value = str(value).replace(" (", " ").replace(")", "")
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d %a %H:%M", "%Y/%m/%d %A %H:%M", "%I:%M %p on %d %B, %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc).isoformat()


def normalize_lme(raw, label, lme):
    q = lme.parse_question(raw)  # Production loader and abstention semantics.
    dates = raw.get("haystack_dates", [""] * len(q.haystack_sessions))
    if len(q.haystack_session_ids) != len(q.haystack_sessions) or len(dates) != len(q.haystack_sessions):
        raise ValueError("unaligned session metadata")
    sessions = [{"session_id": sid, "date": iso(date), "turns": [
        {"role": t["role"], "content": t["content"]} for t in turns
    ]} for sid, date, turns in zip(q.haystack_session_ids, dates, q.haystack_sessions)]
    sessions, mapping = opaque(sessions)
    return Question(q.question_id, label, q.question_type, q.question, str(raw["answer"]),
                    iso(raw.get("question_date")), q.is_abstention, sessions, q.question_id, mapping)


def normalize_locomo(item, adversarial=False):
    conv, cid = item["conversation"], str(item["sample_id"])
    sessions = []
    for key, turns in conv.items():
        if not key.startswith("session_") or not isinstance(turns, list):
            continue
        sessions.append({"session_id": key, "date": iso(conv.get(key + "_date_time")), "turns": [
            {"role": "user" if t["speaker"] == conv["speaker_a"] else "assistant",
             "speaker": t["speaker"], "content": t["text"] +
             ("\n[Image: " + t["blip_caption"] + "]" if t.get("blip_caption") else "")} for t in turns]})
    sessions, mapping = opaque(sessions)
    # D2: category 5 (adversarial) is excluded by default, as Mem0 and OmniMemEval do.
    return [Question(f"{cid}:{i}", "locomo", str(qa["category"]), qa["question"],
                     str(qa.get("answer", qa.get("adversarial_answer", ""))),
                     max(s["date"] for s in sessions), qa["category"] == 5, sessions, cid, mapping)
            for i, qa in enumerate(item["qa"]) if adversarial or qa["category"] != 5]


def load(path, label, lme, ids_file=None):
    # Use upstream's byte loader, then preserve fields that its retrieval-only
    # Question intentionally omits (gold and dates). Read once, hash same bytes.
    payload = Path(path).read_bytes()
    rows = json.loads(payload)
    if label.startswith("lme"):
        parsed = lme._questions_from_bytes(payload)
        if len(parsed) != len(rows):
            raise ValueError("loader row count mismatch")
        questions = [normalize_lme(row, label, lme) for row in rows]
        excluded = {}
    else:
        questions = [q for row in rows for q in normalize_locomo(row)]
        excluded = {"locomo_category_5_adversarial": sum(qa["category"] == 5 for row in rows for qa in row["qa"])}
    if len({q.qid for q in questions}) != len(questions):
        raise ValueError("duplicate question ids")
    ids_hash = None
    if ids_file:
        data = Path(ids_file).read_bytes()
        ids_hash = sha(data)
        ids = json.loads(data) if data.lstrip().startswith(b"[") else data.decode().splitlines()
        if len(set(ids)) != len(ids) or not set(ids) <= {q.qid for q in questions}:
            raise ValueError("invalid question id selection")
        selected = set(ids)
        questions = [q for q in questions if q.qid in selected]
    return questions, sha(payload), ids_hash, excluded
