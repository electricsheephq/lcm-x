"""C1-only checkpoint answer keys and post-compaction #659 probes."""
from __future__ import annotations

import json
from pathlib import Path

from . import reader as R


def enabled(man):
    return man.get("material_version") in ("track-s-deep-v1", "track-s-deep-v2")


def facts_at(directory, checkpoint):
    path = Path(directory) / "deep_metadata.json"
    if not path.exists():
        return None
    values = json.loads(path.read_text())["checkpoint_answers"][str(checkpoint)]
    return list(values.values()) if isinstance(values, dict) else values


def event_probes(run, event, view):
    if not enabled(run.man) or not event.get("is_compaction"):
        return
    rd = R.SolLowReader(run.dir / "event-reader-scratch")
    specs = run.man["continuity_probes"]
    text = R.flatten(run.system, view)
    prompt = ("Reply only as a JSON object mapping each probe id to its answer string. "
              "Use only the conversation; do not invent missing values.\n" +
              "\n".join(f"{p['id']}: {p['text']}" for p in specs))
    answers, meta = R.answer(rd, run.system, view, prompt, count_tokens=run.m.tokens.count_tokens)
    checks = []
    for p in specs:
        answer = (answers or {}).get(p["id"])
        exact = answer == p["answer"] if isinstance(answer, str) and answer.strip() else None
        value = p.get("verbatim", p["answer"])
        present = value in text
        checks.append(dict(kind=p["kind"], passed=exact if present else False,
                           reader_pass=exact, present_verbatim=present))
    event.update(deep_probes=checks, deep_reader_readback=dict(rd.readback),
                 deep_reader_usage=meta.get("usage"))
    directory = run.dir / "deep-probes"
    directory.mkdir(exist_ok=True)
    (directory / f"event-{event['event']}.json").write_text(json.dumps(dict(
        event=event["event"], row_index=event["row_index"], checks=checks,
        answers=answers, readback=rd.readback, metadata=meta), indent=1))
