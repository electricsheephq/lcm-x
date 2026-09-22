#!/usr/bin/env python3
"""Line-model record of the units the shipped posture refused (no provider calls, no raw text).

For every distinct refused unit in the inventory (question_id + unit_index), runs `_pem_line_model` at the pinned head and records,
per model segment: 1-based segment number, kind name, the 1-based physical line it starts on, segment width
(`content_end - redact_start`, the exact quantity `_has_orphan_full_width_base64_run` compares with 40), token count and longest
token of the segment, and whether the orphan-run backstop counts it (kind STRICT_B64 or PREFIXED_B64 and width >= 40, no placeholder).
Usage: refused_line_model.py <repo> <prepared-dir> <inventory.json> <out.json>
"""
import hashlib
import json
import sys
from pathlib import Path

repo, prepared_dir, inventory_path, out_path = sys.argv[1:5]
sys.path.insert(0, repo)
import benchmarking.longmemeval as lme  # noqa: E402

lme._ensure_hermes_lcm_package()
from hermes_lcm import ingest_protection as ip  # noqa: E402

KIND_NAME = {value: name[len("_PEM_LINE_KIND_"):] for name, value in vars(ip).items() if name.startswith("_PEM_LINE_KIND_")}
COUNTED_KINDS = {ip._PEM_LINE_KIND_STRICT_B64, ip._PEM_LINE_KIND_PREFIXED_B64}
PLACEHOLDERS = ("[LCM embedding privacy:", "[LCM sensitive redaction:")

inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
wanted = {}
for entry in inventory["blocked"]:
    wanted.setdefault(entry["question_id"], set()).add(entry["unit_index"])
prepared = lme.load_prepared_dataset(Path(prepared_dir), dataset_label="m")

texts = {}
for question in prepared.iter_question_ids(sorted(wanted)):
    for unit_index, text in enumerate(lme.iter_ingest_embedding_request_units(question)):
        if unit_index in wanted[question.question_id]:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            texts.setdefault(digest, {"text": text, "occurrences": []})["occurrences"].append(
                {"question_id": question.question_id, "unit_index": unit_index})

result = {}
for digest, item in texts.items():
    text = item["text"]
    segments = []
    run = 0
    for number, (kind, redact_start, content_end, line_start, _marker_end) in enumerate(ip._pem_line_model(text), start=1):
        segment = text[redact_start:content_end]
        tokens = segment.split()
        width = content_end - redact_start
        has_placeholder = any(p in segment for p in PLACEHOLDERS)
        counted = kind in COUNTED_KINDS and width >= 40 and not has_placeholder
        run = run + 1 if counted else 0
        segments.append({
            "segment": number,
            "kind": KIND_NAME.get(kind, str(kind)),
            "physical_line": text.count("\n", 0, line_start) + 1,
            "width": width,
            "prefix_chars": redact_start - line_start,
            "tokens": len(tokens),
            "longest_token": max((len(t) for t in tokens), default=0),
            "counted_by_orphan_run_backstop": counted,
            "run_after_this_segment": run,
        })
    result[digest[:16]] = {
        "raw_sha256": digest,
        "occurrences": item["occurrences"],
        "physical_lines": len(text.splitlines()),
        "escaped_newlines": text.count("\\n"),
        "model_segments": len(segments),
        "detector_fires": ip._has_orphan_full_width_base64_run(text),
        "counted_segments": [s for s in segments if s["counted_by_orphan_run_backstop"]],
        "segments": segments,
    }
Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
for key, rec in result.items():
    print(key, "occurrences", len(rec["occurrences"]), "physical_lines", rec["physical_lines"], "segments", rec["model_segments"],
          "fires", rec["detector_fires"], "counted", [(s["segment"], s["physical_line"], s["kind"], s["width"], s["tokens"], s["longest_token"]) for s in rec["counted_segments"]])
