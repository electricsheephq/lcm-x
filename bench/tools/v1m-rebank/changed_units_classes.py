#!/usr/bin/env python3
"""For the units listed in a changed-manifest, replay the transform and report WHICH sensitive-pattern placeholders
were inserted (pattern classes), plus redacted shape stats. Never prints raw text.
Usage: changed_units_classes.py <repo> <prepared-dir> <changed-manifest.jsonl> <out.json>"""
import json
import re
import sys
import hashlib
import collections
from pathlib import Path
repo, prepared_dir, manifest, out_path = sys.argv[1:5]
sys.path.insert(0, repo)
import benchmarking.longmemeval as lme  # noqa: E402
lme._ensure_hermes_lcm_package()
from hermes_lcm import ingest_protection as ip  # noqa: E402
config, revision = lme._embedding_privacy_context("voyage", "voyage-context-3")
prepared = lme.load_prepared_dataset(Path(prepared_dir), dataset_label="m")
rows = [json.loads(line_) for line_ in open(manifest) if line_.strip()]
wanted = collections.defaultdict(set)
for r in rows:
    wanted[r["question_id"]].add(r["unit_index"])
ph_re = ip._EMBEDDING_PRIVACY_PLACEHOLDER_RE
out = {"revision": revision, "units": [], "pattern_totals": collections.Counter(), "unit_kind_totals": collections.Counter()}
for question in prepared.iter_question_ids(sorted(wanted)):
    for unit_index, text in enumerate(lme.iter_ingest_embedding_request_units(question)):
        if unit_index not in wanted[question.question_id]:
            continue
        protected, _rev, changed = ip.protect_embedding_text(text, config, expected_revision=revision)
        names = collections.Counter(m.group(1) if m.groups() else m.group(0) for m in ph_re.finditer(protected))
        for n, c in names.items():
            out["pattern_totals"][n] += c
        kind = "summary" if unit_index < 1000 else "chunk"  # heuristic label only; the ordinal space is the harness's unit walk
        out["unit_kind_totals"][kind] += 1
        out["units"].append({"question_id": question.question_id, "unit_index": unit_index, "changed": bool(changed),
            "raw_len": len(text), "protected_len": len(protected), "placeholders": dict(names),
            "raw_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "shape": {"lines": text.count("\n")+1, "longest_token": max((len(t) for t in text.split()), default=0),
                      "b64_runs_ge40": len(re.findall(r"[A-Za-z0-9+/=_-]{40,}", text)),
                      "has_begin": "-----BEGIN" in text, "mentions_password": bool(re.search(r"passw(or)?d|passphrase", text, re.I))}})
out["pattern_totals"] = dict(out["pattern_totals"])
out["unit_kind_totals"] = dict(out["unit_kind_totals"])
Path(out_path).write_text(json.dumps(out, indent=2, sort_keys=True))
print("units replayed:", len(out["units"]), "pattern totals:", out["pattern_totals"])
for u in out["units"][:61]:
    print(u["question_id"], u["unit_index"], "len", u["raw_len"], "->", u["protected_len"], u["placeholders"], "longest_tok", u["shape"]["longest_token"], "b64runs", u["shape"]["b64_runs_ge40"], "pw" if u["shape"]["mentions_password"] else "")
