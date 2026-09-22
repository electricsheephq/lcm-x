#!/usr/bin/env python3
"""Attribute each BLOCKED unit (from corpus-privacy-inventory.json) to the residual sub-detector(s) that fire, on the raw
text and on the private-key-transformed text. Prints redacted per-line shape only (length, charset class, longest token).
Usage: blocked_units_attribution.py <repo> <prepared-dir> <inventory.json> <out.json>"""
import json
import re
import sys
import inspect
import hashlib
from pathlib import Path
repo, prepared_dir, inv_path, out_path = sys.argv[1:5]
sys.path.insert(0, repo)
import benchmarking.longmemeval as lme  # noqa: E402
lme._ensure_hermes_lcm_package()
from hermes_lcm import ingest_protection as ip  # noqa: E402
inv = json.load(open(inv_path))
src = inspect.getsource(ip._embedding_privacy_residual_patterns)
m = re.search(r"normalized\s*=\s*([A-Za-z_][A-Za-z0-9_.]*)\(text\)", src)
norm = getattr(ip, m.group(1)) if m else (lambda t: t)
subs = {
  "redact_private_keys_changes_text": lambda t: ip._embedding_privacy_redact_private_keys(t) != t,
  "has_orphaned_private_key_body": ip._has_orphaned_private_key_body,
  "pem_marker_with_inline_body": ip._pem_marker_with_inline_body,
  "has_orphan_full_width_base64_run": ip._has_orphan_full_width_base64_run,
  "pem_fragment_near_private_key_placeholder": ip._pem_fragment_near_private_key_placeholder,
  "catalog_private_key_regex": lambda t: ip._SENSITIVE_PATTERN_CATALOG["private_key"].search(t) is not None,
}
def charclass(line):
    s=line.strip()
    if not s:
        return "blank"
    if re.fullmatch(r"[A-Za-z0-9+/=]+", s):
        return f"base64-charset-only(len={len(s)})"
    if re.fullmatch(r"[0-9a-fA-F]+", s):
        return f"hex-only(len={len(s)})"
    if " " not in s:
        return f"single-token(len={len(s)},alnum_ratio={sum(c.isalnum() for c in s)/len(s):.2f})"
    return f"prose(len={len(s)},tokens={len(s.split())},longest={max(len(t) for t in s.split())})"
prepared = lme.load_prepared_dataset(Path(prepared_dir), dataset_label="m")
by_sha = {}
for b in inv["blocked"]:
    by_sha.setdefault(b["shape"]["raw_sha256"], []).append((b["question_id"], b["unit_index"]))
out = {}
for sha, slots in by_sha.items():
    qid, idx = slots[0]
    question = next(prepared.iter_question_ids([qid]))
    text = next(t for i, t in enumerate(lme.iter_ingest_embedding_request_units(question)) if i == idx)
    assert hashlib.sha256(text.encode()).hexdigest() == sha
    pk = ip._embedding_privacy_redact_private_keys(text)
    rec = {"occurrences": slots, "chars": len(text), "lines": text.count("\n")+1,
           "raw": {k: bool(f(text)) for k, f in subs.items()},
           "raw_normalized": {k: bool(f(norm(text))) for k, f in subs.items() if k != "redact_private_keys_changes_text"},
           "after_private_key_transform": {k: bool(f(pk)) for k, f in subs.items()},
           "placeholders_after_transform": len(ip._EMBEDDING_PRIVACY_PLACEHOLDER_RE.findall(pk)),
           "line_shapes": [charclass(line_) for line_ in text.splitlines()],
           "role_prefixes": sorted({line_.split(":")[0][:12] for line_ in text.splitlines() if re.match(r"^(user|assistant|system|[A-Z][a-z]+):", line_)})[:6]}
    out[sha[:16]] = rec
Path(out_path).write_text(json.dumps(out, indent=2))
for sha, rec in out.items():
    print("=== raw_sha", sha, "occurrences", len(rec["occurrences"]), "chars", rec["chars"], "lines", rec["lines"])
    print("  raw fires:", [k for k, v in rec["raw"].items() if v], "| normalized fires:", [k for k, v in rec["raw_normalized"].items() if v], "| after pk-transform fires:", [k for k, v in rec["after_private_key_transform"].items() if v], "placeholders_after_transform", rec["placeholders_after_transform"])
    print("  line shapes:", rec["line_shapes"])
