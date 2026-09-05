#!/usr/bin/env python3
"""Local replay of the prewarm dry run's privacy pass over the WHOLE prepared-m corpus (no provider calls).

The dry run stops at the first validator block; this walks every unit of every question in shard-manifest order,
applies the production transform (protect_embedding_text) and the outbound validator
(validate_embedding_privacy_dispatch) exactly as prewarm_embedding_cache does, and records — WITHOUT raw text —
every changed unit (digests + lengths) and every blocked unit (stage, pattern names from the error, redacted shape
stats of the raw text). Also runs the 500 question texts through the same pair (the harness protects queries at run time).
Usage: corpus_privacy_inventory.py <repo> <prepared-dir> <shards-manifest> <out.json> [expected-repo-sha]
The checkout identity of <repo> is recorded (`repo_head`, `repo_dirty_files`); with [expected-repo-sha] the run refuses a mismatch
(PR #416 review: the artifact must carry evidence of the commit whose transform + validator it replayed).
"""
import json
import re
import subprocess
import sys
import time
import hashlib
from pathlib import Path

repo, prepared_dir, shards, out_path = sys.argv[1:5]
expected_sha = sys.argv[5] if len(sys.argv) > 5 else None
repo_head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
repo_dirty_files = len(subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.splitlines())
if expected_sha and not repo_head.startswith(expected_sha):
    sys.exit(f"repo checkout {repo_head} does not match the expected registration sha {expected_sha}")
sys.path.insert(0, repo)
import benchmarking.longmemeval as lme  # noqa: E402
lme._ensure_hermes_lcm_package()
from hermes_lcm.ingest_protection import (  # noqa: E402
    EmbeddingPrivacyPolicyError, protect_embedding_text, validate_embedding_privacy_dispatch,
)

B64_RUN = re.compile(r"[A-Za-z0-9+/=_-]{40,}")
HEX_RUN = re.compile(r"\b[0-9a-fA-F]{32,}\b")
FULL_B64_LINE = re.compile(r"^[A-Za-z0-9+/=]{40,}$")

def shape(text: str) -> dict:
    lines = text.splitlines()
    tokens = text.split()
    return {
        "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text), "lines": len(lines), "tokens": len(tokens),
        "full_width_b64_lines": sum(1 for line_ in lines if FULL_B64_LINE.match(line_.strip())),
        "b64_runs_ge40": len(B64_RUN.findall(text)),
        "longest_b64_run": max((len(m) for m in B64_RUN.findall(text)), default=0),
        "hex_runs_ge32": len(HEX_RUN.findall(text)),
        "longest_token": max((len(t) for t in tokens), default=0),
        "has_begin_marker": "-----BEGIN" in text, "has_end_marker": "-----END" in text,
        "has_private_key_words": "PRIVATE KEY" in text.upper(),
        "mentions_password": bool(re.search(r"passw(or)?d|passphrase", text, re.I)),
        "mentions_key_words": bool(re.search(r"\b(api[_ -]?key|secret|token|ssh|rsa|pem|certificate)\b", text, re.I)),
        "escaped_newlines": text.count("\\n"),
    }

config, revision = lme._embedding_privacy_context("voyage", "voyage-context-3")
prepared = lme.load_prepared_dataset(Path(prepared_dir), dataset_label="m")
ids = lme.load_shard_question_ids(Path(shards))
t0 = time.time()
inv = {"repo_head": repo_head, "repo_dirty_files": repo_dirty_files, "revision": revision, "prepared_questions": len(prepared.questions), "selected": len(ids),
       "documents": 0, "unique_units": 0, "changed_units": 0, "blocked_units": 0,
       "queries": 0, "queries_changed": 0, "queries_blocked": 0,
       "per_question": {}, "blocked": [], "changed": [], "queries_blocked_detail": [], "queries_changed_detail": []}
seen = set()
def bump(qid, key):
    inv["per_question"].setdefault(qid, {"units": 0, "changed": 0, "blocked": 0})[key] += 1

for question in prepared.iter_question_ids(ids):
    qid = question.question_id
    # query text (protected at run time by the harness + the production lcm_recall arm)
    inv["queries"] += 1
    try:
        pq, _rev, qchanged = protect_embedding_text(question.question, config, expected_revision=revision)
        validate_embedding_privacy_dispatch([pq], config, expected_revision=revision)
        if qchanged:
            inv["queries_changed"] += 1
            inv["queries_changed_detail"].append({"question_id": qid, "raw_len": len(question.question), "protected_len": len(pq)})
    except EmbeddingPrivacyPolicyError as exc:
        inv["queries_blocked"] += 1
        inv["queries_blocked_detail"].append({"question_id": qid, "error": str(exc), "shape": shape(question.question)})
    for unit_index, text in enumerate(lme.iter_ingest_embedding_request_units(question)):
        inv["documents"] += 1
        bump(qid, "units")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
        try:
            protected, _rev, changed = protect_embedding_text(text, config, expected_revision=revision)
        except EmbeddingPrivacyPolicyError as exc:
            inv["blocked_units"] += 1
            bump(qid, "blocked")
            inv["blocked"].append({"question_id": qid, "unit_index": unit_index, "stage": "transform", "error": str(exc), "shape": shape(text)})
            continue
        if changed:
            inv["changed_units"] += 1
            bump(qid, "changed")
            inv["changed"].append({"question_id": qid, "unit_index": unit_index, "raw_sha256": digest,
                                   "protected_sha256": hashlib.sha256(protected.encode("utf-8")).hexdigest(),
                                   "raw_len": len(text), "protected_len": len(protected)})
        try:
            validate_embedding_privacy_dispatch([protected], config, expected_revision=revision)
        except EmbeddingPrivacyPolicyError as exc:
            inv["blocked_units"] += 1
            bump(qid, "blocked")
            inv["blocked"].append({"question_id": qid, "unit_index": unit_index, "stage": "validate", "error": str(exc),
                                   "changed_by_transform": bool(changed), "shape": shape(text),
                                   "protected_shape": shape(protected)})
        if inv["documents"] % 50000 == 0:
            print(f"progress documents={inv['documents']} changed={inv['changed_units']} blocked={inv['blocked_units']} t={time.time()-t0:.0f}s", file=sys.stderr, flush=True)
inv["unique_units"] = len(seen)
inv["elapsed_s"] = round(time.time() - t0, 1)
Path(out_path).write_text(json.dumps(inv, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: inv[k] for k in ("documents", "unique_units", "changed_units", "blocked_units", "queries", "queries_changed", "queries_blocked", "elapsed_s")}, sort_keys=True))
print("blocked units:", [(b["question_id"], b["unit_index"], b["stage"]) for b in inv["blocked"]])
