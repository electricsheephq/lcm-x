"""Outcome-blind development panels; historical #1188 IDs, never confirmation."""
import json
from pathlib import Path

from . import datasets
from .common import digest_file, sha

SEED = 1188
PANELS = Path(__file__).with_name("panels")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def fingerprint(q):
    return sha(encoded([q.question, q.gold, q.question_date, q.qtype, q.is_abstention]))


def population(lme_path, locomo_path, lme, lme_ids):
    """Normalize only existing development IDs and the first two conversations."""
    raw = json.loads(Path(lme_path).read_bytes())
    selected = set(lme_ids)
    questions = [datasets.normalize_lme(r, "lme-s", lme) for r in raw if r["question_id"] in selected]
    if {q.qid for q in questions} != selected:
        raise ValueError("development population mismatch")
    for ci, item in enumerate(json.loads(Path(locomo_path).read_bytes())[:2], 1):
        for q in datasets.normalize_locomo(item, adversarial=True):
            qi = q.qid.rsplit(":", 1)[1]
            q.qid = f"locomo-{ci}-q{qi}"
            questions.append(q)
    return questions


def entry(q):
    abstain = q.is_abstention
    kind = ("abstention" if abstain else "multi-hop" if q.qtype in ("1", "multi-session") else
            "update-temporal" if q.qtype in ("2", "knowledge-update", "temporal-reasoning") else "single-hop")
    # Related LME variants sharing histories are one cluster, rather than pseudo-replicates.
    cluster = (q.qid.split("-q")[0] if q.dataset == "locomo" else
               "lme-" + sha(encoded(q.sessions))[:24])
    return dict(id=q.qid, dataset=q.dataset, cluster=cluster, stratum=kind, sha256=fingerprint(q))


def select(questions):
    entries = [entry(q) for q in questions]
    chosen = []
    for kind in ("multi-hop", "update-temporal", "single-hop", "abstention"):
        group = sorted((e for e in entries if e["stratum"] == kind),
                       key=lambda e: sha(f"{SEED}:{e['id']}".encode()))
        if len(group) < 25:
            raise ValueError("insufficient development stratum")
        chosen.extend(group[:25])
    guard = [e for e in entries if e["stratum"] == "abstention"]
    return sorted(chosen, key=lambda e: e["id"]), sorted(guard, key=lambda e: e["id"])


def freeze(lme_path, locomo_path, lme, split_path, destination=PANELS):
    split = json.loads(Path(split_path).read_bytes())
    if split["locomo_dev_conversations"] != [1, 2] or len(split["lme_dev"]) != 100:
        raise ValueError("unsupported development split")
    questions = population(lme_path, locomo_path, lme, split["lme_dev"])
    main, guard = select(questions)
    provenance = dict(version=1, seed=SEED, rule="25/stratum; ascending sha256(seed:id); all abstention guards",
                      source_split_sha256=digest_file(split_path), source_experiment="77ea17c8",
                      lme_dev_ids=split["lme_dev"],
                      dataset_sha256={"lme-s": digest_file(lme_path), "locomo": digest_file(locomo_path)},
                      population_sha256=sha(encoded(sorted([entry(q) for q in questions], key=lambda e: e["id"]))))
    destination.mkdir(parents=True, exist_ok=True)
    for name, entries in (("dev100.json", main), ("guard.json", guard)):
        # Freeze is deliberately create-only, including during regeneration.
        with (destination / name).open("x") as stream:
            stream.write("{\n" + ",\n".join("  " + json.dumps(k) + ": " + json.dumps(v, sort_keys=True)
                         for k, v in provenance.items()) + ',\n  "items": [\n')
            stream.write(",\n".join("    " + json.dumps(e, sort_keys=True) for e in entries))
            stream.write("\n  ]\n}\n")


def load(panel_path, guard_path, lme_path, locomo_path, lme):
    panel, guard = [json.loads(Path(p).read_bytes()) for p in (panel_path, guard_path)]
    if {k: v for k, v in panel.items() if k != "items"} != {k: v for k, v in guard.items() if k != "items"}:
        raise ValueError("panel provenance mismatch")
    if panel["dataset_sha256"] != {"lme-s": digest_file(lme_path), "locomo": digest_file(locomo_path)}:
        raise ValueError("panel dataset mismatch")
    questions = population(lme_path, locomo_path, lme, panel["lme_dev_ids"])
    expected, expected_guard = select(questions)
    if panel["items"] != expected or guard["items"] != expected_guard:
        raise ValueError("panel freeze mismatch")
    if panel["population_sha256"] != sha(encoded(sorted([entry(q) for q in questions], key=lambda e: e["id"]))):
        raise ValueError("population identity mismatch")
    ids = {e["id"] for e in expected + expected_guard}
    return [q for q in questions if q.qid in ids], panel, guard
