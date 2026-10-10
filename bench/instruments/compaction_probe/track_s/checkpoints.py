"""Manifest checkpoints and probe scheduling shared by the offline Track S writers."""
import json
from pathlib import Path


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []


def manifest(directory):
    return json.loads((Path(directory) / "material.manifest.json").read_text())


def due(directory):
    man = manifest(directory)
    out = {}
    for p in lines(Path(directory) / "lifecycle_probes.jsonl"):
        cp = next((c for c in man["checkpoints"] if c["tokens"] >= p["probe_token_position"]), None)
        if cp is None:
            raise ValueError(f"no checkpoint after lifecycle probe {p['id']}")
        out[p["id"]] = dict(p, checkpoint_id=cp["id"], checkpoint_row=cp["row_index"],
                             schedule="exact" if cp["tokens"] == p["probe_token_position"] else "first_after")
    return out


def select(directory, selector):
    man = manifest(directory)
    if not (Path(directory) / "lifecycle_probes.jsonl").exists():
        return [dict(id=int(x), row_index=int(x)) for x in selector.split(",")]
    cps = {c["id"]: c for c in man["checkpoints"]}
    cps[man["decision_checkpoint"]["id"]] = man["decision_checkpoint"]
    ids = ([p["checkpoint_id"] for p in due(directory).values()] + [man["decision_checkpoint"]["id"]]
           if selector == "lifecycle" else list(cps) if selector == "all" else selector.split(","))
    chosen = []
    for key in ids:
        key = man["decision_checkpoint"]["id"] if key in ("auto", "decision") else key
        key = f"S{man['seed']}-{key}" if key.startswith("CP") else key
        if key not in cps:
            raise ValueError(f"unknown checkpoint id {key}")
        if cps[key] not in chosen:
            chosen.append(cps[key])
    return sorted(chosen, key=lambda c: (c["row_index"], c["tokens"]))


def augment(directory, batches, checkpoint=None):
    """v3 untouched; v4 asks only admitted facts and lifecycle probes due at this id."""
    directory = Path(directory)
    if not (directory / "lifecycle_probes.jsonl").exists():
        return batches
    cp = checkpoint or manifest(directory)["decision_checkpoint"]
    facts = {f["id"]: f for f in json.loads((directory / "facts.json").read_text())}
    cont = json.loads((directory / "continuation.json").read_text())
    out = []
    for b in batches:
        ps = [p for p in b["probes"] if
              (p["id"] not in facts or facts[p["id"]]["row_index"] <= cp["row_index"]) and
              (p["kind"] != "continuation_field" or cont["row_index"] <= cp["row_index"])]
        if ps:
            out.append(dict(b, probes=ps))
    ps = [p for p in due(directory).values() if p["checkpoint_id"] == cp["id"] and p["row_index"] <= cp["row_index"]]
    if ps:
        out.append(dict(id=f"S{manifest(directory)['seed']}-BLIFE", text=batches[0]["text"], probes=ps))
    return out
