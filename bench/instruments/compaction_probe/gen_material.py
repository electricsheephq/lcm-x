#!/usr/bin/env python3
"""Generate deterministic material, canaries, and probes for compaction runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


TOKENS_PER_CHAR = 1 / 3.6
EPOCH_RANGES = {"E0": (1, 5), "E1": (15, 19), "E2": (32, 35)}

# The values are deliberately ordinary words.  The generated value, rather
# than these individual words, is the canary and is never emitted by filler.
VALUE_WORDS = (
    "amber",
    "anchor",
    "apricot",
    "atlas",
    "beacon",
    "birch",
    "canyon",
    "cedar",
    "cinder",
    "citadel",
    "clover",
    "comet",
    "coral",
    "delta",
    "ember",
    "falcon",
    "fjord",
    "harbor",
    "hazel",
    "indigo",
    "keystone",
    "lagoon",
    "lattice",
    "linen",
    "maple",
    "meadow",
    "meridian",
    "meteor",
    "mosaic",
    "nectar",
    "opal",
    "orchard",
    "pebble",
    "pioneer",
    "quartz",
    "raven",
    "reed",
    "ripple",
    "saffron",
    "sierra",
    "spruce",
    "summit",
    "tundra",
    "velvet",
    "vertex",
    "violet",
    "willow",
    "zephyr",
    "acorn",
    "basil",
    "cobalt",
    "dahlia",
    "echo",
    "fable",
    "garden",
    "horizon",
    "island",
    "juniper",
    "kiln",
    "lemon",
    "marble",
    "novel",
    "olive",
    "plume",
    "quiver",
    "rover",
    "sable",
    "thistle",
    "umber",
    "verge",
    "wren",
    "yarrow",
)

NOUNS = (
    "artifact",
    "checkpoint",
    "workspace",
    "release",
    "snapshot",
    "bundle",
    "branch",
    "handoff",
    "manifest",
    "adapter",
    "session",
    "pipeline",
)

FILLER_WORDS = (
    "adapter",
    "assertion",
    "backfill",
    "boundary",
    "cache",
    "checkpoint",
    "codec",
    "commit",
    "contract",
    "cursor",
    "daemon",
    "delta",
    "fixture",
    "frontier",
    "graph",
    "handoff",
    "index",
    "journal",
    "loader",
    "manifest",
    "module",
    "nonce",
    "observability",
    "operator",
    "payload",
    "pipeline",
    "provenance",
    "replay",
    "resolver",
    "retry",
    "schema",
    "snapshot",
    "staging",
    "telemetry",
    "trace",
    "validation",
    "worker",
    "workspace",
)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_hex(seed: int) -> str:
    """Return the four-character seed prefix used by canary values."""

    # Keep the suffix stable for the ordinary non-negative CLI case while
    # still giving negative seeds a deterministic, non-punctuated suffix.
    return format(seed, "x")[:4] if seed >= 0 else format(seed & 0xFFFFFFFF, "x")[:4]


def _canary_probe(class_code: str, noun: str) -> str:
    if class_code == "C1":
        return f"What did we decide to name the {noun}?"
    if class_code == "C2":
        # The noun slot disambiguates the two same-class canaries per epoch —
        # identical probe text with different planted values made the second
        # of each pair structurally unscorable (caught pre-launch).
        return f"Where does the canonical {noun} config now live?"
    if class_code == "C3":
        return f"What must the {noun} limit stay exactly?"
    if class_code == "C4":
        return f"What was the build id printed by the last successful {noun} pipeline run?"
    if class_code == "C5":
        return f"What prefix should all {noun} names use going forward?"
    raise ValueError(f"unknown canary class: {class_code}")


def _canary_sentence(class_code: str, noun: str, value: str) -> str:
    if class_code == "C1":
        return f"After weighing both options we decided to name the {noun} `{value}`."
    if class_code == "C2":
        return f"The canonical {noun} config now lives at src/{value}/settings.toml — remember that path."
    if class_code == "C3":
        return f"Hard constraint from ops: the {noun} limit must stay exactly {value}."
    if class_code == "C4":
        return f"The build id printed by the last successful {noun} pipeline run was {value}."
    if class_code == "C5":
        return f"I prefer that all {noun} names use the `{value}` prefix going forward."
    raise ValueError(f"unknown canary class: {class_code}")


def _filler(turn: int, target_chars: int, rng: random.Random, values: list[str]) -> str:
    """Create varied coding-session prose near the requested character size."""

    vocabulary = list(FILLER_WORDS)
    offset = (turn * 7) % len(vocabulary)
    vocabulary = vocabulary[offset:] + vocabulary[:offset]
    lines: list[str] = []
    paragraph = 0
    while sum(len(line) for line in lines) < target_chars:
        nonce = f"T{turn:02d}-{paragraph:04d}-{rng.getrandbits(64):016x}"
        words = " ".join(rng.choice(vocabulary) for _ in range(11))
        mode = paragraph % 4
        if mode == 0:
            line = (
                f"[{nonce}] review note: the {words} path was checked against "
                "the previous boundary and leaves the fresh tail reachable.\n"
            )
        elif mode == 1:
            line = (
                f"{nonce} $ python -m probe --turn {turn} --cursor {paragraph}: "
                f"{words}; exit=0 duration_ms={rng.randrange(8, 900)}\n"
            )
        elif mode == 2:
            line = (
                f"diff --git a/src/{vocabulary[paragraph % len(vocabulary)]}.py "
                f"b/src/{vocabulary[(paragraph + 3) % len(vocabulary)]}.py\n"
                f"@@ {nonce} @@ {words}\n"
                "+ preserved chronology and explicit ownership metadata\n"
            )
        else:
            line = (
                f"code review {nonce}: I would keep the {words} decision local, "
                "record the reason, and rerun only the affected gate.\n"
            )
        lines.append(line)
        paragraph += 1
    filler = "".join(lines)
    if len(filler) > target_chars:
        filler = filler[:target_chars]
    # A value can only collide accidentally with filler.  Replace any such
    # collision before appending the authoritative canary sentence.
    for value in values:
        if value in filler:
            filler = filler.replace(value, value.replace("-", "_"))
    return filler


def _generate_legacy(seed: int, out_dir: Path, turns: int = 35, tokens_per_turn: int = 17000) -> dict[str, Any]:
    if turns <= 0:
        raise ValueError("--turns must be positive")
    if tokens_per_turn <= 0:
        raise ValueError("--tokens-per-turn must be positive")
    if turns < 35:
        raise ValueError("--turns must be at least 35 for the registered epochs")

    out_dir.mkdir(parents=True, exist_ok=True)
    word_rng = random.Random(seed ^ 0xC0A11A)
    filler_rng = random.Random(seed ^ 0xF111E)
    selected_words = list(VALUE_WORDS)
    word_rng.shuffle(selected_words)
    suffix = _seed_hex(seed)

    canaries: list[dict[str, Any]] = []
    word_cursor = 0
    for epoch, (first_turn, last_turn) in EPOCH_RANGES.items():
        width = last_turn - first_turn + 1
        for class_index in range(1, 6):
            class_code = f"C{class_index}"
            for copy_index in range(1, 3):
                canary_index = len(canaries)
                word1 = selected_words[word_cursor % len(selected_words)]
                word2 = selected_words[(word_cursor + 1) % len(selected_words)]
                word_cursor += 2
                value = f"{word1}-{word2}-{suffix}"
                noun = NOUNS[(canary_index + class_index + seed) % len(NOUNS)]
                canaries.append(
                    {
                        "id": f"{class_code}-{epoch}-{copy_index}",
                        "class": class_code,
                        "epoch": epoch,
                        "turn": first_turn + ((class_index * 2 + copy_index + seed) % width),
                        "value": value,
                        "probe": _canary_probe(class_code, noun),
                        "_noun": noun,
                    }
                )

    values = [row["value"] for row in canaries]
    by_turn: dict[int, list[dict[str, Any]]] = {}
    for row in canaries:
        by_turn.setdefault(row["turn"], []).append(row)

    target_chars = int(round(tokens_per_turn / TOKENS_PER_CHAR))
    turn_rows: list[dict[str, Any]] = []
    turn_estimates: list[dict[str, Any]] = []
    for turn in range(1, turns + 1):
        marker_rows = by_turn.get(turn, [])
        marker_chars = sum(
            len(_canary_sentence(row["class"], row["_noun"], row["value"])) + 1
            for row in marker_rows
        )
        text = _filler(turn, max(1, target_chars - marker_chars), filler_rng, values)
        if marker_rows:
            text = text.rstrip() + "\n"
        for row in marker_rows:
            row["char_offset"] = len(text)
            text += _canary_sentence(row["class"], row["_noun"], row["value"]) + "\n"
        turn_rows.append({"turn": turn, "text": text})
        turn_estimates.append(
            {
                "turn": turn,
                "characters": len(text),
                "estimated_tokens": len(text) * TOKENS_PER_CHAR,
            }
        )

    for row in canaries:
        row.pop("_noun", None)

    traps = [
        "What did we decide to name the glacier-index?",
        "Where does the canonical config for the shadow-market live?",
        "What must the phantom queue limit stay exactly?",
        "What was the build id of the failed midnight pipeline?",
        "What prefix should all nebula names use going forward?",
    ]
    probes = [
        {"id": row["id"], "kind": "canary", "text": row["probe"], "expect": "value"}
        for row in canaries
    ]
    probes.extend(
        {"id": f"TRAP-{index:02d}", "kind": "trap", "text": text, "expect": "ABSTAIN"}
        for index, text in enumerate(traps, 1)
    )
    random.Random(seed ^ 0x5EED).shuffle(probes)

    turns_path = out_dir / "turns.jsonl"
    with turns_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in turn_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    canaries_path = out_dir / "canaries.json"
    _json_write(canaries_path, canaries)
    probes_path = out_dir / "probes.jsonl"
    with probes_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in probes:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    manifest = {
        "seed": seed,
        "params": {"turns": turns, "tokens_per_turn": tokens_per_turn},
        "shas": {
            "turns.jsonl": _sha256(turns_path),
            "canaries.json": _sha256(canaries_path),
            "probes.jsonl": _sha256(probes_path),
        },
        "turn_estimates": turn_estimates,
        "estimated_total_tokens": sum(row["estimated_tokens"] for row in turn_estimates),
    }
    _json_write(out_dir / "material.manifest.json", manifest)
    return manifest


# Track S uses only the checkout's counter, with its documented offline fallback.
def token_counter():
    import importlib.util
    import sys
    import types
    root = Path(__file__).resolve().parents[3]
    package = "_compaction_probe_lcm"
    if package + ".tokens" not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root)]
        sys.modules[package] = module
        spec = importlib.util.spec_from_file_location(package + ".tokens", root / "tokens.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module._encoder_ready = True  # No tiktoken import, download, or cache write.
    return sys.modules[package + ".tokens"].count_tokens


CLASSES = ("name", "path", "limit", "build_id", "prefix", "decision", "superseded_value",
           "error_with_fix", "pending_task", "early_user_constraint", "tool_number", "file_change")
MATERIAL_VERSION = "track-s-v3"
BATCH_INSTRUCTION = "Reply only as a JSON object mapping each probe id to its answer string (use I don't know for ABSTAIN)."
SCENES = (
    "Inspection {n}: the fixture reader keeps the original row order. The dry run checked "
    "a missing cursor and returned an explicit warning. No source files were changed.\n",
    "Test log {n}: read fixture, compare message ownership, preserve call/result pairs. "
    "Expected three rows; observed three rows. Exit 0. The rollback check is still queued.\n",
    "Review {n}: the adapter passes the stored identifier through unchanged. A second read "
    "returns the same bytes. The read-only audit stops before publishing an artifact.\n",
    "Diff note {n}: src/reader.py adds a guard for an absent offset. The fixture contains "
    "an empty result and a resumed cursor. The change keeps both records reachable.\n",
)


def generate(seed: int, out_dir: Path, turns: int = 35, tokens_per_turn: int = 17000,
             placements: bool = False, classes12: bool = False, min_tokens: int = 244800,
             min_events: int = 2, smoke: bool = False, v4: bool = False) -> dict[str, Any]:
    if v4:
        return _generate_v4(seed, out_dir, turns, tokens_per_turn, min_tokens, min_events, smoke)
    if not placements and not classes12:
        return _generate_legacy(seed, out_dir, turns, tokens_per_turn)
    if not placements or not classes12:
        raise ValueError("Track S requires both --placements and --classes12")
    if turns < 10 or tokens_per_turn <= 0 or min_tokens <= 0 or min_events < 2:
        raise ValueError("require turns >= 10, positive token budgets, min-events >= 2")
    count = token_counter()
    rng = random.Random(seed)
    value_rng = random.Random(seed ^ 0x56414C554553)
    trap_rng = random.Random(seed ^ 0x5452415053)
    rows, facts, checkpoints = [], [], []
    serial = 0
    used_payloads = set()

    def value_token(bits=48, bounds=None):
        while True:
            token = (str(value_rng.randrange(*bounds)) if bounds else
                     f"{value_rng.getrandbits(bits):0{bits // 4}x}")
            if token not in used_payloads:
                used_payloads.add(token)
                return token

    def scene(chars):
        nonlocal serial
        pieces, size = [], 0
        while size < chars:
            serial += 1
            part = rng.choice(SCENES).format(n=f"{seed}:{serial:06d}")
            pieces.append(part)
            size += len(part)
        return "".join(pieces)[:chars]

    def add(turn, role, content, call=None):
        index = len(rows)
        rows.append(dict(turn=turn, role=role, content=content, tool_call_id=call,
                         ts=float(seed * 10000 + index), id=f"S{seed}-R{index:05d}"))
        return dict(row_index=index, row_id=rows[-1]["id"], row_role=role)

    def result(turn, content):
        call = f"S{seed}-CALL{len(rows):05d}"
        add(turn, "assistant", f"Read the fixture for {call}.", call)
        rows[-1]["tool_calls"] = [dict(id=call, type="function", function=dict(
            name="read_material", arguments=json.dumps({"path": f"fixtures/{call}.txt"})))]
        return add(turn, "tool", content, call)

    continuity = []
    for role, kind, value in (("system", "host_instruction", "Preserve chronology and abstain when a value is unknown."),
                              ("user", "current_request", "Audit the fixture replay and prepare a rollback check."),
                              ("user", "active_constraint", "Never modify the source fixtures during this audit.")):
        ident = f"S{seed}-CONT-{kind}"
        continuity.append(dict(id=ident, value=value, **add(1, role, f"[{ident}] {value}")))
    for c, cls in enumerate(CLASSES):
        for k in range(5):
            nonce = f"{rng.choice(VALUE_WORDS)}-{seed}-{c:02d}-{k}"
            # Payloads never encode the fixture identifier or a sibling's value.
            payload = value_token()
            values = (f"{payload}-workspace", f"src/{payload}/settings.toml",
                      f"{value_token(bounds=(1000000, 10000000))} MiB", payload,
                      f"{value_token(bits=64)}-",
                      f"snapshot-{payload} over live-{value_token()} because immutable input makes replay repeatable",
                      f"snapshot-{payload} because the mutable cache mixed cursor ownership",
                      f"E_{payload}: cursor missing; fixed by rebuilding the fixture index",
                      f"verify {payload} rollback before release", f"Never modify fixtures/{payload}/source.json",
                      f"rows_verified={value_token(bounds=(10000000, 100000000))}",
                      f"src/{payload}/loader.py: preserve tool_call_id in replay")
            fact = dict(id=f"S{seed}-F{c:02d}-{k}", fixture=nonce, **{"class": cls}, value=values[c],
                        stale=f"mutable-cache-{value_token()} because it avoids snapshot writes" if c == 6 else None,
                        placement=("head", "head", "middle", "tail", "tail")[k],
                        probe=f"What is the current {cls.replace('_', ' ')} for fixture {nonce}?", answer=values[c])
            if fact["stale"]:
                fact["stale_source"] = dict(id=fact["id"] + "-OLD", **add(
                    1, "user", f"[{fact['id']}-OLD; fixture {nonce}] Initial choice: {fact['stale']}."))
            fact["turn"] = 1 + k * 2
            fact["row_role"] = "tool" if k in (2, 3) or c == 10 else ("assistant" if k == 1 else "user")
            facts.append(fact)
    for turn in range(1, 11):
        for role in ("user", "assistant", "tool"):
            group = [f for f in facts if f["turn"] == turn and f["row_role"] == role]
            if role == "tool":
                for f in group:
                    line = f"[{f['id']}; fixture {f['fixture']}] Current {f['class']}: {f['value']}.\n"
                    body = scene(100004 if f["class"] == "file_change" and f["placement"] == "middle" else 6000)
                    offset = {"head": 0, "middle": 3000, "tail": len(body)}[f["placement"]]
                    text = body[:offset] + line + body[offset:]
                    f.update(result(turn, text), char_offset=offset + line.index(f["value"]))
                continue
            for f in group:
                line = f"[{f['id']}; fixture {f['fixture']}] Current {f['class']}: {f['value']}.\n"
                body = scene(4500)
                text = line + body if f["placement"] == "head" else body + line
                f.update(add(turn, role, text), char_offset=text.index(f["value"]))
            if not group:
                add(turn, role, scene(500))
    state = dict(id=f"S{seed}-STATE", next_action="run the read-only rollback check",
                 path=f"fixtures/seed-{seed}/rollback.json", status="pending", decision_id=f"S{seed}-F06-4")
    state.update(add(10, "assistant", f"[{state['id']}] Continuation: " + json.dumps(state, sort_keys=True)))
    counts = [count(r["content"]) for r in rows]
    presented = sum(counts)
    if presented >= min_tokens and not smoke:
        raise ValueError("all scored items must precede the slowest trigger")
    supersession = max(sum(counts[:f["row_index"] + 1]) for f in facts if f["stale"])
    # Conservative fresh-token spans; these are NOT observed runtime events.
    horizon = max(400000, turns * tokens_per_turn, supersession + min_events * min_tokens)
    target = ((horizon + 20000 + 19999) // 20000) * 20000
    total, tool_tokens, turn = presented, sum(n for n, r in zip(counts, rows) if r["role"] == "tool"), 10
    if smoke:
        source = result(10, f"[S{seed}-SMOKE-EVENT] Final wiring-only compaction checkpoint.\n" + scene(4000))
        suffix = dict(id=f"S{seed}-SMOKE-EVENT", **source, action="force_compaction_after_row",
                      timing_population="WIRING-ONLY", runtime_event_required=True)
    else:
        while total < target:
            turn += 1
            add(turn, "user", "Continue the read-only fixture audit; leave the rollback task pending.")
            total += count(rows[-1]["content"])
            checkpoint = ((total // 20000) + 1) * 20000
            budget = min(tokens_per_turn // 2 or 1, target - total, checkpoint - total)
            role = "tool" if tool_tokens < total / 2 else "assistant"
            text = scene(max(1, (budget - 1) * 4))
            if role == "tool":
                result(turn, text)
                total += count(rows[-2]["content"])
                tool_tokens += count(text)
            else:
                add(turn, role, text)
            total += count(text)
            if total >= checkpoint:
                checkpoints.append(dict(id=f"S{seed}-CP{checkpoint}", tokens=total, row_index=len(rows) - 1))
        suffix = None
    checkpoints, cumulative, boundary = [], 0, 20000
    for index, row in enumerate(rows):
        cumulative += count(row["content"])
        while cumulative >= boundary:
            checkpoints.append(dict(id=f"S{seed}-CP{boundary}", tokens=cumulative, row_index=index))
            boundary += 20000
    if smoke:
        decision = dict(row_index=len(rows) - 1, tail_tokens=0)
    else:
        running = 0
        for index, row in enumerate(rows):
            running += count(row["content"])
            if running >= min_tokens:
                first_trigger = dict(row_index=index, tokens=running)
                break
        decision = dict(next(cp for cp in checkpoints if cp["tokens"] >= first_trigger["tokens"] + 20000),
                        trigger=first_trigger)
    admissions = [dict(id=f["id"], value=f["value"], row_id=f["row_id"], row_index=f["row_index"],
                       role=f["row_role"], tool_call_id=rows[f["row_index"]]["tool_call_id"],
                       status="scheduled", runtime_row_id=None) for f in facts]
    targets = [dict(id=f["id"], kind="ancestry", row_id=f["row_id"], row_index=f["row_index"],
                    stale_source=f["stale_source"], required_events=min_events) for f in facts if f["stale"]]
    external = next(f for f in facts if f["class"] == "file_change" and f["placement"] == "middle")
    targets.append(dict(id=external["id"], kind="externalization", row_id=external["row_id"], row_index=external["row_index"]))
    traps = []
    for cls in trap_rng.sample(CLASSES, 5):
        c, k = CLASSES.index(cls), trap_rng.randrange(5, 10)
        name = f"{trap_rng.choice(VALUE_WORDS)}-{seed}-{c:02d}-{k}"
        traps.append(dict(id=f"S{seed}-F{c:02d}-{k}",
                          probe=f"What is the current {cls.replace('_', ' ')} for fixture {name}?", answer="ABSTAIN"))
    probes = [dict(id=f["id"], kind="canary", text=f["probe"], expect="value") for f in facts]
    probes += [dict(id=t["id"], kind="trap", text=t["probe"], expect="ABSTAIN") for t in traps]
    rng.shuffle(probes)
    batches = [dict(id=f"S{seed}-B{k // 10}", probes=probes[k:k + 10],
                    text=BATCH_INSTRUCTION)
               for k in range(0, len(probes), 10)]
    manifest = dict(seed=seed, material_version=MATERIAL_VERSION,
                    mode="smoke" if smoke else "decision", tokenizer="repo-count_tokens:offline-char-estimate",
                    params=dict(turns=turns, tokens_per_turn=tokens_per_turn, min_tokens=min_tokens, min_events=min_events),
                    checkpoints=checkpoints, continuity=continuity, receipt_targets=targets, smoke_suffix=suffix,
                    presented_tokens=presented, planned_trigger_spans=min_events,
                    decision_checkpoint=decision,
                    proof_boundary="Fresh-token planning only; actual events, ancestry, externalization and admission require runtime receipts.")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_names = []
    for name, payload in (("facts.json", facts), ("canaries.json", facts), ("traps.json", traps),
                          ("continuation.json", state), ("admission.manifest.json", admissions)):
        _json_write(out_dir / name, payload)
        generated_names.append(name)
    for name, payload in (("transcript.jsonl", rows), ("turns.jsonl", [dict(
            turn=r["turn"], text=r["content"], role=r["role"], id=r["id"],
            tool_call_id=r["tool_call_id"], tool_calls=r.get("tool_calls", [])) for r in rows]),
                          ("probes.jsonl", probes), ("probe_batches.jsonl", batches)):
        (out_dir / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in payload), encoding="utf-8")
        generated_names.append(name)
    manifest["shas"] = {name: _sha256(out_dir / name) for name in sorted(generated_names)}
    _json_write(out_dir / "material.manifest.json", manifest)
    return manifest



def _generate_v4(seed, out_dir, turns, tokens_per_turn, min_tokens, min_events, smoke):
    """Independent values and interleaved source roles across a full offline horizon."""
    if turns < 10 or tokens_per_turn <= 0 or min_tokens <= 0 or min_events < 2:
        raise ValueError("require turns >= 10, positive token budgets, min-events >= 2")
    count, rng = token_counter(), random.Random(seed ^ 0x5634)
    rows, facts, lifecycle, probes, continuity = [], [], [], [], []
    grams, used = {}, set()

    def value(cls):
        while True:
            words = rng.sample(VALUE_WORDS, rng.randrange(2, 5))
            parts = [rng.choice((str.lower, str.upper, str.title))(w) for w in words]
            parts.insert(rng.randrange(len(parts) + 1), str(rng.randrange(10000, 10**9)))
            parts.append(rng.choice(("MiB", "rows", "kg", "ms", "jobs", "bytes")))
            rng.shuffle(parts)
            text = rng.choice(("-", "/", " ", "_", ":", ".")).join(parts)
            fragments = {text.casefold()[i:i + 6] for i in range(len(text) - 5)}
            if text not in used and not fragments & grams.get(cls, set()):
                used.add(text)
                grams.setdefault(cls, set()).update(fragments)
                return text

    def add(role, text, call=None):
        i = len(rows)
        rows.append(dict(id=f"S{seed}-R{i:05d}", turn=step + 1, role=role, content=text,
                         tool_call_id=call, ts=float(seed * 10000 + i)))
        return dict(row_index=i, row_id=rows[-1]["id"], row_role=role)

    def filler(chars):
        return "".join(rng.choice(SCENES).format(n=f"{seed}:{rng.getrandbits(64):016x}")
                       for _ in range(chars // 140 + 1))[:chars]

    def statement(item, text, role="user"):
        return dict(id=item, value=text, **add(role, f"[{item}] {text}"))

    groups = {role: [] for role in ("user", "assistant", "tool")}
    for c, cls in enumerate(CLASSES):
        for k in range(5):
            val = value(cls)
            role = "tool" if k in (2, 3) or c == 10 else "assistant" if k == 1 else "user"
            fixture = f"{rng.choice(VALUE_WORDS)}-{rng.randrange(100000, 1000000)}"
            f = dict(id=f"S{seed}-F{c:02d}-{k}", fixture=fixture, **{"class": cls}, value=val,
                     answer=val, stale=value(cls) if c == 6 else None, row_role=role,
                     placement=("head", "head", "middle", "tail", "tail")[k],
                     probe=f"What is the current {cls.replace('_', ' ')} for fixture {fixture}?")
            groups[role].append(f)
            facts.append(f)
    schedule = {}
    for group in groups.values():
        rng.shuffle(group)
    # Stratify each source role, including user facts, across all three thirds.
    for third in range(3):
        users = groups["user"][third::3]
        for j, f in enumerate(users):
            schedule[third * 30 + 3 + j * 3] = f
        group = [f for role in ("assistant", "tool") for f in groups[role][third::3]]
        rng.shuffle(group)
        available = [i for i in range(third * 30, (third + 1) * 30) if i not in schedule]
        for j, f in enumerate(group):
            schedule[available[int((j + .5) * len(available) / len(group))]] = f
    # Supersession sources need two full fresh-token spans after their update.
    for slot, f in list(schedule.items()):
        if f["stale"] and slot >= 30:
            early = next(i for i, other in schedule.items() if i < 30 and not other["stale"] and
                         other["row_role"] == f["row_role"])
            schedule[slot], schedule[early] = schedule[early], f
    user_middle = [f for f in facts if f["row_role"] == "user"][:8]
    for f in user_middle:
        f["placement"] = "middle"
    corrected = [f for i, f in sorted(schedule.items()) if i < 88 and
                 f["row_role"] == "user" and f not in user_middle][:3]
    correction_steps = {}
    request_names = [f"check-{rng.randrange(100000, 1000000)}" for _ in range(6)]
    starts = {5: 0, 10: 1, 15: 2, 20: 3, 25: 4, 45: 5}
    ends = {12: (0, "completed", None), 18: (1, "cancelled", 2),
            27: (2, "superseded", 3), 35: (3, "completed", None), 50: (4, "cancelled", 5)}
    step = 0
    continuity.append(statement(f"S{seed}-CONT-host_instruction",
                                "Preserve chronology and answer from stated values.", "system"))
    # Full smoke also meets the token floor; the forced event remains wiring-only.
    chars = max(28000, (min_events + 2) * min_tokens * 4 // 90,
                turns * tokens_per_turn * 4 // 90)
    for step in range(90):
        f = schedule.get(step)
        if f and f["stale"]:
            f["stale_source"] = statement(f["id"] + "-OLD", f"Initial {f['fixture']}: {f['stale']}.")
        for role, size in (("user", chars // 3), ("tool", chars // 3), ("assistant", chars // 3)):
            target = f if f and f["row_role"] == role else None
            size = max(size, 100004 if target and target["class"] == "file_change" and
                       target["placement"] == "middle" else 8000)
            body = filler(size)
            if target:
                initial = value(target["class"]) if target in corrected else target["value"]
                line = f"[{target['id']}; fixture {target['fixture']}] Current {target['class']}: {initial}.\n"
                offset = {"head": 0, "middle": size // 2, "tail": size}[target["placement"]]
                body = body[:offset] + line + body[offset:]
            call = f"S{seed}-CALL{len(rows):05d}" if role == "tool" else None
            if call:
                add("assistant", f"Read the fixture for {call}.", call)
                rows[-1]["tool_calls"] = [dict(id=call, type="function", function=dict(
                    name="read_material", arguments=json.dumps({"path": f"fixtures/{call}.txt"})))]
            source = add(role, body, call)
            if target:
                target.update(source, turn=step + 1, char_offset=body.index(initial))
                if target in corrected:
                    target["correction_source"] = dict(source, value=initial)
                    correction_steps.setdefault(step + 1, []).append(target)
        for target in correction_steps.get(step, []):
            source = statement(target["id"], f"Use {target['value']}.")
            target.update({k: v for k, v in source.items() if k not in ("id", "value")}, placement="head", turn=step + 1,
                          char_offset=rows[-1]["content"].index(target["value"]))
            probes.append(dict(id=target["id"] + "-CORRECTION", kind="corrected_value",
                               text=target["probe"], expect="value", answer=target["value"], **{k: v for k, v in source.items() if k not in ("id", "value")}))
        if step in starts:
            i = starts[step]
            source = statement(f"S{seed}-REQUEST-{i}", f"Please do {request_names[i]}.")
            lifecycle.append(dict(**source, task=request_names[i], status="pending", replacement=None))
        if step in ends:
            i, status, replacement = ends[step]
            task = request_names[i]
            text = (f"Completed {task}." if status == "completed" else
                    f"Drop {task}, don't do it; do {request_names[replacement]}." if status == "cancelled" else
                    f"Instead of {task} do {request_names[replacement]}.")
            source = statement(f"S{seed}-RESOLUTION-{i}", text, "assistant" if status == "completed" else "user")
            lifecycle[i].update(status=status, replacement=request_names[replacement] if replacement is not None else None,
                                resolution=source)
            if replacement is not None:
                probes.append(dict(id=f"S{seed}-STALE-{i}", kind="stale_task", expect="value",
                                   text=f"Is {task} still to be done?", answer=f"No; do {request_names[replacement]}.",
                                   **{k: v for k, v in source.items() if k not in ("id", "value")}))
        if step in (45, 75):
            for kind, text in (("current_request", f"Please do {request_names[5]}."),
                               ("active_constraint", "Keep source fixtures read-only; write audit results only.")):
                item = statement(f"S{seed}-CONT-{kind}", text)
                if step == 45:
                    continuity.append(item)
                else:
                    previous = next(c for c in continuity if c["id"] == item["id"])
                    previous.update(item)
    state = dict(id=f"S{seed}-STATE", next_action=request_names[5], status="pending",
                 path=f"fixtures/seed-{seed}/rollback.json", decision_id=f"S{seed}-F06-4")
    state.update(add("assistant", f"[{state['id']}] Continuation: " + json.dumps(state, sort_keys=True)))
    current = next(c for c in continuity if c["id"].endswith("current_request"))
    probes.append(dict(id=f"S{seed}-CURRENT", kind="current_request", text="What is the user asking now?",
                       expect="value", answer=current["value"], row_index=current["row_index"], row_id=current["row_id"]))
    presented = sum(count(r["content"]) for r in rows)
    add("user", "Continue with the current request, keeping the active constraint.")
    add("assistant", filler(80004))
    suffix = None
    if smoke:
        suffix = statement(f"S{seed}-SMOKE-EVENT", "Final wiring-only compaction checkpoint.")
        suffix.update(action="force_compaction_after_row", timing_population="WIRING-ONLY", runtime_event_required=True)
    cumulative, checkpoints, boundary, prefix = 0, [], 20000, []
    for i, r in enumerate(rows):
        cumulative += count(r["content"])
        prefix.append(cumulative)
        while cumulative >= boundary:
            checkpoints.append(dict(id=f"S{seed}-CP{boundary}", tokens=cumulative, row_index=i))
            boundary += 20000
    for i, p in enumerate(probes):
        h = (1, 3, 5)[i % 3]
        p.update(compaction_horizon=h, source_token_position=prefix[p["row_index"]],
                 probe_token_position=prefix[p["row_index"]] + h * 8000)
    admissions = [dict(id=f["id"], value=f["value"], row_id=f["row_id"], row_index=f["row_index"],
                       role=f["row_role"], tool_call_id=rows[f["row_index"]]["tool_call_id"],
                       status="scheduled", runtime_row_id=None) for f in facts]
    targets = [dict(id=f["id"], kind="ancestry", row_id=f["row_id"], row_index=f["row_index"],
                    stale_source=f["stale_source"], required_events=min_events) for f in facts if f["stale"]]
    external = next(f for f in facts if f["class"] == "file_change" and f["placement"] == "middle")
    targets.append(dict(id=external["id"], kind="externalization", row_id=external["row_id"], row_index=external["row_index"]))
    traps = []
    for c in rng.sample(range(12), 5):
        sibling = next(f for f in facts if f["class"] == CLASSES[c])
        fixture = sibling["fixture"] + "-annex"
        traps.append(dict(id=f"S{seed}-F{c:02d}-5", probe=sibling["probe"].replace(sibling["fixture"], fixture),
                          answer="ABSTAIN", **{"class": CLASSES[c]}, sibling_id=sibling["id"]))
    flat = [dict(id=f["id"], kind="canary", text=f["probe"], expect="value") for f in facts]
    flat += [dict(id=t["id"], kind="trap", text=t["probe"], expect="ABSTAIN") for t in traps]
    rng.shuffle(flat)
    batches = [dict(id=f"S{seed}-B{k // 10}", probes=flat[k:k + 10], text=BATCH_INSTRUCTION) for k in range(0, 65, 10)]
    manifest = dict(seed=seed, material_version="track-s-v4", mode="smoke" if smoke else "decision",
                    tokenizer="repo-count_tokens:offline-char-estimate", params=dict(turns=turns,
                    tokens_per_turn=tokens_per_turn, min_tokens=min_tokens, min_events=min_events),
                    continuity=continuity, lifecycle=lifecycle, default_leaf_tokens=8000, checkpoints=checkpoints,
                    receipt_targets=targets, smoke_suffix=suffix, presented_tokens=presented,
                    planned_trigger_spans=min_events, decision_checkpoint=checkpoints[-1],
                    proof_boundary="Offline fresh-token planning; compactions and all runtime receipts remain untested.")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, payload in (("facts.json", facts), ("canaries.json", facts), ("traps.json", traps),
                          ("continuation.json", state), ("admission.manifest.json", admissions),
                          ("transcript.jsonl", rows), ("turns.jsonl", [dict(turn=r["turn"], text=r["content"],
                           role=r["role"], id=r["id"], tool_call_id=r["tool_call_id"], tool_calls=r.get("tool_calls", [])) for r in rows]),
                          ("probes.jsonl", flat), ("probe_batches.jsonl", batches), ("lifecycle_probes.jsonl", probes)):
        if name.endswith(".jsonl"):
            (out_dir / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in payload), encoding="utf-8")
        else:
            _json_write(out_dir / name, payload)
        names.append(name)
    manifest["shas"] = {name: _sha256(out_dir / name) for name in sorted(names)}
    _json_write(out_dir / "material.manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=35)
    parser.add_argument("--tokens-per-turn", type=int, default=17000)
    parser.add_argument("--placements", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--classes12", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-tokens", type=int, default=244800)
    parser.add_argument("--min-events", type=int, default=2)
    parser.add_argument("--v4", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="ten-turn prefix plus forced-event wiring suffix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate(args.seed, args.out_dir, args.turns, args.tokens_per_turn,
             args.placements, args.classes12, args.min_tokens, args.min_events, args.smoke, args.v4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
