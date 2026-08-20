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
        return "Where does the canonical config now live?"
    if class_code == "C3":
        return f"What must the {noun} limit stay exactly?"
    if class_code == "C4":
        return "What was the build id printed by the last successful pipeline run?"
    if class_code == "C5":
        return f"What prefix should all {noun} names use going forward?"
    raise ValueError(f"unknown canary class: {class_code}")


def _canary_sentence(class_code: str, noun: str, value: str) -> str:
    if class_code == "C1":
        return f"After weighing both options we decided to name the {noun} `{value}`."
    if class_code == "C2":
        return f"The canonical config now lives at src/{value}/settings.toml — remember that path."
    if class_code == "C3":
        return f"Hard constraint from ops: the {noun} limit must stay exactly {value}."
    if class_code == "C4":
        return f"The build id printed by the last successful pipeline run was {value}."
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


def generate(seed: int, out_dir: Path, turns: int = 35, tokens_per_turn: int = 17000) -> dict[str, Any]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=35)
    parser.add_argument("--tokens-per-turn", type=int, default=17000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate(args.seed, args.out_dir, args.turns, args.tokens_per_turn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
