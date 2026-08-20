#!/usr/bin/env python3
"""Render numeric pilot summaries from one or more arm directories."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_required_scores(path: Path) -> dict[str, Any]:
    """Read one arm's scores artifact without defaulting failures to zeros."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"scores artifact missing or unreadable: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"scores artifact invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"scores artifact must be a JSON object: {path}")
    if not isinstance(value.get("probes"), list) or not isinstance(value.get("totals"), dict):
        raise ValueError(f"scores artifact missing probes/totals: {path}")
    return value


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _find_first(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in keys:
                return item
        for item in value.values():
            found = _find_first(item, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first(item, keys)
            if found is not None:
                return found
    return None


def _sum_duration_ms(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("duration_ms", "stall_ms", "wall_ms", "duration", "stall_seconds"):
            if key in value and _number(value[key]) is not None:
                number = float(value[key])
                return number * 1000 if key in {"duration", "stall_seconds"} else number
        return None
    if isinstance(value, list):
        values = [_sum_duration_ms(item) for item in value]
        values = [item for item in values if item is not None]
        return sum(values) if values else None
    return None


def _rate(aggregate: Any, key: str, numerator: str, denominator: str) -> float | None:
    if not isinstance(aggregate, dict):
        return None
    direct = _number(aggregate.get(key))
    if direct is not None:
        return float(direct)
    num = _number(aggregate.get(numerator))
    den = _number(aggregate.get(denominator))
    return float(num / den) if num is not None and den else None


def _format_map(section: Any) -> str:
    if not isinstance(section, dict):
        return ""
    bits = []
    for key in sorted(section):
        aggregate = section[key]
        retention = _rate(aggregate, "retention", "correct", "canary_total")
        if retention is None:
            retention = _rate(aggregate, "accuracy", "correct", "total")
        bits.append(f"{key}={retention:.6g}" if retention is not None else f"{key}=0")
    return "; ".join(bits)


def _hallucination_rate(scores: dict[str, Any]) -> float:
    totals = scores.get("totals") if isinstance(scores.get("totals"), dict) else scores
    value = _rate(totals, "hallucination_rate", "hallucinate", "total")
    return value if value is not None else 0.0


def _config_sha(manifest: dict[str, Any]) -> str:
    for key in ("config_sha256", "config_sha", "config_hash"):
        if manifest.get(key) is not None:
            return str(manifest[key])
    config = manifest.get("config")
    if isinstance(config, dict):
        for key in ("sha256", "sha"):
            if config.get(key) is not None:
                return str(config[key])
    return ""


def _arm_metrics(name: str, directory: Path) -> dict[str, Any]:
    scores = _read_required_scores(directory / "scores.json")
    manifest_path = directory / "run.manifest.json"
    if not manifest_path.exists():
        manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    rollout = _read_json(directory / "report.json")
    if not rollout:
        rollout = _read_json(directory / "parse_rollout" / "report.json")

    compaction_count = _find_first(
        rollout,
        {"compaction_count", "compactions_count", "compact_count", "num_compactions"},
    )
    if compaction_count is None:
        compactions = _find_first(rollout, {"compactions", "compaction_events"})
        if isinstance(compactions, list):
            compaction_count = len(compactions)
        elif isinstance(compactions, dict):
            compaction_count = _find_first(compactions, {"count", "total"}) or 0
        else:
            compaction_count = _number(compactions) or 0
    elif isinstance(compaction_count, dict):
        compaction_count = _find_first(compaction_count, {"count", "total"}) or 0
    compaction_count = _number(compaction_count) or 0

    stall_total = _find_first(
        rollout,
        {"stall_total_ms", "stall_ms_total", "total_stall_ms", "stall_ms"},
    )
    if stall_total is None:
        stall_total = _find_first(rollout, {"stall_total_seconds", "stall_seconds"})
        if stall_total is not None:
            stall_total = float(stall_total) * 1000
    if stall_total is None:
        stalls = _find_first(rollout, {"stalls", "stall_events"})
        stall_total = _sum_duration_ms(stalls) if isinstance(stalls, list) else 0
    elif isinstance(stall_total, dict):
        stall_total = _find_first(stall_total, {"total", "value", "ms"}) or 0
    stall_total = _number(stall_total) or 0

    token_total = _find_first(
        rollout,
        {"token_total", "total_tokens", "tokens_total", "token_count"},
    )
    if isinstance(token_total, dict):
        token_total = _find_first(token_total, {"total", "count", "value"})
    if token_total is None:
        token_values_map = _find_first(rollout, {"tokens"})
        if isinstance(token_values_map, dict):
            token_total = _find_first(token_values_map, {"total", "count"})
    if token_total is None:
        token_values = [
            _number(_find_first(rollout, {key}))
            for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")
        ]
        token_total = sum(value for value in token_values if value is not None)
    token_total = _number(token_total) or 0

    return {
        "name": name,
        "directory": str(directory),
        "scores": scores,
        "probes": {
            str(row.get("probe_id")): row
            for row in scores.get("probes", [])
            if isinstance(row, dict) and row.get("probe_id") is not None
        },
        "epoch_retention": _format_map(scores.get("per_epoch", scores.get("by_epoch"))),
        "class_retention": _format_map(scores.get("per_class", scores.get("by_class"))),
        "hallucination_rate": _hallucination_rate(scores),
        "compaction_count": compaction_count,
        "stall_total_ms": stall_total,
        "token_total": token_total,
        "config_sha": _config_sha(manifest),
        "timeout_count": _number(
            (scores.get("totals") or {}).get(
                "timeout", (scores.get("totals") or {}).get("timed_out", 0)
            )
        )
        or 0,
    }


def parse_arm_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    arms = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"arm must be name=path: {spec!r}")
        name, path = spec.split("=", 1)
        if not name or not path:
            raise ValueError(f"arm must be name=path: {spec!r}")
        arms.append((name, Path(path)))
    if not arms:
        raise ValueError("at least one --arm-dirs name=path is required")
    return arms


def _aa_prefix(name: str) -> str:
    value = name.replace("′", "'").replace("’", "'").strip()
    for suffix in ("-prime", "_prime", " prime"):
        if value.casefold().endswith(suffix):
            return value[: -len(suffix)]
    if value.endswith("'"):
        return value[:-1]
    return value


def _render_aa(arms: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        groups.setdefault(_aa_prefix(arm["name"]), []).append(arm)
    lines: list[str] = []
    for prefix, group in sorted(groups.items()):
        if len(group) < 2 or not any(arm["name"] != prefix for arm in group):
            continue
        for left, right in itertools.combinations(group, 2):
            probe_ids = sorted(set(left["probes"]) | set(right["probes"]))
            discordant = 0
            lines.extend(
                [
                    f"### {left['name']} / {right['name']}",
                    "",
                    "| probe_id | left | right | discordant |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for probe_id in probe_ids:
                left_row = left["probes"].get(probe_id, {})
                right_row = right["probes"].get(probe_id, {})
                left_class = left_row.get("classification", left_row.get("score", ""))
                right_class = right_row.get("classification", right_row.get("score", ""))
                differs = int(left_class != right_class)
                discordant += differs
                lines.append(f"| {probe_id} | {left_class} | {right_class} | {differs} |")
            left_total = left["scores"].get("totals", {})
            right_total = right["scores"].get("totals", {})
            left_retention = _rate(left_total, "retention", "correct", "canary_total") or 0.0
            right_retention = _rate(right_total, "retention", "correct", "canary_total") or 0.0
            lines.extend(
                [
                    "",
                    "| aggregate | left | right | spread |",
                    "| --- | ---: | ---: | ---: |",
                    f"| retention | {left_retention:.6g} | {right_retention:.6g} | {abs(left_retention - right_retention):.6g} |",
                    f"| hallucination_rate | {left['hallucination_rate']:.6g} | {right['hallucination_rate']:.6g} | {abs(left['hallucination_rate'] - right['hallucination_rate']):.6g} |",
                    f"| discordance_count | {discordant} | {len(probe_ids)} | {discordant / len(probe_ids) if probe_ids else 0:.6g} |",
                    "",
                ]
            )
    return lines


def render(arm_specs: Iterable[str]) -> str:
    arms = [_arm_metrics(name, path) for name, path in parse_arm_specs(arm_specs)]
    lines = [
        "# Compaction pilot report",
        "",
        "| arm | epoch_retention | class_retention | hallucination_rate | compaction_count | stall_total_ms | token_total | config_sha | timeout_count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for arm in arms:
        lines.append(
            f"| {arm['name']} | {arm['epoch_retention']} | {arm['class_retention']} | "
            f"{arm['hallucination_rate']:.6g} | {arm['compaction_count']} | "
            f"{arm['stall_total_ms']} | {arm['token_total']} | {arm['config_sha']} | "
            f"{arm['timeout_count']} |"
        )
    aa_lines = _render_aa(arms)
    if aa_lines:
        lines.extend(["", "## A/A′", "", *aa_lines])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dirs", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output = render(args.arm_dirs)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"report_pilot.py: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
