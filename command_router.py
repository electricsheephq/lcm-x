"""Ordered grammar for /lcm; authorization and renderers remain in command."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Route:
    path: tuple[str, ...]
    handler: str
    # exact: engine only; tail: remaining tokens then engine; ignore: no args.
    arity: Literal["exact", "tail", "ignore"] = "exact"


ROUTES = (
    Route(("status",), "_status_text"),
    Route(("doctor",), "_doctor_text"),
    Route(("doctor", "clean"), "_doctor_clean_text"),
    Route(("doctor", "repair"), "_doctor_repair_text"),
    Route(("doctor", "source"), "_doctor_source_text"),
    Route(("doctor", "retention"), "_doctor_retention_text"),
    Route(("doctor", "clean", "apply"), "_doctor_clean_apply_text"),
    Route(("doctor", "clean", "lifecycle"), "_doctor_clean_lifecycle_text"),
    Route(("doctor", "clean", "lifecycle", "apply"), "_doctor_clean_lifecycle_apply_text"),
    Route(("doctor", "repair", "apply"), "_doctor_repair_apply_text"),
    Route(("doctor", "repair", "level3"), "_doctor_repair_level3_text"),
    Route(("doctor", "repair", "level3", "apply"), "_doctor_repair_level3_apply_text"),
    Route(("doctor", "repair", "schema-stamp"), "_doctor_repair_schema_stamp_text"),
    Route(("doctor", "repair", "schema-stamp", "apply"), "_doctor_repair_schema_stamp_apply_text"),
    Route(("doctor", "source", "apply"), "_doctor_source_apply_text"),
    Route(("backup",), "_backup_text"),
    Route(("rotate",), "_rotate_text"),
    Route(("rotate", "apply"), "_rotate_apply_text"),
    Route(("rollups",), "_rollups_text", "tail"),
    Route(("assertions",), "_assertions_text", "tail"),
    Route(("preset",), "_preset_text", "tail"),
    Route(("embed", "warmup"), "_embedding_warmup_text"),
    Route(("embed", "backfill"), "_embedding_backfill_text", "tail"),
    Route(("help",), "_help_text", "ignore"),
)

FALLBACK_HELP = {
    "status": "`/lcm status` does not accept extra arguments.",
    "doctor": (
        "`/lcm doctor` currently supports `clean`, `clean apply`, `clean lifecycle`, "
        "`clean lifecycle apply`, `repair`, `repair apply`, `repair schema-stamp`, "
        "`repair schema-stamp apply`, `source`, `source apply`, and `retention` as "
        "extra subcommands, plus `repair level3` and `repair level3 apply`."
    ),
    "backup": "`/lcm backup` does not accept extra arguments.",
    "rotate": "`/lcm rotate` accepts an optional `apply` subcommand.",
    "embed": "`/lcm embed` requires the `warmup` or `backfill` subcommand.",
}


def route(head: str, rest: list[str], engine, handlers: Mapping[str, Callable]) -> str:
    """Resolve only the selected renderer from the caller's live globals.

    Match case-insensitively while passing the original tokens to delegates.
    Do not inspect engine state here: authorization already ran in the caller.
    """
    path = (head.lower(), *(part.lower() for part in rest))
    for entry in ROUTES:
        if path[:len(entry.path)] != entry.path:
            continue
        if entry.arity == "exact" and len(path) != len(entry.path):
            continue
        handler = handlers[entry.handler]
        if entry.arity == "tail":
            return handler(rest[len(entry.path) - 1:], engine)
        if entry.arity == "ignore":
            return handler()
        return handler(engine)
    return handlers["_help_text"](FALLBACK_HELP.get(path[0], f"Unknown subcommand: {head}"))
