"""Public command grammar characterized before extracting the router."""

import pytest

from hermes_lcm import command


# Literal output snapshot, independent of the production help renderer.
HELP = """LCM command help
- /lcm or /lcm status: show current LCM runtime/session status
- /lcm doctor: run read-only LCM health checks
- /lcm doctor clean: best-effort scan of obvious junk/noise session candidates without deleting anything
- /lcm doctor clean apply: backup-first cleanup for safe pattern-matched candidates only
- /lcm doctor clean lifecycle: read-only scan for lifecycle rows with zero messages/nodes
- /lcm doctor clean lifecycle apply: backup-first cleanup of empty lifecycle rows only
- /lcm doctor repair: read-only scan for SQLite/FTS index repair needs
- /lcm doctor repair apply: backup-first repair/rebuild of message and summary FTS indexes
- /lcm doctor repair level3: read-only scan for level 3 truncation fragments and the condensed nodes built on them
- /lcm doctor repair level3 apply: backup-first re-summary of those nodes in place (ids and links kept)
- /lcm doctor repair schema-stamp: read-only scan for an interim-build schema_version stamp ahead of the actual v5 shape
- /lcm doctor repair schema-stamp apply: backup-first reset of an interim schema_version stamp back to the supported version
- /lcm doctor source: read-only scan for legacy blank-source rows
- /lcm doctor source apply: backup-first normalization of legacy blank-source rows to unknown
- /lcm doctor retention: read-only retention analysis for stored session footprint and age
- /lcm backup: create a timestamped SQLite backup before any future cleanup workflow
- /lcm rotate: preview a tail-preserving in-place compact of the active session (read-only)
- /lcm rotate apply: refused on Hermes (#1141); use /compress here N
- /lcm rollups: show temporal-rollup status for the current session
- /lcm rollups rebuild <day|week|month|all> [date]: synchronously rebuild a bounded UTC target set
- /lcm assertions rebuild [--apply] [--limit N]: preview or run one bounded exact-row assertion batch
- /lcm preset show [name]: inspect shipped preset metadata and benchmark provenance
- /lcm preset suggest: preview the best shipped preset for the current engine state
- /lcm preset apply <name> --dry-run: preview env-var changes without mutating live config
- /lcm embed warmup: download/probe the configured embedding model and register its dimension
- /lcm embed backfill [--apply] [--limit N]: preview or populate missing leaf-summary embeddings
- /lcm help: show this help"""
DOCTOR_ERROR = (
    "`/lcm doctor` currently supports `clean`, `clean apply`, `clean lifecycle`, "
    "`clean lifecycle apply`, `repair`, `repair apply`, `repair schema-stamp`, "
    "`repair schema-stamp apply`, `source`, `source apply`, and `retention` as "
    "extra subcommands, plus `repair level3` and `repair level3 apply`."
)
ERRORS = {
    "status": "`/lcm status` does not accept extra arguments.",
    "doctor": DOCTOR_ERROR,
    "backup": "`/lcm backup` does not accept extra arguments.",
    "rotate": "`/lcm rotate` accepts an optional `apply` subcommand.",
    "embed": "`/lcm embed` requires the `warmup` or `backfill` subcommand.",
}

# None means the handler takes only the engine; a tuple is the delegated tail.
HANDLER_CASES = (
    (None, "_status_text", None),
    ("", "_status_text", None),
    (" \t\n ", "_status_text", None),
    ("status", "_status_text", None),
    ("doctor", "_doctor_text", None),
    ("doctor clean", "_doctor_clean_text", None),
    ("doctor repair", "_doctor_repair_text", None),
    ("doctor source", "_doctor_source_text", None),
    ("doctor retention", "_doctor_retention_text", None),
    ("doctor clean apply", "_doctor_clean_apply_text", None),
    ("doctor clean lifecycle", "_doctor_clean_lifecycle_text", None),
    ("doctor clean lifecycle apply", "_doctor_clean_lifecycle_apply_text", None),
    ("doctor repair apply", "_doctor_repair_apply_text", None),
    ("doctor repair level3", "_doctor_repair_level3_text", None),
    ("doctor repair level3 apply", "_doctor_repair_level3_apply_text", None),
    ("doctor repair schema-stamp", "_doctor_repair_schema_stamp_text", None),
    ("doctor repair schema-stamp apply", "_doctor_repair_schema_stamp_apply_text", None),
    ("doctor source apply", "_doctor_source_apply_text", None),
    ("backup", "_backup_text", None),
    ("rotate", "_rotate_text", None),
    ("rotate apply", "_rotate_apply_text", None),
    ("rollups", "_rollups_text", ()),
    ("rollups ReBuild DaY 2026-01-01", "_rollups_text", ("ReBuild", "DaY", "2026-01-01")),
    ("assertions", "_assertions_text", ()),
    ("assertions ReBuild --Limit 5", "_assertions_text", ("ReBuild", "--Limit", "5")),
    ("preset", "_preset_text", ()),
    ("preset ShOw Example", "_preset_text", ("ShOw", "Example")),
    ("preset suggest", "_preset_text", ("suggest",)),
    ("preset apply Example --dry-run", "_preset_text", ("apply", "Example", "--dry-run")),
    ("embed warmup", "_embedding_warmup_text", None),
    ("embed backfill", "_embedding_backfill_text", ()),
    ("embed backfill --Corpus Chunks --Apply", "_embedding_backfill_text", ("--Corpus", "Chunks", "--Apply")),
    ("rollups Unknown Extra", "_rollups_text", ("Unknown", "Extra")),
    ("assertions Unknown Extra", "_assertions_text", ("Unknown", "Extra")),
    ("preset Unknown Extra", "_preset_text", ("Unknown", "Extra")),
    ("embed backfill Unknown Extra", "_embedding_backfill_text", ("Unknown", "Extra")),
)
HELP_CASES = (
    ("help", None), ("HeLp Ignored Extra", None),
    ("WhAt", "Unknown subcommand: WhAt"),
    ("WhAt doctor clean", "Unknown subcommand: WhAt"),
    ("status extra", ERRORS["status"]),
    ("backup extra", ERRORS["backup"]),
    ("rotate foo", ERRORS["rotate"]),
    ("rotate apply extra", ERRORS["rotate"]),
    ("embed", ERRORS["embed"]),
    ("embed foo", ERRORS["embed"]),
    ("embed warmup extra", ERRORS["embed"]),
    ("doctor foo", DOCTOR_ERROR),
    ("doctor clean foo", DOCTOR_ERROR),
    ("doctor repair level3 foo", DOCTOR_ERROR),
    ("doctor repair schema-stamp foo", DOCTOR_ERROR),
    ("doctor clean lifecycle foo", DOCTOR_ERROR),
) + tuple(
    (raw + " extra", DOCTOR_ERROR)
    for raw, _, _ in HANDLER_CASES if raw and raw.startswith("doctor")
)


class UntouchedEngine:
    def __getattribute__(self, name):
        raise AssertionError(f"unexpected engine read: {name}")


@pytest.fixture
def recordings(monkeypatch):
    calls = []
    help_renderer = command._help_text

    def record(name):
        def render(*args):
            calls.append((name, args))
            return help_renderer(*args) if name == "_help_text" else "recorded output"
        return render

    for name in {name for _, name, _ in HANDLER_CASES} | {"_help_text"}:
        monkeypatch.setattr(command, name, record(name))
    return calls


def mixed_case(text):
    return "".join(char.upper() if index % 2 else char.lower() for index, char in enumerate(text))


@pytest.mark.parametrize("transform", [str, mixed_case], ids=["original", "mixed-case"])
@pytest.mark.parametrize("raw,handler,tail", HANDLER_CASES)
def test_public_handler_truth_table(recordings, raw, handler, tail, transform):
    engine = UntouchedEngine()
    transformed = transform(raw) if raw is not None else None
    if tail is None:
        args = (engine,)
    else:
        # Preserve the caller's spelling, including case-sensitive option values.
        prefix_length = 2 if handler == "_embedding_backfill_text" else 1
        args = (transformed.split()[prefix_length:], engine)
        if transform is str:
            assert args[0] == list(tail)
    assert command.handle_lcm_command(transformed, engine) == "recorded output"
    assert recordings == [(handler, args)]


@pytest.mark.parametrize("raw,error", HELP_CASES)
@pytest.mark.parametrize("padded", [False, True])
def test_public_handler_exact_help(recordings, raw, error, padded):
    if padded:
        raw = " \t" + "  ".join(raw.split()) + "\n "
    expected = HELP if error is None else error + "\n\n" + HELP
    assert command.handle_lcm_command(raw, UntouchedEngine()) == expected
    assert recordings == [("_help_text", () if error is None else (error,))]


def test_handler_patch_is_resolved_on_each_call(monkeypatch):
    engine = UntouchedEngine()
    calls = []

    def first(value):
        calls.append(("first", value))
        return "first output"

    def second(value):
        calls.append(("second", value))
        return "second output"

    monkeypatch.setattr(command, "_doctor_text", first)
    assert command.handle_lcm_command("doctor", engine) == "first output"
    monkeypatch.setattr(command, "_doctor_text", second)
    assert command.handle_lcm_command("doctor", engine) == "second output"
    assert calls == [("first", engine), ("second", engine)]


def test_handler_exception_propagates(monkeypatch):
    error = RuntimeError("handler failed")

    def fail(_engine):
        raise error

    monkeypatch.setattr(command, "_doctor_text", fail)
    with pytest.raises(RuntimeError) as caught:
        command.handle_lcm_command("doctor", UntouchedEngine())
    assert caught.value is error
