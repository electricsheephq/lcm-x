#!/usr/bin/env python3
"""LongMemEval retrieval-quality harness CLI for hermes-lcm.

Five subcommands:

    fetch   Download the pinned LongMemEval_S dataset file once (operator step).
    prepare Stream a corpus into checksum-verified per-question files.
    run     Ingest histories into fresh temp LCM stores and score the arms.
    prewarm-cache Populate the optional content-hash embedding cache.
    determinism-probe Compare two live embeddings of 20 unique sessions.

Offline by default: `run` never downloads. Deterministic with `--provider stub`;
`--provider fastembed` uses the local FastEmbed model (CI-grade, no network at
query time once cached). See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarking.longmemeval as _longmemeval  # noqa: E402

from benchmarking.longmemeval import (  # noqa: E402
    DATASET_COORDS,
    EMBED_CACHE_ENV,
    PER_QUESTION_CHECKPOINT_FILENAME,
    PROVIDERS,
    dataset_coordinates,
    embedding_determinism_report,
    load_shard_question_ids,
    load_questions_with_sha256,
    load_prepared_dataset,
    prewarm_embedding_cache,
    prepare_dataset,
    render_markdown,
    resolve_harness_provider,
    run_harness,
    validate_dataset_path_label,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download a pinned LongMemEval dataset file.")
    fetch.add_argument("--output", required=True, help="Directory to write the dataset file into.")
    fetch.add_argument(
        "--dataset-label", default="s", choices=DATASET_COORDS,
        help="Dataset tier to download (default: s).",
    )

    prepare = sub.add_parser(
        "prepare", help="Stream a dataset into per-question JSON files plus a manifest."
    )
    prepare.add_argument("--dataset", required=True, help="Path to the LongMemEval corpus.")
    prepare.add_argument("--prepared-dir", required=True, help="Empty output directory.")
    prepare.add_argument(
        "--dataset-label", default="s", choices=DATASET_COORDS,
        help="Dataset tier label used to validate and stamp provenance (default: s).",
    )
    prepare.add_argument(
        "--allow-external-output",
        action="store_true",
        help="Allow --prepared-dir outside this repository.",
    )

    run = sub.add_parser("run", help="Run the retrieval harness over the dataset.")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Path to the downloaded LongMemEval corpus.")
    source.add_argument(
        "--prepared-dir", help="Path to a checksum-verified prepared dataset directory."
    )
    run.add_argument(
        "--dataset-label", default="s", choices=DATASET_COORDS,
        help="Dataset tier label used to validate and stamp provenance (default: s).",
    )
    run.add_argument("--output", required=True, help="Output directory for metrics JSON + markdown.")
    run.add_argument(
        "--provider",
        default="stub",
        choices=PROVIDERS,
        help="Embedding provider. 'stub' is deterministic/offline (scores meaningless).",
    )
    run.add_argument("--model", default="", help="Embedding model id (required for non-stub).")
    run.add_argument("--limit", type=int, default=None, help="Score only the first N questions.")
    run.add_argument(
        "--rerank",
        action="store_true",
        help="Use the real cross-encoder rerank arm (VoyageProvider.rerank, "
        "rerank-2.5-lite) when --provider voyage; otherwise the deterministic "
        "placeholder cosine reranker is used and labeled as such.",
    )
    run.add_argument(
        "--recall-rerank",
        action="store_true",
        help="Enable the production lcm_recall rerank stage (voyage provider required); "
        "record the binding in the checkpoint header.",
    )
    run.add_argument(
        "--recall-rerank-window",
        type=int,
        default=0,
        metavar="N",
        help="Bound the production lcm_recall rerank window (default: 0, historical behavior); "
        "requires --recall-rerank.",
    )
    run.add_argument(
        "--recall-rerank-margin",
        type=float,
        default=0.0,
        metavar="X",
        help="Hold the incoming production lcm_recall rank-1 candidate unless the "
        "rerank relevance gap reaches X (default: 0.0); requires --recall-rerank.",
    )
    run.add_argument(
        "--no-db-template",
        dest="reuse_db_template",
        action="store_false",
        help="Disable the reused pre-migrated DB template (bootstrap each question "
        "from scratch). Mainly for measuring the F7 ingest speedup.",
    )
    run.add_argument("--json", action="store_true", help="Print the metrics JSON to stdout.")
    run.add_argument(
        "--resume",
        action="store_true",
        help="Resume from per_question_checkpoint.jsonl in --output, failing closed "
        "if it belongs to a different question selection.",
    )
    run.add_argument(
        "--dump-candidates",
        metavar="PATH",
        help="Append per-question ranked retrieval candidates as JSONL to PATH; "
        "questions skipped by --resume are not dumped.",
    )
    run.add_argument(
        "--allow-external-output",
        action="store_true",
        help="Allow --output outside this repository.",
    )

    prewarm = sub.add_parser(
        "prewarm-cache",
        help="Populate LCM_LONGMEMEVAL_EMBED_CACHE from selected prepared shards.",
    )
    prewarm.add_argument("--prepared-dir", required=True)
    prewarm.add_argument(
        "--shards-manifest",
        required=True,
        help="One manifest.json or a directory containing shard-*/manifest.json files.",
    )
    prewarm.add_argument("--dataset-label", default="m", choices=DATASET_COORDS)
    prewarm.add_argument("--provider", default="voyage", choices=PROVIDERS)
    prewarm.add_argument("--model", required=True)
    prewarm.add_argument("--timeout", type=float, default=300.0)
    prewarm.add_argument(
        "--dry-run",
        action="store_true",
        help="Report uncached embedding units without populating the cache.",
    )
    prewarm.add_argument(
        "--changed-manifest",
        metavar="PATH",
        help=(
            "Write one JSONL record per privacy-transformed request unit for this scan "
            "(the file is rewritten each scan and must not alias the cache or any input)."
        ),
    )

    probe = sub.add_parser(
        "determinism-probe",
        help="Spend-bearing Voyage probe: embed random unique sessions twice.",
    )
    probe.add_argument("--prepared-dir", required=True)
    probe.add_argument(
        "--shards-manifest",
        required=True,
        help="One manifest.json or a directory containing shard-*/manifest.json files.",
    )
    probe.add_argument("--dataset-label", default="m", choices=DATASET_COORDS)
    probe.add_argument("--model", required=True)
    probe.add_argument("--sample-size", type=int, default=20)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    args._recall_rerank_window_given = any(
        token == "--recall-rerank-window"
        or token.startswith("--recall-rerank-window=")
        for token in argv
    )
    args._recall_rerank_margin_given = any(
        token == "--recall-rerank-margin"
        or token.startswith("--recall-rerank-margin=")
        for token in argv
    )
    return args


def _validate_output_path(path: Path, *, allow_external: bool) -> Path:
    resolved = path.resolve()
    repo_root = REPO_ROOT.resolve()
    if not allow_external and not resolved.is_relative_to(repo_root):
        raise SystemExit(
            f"Refusing output outside repo: {resolved}. Pass --allow-external-output to override."
        )
    return resolved


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required for `fetch`; install it, then re-run. "
            "The benchmark `run` step itself needs no network."
        )
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = dataset_coordinates(args.dataset_label)
    path = hf_hub_download(
        repo_id=coordinates["repo_id"],
        filename=coordinates["file"],
        repo_type="dataset",
        revision=coordinates["revision"],
        local_dir=str(output_dir),
    )
    print(json.dumps({"dataset_path": path, "revision": coordinates["revision"]}, indent=2))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    prepared_dir = _validate_output_path(
        Path(args.prepared_dir), allow_external=args.allow_external_output
    )
    try:
        manifest = prepare_dataset(
            Path(args.dataset), prepared_dir, dataset_label=args.dataset_label
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _validated_dump_candidates_path(
    raw: str | None, *, output_dir: Path, dataset: str | None
) -> Path | None:
    """Reject --dump-candidates targets other run files would clobber or read.

    The metrics json/md are written unconditionally after the run (a dump
    aliasing them would be silently overwritten AFTER an expensive run), the
    checkpoint file is interleaved with dump writes, and the source dataset
    must never double as a write target.
    """
    if not raw:
        return None
    dump_path = Path(raw).resolve()
    reserved = {
        (output_dir / PER_QUESTION_CHECKPOINT_FILENAME).resolve(),
        (output_dir / "longmemeval_metrics.json").resolve(),
        (output_dir / "longmemeval_metrics.md").resolve(),
    }
    if dataset:
        reserved.add(Path(dataset).resolve())
    if dump_path in reserved:
        raise ValueError(
            f"--dump-candidates must not alias a run input/output file: {dump_path}"
        )
    return dump_path


def _checkpoint_has_header(checkpoint_path: Path) -> bool:
    """True only when ``--resume`` could actually continue: the checkpoint's first
    non-empty line is the harness header object. A refusal raised before the header
    is written (policy resolution, provider/template initialization) leaves no such
    file, and the honest recovery is a fresh run, not a resume."""
    try:
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                return (
                    isinstance(record, dict)
                    and _longmemeval._CHECKPOINT_HEADER_KEY in record
                )
    except (OSError, ValueError):
        return False
    return False


def _run_privacy_block_report(
    exc: BaseException, *, checkpoint_path: Path
) -> dict | None:
    """Structured, redacted receipt for a privacy refusal raised inside a run.

    Documents are validated by the prewarm dry run before any spend, but question
    texts are protected only at run time, so a query-only refusal surfaces here.
    The blocked question has no checkpoint row (``--resume`` re-runs it); the
    receipt carries its id and the per-question counters the harness attached.
    ``resumable`` reflects the durable state: true only when the checkpoint header
    already exists on disk; otherwise ``checkpoint`` is null and the run is
    re-launched fresh. Returns None for any other exception.
    """
    try:
        from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError
    except ImportError:
        return None
    if not isinstance(exc, EmbeddingPrivacyPolicyError):
        return None
    privacy = getattr(exc, "question_privacy", None)
    if privacy is None:
        privacy = getattr(exc, "privacy_counts", None)
    if privacy is None:
        privacy = dict(_longmemeval._PRIVACY_COUNTS)
    resumable = _checkpoint_has_header(checkpoint_path)
    return {
        "status": "blocked",
        "question_id": getattr(exc, "question_id", None),
        "privacy": privacy,
        "checkpoint": str(checkpoint_path) if resumable else None,
        "resumable": resumable,
        "error": str(exc),
    }


def _cmd_run(args: argparse.Namespace) -> int:
    if args.provider != "stub" and not args.model:
        raise SystemExit(f"--model is required for --provider {args.provider}")
    if (
        getattr(args, "_recall_rerank_window_given", False)
        or args.recall_rerank_window != 0
    ) and not args.recall_rerank:
        raise SystemExit("--recall-rerank-window requires --recall-rerank")
    if (
        getattr(args, "_recall_rerank_margin_given", False)
        or args.recall_rerank_margin != 0.0
    ) and not args.recall_rerank:
        raise SystemExit("--recall-rerank-margin requires --recall-rerank")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    output_dir = _validate_output_path(Path(args.output), allow_external=args.allow_external_output)
    if args.prepared_dir is not None:
        prepared_dir = Path(args.prepared_dir).resolve()
        if output_dir.is_relative_to(prepared_dir):
            raise SystemExit(
                f"Refusing --output equal to or inside --prepared-dir: {output_dir}"
            )
        if prepared_dir.is_relative_to(output_dir):
            raise SystemExit(
                f"Refusing --prepared-dir inside --output (the run could clobber the corpus): {prepared_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.dataset is not None:
            dataset_path = Path(args.dataset)
            if not dataset_path.is_file():
                raise SystemExit(
                    f"dataset file not found: {dataset_path}. Run `fetch` first."
                )
            validate_dataset_path_label(dataset_path, args.dataset_label)
            if args.dataset_label == "m":
                raise SystemExit(
                    "run --dataset with --dataset-label m is unsupported; run `prepare` "
                    "and then use `run --prepared-dir` for the medium tier"
                )
            questions, parsed_sha256 = load_questions_with_sha256(
                dataset_path, limit=args.limit
            )
            # The direct small-tier path intentionally preserves the banked v3
            # dataset block. The digest still covers the exact bytes parsed above,
            # but provenance hashes are emitted only for prepared/medium runs.
            source_sha256 = parsed_sha256 if args.dataset_label != "s" else None
            direct_source_sha256 = parsed_sha256
            manifest_sha256 = None
            question_count = len(questions)
            selected_question_ids = tuple(question.question_id for question in questions)
        else:
            prepared = load_prepared_dataset(
                Path(args.prepared_dir), dataset_label=args.dataset_label
            )
            source_sha256 = prepared.source_sha256
            direct_source_sha256 = None
            manifest_sha256 = prepared.manifest_sha256
            question_count = (
                prepared.question_count
                if args.limit is None
                else min(prepared.question_count, args.limit)
            )
            # Consume a bounded qid-only preflight before scoring so a short or
            # reordered prepared iterator fails before an expensive medium run.
            prepared.validate_question_ids(limit=args.limit)
            selected_question_ids = prepared.selected_question_ids(limit=args.limit)
            questions = prepared.iter_questions(limit=args.limit)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    # PreparedDataset.iter_questions is lazy: checksum and id failures occur
    # only while run_harness consumes it. Keep that consumption inside the
    # same clean CLI error boundary as the initial load/manifest checks.
    try:
        with tempfile.TemporaryDirectory(prefix="lcm-longmemeval-") as tmp:
            tmp_dir = Path(tmp)
            os.environ.setdefault("HERMES_HOME", str(tmp_dir / "hermes-home"))
            report = run_harness(
                questions,
                provider_name=args.provider,
                model=args.model,
                tmp_dir=tmp_dir,
                use_rerank=args.rerank,
                recall_rerank=args.recall_rerank,
                recall_rerank_window=args.recall_rerank_window,
                recall_rerank_margin=args.recall_rerank_margin,
                reuse_db_template=args.reuse_db_template,
                question_count=question_count,
                dataset_label=args.dataset_label,
                source_sha256=source_sha256,
                direct_source_sha256=direct_source_sha256,
                manifest_sha256=manifest_sha256,
                checkpoint_path=output_dir / PER_QUESTION_CHECKPOINT_FILENAME,
                dump_candidates_path=_validated_dump_candidates_path(
                    args.dump_candidates, output_dir=output_dir, dataset=args.dataset
                ),
                resume=args.resume,
                selected_question_ids=selected_question_ids,
            )
    except Exception as exc:
        blocked = _run_privacy_block_report(
            exc, checkpoint_path=output_dir / PER_QUESTION_CHECKPOINT_FILENAME
        )
        if blocked is not None:
            print(json.dumps(blocked, indent=2, sort_keys=True))
            return 2
        if isinstance(exc, (OSError, ValueError)):
            raise SystemExit(str(exc)) from exc
        raise

    metrics_path = output_dir / "longmemeval_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_markdown(report)
    (output_dir / "longmemeval_metrics.md").write_text(markdown + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(markdown)
        print(f"\nmetrics: {metrics_path}")
    return 0


class _ShardSelection:
    """Manifest-selected questions plus how much of the prepared corpus they cover.

    ``coverage`` lets ``prewarm-cache`` label its privacy counters truthfully: a
    shard union that misses a manifest or a question is a partial scan, and its
    report must say so instead of claiming ``privacy_scope: corpus``. Extra ids are
    refused by the prepared dataset and duplicates by the shard loader, so equal
    counts mean an exact partition.
    """

    def __init__(self, prepared, question_ids: tuple[str, ...]) -> None:
        self._prepared = prepared
        self._question_ids = question_ids
        self.coverage = {
            "selected": len(set(question_ids)),
            "prepared": len(prepared.questions),
        }

    def __iter__(self):
        return self._prepared.iter_question_ids(self._question_ids)


def _prepared_shard_questions(args: argparse.Namespace) -> _ShardSelection:
    prepared = load_prepared_dataset(
        Path(args.prepared_dir), dataset_label=args.dataset_label
    )
    question_ids = load_shard_question_ids(Path(args.shards_manifest))
    return _ShardSelection(prepared, tuple(question_ids))


def _refuse_changed_manifest_input_alias(
    changed_manifest: Path, args: argparse.Namespace
) -> None:
    """Refuse a --changed-manifest that names or sits inside one of the command's inputs.

    The manifest is opened for writing before the scan; an alias of the prepared
    corpus manifest or a shard manifest (path, symlink or hard link) would destroy
    the run's own provenance. The harness separately refuses the cache file.
    """
    target = changed_manifest.resolve()
    shards = Path(args.shards_manifest)
    guarded_dirs = [
        Path(args.prepared_dir).resolve(),
        shards.resolve() if shards.is_dir() else shards.resolve().parent,
    ]
    for directory in guarded_dirs:
        if target == directory or target.is_relative_to(directory):
            raise SystemExit(
                "Refusing --changed-manifest inside an input location "
                f"(it is rewritten each scan): {changed_manifest}"
            )
    if not changed_manifest.exists():
        return
    prepared_dir = Path(args.prepared_dir)
    manifests = [shards, prepared_dir / "manifest.json"]
    if shards.is_dir():
        manifests.extend(sorted(shards.glob("shard-*/manifest.json")))
    inputs: list[Path] = list(manifests)
    # Every question file a manifest references is an input too: a hard link to
    # one resolves outside the guarded directories yet names the same inode.
    for manifest in manifests:
        if not manifest.is_file():
            continue
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8")).get("questions", [])
        except (OSError, ValueError, AttributeError):
            entries = []
        for entry in entries if isinstance(entries, list) else []:
            name = entry.get("file") if isinstance(entry, dict) else None
            if isinstance(name, str) and name:
                inputs.append(manifest.parent / name)
                inputs.append(prepared_dir / name)
    seen: set[Path] = set()
    for candidate in inputs:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and os.path.samefile(changed_manifest, candidate):
            raise SystemExit(
                f"Refusing --changed-manifest that is the same file as an input: {candidate}"
            )


def _cmd_prewarm_cache(args: argparse.Namespace) -> int:
    if os.environ.get(EMBED_CACHE_ENV) is None:
        raise SystemExit(f"{EMBED_CACHE_ENV} must name the SQLite cache file")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.changed_manifest is not None:
        _refuse_changed_manifest_input_alias(Path(args.changed_manifest), args)
    _longmemeval._ensure_hermes_lcm_package()
    from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError

    try:
        questions = _prepared_shard_questions(args)
        provider = resolve_harness_provider(
            args.provider,
            args.model,
            timeout=args.timeout,
            warmup=args.provider == "fastembed",
        )
        report = prewarm_embedding_cache(
            questions,
            provider,
            # Progress is operator chatter; stdout stays one JSON object so
            # consumers can json.loads() the success, dry-run and blocked reports.
            progress=lambda processed: print(
                f"prewarm processed={processed}", file=sys.stderr, flush=True
            ),
            dry_run=args.dry_run,
            changed_manifest=args.changed_manifest,
            # None when a caller hands in a bare iterable; the shard selection
            # carries the counts that decide "corpus" vs "partial".
            question_coverage=getattr(questions, "coverage", None),
        )
    except EmbeddingPrivacyPolicyError as exc:
        privacy = getattr(exc, "privacy_counts", None)
        if privacy is None:
            privacy = dict(_longmemeval._PRIVACY_COUNTS)
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "privacy": privacy,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _cmd_determinism_probe(args: argparse.Namespace) -> int:
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    _longmemeval._ensure_hermes_lcm_package()
    from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError

    try:
        questions = _prepared_shard_questions(args)
        # Measurement-neutrality requires two fresh live API passes. Ignore the
        # optional cache and avoid a separate query warmup/API call.
        provider = resolve_harness_provider(
            "voyage",
            args.model,
            timeout=args.timeout,
            use_embed_cache=False,
            warmup=False,
        )
        report = embedding_determinism_report(
            questions,
            provider,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except EmbeddingPrivacyPolicyError as exc:
        privacy = getattr(exc, "privacy_counts", None)
        if privacy is None:
            privacy = dict(_longmemeval._PRIVACY_COUNTS)
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "privacy": privacy,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "prewarm-cache":
        return _cmd_prewarm_cache(args)
    if args.command == "determinism-probe":
        return _cmd_determinism_probe(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
