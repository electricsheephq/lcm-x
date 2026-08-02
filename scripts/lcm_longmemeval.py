#!/usr/bin/env python3
"""LongMemEval retrieval-quality harness CLI for hermes-lcm.

Three subcommands:

    fetch   Download the pinned LongMemEval_S dataset file once (operator step).
    prepare Stream a corpus into checksum-verified per-question files.
    run     Ingest histories into fresh temp LCM stores and score the arms.

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

from benchmarking.longmemeval import (  # noqa: E402
    DATASET_COORDS,
    PROVIDERS,
    dataset_coordinates,
    load_questions_with_sha256,
    load_prepared_dataset,
    prepare_dataset,
    render_markdown,
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
        "--no-db-template",
        dest="reuse_db_template",
        action="store_false",
        help="Disable the reused pre-migrated DB template (bootstrap each question "
        "from scratch). Mainly for measuring the F7 ingest speedup.",
    )
    run.add_argument("--json", action="store_true", help="Print the metrics JSON to stdout.")
    run.add_argument(
        "--allow-external-output",
        action="store_true",
        help="Allow --output outside this repository.",
    )
    return parser.parse_args(argv)


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


def _cmd_run(args: argparse.Namespace) -> int:
    if args.provider != "stub" and not args.model:
        raise SystemExit(f"--model is required for --provider {args.provider}")
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
            manifest_sha256 = None
            question_count = len(questions)
        else:
            prepared = load_prepared_dataset(
                Path(args.prepared_dir), dataset_label=args.dataset_label
            )
            source_sha256 = prepared.source_sha256
            manifest_sha256 = prepared.manifest_sha256
            question_count = (
                prepared.question_count
                if args.limit is None
                else min(prepared.question_count, args.limit)
            )
            # Consume a bounded qid-only preflight before scoring so a short or
            # reordered prepared iterator fails before an expensive medium run.
            prepared.validate_question_ids(limit=args.limit)
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
                reuse_db_template=args.reuse_db_template,
                question_count=question_count,
                dataset_label=args.dataset_label,
                source_sha256=source_sha256,
                manifest_sha256=manifest_sha256,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "run":
        return _cmd_run(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
