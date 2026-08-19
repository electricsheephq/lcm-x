#!/usr/bin/env python3
"""389x single-store retrieval scale instrument.

The promoted instrument preserves F34's session-level metric and adds
answer-turn completeness over the exact delivered hit payload.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.instruments.scale389.metrics import (  # noqa: E402
    DEFAULT_UNCENSORED_N,
    UNCENSORED_ARMS,
    answer_turn_delivery_metrics,
    answer_turns_from_question,
    emit_probe_question_pin,
    select_questions,
    session_gold_metrics,
    summarise_rows,
)


CORPUS = Path(
    os.environ.get(
        "SCALE389_CORPUS",
        "/Volumes/LEXAR/Codex/session-notes/2026-07-25/"
        "hermes-v1-scale/artifacts/corpus",
    )
)
RESULTS = Path(
    os.environ.get(
        "SCALE389_RESULTS",
        "/Volumes/LEXAR/Codex/session-notes/2026-07-29/"
        "hermes-r3-1/artifacts/laneINSTPREP-logs/scale389-results",
    )
)
FILESCAN = Path(
    os.environ.get(
        "SCALE389_FILESCAN", "/Volumes/LEXAR/hermes-work/phase1a-filescan"
    )
)
WORKDIR = Path(
    os.environ.get(
        "SCALE389_WORKDIR", "/Volumes/LEXAR/hermes-work/mb-workdir-phase1a"
    )
)
PRODUCT_REPO = Path(
    os.environ.get("HERMES_LCM_REPO", "/Volumes/LEXAR/hermes-work/hermes-lcm")
)
BRIDGE = Path(
    os.environ.get(
        "SCALE389_BRIDGE",
        str(
            REPO_ROOT
            / "benchmarks/qa-harness/src/providers/hermes-lcm/bridge/"
            "hermes_lcm_bridge.py"
        ),
    )
)
LIMIT = 25
REPS = 3
SCALES = (500, 2000, 8000, 19829)
NO_DEADLINE_S = 3600.0
UNCENSORED_N = int(
    os.environ.get("SCALE389_UNCENSORED_N", str(DEFAULT_UNCENSORED_N))
)

os.environ.setdefault("HERMES_LCM_REPO", str(PRODUCT_REPO))
os.environ.setdefault("HERMES_MB_WORKDIR", str(WORKDIR))
os.environ.setdefault("HERMES_MB_PROVIDER", "fastembed")
# The engine's interval-gated FTS integrity check WRITES metadata timestamps on
# the read path, which breaks store-freeze sha verification mid-run. Disable
# startup checks for instrument runs (structural checks + doctor still exist).
os.environ.setdefault("LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS", "-1")
os.environ.setdefault(
    "LCM_LONGMEMEVAL_FASTEMBED_CACHE",
    "/Volumes/LEXAR/hermes-work/fastembed-cache",
)

_PROBE_BRIDGE_CLASS: type | None = None

def _probe_bridge_class() -> type:
    global _PROBE_BRIDGE_CLASS
    if _PROBE_BRIDGE_CLASS is not None:
        return _PROBE_BRIDGE_CLASS
    spec = importlib.util.spec_from_file_location("scale389_hermes_bridge", BRIDGE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import bridge: {BRIDGE}")
    module = importlib.util.module_from_spec(spec)
    saved = sys.stdout
    spec.loader.exec_module(module)
    sys.stdout = saved

    class ProbeBridge(module.Bridge):
        search_embeddings = True
        query_timeout_s: float | None = None

        def _config(self, db_path: Path):
            config = super()._config(db_path)
            if not self.search_embeddings:
                config = replace(config, embeddings_enabled=False)
            if self.query_timeout_s is not None:
                config = replace(
                    config, recall_query_timeout_s=self.query_timeout_s
                )
            return config

        def search_instrumented(
            self, req: dict[str, Any]
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            from types import SimpleNamespace

            import hermes_lcm.tools as lcm_tools
            from hermes_lcm.dag import SummaryDAG
            from hermes_lcm.store import MessageStore
            from hermes_lcm.vector_store import VectorStore

            container_tag = str(req["containerTag"])
            query = str(req.get("query", ""))
            limit = int(req.get("limit", LIMIT))
            db_path = self._db_path(container_tag)
            config = self._config(db_path)
            store = MessageStore(str(db_path), ingest_protection_config=config)
            dag = SummaryDAG(str(db_path))
            vector_store = VectorStore(str(db_path), config=config)
            try:
                engine = SimpleNamespace(
                    _config=config,
                    _store=store,
                    _dag=dag,
                    _hermes_home=str(self.workdir),
                    current_session_id=(
                        f"__hermes_lcm_recall_probe__{container_tag}"
                    ),
                )
                cache_key = (
                    self.provider_name.strip().lower(),
                    str(self.embedder.model_id).strip() if self.embedder else "",
                )
                if self.embedder is not None:
                    engine._lcm_embedding_provider_cache = (
                        cache_key,
                        self.embedder,
                    )
                payload = json.loads(
                    lcm_tools.lcm_recall(
                        {"query": query, "limit": limit}, engine=engine
                    )
                )
            finally:
                vector_store.close()
                dag.close()
                store.close()
            if "error" in payload:
                raise RuntimeError(f"lcm_recall error: {payload['error']}")
            hits = payload.get("hits", [])[:limit]
            metadata = {key: value for key, value in payload.items() if key != "hits"}
            return hits, metadata

    _PROBE_BRIDGE_CLASS = ProbeBridge
    return ProbeBridge


def load_union() -> dict[str, dict[str, Any]]:
    union: dict[str, dict[str, Any]] = {}
    with (CORPUS / "union.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            union[row["sid"]] = row
    return union


def load_qeval() -> dict[str, Any]:
    qeval = json.loads((CORPUS / "qeval.json").read_text(encoding="utf-8"))
    missing = [
        row for row in qeval["questions"] if "answer_turns" not in row
    ]
    if missing:
        union_path = CORPUS / "union.jsonl"
        try:
            union = load_union()
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"cannot read corpus union {union_path} for question "
                f"{missing[0]['question_id']!r}: {exc}"
            ) from exc
        for row in missing:
            row["answer_turns"] = answer_turns_from_question(
                row, union_path, union
            )
    return qeval


def eval_questions(
    arm: str, uncensored_n: int = UNCENSORED_N
) -> list[dict[str, Any]]:
    return select_questions(
        load_qeval(), arm, uncensored_n=uncensored_n
    )


def load_ladder() -> dict[str, Any]:
    return json.loads((CORPUS / "ladder.json").read_text(encoding="utf-8"))


def session_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": record["messages"],
        "metadata": {"date": record["date"]},
        "sessionId": record["sid"],
    }


def tag_for_scale(n: int) -> str:
    return f"phase1a-scale-{n}"


def tag_for_s0(question_id: str) -> str:
    return f"phase1a-s0-{question_id}"


def dates_path(scale: str, question_id: str) -> Path:
    tag = (
        tag_for_s0(question_id)
        if scale == "S0"
        else tag_for_scale(int(scale))
    )
    return WORKDIR / f"{tag}.dates.json"


def load_dates(scale: str, question_id: str) -> dict[str, Any]:
    path = dates_path(scale, question_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing stable identity sidecar: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"sidecar is not an object: {path}")
    return value


def store_stats(db_path: Path) -> dict[str, Any]:
    import sqlite3

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    stats: dict[str, Any] = {}
    for name, sql in (
        ("messages", "SELECT COUNT(*) FROM messages"),
        ("summary_nodes", "SELECT COUNT(*) FROM summary_nodes"),
        ("summary_vectors", "SELECT COUNT(*) FROM lcm_embedding_vectors"),
        ("chunk_vectors", "SELECT COUNT(*) FROM lcm_chunk_vectors"),
    ):
        try:
            stats[name] = connection.execute(sql).fetchone()[0]
        except sqlite3.Error as exc:
            stats[name] = f"n/a ({exc})"
    connection.close()
    return stats


def ingest_scale(n: int) -> None:
    union = load_union()
    order = load_ladder()["scales"][str(n)]["ingest_order"]
    bridge = _probe_bridge_class()()
    bridge.initialize({})
    db_path = bridge._db_path(tag_for_scale(n))
    if db_path.exists():
        raise SystemExit(f"refusing to re-ingest existing store: {db_path}")
    seen: set[str] = set()
    started = time.monotonic()
    n_messages = 0
    for index, session_id in enumerate(order, 1):
        if session_id in seen:
            raise SystemExit(
                f"duplicate session in ingest order at {index}: {session_id}"
            )
        seen.add(session_id)
        bridge.ingest(
            {
                "containerTag": tag_for_scale(n),
                "session": session_payload(union[session_id]),
            }
        )
        n_messages += len(union[session_id]["messages"])
        if index % 500 == 0 or index == len(order):
            elapsed = time.monotonic() - started
            print(
                f"  {index}/{len(order)} sessions {n_messages} msgs "
                f"{elapsed:.0f}s ({index / elapsed:.1f} sess/s)",
                flush=True,
            )
    output = {
        "db_bytes": db_path.stat().st_size,
        "ingest_seconds": round(time.monotonic() - started, 2),
        "messages": n_messages,
        "scale": n,
        "sessions": len(order),
        "tag": tag_for_scale(n),
        **store_stats(db_path),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"ingest-scale-{n}.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


def ingest_s0(uncensored_n: int) -> None:
    union = load_union()
    questions = eval_questions("A2", uncensored_n)
    bridge = _probe_bridge_class()()
    bridge.initialize({})
    started = time.monotonic()
    per_question: list[dict[str, Any]] = []
    for index, question in enumerate(questions, 1):
        tag = tag_for_s0(question["question_id"])
        db_path = bridge._db_path(tag)
        if db_path.exists():
            raise SystemExit(f"refusing to re-ingest existing store: {db_path}")
        seen: set[str] = set()
        question_started = time.monotonic()
        n_messages = 0
        for session_id in question["haystack"]:
            if session_id in seen:
                continue
            seen.add(session_id)
            bridge.ingest(
                {
                    "containerTag": tag,
                    "session": session_payload(union[session_id]),
                }
            )
            n_messages += len(union[session_id]["messages"])
        per_question.append(
            {
                "ingest_seconds": round(
                    time.monotonic() - question_started, 3
                ),
                "messages": n_messages,
                "question_id": question["question_id"],
                "sessions": len(seen),
            }
        )
        if index % 10 == 0:
            print(
                f"  {index}/{len(questions)} stores "
                f"{time.monotonic() - started:.0f}s",
                flush=True,
            )
    output = {
        "ingest_seconds_total": round(time.monotonic() - started, 2),
        "messages_mean": round(
            statistics.mean(row["messages"] for row in per_question), 1
        ),
        "per_question": per_question,
        "scale": "S0",
        "sessions_mean": round(
            statistics.mean(row["sessions"] for row in per_question), 1
        ),
        "stores": len(per_question),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ingest-s0.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "per_question"},
            indent=2,
        )
    )


def materialise_files(uncensored_n: int) -> None:
    union = load_union()
    ladder = load_ladder()
    base = FILESCAN / "sessions"
    base.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for session_id, record in union.items():
        path = base / f"{session_id}.txt"
        if path.exists():
            continue
        path.write_text(
            "\n".join(
                f"{message.get('role', 'user')}: "
                f"{message.get('content', '')}"
                for message in record["messages"]
            ),
            encoding="utf-8",
        )
    setup: dict[str, Any] = {
        "scales": {},
        "write_seconds": round(time.monotonic() - started, 2),
    }
    for n in SCALES:
        directory = FILESCAN / f"scope-{n}"
        directory.mkdir(parents=True, exist_ok=True)
        members = ladder["scales"][str(n)]["ingest_order"]
        scale_started = time.monotonic()
        linked = 0
        for session_id in members:
            link = directory / f"{session_id}.txt"
            if not link.exists():
                os.link(base / f"{session_id}.txt", link)
                linked += 1
        setup["scales"][str(n)] = {
            "bytes": sum(
                (directory / f"{session_id}.txt").stat().st_size
                for session_id in members
            ),
            "files": len(members),
            "link_seconds": round(time.monotonic() - scale_started, 2),
            "linked": linked,
        }
    s0_started = time.monotonic()
    questions = eval_questions("A2", uncensored_n)
    for question in questions:
        directory = FILESCAN / f"scope-s0-{question['question_id']}"
        directory.mkdir(parents=True, exist_ok=True)
        for session_id in dict.fromkeys(question["haystack"]):
            link = directory / f"{session_id}.txt"
            if not link.exists():
                os.link(base / f"{session_id}.txt", link)
    setup["s0"] = {
        "link_seconds": round(time.monotonic() - s0_started, 2),
        "stores": len(questions),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "setup-filescan.json").write_text(
        json.dumps(setup, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(setup, indent=2))


_STOP = set(
    """a an the and or but if then than that this these those of in on at to for
    from with without by about into over under again further once here there when
    where why how all any both each few more most other some such no nor not only
    own same so too very s t can will just don should now i me my we our you your
    he him his she her it its they them what which who whom be been being am is
    are was were do does did doing have has had having as up down out off yes""".split()
)


def query_terms(question: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for character in question.lower():
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    result: list[str] = []
    for token in tokens:
        if len(token) > 2 and token not in _STOP and token not in result:
            result.append(token)
    return result[:20]


def fts_safe(question: str) -> str:
    characters = [
        character if character.isalnum() or character.isspace() else " "
        for character in question
    ]
    return " ".join("".join(characters).split())


def filescan_search(
    scale: int | str,
    question: str,
    *,
    question_id: str,
    limit: int = LIMIT,
) -> list[dict[str, Any]]:
    terms = query_terms(question)
    if not terms:
        return []
    args = ["rg", "--count-matches", "--no-messages", "-i", "-w"]
    for term in terms:
        args.extend(("-e", term))
    scope = (
        f"scope-s0-{question_id}" if scale == "S0" else f"scope-{scale}"
    )
    directory = FILESCAN / scope
    args.append(str(directory))
    completed = subprocess.run(args, capture_output=True, text=True)
    counts: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        path, _, count = line.rpartition(":")
        if not path:
            continue
        try:
            counts.append((int(count), Path(path).stem))
        except ValueError:
            continue
    counts.sort(key=lambda pair: (-pair[0], pair[1]))
    return [
        {
            "content": (directory / f"{session_id}.txt").read_text(
                encoding="utf-8"
            ),
            "kind": "file",
            "session_id": session_id,
        }
        for _, session_id in counts[:limit]
    ]


def _persisted_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": str(hit.get("snippet") or hit.get("content") or ""),
        "kind": hit.get("kind"),
        "session_id": (
            str(hit["session_id"]) if hit.get("session_id") is not None else None
        ),
    }


def run_query_arm(arm: str, scale: str, uncensored_n: int) -> None:
    questions = eval_questions(arm, uncensored_n)
    probe_pin = emit_probe_question_pin(arm, questions, RESULTS)
    reps = 1 if arm in UNCENSORED_ARMS and scale in {"8000", "19829"} else REPS
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if arm == "B":
        numeric_scale: int | str = scale if scale == "S0" else int(scale)
        filescan_search(
            numeric_scale,
            questions[0]["question"],
            question_id=questions[0]["question_id"],
        )
        for rep in range(1, reps + 1):
            for question in questions:
                started = time.perf_counter()
                hits = filescan_search(
                    numeric_scale,
                    question["question"],
                    question_id=question["question_id"],
                )
                latency_ms = (time.perf_counter() - started) * 1000
                persisted = [_persisted_hit(hit) for hit in hits]
                row = {
                    "arm": arm,
                    "cov_chunk": None,
                    "cov_fts": None,
                    "cov_summary": None,
                    "degraded": False,
                    "degraded_reason": None,
                    "delivered_hits": persisted,
                    "empty": int(not hits),
                    "latency_ms": latency_ms,
                    "question_id": question["question_id"],
                    "rep": rep,
                    "rows_returned": len(hits),
                    "scale": scale,
                    **session_gold_metrics(question, persisted),
                    **answer_turn_delivery_metrics(
                        question,
                        persisted,
                        load_dates(scale, question["question_id"]),
                        CORPUS / "union.jsonl",
                    ),
                }
                rows.append(row)
            print(f"  rep {rep} done", flush=True)
    else:
        bridge = _probe_bridge_class()()
        bridge.search_embeddings = arm != "A1"
        bridge.query_timeout_s = (
            NO_DEADLINE_S if arm in UNCENSORED_ARMS else None
        )
        if bridge.search_embeddings:
            bridge.initialize({})
        else:
            bridge.embedder = None

        def query_one(question: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
            tag = (
                tag_for_s0(question["question_id"])
                if scale == "S0"
                else tag_for_scale(int(scale))
            )
            sent = (
                fts_safe(question["question"])
                if arm in {"A3", "A3u"}
                else question["question"]
            )
            started = time.perf_counter()
            hits, metadata = bridge.search_instrumented(
                {"containerTag": tag, "limit": LIMIT, "query": sent}
            )
            return hits, metadata, (time.perf_counter() - started) * 1000

        query_one(questions[0])
        for rep in range(1, reps + 1):
            for question in questions:
                hits, metadata, latency_ms = query_one(question)
                provenance = metadata.get("provenance") or {}
                coverage = provenance.get("coverage") or {}
                persisted = [_persisted_hit(hit) for hit in hits]
                sessions = {
                    hit["session_id"]
                    for hit in persisted
                    if hit["session_id"] is not None
                }
                row = {
                    "arm": arm,
                    "cov_chunk": coverage.get("chunk"),
                    "cov_fts": coverage.get("fts"),
                    "cov_summary": coverage.get("summary"),
                    "degraded": bool(metadata.get("degraded")),
                    "degraded_reason": metadata.get("degraded_reason"),
                    "delivered_hits": persisted,
                    "empty": int(not hits),
                    "latency_ms": latency_ms,
                    "question_id": question["question_id"],
                    "rep": rep,
                    "rows_returned": len(hits),
                    "scale": scale,
                    "sessions_returned": len(sessions),
                    **session_gold_metrics(question, persisted),
                    **answer_turn_delivery_metrics(
                        question,
                        persisted,
                        load_dates(scale, question["question_id"]),
                        CORPUS / "union.jsonl",
                    ),
                }
                rows.append(row)
            rep_rows = [row for row in rows if row["rep"] == rep]
            print(
                f"  rep {rep} done "
                f"({statistics.mean(row['latency_ms'] for row in rep_rows):.0f}ms mean)",
                flush=True,
            )

    summary = summarise_rows(rows)
    summary["probe_questions"] = probe_pin
    query_path = RESULTS / f"query-{arm}-{scale}.jsonl"
    with query_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (RESULTS / f"summary-{arm}-{scale}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_scale_parser = subparsers.add_parser("ingest-scale")
    ingest_scale_parser.add_argument("scale", type=int, choices=SCALES)

    ingest_s0_parser = subparsers.add_parser("ingest-s0")
    ingest_s0_parser.add_argument(
        "--uncensored-n", type=int, default=DEFAULT_UNCENSORED_N
    )

    materialise_parser = subparsers.add_parser("materialise-files")
    materialise_parser.add_argument(
        "--uncensored-n", type=int, default=DEFAULT_UNCENSORED_N
    )

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument(
        "arm", choices=("A1", "A2", "A3", "A2u", "A3u", "B")
    )
    query_parser.add_argument(
        "scale", choices=("S0", "500", "2000", "8000", "19829")
    )
    query_parser.add_argument(
        "--uncensored-n",
        type=int,
        default=UNCENSORED_N,
        help="u-arm population (default: 50, the full primary set)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest-scale":
        ingest_scale(args.scale)
    elif args.command == "ingest-s0":
        ingest_s0(args.uncensored_n)
    elif args.command == "materialise-files":
        materialise_files(args.uncensored_n)
    else:
        run_query_arm(args.arm, args.scale, args.uncensored_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
