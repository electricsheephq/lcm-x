"""Opt-in local recall around host-owned compaction; no Hermes lifecycle imports.

The launch namespace is operator-owned. MCP tools cannot choose storage paths,
ingest content, or widen session scope. Capture consumes finalized JSONL only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .config import LCMConfig
from .dag import SummaryDAG
from .store import MessageStore
from .ingest_protection import _is_hermes_persisted_output_marker
from . import tools


PROJECTION = "portable-jsonl-v1"
PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
MAX_LINE = 16 * 1024 * 1024
MAX_CAPTURE_EVENTS = 1000
MAX_WIRE = 1024 * 1024
REF = re.compile(r"^lcmx:([0-9a-f]{64}):lcm:([1-9][0-9]*):([0-9]+)-([0-9]+)$")
LOCAL_REF = re.compile(r"^lcm:([1-9][0-9]*):([0-9]+)-([0-9]+)$")


class PortableError(ValueError):
    """Stable, content-free error codes suitable for host-facing receipts."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _label(value: Any, field: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise PortableError("invalid_" + field)
    if any(ord(c) < 32 for c in value):
        raise PortableError("invalid_" + field)
    return value.strip()


def _integer(value: Any, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise PortableError("invalid_" + field)
    return value


def _root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PortableError("absolute_root_required")
    return path.resolve()


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PortableError("transcript_outside_configured_root")
    return resolved


@dataclass
class RetrievalContext:
    """The real storage contract needed by the production read tools."""
    _store: MessageStore
    _dag: SummaryDAG
    _config: LCMConfig
    _hermes_home: str
    current_session_id: str

    @property
    def current_conversation_id(self) -> str:
        return self.current_session_id

    def close(self) -> None:
        self._dag.close()
        self._store.close()


class PortableRecall:
    def __init__(self, root: str | Path, *, project: str, instance: str, host: str):
        self.root = _root(root)
        self.project = _label(project, "project", 1024)
        self.instance = _label(instance, "instance")
        if host not in {"claude", "codex", "manual"}:
            raise PortableError("unsupported_host")
        self.host = host

    def binding(self, session: str) -> dict[str, str]:
        return {"host": self.host, "project": self.project, "instance": self.instance,
                "session": _label(session, "session"), "projection": PROJECTION}

    def corpus_id(self, session: str) -> str:
        return _hash(self.binding(session))

    def _database(self, session: str) -> Path:
        path = self.root / self.corpus_id(session) / "lcm.db"
        # Corpus paths are generated, never user-provided. Refuse symlink escape.
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise PortableError("corpus_path_escape")
        return path

    def open(self, session: str, *, create: bool = False) -> RetrievalContext:
        binding = self.binding(session)
        path = self._database(session)
        if path.exists():
            # Verify ownership before the normal store bootstrap can migrate it.
            try:
                conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
                try:
                    row = conn.execute("SELECT value FROM metadata WHERE key='portable:binding'").fetchone()
                    actual = json.loads(row[0]) if row else None
                finally:
                    conn.close()
            except (sqlite3.Error, ValueError, OSError) as exc:
                raise PortableError("unrecognized_corpus") from exc
            if actual != binding:
                raise PortableError("corpus_binding_mismatch")
        elif not create:
            raise PortableError("session_not_captured")
        config = LCMConfig(database_path=str(path), embeddings_enabled=False,
                           rerank_enabled=False, recall_reference_strict=True)
        store = MessageStore(path, ingest_protection_config=config, hermes_home=str(path.parent))
        try:
            if store.read_metadata_json("portable:binding") is None:
                store.write_metadata_json("portable:binding", binding)
            dag = SummaryDAG(path)
        except BaseException:
            store.close()
            raise
        return RetrievalContext(store, dag, config, str(path.parent), binding["session"])

    def _decorate(self, value: Any, context: RetrievalContext, corpus: str) -> Any:
        if isinstance(value, list):
            return [self._decorate(item, context, corpus) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: self._decorate(item, context, corpus) for key, item in value.items()}
        exact = result.get("exact_ref")
        if isinstance(exact, str) and LOCAL_REF.fullmatch(exact):
            result["portable_ref"] = f"lcmx:{corpus}:{exact}"
            result["corpus_id"] = corpus
            result["projection_version"] = PROJECTION
            row_id = int(LOCAL_REF.fullmatch(exact).group(1))
            provenance = context._store.read_metadata_json(f"portable:row:{row_id}") or {}
            # The envelope is preserved in storage, not dumped into every recall.
            result["source_identity"] = {key: provenance.get(key) for key in
                                         ("identity", "digest", "version", "part_index")}
        return result

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        schemas = {item["name"]: item["inputSchema"] for item in tool_schemas()}
        if name not in schemas or not isinstance(arguments, dict):
            raise PortableError("unknown_tool_or_arguments")
        schema = schemas[name]
        if set(arguments) - set(schema["properties"]):
            raise PortableError("unsupported_argument")
        if any(key not in arguments for key in schema["required"]):
            raise PortableError("required_argument_missing")
        session = _label(arguments["session"], "session")
        context = self.open(session)
        corpus = self.corpus_id(session)
        try:
            if name == "lcm_portable_status":
                return self.status(session, context)
            if name == "lcm_recall":
                query = _label(arguments["query"], "query", 4096)
                limit = _integer(arguments.get("limit", 10), "limit", 1, 25)
                payload = json.loads(tools.lcm_recall(
                    {"query": query, "limit": limit, "detail": "answer_ready", "include": "verbatim"},
                    engine=context,
                ))
            elif name == "lcm_describe":
                payload = json.loads(tools.lcm_describe({}, engine=context))
            else:
                reference = arguments["reference"]
                match = REF.fullmatch(reference) if isinstance(reference, str) else None
                if match is None or match.group(1) != corpus:
                    raise PortableError("reference_corpus_mismatch")
                row_id, start, end = map(int, match.groups()[1:])
                row = context._store.get(row_id)
                content = str(row.get("content") or "") if row else ""
                if row is None or row["session_id"] != session or not 0 <= start < end <= len(content):
                    raise PortableError("reference_not_found")
                offset = _integer(arguments.get("offset", 0), "offset", 0, end - start)
                budget = _integer(arguments.get("max_tokens", 1000), "max_tokens", 1, 2000)
                payload = json.loads(tools.lcm_expand(
                    {"store_id": row_id, "content_offset": start + offset,
                     "max_tokens": budget, "include_exact_ref": True}, engine=context,
                ))
                if "error" not in payload:
                    page = str(payload.get("content") or "")[:end - start - offset]
                    absolute = start + offset
                    payload.update({"content": page, "content_returned_chars": len(page),
                                    "exact_ref": f"lcm:{row_id}:{absolute}-{absolute + len(page)}",
                                    "reference": reference, "offset": offset,
                                    "next_offset": offset + len(page) if absolute + len(page) < end else None,
                                    "has_more": absolute + len(page) < end,
                                    "content_truncated": absolute + len(page) < end,
                                    "next_content_offset": absolute + len(page) if absolute + len(page) < end else None})
            payload = self._decorate(payload, context, corpus)
            payload.update({"corpus_id": corpus, "session": session,
                            "projection_version": PROJECTION, "evidence_trust": "untrusted"})
            return payload
        finally:
            context.close()

    def status(self, session: str, context: RetrievalContext) -> dict[str, Any]:
        pairing = _tool_pairing(context._store, session)
        return {"corpus_id": self.corpus_id(session), "session": session,
                "message_count": context._store.get_session_count(session),
                "projection_version": PROJECTION, "embeddings": False, "scorer": "baseline",
                "native_context_engine": False, "context_replacement": False,
                "capture": context._store.read_metadata_json("portable:capture_status") or
                {"configured": False, "observed": False, "qualified": False},
                "hooks": context._store.read_metadata_json("portable:hooks") or {},
                "native_host_qualified": False, "tool_pairing": pairing}

    def ingest(self, session: str, *, event_id: str, messages: list[dict[str, Any]],
               generation: str = "manual-v1", envelope: dict[str, Any] | None = None,
               checkpoint_key: str = "", checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
        event_id = _label(event_id, "event_id", 1024)
        generation = _label(generation, "generation")
        if not isinstance(messages, list) or len(messages) > 100:
            raise PortableError("invalid_messages")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "tool", "system", "unknown"}:
                raise PortableError("invalid_message")
            if message.get("role") == "tool" and _is_hermes_persisted_output_marker(message.get("content")):
                raise PortableError("unsupported_host_file_marker")
        metadata = {"binding": self.binding(session), "generation": generation,
                    "event_id": event_id, "envelope": envelope or {}, "projection": PROJECTION}
        identity = _hash({"binding": self.binding(session), "generation": generation, "event_id": event_id})
        # Cursor/location are provenance, not event content. Re-emission of the
        # same host UUID at a later byte position is still the same event.
        digest_envelope = (envelope or {}).get("event") if checkpoint_key else envelope or {}
        digest = _hash({"messages": messages, "envelope": digest_envelope, "projection": PROJECTION})
        context = self.open(session, create=True)
        try:
            result = context._store.append_source_event(
                session, messages, identity=identity, digest=digest, source_metadata=metadata,
                checkpoint_key=checkpoint_key, checkpoint=checkpoint, source="portable:" + self.host,
            )
            return {**result, "corpus_id": self.corpus_id(session), "event_identity": identity}
        finally:
            context.close()

    def capture(self, session: str, *, transcript: str | Path, transcript_root: str | Path,
                max_events: int = MAX_CAPTURE_EVENTS) -> dict[str, Any]:
        session = _label(session, "session")
        max_events = _integer(max_events, "max_events", 1, MAX_CAPTURE_EVENTS)
        root = _root(transcript_root)
        path = _within(Path(transcript).expanduser(), root)
        if self.host == "manual":
            raise PortableError("manual_host_requires_explicit_ingest")
        checkpoint_key = "portable:checkpoint:" + _hash(str(path))
        context = self.open(session, create=True)
        try:
            prior = context._store.read_metadata_json(checkpoint_key) or {}
        finally:
            context.close()
        report = {"configured": True, "observed": True, "qualified": False,
                  "appended": 0, "duplicates": 0, "conflicts": 0, "unsupported": 0,
                  "incomplete": False, "coverage_complete": False, "generation_changed": False}
        started = time.monotonic()
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            first = stream.readline(MAX_LINE + 1)
            if not first.endswith(b"\n") or len(first) > MAX_LINE:
                return {**report, "incomplete": True, "status": "incomplete_header"}
            # Codex session_meta is mandatory; Claude message records carry sessionId.
            if self.host == "codex":
                try:
                    header = json.loads(first)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise PortableError("invalid_session_header") from exc
                if not isinstance(header, dict) or header.get("type") != "session_meta" or not isinstance(header.get("payload"), dict) or header["payload"].get("id") != session:
                    raise PortableError("transcript_session_mismatch")
            prefix = hashlib.sha256(first).hexdigest()
            changed = bool(prior) and (prior.get("inode") != [stat.st_dev, stat.st_ino]
                                      or prior.get("prefix") != prefix or stat.st_size < prior.get("offset", 0))
            # Validate all previously consumed bytes. Same-inode rewrites away
            # from the first line must also start a new preserved generation.
            consumed = hashlib.sha256()
            stream.seek(0)
            remaining = int(prior.get("offset", 0)) if not changed else 0
            while remaining:
                if time.monotonic() - started >= 15:
                    return {**report, "status": "checkpoint_validation_budget_exhausted"}
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    changed = True
                    break
                consumed.update(chunk)
                remaining -= len(chunk)
            if prior and not changed and consumed.hexdigest() != prior.get("consumed_sha256"):
                changed = True
            generation = int(prior.get("generation", 0)) + int(changed)
            report["generation_changed"] = changed
            offset = 0 if changed else int(prior.get("offset", 0))
            line_index = 0 if changed else int(prior.get("line_index", 0))
            if changed:
                consumed = hashlib.sha256()
            unsupported_total = int(prior.get("unsupported_total", 0))
            stream.seek(offset)
            processed = 0
            while processed < max_events and time.monotonic() - started < 15:
                position = stream.tell()
                raw = stream.readline(MAX_LINE + 1)
                if not raw:
                    report["coverage_complete"] = report["unsupported"] == 0
                    report["status"] = "synced" if report["coverage_complete"] else "unsupported_events"
                    break
                if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
                    report.update({"incomplete": True, "status": "incomplete_or_oversized_event"})
                    break
                try:
                    envelope = json.loads(raw)
                    if not isinstance(envelope, dict):
                        raise ValueError("object required")
                    messages, supported = project_event(self.host, session, envelope)
                    if any(message.get("role") == "tool" and _is_hermes_persisted_output_marker(message.get("content"))
                           for message in messages):
                        # Hermes file dereference is not a portable capture
                        # capability. Retain the protected envelope and gap.
                        messages, supported = [], False
                except (ValueError, UnicodeDecodeError) as exc:
                    if isinstance(exc, PortableError):
                        raise
                    envelope, messages, supported = {"unparsed_line": raw.decode("utf-8", errors="replace")}, [], False
                payload = envelope.get("payload")
                event_id = str(envelope.get("uuid") or envelope.get("id") or
                               (payload.get("id") if isinstance(payload, dict) else None) or f"line:{line_index}")
                consumed.update(raw)
                unsupported_total += int(not supported)
                checkpoint = {"inode": [stat.st_dev, stat.st_ino], "prefix": prefix,
                              "generation": generation, "offset": stream.tell(), "line_index": line_index + 1,
                              "consumed_sha256": consumed.hexdigest(), "unsupported_total": unsupported_total}
                source = {"event": envelope, "source": {"file_identity": _hash(str(path)),
                          "byte_offset": position, "line_index": line_index,
                          "source_generation": generation}, "supported_projection": supported}
                result = self.ingest(session, event_id=event_id, messages=messages,
                                     generation=f"jsonl:{_hash(str(path))}:{generation}", envelope=source,
                                     checkpoint_key=checkpoint_key, checkpoint=checkpoint)
                report[{"appended": "appended", "duplicate": "duplicates", "conflict": "conflicts"}[result["status"]]] += len(result["store_ids"]) if result["status"] != "duplicate" else 1
                report["unsupported"] += int(not supported)
                processed += 1
                line_index += 1
            else:
                report["status"] = "capture_budget_exhausted"
        context = self.open(session)
        try:
            # The coverage ledger travels in the same transaction as each
            # event, so a crash before this cosmetic status write loses no gap.
            final = context._store.read_metadata_json(checkpoint_key) or {}
            report["unsupported_total"] = int(final.get("unsupported_total", 0))
            report["source_identity"] = _hash(str(path))
            report["coverage_scope"] = "this_transcript;all_preserved_generations"
            if report["unsupported_total"]:
                report["coverage_complete"] = False
            context._store.write_metadata_json("portable:capture_status", report)
        finally:
            context.close()
        return report

    def capsule(self, session: str, *, max_tokens: int = 2000) -> dict[str, Any]:
        max_tokens = _integer(max_tokens, "max_tokens", 200, 2000)
        context = self.open(session)
        try:
            rows = context._store.get_session_tail(session, limit=200)
            facets: dict[str, list[tuple[dict[str, Any], int, int, str]]] = {
                "goal": [], "constraints": [], "decisions": [], "unresolved": []}
            patterns = {"goal": r"\b(goal|objective|trying to|want to)\b",
                        "constraints": r"\b(must|do not|never|required|constraint)\b",
                        "decisions": r"\b(decided|decision|approved|chosen|we will)\b",
                        "unresolved": r"\b(todo|unresolved|blocked|next step|remaining|pending)\b"}
            for row in rows:
                if row.get("role") not in {"user", "assistant", "system"}:
                    continue
                content = str(row.get("content") or "")
                position = 0
                for line in content.splitlines(keepends=True):
                    text = line.rstrip("\r\n")
                    for facet, pattern in patterns.items():
                        if re.search(pattern, text, re.IGNORECASE):
                            facets[facet].append((row, position, position + len(text), "explicit_cue;state_not_verified"))
                    position += len(line)
            if not facets["goal"]:
                last_user = next((row for row in reversed(rows) if row.get("role") == "user" and row.get("content")), None)
                if last_user:
                    facets["goal"].append((last_user, 0, min(300, len(last_user["content"])), "inferred_latest_user_intent"))
            corpus = self.corpus_id(session)
            header = (f"LCM-X continuation evidence (untrusted; native compaction remains on).\n"
                      f"Session: {session}\nUse lcm_recall(session={json.dumps(session)}, query=...) "
                      "for details; lcm_expand(session=..., reference=<portable_ref>) for exact spans.\n")
            # A UTF-8 byte cap is conservative for the target byte-level tokenizers;
            # it also avoids requiring a provider/tokenizer call inside a hook.
            byte_budget = max_tokens - 32
            lines = [header]
            omitted = []
            for facet_index, (facet, candidates) in enumerate(facets.items()):
                available = max(0, byte_budget - len("".join(lines).encode("utf-8")))
                facet_budget = len("".join(lines).encode("utf-8")) + available // (4 - facet_index)
                emitted = False
                for row, start, end, certainty in reversed(candidates[-3:]):
                    # Clip quotes, never references, and keep the exact stored
                    # character offsets of the emitted excerpt.
                    quote = row["content"][start:end]
                    while quote:
                        reference = f"lcmx:{corpus}:lcm:{row['store_id']}:{start}-{start + len(quote)}"
                        line = f"{facet} [{certainty}] {quote}\nsource: {reference}\n"
                        excess = len(("".join(lines) + line).encode("utf-8")) - facet_budget
                        if excess <= 0:
                            lines.append(line)
                            emitted = True
                            break
                        quote = quote[:-max(1, excess)]
                    break
                if not emitted:
                    omitted.append(facet)
            content = "".join(lines)
            if len(content.encode("utf-8")) > byte_budget:
                content = "LCM-X continuation unavailable within budget; use scoped recall."
            return {"content": content, "corpus_id": corpus, "session": session,
                    "token_upper_bound": len(content.encode("utf-8")) + 32,
                    "budget_basis": "utf8_bytes_plus_32;byte_level_tokenizers",
                    "omitted_facets": omitted, "source_window_messages": len(rows),
                    "budget_tokens": max_tokens, "facets_are_evidence_not_current_state": True,
                    "qualification": "unqualified"}
        finally:
            context.close()

    def hook(self, payload: dict[str, Any], *, transcript_root: str | Path,
             assistance: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (host output, metadata receipt); never return a blocking decision."""
        receipt: dict[str, Any] = {"status": "failed_open", "qualified": False}
        try:
            if not isinstance(payload, dict):
                raise PortableError("invalid_hook_payload")
            event = payload.get("hook_event_name")
            if event not in {"PreCompact", "PostCompact", "SessionStart", "Stop", "SessionEnd"}:
                raise PortableError("unsupported_hook_event")
            session = _label(payload.get("session_id"), "session")
            agent = payload.get("agent_id")
            if agent:
                session += "#agent:" + _label(agent, "agent")
            if Path(self.project).is_absolute() and payload.get("cwd"):
                if Path(payload["cwd"]).resolve() != Path(self.project).resolve():
                    raise PortableError("hook_project_mismatch")
            capture = self.capture(session, transcript=payload.get("transcript_path", ""), transcript_root=transcript_root)
            receipt = {"status": "observed", "hook_event": event, "capture": capture, "qualified": False}
            output: dict[str, Any] = {}
            if event == "SessionStart" and assistance:
                capsule = self.capsule(session)
                output = {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                                "additionalContext": capsule["content"]}}
                receipt["capsule_emitted"] = True
            context = self.open(session)
            try:
                observed = context._store.read_metadata_json("portable:hooks") or {}
                observed[event] = {"configured": True, "observed": True, "qualified": False,
                                   "capsule_emitted": bool(receipt.get("capsule_emitted")),
                                   "injection_confirmed": False}
                context._store.write_metadata_json("portable:hooks", observed)
            finally:
                context.close()
            return output, receipt
        except Exception as exc:  # fail-open includes store/capture/provider-free failures
            receipt["error_code"] = str(exc) if isinstance(exc, PortableError) else type(exc).__name__
            return {}, receipt


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if all(isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"} for part in value):
            return "\n".join(str(part.get("text") or "") for part in value)
    return _json(value)


def project_event(host: str, session: str, event: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Project only supported finalized envelopes; preserve others in receipts."""
    timestamp = event.get("timestamp")
    if host == "claude":
        declared = event.get("sessionId") or event.get("session_id")
        base_session, _, agent = session.partition("#agent:")
        if declared and declared != base_session:
            raise PortableError("transcript_session_mismatch")
        if event.get("agentId") and event["agentId"] != agent:
            raise PortableError("transcript_branch_mismatch")
        message = event.get("message")
        if event.get("type") not in {"user", "assistant"} or not isinstance(message, dict):
            return [], False
        if not declared:
            raise PortableError("message_session_identity_missing")
        if agent and event.get("agentId") != agent:
            raise PortableError("message_branch_identity_missing")
        if event.get("type") == "assistant" and message.get("stop_reason") is None and event.get("is_partial"):
            return [], False
        content = message.get("content", "")
        role = message.get("role", event["type"])
        parts = content if isinstance(content, list) else []
        output, remaining, calls = [], [], []
        for part in parts:
            if not isinstance(part, dict):
                remaining.append(part)
            elif part.get("type") == "tool_result" and role == "user":
                output.append({"role": "tool", "content": _content(part.get("content", "")),
                               "tool_call_id": part.get("tool_use_id"), "timestamp": timestamp})
            else:
                remaining.append(part)
                if part.get("type") == "tool_use" and role == "assistant":
                    calls.append({"id": part.get("id"), "type": "function",
                                  "function": {"name": part.get("name"), "arguments": _json(part.get("input", {}))}})
        if not parts or remaining:
            output.insert(0, {"role": role, "content": _content(remaining if parts else content),
                              "tool_calls": calls or None, "timestamp": timestamp})
        return output, True
    if host != "codex":
        raise PortableError("unsupported_host")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return [], False
    if event.get("type") == "session_meta":
        if payload.get("id") != session:
            raise PortableError("transcript_session_mismatch")
        return [], True
    if event.get("type") == "turn_context":
        return [], True
    if event.get("type") != "response_item":
        return [], False
    kind = payload.get("type")
    if kind == "message" and payload.get("role") in {"user", "assistant", "system", "developer"}:
        role = "system" if payload["role"] == "developer" else payload["role"]
        return [{"role": role, "content": _content(payload.get("content", "")), "timestamp": timestamp}], True
    if kind in {"function_call", "custom_tool_call"}:
        return [{"role": "assistant", "content": "", "timestamp": timestamp,
                 "tool_calls": [{"id": payload.get("call_id"), "type": "function", "function": {
                     "name": payload.get("name"), "arguments": payload.get("arguments", payload.get("input", ""))}}]}], True
    if kind in {"function_call_output", "custom_tool_call_output"}:
        return [{"role": "tool", "content": _content(payload.get("output", "")),
                 "tool_call_id": payload.get("call_id"), "timestamp": timestamp}], True
    return [], False


def _tool_pairing(store: MessageStore, session: str) -> dict[str, int]:
    calls: dict[str, int] = {}
    call_count = result_count = unmatched = reused = 0
    cursor = 0
    while True:
        rows = store.load_session_page(session, after_store_id=cursor, limit=500)
        if not rows:
            break
        for row in rows:
            for call in row.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    key = str(call["id"])
                    reused += int(key in calls)
                    calls[key] = calls.get(key, 0) + 1
                    call_count += 1
            if row.get("role") == "tool":
                key = str(row.get("tool_call_id") or "")
                result_count += 1
                if not key or calls.get(key, 0) == 0:
                    unmatched += 1
                else:
                    calls[key] -= 1
        cursor = rows[-1]["store_id"]
    return {"call_ids": len(calls), "call_rows": call_count, "result_rows": result_count,
            "unmatched_results": unmatched, "unresolved_calls": sum(calls.values()),
            "reused_call_ids": reused}


def tool_schemas() -> list[dict[str, Any]]:
    session = {"type": "string", "minLength": 1, "description": "Explicit captured session in this launch namespace."}
    definitions = [
        ("lcm_recall", "Read cited, untrusted evidence in one captured session; maximum 25 hits.",
         {"session": session, "query": {"type": "string", "minLength": 1},
          "limit": {"type": "integer", "minimum": 1, "maximum": 25}}, ["session", "query"]),
        ("lcm_describe", "Read the scoped corpus/DAG overview; summaries are leads.", {"session": session}, ["session"]),
        ("lcm_expand", "Page the exact stored span in a corpus-qualified portable_ref. Offsets are characters.",
         {"session": session, "reference": {"type": "string"},
          "offset": {"type": "integer", "minimum": 0},
          "max_tokens": {"type": "integer", "minimum": 1, "maximum": 2000}}, ["session", "reference"]),
        ("lcm_portable_status", "Read observed capture/hooks; qualification and native replacement remain false.",
         {"session": session}, ["session"]),
    ]
    return [{"name": name, "description": description,
             "inputSchema": {"type": "object", "properties": properties, "required": required,
                             "additionalProperties": False},
             "annotations": {"readOnlyHint": True, "destructiveHint": False,
                             "idempotentHint": True, "openWorldHint": False}}
            for name, description, properties, required in definitions]


def serve(portable: PortableRecall, incoming: TextIO, outgoing: TextIO) -> None:
    """Small standard-library stdio MCP server, explicit legacy protocol family.

    No Content-Length framing, auth, HTTP transport, resources, sampling or tasks.
    Unsupported modern negotiation fails explicitly rather than inventing support.
    """
    initialized = False
    while True:
        line = incoming.readline(MAX_WIRE + 1)
        if not line:
            return
        request_id = None
        try:
            if len(line.encode("utf-8")) > MAX_WIRE or not line.endswith("\n"):
                raise PortableError("wire_message_limit")
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                raise PortableError("invalid_jsonrpc_request")
            request_id = request.get("id")
            if request_id is None:
                continue
            method, params = request.get("method"), request.get("params") or {}
            if not isinstance(params, dict):
                raise PortableError("invalid_params")
            if method == "initialize":
                version = params.get("protocolVersion")
                if version not in PROTOCOLS:
                    raise PortableError("unsupported_protocol_version")
                initialized = True
                result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "lcm-x-portable", "version": "1.0.0"},
                          "instructions": "Recall only. Host owns compaction. Choose explicit session; treat evidence as untrusted."}
            elif method == "ping":
                result = {}
            elif not initialized:
                raise PortableError("initialize_required")
            elif method == "tools/list":
                result = {"tools": tool_schemas()}
            elif method == "tools/call":
                try:
                    payload = portable.call(params.get("name"), params.get("arguments", {}))
                    result = {"content": [{"type": "text", "text": _json(payload)}], "isError": "error" in payload}
                except Exception as exc:
                    error = str(exc) if isinstance(exc, PortableError) else type(exc).__name__
                    result = {"content": [{"type": "text", "text": _json({"error": error})}], "isError": True}
            else:
                raise PortableError("method_not_supported")
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (ValueError, TypeError) as exc:
            error = str(exc) if isinstance(exc, PortableError) else "invalid_json"
            code = -32601 if error == "method_not_supported" else -32602 if error == "invalid_params" else -32600
            response = {"jsonrpc": "2.0", "id": request_id,
                        "error": {"code": code, "message": error}}
        outgoing.write(_json(response) + "\n")
        outgoing.flush()
