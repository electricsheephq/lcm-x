"""Phase A: isolated, runtime-enumerated live LCM tool matrix."""
from __future__ import annotations
import argparse

if not __debug__:  # pragma: no cover - guarded before anything runs
    raise SystemExit(
        "phase_a_tool_matrix refuses optimized Python (-O/PYTHONOPTIMIZE): "
        "assert statements implement the matrix postconditions"
    )
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from types import ModuleType
SCENARIOS = {
    "lcm_grep", "lcm_recall", "lcm_query_state", "lcm_compute", "lcm_compile_evidence", "lcm_evidence_pack", "lcm_retrieve", "lcm_recent",
    "lcm_load_session", "lcm_describe", "lcm_expand", "lcm_expand_query", "lcm_status", "lcm_inspect", "lcm_doctor",
}
PLANTED = {
    "pem_complete": "GAUNTLETPEMCOMPLETEA1", "pem_truncated": "GAUNTLETPEMTRUNCATEDB2", "encrypted_armor": "GAUNTLETENCRYPTEDC3",
    "json_serialized": "GAUNTLETJSONSERIALIZEDD4", "log_prefixed": "GAUNTLETLOGPREFIXE5", "password": "GAUNTLETPASSWORDF6", "api_key": "GAUNTLETAPIKEYG7",
    "password_pem": "GAUNTLETPASSWORDPEMH8", "passphrase_pem": "GAUNTLETPASSPHRASEPEMI9", "quoted_password_pem": "GAUNTLETQUOTEDPASSWORDPEMJ0", "chunk_path": "GAUNTLETCHUNKPATHK1",
}
_RELEASE_TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$")
@contextmanager
def _scrubbed_environment(worktree: Path):
    original = dict(os.environ)
    preserved = {
        "LCM_EMBEDDING_API_KEY", "LCM_EMBEDDING_API_KEY_ENV",
        "LCM_EMBEDDING_BASE_URL", "LCM_GAUNTLET_CLOUD_MODEL",
        "LCM_GAUNTLET_CLOUD_PROVIDER", "VOYAGE_API_KEY",
    }
    scrubbed = sorted(key for key in original if key.startswith(("LCM_", "HERMES_")) and key not in preserved)
    clean = {key: value for key, value in original.items() if key not in scrubbed}
    clean["HERMES_LCM_REPO"] = str(worktree)
    os.environ.clear()
    os.environ.update(clean)
    try:
        yield scrubbed
    finally:
        os.environ.clear()
        os.environ.update(original)
def _default_hermes_repo(worktree: Path) -> Path | None:
    candidates = (worktree.parent / "hermes-agent", worktree.parent / "hermes")
    return next((path for path in candidates if (path / "agent/context_engine.py").is_file()), None)
def _module_world_snapshot():
    return {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name in ("agent", "hermes_lcm")
        or name.startswith(("agent.", "hermes_lcm."))
    }


def _restore_module_world(snapshot, inserted_path):
    if inserted_path is not None:
        try:
            sys.path.remove(inserted_path)
        except ValueError:
            pass
    for name in list(sys.modules):
        if name in ("agent", "hermes_lcm") or name.startswith(("agent.", "hermes_lcm.")):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _load(worktree: Path, hermes_repo: Path | None = None, *, release: bool = False):
    hermes_repo = hermes_repo.resolve() if hermes_repo else _default_hermes_repo(worktree)
    if release and hermes_repo is None:
        raise RuntimeError(
            "release mode requires --hermes-repo resolving to a Hermes checkout"
        )
    # Everything below mutates the interpreter's module world (sys.path, the
    # agent.* family, the hermes_lcm family). Snapshot it so EVERY exit —
    # exception or the cleanup returned to run() — restores the host process
    # exactly (in-process callers like tests must see zero residue).
    snapshot = _module_world_snapshot()
    inserted_path = None
    if hermes_repo is not None:
        inserted_path = str(hermes_repo)
        sys.path.insert(0, inserted_path)
        for name in list(sys.modules):
            if name == "agent" or name.startswith("agent."):
                del sys.modules[name]
    try:
        return _load_inner(worktree, hermes_repo, release, snapshot, inserted_path)
    except BaseException:
        _restore_module_world(snapshot, inserted_path)
        raise


def _load_inner(worktree, hermes_repo, release, snapshot, inserted_path):
    try:
        context = importlib.import_module("agent.context_engine")
    except ModuleNotFoundError as exc:
        if exc.name not in {"agent", "agent.context_engine"}:
            raise
        if release:
            raise RuntimeError(f"agent.context_engine import failed: {exc}; pass --hermes-repo") from exc
        agent, context = ModuleType("agent"), ModuleType("agent.context_engine")
        context.ContextEngine = type("ContextEngine", (), {"get_status": lambda self: {}})
        context.__phase_a_stub__ = True
        sys.modules["agent"], sys.modules["agent.context_engine"] = agent, context
        engine_surface = "engine=stubbed-context-base (NOT the hermes surface)"
        hermes_identity = None
    else:
        module_file = getattr(context, "__file__", None)
        if not module_file:
            raise RuntimeError("agent.context_engine has no module __file__")
        source = Path(str(module_file)).resolve()
        if release and hermes_repo is not None and hermes_repo not in source.parents:
            raise RuntimeError(f"agent.context_engine imported from {source}, not --hermes-repo {hermes_repo}")
        context_engine = getattr(context, "ContextEngine", None)
        markers = {
            "select_context (assemble marker)": getattr(context_engine, "select_context", None),
            "on_turn_complete (ingest marker)": getattr(context_engine, "on_turn_complete", None),
        }
        missing_markers = [name for name, value in markers.items() if not callable(value)]
        if release and (not isinstance(context_engine, type) or missing_markers):
            found = type(context_engine).__name__ if context_engine is not None else "missing"
            detail = ", ".join(missing_markers) or "ContextEngine is not a class"
            raise RuntimeError(
                f"Hermes ContextEngine surface invalid at {source}: found {found}; missing {detail}"
            )
        engine_surface = f"engine=hermes-context-engine ({source})"
        hermes_identity = {
            "module_path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    for name in list(sys.modules):
        if name == "hermes_lcm" or name.startswith("hermes_lcm."):
            del sys.modules[name]
    package = importlib.util.module_from_spec(importlib.util.spec_from_file_location("hermes_lcm", worktree / "__init__.py", submodule_search_locations=[str(worktree)]))
    sys.modules["hermes_lcm"] = package
    names = ("assertion_store", "command", "config", "dag", "embedding_provider", "engine", "ingest_protection", "tools")
    modules = {name: importlib.import_module(f"hermes_lcm.{name}") for name in names}

    def cleanup():
        _restore_module_world(snapshot, inserted_path)

    return modules, engine_surface, hermes_identity, cleanup
def _engine(
    mod,
    state: Path,
    posture: str,
    *,
    patterns: bool,
    embedding_privacy: bool | None = None,
):
    provider = "fastembed" if posture == "local" else os.getenv(
        "LCM_GAUNTLET_CLOUD_PROVIDER", "voyage"
    ).strip().lower()
    model = (
        "BAAI/bge-small-en-v1.5" if posture == "local"
        else os.getenv("LCM_GAUNTLET_CLOUD_MODEL", "voyage-3.5-lite").strip()
    )
    config = mod["config"].LCMConfig(
        database_path=str(state / "lcm.db"), embeddings_enabled=True,
        embedding_provider=provider, embedding_model=model,
        sensitive_patterns_enabled=patterns,
        embedding_privacy_enabled=embedding_privacy, assertions_enabled=True,
        adaptive_retrieval_enabled=True, proactive_recall_enabled=True,
    )
    return mod["engine"].LCMEngine(config=config, hermes_home=str(state / "home"))
def _seed(mod, engine):
    corpus = {
        "gauntlet-a": [
            "The taxi costs $60. The train costs $20.",
            "The Atlas decision is approved for the release.",
        ],
        "gauntlet-b": ["Orchid rendezvous notes belong to the second session."],
        "gauntlet-c": ["Constellation anchor proves the live gauntlet corpus."],
    }
    context = {}
    for session, leading in corpus.items():
        messages = [{"role": "user", "content": text} for text in leading]
        messages += [
            {"role": "assistant" if index % 2 else "user",
             "content": f"Deterministic {session} corpus message {index}."}
            for index in range(len(messages), 10)
        ]
        engine.on_session_start(
            session, conversation_id=f"conversation-{session}",
            platform="phase-a", context_length=200_000,
        )
        engine._ingest_messages(messages)
        rows = engine._store.get_session_messages(session)
        source_ids = [int(row["store_id"]) for row in rows]
        node = mod["dag"].SummaryNode(
            session_id=session, depth=0,
            summary=" ".join(str(row["content"]) for row in rows),
            token_count=max(20, sum(int(row["token_estimate"]) for row in rows) // 2),
            source_token_count=sum(int(row["token_estimate"]) for row in rows),
            source_ids=source_ids, source_type="messages", created_at=time.time(),
            earliest_at=time.time(), latest_at=time.time(), expand_hint="Phase A corpus",
        )
        context[f"node:{session}"] = engine._dag.add_node(node)
    rows = engine._store.search("taxi", session_id="gauntlet-a")
    context["finance"] = rows[0]
    atlas = engine._store.search("Atlas", session_id="gauntlet-a")[0]
    context["atlas"] = atlas
    snapshot = engine._assertions.snapshot_source(int(atlas["store_id"]))
    quote = "Atlas decision is approved"
    start = snapshot.content.index(quote)
    candidate = mod["assertion_store"].AssertionCandidate(
        start, start + len(quote), "project:atlas", "release.decision",
        "approved", "approved", "status",
    )
    engine._assertions.publish_source(snapshot, [candidate])
    return context
def _payload(engine, tool, args):
    value = json.loads(engine.handle_tool_call(tool, args))
    if isinstance(value, dict) and value.get("error"):
        raise AssertionError(value["error"])
    return value
def _exact(row, quote=None, **facets):
    content = str(row["content"])
    quote = quote or content
    start = content.index(quote)
    return {
        "exact_ref": f"lcm:{row['store_id']}:{start}-{start + len(quote)}",
        "quote": quote, **facets,
    }
def _scenario(engine, tool, ctx, *, patterns, posture):
    finance, atlas = ctx["finance"], ctx["atlas"]
    finance_text = str(finance["content"])
    taxi, train = "taxi costs $60", "train costs $20"
    operands = [
        {**_exact(finance, text, value=value, unit="USD", key=key, label=key),
         "store_id": finance["store_id"],
         "span_start": finance_text.index(text),
         "span_end": finance_text.index(text) + len(text)}
        for text, value, key in ((taxi, 60, "taxi"), (train, 20, "train"))
    ]
    if tool == "lcm_grep":
        assert "Constellation anchor" in json.dumps(_payload(engine, tool, {"query": "Constellation anchor"})["results"])
    elif tool == "lcm_recall":
        assert "Constellation anchor" in json.dumps(_payload(engine, tool, {"query": "Constellation anchor"})["hits"]).replace(">>>", "").replace("<<<", "")
    elif tool == "lcm_query_state":
        assert "approved" in json.dumps(_payload(engine, tool, {"subject_key": "project:atlas"})["assertions"])
    elif tool == "lcm_compute":
        value = _payload(engine, tool, {"question": "What is the difference between the taxi and train fares?", "operands": operands})
        assert value["status"] == "computed" and value["trace"]["result"] == "$40" and value["trace"]["result_value"] == 40 and value["answer"].startswith("$40 ")
    elif tool == "lcm_evidence_pack":
        value = _payload(engine, tool, {"question": "What is the difference between the taxi and train fares?", "baseline_refs": operands})
        assert value["status"] == "computed" and value["computation"]["result"] == "$40"
    elif tool == "lcm_compile_evidence":
        ref = _exact(atlas)
        proposal = {"version": "evidence-selector-v1", "selections": [{"claim_id": "atlas", "facet": "decision", **ref}], "missing_facets": []}
        value = _payload(engine, tool, {"question": "What was the Atlas decision?", "baseline_refs": [ref], "proposal": proposal})
        assert "Atlas decision is approved" in json.dumps(value["evidence"])
    elif tool == "lcm_retrieve":
        started = _payload(engine, tool, {"action": "start", "question": "What did Atlas decide?", "identity": {"intent_type": "release decision", "operation": "evidence_only"}, "requirements": [{"slot_id": "decision", "description": "Atlas decision"}]})
        found = _payload(engine, tool, {"action": "search", "retrieval_id": started["retrieval_id"], "missing_slot": "decision", "tool": "lcm_expand", "tool_args": {"store_id": atlas["store_id"]}})
        assert "Atlas decision is approved" in json.dumps(found["evidence"])
    elif tool == "lcm_recent":
        assert "gauntlet" in json.dumps(_payload(engine, tool, {"period": "today"})["sections"]).lower()
    elif tool == "lcm_load_session":
        value = _payload(engine, tool, {"session_id": "gauntlet-a"})
        assert value["total_messages"] == value["returned_messages"] == 10 and "The Atlas decision is approved" in json.dumps(value["messages"])
    elif tool == "lcm_describe":
        value = _payload(engine, tool, {"node_id": ctx["node:gauntlet-c"]})
        assert (value["node_id"], value["source_type"], value["num_sources"], value["expand_hint"]) == (ctx["node:gauntlet-c"], "messages", 10, "Phase A corpus")
    elif tool == "lcm_expand":
        assert "Constellation anchor" in json.dumps(_payload(engine, tool, {"node_id": ctx["node:gauntlet-c"]}))
    elif tool == "lcm_expand_query":
        assert "Constellation anchor" in json.dumps(_payload(engine, tool, {"prompt": "Show the anchor", "node_ids": [ctx["node:gauntlet-c"]], "output": "evidence"})["evidence"])
    elif tool == "lcm_status":
        status = _payload(engine, tool, {})
        # The cloud-default posture's privacy battery ingests 10 extra rows before
        # the matrix runs, so totals are posture-aware exact values.
        expected_total = 40 if posture == "cloud-default" else 30
        assert (
            status["store"]["messages"] == 10
            and status["lifecycle_fragmentation"]["messages_total"] == expected_total
            and status["ingest_protection"]["enabled"] is patterns
            and status["proactive_recall"]["privacy_policy_errors"] == 0
        )
    elif tool == "lcm_inspect":
        inspected = _payload(engine, tool, {})
        assert inspected["read_only"] is True and inspected["session_id"] == "gauntlet-c" and inspected["messages"]["total"] == inspected["messages"]["fresh_tail"]["returned"] == 10
    elif tool == "lcm_doctor":
        assert (value := _payload(engine, tool, {}))["overall"] == "healthy" and all(check["status"] == "pass" for check in value["checks"])
    else:
        raise AssertionError(f"no scenario implementation for {tool}")
    return "exact named postcondition passed"
class _Capture:
    def __init__(self, provider, outbound):
        self._provider, self._outbound = provider, outbound
    def __getattr__(self, name):
        return getattr(self._provider, name)
    def embed_query(self, text):
        self._outbound.append(str(text))
        return self._provider.embed_query(text)
    def embed_query_interactive(self, text, *, timeout):
        self._outbound.append(str(text))
        return self._provider.embed_query_interactive(text, timeout=timeout)
    def embed_documents(self, texts):
        self._outbound.extend(map(str, texts))
        return self._provider.embed_documents(texts)
    def embed_document_batches(self, texts, *, before_dispatch=None):
        self._outbound.extend(map(str, texts))
        yield from self._provider.embed_document_batches(texts, before_dispatch=before_dispatch)
    def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
        self._outbound.extend(str(text) for group in groups for _index, text in group)
        yield from self._provider.embed_chunk_group_batches(groups, before_dispatch=before_dispatch)
def _privacy_fixtures():
    long_chunk = " ".join(f"Chunk safety sentence {index}." for index in range(180))
    return [
        {"kind": "standard", "content": f"privacy fixture complete -----BEGIN PRIVATE KEY-----\n{PLANTED['pem_complete']}\n-----END PRIVATE KEY-----", "secrets": [PLANTED["pem_complete"]]},
        {"kind": "standard", "content": f"privacy fixture truncated -----BEGIN PRIVATE KEY-----\n{PLANTED['pem_truncated']}", "secrets": [PLANTED["pem_truncated"]]},
        {"kind": "standard", "content": f"privacy fixture encrypted -----BEGIN ENCRYPTED PRIVATE KEY-----\n{PLANTED['encrypted_armor']}\n-----END ENCRYPTED PRIVATE KEY-----", "secrets": [PLANTED["encrypted_armor"]]},
        {"kind": "standard", "content": json.dumps({"private_key": f"-----BEGIN PRIVATE KEY-----\n{PLANTED['json_serialized']}\n-----END PRIVATE KEY-----"}), "secrets": [PLANTED["json_serialized"]]},
        {"kind": "standard", "content": f"ERROR credential=-----BEGIN PRIVATE KEY-----\n{PLANTED['log_prefixed']}\n-----END PRIVATE KEY-----", "secrets": [PLANTED["log_prefixed"]]},
        {"kind": "standard", "content": f"password={PLANTED['password']} api_key={PLANTED['api_key']}", "secrets": [PLANTED["password"], PLANTED["api_key"]]},
        {"kind": "365", "content": f"password: -----BEGIN PRIVATE KEY-----\n{PLANTED['password_pem']}\n-----END PRIVATE KEY-----", "secrets": [PLANTED["password_pem"]]},
        {"kind": "365", "content": f"passphrase=-----BEGIN PRIVATE KEY-----\n{PLANTED['passphrase_pem']}\n-----END PRIVATE KEY-----", "secrets": [PLANTED["passphrase_pem"]]},
        {"kind": "365", "content": f"password=\"-----BEGIN PRIVATE KEY-----\n{PLANTED['quoted_password_pem']}\n-----END PRIVATE KEY-----\"", "secrets": [PLANTED["quoted_password_pem"]]},
        {"kind": "chunk", "content": f"{long_chunk} api_key={PLANTED['chunk_path']}", "secrets": [PLANTED["chunk_path"]]},
    ]
def _privacy_corpus(mod, engine):
    fixtures = _privacy_fixtures()
    messages = [{"role": "user", "content": item["content"]} for item in fixtures]
    engine.on_session_start("gauntlet-secrets", conversation_id="conversation-secrets", platform="phase-a", context_length=200_000)
    engine._ingest_messages(messages)
    rows = engine._store.get_session_messages("gauntlet-secrets")
    assert len(rows) == len(fixtures), "privacy ingest did not preserve every planted turn"
    summary = " ".join(str(row["content"]) for row in rows)
    node = mod["dag"].SummaryNode(session_id="gauntlet-secrets", depth=0, summary=summary, token_count=60, source_token_count=120, source_ids=[row["store_id"] for row in rows], source_type="messages", created_at=time.time(), earliest_at=time.time(), latest_at=time.time(), expand_hint="Phase A privacy corpus")
    engine._dag.add_node(node)
    return fixtures, rows, summary


def _planted_secret(mod, engine, outbound, *, expect_365_fixed):
    fixtures, rows, _summary = _privacy_corpus(mod, engine)
    known = []
    for index, (fixture, row) in enumerate(zip(fixtures, rows)):
        durable = str(row["content"])
        assert durable == fixture["content"], f"durable row {index} is not byte-identical to the ingested fixture"
        assert "[LCM sensitive redaction:" not in durable, f"durable row {index} was redacted under the lossless default"
    assert "status: ready" in mod["command"].handle_lcm_command("embed warmup", engine)
    assert "status: complete" in mod["command"].handle_lcm_command("embed backfill --apply --limit 50", engine)
    chunk_start = len(outbound)
    assert "status: complete" in mod["command"].handle_lcm_command("embed backfill --corpus chunks --apply --confirm-raw-text --limit 100", engine)
    chunk_outbound = outbound[chunk_start:]
    assert chunk_outbound, "chunk backfill made no embedding dispatch"
    assert any("Chunk safety sentence" in text for text in chunk_outbound)
    revision = mod["ingest_protection"].embedding_privacy_revision(engine._config)
    assert revision != "privacy:off"
    for text in outbound:
        assert not [secret for fixture in fixtures if fixture["kind"] != "365" for secret in fixture["secrets"] if secret in text], "embedding dispatch leaked a non-#365 planted secret"
        if any(secret in text for fixture in fixtures if fixture["kind"] == "365" for secret in fixture["secrets"]):
            known.append("embedding-dispatch")
        else:
            mod["ingest_protection"].validate_embedding_privacy_dispatch([text], engine._config, expected_revision=revision)
    assert outbound
    assert any("[LCM embedding privacy:" in text for text in outbound)
    recall_start = len(outbound)
    recall_query = f"privacy fixture api_key={PLANTED['api_key']}"
    recall = engine.handle_tool_call("lcm_recall", {"query": recall_query})
    recall_dispatches = outbound[recall_start:]
    assert recall_dispatches, "recall made no semantic query dispatch to audit"
    for text in recall_dispatches:
        assert PLANTED["api_key"] not in text, "recall query dispatch leaked the planted secret"
        mod["ingest_protection"].validate_embedding_privacy_dispatch([text], engine._config, expected_revision=revision)
    recall_hits = json.loads(recall)["hits"]
    assert recall_hits
    assert any(secret in recall for fixture in fixtures for secret in fixture["secrets"]), "recall did not return raw lossless durable text"
    if known and expect_365_fixed:
        raise AssertionError("#365 defining composition leaked on: " + ", ".join(sorted(set(known))))
    engine.on_session_start("gauntlet-c", conversation_id="conversation-gauntlet-c", platform="phase-a", context_length=200_000)
    return bool(known)


def _opt_out(mod, engine, outbound):
    _fixtures, _rows, summary = _privacy_corpus(mod, engine)
    revision = mod["ingest_protection"].embedding_privacy_revision(engine._config)
    assert revision == "privacy:off"
    assert "status: ready" in mod["command"].handle_lcm_command("embed warmup", engine)
    assert "status: complete" in mod["command"].handle_lcm_command("embed backfill --apply --limit 50", engine)
    assert outbound, "opt-out made no embedding dispatch"
    assert summary in outbound, "opt-out provider input did not preserve the durable summary byte-for-byte"
    query_start = len(outbound)
    query_text = f"opt-out probe password={PLANTED['password']}"
    hits = json.loads(engine.handle_tool_call("lcm_recall", {"query": query_text}))["hits"]
    assert hits, "opt-out recall returned no hits"
    query_dispatches = outbound[query_start:]
    assert query_dispatches, "opt-out recall made no semantic query dispatch"
    assert any(PLANTED["password"] in text for text in query_dispatches), "opt-out query dispatch did not carry the raw query by explicit choice"
    assert not any("[LCM embedding privacy:" in text for text in outbound)


def _durable_redaction(mod, engine, outbound, *, expect_365_fixed):
    fixtures, rows, _summary = _privacy_corpus(mod, engine)
    known = []
    for index, (fixture, row) in enumerate(zip(fixtures, rows)):
        durable = str(row["content"])
        leaks = [secret for secret in fixture["secrets"] if secret in durable]
        if leaks and fixture["kind"] != "365":
            raise AssertionError(f"durable row {index} leaked its planted secret")
        if leaks:
            known.append(f"durable-row-{index}")
        else:
            assert "[LCM sensitive redaction:" in durable, f"durable row {index} lacks its canonical per-turn placeholder"
    assert "status: ready" in mod["command"].handle_lcm_command("embed warmup", engine)
    assert "status: complete" in mod["command"].handle_lcm_command("embed backfill --apply --limit 50", engine)
    chunk_start = len(outbound)
    assert "status: complete" in mod["command"].handle_lcm_command("embed backfill --corpus chunks --apply --confirm-raw-text --limit 100", engine)
    chunk_outbound = outbound[chunk_start:]
    assert chunk_outbound, "chunk backfill made no embedding dispatch"
    assert any("Chunk safety sentence" in text for text in chunk_outbound)
    revision = mod["ingest_protection"].embedding_privacy_revision(engine._config)
    assert revision != "privacy:off"
    for text in outbound:
        assert not [secret for fixture in fixtures if fixture["kind"] != "365" for secret in fixture["secrets"] if secret in text], "embedding dispatch leaked a non-#365 planted secret"
        if any(secret in text for fixture in fixtures if fixture["kind"] == "365" for secret in fixture["secrets"]):
            known.append("embedding-dispatch")
        else:
            mod["ingest_protection"].validate_embedding_privacy_dispatch([text], engine._config, expected_revision=revision)
    assert outbound
    assert any("[LCM embedding privacy:" in text for text in outbound)
    recall = engine.handle_tool_call("lcm_recall", {"query": "privacy fixture"})
    assert not [secret for fixture in fixtures if fixture["kind"] != "365" for secret in fixture["secrets"] if secret in recall], "recall leaked a non-#365 planted secret"
    if any(secret in recall for fixture in fixtures if fixture["kind"] == "365" for secret in fixture["secrets"]):
        known.append("recall")
    recall_hits = json.loads(recall)["hits"]
    assert recall_hits
    assert any(
        "[LCM sensitive redaction:" in json.dumps(hit)
        or "[LCM embedding privacy:" in json.dumps(hit)
        for hit in recall_hits
    ), "recalled privacy-fixture hits carry no canonical redaction placeholder"
    if known and expect_365_fixed:
        raise AssertionError("#365 defining composition leaked on: " + ", ".join(sorted(set(known))))
    return bool(known)


def _misconfiguration(mod, state: Path):
    engine = _engine(mod, state, "cloud", patterns=False, embedding_privacy=True)
    try:
        engine._config.sensitive_patterns = ["phase_a_unrecognized_pattern"]
        engine._config.sensitive_patterns_source = "phase-a-misconfiguration"
        _seed(mod, engine)
        error = mod["ingest_protection"].EmbeddingPrivacyPolicyError
        try:
            engine.handle_tool_call("lcm_recall", {"query": "Constellation anchor"})
        except error:
            pass
        else:
            raise AssertionError("lcm_recall did not raise EmbeddingPrivacyPolicyError")
        before = engine._proactive_recall_privacy_error_count
        assert engine._build_proactive_recall_message([{"role": "user", "content": "Constellation anchor"}], "system", set()) is None
        status = _payload(engine, "lcm_status", {})
        assert status["proactive_recall"]["privacy_policy_errors"] == before + 1
    finally:
        engine.shutdown()
def _key_gate(mod, provider: str):
    requires_privacy = mod["ingest_protection"].embedding_provider_requires_privacy(
        provider
    )
    if not requires_privacy:
        return (
            "provider credential",
            False,
            f"configuration error: CLOUD posture provider {provider!r} is not in the product cloud set",
        )
    if provider in {"voyage", "voyageai"}:
        return "VOYAGE_API_KEY", bool(os.getenv("VOYAGE_API_KEY", "").strip()), None
    if provider in {"openai", "openai-compatible", "siliconflow"}:
        present = bool(os.getenv("LCM_EMBEDDING_API_KEY", "").strip() or os.getenv("SILICONFLOW_API_KEY", "").strip())
        return "LCM_EMBEDDING_API_KEY or SILICONFLOW_API_KEY", present, None
    return (
        "provider credential",
        False,
        f"configuration error: product cloud provider {provider!r} has no Phase A credential gate",
    )
def _safe_detail(exc):
    detail = f"{type(exc).__name__}: {exc}"
    for value in PLANTED.values():
        detail = detail.replace(value, "<redacted>")
    return detail.replace("\n", " ")[:300]
def _git_identity(worktree: Path):
    def git(*args, check=True):
        return subprocess.run(["git", "-C", str(worktree), *args], check=check, text=True, capture_output=True).stdout.strip()
    tag = git("describe", "--tags", "--exact-match", "HEAD", check=False)
    return {"head": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "tag": tag or "not-an-exact-tag", "dirty": bool(git("status", "--porcelain", "--untracked-files=all"))}
def _hermes_git_identity(hermes_repo: Path | None):
    unavailable = {"head": "unavailable", "dirty": None}
    if hermes_repo is None:
        return {**unavailable, "error": "Hermes checkout not configured"}
    try:
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(hermes_repo), *args], check=True,
                text=True, capture_output=True,
            ).stdout.strip()
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        if top != hermes_repo.resolve():
            return {**unavailable, "error": "not a git checkout root"}
        return {
            "head": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
            "error": None,
        }
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {**unavailable, "error": "not a git checkout"}
def _release_identity_failures(identity, rc_tag):
    failures = []
    if identity.get("dirty"):
        failures.append("release worktree is dirty")
    if not rc_tag:
        failures.append("release mode requires --rc-tag")
    elif _RELEASE_TAG_RE.fullmatch(rc_tag) is None:
        failures.append(
            "release mode requires an RC-suffixed --rc-tag matching "
            "^v[0-9]+\\.[0-9]+\\.[0-9]+-rc[0-9]+$; "
            f"got {rc_tag!r}"
        )
    elif identity.get("tag") != rc_tag:
        failures.append(f"HEAD exact tag is {identity.get('tag')!r}, not requested --rc-tag {rc_tag!r}")
    return failures
def _run_matrix(engine, registered, context, *, patterns, posture, missing):
    records = []
    for tool in registered:
        try:
            if tool in missing:
                raise AssertionError("registered tool has no matrix scenario")
            detail = _scenario(engine, tool, context, patterns=patterns, posture=posture)
            if not detail:
                raise AssertionError("scenario returned no postcondition evidence")
            records.append((posture, tool, "PASS", detail))
        except Exception as exc:
            records.append((posture, tool, "FAIL", _safe_detail(exc)))
    return records
def _verdict(records, batteries, *, release, coverage_failed, preflight):
    skipped = any(row[2] == "SKIP" for row in records) or any(row[1] == "SKIP" for row in batteries)
    blocked = any(row[2] == "BLOCKED" for row in records) or any(row[1] == "BLOCKED" for row in batteries)
    known_leak = any(row[2] == "KNOWN-LEAK-ON-BASE" for row in records) or any(
        row[1] == "KNOWN-LEAK-ON-BASE" for row in batteries
    )
    failed = coverage_failed or any(row[2] == "FAIL" for row in records) or any(row[1] == "FAIL" for row in batteries)
    if preflight or blocked or (release and (skipped or known_leak)):
        return "BLOCKED"
    return "FAIL" if failed else "PASS"
def _cloud_rows(mod, state_root, registered, missing, *, expect_365_fixed):
    provider = os.getenv("LCM_GAUNTLET_CLOUD_PROVIDER", "voyage").strip().lower()
    key_name, present, gate_error = _key_gate(mod, provider)
    if gate_error:
        reason = f"BLOCKED: {gate_error}"
        return [
            ("cloud-default", tool, "BLOCKED", reason) for tool in registered
        ], [
            (name, "BLOCKED", reason) for name in ("planted-secret", "opt-out", "durable-redaction", "misconfiguration")
        ]
    if not present:
        reason = f"SKIP: {key_name} is absent"
        return [("cloud-default", tool, "SKIP", reason) for tool in registered], [(name, "SKIP", reason) for name in ("planted-secret", "opt-out", "durable-redaction", "misconfiguration")]
    outbound, original = [], mod["embedding_provider"].resolve_provider
    def resolve(config, **kwargs):
        value = original(config, **kwargs)
        return _Capture(value, outbound) if value is not None else None
    mod["command"].resolve_provider = resolve
    mod["tools"].resolve_provider = resolve
    batteries = []
    try:
        outbound = []
        cloud = _engine(mod, Path(tempfile.mkdtemp(prefix="cloud-default-", dir=state_root)), "cloud", patterns=False)
        try:
            context = _seed(mod, cloud)
            try:
                known = _planted_secret(mod, cloud, outbound, expect_365_fixed=expect_365_fixed)
                batteries.append(("planted-secret", "KNOWN-LEAK-ON-BASE" if known else "PASS", "lossless durable rows, protected dispatch, revision, raw recall checks passed"))
            except Exception as exc:
                batteries.append(("planted-secret", "FAIL", _safe_detail(exc)))
            try:
                assert "Constellation anchor" in json.dumps(_payload(cloud, "lcm_recall", {"query": "Constellation anchor"})["hits"])
                default_control_error = None
            except Exception as exc:
                default_control_error = exc
            records = _run_matrix(cloud, registered, context, patterns=False, posture="cloud-default", missing=missing)
        finally:
            cloud.shutdown()

        outbound = []
        opt_out = _engine(mod, Path(tempfile.mkdtemp(prefix="cloud-opt-out-", dir=state_root)), "cloud", patterns=False, embedding_privacy=False)
        try:
            _opt_out(mod, opt_out, outbound)
            batteries.append(("opt-out", "PASS", "privacy:off revision and byte-identical raw provider input checks passed"))
        except Exception as exc:
            batteries.append(("opt-out", "FAIL", _safe_detail(exc)))
        finally:
            opt_out.shutdown()

        outbound = []
        redacted = _engine(mod, Path(tempfile.mkdtemp(prefix="cloud-durable-redaction-", dir=state_root)), "cloud", patterns=True)
        try:
            known = _durable_redaction(mod, redacted, outbound, expect_365_fixed=expect_365_fixed)
            batteries.append(("durable-redaction", "KNOWN-LEAK-ON-BASE" if known else "PASS", "opt-in durable, dispatch, revision, recall checks passed"))
        except Exception as exc:
            batteries.append(("durable-redaction", "FAIL", _safe_detail(exc)))
        finally:
            redacted.shutdown()

        try:
            if default_control_error is not None:
                raise AssertionError(f"shipped default recall failed: {_safe_detail(default_control_error)}")
            _misconfiguration(mod, Path(tempfile.mkdtemp(prefix="cloud-misconfiguration-", dir=state_root)))
            batteries.append(("misconfiguration", "PASS", "default recall success plus raise, counter, status, assembly checks passed"))
        except Exception as exc:
            batteries.append(("misconfiguration", "FAIL", _safe_detail(exc)))
    finally:
        mod["command"].resolve_provider = original
        mod["tools"].resolve_provider = original
    return records, batteries
def run(worktree: Path, out: Path, *, release: bool = False, rc_tag: str | None = None, hermes_repo: Path | None = None, expect_365_fixed: bool = True):
    started = time.monotonic()
    worktree, out = worktree.resolve(), out.resolve()
    if not (worktree / "engine.py").is_file():
        raise ValueError(f"not an LCM-X worktree: {worktree}")
    out.mkdir(parents=True, exist_ok=True)
    state_root = out / "state"
    state_root.mkdir(exist_ok=True)
    hermes_repo = hermes_repo.resolve() if hermes_repo else _default_hermes_repo(worktree)
    identity = _git_identity(worktree)
    preflight = _release_identity_failures(identity, rc_tag) if release else []
    hermes_git_identity = _hermes_git_identity(hermes_repo)
    if release and hermes_git_identity["error"]:
        preflight.append(f"Hermes provenance unavailable: {hermes_git_identity['error']}")
    elif release and hermes_git_identity["dirty"]:
        preflight.append("Hermes checkout is dirty")
    records, batteries, registered = [], [], []
    missing, unexpected = [], []
    engine_surface = "engine=not-loaded"
    hermes_identity = None
    with _scrubbed_environment(worktree) as scrubbed:
        mod = None
        world_cleanup = None
        if not preflight:
            try:
                mod, engine_surface, hermes_identity, world_cleanup = _load(
                    worktree, hermes_repo=hermes_repo, release=release
                )
            except Exception as exc:
                preflight.append(_safe_detail(exc))
        try:
            if mod is not None and not preflight:
                local = _engine(mod, Path(tempfile.mkdtemp(prefix="local-", dir=state_root)), "local", patterns=False)
                try:
                    context = _seed(mod, local)
                    registered = sorted({schema["name"] for schema in local.get_tool_schemas() if str(schema.get("name", "")).startswith("lcm_")})
                    missing = sorted(set(registered) - SCENARIOS)
                    unexpected = sorted(SCENARIOS - set(registered))
                    records += _run_matrix(local, registered, context, patterns=False, posture="local", missing=missing)
                finally:
                    local.shutdown()
                cloud_records, cloud_batteries = _cloud_rows(mod, state_root, registered, missing, expect_365_fixed=expect_365_fixed)
                records += cloud_records
                batteries += cloud_batteries
        finally:
            if world_cleanup is not None:
                # Restore the host interpreter's module world (agent.* and
                # hermes_lcm families + the inserted sys.path entry): the
                # runner must leave zero residue in an embedding process.
                world_cleanup()
    final_identity = _git_identity(worktree)
    if release and final_identity != identity:
        preflight.append("release identity changed while the runner was executing")
    identity = final_identity
    final_hermes_git_identity = _hermes_git_identity(hermes_repo)
    if release and final_hermes_git_identity != hermes_git_identity:
        preflight.append("Hermes checkout identity changed while the runner was executing")
    hermes_git_identity = final_hermes_git_identity
    coverage_failed = bool(missing or unexpected)
    verdict = _verdict(records, batteries, release=release, coverage_failed=coverage_failed, preflight=preflight)
    identity_note = f"release bound to requested tag `{rc_tag}`" if release else "DEV RUN — identity unbound"
    hermes_module_path = hermes_identity["module_path"] if hermes_identity else "not-loaded"
    hermes_module_sha256 = hermes_identity["sha256"] if hermes_identity else "not-loaded"
    hermes_dirty = "unavailable" if hermes_git_identity["dirty"] is None else ("dirty" if hermes_git_identity["dirty"] else "clean")
    lines = ["# PHASE-A RECEIPT", "", f"- Mode: `{'release' if release else 'dev'}`", f"- Identity: {identity_note}", f"- Requested release tag: `{rc_tag if rc_tag is not None else 'none'}`", f"- RC tag at HEAD: `{identity['tag']}`", f"- Tree SHA: `{identity['tree']}`", f"- HEAD SHA: `{identity['head']}`", f"- Dirty state: `{'dirty' if identity['dirty'] else 'clean'}`", f"- Engine surface: {engine_surface}", f"- Hermes checkout: `{hermes_repo if hermes_repo else 'not-configured'}`", f"- Hermes checkout HEAD SHA: `{hermes_git_identity['head']}`", f"- Hermes checkout dirty state: `{hermes_dirty}`", f"- Hermes checkout provenance: `{hermes_git_identity['error'] or 'git -C verified'}`", f"- Hermes ContextEngine module __file__: `{hermes_module_path}`", f"- Hermes ContextEngine module sha256: `{hermes_module_sha256}`", f"- Environment scrubbed: `{', '.join(scrubbed) if scrubbed else 'none'}`", f"- #365 expectation: `{'fixed' if expect_365_fixed else 'KNOWN-LEAK-ON-BASE allowed'}`", f"- Command: `{shlex.join([sys.executable, *sys.argv])}`", f"- HERMES_LCM_REPO: `{worktree}`", "- Claim class: `code_green_local`", f"- Runtime registry tools: {len(registered)}", f"- Registry coverage: `{'FAIL' if coverage_failed else 'COMPLETE'}`", f"- Missing matrix rows: `{', '.join(missing) if missing else 'none'}`", f"- Unexpected scenario rows: `{', '.join(unexpected) if unexpected else 'none'}`", "", "## Preflight"]
    lines += [f"- BLOCKED: {detail}" for detail in preflight] or ["- PASS"]
    lines += ["", "## Tool matrix", "", "| Posture | Tool | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {posture} | `{tool}` | {result} | {detail} |" for posture, tool, result, detail in records]
    lines += ["", "## Batteries", "", "| Battery | Result | Detail |", "|---|---|---|"]
    lines += [f"| {name} | {result} | {detail} |" for name, result, detail in batteries]
    lines += ["", f"- Wall time: `{time.monotonic() - started:.3f}s`", f"- Exit verdict: `{verdict}`", "- Proof boundary: this receipt proves only the executed rows against this tree; skipped cloud rows are not release-readiness proof; rerank transform covered by product regressions, not a live battery.", ""]
    receipt = out / "PHASE-A-RECEIPT.md"
    receipt.write_text("\n".join(lines), encoding="utf-8")
    return (0 if verdict == "PASS" else 1), receipt
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--rc-tag")
    parser.add_argument("--hermes-repo", type=Path)
    parser.add_argument("--expect-365-fixed", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    code, receipt = run(args.worktree, args.out, release=args.release, rc_tag=args.rc_tag, hermes_repo=args.hermes_repo, expect_365_fixed=args.expect_365_fixed)
    print(receipt)
    return code
if __name__ == "__main__":
    raise SystemExit(main())
