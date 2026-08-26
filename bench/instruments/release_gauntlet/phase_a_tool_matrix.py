"""Phase A: isolated, runtime-enumerated live LCM tool matrix."""
from __future__ import annotations
import argparse, importlib, importlib.util, json, os, shlex
from pathlib import Path
import subprocess, sys, tempfile, time
from types import ModuleType
SCENARIOS = {
    "lcm_grep", "lcm_recall", "lcm_query_state", "lcm_compute", "lcm_compile_evidence", "lcm_evidence_pack", "lcm_retrieve", "lcm_recent",
    "lcm_load_session", "lcm_describe", "lcm_expand", "lcm_expand_query", "lcm_status", "lcm_inspect", "lcm_doctor",
}
PLANTED = {
    "pem_complete": "GAUNTLETPEMCOMPLETEA1", "pem_truncated": "GAUNTLETPEMTRUNCATEDB2", "encrypted_armor": "GAUNTLETENCRYPTEDC3",
    "json_serialized": "GAUNTLETJSONSERIALIZEDD4", "log_prefixed": "GAUNTLETLOGPREFIXE5", "password": "GAUNTLETPASSWORDF6", "api_key": "GAUNTLETAPIKEYG7",
}
def _load(worktree: Path):
    try: importlib.import_module("agent.context_engine")
    except ModuleNotFoundError as exc:
        if exc.name not in {"agent", "agent.context_engine"}: raise
        agent, context = ModuleType("agent"), ModuleType("agent.context_engine")
        context.ContextEngine = type("ContextEngine", (), {"get_status": lambda self: {}})
        sys.modules["agent"], sys.modules["agent.context_engine"] = agent, context
    for name in list(sys.modules):
        if name == "hermes_lcm" or name.startswith("hermes_lcm."):
            del sys.modules[name]
    package = importlib.util.module_from_spec(importlib.util.spec_from_file_location("hermes_lcm", worktree / "__init__.py", submodule_search_locations=[str(worktree)]))
    sys.modules["hermes_lcm"] = package
    return {name: importlib.import_module(f"hermes_lcm.{name}") for name in (
        "assertion_store", "command", "config", "dag", "embedding_provider",
        "engine", "ingest_protection", "tools",
    )}
def _engine(mod, state: Path, posture: str, *, patterns: bool):
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
        sensitive_patterns_enabled=patterns, assertions_enabled=True,
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
def _scenario(engine, tool, ctx, *, patterns):
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
        assert _payload(engine, tool, {"query": "Constellation anchor"})["results"]
    elif tool == "lcm_recall":
        assert _payload(engine, tool, {"query": "Constellation anchor"})["hits"]
    elif tool == "lcm_query_state":
        assert _payload(engine, tool, {"subject_key": "project:atlas"})["assertions"]
    elif tool == "lcm_compute":
        assert _payload(engine, tool, {"question": "What is the difference between the taxi and train fares?", "operands": operands})["status"] == "computed"
    elif tool == "lcm_evidence_pack":
        assert _payload(engine, tool, {"question": "What is the difference between the taxi and train fares?", "baseline_refs": operands})["status"] == "computed"
    elif tool == "lcm_compile_evidence":
        ref = _exact(atlas)
        proposal = {"version": "evidence-selector-v1", "selections": [{"claim_id": "atlas", "facet": "decision", **ref}], "missing_facets": []}
        assert _payload(engine, tool, {"question": "What was the Atlas decision?", "baseline_refs": [ref], "proposal": proposal})["evidence"]
    elif tool == "lcm_retrieve":
        started = _payload(engine, tool, {"action": "start", "question": "What did Atlas decide?", "identity": {"intent_type": "release decision", "operation": "evidence_only"}, "requirements": [{"slot_id": "decision", "description": "Atlas decision"}]})
        found = _payload(engine, tool, {"action": "search", "retrieval_id": started["retrieval_id"], "missing_slot": "decision", "tool": "lcm_expand", "tool_args": {"store_id": atlas["store_id"]}})
        assert found["evidence"]
    elif tool == "lcm_recent":
        assert _payload(engine, tool, {"period": "today"})["sections"]
    elif tool == "lcm_load_session":
        assert len(_payload(engine, tool, {"session_id": "gauntlet-a"})["messages"]) == 10
    elif tool == "lcm_describe":
        assert _payload(engine, tool, {"node_id": ctx["node:gauntlet-c"]})["node_id"] == ctx["node:gauntlet-c"]
    elif tool == "lcm_expand":
        assert "Constellation anchor" in json.dumps(_payload(engine, tool, {"node_id": ctx["node:gauntlet-c"]}))
    elif tool == "lcm_expand_query":
        assert _payload(engine, tool, {"prompt": "Show the anchor", "node_ids": [ctx["node:gauntlet-c"]], "output": "evidence"})["evidence"]
    elif tool == "lcm_status":
        status = _payload(engine, tool, {})
        assert status["store"]["messages"] >= 10 and status["ingest_protection"]["enabled"] is patterns
        assert "privacy_policy_errors" in status["proactive_recall"]
    elif tool == "lcm_inspect":
        inspected = _payload(engine, tool, {}); assert inspected["read_only"] is True and inspected["messages"]
    elif tool == "lcm_doctor":
        assert _payload(engine, tool, {})["overall"] == "healthy"
class _Capture:
    def __init__(self, provider, outbound):
        self._provider, self._outbound = provider, outbound
    def __getattr__(self, name): return getattr(self._provider, name)
    def embed_query(self, text):
        self._outbound.append(str(text)); return self._provider.embed_query(text)
    def embed_query_interactive(self, text, *, timeout):
        self._outbound.append(str(text)); return self._provider.embed_query_interactive(text, timeout=timeout)
    def embed_documents(self, texts):
        self._outbound.extend(map(str, texts)); return self._provider.embed_documents(texts)
    def embed_document_batches(self, texts, *, before_dispatch=None):
        self._outbound.extend(map(str, texts))
        yield from self._provider.embed_document_batches(texts, before_dispatch=before_dispatch)
def _privacy(mod, engine, ctx, outbound):
    messages = [
        {"role": "user", "content": f"privacy fixture complete -----BEGIN PRIVATE KEY-----\n{PLANTED['pem_complete']}\n-----END PRIVATE KEY-----"},
        {"role": "user", "content": f"privacy fixture truncated -----BEGIN PRIVATE KEY-----\n{PLANTED['pem_truncated']}"},
        {"role": "user", "content": f"privacy fixture encrypted -----BEGIN ENCRYPTED PRIVATE KEY-----\n{PLANTED['encrypted_armor']}\n-----END ENCRYPTED PRIVATE KEY-----"},
        {"role": "user", "content": json.dumps({"private_key": f"-----BEGIN PRIVATE KEY-----\n{PLANTED['json_serialized']}\n-----END PRIVATE KEY-----"})},
        {"role": "user", "content": f"ERROR credential=-----BEGIN PRIVATE KEY-----\n{PLANTED['log_prefixed']}\n-----END PRIVATE KEY-----"},
        {"role": "user", "content": f"password={PLANTED['password']} api_key={PLANTED['api_key']}"},
    ]
    engine.on_session_start("gauntlet-secrets", conversation_id="conversation-secrets", platform="phase-a", context_length=200_000)
    engine._ingest_messages(messages)
    rows = engine._store.get_session_messages("gauntlet-secrets")
    durable = json.dumps(rows)
    assert all(secret not in durable for secret in PLANTED.values()) and "[LCM sensitive redaction:" in durable
    node = mod["dag"].SummaryNode(
        session_id="gauntlet-secrets", depth=0,
        summary=" ".join(str(row["content"]) for row in rows), token_count=60,
        source_token_count=120, source_ids=[row["store_id"] for row in rows],
        source_type="messages", created_at=time.time(), earliest_at=time.time(),
        latest_at=time.time(), expand_hint="Phase A privacy corpus",
    )
    engine._dag.add_node(node)
    warmup = mod["command"].handle_lcm_command("embed warmup", engine); assert "status: ready" in warmup
    backfill = mod["command"].handle_lcm_command("embed backfill --apply --limit 50", engine); assert "status: complete" in backfill
    revision = mod["ingest_protection"].embedding_privacy_revision(engine._config)
    for text in outbound:
        mod["ingest_protection"].validate_embedding_privacy_dispatch([text], engine._config, expected_revision=revision)
    assert outbound and any("[LCM embedding privacy:" in text for text in outbound)
    assert all(secret not in text for text in outbound for secret in PLANTED.values())
    recall = engine.handle_tool_call("lcm_recall", {"query": "privacy fixture"})
    assert all(secret not in recall for secret in PLANTED.values())
    assert json.loads(recall)["hits"]
    engine.on_session_start("gauntlet-c", conversation_id="conversation-gauntlet-c", platform="phase-a", context_length=200_000)
def _loud_fail(mod, state: Path):
    engine = _engine(mod, state, "cloud", patterns=False)
    try:
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
def _key_gate(provider: str):
    if provider in {"voyage", "voyageai"}: return "VOYAGE_API_KEY", bool(os.getenv("VOYAGE_API_KEY", "").strip())
    if provider in {"openai", "openai-compatible", "siliconflow"}:
        present = bool(os.getenv("LCM_EMBEDDING_API_KEY", "").strip() or os.getenv("SILICONFLOW_API_KEY", "").strip())
        return "LCM_EMBEDDING_API_KEY or SILICONFLOW_API_KEY", present
    return "provider-specific standard environment", True
def _safe_detail(exc):
    detail = f"{type(exc).__name__}: {exc}"
    for value in PLANTED.values():
        detail = detail.replace(value, "<redacted>")
    return detail.replace("\n", " ")[:300]
def run(worktree: Path, out: Path):
    started = time.monotonic(); worktree, out = worktree.resolve(), out.resolve()
    if not (worktree / "engine.py").is_file():
        raise ValueError(f"not an LCM-X worktree: {worktree}")
    out.mkdir(parents=True, exist_ok=True); state_root = out / "state"; state_root.mkdir(exist_ok=True)
    os.environ["HERMES_LCM_REPO"] = str(worktree)
    mod = _load(worktree)
    records, batteries = [], []
    local_state = Path(tempfile.mkdtemp(prefix="local-", dir=state_root)); local = _engine(mod, local_state, "local", patterns=False)
    try:
        local_ctx = _seed(mod, local)
        registered = sorted({s["name"] for s in local.get_tool_schemas() if str(s.get("name", "")).startswith("lcm_")})
        missing = sorted(set(registered) - SCENARIOS)
        for tool in registered:
            try:
                if tool in missing: raise AssertionError("registered tool has no matrix scenario")
                _scenario(local, tool, local_ctx, patterns=False)
                records.append(("local", tool, "PASS", "real post-condition passed"))
            except Exception as exc:
                records.append(("local", tool, "FAIL", _safe_detail(exc)))
    finally:
        local.shutdown()
    provider = os.getenv("LCM_GAUNTLET_CLOUD_PROVIDER", "voyage").strip().lower()
    key_name, key_present = _key_gate(provider)
    if not key_present:
        reason = f"SKIP: {key_name} is absent"
        records.extend(("cloud", tool, "SKIP", reason) for tool in registered)
        batteries.extend((name, "SKIP", reason) for name in ("planted-secret", "loud-fail"))
    else:
        outbound, original = [], mod["embedding_provider"].resolve_provider
        def resolve(config, **kwargs):
            provider_value = original(config, **kwargs)
            return _Capture(provider_value, outbound) if provider_value is not None else None
        mod["command"].resolve_provider = resolve
        mod["tools"].resolve_provider = resolve
        cloud_state = Path(tempfile.mkdtemp(prefix="cloud-", dir=state_root)); cloud = _engine(mod, cloud_state, "cloud", patterns=True)
        try:
            cloud_ctx = _seed(mod, cloud)
            try:
                _privacy(mod, cloud, cloud_ctx, outbound)
                batteries.append(("planted-secret", "PASS", "durable, dispatch, revision, recall checks passed"))
            except Exception as exc:
                batteries.append(("planted-secret", "FAIL", _safe_detail(exc)))
            for tool in registered:
                try:
                    if tool in missing: raise AssertionError("registered tool has no matrix scenario")
                    _scenario(cloud, tool, cloud_ctx, patterns=True)
                    records.append(("cloud", tool, "PASS", "real post-condition passed"))
                except Exception as exc:
                    records.append(("cloud", tool, "FAIL", _safe_detail(exc)))
        finally:
            cloud.shutdown()
        try:
            _loud_fail(mod, Path(tempfile.mkdtemp(prefix="cloud-loud-", dir=state_root)))
            batteries.append(("loud-fail", "PASS", "raise, counter, status, assembly checks passed"))
        except Exception as exc:
            batteries.append(("loud-fail", "FAIL", _safe_detail(exc)))
    tree = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD^{tree}"], check=True, text=True, capture_output=True).stdout.strip(); tag = subprocess.run(["git", "-C", str(worktree), "describe", "--tags", "--exact-match", "HEAD"], text=True, capture_output=True).stdout.strip() or "not-an-exact-tag"
    failed = bool(missing or any(row[2] == "FAIL" for row in records) or any(row[1] == "FAIL" for row in batteries))
    lines = ["# PHASE-A RECEIPT", "", f"- RC tag: `{tag}`", f"- Tree SHA: `{tree}`", f"- Command: `{shlex.join([sys.executable, *sys.argv])}`", f"- HERMES_LCM_REPO: `{worktree}`", f"- Claim class: `code_green_local`", f"- Runtime registry tools: {len(registered)}", f"- Registry coverage: `{'FAIL' if missing else 'COMPLETE'}`", f"- Missing matrix rows: `{', '.join(missing) if missing else 'none'}`", "", "## Tool matrix", "", "| Posture | Tool | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {posture} | `{tool}` | {result} | {detail} |" for posture, tool, result, detail in records]
    lines += ["", "## Batteries", "", "| Battery | Result | Detail |", "|---|---|---|"]
    lines += [f"| {name} | {result} | {detail} |" for name, result, detail in batteries]
    lines += ["", f"- Wall time: `{time.monotonic() - started:.3f}s`", f"- Exit verdict: `{'FAIL' if failed else 'PASS'}`", "- Proof boundary: this receipt proves only the executed rows against this tree; skipped cloud rows are not release-readiness proof.", ""]
    receipt = out / "PHASE-A-RECEIPT.md"
    receipt.write_text("\n".join(lines), encoding="utf-8")
    return 1 if failed else 0, receipt
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code, receipt = run(args.worktree, args.out)
    print(receipt); return code
if __name__ == "__main__":
    raise SystemExit(main())
