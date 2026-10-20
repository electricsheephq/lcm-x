"""One tree per process; synthetic providers and a scripted host clock only."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import time
from types import SimpleNamespace

SENTINEL = re.compile(r"⟦(?:r[^⟧]+|objective|newest)⟧")


def census(rows):
    return sorted(set(SENTINEL.findall(json.dumps(rows, ensure_ascii=False))))


MANAGED_BASELINE = dict(
    context_threshold=0.75, fresh_tail_count=24, fresh_tail_max_tokens=24000,
    leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
    large_output_externalization_enabled=True, large_output_active_replay_stubbing_enabled=True,
)


def visible_sentinels(view, produced):
    visible = set(census(view))
    visible.discard("⟦newest⟧")
    newest = next((row for row in reversed(produced) if row["role"] == "user"), None)
    if newest is not None:
        if "⟦objective⟧" not in census(newest):
            visible.discard("⟦objective⟧")
        row_tag = next(tag for tag in census(newest) if tag.startswith("⟦r"))
        if row_tag in visible:
            visible.add("⟦newest⟧")
    return sorted(visible)


def uncovered_rows(stored, view, nodes):
    """Synthetic row IDs identify live rows; all DAG message sources count."""
    live = set(census(view))
    covered = {sid for node in nodes if node.source_type == "messages" for sid in node.source_ids}
    return sum(1 for row in stored if row.get("role") != "system" and row["store_id"] not in covered
               and not {tag for tag in census(row) if tag.startswith("⟦r")} & live)


def config_overrides(case, baseline_config):
    return {**(MANAGED_BASELINE if baseline_config == "managed" else {}), **case["overrides"]}


def pair_errors(rows):
    calls, results, errors = set(), set(), []
    seen_calls, missing = set(), []
    pending = []
    for row in rows:
        if row.get("role") != "tool":
            pending = []
            if row.get("tool_calls"):
                missing.extend(sorted(calls - results))
                calls, results = set(), set()
        for call in row.get("tool_calls") or []:
            if call["id"] in calls:
                errors.append(f"duplicate-call:{call['id']}")
            calls.add(call["id"])
            seen_calls.add(call["id"])
            pending.append(call["id"])
        if row.get("role") == "tool":
            cid = row.get("tool_call_id")
            if cid not in seen_calls:
                errors.append(f"result-before-call:{cid}")
            if cid in results:
                errors.append(f"duplicate-result:{cid}")
            elif cid in seen_calls:
                if cid not in pending:
                    errors.append(f"result-not-contiguous:{cid}")
                else:
                    if cid != pending[0]:
                        errors.append(f"result-out-of-order:{cid}")
                    pending.remove(cid)
            results.add(cid)
    return errors + [f"missing-result:{cid}" for cid in missing + sorted(calls - results)]


def user_row_evidence(view, original, count_tokens, normalize, mark_re, store):
    """Bind user token counts to input row sentinels, even when a mark hides them."""
    by_tag = {}
    for row in view:
        if row.get("role") != "user":
            continue
        text = normalize(row.get("content")) or ""
        match = mark_re.search(text) if mark_re is not None else None
        stored = store.get(int(match[3])) if match is not None and match[1] == "user" else None
        for tag in census(row) + (census(stored) if stored and stored["role"] == "user" else []):
            if tag.startswith("⟦r"):
                by_tag[tag] = (row, stored, match)
    evidence = {}
    for i, source in enumerate(original):
        if source.get("role") != "user":
            continue
        found = next((by_tag[tag] for tag in census(source) if tag.startswith("⟦r") and tag in by_tag), None)
        if found is None:
            continue
        row, stored, match = found
        marked = bool(stored and stored["role"] == "user"
                      and normalize(stored.get("content")) == normalize(source.get("content")))
        evidence[str(i)] = dict(tokens=count_tokens(row), marked=marked,
                                preserved=row.get("content") == source.get("content") and match is None)
    return evidence


@contextmanager
def module_patches():
    """Restore each patched binding by identity, including partial setup failures."""
    originals = []

    def patch(module, attr, value):
        originals.append((module, attr, getattr(module, attr)))
        setattr(module, attr, value)

    try:
        yield patch
    finally:
        for module, attr, original in reversed(originals):
            setattr(module, attr, original)


def load_tree(tree):
    spec = importlib.util.spec_from_file_location("hermes_lcm", tree / "__init__.py", submodule_search_locations=[str(tree)])
    sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
    tokens = importlib.import_module("hermes_lcm.tokens")
    missing = []
    for name in ("_encoder_ready", "_encoder"):
        if not hasattr(tokens, name):
            missing.append(f"tokens.{name}")
    modules = {name: importlib.import_module(f"hermes_lcm.{name}") for name in
               ("engine", "config", "escalation", "survival_fit", "externalize", "message_content", "dag")}
    # Newer trees moved compress() out of engine.py; both bind this provider.
    if (tree / "compaction.py").exists():
        modules["compaction"] = importlib.import_module("hermes_lcm.compaction")
    if not hasattr(modules["survival_fit"], "_host_estimate"):
        missing.append("survival_fit._host_estimate")
    modules["tokens"] = tokens
    return modules, missing


def run_case(case, modules, directory, initial_missing, baseline_config="bare", side="candidate"):
    with module_patches() as patch:
        tokens = modules["tokens"]
        if hasattr(tokens, "_encoder_generation"):
            patch(tokens, "_encoder_generation", tokens._encoder_generation + 1)
        for name, value in (("_encoder_ready", True), ("_encoder", None)):
            if hasattr(modules["tokens"], name):
                patch(modules["tokens"], name, value)
        if hasattr(modules["survival_fit"], "_host_estimate"):
            patch(modules["survival_fit"], "_host_estimate", lambda messages: None)
        return _run_case(case, modules, directory, initial_missing, baseline_config, side, patch)


def _run_case(case, modules, directory, initial_missing, baseline_config, side, patch):
    missing = list(initial_missing)
    now = [100000.0]
    # Replace module bindings, not the interpreter's time module or thread clocks.
    clock = SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith("__")})
    clock.time = clock.monotonic = lambda: now[0]
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("hermes_lcm.") and hasattr(module, "time"):
            if hasattr(module.time, "monotonic"):
                patch(module, "time", clock)
    calls = [0]

    def provider(*args, **kwargs):
        calls[0] += 1
        now[0] += 1
        if case["summarizer"] == "outage":
            raise RuntimeError("synthetic summary route outage")
        return "Earlier work. " + "fact " * max(1, (case.get("summary_tokens", 80) * 4 - 80) // 5) + "\nExpand for details about: records"

    if hasattr(modules["escalation"], "_call_llm_for_summary"):
        patch(modules["escalation"], "_call_llm_for_summary", provider)
    else:
        missing.append("escalation._call_llm_for_summary")
    config = modules["config"].LCMConfig(database_path=str(directory / "lcm.db"), temporal_rollups_enabled=False)
    for key, value in config_overrides(case, baseline_config).items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            missing.append(f"config.{key}")
    absent_overrides = [key for key in case["overrides"] if not hasattr(config, key)]
    if side == "candidate" and absent_overrides:
        return dict(id=case["id"], missing_seams=missing,
                    error=f"candidate override seam unavailable: {absent_overrides}")
    home = str(directory / "home")
    engine = modules["engine"].LCMEngine(config=config, hermes_home=home)
    engine.on_session_start("S", platform="cli", conversation_id="conv", context_length=case["window"])
    tokens = modules["tokens"]
    sf = modules["survival_fit"]
    mark_re = getattr(sf, "_PROJECTED_RE", None)
    if mark_re is None:
        missing.append("survival_fit._PROJECTED_RE")
    normalize = modules["message_content"].normalize_content_value

    def require(names):
        absent = [name for name in names if not hasattr(engine, name)]
        if absent:
            missing.extend(absent)
            raise RuntimeError(f"historical seam unavailable: {absent}")

    if case.get("fixture_summarizer"):
        require(("_summarize_leaf_chunk_with_rescue",))

        def fake_summary(**kwargs):
            return provider(), 1

        def fake_leaf(chunk, focus_topic=None, deadline=None):
            return chunk, tokens.count_messages_tokens(chunk), provider(), 1, 0

        patch(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
        for module in modules.values():
            if hasattr(module, "summarize_with_escalation"):
                patch(module, "summarize_with_escalation", fake_summary)

    def marks(view):
        return [i for i, row in enumerate(view) if row.get("role") == "user" and mark_re is not None
                and mark_re.fullmatch(normalize(row.get("content")) or "")]

    def user_rows(view, original):
        return user_row_evidence(view, original, tokens.count_message_tokens, normalize, mark_re, engine._store)

    try:
        if case["layer"] == "projection":
            needed = ("_store", "_survival_projection", "_survival_measure")
            absent = [name for name in needed if not hasattr(engine, name)]
            if absent:
                return dict(id=case["id"], missing_seams=missing + absent, error="projection unavailable")
            turn, ids = [], {}
            for entry in case["rows"]:
                message = deepcopy(entry["message"])
                turn.append(message)
                if entry["is_stored"]:
                    ids[id(message)] = engine._store.append("S", entry["stored"], conversation_id="conv")
            before = deepcopy(turn)
            out = engine._survival_projection(turn, ids, case["limit"])
            mutated = turn != before
            return dict(id=case["id"], out=out, out_tokens=engine._survival_measure(out),
                        in_tokens=engine._survival_measure(before), limit=case["limit"], user_marks=marks(out),
                        user_rows=user_rows(out, before),
                        user_sources={str(i): dict(store_id=ids.get(id(turn[i])), tokens=tokens.count_message_tokens(row),
                                                   text=normalize(row.get("content")) or "")
                                      for i, row in enumerate(before) if row.get("role") == "user"},
                        user_changed=[i for i, (a, b) in enumerate(zip(turn, out)) if a["role"] == "user" and a is not b],
                        ext=[i for i, row in enumerate(case["rows"]) if row["ext"]],
                        unstored=[i for i, row in enumerate(case["rows"]) if not row["is_stored"]],
                        input_mutated=mutated, missing_seams=missing)
        view, produced, sent_calls = [], [], []
        for index, new_rows in enumerate(case["steps"]):
            now[0] += case["call_seconds"]
            view.extend(deepcopy(new_rows))
            produced.extend(deepcopy(new_rows))
            engine.ingest(view)
            if index == 0 and case.get("seed_frontier"):
                require(("_dag",))
                for number in range(case["seed_frontier"]):
                    engine._dag.add_node(modules["dag"].SummaryNode(
                        session_id="S", depth=0, summary=f"Earlier fact group {number}",
                        token_count=2000, source_token_count=24000, source_ids=[],
                        source_type="messages", created_at=number))
            if index == 0 and case.get("leaf_history"):
                needed = ("_store", "_dag", "_last_compacted_store_id", "_pending_context_anchor_messages")
                absent = [name for name in needed if not hasattr(engine, name)]
                if absent:
                    missing.extend(absent)
                    raise RuntimeError("objective setup leaf seam unavailable")
                history = [row for row in engine._store.get_session_messages("S") if row["role"] != "system"][:case["leaf_history"]]
                engine._dag.add_node(modules["dag"].SummaryNode(
                    session_id="S", depth=0, summary="Earlier work. " + "abcd" * 1990, token_count=2000,
                    source_token_count=tokens.count_messages_tokens(history), source_ids=[row["store_id"] for row in history],
                    source_type="messages", expand_hint="synthetic history", created_at=30.0))
                engine._last_compacted_store_id = history[-1]["store_id"]
                engine._pending_context_anchor_messages = view[1:]
            start_calls, runs = calls[0], 0
            overhead = case.get("host_overhead", 0)
            prompt = tokens.count_messages_tokens(view) + overhead
            if "assembly_tail" in case:
                require(("_assemble_context", "_survival_fit"))
                assembled = engine._assemble_context(view[0], view[-case["assembly_tail"]:], include_lcm_note=False)
                view = engine._survival_fit(view, assembled, prompt, case["id"])
            elif engine.should_compress(prompt):
                view = engine.compress(view, current_tokens=prompt)
                runs += 1
            if "assembly_tail" not in case and tokens.count_messages_tokens(view) + overhead > case["window"]:
                view = engine.compress(view, current_tokens=tokens.count_messages_tokens(view) + overhead, bypass_cooldown=True)
                runs += 1
            sent = tokens.count_messages_tokens(view) + overhead
            failed = getattr(engine, "_last_overflow_recovery_failed", None)
            if not hasattr(engine, "_last_overflow_recovery_failed"):
                missing.append("_last_overflow_recovery_failed")
            visible = visible_sentinels(view, produced)
            sent_calls.append(dict(input_tokens=prompt, tokens=sent, rows=len(view), compress_runs=runs, summarizer_calls=calls[0] - start_calls,
                                   overflow_recovery_failed=failed, sentinels=sorted(visible), user_marks=marks(view),
                                   user_rows=user_rows(view, produced),
                                   pair_errors=pair_errors(view)))
            if hasattr(engine, "update_from_response"):
                engine.update_from_response({"prompt_tokens": sent, "completion_tokens": 50})
            else:
                missing.append("update_from_response")
        if not hasattr(engine, "_store"):
            raise RuntimeError("store census unavailable")
        stored = engine._store.get_session_messages("S", limit=1000000)
        durable = set(census(stored))
        external = modules["externalize"]
        retrieved, unavailable = [], []
        if hasattr(external, "extract_externalized_refs") and hasattr(external, "load_externalized_payload"):
            refs = {ref for row in stored for ref in external.extract_externalized_refs(row.get("content") or "")}
            for ref in sorted(refs):
                payload = external.load_externalized_payload(ref, config=config, hermes_home=home)
                if payload is None:
                    unavailable.append(ref)
                else:
                    durable.update(census(payload.get("content", "")))
                    retrieved.append(ref)
        else:
            missing.append("externalize.payload_retrieval")
        return dict(id=case["id"], calls=sent_calls, produced=census(produced), stored=sorted(durable),
                    uncovered_rows=uncovered_rows(stored, view, engine._dag.get_session_nodes("S")),
                    retrieved_payloads=len(retrieved), unavailable_payloads=unavailable, missing_seams=sorted(set(missing)))
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--layer", choices=["public", "projection"], required=True)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-config", choices=["bare", "managed"], default="bare")
    parser.add_argument("--side", choices=["ga", "candidate"], default="candidate")
    args = parser.parse_args()
    if os.environ.get("LCM_DIFFERENTIAL") != "1":
        parser.error("set LCM_DIFFERENTIAL=1 to opt in")
    logging.basicConfig(level=logging.ERROR)

    def no_network(*args, **kwargs):
        raise RuntimeError("network forbidden in differential driver")

    with module_patches() as patch, args.cases.open() as source, args.out.open("w") as output:
        patch(socket.socket, "connect", no_network)
        modules, missing = load_tree(args.tree)
        for line in source:
            case = json.loads(line)
            assert case["layer"] == args.layer
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="case-", dir=args.out.parent) as directory:
                try:
                    result = run_case(case, modules, Path(directory), missing, args.baseline_config, args.side)
                except Exception as exc:
                    result = dict(id=case["id"], error=f"{type(exc).__name__}: {exc}", missing_seams=missing)
            result["seconds"] = time.perf_counter() - started
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
