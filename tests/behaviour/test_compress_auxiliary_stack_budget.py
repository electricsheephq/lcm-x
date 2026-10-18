"""Keep late auxiliary detection reachable through the compression entry helper."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

harness = importlib.import_module("tests.harness")

# REALDEPTH C-r34.10-0.21.5-1405-delegate-G-1 has the maximum six host
# frames between plugin entry and the auxiliary frame (seven including it).
# Add one for public compress(), skipped by entering _compress_impl below;
# keep the auxiliary frame itself separate and the 32-frame budget unchanged.
OUTSIDE_FRAMES = 6 + 1


def _host_frames(engine: Any, view: list[dict[str, Any]], remaining: int) -> Any:
    if remaining > 1:
        return _host_frames(engine, view, remaining - 1)
    result = engine._compress_impl(view)
    assert engine._in_process_auxiliary_caller_generation("review-session") > 0
    assert engine._thread_context_session_id() == "review-session"
    assert engine.last_compression_noop_reason == "LCM bypassed below threshold: auxiliary thread context"
    return result


class _AuxiliaryAgent:
    session_id = "review-session"
    _parent_session_id = "foreground-session"
    _memory_write_origin = "background_review"

    def compress(self, engine: Any, view: list[dict[str, Any]]) -> Any:
        return _host_frames(engine, view, OUTSIDE_FRAMES)


@pytest.mark.parametrize("bound_session", ["review-session", "foreground-session"])
def test_begin_compression_preserves_late_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_session: str,
) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path, window=200_000, session_id=bound_session) as engine:
        view = [harness.user("auxiliary replay must stay out of durable history")]
        # compress() normally catches an already-marked caller before this
        # helper. Enter the real inner path to exercise its late-marker recheck
        # and its subsequent _current_auxiliary_prompt_tokens generation walk.
        # The review binding requires late generation-based registration;
        # the foreground binding matches the session-id walk via the parent.
        result = _AuxiliaryAgent().compress(engine, view)
        assert result == view
        assert engine._thread_context_has_auxiliary_session("review-session") == (
            bound_session == "review-session"
        )
        assert engine._store.get_session_count("review-session") == 0
        assert engine._store.get_session_count("foreground-session") == 0


class _LateAuxiliaryAgent:
    session_id = "review-session"
    _parent_session_id = "foreground-session"
    _memory_write_origin = ""

    def compress(self, engine: Any, view: list[dict[str, Any]], current_tokens: int | None) -> Any:
        return _phase_host_frames(engine, view, current_tokens, OUTSIDE_FRAMES)


def _phase_host_frames(engine: Any, view: list[dict[str, Any]], current_tokens: int | None, remaining: int) -> Any:
    if remaining > 1:
        return _phase_host_frames(engine, view, current_tokens, remaining - 1)
    return engine._compress_impl(view, current_tokens=current_tokens, force=False)


@pytest.mark.parametrize("phase", ["prepare", "condense", "leaf", "leaf-timeout"])
def test_sweep_phase_host_callbacks_preserve_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    """Keep the real path to each host boundary, including summary route wrappers.

    An already-marked child exits in P0. A host callback can mark its caller
    late; check both real walkers there, without replacing an engine method.
    The estimator and auxiliary-client response are the only host fakes.
    """
    import inspect
    import json
    import re
    import sys
    from types import ModuleType, SimpleNamespace

    from tests.behaviour.compress_trace_harness import Clock, PAD, SCENARIOS, dag, install_clock, transcript

    survival = importlib.import_module("hermes_lcm.survival_fit")
    host_estimate = survival._host_estimate
    harness.pin_char_counter(monkeypatch)
    monkeypatch.setattr(survival, "_host_estimate", host_estimate)
    case = next(case for case in SCENARIOS if case.name == (
        "sweep-stub-first-exit" if phase == "prepare" else "leaf-pass" if phase.startswith("leaf") else "sweep-pre-leaf-condensation"))
    settings = dict(fresh_tail_count=2, fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                    context_threshold=0.001, threshold_full_sweep_enabled=True,
                    max_assembly_tokens=100000, summary_model="trace-model", l3_truncate_tokens=2,
                    temporal_rollups_enabled=False, embeddings_enabled=False,
                    large_output_externalization_enabled=False,
                    large_output_active_replay_stubbing_enabled=False,
                    large_output_externalization_path=str(tmp_path / "payloads"))
    settings.update(dict(case.overrides))
    if phase == "prepare":
        # No host usage observation or explicit prefix cap: seeded summaries
        # reach the estimator through the deeper trial-assembly budget path.
        settings.update(context_threshold=0.02, max_assembly_tokens=0, summary_prefix_target_tokens=0)
    clock = Clock()
    if phase.startswith("leaf"):
        install_clock(monkeypatch, clock)
    agent = _LateAuxiliaryAgent()
    boundary_parent = ("_stub_first_exit" if phase == "prepare" else
                       "_summarize_leaf_chunk_with_rescue" if phase.startswith("leaf") else
                       "_run_threshold_sweep_condensation")
    distances: dict[str, int] = {}
    answers: list[tuple[str, int]] = []
    paths: dict[str, list[str]] = {}
    walkers = {"_in_process_auxiliary_session_id_from_stack", "_in_process_auxiliary_caller_generation"}

    def profile(frame: Any, event: str, value: Any) -> None:
        if event != "return" or frame.f_code.co_name not in walkers:
            return
        chain, current = [], frame.f_back
        while current is not None:
            chain.append(current.f_code.co_name)
            if current.f_locals.get("self") is agent:
                break
            current = current.f_back
        if boundary_parent in chain and agent._memory_write_origin:
            assert current is not None
            name = frame.f_code.co_name
            if len(chain) > distances.get(name, 0):
                distances[name], paths[name] = len(chain), chain
            assert value == "review-session" if name.endswith("from_stack") else value > 0

    with harness.engine_context(tmp_path, window=200000, session_id="foreground-session", **settings) as engine:
        def boundary() -> None:
            frame = inspect.currentframe()
            chain = []
            while frame is not None:
                chain.append(frame.f_code.co_name)
                frame = frame.f_back
            if boundary_parent not in chain:
                return
            agent._memory_write_origin = "background_review"
            try:
                answers.append((engine._thread_context_session_id(),
                                engine._in_process_auxiliary_caller_generation("review-session")))
            finally:
                agent._memory_write_origin = ""

        def estimate(messages: Any) -> int:
            boundary()
            return 0  # real local counter still determines the measured size

        def call_llm(**kwargs: Any) -> Any:
            boundary()
            if phase == "leaf-timeout":
                # Advance the deterministic provider clock through the real
                # configured deadline; production classifies the timeout.
                clock.now += kwargs["timeout"]
                raise TimeoutError("provider reached its deadline")
            nonce = re.search(r'<lcm-summary nonce="([a-f0-9]+)">', kwargs["messages"][0]["content"])
            assert nonce is not None
            content = (f'<lcm-summary nonce="{nonce[1]}">\n'
                       'Earlier turns recorded source evidence and preserved the pending request.\n'
                       'Expand for details about: turns\n</lcm-summary>')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        metadata, auxiliary = ModuleType("agent.model_metadata"), ModuleType("agent.auxiliary_client")
        metadata.estimate_messages_tokens_rough = estimate
        auxiliary.call_llm = call_llm
        monkeypatch.setitem(sys.modules, metadata.__name__, metadata)
        monkeypatch.setitem(sys.modules, auxiliary.__name__, auxiliary)
        for i, depth in enumerate(() if phase.startswith("leaf") else case.seeded_depths or (0,)):
            engine._dag.add_node(dag.SummaryNode(
                session_id="foreground-session", depth=depth, summary=f"Seed {i}. " + PAD,
                token_count=500, source_token_count=1000, source_ids=[],
                source_type="messages", created_at=100.0 + i))
        if phase.startswith("leaf"):
            engine._foreground_budget = engine._new_foreground_budget()
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            agent.compress(engine, transcript(case), None if phase == "prepare" else int(case.current_tokens or 0))
        finally:
            sys.setprofile(previous)
        if phase == "leaf-timeout":
            assert engine._foreground_budget.hung_route == "trace-model"
            assert engine._leaf_rescue_admit is None
        assert answers and all(session == "review-session" and generation > 0 for session, generation in answers)
        assert set(distances) == walkers
        assert max(distances.values()) < 28, "keep calls within four frames of the limit in their original frame"
        print(json.dumps(dict(phase=phase, distances=distances, paths=paths), sort_keys=True))


@pytest.mark.parametrize("path", ["tail", "yield", "replay", "filter"])
def test_leaf_phase_image_pricing_preserves_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    """Exercise the host pricing callback through each leaf-phase token path."""
    import ast
    import inspect
    import json
    import sys
    from types import ModuleType

    from tests.behaviour.compress_trace_harness import LiteralPattern, PAD, dag

    harness.pin_char_counter(monkeypatch)
    agent = _LateAuxiliaryAgent()
    compaction = importlib.import_module("hermes_lcm.compaction")
    source = inspect.getsource(compaction)
    driver = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef) and n.name == "_compress_impl")
    loop = next(n for n in driver.body if isinstance(n, ast.While))
    # Before extraction the same callback is identified by its original phase
    # statement, so this exact test also measures the immediate parent.
    filtering = next((n for n in loop.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "candidate_start < fresh_tail_start"), None)
    target = "_filter_leaf_backlog" if path == "filter" else "_admit_leaf_and_adopt_replay"
    distances: dict[str, int] = {}
    paths: dict[str, list[str]] = {}
    observed_roots: set[str] = set()
    settings = dict(fresh_tail_count=100 if path in {"yield", "replay"} else 2,
                    fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                    context_threshold=0.001, threshold_full_sweep_enabled=False,
                    max_assembly_tokens=100000, summary_model="trace-model",
                    temporal_rollups_enabled=False, embeddings_enabled=False,
                    fresh_tail_pressure_yield_min_observations=1,
                    large_output_externalization_enabled=False,
                    ignore_message_patterns=["IGNORE THIS ROW"] if path == "filter" else [])
    with harness.engine_context(tmp_path, window=200000, session_id="foreground-session", **settings) as engine:
        harness.ScriptedSummarizer().install(monkeypatch)
        view = []
        for i in range(6):
            view.extend([harness.user(f"Request {i}. " + PAD), harness.assistant(f"Reply {i}. " + PAD)])
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        for row in view:
            row["content"] = [{"type": "text", "text": row["content"]}, image]
        if path == "filter":
            view[2]["content"] = "IGNORE THIS ROW " + PAD
        if path == "filter":
            engine._compiled_ignore_message_patterns = [LiteralPattern()]
            engine._store.append_batch("foreground-session", view, conversation_id=engine._conversation_id)
            engine._ingest_cursor = len(view)
        else:
            engine.ingest(view)
        if path == "replay":
            engine._dag.add_node(dag.SummaryNode(
                session_id="foreground-session", depth=0, summary="Committed prefix.", token_count=20,
                source_token_count=2000, source_ids=[1, 2, 3, 4], source_type="messages", created_at=100.0))
            engine._lifecycle.advance_frontier(engine._conversation_id, "foreground-session", 4)
            engine._last_compacted_store_id = 4
            for _ in range(3):
                engine._summary_circuit_breaker.record_failure("trace-model")

        def price() -> int:
            frame = inspect.currentframe()
            chain = []
            phase_frame = None
            while frame is not None:
                chain.append(frame.f_code.co_name)
                if frame.f_code.co_name == target:
                    phase_frame = frame
                    break
                if frame.f_code.co_name == "_compress_impl":
                    if loop.lineno < frame.f_lineno <= loop.end_lineno and filtering is not None:
                        is_filter = filtering.lineno <= frame.f_lineno <= filtering.end_lineno
                        if is_filter == (path == "filter") and frame.f_lineno <= filtering.end_lineno:
                            phase_frame = frame
                    break
                frame = frame.f_back
            if phase_frame is None:
                return 1500
            root = chain[-2]
            if path == "yield" and root != "_maybe_engage_fresh_tail_pressure_yield":
                return 1500
            if path == "replay" and root != "count_messages_tokens":
                return 1500
            observed_roots.add(root)
            agent._memory_write_origin = "background_review"
            try:
                assert engine._thread_context_session_id() == "review-session"
                assert engine._in_process_auxiliary_caller_generation("review-session") > 0
            finally:
                agent._memory_write_origin = ""
            return 1500

        def profile(frame: Any, event: str, value: Any) -> None:
            name = frame.f_code.co_name
            if event != "return" or name not in {
                "_in_process_auxiliary_session_id_from_stack", "_in_process_auxiliary_caller_generation"
            } or not agent._memory_write_origin:
                return
            chain, current = [], frame.f_back
            while current is not None:
                chain.append(current.f_code.co_name)
                if current.f_locals.get("self") is agent:
                    break
                current = current.f_back
            assert current is not None
            if len(chain) > distances.get(name, 0):
                distances[name], paths[name] = len(chain), chain

        pricing = ModuleType("agent.image_token_cost")
        pricing.current_image_token_cost = price
        monkeypatch.setitem(sys.modules, pricing.__name__, pricing)
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            with engine._fresh_tail_pressure_yield_invocation():
                agent.compress(engine, view, 3000)
        finally:
            sys.setprofile(previous)
        assert observed_roots and len(distances) == 2
        assert max(distances.values()) < 28
        print(json.dumps(dict(path=path, roots=sorted(observed_roots), distances=distances, paths=paths), sort_keys=True))


@pytest.mark.parametrize("path", ["condense", "sweep", "overflow", "locked", "debt", "reply",
                                  "stop-debt", "stop-overflow", "stop-anchor", "stop-reply"])
def test_finish_phase_callbacks_preserve_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    """Measure real finish-phase estimator, image and summary host paths."""
    import ast
    import inspect
    import json
    import re
    import sys
    from types import ModuleType, SimpleNamespace

    from tests.behaviour.compress_trace_harness import PAD, Scenario, dag, transcript

    survival = importlib.import_module("hermes_lcm.survival_fit")
    host_estimate = survival._host_estimate
    harness.pin_char_counter(monkeypatch)
    monkeypatch.setattr(survival, "_host_estimate", host_estimate)
    compaction = importlib.import_module("hermes_lcm.compaction")
    source = inspect.getsource(compaction)
    driver = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef) and n.name == "_compress_impl")
    f1 = next((n for n in driver.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == "not leaf_compacted_this_turn"), None)
    f2 = next((n.lineno for n in driver.body if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == "pre_condensation_context"), None)
    f3 = next((n.lineno for n in driver.body if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == "self._refresh_raw_backlog_debt"), None)
    agent = _LateAuxiliaryAgent()
    distances: dict[str, dict[str, int]] = {}
    paths: dict[str, dict[str, list[str]]] = {}
    active_root = ""
    estimated_roots: set[str] = set()
    estimate_calls = 0
    settings = dict(fresh_tail_count=2, fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                    context_threshold=0.001, threshold_full_sweep_enabled=path == "sweep",
                    max_assembly_tokens=2000 if path == "overflow" else 20000 if path == "locked" else 100000,
                    deferred_maintenance_enabled=path == "debt",
                    summary_prefix_target_tokens=20, summary_model="trace-model",
                    l3_truncate_tokens=2, condensation_fanin=2,
                    temporal_rollups_enabled=False, embeddings_enabled=False,
                    large_output_externalization_enabled=False)
    if path.startswith("stop-"):
        settings.update(fresh_tail_count=100, fresh_tail_pressure_yield_enabled=False,
                        deferred_maintenance_enabled=path == "stop-debt",
                        max_assembly_tokens=2000 if path == "stop-overflow" else 100000)
    if path in {"reply", "stop-reply"}:
        settings.update(fresh_tail_count=7, leaf_chunk_tokens=400, dynamic_leaf_chunk_max=400,
                        max_assembly_tokens=1300, threshold_full_sweep_enabled=True,
                        summary_prefix_target_tokens=1000)
        if path == "stop-reply":
            settings.update(fresh_tail_count=100, threshold_full_sweep_enabled=False)
    with harness.engine_context(tmp_path, window=200000, session_id="foreground-session", **settings) as engine:
        def boundary() -> None:
            nonlocal active_root
            frame = inspect.currentframe()
            chain = []
            while frame is not None:
                name = frame.f_code.co_name
                if name in {"_finish_without_leaf", "_condense_after_leaves", "_finish_with_leaf"}:
                    phase = {"_finish_without_leaf": "F1", "_condense_after_leaves": "F2",
                             "_finish_with_leaf": "F3"}[name]
                    break
                if name == "_compress_impl":
                    if f3 is not None and frame.f_lineno >= f3:
                        phase = "F3"
                    elif f2 is not None and frame.f_lineno >= f2:
                        phase = "F2"
                    elif f1 is not None and f1.lineno <= frame.f_lineno <= f1.end_lineno:
                        phase = "F1"
                    else:
                        return
                    break
                chain.append(name)
                frame = frame.f_back
            else:
                return
            root = "_assemble_context" if chain[-1] == "inner" and "_assemble_context" in chain else chain[-1]
            active_root = phase + ":" + root
            if "estimate" in chain:
                estimated_roots.add(active_root)
            agent._memory_write_origin = "background_review"
            try:
                assert engine._thread_context_session_id() == "review-session"
                assert engine._in_process_auxiliary_caller_generation("review-session") > 0
            finally:
                agent._memory_write_origin = ""
                active_root = ""

        def estimate(messages: Any) -> int:
            nonlocal estimate_calls
            boundary()
            estimate_calls += 1
            # Reject each kept-reply proposal against its following host-input
            # measurement, exercising the rejection path without an internal mock.
            return 1000000 if path in {"reply", "stop-reply"} and estimate_calls % 2 else 0

        def price() -> int:
            boundary()
            return 1500

        def call_llm(**kwargs: Any) -> Any:
            boundary()
            if path == "locked" and engine._last_compacted_store_id > 0:
                engine._dag._conn.execute("""CREATE TEMP TRIGGER IF NOT EXISTS lock_summary
                    BEFORE INSERT ON summary_nodes
                    BEGIN SELECT RAISE(ABORT, 'database is locked'); END""")
            nonce = re.search(r'<lcm-summary nonce="([a-f0-9]+)">', kwargs["messages"][0]["content"])
            assert nonce is not None
            content = (f'<lcm-summary nonce="{nonce[1]}">\n'
                       'Earlier turns recorded source evidence and preserved the pending request.\n'
                       'Expand for details about: turns\n</lcm-summary>')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        def profile(frame: Any, event: str, value: Any) -> None:
            name = frame.f_code.co_name
            if event != "return" or not active_root or name not in {
                "_in_process_auxiliary_session_id_from_stack", "_in_process_auxiliary_caller_generation"
            }:
                return
            chain, current = [], frame.f_back
            while current is not None:
                chain.append(current.f_code.co_name)
                if current.f_locals.get("self") is agent:
                    break
                current = current.f_back
            assert current is not None
            row = distances.setdefault(active_root, {})
            if len(chain) > row.get(name, 0):
                row[name] = len(chain)
                paths.setdefault(active_root, {})[name] = chain

        metadata = ModuleType("agent.model_metadata")
        metadata.estimate_messages_tokens_rough = estimate
        pricing = ModuleType("agent.image_token_cost")
        pricing.current_image_token_cost = price
        auxiliary = ModuleType("agent.auxiliary_client")
        auxiliary.call_llm = call_llm
        for module in (metadata, pricing, auxiliary):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        if path in {"condense", "locked"}:
            for i, depth in enumerate((0, 0, 1)):
                engine._dag.add_node(dag.SummaryNode(
                    session_id="foreground-session", depth=depth, summary=f"Seed {i}. " + PAD,
                    token_count=500, source_token_count=1000, source_ids=[],
                    source_type="messages", created_at=100.0 + i))
        view = transcript(Scenario("finish-stack", turns=4 if path == "sweep" else 8))
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        for row in view[1:]:
            row["content"] = [{"type": "text", "text": row["content"]}, image]
        if path == "stop-anchor":
            engine.ingest(view)
            engine._dag.add_node(dag.SummaryNode(
                session_id="foreground-session", depth=0, summary="Committed prefix. " + PAD,
                token_count=500, source_token_count=1200, source_ids=[3, 4, 5],
                source_type="messages", created_at=100.0))
            engine._lifecycle.advance_frontier(engine._conversation_id, "foreground-session", 5)
            engine._last_compacted_store_id = 5
            for _ in range(3):
                engine._summary_circuit_breaker.record_failure("trace-model")
        if path in {"reply", "stop-reply"}:
            from tests.test_issue_1162_keep_reply import PAD as REPLY_PAD, _Host

            host = _Host(True)
            view = [row for i in range(1, 5) for row in (host.user(f"[T{i}] ask"), host.reply(f"reply to T{i}"))]
            view += [host.user("continue", pad=REPLY_PAD * 3), host.call(1), host.result(1),
                     host.user("continue", pad=REPLY_PAD * 3), host.call(2), host.result(2)]
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            agent.compress(engine, view, None if path in {"locked", "reply", "stop-reply"} else 250000 if path in {"overflow", "stop-overflow"} else 3000)
        finally:
            sys.setprofile(previous)
        assert {key[:2] for key in distances} == (
            {"F1"} if path.startswith("stop-") else {"F2"} if path in {"locked", "reply"} else {"F2", "F3"})
        if path in {"reply", "stop-reply"}:
            assert estimated_roots == ({"F1:_assemble_overflow_recovery_context"} if path == "stop-reply"
                                       else {"F2:_assemble_committed_compaction_context"})
            assert estimate_calls == 2
        required = {"condense": "F2:_maybe_condense", "sweep": "F2:_run_threshold_sweep_condensation",
                    "locked": "F2:_fail_open_after_publication_failure", "debt": "F3:_refresh_raw_backlog_debt",
                    "stop-debt": "F1:_refresh_raw_backlog_debt", "stop-anchor": "F1:_assemble_context",
                    "stop-overflow": "F1:_assemble_overflow_recovery_context"}
        if path in required:
            assert required[path] in distances
        print(json.dumps(dict(path=path, distances=distances, paths=paths), sort_keys=True))
        assert max(distance for row in distances.values() for distance in row.values()) < 28


@pytest.mark.parametrize("path", ["live", "prepared", "dependent", "extraction"])
def test_leaf_summary_and_publication_callbacks_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    """Measure every leaf-phase root reaching pricing or an auxiliary provider."""
    import ast
    import inspect
    import json
    import re
    import sys
    from types import ModuleType, SimpleNamespace

    from tests.behaviour.compress_trace_harness import PAD, Scenario, leaf_prep, transcript

    harness.pin_char_counter(monkeypatch)
    compaction = importlib.import_module("hermes_lcm.compaction")
    driver = next(n for n in ast.walk(ast.parse(inspect.getsource(compaction)))
                  if isinstance(n, ast.FunctionDef) and n.name == "_compress_impl")
    loop = next(n for n in driver.body if isinstance(n, ast.While))
    l4 = next((n for n in loop.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == "not summary_input_chunk"), None)
    l5_end = next((n for n in loop.body if isinstance(n, ast.Expr)
                   and ast.unparse(n.value) == "publication_excluded_store_ids.extend(filter_exclusion_proofs)"), None)
    l5_start = next((n for n in loop.body if isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "self._leaf_consumption"), None)
    agent = _LateAuxiliaryAgent()
    distances: dict[str, dict[str, int]] = {}
    paths: dict[str, dict[str, list[str]]] = {}
    active_root = ""
    settings = dict(fresh_tail_count=2, fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                    context_threshold=0.001, max_assembly_tokens=100000, summary_model="trace-model",
                    threshold_full_sweep_enabled=False, embeddings_enabled=False, temporal_rollups_enabled=False,
                    large_output_externalization_enabled=False, extraction_enabled=path == "extraction",
                    background_leaf_prep_enabled=path == "prepared")
    if path == "prepared":
        settings.update(context_threshold=0.75, leaf_chunk_tokens=4000, dynamic_leaf_chunk_max=4000,
                        max_assembly_tokens=200000, background_leaf_prep_max_calls=64)
    if path == "dependent":
        settings.update(leaf_chunk_tokens=1)
    monkeypatch.setattr(leaf_prep, "_LEAF_PREP", leaf_prep._LeafPrep())
    with harness.engine_context(tmp_path, window=200000, session_id="foreground-session", **settings) as engine:
        def boundary() -> None:
            nonlocal active_root
            frame, chain = inspect.currentframe(), []
            while frame is not None:
                name = frame.f_code.co_name
                if name in {"_summarize_selected_leaf", "_build_leaf_publication"}:
                    phase = "L4" if name == "_summarize_selected_leaf" else "L5"
                    break
                if name == "_compress_impl":
                    if l4 is not None and l4.lineno <= frame.f_lineno <= l4.end_lineno:
                        phase = "L4"
                    elif l5_start is not None and l5_end is not None and l5_start.lineno <= frame.f_lineno <= l5_end.end_lineno:
                        phase = "L5"
                    else:
                        return
                    break
                chain.append(name)
                frame = frame.f_back
            else:
                return
            active_root = phase + ":" + chain[-1]
            agent._memory_write_origin = "background_review"
            try:
                assert engine._thread_context_session_id() == "review-session"
                assert engine._in_process_auxiliary_caller_generation("review-session") > 0
            finally:
                agent._memory_write_origin = ""
                active_root = ""

        def price() -> int:
            boundary()
            return 1500

        def call_llm(**kwargs: Any) -> Any:
            boundary()
            nonce = re.search(r'<lcm-summary nonce="([a-f0-9]+)">', kwargs["messages"][0]["content"])
            content = (f'<lcm-summary nonce="{nonce[1]}">\nEarlier turns preserved evidence.\n'
                       'Expand for details about: turns\n</lcm-summary>') if nonce else "[]"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        def profile(frame: Any, event: str, value: Any) -> None:
            name = frame.f_code.co_name
            if event != "return" or not active_root or name not in {
                "_in_process_auxiliary_session_id_from_stack", "_in_process_auxiliary_caller_generation"
            }:
                return
            chain, current = [], frame.f_back
            while current is not None:
                chain.append(current.f_code.co_name)
                if current.f_locals.get("self") is agent:
                    break
                current = current.f_back
            assert current is not None
            row = distances.setdefault(active_root, {})
            if len(chain) > row.get(name, 0):
                row[name] = len(chain)
                paths.setdefault(active_root, {})[name] = chain

        pricing, auxiliary = ModuleType("agent.image_token_cost"), ModuleType("agent.auxiliary_client")
        pricing.current_image_token_cost, auxiliary.call_llm = price, call_llm
        monkeypatch.setitem(sys.modules, pricing.__name__, pricing)
        monkeypatch.setitem(sys.modules, auxiliary.__name__, auxiliary)
        view = transcript(Scenario("leaf-stack", turns=8))
        if path == "dependent":
            view = [harness.user("Previously consumed request. " + PAD),
                    harness.assistant("Dependent answer. " + PAD), harness.assistant("Dependent continuation. " + PAD),
                    harness.user("Fresh request."), harness.assistant("Fresh answer.")]
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        for row in view:
            if row["role"] != "system":
                row["content"] = [{"type": "text", "text": row["content"]}, image]
        engine.ingest(view)
        if path == "dependent":
            engine._compress_run.store_ids_by_message_id = engine._get_store_id_map_for_messages(view)
            for row in view[1:3]:
                engine._remember_generated_ignored_dependent_reply(row, row["content"][0]["text"])
            engine._lifecycle.advance_frontier(engine._conversation_id, "foreground-session", 1)
            engine._last_compacted_store_id = 1
            view = view[1:]
            engine._ingest_cursor = len(view)
        if path == "prepared":
            engine.last_prompt_tokens = 150000
            engine._prepare_leaves_after_turn(view)
            assert leaf_prep._LEAF_PREP.wait_idle(10)
            assert leaf_prep._LEAF_PREP.status()["cached_leaves"] > 0
        engine._foreground_budget = engine._new_foreground_budget()
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            agent.compress(engine, view, 150000 if path == "prepared" else 3000)
        finally:
            sys.setprofile(previous)
        required = {"live": "L4:_summarize_leaf_chunk_with_rescue", "prepared": "L4:summarize",
                    "dependent": "L4:count_messages_tokens", "extraction": "L4:_run_pre_compaction_extraction"}
        assert required[path] in distances
        assert "L5:count_messages_tokens" in distances
        print(json.dumps(dict(path=path, distances=distances, paths=paths), sort_keys=True))
        assert max(distance for row in distances.values() for distance in row.values()) < 28
