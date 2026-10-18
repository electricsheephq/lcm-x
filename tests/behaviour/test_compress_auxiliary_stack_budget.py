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


@pytest.mark.parametrize("phase", ["prepare", "condense"])
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

    from tests.behaviour.compress_trace_harness import PAD, SCENARIOS, dag, transcript

    survival = importlib.import_module("hermes_lcm.survival_fit")
    host_estimate = survival._host_estimate
    harness.pin_char_counter(monkeypatch)
    monkeypatch.setattr(survival, "_host_estimate", host_estimate)
    case = next(case for case in SCENARIOS if case.name == (
        "sweep-stub-first-exit" if phase == "prepare" else "sweep-pre-leaf-condensation"))
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
    agent = _LateAuxiliaryAgent()
    boundary_parent = "_stub_first_exit" if phase == "prepare" else "_run_threshold_sweep_condensation"
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
        for i, depth in enumerate(case.seeded_depths or (0,)):
            engine._dag.add_node(dag.SummaryNode(
                session_id="foreground-session", depth=depth, summary=f"Seed {i}. " + PAD,
                token_count=500, source_token_count=1000, source_ids=[],
                source_type="messages", created_at=100.0 + i))
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            agent.compress(engine, transcript(case), None if phase == "prepare" else int(case.current_tokens or 0))
        finally:
            sys.setprofile(previous)
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
