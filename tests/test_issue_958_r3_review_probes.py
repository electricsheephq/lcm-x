"""#958 r3: compare review claims with the same inputs on main."""

import pytest

import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine, strip_injected_context_blocks
from hermes_lcm.externalize import load_externalized_payload
from hermes_lcm.ingest_protection import (
    externalized_payload_stats,
    extract_all_externalized_payload_refs,
)
from hermes_lcm.reconcile import (
    _PRESERVED_OBJECTIVE_CONTEXT_PREFIX as PREFIX,
    _finalize_emission_descriptors,
)
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens


SEP = "\n\n---\n\n"
TAIL = [{"role": "assistant", "content": "Continuing."}]
PAYLOAD = "data:image/png;base64," + "QUJD" * 2048


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
    ), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("issue-958-r3", context_length=128_000)
    try:
        yield instance
    finally:
        instance.shutdown()
        tokens._count_tokens_cached.cache_clear()


def _part(engine):
    summary = "Synthetic stored history: " + "older work " * 80
    node_id = engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, depth=0, summary=summary,
        token_count=tokens.count_tokens(summary), source_token_count=1000,
        source_ids=[], source_type="messages", expand_hint="synthetic details",
    ))
    return f"[Recent Summary (d0, node {node_id})]\n{summary}\n[Expand for details: synthetic details]"


def _assemble(engine, content, budget=None, persist=False):
    engine._pending_context_anchor_messages = [{"role": "user", "content": content}, *TAIL]
    return engine._assemble_context(
        None, TAIL, include_lcm_note=False, persist=persist,
        assembly_cap_override=None if budget is None else budget + count_messages_tokens(TAIL),
    )


def _cost(content):
    return count_message_tokens({"role": "user", "content": content})


def _text(out):
    return "\n".join(row["content"] for row in out)


def test_c1_uncapped_merged_suffix_writes_only_emitted_payload(engine):
    part = _part(engine)
    content = PREFIX + "\nOlder objective." + SEP + part + "\n\nInspect " + PAYLOAD
    counts = [externalized_payload_stats(engine._config, engine._hermes_home)["externalized_payload_count"]]
    for _ in range(2):
        out = _assemble(engine, content)
        refs = extract_all_externalized_payload_refs(out[0]["content"])
        assert len(refs) == 1
        assert load_externalized_payload(refs[0], config=engine._config, hermes_home=engine._hermes_home)["content"] == PAYLOAD
        # Candidate 0 still emits the whole merged row plus the stored part.
        prefix, suffix = out[0]["content"].split("Inspect ", 1)
        assert prefix == content.split("Inspect ", 1)[0]
        assert suffix.endswith(SEP + part)
        counts.append(externalized_payload_stats(engine._config, engine._hermes_home)["externalized_payload_count"])
    print(f"C1 payload counts: {counts}")
    assert counts[0] == 0
    assert [after - before for before, after in zip(counts, counts[1:])] == [1, 1]


def test_c2_stripped_anchor_descriptor_and_roomy_control(engine):
    content = PREFIX + "\nKeep this objective.\n<relevant-memories>\n" + "mem " * 400 + "\n</relevant-memories>"
    stripped = strip_injected_context_blocks(content)
    tight = _cost(stripped) + 5
    assert _cost(content) > tight
    outcomes = []
    for budget in (tight, _cost(content) + 5):
        out = _assemble(engine, content, budget, persist=True)
        descriptors = _finalize_emission_descriptors(out, engine._pending_emission_candidates, {})
        emitted = PREFIX in _text(out)
        objective_descriptors = sum(item["kind"] == "objective" for item in descriptors)
        outcomes.append((emitted, objective_descriptors))
        # Accepted: sanitizer/descriptor mismatch already exists on main with a roomy cap.
        assert objective_descriptors == 0
        if emitted:
            assert out[0]["content"] == stripped
    print(f"C2 (emitted, objective descriptors), tight/roomy: {outcomes}")
    assert outcomes == ([(True, 0), (True, 0)] if hasattr(engine, "_latest_user_context_anchor_candidates") else [(False, 0), (True, 0)])


def test_c3_full_payload_sanitization_would_fit_whole_anchor(engine):
    part = _part(engine)
    older = PREFIX + "\nOLDER-OBJECTIVE keep me."
    newest = "Inspect " + PAYLOAD
    content = older + SEP + part + "\n\n" + newest
    sanitized = engine._sanitize_preserved_objective_content(content)
    budget = _cost(sanitized + SEP + part) + 10
    assert _cost(content) > budget
    assert _cost(older + "\n\n" + newest) > budget
    out = _assemble(engine, content, budget)
    text = _text(out)
    emitted = PREFIX in text
    refs = extract_all_externalized_payload_refs(text)
    print(f"C3 objective={emitted}, older={'OLDER-OBJECTIVE' in text}, payload refs={len(refs)}")
    # Accepted: main drops the anchor; branch's sanitized newest-only fallback is better.
    assert "OLDER-OBJECTIVE" not in text
    assert emitted == hasattr(engine, "_latest_user_context_anchor_candidates")
    assert len(refs) == int(emitted)
    if refs:
        assert load_externalized_payload(refs[0], config=engine._config, hermes_home=engine._hermes_home)["content"] == PAYLOAD


def test_c4_raw_anchor_fits_alone_but_displaces_summary(engine):
    part = _part(engine)
    content = PREFIX + "\nKeep this objective.\n<relevant-memories>\n" + "mem " * 300 + "\n</relevant-memories>"
    stripped = strip_injected_context_blocks(content)
    budget = max(_cost(content), _cost(stripped + SEP + part)) + 5
    assert _cost(content) < budget < _cost(content + SEP + part)
    out = _assemble(engine, content, budget)
    kept = _text(out).count(part)
    print(f"C4 summaries kept: {kept}")
    # Accepted: both main and branch pack the raw fitting anchor and omit the summary.
    assert out[0]["content"] == stripped
    assert kept == 0


def test_c5_quoted_verified_summary_fallback_is_after_whole_row(engine):
    part = _part(engine)
    instruction = "Then perform the final instruction."
    content = PREFIX + "\nUser request quotes:" + SEP + part + "\n\n" + instruction + SEP + part
    whole = content[:content.rfind(SEP)]
    budget = _cost(PREFIX + "\n" + instruction)
    assert engine._latest_user_context_anchor([{"role": "user", "content": content}], []) == whole
    assert _cost(whole) > budget
    if hasattr(engine, "_latest_user_context_anchor_candidates"):
        candidates = engine._latest_user_context_anchor_candidates([{"role": "user", "content": content}], [])
        assert candidates[0] == whole
        assert candidates[-1] == PREFIX + "\n" + instruction
    out = _assemble(engine, content, budget)
    emitted = PREFIX in _text(out)
    print(f"C5 objective={emitted}, quoted summary retained={part in _text(out)}")
    # Accepted J6/J13 tradeoff: fallback is tried only after main's whole anchor fails.
    assert emitted == hasattr(engine, "_latest_user_context_anchor_candidates")
    assert part not in _text(out)
    if emitted:
        assert out[0]["content"] == PREFIX + "\n" + instruction
