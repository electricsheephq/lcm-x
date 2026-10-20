"""#1194: an empty optional array parameter behaves like an omitted one.

GPT-family models routinely fill optional array parameters with ``[]``. For
every optional array parameter of the retrieval tools, the call with ``[]``
must return the same JSON as the call without it, unless an empty list has a
documented meaning (marked ``exception`` in the table below).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

CURRENT = "session-a"
OTHER = "session-b"


def _engine(tmp_path: Path, name: str = "lcm.db") -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / name),
            embeddings_enabled=False,
            assertions_enabled=True,
            adaptive_retrieval_enabled=True,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        ),
        hermes_home=str(tmp_path / f"home-{name}"),
    )
    engine.on_session_start(CURRENT, conversation_id="conversation-a")
    return engine


def _seed(engine: LCMEngine) -> dict:
    seeded = {}
    for session_id, content in (
        (CURRENT, "We booked the kanban workshop in Lisbon for March."),
        (CURRENT, "The kanban board tracks the Lisbon workshop budget."),
        (OTHER, "Another kanban note about the Lisbon venue."),
    ):
        store_id = engine._store.append(
            session_id,
            {"role": "user", "content": content, "conversation_id": "conversation-a"},
        )
        # A fixed timestamp keeps encoded evidence sizes identical across replays.
        engine._store._conn.execute(
            "UPDATE messages SET timestamp=? WHERE store_id=?", (1_709_294_400.0, store_id)
        )
        engine._store._conn.commit()
        seeded.setdefault("store_ids", []).append(store_id)
        seeded.setdefault("contents", []).append(content)
    seeded["node_id"] = engine._dag.add_node(
        SummaryNode(
            session_id=CURRENT,
            depth=0,
            summary="Kanban workshop planning in Lisbon.",
            token_count=10,
            source_token_count=20,
            source_ids=list(seeded["store_ids"][:2]),
            source_type="messages",
            created_at=1_700_000_000.0,
            earliest_at=1_700_000_000.0,
            latest_at=1_700_000_000.0,
            expand_hint="Expand kanban workshop",
        )
    )
    storage = Path(engine._hermes_home, "lcm-large-outputs")
    engine._serialize_messages([
        {
            "role": "tool",
            "tool_call_id": "call-1194",
            "content": "kanban externalized payload " * 20,
        }
    ])
    seeded["externalized_ref"] = next(path.name for path in storage.glob("*.json"))
    return seeded


@pytest.fixture
def seeded(tmp_path):
    engine = _engine(tmp_path)
    try:
        yield engine, _seed(engine)
    finally:
        engine.shutdown()


def _exact_ref(seed: dict, index: int = 0) -> str:
    return f"lcm:{seed['store_ids'][index]}:0-{len(seed['contents'][index])}"


def _proposal(**fields):
    return {
        "version": "evidence-selector-v1",
        "selections": [],
        "missing_facets": [],
        **fields,
    }


# (tool, base args, path of the optional array parameter, documented exception or None)
_CASES = [
    ("lcm_grep", {"query": "kanban"}, ("externalized_refs",), None),
    (
        "lcm_grep",
        {"query": "kanban", "content_scope": "externalized"},
        ("externalized_refs",),
        None,
    ),
    ("lcm_grep", {"query": "kanban", "content_scope": "both"}, ("externalized_refs",), None),
    ("lcm_grep", {"query": "kanban"}, ("exclude_session_ids",), None),
    ("lcm_grep", {"query": "kanban", "session_scope": "all"}, ("exclude_session_ids",), None),
    ("lcm_recall", {"query": "kanban Lisbon"}, ("exclude_session_ids",), None),
    ("lcm_recall", {"query": "kanban Lisbon"}, ("seen_refs",), None),
    (
        "lcm_recall",
        {"query": "kanban Lisbon", "detail": "answer_ready"},
        ("seen_refs",),
        "seen_refs schema: supplied with detail='answer_ready' it is the delta cursor "
        "(exact-ref-delta-v1); omitting it preserves the baseline response bytes",
    ),
    ("lcm_query_state", {"subject_key": "user:self"}, ("kinds",), None),
    ("lcm_load_session", {"session_id": CURRENT}, ("roles",), None),
    (
        "lcm_expand_query",
        {"prompt": "What about Lisbon?", "query": "kanban", "output": "evidence"},
        ("node_ids",),
        None,
    ),
    (
        "lcm_expand_query",
        {"prompt": "What about Lisbon?", "query": "kanban", "output": "evidence"},
        ("session_ids",),
        None,
    ),
    # baseline_refs is required (minItems 1); [] and omission both fall back.
    ("lcm_evidence_pack", {"question": "Where is the workshop?"}, ("baseline_refs",), None),
    (
        "lcm_compile_evidence",
        {"question": "Where is the workshop?", "proposal": _proposal()},
        ("baseline_refs",),
        None,
    ),
    (
        "lcm_compile_evidence",
        {"question": "Where is the workshop?", "mode": "auto"},
        ("baseline_refs",),
        None,
    ),
    (
        "lcm_compile_evidence",
        {"question": "Where is the workshop?", "proposal": _proposal()},
        ("proposal", "requested_facets"),
        None,
    ),
    (
        "lcm_compile_evidence",
        {"question": "Where is the workshop?", "proposal": _proposal()},
        ("proposal", "selections"),
        "proposal schema: selections is required; [] means no selection",
    ),
    (
        "lcm_compile_evidence",
        {"question": "Where is the workshop?", "proposal": _proposal()},
        ("proposal", "missing_facets"),
        "proposal schema: missing_facets is required; [] means nothing is missing",
    ),
]


_OMIT = object()


def _with(args: dict, path: tuple[str, ...], value) -> dict:
    result = copy.deepcopy(args)
    target = result
    for key in path[:-1]:
        target = target[key]
    if value is _OMIT:
        target.pop(path[-1], None)
    else:
        target[path[-1]] = value
    return result


# proposal_sha256 / selector_sha256 / trace_sha256 hash the caller's literal
# proposal bytes by design (an audit record of exactly what was proposed), so a
# proposal that spells out an empty optional key hashes differently. Only these
# digests may differ for a parameter nested in the proposal.
_LITERAL_DIGEST_KEYS = {"proposal_sha256", "selector_sha256", "trace_sha256"}


# Always volatile: wall-clock latency, the random episode id, and the compiler
# trace digest, which hashes the latency metrics. Two identical calls can still
# round to the same latency, so the noise control alone cannot be relied on.
_VOLATILE_KEYS = {"latency_ms", "retrieval_id", "digest_sha256"}


def _diff_paths(left, right, prefix=()):
    """Return the leaf paths where two decoded JSON values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                found.add(prefix + (key,))
            else:
                found |= _diff_paths(left[key], right[key], prefix + (key,))
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        found = set()
        for index, (a, b) in enumerate(zip(left, right)):
            found |= _diff_paths(a, b, prefix + (index,))
        return found
    return set() if left == right else {prefix}


def _empty_vs_omitted(call, omitted: dict, empty: dict, *, literal_digests: bool):
    """Paths where ``[]`` differs from omission beyond run-to-run noise.

    The noise control is a second identical omitted call: fields that differ
    between two identical calls (latency_ms, timing-bearing digests) are noise.
    """
    first = call(omitted)
    noise = _diff_paths(first, call(omitted))
    with_empty = call(empty)
    changed = {
        path for path in _diff_paths(first, with_empty) - noise
        if path[-1] not in _VOLATILE_KEYS
    }
    if literal_digests:
        changed = {path for path in changed if path[-1] not in _LITERAL_DIGEST_KEYS}
    return changed, with_empty


def _fill(args: dict, seed: dict) -> dict:
    if args.get("baseline_refs") == "<seeded>":
        args["baseline_refs"] = [_exact_ref(seed)]
    return args


@pytest.mark.parametrize(
    ("tool", "args", "path", "exception"),
    _CASES,
    ids=[f"{case[0]}-{'.'.join(case[2])}-{index}" for index, case in enumerate(_CASES)],
)
def test_empty_optional_array_matches_omission(seeded, tool, args, path, exception):
    engine, seed = seeded
    if tool == "lcm_compile_evidence" and path != ("baseline_refs",):
        args = {**args, "baseline_refs": "<seeded>"}
    omitted = _fill(_with(args, path, _OMIT), seed)
    empty = _fill(_with(args, path, []), seed)

    changed, with_empty = _empty_vs_omitted(
        lambda call_args: json.loads(engine.handle_tool_call(tool, call_args)),
        omitted,
        empty,
        literal_digests=path[0] == "proposal",
    )

    if exception is None:
        assert changed == set(), with_empty
    else:
        # Documented meaning: the empty list is a real value, not "not provided".
        assert changed


def test_grep_empty_externalized_refs_with_default_scope_returns_results(seeded):
    engine, _seed_data = seeded
    payload = json.loads(
        engine.handle_tool_call("lcm_grep", {"query": "kanban", "externalized_refs": []})
    )

    assert "error" not in payload
    assert payload["total_results"] > 0
    assert "externalized_refs" not in payload


def test_grep_empty_externalized_refs_searches_like_auto_discovery(seeded):
    engine, seed = seeded
    payload = json.loads(
        engine.handle_tool_call(
            "lcm_grep",
            {"query": "kanban", "content_scope": "externalized", "externalized_refs": []},
        )
    )

    assert [item["ref"] for item in payload["results"]] == [seed["externalized_ref"]]


def test_grep_non_empty_externalized_refs_with_history_scope_still_errors(seeded):
    engine, seed = seeded
    payload = json.loads(
        engine.handle_tool_call(
            "lcm_grep",
            {
                "query": "kanban",
                "content_scope": "history",
                "externalized_refs": [seed["externalized_ref"]],
            },
        )
    )

    assert payload == {"error": "externalized_refs requires content_scope=externalized or both"}


def test_query_state_and_expand_query_still_reject_malformed_arrays(seeded):
    engine, _seed_data = seeded
    kinds = json.loads(
        engine.handle_tool_call(
            "lcm_query_state", {"subject_key": "user:self", "kinds": "fact"}
        )
    )
    session_ids = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "p", "query": "kanban", "output": "evidence", "session_ids": "x"},
        )
    )

    assert kinds == {"error": "kinds must be a non-empty array"}
    assert session_ids == {"error": "session_ids must be a non-empty array of strings"}


def test_recall_seen_refs_still_rejects_non_empty_cursor_without_answer_ready(seeded):
    engine, seed = seeded
    payload = json.loads(
        engine.handle_tool_call(
            "lcm_recall", {"query": "kanban", "seen_refs": [_exact_ref(seed)]}
        )
    )

    assert payload == {"error": "seen_refs requires detail='answer_ready'"}




# lcm_retrieve keeps episode state, so every call variant replays the episode on
# its own fresh store (store ids and refs are identical across replays).
def _retrieve(engine: LCMEngine, **args) -> dict:
    return json.loads(engine.handle_tool_call("lcm_retrieve", args))


_IDENTITY = {
    "intent_type": "workshop memory",
    "operation": "evidence_only",
    "subject_key": "user:self",
    "conversation_id": "conversation-a",
    "time_mode": "none",
    "policy_version": "adaptive-v1",
}
_START = {
    "action": "start",
    "question": "Where is the kanban workshop?",
    "identity": _IDENTITY,
    "requirements": [{"slot_id": "venue", "description": "workshop venue"}],
}


def _episode(root: Path, final: str, final_args: dict) -> dict:
    """start -> search, then ``final`` with ``final_args`` (CITATION = first ref)."""
    root.mkdir()
    engine = _engine(root)
    try:
        seed = _seed(engine)
        if final == "start":
            return _retrieve(engine, **final_args)
        retrieval_id = _retrieve(engine, **_START)["retrieval_id"]
        searched = _retrieve(
            engine,
            action="search",
            retrieval_id=retrieval_id,
            missing_slot="venue",
            tool="lcm_expand",
            tool_args={"store_id": seed["store_ids"][0]},
        )
        citation = searched["evidence"][0]["citation"]
        resolved = [{"slot_id": "venue", "evidence_refs": [citation]}]
        final_args = json.loads(
            json.dumps(final_args)
            .replace('"<citation>"', json.dumps(citation))
            .replace('"<resolved>"', json.dumps(resolved))
        )
        if final == "search":
            final_args = {
                "missing_slot": "venue",
                "tool": "lcm_expand",
                "tool_args": {"store_id": seed["store_ids"][1]},
                **final_args,
            }
        elif not final_args.get("resolved_slots"):
            # An earlier search resolves the slot, so finish may omit it or send [].
            _retrieve(
                engine,
                action="search",
                retrieval_id=retrieval_id,
                missing_slot="venue",
                tool="lcm_expand",
                tool_args={"store_id": seed["store_ids"][1]},
                resolved_slots=resolved,
            )
        return _retrieve(engine, action=final, retrieval_id=retrieval_id, **final_args)
    finally:
        engine.shutdown()


# (final action, args with the parameter omitted, args with it empty, documented exception)
_RETRIEVE_CASES = [
    # requirements is required for action=start (minItems 1): [] and omission both error.
    (
        "start",
        {key: value for key, value in _START.items() if key != "requirements"},
        {**_START, "requirements": []},
        None,
    ),
    ("search", {}, {"resolved_slots": []}, None),
    ("finish", {"resolved_slots": "<resolved>"}, {"resolved_slots": "<resolved>", "selected_refs": []}, None),
    ("finish", {"selected_refs": ["<citation>"]}, {"selected_refs": ["<citation>"], "resolved_slots": []}, None),
    # evidence_refs is required in each resolved_slots item (minItems 1): both error alike.
    (
        "finish",
        {"resolved_slots": [{"slot_id": "venue"}]},
        {"resolved_slots": [{"slot_id": "venue", "evidence_refs": []}]},
        None,
    ),
    (
        "finish",
        {"resolved_slots": "<resolved>", "computation": {}},
        {"resolved_slots": "<resolved>", "computation": {"operands": []}},
        "computation schema: operands is required (minItems 1); an empty operand "
        "list is a computation request, not an omitted parameter",
    ),
]


@pytest.mark.parametrize(
    ("final", "omitted", "empty", "exception"),
    _RETRIEVE_CASES,
    ids=["start-requirements", "search-resolved_slots", "finish-selected_refs",
         "finish-resolved_slots", "finish-evidence_refs", "finish-operands"],
)
def test_retrieve_empty_optional_array_matches_omission(
    tmp_path, final, omitted, empty, exception
):
    runs = iter(range(3))

    def call(final_args):
        return _episode(tmp_path / f"run-{next(runs)}", final, final_args)

    changed, with_empty = _empty_vs_omitted(call, omitted, empty, literal_digests=False)

    if exception is None:
        assert changed == set(), with_empty
    else:
        assert changed
