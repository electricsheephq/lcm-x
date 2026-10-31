"""Pin the pure B8 seams before moving them; no engine or provider is needed."""
from __future__ import annotations

import hashlib
import importlib
import logging
from typing import Any

import pytest

from hermes_lcm import identity_anchor, reconcile


def test_decomposition_preserves_order_remainder_and_ambiguity() -> None:
    split = identity_anchor._decompositions
    assert split("alpha\n\nbeta", {"alpha", "beta"}, partial=False) == [(["alpha", "beta"], "")]
    assert split("alpha\n\nnew", {"alpha"}, partial=True) == [(["alpha"], "new")]
    assert split("alpha\n\nnew", {"alpha"}, partial=False) == []
    assert split("alpha", {"alpha"}, partial=False) == []
    assert split("\n\n".join(["alpha"] * 70), {"alpha"}, partial=False) is None


def test_matching_merges_store_ids_and_uses_augmenting_paths() -> None:
    rows: list[dict[str, Any]] = [{"store_id": 1, "keys": {"a"}}, {"store_id": 1, "keys": {"b"}},
            {"store_id": 2, "keys": {"a"}}]
    assert identity_anchor._match_occurrences(rows, lambda row: row["keys"], [(10, "a"), (20, "b")]) == {1: 20, 2: 10}


def test_null_reservation_keeps_stamped_holder_and_alias_ownership() -> None:
    rows: list[dict[str, Any]] = [{"store_id": 1, "observed_at": 1.0, "forms": {"a", "b"}},
            {"store_id": 2, "observed_at": None, "forms": {"a"}}]
    forms = lambda row: row["forms"]  # noqa: E731
    assert identity_anchor._reserve_shown(rows, forms, [(10, (1.0, "a")), (20, (1.0, "b"))]) == {1, 2}
    assert identity_anchor._reserve_shown(rows, forms, [(10, (1.0, "a"))]) == {1}
    assert identity_anchor._reserve_shown(rows[:1], forms, [(10, (2.0, "a"))], {1: {2.0}}) == {1}


def _emission() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    span = "generated λ\n\n"
    row = {"role": "user", "content": span + " authored ", "message_uid": "uid"}
    scope = {"hermes_home": "home", "session_id": "session", "conversation_id": "conversation", "reset_epoch": None}
    descriptors = reconcile._finalize_emission_descriptors([row], [{
        "kind": "carrier", "span": span, "row": row,
        "full_identity": reconcile._emission_identity(row),
        "retained_source": {"store_id": 7}, "engine_uid": "uid",
    }], scope)
    proof = {"version": 4, **scope, "emissions": descriptors, "output": [reconcile._emission_identity(row)]}
    return [row], proof, span


def test_descriptor_bytes_projection_slice_and_retirement_key() -> None:
    rows, proof, span = _emission()
    descriptor = proof["emissions"][0]
    assert descriptor == {
        "kind": "carrier", "role": "user", "same_prefix_ordinal": 0,
        "output_occurrence": {"index": 0, "same_identity_ordinal": 0, "of": 1},
        "generated_span_sha256": hashlib.sha256(span.encode()).hexdigest(),
        "generated_span_bytes": len(span.encode()),
        "suffix_sha256": hashlib.sha256(b"authored").hexdigest(), "suffix_length": 8,
        "retained_source": {"store_id": 7}, "engine_uid": "uid",
        "scope": {key: proof[key] for key in ("hermes_home", "session_id", "conversation_id", "reset_epoch")},
    }
    projection = reconcile._project_emitted_occurrences(rows + [dict(rows[0])], proof=proof)
    entry, duplicate = projection.entries
    assert entry.full_identity == ("user", span + " authored ", "", "[]", "")
    assert entry.effective_identity == ("user", " authored ", "", "[]", "")
    assert entry.generated_span == span and entry.output_index == 0
    assert duplicate.generated_span is None and duplicate.full_identity == duplicate.effective_identity
    assert projection.slice(1).entries == (duplicate,)
    assert projection.slice(1).offset == 1 and projection.slice(1).complete_length == 2
    assert reconcile._carrier_emission_key(entry) == reconcile._carrier_emission_key(descriptor)
    with pytest.raises(TypeError):
        entry.retained_source["store_id"] = 8
    assert rows[0]["content"] == span + " authored "


@pytest.mark.parametrize("change", ["legacy", "scope", "digest", "index"])
def test_unproven_emission_stays_whole(change: str) -> None:
    rows, proof, _span = _emission()
    if change == "legacy":
        proof["version"] = 3
    elif change == "scope":
        proof["session_id"] = "different"
    elif change == "digest":
        proof["emissions"][0]["generated_span_sha256"] = "bad"
    else:
        proof["emissions"][0]["output_occurrence"]["index"] = -1
    entry = reconcile._project_emitted_occurrences(rows, proof=proof).entries[0]
    assert entry.generated_span is None and entry.full_identity == entry.effective_identity


def test_descriptor_shape_declines_invalid_occurrence() -> None:
    assert reconcile._descriptor_shape_is_well_formed({"scope": {}, "output_occurrence": {"index": 0, "of": 1}})
    for descriptor in (None, {}, {"scope": {}, "output_occurrence": {"index": True}},
                       {"scope": {}, "output_occurrence": {"index": 1, "of": 1}}):
        assert not reconcile._descriptor_shape_is_well_formed(descriptor)


def test_null_reroute_budget_retains_warning_namespace(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    matching = importlib.import_module(identity_anchor._reserve_shown.__module__)
    monkeypatch.setattr(matching, "_MATCH_WORK_PER_ITEM", 0)
    monkeypatch.setattr(matching, "_MATCH_WORK_FLOOR", 0)
    rows: list[dict[str, Any]] = [{"store_id": 1, "observed_at": 1.0, "forms": {"a", "b"}},
            {"store_id": 2, "observed_at": None, "forms": {"a"}}]
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.identity_anchor"):
        assert identity_anchor._reserve_shown(rows, lambda row: row["forms"],
                                              [(10, (1.0, "a")), (20, (1.0, "b"))]) == {1}
    assert [(record.name, record.getMessage()) for record in caplog.records] == [
        ("hermes_lcm.identity_anchor",
         "LCM identity-anchor NULL re-route spent its work budget 0: rows=2 occurrences=2 reserved=0"),
    ]
