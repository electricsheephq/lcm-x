"""Computed dates must not promote a carried observation into certainty."""
from dataclasses import replace
import json

import pytest

from hermes_lcm import evidence_pack
from tests.harness import engine_context


STAMP = 1_710_936_000.0  # 2024-03-20T12:00:00Z


@pytest.mark.parametrize("carried_index", [None, 0, 1])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("question_date", [None, "2024-03-22"])
def test_interval_uses_only_certain_operand_dates(
    tmp_path, monkeypatch, carried_index, absolute, question_date,
):
    monkeypatch.setattr("time.time", lambda: 1_800_000_000.0)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = (
            ["The PlankChallenge happened on 2024-03-19.",
             "The launch happened on 2024-03-17."]
            if absolute else
            ["The PlankChallenge happened yesterday.",
             "The launch happened 3 days ago."]
        )
        ids = [engine._store.append(str(index), dict(
            role="user", content=text, timestamp=STAMP,
        )) for index, text in enumerate(texts)]
        if carried_index is not None:
            engine._store.add_message_relations([
                [(ids[carried_index], "carried_stamp", None, 0, STAMP)],
            ])
        refs = [dict(exact_ref=f"lcm:{sid}:0-{len(text)}", label=label)
                for sid, text, label in zip(ids, texts, ["PlankChallenge", "launch"])]
        obj = json.loads(evidence_pack.build_evidence_pack(dict(
            question="How many days between the PlankChallenge and launch?",
            baseline_refs=refs, question_date=question_date,
        ), engine=engine))
        assert len(obj["evidence"]) == 2
        for index, item in enumerate(obj["evidence"]):
            assert item["observation_time"].get("time_uncertain") == (
                "carried" if index == carried_index else None
            )
        blocked = carried_index is not None and (not absolute or question_date is not None)
        if blocked:
            assert obj["status"] == "evidence_ready"
            assert obj["computation"] is None
            assert obj["computation_fallback_reason"] == "uncertain_source_observation_time"
        else:
            assert obj["status"] == "computed"
            assert obj["computation"]["result"] == "2 days"
            assert obj["computation"]["evidence_dates"] == ["2024-03-19", "2024-03-17"]


def test_carried_evidence_outside_grounded_operands_does_not_block(tmp_path, monkeypatch):
    """Isolate the evidence/operand seam without adding a new selection path."""
    monkeypatch.setattr("time.time", lambda: 1_800_000_000.0)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = ["The PlankChallenge happened yesterday.",
                 "The launch happened 3 days ago.", "Background happened yesterday."]
        ids = [engine._store.append(str(index), dict(
            role="user", content=text, timestamp=STAMP,
        )) for index, text in enumerate(texts)]
        engine._store.add_message_relations([[(ids[2], "carried_stamp", None, 0, STAMP)]])
        original = evidence_pack._ground_pack_evidence

        def ground_only_operands(resolved, **kwargs):
            grounded = original(resolved, **kwargs)
            assert grounded.status == "grounded"
            return replace(grounded, operands=grounded.operands[:2])

        monkeypatch.setattr(evidence_pack, "_ground_pack_evidence", ground_only_operands)
        obj = json.loads(evidence_pack.build_evidence_pack(dict(
            question="How many days between the PlankChallenge and launch?",
            baseline_refs=[dict(exact_ref=f"lcm:{sid}:0-{len(text)}")
                           for sid, text in zip(ids, texts)],
        ), engine=engine))
        assert obj["evidence"][2]["observation_time"]["time_uncertain"] == "carried"
        assert obj["status"] == "computed"
        assert obj["computation"]["result"] == "2 days"
