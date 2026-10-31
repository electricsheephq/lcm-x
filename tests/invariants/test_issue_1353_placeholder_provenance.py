"""Restored generated replay rows retain object provenance across a fault.

The boundary filtering state follows the existing literal-before-generated fixture;
this is a synthetic supported boundary shape, not a natural lifecycle run.
"""

from copy import deepcopy
from tests.behaviour.reconcile_trace_harness import LiteralPattern
from tests.invariants.test_issue_1353_prematch_fault_prefix import _base, _boom
from tests.test_host_uid_shadow import _engine, _m, _state_db


def test_restored_generated_row_keeps_provenance(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        e._compiled_ignore_message_patterns = [LiteralPattern()]
        original = _base()
        original[1]["content"] = "IGNORE THIS ROW synthetic"
        e.ingest(deepcopy(original))
        prior = e._last_active_replay_messages[1]
        digest = e._active_replay_placeholder_digest(prior["content"])
        assert digest is not None
        assert id(prior) in e._generated_ignored_active_replay_placeholder_message_ids
        changed = deepcopy(original)
        changed[0] = _m("user", "first rewrite", 20.0)
        changed[4] = _m("user", "later rewrite", 24.0)
        with monkeypatch.context() as patch:
            patch.setattr(e, "_identity_anchor_prematch", _boom)
            e.ingest(deepcopy(changed))
        restored = e._last_active_replay_messages[1]
        retained = (
            id(restored) in e._generated_ignored_active_replay_placeholder_message_ids
        )
        budget = e._active_replay_generated_placeholder_digest_budget()
        print(
            "restored_same_object",
            restored is prior,
            "generated_retained",
            retained,
            "budget_has_digest",
            digest in budget,
        )
        # Existing boundary filtering shape: literal row followed by generated row.
        monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "0")
        e._ingest_cursor = 0
        e._ingest_cursor_needs_reconcile = False
        e._compression_boundary_ingest_pending = True
        e._compression_boundary_active_placeholder_digest_budget = {digest: 1}
        e._compression_boundary_active_placeholder_digest_ordinals = {}
        e._ingest_messages(
            [
                {"role": "user", "content": restored["content"]},
                {"role": "user", "content": "boundary separator"},
                restored,
            ]
        )
        literal_count = sum(
            row["content"] == restored["content"]
            for row in e._store.get_session_messages("S")
        )
        print("boundary_literal_count", literal_count)
        assert literal_count == 1
        assert retained
        assert (
            e._generated_ignored_active_replay_placeholder_messages.get(id(restored))
            is restored
        )
        assert digest in budget
    finally:
        e.shutdown()
