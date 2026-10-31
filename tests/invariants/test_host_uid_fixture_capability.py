"""Pair recovery/replay controls across explicit non-UID and UID-capable hosts."""
from copy import deepcopy
import hashlib

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1511_cold_later_stamp import snapshot
from tests.invariants.test_issue_1519_alias_uid_conflict import (
    TEXT, deterministic_host, message, prepare,
)

__all__ = ["deterministic_host"]


@pytest.mark.parametrize("mode", ["off", "shadow", "on"])
@pytest.mark.parametrize("shape", ["alias", "ordered-resend"])
def test_uid_capable_recovery_preserves_rows_and_adds_exact_binding(tmp_path, monkeypatch, mode, shape):
    from hermes_lcm import host_uid_emit

    controls = []
    for capable in (False, True):
        root = tmp_path / str(capable)
        root.mkdir()
        # Both arms get the same verified lineage topology and deterministic clock.
        deterministic_host.__wrapped__(root, monkeypatch)
        monkeypatch.setattr(host_uid_emit, "_host_uid_capability", capable)
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
        events = []
        with engine_context(root, window=128000, session_id="parent", conversation_id="conv") as eng:
            view = prepare(eng, "same_known_key")
            events.append(snapshot(eng))
        if shape == "ordered-resend":
            view = [message(TEXT, 30., message_uid=uid) for uid in ("host-old", "m-b", "c-c")]
        # Deliberately seed before switching modes, as the round2 contract does.
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
        for _ in range(2):
            with engine_context(root, window=128000, session_id="parent", conversation_id="conv") as eng:
                eng.ingest(deepcopy(view))
                events.append(snapshot(eng))
        controls.append(events)

    # Independent oracle for the generated recovery note's home-derived UID.
    home_tag = hashlib.sha256(str((root / "home/state.db").resolve()).encode()).hexdigest()[:16]
    note = ("[LCM overflow recovery] Your latest message (5505 tokens) is stored but exceeds the "
            "recovery budget of 120 tokens; it was not included. Re-send a shorter version or ask LCM to recall it.")
    basis = hashlib.sha256(note.encode()).hexdigest()
    uid = hashlib.sha256(f"lcmx-engine-uid\0{home_tag}:parent\0overflow_note\0{basis}\0{0}".encode()).hexdigest()[:32]
    extra = dict(store_id=3, uid=uid, lineage_key="HOME:parent", kind="alias_candidate",
                 binding_version=1, proof_kind="unknown", created_at=1700000000.,
                 first_check=None, disagree_seen=0)
    for plain, emitting in zip(*controls):
        # Includes every ordered row field, observed stamp and relation.
        assert {k: v for k, v in emitting.items() if k != "bindings"} == {
            k: v for k, v in plain.items() if k != "bindings"}
        assert emitting["bindings"] == plain["bindings"][:1] + [extra] + plain["bindings"][1:]
