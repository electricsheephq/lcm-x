"""Generated occurrence schedules: sender safety, deliberately no capture-rate gate."""
from copy import deepcopy
import random

import pytest

from tests.harness import engine_context
from tests.test_sender_capture import begin, capture_module, expected, message, senders
from tests.test_sender_capture import clean_capture_registry  # noqa: F401


@pytest.mark.parametrize("uid_host", [False, True], ids=["uidless-deepcopy", "uid-deepcopy"])
def test_t4_generated_sender_occurrences(tmp_path, uid_host):
    stamped_count = 0
    for seed in range(12):
        rng = random.Random(seed)
        with engine_context(tmp_path / str(seed), window=128000, conversation_id="conv") as e:
            view, owners = [], {}
            for turn in range(8):
                row = message(0)  # Deliberately identical content and same-second stamp.
                if uid_host:
                    row["message_uid"] = f"{seed * 100 + turn:032x}"
                current = [*view, row]
                sender = f"synthetic-{turn}"
                begin(e, current, turn=f"turn-{turn}", sender_id=sender)
                cloned = deepcopy(current)
                if rng.choice([False, True]):
                    cloned.append(dict(role="assistant", content=f"synthetic reply {turn}"))
                e.ingest(cloned)
                rows = e._store.get_session_messages("S")
                stamped = e._store.get_message_senders([r["store_id"] for r in rows])
                for stored in rows:
                    sid = stored["store_id"]
                    owners.setdefault(sid, turn)
                    value = stamped[sid]
                    if owners[sid] == turn:
                        stamped_count += value != (None, None)
                    allowed = (f"telegram:synthetic-{owners[sid]}", "gateway_principal")
                    if value not in {(None, None), allowed}:
                        raise AssertionError("generated cross-turn sender violation")
                    if not uid_host and value != (None, None):
                        raise AssertionError("uidless deepcopy must remain unresolved")
                e.on_turn_complete(cloned, turn_id=f"turn-{turn}")
                view = cloned
    if uid_host:
        assert stamped_count > 0, "uid schedules must exercise sender capture"


@pytest.mark.parametrize("shape", ["first-uid-survivor", "uidless-inplace", "empty-witness", "r3-remainder",
                                   "first-prefix-extended", "first-prefix-unchanged"])
def test_t4_merge_survivors_are_null(tmp_path, shape):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        row = message()
        if shape == "first-uid-survivor" or shape.startswith("first-prefix"):
            row["message_uid"] = "c" * 32
        p = begin(e, [row])
        if shape == "first-uid-survivor":
            row["_absorbed_message_uids"] = ["d" * 32]
        elif shape.startswith("first-prefix"):
            # Hermes 62b06323b2 keeps the first uid; an empty second turn
            # leaves content unchanged but still carries the prefix witness.
            row["_merged_turn_prefix"] = row["content"]
            if shape.endswith("extended"):
                # W1's identity guard catches this extension before the prefix check matters.
                row["content"] += "\n\nsynthetic second constituent"
        elif shape == "empty-witness":
            row["_absorbed_message_uids"] = []
            row["content"] += " synthetic merged suffix"
        elif shape == "r3-remainder":
            row["_merged_turn_prefix"] = "synthetic prior prefix"
        else:
            row["content"] += " synthetic merged suffix"
        assert not capture_module().matches(e, p, row)
        e.ingest([row])
        assert all(value == (None, None) for value in senders(e))


@pytest.mark.parametrize("anchor", ["0", "1"])
@pytest.mark.parametrize("cold", [False, True], ids=["warm-r5", "cold-value-match"])
def test_t4_value_match_never_authorizes_old_row_update(tmp_path, monkeypatch, anchor, cold):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    old, current = message(), message()
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        e.ingest([old])
        if not cold:
            begin(e, [current])
            e.ingest([current])
            if senders(e)[0] != (None, None):
                raise AssertionError("R5 value match stamped an older occurrence")
            return
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        begin(e, [current])
        e.ingest([current])
        if senders(e)[0] != (None, None):
            raise AssertionError("cold anchor value match stamped an older occurrence")


def test_registered_compress_copy_matches_but_unregistered_copy_does_not(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        row, copy = message(), message()
        e.ingest([row])
        # The bind-time rollup pass writes on its own connection; the guarded
        # UPDATE is non-blocking by design, so let that pass finish first.
        assert e.drain_rollup_maintenance(timeout=10)
        p = begin(e, [row])
        assert not capture_module().matches(e, p, copy)
        e._rekey_host_rewrite_watch([row], [copy])
        assert capture_module().matches(e, p, copy)
        assert senders(e) == [expected()]
