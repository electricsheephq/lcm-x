"""I1/I2: PR #1780 observes conflicts without reordering host occurrences."""
from copy import copy, deepcopy
from itertools import permutations
import pickle
import random

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1519_alias_uid_conflict import (
    TEXT, bindings, deterministic_host, message, prepare, targets,
)

# Register the deterministic clock/host fixture in this module too.
__all__ = ["deterministic_host"]

_rng = random.Random(1780)
RANDOM_UIDS = [tuple(f"{_rng.getrandbits(128):032x}" for _ in range(3)) for _ in range(24)]


@pytest.mark.parametrize("uids", [("host-old", "zz-b", "aa-new"),
                                  ("host-old", "hz-b", "zz-new"), *RANDOM_UIDS])
def test_non_alias_view_order(tmp_path, uids):
    """Adapt Claude m16/m20 and its 24 deterministic random-UID trials."""
    old, other, _fresh = uids
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        prepare(eng, "same_known_key")
        eng._store._conn.execute("UPDATE host_uid_bindings SET uid=? WHERE uid='host-old'", (old,))
        eng._store._conn.commit()
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        eng.ingest([message(TEXT, 30., message_uid=u) for u in (old, other)])
        prior = [r["store_id"] for r in targets(eng)]
        assert len(prior) == 2
    view = [message(TEXT, 30., message_uid=u) for u in uids]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        plan = eng._identity_anchor_prematch(view, view, 0)
        assert {i: rs[0]["store_id"] for i, rs in plan["matched"].items()} == dict(enumerate(prior))
        eng.ingest(deepcopy(view))
        assert len(targets(eng)) == 3
        for row, uid in zip(targets(eng), uids):
            assert [(b[0], b[4], b[5]) for b in bindings(eng, row["store_id"])] == [(uid, None if uid == _fresh else "agree", 0)]
    for _ in range(3):
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
            eng.ingest(deepcopy(view))
            assert len(targets(eng)) == 3


def test_exact_resend_preserves_binding_agreement(tmp_path):
    """Claude m21: no new UID, no sticky disagreement after an exact resend."""
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        prepare(eng, "same_known_key")
    view = [message(TEXT, 30., message_uid=u) for u in ("host-old", "m-b", "c-c")]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        eng.ingest(deepcopy(view))
        before = targets(eng)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        eng.ingest(deepcopy(view))
        assert targets(eng) == before
        assert all(b[4:] == ("agree", 0) for r in targets(eng) for b in bindings(eng, r["store_id"]))


@pytest.mark.parametrize("uids", list(permutations(("uid-one", "uid-new", None))))
def test_conflict_inside_cursor_is_observed(tmp_path, uids, caplog):
    """Astra A1 and its sibling orders: two aliases and three host occurrences."""
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        eng.ingest([message(TEXT, 10., message_uid="uid-one"), message(TEXT, 20., message_uid="uid-two")])
        eng._store.add_message_relations([[(r["store_id"], "alt_stamp", None, None, 99.)] for r in targets(eng)])
    view = [message(TEXT, 99., **({"message_uid": u} if u else {})) for u in uids]
    for turn in range(4):
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
            if not turn:
                with caplog.at_level("DEBUG", logger="hermes_lcm.identity_anchor"):
                    plan = eng._identity_anchor_prematch(view, view, 0)
                assert set(plan["matched"]) == {0, 1}
                assert any("would reject" in r.message for r in caplog.records)
            eng.ingest(deepcopy(view))
            assert len(targets(eng)) == 3



@pytest.mark.parametrize("transport", [copy, deepcopy, lambda v: pickle.loads(pickle.dumps(v))],
                         ids=["copy", "deepcopy", "pickle"])
def test_lookup_metadata_is_ephemeral(tmp_path, transport):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        prepare(eng, "same_known_key")
        value = eng._store.find_rows_by_observed_at("conv", ["parent"], [30.])
        plain = transport(value)
        assert type(plain) is list
        assert plain == list(value)
        assert not hasattr(plain, "canonical_stamps")
        assert b"canonical_stamps" not in pickle.dumps(value)


def test_plain_list_lookup_skips_observation_without_provenance(tmp_path, monkeypatch, caplog):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        prepare(eng, "same_known_key")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        lookup = eng._store.find_rows_by_observed_at
        monkeypatch.setattr(eng._store, "find_rows_by_observed_at", lambda *a, **k: list(lookup(*a, **k)))
        def unexpected(*args, **kwargs):
            pytest.fail("discarded provenance triggered a diagnostic re-read")

        observe = eng._observe_alias_uid_conflicts

        def without_diagnostic_reread(*args, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(eng._store, "get_batch", unexpected)
                return observe(*args, **kwargs)

        monkeypatch.setattr(eng, "_observe_alias_uid_conflicts", without_diagnostic_reread)
        with caplog.at_level("DEBUG", logger="hermes_lcm.identity_anchor"):
            eng.ingest([message(TEXT, 30., message_uid="host-new")])
        assert len(targets(eng)) == 1
        assert not any("would reject" in r.message for r in caplog.records)
