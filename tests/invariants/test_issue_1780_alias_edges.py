"""I1/I2: PR #1780 forbids edges without reordering other host occurrences."""
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
def test_rejected_occurrence_inside_cursor_is_stored(tmp_path, uids):
    """Astra A1 and its sibling orders: two aliases and three host occurrences."""
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        eng.ingest([message(TEXT, 10., message_uid="uid-one"), message(TEXT, 20., message_uid="uid-two")])
        eng._store.add_message_relations([[(r["store_id"], "alt_stamp", None, None, 99.)] for r in targets(eng)])
    view = [message(TEXT, 99., **({"message_uid": u} if u else {})) for u in uids]
    for _ in range(4):
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
            eng.ingest(deepcopy(view))
            assert len(targets(eng)) == 3
            found = [r for r in targets(eng) if any(b[0] == "uid-new" for b in bindings(eng, r["store_id"]))]
            assert len(found) == 1
            assert found[0]["observed_at"] == 99.


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


def test_plain_list_lookup_fails_closed(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        prepare(eng, "same_known_key")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as eng:
        lookup = eng._store.find_rows_by_observed_at
        monkeypatch.setattr(eng._store, "find_rows_by_observed_at", lambda *a, **k: list(lookup(*a, **k)))
        eng.ingest([message(TEXT, 30., message_uid="host-new")])
        assert len(targets(eng)) == 2
        assert any(b[0] == "host-new" for b in bindings(eng, targets(eng)[1]["store_id"]))


def test_forbidden_pairs_keep_maximum_and_allowed_base_assignments():
    from hermes_lcm.recognition_matching import _match_occurrences
    from tests.test_issue_436_matching import _instance, _maximum

    rng = random.Random(1780)
    for _ in range(2000):
        rows, occurrences = _instance(rng, 2)
        forbidden = {(r["store_id"], o) for r in rows for o, _key in occurrences if rng.random() < .3}
        def keys_of(r):
            return r["keys"]
        base = _match_occurrences(rows, keys_of, occurrences)
        result = _match_occurrences(rows, keys_of, occurrences, forbidden=forbidden)
        allowed = [{**r, "keys": [o for o, key in occurrences
                                  if key in r["keys"] and (r["store_id"], o) not in forbidden]} for r in rows]
        assert len(result) == _maximum(allowed, [(o, o) for o, _key in occurrences])
        assert len(set(result.values())) == len(result)
        assert all((sid, o) not in forbidden for sid, o in result.items())
        assert all(dict(occurrences)[o] in next(r["keys"] for r in rows if r["store_id"] == sid)
                   for sid, o in result.items())
        assert result == _match_occurrences(rows, keys_of, occurrences, forbidden=forbidden)
        if not any((sid, o) in forbidden for sid, o in base.items()):
            assert list(result.items()) == list(base.items())
