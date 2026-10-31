"""R3 diagnostic keys equal the frozen r2 callback on seeded synthetic shapes."""
from collections import defaultdict
from copy import deepcopy
import random

import pytest

from tests.invariants.test_issue_1519_alias_uid_conflict import deterministic_host, message
from tests.invariants.test_issue_1780_round3 import context

__all__ = ["deterministic_host"]


def r2_diagnostic_keys(engine, forms_of, occurrences, identity_messages):
    """Copied r2 implementation; independent predicate pins the oracle as well."""
    def keys_of(r):
        return {(float(r["observed_at"]), form) for form in forms_of[id(r)]}

    def _ws_equivalent(stored, identity):
        return (identity[0] == "user" and bool(identity[1].strip()) and stored != identity
                and stored[1].strip() == identity[1].strip() and tuple(stored[2:]) == tuple(identity[2:]))

    ws_index = None

    def diag_keys_of(r):
        nonlocal ws_index
        if ws_index is None:  # Only reached after the observer's mode/valid-UID gates.
            from hermes_lcm.host_uid import _valid_uid

            ws_index = defaultdict(set)
            for idx, key in occurrences:
                if not _valid_uid(identity_messages[idx].get("message_uid")):
                    continue
                ident = key[1]
                if ident[0] == "user" and ident[1].strip():
                    ws_index[(key[0], ident[1].strip(), tuple(ident[2:]))].add(key)
        keys = keys_of(r)
        if r.get("role") == "user":
            stored = engine._message_replay_identity(r, stored_row=True)
            bucket = (float(r["observed_at"]), stored[1].strip(), tuple(stored[2:]))
            keys.update(key for key in ws_index.get(bucket, ()) if _ws_equivalent(stored, key[1]))
        return keys

    return diag_keys_of


def test_seeded_r3_keys_equal_r2_and_reuse_singletons(tmp_path, monkeypatch, record_property):
    rng = random.Random(1787032026)
    stats = defaultdict(int)
    pending_rows = []
    forms_of = {}
    phase = None

    class KeysCompared(Exception):
        pass

    with context(tmp_path) as eng:
        identity = eng._message_replay_identity
        stored_forms = eng._stored_row_forms

        def count_identity(row, **kwargs):
            if phase == "candidate":
                stats["candidate_identity_calls"] += 1
            return identity(row, **kwargs)

        def forms(row):
            result = stored_forms(row)
            if row.get("extra_form"):
                original = identity(row, stored_row=True)
                result.add((original[0], "override fixture", *original[2:]))
            forms_of[id(row)] = result
            return result

        def compare(rows, canonical_stamps, keys_of, occurrences, messages):
            nonlocal phase
            oracle = r2_diagnostic_keys(eng, forms_of, occurrences, messages)
            for row in rows:
                phase = "candidate"
                before = stats["candidate_identity_calls"]
                actual = keys_of(row)
                calls = stats["candidate_identity_calls"] - before
                expected_calls = int(row["role"] == "user" and len(forms_of[id(row)]) != 1)
                assert calls == expected_calls
                stats["singleton_user_rows"] += int(row["role"] == "user" and not expected_calls)
                stats["multiple_form_user_rows"] += expected_calls
                phase = "oracle"
                expected = oracle(row)
                assert actual == expected
                original = {(float(row["observed_at"]), form) for form in forms_of[id(row)]}
                stats["ws_additions"] += len(actual - original)
                stats["rows"] += 1
            phase = None
            stats["shapes"] += 1
            raise KeysCompared

        monkeypatch.setattr(eng, "_message_replay_identity", count_identity)
        monkeypatch.setattr(eng, "_stored_row_forms", forms)
        monkeypatch.setattr(eng, "_identity_anchor_rewritten", lambda _messages, _cursor: set())
        monkeypatch.setattr(eng, "_load_host_rewrite_overrides", lambda _rows: None)
        monkeypatch.setattr(eng._store, "find_rows_by_observed_at", lambda *_args: pending_rows)
        monkeypatch.setattr(eng, "_observe_alias_uid_conflicts", compare)
        for _shape in range(2048):
            view = []
            for i in range(rng.randrange(4, 13)):
                text = rng.choice(["shared", "other", "", " ", "fixture α"])
                content = rng.choice(["", " ", "\t"]) + text + rng.choice(["", "\n", "  "])
                kind = rng.randrange(5)
                if kind == 0:
                    content = [{"type": "text", "text": content}]
                elif kind == 1:
                    content = {"fixture": content}
                uid = rng.choice([None, "", 123, [], "x" * 257, "uid-a", "uid-b"])
                item = message(content, float(rng.randrange(3)), rng.choice(["user", "user", "assistant"]),
                               tool_call_id=rng.choice(["", "tail-a", "tail-b"]))
                if uid is not None:
                    item["message_uid"] = uid
                view.append(item)
                stats["non_string_content"] += int(not isinstance(content, str))
            # Pin one qualifying mixed-whitespace pair in every shape, alongside random siblings.
            view.append(message("  shared\n", 1., message_uid="uid-conflict"))
            pending_rows.clear()
            forms_of.clear()
            for i in range(rng.randrange(4, 13)):
                row = deepcopy(rng.choice(view))
                if isinstance(row["content"], str) and rng.choice([False, True]):
                    row["content"] = rng.choice(["", "\n"]) + row["content"].strip() + rng.choice(["", "\t"])
                row.update(store_id=i + 1, observed_at=float(rng.randrange(3)), extra_form=rng.randrange(4) == 0)
                pending_rows.append(row)
            pending_rows.append({"role": "user", "content": "shared", "store_id": 100, "observed_at": 1.})
            with pytest.raises(KeysCompared):
                eng._identity_anchor_prematch(view, view, 0)
    assert stats["shapes"] == 2048
    assert stats["ws_additions"] >= 2048
    assert stats["singleton_user_rows"] > 0
    assert stats["multiple_form_user_rows"] > 0
    assert stats["non_string_content"] > 0
    record_property("r3_r2_key_equivalence", dict(stats))
