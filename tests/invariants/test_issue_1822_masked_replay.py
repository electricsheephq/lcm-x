"""I1/I2: masked replay decisions and complete stored histories equal full replay."""
from copy import deepcopy
import random
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.invariants.test_issue_1628_rewind_rescan import LOOP, MARK, SESSION, _ContainsPattern, engine as engine


def history(seed):
    rng = random.Random(seed)
    messages = [{"role": "user" if idx % 2 == 0 else "assistant",
                 "content": f"seed {seed} row {idx}", "timestamp": float(100 + idx)}
                for idx in range(rng.randint(3, 12))]
    # Equal occurrences must retain their full-list ordinal in the store map.
    if seed % 3 == 0:
        messages[1] = deepcopy(messages[0])
    if seed % 5 == 0:
        messages[0]["content"] = "  " + messages[0]["content"] + "  "
    changed = rng.randrange(len(messages))
    messages[changed]["content"] += " original"
    return messages, changed, rng


def exercise(seed, root, *, full=False, engine_class=LCMEngine, decisions=True):
    messages, changed, rng = history(seed)
    config = LCMConfig(database_path=str(root / "lcm.db"), embeddings_enabled=False,
                       large_output_externalization_enabled=False,
                       large_output_externalization_path=str(root.parent / "payloads"))
    engine = engine_class(config=config, hermes_home=str(root.parent / "home"))
    engine.on_session_start("s1822", platform="cli", context_length=1_000_000)
    try:
        engine._ingest_messages(messages)
        if seed % 2:
            engine._compiled_ignore_message_patterns = [_ContainsPattern(MARK)]
        messages[changed]["content"] = (f"{MARK} revised {seed}" if seed % 4 < 2
                                         else LOOP if messages[changed]["role"] == "assistant"
                                         else f"revised {seed}")
        if seed % 5 == 0 and changed != 0:
            messages[0]["content"] = messages[0]["content"].strip()
        if seed % 7 == 0:
            messages[-1]["timestamp"] += 1000.0
        messages += [{"role": "user", "content": f"tail {seed}", "timestamp": 200.0},
                     {"role": "assistant", "content": "Done.", "timestamp": 201.0}]
        build = engine._build_ingest_replay
        if decisions:
            cursor = rng.randrange(changed + 1)
            only = {changed} | {idx for idx in range(cursor, len(messages)) if rng.random() < .3}
            expected = build(messages=messages, n=len(messages), cursor=cursor)
            actual = build(messages=messages, n=len(messages), cursor=cursor, only=only,
                                                  store_ids=getattr(expected[1], "store_ids", None))
            assert actual[0] == expected[0]
            for idx in only:
                assert (actual[1][idx], actual[2][idx]) == (expected[1][idx], expected[2][idx]), (seed, idx)
            assert all(not actual[1][idx] for idx in range(len(messages)) if idx not in only)
            assert all(actual[2][idx] == messages[idx] for idx in range(len(messages)) if idx not in only)
        if full:
            def unmasked(**kwargs):
                kwargs.pop("only", None)
                kwargs.pop("store_ids", None)
                return build(**kwargs)
            engine._build_ingest_replay = unmasked
        engine._ingest_messages(messages)
        rows = engine._store.get_session_messages("s1822")
        # A second ingest must not amplify the histories either.
        engine._ingest_messages(messages)
        assert engine._store.get_session_messages("s1822") == rows
        return rows
    finally:
        engine.shutdown()


@pytest.mark.parametrize("seed", range(64))
def test_masked_decisions_and_stored_rows_equal_full(seed, tmp_path, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)
    actual = exercise(seed, tmp_path / "masked")
    expected = exercise(seed, tmp_path / "full", full=True, decisions=False)
    assert actual == expected


def test_merge_head_override_invalidates_initial_full_list_map(engine, monkeypatch):
    engine._ingest_messages([{"role": "user", "content": "A", "timestamp": 100.0},
                             {"role": "user", "content": "other", "timestamp": 101.0}])
    engine._compiled_ignore_message_patterns = [SimpleNamespace(
        pattern="^A$", search=lambda text, timeout=None: text == "A")]
    build = engine._build_ingest_replay
    rescans = []

    def record(**kwargs):
        result = build(**kwargs)
        if kwargs.get("only") is not None:
            rescans.append((kwargs.get("store_ids"), list(result[1])))
        return result

    monkeypatch.setattr(engine, "_build_ingest_replay", record)
    engine._ingest_messages([
        {"role": "user", "content": " A ", "timestamp": 999.0},
        {"role": "user", "content": " A \n\nT", "timestamp": 100.0, "_merged_turn_prefix": " A "},
        {"role": "user", "content": "T", "timestamp": 200.0},
    ])
    assert rescans == [(None, [True, False, False])]
    assert [(row["role"], row["content"]) for row in engine._store.get_session_messages(SESSION)] == [
        ("user", "A"), ("user", "other"), ("user", " A \n\nT"), ("user", "T")]
    assert engine._ignore_pattern_dropped_count == 1
