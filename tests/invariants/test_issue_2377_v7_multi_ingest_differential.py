"""Seeded multi-ingest differential for #2377 v7 (release-manager condition, Codex P1).

The single-ingest differential seeds the store with one fresh ingest, so it never sees a
replay copy that v7 stored and linked in an EARLIER ingest. Each plan here runs 2-4
ingests of one growing true conversation: warm (same engine, positional cursor) or cold
(new engine, cursor pinned to a true replay boundary), with stamped, unstamped and
re-stamped presentations, rewritten results, and genuine repeats of the same call.

Every plan runs twice on the same tree: v7 as shipped, and "main mode", which replaces
the tool-segment scan with main's frozen predicate (the ceiling hook) and writes no v7
tool links. Each view is a prefix of the true rows, so a host index IS its true
occurrence.
- No new loss, per occurrence over every ingest of the plan: an occurrence main mode ever
  stored, v7 stored too. Per-key counts cannot check this: a linked replay copy and a lost
  new row cancel.
- No new silent copy, per scorer key over the final store: v7 holds no more rows without a
  link than main mode. Which row of a key carries the link is not observable when no
  equal-stamp evidence exists (the oracle's pooling rule), so this is a count per key.
LCM_MULTI_PLANS sets the plan count (default 600, split over six chunks for xdist).

Ownership variants (blank-id stores and a second conversation): right after the first ingest,
while both runs hold identical rows, a plan relabels some delivered occurrences to one value: NULL,
'', whitespace (legacy blank ids, which the active conversation owns, #581) or another
conversation. Rows never change owner later, as on a real upgraded or shared store. The relabel is applied by true occurrence
in both runs. A link into a row of another conversation is a failure of its own. Main's predicate
may skip a row on another conversation's twin (a loss on main); v7 stores it and must not link
across conversations. The variant silent check is strict, per key inside this conversation (blank
and active spellings are one owner): v7's unlinked rows, less its OWN rows, may not exceed main's
unlinked rows. An own row is an unlinked row of an (occurrence, key) that main never held in this
conversation, so main lost it and v7 kept it; a linked row is never an own row. In a plan with
another named conversation, one strict flag shape is exempt and counted (the disclosed limit): v7
stored a row of that key that main lost, the link went to that new row, and per key links equal
copies (rows - occurrences), so a later replay of the key is the unlinked one. Any other flag fails.
"""
from copy import deepcopy
from collections import Counter, defaultdict
import json
import os
import random
import time

import pytest

from tests.harness import make_engine
from tests.invariants._ingest_shapes_2377 import U, call, key, relations

pytestmark = pytest.mark.issue(2377)

PLANS = int(os.environ.get("LCM_MULTI_PLANS", "600"))
VARIANT_PLANS = int(os.environ.get("LCM_MULTI_VARIANT_PLANS", "300"))
VARIANT_FIRST_SEED = 63770
RELABELS = ("NULL", "''", "'  '", "'other'")
CHUNKS = 6
FIRST_SEED = 23770


def generate(seed, variant=False):
    """A plan: the true rows, then 2-4 ingests, each (view, mode, cursor[, relabel])."""
    rng = random.Random(seed)
    relabel_rng = random.Random(seed * 7919 + 1)  # a separate stream: base plans stay unchanged
    label = relabel_rng.choice(RELABELS)  # one owner per plan: a row never moves between conversations twice
    alphabet = []
    for identity in range(rng.randint(1, 2)):
        ids = tuple(f"id-{identity}-{i}" for i in range(rng.randint(1, 2)))
        alphabet.append((ids, json.dumps({"identity": identity})))
    true, segments = [deepcopy(U)], []  # segments: (start, length) in true

    def add_segment():
        ids, args = alphabet[rng.randrange(len(alphabet))]
        stamp = rng.choice((None, float(len(true) + 2)))
        rows = [call(stamp, ids, args), *[dict(role="tool", tool_call_id=cid, content=f"result {cid}",
                                               timestamp=rng.choice((None, stamp))) for cid in ids]]
        segments.append((len(true), len(rows)))
        true.extend(rows)

    ingests, delivered = [], 0
    for number in range(rng.randint(2, 4)):
        for _ in range(rng.randint(1, 2) if number else rng.randint(1, 3)):
            add_segment()
        true.append({"role": "user", "content": f"turn {number}", "timestamp": 1000.0 + number})
        view = deepcopy(true)
        if number:
            # Presentation changes only touch segments already delivered (replays).
            for start, length in segments:
                if start + length > delivered:
                    continue
                mode = rng.choice(("same", "same", "absent", "restamped"))
                for row in view[start:start + length]:
                    if mode == "absent":
                        row.pop("timestamp", None)
                    elif mode == "restamped":
                        row["timestamp"] = float(100 + start + 10 * number)
                if length > 1 and rng.randrange(4) == 0:
                    at = start + rng.randrange(1, length)
                    view[at]["content"] = f"rewritten {start} {at} {number}"
        warm = number > 0 and rng.random() < 0.4
        boundaries = [1] + [s for s, _length in segments if s <= delivered] + [
            s + length for s, length in segments if s + length <= delivered]
        cursor = rng.choice(sorted(set(b for b in boundaries if 1 <= b <= delivered))) if number else 0
        ingests.append(dict(view=view, warm=warm, cursor=cursor))
        delivered = len(true)
        if variant and number == 0:
            chosen = sorted(relabel_rng.sample(range(delivered), relabel_rng.randint(1, delivered)))
            ingests[-1]["relabel"] = (label, chosen)
    return ingests


def run_plan(ingests, folder, main_mode):
    folder.mkdir(parents=True, exist_ok=True)
    engines = []

    def build():
        engine = make_engine(folder, window=200_000, session_id="S", conversation_id="conversation",
                             fresh_tail_count=9, leaf_chunk_tokens=400,
                             large_output_externalization_enabled=False,
                             large_output_active_replay_stubbing_enabled=False,
                             temporal_rollups_enabled=False, embeddings_enabled=False,
                             deferred_maintenance_enabled=False)
        engines.append(engine)
        if main_mode:
            def main_scan(messages, cursor, *, consumed_store_ids=(), consumed_pairs=None):
                engine._tool_cursor_spent, engine._tool_borrow_proofs = set(), {}
                if not engine._session_id or cursor >= len(messages):
                    return set()
                tail = engine._store.get_session_tail(engine._session_id, limit=max(len(messages) * 4, 64))
                return set(engine._main_tool_segment_skip_ceiling(
                    messages, cursor, consumed_store_ids=consumed_store_ids,
                    consumed_pairs=consumed_pairs, tail_rows=tail))
            engine._replayed_tool_segment_indexes_after_cursor = main_scan
            engine._prepare_tool_replay_links = lambda *a, **kw: []
        return engine

    stored = defaultdict(list)  # true occurrence -> store ids, in ingest order
    foreign, seen = [], set()  # links into another conversation's row, judged when each link is written
    try:
        engine = None
        with pytest.MonkeyPatch.context() as patch:
            for step in ingests:
                if engine is None or not step["warm"]:
                    if engine is not None:
                        engine.shutdown()
                    engine = build()
                    write = engine._store_ingest_rows

                    def capture(write=write, **kwargs):
                        ids = write(**kwargs)
                        for (idx, _row), sid in zip(kwargs["messages_to_store_with_index"], ids):
                            stored[idx].append(sid)
                        return ids
                    engine._store_ingest_rows = capture
                    if step["cursor"]:
                        at = step["cursor"]
                        patch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, at=at, **kw: at)
                        patch.setattr(engine, "_ledger_cold_cursor", lambda *a, at=at, **kw: dict(k=at, pairs={}))
                engine.ingest(deepcopy(step["view"]))
                assert engine._ingest_failure_count == 0, engine._last_ingest_error
                owner = dict(engine._store._conn.execute("SELECT store_id, conversation_id FROM messages"))
                for link in relations(engine):
                    if link not in seen:
                        seen.add(link)
                        if link[1] == "possible_replay_of" and (owner.get(link[2]) or "").strip() not in (
                                "", "conversation"):
                            foreign.append(link)
                if step.get("relabel"):
                    value, chosen = step["relabel"]
                    ids = [sid for idx in chosen for sid in stored.get(idx, ())]
                    conn = engine._store._conn
                    conn.execute(f"UPDATE messages SET conversation_id = {value} WHERE store_id IN "
                                 "(SELECT value FROM json_each(?))", (json.dumps(ids),))
                    conn.commit()
        assert len(engine._store.get_session_messages("S")) == sum(map(len, stored.values())), "capture missed rows"
        sources = {s for s, _kind, _t in relations(engine)}
        rows = engine._store.get_session_messages("S")
        unlinked = Counter(key(r) for r in rows if r["store_id"] not in sources)
        # This conversation's rows (blank ids included), keyed without the conversation spelling.
        local = {r["store_id"]: key(r)[1:] for r in rows if (r["conversation_id"] or "").strip() in ("", "conversation")}
        pairs = [(idx, local[sid], sid in sources) for idx, ids in stored.items() for sid in ids if sid in local]
        ownership = dict(
            unlinked=Counter(k for sid, k in local.items() if sid not in sources),
            held={(idx, k) for idx, k, _linked in pairs},
            own={(idx, k) for idx, k, linked in pairs if not linked},
            linked_pairs={(idx, k) for idx, k, linked in pairs if linked},
            rows=Counter(local.values()), links=Counter(k for sid, k in local.items() if sid in sources),
            occurrences=Counter(k for _idx, k in {(idx, k) for idx, k, _linked in pairs}))
        return set(stored), unlinked, sorted(foreign), ownership
    finally:
        for engine in engines:
            engine.shutdown()


def check(seed, root, variant=False):
    ingests = generate(seed, variant)
    v7 = run_plan(ingests, root / f"v7-{seed}", main_mode=False)
    main = run_plan(ingests, root / f"main-{seed}", main_mode=True)
    loss, silent = compare(v7, main)
    if not variant:
        return loss, silent
    flags, exempt = ownership_flags(v7[3], main[3], other=any(
        (step.get("relabel") or ("",))[0] == "'other'" for step in ingests))
    return loss, flags, v7[2], exempt


def ownership_flags(v7, main, other):
    """Strict per-key silent check inside the conversation, with the one exempt shape for 'other' plans."""
    # An (occurrence, key) with any linked row has no own row: its unlinked rows are copies.
    own = Counter(k for _idx, k in (v7["own"] - v7["linked_pairs"]) - main["held"])
    flags, exempt = [], []
    for k, n in v7["unlinked"].items():
        if n - own[k] <= main["unlinked"][k]:
            continue
        # Exempt at most one unlinked copy per link held by a row main lost (the disclosed shape).
        new_linked = len({pair for pair in v7["linked_pairs"] - main["held"] if pair[1] == k})
        if other and new_linked and n - own[k] - main["unlinked"][k] <= new_linked:
            exempt.append(k[1])
        else:
            flags.append(k[1])
    return sorted(flags), sorted(exempt)


def compare(v7, main):
    loss = sorted(main[0] - v7[0])
    silent = sorted(k[2] for k, n in v7[1].items() if n > main[1][k])
    return loss, silent


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_multi_ingest_differential(tmp_path, chunk, record_property):
    seeds = range(FIRST_SEED + chunk, FIRST_SEED + PLANS, CHUNKS)
    started = time.perf_counter()
    failures = {"loss": [], "silent": []}
    for seed in seeds:
        loss, silent = check(seed, tmp_path)
        if loss:
            failures["loss"].append(seed)
        if silent:
            failures["silent"].append(seed)
    result = dict(chunk=chunk, plans=len(seeds), seconds=round(time.perf_counter() - started, 3),
                  failed={k: len(v) for k, v in failures.items()}, first_ten={k: v[:10] for k, v in failures.items()})
    record_property("multi_ingest_result", json.dumps(result, sort_keys=True))
    print("\nMULTI_INGEST_RESULT " + json.dumps(result, sort_keys=True))
    assert not any(failures.values()), json.dumps(result, sort_keys=True)


def test_multi_ingest_catches_the_codex_p1_chain(tmp_path):
    """The A31 chain, written as a plan: the oracle must flag it when capacity counts linked copies."""
    X = {"role": "tool", "tool_call_id": "c", "content": "X"}
    plan = [dict(view=[U, call(2), dict(X, timestamp=2)], warm=False, cursor=0),
            dict(view=[U, call(4), dict(X, timestamp=4), {"role": "user", "content": "next", "timestamp": 1000.0}],
                 warm=False, cursor=1),
            dict(view=[U, call(), X, call(), X, {"role": "user", "content": "next2", "timestamp": 2000.0}],
                 warm=False, cursor=1)]
    loss, silent = compare(run_plan(plan, tmp_path / "v7", main_mode=False),
                           run_plan(plan, tmp_path / "main", main_mode=True))
    assert (loss, silent) == ([], [])


def test_multi_ingest_seed_26092_cross_class_copy(tmp_path):
    """A replay copy linked to an original in ANOTHER class (re-delivered unstamped) must not fund a skip:
    the unstamped class held that copy plus two new pairs, and a later new A@16 pair was lost."""
    assert check(26092, tmp_path) == ([], [])


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_multi_ingest_ownership_variants(tmp_path, chunk, record_property):
    """Blank-id stores and a second conversation: no loss, no unlinked extra, no foreign link."""
    seeds = range(VARIANT_FIRST_SEED + chunk, VARIANT_FIRST_SEED + VARIANT_PLANS, CHUNKS)
    started = time.perf_counter()
    failures = {"loss": [], "silent": [], "foreign": []}
    exempt = []
    for seed in seeds:
        loss, silent, foreign, shape = check(seed, tmp_path, variant=True)
        for name, found in zip(("loss", "silent", "foreign"), (loss, silent, foreign)):
            if found:
                failures[name].append(seed)
        if shape:
            exempt.append(seed)
    result = dict(chunk=chunk, plans=len(seeds), seconds=round(time.perf_counter() - started, 3),
                  failed={k: len(v) for k, v in failures.items()}, first_ten={k: v[:10] for k, v in failures.items()},
                  exempt_disclosed_shape=len(exempt), exempt_seeds=exempt[:10])
    record_property("multi_ingest_variant_result", json.dumps(result, sort_keys=True))
    print("\nMULTI_INGEST_VARIANT_RESULT " + json.dumps(result, sort_keys=True))
    assert not any(failures.values()), json.dumps(result, sort_keys=True)


@pytest.mark.parametrize("value", ["NULL", "''", "'  '"], ids=["null", "empty", "spaces"])
def test_multi_ingest_blank_id_copy_is_linked(tmp_path, value):
    """Astra r5: an unlinked replay copy of a blank-id original funded a later skip, and the new pair
    was lost. Blank ids belong to the active conversation (#581), so the copy is linked and adds no capacity."""
    X = {"role": "tool", "tool_call_id": "c", "content": "X"}
    plan = [dict(view=[U, call(2), dict(X, timestamp=2)], warm=False, cursor=0, relabel=(value, [0, 1, 2])),
            dict(view=[U, call(4), dict(X, timestamp=4), {"role": "user", "content": "next", "timestamp": 1000.0}],
                 warm=False, cursor=1),
            dict(view=[U, call(), X, call(), X, {"role": "user", "content": "next2", "timestamp": 2000.0}],
                 warm=False, cursor=1)]
    v7 = run_plan(plan, tmp_path / "v7", main_mode=False)
    assert compare(v7, run_plan(plan, tmp_path / "main", main_mode=True)) == ([], [])
    assert v7[2] == []


def test_multi_ingest_foreign_original_is_never_a_target(tmp_path):
    """The same chain with the original in another conversation: no link crosses into it."""
    X = {"role": "tool", "tool_call_id": "c", "content": "X"}
    plan = [dict(view=[U, call(2), dict(X, timestamp=2)], warm=False, cursor=0, relabel=("'other'", [1, 2])),
            dict(view=[U, call(4), dict(X, timestamp=4), {"role": "user", "content": "next", "timestamp": 1000.0}],
                 warm=False, cursor=1)]
    v7 = run_plan(plan, tmp_path / "v7", main_mode=False)
    main = run_plan(plan, tmp_path / "main", main_mode=True)
    # Main skips the pair on the other conversation's row; v7 stores it, unlinked: it is new to this
    # conversation, and a link would cross conversations.
    assert compare(v7, main) == ([], ["", "X"])
    assert ownership_flags(v7[3], main[3], other=True) == ([], [])
    assert v7[2] == []
