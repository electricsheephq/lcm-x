"""Occurrence-grounded #1232 v2 twins for SPEC-L2a-5 D, engine-direct.

Ports the 256 twins and 192 C1 cells, other-tool and upgrade probes; expands
the uid modes and adds the third view using the classifier's own write plan.
Source fixtures: #1232 v2 design probes; Astra's two collision traces; #2259.
Distinct-occurrence stamps prove this fixture's witness model only.
"""
from copy import deepcopy
from itertools import product
import json

import pytest

from hermes_lcm import ingest_classifier as classifier, ingest_ledger as ledger
from tests.invariants.test_p2_classifier import (
    classify, factory as classifier_factory, materialize, possible, snapshot, verdicts,
)

factory = classifier_factory


P = "[Note: model was just switched from X to Y]\n\nok"
MODES = ("off", "shadow", "on")
LIVE = ("name_only", "named")


def A(cid="c", name=None):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {
            "name": name or ("read" if cid == "c" else "other"), "arguments": "{}"}}]}


def T(live="named", text="ok", cid="c", name="read"):
    return {"role": "tool", "content": text, "tool_call_id": cid, "name": name,
            **({"tool_name": name} if live == "named" else {})}


def NL(text="ok", cid="c"):
    return {"role": "tool", "content": text, "tool_call_id": cid}


def U(text):
    return {"role": "user", "content": text}


def D(text="done"):
    return {"role": "assistant", "content": text}


def witness(occ, message, kind):
    row = deepcopy(message)
    if row.pop("_nowit", None):
        return row
    if kind == "stamp" or (kind == "gw" and row["role"] == "user"):
        row["timestamp"] = float(1000 + occ)
    if kind in ("uid", "gw"):
        row["message_uid"] = f"u{occ:031d}"
    return row


def reload_shape(row, shape):
    row = deepcopy(row)
    if row["role"] == "tool" and shape not in ("same", "warm"):
        had = "tool_name" in row
        name = row.pop("name", None)
        row.pop("tool_name", None)
        if shape == "P" or had:
            row["tool_name"] = name
    return row


def cases(live):
    def t(text="ok"):
        return T(live, text)

    def r():
        return T("named", "r", "b", "other")

    return {
        "D3_1": ([(1, A()), (2, t()), (3, U("again"))],
                 [(1, A()), (2, t()), (3, U("again"))],
                 [(4, A()), (5, t()), (6, U("again"))], "shadow"),
        "D3_2a_result_first": ([(1, t()), (2, A()), (3, U("again"))],
                               [(1, t()), (2, A()), (3, U("again"))],
                               [(4, t()), (5, A()), (6, U("again"))], "shadow"),
        "D3_2b_E_issuer_outside": ([(1, t()), (2, U("again"))],
                                   [(1, t()), (2, U("again"))],
                                   [(3, t()), (4, U("again"))], "shadow"),
        "C4_switch_note": ([(1, A()), (2, t(P)), (3, D())],
                           [(1, A()), (2, NL()), (3, D()), (4, t()), (5, U("next"))],
                           [(6, A()), (7, NL()), (8, D()), (9, t()), (10, U("next"))], "shadow"),
        "SUP_suppressor": ([(1, A("b")), (2, r()), (3, U("old")), (4, A()), (5, t()), (6, D())],
                           [(4, A()), (5, t()), (6, D()), (7, A("b")), (8, r()), (9, U("new"))],
                           [(10, A()), (11, t()), (12, D()), (13, A("b")), (14, r()), (15, U("new"))], "shadow"),
        "RST_restart": ([(1, U("question")), (2, A()), (3, t("result")), (4, D("answer"))],
                        [(1, U("question")), (2, A()), (3, t("result")), (4, D("answer")), (5, U("next"))],
                        [(6, U("question")), (7, A()), (8, t("result")), (9, D("answer")), (10, U("next"))], "shadow"),
        "MIX_unwitnessed_neighbour": (
            [(1, U("question")), (2, A()), (3, t("result")), (4, dict(D("answer"), _nowit=1))],
            [(1, U("question")), (2, A()), (3, t("result")), (4, dict(D("answer"), _nowit=1)), (5, U("next"))],
            [(1, U("question")), (2, A()), (3, t("result")), (6, dict(D("answer"), _nowit=1)), (7, U("next"))], "shadow"),
        "PRE_preledger": ([(1, U("question")), (2, A()), (3, t("result")), (4, D("answer"))],
                          [(1, U("question")), (2, A()), (3, t("result")), (4, D("answer")), (5, U("next"))],
                          [(6, U("question")), (7, A()), (8, t("result")), (9, D("answer")), (10, U("next"))], "off"),
    }


def m2(row, target, snap, mode, unreliable):
    """Evaluate the fixture's keys independently of the returned evidence label."""
    if target not in snap.bindings:
        return False
    bound = snap.bindings[target]
    incoming_stamp = ledger.stamp_text(row.get("timestamp"))
    if not unreliable and (incoming_stamp is not None or snap.stamps(target)):
        return False
    known = {row["message_uid"], *row.get("_absorbed_message_uids", ())} if row.get("message_uid") else set()
    stored = bound["uids"] | bound["constituents"]
    if mode != "off" and known and stored and known != stored:
        return False
    if mode == "on":
        for uid in known | bound["uids"]:
            if any(not u["disagree_seen"] and (u["proof_kind"] == "stored_new" or any(
                    b["store_id"] == u["store_id"] for b in snap.bindings.values()))
                   for u in snap.uid_bindings.get(uid, ()) if u["kind"] in ("canonical", "version")):
                return False
    return ledger.identity_digest(snap.salt, row) == bound["identity_digest"]


def check_cell(factory, first, second, *, kind, mode, shape="same", ledger1="shadow", label="cell", evict=False):
    """Return the second plan; assert every cell without a failure cap."""
    history_rows = first + ([(90, U("evict1")), (91, U("evict2"))] if evict else [])
    seen = {occ for occ, _ in history_rows}
    v1 = [witness(occ, row, kind) for occ, row in first]
    v2 = [witness(occ, reload_shape(row, shape) if occ in seen else row, kind) for occ, row in second]
    engines = []
    for name in ("module", "base"):
        engine = factory(name, mode, ledger_mode=ledger1)
        engine._ingest_messages(deepcopy(v1))
        if evict:
            for occ, row in history_rows[len(first):]:
                engine._ingest_messages([witness(occ, row, kind)])
        if shape != "warm":
            engine = factory(name, mode, cold_of=engine)
        engines.append(engine)
    engine, base = engines
    snap = snapshot(engine)
    initial = engine._store.get_session_messages("S")
    assert len(initial) == len(history_rows), "fixture must assign each initial store row independently"
    ground = {row["store_id"]: occ for row, (occ, _) in zip(initial, history_rows)}
    result = classify(engine, v2, mode, snap=snap)
    base._ingest_messages(deepcopy(v2))
    base_count = len(base._store.get_session_messages("S"))
    stored_count = len(initial) + sum(v.verdict == "new" for v in result.rows)
    table = dict(cell=label, witness=kind, uid_mode=mode, shape=shape, base=base_count, module=stored_count,
                 expected=len(seen | {o for o, _ in second}), coverage=result.coverage.kind,
                 rows=[dict(occ=o, verdict=v.verdict, evidence=v.evidence, target=v.target,
                            reason=v.reason, possible=possible(v)) for (o, _), v in zip(second, result.rows)])
    print("TWIN " + json.dumps(table, sort_keys=True))
    # Snapshot truth, rather than the fixture's pre-ledger origin, decides coverage.
    unbound = engine._store._conn.execute(
        "SELECT count(*) FROM messages m WHERE session_id='S' AND NOT EXISTS "
        "(SELECT 1 FROM occurrence_bindings b WHERE b.store_id=m.store_id)").fetchone()[0]
    assert result.coverage == classifier.Coverage("partial" if unbound else "complete", unbound)
    cold_c1_preledger = label.startswith("C1/") and shape != "warm" and ledger1 == "off"
    if label.startswith("C1/"):
        assert bool(unbound) == cold_c1_preledger, ("HARD_BAR C1 partial coverage class", table)
    assert not unbound or cold_c1_preledger or label.startswith("PRE_preledger/") or label == "UPGRADE", (
        "HARD_BAR unexpected partial coverage", table)
    explanations, replay_targets = [], set()
    if not unbound:  # Full pre-ledger acceptance belongs to L2b.
        for i, ((occ, raw), row, verdict) in enumerate(zip(second, v2, result.rows)):
            is_new = occ not in seen
            if verdict.verdict == "replay":
                target_occ = ground[snap.bindings[verdict.target]["store_id"]]
                verified_m2 = m2(row, verdict.target, snap, mode, result.writes.stamp_unreliable or
                                 snap.stamp_policy == "stamp_unreliable")
                assert verdict.target not in replay_targets, ("HARD_BAR reused target", i, table)
                replay_targets.add(verdict.target)
                # Item 12: occurrence ids remain ground truth for freshness, but
                # old rows with identical identity/stamp/uid may exchange targets.
                originals = [b for b in snap.bindings.values() if ground[b["store_id"]] == occ]
                indistinguishable = not is_new and any(
                    b["identity_digest"] == snap.bindings[verdict.target]["identity_digest"]
                    and b["stamp"] == snap.bindings[verdict.target]["stamp"]
                    and (mode == "off" or b["uids"] | b["constituents"] ==
                         snap.bindings[verdict.target]["uids"] | snap.bindings[verdict.target]["constituents"])
                    for b in originals)
                assert target_occ == occ or (is_new and verified_m2) or indistinguishable, ("HARD_BAR loss", i, table)
                explanations.append(dict(row=i, proof="M2" if verified_m2 else "occurrence", target_occ=target_occ))
            elif not is_new:
                assert possible(verdict), ("HARD_BAR silent surplus", i, table)
                # C4's shadow uid/gateway aliases are explicitly unwitnessed.
                c4 = label.startswith("C4_switch_note") and (kind == "none" or
                     (kind in ("uid", "gw") and mode != "on"))
                raw_unkeyed = raw.get("_nowit") or kind == "none" or (kind == "uid" and mode == "off")
                assert c4 or raw_unkeyed, ("HARD_BAR witnessed surplus", i, table)
    if stored_count < base_count:
        print("REDUCTION " + json.dumps(dict(cell=label, uid_mode=mode, witness=kind, shape=shape,
                                            base=base_count, module=stored_count, attribution=explanations), sort_keys=True))
    stored = materialize(engine, v2, result)
    actual = engine._store.get_session_messages("S")
    expected_rows = initial + [dict(v2[i], store_id=store_id) for i, store_id in stored.items()]
    fields = ("store_id", "role", "content", "tool_call_id", "tool_name")
    assert [tuple(r.get(k) or "" for k in fields) for r in actual] == [
        tuple(r.get(k) or "" for k in fields) for r in expected_rows], ("HARD_BAR stored rows", table)
    third = classify(engine, v2, mode)
    print("THIRD " + json.dumps(dict(cell=label, witness=kind, uid_mode=mode, shape=shape,
                                    verdicts=verdicts(third), reasons=[v.reason for v in third.rows])))
    assert all(v.verdict == "replay" for v in third.rows), ("HARD_BAR third-view surplus", table, third)
    return result, stored_count, base_count, engine


TWIN_CELLS = [(case, live, occurrence, shape, kind, mode)
              for case, live, occurrence, shape, kind, mode in product(
                  cases("named"), LIVE, ("replay", "fresh"), ("same", "P"),
                  ("none", "stamp", "uid", "gw"), MODES)]


@pytest.mark.parametrize("case,live,occurrence,shape,kind,mode", TWIN_CELLS)
def test_classifier_twin(factory, case, live, occurrence, shape, kind, mode):
    first, replay, fresh, ledger1 = cases(live)[case]
    check_cell(factory, first, replay if occurrence == "replay" else fresh, kind=kind, mode=mode,
               shape=shape, ledger1=ledger1, label=f"{case}/{live}/{occurrence}")


@pytest.mark.parametrize("occurrence", ["replay", "fresh"])
def test_gateway_no_run_twin(factory, occurrence):
    first = list(enumerate([U("q"), A(), T(), dict(D(), _nowit=1)], 1))
    second = (first if occurrence == "replay" else
              list(enumerate([U("q"), A(), T(), dict(D(), _nowit=1)], 5))) + [(9, U("next"))]
    result, _, _, _ = check_cell(factory, first, second, kind="gw", mode="off", evict=True,
                                label=f"GATEWAY_no_run/{occurrence}")
    assert result.run is None
    if occurrence == "replay":
        assert result.head.kind == "h2"
        assert verdicts(result) == ["replay", "replay", "replay", "new", "new"]
        assert [v.evidence for v in result.rows[:3]] == ["rank2", "rank3", "rank3"]
        assert result.rows[3].deferred == "rank5" and possible(result.rows[3]) == [4]
    else:
        assert verdicts(result) == ["new"] * 5


def c1(live, seq):
    first = [U("q"), A(), T(live)]
    if seq != "V1_retry_tail":
        first.append(D())
    if seq.startswith(("V5", "V6")):
        first += [U("u2"), D("a2")]
    pairs = list(enumerate(first, 1))
    n = len(first) + 1
    if seq == "V1_retry_tail":
        second = pairs + [(n, T(live)), (n + 1, U("again"))]
    elif seq == "V2_reissue_pair":
        second = pairs + list(enumerate([U("more"), A(), T(live), U("again")], n))
    elif seq == "V3_pair_then_user":
        second = pairs + list(enumerate([A(), T(live), U("again")], n))
    elif seq == "V4_dup_result_mid":
        second = pairs[:3] + [(n, T(live))] + pairs[3:] + [(n + 1, U("again"))]
    else:
        tail = [A(), T(live), *([D()] if seq.startswith("V6") else []), U("again")]
        second = pairs[4:] + list(enumerate(tail, n))
    return pairs, second


C1_CELLS = list(product(LIVE, ("V1_retry_tail", "V2_reissue_pair", "V3_pair_then_user", "V4_dup_result_mid",
                               "V5_unreserved_pair", "V6_unreserved_pair_reply"),
                       ("warm", "same", "P", "Q"), ("none", "stamp"), ("shadow", "off"), MODES))


@pytest.mark.parametrize("live,seq,shape,kind,ledger1,mode", C1_CELLS)
def test_c1_occurrences(factory, live, seq, shape, kind, ledger1, mode):
    if shape == "warm" and ledger1 == "off":
        pytest.skip("L1 ledger-off assertion: no ledger schema exists, so a snapshot cannot be built")
    first, second = c1(live, seq)
    check_cell(factory, first, second, kind=kind, mode=mode, shape=shape, ledger1=ledger1,
               label=f"C1/{live}/{seq}/{ledger1}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["same", "P"])
@pytest.mark.parametrize("kind", ["none", "stamp"])
def test_other_tool(factory, mode, shape, kind):
    first = list(enumerate([U("q"), A(), T("name_only"), D()], 1))
    second = first + list(enumerate([A(name="delete"), T("name_only", name="delete"), U("after delete")], 5))
    check_cell(factory, first, second, kind=kind, mode=mode, shape=shape, label="OTHER")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("live", LIVE)
@pytest.mark.parametrize("shape", ["same", "P", "Q"])
@pytest.mark.parametrize("kind", ["none", "stamp"])
def test_upgrade(factory, mode, live, shape, kind):
    first, _, _, _ = cases(live)["PRE_preledger"]
    second = first + [(5, U("next1"))]
    _, _, _, engine = check_cell(factory, first, second, kind=kind, mode=mode, shape=shape,
                                  ledger1="off", label="UPGRADE")
    # The original probe's next growing view, after the classifier's third view.
    engine = factory("module", mode, cold_of=engine)
    view = [witness(o, reload_shape(m, shape), kind) for o, m in second] + [witness(6, U("next2"), kind)]
    result = classify(engine, view, mode)
    assert verdicts(result) == ["replay"] * 5 + ["new"]


@pytest.mark.parametrize("mode", MODES)
def test_astra_shared_stamp_fresh_uid(factory, mode):
    # The result's issuer is outside the view; the shared stamp is intentional.
    first = [(1, dict(T("name_only"), timestamp=1., message_uid="old")), (2, dict(U("old"), timestamp=2.))]
    second = [(3, dict(T("name_only", name="delete"), timestamp=1., message_uid="fresh")),
              (4, dict(U("new"), timestamp=3.))]
    result, _, _, _ = check_cell(factory, first, second, kind="none", mode=mode, label="ASTRA1")
    assert verdicts(result) == ["new", "new"]


@pytest.mark.parametrize("mode", MODES)
def test_astra_mixed_same_stamp_replays(factory, mode):
    first = [(1, dict(T("name_only"), timestamp=1.)), (2, dict(T(), timestamp=1.)),
             (3, dict(U("old"), timestamp=2.))]
    second = [(4, dict(U("new"), timestamp=3.)), (2, dict(T(), timestamp=1.)),
              (1, dict(NL(), timestamp=1.))]
    result, count, _, _ = check_cell(factory, first, second, kind="none", mode=mode, label="ASTRA2")
    assert count == 4 and verdicts(result) == ["new", "replay", "replay"]
    assert len({v.target for v in result.rows if v.verdict == "replay"}) == 2


@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("kind", ["stamp", "uid", "gw"])
@pytest.mark.parametrize("case,expected", [("D3_1", 6), ("D3_2b_E_issuer_outside", 4),
                                          ("RST_restart", 9), ("SUP_suppressor", 12)])
def test_2259_and_bs_twelve_named_cells(factory, mode, kind, case, expected):
    first, _, fresh, _ = cases("named")[case]
    result, count, base_count, _ = check_cell(factory, first, fresh, kind=kind, mode=mode, label=f"2259/{case}")
    print("BS_CROSSCHECK " + json.dumps(dict(case=case, witness=kind, uid_mode=mode,
                                           expected=expected, module=count, base=base_count)))
    assert verdicts(result) == ["new"] * len(fresh)
    assert count == expected
