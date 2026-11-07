"""Read-only replay-rule audit; stdout only. Run with ``python -m bench.tools.dryrun_hostrewrite``.

The frozen audit exposes todo_expected IDs, but not host_expected matched IDs. Reconstruct the
 latter using its resolve/undo/recovery pipeline and multiset.score's surplus_keys: within each
 assistant key, parity reserves the earliest stored copies and the last surplus copies are
 unmatched. These are the only host_expected keys Arm M can flag; Arm S uses todo_expected IDs.
 Both databases must be stopped/checkpointed, as required by audit.ro (mode=ro&immutable=1).
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from benchmarking.longmemeval import _ensure_hermes_lcm_package
from bench.instruments.gate_kit import audit

_ensure_hermes_lcm_package()
from hermes_lcm.host_rewrite import host_rewrite_match  # noqa: E402
from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX as HEADER, _todo_annotation_span  # noqa: E402


def run(state_db, lcm_db, session=None):
    sessions, host, _ = audit.read_host(state_db)
    stored, _rels, _uids, _lineage, _leaves, nodes = audit.read_store(lcm_db)
    group = audit.lineage_of(sessions)
    markers = audit.plugin_tree.carrier_markers(audit.REPO)
    def strip(text, sid):
        return audit.carrier_remainder(text, markers[0], nodes, lambda s: group(s) == group(sid))
    expected, _, _ = audit.host_expected(host, group, markers, strip, set(sessions))
    resolved, _, _ = audit.externalized.resolve([r[:6] for r in stored], Path(state_db).resolve().parent)
    extra = {r[0]: r for r in stored}
    rows = [(*r, extra[r[0]][6]) for r in resolved]
    retained = set(audit.undo_retained_rows(rows, host, strip))
    recovery, _ = audit.empty_recovery_rows(stored, host, group, strip)
    rows = [r for r in rows if r[0] not in retained | recovery]
    todo = {i for ids in audit.todo_expected(host, rows, group, HEADER).values() for i in ids}
    matched, by_lineage = set(todo), defaultdict(list)
    for r in rows:
        if r[0] not in todo:
            by_lineage[group(r[1])].append(r)
    for g, seq in by_lineage.items():
        surplus = []
        audit.multiset.score(expected.get(g, ([], Counter()))[0], [r[:4] for r in seq], None,
                             surplus_keys=surplus)
        excess = {i for _key, n, ids, *_ in surplus for i in sorted(ids)[-n:]}
        matched.update(r[0] for r in seq if r[2] == "assistant" and r[0] not in excess)
    counts, expected_yes, prefixes = Counter(), 0, defaultdict(list)
    for sid, owner, role, content, calls, _call_id, stamp, *_ in stored:
        prefix = prefixes[owner]
        incoming = dict(role=role, content=content, tool_calls=calls, stamp=stamp)
        candidates = []
        if role == "assistant" and not calls and stamp is not None and isinstance(content, str) and "\n" in content:
            candidates = sorted([r for r in prefix if r["observed_at"] is not None and r["observed_at"] >= stamp],
                                key=lambda r: (r["observed_at"], r["store_id"]))[:64]
        elif role == "user" and stamp is None and isinstance(content, str) and content.startswith(HEADER):
            candidates = [r for r in reversed(prefix) if r["role"] == "user"
                          and (r["content"] or "").startswith(HEADER)][:10000]
        match = host_rewrite_match(incoming, candidates, header=HEADER,
                                  todo_span=_todo_annotation_span(content, 0) if isinstance(content, str) else None)
        if match and (session is None or owner == session):
            counts[match.arm] += 1
            expected_yes += sid in matched
            print(f"store_id={sid} role={role} arm={match.arm} stamp={stamp} length={len(content)} "
                  f"parts={list(match.consumed)} EXPECTED={'yes' if sid in matched else 'no'}")
        previous = next((r["store_id"] for r in reversed(prefix) if r["role"] != "user"), None)
        prefix.append(dict(store_id=sid, role=role, content=content, tool_calls=calls, observed_at=stamp,
                           previous=previous))
    print(f"summary M={counts['M']} S={counts['S']} EXPECTED=yes:{expected_yes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--lcm-db", type=Path, required=True)
    parser.add_argument("--session")
    args = parser.parse_args()
    run(args.state_db, args.lcm_db, args.session)
