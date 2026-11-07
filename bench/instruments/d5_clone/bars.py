"""Bars 1-6 and the raw verdict of the D5 clone run (bench/specs/RELEASE-READINESS-V1.md, "D5 clone run").

Pure functions over plain inputs: store and host rows as dicts, the edition's bind receipts as dicts. Row identity
and the two host-rewrite credits are passed in from the pinned instruments; nothing here defines them. A row's key
is (session, identity), so matching never crosses sessions. Results carry counts, ids and verdicts only.
"""

from collections import Counter
import re

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
BARS = ("bar1", "bar2", "bar3", "bar4", "bar5", "bar6")
# The edition's receipt keys, read by name only: (i) validity, then every (ii) comparison check.
VALIDITY = "validity"
COMPARISONS = ("candidate_subset_of_base", "candidate_non_new_rows_subset_of_base")
CONFLICT = "publication_invariant_conflict"
CHECK_KINDS = ("doctor", "quick_check", "integrity", "schema")


def _worst(verdicts):
    verdicts = list(verdicts)
    return FAIL if FAIL in verdicts else PASS if verdicts and all(v == PASS for v in verdicts) else INCONCLUSIVE


def keys(rows, identity):
    return {(row["session_id"], identity(row)) for row in rows}


def bar2_bind(receipt, *, range_new_rows=None, error=False):
    """One rc-arm cold bind: (i) validity, then (ii) comparison. A missing or NOT SHOWN receipt, a replay error, or
    range_new_rows > 0 is INCONCLUSIVE, never PASS; a bind that passes (i) and fails (ii) fails (spec bar 2)."""
    receipt = receipt if isinstance(receipt, dict) else {}
    checks = receipt.get("checks") or {}
    if range_new_rows is None:
        range_new_rows = receipt.get("range_new_rows")
    out = {"validity": (checks.get(VALIDITY) or {}).get("verdict"), "range_new_rows": range_new_rows,
           "error": bool(error), "comparison": {k: (checks.get(k) or {}).get("verdict") for k in COMPARISONS},
           "copies": len((receipt.get("bind_identity") or {}).get("dup_store_ids") or [])}
    shown = type(range_new_rows) is int and range_new_rows == 0 and out["validity"] == PASS  # Never a bool.
    return {**out, "verdict": _worst(out["comparison"].values()) if shown and not error else INCONCLUSIVE}


def dup_store_ids(receipt):
    """The copies the edition judged at a bind (its positional, credited set), or None when the receipt has none."""
    ids = ((receipt if isinstance(receipt, dict) else {}).get("bind_identity") or {}).get("dup_store_ids")
    return ids if isinstance(ids, list) and all(type(i) is int for i in ids) else None


def bar2(binds):
    return {"verdict": _worst(b["verdict"] for b in binds), "binds": len(binds),
            "by_verdict": dict(Counter(b["verdict"] for b in binds))}


def _ordered(rows, hosts, same):
    """A largest one-to-one matching that keeps both orders (longest common subsequence): (row, host) index pairs."""
    ok = [[same(r, h) for h in hosts] for r in rows]
    best = [[0] * (len(hosts) + 1) for _ in range(len(rows) + 1)]
    for i in range(len(rows) - 1, -1, -1):
        for j in range(len(hosts) - 1, -1, -1):
            best[i][j] = max(best[i + 1][j], best[i][j + 1], best[i + 1][j + 1] + 1 if ok[i][j] else 0)
    pairs, i, j = [], 0, 0
    while i < len(rows) and j < len(hosts):
        if ok[i][j] and best[i][j] == best[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def bar1(rc_rows, host_rows, snapshot_rows, ga_rows, binds, bar2_by_bind, identity, credits):
    """Message rows in the rc arm (spec bar 1, with the bar-1 ruling and RM ruling (b)).

    rc_rows and ga_rows are the rows each arm stored during the run, in store order; host_rows are the run's host
    rows in host order; snapshot_rows are the snapshot's store and host rows. Each bind is
    {bind, session_id, t0_max, h8_max, dup_store_ids}: the copies bar 2 judged at that bind (the edition's
    positional, credited set; None when its receipt has none)."""
    dups = {b["bind"]: set(b["dup_store_ids"]) for b in binds if b.get("dup_store_ids") is not None}

    def key(row):
        return row["session_id"], identity(row)

    def same(row, host):
        return key(row) == key(host) or any(credit(row, host) for credit in credits)

    def bind_of(row):
        return next((b for b in binds if b["session_id"] == row["session_id"]
                     and b["t0_max"] < row["store_id"] <= b["h8_max"]), None)

    copies, leftovers, matched, lost = Counter(), [], 0, 0
    for session in dict.fromkeys(r["session_id"] for r in [*rc_rows, *host_rows]):  # Order is per session only.
        rest = []
        for row in (r for r in rc_rows if r["session_id"] == session):
            bind = bind_of(row)
            if bind and row["store_id"] in dups.get(bind["bind"], ()):  # 1. The copies bar 2 judged at the bind.
                copies[bind["bind"]] += 1
            else:
                rest.append(row)
        hosts = [h for h in host_rows if h["session_id"] == session]
        pairs = _ordered(rest, hosts, same)  # 2. One to one, in order, to the run's host occurrences.
        used = {i for i, _ in pairs}
        matched, lost = matched + len(pairs), lost + len(hosts) - len(pairs)
        leftovers += [(row, hosts) for i, row in enumerate(rest) if i not in used]
    host_keys, snapshot_keys, ga = keys(host_rows, identity), keys(snapshot_rows, identity), Counter(map(key, ga_rows))
    class3, unassigned = {}, []
    for row, hosts in leftovers:  # 3. Class 3 only for an identity matching no host row and no snapshot row.
        if key(row) in host_keys or key(row) in snapshot_keys or any(same(row, h) for h in hosts):
            unassigned.append(row)
        else:
            class3.setdefault(key(row), []).append(row)
    over = [k for k, rows in class3.items() if len(rows) > ga[k]]  # Per identity over the whole run, rc <= GA.
    groups = [[row] for row in unassigned] + [class3[k] for k in over]
    # A leftover at a bind whose bar 2 is INCONCLUSIVE, or whose copies are unknown, makes the bar INCONCLUSIVE,
    # never FAIL (design r4 I-2).
    def open_bind(row):
        bind = bind_of(row)
        return bind is not None and (bar2_by_bind.get(bind["bind"]) == INCONCLUSIVE or bind["bind"] not in dups)

    pending = [g for g in groups if all(open_bind(r) for r in g)]
    verdict = FAIL if lost or len(pending) < len(groups) else INCONCLUSIVE if pending else PASS
    return {"verdict": verdict, "rows": len(rc_rows), "host_occurrences": len(host_rows), "matched": matched,
            "copies_by_bind": dict(copies), "lost": lost, "class3_rows": sum(map(len, class3.values())),
            "class3_over_ga": len(over), "unassigned": len(unassigned), "inconclusive_leftover_groups": len(pending)}


def bar3(snapshot_rows, final_rows, identity):
    """Row preservation, keyed by store_id: same session, equal identity, same relative order within its session."""
    final = {row["store_id"]: (i, row) for i, row in enumerate(final_rows)}
    counts, last = Counter(), {}
    for row in snapshot_rows:
        if row["store_id"] not in final:
            counts["missing"] += 1
            continue
        position, now = final[row["store_id"]]
        counts["wrong_session"] += now["session_id"] != row["session_id"]
        counts["identity_changed"] += identity(now) != identity(row)
        counts["order_changed"] += position < last.get(row["session_id"], -1)
        last[row["session_id"]] = max(position, last.get(row["session_id"], -1))
        counts["observed_at_filled"] += row.get("observed_at") is None and now.get("observed_at") is not None
    bad = sum(counts[k] for k in ("missing", "wrong_session", "identity_changed", "order_changed"))
    return {"verdict": FAIL if bad else PASS if snapshot_rows else INCONCLUSIVE, "rows": len(snapshot_rows),
            **{k: counts[k] for k in ("missing", "wrong_session", "identity_changed", "order_changed",
                                      "observed_at_filled")}}


_FRAME = re.compile(r'^\s*File "([^"]+)", line \d+, in (\S+)')
_TYPE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")


def traceback_signatures(lines, roots):
    """Per traceback with an LCM-X frame: exception type + innermost LCM-X frame (path below its root, function)."""
    found, frames = Counter(), None
    for line in lines:
        if "Traceback (most recent call last):" in line:
            frames = []
        elif frames is not None and (frame := _FRAME.match(line)):
            root = next((r.rstrip("/") for r in roots if frame.group(1).startswith(r.rstrip("/") + "/")), None)
            if root:
                frames.append(f"{frame.group(1)[len(root) + 1:]}:{frame.group(2)}")
        elif frames is not None and not line.startswith((" ", "\t")):
            if frames:
                kind = _TYPE.match(line)
                found[f"{kind.group(1) if kind else 'unparsed'} @ {frames[-1]}"] += 1
            frames = None
    return found


def bar4(rc_lines, ga_lines, rc_roots, ga_roots):
    """0 publication-invariant lines in the rc arm; each rc traceback signature also in GA, with rc <= GA."""
    conflicts = sum(CONFLICT in line for line in rc_lines)
    rc, ga = traceback_signatures(rc_lines, rc_roots), traceback_signatures(ga_lines, ga_roots)
    over = sorted(s for s, n in rc.items() if n > ga[s])
    return {"verdict": FAIL if conflicts or over else PASS, "publication_invariant_lines": conflicts,
            "rc_signatures": dict(rc), "ga_signatures": dict(ga), "over_ga": over}


def failed_checks(checks):
    return sorted({c["check"] for c in checks if c["status"] not in ("ok", "warn")})  # Any other status fails.


def kinds_missing(checks):
    return [k for k in CHECK_KINDS if not any(c["kind"] == k for c in checks)]


def check_counts(checks):
    """Counts only: checks by kind and status, and warnings by check name."""
    return {"by_kind_status": dict(Counter(f"{c['kind']}:{c['status']}" for c in checks)),
            "warnings": dict(Counter(c["check"] for c in checks if c["status"] == "warn"))}


def bar5(rc_checks, ga_checks, ga_rerun_checks=None, ga_rerun_turns_complete=True):
    """No failed doctor, quick_check, integrity or schema check in the rc arm; doctor warnings by check name, with
    rc <= GA. Checks are {kind, check, status}; a kind with no check is not shown (INCONCLUSIVE). A failed check in
    the previous-GA arm is never PASS (release manager, 2026-11-07): the GA arm is re-run once; if the re-run fails
    too, the bar stays INCONCLUSIVE until the release manager records a disposition. A re-run is clean only when it
    completed every planned turn, shows every check kind and fails none; a clean re-run supplies the doctor-warning
    baseline."""
    failed, ga_failed = failed_checks(rc_checks), failed_checks(ga_checks)
    rerun_failed = failed_checks(ga_rerun_checks) if ga_rerun_checks is not None else None
    rerun_missing = kinds_missing(ga_rerun_checks) if ga_rerun_checks is not None else None
    rerun_clean = rerun_failed == [] and rerun_missing == [] and ga_rerun_turns_complete is True
    baseline = ga_rerun_checks if ga_failed and rerun_clean else ga_checks
    rc_w, ga_w = (Counter(c["check"] for c in checks if c["kind"] == "doctor" and c["status"] == "warn")
                  for checks in (rc_checks, baseline))
    over = sorted(name for name, n in rc_w.items() if n > ga_w[name])
    missing = kinds_missing(rc_checks)
    ga_open = bool(ga_failed) and not rerun_clean
    verdict = FAIL if failed or over else INCONCLUSIVE if missing or ga_open else PASS
    return {"verdict": verdict, "failed": failed, "warnings_over_ga": over, "kinds_missing": missing,
            "ga_failed": ga_failed, "ga_rerun_needed": bool(ga_failed) and rerun_failed is None,
            "ga_rerun_failed": rerun_failed or [], "ga_rerun_kinds_missing": rerun_missing or [],
            "ga_rerun_turns_complete": None if ga_rerun_checks is None else ga_rerun_turns_complete is True,
            "ga_disposition_needed": bool(rerun_failed)}


def workload_items(w):
    """Spec workload, item by item, from one arm's observations (a missing observation is a missing item)."""
    names = set(w.get("conversation_names") or [])
    restarts = [r for r in w.get("restarts") or [] if names and set(r["cold_binds"]) >= names]
    return {"conversations": len(names) >= 3, "large_conversation": (w.get("large_conversation_rows") or 0) >= 1000,
            "restarts_with_cold_binds": len(restarts) >= 3, "cron_fired": (w.get("cron_fired") or 0) >= 1,
            "tool_heavy_turns": (w.get("tool_heavy_turns") or 0) >= 1,
            "payload_created": (w.get("payload_files_created") or 0) >= 1,
            "compaction_published": (w.get("compactions_published") or 0) >= 1,
            "whitespace_rewrite": (w.get("whitespace_rewrites") or 0) >= 1,
            "payload_swap": (w.get("payload_swaps") or 0) >= 1, "replay_misses_zero": w.get("replay_misses") == 0,
            "turns_completed": bool(w.get("turns_planned")) and w.get("turns_completed") == w.get("turns_planned")}


def bar6(arms):
    items = {arm: workload_items(w) for arm, w in arms.items()}
    shown = set(arms) == {"rc", "ga"} and all(all(v.values()) for v in items.values())
    return {"verdict": PASS if shown else INCONCLUSIVE, "items": items,
            "replay_misses": {arm: w.get("replay_misses") for arm, w in arms.items()}}


def verdict(results, prior=()):
    """The raw verdict (spec "Verdict", RELEASE-READINESS-V1.md, D5 clone run): PASS when every bar holds, FAIL when
    any bar fails, and a FAIL is never re-run. Otherwise INCONCLUSIVE, which is not green, and the run may re-run
    once on the same rc. A second INCONCLUSIVE is an instrument defect: D5 stays INCONCLUSIVE for that rc until a
    reviewed fix to the instrument lands and the run passes. *prior* is the earlier raw verdicts on this rc."""
    raw = _worst(results.get(b, {}).get("verdict") for b in BARS)
    return {"raw": raw, "attempt": len(prior) + 1, "rerun_allowed": raw == INCONCLUSIVE and not prior,
            "instrument_defect": raw == INCONCLUSIVE and len(prior) >= 1}


def may_run(prior, instrument_fix):
    """None when this rc may run again, else the reason it may not."""
    if FAIL in prior or PASS in prior:
        return "a FAIL is never re-run, and a PASS needs no re-run"
    if len(prior) >= 2 and not instrument_fix:
        return "instrument defect: a reviewed instrument fix must land first"
    return None
