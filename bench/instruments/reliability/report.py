"""MATRIX.md and ISSUE-MAP.md from results (a list of results.jsonl records)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.instruments.reliability.cells import ISSUES, registry  # noqa: E402

BOUNDARY = ("Claim class: advisory / code_green_local. A PASS proves this lcm-x tree, on this host sha, under this "
            "scripted in-process scenario, meets the bars; not behaviour under real models, the real ACP/gateway "
            "processes, real transports or customer boxes.")
BOUNDARY_R2 = ("Claim class: advisory / code_green_local. Transport {t}: real host processes with a scripted localhost "
               "model. A PASS says nothing about real-model behaviour, real messaging platforms or customer boxes.")
VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE", "ERROR", "UNSUPPORTED")


def cell_word(r: dict) -> str:
    if r["verdict"] == "FAIL":
        return "FAIL " + "/".join(sorted(r.get("failed_bars", {})))
    if r["verdict"] == "INCONCLUSIVE":
        return "INCONCLUSIVE " + "/".join(sorted(r.get("inconclusive_bars", {})))
    return r["verdict"]


def licensed(r: dict) -> tuple[int, int]:
    """(B2, B1) rows licensed by host parity (D-A, scorers/host_parity.py)."""
    n = r.get("numbers") or {}
    return tuple((n.get(b, {}).get("host_parity_licensed") or {}).get("rows", 0) for b in ("B2", "B1"))


def host_dup(r: dict) -> str:
    """One cell's licences with their evidence: LCM store ids, host state.db row ids, content hash, tags."""
    recs = (r.get("numbers") or {}).get("B2", {}).get("host_parity_licensed", {}).get("records") or []
    return "; ".join(f"{x['session']} sha256 {x['sha256'][:12]} tags {','.join(x.get('tags') or []) or '-'} "
                     f"expected {x['expected']} stored {x['stored']} host {x['host']} licensed {x['licensed']} "
                     f"lcm {x['store_ids']} host rows {x['host_row_ids']}" for x in recs)


def signature(r: dict) -> str:
    """Failed bars plus the counters a human needs to attribute the failure."""
    n = r.get("numbers") or {}
    parts = ["/".join(sorted(r.get("failed_bars", {}))) or r["verdict"]]
    conflicts = n.get("B3", {}).get("publication_invariant_conflict")
    b2 = n.get("B2", {})
    failed = n.get("B4", {}).get("failed_turns") or []
    rejections = n.get("diagnostic", {}).get("native_rejections") or {}
    if conflicts:
        parts.append(f"conflicts={conflicts}")
    if n.get("B8", {}).get("survival_fit"):
        parts.append(f"survival_fits={n['B8']['survival_fit']}")
    if n.get("B8", {}).get("fit_unshortened"):
        parts.append(f"fits_unshortened={n['B8']['fit_unshortened']}")
    if n.get("B8", {}).get("exit_fit"):
        parts.append(f"exit_fits={n['B8']['exit_fit']}")
    if n.get("B8", {}).get("exit_fit_skipped"):
        parts.append(f"exit_fits_skipped={n['B8']['exit_fit_skipped']}")
    if b2.get("surplus_rows") or b2.get("deficit_rows"):
        parts.append(f"surplus/deficit={b2.get('surplus_rows')}/{b2.get('deficit_rows')}")
    if failed:
        parts.append(f"failed_turns={len(failed)} from {failed[0]}")
    if rejections:
        parts.append("rejections=" + ",".join(f"{k}x{v}" for k, v in sorted(rejections.items())))
    if n.get("B6", {}).get("split_groups"):
        parts.append(f"split_groups={n['B6']['split_groups']}")
    if any(licensed(r)):
        parts.append("host-dup=B2:{}/B1:{}".format(*licensed(r)))
    return " ".join(parts)


def issue_status(rows: list[dict], capability: str, bars: tuple) -> tuple[str, str]:
    live = [r for r in rows if r["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")]
    if not live:
        why = "; ".join(sorted({str(r.get("reason"))[:90] for r in rows})) if rows else capability
        return f"NOT COVERED ({why or 'no targeting cell'})", ""
    unapplied = [b for b in bars if not any(b in (r.get("applicable_bars") or ()) for r in live)]
    if unapplied:  # a bar no targeting cell evaluated is not a pass
        return f"NOT COVERED on {'/'.join(unapplied)} (no targeting cell applies it)", ""
    hits = sorted((r for r in live if set(r.get("failed_bars", {})) & set(bars)), key=lambda r: r["cell"])
    if hits:
        return ("target cell FAILS (" + ", ".join(r["cell"] for r in hits) + ")",
                "; ".join(f"`{r['cell']}`: {signature(r)}" for r in hits))
    unsure = sorted(r["cell"] for r in live if set(r.get("inconclusive_bars", {})) & set(bars))
    if unsure:
        return "target cell INCONCLUSIVE (" + ", ".join(unsure) + ")", "; ".join(
            f"`{r['cell']}`: {json.dumps(r['inconclusive_bars'])[:200]}" for r in live if r["cell"] in unsure)
    return "target cells PASS on this bar" + (" (some ERROR/UNSUPPORTED)" if len(live) < len(rows) else ""), ""


def write(out: Path, results: list[dict], wall: float, lcm_env: dict | None = None) -> None:
    hosts = sorted({r["host"] for r in results})
    refs = sorted({(r["plugin_ref"], r["plugin_sha"][:12]) for r in results})
    order = [c["id"] for c in registry()]
    order += sorted({r["cell"] for r in results} - set(order))  # transport-only (R2) cells
    env = f"Global LCM env override: `{json.dumps(lcm_env)}`." if lcm_env else "No global LCM env override."
    transports = sorted({r["transport"] for r in results if r.get("transport")})
    boundary = BOUNDARY_R2.format(t="/".join(transports)) if transports else BOUNDARY
    lines = ["# Reliability matrix", "", boundary, "", env, f"Wall clock: {wall:.0f} s for {len(results)} cells.", ""]
    for ref, sha in refs:
        rs = [r for r in results if r["plugin_ref"] == ref and r["plugin_sha"][:12] == sha]
        by = {(r["cell"], r["host"]): r for r in rs}
        lines += [f"## lcm-x `{ref}` ({sha})", "", "| cell | " + " | ".join(hosts) + " | host-dup |",
                  "|---|" + "---|" * (len(hosts) + 1)]
        for cid in [c for c in order if any((c, h) in by for h in hosts)]:
            dup = ", ".join("{}: B2 {}/B1 {}".format(h, *licensed(by[(cid, h)])) for h in hosts
                            if (cid, h) in by and any(licensed(by[(cid, h)])))
            lines.append(f"| `{cid}` | " + " | ".join(cell_word(by[(cid, h)]) if (cid, h) in by else "-" for h in hosts)
                         + f" | {dup or '-'} |")
        lines += ["", "| host | host sha | " + " | ".join(VERDICTS) + " |", "|---|---|" + "---|" * len(VERDICTS)]
        for h in hosts:
            n = Counter(r["verdict"] for r in rs if r["host"] == h)
            sha_h = next((r["host_sha"][:10] for r in rs if r["host"] == h), "")
            lines.append(f"| {h} | {sha_h} | " + " | ".join(str(n[v]) for v in VERDICTS) + " |")
        lines += ["", "### Signatures of non-PASS cells", ""]
        for r in sorted(rs, key=lambda r: (r["cell"], r["host"])):
            if r["verdict"] == "FAIL":
                lines.append(f"- `{r['cell']}` on {r['host']}: {signature(r)}")
            elif r["verdict"] == "INCONCLUSIVE":
                lines.append(f"- INCONCLUSIVE `{r['cell']}` on {r['host']}: {json.dumps(r['inconclusive_bars'])[:300]}")
            elif r["verdict"] in ("ERROR", "UNSUPPORTED"):
                lines.append(f"- {r['verdict']} `{r['cell']}` on {r['host']}: {str(r.get('reason'))[:300]}")
        lines += ["", "### PASS with host-parity licences", ""]
        lines += [f"- `{r['cell']}` on {r['host']}: " + "B2 {} / B1 {} rows licensed".format(*licensed(r))
                  for r in sorted(rs, key=lambda r: (r["cell"], r["host"])) if r["verdict"] == "PASS" and any(licensed(r))] or ["- none"]
        lines.append("")
    (out / "MATRIX.md").write_text("\n".join(lines) + "\n")

    targeting = defaultdict(list)
    for c in registry():
        for t in c["targets"]:
            targeting[t].append(c["id"])
    im = ["# Issue map", "", boundary, "", env, "",
          "\"target cell FAILS\" = a cell targeting the issue fails that issue's bar at the evaluated ref. It is a "
          "signal, not an attribution: whether the failure IS that issue is a human call from the signature.", ""]
    for ref, sha in refs:
        rs = [r for r in results if r["plugin_ref"] == ref and r["plugin_sha"][:12] == sha]
        im += [f"## lcm-x `{ref}` ({sha})", "",
               "| issue | bar | targeting cells | host | status | signature |", "|---|---|---|---|---|---|"]
        for issue, (issue_bars, capability) in ISSUES.items():
            cells_for = targeting.get(issue, [])
            for h in hosts:
                hr = [r for r in rs if r["host"] == h and r["cell"] in cells_for]
                status, sig = ("not run", "") if not hr and cells_for else issue_status(hr, capability, issue_bars)
                im.append(f"| {issue} | {'/'.join(issue_bars)} | {', '.join(f'`{c}`' for c in cells_for) or '-'} "
                          f"| {h} | {status} | {sig} |")
        im += ["", "### Host-parity licences (D-A): licensed rows per cell", ""]
        im += [f"- `{r['cell']}` on {r['host']} ({r['verdict']}): " + "B2 {} / B1 {} rows; ".format(*licensed(r)) + host_dup(r)
               for r in sorted(rs, key=lambda r: (r["cell"], r["host"])) if any(licensed(r))] or ["- none"]
        im.append("")
    (out / "ISSUE-MAP.md").write_text("\n".join(im) + "\n")


if __name__ == "__main__":  # re-render from an existing results.jsonl: python report.py <out>
    target = Path(sys.argv[1])
    recs = [json.loads(x) for x in (target / "results.jsonl").read_text().splitlines() if x.strip()]
    run = json.loads((target / "run.json").read_text()) if (target / "run.json").exists() else {}
    write(target, recs, sum(r.get("wall_s", 0) for r in recs), run.get("lcm_env"))
