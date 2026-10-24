import json
import math
from pathlib import Path

R = Path(__file__).parent
TOL = 4 * 2**-23


def load(n):
    return json.loads((R / n).read_text())


def rows(n):
    return [json.loads(x) for x in (R / n / "recalls.jsonl").read_text().splitlines()]


def full(r):
    return (
        not r["timeout"]
        and not r["error"]
        and all(
            (r.get("coverage") or {}).get(k) == "full" for k in ("summary", "chunk")
        )
    )


def exact(a, b, uncensored):
    aa, bb = rows(a), rows(b)
    assert len(aa) == len(bb)
    checked = 0
    censored = 0
    maximum = 0
    for x, y in zip(aa, bb):
        assert (x["i"], x["query_class"], x["query"]) == (
            y["i"],
            y["query_class"],
            y["query"],
        )
        if not (full(x) and full(y)):
            censored += 1
            continue
        xx, yy = x["payload"].get("hits", []), y["payload"].get("hits", [])

        def keys(hh):
            return [
                (
                    h.get("kind"),
                    h.get("session_id"),
                    h.get("node_id"),
                    h.get("store_id"),
                    h.get("chunk_span"),
                )
                for h in hh
            ]

        assert keys(xx) == keys(yy), "EXACTNESS_STOP ordered IDs"
        differences = [abs(h["score"] - k["score"]) for h, k in zip(xx, yy)]
        maximum = max([maximum] + differences)
        assert all(d <= TOL for d in differences), "EXACTNESS_STOP scores"
        checked += 1
    if uncensored:
        assert censored == 0, "uncensored pass incomplete"
    return dict(
        pairs=len(aa),
        checked=checked,
        censored=censored,
        ordered_ids_identical=True,
        max_abs_score_difference=maximum,
        tolerance=TOL,
    )


def cell(prefix, left, right):
    result = {}
    for arm, letter in [(left, "A"), (right, "B")]:
        rr = rows(prefix + "-" + letter + "1") + rows(prefix + "-" + letter + "2")
        result[arm] = {}
        for c in ("a", "b"):
            g = [r for r in rr if r["query_class"] == c]
            lat = sorted(r["seconds"] for r in g)
            result[arm][c] = dict(
                n=len(g),
                p50=lat[math.ceil(0.5 * len(lat)) - 1],
                p90=lat[math.ceil(0.9 * len(lat)) - 1],
                max=max(lat),
                timeouts=sum(bool(r["timeout"]) for r in g),
                errors=sum(bool(r["error"]) for r in g),
                full=sum(full(r) for r in g),
            )
        result[arm]["peak_rss_bytes"] = max(
            load(prefix + "-" + letter + str(i) + "/summary.json")["_process"][
                "peak_rss_bytes"
            ]
            for i in (1, 2)
        )
    result["exactness"] = (
        [
            exact(prefix + "-A" + str(i), prefix + "-B" + str(i), prefix in ("N", "V"))
            for i in (1, 2)
        ]
        if prefix != "C2"
        else "not scored; latency events retained"
    )
    if prefix == "N":
        result["memory_pass"] = (
            result[right]["peak_rss_bytes"]
            <= result[left]["peak_rss_bytes"] + 256 * 2**20
        )
    if prefix == "C1":
        result["latency_pass"] = all(
            result[right][c][p] <= result[left][c][p] * 1.05
            for c in ("a", "b")
            for p in ("p50", "p90")
        )
    if prefix == "C2":
        result["latency_pass"] = all(
            result[right][c]["timeouts"] == 0
            and result[right][c]["errors"] == 0
            and result[right][c]["full"] == result[right][c]["n"]
            and result[right][c]["p90"] <= 4
            for c in ("a", "b")
        )
    return result


if __name__ == "__main__":
    score = {}
    for p, left, right in [
        ("V", "option1", "variant"),
        ("N", "base", "candidate"),
        ("C1", "base", "candidate"),
        ("C2", "base", "candidate"),
    ]:
        if all(
            (R / (p + "-" + x + str(i)) / "summary.json").exists()
            for x in ("A", "B")
            for i in (1, 2)
        ):
            score[p] = cell(p, left, right)
        else:
            score[p] = dict(
                status="PENDING",
                reason="incomplete or not admitted; see clock and load.log",
            )
    if "candidate" in score["N"]:
        peak = score["N"]["candidate"]["peak_rss_bytes"]
        score["rss_projection"] = dict(
            formula="RSS(D) = measured_RSS384 + 64*1000*4*(D-384); RCA same-policy model with raw transfer instead of hex",
            dimensions_384_bytes=peak,
            dimensions_768_bytes=peak + 64 * 1000 * 4 * (768 - 384),
            batch_rows=1000,
            bytes_per_dim=4,
            cache_entries_assumption=64,
            not_proven="768d/16GB RSS and model residency; fixed overhead, unchanged allocator policy, no corpus-sized matrix",
        )
    (R / "SCORE.json").write_text(json.dumps(score, indent=2) + "\n")
    print(json.dumps(score, indent=2))
