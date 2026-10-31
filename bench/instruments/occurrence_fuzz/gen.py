"""Seeded host views with shrink, resend, reset and restart operations."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

OPS = [
    "shrink",
    "shrink",
    "append",
    "append",
    "frontdrop",
    "reorder",
    "regrow",
    "resend",
    "reset",
    "restart",
]
# Episode-heavy mode: more shrink/append/regrow and longer sequences.
OPS_EP = (
    ["shrink"] * 3
    + ["append"] * 3
    + ["regrow"] * 2
    + ["resend", "frontdrop", "reorder", "reset", "restart"]
)


def gen(seed: int, mode: str = "base") -> dict[str, Any]:
    rng = random.Random(seed)
    ops_pool = OPS_EP if mode == "ep" else OPS
    nsteps = (4, 9) if mode == "ep" else (3, 7)
    n = rng.randint(6, 10)
    cls = rng.random()
    if cls < 0.25:
        masks = rng.choice("nsu") * n
    else:
        masks = "".join(rng.choice("nsu") for _ in range(n))
    rows: dict[str, dict[str, Any]] = {}
    for i, m in enumerate(masks):
        r: dict[str, Any] = {"role": "user" if i % 2 == 0 else "assistant", "content": f"row-{i}"}
        if m == "s":
            r["timestamp"] = float(10 + i)
        elif m == "u":
            r["message_uid"] = f"old-{i}"
        rows[f"B{i}"] = r
    duptext = None
    if rng.random() < 0.2 and n >= 4:
        i = rng.randint(2, n - 1)
        rows[f"B{i}"]["content"] = rows[f"B{i - 2}"]["content"]
        duptext = i
    base = [f"B{i}" for i in range(n)]
    H_ins = list(base)
    H_end = list(base)
    view = list(base)
    views: list[list[str] | str] = [list(view)]
    ops: list[str] = []
    j = 0
    for _ in range(rng.randint(*nsteps)):
        op = rng.choice(ops_pool)
        if op in ("shrink", "frontdrop", "reorder") and len(view) < 2:
            op = "append"
        if op == "shrink":
            k = rng.randint(1, len(view) - 1)
            view = view[:k]
            ops.append(f"shrink:{k}")
        elif op == "append":
            m = rng.choice("nsu")
            fid = f"F{j}"
            role = (
                "user"
                if not view or rows[view[-1]]["role"] == "assistant"
                else "assistant"
            )
            r = {"role": role, "content": f"fresh-{j}"}
            if m == "s":
                r["timestamp"] = float(1000 + j)
            elif m == "u":
                r["message_uid"] = f"new-{j}"
            rows[fid] = r
            j += 1
            anchor = view[-1] if view else None
            pos = (H_ins.index(anchor) + 1) if anchor in H_ins else len(H_ins)
            # keep fresh rows that already follow the anchor ahead of this one
            while pos < len(H_ins) and H_ins[pos].startswith("F"):
                pos += 1
            H_ins.insert(pos, fid)
            H_end.append(fid)
            view = view + [fid]
            ops.append(f"append:{m}")
        elif op == "frontdrop":
            d = rng.randint(1, min(3, len(view) - 1))
            view = view[d:]
            ops.append(f"frontdrop:{d}")
        elif op == "reorder":
            if rng.random() < 0.5:
                a = rng.randint(0, len(view) - 2)
                b = a + 1
            else:
                a, b = sorted(rng.sample(range(len(view)), 2))
            view = list(view)
            view[a], view[b] = view[b], view[a]
            ops.append(f"reorder:{a},{b}")
        elif op == "regrow":
            if rng.random() < 0.5:
                view = list(H_ins)
                ops.append("regrow:insert")
            else:
                view = list(H_end)
                ops.append("regrow:end")
        elif op == "resend":
            old = rng.choice(H_end)
            view = view + [old]
            ops.append(f"resend:{old}")
        elif op in ("reset", "restart"):
            ops.append(op)
            views.append(op.upper())
            if rng.random() < 0.5:
                ops.append("same")
                views.append(list(view))
            continue
        views.append(list(view))
    return {
        "seed": seed,
        "masks": masks,
        "duptext": duptext,
        "rows": rows,
        "views": views,
        "ops": ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lo", type=int)
    parser.add_argument("hi", type=int)
    parser.add_argument("out")
    parser.add_argument("mode", nargs="?", choices=("base", "ep"), default="base")
    args = parser.parse_args()
    with open(args.out, "w") as fh:
        for seed in range(args.lo, args.hi):
            fh.write(json.dumps(gen(seed, args.mode)) + "\n")


if __name__ == "__main__":
    main()
