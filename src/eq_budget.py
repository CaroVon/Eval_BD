"""Plan metric (iv): worst-slice certifiable bound at equal compute budget (per version).

For each method and budget b (fraction of the full-evaluation cost of that version), the
inclusion-probability floor is set by bisection so that the expected audit cost equals b; the
reported value is the mean worst-slice UCB over Monte Carlo reps (lower = more precise).
'inf' means the method's fixed screening cost alone exceeds b.
Usage: eq_budget.py <truth_dir> [r]
"""
import json
import sys
from pathlib import Path

import numpy as np

from simulate_cert import build_slices, load_version, read_jsonl
from simulate_net import simulate

truth = Path(sys.argv[1])
r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.365
gen = read_jsonl(truth / "parent_gen" / "gen_A.jsonl")
meta = [{"id": g["id"], "prompt_len": g["prompt_len"], "parent_len": len(g["out_ids"])} for g in gen]
ids = [m["id"] for m in meta]
by_id = {m["id"]: m for m in meta}
slices = build_slices(meta, 4, None, 100)
vers = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
dp = 0.05 / (len(vers) * len(slices))
V = {v: load_version(truth, v, by_id, ids) for v in vers}
methods = ["harm_B1", "harm_ours", "net_B1", "net_ours"]
budgets = [0.2, 0.4, 0.6, 0.8, 1.0]
res = {"r": r, "budgets": budgets, "values": {}}


def cost_at(m, v, f):
    rng = np.random.default_rng(0)
    out = simulate(m, V[v], slices, 0.05, dp, r, 1, rng, f)
    return out["cost"] / out["full_cost"]


print(f"== {truth} r={r}: mean worst-slice UCB at equal per-version budget (fraction of full eval)")
print("   " + " ".join(f"{b:>8}" for b in budgets))
for v in ["Q8_0", "Q6_K"]:
    res["values"][v] = {}
    for m in methods:
        row = []
        for b in budgets:
            lo, hi = 0.0, 1.0
            if cost_at(m, v, lo) > b:
                row.append(float("inf"))
                continue
            if cost_at(m, v, hi) <= b:
                f = hi
            else:
                for _ in range(18):
                    mid = (lo + hi) / 2
                    (lo, hi) = (mid, hi) if cost_at(m, v, mid) <= b else (lo, mid)
                f = lo
            out = simulate(m, V[v], slices, 0.05, dp, r, 400, np.random.default_rng(1), f)
            row.append(out["worst_ucb_mean"])
        res["values"][v][m] = row
        print(f"  {v:6} {m:10} " + " ".join(f"{x:8.3f}" if np.isfinite(x) else "     inf" for x in row))
(truth / f"eq_budget_r{r}.json").write_text(json.dumps(res, indent=1))
