"""Fit the wall-clock model (wallclock_model.py) to the speed_bs runs at a capped decode batch.

Rows per parent: full generation and continuation (no prefix caching) for each timed version, with
lengths from the truth matrix. The screening prefill is taken per token from the same runs.
Usage: wallclock_fit_bs.py <scratch_root> <speed_bs_dir> <slots> <out.json>
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

root, sb, slots, outp = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
RUNS = {"full_truth": "qwen3b", "q7_truth": "qwen7b", "q14_truth": "qwen14b", "l8_truth": "llama8b",
        "phi_truth": "phi35"}


def makespan(L, s):
    h = [0] * min(s, len(L))
    for x in L:
        t = heapq.heappop(h)
        heapq.heappush(h, t + max(int(x), 1))
    return max(h) if h else 0


out = {}
for td, tag in RUNS.items():
    d = sb / f"{tag}_bs{slots}"
    if not d.exists():
        continue
    prompt = {json.loads(l)["id"]: json.loads(l)["prompt_len"]
              for l in open(root / td / "parent_gen" / "gen_A.jsonl", encoding="utf-8")}
    X, y, lab, pp = [], [], [], []
    for v in ["Q8_0", "Q6_K"]:
        if not (d / f"gen_{v}.json").exists():
            continue
        R = [json.loads(l) for l in open(root / td / "screen" / v / "records.jsonl", encoding="utf-8")]
        g = json.loads((d / f"gen_{v}.json").read_text())
        s = json.loads((d / f"scr_{v}_k1.json").read_text())
        full = [r["child_len"] for r in R]
        X.append([makespan(full, slots), sum(full), sum(prompt[r["id"]] for r in R)])
        y.append(g["seconds"]["full_s"]); lab.append(f"{v}/full")
        D = [r for r in R if r["in_D"]]
        c = [r["child_len"] - r["t_star"] for r in D]
        X.append([makespan(c, slots), sum(c), sum(prompt[r["id"]] + r["t_star"] for r in D)])
        y.append(s["seconds"]["cont_s"]); lab.append(f"{v}/cont")
        pp.append(s["seconds"]["prefill_s"] / s["prefill_tokens"])
    X, y = np.array(X, float), np.array(y)
    c, _ = nnls(X, y)
    err = (X @ c - y) / y
    out[td] = {"alpha_s_per_step": c[0], "beta_s_per_token": c[1], "gamma_s_per_prompt_token": c[2],
               "prefill_pass_s_per_token": float(np.median(pp)), "rel_err": dict(zip(lab, err.tolist())),
               "slots": slots}
    print(f"{td:10} alpha={1e3 * c[0]:6.2f} ms/step beta={1e6 * c[1]:7.2f} us gamma={1e6 * c[2]:7.2f} us "
          f"max|err|={np.abs(err).max():.3f}")
outp.write_text(json.dumps(out, indent=1))
