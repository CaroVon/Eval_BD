"""Wall-clock cost model for batched greedy generation, fitted to measured truth-matrix runs.

Continuous batching with S slots, FCFS in question order: a request occupies a slot for L decode steps,
and the engine runs one step for all active requests. The model is
    time = alpha * makespan + beta * sum(L) + gamma * prefill_tokens,
where makespan is the number of engine steps of list scheduling on S slots. alpha is the fixed cost per
step (sec. 36), beta the marginal cost per decoded token, gamma the cost per prompt token.
Rows per truth matrix: full generation and continuation from t* for every version, with
lengths from records.jsonl. The screening prefill pass is reported separately as seconds per token.
Usage: wallclock_model.py <scratch_root> <seconds_all.json> <out.json>
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

root, secf, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEC = json.loads(secf.read_text())
MAP = {"full_truth": "gsm8k_full_fp32_qwen2.5-3b_2273994", "q7_truth": "gsm8k_full_fp32_qwen2.5-7b_eager_2276789",
       "q14_truth": "gsm8k_full_fp32_qwen14b_2280619", "l8_truth": "gsm8k_full_fp32_llama8b_2280621",
       "phi_truth": "gsm8k_full_fp32_phi-3.5-mini_eager_2276333", "smol_truth": "gsm8k_full_fp32_smollm2-1.7b_eager_2276609",
       "mmlu_truth": "mmlupro_pilot1000_fp32_qwen2.5-3b_2274801", "q14m_truth": "mmlupro_pilot1000_fp32_qwen14b_eager_2281325",
       "l8m_truth": "mmlupro_pilot1000_fp32_llama8b_eager_2281326", "phim_truth": "mmlupro_pilot1000_fp32_phi-3.5-mini_eager_2276607",
       "imat_truth": "gsm8k_full_fp32_qwen2.5-3b_imat_2283849", "mimat_truth": "mmlupro_pilot1000_fp32_qwen2.5-3b_imat_2291668"}
SLOTS = 256


def makespan(lengths, slots=SLOTS):
    h = [0] * min(slots, len(lengths))
    for L in lengths:
        t = heapq.heappop(h)
        heapq.heappush(h, t + max(int(L), 1))
    return max(h) if h else 0


def rows_for(td):
    prompt = {}
    for l in open(root / td / "parent_gen" / "gen_A.jsonl", encoding="utf-8"):
        r = json.loads(l)
        prompt[r["id"]] = r["prompt_len"]
    X, y, lab = [], [], []
    for v, s in SEC[MAP[td]].items():
        f = root / td / "screen" / v / "records.jsonl"
        if not f.exists() or not s["seconds"]:
            continue
        R = [json.loads(l) for l in open(f, encoding="utf-8")]
        full = [r["child_len"] for r in R]
        X.append([makespan(full), sum(full), sum(prompt[r["id"]] for r in R)])
        y.append(s["seconds"]["full_s"]); lab.append(f"{v}/full")
        D = [r for r in R if r["in_D"]]
        if not D:  # parent vs itself: no continuation
            continue
        cont = [r["child_len"] - r["t_star"] for r in D]
        X.append([makespan(cont), sum(cont), sum(prompt[r["id"]] + r["t_star"] for r in D)])
        y.append(s["seconds"]["cont_s"]); lab.append(f"{v}/cont")
    return np.array(X, float), np.array(y), lab


res = {}
print(f"{'truth':12} {'alpha ms/step':>13} {'beta us/tok':>11} {'gamma us/tok':>12} {'max |err|':>9} "
      f"{'LOO max|err|':>12} {'prefill-pass us/tok':>19}")
for td in MAP:
    if not (root / td / "parent_gen" / "gen_A.jsonl").exists():
        continue
    X, y, lab = rows_for(td)
    coef, _ = nnls(X, y)
    err = (X @ coef - y) / y
    loo = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        c, _ = nnls(X[m], y[m])
        loo.append((X[i] @ c - y[i]) / y[i])
    pp = [s["seconds"]["prefill_s"] / s["prefill_tokens"] for s in SEC[MAP[td]].values() if s["seconds"]]
    res[td] = {"alpha_s_per_step": coef[0], "beta_s_per_token": coef[1], "gamma_s_per_prompt_token": coef[2],
               "rel_err": dict(zip(lab, err.tolist())), "loo_rel_err": dict(zip(lab, loo)),
               "prefill_pass_s_per_token": float(np.median(pp))}
    print(f"{td:12} {1e3*coef[0]:13.2f} {1e6*coef[1]:11.2f} {1e6*coef[2]:12.2f} {np.abs(err).max():9.3f} "
          f"{np.abs(loo).max():12.3f} {1e6*np.median(pp):19.2f}")
outp.write_text(json.dumps(res, indent=1))
