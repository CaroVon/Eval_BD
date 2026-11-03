"""G3 at equal wall-clock budget, with costs from the fitted wall-clock model (wallclock_model.py).

Same certificates and designs as eq_budget.py (harm, worst slice, delta' = 0.05 / |S|); only the cost
changes. For a floor f on the inclusion probabilities, the expected time of a method is averaged over
N_DRAW Poisson draws of the audited set:
  B1     full generation of the audited parent-correct questions
  ours   screening prefill of every parent-correct question + continuation of the audited ones
         prefill 'full' : prompt + whole parent output, at the measured prefill-pass cost per token
                          (current implementation)
         prefill 'lazy' : prompt + parent output up to t* + 1 on D (chunked early stop, DECISIONS D7)
The budget b is a fraction of the modelled time of a full evaluation of that version.
Usage: g3_wallclock.py <scratch_root> <wallclock_model.json> <out.json> [slots]
  slots: decode batch of the modelled engine (default 256); only truth dirs present in the model json run.
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np

from cert import cover_design
from simulate_cert import build_slices, load_version, read_jsonl
from simulate_net import simulate


def makespan(lengths, slots=None):
    slots = slots or SLOTS
    """Engine steps of FCFS list scheduling on `slots` slots (as in wallclock_model.py)."""
    h = [0] * min(slots, len(lengths))
    for L in lengths:
        t = heapq.heappop(h)
        heapq.heappush(h, t + max(int(L), 1))
    return max(h) if h else 0


root, wmf, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
WM = json.loads(wmf.read_text())
SLOTS = int(sys.argv[4]) if len(sys.argv) > 4 else 256
TRUTH = [("Qwen2.5-3B GSM8K", "full_truth"), ("Qwen2.5-7B GSM8K", "q7_truth"), ("Qwen2.5-14B GSM8K", "q14_truth"),
         ("Llama-3.1-8B GSM8K", "l8_truth"), ("Phi-3.5-mini GSM8K", "phi_truth"),
         ("Qwen2.5-3B MMLU-Pro 1k", "mmlu_truth"), ("Qwen2.5-14B MMLU-Pro 1k", "q14m_truth")]
BUDGETS = [0.4, 0.6, 0.8]
N_DRAW = 5


def gen_time(w, lengths, prompt_tokens):
    return w["alpha_s_per_step"] * makespan(lengths) + w["beta_s_per_token"] * float(np.sum(lengths)) \
        + w["gamma_s_per_prompt_token"] * float(prompt_tokens)


def method_time(method, V, slices, dp, f, w, prefill, rng):
    cand = V["P"] & V["D"] if method == "ours" else V["P"].copy()
    pi = cover_design(slices, cand, 0.05, dp)
    pi = np.where(cand, np.maximum(pi, f), 0.0)
    if method == "ours":
        plen = V["parent_len"] if prefill == "full" else np.where(V["D"], V["t_star"] + 1, V["parent_len"])
        pre = w["prefill_pass_s_per_token"] * float(((V["prompt_len"] + plen) * V["P"]).sum())
    else:
        pre = 0.0
    ts = []
    for _ in range(N_DRAW):
        S = rng.random(len(pi)) < pi
        if method == "ours":
            ts.append(pre + gen_time(w, V["cont_tokens"][S], (V["prompt_len"] + V["t_star"])[S].sum()))
        else:
            ts.append(gen_time(w, V["child_len"][S], V["prompt_len"][S].sum()))
    return float(np.mean(ts))


def floor_for(method, V, slices, dp, b, full_t, w, prefill):
    rng = lambda: np.random.default_rng(0)
    if method_time(method, V, slices, dp, 0.0, w, prefill, rng()) / full_t > b:
        return None
    if method_time(method, V, slices, dp, 1.0, w, prefill, rng()) / full_t <= b:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if method_time(method, V, slices, dp, mid, w, prefill, rng()) / full_t <= b else (lo, mid)
    return lo


res, md = {}, ["| Parent / task | version | true worst H | " +
               " | ".join(f"b={b}: B1 / ours full / ours lazy" for b in BUDGETS) + " |",
               "|---|---|---|" + "---|" * len(BUDGETS)]
for lab, td in TRUTH:
    if td not in WM:
        continue
    T = root / td
    gen = read_jsonl(T / "parent_gen" / "gen_A.jsonl")
    meta = [{"id": g["id"], "prompt_len": g["prompt_len"], "parent_len": len(g["out_ids"])} for g in gen]
    ids = [m["id"] for m in meta]
    by_id = {m["id"]: m for m in meta}
    slices = build_slices(meta, 4, None, 100)
    vers = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
    dp = 0.05 / (len(vers) * len(slices))
    w = WM[td]
    res[lab] = {}
    for v in ["Q8_0", "Q6_K"]:
        V = load_version(T, v, by_id, ids)
        full_t = gen_time(w, V["child_len"], V["prompt_len"].sum())
        row, cells = {}, []
        for b in BUDGETS:
            vals = {}
            for m, pf in [("B1", None), ("ours", "full"), ("ours", "lazy")]:
                f = floor_for(m, V, slices, dp, b, full_t, w, pf)
                key = m if pf is None else f"ours_{pf}"
                if f is None:
                    vals[key] = float("inf")
                    continue
                sim = simulate("harm_B1" if m == "B1" else "harm_ours", V, slices, 0.05, dp, 0.0, 400,
                               np.random.default_rng(1), f)
                vals[key] = sim["worst_ucb_mean"]
            row[b] = vals
            fm = lambda x: f"{x:.3f}" if np.isfinite(x) else "inf"
            rat = lambda k: vals["B1"] / vals[k] if np.isfinite(vals[k]) else 0.0
            cells.append(f"{fm(vals['B1'])} / {fm(vals['ours_full'])} ({rat('ours_full'):.2f}x) / "
                         f"{fm(vals['ours_lazy'])} ({rat('ours_lazy'):.2f}x)")
        trueH = max((V["F"] & V["P"] & m).sum() / m.sum() for m in slices.values())
        res[lab][v] = {"true_worst_H": float(trueH), "values": row}
        md.append(f"| {lab} | {v} | {trueH:.3f} | " + " | ".join(cells) + " |")
        print(md[-1], flush=True)
outp.write_text(json.dumps(res, indent=1, default=str))
(outp.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
