"""Diagnostics on a truth matrix: why |A|/|P| is high, why flips are rare inside A, harm/repair symmetry."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tokdecode  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


truth, vocab_path = Path(sys.argv[1]), sys.argv[2]
_text = tokdecode.load(vocab_path)
gen = {r["id"]: r for r in read_jsonl(truth / "parent_gen" / "gen_A.jsonl")}
print(f"== {truth}")
for ver in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]:
    R = read_jsonl(truth / "screen" / ver / "records.jsonl")
    T = np.array([r["parent_len"] for r in R], float)
    inD = np.array([r["in_D"] for r in R])
    P = np.array([r["parent_correct"] for r in R])
    C = np.array([r["child_correct"] for r in R])
    ts = np.array([r["t_star"] if r["t_star"] is not None else -1 for r in R])

    # 1. constant-hazard model: P(no divergence) = (1 - rho)^T; MLE of rho from survival data
    events = inD.sum()
    exposure = np.where(inD, ts + 1, T).sum()
    rho = events / exposure
    pred = 1 - (1 - rho) ** T
    bins = np.quantile(T, [0, .25, .5, .75, 1])
    b = np.clip(np.searchsorted(bins[1:-1], T, side="right"), 0, 3)
    obs_q = [inD[b == j].mean() for j in range(4)]
    pred_q = [pred[b == j].mean() for j in range(4)]

    # 2. where divergences happen
    tD = ts[inD]
    frac0 = (tD == 0).mean()
    rel = (tD / T[inD])
    # hazard at position 0 vs later positions
    h0 = (tD == 0).sum() / len(T)
    later = inD & (ts > 0)
    rho_later = later.sum() / np.where(inD, ts, T).clip(min=0).sum() if later.any() else 0

    # 3. flips inside A
    A = P & inD
    harm = A & ~C
    flip_rate_A = harm.sum() / max(A.sum(), 1)
    early = A & (ts < np.median(ts[A])) if A.any() else A
    late = A & (ts >= np.median(ts[A])) if A.any() else A

    # 4. harm vs repair
    repair = ~P & C
    print(f"{ver:7} rho={rho:.4f}  |D|/N obs by length quartile {np.round(obs_q, 2)} pred {np.round(pred_q, 2)}")
    print(f"        t*=0 share of D {frac0:.2f}  hazard@0={h0:.3f}  rho(t>0)={rho_later:.4f}  median t*/T={np.median(rel):.2f}")
    print(f"        flips in A: {harm.sum()}/{A.sum()} = {flip_rate_A:.3f}  "
          f"(early t* {(harm & early).sum()}/{early.sum()}, late t* {(harm & late).sum()}/{late.sum()})  "
          f"harm/repair = {harm.sum()}/{repair.sum()}")
    if frac0 > 0.2:
        pairs = Counter()
        for r in R:
            if r["in_D"] and r["t_star"] == 0:
                p_tok = gen[r["id"]]["out_ids"][0]
                c_tok = r["child_out_ids"][0]
                pairs[(_text([p_tok]), _text([c_tok]))] += 1
        print("        top t*=0 (parent -> child) first tokens:", pairs.most_common(5))
