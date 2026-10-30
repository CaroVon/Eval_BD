"""Predict screening outcomes from parent margins, calibrating sigma_q on a small pilot.

Uses the margin profiles (analysis 1). sigma_q is fitted on the first n_cal questions only
(Bernoulli likelihood of teacher-forced divergences given parent margins); the remaining
questions get P(in D) = 1 - prod_t Phi(m_t / sigma_q) from the parent margins alone.
This is "prediction information" in the plan's sense: usable for design (proposition 12),
never inside a certificate.
Usage: predict_D.py <truth_dir> [n_cal]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


truth = Path(sys.argv[1])
n_cal = int(sys.argv[2]) if len(sys.argv) > 2 else 100
M = np.load(truth / "margins" / "margins_parent.npz")
m, off = M["gap"].astype(float), M["offsets"]
nq = len(off) - 1
qidx = np.repeat(np.arange(nq), np.diff(off))
recs = {}
for ver in ["Q8_0", "Q6_K", "Q4_K_M"]:
    recs[ver] = [json.loads(l) for l in open(truth / "screen" / ver / "records.jsonl", encoding="utf-8")]
P = np.array([r["parent_correct"] for r in recs["Q8_0"]])
cal = np.arange(nq) < n_cal
res = {"n_cal": n_cal}
print(f"== {truth}: sigma calibrated on first {n_cal} questions, evaluated on the other {nq - n_cal}")
for ver, R in recs.items():
    g = np.load(truth / "margins" / f"margins_{ver}.npz")["gap"].astype(float)
    div = g < 0
    pc = cal[qidx]

    def nll(log_s):
        p = np.clip(norm.cdf(-m[pc] / np.exp(log_s)), 1e-15, 1 - 1e-15)
        return -(div[pc] * np.log(p) + (~div[pc]) * np.log(1 - p)).sum()

    s = float(np.exp(minimize_scalar(nll, bounds=(-6, 3), method="bounded").x))
    lp = np.zeros(nq)
    np.add.at(lp, qidx, norm.logcdf(m / s))
    pred = 1 - np.exp(lp)
    inD = np.array([r["in_D"] for r in R])
    ev = ~cal
    evP = ev & P
    res[ver] = {"sigma_cal": s, "AP_obs": float(inD[evP].mean()), "AP_pred": float(pred[evP].mean()),
                "auc": auc(pred[ev], inD[ev])}
    print(f"  {ver:7} sigma={s:.3f}  |A|/|P| obs {inD[evP].mean():.3f} pred {pred[evP].mean():.3f}  "
          f"question-level AUC {res[ver]['auc']:.3f}")
(truth / "margins" / f"predict_D_cal{n_cal}.json").write_text(json.dumps(res, indent=1))
