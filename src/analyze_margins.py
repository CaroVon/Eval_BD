"""Analysis 1: does divergence happen where the parent margin is smaller than the quantization
perturbation of that margin?

Per parent-output position t: m_t = parent margin of y0_t (>= 0), g_t = child margin of y0_t,
d_t = g_t - m_t. Teacher-forced divergence at t iff g_t < 0.
Model: d_t ~ N(0, sigma_q) independent of m_t gives P(div at t) = Phi(-m_t / sigma_q) and
P(question not in D) = prod_t Phi(m_t / sigma_q).
Usage: analyze_margins.py <truth_dir> <vocab.json>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from classify_divergence import BYTE_DEC  # noqa: E402

truth, vocab_path = Path(sys.argv[1]), sys.argv[2]
inv = {v: k for k, v in json.load(open(vocab_path, encoding="utf-8")).items()}


def tok_text(t):
    s = inv.get(int(t), "")
    return bytearray(BYTE_DEC[c] for c in s if c in BYTE_DEC).decode("utf-8", errors="replace")


def tok_class(t, pos):
    if pos == 0:
        return "opener"
    a = tok_text(t)
    if a.strip() == "":
        return "space/newline"
    if re.search(r"\d", a):
        return "digit"
    if not re.search(r"[A-Za-z]", a):
        return "punct/markup"
    return "word"


M = np.load(truth / "margins" / "margins_parent.npz")
m, toks, off = M["gap"].astype(float), M["tok"], M["offsets"]
qidx = np.repeat(np.arange(len(off) - 1), np.diff(off))
pos = np.arange(len(m)) - off[qidx]
cls = np.array([tok_class(t, p) for t, p in zip(toks, pos)])
print(f"== {truth}  positions={len(m)}  parent margin quantiles (5/25/50/75%): "
      f"{np.round(np.quantile(m, [.05, .25, .5, .75]), 3)}  min={m.min():.4f}")
print("   share of positions with margin < 0.1 / 0.5 / 1 / 2:",
      [round(float((m < x).mean()), 4) for x in (0.1, 0.5, 1, 2)])
bins = np.array([0, 0.1, 0.25, 0.5, 1, 2, 4, np.inf])
summary = {}
for ver in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]:
    f = truth / "margins" / f"margins_{ver}.npz"
    if not f.exists():
        continue
    C = np.load(f)
    assert np.array_equal(C["tok"], toks) and np.array_equal(C["offsets"], off)
    g = C["gap"].astype(float)
    d = g - m
    div = g < 0

    # fit sigma: maximise Bernoulli likelihood of div given m
    def nll(log_s):
        p = np.clip(norm.cdf(-m / np.exp(log_s)), 1e-15, 1 - 1e-15)
        return -(div * np.log(p) + (~div) * np.log(1 - p)).sum()

    s = float(np.exp(minimize_scalar(nll, bounds=(-6, 3), method="bounded").x))
    b = np.digitize(m, bins) - 1
    obs = [div[b == j].mean() if (b == j).any() else np.nan for j in range(len(bins) - 1)]
    pred = [norm.cdf(-m[b == j] / s).mean() if (b == j).any() else np.nan for j in range(len(bins) - 1)]
    # question level: not in D <=> no divergence anywhere along y0
    nq = len(off) - 1
    inD = np.zeros(nq, bool)
    np.logical_or.at(inD, qidx, div)
    logp_ok = np.zeros(nq)
    np.add.at(logp_ok, qidx, norm.logcdf(m / s))
    pred_inD = 1 - np.exp(logp_ok)
    # calibration of predicted P(in D) by decile
    order = np.argsort(pred_inD)
    dec = np.array_split(order, 5)
    calib = [(round(float(pred_inD[i].mean()), 2), round(float(inD[i].mean()), 2)) for i in dec]
    # perturbation size vs margin, and class shares
    sd_by_m = [float(np.std(d[b == j])) if (b == j).sum() > 50 else np.nan for j in range(len(bins) - 1)]
    cls_rows = []
    for c in ["opener", "word", "punct/markup", "digit", "space/newline"]:
        k = cls == c
        if k.any():
            cls_rows.append((c, round(float(k.mean()), 3), round(float(np.median(m[k])), 2),
                             round(float(div[k].mean()), 4), round(float(div[k].sum() / max(div.sum(), 1)), 3)))
    print(f"-- {ver}: sigma={s:.3f}  teacher-forced divergence rate {div.mean():.4f}  "
          f"|D|/N obs {inD.mean():.3f} pred {pred_inD.mean():.3f}  corr(d, m)={np.corrcoef(d, m)[0, 1]:.3f}")
    print("   P(div | parent margin bin) obs :", [round(x, 4) for x in obs])
    print("                              pred:", [round(x, 4) for x in pred])
    print("   sd of perturbation by margin bin:", [round(x, 3) for x in sd_by_m])
    print("   question-level calibration (pred, obs) by quintile:", calib)
    print("   by token class (class, share of positions, median margin, div rate, share of divs):", cls_rows)
    summary[ver] = {"sigma": s, "div_rate": float(div.mean()), "D_obs": float(inD.mean()),
                    "D_pred": float(pred_inD.mean()), "obs_by_bin": obs, "pred_by_bin": pred,
                    "sd_by_bin": sd_by_m, "calibration": calib, "classes": cls_rows,
                    "bins": [float(x) for x in bins[:-1]]}
(truth / "margins" / "analysis.json").write_text(json.dumps(summary, indent=1))
