"""Robustness of the margin law rho = f_m(0) sigma / sqrt(2 pi) (RESULTS.md sec. 13, 26-27).

For each truth matrix and version: ratio = observed rho / (f_m(0) sigma / sqrt(2 pi)), with
  rho     teacher-forced divergence rate per position (child margin of y0_t < 0)
  sigma   sd of d_t = child margin - parent margin, over positions with parent margin in [0, 4)
  f_m(0)  share of parent margins below w, divided by w, for windows w = 0.05, 0.1, 0.2
95% intervals from a bootstrap over questions (B resamples).
Usage: law_robustness.py <truth_dir>... (labels taken from directory names)
"""
import json
import sys
from pathlib import Path

import numpy as np

B = 200
WINDOWS = (0.05, 0.1, 0.2)
rng = np.random.default_rng(0)
out = {}
for td in sys.argv[1:]:
    T = Path(td)
    M = np.load(T / "margins" / "margins_parent.npz")
    m, off = M["gap"].astype(float), M["offsets"]
    nq = len(off) - 1
    qidx = np.repeat(np.arange(nq), np.diff(off))
    out[T.name] = {}
    for ver in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]:
        f = T / "margins" / f"margins_{ver}.npz"
        if not f.exists():
            continue
        g = np.load(f)["gap"].astype(float)
        d = g - m
        div = (g < 0).astype(float)
        core = (m < 4).astype(float)
        # per-question sufficient statistics, so bootstrap resamples questions cheaply
        def qsum(x):
            s = np.zeros(nq)
            np.add.at(s, qidx, x)
            return s
        n_pos = qsum(np.ones_like(m)); n_div = qsum(div)
        n_core = qsum(core); s1 = qsum(d * core); s2 = qsum(d * d * core)
        n_win = {w: qsum((m < w).astype(float)) for w in WINDOWS}

        def ratio(idx, w):
            npos = n_pos[idx].sum()
            rho = n_div[idx].sum() / npos
            nc = n_core[idx].sum()
            mu = s1[idx].sum() / nc
            sig = np.sqrt(max(s2[idx].sum() / nc - mu * mu, 1e-12))
            f0 = n_win[w][idx].sum() / npos / w
            return rho / (f0 * sig / np.sqrt(2 * np.pi)), rho, sig, f0

        res = {}
        full = np.arange(nq)
        for w in WINDOWS:
            r0, rho, sig, f0 = ratio(full, w)
            boots = [ratio(rng.integers(0, nq, nq), w)[0] for _ in range(B)]
            lo, hi = np.quantile(boots, [0.025, 0.975])
            res[str(w)] = {"ratio": r0, "ci": [float(lo), float(hi)], "rho": rho, "sigma": sig, "f0": f0}
        out[T.name][ver] = res
        print(f"{T.name:14} {ver:7} sigma={res['0.1']['sigma']:.3f} rho={res['0.1']['rho']:.4f}  "
              + "  ".join(f"w={w}: ratio {res[str(w)]['ratio']:.2f} [{res[str(w)]['ci'][0]:.2f},{res[str(w)]['ci'][1]:.2f}]"
                          for w in WINDOWS))
Path("law_robustness.json").write_text(json.dumps(out, indent=1))
