"""Tighter B2 and the design baselines B3/B4 at equal compute (follow-up to baselines.py, sec. 35).

All methods certify the worst-slice harm H_g = T_g / N_g with T_g = sum_{g n P} F_i, simultaneous
over versions x slices by Bonferroni (delta' as in baselines.py). Candidates C = g n P, cost of an
audit c_i = child_len + r prompt_len, Poisson sampling with inclusion probabilities pi_i whose
expected cost equals b x (full evaluation).

  B0m  uniform pi, f = 0                        (sanity: should be no better than B1)
  B2m  uniform pi, AIPW with f_i = q pD_i       (B2 with the bound below instead of Bernstein)
  B3   pi_i ~ sqrt(f_i (1 - f_i)), AIPW         (active inference: variance-optimal design)
  B4   pi_i ~ sqrt(f_i (1 - f_i) / c_i), AIPW   (cost-optimal active evaluation)
  B3/B4 mix the tilt with 10% uniform weight so that every pi_i > 0.

Bound (finite-sample valid, no CLT). With Z_i = f_i + R_i (F_i - f_i) / pi_i, T_hat = sum Z_i has
E T_hat = T. For a hypothesised T0, P(T_hat <= t) <= exp(lam t) prod_i E exp(-lam Z_i) for every
lam >= 0 (Chernoff with the exact two-point MGF of each Z_i). The flip placement is unknown, so the
bound is maximised over all placements with sum F = T0: for fixed lam this takes the T0 units with
the largest log M_i(1) - log M_i(0). T0 is rejected if the minimum over a lam grid is < delta'; the
UCB is the largest T0 not rejected. This is at least as tight as any fixed-bet betting bound on the
same statistic and tighter than Bernstein, but uses the same information.
Predictors f, q, pD: as baselines.py (parent signals only; q transferred from another parent).
Usage: baselines_mgf.py <truth_dir> <transfer_truth_dir> [r] [n_cal]
Writes <truth_dir>/baselines_mgf_r{r}.json; B1 and ours are read from baselines_r{r}.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from simulate_cert import build_slices, load_version, read_jsonl

truth, transfer = Path(sys.argv[1]), Path(sys.argv[2])
r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.365
n_cal = int(sys.argv[4]) if len(sys.argv) > 4 else 50
VERS = ["Q8_0", "Q6_K"]
ALLV = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
BUDGETS = [0.2, 0.4, 0.6, 0.8, 1.0]
REPS = 400
LAMS = np.geomspace(1e-4, 1e3, 160)
MIX = 0.1

gen = read_jsonl(truth / "parent_gen" / "gen_A.jsonl")
meta = [{"id": g["id"], "prompt_len": g["prompt_len"], "parent_len": len(g["out_ids"])} for g in gen]
ids = [m["id"] for m in meta]
by_id = {m["id"]: m for m in meta}
slices = build_slices(meta, 4, None, 100)
dp = 0.05 / (len(ALLV) * len(slices))
logdp = np.log(dp)


def transfer_q(ver):
    g = read_jsonl(transfer / "parent_gen" / "gen_A.jsonl")
    tm = {x["id"]: {"prompt_len": x["prompt_len"]} for x in g}
    v = load_version(transfer, ver, tm, [x["id"] for x in g])
    A = v["P"] & v["D"]
    return float((v["F"] & A).sum() / A.sum())


Mp = np.load(truth / "margins" / "margins_parent.npz")
m_par, off = Mp["gap"].astype(float), Mp["offsets"]
qidx = np.repeat(np.arange(len(off) - 1), np.diff(off))


def parent_only_pD(ver):
    g = np.load(truth / "margins" / f"margins_{ver}.npz")["gap"].astype(float)
    pc = qidx < n_cal
    div = g < 0

    def nll(log_s):
        p = np.clip(norm.cdf(-m_par[pc] / np.exp(log_s)), 1e-15, 1 - 1e-15)
        return -(div[pc] * np.log(p) + (~div[pc]) * np.log(1 - p)).sum()

    s = float(np.exp(minimize_scalar(nll, bounds=(-6, 3), method="bounded").x))
    lp = np.zeros(len(off) - 1)
    np.add.at(lp, qidx, norm.logcdf(m_par / s))
    return 1 - np.exp(lp), s


def log_mgf(lam, f, pi, F):
    """log E exp(-lam Z), Z = f + R (F - f) / pi, R ~ Bernoulli(pi); shape (len(lam), n)."""
    lam = lam[:, None]
    with np.errstate(divide="ignore"):
        l1p = np.log1p(-np.minimum(pi, 1.0))  # -inf at pi = 1
    return -lam * f + np.logaddexp(l1p, np.log(pi) - lam * (F - f) / pi)


def slice_table(f, pi):
    """G[l, T0] = max over placements with T0 flips of sum_i log M_i(lam_l)."""
    m0, m1 = log_mgf(LAMS, f, pi, 0.0), log_mgf(LAMS, f, pi, 1.0)
    d = -np.sort(-(m1 - m0), axis=1)
    return m0.sum(axis=1)[:, None] + np.concatenate([np.zeros((len(LAMS), 1)), np.cumsum(d, axis=1)], axis=1)


def ucb_T(t_obs, G):
    """Largest T0 not rejected, per replicate."""
    best = np.full((len(t_obs), G.shape[1]), np.inf)
    for l, lam in enumerate(LAMS):
        np.minimum(best, lam * t_obs[:, None] + G[l][None, :], out=best)
    keep = best >= logdp
    keep[:, 0] = True
    return keep.shape[1] - 1 - np.argmax(keep[:, ::-1], axis=1)


def design(method, f, c, C, spend):
    """Inclusion probabilities on C with expected cost = spend (pi = min(1, a w))."""
    if method in ("B0m", "B2m"):
        w = np.ones(C.sum())
    else:
        s = np.sqrt(np.clip(f[C] * (1 - f[C]), 0, None))
        if method == "B4":
            s = s / np.sqrt(c[C])
        w = (1 - MIX) * s / s.mean() + MIX
    cc = c[C]
    if (cc).sum() <= spend:
        return np.ones(C.sum())
    lo, hi = 0.0, 1.0 / w.min()
    for _ in range(60):
        a = (lo + hi) / 2
        lo, hi = (a, hi) if (np.minimum(1, a * w) * cc).sum() <= spend else (lo, a)
    return np.minimum(1, lo * w)


def run(method, v, f, b, rng):
    c = v["child_len"] + r * v["prompt_len"]
    full = float(c.sum())
    C = v["P"].copy()
    ff = np.zeros(len(f)) if method == "B0m" else f
    pi_C = design(method, ff, c, C, b * full)
    pi = np.zeros(len(f))
    pi[C] = pi_C
    R = rng.random((REPS, int(C.sum()))) < pi_C
    Rfull = np.zeros((REPS, len(f)), bool)
    Rfull[:, C] = R
    worst = np.full(REPS, -np.inf)
    fail = np.zeros(REPS, bool)
    for mask in slices.values():
        Ng = int(mask.sum())
        Cg = C & mask
        T = int((v["F"] & Cg).sum())
        fg, pg, Fg = ff[Cg], pi[Cg], v["F"][Cg].astype(float)
        t_obs = fg.sum() + (Rfull[:, Cg] * (Fg - fg) / pg).sum(axis=1)
        u = ucb_T(t_obs, slice_table(fg, pg)) / Ng
        fail |= u < T / Ng - 1e-9
        worst = np.maximum(worst, u)
    return float(worst.mean()), float(1 - fail.mean())


ref = json.loads((truth / f"baselines_r{r}.json").read_text())["values"]
res = {"r": r, "budgets": BUDGETS, "delta_prime": dp, "n_cal": n_cal, "transfer": str(transfer),
       "mix": MIX, "values": {}}
print(f"== {truth} r={r}: mean worst-slice harm UCB (coverage) at equal budget; delta'={dp:.2e}")
print("               " + " ".join(f"{b:>13}" for b in BUDGETS))
for ver in VERS:
    v = load_version(truth, ver, by_id, ids)
    q = transfer_q(ver)
    pD, s = parent_only_pD(ver)
    f = q * pD
    out = {"q_transfer": q, "sigma_cal": s, "true_worst_H": ref[ver]["true_worst_H"],
           "methods": {m: ref[ver]["methods"][m] for m in ("harm_B1", "harm_ours", "B2")}}
    print(f"  {ver}: q={q:.3f} sigma={s:.3f} true worst-slice H={out['true_worst_H']:.3f}")
    for m in ["B0m", "B2m", "B3", "B4"]:
        out["methods"][m] = [run(m, v, f, b, np.random.default_rng(2)) for b in BUDGETS]
    for m, row in out["methods"].items():
        print(f"    {m:10} " + " ".join(f"{u:6.3f}({c:4.2f})" if u is not None and np.isfinite(u) else "     inf     "
                                       for u, c in row))
    res["values"][ver] = out
(truth / f"baselines_mgf_r{r}.json").write_text(json.dumps(res, indent=1))
