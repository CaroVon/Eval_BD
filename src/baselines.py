"""Baselines B2 and B6 against exact screening at equal compute (plan sec. 2.5, metric iv).

Harm certificate for every slice g: upper bound on H_g = (1/N_g) sum_{g n P} F_i, simultaneous over
versions x slices by Bonferroni (delta' = delta / |S|, as in eq_budget.py). Uniform Poisson sampling
at rate pi on the candidate set C; pi is set so that the expected cost equals b x (full evaluation).

  B1    C = g n P, exact binomial inversion (theorem 4)                  [reference, from simulate_net]
  ours  C = g n P n D, slice-cover design + theorem 4, prefill paid       [reference, from simulate_net]
  B6    C = g n P, Horvitz-Thompson + CLT interval (z_{1-delta'})        asymptotic, no finite-sample guarantee
  B2    C = g n P, AIPW with a parent-only predictor f_i + Bernstein     finite-sample valid
  B2s   C = g n P n D, AIPW with f_i = q (prefill signal) + Bernstein   finite-sample valid, prefill paid

AIPW: T_hat = sum_C f_i + sum_C R_i (F_i - f_i) / pi. T - T_hat = sum_i Y_i with independent zero-mean
Y_i = (1 - R_i/pi)(F_i - f_i) <= b = max(1, (1/pi - 1) max f), Var Y_i = (F_i - f_i)^2 (1-pi)/pi and
(F_i - f_i)^2 <= F_i + f_i^2. Bernstein then gives, with prob >= 1 - delta', T <= the largest T with
T - T_hat <= bL/3 + sqrt((bL/3)^2 + 2 (1-pi)/pi (T + sum f^2) L), L = log(1/delta'), solved in closed form.

Predictors (no labels of the target version are used):
  q_v   flip rate among divergent parent-correct questions, transferred from another truth matrix
  pD_i  P(i in D) from parent margins: 1 - prod_t Phi(m_t / sigma_v), sigma_v fitted on the first
        n_cal questions' teacher-forced divergences (as predict_D.py); f_i = q_v pD_i for B2.
Usage: baselines.py <truth_dir> <transfer_truth_dir> [r] [n_cal]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from simulate_cert import build_slices, load_version, read_jsonl
from simulate_net import simulate

truth, transfer = Path(sys.argv[1]), Path(sys.argv[2])
r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.365
n_cal = int(sys.argv[4]) if len(sys.argv) > 4 else 50
VERS = ["Q8_0", "Q6_K"]
ALLV = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
BUDGETS = [0.2, 0.4, 0.6, 0.8, 1.0]
REPS = 400

gen = read_jsonl(truth / "parent_gen" / "gen_A.jsonl")
meta = [{"id": g["id"], "prompt_len": g["prompt_len"], "parent_len": len(g["out_ids"])} for g in gen]
ids = [m["id"] for m in meta]
by_id = {m["id"]: m for m in meta}
slices = build_slices(meta, 4, None, 100)
dp = 0.05 / (len(ALLV) * len(slices))
L = np.log(1 / dp)
z = norm.ppf(1 - dp)


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


def bernstein_ucb(T_hat, S2, pi, fmax):
    a = (1 - pi) / pi
    b = 0.0 if pi >= 1 else max(1.0, (1 / pi - 1) * fmax)  # at pi = 1 every Y_i is 0
    c = b * L / 3
    w = np.maximum(c * c + 2 * a * L * (T_hat + c + S2) + (a * L) ** 2, 0.0)
    return T_hat + c + a * L + np.sqrt(w)


def run_uniform(method, v, f, b, rng):
    """Uniform-pi methods. Returns (mean worst-slice UCB, coverage) or (inf, nan) if over budget."""
    full = float((v["child_len"] + r * v["prompt_len"]).sum())
    if method == "B2s":
        C = v["P"] & v["D"]
        plen = np.where(v["D"], v["t_star"] + 1, v["parent_len"])
        pre = r * float(((v["prompt_len"] + plen) * v["P"]).sum())
        unit = v["cont_tokens"] + r * (v["prompt_len"] + v["t_star"])
    else:
        C = v["P"].copy()
        pre = 0.0
        unit = v["child_len"] + r * v["prompt_len"]
    spend = b * full - pre
    if spend <= 0:
        return float("inf"), float("nan")
    pi = min(1.0, spend / float(unit[C].sum()))
    R = rng.random((REPS, len(C))) < pi
    worst = np.full(REPS, -np.inf)
    fail = np.zeros(REPS, bool)
    for mask in slices.values():
        Ng = int(mask.sum())
        Cg = C & mask
        T = int((v["F"] & Cg).sum())
        RF = R[:, v["F"] & Cg]
        if method == "B6":
            K = RF.sum(axis=1)
            ucb_T = K / pi + z * np.sqrt(K * (1 - pi)) / pi
        else:
            fg = f[Cg]
            Rg = R[:, Cg]
            Fg = v["F"][Cg].astype(float)
            T_hat = fg.sum() + (Rg * (Fg - fg)).sum(axis=1) / pi
            ucb_T = bernstein_ucb(T_hat, float((fg ** 2).sum()), pi, float(fg.max()) if len(fg) else 0.0)
        ucb_T = np.clip(ucb_T, 0, Cg.sum())
        ucb = ucb_T / Ng
        fail |= ucb < T / Ng - 1e-9  # pi = 1 gives T_hat = T up to rounding
        worst = np.maximum(worst, ucb)
    return float(worst.mean()), float(1 - fail.mean())


def ref_method(m, v, b):
    """B1 / ours from simulate_net with the same bisection as eq_budget.py."""
    cost = lambda fl: (lambda o: o["cost"] / o["full_cost"])(simulate(m, v, slices, 0.05, dp, r, 1, np.random.default_rng(0), fl))
    if cost(0.0) > b:
        return float("inf"), float("nan")
    if cost(1.0) <= b:
        fl = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(18):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cost(mid) <= b else (lo, mid)
        fl = lo
    o = simulate(m, v, slices, 0.05, dp, r, REPS, np.random.default_rng(1), fl)
    return o["worst_ucb_mean"], o["coverage"]


res = {"r": r, "budgets": BUDGETS, "delta_prime": dp, "n_cal": n_cal, "transfer": str(transfer), "values": {}}
print(f"== {truth} r={r}: mean worst-slice harm UCB (coverage) at equal budget; delta'={dp:.2e}")
print("               " + " ".join(f"{b:>13}" for b in BUDGETS))
for ver in VERS:
    v = load_version(truth, ver, by_id, ids)
    q = transfer_q(ver)
    pD, s = parent_only_pD(ver)
    f_par = q * pD
    f_scr = np.full(len(ids), q)
    H_true = max((v["F"] & v["P"] & mk).sum() / mk.sum() for mk in slices.values())
    res["values"][ver] = {"q_transfer": q, "sigma_cal": s, "true_worst_H": float(H_true), "methods": {}}
    print(f"  {ver}: q={q:.3f} sigma={s:.3f} true worst-slice H={H_true:.3f}")
    for m in ["harm_B1", "harm_ours", "B6", "B2", "B2s"]:
        row = []
        for b in BUDGETS:
            if m.startswith("harm_"):
                row.append(ref_method(m, v, b))
            else:
                row.append(run_uniform(m, v, f_scr if m == "B2s" else f_par, b, np.random.default_rng(2)))
        res["values"][ver]["methods"][m] = row
        print(f"    {m:10} " + " ".join(f"{u:6.3f}({c:4.2f})" if np.isfinite(u) else "     inf     " for u, c in row))
(truth / f"baselines_r{r}.json").write_text(json.dumps(res, indent=1))
