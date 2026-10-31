"""Joint net-degradation certificate: one exact test on the difference K_A - K_B (follow-up to
simulate_net.py; docs/net_certificate.md, section "Joint certificate").

The split certificate bounds T_A from above and T_B from below, each at delta'/2, so its width is
the sum of two one-sided widths. The joint certificate tests every (a, b) with the statistic
W = K_A - K_B at level 0.9 delta', intersected with b <= T_U^B(K_B) (theorem 4 on B at 0.1 delta'),
and reports UCB = max{a - b : (a, b) kept} / N_g. (The difference test alone keeps pairs with
many repairs, whose large variance it cannot exclude; it was worse than the split certificate.)

Validity. With V_i uniform, R_i = 1[V_i <= pi_i], pi_i >= pA on A and pi_i <= pB on B:
K_A >= X = sum_{A, F=1} 1[V_i <= pA] ~ Bin(T_A, pA) and K_B <= Y = sum_{B, F=1} 1[V_i <= pB] ~
Bin(T_B, pB) pointwise, X and Y independent. So W >= X - Y and
P(W <= c) <= P(X - Y <= c). With crit(a, b) = max{c : P(X_a - Y_b <= c) <= 0.9 delta'}, the test
rejects (a, b) iff W <= crit(a, b); at the true (T_A, T_B) it rejects with probability
<= 0.9 delta', and the B bound fails with probability <= 0.1 delta'. Hence
P(theta_g > UCB) <= delta' for every configuration; uniformity over versions x slices by the
union bound, as in theorem 5.
Result (RESULTS.md sec. 42): not tighter than the split certificate on GSM8K Qwen-3B.

Same design and cost as the split certificate (cover design at delta'/2 plus the pi floor), so the
comparison isolates the bound. Usage:
  net_joint.py --truth DIR --prefill-ratio R --floors 0.6,0.8,0.9 [--reps N] --out FILE
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from cert import cover_design, tl_table, tu_table
from simulate_cert import build_slices, load_version, read_jsonl


def crit_table(nA, nB, pA, pB, delta):
    """C[a, b] = index (x + nB) of crit(a, b); -1 if nothing is rejected. W = x is rejected iff
    x + nB <= C[a, b]."""
    size = nA + nB + 1
    C = np.empty((nA + 1, nB + 1), int)
    for a in range(nA + 1):
        pmf = np.zeros(size + 1)  # index z + nB, one spare cell for the shift
        pmf[nB:nB + a + 1] = binom.pmf(np.arange(a + 1), a, pA) if pA < 1 else np.eye(1, a + 1, a)[0]
        for b in range(nB + 1):
            if b:
                pmf[:size] = (1 - pB) * pmf[:size] + pB * pmf[1:size + 1]
            cdf = np.cumsum(pmf[:size])
            C[a, b] = int(np.searchsorted(cdf, delta * (1 - 1e-12), side="right")) - 1  # last cdf <= delta
    return C


def joint_ucb(KA, KB, nA, nB, pA, pB_min, pB_max, delta):
    """max{a - b : W test at 0.9 delta keeps (a, b), b <= T_U^B(K_B) at 0.1 delta}, per replicate.

    The second condition is theorem 4 on B (pi_i >= pB_min there); it removes pairs with many
    repairs, whose large variance the difference test alone cannot exclude."""
    C = crit_table(nA, nB, pA, pB_max, 0.9 * delta)
    bU = tu_table(nB, pB_min, 0.1 * delta) if nB else np.zeros(1, int)
    d = np.arange(nA + 1)[:, None] - np.arange(nB + 1)[None, :]
    bidx = np.arange(nB + 1)[None, :]
    out = np.empty(len(KA))
    for (x, kb) in set(zip((KA - KB + nB).tolist(), KB.tolist())):
        keep = (C < x) & (bidx <= bU[kb])
        sel = (KA - KB + nB == x) & (KB == kb)
        out[sel] = d[keep].max() if keep.any() else nA
    return out


def simulate_joint(method, v, slices, tau, delta_p, ratio, reps, rng, pi_floor):
    screen = method.endswith("ours")
    if screen:
        candA, candB = v["P"] & v["D"], (~v["P"]) & v["D"]
    else:
        candA, candB = v["P"].copy(), ~v["P"]
    cand = candA | candB
    pi = cover_design(slices, cand, tau, delta_p / 2)  # same design as the split certificate
    pi = np.where(cand, np.maximum(pi, pi_floor), 0.0)
    if screen:
        plen = np.where(v["D"], v["t_star"] + 1, v["parent_len"])
        pre = ratio * float((v["prompt_len"] + plen).sum())
        unit = np.where(v["D"], v["cont_tokens"] + ratio * (v["prompt_len"] + v["t_star"]), 0.0)
    else:
        pre, unit = 0.0, v["child_len"] + ratio * v["prompt_len"]
    cost = pre + float((pi * unit).sum())
    R = rng.random((reps, len(pi))) < pi
    out = {}
    for kind in ("split", "joint"):
        out[kind] = {"fails": np.zeros(reps, bool), "passes": np.ones(reps, bool),
                     "worst": np.full(reps, -np.inf)}
    for g, mask in slices.items():
        Ng = int(mask.sum())
        theta = ((v["F"] & v["P"] & mask).sum() - (v["F"] & ~v["P"] & mask).sum()) / Ng
        A, B = candA & mask, candB & mask
        nA, nB = int(A.sum()), int(B.sum())
        pA = float(pi[A].min()) if nA else 1.0
        pB = float(pi[B].max()) if nB else 0.0
        KA = R[:, v["F"] & A].sum(axis=1)
        KB = R[:, v["F"] & B].sum(axis=1)
        split = tu_table(nA, pA, delta_p / 2)[KA] / Ng
        if nB:
            split = split - tl_table(nB, pB, delta_p / 2)[KB] / Ng
        pBmin = float(pi[B].min()) if nB else 1.0
        joint = joint_ucb(KA, KB, nA, nB, pA, pBmin, pB if nB else 1.0, delta_p) / Ng
        for kind, u in (("split", split), ("joint", joint)):
            o = out[kind]
            o["fails"] |= u < theta - 1e-12
            o["passes"] &= u < tau
            o["worst"] = np.maximum(o["worst"], u)
    full = float((v["child_len"] + ratio * v["prompt_len"]).sum())
    return {k: {"cost_over_full": cost / full, "coverage": float(1 - o["fails"].mean()),
                "certified_rate": float(o["passes"].mean()), "worst_ucb_mean": float(o["worst"].mean())}
            for k, o in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--versions", default="Q8_0,Q6_K")
    ap.add_argument("--n-versions", type=int, default=4, help="versions in the union bound")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--prefill-ratio", type=float, default=0.365)
    ap.add_argument("--floors", default="0.5,0.6,0.7,0.8,0.9,0.95,0.99")
    ap.add_argument("--methods", default="net_ours,net_B1")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    truth = Path(args.truth)
    gen = read_jsonl(truth / "parent_gen" / "gen_A.jsonl")
    meta = [{"id": r["id"], "prompt_len": r["prompt_len"], "parent_len": len(r["out_ids"])} for r in gen]
    ids = [r["id"] for r in meta]
    by_id = {r["id"]: r for r in meta}
    slices = build_slices(meta, 4, None, 100)
    dp = args.delta / (args.n_versions * len(slices))
    vers = args.versions.split(",")
    V = {ver: load_version(truth, ver, by_id, ids) for ver in vers}
    res = {"truth": str(truth), "tau": args.tau, "delta_prime": dp, "r": args.prefill_ratio,
           "reps": args.reps, "rows": []}
    print(f"{truth} tau={args.tau} r={args.prefill_ratio} delta'={dp:.2e}")
    for ver in vers:
        v = V[ver]
        theta = max(((v["F"] & v["P"] & m).sum() - (v["F"] & ~v["P"] & m).sum()) / m.sum() for m in slices.values())
        print(f"  {ver}: true worst-slice theta = {theta:.3f}")
        for mth in args.methods.split(","):
            for fl in map(float, args.floors.split(",")):
                o = simulate_joint(mth, v, slices, args.tau, dp, args.prefill_ratio, args.reps,
                                   np.random.default_rng(0), fl)
                res["rows"].append({"version": ver, "method": mth, "floor": fl, "true_theta": theta, **o})
                print(f"    {mth:8} floor {fl:4.2f} cost {o['split']['cost_over_full']:.3f} | "
                      f"split cert {o['split']['certified_rate']:.2f} ucb {o['split']['worst_ucb_mean']:.3f} "
                      f"cov {o['split']['coverage']:.3f} | joint cert {o['joint']['certified_rate']:.2f} "
                      f"ucb {o['joint']['worst_ucb_mean']:.3f} cov {o['joint']['coverage']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
