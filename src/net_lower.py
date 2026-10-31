"""Variance-type lower bound on the audit rate of any valid net (theta) certificate on one slice.

Setting (docs/net_certificate.md): harm candidates A (|A| = nA, T_A = a flips), repair candidates
B (nB, b), theta = (a - b) / N_g < tau. A procedure certifies when UCB < tau and must be valid at
every configuration, in particular at configurations with theta >= tau. Uniform Poisson sampling
at rate pi on A u B, fixed in advance; the procedure sees which sampled units flip.

Two-point mixture argument. Precision is required at every placement of the a (b) flips, so in
particular on average over uniformly random placements (mixture M0). Validity is required at
every configuration with a + j harms and b - (m - j) repairs, m = ceil(tau N_g) - (a - b), so in
particular on average over random placements of those (mixture M1_j). Under random placement the
observation is summarised without loss by (S_A, K_A, S_B, K_B) (sampled units, sampled flips;
identities are exchangeable), with S ~ Bin(n, pi) the same under M0 and M1 and K | S
hypergeometric. Any certificate gives an event E with P_M1(E) <= delta_v and P_M0(E) = certify
probability. Neyman-Pearson bounds the latter by the most powerful test of M1 vs M0 at size
delta_v; the minimum pi at which this power reaches the target is a lower bound for every valid
procedure whose certification probability reaches the target at every placement of the flips
(for every j; we report the max over j in {0, m/2, m}, which is still a valid lower bound).
Reported at delta_v = delta' (per-slice level of Bonferroni procedures such as ours and B1) and at
delta_v = delta (any procedure valid simultaneously at delta). The worst slice is reported.
Usage: net_lower.py --truth DIR --out FILE [--versions Q8_0,Q6_K] [--target 0.5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, hypergeom

from simulate_cert import build_slices, load_version, read_jsonl


def side(n, t, pi, eps=1e-11):
    """Joint pmf of (S, K) for t flips among n units, flattened, with the (S, K) index pairs."""
    s = np.arange(n + 1)
    ps = binom.pmf(s, n, pi)
    s = s[ps > eps]
    out_s, out_k, out_p = [], [], []
    for si in s:
        k = np.arange(max(0, si - (n - t)), min(si, t) + 1)
        pk = hypergeom.pmf(k, n, t, si) * binom.pmf(si, n, pi)
        out_s.append(np.full(len(k), si))
        out_k.append(k)
        out_p.append(pk)
    return np.concatenate(out_s), np.concatenate(out_k), np.concatenate(out_p)


def aligned(n, t0, t1, pi):
    """pmf of (S, K) under t0 and t1 flips on a common support."""
    s0, k0, p0 = side(n, t0, pi)
    s1, k1, p1 = side(n, t1, pi)
    key = lambda s, k: s * (n + 2) + k
    keys = np.union1d(key(s0, k0), key(s1, k1))
    P0, P1 = np.zeros(len(keys)), np.zeros(len(keys))
    P0[np.searchsorted(keys, key(s0, k0))] = p0
    P1[np.searchsorted(keys, key(s1, k1))] = p1
    return P0, P1


def np_power(P0, P1, alpha):
    """Max P0(E) subject to P1(E) <= alpha (randomised Neyman-Pearson)."""
    with np.errstate(divide="ignore"):
        lr = np.where(P0 > 0, P1 / P0, np.inf)
    o = np.argsort(lr, kind="stable")
    c1 = np.cumsum(P1[o])
    c0 = np.cumsum(P0[o])
    j = np.searchsorted(c1, alpha, side="right")  # first index where c1 > alpha
    base = c0[j - 1] if j > 0 else 0.0
    if j < len(o) and P1[o[j]] > 0:
        prev = c1[j - 1] if j > 0 else 0.0
        base += P0[o[j]] * (alpha - prev) / P1[o[j]]
    return min(base, 1.0)


def power(nA, a, nB, b, j, m, pi, alpha):
    PA0, PA1 = aligned(nA, a, a + j, pi)
    if nB:
        PB0, PB1 = aligned(nB, b, b - (m - j), pi)
    else:
        PB0 = PB1 = np.ones(1)
    P0 = np.outer(PA0, PB0).ravel()
    P1 = np.outer(PA1, PB1).ravel()
    keep = (P0 > 1e-16) | (P1 > 1e-16)
    return np_power(P0[keep], P1[keep], alpha)


def min_pi(nA, a, nB, b, Ng, tau, alpha, target):
    m = int(np.ceil(tau * Ng)) - (a - b)
    if m <= 0:
        return 1.0, None  # theta >= tau: nothing can certify
    js = [j for j in sorted({0, m // 2, m}) if a + j <= nA and b - (m - j) >= 0]
    best, arg = 0.0, None
    for j in js:
        lo, hi = 0.0, 1.0
        if power(nA, a, nB, b, j, m, hi, alpha) < target:
            continue
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if power(nA, a, nB, b, j, m, mid, alpha) < target else (lo, mid)
        if hi > best:
            best, arg = hi, j
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--versions", default="Q8_0,Q6_K")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-versions", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    truth = Path(args.truth)
    gen = read_jsonl(truth / "parent_gen" / "gen_A.jsonl")
    meta = [{"id": r["id"], "prompt_len": r["prompt_len"], "parent_len": len(r["out_ids"])} for r in gen]
    ids = [r["id"] for r in meta]
    by_id = {r["id"]: r for r in meta}
    slices = build_slices(meta, 4, None, 100)
    dp = args.delta / (args.n_versions * len(slices))
    res = {"truth": str(truth), "tau": args.tau, "target": args.target, "delta_prime": dp, "rows": []}
    print(f"{truth} tau={args.tau} target certify prob={args.target}  (pi lower bound at level delta' / delta)")
    for ver in args.versions.split(","):
        v = load_version(truth, ver, by_id, ids)
        for screen in (True, False):
            for kind in ("harm", "net"):
                worst = {"dp": (0.0, ""), "d": (0.0, "")}
                for g, mask in slices.items():
                    Ng = int(mask.sum())
                    A = mask & v["P"] & (v["D"] if screen else True)
                    B = mask & ~v["P"] & (v["D"] if screen else True) if kind == "net" else np.zeros_like(mask)
                    a, b = int((v["F"] & A).sum()), int((v["F"] & B).sum())
                    for lev, alpha in (("dp", dp), ("d", args.delta)):
                        p, j = min_pi(int(A.sum()), a, int(B.sum()), b, Ng, args.tau, alpha, args.target)
                        res["rows"].append({"version": ver, "screen": screen, "kind": kind, "slice": g,
                                            "level": lev, "Ng": Ng, "nA": int(A.sum()), "a": a,
                                            "nB": int(B.sum()), "b": b, "pi_min": p, "j": j})
                        if p > worst[lev][0]:
                            worst[lev] = (p, g)
                print(f"  {ver} {'ours' if screen else 'B1  '} {kind}: pi >= {worst['dp'][0]:.3f} "
                      f"[{worst['dp'][1]}] at delta', {worst['d'][0]:.3f} at delta", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
