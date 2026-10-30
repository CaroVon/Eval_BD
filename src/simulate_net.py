"""Net-degradation certificate vs harm-only certificate (offline Monte Carlo).

theta_g = H_g - Rep_g, with H_g = (1/N_g) sum_{g n P} F_i and Rep_g = (1/N_g) sum_{g \\ P} F_i.
By lemma 3, F_i = 0 outside D for every question, so harm candidates are A = g n P n D and
repair candidates are B = g \\ P n D. Certificates (each side at delta'/2 for the net one):
  harm  UCB = T_U^A(K_A) / N_g                       (theorem 4; theta <= H)
  net   UCB = (T_U^A(K_A) - T_L^B(K_B)) / N_g        (theorem 4 + mirrored lower bound)
Without screening (B1-style) the candidate sets are g n P and g \\ P.
A version is certified at tau when every slice has UCB < tau (simultaneous over slices and
versions via delta' = delta / |S|).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cert import cover_design, tl_table, tu_table
from simulate_cert import build_slices, load_version, read_jsonl


def simulate(method, v, slices, tau, delta_p, ratio, reps, rng, pi_floor=0.0):
    screen = method.endswith("ours")
    net = method.startswith("net")
    if method == "full":
        candA, candB = v["P"].copy(), ~v["P"]
    elif screen:
        candA, candB = v["P"] & v["D"], (~v["P"]) & v["D"]
    else:
        candA, candB = v["P"].copy(), ~v["P"]
    if not net and method != "full":
        candB = np.zeros_like(candB)
    dp = delta_p / 2 if net else delta_p
    cand = candA | candB
    pi = np.ones(len(cand)) * cand if method == "full" else cover_design(slices, cand, tau, dp)
    if pi_floor > 0:
        pi = np.where(cand, np.maximum(pi, pi_floor), 0.0)
    if screen:
        scope = np.ones_like(v["P"]) if net else v["P"]  # net needs D on non-P questions too
        plen = np.where(v["D"], v["t_star"] + 1, v["parent_len"])  # lazy prefill (D7)
        pre = ratio * float(((v["prompt_len"] + plen) * scope).sum())
        unit = np.where(v["D"], v["cont_tokens"] + ratio * (v["prompt_len"] + v["t_star"]), 0.0)
    else:
        pre = 0.0
        unit = v["child_len"] + ratio * v["prompt_len"]
    cost = pre + float((pi * unit).sum())
    R = rng.random((reps, len(pi))) < pi
    fails = np.zeros(reps, bool)
    passes = np.ones(reps, bool)
    worst = np.full(reps, -np.inf)
    rows = {}
    for g, mask in slices.items():
        Ng = int(mask.sum())
        H = (v["F"] & v["P"] & mask).sum() / Ng
        Rep = (v["F"] & ~v["P"] & mask).sum() / Ng
        theta = H - Rep
        A, B = candA & mask, candB & mask
        if method == "full":
            ucb = np.full(reps, theta)
        else:
            nA, nB = int(A.sum()), int(B.sum())
            tA = tu_table(nA, float(pi[A].min()) if nA else 1.0, dp)
            KA = R[:, v["F"] & A].sum(axis=1)
            ucb = tA[KA] / Ng
            if net and nB:
                tB = tl_table(nB, float(pi[B].max()), dp)
                KB = R[:, v["F"] & B].sum(axis=1)
                ucb = ucb - tB[KB] / Ng
        fails |= ucb < theta
        passes &= ucb < tau
        worst = np.maximum(worst, ucb)
        rows[g] = {"Ng": Ng, "H": H, "Rep": Rep, "theta": theta, "ucb_mean": float(ucb.mean())}
    return {"cost": cost, "full_cost": float((v["child_len"] + ratio * v["prompt_len"]).sum()),
            "expected_audits": float(pi.sum()), "coverage": float(1 - fails.mean()),
            "certified_rate": float(passes.mean()), "worst_ucb_mean": float(worst.mean()),
            "slices": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--versions", default="Q8_0,Q6_K,Q4_K_M,Q3_K_M")
    ap.add_argument("--data", default=None)
    ap.add_argument("--cat-field", default=None)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--nbins", type=int, default=4)
    ap.add_argument("--prefill-ratio", type=float, default=0.365)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pi-floor", type=float, default=0.0, help="minimum inclusion probability on candidates")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    truth = Path(args.truth)
    gen = read_jsonl(truth / "parent_gen" / "gen_A.jsonl")
    extra = {r["id"]: r for r in read_jsonl(args.data)} if args.data else {}
    meta = [{"id": r["id"], "prompt_len": r["prompt_len"], "parent_len": len(r["out_ids"]),
             **({args.cat_field: extra[r["id"]][args.cat_field]} if args.cat_field else {})} for r in gen]
    ids = [r["id"] for r in meta]
    by_id = {r["id"]: r for r in meta}
    slices = build_slices(meta, args.nbins, args.cat_field, args.m)
    vers = args.versions.split(",")
    S = len(vers) * len(slices)
    dp = args.delta / S
    rng = np.random.default_rng(args.seed)
    res = {"N": len(ids), "n_slices": len(slices), "S": S, "tau": args.tau, "delta": args.delta,
           "pi_floor": args.pi_floor,
           "prefill_ratio": args.prefill_ratio, "reps": args.reps, "methods": {}}
    methods = ["harm_ours", "net_ours", "harm_B1", "net_B1", "full"]
    for mth in methods:
        res["methods"][mth] = {ver: simulate(mth, load_version(truth, ver, by_id, ids), slices, args.tau,
                                             dp, args.prefill_ratio, args.reps, rng,
                                             args.pi_floor) for ver in vers}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"N={len(ids)} slices={len(slices)} |S|={S} tau={args.tau} r={args.prefill_ratio}")
    print(f"{'method':10} " + " ".join(f"{v:>26}" for v in vers) + "   total cost / full")
    tot_full = sum(res["methods"]["full"][v]["full_cost"] for v in vers)
    for mth in methods:
        cells = []
        for v in vers:
            x = res["methods"][mth][v]
            cells.append(f"cert {x['certified_rate']:.2f} ucb {x['worst_ucb_mean']:+.3f} cov {x['coverage']:.2f}")
        tot = sum(res["methods"][mth][v]["cost"] for v in vers)
        print(f"{mth:10} " + " ".join(f"{c:>26}" for c in cells) + f"   {tot / tot_full:.3f}")
    print("true theta (worst slice) / H (worst slice):",
          {v: (round(max(s["theta"] for s in res["methods"]["full"][v]["slices"].values()), 3),
               round(max(s["H"] for s in res["methods"]["full"][v]["slices"].values()), 3)) for v in vers})


if __name__ == "__main__":
    main()
