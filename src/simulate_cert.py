"""Offline Monte Carlo of the certification pipeline on a truth matrix (plan sec 2.3 steps 2-4).

Methods (same exact certificate, theorems 4-5, uniform over versions x slices):
  ours  candidates A = P n D (exact screening), audit = continuation from t*
  B1    candidates g n P (no screening), audit = full child generation
  sel   'ours' for versions in --screen-versions, B1 for the rest (design fixed from
        first-stage data, proposition 12)
Inclusion probabilities follow the slice-cover design (proposition 10). Costs are in
decode-token equivalents: a prefill token costs --prefill-ratio decode tokens.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cert import audit_rate_bounds, cover_design, pi_star, tu_table


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_slices(meta, nbins, cat_field, m):
    """'all', quantile bins of parent output length and prompt length, optional category."""
    n = len(meta)
    sl = {"all": np.ones(n, bool)}
    for feat in ("parent_len", "prompt_len"):
        x = np.array([r[feat] for r in meta], float)
        edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
        b = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, nbins - 1)
        for j in range(nbins):
            sl[f"{feat}_q{j + 1}of{nbins}"] = b == j
    if cat_field:
        for c in sorted({r[cat_field] for r in meta}):
            sl[f"{cat_field}={c}"] = np.array([r[cat_field] == c for r in meta])
    return {k: v for k, v in sl.items() if v.sum() >= m}


def run_method(method, V, slices, tau, delta_p, ratio, reps, rng, screened=(), opts=None):
    opts = opts or {"scope": "all", "lazy": False}
    """Returns per-version expected cost, plus MC coverage/width per (version, slice)."""
    out = {"versions": {}, "cover_fail": np.zeros(reps, bool)}
    for name, v in V.items():
        mv = method if method != "sel" else ("ours" if name in screened else "B1")
        cand = v["P"] & v["D"] if mv == "ours" else v["P"].copy()
        pi = cover_design(slices, cand, tau, delta_p)
        if mv == "ours":
            # Screening prefill. scope=P: only parent-correct questions can be candidates, so
            # only they are screened. lazy: prefill stops at the first divergence (t* + 1
            # tokens of the parent output); D and t* are unchanged under determinism.
            scope = v["P"] if opts["scope"] == "P" else np.ones_like(v["P"])
            plen = np.where(v["D"], v["t_star"] + 1, v["parent_len"]) if opts["lazy"] else v["parent_len"]
            pre = ratio * float(((v["prompt_len"] + plen) * scope).sum())
            unit = v["cont_tokens"] + ratio * (v["prompt_len"] + v["t_star"])
        else:
            pre = 0.0
            unit = v["child_len"] + ratio * v["prompt_len"]
        exp_cost = pre + float((pi * unit).sum())
        R = rng.random((reps, len(pi))) < pi  # independent Poisson sampling per version
        realized = pre + (R * unit).sum(axis=1)
        per_slice = {}
        for g, mask in slices.items():
            Ng = int(mask.sum())
            A = cand & mask
            nA = int(A.sum())
            pmin = float(pi[A].min()) if nA else 1.0
            table = tu_table(nA, pmin, delta_p)
            harm = v["F"] & A  # harm flips inside the candidate set
            K = R[:, harm].sum(axis=1) if harm.any() else np.zeros(reps, int)
            ucb = table[K] / Ng
            H = (v["F"] & v["P"] & mask).sum() / Ng
            theta = H - (v["F"] & ~v["P"] & mask).sum() / Ng
            fail = (ucb < H) | (ucb < theta)
            out["cover_fail"] |= fail
            per_slice[g] = {"Ng": Ng, "n_cand": nA, "n_gP": int((v["P"] & mask).sum()),
                            "pi_min": pmin, "H": H, "theta": theta,
                            "ucb_mean": float(ucb.mean()), "width_mean": float((ucb - H).mean()),
                            "pass_tau_rate": float((ucb < tau).mean()), "cover_rate": float(1 - fail.mean())}
        out["versions"][name] = {
            "expected_cost": exp_cost, "realized_cost_mean": float(realized.mean()),
            "realized_cost_p95": float(np.quantile(realized, 0.95)),
            "full_eval_cost": float((v["child_len"] + ratio * v["prompt_len"]).sum()),
            "n_cand": int(cand.sum()), "expected_audits": float(pi.sum()),
            "worst_slice_ucb_mean": max(s["ucb_mean"] for s in per_slice.values()),
            "slices": per_slice}
    out["simultaneous_coverage"] = float(1 - out["cover_fail"].mean())
    del out["cover_fail"]
    tot_full = sum(x["full_eval_cost"] for x in out["versions"].values())
    tot = sum(x["expected_cost"] for x in out["versions"].values())
    out["speedup_expected"] = tot_full / tot
    return out


def load_version(truth, name, meta_by_id, ids):
    R = {r["id"]: r for r in read_jsonl(truth / "screen" / name / "records.jsonl")}
    rs = [R[i] for i in ids]
    a = lambda f, d=0: np.array([r.get(f) if r.get(f) is not None else d for r in rs])
    return {"P": a("parent_correct").astype(bool), "D": a("in_D").astype(bool),
            "F": (a("parent_correct") != a("child_correct")),
            "child_len": a("child_len").astype(float), "parent_len": a("parent_len").astype(float),
            "cont_tokens": a("cont_decode_tokens").astype(float), "t_star": a("t_star").astype(float),
            "prompt_len": np.array([meta_by_id[i]["prompt_len"] for i in ids], float)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="dir with parent_gen/gen_A.jsonl and screen/<ver>/records.jsonl")
    ap.add_argument("--parent-gen", default=None)
    ap.add_argument("--versions", default="Q8_0,Q6_K,Q4_K_M,Q3_K_M")
    ap.add_argument("--data", default=None, help="test.jsonl, for --cat-field")
    ap.add_argument("--cat-field", default=None)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--nbins", type=int, default=4)
    ap.add_argument("--prefill-ratio", type=float, required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--screen-versions", default="Q8_0")
    ap.add_argument("--scope", default="P", choices=["all", "P"])
    ap.add_argument("--lazy", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    truth = Path(args.truth)
    gen = read_jsonl(args.parent_gen or truth / "parent_gen" / "gen_A.jsonl")
    extra = {r["id"]: r for r in read_jsonl(args.data)} if args.data else {}
    meta = [{"id": r["id"], "prompt_len": r["prompt_len"], "parent_len": len(r["out_ids"]),
             **({args.cat_field: extra[r["id"]][args.cat_field]} if args.cat_field else {})} for r in gen]
    ids = [r["id"] for r in meta]
    meta_by_id = {r["id"]: r for r in meta}
    slices = build_slices(meta, args.nbins, args.cat_field, args.m)
    vers = args.versions.split(",")
    V = {v: load_version(truth, v, meta_by_id, ids) for v in vers}
    S = len(vers) * len(slices)
    delta_p = args.delta / S

    rng = np.random.default_rng(args.seed)
    res = {"N": len(ids), "scope": args.scope, "lazy": bool(args.lazy), "n_slices": len(slices), "S": S, "tau": args.tau, "delta": args.delta,
           "delta_prime": delta_p, "prefill_ratio": args.prefill_ratio, "reps": args.reps,
           "slice_sizes": {g: int(mk.sum()) for g, mk in slices.items()},
           "pi_star": {g: pi_star(int(mk.sum()), args.tau, delta_p) for g, mk in slices.items()},
           "E1_rates_smallest_slice": audit_rate_bounds(min(int(mk.sum()) for mk in slices.values()),
                                                        args.tau, args.delta, S)}
    screened = set(args.screen_versions.split(","))
    for method in ("ours", "B1", "sel"):
        res[method] = run_method(method, V, slices, args.tau, delta_p, args.prefill_ratio, args.reps,
                                 rng, screened, {"scope": args.scope, "lazy": bool(args.lazy)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))

    print(f"N={res['N']} slices={res['n_slices']} |S|={S} delta'={delta_p:.2e} r={args.prefill_ratio}")
    for method in ("ours", "B1", "sel"):
        r = res[method]
        print(f"[{method}] simultaneous coverage={r['simultaneous_coverage']:.4f} "
              f"speedup(expected)={r['speedup_expected']:.2f}")
        for v, x in r["versions"].items():
            print(f"   {v:7} cand={x['n_cand']:5d} E[audits]={x['expected_audits']:7.1f} "
                  f"cost={x['expected_cost']:10.0f} full={x['full_eval_cost']:10.0f} "
                  f"worst-slice UCB={x['worst_slice_ucb_mean']:.3f}")


if __name__ == "__main__":
    main()
