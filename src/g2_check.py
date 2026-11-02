"""G2 gate of plan v4.0 sec. 2.7: on slices with N_g >= 500, the audit rate the uniform certificate
needs at tau = 0.05 should not exceed 0.5.

Exact computation, no Monte Carlo. For version v and slice g with candidate set C (ours: g n P n D;
B1: g n P), uniform inclusion probability pi on C, and true harm count T on C, K ~ Bin(T, pi). The
slice is certified when T_U(K) <= floor(tau N_g) (theorem 4 at delta' = delta / |S|). The required
rate is the smallest pi with P(certified) >= 1/2, found by bisection; P(certified) is
non-decreasing in pi.
Slices: 'all' plus quantile bins of parent output length and prompt length, with
nbins = min(4, N // 500), keeping bins with N_g >= 500.
Usage: g2_check.py <scratch_root> <out.json> [tau] [delta]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

from cert import audit_rate_bounds, tu_table
from simulate_cert import build_slices, load_version, read_jsonl

root, outp = Path(sys.argv[1]), Path(sys.argv[2])
tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
VERS = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
TRUTH = [("Qwen2.5-3B GSM8K", "full_truth"), ("Qwen2.5-7B GSM8K", "q7_truth"),
         ("Qwen2.5-14B GSM8K", "q14_truth"), ("Llama-3.1-8B GSM8K", "l8_truth"),
         ("Phi-3.5-mini GSM8K", "phi_truth"), ("SmolLM2-1.7B GSM8K", "smol_truth"),
         ("Qwen2.5-3B MMLU-Pro 1k", "mmlu_truth"), ("Qwen2.5-14B MMLU-Pro 1k", "q14m_truth"),
         ("Llama-3.1-8B MMLU-Pro 1k", "l8m_truth"), ("Phi-3.5-mini MMLU-Pro 1k", "phim_truth"),
         ("Qwen2.5-3B GSM8K imatrix", "imat4_truth"), ("Qwen2.5-3B MMLU-Pro 1k imatrix", "mimat4_truth")]


def p_cert(n, T, pi, kmax, dp):
    # T_U(k) = max{t <= n : P(Bin(t, pi) <= k) > dp} and the cdf falls in t, so
    # T_U(k) <= kmax  <=>  n <= kmax  or  P(Bin(kmax + 1, pi) <= k) <= dp. O(T) per call.
    if n <= kmax:
        return 1.0
    if pi <= 0:
        return 0.0
    ks = np.arange(T + 1)
    ok = binom.cdf(ks, kmax + 1, pi) <= dp
    return float(binom.pmf(ks, T, pi) @ ok)


def need(n, T, kmax, dp):
    if p_cert(n, T, 1.0, kmax, dp) < 0.5:
        return None  # not certifiable even with a full audit (T > kmax)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if p_cert(n, T, mid, kmax, dp) >= 0.5 else (mid, hi)
    return hi


res, md = {}, ["| Parent / task | slices (N_g) | version | worst H_g | pi ours | pi B1 | "
               "audited share of slice, ours / B1 | G2 |", "|---|---|---|---|---|---|---|---|"]
for lab, td in TRUTH:
    T_dir = root / td
    if not (T_dir / "parent_gen" / "gen_A.jsonl").exists():
        continue
    gen = read_jsonl(T_dir / "parent_gen" / "gen_A.jsonl")
    meta = [{"id": r["id"], "prompt_len": r["prompt_len"], "parent_len": len(r["out_ids"])} for r in gen]
    ids = [r["id"] for r in meta]
    mb = {r["id"]: r for r in meta}
    N = len(ids)
    slices = build_slices(meta, max(1, min(4, N // 500)), None, 500)
    vers = [v for v in VERS if (T_dir / "screen" / v / "records.jsonl").exists()]
    S = len(vers) * len(slices)
    dp = delta / S
    res[lab] = {"N": N, "slices": {g: int(m.sum()) for g, m in slices.items()}, "S": S, "delta_prime": dp,
                "E1_zero_harm": audit_rate_bounds(min(int(m.sum()) for m in slices.values()), tau, delta, S),
                "versions": {}}
    for v in vers:
        V = load_version(T_dir, v, mb, ids)
        rows = {}
        for g, m in slices.items():
            Ng, kmax = int(m.sum()), math.floor(tau * int(m.sum()))
            P, D, F = V["P"] & m, V["P"] & m & V["D"], V["F"]
            H = int((F & P).sum())
            r = {"N_g": Ng, "H": H, "H_rate": H / Ng, "nA": int(D.sum()), "nP": int(P.sum())}
            r["pi_ours"] = need(int(D.sum()), H, kmax, dp)
            r["pi_B1"] = need(int(P.sum()), H, kmax, dp)
            rows[g] = r
        res[lab]["versions"][v] = rows
        cert = [r for r in rows.values() if r["pi_ours"] is not None]
        worstH = max(r["H_rate"] for r in rows.values())
        if len(cert) < len(rows):
            md.append(f"| {lab} | {len(slices)} ({min(r['N_g'] for r in rows.values())}-{N}) | {v} | "
                      f"{worstH:.3f} | - | - | - | H >= tau on {len(rows) - len(cert)} slice(s) |")
            continue
        po = max(r["pi_ours"] for r in rows.values())
        pb = max(r["pi_B1"] for r in rows.values())
        sh_o = max(r["pi_ours"] * r["nA"] / r["N_g"] for r in rows.values())
        sh_b = max(r["pi_B1"] * r["nP"] / r["N_g"] for r in rows.values())
        md.append(f"| {lab} | {len(slices)} ({min(r['N_g'] for r in rows.values())}-{N}) | {v} | {worstH:.3f} | "
                  f"{po:.3f} | {pb:.3f} | {sh_o:.3f} / {sh_b:.3f} | {'pass' if po <= 0.5 else 'fail'} |")
outp.write_text(json.dumps(res, indent=1))
print("\n".join(md))
