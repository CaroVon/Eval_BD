"""Compare Track B (llama.cpp GGUF) greedy outputs with Track A (dequantized fp32 under vLLM).

For every version: accuracy, share of outputs identical to the Track B parent (bf16.gguf) and to the
Track A child, divergence set and first-divergence position relative to the Track B parent,
harm / repair flips relative to the Track B parent and to the Track A parent, and agreement of the
per-question flip indicators between the two tracks.
Usage: trackb_compare.py <trackb_dir> <trackA_truth_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

B, A = Path(sys.argv[1]), Path(sys.argv[2])
rd = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
EOS = None


def strip(x):
    return x[:-1] if x and x[-1] == EOS else x


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


PA = rd(A / "parent_gen" / "gen_A.jsonl")
EOS = next(iter(PA.values()))["out_ids"][-1]
PB = rd(B / "gen_bf16.jsonl")
ids = [i for i in PA if i in PB]
pa_ok = {i: None for i in ids}
recA = {v: rd(A / "screen" / v / "records.jsonl") for v in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]}
for i in ids:
    pa_ok[i] = recA["Q8_0"][i]["parent_correct"]
pb_ok = {i: PB[i]["correct"] for i in ids}
res = {"n": len(ids), "parent": {"acc_A": float(np.mean([pa_ok[i] for i in ids])),
                                  "acc_B": float(np.mean([pb_ok[i] for i in ids])),
                                  "identical_A_B": int(sum(strip(PB[i]["out_ids"]) == strip(PA[i]["out_ids"]) for i in ids)),
                                  "answer_flips_B_vs_A": int(sum(pb_ok[i] != pa_ok[i] for i in ids))}}
print(f"== {B} vs {A}: n={len(ids)}")
print("parent: acc A %.3f B %.3f, identical %d, correctness differs %d" % (
    res["parent"]["acc_A"], res["parent"]["acc_B"], res["parent"]["identical_A_B"], res["parent"]["answer_flips_B_vs_A"]))
for v in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]:
    f = B / f"gen_{v}.jsonl"
    if not f.exists():
        continue
    CB = rd(f)
    vi = [i for i in ids if i in CB]
    if len(vi) < len(ids):
        print(f"{v}: incomplete ({len(vi)}/{len(ids)})")
        continue
    ra = recA[v]
    cA = {i: strip(ra[i]["child_out_ids"]) if ra[i]["in_D"] else strip(PA[i]["out_ids"]) for i in vi}
    d = {i: first_div(strip(CB[i]["out_ids"]), strip(PB[i]["out_ids"])) for i in vi}
    inD = np.array([d[i] is not None for i in vi])
    P = np.array([pb_ok[i] for i in vi])
    cb = np.array([CB[i]["correct"] for i in vi])
    harmB = int(((P == 1) & (cb == 0)).sum()); repB = int(((P == 0) & (cb == 1)).sum())
    flipA = np.array([ra[i]["parent_correct"] != ra[i]["child_correct"] for i in vi])
    flipB = P != cb
    tstar = [d[i] for i in vi if d[i] is not None]
    pos = sum((d[i] + 1) if d[i] is not None else len(strip(PB[i]["out_ids"])) for i in vi)
    r = {"acc_B": float(cb.mean()), "acc_A": float(np.mean([ra[i]["child_correct"] for i in vi])),
         "identical_to_B_parent": int((~inD).sum()), "identical_to_A_child": int(sum(strip(CB[i]["out_ids"]) == cA[i] for i in vi)),
         "A_over_P_B": float(inD[P == 1].mean()), "A_over_P_A": float(np.mean([ra[i]["in_D"] for i in vi if ra[i]["parent_correct"]])),
         "harm_B": harmB, "repair_B": repB,
         "harm_A": int(sum(ra[i]["parent_correct"] and not ra[i]["child_correct"] for i in vi)),
         "repair_A": int(sum((not ra[i]["parent_correct"]) and ra[i]["child_correct"] for i in vi)),
         "flips_outside_D_B": int((flipB & ~inD).sum()),
         "flip_agreement": float((flipA == flipB).mean()), "both_flip": int((flipA & flipB).sum()),
         "t_star_median_B": float(np.median(tstar)) if tstar else None, "rho_B": len(tstar) / pos}
    res[v] = r
    print(f"{v:7} acc A {r['acc_A']:.3f} B {r['acc_B']:.3f} | |A|/|P| A {r['A_over_P_A']:.3f} B {r['A_over_P_B']:.3f} "
          f"| harm/rep A {r['harm_A']}/{r['repair_A']} B {r['harm_B']}/{r['repair_B']} | flips outside D_B {r['flips_outside_D_B']} "
          f"| flip indicators agree {r['flip_agreement']:.3f} (both {r['both_flip']}) | B==A child {r['identical_to_A_child']} "
          f"| t*_B med {r['t_star_median_B']} rho_B {r['rho_B']:.4f}")
(B / "trackb_compare.json").write_text(json.dumps(res, indent=1))
