"""Break down pilot screening: how much of D comes from exact ties vs strict divergences."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(f"{'model':8} {'|D|':>4} {'strictD':>7} {'tieOnly':>7} {'t*=tie':>6} {'t*=0':>5} "
      f"{'A/P':>6} {'A/P strict':>10} {'save%':>6} {'save% strict':>12}")
for name in ["parent", "Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]:
    f = root / name / "records.jsonl"
    if not f.exists():
        continue
    R = [json.loads(l) for l in open(f)]
    D = [r for r in R if r["in_D"]]
    strict = [r for r in D if r["n_div"] > 0]
    tie_only = [r for r in D if r["n_div"] == 0]
    tstar_tie = [r for r in D if r["t_star"] in r["tie_pos"]]
    t0 = [r for r in D if r["t_star"] == 0]
    P = [r for r in R if r["parent_correct"]]
    A = [r for r in P if r["in_D"]]
    A_strict = [r for r in P if r["n_div"] > 0]
    full = sum(r["child_len"] for r in R)
    cont = sum(r.get("cont_decode_tokens", 0) for r in R)
    print(f"{name:8} {len(D):4d} {len(strict):7d} {len(tie_only):7d} {len(tstar_tie):6d} {len(t0):5d} "
          f"{len(A)/len(P):6.3f} {len(A_strict)/len(P):10.3f} {100*(1-cont/full):6.1f}")
    if name == "Q4_K_M":
        ex = [(r["id"], r["t_star"], r["tie_pos"][:3]) for r in t0[:5]]
        print("   t*=0 examples:", ex)
