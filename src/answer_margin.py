"""Do answer flips concentrate where the parent is unsure of its own answer? (root cause D)

Answer tokens in the parent output: the digits after the last '####' (GSM8K) or the letter after
the last 'answer is (' (MMLU-Pro). The parent's answer margin is the minimum parent margin over
those tokens (from the analysis-1 margin profile). Flip rates (harm on P, repair off P) are then
tabulated by answer-margin bin.
Usage: answer_margin.py <truth_dir> <vocab.json> <gsm8k|mmlu_pro>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from classify_divergence import BYTE_DEC  # noqa: E402

truth, vocab_path, task = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
inv = {v: k for k, v in json.load(open(vocab_path, encoding="utf-8")).items()}


def piece(t):
    s = inv.get(int(t), "")
    return bytearray(BYTE_DEC[c] for c in s if c in BYTE_DEC).decode("utf-8", errors="replace")


M = np.load(truth / "margins" / "margins_parent.npz")
m, toks, off = M["gap"], M["tok"], M["offsets"]
nq = len(off) - 1
amarg = np.full(nq, np.nan)
for i in range(nq):
    ts = toks[off[i]:off[i + 1]]
    text, starts = "", []
    for t in ts:
        starts.append(len(text))
        text += piece(t)
    starts = np.array(starts)
    if task == "gsm8k":
        mt = list(re.finditer(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text))
        span = mt[-1].span(1) if mt else None
    else:
        mt = list(re.finditer(r"answer is \(?([A-J])\)?", text))
        span = mt[-1].span(1) if mt else None
    if span is None:
        continue
    idx = [j for j in range(len(ts)) if starts[j] < span[1] and starts[j] + len(piece(ts[j])) > span[0]]
    if idx:
        amarg[i] = float(m[off[i] + np.array(idx)].min())

ok = ~np.isnan(amarg)
print(f"== {truth}: answer span found for {ok.sum()}/{nq} parent outputs; "
      f"answer-margin quantiles 10/50/90%: {np.round(np.nanquantile(amarg, [.1, .5, .9]), 2)}")
bins = [0, 0.5, 1, 2, 4, 8, np.inf]
b = np.digitize(np.nan_to_num(amarg, nan=-1), bins) - 1
out = {"bins": bins[:-1], "versions": {}}
for ver in ["Q8_0", "Q6_K", "Q4_K_M"]:
    R = [json.loads(l) for l in open(truth / "screen" / ver / "records.jsonl", encoding="utf-8")]
    P = np.array([r["parent_correct"] for r in R])
    C = np.array([r["child_correct"] for r in R])
    harm, rep = P & ~C, ~P & C
    rows = []
    for j in range(len(bins) - 1):
        k = ok & (b == j)
        rows.append({"n": int(k.sum()), "n_P": int((k & P).sum()), "n_notP": int((k & ~P).sum()),
                     "harm_rate": float(harm[k & P].mean()) if (k & P).any() else None,
                     "repair_rate": float(rep[k & ~P].mean()) if (k & ~P).any() else None,
                     "harm": int(harm[k].sum()), "repair": int(rep[k].sum())})
    out["versions"][ver] = rows
    share_low = (harm & ok & (amarg < 1)).sum() / max((harm & ok).sum(), 1)
    print(f"  {ver:7} flips by parent answer margin " + " ".join(
        f"[{bins[j]},{bins[j + 1]}): harm {r['harm']}/{r['n_P']} rep {r['repair']}/{r['n_notP']};" for j, r in enumerate(rows))
          + f"  share of harms at margin<1: {share_low:.2f}")
(truth / "answer_margin.json").write_text(json.dumps(out, indent=1))
