"""Download MMLU-Pro test split and fix the stratified subsets (decision D6).

3,000 questions with equal allocation across the 14 categories; the 1,000-question pilot is
the first ceil/floor(1000/14) of each category's shuffled list, so it is nested in the 3,000.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tasks import mmlu_pro_prompt  # noqa: E402

REPO = "TIGER-Lab/MMLU-Pro"
SEED = 20261030
SIZES = {"subset_3000": 3000, "pilot_1000": 1000}

out = Path(os.environ["KLAB"]) / "data" / "ICML" / "datasets" / "mmlu_pro"
out.mkdir(parents=True, exist_ok=True)
sha = HfApi().dataset_info(REPO).sha
ds = load_dataset(REPO, split="test", revision=sha)

recs = []
for row in ds:
    recs.append({
        "id": f"mmlupro-{row['question_id']}", "task": "mmlu_pro", "category": row["category"],
        "src": row["src"], "n_options": len(row["options"]), "gold": row["answer"],
        "prompt": mmlu_pro_prompt(row["question"], row["options"], row["category"]),
    })
with open(out / "test.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

cats = sorted({r["category"] for r in recs})
rng = np.random.default_rng(SEED)
by_cat = {c: [r["id"] for r in recs if r["category"] == c] for c in cats}
perm = {c: [by_cat[c][j] for j in rng.permutation(len(by_cat[c]))] for c in cats}


def take(total):
    base, extra = divmod(total, len(cats))
    ids = []
    for j, c in enumerate(cats):
        n = base + (1 if j < extra else 0)
        assert n <= len(perm[c]), (c, n, len(perm[c]))
        ids += perm[c][:n]
    return sorted(ids)


subsets = {name: take(n) for name, n in SIZES.items()}
assert set(subsets["pilot_1000"]) <= set(subsets["subset_3000"])
for name, ids in subsets.items():
    (out / f"{name}_ids.json").write_text(json.dumps(ids, indent=1))

meta = {"repo": REPO, "split": "test", "revision": sha, "n": len(recs), "seed": SEED,
        "categories": {c: len(by_cat[c]) for c in cats},
        "allocation": "equal per category; pilot nested in subset_3000",
        "subset_sizes": {k: len(v) for k, v in subsets.items()}}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
