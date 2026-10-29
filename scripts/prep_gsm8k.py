"""Download GSM8K test split, save as JSONL, and fix the determinism-check subset."""
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

REPO = "openai/gsm8k"
SEED = 20261028
N_SUBSET = 200

out_dir = Path(os.environ["KLAB"]) / "data" / "ICML" / "datasets" / "gsm8k"
out_dir.mkdir(parents=True, exist_ok=True)

sha = HfApi().dataset_info(REPO).sha
ds = load_dataset(REPO, "main", split="test", revision=sha)

with open(out_dir / "test.jsonl", "w", encoding="utf-8") as f:
    for idx, row in enumerate(ds):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        rec = {"id": f"gsm8k-test-{idx:04d}", "question": row["question"],
               "answer": row["answer"], "gold": gold}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

rng = np.random.default_rng(SEED)
subset = sorted(rng.choice(len(ds), size=N_SUBSET, replace=False).tolist())
ids = [f"gsm8k-test-{i:04d}" for i in subset]
(out_dir / "determinism_subset_ids.json").write_text(json.dumps(ids, indent=1))

meta = {"repo": REPO, "config": "main", "split": "test", "revision": sha,
        "n": len(ds), "subset_seed": SEED, "subset_size": N_SUBSET}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
