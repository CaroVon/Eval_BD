"""GSM8K test with fixed k-shot exemplars as prior chat turns (datasets/gsm8k_{k}shot).

Exemplars: the first k train examples at the same pinned revision as the test split
(datasets/gsm8k/meta.json), with calculator annotations <<...>> removed. Each test record gets
a 'fewshot' list of {question, answer}; det_check.build_prompt_ids turns it into user/assistant
turns before the question. Ids and gold answers are unchanged.
Env: KLAB, K (default 8).
"""
import json
import os
import re
from pathlib import Path

from datasets import load_dataset

K = int(os.environ.get("K", "8"))
base = Path(os.environ["KLAB"]) / "data" / "ICML" / "datasets"
src = base / "gsm8k"
meta0 = json.loads((src / "meta.json").read_text())
out = base / f"gsm8k_{K}shot"
out.mkdir(parents=True, exist_ok=True)

train = load_dataset(meta0["repo"], meta0["config"], split="train", revision=meta0["revision"])
shots = [{"question": train[i]["question"], "answer": re.sub(r"<<[^>]*>>", "", train[i]["answer"])}
         for i in range(K)]
n = 0
with open(src / "test.jsonl", encoding="utf-8") as fi, open(out / "test.jsonl", "w", encoding="utf-8") as fo:
    for line in fi:
        r = json.loads(line)
        r["fewshot"] = shots
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
for f in ("all_test_ids.json", "determinism_subset_ids.json"):
    (out / f).write_text((src / f).read_text())
meta = {**meta0, "fewshot_k": K, "fewshot_source": "train[:K], <<...>> removed", "n": n}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
