"""Margin profile along the parent outputs (analysis 1).

For one model, prefill prompt + parent output with top-2 prompt logprobs and record, at every
parent-output position t, the margin of the parent token y0_t:
    gap_t = logprob(y0_t) - max_{v != y0_t} logprob(v)
For the parent itself gap_t >= 0 (y0_t is its greedy choice); for a child, gap_t < 0 exactly at
teacher-forced divergences. Log-prob differences equal logit differences.
Output: npz with concatenated gaps, parent token ids, and per-question offsets.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from det_check import build_prompt_ids, load_questions, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parent-gen", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    qs = load_questions(Path(args.data_dir), args.ids)
    parent = {r["id"]: r for r in read_jsonl(args.parent_gen)}
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt_ids(tok, q) for q in qs]
    y0 = [parent[q["id"]]["out_ids"] for q in qs]

    llm = LLM(model=args.model, dtype=args.dtype, seed=0, enable_prefix_caching=False,
              async_scheduling=False, max_num_seqs=256, max_model_len=4096,
              gpu_memory_utilization=0.85)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p + y) for p, y in zip(prompts, y0)],
                        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=2))
    dt = time.time() - t0

    gaps, toks, offsets = [], [], [0]
    for i, o in enumerate(outs):
        P, pl = len(prompts[i]), o.prompt_logprobs
        for t, tk in enumerate(y0[i]):
            e = pl[P + t]
            other = max(v.logprob for k, v in e.items() if k != tk)
            gaps.append(e[tk].logprob - other)
            toks.append(tk)
        offsets.append(len(gaps))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"margins_{args.name}.npz", gap=np.array(gaps, np.float32),
                        tok=np.array(toks, np.int32), offsets=np.array(offsets, np.int64),
                        ids=np.array([q["id"] for q in qs]))
    g = np.array(gaps)
    print(f"MARGIN {args.name} positions={len(g)} neg={int((g < 0).sum())} zero={int((g == 0).sum())} "
          f"median={np.median(g):.3f} seconds={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
