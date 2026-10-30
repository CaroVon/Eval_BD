"""Measure prefill cost vs decode cost for one child (plan E3, RQ1 'prefill overhead').

All timings are in one process on the same GPU; each setting is run --reps times and the
minimum is reported, to reduce noise from other jobs on the node.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from det_check import build_prompt_ids, env_record, load_questions, read_jsonl  # noqa: E402


def timed(fn, reps):
    best = None
    for _ in range(reps):
        t0 = time.time()
        fn()
        dt = time.time() - t0
        best = dt if best is None else min(best, dt)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parent-gen", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    qs = load_questions(Path(args.data_dir), args.ids)
    parent = {r["id"]: r for r in read_jsonl(args.parent_gen)}
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt_ids(tok, q) for q in qs]
    full = [TokensPrompt(prompt_token_ids=p + parent[q["id"]]["out_ids"]) for p, q in zip(prompts, qs)]
    plain = [TokensPrompt(prompt_token_ids=p) for p in prompts]

    llm = LLM(model=args.model, dtype=args.dtype, seed=0, enable_prefix_caching=False,
              async_scheduling=False, max_num_seqs=256, max_model_len=4096,
              gpu_memory_utilization=0.85)
    res = {"n": len(qs),
           "prompt_tokens": sum(len(p) for p in prompts),
           "prefill_tokens": sum(len(f["prompt_token_ids"]) for f in full)}

    dec_sp = SamplingParams(temperature=0.0, max_tokens=1024)
    out = llm.generate(plain, dec_sp)
    res["decode_tokens"] = sum(len(o.outputs[0].token_ids) for o in out)
    res["decode_s"] = timed(lambda: llm.generate(plain, dec_sp), args.reps)
    res["prefill_s"] = {}
    for k in [None, 1, 2, 20]:
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=k)
        res["prefill_s"][str(k)] = timed(lambda: llm.generate(full, sp), args.reps)
    res["ratio_prefill_over_decode"] = {k: v / res["decode_s"] for k, v in res["prefill_s"].items()}
    res["config"] = env_record(args, llm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("COST", json.dumps({k: v for k, v in res.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()
