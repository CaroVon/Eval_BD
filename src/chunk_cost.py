"""Per-token cost of verifying short chunks on top of a cached prefix, vs decode (cost only).

Prefix caching is ON here (it is off in every truth-matrix run); this script produces no truth
data. For each question the base is prompt + first k parent tokens, truncated so the base length
is a multiple of the KV block size (so the cached prefix is fully reused). Each chunk request
extends the base by L parent tokens with max_tokens = 1; its time over N*L tokens is the chunk
cost per token. Decode cost per token is measured from the same bases with max_tokens = 64.
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

from det_check import build_prompt_ids, load_questions, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parent-gen", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    qs = load_questions(Path(args.data_dir), args.ids)
    par = {r["id"]: r["out_ids"] for r in read_jsonl(args.parent_gen)}
    tok = AutoTokenizer.from_pretrained(args.model)
    bases, rests = [], []
    for q in qs:
        y = par[q["id"]]
        if len(y) < args.k + 80:
            continue
        b = build_prompt_ids(tok, q) + y[:args.k]
        cut = len(b) - len(b) % args.block
        bases.append(b[:cut])
        rests.append((b[cut:] + y[args.k:]))
        if len(bases) == args.n:
            break
    llm = LLM(model=args.model, dtype=args.dtype, seed=0, enable_prefix_caching=True,
              async_scheduling=False, max_num_seqs=256, max_model_len=4096,
              gpu_memory_utilization=0.85, block_size=args.block)
    one = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate([TokensPrompt(prompt_token_ids=b) for b in bases], one)  # warm the prefix cache
    res = {"n": len(bases), "k": args.k, "block": args.block, "chunk": {}}
    for L in (1, 2, 4, 8, 16, 32, 64):
        reqs = [TokensPrompt(prompt_token_ids=b + r[:L]) for b, r in zip(bases, rests)]
        llm.reset_prefix_cache()  # drop blocks cached by earlier chunk lengths
        llm.generate([TokensPrompt(prompt_token_ids=b) for b in bases], one)  # re-warm base only
        t0 = time.time()
        llm.generate(reqs, one)
        dt = time.time() - t0
        res["chunk"][L] = {"seconds": dt, "per_token": dt / (len(bases) * L)}
    llm.reset_prefix_cache()
    llm.generate([TokensPrompt(prompt_token_ids=b) for b in bases], one)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=b) for b in bases],
                        SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    res["decode"] = {"seconds": dt, "tokens": ntok, "per_token": dt / ntok}
    res["r_chunk"] = {L: v["per_token"] / res["decode"]["per_token"] for L, v in res["chunk"].items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("CHUNK", json.dumps({"r_chunk": res["r_chunk"], "decode": res["decode"]}), flush=True)


if __name__ == "__main__":
    main()
