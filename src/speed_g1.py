"""End-to-end wall-clock of exact screening for the G1 speedup gate (status doc sec. 7).

Each variant runs in its own process after a warm-up, so JIT compilation and engine start-up are not
timed. Variants differ only in implementation choices that do not change the certificate:
  --topk K            prompt logprobs per position in the screening prefill (1 suffices without
                      exact ties; ties are still detected when the parent token ties the top-1)
  --prefix-caching    continuation reuses the KV blocks of the screening prefill
  --dtype             float32 (adopted) or bfloat16 (exact ties possible; tie rule as child_screen)
Modes:
  gen     greedy generation, writes gen.jsonl (out_ids) and timing (used for a bf16 parent and as the
          bf16 child reference)
  screen  [full generation, unless --skip-full] + screening prefill + continuation from t*.
          Lemma 3 is checked against --ref (records.jsonl with child_out_ids, or gen.jsonl), or
          against the in-process full generation.
Output: <out-dir>/<name>.json
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

from det_check import build_prompt_ids, env_record, load_questions, read_jsonl, write_jsonl  # noqa: E402

N_WARM = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "screen"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parent-gen", help="gen_A.jsonl or gen.jsonl of the parent (screen mode)")
    ap.add_argument("--ref", help="reference child outputs: records.jsonl or gen.jsonl")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qs = load_questions(Path(args.data_dir), args.ids)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt_ids(tok, q) for q in qs]
    llm = LLM(model=args.model, dtype=args.dtype, seed=0, enable_prefix_caching=args.prefix_caching,
              async_scheduling=False, max_num_seqs=args.max_num_seqs, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=args.enforce_eager)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # warm-up of every request type (JIT kernels, graph capture); cache cleared afterwards
    w = [TokensPrompt(prompt_token_ids=p) for p in prompts[:N_WARM]]
    llm.generate(w, SamplingParams(temperature=0.0, max_tokens=16))
    llm.generate(w, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk))
    if args.prefix_caching:
        llm.reset_prefix_cache()
    res = {"name": args.name, "mode": args.mode, "n": len(qs), "dtype": args.dtype, "topk": args.topk,
           "prefix_caching": args.prefix_caching, "seconds": {}}
    sec = res["seconds"]

    full_ids = None
    if args.mode == "gen" or not args.skip_full:
        t0 = time.time()
        full = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], greedy)
        sec["full_s"] = time.time() - t0
        full_ids = [list(o.outputs[0].token_ids) for o in full]
        res["decode_tokens_full"] = sum(map(len, full_ids))
        if args.mode == "gen":
            write_jsonl(out / f"{args.name}_gen.jsonl",
                        [{"id": q["id"], "out_ids": y, "text": o.outputs[0].text}
                         for q, y, o in zip(qs, full_ids, full)])
        if args.prefix_caching:
            llm.reset_prefix_cache()  # the screen must not reuse blocks of the full generation

    if args.mode == "screen":
        parent = {r["id"]: r["out_ids"] for r in read_jsonl(args.parent_gen)}
        y0 = [parent[q["id"]] for q in qs]
        t0 = time.time()
        pre = llm.generate([TokensPrompt(prompt_token_ids=p + y) for p, y in zip(prompts, y0)],
                           SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk))
        sec["prefill_s"] = time.time() - t0
        t1 = time.time()
        tstar, n_ties = {}, 0
        for i, o in enumerate(pre):
            P, pl = len(prompts[i]), o.prompt_logprobs
            for t, tk in enumerate(y0[i]):
                e = pl[P + t]
                best = max(v.logprob for v in e.values())
                tied = sorted(k for k, v in e.items() if v.logprob == best)
                n_ties += len(tied) > 1
                if tied[0] != tk:  # smallest id among maximisers (definition 2)
                    tstar[i] = t
                    break
        sec["scan_s"] = time.time() - t1
        D = sorted(tstar)
        t0 = time.time()
        cont = llm.generate([TokensPrompt(prompt_token_ids=prompts[i] + y0[i][:tstar[i]]) for i in D],
                            [SamplingParams(temperature=0.0, max_tokens=args.max_tokens - tstar[i])
                             for i in D]) if D else []
        sec["cont_s"] = time.time() - t0
        cont_ids = {i: y0[i][:tstar[i]] + list(o.outputs[0].token_ids) for i, o in zip(D, cont)}

        if full_ids is None:
            ref = {r["id"]: r.get("child_out_ids", r.get("out_ids")) for r in read_jsonl(args.ref)}
            full_ids = [ref[q["id"]] for q in qs]
        res["lemma3a_violations"] = sum(full_ids[i] != y0[i] for i in range(len(qs)) if i not in tstar)
        res["lemma3b_violations"] = sum(cont_ids[i] != full_ids[i] for i in D)
        res["D_frac"] = len(D) / len(qs)
        res["tie_positions"] = n_ties
        res["decode_tokens_cont"] = sum(len(full_ids[i]) - tstar[i] for i in D)
        res["decode_tokens_full_ref"] = sum(map(len, full_ids))
        res["prefill_tokens"] = sum(len(p) + len(y) for p, y in zip(prompts, y0))
        if "full_s" in sec:
            res["speedup"] = sec["full_s"] / (sec["prefill_s"] + sec["scan_s"] + sec["cont_s"])
    res["config"] = env_record(args, llm)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    print("SPEED", json.dumps({k: v for k, v in res.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()
