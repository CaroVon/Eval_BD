"""Four-layer determinism check for vLLM batch-invariant mode (research plan v4.0, sec 2.2).

One process = one engine config. Phases (--phases):
  gen      full greedy generation (run tag given by --tag); saves token ids
  prefill  teacher-forced prefill along a reference generation; per-position top-1
  cont     continuation from cut points of a reference generation
Comparison/report is done by det_report.py.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import torch  # noqa: E402
import vllm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from tasks import user_content  # noqa: E402


def load_questions(data_dir, ids_file):
    rows = {}
    with open(data_dir / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    ids = json.loads(Path(ids_file).read_text())
    return [rows[i] for i in ids]


def build_prompt_ids(tok, rec):
    msgs = [{"role": "user", "content": user_content(rec)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    # Newer transformers return a BatchEncoding here rather than a list of ids.
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    ids = [int(t) for t in ids]
    assert ids and all(isinstance(t, int) for t in ids)
    return ids


def env_record(args, llm):
    return {
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "node": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("VLLM_", "CUBLAS", "CUDA_VISIBLE"))},
        "args": vars(args),
    }


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def phase_gen(llm, qs, prompts, args, out_dir):
    order = list(range(len(qs)))
    if args.shuffle:
        random.Random(args.shuffle).shuffle(order)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in order], sp)
    dt = time.time() - t0
    recs = [None] * len(qs)
    for j, i in enumerate(order):
        o = outs[j].outputs[0]
        recs[i] = {"id": qs[i]["id"], "prompt_len": len(prompts[i]),
                   "out_ids": list(o.token_ids), "text": o.text, "finish": o.finish_reason}
    write_jsonl(out_dir / f"gen_{args.tag}.jsonl", recs)
    return {"phase": "gen", "tag": args.tag, "seconds": dt,
            "tokens": sum(len(r["out_ids"]) for r in recs)}


def phase_prefill(llm, qs, prompts, args, out_dir):
    ref = read_jsonl(out_dir / f"gen_{args.ref}.jsonl")
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    full = [prompts[i] + ref[i]["out_ids"] for i in range(len(qs))]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=f) for f in full], sp)
    dt = time.time() - t0
    recs = []
    for i, o in enumerate(outs):
        pl, P = o.prompt_logprobs, len(prompts[i])
        mism, ties = [], []
        for t, tok in enumerate(ref[i]["out_ids"]):
            d = pl[P + t]
            top = min(d.items(), key=lambda kv: kv[1].rank)
            if top[0] != tok:
                if abs(top[1].logprob - d[tok].logprob) == 0.0:
                    ties.append(t)
                else:
                    mism.append(t)
        recs.append({"id": qs[i]["id"], "n_out": len(ref[i]["out_ids"]),
                     "mismatch_pos": mism, "tie_pos": ties})
    write_jsonl(out_dir / f"prefill_vs_{args.ref}.jsonl", recs)
    return {"phase": "prefill", "ref": args.ref, "seconds": dt,
            "tokens": sum(len(f) for f in full)}


def phase_cont(llm, qs, prompts, args, out_dir):
    ref = read_jsonl(out_dir / f"gen_{args.ref}.jsonl")
    reqs, meta = [], []
    for i, r in enumerate(ref):
        n = len(r["out_ids"])
        for frac in (1 / 3, 2 / 3):
            t = max(1, int(n * frac))
            if t >= n:
                continue
            reqs.append(TokensPrompt(prompt_token_ids=prompts[i] + r["out_ids"][:t]))
            meta.append((i, t))
    sp = [SamplingParams(temperature=0.0, max_tokens=args.max_tokens - t) for _, t in meta]
    t0 = time.time()
    outs = llm.generate(reqs, sp)
    dt = time.time() - t0
    recs = []
    for (i, t), o in zip(meta, outs):
        cont = ref[i]["out_ids"][:t] + list(o.outputs[0].token_ids)
        recs.append({"id": qs[i]["id"], "cut": t, "equal": cont == ref[i]["out_ids"],
                     "first_diff": next((k for k, (a, b) in enumerate(zip(cont, ref[i]["out_ids"]))
                                         if a != b), None),
                     "len_cont": len(cont), "len_ref": len(ref[i]["out_ids"])})
    write_jsonl(out_dir / f"cont_vs_{args.ref}.jsonl", recs)
    return {"phase": "cont", "ref": args.ref, "seconds": dt, "n": len(recs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--phases", default="gen")
    ap.add_argument("--tag", default="A")
    ap.add_argument("--ref", default="A")
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = load_questions(Path(args.data_dir), args.ids)
    if args.limit:
        qs = qs[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt_ids(tok, q) for q in qs]

    llm = LLM(model=args.model, dtype=args.dtype, seed=0,
              enable_prefix_caching=False, async_scheduling=False,
              max_num_seqs=args.max_num_seqs, max_model_len=4096,
              gpu_memory_utilization=0.85)

    log = {"config": env_record(args, llm), "phases": []}
    for ph in args.phases.split(","):
        fn = {"gen": phase_gen, "prefill": phase_prefill, "cont": phase_cont}[ph]
        res = fn(llm, qs, prompts, args, out_dir)
        print(json.dumps(res), flush=True)
        log["phases"].append(res)
    name = f"run_{args.tag}_{args.phases.replace(',', '-')}.json"
    (out_dir / name).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
