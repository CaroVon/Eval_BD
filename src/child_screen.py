"""Pilot exact screening for one child model (plan sec 2.3 steps 0-1, lemma 3).

For each question:
  full     child greedy generation (ground truth)
  prefill  child teacher-forced along the parent output -> divergence positions, t*
  cont     for questions in D: parent prefix y0[:t*] + child greedy continuation
Checks lemma 3: (a) i not in D -> child output == parent output;
                (b) i in D     -> y0[:t*] + continuation == child full output.
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
from tasks import grade  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parent-gen", required=True, help="gen_A.jsonl of the parent")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable torch.compile and CUDA graphs")
    args = ap.parse_args()

    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    qs = load_questions(data, args.ids)
    by_id = {r["id"]: r for r in read_jsonl(data / "test.jsonl")}
    parent = {r["id"]: r for r in read_jsonl(args.parent_gen)}
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt_ids(tok, q) for q in qs]
    y0 = [parent[q["id"]]["out_ids"] for q in qs]

    llm = LLM(model=args.model, dtype=args.dtype, seed=0, enable_prefix_caching=False,
              async_scheduling=False, max_num_seqs=args.max_num_seqs, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=args.enforce_eager)
    cost = {}

    # full child generation (ground truth)
    t0 = time.time()
    full = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                        SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    cost["full_s"] = time.time() - t0
    yq = [list(o.outputs[0].token_ids) for o in full]
    txt_q = [o.outputs[0].text for o in full]

    # prefill along parent outputs. Exact ties at bf16 logit resolution are common, so
    # take the top-K set and predict the decoded token as the smallest id among the
    # maximisers (torch.argmax rule, definition 2); the prediction is checked below.
    t0 = time.time()
    pre = llm.generate([TokensPrompt(prompt_token_ids=p + y) for p, y in zip(prompts, y0)],
                       SamplingParams(temperature=0.0, max_tokens=1,
                                      prompt_logprobs=args.topk))
    cost["prefill_s"] = time.time() - t0
    div, ties, pred_at, saturated = [], [], [], 0
    for i, o in enumerate(pre):
        P, pl, d, tz, pr = len(prompts[i]), o.prompt_logprobs, [], [], {}
        for t, tk in enumerate(y0[i]):
            e = pl[P + t]
            best = max(v.logprob for v in e.values())
            tied = sorted(k for k, v in e.items() if v.logprob == best)
            if len(tied) >= args.topk:
                saturated += 1
            if len(tied) > 1 and tk in tied:
                tz.append(t)
            pred = tied[0]
            if pred != tk:
                d.append(t)
                pr[t] = pred
        div.append(d)
        ties.append(tz)
        pred_at.append(pr)

    # continuation from t* for divergent questions
    D = [i for i in range(len(qs)) if div[i]]
    tstar = {i: div[i][0] for i in D}
    t0 = time.time()
    cont = llm.generate([TokensPrompt(prompt_token_ids=prompts[i] + y0[i][:tstar[i]]) for i in D],
                        [SamplingParams(temperature=0.0, max_tokens=args.max_tokens - tstar[i])
                         for i in D]) if D else []
    cost["cont_s"] = time.time() - t0
    cont_ids = {i: y0[i][:tstar[i]] + list(o.outputs[0].token_ids) for i, o in zip(D, cont)}

    recs = []
    for i, q in enumerate(qs):
        par_txt = parent[q["id"]]["text"]
        rec = {"id": q["id"], "parent_len": len(y0[i]), "child_len": len(yq[i]),
               "in_D": i in tstar, "t_star": tstar.get(i), "n_div": len(div[i]),
               "tie_pos": ties[i],
               "in_D_conservative": bool(div[i] or ties[i]),
               "parent_correct": grade(par_txt, by_id[q["id"]]),
               "child_correct": grade(txt_q[i], by_id[q["id"]]),
               "child_out_ids": yq[i]}
        if i in tstar:
            t = tstar[i]
            rec["lemma3b_ok"] = cont_ids[i] == yq[i]
            rec["cont_decode_tokens"] = len(yq[i]) - t
            # tie rule: child agrees with parent before t* and decodes the predicted token at t*
            rec["tie_rule_ok"] = yq[i][:t] == y0[i][:t] and len(yq[i]) > t and yq[i][t] == pred_at[i][t]
        else:
            rec["lemma3a_ok"] = yq[i] == y0[i]
            rec["tie_rule_ok"] = rec["lemma3a_ok"]
        recs.append(rec)
    write_jsonl(out / "records.jsonl", recs)

    Pset = [r for r in recs if r["parent_correct"]]
    A = [r for r in Pset if r["in_D"]]
    summ = {
        "name": args.name, "n": len(recs),
        "acc_parent": len(Pset) / len(recs),
        "acc_child": sum(r["child_correct"] for r in recs) / len(recs),
        "D_frac": sum(r["in_D"] for r in recs) / len(recs),
        "A_over_P": len(A) / max(len(Pset), 1),
        "harm_flips_in_P": sum(not r["child_correct"] for r in Pset),
        "repair_flips": sum(r["child_correct"] for r in recs if not r["parent_correct"]),
        "flips_outside_D": sum(r["parent_correct"] != r["child_correct"] for r in recs if not r["in_D"]),
        "lemma3a_violations": sum(not r["lemma3a_ok"] for r in recs if "lemma3a_ok" in r),
        "lemma3b_violations": sum(not r["lemma3b_ok"] for r in recs if "lemma3b_ok" in r),
        "questions_with_ties": sum(bool(r["tie_pos"]) for r in recs),
        "tie_positions": sum(len(r["tie_pos"]) for r in recs),
        "tie_rule_violations": sum(not r["tie_rule_ok"] for r in recs),
        "topk_saturated_positions": saturated,
        "A_over_P_conservative": sum(r["in_D_conservative"] for r in Pset) / max(len(Pset), 1),
        "t_star_median": sorted(r["t_star"] for r in recs if r["in_D"])[len(D) // 2] if D else None,
        "decode_tokens_full": sum(r["child_len"] for r in recs),
        "decode_tokens_cont": sum(r.get("cont_decode_tokens", 0) for r in recs),
        "prefill_tokens": sum(len(p) + len(y) for p, y in zip(prompts, y0)),
        "seconds": cost,
        "config": env_record(args, llm),
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("SCREEN", json.dumps({k: v for k, v in summ.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()
