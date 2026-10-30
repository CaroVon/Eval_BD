"""Summarize the four-layer determinism check written by det_check.py."""
import json
import sys
from pathlib import Path

from tasks import answer, grade


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def compare_gen(a, b, recs):
    diff = [i for i in range(len(a)) if a[i]["out_ids"] != b[i]["out_ids"]]
    score = [i for i in diff if answer(a[i]["text"], recs.get(a[i]["id"], {})) != answer(b[i]["text"], recs.get(b[i]["id"], {}))]
    return {"n": len(a), "token_seq_diff": len(diff), "answer_diff": len(score),
            "diff_ids": [a[i]["id"] for i in diff][:20]}


def main(out_dir):
    d = Path(out_dir)
    recs = {}
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if data:
        for r in read_jsonl(data / "test.jsonl"):
            recs[r["id"]] = r
    A = read_jsonl(d / "gen_A.jsonl")
    rep = {}
    if recs:
        rep["accuracy_A"] = sum(grade(r["text"], recs[r["id"]]) for r in A) / len(A)
    rep["hit_max_tokens_A"] = sum(r["finish"] == "length" for r in A)
    rep["mean_out_tokens_A"] = sum(len(r["out_ids"]) for r in A) / len(A)

    # Layer 1: same config, repeated run
    if (d / "gen_B.jsonl").exists():
        rep["L1_repeat_A_vs_B"] = compare_gen(A, read_jsonl(d / "gen_B.jsonl"), recs)

    # Layer 2: prefill top-1 vs decoded token
    p = read_jsonl(d / "prefill_vs_A.jsonl")
    ntok = sum(r["n_out"] for r in p)
    nmis = sum(len(r["mismatch_pos"]) for r in p)
    rep["L2_prefill_vs_decode"] = {
        "tokens": ntok, "mismatch_tokens": nmis, "token_rate": nmis / max(ntok, 1),
        "questions_with_mismatch": sum(bool(r["mismatch_pos"]) for r in p),
        "questions_with_exact_tie": sum(bool(r["tie_pos"]) for r in p),
        "examples": [(r["id"], r["mismatch_pos"][:5]) for r in p if r["mismatch_pos"]][:10]}

    # Layer 3: continuation from cut point equals full generation
    c = read_jsonl(d / "cont_vs_A.jsonl")
    bad = [r for r in c if not r["equal"]]
    rep["L3_continuation"] = {"n": len(c), "not_equal": len(bad),
                              "questions_affected": len({r["id"] for r in bad}),
                              "examples": bad[:10]}

    # Layer 4: different max_num_seqs + shuffled order
    for tag in ("C",):
        f = d / f"gen_{tag}.jsonl"
        if f.exists():
            rep[f"L4_batch_A_vs_{tag}"] = compare_gen(A, read_jsonl(f), recs)

    verdict = all([
        rep.get("L1_repeat_A_vs_B", {}).get("token_seq_diff", 1) == 0,
        rep["L2_prefill_vs_decode"]["mismatch_tokens"] == 0,
        rep["L3_continuation"]["not_equal"] == 0,
        rep.get("L4_batch_A_vs_C", {}).get("token_seq_diff", 1) == 0,
    ])
    rep["ALL_FOUR_LAYERS_PASS"] = verdict
    (d / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
