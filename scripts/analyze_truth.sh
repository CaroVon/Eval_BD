#!/bin/bash
# Run every local analysis on the Phi-3.5 GSM8K truth matrix once it has been downloaded to phi_truth/.
set -e
cd "$(dirname "$0")"
PY="/c/Users/ZYF/anaconda3/python -X utf8"
T=${1:-phi_truth}
TOK=${2:-phi_tokenizer.json}
echo "### decode check"
$PY - $TOK $T <<'EOF'
import json, sys
sys.path.insert(0, 'src')
import tokdecode
text = tokdecode.load(sys.argv[1])
G = [json.loads(l) for l in open(sys.argv[2] + '/parent_gen/gen_A.jsonl', encoding='utf-8')]
ok = sum(text(g['out_ids']).strip() == g['text'].strip() for g in G)
print(f'decode matches vLLM text: {ok}/{len(G)}')
EOF
echo "### hazard model + cascade"; $PY src/diagnose.py $T $TOK 2>&1 | grep -v "top t\*=0" || true
echo "### margins"; [ -f $T/margins/margins_Q4_K_M.npz ] && $PY src/analyze_margins.py $T $TOK
echo "### divergence classes"; $PY src/classify_divergence.py $T $TOK full_truth/test.jsonl
echo "### predictor"; [ -f $T/margins/margins_Q4_K_M.npz ] && $PY src/predict_D.py $T 50
echo "### answer margin"; [ -f $T/margins/margins_parent.npz ] && $PY src/answer_margin.py $T $TOK gsm8k
echo "### certificate sims"
$PY src/simulate_cert.py --truth $T --tau 0.05 --nbins 4 --m 100 --prefill-ratio 0.365 --reps 2000 --scope P --lazy 1 --out $T/sim_t05_q4_r0.365.json | tail -16
(cd src && $PY eq_budget.py ../$T 0.365)
