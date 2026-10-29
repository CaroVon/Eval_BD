#!/bin/bash
set -e
cd ~/myproject/ICML
mv -f gitignore .gitignore
bash scripts/collect_results.sh
[ -d .git ] || git init -q -b main
git config user.name "CaroVon"
git config user.email "caroyf@uchicago.edu"
git add .gitignore README.md DECISIONS.md RESULTS.md src scripts results configs
git status --short | head -80
echo "--- files: $(git diff --cached --name-only | wc -l)  size: $(git diff --cached --name-only | xargs du -ch 2>/dev/null | tail -1)"
git commit -q -F - <<'EOF'
Track A pilot: determinism checks, quantization ladder, exact screening

- Four-layer determinism check for vLLM batch-invariant mode, with a
  batch-invariance-off control
- llama.cpp quantization ladder (Q8_0, Q6_K, Q4_K_M, Q3_K_M) and
  dequantization to bf16/fp32 safetensors
- Exact screening along parent outputs with lemma 3 and tie-rule checks
- Pilot results on 200 GSM8K questions; execution dtype switched to fp32
  (DECISIONS.md D4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git remote get-url origin >/dev/null 2>&1 || git remote add origin git@github.com:CaroVon/Eval_BD.git
git push -u origin main 2>&1 | tail -5
git log --oneline -1
