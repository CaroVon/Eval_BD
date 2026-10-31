#!/bin/bash
# Copy small result files from the cluster outputs into results/ for version control.
set -e
K=/net/spaces/jevans/klab/caroyf
R=~/myproject/ICML/results
O=$K/outputs/ICML
mkdir -p $R/detcheck $R/screen $R/manifests $R/pins $R/data

for d in qwen2.5-3b_bf16_2272378 qwen2.5-3b_bf16_NOBI_2272437 qwen2.5-3b_fp32_2273534; do
  mkdir -p $R/detcheck/$d
  cp $O/detcheck/$d/report.json $O/detcheck/$d/run_*.json $R/detcheck/$d/
  [ -f $O/detcheck/$d/gpu.csv ] && cp $O/detcheck/$d/gpu.csv $R/detcheck/$d/
done
for d in qwen2.5-3b_2273233 qwen2.5-3b_fp32_2273534; do
  for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
    mkdir -p $R/screen/$d/$m
    cp $O/pilot_screen/$d/$m/summary.json $R/screen/$d/$m/
  done
done
for t in bf16 fp32; do
  for q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
    cp $K/data/ICML/models/dequant_$t/qwen2.5-3b-instruct-$q/MANIFEST.json $R/manifests/${t}_$q.json
  done
done
cp $K/data/ICML/envs_lock/*.txt $R/pins/
cp ~/myproject/ICML/third_party/llama.cpp.PINNED $R/pins/
cp $K/data/ICML/models/hf/Qwen2.5-3B-Instruct/PINNED_REVISION.json $R/pins/qwen2.5-3b-instruct.json
cp $K/data/ICML/datasets/gsm8k/meta.json $K/data/ICML/datasets/gsm8k/determinism_subset_ids.json $R/data/

# full GSM8K truth matrix: screening summaries + certificate simulations
T=$O/truth/gsm8k_full_fp32_qwen2.5-3b_2273994
mkdir -p $R/truth/gsm8k_full_fp32
for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  cp $T/screen/$m/summary.json $R/truth/gsm8k_full_fp32/summary_$m.json
done
cp $T/sim_*.json $R/truth/gsm8k_full_fp32/ 2>/dev/null || true
cp $T/divergence_classes.json $R/truth/gsm8k_full_fp32/ 2>/dev/null || true
cp $T/margins/analysis.json $R/truth/gsm8k_full_fp32/margin_analysis.json 2>/dev/null || true
T=$O/truth/mmlupro_pilot1000_fp32_qwen2.5-3b_2274801
mkdir -p $R/truth/mmlupro_pilot1000_fp32
cp $T/divergence_classes.json $R/truth/mmlupro_pilot1000_fp32/ 2>/dev/null || true
cp $T/net_*.json $R/truth/mmlupro_pilot1000_fp32/ 2>/dev/null || true
cp $O/truth/gsm8k_full_fp32_qwen2.5-3b_2273994/net_*.json $R/truth/gsm8k_full_fp32/ 2>/dev/null || true
for pair in "gsm8k_full_fp32_qwen2.5-3b_2273994 gsm8k_full_fp32" "mmlupro_pilot1000_fp32_qwen2.5-3b_2274801 mmlupro_pilot1000_fp32"; do
  set -- $pair
  cp $O/truth/$1/reanchor*.json $R/truth/$2/ 2>/dev/null || true
  cp $O/truth/$1/margins/predict_D_*.json $R/truth/$2/ 2>/dev/null || true
  cp $O/truth/$1/answer_margin.json $R/truth/$2/ 2>/dev/null || true
  cp $O/truth/$1/eq_budget_*.json $R/truth/$2/ 2>/dev/null || true
done
cp $O/cost/*.json $R/cost/ 2>/dev/null || true
cp $T/margins/analysis.json $R/truth/mmlupro_pilot1000_fp32/margin_analysis.json 2>/dev/null || true
mkdir -p $R/truth/mmlupro_pilot1000_fp32
for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  cp $T/screen/$m/summary.json $R/truth/mmlupro_pilot1000_fp32/summary_$m.json
done
cp $T/sim_*.json $R/truth/mmlupro_pilot1000_fp32/ 2>/dev/null || true
mkdir -p $R/cost
cp $O/cost/*.json $R/cost/ 2>/dev/null || true
mkdir -p $R/data/mmlu_pro
cp $K/data/ICML/datasets/mmlu_pro/meta.json $K/data/ICML/datasets/mmlu_pro/*_ids.json $R/data/mmlu_pro/
du -sh $R
# additional parents: determinism reports and truth-matrix summaries
for d in $O/detcheck/phi-3.5-mini*; do
  [ -d "$d" ] || continue
  b=$(basename $d); mkdir -p $R/detcheck/$b
  cp $d/report.json $d/run_*.json $R/detcheck/$b/ 2>/dev/null || true
done
for d in $O/truth/gsm8k_full_fp32_phi-3.5-mini*; do
  [ -d "$d/screen" ] || continue
  b=$(basename $d); mkdir -p $R/truth/$b
  for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
    cp $d/screen/$m/summary.json $R/truth/$b/summary_$m.json 2>/dev/null || true
  done
  cp $d/margins/analysis.json $R/truth/$b/margin_analysis.json 2>/dev/null || true
done
for m in phi-3.5-mini-instruct; do
  for q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
    cp $K/data/ICML/models/dequant_fp32/$m-$q/MANIFEST.json $R/manifests/fp32_${m}_$q.json 2>/dev/null || true
  done
done
cp $K/data/ICML/models/hf/Phi-3.5-mini-instruct/PINNED_REVISION.json $R/pins/phi-3.5-mini-instruct.json 2>/dev/null || true
# phi extras
for d in $O/truth/gsm8k_full_fp32_phi-3.5-mini*; do
  [ -d "$d/screen" ] || continue
  b=$(basename $d)
  cp $d/divergence_classes.json $d/answer_margin.json $d/sim_*.json $d/eq_budget_*.json $R/truth/$b/ 2>/dev/null || true
  cp $d/margins/predict_D_*.json $R/truth/$b/ 2>/dev/null || true
done
# MMLU-Pro 3000 (Qwen)
T=$O/truth/mmlupro_3000_fp32_qwen2.5-3b_2276120
if [ -d $T/screen ]; then
  mkdir -p $R/truth/mmlupro_3000_fp32
  for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do cp $T/screen/$m/summary.json $R/truth/mmlupro_3000_fp32/summary_$m.json 2>/dev/null || true; done
  cp $T/sim_*.json $T/net_*.json $R/truth/mmlupro_3000_fp32/ 2>/dev/null || true
fi
# other parents generic (smollm2, phi on other datasets)
for d in $O/truth/*smollm2* $O/truth/*phi-3.5-mini*; do
  [ -d "$d/screen" ] || continue
  b=$(basename $d); mkdir -p $R/truth/$b
  for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do cp $d/screen/$m/summary.json $R/truth/$b/summary_$m.json 2>/dev/null || true; done
  cp $d/divergence_classes.json $d/answer_margin.json $d/sim_*.json $d/eq_budget_*.json $d/*analysis.txt $R/truth/$b/ 2>/dev/null || true
  cp $d/margins/analysis.json $R/truth/$b/margin_analysis.json 2>/dev/null || true
  cp $d/margins/predict_D_*.json $R/truth/$b/ 2>/dev/null || true
done
for d in $O/detcheck/smollm2*; do
  [ -d "$d" ] || continue
  b=$(basename $d); mkdir -p $R/detcheck/$b; cp $d/report.json $d/run_*.json $R/detcheck/$b/ 2>/dev/null || true
done
for q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
  cp $K/data/ICML/models/dequant_fp32/smollm2-1.7b-instruct-$q/MANIFEST.json $R/manifests/fp32_smollm2-1.7b-instruct_$q.json 2>/dev/null || true
done
cp $K/data/ICML/models/hf/SmolLM2-1.7B-Instruct/PINNED_REVISION.json $R/pins/smollm2-1.7b-instruct.json 2>/dev/null || true
# qwen2.5-7b extras
for d in $O/truth/*qwen2.5-7b*; do
  [ -d "$d/screen" ] || continue
  b=$(basename $d); mkdir -p $R/truth/$b
  for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do cp $d/screen/$m/summary.json $R/truth/$b/summary_$m.json 2>/dev/null || true; done
  cp $d/divergence_classes.json $d/answer_margin.json $d/sim_*.json $d/eq_budget_*.json $d/*analysis.txt $R/truth/$b/ 2>/dev/null || true
  cp $d/margins/analysis.json $R/truth/$b/margin_analysis.json 2>/dev/null || true
  cp $d/margins/predict_D_*.json $R/truth/$b/ 2>/dev/null || true
done
for d in $O/detcheck/qwen2.5-7b*; do
  [ -d "$d" ] || continue
  b=$(basename $d); mkdir -p $R/detcheck/$b; cp $d/report.json $d/run_*.json $R/detcheck/$b/ 2>/dev/null || true
done
for q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
  cp $K/data/ICML/models/dequant_fp32/qwen2.5-7b-instruct-$q/MANIFEST.json $R/manifests/fp32_qwen2.5-7b-instruct_$q.json 2>/dev/null || true
done
cp $K/data/ICML/models/hf/Qwen2.5-7B-Instruct/PINNED_REVISION.json $R/pins/qwen2.5-7b-instruct.json 2>/dev/null || true
# qwen2.5-14b and llama-3.1-8b (eager)
for d in $O/truth/*qwen14b* $O/truth/*llama8b*; do
  [ -d "$d/screen" ] || continue
  b=$(basename $d); mkdir -p $R/truth/$b
  for m in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do cp $d/screen/$m/summary.json $R/truth/$b/summary_$m.json 2>/dev/null || true; done
  cp $d/divergence_classes.json $d/answer_margin.json $d/sim_*.json $d/eq_budget_*.json $d/*analysis.txt $R/truth/$b/ 2>/dev/null || true
  cp $d/margins/analysis.json $R/truth/$b/margin_analysis.json 2>/dev/null || true
  cp $d/margins/predict_D_*.json $R/truth/$b/ 2>/dev/null || true
done
for d in $O/detcheck/qwen14b* $O/detcheck/llama8b*; do
  [ -d "$d" ] || continue
  b=$(basename $d); mkdir -p $R/detcheck/$b; cp $d/report.json $d/run_*.json $R/detcheck/$b/ 2>/dev/null || true
done
for m in qwen2.5-14b-instruct llama-3.1-8b-instruct; do
  for q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
    cp $K/data/ICML/models/dequant_fp32/$m-$q/MANIFEST.json $R/manifests/fp32_${m}_$q.json 2>/dev/null || true
  done
done
cp $K/data/ICML/models/hf/Qwen2.5-14B-Instruct/PINNED_REVISION.json $R/pins/qwen2.5-14b-instruct.json 2>/dev/null || true
cp $K/data/ICML/models/hf/Llama-3.1-8B-Instruct/PINNED_REVISION.json $R/pins/llama-3.1-8b-instruct.json 2>/dev/null || true
cp $O/law_robustness_*.json $R/ 2>/dev/null || true
# perplexity check (WikiText-2): HF results and llama.cpp final estimates
for d in $O/ppl/*; do
  [ -d "$d" ] || continue
  b=$(basename $d); mkdir -p $R/ppl/$b
  cp $d/hf_*.json $R/ppl/$b/ 2>/dev/null || true
  for f in $d/llamacpp_*.log; do grep -H "Final estimate" $f; done > $R/ppl/$b/llamacpp_final.txt 2>/dev/null || true
done
cp $K/data/ICML/datasets/wikitext2/meta.json $R/data/wikitext2_meta.json 2>/dev/null || true
# Track B (llama.cpp) pilot: comparison summaries and continuation checks (outputs are small)
for d in $O/trackb/*; do
  [ -d "$d" ] || continue
  b=$(basename $d); mkdir -p $R/trackb/$b
  cp $d/trackb_compare.json $d/cont_*.jsonl $d/llamacpp_pin.txt $R/trackb/$b/ 2>/dev/null || true
done
