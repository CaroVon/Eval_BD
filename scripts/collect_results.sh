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
