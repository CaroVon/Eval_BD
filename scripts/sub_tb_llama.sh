#!/bin/bash
# Track B for Llama-3.1-8B on GSM8K: one job per version (parallel GPUs), then continuation checks
# on the 200-question subset for bf16 and Q8_0 once their generations exist.
export NAME=llama-3.1-8b-instruct HFDIR=Llama-3.1-8B-Instruct TAG=llama8b_gsm8k_full
ids=""
for V in bf16 Q8_0 Q6_K; do
  j=$(VERSIONS=$V sbatch --parsable --time=03:00:00 --job-name=tb_l8_$V --export=ALL ~/myproject/ICML/scripts/trackb.sbatch)
  echo "$V $j"; ids="$ids $j"
done
set -- $ids
VERSIONS="bf16 Q8_0" IDSFILE=determinism_subset_ids.json sbatch --parsable --dependency=afterok:$1:$2 \
  --job-name=tb_l8_cont --export=ALL ~/myproject/ICML/scripts/trackb_cont.sbatch
