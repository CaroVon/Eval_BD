#!/bin/bash
# Track B control: llama-server with batch and ubatch 1, so the prompt is processed token by token
# on the same kernel path as decoding. Then the continuation check against that run.
export NAME=qwen2.5-3b-instruct HFDIR=Qwen2.5-3B-Instruct VERSIONS=Q8_0 IDSFILE=determinism_subset_ids.json TAG=qwen3b_gsm8k_det200_ub1
export SERVER_EXTRA="-b 1 -ub 1"
j=$(sbatch --parsable --time=02:00:00 --job-name=trackb_ub1 --export=ALL ~/myproject/ICML/scripts/trackb.sbatch)
echo $j
sbatch --parsable --dependency=afterok:$j --job-name=trackb_cont_ub1 --export=ALL ~/myproject/ICML/scripts/trackb_cont.sbatch
