#!/bin/bash
# Split the Llama continuation checks so bf16 runs as soon as its generation finishes.
scancel 2281964
export NAME=llama-3.1-8b-instruct HFDIR=Llama-3.1-8B-Instruct TAG=llama8b_gsm8k_full IDSFILE=determinism_subset_ids.json
VERSIONS=bf16 sbatch --parsable --dependency=afterok:2281961 --job-name=tb_l8_cont_bf16 --export=ALL ~/myproject/ICML/scripts/trackb_cont.sbatch
VERSIONS=Q8_0 sbatch --parsable --dependency=afterok:2281962 --job-name=tb_l8_cont_q8 --export=ALL ~/myproject/ICML/scripts/trackb_cont.sbatch
