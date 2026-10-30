#!/bin/bash
date
grep -E 'DEQUANT_DONE|quant size' /net/spaces/jevans/klab/caroyf/logs/ICML/quant_qwen7b_2276788.out | cut -c1-120
sacct -nX -j 2276787,2276789 -o JobID,JobName%20,State,Elapsed
grep -h '^COST' /net/spaces/jevans/klab/caroyf/logs/ICML/cost_qwen7b_2276787.out 2>/dev/null | cut -c1-700