#!/bin/bash
date
sacct -nX -j 2276789,2276900 -o JobID,JobName%20,State,Elapsed
grep -h -E '^\{"phase|ALL_FOUR|DETERMINISM' /net/spaces/jevans/klab/caroyf/logs/ICML/suite_qwen7b_eager_2276789.out | cut -c1-120
ls /net/spaces/jevans/klab/caroyf/outputs/ICML/truth/gsm8k_full_fp32_qwen2.5-7b_eager_2276789/screen 2>/dev/null
grep -h '^COST' /net/spaces/jevans/klab/caroyf/logs/ICML/cost_bs32_2276900.out 2>/dev/null | cut -c1-600