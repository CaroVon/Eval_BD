#!/bin/bash
date
sacct -nX -j 2276120,2276333 -o JobID,JobName%18,State,Elapsed
ls /net/spaces/jevans/klab/caroyf/outputs/ICML/truth/mmlupro_3000_fp32_qwen2.5-3b_2276120/screen 2>/dev/null
grep -h '^{"phase' /net/spaces/jevans/klab/caroyf/logs/ICML/suite_phi35_eager_2276333.out 2>/dev/null
grep -h -E 'DETERMINISM|ALL_FOUR' /net/spaces/jevans/klab/caroyf/logs/ICML/suite_phi35_eager_2276333.out 2>/dev/null