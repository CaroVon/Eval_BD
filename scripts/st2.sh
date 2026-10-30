#!/bin/bash
date
sacct -nX -j 2276120,2276156,2276157,2276195 -o JobID,JobName%14,State,Elapsed,NodeList
L=/net/spaces/jevans/klab/caroyf/logs/ICML
grep -E 'DEQUANT_DONE|PIPELINE_DONE|Error|Traceback|quant size|tensor mismatch' $L/quant_phi35_2276156.out 2>/dev/null | cut -c1-220 | tail -12
ls /net/spaces/jevans/klab/caroyf/outputs/ICML/truth/mmlupro_3000_fp32_qwen2.5-3b_2276120/screen 2>/dev/null