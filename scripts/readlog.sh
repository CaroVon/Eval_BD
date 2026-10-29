#!/bin/bash
J=${1:?jobid}
L=$(ls /net/spaces/jevans/klab/caroyf/logs/ICML/*_${J}.out)
sacct -j $J -X --format=JobID,State,Elapsed,NodeList,ExitCode
grep -E 'stage_env|^\{"phase|Error|Traceback|ALL_FOUR' "$L" | cut -c1-250 | head -40
echo "=== TAIL"
tail -n 45 "$L" | cut -c1-250