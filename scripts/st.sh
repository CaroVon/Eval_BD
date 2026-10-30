#!/bin/bash
date
squeue -u caroyf -o '%.10i %.14j %.8T %.12r %.20S' 
cat /net/spaces/jevans/klab/caroyf/logs/ICML/prep_mmlu_pro.log | cut -c1-1200
ls /net/spaces/jevans/klab/caroyf/data/ICML/datasets/mmlu_pro/ 2>&1