#!/bin/bash
pkill -u caroyf -x grep
grep -E 'DEQUANT_DONE|quant size' /net/spaces/jevans/klab/caroyf/logs/ICML/quant_smol_2276608.out | cut -c1-120
sacct -nX -j 2276609 -o JobID,State