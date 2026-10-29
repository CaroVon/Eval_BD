#!/bin/bash
# Pack a conda env into an uncompressed tarball for fast staging to node-local disk.
set -e
name=${1:?env name}
K=/net/spaces/jevans/klab/caroyf
mkdir -p $K/data/ICML/envs_pack
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qc-tools
t0=$(date +%s)
conda-pack -n "$name" -o $K/data/ICML/envs_pack/${name}.tar --format tar --force --ignore-missing-files
ls -la $K/data/ICML/envs_pack/${name}.tar
echo "PACK_DONE $name $(( $(date +%s) - t0 ))s"
