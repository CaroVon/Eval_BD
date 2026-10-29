# Source this inside a Slurm job that requested --gres=local:<N>G.
# Unpacks a conda-pack tarball onto node-local NVMe and activates it.
# Usage: source stage_env.sh qc-vllm
_name=${1:?env name}
_pack=/net/spaces/jevans/klab/caroyf/data/ICML/envs_pack/${_name}.tar
_local=/local/scratch/${USER}_${SLURM_JOB_ID}
if [ ! -d "$_local" ]; then
    echo "stage_env: $_local missing; request --gres=local:<N>G" >&2
    return 1
fi
_dest=$_local/env_${_name}
if [ ! -x "$_dest/bin/python" ]; then
    _t0=$(date +%s)
    mkdir -p "$_dest"
    tar -xf "$_pack" -C "$_dest"
    # The packed setuptools is a broken mix of conda 84.2.1 and pip 80.10.2 files;
    # overlay the repaired copy from the source env until the env is repacked.
    _src=/net/spaces/jevans/klab/caroyf/conda/envs/${_name}/lib/python3.12/site-packages
    _sp=$_dest/lib/python3.12/site-packages
    if [ "$_name" = "qc-vllm" ]; then
        rm -rf "$_sp"/setuptools "$_sp"/_distutils_hack "$_sp"/setuptools-*.dist-info "$_sp"/distutils-precedence.pth
        cp -r "$_src"/setuptools "$_src"/_distutils_hack "$_src"/setuptools-*.dist-info "$_sp"/
        cp "$_src"/distutils-precedence.pth "$_sp"/ 2>/dev/null || true
    fi
    source "$_dest/bin/activate"
    conda-unpack
    echo "stage_env: unpacked $_name in $(( $(date +%s) - _t0 ))s to $_dest"
else
    source "$_dest/bin/activate"
fi
export TRITON_CACHE_DIR=$_local/triton_cache
