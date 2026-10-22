#!/bin/bash
# Build ngram_rows_parity (sm_90a) against the trainer build's libsurogate-common.so:
#   bash build.sh REPO_ROOT BUILD_DIR OUT_BIN        (toolchain: NVCC, CXX env or the ngram-poc defaults)
set -euo pipefail
ROOT=$(readlink -f ${1:?repo}); B=${2:?trainer build dir}; OUT=${3:?output binary}
T=/mnt/data/tools/ngram-poc-build
NVCC=${NVCC:-$T/cuda-13.0.2/bin/nvcc}; CXX=${CXX:-$T/env/bin/x86_64-conda-linux-gnu-g++}
D=$(dirname $(readlink -f $0))
python3 $D/extract_engine_rows.py $ROOT $(dirname $OUT)/engine_rows_extract.cuh
$NVCC -O2 -std=c++20 -ccbin $CXX --generate-code=arch=compute_90a,code=sm_90a -I$(dirname $OUT) -I$ROOT/csrc/src \
  $D/ngram_rows_parity.cu -o $OUT -L$B -lsurogate-common -Xlinker --allow-shlib-undefined -Xlinker -rpath -Xlinker '$ORIGIN'
echo built $OUT
