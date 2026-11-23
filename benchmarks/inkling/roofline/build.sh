#!/bin/bash
# ENGINE-INKLING lane h: build the roofline kit on gpubox2 (standalone nvcc, outside the engine's CMake).
# Run from node-3 (rsync + ssh only; node-3 never compiles). Rules (LANE-RULES 10/16): own dir
# /home/densemax2/effrl/build-inkling/h/, nice -n 19 ionice -c3, one nvcc at a time, only with MemAvailable >= 60 GB.
# Product: lanes/h/bin/inkling_roofline (+ SHA256SUMS, SOURCE_HEAD) on NFS.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/densemax2/effrl/build-inkling/h/roofline
OUT=/mnt/data/work/rl/engine-inkling-2026-11-21/lanes/h/bin
mkdir -p "$OUT"
ssh gpubox2 "mkdir -p $R"
rsync -a --delete --include '*.cu' --include '*.cuh' --exclude '*' "$HERE/" "gpubox2:$R/src/"
ssh gpubox2 bash -s <<EOF
set -euo pipefail
cd $R
while [ "\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo)" -lt 60 ]; do
  echo "waiting: gpubox2 MemAvailable < 60 GB"; sleep 30
done
exec 9>$R/nvcc.lock
flock 9
nice -n 19 ionice -c3 /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -lineinfo \
  -Xcompiler -Wall src/main.cu src/dram.cu src/mma.cu src/host.cu src/gap.cu -lcublasLt -o inkling_roofline.new
mv inkling_roofline.new inkling_roofline
sha256sum inkling_roofline
EOF
rsync -a "gpubox2:$R/inkling_roofline" "$OUT/inkling_roofline"
(cd "$OUT" && sha256sum inkling_roofline > SHA256SUMS.roofline)
git -C "$HERE" rev-parse HEAD > "$OUT/SOURCE_HEAD.roofline"
git -C "$HERE" status --porcelain -- . > "$OUT/SOURCE_DIRTY.roofline"
echo "built: $OUT/inkling_roofline"
