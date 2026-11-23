#!/bin/bash
# Build the Inkling-2-Mini roofline kit on a remote build host (standalone nvcc, outside the engine's CMake).
# Run from a control machine (rsync + ssh only; nothing compiles locally). On the build host: its own directory,
# nice -n 19 ionice -c3, one nvcc at a time (flock), only with MemAvailable >= 60 GB.
# Environment: BUILD_HOST (ssh host that builds the kit, required); REMOTE_DIR (directory on BUILD_HOST, default
# $HOME/inkling-roofline there); OUT_DIR (local product directory, default bin/ next to this script).
# Product: $OUT_DIR/inkling_roofline (+ SHA256SUMS, SOURCE_HEAD).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD_HOST=${BUILD_HOST:?set BUILD_HOST to the ssh host that builds the kit}
R=${REMOTE_DIR:-$(ssh "$BUILD_HOST" 'echo "$HOME/inkling-roofline"')}
OUT=${OUT_DIR:-$HERE/bin}
mkdir -p "$OUT"
ssh "$BUILD_HOST" "mkdir -p $R"
rsync -a --delete --include '*.cu' --include '*.cuh' --exclude '*' "$HERE/" "$BUILD_HOST:$R/src/"
ssh "$BUILD_HOST" bash -s <<EOF
set -euo pipefail
cd $R
while [ "\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo)" -lt 60 ]; do
  echo "waiting: build host MemAvailable < 60 GB"; sleep 30
done
exec 9>$R/nvcc.lock
flock 9
nice -n 19 ionice -c3 /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -lineinfo \
  -Xcompiler -Wall src/main.cu src/dram.cu src/mma.cu src/host.cu src/gap.cu -lcublasLt -o inkling_roofline.new
mv inkling_roofline.new inkling_roofline
sha256sum inkling_roofline
EOF
rsync -a "$BUILD_HOST:$R/inkling_roofline" "$OUT/inkling_roofline"
(cd "$OUT" && sha256sum inkling_roofline > SHA256SUMS.roofline)
git -C "$HERE" rev-parse HEAD > "$OUT/SOURCE_HEAD.roofline"
git -C "$HERE" status --porcelain -- . > "$OUT/SOURCE_DIRTY.roofline"
echo "built: $OUT/inkling_roofline"
