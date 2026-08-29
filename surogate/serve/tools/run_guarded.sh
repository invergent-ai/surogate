#!/bin/bash
# Launch a serving binary inside a memory-limited cgroup so a runaway run kills itself instead
# of the machine.
#
# Why: a Flash-Next process pins its expert bank in host memory (~55 GB shmem RSS, ~300 GB
# virtual, on top of the 152 GB artifact's page cache). Unreclaimable pages plus this box's
# k3s/docker/desktop workload reach global OOM, and on 2026-08-29 that took the machine down
# three times — the OOM killer took out the desktop and journald, and the last one needed a
# hard reboot. A cgroup limit turns that into a killed process (verified: the kernel reports
# `constraint=CONSTRAINT_MEMCG` and the box keeps running).
#
#   run_guarded.sh [--mem 300G] [--numa interleave|none] -- <binary> [args...]
#
# Env: GUARD_MIN_AVAIL_GB (default 200) is the free-memory floor the launch requires.
set -u
mem=300G; numa=interleave
while [ $# -gt 0 ]; do
  case "$1" in
    --mem) mem=$2; shift 2 ;;
    --numa) numa=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "run_guarded.sh: unexpected argument '$1'" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "run_guarded.sh: no command given" >&2; exit 2; }

# One offloaded model at a time: a second copy of the bank is what exhausts the host.
live=$(pgrep -x surogate-engine; pgrep -f 'surogate-engine-cl[i]') || true
if [ -n "${live//[$'\n' ]/}" ]; then
  echo "run_guarded.sh: a serving process is already running (${live//$'\n'/ }); refusing to start a second" >&2
  exit 1
fi
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
floor=${GUARD_MIN_AVAIL_GB:-200}
if [ "$avail" -lt "$floor" ]; then
  echo "run_guarded.sh: only ${avail} GB available, need ${floor} GB (drop caches or wait)" >&2
  exit 1
fi
prefix=()
[ "$numa" = interleave ] && prefix=(numactl --interleave=all)   # never --membind a single node:
                                                               # the whole bank must not be
                                                               # confined to one node's 251 GB
echo "run_guarded.sh: MemoryMax=$mem, ${avail} GB available, numa=$numa" >&2
exec systemd-run --user --scope -q --unit "surogate-guarded-$$" \
     -p MemoryMax="$mem" -p MemorySwapMax=0 -- "${prefix[@]}" "$@"
