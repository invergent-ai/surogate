#!/bin/bash
# ENGINE-INKLING lane h: run the roofline kit on ONE gpubox2 card, only after agent-EN relayed "go" for that card.
# Run from node-3. Usage:
#   run.sh start <card>[,<card2>] --go [kit options, e.g. --only gap,dram]  start detached on gpubox2, poll, fetch
#                                     (two cards: device 0 = the first; the host part then also times P2P 0 <-> 1)
#   run.sh stop                                                            kill our running kit (PID + cmdline check)
#   run.sh fetch <stamp>                                                   copy a finished run back to NFS
# Guards: refuses a card with any compute process or > 1 GiB used; CUDA_DEVICE_ORDER=PCI_BUS_ID so <card> is the
# nvidia-smi index. Results: rl/engine-inkling-2026-11-21/perf/results/roofline-<stamp>/ (roofline.jsonl = stdout,
# roofline.stderr, smi.csv sampled every 0.5 s, gpu-info.txt).
set -euo pipefail
R=/home/densemax2/effrl/build-inkling/h/roofline
NFS=/mnt/data/work/rl/engine-inkling-2026-11-21/perf/results
cmd=${1:?start|stop|fetch}; shift

fetch() {
  local stamp=$1
  mkdir -p "$NFS/roofline-$stamp"
  rsync -a "gpubox2:$R/runs/$stamp/" "$NFS/roofline-$stamp/"
  echo "fetched: $NFS/roofline-$stamp"
}

case "$cmd" in
start)
  card=${1:?card}; shift
  [ "${1:-}" = "--go" ] || { echo "refusing: pass --go only after agent-EN relayed go for card $card"; exit 2; }
  shift
  stamp=$(TZ=Europe/Bucharest date +%m%d-%H%M%S)
  ssh gpubox2 bash -s -- "$card" "$stamp" "$@" <<'EOF'
set -euo pipefail
card=$1; stamp=$2; shift 2
R=/home/densemax2/effrl/build-inkling/h/roofline
D=$R/runs/$stamp; mkdir -p $D
for c in ${card//,/ }; do   # <card> may be a list, e.g. 6,7 (P2P): every card must be idle
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $c)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)
  procs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid" || true)
  if [ "$procs" -ne 0 ] || [ "$used" -gt 1024 ]; then
    echo "REFUSED: card $c busy (compute processes $procs, used $used MiB)"; exit 3
  fi
done
if [ -f $R/kit.pid ] && kill -0 "$(cat $R/kit.pid)" 2>/dev/null; then echo "REFUSED: a kit run is still alive"; exit 3; fi
nvidia-smi -q -i $card -d CLOCK,POWER,PERFORMANCE,TEMPERATURE > $D/gpu-info.txt 2>&1 || true
nvidia-smi topo -m >> $D/gpu-info.txt 2>&1 || true
echo "cards=$card args=$*" >> $D/gpu-info.txt
setsid nohup nvidia-smi -i $card --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 500 > $D/smi.csv 2>/dev/null < /dev/null &
echo $! > $D/smi.pid
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$card setsid nohup $R/inkling_roofline "$@" \
  > $D/roofline.jsonl 2> $D/roofline.stderr < /dev/null &
echo $! > $R/kit.pid
echo "started stamp=$stamp pid=$(cat $R/kit.pid) card=$card"
EOF
  # poll until the kit exits (it prints {"bench":"done"} last), then stop the sampler and fetch
  while ssh gpubox2 "kill -0 \$(cat $R/kit.pid) 2>/dev/null"; do
    ssh gpubox2 "tail -n 1 $R/runs/$stamp/roofline.stderr" 2>/dev/null || true
    sleep 15
  done
  ssh gpubox2 "p=\$(cat $R/runs/$stamp/smi.pid); ps -o args= -p \$p | grep -q '^nvidia-smi -i' && kill \$p || true"
  fetch "$stamp"
  grep -c '"ok":false' "$NFS/roofline-$stamp/roofline.jsonl" | sed 's/^/ok:false lines: /' || true
  ;;
stop)
  ssh gpubox2 bash -s <<'EOF'
R=/home/densemax2/effrl/build-inkling/h/roofline
p=$(cat $R/kit.pid 2>/dev/null) || { echo "no pid file"; exit 0; }
if ps -o args= -p "$p" | grep -q "^$R/inkling_roofline"; then kill "$p"; echo "killed kit pid $p"; else echo "pid $p is not our kit (or gone)"; fi
for f in $R/runs/*/smi.pid; do q=$(cat "$f"); ps -o args= -p "$q" 2>/dev/null | grep -q '^nvidia-smi -i' && kill "$q" && echo "killed sampler $q"; done
true
EOF
  ;;
fetch) fetch "${1:?stamp}" ;;
*) echo "usage: run.sh start <card> --go [opts] | stop | fetch <stamp>"; exit 2 ;;
esac
