#!/bin/bash
# Harbor compose garbage collector (belt-and-braces behind HarborHarness.close()).
#
# Every Harbor rollout runs a docker compose project; a leaked project pins one
# docker network, and the ~31-network address pool then starves ALL repo-lane
# rollouts ("no Harbor verifier rewards found" / instant-zero episodes — the
# 2026-08-03 incident). close() tears projects down in-process, but any crash,
# version drift in compose project naming, or kill between run and close leaks.
#
# Removal criterion (exact, no name patterns): the container carries the
# compose working_dir label pointing under .ultra_harbor_runs AND is older
# than MAX_AGE_S. No live Harbor job exceeds the 900s agent cap + verifier +
# teardown, so 1500s cannot match an in-flight rollout.
MAX_AGE_S=1500
INTERVAL_S=300
RUNS_MARKER="/.ultra_harbor_runs/"
while true; do
  now=$(date +%s)
  docker ps -q 2>/dev/null | while read -r id; do
    wd=$(docker inspect "$id" --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' 2>/dev/null)
    case "$wd" in
      *"$RUNS_MARKER"*)
        started=$(docker inspect "$id" --format '{{.State.StartedAt}}' 2>/dev/null)
        started_s=$(date -d "$started" +%s 2>/dev/null) || continue
        if [ $((now - started_s)) -gt "$MAX_AGE_S" ]; then
          echo "$(date -Is) sweeping $(docker inspect "$id" --format '{{.Name}}' 2>/dev/null) age=$((now - started_s))s"
          docker rm -f "$id" >/dev/null 2>&1
        fi
        ;;
    esac
  done
  docker network prune -f >/dev/null 2>&1
  # Per-trial compose BUILD images (inferredbugs-XXXX__<trial>__env-main) are
  # garbage once their trial ends; without GC they accumulate one 500MB tag per
  # rollout-step (~700/day measured 2026-08-04). In-use ones survive rmi.
  docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep -E '^inferredbugs-[0-9]+__' | xargs -r docker rmi >/dev/null 2>&1
  docker image prune -f >/dev/null 2>&1
  # Base task images (ECR swe-bench) are multi-GB and unbounded across the
  # registry; below 80GB free, evict not-in-use bases oldest-first until
  # 100GB free (they re-pull on demand at ~1-2 min).
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${free_gb:-999}" -lt 80 ]; then
    in_use=$(docker ps --format '{{.Image}}' | sort -u)
    docker images --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' \
      | grep -E 'public\.ecr\.aws/.*swe-bench' | sort | cut -f2 | while read -r img; do
      echo "$in_use" | grep -qx "$img" && continue
      echo "$(date -Is) evicting base image $img (free ${free_gb}GB)"
      docker rmi "$img" >/dev/null 2>&1
      free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
      [ "$free_gb" -ge 100 ] && break
    done
  fi
  sleep "$INTERVAL_S"
done
