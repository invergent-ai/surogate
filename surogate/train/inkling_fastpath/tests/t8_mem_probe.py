"""Lane T8, U0: where a worker's host memory goes (plan item U0). One card, about 10 minutes, no adapter saved, nothing trained.
Reads /proc/self/status and torch's pinned-host allocator stats after each phase:
 1. after import; 2. after the engine load (does the engine keep a host copy of its weights?);
 3. the pinned allocator alone: tensors of 61 / 70 / 140 / 267 MB, freed, then its cache emptied (does it round to powers of two and keep freed blocks?);
 4. masters and banks; 5. forward_host + the stock full backward on real rows in ascending length (does the visit buffer leave stale blocks?);
 6. the host cache emptied at the end.
PASS LINE (written before the run): at least 90 % of the peak RSS is attributed to named items (engine / python baseline, pinned allocator reserved
bytes, masters); the report states the engine's host bytes and the pinned rounding factor.
Usage: t8_mem_probe.py --bin B --artifact A --rollouts rollouts.bin --T 16150,28542 --out report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
GIB = 2 ** 30


def status():
    out = {}
    for line in open("/proc/self/status"):
        k = line.split(":")[0]
        if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmPin", "VmLck", "VmHWM"):
            out[k] = int(line.split()[1]) * 1024 / GIB
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            out["MemAvailable"] = int(line.split()[1]) * 1024 / GIB
    return out


def host_stats():
    try:
        s = torch.cuda.host_memory_stats()
    except Exception as e:  # not in this torch build: the report says so
        return {"error": str(e)[:120]}
    return {k: (v / GIB if "bytes" in k else v) for k, v in s.items() if any(x in k for x in ("reserved", "allocated", "active", "bytes"))}


REC = []


def mark(name, **extra):
    r = {"phase": name, "t": time.time(), "status": status(), "pinned": host_stats(),
         "card_alloc_gib": torch.cuda.memory_allocated() / GIB if torch.cuda.is_initialized() else None,
         "card_reserved_gib": torch.cuda.memory_reserved() / GIB if torch.cuda.is_initialized() else None}
    r.update(extra)
    REC.append(r)
    s = r["status"]
    print(f"[{name}] RSS {s['VmRSS']:.2f} anon {s['RssAnon']:.2f} shmem {s['RssShmem']:.2f} file {s['RssFile']:.2f} pin {s['VmPin']:.2f} | "
          f"pinned reserved {r['pinned'].get('reserved_bytes.current', r['pinned'].get('reserved_bytes.all.current', '?'))}", flush=True)
    return r


def empty_host():
    try:
        torch._C._host_emptyCache()
        return True
    except Exception as e:
        print("host cache call:", e, flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--T", default="16150,28542")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from surogate.inkling_rl.records import load_rows
    import ift_backward as B
    import ift_engine as E
    import ift_long as LG

    mark("import")
    eng = E.InklingEngine(a.bin, a.artifact, 0)
    torch.cuda.synchronize()
    mark("engine_loaded")

    # 3. the pinned allocator alone
    sizes_mb = [61, 70, 140, 267]
    alloc = []
    for mb in sizes_mb:
        before = status()["RssShmem"]
        t = torch.empty(mb * 10 ** 6, dtype=torch.uint8, pin_memory=True)
        t.fill_(1)
        after = status()["RssShmem"]
        alloc.append({"logical_mb": mb, "rss_shmem_delta_mib": (after - before) * 1024})
        del t
    mark("pinned_freed", pinned_test=alloc)
    empty_host()
    mark("pinned_cache_emptied")

    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=0, b_std=0.0)
    banks = B.EngineBanks(lora, eng.dev)
    torch.cuda.empty_cache()  # the masters stay on the card: the stock LongModel path needs them there
    mark("masters_banks")
    tf = LG.TrainForwardLong(eng, W)
    _, rows = load_rows(a.rollouts)
    want = [int(x) for x in a.T.split(",")]
    for t in want:
        r = next(x for x in rows if x.T == t)
        ids = torch.as_tensor(r.ids.copy(), dtype=torch.int32, device=eng.dev)
        t0 = time.time()
        res = LG.forward_host(eng, ids, [L["adapters"] for L in banks.layers])
        mark(f"forward_T{t}", seconds=time.time() - t0)
        masks = torch.as_tensor(r.loss_mask[1:].copy(), device=eng.dev)
        lt = [lo[k] for lo in lora for k in B.LORA_KEYS]
        t0 = time.time()
        lp = LG.LongModel.apply(tf, ids, res, lora[0]["scale"], *lt)
        (masks.float() * lp).sum().backward()
        torch.cuda.synchronize()
        mark(f"backward_T{t}", seconds=time.time() - t0, card_peak_gib=torch.cuda.max_memory_allocated() / GIB)
        for p in lt:
            p.grad = None
        del lp, res, ids, masks
        torch.cuda.empty_cache()
        mark(f"after_row_T{t}")
    LG._PINNED.clear()
    mark("visit_buffers_dropped")
    empty_host()
    mark("host_cache_emptied")
    rep = {"phases": REC, "peak_rss_gib": max(x["status"]["VmHWM"] for x in REC)}
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    eng.close()
    print("U0 DONE", json.dumps({"peak_rss_gib": rep["peak_rss_gib"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
