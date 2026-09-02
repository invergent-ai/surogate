# KV-pool physical occupancy under a scripted multi-model traffic pattern.
#
#   python kvocc.py <port> <out.jsonl> <phase>... [--prompt-tokens N] [--max-tokens N]
#
#   phase = <seconds>:idle
#         | <seconds>:<model>=<users>[,<model>=<users>...]
#
#   python kvocc.py 8000 kv.jsonl 60:idle 180:big=8 180:big=8,small=8 180:small=8 90:idle
#
# The server's /kv_stats reports, per model, the KV pool's physical page count, the pages
# actually holding a sequence's KV, and the pages a demand-mapped (CUDA VMM) pool could not
# have released -- a mapping granule stays resident while any one of its pages is in use.
# This samples that once a second while driving traffic, then reports time-weighted means.
#
# Two numbers come out of it. `idle` (pool - in_use) is the whole prize of an elastic KV
# pool. `unreclaimable` (resident - in_use) is what the current plane-major page layout
# would give away to granule fragmentation, and so is what interleaving the planes would buy
# on top. If idle is small, elastic KV is not worth building; if idle is large but
# unreclaimable eats most of it, the layout change is not optional.
import json
import statistics
import sys
import threading
import time
import urllib.request

GIB = 1024.0**3


def post_chat(port, model, prompt_tokens, max_tokens, tag):
    prompt = f"request {tag}: " + "word " * prompt_tokens
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


class Load:
    """One phase's worth of concurrent users against one model."""

    def __init__(self, port, model, users, deadline, prompt_tokens, max_tokens, counters):
        self.port, self.model, self.deadline = port, model, deadline
        self.prompt_tokens, self.max_tokens = prompt_tokens, max_tokens
        self.counters, self.lock = counters, threading.Lock()
        self.serial = 0
        self.threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(users)]

    def start(self):
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            thread.join()

    def _worker(self):
        while time.time() < self.deadline:
            with self.lock:
                self.serial += 1
                tag = f"{self.model}-{self.serial}-{time.time()}"
            try:
                done = post_chat(self.port, self.model, self.prompt_tokens, self.max_tokens, tag)
                tokens = done.get("usage", {}).get("completion_tokens", 0)
                with self.lock:
                    self.counters["ok"] += 1
                    self.counters["tokens"] += tokens
            except Exception as error:  # a refusal is data, not a crash
                with self.lock:
                    self.counters["err"] += 1
                    self.counters.setdefault("last_error", str(error)[:200])


class Sampler(threading.Thread):
    def __init__(self, port, out_path, interval=1.0):
        super().__init__(daemon=True)
        self.port, self.out_path, self.interval = port, out_path, interval
        self.phase = "start"
        self.samples = []
        self.stop_flag = threading.Event()

    def run(self):
        with open(self.out_path, "w") as out:
            while not self.stop_flag.is_set():
                began = time.time()
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/kv_stats", timeout=30
                    ) as response:
                        record = json.load(response)
                except Exception as error:
                    record = {"error": str(error)[:200]}
                record["wall"] = began
                record["phase"] = self.phase
                out.write(json.dumps(record) + "\n")
                out.flush()
                self.samples.append(record)
                self.stop_flag.wait(max(0.0, self.interval - (time.time() - began)))


def parse_phase(text):
    seconds, _, spec = text.partition(":")
    if not spec:
        raise SystemExit(f"phase {text!r} must be <seconds>:idle or <seconds>:<model>=<users>,...")
    users = {}
    if spec != "idle":
        for part in spec.split(","):
            model, _, count = part.partition("=")
            if not count:
                raise SystemExit(f"phase {text!r}: expected <model>=<users>")
            users[model] = int(count)
    return float(seconds), spec, users


def summarize(samples):
    """Time-weighted per-model means over the samples of one phase."""
    per_model = {}
    for record in samples:
        for model in record.get("models", []):
            per_model.setdefault(model["model"], []).append(model)
    rows = []
    for name, entries in sorted(per_model.items()):

        def mean(key):
            return statistics.fmean(entry[key] for entry in entries)

        def peak(key):
            return max(entry[key] for entry in entries)

        pool = peak("pool_bytes")
        rows.append(
            {
                "model": name,
                "pool_gib": pool / GIB,
                "in_use_gib": mean("in_use_bytes") / GIB,
                "in_use_peak_gib": peak("in_use_bytes") / GIB,
                "resident_gib": mean("resident_at_granule_bytes") / GIB,
                # What an elastic pool actually holds: in use plus its reserve. Equals the
                # pool for a pool that lives in the arena.
                "mapped_gib": mean("mapped_bytes") / GIB if "mapped_bytes" in entries[-1] else 0.0,
                "mapped_peak_gib": peak("mapped_bytes") / GIB if "mapped_bytes" in entries[-1] else 0.0,
                "idle_gib": (pool - mean("in_use_bytes")) / GIB,
                "reclaimable_gib": (pool - mean("resident_at_granule_bytes")) / GIB,
                "unreclaimable_gib": (mean("resident_at_granule_bytes") - mean("in_use_bytes"))
                / GIB,
                "granule_pages": entries[-1]["granule_pages"],
                "page_bytes": entries[-1]["page_bytes"],
                "samples": len(entries),
            }
        )
    return rows


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__ or "usage: kvocc.py <port> <out.jsonl> <phase>...")
    port, out_path = sys.argv[1], sys.argv[2]
    phases, prompt_tokens, max_tokens = [], 1024, 256
    args = sys.argv[3:]
    index = 0
    while index < len(args):
        if args[index] == "--prompt-tokens":
            prompt_tokens = int(args[index + 1])
            index += 2
        elif args[index] == "--max-tokens":
            max_tokens = int(args[index + 1])
            index += 2
        else:
            phases.append(parse_phase(args[index]))
            index += 1

    sampler = Sampler(port, out_path)
    sampler.start()
    print(f"sampling /kv_stats on port {port} -> {out_path}", flush=True)

    phase_windows = []
    for seconds, spec, users in phases:
        sampler.phase = spec
        began = time.time()
        deadline = began + seconds
        counters = {model: {"ok": 0, "err": 0, "tokens": 0} for model in users}
        loads = [
            Load(port, model, count, deadline, prompt_tokens, max_tokens, counters[model])
            for model, count in users.items()
        ]
        print(f"[{time.strftime('%H:%M:%S')}] phase {spec} for {seconds:.0f}s", flush=True)
        for load in loads:
            load.start()
        # Idle phases have no threads to join on; wait out the clock either way.
        while time.time() < deadline:
            time.sleep(0.5)
        for load in loads:
            load.join()
        ended = time.time()
        phase_windows.append((spec, began, ended, counters))
        for model, counts in counters.items():
            rate = counts["tokens"] / max(1e-9, ended - began)
            print(
                f"    {model}: {counts['ok']} ok, {counts['err']} err, {rate:.0f} decode tok/s",
                flush=True,
            )

    sampler.stop_flag.set()
    sampler.join(timeout=10)

    print("\n=== KV pool occupancy by phase (time-weighted means, GiB) ===")
    header = (
        f"{'phase':<22} {'model':<14} {'pool':>7} {'in_use':>8} {'peak':>7} "
        f"{'idle':>7} {'reclaim':>8} {'lost_to_granule':>16} {'mapped':>7} {'map_pk':>7}"
    )
    print(header)
    print("-" * len(header))
    for spec, began, ended, _ in phase_windows:
        window = [
            record
            for record in sampler.samples
            if began <= record.get("wall", 0) <= ended and "models" in record
        ]
        if not window:
            print(f"{spec:<22} (no samples)")
            continue
        for row in summarize(window):
            print(
                f"{spec:<22} {row['model']:<14} {row['pool_gib']:>7.2f} {row['in_use_gib']:>8.2f} "
                f"{row['in_use_peak_gib']:>7.2f} {row['idle_gib']:>7.2f} "
                f"{row['reclaimable_gib']:>8.2f} {row['unreclaimable_gib']:>16.2f} "
                f"{row['mapped_gib']:>7.2f} {row['mapped_peak_gib']:>7.2f}"
            )
    every = [record for record in sampler.samples if "models" in record]
    if every:
        print("\ngranule: ", end="")
        for row in summarize(every):
            print(
                f"{row['model']} = {row['granule_pages']} pages "
                f"({row['granule_pages'] * row['page_bytes'] / (1024 * 1024):.0f} MiB)  ",
                end="",
            )
        print()


if __name__ == "__main__":
    main()
