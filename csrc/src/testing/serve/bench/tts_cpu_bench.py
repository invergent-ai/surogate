"""Compare CPU HTTP throughput and frozen audio with identical thread/affinity settings."""

import argparse
import hashlib
import io
import json
import os
import socket
import subprocess
import time
import wave
from pathlib import Path

import httpx


def main():
    root = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Verified native model directory")
    parser.add_argument("--binary", type=Path, default=root / "csrc/build-serve/surogate-tts")
    parser.add_argument("--cases", type=Path, default=root / "tests/serve/fixtures/tts-cpu-controls.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--codec-threads", type=int, default=4)
    parser.add_argument("--cpus", help="Optional taskset CPU list; choose physical cores on one socket")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.read_text())
    report = {"threads": args.threads, "codec_threads": args.codec_threads,
              "cpu_affinity": args.cpus, "rounds": args.rounds, "requests": []}
    for repeat in range(args.rounds):
        modes = ["reference", "optimized"] if repeat % 2 == 0 else ["optimized", "reference"]
        for mode in modes:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            command = [str(args.binary), args.model, "--port", str(port), "--threads", str(args.threads),
                       "--codec-threads", str(args.codec_threads), "--cpu-kernels", mode]
            if args.cpus:
                command = ["taskset", "-c", args.cpus, *command]
            log_path = args.output.with_suffix(f".{mode}.{repeat}.log")
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(log_path.read_text()[-4000:])
                        try:
                            if client.get("/health", timeout=0.2).status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(0.1)
                    else:
                        raise TimeoutError("TTS startup timed out")
                    for case in cases:
                        begin = time.monotonic()
                        response = client.post("/v1/audio/speech", json={
                            "input": case["input"], "voice": case["voice"], "seed": case["seed"],
                        })
                        elapsed = time.monotonic() - begin
                        response.raise_for_status()
                        digest = hashlib.sha256(response.content).hexdigest()
                        if digest != case["sha256"]:
                            raise RuntimeError(f"Audio mismatch: {mode}, {case['voice']}, {case['id']}")
                        with wave.open(io.BytesIO(response.content)) as wav:
                            duration = wav.getnframes() / wav.getframerate()
                        row = {"mode": mode, "repeat": repeat, "voice": case["voice"], "id": case["id"],
                               "audio_s": duration, "elapsed_s": elapsed, "rtf": elapsed / duration,
                               "sha256": digest, "byte_exact": True}
                        report["requests"].append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(row), flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise
    report["summary"] = {}
    for mode in ("reference", "optimized"):
        rows = [row for row in report["requests"] if row["mode"] == mode]
        duration = sum(row["audio_s"] for row in rows)
        elapsed = sum(row["elapsed_s"] for row in rows)
        report["summary"][mode] = {"requests": len(rows), "audio_s": duration,
                                   "elapsed_s": elapsed, "audio_seconds_per_second": duration / elapsed,
                                   "rtf": elapsed / duration}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
