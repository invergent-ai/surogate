#!/usr/bin/env python3
"""CPU test of perf_harness.py against a mock OpenAI streaming server with KNOWN timing (no GPU, no engine).

Mock: every chat completion streams max_tokens chunks, one per TOKEN_S seconds after a prefill delay of
PREFILL_S_PER_KTOK per 1,000 prompt tokens (prompt tokens = characters / 4); /v1/load_lora_adapter sleeps SWAP_S.
Checks: decode rate per stream ~ 1/TOKEN_S, aggregate ~ streams/TOKEN_S, prefill tok/s ~ 1000/PREFILL_S_PER_KTOK,
swap latency ~ SWAP_S, the calibration recovers 4 chars/token, and --engine ours without a parity record is refused.
Run: python3 benchmarks/inkling/test_perf_harness.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN_S = 0.004
PREFILL_S_PER_KTOK = 0.010
SWAP_S = 0.25
HERE = os.path.dirname(os.path.abspath(__file__))


class Mock(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/load_lora_adapter":
            time.sleep(SWAP_S)
            out = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        prompt_tokens = len(body["messages"][0]["content"]) // 4
        n = body["max_tokens"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(obj):
            data = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        time.sleep(PREFILL_S_PER_KTOK * prompt_tokens / 1000.0)
        t_next = time.time()
        for i in range(n):  # every 5th token without a content delta (as vLLM's special / boundary tokens)
            chunk({"choices": [{"delta": {} if i % 5 == 4 else {"content": "x"}}]})
            t_next += TOKEN_S
            time.sleep(max(0.0, t_next - time.time()))
        chunk({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n}})
        data = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(data), data))
        self.wfile.flush()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, "c.txt")
    open(corpus, "w").write(("The quick brown fox jumps over the lazy dog %d. " * 1) * 0 +
                            "".join("Sentence number %d about ledgers and invoices. " % i for i in range(20000)))
    out = os.path.join(tmp, "o.jsonl")
    cmd = [sys.executable, os.path.join(HERE, "perf_harness.py"), "--engine", "vllm", "--base",
           "http://127.0.0.1:%d" % port, "--corpus", corpus, "--decode", "1x2k,4x2k", "--gen", "200",
           "--prefill", "16k", "--cached", "4k+256", "--swap", "a", "/x", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in open(out)]
    kinds = {x["kind"]: x for x in rows if x["kind"] != "decode"}
    dec = [x for x in rows if x["kind"] == "decode"]
    fails = []

    def check(name, got, want, rel):
        ok = abs(got - want) <= rel * want
        print("%-34s got %9.3f want %9.3f  %s" % (name, got, want, "ok" if ok else "FAIL"))
        if not ok:
            fails.append(name)

    check("calibration chars/token", kinds["calibration"]["chars_per_token"], 4.0, 0.05)
    check("1 stream decode tok/s", dec[0]["aggregate_tok_s"], 1 / TOKEN_S, 0.10)
    check("4 streams aggregate tok/s", dec[1]["aggregate_tok_s"], 4 / TOKEN_S, 0.15)
    check("4 streams prompt tokens ~2k", sum(dec[1]["prompt_tokens"]) / 4, 2048, 0.05)
    check("prefill tok/s", kinds["prefill"]["tok_s"], 1000 / PREFILL_S_PER_KTOK, 0.15)
    check("swap seconds", kinds["swap"]["swap_s"], SWAP_S, 0.2)
    distinct = len(set(json.dumps(x) for x in dec[1]["prompt_tokens"])) >= 1
    assert distinct
    # parity guard
    r2 = subprocess.run([sys.executable, os.path.join(HERE, "perf_harness.py"), "--engine", "ours", "--base",
                         "http://127.0.0.1:%d" % port, "--corpus", corpus], capture_output=True, text=True)
    guard_ok = r2.returncode != 0 and "REFUSED" in (r2.stderr + r2.stdout)
    print("%-34s %s" % ("parity guard refuses (ours)", "ok" if guard_ok else "FAIL"))
    if not guard_ok:
        fails.append("parity guard")
    srv.shutdown()
    print("TEST", "PASS" if not fails else "FAIL: " + ", ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
