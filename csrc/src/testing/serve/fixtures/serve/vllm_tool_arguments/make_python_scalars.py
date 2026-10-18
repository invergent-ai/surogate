"""Regenerate python_scalars.jsonl (consumed by test_vllm_tool_arguments.cpp): what CPython itself returns for the
scalar conversions vLLM's tool-call argument coercion uses, on random and edge inputs.

Lines: {"fn": "repr", "bits": "<16 hex digits of the double>", "out": repr(x)}
       {"fn": "int_of_float", "bits": ..., "out": str(int(x))}
       {"fn": "int", "in": text, "out": str(int(text)) | null}
       {"fn": "float", "in": text, "bits": hex of float(text) | null}   (non-finite included)
Run with the Python vLLM 0.31 runs on (CPython 3.12; the seed is fixed): python make_python_scalars.py python_scalars.jsonl
"""
import json
import math
import random
import struct
import sys


def bits(x):
    return struct.pack(">d", x).hex()


def main(out):
    rnd = random.Random(20261018)
    rows = []
    doubles = [0.1, 0.2, 0.3, 1 / 3, 2 / 3, 1e15, 1e16, 1e17, 1e-4, 1e-5, 9.999999999999999e15, 1e16 - 2,
               123456789012345.6, 1234567890123456.7, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
               math.pi, math.e, 100.0, 1.5, -2.5, 1e22, 1e21, 1e23, 0.5e-3, 4.35, 0.1 + 0.2, 1 / 7, 2.0 ** 53,
               2.0 ** 63, 2.0 ** 64, 2.0 ** 100, 1e300, -1e-300]
    for _ in range(1000):
        kind = rnd.random()
        if kind < 0.4:
            x = struct.unpack(">d", rnd.getrandbits(64).to_bytes(8, "big"))[0]
            if not math.isfinite(x):
                continue
        elif kind < 0.7:
            x = rnd.uniform(-1, 1) * 10 ** rnd.randint(-30, 30)
        elif kind < 0.9:
            x = round(rnd.uniform(-1e6, 1e6), rnd.randint(0, 8))
        else:
            x = float(rnd.randint(-10 ** 20, 10 ** 20))
        doubles.append(x)
    for x in doubles:
        if x == 0:
            continue
        rows.append({"fn": "repr", "bits": bits(x), "out": repr(x)})
        rows.append({"fn": "repr", "bits": bits(-x), "out": repr(-x)})
    for x in [2.0 ** k for k in range(50, 1024, 7)] + [1e300, 1.7976931348623157e308, 9e18, 9.3e18, 1e19, 12345678901234567890.0]:
        for y in (x, -x):
            rows.append({"fn": "int_of_float", "bits": bits(y), "out": str(int(y))})
    texts = ["5", "+5", "-5", "007", "1_000", "1__0", "_1", "1_", " 42 ", "\t7\n", " 7", "7　", "٣",
             "１２", "1e3", "5.0", "0x10", "", " ", "-", "+", "-0", "00", "5 6", "\x1c5", "5\x1f", "\x0b5\x0c",
             "1_000_000", "9" * 30, "-" + "9" * 30, "9" * 4300, "9" * 4301, "١٢٣", "5\x00", " 5", "​5",
             "1.5", ".5", "5.", "+.5", "1e", "1e+", "1e5", "1E-5", "inf", "-Infinity", "nAn", "+inf", "infinit",
             "1_0.5", "1._5", "1_.5", "1e1_0", "1e_10", "1.0_1", "3.14159265358979323846", "1e400", "-1e400",
             "1e-400", "2e-324", "3e-324", "4.9e-324", "1.7976931348623157e308", "1.7976931348623159e308",
             "9007199254740993", "0.000001", "  -0.0  ", "٣.٥", "1,5", "1..5", "--1", "+-1", ".", "e5",
             "0e0", "00.00", "١e٢"]
    for t in texts:
        try:
            iv = str(int(t))
        except ValueError:
            iv = None
        try:
            fv = bits(float(t))
        except ValueError:
            fv = None
        rows.append({"fn": "int", "in": t, "out": iv})
        rows.append({"fn": "float", "in": t, "bits": fv})
    with open(out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
