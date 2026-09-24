"""Assemble the GPU variant of a native TTS package.

The GPU variant is the CPU package -- the same model, codec, tokenizer and voices -- with lib/
replaced by the same Magpie runtime built with CUDA: libnemo_speech_tts, libggml, libggml-base,
libggml-cpu and libggml-cuda. voices.json lists the new libraries' checksums, which is what the
launcher verifies and how `surogate-tts --device N` recognizes the variant. Large assets are hard
links when the output is on the same filesystem, copies otherwise.

usage: python -m surogate.serve.tools.tts.gpu_variant CPU_PACKAGE CUDA_LIB_DIR OUT_DIR

CUDA_LIB_DIR is the `bin/` of a CUDA build of the runtime; see docs/inference/tts.md.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# The CUDA runtime surogate-tts-worker accepts (csrc/src/serve/tts/native_worker.cpp pins the same
# checksums; tests/serve/test_tts.py checks the two agree). Another build must be qualified and
# pinned in both places before a package can carry it.
PINNED = {
    "libnemo_speech_tts.so.1": "02fa87a8479ccde6b188b204c0086e87624034e5fb177c94b5ab0c2bb666e395",
    "libggml-base.so.0": "5916a79c9ece31dcbef048992230d1edd75c585c5c8534fc4f94100a40639718",
    "libggml-cuda.so.0": "285f960f40ade7e1ec7361d74ebbbb77809b8bfc2fdc34a355876a564c55dc2f",
}

LIBRARIES = {
    "libnemo_speech_tts": ["libnemo_speech_tts.so", "libnemo_speech_tts.so.1"],
    **{base: [f"{base}.so", f"{base}.so.0", f"{base}.so.0.12.0"]
       for base in ("libggml", "libggml-base", "libggml-cpu", "libggml-cuda")},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(cpu_package, cuda_lib_dir, out, *, check_pins=True):
    cpu_package, cuda_lib_dir, out = Path(cpu_package).resolve(), Path(cuda_lib_dir), Path(out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    if out.is_relative_to(cpu_package):
        raise SystemExit("the GPU variant cannot be written inside the CPU package")
    if check_pins:
        for name, digest in PINNED.items():
            actual = sha256((cuda_lib_dir / name).resolve())
            if actual != digest:
                raise SystemExit(f"{name} is not the pinned CUDA runtime build ({actual}); qualify it and "
                                 "update the pins in native_worker.cpp and this tool first")
    for path in sorted(cpu_package.rglob("*")):
        relative = path.relative_to(cpu_package)
        target = out / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif relative.parts[0] != "lib" and relative not in (Path("voices.json"), Path("build_info.json")):
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
    lib = out / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    for names in LIBRARIES.values():
        real = (cuda_lib_dir / names[-1]).resolve()
        for name in names:
            shutil.copyfile(real, lib / name)
    build_info = json.loads((cpu_package / "build_info.json").read_text(encoding="utf-8"))
    build_info = {
        "scope": "GPU variant: the CPU package's model, codec, tokenizer and voices with the same native "
                 "runtime built with CUDA. It runs on compute capability 12.0 (RTX 50-series) only.",
        "native_commit": build_info.get("native_commit"),
        "cmake": {"GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "120", "GGML_CUDA_NCCL": "OFF",
                  "GGML_NATIVE": "ON", "GGML_OPENMP": "ON"},
        "native_library_sha256": {f"lib/{path.name}": sha256(path) for path in sorted(lib.iterdir())},
    }
    (out / "build_info.json").write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")
    profile = json.loads((cpu_package / "voices.json").read_text(encoding="utf-8"))
    files = {name: digest for name, digest in profile["files"].items() if not name.startswith("lib/")}
    files.update({f"lib/{path.name}": sha256(path) for path in sorted(lib.iterdir())})
    files["build_info.json"] = sha256(out / "build_info.json")
    profile["files"] = files
    profile["scope"] = ("GPU variant of the native package: the same voices, decoded by the Magpie runtime "
                        "built with CUDA for compute capability 12.0.")
    (out / "voices.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(assemble(*sys.argv[1:]))
