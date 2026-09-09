#!/usr/bin/env python3
# Add CUDA tag to wheel version
#
# PyPI/PEP 440 supports local version identifiers (e.g., 0.1.1+cu128)
# This script modifies pyproject.toml to add the CUDA tag to the version
# before building the wheel.
import subprocess
import sys
import tomlkit

# What a wheel declares, so that `pip install <wheel-url>` resolves from PyPI alone.
#
# cu130 asks for plain `torch==2.11.0`. The pin used to carry a local version --
# `torch==2.11.0+cu130` -- which exists only on download.pytorch.org, so the install
# died in resolution unless the caller passed --index-url. Nothing here links torch
# (there is no find_package(Torch); the extensions pull libcudart, libcublasLt,
# libcuda and libgomp only), so its CUDA variant is not our concern -- and PyPI's
# torch 2.11.0 is itself a CUDA 13 build, which is the one that matches.
#
# The CUDA runtime *is* our concern, because our own binaries link it, so we name it
# rather than hope torch's transitive set covers the right major. Bounds stay loose:
# the cu13 packages pin each other through `cuda-toolkit`, and a tighter floor here
# makes that graph unsatisfiable.
#
# cu128 keeps its local-version pin. PyPI publishes one Linux torch and it is CUDA 13,
# so a CUDA 12 host needs the pytorch index either way -- which is what install.sh is
# for. Its users are on drivers that predate CUDA 13 and cannot take the PyPI build.
CUDA_DEPS = {
    "cu128": [
        "torch==2.11.0+cu128",
        "torchvision==0.26.0+cu128",
    ],
    "cu130": [
        "torch==2.11.0",
        "torchvision==0.26.0",
        "nvidia-cuda-runtime>=13,<14",
        "nvidia-cublas>=13,<14",
        "nvidia-cufile>=1.15,<2",
    ],
}


def add_cuda_version_tag(cuda_tag: str):
    with open("pyproject.toml", "r") as f:
        data = tomlkit.load(f)

    # Get current version from git or fallback
    if "version" in data["project"]:
        current_version = data["project"]["version"]
    else:
        # First check if we're exactly on a tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "--match", "v*"], capture_output=True, text=True
        )
        if result.returncode == 0:
            # We're on an exact tag - use it directly
            current_version = result.stdout.strip().lstrip("v")
            print(f"On exact tag, using version: {current_version}")
        else:
            # Not on exact tag, use git describe
            result = subprocess.run(["git", "describe", "--tags", "--match", "v*"], capture_output=True, text=True)
            if result.returncode == 0:
                # Convert git describe output (e.g., v0.1.0-5-gabcdef) to PEP 440
                git_version = result.stdout.strip().lstrip("v")
                # Extract base version (before the commit count)
                if "-" in git_version:
                    current_version = git_version.split("-")[0]
                else:
                    current_version = git_version
                print(f"From git describe, using version: {current_version}")
            else:
                # Fallback version from setuptools_scm config
                fallback = data.get("tool", {}).get("setuptools_scm", {}).get("fallback_version", "0.0.1")
                current_version = fallback
                print(f"Using fallback version: {current_version}")

    # Remove 'version' from dynamic list if present
    if "dynamic" in data["project"] and "version" in data["project"]["dynamic"]:
        data["project"]["dynamic"].remove("version")
        if not data["project"]["dynamic"]:
            del data["project"]["dynamic"]

    # Add CUDA tag to version
    data["project"]["version"] = f"{current_version}+{cuda_tag}"
    print(f"Set version to {data['project']['version']}")

    existing_deps = list(data["project"].get("dependencies", []))

    # Remove 'torch*' dependencies if they exist to avoid conflicts
    existing_deps = [dep for dep in existing_deps if not dep.startswith("torch")]

    data["project"]["dependencies"] = existing_deps + CUDA_DEPS[cuda_tag]

    with open("pyproject.toml", "w") as f:
        tomlkit.dump(data, f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: set_cuda_version_tag.py <cuda_tag>")
        print("Example: set_cuda_version_tag.py cu128")
        sys.exit(1)

    cuda_tag = sys.argv[1]
    add_cuda_version_tag(cuda_tag)
