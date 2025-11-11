# SPDX-License-Identifier: AGPL-3.0
# SPDX-FileCopyrightText: Copyright contributors to the Surogate project

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
from packaging.version import Version, parse
from setuptools import setup
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _no_device() -> bool:
    return SUROGATE_TARGET_DEVICE == "empty"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return SUROGATE_TARGET_DEVICE == "cuda" and has_cuda and not _is_tpu()


def _is_tpu() -> bool:
    return SUROGATE_TARGET_DEVICE == "tpu"


def _is_cpu() -> bool:
    return SUROGATE_TARGET_DEVICE == "cpu"


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_surogate_version() -> str:
    if env_version := os.getenv("SUROGATE_VERSION_OVERRIDE"):
        return env_version
    version = get_version(write_to="surogate/_version.py")
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _no_device():
        if envs.SUROGATE_TARGET_DEVICE == "empty":
            version += f"{sep}empty"
    elif _is_cuda():
        cuda_version = str(get_nvcc_cuda_version())
        if cuda_version != envs.SUROGATE_MAIN_CUDA_VERSION:
            cuda_version_str = cuda_version.replace(".", "")[:3]
            # skip this for source tarball, required for pypi
            if "sdist" not in sys.argv:
                version += f"{sep}cu{cuda_version_str}"
    elif _is_cpu():
        if envs.SUROGATE_TARGET_DEVICE == "cpu":
            version += f"{sep}cpu"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version

if not (sys.platform.startswith("linux")):
    logger.fatal("Surogate only supports Linux platform")
    sys.exit(1)

envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "surogate", "envs.py"))
SUROGATE_TARGET_DEVICE = envs.SUROGATE_TARGET_DEVICE

package_data = {
    "surogate": [
        "py.typed",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
    ]
}

setup(
    version=get_surogate_version(),
    extras_require={},
    package_data=package_data,
)
