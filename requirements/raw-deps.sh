#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(cd ${SCRIPT_DIR}/..; pwd)

uv pip install "vllm==0.12.0" --no-deps --pre --torch-backend=auto
uv pip install --no-deps "sglang==0.5.6.post2" "llmcompressor==0.8.1" "evalscope==1.1.1"
uv pip install --no-deps --index-url https://pypi.org/simple --extra-index-url https://test.pypi.org/simple lmcache==0.3.10.post2
