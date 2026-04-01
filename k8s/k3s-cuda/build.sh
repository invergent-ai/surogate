#!/bin/bash

set -euxo pipefail

K3S_TAG=${K3S_TAG:="v1.35.3-k3s1"} 
CUDA_TAG=${CUDA_TAG:="12.9.1-cudnn-runtime-ubuntu24.04"}
IMAGE=${IMAGE:="ghcr.io/invergent-ai/k3s-${K3S_TAG}-cuda-${CUDA_TAG}"}

echo "IMAGE=$IMAGE"

docker build \
  --build-arg K3S_TAG=$K3S_TAG \
  --build-arg CUDA_TAG=$CUDA_TAG \
  -t $IMAGE .
# docker push $IMAGE
echo "Done!"