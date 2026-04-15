#!/bin/bash

docker build -t ghcr.io/invergent-ai/surogate-llama-cpp:full-cuda12 -f Dockerfile .
docker push ghcr.io/invergent-ai/surogate-llama-cpp:full-cuda12
