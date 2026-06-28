#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-autoresearch:gpu}"

docker run --rm \
    --gpus all \
    --ipc host \
    --shm-size 16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1 \
    --env WANDB_MODE=disabled \
    --volume "${PROJECT_DIR}:/workspace" \
    --workdir /workspace \
    "${IMAGE}" \
    ./scripts/container_train_test.sh
