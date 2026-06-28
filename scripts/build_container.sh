#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-autoresearch:gpu}"

docker build \
    --pull \
    --tag "${IMAGE}" \
    "${PROJECT_DIR}"
