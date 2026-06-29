#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-autoresearch:gpu}"

# Ensure configuration folders exist on the host so Docker mounts them with the correct permissions
mkdir -p "${HOME}/.claude"
mkdir -p "${HOME}/.config/@anthropic-ai/claude-code"
mkdir -p "${HOME}/.npm"

echo "=========================================================="
echo "Starting Interactive Autoresearch Container: ${IMAGE}"
echo "Workspace mounted at: /workspace"
echo "Claude Code auth & configs are persisted in: ${HOME}/.claude"
echo "=========================================================="

# Run container with interactive terminal, GPU support, and resource configurations.
# Forwards API keys from host environment if available.
docker run -it --rm \
    --gpus all \
    --ipc host \
    --shm-size 16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1 \
    --volume "${PROJECT_DIR}:/workspace" \
    --volume "${HOME}/.claude:/root/.claude" \
    --volume "${HOME}/.config/@anthropic-ai/claude-code:/root/.config/@anthropic-ai/claude-code" \
    --volume "${HOME}/.npm:/root/.npm" \
    --workdir /workspace \
    "${IMAGE}" \
    bash -c "
      if [ ! -d 'nemo_experiments/data/stage1_sft_pt/splits/train' ] || [ -z \"\$(ls -A nemo_experiments/data/stage1_sft_pt/splits/train 2>/dev/null)\" ]; then
          echo '=========================================================='
          echo 'Dataset não encontrado em nemo_experiments/data/stage1_sft_pt/splits/train.'
          echo 'Iniciando o download do dataset TucanoBR/Tucano-SFT...'
          echo '=========================================================='
          python scripts/download_tucano_sft.py
      fi
      exec bash
    "

