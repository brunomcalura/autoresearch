#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/nemo_experiments/data/synthetic_smoke}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/nemo_experiments/pretrained/megatron/NVIDIA-Nemotron-3-Nano-4B-BF16}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
TOKENIZER="${TOKENIZER:-${PROJECT_DIR}/nemo_experiments/pretrained/NVIDIA-Nemotron-3-Nano-4B-BF16}"

cd "${PROJECT_DIR}"

uv run --offline python scripts/make_synthetic_data.py \
    --output-dir "${DATA_DIR}" \
    --seq-length "${SEQ_LENGTH}"

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint não encontrado: ${CHECKPOINT_PATH}" >&2
    echo "Defina CHECKPOINT_PATH para um checkpoint Megatron Bridge NemotronH 4B válido." >&2
    exit 2
fi

if ! uv run --offline python -c "import transformer_engine.pytorch" >/dev/null 2>&1; then
    echo "Dependência ausente: transformer_engine com suporte a PyTorch/CUDA." >&2
    echo "Use um ambiente NVIDIA NeMo/Megatron Bridge compatível antes de treinar." >&2
    exit 3
fi

CUDA_VISIBLE_DEVICES=0 uv run --offline torchrun \
    --standalone \
    --nproc-per-node=1 \
    train.py \
    --pretrained-checkpoint "${CHECKPOINT_PATH}" \
    --packed-data-dir "${DATA_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --seq-length "${SEQ_LENGTH}" \
    --train-iters 2 \
    --global-batch-size 1 \
    --micro-batch-size 1 \
    --context-parallel-size 1 \
    --warmup-iters 1 \
    --eval-interval 1 \
    --eval-iters 1 \
    --save-interval 2 \
    --experiment-name "smoke_nemotronh_4b_1gpu_2iters" \
    --wandb-project ""
