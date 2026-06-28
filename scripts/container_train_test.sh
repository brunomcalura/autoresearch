#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-nemo_experiments/data/stage1_sft_pt}"
HF_MODEL="${HF_MODEL:-nemo_experiments/pretrained/NVIDIA-Nemotron-3-Nano-4B-BF16}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-nemo_experiments/pretrained/megatron/NVIDIA-Nemotron-3-Nano-4B-BF16}"
TOKENIZER="${TOKENIZER:-${HF_MODEL}}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"

cd "${PROJECT_DIR}"

python scripts/download_tucano_sft.py \
    --output-dir "${DATA_DIR}" \
    --seq-length "${SEQ_LENGTH}"

if [[ ! -f "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]]; then
    python scripts/convert_checkpoint.py --hf-model "${HF_MODEL}" --output "${CHECKPOINT_PATH}"
fi

python -c "import torch, transformer_engine.pytorch; assert torch.cuda.is_available()"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
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
