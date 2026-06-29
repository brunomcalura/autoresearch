"""
Autoresearch SFT script — Nemotron 3 Nano 4B fine-tuning via Megatron Bridge.
This is the file the agent modifies. Everything is fair game: hyperparameters,
model config, optimizer settings, scheduler, etc.

Usage:
    torchrun --nproc-per-node=<N_GPUS> train.py \
        --pretrained-checkpoint <path> \
        --packed-data-dir <path> \
        [--seq-length 8192] [--train-iters 512] [--lr 1.5e-6] ...
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.recipes.nemotronh.nemotronh import (
    nemotronh_4b_peft_config,
)
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from transformers import AutoConfig

from prepare import resolve_packed_paths, validate_config


def _get_git_short_hash() -> str:
    """Return the short git commit hash, or a timestamp fallback."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT Nemotron 3 Nano 4B from pre-packed Parquet SFT shards.",
    )
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument(
        "--packed-data-dir",
        type=Path,
        default=Path("./nemo_experiments/data/stage1_sft_pt"),
        help="Directory containing splits/train and optionally splits/valid.",
    )
    parser.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    parser.add_argument("--seq-length", type=int, default=8192)

    parser.add_argument("--train-iters", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--cp-comm-type", default=None, help="Optional Megatron CP comm type, e.g. p2p or a2a.")

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--warmup-iters", type=int, default=16)

    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=400)

    parser.add_argument("--experiment-name", default="nemotron_3_nano_4b_sft_stage1_8k_pt")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save checkpoints. Defaults to None (checkpoint saving disabled to prevent auto-resume).",
    )
    parser.add_argument("--wandb-project", default="portuguese-nemotron-sft")
    parser.add_argument("--wandb-save-dir", default="./nemo_experiments/wandb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_config(args.seq_length, args.micro_batch_size, args.context_parallel_size)

    train_path, valid_path = resolve_packed_paths(args.packed_data_dir)

    config = nemotronh_4b_peft_config()
    hf_config = AutoConfig.from_pretrained(args.tokenizer, trust_remote_code=True)
    config.model = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)

    config.model.seq_length = args.seq_length
    config.model.context_parallel_size = args.context_parallel_size
    config.model.cp_comm_type = args.cp_comm_type
    if args.context_parallel_size > 1:
        config.model.calculate_per_token_loss = True
        config.ddp.average_in_collective = False

    config.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    config.tokenizer.tokenizer_model = args.tokenizer

    config.dataset = FinetuningDatasetConfig(
        dataset_root=args.packed_data_dir,
        seq_length=args.seq_length,
        seed=5678,
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=args.seq_length,
            packed_train_data_path=train_path,
            packed_val_data_path=valid_path,
        ),
        dataloader_type="batch",
        do_validation=valid_path is not None,
        do_test=False,
    )
    config.train.train_iters = args.train_iters
    if hasattr(config.train, "train_samples"):
        config.train.train_samples = None
    config.train.global_batch_size = args.global_batch_size
    config.train.micro_batch_size = args.micro_batch_size

    config.optimizer.lr = args.lr
    config.optimizer.min_lr = args.min_lr
    config.scheduler.lr_warmup_iters = args.warmup_iters
    config.scheduler.lr_decay_iters = args.train_iters

    config.validation.eval_interval = args.eval_interval
    config.validation.eval_iters = args.eval_iters

    config.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    if args.save_dir is not None:
        config.checkpoint.save = f"{args.save_dir}/{args.experiment_name}"
        # Ensure the checkpoint directory exists
        Path(config.checkpoint.save).mkdir(parents=True, exist_ok=True)
        config.checkpoint.save_interval = args.save_interval
    else:
        config.checkpoint.save = None
    if hasattr(config.checkpoint, "finetune"):
        config.checkpoint.finetune = True

    config.logger.log_interval = 10
    config.logger.wandb_project = args.wandb_project
    wandb_run_name = f"{args.experiment_name}_{_get_git_short_hash()}"
    config.logger.wandb_exp_name = wandb_run_name
    config.logger.wandb_save_dir = args.wandb_save_dir

    print("Running packed-Parquet SFT v2 with:")
    print(f"  train_path: {train_path}")
    print(f"  valid_path: {valid_path}")
    print(f"  seq_length: {args.seq_length}")
    print(f"  global_batch_size: {args.global_batch_size}")
    print(f"  micro_batch_size: {args.micro_batch_size}")
    print(f"  context_parallel_size: {args.context_parallel_size}")
    print(f"  cp_comm_type: {config.model.cp_comm_type}")
    print(f"  lr: {args.lr}")
    print(f"  min_lr: {args.min_lr}")
    print(f"  warmup_iters: {args.warmup_iters}")
    print(f"  train_iters: {args.train_iters}")
    print(f"  checkpoint: {args.pretrained_checkpoint}")
    print(f"  save: {config.checkpoint.save}")
    print(f"  wandb_run: {wandb_run_name}")

    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
