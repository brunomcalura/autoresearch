"""
Fixed utilities for autoresearch SFT experiments.
This file is READ-ONLY — the agent must NOT modify it.

Contains:
- Path resolution for packed Parquet SFT data
- Configuration validation for Megatron Bridge constraints
- Model download (run directly: ``python prepare.py``)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_LOCAL_DIR = Path("./nemo_experiments/pretrained/NVIDIA-Nemotron-3-Nano-4B-BF16")


def has_parquet(path: Path) -> bool:
    """Check whether *path* points to (or contains) at least one Parquet file."""
    return (path.is_file() and path.suffix == ".parquet") or (
        path.is_dir() and any(path.glob("*.parquet"))
    )


def resolve_packed_paths(packed_data_dir: Path) -> tuple[Path, Path | None]:
    """Resolve train/valid split directories inside *packed_data_dir*.

    Expected layout::

        packed_data_dir/
        └── splits/
            ├── train/   ← must contain *.parquet
            └── valid/   ← optional

    Returns (train_path, valid_path). *valid_path* is ``None`` when no
    validation Parquet files are found.

    Raises:
        FileNotFoundError: If the train split has no Parquet files.
    """
    train_path = packed_data_dir / "splits" / "train"
    valid_path = packed_data_dir / "splits" / "valid"

    if not has_parquet(train_path):
        raise FileNotFoundError(f"No packed train parquet files found in: {train_path}")

    if not has_parquet(valid_path):
        valid_path = None

    return train_path, valid_path


def validate_config(seq_length: int, micro_batch_size: int, context_parallel_size: int) -> None:
    """Validate Megatron Bridge SFT constraints. Raises ValueError on failure."""
    if micro_batch_size != 1:
        raise ValueError("Packed SFT in Megatron Bridge requires --micro-batch-size 1.")
    if seq_length % context_parallel_size != 0:
        raise ValueError("--seq-length must be divisible by --context-parallel-size.")
    if context_parallel_size > 1 and seq_length % (context_parallel_size * 2) != 0:
        raise ValueError(
            "With CP enabled, --seq-length must be divisible by context_parallel_size * 2."
        )


# ---------------------------------------------------------------------------
# CLI: download pretrained checkpoint
# ---------------------------------------------------------------------------

def _download_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pretrained Nemotron 3 Nano 4B checkpoint.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo to download (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Local directory to save the model (default: {DEFAULT_LOCAL_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _download_args()
    local_dir: Path = args.local_dir

    local_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} → {local_dir.resolve()} ...")
    snapshot_download(repo_id=args.repo_id, local_dir=str(local_dir))
    print()
    print("Done! Use this path with train.py:")
    print(f"  --pretrained-checkpoint {local_dir}")

