"""Convert a local Hugging Face checkpoint to Megatron torch_dist format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from megatron.bridge import AutoBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    AutoBridge.import_ckpt(
        args.hf_model,
        args.output,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    main()
