#!/usr/bin/env python3
"""Download Tucano-SFT and convert it to packed Parquet for this project."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "nemo_experiments/data/stage1_sft_pt"
DEFAULT_CACHE = PROJECT_ROOT / "nemo_experiments/cache/huggingface"
LOCAL_TOKENIZER = PROJECT_ROOT / "nemo_experiments/pretrained/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_TOKENIZER = (
    str(LOCAL_TOKENIZER) if LOCAL_TOKENIZER.exists() else "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, tokenize, and pack TucanoBR/Tucano-SFT for Megatron Bridge."
    )
    parser.add_argument("--repo-id", default="TucanoBR/Tucano-SFT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument(
        "--max-samples", type=int, default=10_000,
        help="Usable conversations to process; 0 downloads the complete dataset.",
    )
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--rows-per-shard", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--long-sample", choices=("truncate", "skip"), default="truncate")
    parser.add_argument("--force", action="store_true", help="Replace output-dir if it exists.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.seq_length < 2:
        raise ValueError("--seq-length must be >= 2")
    if not 0 < args.valid_ratio < 1:
        raise ValueError("--valid-ratio must be between 0 and 1")
    if args.rows_per_shard < 1:
        raise ValueError("--rows-per-shard must be >= 1")


def normalize_messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("conversations is not a list")
    messages = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("conversation message is not an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("unsupported role or non-string content")
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("conversation has no assistant response")
    return messages


def as_ids(value) -> list[int]:
    return value if isinstance(value, list) else value["input_ids"]


def tokenize_conversation(tokenizer, messages, seq_length, long_sample):
    template_args = {"tokenize": True, "enable_thinking": False}
    input_ids = as_ids(tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, **template_args
    ))
    target_mask = [0] * len(input_ids)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt_ids = as_ids(tokenizer.apply_chat_template(
            messages[:index], add_generation_prompt=True, **template_args
        ))
        through_ids = as_ids(tokenizer.apply_chat_template(
            messages[: index + 1], add_generation_prompt=False, **template_args
        ))
        if through_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("assistant prompt is not a stable template prefix")
        if input_ids[: len(through_ids)] != through_ids:
            raise ValueError("conversation is not a stable template prefix")
        target_mask[len(prompt_ids) : len(through_ids)] = [1] * (
            len(through_ids) - len(prompt_ids)
        )

    max_tokens = seq_length + 1
    if len(input_ids) > max_tokens:
        if long_sample == "skip":
            return None
        input_ids, target_mask = input_ids[:max_tokens], target_mask[:max_tokens]
    if len(input_ids) < 2 or not any(target_mask[1:]):
        return None

    # The loader predicts token i+1 at position i, so align the target mask likewise.
    return {"input_ids": input_ids, "loss_mask": target_mask[1:] + [0]}


def pack_sequences(sequences, seq_length):
    """First-fit-decreasing packing using effective next-token lengths."""
    bins = []
    for sequence in sorted(sequences, key=lambda row: len(row["input_ids"]), reverse=True):
        cost = len(sequence["input_ids"]) - 1
        packed = next((item for item in bins if item["used"] + cost <= seq_length), None)
        if packed is None:
            packed = {"used": 0, "sequences": []}
            bins.append(packed)
        packed["used"] += cost
        packed["sequences"].append(sequence)

    rows = []
    for packed in bins:
        input_ids, loss_mask, seq_start_id = [], [], []
        for sequence in packed["sequences"]:
            seq_start_id.append(len(input_ids))
            input_ids.extend(sequence["input_ids"])
            loss_mask.extend(sequence["loss_mask"])
        rows.append({
            "input_ids": input_ids,
            "seq_start_id": seq_start_id,
            "loss_mask": loss_mask,
        })
    return rows


def write_shards(rows, split_dir: Path, rows_per_shard: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("seq_start_id", pa.list_(pa.int32())),
        pa.field("loss_mask", pa.list_(pa.int8())),
    ])
    for index, start in enumerate(range(0, len(rows), rows_per_shard)):
        table = pa.Table.from_pylist(rows[start : start + rows_per_shard], schema=schema)
        pq.write_table(
            table, split_dir / f"shard_{index:05d}.idx.parquet",
            compression="zstd", row_group_size=500,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir, cache_dir = args.output_dir.resolve(), args.cache_dir.resolve()
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"{output_dir} already exists; pass --force to replace it")

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, cache_dir=str(cache_dir)
    )
    dataset = load_dataset(
        args.repo_id, split=args.split, streaming=True, cache_dir=str(cache_dir)
    )

    sequences, rejected = [], 0
    print(f"Streaming {args.repo_id} ({args.split})...")
    for row in dataset:
        try:
            messages = normalize_messages(row.get("conversations"))
            sequence = tokenize_conversation(
                tokenizer, messages, args.seq_length, args.long_sample
            )
        except (KeyError, TypeError, ValueError):
            sequence = None
        if sequence is None:
            rejected += 1
            continue
        sequences.append(sequence)
        if args.max_samples and len(sequences) >= args.max_samples:
            break
        if len(sequences) % 1000 == 0:
            print(f"  tokenized {len(sequences):,} conversations")

    if len(sequences) < 2:
        raise RuntimeError("fewer than two usable conversations were downloaded")
    random.Random(args.seed).shuffle(sequences)
    valid_count = min(max(1, round(len(sequences) * args.valid_ratio)), len(sequences) - 1)
    valid_sequences, train_sequences = sequences[:valid_count], sequences[valid_count:]
    train_rows = pack_sequences(train_sequences, args.seq_length)
    valid_rows = pack_sequences(valid_sequences, args.seq_length)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        write_shards(train_rows, staging / "splits/train", args.rows_per_shard)
        write_shards(valid_rows, staging / "splits/valid", args.rows_per_shard)
        manifest = {
            "source": f"https://huggingface.co/datasets/{args.repo_id}",
            "dataset": args.repo_id,
            "source_split": args.split,
            "tokenizer": args.tokenizer,
            "seq_length": args.seq_length,
            "seed": args.seed,
            "conversations": len(sequences),
            "rejected_conversations": rejected,
            "train_conversations": len(train_sequences),
            "valid_conversations": len(valid_sequences),
            "train_packs": len(train_rows),
            "valid_packs": len(valid_rows),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Created packed dataset at {output_dir}")
    print(
        f"  conversations: {len(sequences):,} "
        f"({len(train_sequences):,} train, {len(valid_sequences):,} valid; "
        f"{rejected:,} rejected)"
    )
    print(f"  packs: {len(train_rows):,} train, {len(valid_rows):,} valid")
    print(f"Use with train.py: --packed-data-dir {output_dir}")


if __name__ == "__main__":
    main()
