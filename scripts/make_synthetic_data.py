"""Generate a minimal packed-Parquet dataset for a training smoke test."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=128)
    return parser.parse_args()


def make_rows(count: int, seq_length: int, seed: int) -> list[dict[str, list[int]]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        # Packed SFT stores seq_length + 1 tokens so next-token shifting
        # produces exactly seq_length training tokens.
        input_ids = [rng.randint(100, 2000) for _ in range(seq_length + 1)]
        rows.append(
            {
                "input_ids": input_ids,
                "seq_start_id": [0],
                "loss_mask": [0] + [1] * seq_length,
            }
        )
    return rows


def write_split(path: Path, count: int, seq_length: int, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = make_rows(count, seq_length, seed)
    schema = pa.schema(
        [
            pa.field("input_ids", pa.list_(pa.int32())),
            pa.field("seq_start_id", pa.list_(pa.int32())),
            pa.field("loss_mask", pa.list_(pa.int8())),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path / "shard_0.parquet")


def main() -> None:
    args = parse_args()
    if args.seq_length < 2:
        raise ValueError("--seq-length must be at least 2")

    write_split(args.output_dir / "splits" / "train", 4, args.seq_length, seed=1234)
    write_split(args.output_dir / "splits" / "valid", 2, args.seq_length, seed=5678)
    print(f"Synthetic packed data created at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
