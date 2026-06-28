# autoresearch

Autonomous SFT (Supervised Fine-Tuning) optimization for the Nemotron 3 Nano 4B model using [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted from pretraining-from-scratch to SFT of production LLMs.

## How it works

Give an AI agent a real SFT setup and let it experiment autonomously. It modifies hyperparameters, trains, checks if the result improved, keeps or discards, and repeats.

The repo has three files that matter:

- **`prepare.py`** -- fixed utilities (path resolution, config validation). Not modified by the agent.
- **`train.py`** -- the single file the agent edits. Contains SFT configuration via Megatron Bridge: hyperparameters, optimizer settings, scheduler, batch sizes, etc.
- **`program.md`** -- agent instructions. Edited by the human.

## Quick start

**Requirements:** NVIDIA GPUs, Python 3.12, [uv](https://docs.astral.sh/uv/), [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) (or NVIDIA NeMo container).

### 1. Install dependencies

```bash
uv sync
```

### 2. Prepare data

SFT requires pre-packed Parquet data in the following layout:

```
<packed-data-dir>/
  splits/
    train/    # *.parquet files (required)
    valid/    # *.parquet files (optional)
```

For a reproducible Portuguese test dataset, stream a subset of
[`TucanoBR/Tucano-SFT`](https://huggingface.co/datasets/TucanoBR/Tucano-SFT)
from Hugging Face and convert it to the required packed format:

```bash
uv run python scripts/download_tucano_sft.py
```

This processes 10,000 conversations by default and writes them to
`nemo_experiments/data/stage1_sft_pt`. Use `--max-samples 0` for the complete
dataset or `--help` to see the tokenizer, sequence-length, cache, and output
options.

### 3. Run SFT

```bash
torchrun --nproc-per-node=<N_GPUS> train.py \
    --pretrained-checkpoint <path-to-checkpoint> \
    --packed-data-dir <path-to-packed-data> \
    --seq-length 8192 \
    --train-iters 512 \
    --warmup-iters 20 \
    --lr 5e-6
```

The number of GPUs is a user-chosen hyperparameter.

### Validate and smoke-test

Run the offline checks without a GPU:

```bash
uv run python validate.py
```

The repository also includes a native smoke test and a reproducible Docker flow:

```bash
./scripts/smoke_test.sh

./scripts/build_container.sh
./scripts/run_train_test.sh
```

Models, datasets, converted checkpoints, logs, W&B files, and `results.tsv` are
local experiment state. They belong under `nemo_experiments/` (or `wandb/`) and
are intentionally excluded from Git.

## Running the agent

Spin up your AI coding agent in this repo and prompt something like:

```
Have a look at program.md and let's kick off a new SFT experiment!
```

The agent will read `program.md`, set up a branch, and start experimenting autonomously.

## Project structure

```
.
├── prepare.py       # fixed utilities; the agent must not modify this
├── train.py         # SFT configuration modified by the agent
├── validate.py      # offline validation (no GPU required)
├── program.md       # autonomous research instructions
├── scripts/         # checkpoint, data, container, and smoke-test utilities
├── docker/          # Docker build support files
├── notebooks/       # optional experiment analysis
├── Dockerfile       # reproducible NVIDIA/Megatron Bridge environment
├── pyproject.toml   # Python project and direct dependencies
└── uv.lock          # reproducible dependency lock
```

The root stays intentionally small to preserve the original autoresearch model:
`prepare.py`, `train.py`, and `program.md` are the core workflow.

## Key differences from original autoresearch

| | Original | This fork |
|---|---|---|
| **Task** | Pretraining from scratch | SFT of Nemotron 3 Nano 4B |
| **Framework** | Custom GPT + Muon optimizer | Megatron Bridge |
| **GPUs** | Single GPU | Multi-GPU via `torchrun` |
| **Data** | ClimbMix-400B shards | Pre-packed Parquet SFT data |
| **Agent modifies** | Model architecture + training | Hyperparameters + config |
| **Metric** | val_bpb | Training/validation loss |

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained-checkpoint` | (required) | Path to pretrained model checkpoint |
| `--packed-data-dir` | `./nemo_experiments/data/stage1_sft_pt` | Directory with packed Parquet splits |
| `--tokenizer` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | HuggingFace tokenizer |
| `--seq-length` | `8192` | Sequence length |
| `--train-iters` | `512` | Training iterations |
| `--global-batch-size` | `64` | Global batch size |
| `--micro-batch-size` | `1` | Micro batch size (must be 1 for packed SFT) |
| `--context-parallel-size` | `1` | Context parallelism degree |
| `--lr` | `5e-6` | Learning rate |
| `--min-lr` | `5e-7` | Minimum learning rate |
| `--warmup-iters` | `32` | Warmup iterations |
| `--experiment-name` | `nemotron_3_nano_4b_sft_stage1_8k_pt` | Experiment name (wandb + checkpoint dir) |

## License

MIT
