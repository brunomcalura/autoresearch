# autoresearch

This is an experiment to have the LLM optimize SFT (Supervised Fine-Tuning) of the Nemotron 3 Nano 4B model using Megatron Bridge.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun28`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` -- repository context.
   - `prepare.py` -- fixed utilities (path resolution, validation). Do not modify.
   - `train.py` -- the file you modify. SFT configuration, hyperparameters, model config.
4. **Verify the environment**: Confirm that Megatron Bridge is installed (`python -c "import megatron.bridge"`). If not, tell the human to install it.
5. **Verify data exists**: Confirm the packed Parquet data directory exists and contains `splits/train/*.parquet`. The default path is `./nemo_experiments/data/stage1_sft_pt`.
6. **Verify checkpoint exists**: Confirm the pretrained checkpoint path is valid.
7. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment uses `torchrun` for distributed training across the GPUs configured by the user. Launch it as:

```bash
torchrun --nproc-per-node=<N_GPUS> train.py \
    --pretrained-checkpoint <path> \
    --packed-data-dir <path> \
    [--seq-length 8192] [--train-iters 512] [--lr 5e-6] ...
```

The number of GPUs (`<N_GPUS>`) is chosen **exclusively by the user**. The agent must never change `--nproc-per-node`.

**Iteration budget**: The goal of each run is to **validate code changes**, not to train to convergence. Keep the number of training iterations low — around **512 iterations** is the target. This is enough to confirm that the training loop runs correctly, losses decrease, and there are no crashes or regressions. Do not increase `--train-iters` significantly beyond this unless the user explicitly asks.

**Warmup rule**: `--warmup-iters` should always be set to **2% – 5%** of `--train-iters`. For example, with 512 iterations, warmup should be between ~10 and ~26 iterations. This ensures a brief but sufficient learning rate ramp-up without wasting a large fraction of a short validation run on warmup.

**What you CAN do:**
- Modify `train.py` -- this is the only file you edit. Everything is fair game: hyperparameters (learning rate, batch size, warmup, decay), optimizer settings, scheduler, context parallelism, sequence length, evaluation intervals, etc.

**What you CANNOT do:**
- Change the number of GPUs (`--nproc-per-node`). This is set by the user and must not be modified by the agent.
- Modify `prepare.py`. It is read-only. It contains the fixed path resolution and validation utilities.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the model architecture (the Nemotron 3 Nano 4B recipe is fixed via `nemotronh_4b_finetune_config()`).

**The goal is simple: get the best SFT quality.** You are optimizing the fine-tuning process -- learning rate schedules, batch sizes, training iterations, warmup strategies, etc. The model architecture itself is fixed (Nemotron 3 Nano 4B), but everything about how you train it is up for grabs.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline with default hyperparameters.

## Output format

Megatron Bridge logs training progress to stdout/wandb. Each W&B run is automatically named `<experiment_name>_<git_short_hash>`, so every iteration of the loop gets a unique, traceable run in the W&B dashboard. The key metrics to track:

- **Training loss** -- lower is better, monitor for convergence
- **Validation loss** -- if validation data is present (`splits/valid/`)
- **Training iterations completed** -- did it finish all `--train-iters`?

Extract results from the log:

```bash
grep "lm loss" run.log | tail -5
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	train_loss	val_loss	status	description
```

1. git commit hash (short, 7 chars)
2. final training loss (e.g. 1.234567) -- use 0.000000 for crashes
3. final validation loss (e.g. 1.345678) -- use 0.000000 if no validation or crash
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	train_loss	val_loss	status	description
a1b2c3d	1.234567	1.345678	keep	baseline
b2c3d4e	1.200000	1.310000	keep	lr=1.5e-6 warmup=100
c3d4e5f	1.250000	1.400000	discard	lr=1e-4 (too aggressive)
d4e5f6g	0.000000	0.000000	crash	gbs=256 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun28`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly editing the code.
3. git commit
4. Run the experiment: `torchrun --nproc-per-node=<N_GPUS> train.py --pretrained-checkpoint <path> --packed-data-dir <path> > run.log 2>&1`
5. Read out the results: `grep "lm loss" run.log | tail -5`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If loss improved (lower), you "advance" the branch, keeping the git commit
9. If loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous.
