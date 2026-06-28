"""
Offline validation for autoresearch SFT — runs without GPU or Megatron Bridge.

Tests:
1. Syntax check on train.py and prepare.py
2. prepare.py exports (has_parquet, resolve_packed_paths, validate_config)
3. resolve_packed_paths with valid/invalid directory structures
4. validate_config with valid/invalid constraint combinations
5. train.py CLI argument parsing
"""

import ast
import sys
import tempfile
from pathlib import Path

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    msg = f"  {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if ok:
        PASS += 1
    else:
        FAIL += 1


# ── 1. Syntax ──────────────────────────────────────────────────────────

print("1. Syntax check")
for filename in ("train.py", "prepare.py"):
    path = Path(__file__).parent / filename
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=filename)
        check(filename, True)
    except SyntaxError as e:
        check(filename, False, str(e))


# ── 2. prepare.py exports ─────────────────────────────────────────────

print("\n2. prepare.py exports")
try:
    from prepare import has_parquet, resolve_packed_paths, validate_config

    check("has_parquet", callable(has_parquet))
    check("resolve_packed_paths", callable(resolve_packed_paths))
    check("validate_config", callable(validate_config))
except ImportError as e:
    check("imports", False, str(e))


# ── 3. resolve_packed_paths ────────────────────────────────────────────

print("\n3. resolve_packed_paths")

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)

    # 3a. Valid: train + valid
    train_dir = base / "full" / "splits" / "train"
    valid_dir = base / "full" / "splits" / "valid"
    train_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)
    (train_dir / "shard_0.parquet").write_bytes(b"fake")
    (valid_dir / "shard_0.parquet").write_bytes(b"fake")

    tp, vp = resolve_packed_paths(base / "full")
    check("train+valid", tp == train_dir and vp == valid_dir)

    # 3b. Valid: train only (no valid)
    train_only = base / "train_only" / "splits" / "train"
    valid_empty = base / "train_only" / "splits" / "valid"
    train_only.mkdir(parents=True)
    valid_empty.mkdir(parents=True)
    (train_only / "shard_0.parquet").write_bytes(b"fake")

    tp, vp = resolve_packed_paths(base / "train_only")
    check("train only (no valid)", tp == train_only and vp is None)

    # 3c. Invalid: no train parquet
    empty_train = base / "empty" / "splits" / "train"
    empty_train.mkdir(parents=True)
    try:
        resolve_packed_paths(base / "empty")
        check("missing train raises", False, "no exception raised")
    except FileNotFoundError:
        check("missing train raises", True)


# ── 4. validate_config ────────────────────────────────────────────────

print("\n4. validate_config")

# Valid
try:
    validate_config(8192, 1, 1)
    check("valid config (cp=1)", True)
except ValueError as e:
    check("valid config (cp=1)", False, str(e))

try:
    validate_config(8192, 1, 2)
    check("valid config (cp=2)", True)
except ValueError as e:
    check("valid config (cp=2)", False, str(e))

# Invalid: micro_batch_size != 1
try:
    validate_config(8192, 2, 1)
    check("mbs!=1 raises", False, "no exception")
except ValueError:
    check("mbs!=1 raises", True)

# Invalid: seq_length % cp != 0
try:
    validate_config(8193, 1, 2)
    check("seq%cp!=0 raises", False, "no exception")
except ValueError:
    check("seq%cp!=0 raises", True)

# Invalid: cp>1 and seq_length % (cp*2) != 0
try:
    validate_config(8194, 1, 4)
    check("seq%(cp*2)!=0 raises", False, "no exception")
except ValueError:
    check("seq%(cp*2)!=0 raises", True)


# ── 5. train.py CLI parsing ───────────────────────────────────────────

print("\n5. train.py CLI parsing")

# Parse the train.py AST to verify parse_args exists and has expected arguments
train_src = (Path(__file__).parent / "train.py").read_text(encoding="utf-8")
tree = ast.parse(train_src, "train.py")

func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
check("parse_args defined", "parse_args" in func_names)
check("main defined", "main" in func_names)

# Verify key CLI arguments are present in source
expected_args = [
    "--pretrained-checkpoint",
    "--packed-data-dir",
    "--seq-length",
    "--train-iters",
    "--global-batch-size",
    "--micro-batch-size",
    "--context-parallel-size",
    "--lr",
    "--min-lr",
    "--experiment-name",
]
for arg in expected_args:
    check(f"arg {arg}", arg in train_src)

# Verify imports from prepare
check("imports from prepare", "from prepare import" in train_src)


# ── Summary ────────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    print("VALIDATION FAILED")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
