"""
crvae_model/utils/common.py
Generic infrastructure used by the surrogate-training stage only (seeding,
device resolution, logging, json/csv IO, the run registry, timing, env
info). Nothing here is specific to the cLSTM architecture itself -- that
lives in crvae_model/model.py.

This is the training-side counterpart of utils/common.py at the project
root, which serves the attack stage (attack.py, run_attack.py,
analyze_results.py). The two are deliberately separate, near-identical
copies, not one shared module: crvae_model/ (the model definition, its
training loop, and everything the training loop needs) is meant to be
self-contained and importable on its own, with no dependency on any
attack-side code, and vice versa. Only data/ -- the dataset .npz files and
their generators -- is shared between the two stages; the checkpoint a
training run writes is the other cross-stage handoff, and it travels as a
plain file (via the dataset's registry.json + checkpoint.pt), not as a
shared Python import. See crvae_model/train.py's module docstring.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_deterministic(seed: int = 42) -> None:
    """Seed every RNG we touch and ask cuDNN/torch for deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("torch.use_deterministic_algorithms(True) not available on this "
              f"PyTorch or may raise for certain ops: {e}")


def resolve_device(device: str = "auto") -> torch.device:
    """'auto' picks cuda when available, else cpu. The original crvae_model
    code hardcoded '.cuda()' (and even a bare device='cuda' string inside
    LSTMEncoder) everywhere, so it could only ever run on a GPU machine;
    routing every module through this instead means the same code also runs
    on a CPU-only machine, just slower."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Naming and paths
# --------------------------------------------------------------------------- #
def timestamp() -> str:
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def build_run_name(stage: str, tag: str, seed: int) -> str:
    """e.g. crvae__grid__seed42__14-08-2026_10-06-40"""
    return "__".join([stage, tag, f"seed{seed}", timestamp()])


def seed_artifact_root(seed: int, base: str | Path = "artifacts") -> Path:
    """artifacts/seed<S>/ -- every dataset run for this seed lands underneath.
    Each seed gets its own root so that concurrent per-seed runs never touch
    the same registry file."""
    return Path(base) / f"seed{seed}"


def dataset_artifact_dir(artifacts_root: str | Path, dataset: str) -> Path:
    """artifacts/seed<S>/artifacts_<dataset>/ ; created if absent."""
    p = Path(artifacts_root) / f"artifacts_{dataset}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique_dir(path: Path) -> Path:
    """Guard against two runs starting inside the same second."""
    if not path.exists():
        return path
    for k in range(1, 100):
        cand = path.with_name(f"{path.name}__r{k}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot allocate a unique run directory near {path}")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: str | Path, level: int = logging.INFO) -> logging.Logger:
    """One logger per run: writes every line to <run_dir>/run.log AND stdout,
    so a single file has the whole grid's output instead of one log file per
    hyperparameter combination (the original train_<dataset>.py scripts
    redirected sys.stdout to a fresh file per combination -- fragile, and
    impossible to tail while a grid is running). Exceptions should be logged
    with logger.error(..., exc_info=True) (or with traceback.format_exc()
    folded into the message) so the real error is always visible here, not
    just a bare exit code."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_csv_row(path: str | Path, row: dict) -> None:
    """Append one row to a CSV, writing the header if the file is new.
    Every row appended through one run should share the same keys, in the
    same order, or the file becomes ragged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: str | Path, model: torch.nn.Module, model_hp: dict) -> None:
    """Weights plus just enough architecture info to rebuild the model later:
    n_dim, hidden_size, AND the causal_graph itself (see crvae_model.model.
    cLSTM's docstring on why the graph is part of the model's own
    hyperparameters, not an inference-time extra). This is the file format
    the attack-side utils/data_utils.load_model reads -- the only contract
    between the two stages, deliberately a plain file rather than a shared
    loader function."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_hp": model_hp}, path)


def save_history(path: str | Path, history: list[dict]) -> None:
    """Per-epoch training curves as a compressed .npz instead of nested JSON
    lists. `history` is a list of one flat dict per epoch, all epochs sharing
    the same keys (e.g. train_loss, val_loss, ...); this stacks each key into
    its own array. Callers should keep only a pointer to this file in their
    JSON metadata (see train.py's "history_path" field)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.array([h[k] for h in history]) for k in history[0]} if history else {}
    np.savez_compressed(path, **arrays)


# --------------------------------------------------------------------------- #
# Registry: which run of a dataset+seed is "the" one the attack stage should
# load a checkpoint from.
# --------------------------------------------------------------------------- #
def update_registry(
    registry_path: str | Path,
    run_name: str,
    entry: dict,
    promote: bool = True,
    logger=None,
) -> dict:
    """Record `entry` under `run_name` and decide whether it becomes `selected`.

    artifacts/seed<S>/artifacts_<dataset>/registry.json is the one file the
    attack stage reads to find which experiment directory -- and, inside it,
    which checkpoint.pt -- to load; see the attack-side utils/data_utils.py.
    Every grid run for a (dataset, seed) is self-contained, so the default
    policy promotes each new run automatically; pass promote=False to only
    record it.
    """
    registry_path = Path(registry_path)
    reg = load_json(registry_path) if registry_path.exists() else {"selected": None, "runs": {}}
    entry = {**entry, "run_name": run_name, "recorded_at": timestamp()}
    reg["runs"][run_name] = entry

    if reg["selected"] is None:
        reg["selected"] = entry
        action = "promoted (first run)"
    elif promote:
        previous = reg["selected"]["run_name"]
        reg["selected"] = entry
        action = f"promoted (replaces {previous})"
    else:
        action = f"recorded only; selected run remains {reg['selected']['run_name']}"

    save_json(reg, registry_path)
    if logger is not None:
        logger.info(f"[registry] {run_name}: {action}")
    return reg


# --------------------------------------------------------------------------- #
# Environment / provenance
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "n/a"


def env_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
    return info


class Timer:
    """with Timer() as t: ...   ->   t.elapsed (seconds)"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


def fmt_seconds(s: float) -> str:
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:d}m{sec:02d}s"
