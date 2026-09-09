from __future__ import annotations

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

def set_deterministic(seed: int = 42) -> None:
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
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def timestamp() -> str:
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def build_run_name(stage: str, tag: str, seed: int) -> str:
    return "__".join([stage, tag, f"seed{seed}", timestamp()])


def seed_artifact_root(seed: int, base: str | Path = "artifacts") -> Path:
    return Path(base) / f"seed{seed}"


def dataset_artifact_dir(artifacts_root: str | Path, dataset: str) -> Path:
    p = Path(artifacts_root) / f"artifacts_{dataset}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique_dir(path: Path) -> Path:
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


def dumps(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
