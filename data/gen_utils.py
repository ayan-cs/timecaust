from __future__ import annotations

import json
import os

import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return (X - mu) / sigma


def save_dataset(out_dir: str, filename: str, X_raw: np.ndarray, A: np.ndarray, meta: dict, **extra_arrays) -> str:
    os.makedirs(out_dir, exist_ok=True)
    X_raw = np.asarray(X_raw, dtype=np.float64)
    A = np.asarray(A)
    kwargs = {
        "X_raw": X_raw,
        "X_std": standardize(X_raw),
        "A": A,
        "meta": np.array(json.dumps(meta, default=_json_default)),
        **{k: np.asarray(v) for k, v in extra_arrays.items()},
    }
    fpath = os.path.join(out_dir, filename)
    np.savez_compressed(fpath, **kwargs)
    print(f"  saved -> {fpath}  (X_raw {X_raw.shape}, A {A.shape})", flush=True)
    return fpath


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
