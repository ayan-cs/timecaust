"""
data/gen_utils.py
Shared save routine for every data/<name>_gen.py generator, so the
"standardize + write .npz" boilerplate exists in exactly one place instead
of being copy-pasted (with the standardization applied inconsistently) into
each dataset's data_prep.ipynb cell the way TimeCAT_new's notebook did:
henon and synhill kept the raw trajectory, ecoli/yeast/ecoliM/yeastM applied
sklearn's StandardScaler, and lorenz applied a MinMaxScaler to [0, 1] --
three different rescalings, chosen per-dataset, with no raw copy kept
alongside any of them.

Every generator here now keeps both: `X_raw` exactly as produced by its
map/ODE/SCM, and `X_std`, a zero-mean/unit-variance standardization of that
same trajectory computed the same way for every dataset (matching the
convention codebase_caufrts's data/<name>_anm.py generators already use).
This is a real behavior change for the datasets that did not previously use
z-score standardization (henon, lorenz, synhill) -- see the project README's
"what changed" section. config.CRVAE_TRAIN["data_version"] selects "raw" or
"std" at training time, per codebase_caufrts's precedent.
"""

from __future__ import annotations

import json
import os

import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance standardization, one scale factor per
    channel (column), matching codebase_caufrts's data/<name>_anm.py."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return (X - mu) / sigma


def save_dataset(out_dir: str, filename: str, X_raw: np.ndarray, A: np.ndarray, meta: dict, **extra_arrays) -> str:
    """Writes <out_dir>/<filename> as one compressed .npz with:
        X_raw   original trajectory, [T, D], exactly as generated
        X_std   standardize(X_raw), [T, D]
        A       ground-truth causal adjacency, [D, D]; A[i, j] == 1 means j
                is a parent of i (matches crvae_model.model.cLSTM's
                causal_graph convention)
        meta    a JSON string (parsed back with json.loads) of generation
                parameters, for provenance
    plus any extra dataset-specific arrays (e.g. synhill's sign matrix)
    passed as keyword arguments.
    """
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
