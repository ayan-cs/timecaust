from __future__ import annotations

import json

import numpy as np
import torch

from crvae_model.model import cLSTM

def load_trajectory(npz_path, version="std"):
    key = {"std": "X_std", "raw": "X_raw"}.get(version)
    if key is None:
        raise ValueError(f"unknown data version {version!r}; expected one of ['std', 'raw']")
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d[key], dtype=np.float32)
    meta = json.loads(str(d["meta"]))
    return X, meta


def load_ground_truth(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return np.asarray(d["A"], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Chronological split + windowing
# --------------------------------------------------------------------------- #
def create_split_windows(X, context, val_frac, test_frac, verbose=True, logger=None):
    if context < 2:
        raise ValueError("Context value should have minimum length = 2.")

    T = len(X)
    n_test = int(T * test_frac)
    n_val = int(T * val_frac)
    n_train = T - n_val - n_test
    segments = [
        ("train", X[:n_train]),
        ("val", X[n_train:n_train + n_val]),
        ("test", X[n_train + n_val:]),
    ]

    log = logger.info if logger is not None else (lambda m: print(m, flush=True))
    if verbose:
        log("Complete!")

    out = {}
    for name, seg in segments:
        seg = seg.tolist() if isinstance(seg, np.ndarray) else seg
        if context > len(seg):
            raise ValueError(
                f"Context value cannot exceed the {name} split's length ({len(seg)} of T={T}); "
                f"retry with a smaller context, or a smaller val_frac/test_frac."
            )
        out[name] = [seg[i: i + context] for i in range(len(seg) - context + 1)]
        log(f"{name} windows : {len(out[name]), len(out[name][0]), len(out[name][0][0])}")

    return out["train"], out["val"], out["test"]


def split_window(window):
    return window[:-1], window[-1:]


def get_parent_mask(causal_graph, target_dim):
    mask = causal_graph[target_dim] != 0
    return mask.astype(bool)


def get_nonparent_mask(causal_graph, target_dim):
    parent_mask = get_parent_mask(causal_graph, target_dim)
    nonparent_mask = ~parent_mask
    nonparent_mask[target_dim] = False
    return nonparent_mask


def get_random_mask(n_dim, n_select, exclude_dim=None, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    candidates = list(range(n_dim))
    if exclude_dim is not None:
        candidates.remove(exclude_dim)

    n_select = min(n_select, len(candidates))
    chosen = rng.choice(candidates, size=n_select, replace=False)

    mask = np.zeros(n_dim, dtype=bool)
    mask[chosen] = True
    return mask


def load_model(checkpoint_path, device) -> cLSTM:
    ckpt = torch.load(checkpoint_path, map_location=device)
    hp = ckpt["model_hp"]
    model = cLSTM(
        n_dim=hp["n_dim"], hidden_size=hp["hidden_size"],
        causal_graph=np.asarray(hp["causal_graph"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
