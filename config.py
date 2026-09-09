from __future__ import annotations
import os
from itertools import product

DEVICE = os.environ.get("TCAT_DEVICE", "cuda")   # "auto" | "cuda" | "cuda:0" | "cpu"
PROMOTE = True    # each new grid run becomes the dataset's selected run in its registry.json

DATA_DIR = "data"            # generator artifacts: data/<name>/<name>_*.npz
ARTIFACTS_DIR = "artifacts"  # per-seed subfolders (artifacts/seed<S>/...) are added automatically

CONTEXT = 50
DATASETS = {
    "henon":   dict(npz=os.path.join(DATA_DIR, "henon",   "henon_10000_10.npz"),   context=CONTEXT),
    # "lorenz":  dict(npz=os.path.join(DATA_DIR, "lorenz",  "lorenz_8000_10.npz"),   context=CONTEXT),
    # "ecoli":   dict(npz=os.path.join(DATA_DIR, "ecoli",   "ecoli_5000_10.npz"),    context=CONTEXT),
    # "yeast":   dict(npz=os.path.join(DATA_DIR, "yeast",   "yeast_5000_10.npz"),    context=CONTEXT),
    # "ecoliM":  dict(npz=os.path.join(DATA_DIR, "ecoliM",  "ecoliM_5000_10.npz"),   context=CONTEXT),
    # "yeastM":  dict(npz=os.path.join(DATA_DIR, "yeastM",  "yeastM_5000_10.npz"),   context=CONTEXT),
    # "synhill": dict(npz=os.path.join(DATA_DIR, "synhill", "synhill_5000_10.npz"),  context=CONTEXT),
}

RUN_DATASETS = list(DATASETS.keys())   # e.g. ["henon"] for a single dataset
RUN_SEEDS = [42]                       # e.g. [42, 71, 97] for multiple seeds

CRVAE_PARAM_GRID = {
    "lr":          [0.001],
    "batch_size":  [512],
    "hidden_size": [64],
    "num_layers":  [1],
    "dropout":     [0],
    "beta_kl":     [0.01, 0.05],
}
CRVAE_GRID_KEYS = ["lr", "batch_size", "hidden_size", "num_layers", "dropout", "beta_kl"]

CRVAE_TRAIN = dict(
    data_version="std",       # "std" | "raw" -- which trajectory array data_utils.load_trajectory loads
    val_frac=0.15,            # fraction of the trajectory held out for early stopping / model selection
    test_frac=0.15,           # fraction held out, untouched by training (see test_windows.npz)
    patience=100,             # early stopping on val loss
    lr_patience=20,           # ReduceLROnPlateau patience
    lr_factor=0.1,
    epochs=5000,
    select_metric="best_val_loss",  # lower is better; picks grid_results.csv's "best" trial
)

ATTACK_PARAM_GRID = {
    "lam_ntgt":   [0.5, 0.8],
    "lam_smooth": [0.1, 0.5],
    "epsilon":    [0.1, 0.5],
    "alpha":      [0.01, 0.05],   # alpha = epsilon / pgd_steps
    "pgd_steps":  [100],
    "batch_size": [1024],
}
ATTACK_GRID_KEYS = ["lam_ntgt", "lam_smooth", "epsilon", "alpha", "pgd_steps", "batch_size"]

ATTACK_CONFIG = dict(
    target_dims=None,      # None -> range(n_dim); or e.g. [0, 3, 7] for specific channels
    n_random_trials=3,     # random-mask control trials per (dim, combination)
)

BASE_TAG = None                                 # <basetag>___<sectag>___<rest ... portion>
CRVAE_SEC_TAG = CRVAE_TRAIN["data_version"]      # Default: crvae___grid___<rest ... portion>
ATTACK_SEC_TAG = "grid"                          # Default: timecat___grid___<rest ... portion>


def expand_crvae_grid() -> list[dict]:
    return [dict(zip(CRVAE_GRID_KEYS, values)) for values in product(*(CRVAE_PARAM_GRID[k] for k in CRVAE_GRID_KEYS))]


def expand_attack_grid() -> list[dict]:
    return [dict(zip(ATTACK_GRID_KEYS, values)) for values in product(*(ATTACK_PARAM_GRID[k] for k in ATTACK_GRID_KEYS))]
