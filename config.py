"""
config.py
Every tunable for the surrogate-model + TimeCAT attack pipeline lives here,
not on the command line -- there is no argparse anywhere in this project.
Edit this file once, then `python run_model.py` (surrogate training) and/or
`python run_attack.py` (the TimeCAT attack).

Mirrors config.py in the sibling codebase_caufrts project: a dataset
registry, a hyperparameter grid per stage, fixed training settings, and
which of those this invocation actually runs. The two projects are
otherwise unrelated -- this one attacks a per-dataset cLSTM surrogate
forecaster (crvae_model/), caufrts trains a causal-discovery forecaster.
"""

from __future__ import annotations
import os
from itertools import product

DEVICE = os.environ.get("TCAT_DEVICE", "cuda")   # "auto" | "cuda" | "cuda:0" | "cpu"
# TCAT_DEVICE lets run_model.py/run_attack.py override this per-subprocess: when
# DEVICE is "cuda:N" it restricts the subprocess to that GPU via
# CUDA_VISIBLE_DEVICES and passes TCAT_DEVICE=cuda so the subprocess (which
# re-imports this file fresh) resolves to the now-remapped device 0 instead
# of the stale index N.
PROMOTE = True    # each new grid run becomes the dataset's selected run in its registry.json

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = "data"            # generator artifacts: data/<name>/<name>_*.npz
ARTIFACTS_DIR = "artifacts"  # per-seed subfolders (artifacts/seed<S>/...) are added automatically
# No LOGS_DIR: run_model.py/run_attack.py's multi-job subprocess dispatch used
# to redirect each subprocess's stdout/stderr into logs_all/seed<S>/, but
# that was pure duplication of the same run's own run.log/global_logs.log
# (crvae_model.utils.common.get_logger / utils.common.get_logger already
# dual-write every line to both stdout and that file) -- and it was never
# created at all whenever RUN_DATASETS/RUN_SEEDS resolved to a single job,
# since that path trains/attacks in-process instead of via subprocess.
# Subprocesses now just inherit this process's own stdout/stderr.

# --------------------------------------------------------------------------- #
# Dataset registry -- one entry per generator artifact under DATA_DIR (see
# data/<name>_gen.py; run `python -m data.<name>_gen` once per dataset to
# (re)create its .npz). `context` is the window length fed to the surrogate
# encoder (context - 1 history steps + 1 one-step-ahead target); it does not
# affect how the .npz itself was generated (every generator's own RNG seed
# is fixed inside data/<name>_gen.py, independent of this file).
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# What run_model.py (surrogate training) and run_attack.py (TimeCAT attack) run.
#   - One dataset and one seed -> trains/attacks in this process (easiest to
#     debug: one process, one traceback if something goes wrong).
#   - More than one dataset and/or seed -> each (dataset, seed) job runs
#     sequentially, one subprocess per job (mirrors codebase_caufrts's
#     run.py, and the old ztimecat_run_all.py's subprocess-per-dataset
#     pattern, minus the per-dataset script duplication).
# Trim either list to work with a single dataset or a single seed.
# --------------------------------------------------------------------------- #
RUN_DATASETS = list(DATASETS.keys())   # e.g. ["henon"] for a single dataset
RUN_SEEDS = [42]                       # e.g. [42, 71, 97] for multiple seeds

# --------------------------------------------------------------------------- #
# Stage 1 -- surrogate model (crvae_model.model.cLSTM) hyperparameter grid.
# Values are the union of the per-dataset "quick" grids the original
# train_<dataset>.py scripts hardcoded; trim/expand freely. Every combination
# is trained to early stopping and kept (trials/comb_XXXX/); the best by
# val_loss is promoted to best/ and to the dataset's registry.json.
# --------------------------------------------------------------------------- #
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
    lr_patience=20,           # ReduceLROnPlateau patience (was `step_size`)
    lr_factor=0.1,
    epochs=5000,
    select_metric="best_val_loss",  # lower is better; picks grid_results.csv's "best" trial
)

# --------------------------------------------------------------------------- #
# Stage 2 -- TimeCAT attack hyperparameter grid. `target_dims=None` means
# "every channel of the dataset" (resolved at runtime from n_dim); trim to a
# list of ints to attack only specific channels.
# --------------------------------------------------------------------------- #
ATTACK_PARAM_GRID = {
    "lam_ntgt":   [0.5, 0.8],
    "lam_smooth": [0.1, 0.5],
    "epsilon":    [0.1, 0.5],
    "alpha":      [0.01, 0.05],   # alpha = epsilon / pgd_steps, usually
    "pgd_steps":  [100],
    "batch_size": [1024],         # lower this on smaller GPUs
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
    """Cartesian product of CRVAE_PARAM_GRID as a list of flat dicts, in the
    same trial order as the original nested `product(...)` loop."""
    return [dict(zip(CRVAE_GRID_KEYS, values)) for values in product(*(CRVAE_PARAM_GRID[k] for k in CRVAE_GRID_KEYS))]


def expand_attack_grid() -> list[dict]:
    """Cartesian product of ATTACK_PARAM_GRID as a list of flat dicts."""
    return [dict(zip(ATTACK_GRID_KEYS, values)) for values in product(*(ATTACK_PARAM_GRID[k] for k in ATTACK_GRID_KEYS))]
