"""
data/henon_gen.py
Coupled-Henon-map dataset: node i's next value depends on its own lag-1/lag-2
history and (for i > 0) its left neighbor's lag-1 value -- see
data.graphs.henon_graph for the resulting ground-truth adjacency.

Ported from data_prep.ipynb's "Henon" section. One indexing bug fixed: the
notebook seeded its whole (T+1, D) array with np.random.rand up front, then
ran the recurrence starting at i=0, whose `X[i-1][j]` term is `X[-1][j]` for
i=0 -- i.e. it read the *last* (still-uninitialized, not-yet-simulated) row
of the array as if it were the previous step, purely because of Python/NumPy
negative-index wraparound. burn_in discards the first 100 steps regardless,
so this never reached the retained data, but it is not something to
reproduce on purpose: this version seeds two proper initial rows and starts
the recurrence at i=1, so every step reads a real, already-simulated
predecessor. Run directly (`python -m data.henon_gen`) to (re)write the .npz.
"""

from __future__ import annotations

import os

import numpy as np

from timecaust.data.gen_utils import save_dataset
from data.graphs import henon_graph

D = 10
T = 10000
BURN_IN = 100
A_PARAM = 1.4
B_PARAM = 0.3
SEED = 42


def generate_henon(D=10, T=10000, burn_in=100, a=1.4, b=0.3, seed=42):
    rng = np.random.default_rng(seed)
    A = henon_graph(D)

    total = T + burn_in
    X = rng.random((total + 1, D))  # X[0] is the only "invented" initial row
    for i in range(1, total):
        x_next = np.zeros(D)
        for j in range(D):
            if j != 0:
                x_next[j] = a - (b * X[i][j - 1] + (1 - b) * X[i][j]) ** 2 + b * X[i - 1][j]
            else:
                x_next[j] = a - X[i][j] ** 2 + b * X[i - 1][j]
        X[i + 1] = x_next

    X_raw = X[-T:]  # drop the invented row + burn-in; keep exactly T rows
    meta = {"system": "henon", "D": D, "T": T, "burn_in": burn_in, "a": a, "b": b, "seed": seed}
    return X_raw, A, meta


if __name__ == "__main__":
    X_raw, A, meta = generate_henon(D=D, T=T, burn_in=BURN_IN, a=A_PARAM, b=B_PARAM, seed=SEED)
    print(f"[henon] range=[{X_raw.min():.3f}, {X_raw.max():.3f}]  shape={X_raw.shape}", flush=True)
    out_dir = os.path.join(os.path.dirname(__file__), "henon")
    save_dataset(out_dir, f"henon_{T}_{D}.npz", X_raw, A, meta)
