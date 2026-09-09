from __future__ import annotations

import numpy as np


def generate_synthetic_from_gc(graph: np.ndarray, n_steps: int, n_dim: int, burn_in: int, a: float = 1.4, b: float = 0.2, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parents = {d: list(np.where(graph[d, :] != 0)[0]) for d in range(graph.shape[0])}

    total = n_steps + burn_in
    X = rng.random((total, n_dim))
    for t in range(total - 1):
        x_next = np.zeros_like(X[t])
        for d in range(n_dim):
            if len(parents[d]) == 1:
                x_next[d] = a - X[t, d] ** 2 + b * X[t - 1, d]
            else:
                temp = sum(b * X[t, p] for p in parents[d] if p != d)
                temp /= (len(parents[d]) - 1)
                x_next[d] = a - (temp + (1 - b) * X[t, d]) ** 2 + b * X[t - 1, d]
        X[t + 1] = x_next

    return X[burn_in:]
