"""
data/scm_sim.py
Shared synthetic-SCM simulator for the ecoli and yeast datasets: a nonlinear
autoregressive system built to be consistent with a given fixed causal
graph (data.graphs.ECOLI_GRAPH / YEAST_GRAPH). Both data/ecoli_gen.py and
data/yeast_gen.py call this with only the graph swapped -- the original
data_prep.ipynb notebook instead pasted the whole function twice (once per
dataset, both named generate_synthetic_from_gc), byte-for-byte identical
except which graph was in scope.

One real bug fixed here, not just de-duplicated: the notebook version built
each node d's parent set as `np.where(gc[:, d] != 0)` -- column d of the
adjacency matrix. Every other convention in this codebase (crvae_model's
causal_graph argument, data.graphs.henon_graph/lorenz96_graph, and this same
dataset's own "modified" sibling generator in ecoli_mod_gen.py/
yeast_mod_gen.py, which correctly uses `np.where(adj[i, :] > 0)` -- row i)
treats A[i, j] == 1 as "j is a parent of i", i.e. parents live in *rows*.
Building the simulator from columns instead meant the trajectory these two
datasets originally produced was actually driven by each node's *children*
in ECOLI_GRAPH/YEAST_GRAPH, not its parents -- silently mismatched against
the "ground truth" graph the rest of the pipeline (get_parent_mask, the
attack's parent/non-parent conditions) would have paired it with. This
version uses rows, consistent with every other generator and with
crvae_model.model.cLSTM's own causal_graph convention.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_from_gc(graph: np.ndarray, n_steps: int, n_dim: int, burn_in: int, a: float = 1.4, b: float = 0.2, seed: int = 42) -> np.ndarray:
    """Nonlinear autoregressive SCM: node d with a single parent (itself)
    follows the plain Henon-map recursion; a node with more parents follows
    the same recursion driven by the mean of its parents' current values
    (excluding itself) in place of its own value's contribution."""
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
