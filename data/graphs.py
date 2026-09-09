"""
data/graphs.py
Ground-truth causal adjacency matrices shared by more than one generator.
A[i, j] == 1 means j is a parent of i (row i lists i's parents) -- the same
convention crvae_model.model.cLSTM's `causal_graph` argument expects.

ECOLI_GRAPH and YEAST_GRAPH are the fixed 10-node graphs TimeCAT_new's
utils.py hardcoded (as getCausalMatrix('ecoli') / getCausalMatrix('yeast'));
they are reused as-is by both the ecoli/yeast generators (which simulate a
synthetic SCM consistent with the graph) and the ecoliM/yeastM generators
(a modified-Henon system driven by the same graph). Keeping them here once,
instead of copy-pasted into every generator (as utils.py's getCausalMatrix
was called from three different places in the original code), is what lets
data_gen and the eventual attack's parent/non-parent masks never drift apart
-- see utils/data_utils.py's module docstring.
"""

import numpy as np


def henon_graph(D: int) -> np.ndarray:
    """Coupled-Henon chain: node i's parents are {i-1, i} (i's own history
    plus its left neighbor's), node 0's only parent is itself."""
    A = np.zeros((D, D), dtype=int)
    for i in range(D):
        A[i, i] = 1
        if i > 0:
            A[i, i - 1] = 1
    return A


def lorenz96_graph(D: int) -> np.ndarray:
    """Standard Lorenz-96 Granger structure: dx_i/dt = (x_{i+1} - x_{i-2}) *
    x_{i-1} - x_i + F depends on exactly {i-2, i-1, i, i+1} (mod D)."""
    A = np.zeros((D, D), dtype=int)
    for i in range(D):
        A[i, i] = 1
        A[i, (i + 1) % D] = 1
        A[i, (i - 1) % D] = 1
        A[i, (i - 2) % D] = 1
    return A


ECOLI_GRAPH = np.array([
    [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 1., 0., 0., 0., 0., 1.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
])

YEAST_GRAPH = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
])
