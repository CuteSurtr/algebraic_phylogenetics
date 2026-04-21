from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
from .tensor import TreeModel
from .tree import Node

def site_likelihood(model: TreeModel, site_pattern: Dict[str, int]) -> float:
    kappa = model.kappa

    def rec(v: Node) -> np.ndarray:
        if v.is_leaf:
            s = site_pattern.get(v.name, -1)
            vec = np.zeros(kappa)
            if s == -1:
                vec[:] = 1.0
            else:
                vec[s] = 1.0
            return vec
        out = np.ones(kappa)
        for c in v.children:
            child_vec = rec(c)
            M = model.transition_fn(c.length)
            out *= M @ child_vec
        return out
    root_vec = rec(model.tree.root)
    return float(model.root_distribution @ root_vec)

def alignment_log_likelihood(model: TreeModel, alignment: Dict[str, Sequence[int]]) -> float:
    taxa = list(alignment.keys())
    L = len(alignment[taxa[0]])
    assert all((len(alignment[t]) == L for t in taxa))
    total = 0.0
    for s in range(L):
        pattern = {t: int(alignment[t][s]) for t in taxa}
        p = site_likelihood(model, pattern)
        if p > 0:
            total += np.log(p)
        else:
            total += -np.inf
    return total