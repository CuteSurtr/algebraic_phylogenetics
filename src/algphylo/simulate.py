from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from .tensor import TreeModel
from .tree import Node

def simulate_alignment(model: TreeModel, length: int, rng: Optional[np.random.Generator]=None) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    kappa = model.kappa
    taxa = list(model.tree.taxa)
    out = {t: np.empty(length, dtype=int) for t in taxa}
    for s in range(length):
        root_state = int(rng.choice(kappa, p=model.root_distribution))
        state_of: Dict[int, int] = {id(model.tree.root): root_state}

        def walk(v: Node) -> None:
            for c in v.children:
                parent_s = state_of[id(v)]
                M = model.transition_fn(c.length)
                child_s = int(rng.choice(kappa, p=M[parent_s]))
                state_of[id(c)] = child_s
                walk(c)
        walk(model.tree.root)
        for n in model.tree.postorder():
            if n.is_leaf and n.name:
                out[n.name][s] = state_of[id(n)]
    return out

def empirical_joint_tensor(alignment: Dict[str, np.ndarray], taxon_order, kappa: int=4) -> np.ndarray:
    taxon_order = list(taxon_order)
    n = len(taxon_order)
    sizes = (kappa,) * n
    P = np.zeros(sizes)
    length = len(next(iter(alignment.values())))
    for s in range(length):
        idx = tuple((int(alignment[t][s]) for t in taxon_order))
        P[idx] += 1
    P /= P.sum()
    return P