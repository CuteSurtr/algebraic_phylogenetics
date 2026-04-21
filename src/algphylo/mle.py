from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
from scipy.optimize import minimize
from .felsenstein import alignment_log_likelihood
from .tensor import TreeModel
from .tree import Node, Tree

def _internal_edges_in_order(tree: Tree) -> List[Node]:
    out = []
    for n in tree.postorder():
        if n.parent is not None:
            out.append(n)
    return out

def fit_branch_lengths(tree: Tree, transition_fn_of: Callable[[float], np.ndarray], root_distribution: np.ndarray, alignment: Dict[str, Sequence[int]], kappa: int=4, initial: float=0.1, method: str='L-BFGS-B') -> List[float]:
    edges = _internal_edges_in_order(tree)

    def set_lengths(x_log):
        for e, xl in zip(edges, x_log):
            e.length = float(np.exp(xl))

    def neg_logL(x_log):
        set_lengths(x_log)
        model = TreeModel(tree=tree, root_distribution=root_distribution, transition_fn=transition_fn_of, kappa=kappa)
        return -alignment_log_likelihood(model, alignment)
    x0 = np.log(np.full(len(edges), initial))
    res = minimize(neg_logL, x0, method=method, options={'maxiter': 500})
    set_lengths(res.x)
    return [np.exp(xl) for xl in res.x]