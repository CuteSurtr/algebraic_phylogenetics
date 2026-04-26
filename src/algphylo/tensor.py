from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
from .tree import Node, Tree
TransitionFn = Callable[[float], np.ndarray]
'A function t ? M(t) returning a kappaxkappa row-stochastic matrix.'

@dataclass
class TreeModel:
    tree: Tree
    root_distribution: np.ndarray
    transition_fn: TransitionFn
    kappa: int = 4

    def joint_tensor(self, leaf_order: Optional[Sequence[str]]=None) -> np.ndarray:
        if leaf_order is None:
            leaf_order = list(self.tree.taxa)
        leaf_order = list(leaf_order)
        axis_of = {name: i for i, name in enumerate(leaf_order)}
        n = len(leaf_order)

        def build(v: Node) -> tuple[np.ndarray, List[int]]:
            if v.is_leaf:
                T = np.eye(self.kappa)
                return (T, [axis_of[v.name]])
            parts: List[np.ndarray] = []
            axes_lists: List[List[int]] = []
            for c in v.children:
                T_c, axes_c = build(c)
                M = self.transition_fn(c.length)
                T_c = np.tensordot(T_c, M.T, axes=([T_c.ndim - 1], [0]))
                parts.append(T_c)
                axes_lists.append(axes_c)
            combined = parts[0]
            combined_axes = list(axes_lists[0])
            for i in range(1, len(parts)):
                T_i = parts[i]
                axes_i = axes_lists[i]
                combined = _outer_broadcast_last(combined, T_i)
                combined_axes = combined_axes + axes_i
            return (combined, combined_axes)
        T_root, axes_list = build(self.tree.root)
        P = np.tensordot(T_root, self.root_distribution, axes=([T_root.ndim - 1], [0]))
        perm = [axes_list.index(i) for i in range(n)]
        P = np.transpose(P, perm)
        return P

def _outer_broadcast_last(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape[-1] == B.shape[-1]
    kappa = A.shape[-1]
    A_exp = A.reshape(A.shape[:-1] + (1,) * (B.ndim - 1) + (kappa,))
    B_exp = B.reshape((1,) * (A.ndim - 1) + B.shape[:-1] + (kappa,))
    return A_exp * B_exp