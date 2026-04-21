from __future__ import annotations
from itertools import product
from typing import List, Sequence, Tuple
import numpy as np
from .flatten import flatten
from .fourier import hadamard_transform

def rank_invariant_residual(P: np.ndarray, A: Sequence[int], B: Sequence[int], kappa: int=4) -> float:
    F = flatten(P, A, B)
    sv = np.linalg.svd(F, compute_uv=False)
    total = float((sv ** 2).sum())
    if total == 0:
        return 0.0
    tail = float((sv[kappa:] ** 2).sum())
    return tail / total

def _global_parity_mask(n: int, kappa: int=4) -> np.ndarray:
    shape = (kappa,) * n
    mask = np.zeros(shape, dtype=bool)
    for idx in product(range(kappa), repeat=n):
        bits = [(b // 2, b % 2) for b in idx]
        s = (0, 0)
        for bi in bits:
            s = (s[0] ^ bi[0], s[1] ^ bi[1])
        if s == (0, 0):
            mask[idx] = True
    return mask

def global_parity_off_support_mass(P: np.ndarray) -> float:
    Phat = hadamard_transform(P)
    mask = _global_parity_mask(P.ndim)
    idx0 = (0,) * P.ndim
    total = float((Phat ** 2).sum()) - float(Phat[idx0] ** 2)
    if total <= 0:
        return 0.0
    off = float((Phat[~mask] ** 2).sum())
    return off / total

def split_flattening_rank_residual(P: np.ndarray, A_axes: Sequence[int], B_axes: Sequence[int], kappa: int=4) -> float:
    return rank_invariant_residual(P, A_axes, B_axes, kappa=kappa)

def off_support_mass(P: np.ndarray, A_axes: Sequence[int], B_axes: Sequence[int]) -> float:
    return rank_invariant_residual(P, A_axes, B_axes, kappa=4)