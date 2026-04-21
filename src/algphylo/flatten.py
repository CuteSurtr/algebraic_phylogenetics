from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np

def flatten(P: np.ndarray, A: Sequence[int], B: Sequence[int]) -> np.ndarray:
    n = P.ndim
    A = list(A)
    B = list(B)
    assert set(A) | set(B) == set(range(n))
    assert set(A) & set(B) == set()
    kappa = P.shape[0]
    perm = A + B
    P2 = np.transpose(P, perm)
    rA = kappa ** len(A)
    rB = kappa ** len(B)
    return P2.reshape(rA, rB)

def svd_score(F: np.ndarray, rank: int) -> float:
    sv = np.linalg.svd(F, compute_uv=False)
    return float(np.sum(sv[rank:] ** 2))

def approximate_rank(F: np.ndarray, tol: float=1e-06) -> int:
    sv = np.linalg.svd(F, compute_uv=False)
    if sv.size == 0:
        return 0
    cutoff = tol * sv[0]
    return int((sv > cutoff).sum())