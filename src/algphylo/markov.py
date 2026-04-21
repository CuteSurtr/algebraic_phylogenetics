from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.linalg import expm

def jc_rate_matrix(kappa: int=4) -> np.ndarray:
    Q = np.full((kappa, kappa), 1.0 / (kappa - 1))
    np.fill_diagonal(Q, -1.0)
    pi = np.full(kappa, 1.0 / kappa)
    mu = -np.sum(pi * np.diag(Q))
    Q /= mu
    return Q

def jc_transition(t: float, kappa: int=4) -> np.ndarray:
    rate = kappa / (kappa - 1)
    e = np.exp(-rate * t)
    same = 1.0 / kappa + (1 - 1.0 / kappa) * e
    diff = 1.0 / kappa - 1.0 / kappa * e
    M = np.full((kappa, kappa), diff)
    np.fill_diagonal(M, same)
    return M

def _is_transition(i: int, j: int) -> bool:
    return (i, j) in ((0, 2), (2, 0), (1, 3), (3, 1))

def k80_rate_matrix(alpha: float=1.0, beta: float=0.5) -> np.ndarray:
    Q = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            Q[i, j] = alpha if _is_transition(i, j) else beta
    for i in range(4):
        Q[i, i] = -Q[i].sum()
    pi = np.full(4, 0.25)
    mu = -np.sum(pi * np.diag(Q))
    Q /= mu
    return Q

def k80_transition(t: float, alpha: float=1.0, beta: float=0.5) -> np.ndarray:
    return expm(k80_rate_matrix(alpha, beta) * t)

def hky_rate_matrix(pi: np.ndarray, kappa: float=2.0) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    assert pi.shape == (4,) and np.isclose(pi.sum(), 1.0)
    Q = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            rate = kappa if _is_transition(i, j) else 1.0
            Q[i, j] = rate * pi[j]
    for i in range(4):
        Q[i, i] = -Q[i].sum()
    mu = -np.sum(pi * np.diag(Q))
    Q /= mu
    return Q

def hky_transition(t: float, pi: np.ndarray, kappa: float=2.0) -> np.ndarray:
    return expm(hky_rate_matrix(pi, kappa) * t)

@dataclass
class GMMMatrix:
    M: np.ndarray

    def __post_init__(self) -> None:
        self.M = np.asarray(self.M, dtype=float)
        assert self.M.ndim == 2 and self.M.shape[0] == self.M.shape[1]
        assert np.all(self.M >= -1e-12)
        assert np.allclose(self.M.sum(axis=1), 1.0, atol=1e-08), f'GMM rows must sum to 1; got {self.M.sum(axis=1)}'