from __future__ import annotations
from typing import Tuple
import numpy as np
_H4 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float)

def hadamard_transform(P: np.ndarray) -> np.ndarray:
    Pbar = P.copy()
    for axis in range(P.ndim):
        Pbar = np.tensordot(_H4, Pbar, axes=([1], [axis]))
        Pbar = np.moveaxis(Pbar, 0, axis)
    return Pbar

def inverse_hadamard_transform(F: np.ndarray) -> np.ndarray:
    P = F.copy()
    for axis in range(F.ndim):
        P = np.tensordot(_H4, P, axes=([1], [axis]))
        P = np.moveaxis(P, 0, axis)
    return P / 4 ** F.ndim