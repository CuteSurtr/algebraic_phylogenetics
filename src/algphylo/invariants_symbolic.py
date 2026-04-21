from __future__ import annotations
from itertools import combinations, product
from typing import Dict, List, Sequence, Tuple
import numpy as np
import sympy as sp

def symbolic_tensor(n_taxa: int, kappa: int=4) -> Tuple[np.ndarray, List[sp.Symbol]]:
    shape = (kappa,) * n_taxa
    symbols: List[sp.Symbol] = []
    P = np.empty(shape, dtype=object)
    for idx in product(range(kappa), repeat=n_taxa):
        name = 'p_' + ''.join(('ACGT'[i] if kappa == 4 else str(i) for i in idx))
        s = sp.Symbol(name)
        P[idx] = s
        symbols.append(s)
    return (P, symbols)

def flatten_symbolic(P: np.ndarray, A: Sequence[int], B: Sequence[int]) -> sp.Matrix:
    from .flatten import flatten
    F_np = flatten(P, list(A), list(B))
    return sp.Matrix(F_np)

def three_by_three_minors(F: sp.Matrix, limit: int=10) -> List[sp.Expr]:
    rows, cols = F.shape
    out: List[sp.Expr] = []
    seen = set()
    for r_set in combinations(range(rows), 3):
        for c_set in combinations(range(cols), 3):
            if len(out) >= limit:
                return out
            sub = F[list(r_set), list(c_set)]
            minor = sub.det()
            expanded = sp.expand(minor)
            key = str(expanded)
            if key in seen or expanded == 0:
                continue
            seen.add(key)
            out.append(expanded)
    return out

def evaluate_on_tensor(expr: sp.Expr, symbols: Sequence[sp.Symbol], P: np.ndarray) -> float:
    subs = {}
    for sym in symbols:
        name = str(sym)
        assert name.startswith('p_')
        label = name[2:]
        if P.shape[0] == 4:
            idx = tuple(('ACGT'.index(ch) for ch in label))
        else:
            idx = tuple((int(ch) for ch in label))
        subs[sym] = float(P[idx])
    return float(expr.xreplace(subs))

def jc_fourier_support_4taxon(split_A: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    kappa = 4
    n = 4
    A_set = set(split_A)

    def parity(idx: Tuple[int, ...]) -> Tuple[int, int]:
        s = (0, 0)
        for b in idx:
            s = (s[0] ^ b // 2, s[1] ^ b % 2)
        return s

    def signature(idx: Tuple[int, ...]):
        zero_pattern = tuple((v == 0 for v in idx))
        sA = (0, 0)
        for i in A_set:
            sA = (sA[0] ^ idx[i] // 2, sA[1] ^ idx[i] % 2)
        internal_zero = sA == (0, 0)
        return (zero_pattern, internal_zero)
    from collections import defaultdict
    groups: Dict[tuple, List[Tuple[int, ...]]] = defaultdict(list)
    for idx in product(range(kappa), repeat=n):
        if parity(idx) != (0, 0):
            continue
        groups[signature(idx)].append(idx)
    binomials: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for grp in groups.values():
        if len(grp) < 2:
            continue
        for i in range(1, len(grp)):
            binomials.append((grp[0], grp[i]))
    return binomials

def evaluate_fourier_binomial(Phat: np.ndarray, alpha: Tuple[int, ...], beta: Tuple[int, ...]) -> float:
    return float(Phat[alpha] - Phat[beta])