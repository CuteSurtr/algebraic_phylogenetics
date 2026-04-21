from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple
import numpy as np
from .flatten import flatten, svd_score
from .simulate import empirical_joint_tensor

@dataclass
class QuartetResult:
    taxa: Tuple[str, str, str, str]
    scores: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float]

    @property
    def best(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return min(self.scores, key=lambda k: self.scores[k])

    def summary(self) -> str:
        lines = [f'quartet {self.taxa}:']
        for split, s in sorted(self.scores.items(), key=lambda kv: kv[1]):
            a, b = split
            lines.append(f"  {'+'.join(a):>15} | {'+'.join(b):<15}  SVD score = {s:.4f}")
        return '\n'.join(lines)

def quartet_scores(alignment: Dict[str, np.ndarray], taxa: Sequence[str], kappa: int=4) -> QuartetResult:
    assert len(taxa) == 4
    a, b, c, d = taxa
    P = empirical_joint_tensor(alignment, taxon_order=[a, b, c, d], kappa=kappa)
    splits = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    scores = {}
    for A_names, B_names in splits:
        A_idx = [taxa.index(x) for x in A_names]
        B_idx = [taxa.index(x) for x in B_names]
        F = flatten(P, A_idx, B_idx)
        scores[A_names, B_names] = svd_score(F, rank=kappa)
    return QuartetResult(taxa=tuple(taxa), scores=scores)

def all_quartets(alignment: Dict[str, np.ndarray], kappa: int=4) -> List[QuartetResult]:
    taxa = list(alignment.keys())
    out = []
    for quad in combinations(taxa, 4):
        out.append(quartet_scores(alignment, quad, kappa=kappa))
    return out