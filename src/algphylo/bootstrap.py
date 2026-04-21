from __future__ import annotations
from collections import Counter
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from .svdquartets import quartet_scores

def bootstrap_quartet(alignment: Dict[str, np.ndarray], taxa: Sequence[str], n_bootstraps: int=200, kappa: int=4, rng: Optional[np.random.Generator]=None) -> Dict[Tuple, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    taxa = list(taxa)
    L = len(next(iter(alignment.values())))
    counts: Counter = Counter()
    for _ in range(n_bootstraps):
        cols = rng.integers(0, L, size=L)
        sub = {t: alignment[t][cols] for t in taxa}
        qr = quartet_scores(sub, taxa, kappa=kappa)
        counts[qr.best] += 1
    return {split: c / n_bootstraps for split, c in counts.items()}