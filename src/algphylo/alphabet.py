from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class Alphabet:
    symbols: str
    name: str = ''

    @property
    def size(self) -> int:
        return len(self.symbols)

    @property
    def index_of(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.symbols)}

    def encode(self, seq: str, skip_ambiguous: bool=True):
        import numpy as np
        idx = self.index_of
        out = []
        for c in seq.upper():
            if c in idx:
                out.append(idx[c])
            elif not skip_ambiguous:
                out.append(-1)
        return np.asarray(out, dtype=int)
DNA = Alphabet(symbols='ACGT', name='DNA')
BINARY = Alphabet(symbols='01', name='2-state')