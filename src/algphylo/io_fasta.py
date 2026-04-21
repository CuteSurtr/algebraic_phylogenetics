from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
from .alphabet import Alphabet, DNA

@dataclass
class FastaRecord:
    header: str
    sequence: str

    @property
    def name(self) -> str:
        return self.header.split()[0] if self.header else ''

def read_fasta(path) -> List[FastaRecord]:
    text = Path(path).read_text()
    recs: List[FastaRecord] = []
    header = None
    parts: List[str] = []
    for line in text.splitlines():
        if line.startswith('>'):
            if header is not None:
                recs.append(FastaRecord(header, ''.join(parts).upper()))
            header = line[1:].strip()
            parts = []
        else:
            parts.append(line.strip())
    if header is not None:
        recs.append(FastaRecord(header, ''.join(parts).upper()))
    return recs

def align_by_truncation(records: List[FastaRecord]) -> Dict[str, str]:
    L = min((len(r.sequence) for r in records))
    return {r.name: r.sequence[:L] for r in records}

def encode_alignment(alignment: Dict[str, str], alphabet: Alphabet=DNA, skip_columns_with_gaps: bool=True) -> Dict[str, np.ndarray]:
    idx = alphabet.index_of
    taxa = list(alignment.keys())
    L = len(alignment[taxa[0]])
    keep_cols = []
    for c in range(L):
        column = [alignment[t][c] if c < len(alignment[t]) else 'N' for t in taxa]
        if skip_columns_with_gaps and any((ch not in idx for ch in column)):
            continue
        keep_cols.append(c)
    out: Dict[str, np.ndarray] = {}
    for t in taxa:
        s = alignment[t]
        out[t] = np.array([idx[s[c]] for c in keep_cols], dtype=int)
    return out