from __future__ import annotations
from typing import Dict, List, Optional, Tuple
try:
    from Bio import Align
except ImportError as e:
    raise ImportError('biopython required; `pip install biopython`') from e

def pairwise_align(ref: str, query: str, match: float=2.0, mismatch: float=-1.0, open_gap: float=-5.0, extend_gap: float=-1.0) -> Tuple[str, str, float]:
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap
    alignments = aligner.align(ref, query)
    best = alignments[0]
    return (str(best[0]), str(best[1]), float(best.score))

def star_alignment(sequences: Dict[str, str], reference_name: Optional[str]=None) -> Dict[str, str]:
    names = list(sequences.keys())
    if reference_name is None:
        reference_name = max(names, key=lambda n: len(sequences[n]))
    ref = sequences[reference_name]
    pairwise: Dict[str, Tuple[str, str]] = {}
    for name in names:
        if name == reference_name:
            continue
        r_aln, q_aln, _ = pairwise_align(ref, sequences[name])
        pairwise[name] = (r_aln, q_aln)
    master_ref_cols: List[int] = []
    first = next(iter(pairwise.values()))
    r_aln_first = first[0]
    master = list(r_aln_first)
    ref_pos_at_col = []
    p = 0
    for ch in master:
        if ch == '-':
            ref_pos_at_col.append(-1)
        else:
            ref_pos_at_col.append(p)
            p += 1
    for name, (r_aln, q_aln) in list(pairwise.items())[1:]:
        merged_master: List[str] = []
        merged_positions: List[int] = []
        i = 0
        j = 0
        while i < len(master) or j < len(r_aln):
            m_char = master[i] if i < len(master) else None
            o_char = r_aln[j] if j < len(r_aln) else None
            if m_char is None:
                merged_master.append(o_char)
                merged_positions.append(_refpos(r_aln, j))
                j += 1
                continue
            if o_char is None:
                merged_master.append(m_char)
                merged_positions.append(ref_pos_at_col[i])
                i += 1
                continue
            if m_char == '-' and o_char != '-':
                merged_master.append('-')
                merged_positions.append(-1)
                i += 1
            elif o_char == '-' and m_char != '-':
                merged_master.append('-')
                merged_positions.append(-1)
                j += 1
            else:
                merged_master.append(m_char)
                merged_positions.append(ref_pos_at_col[i])
                i += 1
                j += 1
        master = merged_master
        ref_pos_at_col = merged_positions
    out: Dict[str, str] = {}
    out[reference_name] = ''.join((ref[ref_pos_at_col[c]] if ref_pos_at_col[c] >= 0 else '-' for c in range(len(master))))
    for name, (r_aln, q_aln) in pairwise.items():
        q_by_refpos: Dict[int, str] = {}
        rp = 0
        gap_after_rp: Dict[int, List[str]] = {-1: []}
        cur_ref_pos = -1
        for r_ch, q_ch in zip(r_aln, q_aln):
            if r_ch != '-':
                cur_ref_pos = rp
                q_by_refpos[rp] = q_ch
                rp += 1
            else:
                gap_after_rp.setdefault(cur_ref_pos, []).append(q_ch)
        emitted = []
        leading = list(gap_after_rp.get(-1, []))
        for c in range(len(master)):
            if ref_pos_at_col[c] < 0:
                if leading:
                    emitted.append(leading.pop(0))
                else:
                    emitted.append('-')
            else:
                emitted.append(q_by_refpos.get(ref_pos_at_col[c], '-'))
        out[name] = ''.join(emitted)
    L = len(master)
    for k in out:
        if len(out[k]) < L:
            out[k] += '-' * (L - len(out[k]))
        elif len(out[k]) > L:
            out[k] = out[k][:L]
    return out

def _refpos(r_aln: str, j: int) -> int:
    return sum((1 for c in r_aln[:j] if c != '-')) - (0 if j < len(r_aln) and r_aln[j] != '-' else 1)