from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np

def plot_singular_value_spectrum(F_by_split: Dict[str, np.ndarray], kappa: int=4, ax: Optional[plt.Axes]=None, title: str='Flattening singular spectrum'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, F in F_by_split.items():
        sv = np.linalg.svd(F, compute_uv=False)
        ax.semilogy(np.arange(1, len(sv) + 1), sv + 1e-16, 'o-', label=label, markersize=4)
    ax.axvline(kappa + 0.5, color='gray', linestyle='--', alpha=0.6, label=f'rank κ={kappa}')
    ax.set_xlabel('singular value index')
    ax.set_ylabel('σ (log)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax

def plot_quartet_scores_bar(scores: Dict, ax: Optional[plt.Axes]=None, title: str='SVDQuartets scores (lower = better)'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = []
    values = []
    for split, score in sorted(scores.items(), key=lambda kv: kv[1]):
        a, b = split
        labels.append(f"{'+'.join(a)} | {'+'.join(b)}")
        values.append(score)
    colors = ['tab:green' if i == 0 else 'tab:gray' for i in range(len(values))]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('SVD score')
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
    return ax

def plot_flattening_heatmap(F: np.ndarray, ax: Optional[plt.Axes]=None, title: str='Flattening (log scale)'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    M = np.log10(F + 1e-16)
    im = ax.imshow(M, cmap='magma')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, label='log10 P')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def plot_fourier_spectrum(Phat: np.ndarray, ax: Optional[plt.Axes]=None, title: str='Hadamard-transformed tensor'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    flat = np.abs(Phat).flatten()
    ax.bar(np.arange(len(flat)), flat, color='tab:purple', width=1.0)
    ax.set_xlabel('Fourier-coordinate index')
    ax.set_ylabel('|P̂|')
    ax.set_title(title)
    ax.set_yscale('log')
    return ax