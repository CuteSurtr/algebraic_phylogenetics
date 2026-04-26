from __future__ import annotations
from pathlib import Path
import numpy as np
from . import TreeModel, align_by_truncation, alignment_log_likelihood, all_quartets, empirical_joint_tensor, encode_alignment, flatten, global_parity_off_support_mass, hadamard_transform, jc_transition, k80_transition, parse_newick, quartet_scores, rank_invariant_residual, read_fasta, simulate_alignment, svd_score
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import viz
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
HERE = Path(__file__).resolve().parents[2]
DATA = HERE / 'data'
RESULTS = HERE / 'results'

def section(title: str) -> None:
    print('\n' + '=' * 74)
    print(title)
    print('=' * 74)

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    section('1. Markov models + joint tensor sanity')
    tree = parse_newick('((a:0.3,b:0.3):0.5,(c:0.3,d:0.3):0.2);')
    model = TreeModel(tree=tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    P = model.joint_tensor()
    print(f'  tree: {tree.taxa}     joint tensor shape {P.shape}   sum={P.sum():.6f}')
    print(f'  JC model, root = uniform')
    section('2. Rank theorem validation on exact vs empirical tensors')
    print(f"  {'split':<15}{'exact residual':>18}{'empirical (20k sites)':>24}")
    data = simulate_alignment(model, length=20000, rng=rng)
    P_emp = empirical_joint_tensor(data, tree.taxa)
    for split in [([0, 1], [2, 3], '{a,b}|{c,d}'), ([0, 2], [1, 3], '{a,c}|{b,d}'), ([0, 3], [1, 2], '{a,d}|{b,c}')]:
        A, B, label = split
        r_exact = rank_invariant_residual(P, A, B)
        r_emp = rank_invariant_residual(P_emp, A, B)
        print(f'  {label:<15}{r_exact:>18.2e}{r_emp:>24.2e}')
    print('  ^ exact tensor: only the true edge split has residual ~ 0')
    print('  ^ 20k-site empirical: true edge still has smallest residual')
    section('3. SVDQuartets on simulated data (topology recovery)')
    qr = quartet_scores(data, tree.taxa)
    print(qr.summary())
    best = qr.best
    print(f"  -> winner: {'+'.join(best[0])} | {'+'.join(best[1])}")
    assert set(best) == {('a', 'b'), ('c', 'd')}, 'topology recovery failed'
    section('4. Global Fourier parity support check (group-based sanity)')
    off = global_parity_off_support_mass(P)
    print(f'  exact JC tensor: off-parity-support mass = {off:.2e}  (should be ~0)')
    off_emp = global_parity_off_support_mass(P_emp)
    print(f'  20k-site empirical:                    = {off_emp:.2e}')
    section('5. Real data: 5-primate mitochondrial quartet analysis')
    primate_fasta = DATA / 'primate_cox1.fasta'
    if not primate_fasta.exists():
        print('  (primate COI data not available, skipping)')
    else:
        print('  using cytochrome oxidase I (COI) -- a protein-coding, well-conserved gene')
        records = read_fasta(primate_fasta)
        short_name = {'NC_012920.1:5904-7445': 'Human', 'NC_001643.1:5904-7445': 'Chimp', 'NC_001644.1:5904-7445': 'Bonobo', 'NC_001645.1:5904-7445': 'Gorilla', 'NC_002083.1:5904-7445': 'Orangutan'}
        for r in records:
            r.header = short_name.get(r.name, r.name)
        aligned = align_by_truncation(records)
        aligned = {short_name.get(k, k): v for k, v in aligned.items()}
        enc = encode_alignment(aligned)
        L = len(next(iter(enc.values())))
        print(f'  loaded {len(enc)} primate species, {L} columns after gap removal')
        for name, seq in enc.items():
            print(f"    {name:<10} length={len(seq)}   sample={''.join(('ACGT'[s] for s in seq[:30]))}...")
        results = all_quartets(enc)
        print(f'\n  Scored {len(results)} quartets; best splits:')
        for qr in results:
            best = qr.best
            label = f"{'+'.join(best[0])} | {'+'.join(best[1])}"
            print(f'    taxa={qr.taxa}  ->  best split: {label}')
        for qr in results:
            if set(qr.taxa) == {'Human', 'Chimp', 'Bonobo', 'Gorilla'}:
                best = qr.best
                biology_expected = {('Bonobo', 'Chimp'), ('Gorilla', 'Human')}
                got = {tuple(sorted(best[0])), tuple(sorted(best[1]))}
                if got == biology_expected:
                    print('  [ok] recovered biologically correct chimp-bonobo sister grouping')
                else:
                    print(f'  [warn] expected {biology_expected}, got {got}')
    section('6. Quartet puzzling -> n-taxon tree on primate data')
    try:
        from . import build_tree_via_svdquartets
        from .external.dendropy_bridge import rf_distance, same_topology
        newick, split_support = build_tree_via_svdquartets(enc, n_runs=50, seed=0)
        print(f'  puzzled tree: {newick}')
        print(f'  distinct splits across runs:')
        for split, count in sorted(split_support.items(), key=lambda kv: -kv[1])[:6]:
            print(f'    {split:<50}  support={count}/50')
        reference = '(((Chimp,Bonobo),(Human,Gorilla)),Orangutan);'
        d = rf_distance(newick, reference)
        print(f'\n  reference topology: {reference}')
        print(f'  Robinson-Foulds distance to reference: {d}')
        if d == 0:
            print('  [ok] exact topology match')
        else:
            print(f'  (partial match; differences are in internal-edge rootings)')
    except ImportError as e:
        print(f'  (skipped: {e})')
    section('7. Bootstrap support on a challenging quartet')
    from .bootstrap import bootstrap_quartet
    model_b = TreeModel(tree=parse_newick('((a:0.2,b:0.2):0.2,(c:0.2,d:0.2):0.1);'), root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    data_b = simulate_alignment(model_b, length=500, rng=np.random.default_rng(42))
    support = bootstrap_quartet(data_b, ['a', 'b', 'c', 'd'], n_bootstraps=200, rng=np.random.default_rng(1))
    print('  Bootstrap quartet support (500 sites, 200 replicates):')
    for split, pct in sorted(support.items(), key=lambda kv: -kv[1]):
        a_, b_ = split
        print(f"    {'+'.join(a_):>6} | {'+'.join(b_):<6}   {pct:.1%}")
    section('8. Symbolic invariants -- Allman-Rhodes 3x3 minors + JC binomials')
    from .invariants_symbolic import evaluate_fourier_binomial, evaluate_on_tensor, flatten_symbolic, jc_fourier_support_4taxon, symbolic_tensor, three_by_three_minors
    Psym, syms = symbolic_tensor(n_taxa=4, kappa=4)
    F_sym = flatten_symbolic(Psym, [0, 1], [2, 3])
    minors = three_by_three_minors(F_sym, limit=3)
    print(f'  Symbolic flattening F_{{a,b|c,d}} shape: {F_sym.shape}')
    print(f'  Computed {len(minors)} sample 3x3 minors; first has {minors[0].count_ops()} operations.')
    P_exact = TreeModel(tree=parse_newick('((a:0.2,b:0.2):0.3,(c:0.2,d:0.2):0.1);'), root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4).joint_tensor(leaf_order=['a', 'b', 'c', 'd'])
    for i, m in enumerate(minors):
        v = evaluate_on_tensor(m, syms, P_exact)
        print(f'    minor #{i}: evaluated on exact JC tensor = {v:.2e}')
    binomials = jc_fourier_support_4taxon(split_A=(0, 1))
    print(f'\n  JC-consistent Fourier binomials for split (a,b)|(c,d): {len(binomials)}')
    from .fourier import hadamard_transform
    Phat = hadamard_transform(P_exact)
    max_v = max((abs(evaluate_fourier_binomial(Phat, a, b)) for a, b in binomials))
    print(f'  Max absolute value on exact JC tensor: {max_v:.2e}  (all ~ 0 => invariants validated)')
    section('9. Branch-length MLE on simulated data')
    from .mle import fit_branch_lengths
    true_tree = parse_newick('((a:0.25,b:0.25):0.3,(c:0.25,d:0.25):0.15);')
    true_model = TreeModel(tree=true_tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    data_mle = simulate_alignment(true_model, length=2000, rng=np.random.default_rng(5))
    fit_tree = parse_newick('((a:0.1,b:0.1):0.1,(c:0.1,d:0.1):0.1);')
    fit_lengths = fit_branch_lengths(fit_tree, transition_fn_of=lambda t: jc_transition(t), root_distribution=np.full(4, 0.25), alignment=data_mle, kappa=4)
    true_values = [0.25, 0.25, 0.3, 0.25, 0.25, 0.15]
    print('  Fit branch lengths (MLE, JC, 2000 sites):')
    labels = ['a', 'b', '(ab)', 'c', 'd', '(cd)']
    for lab, tv, fv in zip(labels, true_values, fit_lengths):
        print(f'    {lab:<6}  true={tv:.3f}   fit={fv:.3f}')
    if HAVE_MPL:
        section('10. Figures -> results/')
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        F_by = {}
        for A, B, label in [([0, 1], [2, 3], '{a,b}|{c,d} (true)'), ([0, 2], [1, 3], '{a,c}|{b,d}'), ([0, 3], [1, 2], '{a,d}|{b,c}')]:
            F_by[label] = flatten(P_emp, A, B)
        viz.plot_singular_value_spectrum(F_by, kappa=4, ax=axs[0][0], title='Flattening singular values (simulated 20k)')
        viz.plot_quartet_scores_bar(qr.scores, ax=axs[0][1], title='SVDQuartets on simulated JC data')
        viz.plot_flattening_heatmap(F_by['{a,b}|{c,d} (true)'], ax=axs[1][0], title='True-edge flattening |F_{a,b|c,d}|')
        Phat = hadamard_transform(P_emp)
        viz.plot_fourier_spectrum(Phat, ax=axs[1][1], title='Hadamard transform of empirical P?')
        fig.tight_layout()
        fig.savefig(RESULTS / 'algphylo_demo.png', dpi=140, bbox_inches='tight')
        print(f"  wrote {RESULTS / 'algphylo_demo.png'}")
if __name__ == '__main__':
    main()