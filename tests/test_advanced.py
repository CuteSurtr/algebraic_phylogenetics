import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from algphylo import QuartetCall, TreeModel, bootstrap_quartet, build_tree_via_svdquartets, fit_branch_lengths, jc_transition, parse_newick, puzzle_tree, simulate_alignment

def test_puzzle_tree_trivial_triplet():
    taxa = ['a', 'b', 'c']
    newick, splits = puzzle_tree(taxa, {}, n_runs=1)
    assert all((t in newick for t in taxa))

def test_puzzle_tree_recovers_simulated_topology():
    rng = np.random.default_rng(0)
    tree = parse_newick('(((a:0.1,b:0.1):0.1,c:0.15):0.1,(d:0.1,e:0.1):0.1);')
    model = TreeModel(tree=tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    data = simulate_alignment(model, length=20000, rng=rng)
    newick, splits = build_tree_via_svdquartets(data, n_runs=20, seed=0)
    split_keys = list(splits.keys())
    biology_split = 'a|b|c || d|e'
    assert biology_split in split_keys, f"expected split '{biology_split}' not found in {split_keys}"
    ab_split = 'a|b || c|d|e'
    assert ab_split in split_keys, f"expected split '{ab_split}' not found in {split_keys}"

def test_symbolic_tensor_shape():
    from algphylo.invariants_symbolic import symbolic_tensor
    P, syms = symbolic_tensor(n_taxa=3, kappa=4)
    assert P.shape == (4, 4, 4)
    assert len(syms) == 64

def test_three_by_three_minors_are_nonzero():
    from algphylo.invariants_symbolic import symbolic_tensor, flatten_symbolic, three_by_three_minors
    P, syms = symbolic_tensor(n_taxa=4, kappa=4)
    F = flatten_symbolic(P, [0, 1], [2, 3])
    minors = three_by_three_minors(F, limit=3)
    assert len(minors) == 3
    for m in minors:
        assert m != 0

def test_jc_fourier_binomials_vanish_on_correct_tree():
    from algphylo.fourier import hadamard_transform
    from algphylo.invariants_symbolic import evaluate_fourier_binomial, jc_fourier_support_4taxon
    tree = parse_newick('((a:0.2,b:0.2):0.3,(c:0.2,d:0.2):0.1);')
    model = TreeModel(tree=tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    P = model.joint_tensor(leaf_order=['a', 'b', 'c', 'd'])
    Phat = hadamard_transform(P)
    binomials = jc_fourier_support_4taxon(split_A=(0, 1))
    vals = [evaluate_fourier_binomial(Phat, a, b) for a, b in binomials[:20]]
    assert max((abs(v) for v in vals)) < 1e-12

def test_bootstrap_support_concentrates_on_correct_split():
    rng = np.random.default_rng(0)
    tree = parse_newick('((a:0.3,b:0.3):0.4,(c:0.3,d:0.3):0.2);')
    model = TreeModel(tree=tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    data = simulate_alignment(model, length=5000, rng=rng)
    support = bootstrap_quartet(data, ['a', 'b', 'c', 'd'], n_bootstraps=50, rng=rng)
    best = max(support.items(), key=lambda kv: kv[1])[0]
    assert {frozenset(best[0]), frozenset(best[1])} == {frozenset({'a', 'b'}), frozenset({'c', 'd'})}

def test_branch_length_mle_finds_positive_lengths():
    rng = np.random.default_rng(0)
    true_tree = parse_newick('((a:0.25,b:0.25):0.3,(c:0.25,d:0.25):0.15);')
    model = TreeModel(tree=true_tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    data = simulate_alignment(model, length=2000, rng=rng)
    fit_tree = parse_newick('((a:0.1,b:0.1):0.1,(c:0.1,d:0.1):0.1);')
    fit_lengths = fit_branch_lengths(fit_tree, transition_fn_of=lambda t: jc_transition(t), root_distribution=np.full(4, 0.25), alignment=data, kappa=4)
    assert all((0.05 < l < 1.0 for l in fit_lengths))

def test_dendropy_rf_distance():
    pytest.importorskip('dendropy')
    from algphylo.external.dendropy_bridge import rf_distance, same_topology
    a = '((A:1,B:1):1,(C:1,D:1):1);'
    b = '((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);'
    assert rf_distance(a, b) == 0
    assert same_topology(a, b)
    c = '((A,C),(B,D));'
    assert rf_distance(a, c) > 0

def test_biopython_star_alignment_identical_sequences():
    pytest.importorskip('Bio')
    from algphylo.external.biopython_align import star_alignment
    seqs = {'a': 'ACGTACGT', 'b': 'ACGTACGT', 'c': 'ACGTACGT'}
    aln = star_alignment(seqs)
    lens = {len(v) for v in aln.values()}
    assert len(lens) == 1