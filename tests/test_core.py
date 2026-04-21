import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from algphylo import DNA, GMMMatrix, TreeModel, alignment_log_likelihood, empirical_joint_tensor, flatten, hadamard_transform, hky_transition, inverse_hadamard_transform, jc_rate_matrix, jc_transition, k80_transition, off_support_mass, parse_newick, quartet_scores, rank_invariant_residual, simulate_alignment, site_likelihood, svd_score

def test_jc_transition_rows_sum_to_one():
    M = jc_transition(0.5)
    assert np.allclose(M.sum(axis=1), 1.0)
    assert np.allclose(M, M.T)

def test_jc_closed_form_matches_expm():
    from scipy.linalg import expm
    Q = jc_rate_matrix(4)
    for t in [0.1, 0.5, 1.0, 3.0]:
        M_expm = expm(Q * t)
        M_closed = jc_transition(t)
        assert np.allclose(M_expm, M_closed, atol=1e-10)

def test_k80_distinguishes_transitions_from_transversions():
    M = k80_transition(0.5, alpha=2.0, beta=0.5)
    assert M[0, 2] > M[0, 1]
    assert M[0, 2] > M[0, 3]
    assert M[1, 3] > M[1, 0]

def test_hky_transition_rows_stochastic():
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    M = hky_transition(0.5, pi=pi, kappa=2.0)
    assert np.allclose(M.sum(axis=1), 1.0)

def _primate_subtree():
    return parse_newick('((a:0.1,b:0.1):0.2,(c:0.1,d:0.1):0.1);')

def _jc_model(tree, kappa=4):
    return TreeModel(tree=tree, root_distribution=np.full(kappa, 1.0 / kappa), transition_fn=lambda t: jc_transition(t, kappa), kappa=kappa)

def test_joint_tensor_sums_to_one():
    model = _jc_model(_primate_subtree())
    P = model.joint_tensor()
    assert P.shape == (4, 4, 4, 4)
    assert math.isclose(P.sum(), 1.0, rel_tol=1e-10)
    assert (P >= 0).all()

def test_felsenstein_matches_tensor_entry():
    model = _jc_model(_primate_subtree())
    P = model.joint_tensor()
    pattern = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    lik_pruning = site_likelihood(model, pattern)
    assert math.isclose(P[0, 1, 2, 3], lik_pruning, rel_tol=1e-10)

def test_felsenstein_matches_tensor_entry_random():
    rng = np.random.default_rng(0)
    model = _jc_model(_primate_subtree())
    P = model.joint_tensor()
    taxa = model.tree.taxa
    for _ in range(20):
        idx = tuple((int(rng.integers(0, 4)) for _ in taxa))
        pattern = {t: s for t, s in zip(taxa, idx)}
        assert math.isclose(P[idx], site_likelihood(model, pattern), rel_tol=1e-09)

def test_alignment_log_likelihood_is_finite():
    model = _jc_model(_primate_subtree())
    data = simulate_alignment(model, length=100, rng=np.random.default_rng(0))
    L = alignment_log_likelihood(model, data)
    assert np.isfinite(L)
    assert L < 0

def test_empirical_tensor_sums_to_one():
    model = _jc_model(_primate_subtree())
    data = simulate_alignment(model, length=1000, rng=np.random.default_rng(1))
    P = empirical_joint_tensor(data, ['a', 'b', 'c', 'd'])
    assert math.isclose(P.sum(), 1.0)

def test_flatten_shape_and_reconstruction():
    P = np.arange(256).reshape(4, 4, 4, 4).astype(float) / 10000
    F = flatten(P, [0, 1], [2, 3])
    assert F.shape == (16, 16)
    back = F.reshape(4, 4, 4, 4)
    assert np.allclose(back, P)

def test_svd_score_nonnegative():
    rng = np.random.default_rng(2)
    P = rng.random((4, 4, 4, 4))
    P /= P.sum()
    F = flatten(P, [0, 1], [2, 3])
    assert svd_score(F, rank=4) >= 0

def test_rank_theorem_edge_split_has_low_residual():
    rng = np.random.default_rng(42)
    model = _jc_model(_primate_subtree())
    P = model.joint_tensor()
    edge_residual = rank_invariant_residual(P, [0, 1], [2, 3], kappa=4)
    non_edge_residual = rank_invariant_residual(P, [0, 2], [1, 3], kappa=4)
    assert edge_residual < 1e-10
    assert non_edge_residual >= 0

def test_rank_theorem_edge_split_has_low_residual_noisy():
    rng = np.random.default_rng(7)
    model = _jc_model(_primate_subtree())
    model = TreeModel(tree=parse_newick('((a:0.3,b:0.3):0.5,(c:0.3,d:0.3):0.2);'), root_distribution=model.root_distribution, transition_fn=model.transition_fn, kappa=model.kappa)
    data = simulate_alignment(model, length=20000, rng=rng)
    P = empirical_joint_tensor(data, ['a', 'b', 'c', 'd'])
    r_ab_cd = rank_invariant_residual(P, [0, 1], [2, 3])
    r_ac_bd = rank_invariant_residual(P, [0, 2], [1, 3])
    r_ad_bc = rank_invariant_residual(P, [0, 3], [1, 2])
    assert r_ab_cd < r_ac_bd
    assert r_ab_cd < r_ad_bc

def test_svdquartets_recovers_topology_on_simulated_data():
    rng = np.random.default_rng(11)
    tree = parse_newick('((a:0.3,b:0.3):0.5,(c:0.3,d:0.3):0.2);')
    model = TreeModel(tree=tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    data = simulate_alignment(model, length=10000, rng=rng)
    result = quartet_scores(data, ['a', 'b', 'c', 'd'])
    best = result.best
    assert set(best) == {('a', 'b'), ('c', 'd')}

def test_hadamard_transform_is_involutive_up_to_scale():
    rng = np.random.default_rng(3)
    P = rng.random((4, 4))
    P /= P.sum()
    F = hadamard_transform(P)
    P_back = inverse_hadamard_transform(F)
    assert np.allclose(P_back, P, atol=1e-12)

def test_global_parity_support_for_group_based_model():
    from algphylo import global_parity_off_support_mass
    tree = parse_newick('((a:0.3,b:0.3):0.5,(c:0.3,d:0.3):0.2);')
    model = TreeModel(tree=tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    P = model.joint_tensor(leaf_order=['a', 'b', 'c', 'd'])
    off = global_parity_off_support_mass(P)
    assert off < 1e-10

def test_off_support_mass_small_on_correct_tree():
    tree = parse_newick('((a:0.3,b:0.3):0.5,(c:0.3,d:0.3):0.2);')
    model = TreeModel(tree=tree, root_distribution=np.full(4, 0.25), transition_fn=lambda t: jc_transition(t), kappa=4)
    P = model.joint_tensor(leaf_order=['a', 'b', 'c', 'd'])
    off = off_support_mass(P, [0, 1], [2, 3])
    assert off < 1e-10