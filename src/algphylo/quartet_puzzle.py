from __future__ import annotations
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

@dataclass(frozen=True)
class QuartetCall:
    pair1: FrozenSet[str]
    pair2: FrozenSet[str]

    @staticmethod
    def make(a: str, b: str, c: str, d: str) -> 'QuartetCall':
        p1 = frozenset({a, b})
        p2 = frozenset({c, d})
        if min(p1) > min(p2):
            p1, p2 = (p2, p1)
        return QuartetCall(p1, p2)

    @property
    def taxa(self) -> FrozenSet[str]:
        return self.pair1 | self.pair2

    def __repr__(self) -> str:
        return f"({'+'.join(sorted(self.pair1))} | {'+'.join(sorted(self.pair2))})"

class _Node:
    __slots__ = ('label', 'neighbors')

    def __init__(self, label: Optional[str]=None):
        self.label: Optional[str] = label
        self.neighbors: List['_Node'] = []

class _UnrootedTree:

    def __init__(self):
        self.nodes: List[_Node] = []

    @classmethod
    def trivial_triplet(cls, a: str, b: str, c: str) -> '_UnrootedTree':
        t = cls()
        center = _Node()
        la, lb, lc = (_Node(a), _Node(b), _Node(c))
        for leaf in (la, lb, lc):
            center.neighbors.append(leaf)
            leaf.neighbors.append(center)
        t.nodes.extend([center, la, lb, lc])
        return t

    def edges(self) -> List[Tuple[_Node, _Node]]:
        seen: Set[Tuple[int, int]] = set()
        out = []
        for u in self.nodes:
            for v in u.neighbors:
                key = (min(id(u), id(v)), max(id(u), id(v)))
                if key in seen:
                    continue
                seen.add(key)
                out.append((u, v))
        return out

    def insert_leaf_on_edge(self, edge: Tuple[_Node, _Node], label: str) -> None:
        u, v = edge
        u.neighbors.remove(v)
        v.neighbors.remove(u)
        mid = _Node()
        leaf = _Node(label)
        u.neighbors.append(mid)
        mid.neighbors.extend([u, v, leaf])
        v.neighbors.append(mid)
        leaf.neighbors.append(mid)
        self.nodes.extend([mid, leaf])

    def split_of_edge(self, edge: Tuple[_Node, _Node]) -> Tuple[Set[str], Set[str]]:
        u, v = edge
        A: Set[str] = set()
        stack = [u]
        visited = {id(v)}
        while stack:
            n = stack.pop()
            if id(n) in visited:
                continue
            visited.add(id(n))
            if n.label:
                A.add(n.label)
            for m in n.neighbors:
                if id(m) not in visited:
                    stack.append(m)
        all_leaves = set()
        for n in self.nodes:
            if n.label:
                all_leaves.add(n.label)
        return (A, all_leaves - A)

    def leaves(self) -> Set[str]:
        return {n.label for n in self.nodes if n.label is not None}

    def to_newick(self) -> str:
        start = next((n for n in self.nodes if n.label is not None))
        parent = start.neighbors[0]
        body = self._render(parent, skip_id=id(start))
        return f'({start.label},{body});'

    def _render(self, node: _Node, skip_id: int) -> str:
        if node.label is not None:
            return node.label
        children = [c for c in node.neighbors if id(c) != skip_id]
        rendered = [self._render(c, skip_id=id(node)) for c in children]
        return '(' + ','.join(rendered) + ')'

def puzzle_tree(taxa: Sequence[str], quartet_calls: Dict[FrozenSet[str], QuartetCall], n_runs: int=10, seed: Optional[int]=0) -> Tuple[str, Dict[str, int]]:
    rng = random.Random(seed)
    split_counter: Counter = Counter()
    best_tree_newick: Optional[str] = None
    best_score = -1
    for _ in range(max(1, n_runs)):
        order = list(taxa)
        rng.shuffle(order)
        if len(order) < 3:
            return ('(' + ','.join(order) + ');', {})
        tree = _UnrootedTree.trivial_triplet(order[0], order[1], order[2])
        for new_taxon in order[3:]:
            _insert_best_edge(tree, new_taxon, quartet_calls, rng)
        run_splits = []
        for e in tree.edges():
            A, B = tree.split_of_edge(e)
            if len(A) > 1 and len(B) > 1:
                key = _split_key(A, B)
                split_counter[key] += 1
                run_splits.append(key)
        score = _score_tree_against_calls(tree, quartet_calls)
        if score > best_score:
            best_score = score
            best_tree_newick = tree.to_newick()
    assert best_tree_newick is not None
    return (best_tree_newick, dict(split_counter))

def _insert_best_edge(tree: _UnrootedTree, new_taxon: str, calls: Dict[FrozenSet[str], QuartetCall], rng: random.Random) -> None:
    placed = tree.leaves()
    relevant_calls: List[QuartetCall] = []
    for trip in combinations(placed, 3):
        key = frozenset({new_taxon, *trip})
        if key in calls:
            relevant_calls.append(calls[key])
    best_edges = []
    best_agree = -1
    for edge in tree.edges():
        A, B = tree.split_of_edge(edge)
        agree = 0
        for call in relevant_calls:
            side_A = {new_taxon}
            side_B = set()
            for t in call.taxa:
                if t == new_taxon:
                    continue
                if t in A:
                    side_A.add(t)
                else:
                    side_B.add(t)
            pair_sides = {'A': side_A, 'B': side_B}
            for p in (call.pair1, call.pair2):
                if not (p <= side_A or p <= side_B):
                    break
            else:
                agree += 1
        if agree > best_agree:
            best_agree = agree
            best_edges = [edge]
        elif agree == best_agree:
            best_edges.append(edge)
    choice = rng.choice(best_edges)
    tree.insert_leaf_on_edge(choice, new_taxon)

def _score_tree_against_calls(tree: _UnrootedTree, calls: Dict[FrozenSet[str], QuartetCall]) -> int:
    taxa = tree.leaves()
    splits = [tree.split_of_edge(e) for e in tree.edges()]
    agree = 0
    for call in calls.values():
        if not call.taxa <= taxa:
            continue
        for A, B in splits:
            if call.pair1 <= A and call.pair2 <= B or (call.pair1 <= B and call.pair2 <= A):
                agree += 1
                break
    return agree

def _split_key(A: Iterable[str], B: Iterable[str]) -> str:
    a = '|'.join(sorted(A))
    b = '|'.join(sorted(B))
    return a + ' || ' + b if a < b else b + ' || ' + a

def build_tree_via_svdquartets(alignment, kappa: int=4, n_runs: int=10, seed: int=0):
    from .svdquartets import all_quartets
    taxa = list(alignment.keys())
    results = all_quartets(alignment, kappa=kappa)
    calls: Dict[FrozenSet[str], QuartetCall] = {}
    for qr in results:
        best = qr.best
        a, b = tuple(best[0])
        c, d = tuple(best[1])
        calls[frozenset(qr.taxa)] = QuartetCall.make(a, b, c, d)
    return puzzle_tree(taxa, calls, n_runs=n_runs, seed=seed)