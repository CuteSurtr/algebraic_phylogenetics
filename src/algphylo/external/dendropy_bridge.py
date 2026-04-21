from __future__ import annotations
from typing import Tuple
try:
    import dendropy
except ImportError as e:
    raise ImportError('dendropy required; `pip install dendropy`') from e

def newick_to_dendropy(newick: str) -> 'dendropy.Tree':
    return dendropy.Tree.get(data=newick, schema='newick')

def dendropy_to_newick(tree: 'dendropy.Tree') -> str:
    return tree.as_string(schema='newick').strip()

def rf_distance(tree_a_newick: str, tree_b_newick: str) -> int:
    tns = dendropy.TaxonNamespace()
    a = dendropy.Tree.get(data=tree_a_newick, schema='newick', taxon_namespace=tns)
    b = dendropy.Tree.get(data=tree_b_newick, schema='newick', taxon_namespace=tns)
    a.encode_bipartitions()
    b.encode_bipartitions()
    return int(dendropy.calculate.treecompare.symmetric_difference(a, b))

def same_topology(tree_a_newick: str, tree_b_newick: str) -> bool:
    return rf_distance(tree_a_newick, tree_b_newick) == 0