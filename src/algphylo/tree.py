from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Node:
    name: Optional[str] = None
    length: float = 0.0
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

@dataclass
class Tree:
    root: Node
    taxa: List[str]

    @staticmethod
    def build(root: Node, taxa: List[str]) -> 'Tree':
        return Tree(root=root, taxa=sorted(set(taxa)))

    def leaves(self) -> List[Node]:
        out: List[Node] = []

        def walk(n: Node) -> None:
            if n.is_leaf:
                out.append(n)
            for c in n.children:
                walk(c)
        walk(self.root)
        return out

    def postorder(self) -> List[Node]:
        out: List[Node] = []

        def walk(n: Node) -> None:
            for c in n.children:
                walk(c)
            out.append(n)
        walk(self.root)
        return out

def parse_newick(s: str) -> Tree:
    if not isinstance(s, str) or not s.strip():
        raise ValueError('empty or non-string Newick input')
    tokens = _tokenize(s.strip())
    pos = [0]

    def peek() -> str:
        return tokens[pos[0]]

    def eat(tok: str) -> None:
        if tokens[pos[0]] != tok:
            raise ValueError(f'expected {tok!r}, got {tokens[pos[0]]!r}')
        pos[0] += 1

    def parse_subtree() -> Node:
        if peek() == '(':
            eat('(')
            children = [parse_subtree()]
            while peek() == ',':
                eat(',')
                children.append(parse_subtree())
            eat(')')
            name = None
            if peek() not in (',', ')', ':', ';'):
                name = tokens[pos[0]]
                pos[0] += 1
            length = 0.0
            if peek() == ':':
                eat(':')
                length = float(tokens[pos[0]])
                pos[0] += 1
            n = Node(name=name, length=length, children=children)
            for c in children:
                c.parent = n
            return n
        name = tokens[pos[0]]
        pos[0] += 1
        length = 0.0
        if peek() == ':':
            eat(':')
            length = float(tokens[pos[0]])
            pos[0] += 1
        return Node(name=name, length=length)
    root = parse_subtree()
    if peek() == ';':
        eat(';')
    taxa: List[str] = []

    def collect(n: Node) -> None:
        if n.is_leaf and n.name:
            taxa.append(n.name)
        for c in n.children:
            collect(c)
    collect(root)
    return Tree.build(root, taxa)

def _tokenize(s: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in '(),:;':
            out.append(c)
            i += 1
            continue
        j = i
        while j < len(s) and s[j] not in '(),:;' and (not s[j].isspace()):
            j += 1
        out.append(s[i:j])
        i = j
    out.append(';')
    return out

def to_newick(t: Tree) -> str:

    def render(n: Node) -> str:
        body = n.name or '' if n.is_leaf else '(' + ','.join((render(c) for c in n.children)) + ')'
        if n.parent is not None and n.length != 0:
            body += f':{n.length:g}'
        return body
    return render(t.root) + ';'