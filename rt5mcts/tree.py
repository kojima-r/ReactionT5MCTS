"""Partial-synthesis-tree data structures and helpers.

Split out of ``mcts.py`` so that reward functions (``reward.py``) can traverse
routes without importing the search itself (avoids a circular import).  A
*route* is a tree of :class:`RouteMol` nodes; a molecule with a
:class:`RouteReaction` child has been disconnected into that reaction's
reactant molecules.  Leaves with ``in_stock=True`` are purchasable.
"""
from __future__ import annotations

from typing import List, Optional

from rdkit import Chem


class RouteMol:
    """A molecule node in a (partial) synthesis tree."""

    __slots__ = ("smiles", "in_stock", "depth", "reaction")

    def __init__(self, smiles: str, in_stock: bool, depth: int) -> None:
        self.smiles = smiles
        self.in_stock = in_stock
        self.depth = depth
        self.reaction: Optional["RouteReaction"] = None  # None => leaf

    def clone(self) -> "RouteMol":
        new = RouteMol(self.smiles, self.in_stock, self.depth)
        if self.reaction is not None:
            new.reaction = self.reaction.clone()
        return new


class RouteReaction:
    """A reaction node: its children are the reactant molecules.

    The reaction's *product* is the parent :class:`RouteMol`, so a step is
    fully described by ``(parent.smiles, [child.smiles for child in children])``
    — everything a yield/feasibility model needs.  ``prob`` is the one-step
    model's (normalised) probability for this disconnection, used by the
    template-score reward; ``None`` if unknown.
    """

    __slots__ = ("children", "prob")

    def __init__(self, children: List[RouteMol], prob: Optional[float] = None) -> None:
        self.children = children
        self.prob = prob

    def clone(self) -> "RouteReaction":
        return RouteReaction([c.clone() for c in self.children], self.prob)


def iter_mols(root: RouteMol):
    stack = [root]
    while stack:
        m = stack.pop()
        yield m
        if m.reaction is not None:
            stack.extend(m.reaction.children)


# backwards-compatible private alias (older code imported the underscored name)
_iter_mols = iter_mols


def iter_steps(root: RouteMol):
    """Yield each reaction step as ``(product_smiles, [reactant_smiles, ...])``."""
    for m in iter_mols(root):
        if m.reaction is not None:
            yield m.smiles, [c.smiles for c in m.reaction.children]


def iter_step_details(root: RouteMol):
    """Yield ``(product_smiles, [reactant_smiles, ...], prob)`` per reaction step."""
    for m in iter_mols(root):
        if m.reaction is not None:
            yield (m.smiles, [c.smiles for c in m.reaction.children],
                   m.reaction.prob)


def internal_mols(root: RouteMol) -> List[RouteMol]:
    """Non-root molecules that are further disconnected (route intermediates)."""
    return [m for m in iter_mols(root)
            if m.reaction is not None and m is not root]


def leaves(root: RouteMol) -> List[RouteMol]:
    return [m for m in iter_mols(root) if m.reaction is None]


def open_leaves(root: RouteMol, max_depth: int) -> List[RouteMol]:
    """Leaves that are not in stock and can still be expanded."""
    return [
        m for m in iter_mols(root)
        if m.reaction is None and not m.in_stock and m.depth < max_depth
    ]


def count_reactions(root: RouteMol) -> int:
    return sum(1 for m in iter_mols(root) if m.reaction is not None)


def is_solved(root: RouteMol) -> bool:
    return all(m.in_stock for m in leaves(root))


def in_stock_fraction(root: RouteMol) -> float:
    lvs = leaves(root)
    if not lvs:
        return 0.0
    return sum(1 for m in lvs if m.in_stock) / len(lvs)


def _n_heavy(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def choose_expandable(root: RouteMol, max_depth: int) -> Optional[RouteMol]:
    """Deterministically pick the open leaf to expand (largest molecule first)."""
    opens = open_leaves(root, max_depth)
    if not opens:
        return None
    return max(opens, key=lambda m: (_n_heavy(m.smiles), m.smiles))


def signature(root: RouteMol) -> str:
    """A canonical, order-independent string signature of a route tree."""
    if root.reaction is None:
        return root.smiles
    parts = sorted(signature(c) for c in root.reaction.children)
    return f"{root.smiles}<[{'|'.join(parts)}]"


def to_paroutes(root: RouteMol) -> dict:
    node = {"type": "mol", "smiles": root.smiles, "in_stock": bool(root.in_stock)}
    if root.reaction is not None:
        node["children"] = [
            {
                "type": "reaction",
                "children": [to_paroutes(c) for c in root.reaction.children],
            }
        ]
    return node
