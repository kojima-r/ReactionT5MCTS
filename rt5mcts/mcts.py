"""Monte-Carlo Tree Search for retrosynthetic route planning.

The formulation follows the AiZynthFinder-style MCTS: a search *state* is the
current partial synthesis tree, whose leaves are either purchasable (in stock)
or still "open".  An *action* expands a single open leaf with one reaction
proposed by the ReactionT5 one-step model, replacing that leaf with the
predicted reactants.  A state is *solved* when every leaf is in stock.

Selection uses a PUCT rule (AlphaZero-style) with the model probabilities as
priors; a greedy top-1 rollout provides the value estimate.  Every solved state
encountered during expansion or rollout is collected as a candidate route, and
the collected routes are ranked to produce the top-N output PaRoutes expects.

**The state-evaluation (reward) function is pluggable** — see ``reward.py``.
``RetroMCTS`` takes a ``reward`` callable (route tree -> value in [0, 1]); by
default it is built from ``config.reward_policy`` (``"stock"`` or ``"yield"``).
The reward is the single place that decides what routes are worth, so swapping
it (e.g. to weight in per-step predicted yields) changes the planner's
objective without touching the search.

The search is fully deterministic given a fixed seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from rdkit import Chem

from .model import ReactionT5
from .reward import RouteReward, StockReward, make_reward
from .stock import Stock
# tree structures + helpers (re-exported here for backwards compatibility)
from .tree import (RouteMol, RouteReaction, choose_expandable, count_reactions,
                   in_stock_fraction, is_solved, iter_mols, leaves,
                   open_leaves, signature, to_paroutes)

# backwards-compatible private aliases (older imports used these names)
_iter_mols = iter_mols


def state_score(root: RouteMol) -> float:
    """Historical default value function (kept for backwards compatibility).

    Equivalent to the ``stock`` reward: depth-decayed × in-stock fraction.
    New code should go through a :class:`~rt5mcts.reward.RouteReward`.
    """
    return StockReward().score(root)


# --------------------------------------------------------------------------- #
# MCTS
# --------------------------------------------------------------------------- #
@dataclass
class MCTSConfig:
    expansion_width: int = 5     # candidate reactions generated per expansion
    iterations: int = 100        # MCTS iterations (search budget)
    c_puct: float = 1.4          # exploration constant
    max_depth: int = 7           # max reactions on any path (== max_transforms)
    rollout_width: int = 1       # top-k used in the greedy rollout (1 = greedy)
    rollout_policy: str = "stock"  # "stock" (maximise in-stock reactants) or "prob"
    model_budget: int = 0        # max fresh model calls per target (0 = unlimited)
    seed: int = 42
    # --- pluggable reward (state-evaluation) ------------------------------- #
    reward_policy: str = "stock"     # name in rt5mcts.reward registry
    reward_depth_decay: float = 0.95  # per-reaction decay in the value
    yield_weight: float = 1.0        # exponent on the yield term (yield reward)
    yield_agg: str = "geomean"       # geomean | product | min | mean
    # composite reward: {term_name: weight} and how to combine (product | sum);
    # empty combine = use the reward's own default (product / retrek's sum)
    reward_terms: dict = field(default_factory=dict)
    reward_combine: str = ""


class _Node:
    __slots__ = ("state", "parent", "children", "prior", "N", "W",
                 "expanded", "solved", "dead")

    def __init__(self, state: RouteMol, parent: Optional["_Node"], prior: float):
        self.state = state
        self.parent = parent
        self.children: List[_Node] = []
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.expanded = False
        self.solved = is_solved(state)
        self.dead = False

    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class RetroMCTS:
    def __init__(self, model: ReactionT5, stock: Stock, config: MCTSConfig,
                 reward: Optional[Callable[[RouteMol], float]] = None,
                 yield_model=None) -> None:
        self.model = model
        self.stock = stock
        self.cfg = config
        self.rng = random.Random(config.seed)
        # the pluggable state-evaluation function; explicit `reward` wins,
        # otherwise build it from the config (policy + params).
        self.reward: Callable[[RouteMol], float] = reward or make_reward(
            config.reward_policy, yield_model=yield_model,
            depth_decay=config.reward_depth_decay,
            yield_weight=config.yield_weight, yield_agg=config.yield_agg,
            reward_terms=config.reward_terms, reward_combine=config.reward_combine)
        # signature -> (route_dict, n_reactions, reward)
        self.solutions: Dict[str, Tuple[dict, int, float]] = {}
        # best fraction of leaves reaching stock over any explored state
        self.best_frac = 0.0

    # -- route bookkeeping --------------------------------------------------
    def _record_if_solved(self, state: RouteMol) -> None:
        if not is_solved(state):
            return
        sig = signature(state)
        if sig in self.solutions:
            return
        self.solutions[sig] = (to_paroutes(state), count_reactions(state),
                               self.reward(state))

    # -- expansion ----------------------------------------------------------
    def _make_child_state(self, state: RouteMol, target_smiles: str,
                          reactants: Tuple[str, ...],
                          prob: Optional[float] = None) -> Optional[RouteMol]:
        new_state = state.clone()
        # locate the corresponding leaf in the cloned tree
        target_leaf = None
        for m in iter_mols(new_state):
            if m.reaction is None and m.smiles == target_smiles and not m.in_stock:
                target_leaf = m
                break
        if target_leaf is None:
            return None
        children = []
        for r in reactants:
            child = RouteMol(r, in_stock=(r in self.stock), depth=target_leaf.depth + 1)
            children.append(child)
        target_leaf.reaction = RouteReaction(children, prob=prob)
        return new_state

    def _expand(self, node: _Node) -> None:
        node.expanded = True
        mol = choose_expandable(node.state, self.cfg.max_depth)
        if mol is None:
            node.dead = True
            return
        preds = self.model.predict(mol.smiles, top_k=self.cfg.expansion_width)
        for pred in preds:
            child_state = self._make_child_state(node.state, mol.smiles, pred.reactants, prob=pred.prob)
            if child_state is None:
                continue
            child = _Node(child_state, node, prior=pred.prob)
            node.children.append(child)
            self._record_if_solved(child_state)
        if not node.children:
            node.dead = True

    # -- selection ----------------------------------------------------------
    def _puct(self, parent: _Node, child: _Node) -> float:
        return child.q() + self.cfg.c_puct * child.prior * math.sqrt(parent.N) / (1 + child.N)

    def _best_child(self, node: _Node) -> Optional[_Node]:
        alive = [c for c in node.children if not c.dead]
        if not alive:
            return None
        best, best_score = None, -1e18
        for c in alive:
            s = self._puct(node, c)
            if s > best_score:
                best, best_score = c, s
        return best

    def _select(self, root: _Node) -> List[_Node]:
        path = [root]
        node = root
        while True:
            if node.solved or node.dead:
                return path
            if not node.expanded:
                self._expand(node)
                if node.dead:
                    return path
                child = self._best_child(node)
                if child is None:
                    node.dead = True
                    return path
                path.append(child)
                return path
            child = self._best_child(node)
            if child is None:
                node.dead = True
                return path
            node = child
            path.append(node)

    # -- rollout ------------------------------------------------------------
    def _stock_score(self, pred) -> tuple:
        """Rank a candidate reaction by how close it gets to purchasable stock."""
        n_stock = sum(1 for r in pred.reactants if r in self.stock)
        n_open = len(pred.reactants) - n_stock
        return (n_stock - n_open, pred.prob)

    def _rollout(self, state: RouteMol) -> float:
        work = state.clone()
        for _ in range(self.cfg.max_depth * 4):  # generous safety bound
            if is_solved(work):
                break
            mol = choose_expandable(work, self.cfg.max_depth)
            if mol is None:
                break
            if self.cfg.rollout_policy == "stock":
                # look at the full cached candidate list (no extra model calls)
                preds = self.model.predict(mol.smiles, top_k=self.cfg.expansion_width)
                if not preds:
                    break
                choice = max(preds, key=self._stock_score)
            else:
                preds = self.model.predict(mol.smiles, top_k=self.cfg.rollout_width)
                if not preds:
                    break
                if self.cfg.rollout_width == 1:
                    choice = preds[0]
                else:
                    choice = self.rng.choices(preds, weights=[p.prob for p in preds])[0]
            new_state = self._make_child_state(work, mol.smiles, choice.reactants, prob=choice.prob)
            if new_state is None:
                break
            work = new_state
        self._record_if_solved(work)
        frac = in_stock_fraction(work)
        if frac > self.best_frac:
            self.best_frac = frac
        return self.reward(work)

    # -- backprop -----------------------------------------------------------
    @staticmethod
    def _backprop(path: List[_Node], reward: float) -> None:
        for node in path:
            node.N += 1
            node.W += reward

    # -- driver -------------------------------------------------------------
    def search(self, target_smiles: str) -> None:
        self.model.reset_budget(self.cfg.model_budget if self.cfg.model_budget > 0 else None)
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(target_smiles))
        root_state = RouteMol(canon, in_stock=(canon in self.stock), depth=0)
        self._record_if_solved(root_state)  # target already in stock -> trivial route
        root = _Node(root_state, None, prior=1.0)
        for _ in range(self.cfg.iterations):
            if root.dead:
                break
            path = self._select(root)
            reward = self._rollout(path[-1].state)
            self._backprop(path, reward)

    def ranked_routes(self, max_routes: int = 10) -> List[dict]:
        """Return solved routes ranked best-first (higher reward, then fewer reactions)."""
        items = list(self.solutions.values())
        items.sort(key=lambda t: (-t[2], t[1]))
        return [route for route, _, _ in items[:max_routes]]
