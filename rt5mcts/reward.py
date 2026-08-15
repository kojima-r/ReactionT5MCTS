"""Pluggable route-evaluation (reward) functions for the MCTS.

A reward maps a (partial) route tree to a scalar in ``[0, 1]`` — higher is
better.  The MCTS uses it for the rollout value that is back-propagated, the
score stored for each solved route, and the final ranking.  Swapping the reward
changes what the planner optimises for, without touching the search.

Design (inspired by knowledge-based retrosynthesis planners such as ReTReK,
re-implemented independently — no external code): a reward is either a single
purpose-built class or a **composite of weighted terms**.  A *term* is a small
function ``route -> [0, 1]`` measuring one desirable property; a
:class:`CompositeReward` combines any selection of terms with per-term weights,
either as a weighted product (default) or a weighted average.  This is what lets
you configure "various reward functions" from config alone.

Built-in terms (register more with ``@term("name")``):
  * ``in_stock``  — fraction of leaves already purchasable
  * ``solved``    — 1 if every leaf is in stock, else 0
  * ``shallow``   — ``depth_decay ** n_reactions`` (prefers short routes)
  * ``yield``     — aggregated per-step predicted yield (needs a yield model)
  * ``sa``        — mean synthetic-accessibility *ease* of the leaves
                    (RDKit SA score mapped so easy building blocks score ~1)

Built-in rewards (select with ``MCTSConfig(reward_policy=...)``):
  * ``stock``      — the historical value: ``shallow`` × ``in_stock``
  * ``yield``      — ``stock`` × (aggregated step yield) ** ``yield_weight``
  * ``sa``         — product of ``sa`` × ``in_stock`` × ``shallow``
  * ``composite``  — fully configurable weighted combination of any terms via
                     ``reward_terms={name: weight, ...}`` and ``reward_combine``

Add a new reward in three lines::

    @register
    class MyReward(RouteReward):
        name = "mine"
        def score(self, root): return ...
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .tree import (RouteMol, count_reactions, in_stock_fraction, internal_mols,
                   is_solved, iter_step_details, iter_steps, leaves)


# --------------------------------------------------------------------------- #
# Reward context (shared state the terms may need)
# --------------------------------------------------------------------------- #
@dataclass
class RewardContext:
    """Everything a term might need beyond the route itself, plus caches."""
    yield_model: object = None
    depth_decay: float = 0.95
    yield_agg: str = "geomean"
    _sa_cache: Dict[str, float] = field(default_factory=dict)
    _desc_cache: Dict[str, tuple] = field(default_factory=dict)


def _aggregate(values: List[float], how: str) -> float:
    if not values:
        return 1.0  # no reactions yet => neutral
    if how == "product":
        p = 1.0
        for v in values:
            p *= v
        return p
    if how == "min":
        return min(values)
    if how == "mean":
        return sum(values) / len(values)
    # geometric mean (default): rewards consistently high values, less punishing
    # than the raw product for long routes
    return math.exp(sum(math.log(max(v, 1e-6)) for v in values) / len(values))


# --------------------------------------------------------------------------- #
# Terms  (route, ctx) -> [0, 1]
# --------------------------------------------------------------------------- #
TERMS: Dict[str, Callable[[RouteMol, RewardContext], float]] = {}


def term(name: str):
    """Decorator registering a reward term under ``name``."""
    def deco(fn):
        TERMS[name] = fn
        return fn
    return deco


@term("in_stock")
def _term_in_stock(root: RouteMol, ctx: RewardContext) -> float:
    return in_stock_fraction(root)


@term("solved")
def _term_solved(root: RouteMol, ctx: RewardContext) -> float:
    return 1.0 if is_solved(root) else 0.0


@term("shallow")
def _term_shallow(root: RouteMol, ctx: RewardContext) -> float:
    return ctx.depth_decay ** count_reactions(root)


@term("yield")
def _term_yield(root: RouteMol, ctx: RewardContext) -> float:
    if ctx.yield_model is None:
        return 1.0  # neutral if no yield model wired
    ys = [ctx.yield_model.predict_yield(prod, reacts)
          for prod, reacts in iter_steps(root)]
    return _aggregate(ys, ctx.yield_agg)


_SASCORER = None
_SA_UNAVAILABLE = False


def _sa_ease(smiles: str) -> float:
    """Map RDKit SA score (1=easy .. 10=hard) to an ease value in [0, 1]."""
    global _SASCORER, _SA_UNAVAILABLE
    if _SA_UNAVAILABLE:
        return 1.0
    if _SASCORER is None:
        try:
            from rdkit.Chem import RDConfig
            sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
            import sascorer  # type: ignore
            _SASCORER = sascorer
        except Exception:
            _SA_UNAVAILABLE = True
            return 1.0
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.5
    sa = _SASCORER.calculateScore(mol)  # ~[1, 10]
    return max(0.0, min(1.0, 1.0 - (sa - 1.0) / 9.0))


def _sa_ease_cached(smiles: str, ctx: RewardContext) -> float:
    v = ctx._sa_cache.get(smiles)
    if v is None:
        v = _sa_ease(smiles)
        ctx._sa_cache[smiles] = v
    return v


@term("sa")
def _term_sa(root: RouteMol, ctx: RewardContext) -> float:
    vals = [_sa_ease_cached(m.smiles, ctx) for m in leaves(root)]
    return sum(vals) / len(vals) if vals else 1.0


# --------------------------------------------------------------------------- #
# ReTReK-style knowledge scores (independent re-implementation from the
# published definitions — no external code referenced).  Each is a term in
# [0, 1], aggregated as the mean over the route's reaction steps; a route with
# no reactions returns the neutral value 1.0.  They are designed to be *summed*
# (reward_combine="sum"): use the ``retrek`` preset.
# --------------------------------------------------------------------------- #
def _desc(smiles: str, ctx: RewardContext) -> tuple:
    """(#heavy atoms, #rings) for a SMILES, cached; (0, 0) if unparsable."""
    d = ctx._desc_cache.get(smiles)
    if d is None:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        d = ((mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(mol))
             if mol is not None else (0, 0))
        ctx._desc_cache[smiles] = d
    return d


def _step_mean(values: List[float], empty: float = 1.0) -> float:
    return sum(values) / len(values) if values else empty


@term("cdscore")
def _term_cdscore(root: RouteMol, ctx: RewardContext) -> float:
    """Convergent Disconnection: reward steps that split into balanced fragments.

    Per step, convergence = min/max of the reactant heavy-atom counts (1.0 when
    the two fragments are equal-sized = maximally convergent, →0 for a big+small
    linear disconnection); a single-reactant step is not convergent (0.0).
    """
    scores = []
    for _prod, reactants, _p in iter_step_details(root):
        sizes = [h for h in (_desc(r, ctx)[0] for r in reactants) if h > 0]
        scores.append(min(sizes) / max(sizes) if len(sizes) >= 2 else 0.0)
    return _step_mean(scores)


@term("asscore")
def _term_asscore(root: RouteMol, ctx: RewardContext) -> float:
    """Available Substances: fraction of leaf compounds that are purchasable."""
    return in_stock_fraction(root)


@term("rdscore")
def _term_rdscore(root: RouteMol, ctx: RewardContext) -> float:
    """Ring Disconnection: reward steps that build a ring (retro: break one).

    1.0 for a step whose product has more rings than its reactants combined,
    else 0.0; averaged over steps.
    """
    scores = []
    for prod, reactants, _p in iter_step_details(root):
        r_prod = _desc(prod, ctx)[1]
        r_reac = sum(_desc(r, ctx)[1] for r in reactants)
        scores.append(1.0 if r_prod > r_reac else 0.0)
    return _step_mean(scores)


@term("stscore")
def _term_stscore(root: RouteMol, ctx: RewardContext) -> float:
    """Selective Transformation: prefer few-reactant (more selective) steps.

    Per step = 1 / (number of reactants): a clean 1-reactant transformation
    scores 1.0, a 2-reactant coupling 0.5, and so on.
    """
    scores = []
    for _prod, reactants, _p in iter_step_details(root):
        n = max(1, len(reactants))
        scores.append(1.0 / n)
    return _step_mean(scores)


@term("intermediate")
def _term_intermediate(root: RouteMol, ctx: RewardContext) -> float:
    """Intermediate: reward routes passing through reasonable intermediates.

    Mean synthetic-accessibility ease over the route's internal (non-leaf,
    non-target) molecules — a proxy for "the intermediates are sensible".
    """
    mids = internal_mols(root)
    return _step_mean([_sa_ease_cached(m.smiles, ctx) for m in mids])


@term("template")
def _term_template(root: RouteMol, ctx: RewardContext) -> float:
    """Template: mean one-step model probability of the reactions used.

    Uses the (softmax-normalised) probability the policy assigned to each
    disconnection — higher means more confident / well-precedented templates.
    Steps with no recorded probability are skipped.
    """
    ps = [p for _prod, _r, p in iter_step_details(root) if p is not None]
    return _step_mean(ps)


def available_terms() -> List[str]:
    return sorted(TERMS)


# --------------------------------------------------------------------------- #
# Rewards
# --------------------------------------------------------------------------- #
class RouteReward:
    """Base class: subclass and implement :meth:`score`."""

    name = "base"

    def score(self, root: RouteMol) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, root: RouteMol) -> float:
        return self.score(root)


_REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator registering a reward under its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


@register
class StockReward(RouteReward):
    """Depth-decayed × in-stock fraction (the historical ``state_score``)."""

    name = "stock"

    def __init__(self, depth_decay: float = 0.95, **_ignored) -> None:
        self.depth_decay = depth_decay

    def score(self, root: RouteMol) -> float:
        frac = in_stock_fraction(root)
        if frac == 0.0:
            return 0.0
        return (self.depth_decay ** count_reactions(root)) * frac


@register
class YieldReward(RouteReward):
    """``stock`` × (aggregated per-step predicted yield) ** ``yield_weight``."""

    name = "yield"

    def __init__(self, yield_model, depth_decay: float = 0.95,
                 yield_weight: float = 1.0, yield_agg: str = "geomean",
                 **_ignored) -> None:
        if yield_model is None:
            raise ValueError("YieldReward requires a yield_model")
        self.ctx = RewardContext(yield_model=yield_model, depth_decay=depth_decay,
                                 yield_agg=yield_agg)
        self.depth_decay = depth_decay
        self.yield_weight = yield_weight

    def route_yield(self, root: RouteMol) -> float:
        return _term_yield(root, self.ctx)

    def score(self, root: RouteMol) -> float:
        frac = in_stock_fraction(root)
        if frac == 0.0:
            return 0.0
        base = (self.depth_decay ** count_reactions(root)) * frac
        return base * (self.route_yield(root) ** self.yield_weight)


@register
class CompositeReward(RouteReward):
    """Weighted combination of named terms — the fully configurable reward.

    ``terms`` maps a term name (see :data:`TERMS`) to a non-negative weight.
    ``combine`` is ``"product"`` (weighted geometric product, default) or
    ``"sum"`` (weighted average); both stay in [0, 1].  ``gate_on_stock`` keeps
    the historical behaviour that a route with no purchasable leaves scores 0.
    """

    name = "composite"

    def __init__(self, terms: Optional[Dict[str, float]] = None,
                 combine: str = "product", ctx: Optional[RewardContext] = None,
                 gate_on_stock: bool = True, **ctx_kwargs) -> None:
        self.terms = dict(terms) if terms else {"shallow": 1.0, "in_stock": 1.0}
        for name in self.terms:
            if name not in TERMS:
                raise KeyError(f"unknown reward term {name!r}; "
                               f"available: {available_terms()}")
        self.combine = combine
        self.gate_on_stock = gate_on_stock
        self.ctx = ctx or RewardContext(
            **{k: v for k, v in ctx_kwargs.items()
               if k in ("yield_model", "depth_decay", "yield_agg")})

    def score(self, root: RouteMol) -> float:
        if self.gate_on_stock and in_stock_fraction(root) == 0.0:
            return 0.0
        parts = [(TERMS[n](root, self.ctx), w)
                 for n, w in self.terms.items() if w != 0]
        if not parts:
            return 0.0
        if self.combine == "sum":
            wsum = sum(w for _, w in parts)
            return sum(v * w for v, w in parts) / wsum if wsum else 0.0
        prod = 1.0
        for v, w in parts:
            prod *= max(v, 0.0) ** w
        return prod


@register
class SAReward(CompositeReward):
    """Convenience preset: prefer routes into easy-to-make building blocks.

    Product of synthetic-accessibility ease × in-stock fraction × shallowness.
    """

    name = "sa"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("terms", {"sa": 1.0, "in_stock": 1.0, "shallow": 1.0})
        kwargs.setdefault("combine", "product")
        super().__init__(**kwargs)


@register
class RetrekReward(CompositeReward):
    """The six ReTReK-style knowledge scores, summed with equal weights.

    Terms: CDScore (cdscore), ASScore (asscore), RDScore (rdscore),
    STScore (stscore), IntermediateScore (intermediate), TemplateScore
    (template).  Re-weight or drop any of them via ``reward_terms``; combined
    as a weighted average (``combine="sum"``) as in the original additive
    formulation.  ``gate_on_stock`` keeps unsolved-leaf routes at 0.
    """

    name = "retrek"
    DEFAULT_TERMS = {"cdscore": 1.0, "asscore": 1.0, "rdscore": 1.0,
                     "stscore": 1.0, "intermediate": 1.0, "template": 1.0}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("terms", dict(self.DEFAULT_TERMS))
        kwargs.setdefault("combine", "sum")
        super().__init__(**kwargs)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_reward(policy: str = "stock", *, yield_model=None,
                reward_terms: Optional[Dict[str, float]] = None,
                reward_combine: str = "", depth_decay: float = 0.95,
                yield_weight: float = 1.0, yield_agg: str = "geomean",
                **kwargs) -> RouteReward:
    """Build a reward by name. A single config blob drives any reward.

    ``reward_combine`` empty ("") means "use the reward's own default"
    (``product`` for ``composite``/``sa``, ``sum`` for ``retrek``).  Unknown
    kwargs are ignored so the same call works for every policy.
    """
    if policy not in _REGISTRY:
        raise KeyError(f"unknown reward_policy {policy!r}; "
                       f"available: {available()}")
    if policy == "stock":
        return StockReward(depth_decay=depth_decay)
    if policy == "yield":
        return YieldReward(yield_model=yield_model, depth_decay=depth_decay,
                           yield_weight=yield_weight, yield_agg=yield_agg)
    # composite / sa / retrek (and any CompositeReward subclass): term-based
    ctx = RewardContext(yield_model=yield_model, depth_decay=depth_decay,
                        yield_agg=yield_agg)
    kw = {"ctx": ctx}
    if reward_combine:            # only override when explicitly given
        kw["combine"] = reward_combine
    if reward_terms:              # only override when explicitly given
        kw["terms"] = reward_terms
    return _REGISTRY[policy](**kw)


def reward_needs_yield(policy: str,
                       reward_terms: Optional[Dict[str, float]] = None) -> bool:
    """Whether a reward config requires the (expensive) yield model to be built.

    True for the ``yield`` reward, or any term-based reward whose (explicit)
    terms include the ``yield`` term.  The presets don't use yield by default.
    """
    if policy == "yield":
        return True
    if policy in ("composite", "sa", "retrek") and reward_terms:
        return float(reward_terms.get("yield", 0)) != 0
    return False


def available() -> List[str]:
    return sorted(_REGISTRY)
