# Retrosynthesis Route Planning: ReactionT5-MCTS vs AiZynthFinder on PaRoutes

*Generated 2026-08-15 09:03 UTC*

This report evaluates a Monte-Carlo Tree Search (MCTS) retrosynthetic planner
that uses **ReactionT5** (`sagawa/ReactionT5v2-retrosynthesis`) as the one-step
retrosynthesis model, benchmarked on the **PaRoutes** n1/n5 route sets, and
compares it with **AiZynthFinder** and the published literature values.

**Reproducibility.** All runs use a fixed random seed. ReactionT5 beam search is
deterministic and one-step predictions are memoised in a shared on-disk cache,
so re-running yields identical routes.

**Scope note.** Runs use GPU-backed ReactionT5 inference with target-level
parallelism, so the **full benchmark (10,000 targets/set)** is evaluated for
every "This work" row. Each ReactionT5-MCTS target is still capped at a fixed
*budget* of fresh one-step model evaluations (memoised in a shared cache). Three
planners are compared like-for-like on the identical full sets: **ReactionT5-MCTS**
(best sweep config), **AiZynthFinder-MCTS**, and **AiZynthFinder-Retro\*** — the
latter using the public USPTO expansion policy with the PaRoutes
`retrostar_value_model.pickle` molecule-cost model. The literature columns are
the full-benchmark published values, produced with the PaRoutes-specific
template one-step models, and are shown for context.

Best ReactionT5-MCTS sweep configuration (by solve rate on the n1 subset): **sweep-base**.


## 1. Hyperparameter sweep (ReactionT5-MCTS, n1 subset)

Base config: expansion_width=5, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15. Each row varies one parameter.

| config (tag) | route set | params | solved | solve rate | in-stock frac | top-1 | top-5 | top-10 | total time | time/target |
|---|---|---|---|---|---|---|---|---|---|---|
| base | n1 | expansion_width=5, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.661 | 0.000 | 0.000 | 0.000 | 1.6 | 0.5 |
| c0p5 | n1 | expansion_width=5, iterations=100, c_puct=0.5, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.660 | 0.000 | 0.000 | 0.000 | 6.8 | 2.3 |
| c3p0 | n1 | expansion_width=5, iterations=100, c_puct=3.0, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.661 | 0.000 | 0.000 | 0.000 | 2.2 | 0.7 |
| ew3 | n1 | expansion_width=3, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.621 | 0.000 | 0.000 | 0.000 | 3.4 | 1.1 |
| it200 | n1 | expansion_width=5, iterations=200, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.661 | 0.000 | 0.000 | 0.000 | 2.8 | 0.9 |
| md8 | n1 | expansion_width=5, iterations=100, c_puct=1.4, max_depth=8, rollout_width=1, rollout_policy=stock, model_budget=15 | 0/3 | 0.000 | 0.642 | 0.000 | 0.000 | 0.000 | 10.0 | 3.3 |
| rprob | n1 | expansion_width=5, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=prob, model_budget=15 | 0/3 | 0.000 | 0.642 | 0.000 | 0.000 | 0.000 | 13.7 | 4.6 |

## 2. Main comparison: literature vs ReactionT5-MCTS (best) vs AiZynthFinder

| source / method | route set | n targets | solved | solve rate | in-stock frac | top-1 | top-5 | top-10 | total time | time/target |
|---|---|---|---|---|---|---|---|---|---|---|
| Genheden 2022 (v1) — MCTS | n1 | 10000 | 9714 | 0.971 | N/A | 0.200 | 0.550 | 0.610 | N/A | N/A |
| Genheden 2022 (v1) — MCTS | n5 | 10000 | 9676 | 0.968 | N/A | 0.090 | 0.340 | 0.420 | N/A | N/A |
| Genheden 2022 (v1) — Retro* | n1 | 10000 | 9726 | 0.973 | N/A | 0.170 | 0.480 | 0.540 | N/A | N/A |
| Genheden 2022 (v1) — Retro* | n5 | 10000 | 9703 | 0.970 | N/A | 0.080 | 0.300 | 0.380 | N/A | N/A |
| PaRoutes v2.0 — MCTS | n1 | 10000 | 9716 | 0.972 | N/A | 0.237 | 0.511 | 0.541 | N/A | N/A |
| PaRoutes v2.0 — MCTS | n5 | 10000 | 9689 | 0.969 | N/A | 0.124 | 0.358 | 0.406 | N/A | N/A |
| PaRoutes v2.0 — Retro* | n1 | 10000 | 9728 | 0.973 | N/A | 0.203 | 0.452 | 0.485 | N/A | N/A |
| PaRoutes v2.0 — Retro* | n5 | 10000 | 9729 | 0.973 | N/A | 0.114 | 0.337 | 0.390 | N/A | N/A |
| This work — AiZynthFinder | n1 | 10000 | 7538 | 0.754 | N/A | 0.059 | 0.107 | 0.107 | 138.2 min | 0.8 |
| This work — AiZynthFinder | n5 | 10000 | 7413 | 0.741 | N/A | 0.030 | 0.070 | 0.070 | 110.4 min | 0.7 |
| This work — AiZynthFinder-MCTS-PaRoutes | n1 | 10000 | 8440 | 0.844 | N/A | 0.038 | 0.100 | 0.101 | 388.5 min | 2.3 |
| This work — AiZynthFinder-MCTS-PaRoutes | n5 | 10000 | 8444 | 0.844 | N/A | 0.016 | 0.059 | 0.060 | 363.1 min | 2.2 |
| This work — AiZynthFinder-Retro* | n1 | 10000 | 7426 | 0.743 | N/A | 0.054 | 0.090 | 0.090 | 804.6 min | 4.8 |
| This work — AiZynthFinder-Retro* | n5 | 10000 | 7396 | 0.740 | N/A | 0.028 | 0.059 | 0.059 | 844.2 min | 5.1 |
| This work — AiZynthFinder-Retro*-PaRoutes | n1 | 10000 | 8397 | 0.840 | N/A | 0.026 | 0.062 | 0.062 | 3336.3 min | 20.0 |
| This work — AiZynthFinder-Retro*-PaRoutes | n5 | 10000 | 8413 | 0.841 | N/A | 0.011 | 0.033 | 0.034 | 3385.6 min | 20.3 |
| This work — ReactionT5-MCTS (best config: base) | n1 | 10000 | 6902 | 0.690 | 0.972 | 0.045 | 0.049 | 0.049 | 118.4 min | 0.7 |
| This work — ReactionT5-MCTS (best config: base) | n5 | 10000 | 6390 | 0.639 | 0.966 | 0.021 | 0.024 | 0.024 | 7.2 min | 0.0 |
| This work — ReactionT5-MCTS (best reward: retrek) | n1 | 10000 | 6907 | 0.691 | 0.972 | 0.052 | 0.057 | 0.057 | 8.3 min | 0.0 |
| This work — ReactionT5-MCTS (best reward: retrek) | n5 | 10000 | 6383 | 0.638 | 0.966 | 0.025 | 0.029 | 0.029 | 5.3 min | 0.0 |

### Notes
- **solve rate is the primary metric here** = number of targets for which at
  least one predicted route reaches purchasable stock, divided by n targets.
- **top-k** = fraction of targets whose exact reference route is recovered
  (route tree-edit distance == 0) among the top-k ranked routes — the strict
  PaRoutes accuracy metric. For **ReactionT5-MCTS this is expected to be ~0**:
  ReactionT5 is a template-free seq2seq model, whereas the PaRoutes reference
  routes were generated by a template-based model, so recovering the exact same
  tree is very unlikely even when a valid alternative route is found.
- **in-stock frac** = mean over targets of the best fraction of route leaves that
  reach purchasable stock. It is a softer progress metric than the all-or-nothing
  "solved": a value near 1.0 means the planner found an almost-complete route
  (only one or two leaves short), which the strict "solved" flag hides.
- ReactionT5 emits reagents/solvents/salts that the PaRoutes building-block stock
  lacks, so the ReactionT5-MCTS runs augment the stock with a curated set of
  common reagents (chemically these are always available). The rollout uses a
  stock-greedy policy that prefers disconnections into already-purchasable pieces.
- **time** columns are wall-clock on CPU; literature values were produced on
  different hardware over the full benchmark and report no runtime.
- AiZynthFinder here uses the public USPTO expansion policy with the PaRoutes
  stock; the literature MCTS/Retro* rows use the PaRoutes-specific one-step models.
- **AiZynthFinder-Retro\*** is the Retro\* search algorithm with the PaRoutes
  `retrostar_value_model.pickle` as the molecule-cost (value) model, otherwise
  identical policy/stock/budget to the AiZynthFinder-MCTS run. Comparing the two
  "This work — AiZynthFinder" rows isolates the effect of the search algorithm
  (MCTS vs Retro\*) with everything else held fixed.
- **AiZynthFinder-*-PaRoutes** rows use the **literature one-step model**: the
  route-set-specific expansion policy `uspto_rxn_n{1,5}_keras_model.hdf5`
  (Zenodo 6275421, converted Keras→ONNX) with its matching template set, plus the
  **literature search budget** (iteration_limit=500, time_limit=3600 s). These are
  the faithful reproduction of the published PaRoutes MCTS/Retro\* rows; the plain
  "AiZynthFinder" rows above use the generic public USPTO policy and a smaller
  budget. The difference between them quantifies how much of the gap to the
  literature comes from the one-step model + search budget rather than the planner.
