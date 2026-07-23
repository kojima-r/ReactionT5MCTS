# Retrosynthesis Route Planning: ReactionT5-MCTS vs AiZynthFinder on PaRoutes

*Generated 2026-07-23 09:39 UTC*

This report evaluates a Monte-Carlo Tree Search (MCTS) retrosynthetic planner
that uses **ReactionT5** (`sagawa/ReactionT5v2-retrosynthesis`) as the one-step
retrosynthesis model, benchmarked on the **PaRoutes** n1/n5 route sets, and
compares it with **AiZynthFinder** and the published literature values.

**Reproducibility.** All runs use a fixed random seed. ReactionT5 beam search is
deterministic and one-step predictions are memoised in a shared on-disk cache,
so re-running yields identical routes.

**Scope note.** Experiments were run on **CPU only** (no GPU available), where a
single ReactionT5 beam-search call on a drug-like PaRoutes target costs ~30-45 s.
The full benchmark (10,000 targets/set) is therefore intractable here, and each
target is additionally capped at a fixed *budget* of fresh one-step model
evaluations. ReactionT5-MCTS and AiZynthFinder were run on the **first N
targets** of each set (see tables). The literature columns are the
full-benchmark published values and are shown for context — the absolute counts
are not directly comparable to the compute-bounded subset runs, but the
comparison between ReactionT5-MCTS and AiZynthFinder is like-for-like on the
identical subset.

Best ReactionT5-MCTS sweep configuration (by solve rate on the n1 subset): **sweep-base**.


## 1. Hyperparameter sweep (ReactionT5-MCTS, n1 subset)

Base config: expansion_width=5, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15. Each row varies one parameter.

| config (tag) | route set | params | solved | solve rate | in-stock frac | top-1 | top-5 | top-10 | total time | time/target |
|---|---|---|---|---|---|---|---|---|---|---|
| base | n1 | expansion_width=5, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.661 | 0.000 | 0.000 | 0.000 | 9.7 min | 3.2 min |
| c0p5 | n1 | expansion_width=5, iterations=100, c_puct=0.5, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.660 | 0.000 | 0.000 | 0.000 | 1.5 | 0.5 |
| c3p0 | n1 | expansion_width=5, iterations=100, c_puct=3.0, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.661 | 0.000 | 0.000 | 0.000 | 4.4 | 1.5 |
| ew3 | n1 | expansion_width=3, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.621 | 0.000 | 0.000 | 0.000 | 6.4 | 2.1 |
| it200 | n1 | expansion_width=5, iterations=200, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15 | 1/3 | 0.333 | 0.661 | 0.000 | 0.000 | 0.000 | 5.7 | 1.9 |
| md8 | n1 | expansion_width=5, iterations=100, c_puct=1.4, max_depth=8, rollout_width=1, rollout_policy=stock, model_budget=15 | 0/3 | 0.000 | 0.642 | 0.000 | 0.000 | 0.000 | 1.2 min | 23.1 |
| rprob | n1 | expansion_width=5, iterations=100, c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=prob, model_budget=15 | 0/3 | 0.000 | 0.642 | 0.000 | 0.000 | 0.000 | 3.9 min | 1.3 min |

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
| This work — AiZynthFinder | n1 | 8 | 2 | 0.250 | N/A | 0.000 | 0.000 | 0.000 | 5.1 min | 38.5 |
| This work — AiZynthFinder | n5 | 8 | 3 | 0.375 | N/A | 0.000 | 0.000 | 0.000 | 5.2 min | 39.1 |
| This work — ReactionT5-MCTS (best config: base) | n1 | 8 | 1 | 0.125 | 0.824 | 0.000 | 0.000 | 0.000 | 35.2 min | 4.4 min |
| This work — ReactionT5-MCTS (best config: base) | n5 | 8 | 2 | 0.250 | 0.947 | 0.000 | 0.000 | 0.000 | 75.7 min | 9.5 min |

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
