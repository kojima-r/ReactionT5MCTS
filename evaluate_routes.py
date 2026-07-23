"""Evaluate predicted routes with the official PaRoutes route-quality metrics.

Run this inside the `aizynth` conda env, where `route_distances` (the package
providing the tree-edit distance used by PaRoutes) is installed:

    conda run -n aizynth python evaluate_routes.py \
        --routes results/rt5/routes_base_n1.json \
        --references paroutes/data/n1-routes.json \
        --n 20 --method ReactionT5-MCTS --tag base --route-set n1 \
        --out results/eval/rt5_base_n1.json

Metrics reproduce `paroutes/analysis/route_quality.py`:
    * solved targets  - at least one predicted route has all leaves in stock
    * top-1/5/10      - fraction of targets whose reference route is recovered
                         (route distance == 0) within the top-k ranked routes
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "paroutes"))

# --- robustness patch -------------------------------------------------------
# route_distances precomputes child-order permutations only for reactions with
# 1..7 reactants; ReactionT5 occasionally emits reactions with more fragments
# (reagents/salts), which raises KeyError during the tree-edit-distance (used
# only for the strict top-k metric). The solved-target flag is computed
# separately (is_solved) and is unaffected. We patch the permutation table to
# fall back to the identity ordering for >=8 children, which avoids both the
# crash and the factorial blow-up.
import itertools  # noqa: E402
import route_distances.ted.reactiontree as _rtmod  # noqa: E402


class _PermDict(dict):
    def __missing__(self, n):
        v = list(itertools.permutations(range(n), n)) if n <= 7 else [tuple(range(n))]
        self[n] = v
        return v


_rtmod.ReactionTreeWrapper._index_permutations = _PermDict(
    _rtmod.ReactionTreeWrapper._index_permutations
)

from analysis.route_quality import _analyze_routes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--references", required=True)
    ap.add_argument("--n", type=int, default=0, help="limit to first N targets (0 = all)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--method", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--route-set", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.routes) as fh:
        routes_list = json.load(fh)
    with open(args.references) as fh:
        references = json.load(fh)

    n = args.n if args.n > 0 else len(routes_list)
    routes_list = routes_list[:n]
    references = references[:n]

    stats = []
    for routes, reference in zip(routes_list, references):
        stats.append(_analyze_routes(routes, reference, args.ks))

    n_targets = len(stats)
    n_solved = sum(1 for s in stats if s["solved target"])
    metrics = {
        "method": args.method,
        "tag": args.tag,
        "route_set": args.route_set,
        "n_targets": n_targets,
        "solved_targets": n_solved,
        "solve_rate": n_solved / n_targets if n_targets else 0.0,
    }
    for k in args.ks:
        topk = sum(1 for s in stats if s[f"best-{k}"] == 0) / n_targets if n_targets else 0.0
        metrics[f"top{k}"] = topk

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
