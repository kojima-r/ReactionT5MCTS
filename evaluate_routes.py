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

import signal  # noqa: E402

from analysis.route_quality import _analyze_routes, is_solved  # noqa: E402

# fork-inherited (copy-on-write) worker data: routes/references stay in the
# parent's memory and workers receive only integer indices — never pickle the
# (potentially huge) route lists through the task queue.
_EVAL = {"routes": None, "refs": None, "ks": [1, 5, 10], "timeout": 60}


class _EvalTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _EvalTimeout()


def _analyze_one(i):
    """Analyze one target with a wall-clock cap on the tree-edit-distance.

    The TED permutation count grows factorially with reaction arity, and a few
    ReactionT5 routes make the exact metric intractable.  On timeout we keep
    the (cheap) solved flag and report the reference route as not recovered —
    the same outcome the strict metric would almost surely give.
    """
    routes, ref, ks = _EVAL["routes"][i], _EVAL["refs"][i], _EVAL["ks"]
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(_EVAL["timeout"])
    try:
        return _analyze_routes(routes, ref, ks)
    except _EvalTimeout:
        stats = {"solved target": any(is_solved(r) for r in routes), "timeout": True}
        for k in ks:
            stats[f"best-{k}"] = 1e6
        return stats
    finally:
        signal.alarm(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--references", required=True)
    ap.add_argument("--n", type=int, default=0, help="limit to first N targets (0 = all)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes over targets (1 = serial)")
    ap.add_argument("--target-timeout", type=int, default=60,
                    help="seconds allowed per target for the strict TED metric; "
                         "on timeout the target keeps its solved flag but counts "
                         "as not recovering the reference route")
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

    _EVAL["routes"], _EVAL["refs"], _EVAL["ks"] = routes_list, references, args.ks
    _EVAL["timeout"] = args.target_timeout
    n_items = len(routes_list)
    workers = max(1, min(args.workers, n_items)) if n_items else 1
    if workers == 1:
        stats = [_analyze_one(i) for i in range(n_items)]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")  # children inherit _EVAL and the perm patch
        with ctx.Pool(workers) as pool:
            stats = []
            for s in pool.imap(_analyze_one, range(n_items), chunksize=16):
                stats.append(s)
                if len(stats) % 1000 == 0:
                    print(f"  evaluated {len(stats)}/{n_items}", flush=True)

    n_targets = len(stats)
    n_solved = sum(1 for s in stats if s["solved target"])
    metrics = {
        "method": args.method,
        "tag": args.tag,
        "route_set": args.route_set,
        "n_targets": n_targets,
        "solved_targets": n_solved,
        "solve_rate": n_solved / n_targets if n_targets else 0.0,
        "ted_timeouts": sum(1 for s in stats if s.get("timeout")),
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
