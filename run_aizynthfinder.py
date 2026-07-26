"""Run AiZynthFinder retrosynthesis on a subset of a PaRoutes route set.

Run inside the `aizynth` conda env.  Uses the public USPTO expansion policy
(from ``download_public_data``) together with the PaRoutes stock so that the
"solved" criterion matches the PaRoutes reference routes.  Output is written in
the same PaRoutes route format as the ReactionT5-MCTS driver, plus a metadata
file with timing.  Targets are independent, so the run can be parallelised
over worker processes (--workers); each worker owns one AiZynthFinder
instance (the expansion policy is CPU-bound ONNX).

    conda run -n aizynth python run_aizynthfinder.py \
        --route-set n1 --n-targets 20 --out-dir results/aizynth --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAROUTES = os.path.join(HERE, "paroutes")

_G: dict = {}  # per-worker AiZynthFinder instance


def build_config(route_set: str, iteration_limit: int, time_limit: int) -> dict:
    """Assemble an AiZynthFinder config dict (public USPTO policy + PaRoutes stock)."""
    data = os.path.join(HERE, "aizynth_data")
    stock_file = os.path.join(PAROUTES, "publication", f"aizynth_{route_set}_stock.txt")
    return {
        "expansion": {
            "uspto": [
                os.path.join(data, "uspto_model.onnx"),
                os.path.join(data, "uspto_templates.csv.gz"),
            ]
        },
        "stock": {"paroutes": stock_file},
        "search": {
            "algorithm": "mcts",
            "max_transforms": 10,
            "iteration_limit": iteration_limit,
            "time_limit": time_limit,
        },
    }


def _make_finder(cfg: dict):
    from aizynthfinder.aizynthfinder import AiZynthFinder

    finder = AiZynthFinder(configdict=cfg)
    finder.stock.select("paroutes")
    finder.expansion_policy.select("uspto")
    return finder


def _worker_init(cfg, seed):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    _G["finder"] = _make_finder(cfg)


def _run_target(job):
    """Search one target; returns (index, routes, meta)."""
    i, tgt = job
    finder = _G["finder"]
    finder.target_smiles = tgt
    t0 = time.time()
    finder.tree_search()
    finder.build_routes()
    dt = time.time() - t0
    route_dicts = [r["reaction_tree"].to_dict() for r in finder.routes]
    # a target is solved if any extracted route has all leaves in stock
    solved = any(_route_solved(rd) for rd in route_dicts) if route_dicts else False
    if not route_dicts:
        route_dicts = [{"type": "mol", "smiles": tgt, "in_stock": False}]
    meta = {"index": i, "smiles": tgt, "solved": solved,
            "n_routes": len(finder.routes), "time_s": dt}
    return (i, route_dicts[:10], meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-set", choices=["n1", "n5"], required=True)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--out-dir", default="results/aizynth")
    ap.add_argument("--iteration-limit", type=int, default=100)
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes over targets (1 = serial)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_config(args.route_set, args.iteration_limit, args.time_limit)

    with open(os.path.join(PAROUTES, "data", f"{args.route_set}-targets.txt")) as fh:
        targets = [l.strip() for l in fh if l.strip()][: args.n_targets]

    workers = max(1, min(args.workers, len(targets))) if targets else 1
    print(f"[aizynth/{args.route_set}] {len(targets)} targets, workers={workers}")

    jobs = list(enumerate(targets))
    all_routes = [None] * len(targets)
    meta = [None] * len(targets)
    t_all = time.time()

    if workers == 1:
        _worker_init(cfg, args.seed)
        print(f"[aizynth/{args.route_set}] stock {len(_G['finder'].stock)}")
        results = map(_run_target, jobs)
        for i, routes, m in results:
            all_routes[i], meta[i] = routes, m
            print(f"  target {i:3d}: solved={m['solved']} routes={m['n_routes']:2d} "
                  f"time={m['time_s']:6.1f}s")
    else:
        ctx = mp.get_context("fork")
        # apply_async + get(timeout): a crashed worker loses its in-flight task
        # and a bare imap would wait on it forever (see run_reactiont5.py).
        wait_limit = args.time_limit + 600
        with ctx.Pool(workers, initializer=_worker_init, initargs=(cfg, args.seed)) as pool:
            async_results = [pool.apply_async(_run_target, (job,)) for job in jobs]
            for j, ar in enumerate(async_results):
                try:
                    i, routes, m = ar.get(timeout=wait_limit)
                except Exception as exc:  # mp.TimeoutError or a dead worker
                    i, tgt = jobs[j]
                    routes = [{"type": "mol", "smiles": tgt, "in_stock": False}]
                    m = {"index": i, "smiles": tgt, "solved": False, "n_routes": 0,
                         "time_s": float(wait_limit),
                         "error": f"lost/timed-out task: {type(exc).__name__}"}
                all_routes[i], meta[i] = routes, m
                print(f"  target {i:3d}: solved={m['solved']} routes={m['n_routes']:2d} "
                      f"time={m['time_s']:6.1f}s", flush=True)

    total_time = time.time() - t_all
    routes_path = os.path.join(args.out_dir, f"routes_azf_{args.route_set}.json")
    meta_path = os.path.join(args.out_dir, f"meta_azf_{args.route_set}.json")
    with open(routes_path, "w") as fh:
        json.dump(all_routes, fh)
    n_solved = sum(m["solved"] for m in meta)
    summary = {
        "tag": "azf", "route_set": args.route_set, "method": "AiZynthFinder",
        "n_targets": len(targets), "n_solved": n_solved,
        "solve_rate": n_solved / len(targets) if targets else 0.0,
        "total_time_s": total_time,
        "mean_time_per_target_s": total_time / len(targets) if targets else 0.0,
        "config": {"algorithm": "mcts", "iteration_limit": args.iteration_limit,
                   "time_limit": args.time_limit, "policy": "public USPTO",
                   "stock": f"PaRoutes {args.route_set}"},
        "workers": workers,
        "seed": args.seed, "per_target": meta,
    }
    with open(meta_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[aizynth/{args.route_set}] solved {n_solved}/{len(targets)} "
          f"in {total_time:.0f}s -> {routes_path}")


def _route_solved(node: dict) -> bool:
    """True if every leaf molecule in the route dict is in stock."""
    if node.get("type") == "mol":
        children = node.get("children")
        if not children:
            return bool(node.get("in_stock", False))
        return all(_route_solved(c) for c in children)
    # reaction node
    return all(_route_solved(c) for c in node.get("children", []))


if __name__ == "__main__":
    main()
