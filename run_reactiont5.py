"""Run ReactionT5-MCTS retrosynthesis on a subset of a PaRoutes route set.

Produces a route file in the PaRoutes analysis format (a JSON list with one
entry per target, each entry being a list of ranked route dicts) plus a
per-target metadata JSON with timing and solve information.

Example:
    python run_reactiont5.py --route-set n1 --n-targets 20 \
        --config configs/base.json --tag base --out-dir results/rt5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from rt5mcts.mcts import MCTSConfig, RetroMCTS, count_reactions
from rt5mcts.model import ReactionT5
from rt5mcts.stock import Stock

PAROUTES = os.path.join(os.path.dirname(__file__), "paroutes")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_targets(route_set: str, n: int):
    path = os.path.join(PAROUTES, "data", f"{route_set}-targets.txt")
    with open(path) as fh:
        targets = [line.strip() for line in fh if line.strip()]
    return targets[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-set", choices=["n1", "n5"], required=True)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--config", required=True, help="JSON file with MCTS hyperparameters")
    ap.add_argument("--tag", required=True, help="short name for this run")
    ap.add_argument("--out-dir", default="results/rt5")
    ap.add_argument("--cache", default="cache/reactiont5_shared.sqlite")
    ap.add_argument("--model-budget", type=int, default=-1,
                    help="override config model_budget (fresh calls/target); -1 = keep config")
    ap.add_argument("--max-beams", type=int, default=10)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--augment-stock", dest="augment_stock", action="store_true", default=True,
                    help="treat common reagents/solvents/ions as purchasable (default on)")
    ap.add_argument("--no-augment-stock", dest="augment_stock", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as fh:
        cfg_dict = json.load(fh)
    cfg_dict.pop("seed", None)  # seed is controlled by --seed
    if args.model_budget >= 0:
        cfg_dict["model_budget"] = args.model_budget

    targets = load_targets(args.route_set, args.n_targets)
    stock_path = os.path.join(PAROUTES, "data", f"{args.route_set}-stock.txt")
    with open(stock_path) as fh:
        stock_lines = [l for l in fh if l.strip()]
    if args.augment_stock:
        from rt5mcts.reagents import reagent_smiles
        stock_lines = stock_lines + reagent_smiles()
    stock = Stock(stock_lines)
    print(f"[{args.tag}/{args.route_set}] {len(targets)} targets, stock size {len(stock)} "
          f"(augment={args.augment_stock})")

    model = ReactionT5(
        cache_path=args.cache,
        max_beams=args.max_beams,
        num_threads=args.num_threads,
        seed=args.seed,
    )

    all_routes = []
    meta = []
    t_all = time.time()
    for i, tgt in enumerate(targets):
        cfg = MCTSConfig(seed=args.seed, **cfg_dict)
        mcts = RetroMCTS(model, stock, cfg)
        t0 = time.time()
        try:
            mcts.search(tgt)
        except Exception as exc:  # keep going on a single-target failure
            print(f"  target {i}: ERROR {type(exc).__name__}: {exc}")
            all_routes.append([{"type": "mol", "smiles": tgt, "in_stock": False}])
            meta.append({"index": i, "smiles": tgt, "solved": False,
                         "n_routes": 0, "time_s": time.time() - t0, "error": str(exc)})
            continue
        dt = time.time() - t0
        routes = mcts.ranked_routes(10)
        solved = len(routes) > 0
        if not routes:
            routes = [{"type": "mol", "smiles": tgt, "in_stock": False}]
        all_routes.append(routes)
        best_nrxn = min((count_reactions_of(r) for r in routes), default=0)
        meta.append({"index": i, "smiles": tgt, "solved": solved,
                     "n_routes": len(mcts.solutions), "best_n_reactions": best_nrxn,
                     "best_instock_frac": round(mcts.best_frac, 3), "time_s": dt})
        print(f"  target {i:3d}: solved={solved} routes={len(mcts.solutions):2d} "
              f"time={dt:6.1f}s model_calls={model.stats()['model_calls']}")

    total_time = time.time() - t_all
    routes_path = os.path.join(args.out_dir, f"routes_{args.tag}_{args.route_set}.json")
    meta_path = os.path.join(args.out_dir, f"meta_{args.tag}_{args.route_set}.json")
    with open(routes_path, "w") as fh:
        json.dump(all_routes, fh)
    n_solved = sum(m["solved"] for m in meta)
    summary = {
        "tag": args.tag, "route_set": args.route_set, "method": "ReactionT5-MCTS",
        "n_targets": len(targets), "n_solved": n_solved,
        "solve_rate": n_solved / len(targets) if targets else 0.0,
        "mean_best_instock_frac": round(
            sum(m.get("best_instock_frac", 0.0) for m in meta) / len(meta), 3
        ) if meta else 0.0,
        "total_time_s": total_time,
        "mean_time_per_target_s": total_time / len(targets) if targets else 0.0,
        "config": cfg_dict, "model_stats": model.stats(),
        "seed": args.seed, "per_target": meta,
    }
    with open(meta_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    model.close()
    print(f"[{args.tag}/{args.route_set}] solved {n_solved}/{len(targets)} "
          f"in {total_time:.0f}s -> {routes_path}")


def count_reactions_of(route_dict: dict) -> int:
    n = 0
    stack = [route_dict]
    while stack:
        node = stack.pop()
        for child in node.get("children", []):
            if child.get("type") == "reaction":
                n += 1
                stack.extend(child.get("children", []))
            else:
                stack.append(child)
    return n


if __name__ == "__main__":
    main()
