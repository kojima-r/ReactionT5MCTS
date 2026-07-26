"""Run ReactionT5-MCTS retrosynthesis on a subset of a PaRoutes route set.

Produces a route file in the PaRoutes analysis format (a JSON list with one
entry per target, each entry being a list of ranked route dicts) plus a
per-target metadata JSON with timing and solve information.

Targets are independent, so the run can be parallelised over worker processes
(--workers) and the model can run on GPU (--device cuda); with several GPUs
the workers are spread round-robin across them.  All workers share one SQLite
prediction cache (WAL mode).

Example:
    python run_reactiont5.py --route-set n1 --n-targets 20 \
        --config configs/base.json --tag base --out-dir results/rt5 \
        --device cuda --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch

from rt5mcts.mcts import MCTSConfig, RetroMCTS, count_reactions
from rt5mcts.model import ReactionT5
from rt5mcts.stock import Stock

PAROUTES = os.path.join(os.path.dirname(__file__), "paroutes")

# worker-process globals; the stock is installed before fork (copy-on-write),
# the model is created per worker in _worker_init (CUDA must init after fork)
_G: dict = {}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_targets(route_set: str, n: int):
    path = os.path.join(PAROUTES, "data", f"{route_set}-targets.txt")
    with open(path) as fh:
        targets = [line.strip() for line in fh if line.strip()]
    return targets[:n]


def resolve_devices(device: str, workers: int) -> list:
    """Expand --device into the per-worker device list."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        n = torch.cuda.device_count()
        if n == 0:
            return ["cpu"]
        return [f"cuda:{i}" for i in range(min(n, max(workers, 1)))]
    return [device]


def _make_model(args, device: str, num_threads: int) -> ReactionT5:
    return ReactionT5(
        cache_path=args.cache,
        max_beams=args.max_beams,
        device=device,
        num_threads=num_threads,
        seed=args.seed,
    )


def _worker_init(args, devices, num_threads, counter):
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    _G["model"] = _make_model(args, devices[idx % len(devices)], num_threads)


def _run_target(job):
    """Search one target; returns (index, routes, meta, fresh_calls, cache_hits)."""
    i, tgt, cfg_dict, seed = job
    model: ReactionT5 = _G["model"]
    stock: Stock = _G["stock"]
    calls0, hits0 = model.n_model_calls, model.n_cache_hits
    cfg = MCTSConfig(seed=seed, **cfg_dict)
    mcts = RetroMCTS(model, stock, cfg)
    t0 = time.time()
    try:
        mcts.search(tgt)
    except Exception as exc:  # keep going on a single-target failure
        return (i, [{"type": "mol", "smiles": tgt, "in_stock": False}],
                {"index": i, "smiles": tgt, "solved": False, "n_routes": 0,
                 "time_s": time.time() - t0, "error": str(exc)},
                model.n_model_calls - calls0, model.n_cache_hits - hits0)
    dt = time.time() - t0
    routes = mcts.ranked_routes(10)
    solved = len(routes) > 0
    if not routes:
        routes = [{"type": "mol", "smiles": tgt, "in_stock": False}]
    best_nrxn = min((count_reactions_of(r) for r in routes), default=0)
    meta = {"index": i, "smiles": tgt, "solved": solved,
            "n_routes": len(mcts.solutions), "best_n_reactions": best_nrxn,
            "best_instock_frac": round(mcts.best_frac, 3), "time_s": dt}
    return (i, routes, meta,
            model.n_model_calls - calls0, model.n_cache_hits - hits0)


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
    ap.add_argument("--device", default="cpu",
                    help="torch device for the model: cpu, cuda, cuda:N or auto; "
                         "with --workers>1 and several GPUs, workers are spread round-robin")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes over targets (1 = serial)")
    ap.add_argument("--target-timeout", type=int, default=1800,
                    help="max seconds to wait for one target's result in parallel "
                         "mode; a lost task (crashed worker) is recorded as failed "
                         "instead of hanging the whole run")
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
    _G["stock"] = stock

    workers = max(1, min(args.workers, len(targets))) if targets else 1
    devices = resolve_devices(args.device, workers)
    threads_per_worker = (args.num_threads if workers == 1
                          else max(1, min(args.num_threads, (os.cpu_count() or workers) // workers)))
    print(f"[{args.tag}/{args.route_set}] {len(targets)} targets, stock size {len(stock)} "
          f"(augment={args.augment_stock}) workers={workers} devices={devices} "
          f"threads/worker={threads_per_worker}")

    jobs = [(i, tgt, cfg_dict, args.seed) for i, tgt in enumerate(targets)]
    all_routes = [None] * len(targets)
    meta = [None] * len(targets)
    n_model_calls = 0
    n_cache_hits = 0
    t_all = time.time()

    if workers == 1:
        _G["model"] = _make_model(args, devices[0], threads_per_worker)
        results = map(_run_target, jobs)
        for i, routes, m, calls, hits in results:
            all_routes[i], meta[i] = routes, m
            n_model_calls += calls
            n_cache_hits += hits
            print(f"  target {i:3d}: solved={m['solved']} routes={m['n_routes']:2d} "
                  f"time={m['time_s']:6.1f}s model_calls={calls}")
        _G["model"].close()
    else:
        ctx = mp.get_context("fork")  # CUDA is only initialised inside workers
        counter = ctx.Value("i", 0)
        with ctx.Pool(workers, initializer=_worker_init,
                      initargs=(args, devices, threads_per_worker, counter)) as pool:
            # apply_async + get(timeout) instead of imap: if a worker dies
            # (segfault/OOM) its in-flight task is lost and a bare imap would
            # wait on it forever. Results are collected in submission order, so
            # by the time we wait on result i most of it is already done and
            # the timeout only has to cover one straggler target.
            async_results = [pool.apply_async(_run_target, (job,)) for job in jobs]
            for j, ar in enumerate(async_results):
                try:
                    i, routes, m, calls, hits = ar.get(timeout=args.target_timeout)
                except Exception as exc:  # mp.TimeoutError or a dead worker
                    i = jobs[j][0]
                    routes = [{"type": "mol", "smiles": jobs[j][1], "in_stock": False}]
                    m = {"index": i, "smiles": jobs[j][1], "solved": False,
                         "n_routes": 0, "time_s": float(args.target_timeout),
                         "error": f"lost/timed-out task: {type(exc).__name__}"}
                    calls = hits = 0
                all_routes[i], meta[i] = routes, m
                n_model_calls += calls
                n_cache_hits += hits
                print(f"  target {i:3d}: solved={m['solved']} routes={m['n_routes']:2d} "
                      f"time={m['time_s']:6.1f}s model_calls={calls}", flush=True)

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
        "config": cfg_dict,
        "model_stats": {"model_calls": n_model_calls, "cache_hits": n_cache_hits},
        "device": args.device, "workers": workers,
        "seed": args.seed, "per_target": meta,
    }
    with open(meta_path, "w") as fh:
        json.dump(summary, fh, indent=2)
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
