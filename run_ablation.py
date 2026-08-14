"""Detailed one-at-a-time ablation for ReactionT5-MCTS.

Holds every hyper-parameter at the baseline and sweeps ONE variable at a time
over several values, on the first N n1 targets.  For each config it records
solve rate, top-k, wall-clock time, and the number of *fresh* one-step model
calls (a cache-independent cost signal).  Results go to a single summary JSON
that make_report.py turns into per-variable plots.

Runs are sequential (each run_reactiont5.py already uses all workers/GPUs).
Resumable: a config whose meta+eval already exist is skipped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RT5PY = "/home/kojima/miniconda3/envs/reactiont5/bin/python"
AZPY = "/home/kojima/miniconda3/envs/aizynth/bin/python"
CACHE = "cache/rt5_b5.sqlite"
N = 3000
ROUTE_SET = "n1"
WORKERS = 32
MAXBEAMS = 5
SEED = 42
CFG_DIR = os.path.join(HERE, "configs", "ablation")
OUT_DIR = os.path.join(HERE, "results", "ablation")
EVAL_DIR = os.path.join(HERE, "results", "ablation_eval")
SUMMARY = os.path.join(OUT_DIR, "summary.json")
STATUS = os.path.join(HERE, "results", "STATUS_ablation.txt")

BASE = {"expansion_width": 5, "iterations": 100, "c_puct": 1.4, "max_depth": 12,
        "rollout_width": 1, "rollout_policy": "stock", "model_budget": 20}

# one-at-a-time sweeps (baseline value included in each)
GRID = {
    "expansion_width": [1, 3, 5, 8, 10],
    "iterations": [25, 50, 100, 200, 400],
    "c_puct": [0.5, 1.0, 1.4, 2.0, 3.0],
    "max_depth": [4, 6, 8, 10, 12],
    "model_budget": [5, 10, 20, 40, 80],
}
# rollout policy/width sweep — expressed as labelled points
ROLLOUT = [("stock", 1, "stock"), ("prob", 1, "prob·rw1"),
           ("prob", 3, "prob·rw3"), ("prob", 5, "prob·rw5")]


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as fh:
        fh.write(line + "\n")


def cfg_key(cfg: dict) -> str:
    """Stable short tag for a config."""
    return "abl-{expansion_width}-{iterations}-{c_puct}-{max_depth}-" \
           "{rollout_width}-{rollout_policy}-{model_budget}".format(**cfg).replace(".", "p")


def run_one(cfg: dict) -> dict:
    """Run + evaluate one config; return metrics dict. Cached/resumable."""
    tag = cfg_key(cfg)
    cfg_path = os.path.join(CFG_DIR, tag + ".json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh)
    meta_path = os.path.join(OUT_DIR, f"meta_{tag}_{ROUTE_SET}.json")
    eval_path = os.path.join(EVAL_DIR, f"{tag}.json")

    if not (os.path.exists(meta_path) and os.path.exists(eval_path)):
        say(f"RUN {tag}")
        rc = subprocess.call([
            RT5PY, "-u", "run_reactiont5.py", "--route-set", ROUTE_SET,
            "--n-targets", str(N), "--config", cfg_path, "--tag", tag,
            "--out-dir", OUT_DIR, "--cache", CACHE, "--max-beams", str(MAXBEAMS),
            "--device", "cuda", "--workers", str(WORKERS), "--seed", str(SEED),
        ], stdout=open(os.path.join(HERE, "logs", f"abl_{tag}.log"), "w"),
           stderr=subprocess.STDOUT)
        if rc != 0:
            say(f"  {tag} RUN FAILED rc={rc}")
        say(f"EVAL {tag}")
        rc = subprocess.call([
            AZPY, "-u", "evaluate_routes.py",
            "--routes", os.path.join(OUT_DIR, f"routes_{tag}_{ROUTE_SET}.json"),
            "--references", f"paroutes/data/{ROUTE_SET}-routes.json", "--n", str(N),
            "--method", "ReactionT5-MCTS", "--tag", tag, "--route-set", ROUTE_SET,
            "--workers", "24", "--out", eval_path,
        ], stdout=open(os.path.join(HERE, "logs", f"abl_eval_{tag}.log"), "w"),
           stderr=subprocess.STDOUT)
        if rc != 0:
            say(f"  {tag} EVAL FAILED rc={rc}")
    else:
        say(f"SKIP {tag} (already done)")

    meta = json.load(open(meta_path))
    ev = json.load(open(eval_path))
    stats = meta.get("model_stats", {})
    nt = meta.get("n_targets", ev.get("n_targets", 1)) or 1
    queries = stats.get("model_calls", 0) + stats.get("cache_hits", 0)
    return {
        "tag": tag, "config": cfg,
        "solve_rate": ev.get("solve_rate", 0.0),
        "top1": ev.get("top1", 0.0), "top5": ev.get("top5", 0.0),
        "top10": ev.get("top10", 0.0),
        "n_targets": ev.get("n_targets", 0),
        "total_time_s": meta.get("total_time_s", 0.0),
        "time_per_target_s": meta.get("mean_time_per_target_s", 0.0),
        "model_calls": stats.get("model_calls", 0),
        # cache-independent search-work signal (total one-step queries / target)
        "queries_per_target": round(queries / nt, 1),
    }


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    open(STATUS, "w").close()
    say(f"START ablation (N={N} {ROUTE_SET}, base={BASE})")

    results = {"meta": {"n_targets": N, "route_set": ROUTE_SET, "base": BASE},
               "variables": {}}
    cache_by_tag = {}

    def get(cfg):
        t = cfg_key(cfg)
        if t not in cache_by_tag:
            cache_by_tag[t] = run_one(cfg)
        return cache_by_tag[t]

    # numeric one-at-a-time sweeps
    for var, values in GRID.items():
        pts = []
        for v in values:
            cfg = dict(BASE); cfg[var] = v
            r = get(cfg)
            pts.append({"value": v, "label": str(v), **r})
        results["variables"][var] = pts
        say(f"DONE variable {var}")

    # rollout policy/width sweep
    pts = []
    for policy, rw, label in ROLLOUT:
        cfg = dict(BASE); cfg["rollout_policy"] = policy; cfg["rollout_width"] = rw
        r = get(cfg)
        pts.append({"value": label, "label": label, **r})
    results["variables"]["rollout"] = pts
    say("DONE variable rollout")

    with open(SUMMARY, "w") as fh:
        json.dump(results, fh, indent=2)
    say(f"WROTE {SUMMARY} ({len(cache_by_tag)} unique configs)")

    # rebuild the report with ablation plots
    say("REPORT rebuild")
    subprocess.call([RT5PY, "make_report.py"],
                    stdout=open(os.path.join(HERE, "logs", "report.log"), "a"),
                    stderr=subprocess.STDOUT)
    say("DONE")


if __name__ == "__main__":
    main()
