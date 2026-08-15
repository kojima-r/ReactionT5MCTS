"""Compare reward functions and weight settings for ReactionT5-MCTS.

Runs the planner on a fixed n1 subset under several reward configurations
(different rewards, and different weights of the same reward), evaluates each
with the official PaRoutes metrics, and additionally measures *what kind of
top-ranked routes each reward produces* — mean predicted yield, synthetic-
accessibility ease, convergence (CDScore) and route length.  The reward changes
the value that guides selection and ranking, so these route-property columns are
the substance of the comparison (solve rate is largely reward-independent — it
is driven by the stock-greedy rollout — whereas the ranked top route is not).

Writes results/reward_exp/summary.json (consumed by make_reward_report.py).
Sequential runs (each already uses all workers/GPUs). Resumable.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RT5PY = "/home/kojima/miniconda3/envs/reactiont5/bin/python"
AZPY = "/home/kojima/miniconda3/envs/aizynth/bin/python"
CACHE = "cache/rt5_b5.sqlite"
YIELD_CACHE = "cache/rt5_yield.sqlite"
N = int(os.environ.get("REWARD_EXP_N", "2000"))  # override for quick tests
ROUTE_SET = os.environ.get("REWARD_EXP_SET", "n1")  # n1 | n5
WORKERS = 32
MAXBEAMS = 5
SEED = 42
CFG_DIR = os.path.join(HERE, "configs", "reward_exp")
OUT_DIR = os.path.join(HERE, "results", "reward_exp")
EVAL_DIR = os.path.join(OUT_DIR, "eval")
SUMMARY = os.path.join(OUT_DIR, f"summary_{ROUTE_SET}.json")
STATUS = os.path.join(HERE, "results", f"STATUS_reward_{ROUTE_SET}.txt")

BASE = {"expansion_width": 5, "iterations": 100, "c_puct": 1.4, "max_depth": 12,
        "rollout_width": 1, "rollout_policy": "stock", "model_budget": 20}

# (label, group, reward overrides merged onto BASE)
#   group is used only to organise the report.
REWARD_CONFIGS = [
    ("stock",        "reward",  {"reward_policy": "stock"}),
    ("yield",        "reward",  {"reward_policy": "yield", "yield_weight": 1.0}),
    ("sa",           "reward",  {"reward_policy": "sa"}),
    ("retrek",       "reward",  {"reward_policy": "retrek"}),
    ("stock+yield",  "reward",  {"reward_policy": "composite", "reward_combine": "product",
                                 "reward_terms": {"in_stock": 1, "shallow": 1, "yield": 1}}),
    # yield_weight sweep (weight-setting effect on the same reward)
    ("yield_w0.5",   "weight",  {"reward_policy": "yield", "yield_weight": 0.5}),
    ("yield_w2",     "weight",  {"reward_policy": "yield", "yield_weight": 2.0}),
    ("yield_w4",     "weight",  {"reward_policy": "yield", "yield_weight": 4.0}),
    # retrek term-weight variants (weight-setting effect within a composite)
    ("retrek_cd3",   "weight",  {"reward_policy": "retrek", "reward_combine": "sum",
                                 "reward_terms": {"cdscore": 3, "asscore": 1, "rdscore": 1,
                                                  "stscore": 1, "intermediate": 1, "template": 1}}),
    ("retrek_tmpl3", "weight",  {"reward_policy": "retrek", "reward_combine": "sum",
                                 "reward_terms": {"cdscore": 1, "asscore": 1, "rdscore": 1,
                                                  "stscore": 1, "intermediate": 1, "template": 3}}),
]


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# route-dict property analysis (PaRoutes format) — reward-independent
# --------------------------------------------------------------------------- #
def _dict_steps(node):
    """Yield (product_smiles, [reactant_smiles, ...]) for each reaction in a route dict."""
    stack = [node]
    while stack:
        m = stack.pop()
        for rxn in m.get("children", []):
            reactants = rxn.get("children", [])
            yield m["smiles"], [r["smiles"] for r in reactants]
            stack.extend(reactants)


def _dict_leaves(node):
    out, stack = [], [node]
    while stack:
        m = stack.pop()
        ch = m.get("children", [])
        if not ch:
            out.append(m["smiles"])
        else:
            stack.extend(ch[0].get("children", []))
    return out


def _n_reactions_dict(node):
    return sum(1 for _ in _dict_steps(node))


def analyze_top_routes(routes_path, ym, sa_fn, desc_fn):
    """Mean route-property metrics over the top route of each solved target."""
    routes = json.load(open(routes_path))
    ny, nyv, sa, cd, nr, cnt = 0.0, 0, 0.0, 0.0, 0.0, 0
    for entry in routes:
        top = entry[0] if isinstance(entry, list) else entry
        steps = list(_dict_steps(top))
        if not steps:  # unsolved (single mol) — skip
            continue
        cnt += 1
        nr += len(steps)
        # yield: geomean over steps
        ys = [ym.predict_yield(p, r) for p, r in steps]
        nyv += math.exp(sum(math.log(max(v, 1e-6)) for v in ys) / len(ys))
        ny += 1
        # convergence (cdscore): mean over steps of min/max heavy-atom balance
        cds = []
        for _p, reacts in steps:
            sizes = [h for h in (desc_fn(r)[0] for r in reacts) if h > 0]
            cds.append(min(sizes) / max(sizes) if len(sizes) >= 2 else 0.0)
        cd += sum(cds) / len(cds)
        # SA ease over leaves
        lv = _dict_leaves(top)
        sa += sum(sa_fn(s) for s in lv) / len(lv) if lv else 0.0
    if cnt == 0:
        return {"n_solved_analyzed": 0}
    return {"n_solved_analyzed": cnt,
            "mean_n_reactions": round(nr / cnt, 3),
            "mean_top_yield": round(nyv / ny, 4) if ny else None,
            "mean_top_sa_ease": round(sa / cnt, 4),
            "mean_top_cdscore": round(cd / cnt, 4)}


def run_one(label, group, overrides, ym, sa_fn, desc_fn):
    cfg = dict(BASE); cfg.update(overrides)
    tag = "rex-" + label.replace("+", "_").replace(".", "p")
    cfg_path = os.path.join(CFG_DIR, tag + ".json")
    json.dump(cfg, open(cfg_path, "w"))
    routes_path = os.path.join(OUT_DIR, f"routes_{tag}_{ROUTE_SET}.json")
    meta_path = os.path.join(OUT_DIR, f"meta_{tag}_{ROUTE_SET}.json")
    eval_path = os.path.join(EVAL_DIR, f"{tag}_{ROUTE_SET}.json")

    if not (os.path.exists(meta_path) and os.path.exists(eval_path)):
        say(f"RUN {label} ({tag})")
        rc = subprocess.call([
            RT5PY, "-u", "run_reactiont5.py", "--route-set", ROUTE_SET,
            "--n-targets", str(N), "--config", cfg_path, "--tag", tag,
            "--out-dir", OUT_DIR, "--cache", CACHE, "--max-beams", str(MAXBEAMS),
            "--device", "cuda", "--workers", str(WORKERS), "--seed", str(SEED),
            "--yield-cache", YIELD_CACHE,
        ], stdout=open(os.path.join(HERE, "logs", f"rex_{tag}.log"), "w"),
           stderr=subprocess.STDOUT)
        if rc != 0:
            say(f"  {label} RUN FAILED rc={rc}")
        say(f"EVAL {label}")
        rc = subprocess.call([
            AZPY, "-u", "evaluate_routes.py", "--routes", routes_path,
            "--references", f"paroutes/data/{ROUTE_SET}-routes.json", "--n", str(N),
            "--method", "ReactionT5-MCTS", "--tag", tag, "--route-set", ROUTE_SET,
            "--workers", "24", "--out", eval_path,
        ], stdout=open(os.path.join(HERE, "logs", f"rex_eval_{tag}.log"), "w"),
           stderr=subprocess.STDOUT)
        if rc != 0:
            say(f"  {label} EVAL FAILED rc={rc}")
    else:
        say(f"SKIP {label} (already done)")

    meta = json.load(open(meta_path))
    ev = json.load(open(eval_path))
    stats = meta.get("model_stats", {})
    nt = meta.get("n_targets", 1) or 1
    say(f"ANALYZE {label} (top-route properties)")
    props = analyze_top_routes(routes_path, ym, sa_fn, desc_fn)
    return {"label": label, "group": group, "config": overrides,
            "solve_rate": ev.get("solve_rate", 0.0),
            "top1": ev.get("top1", 0.0), "top5": ev.get("top5", 0.0),
            "top10": ev.get("top10", 0.0),
            "queries_per_target": round((stats.get("model_calls", 0)
                                         + stats.get("cache_hits", 0)) / nt, 1),
            "time_per_target_s": meta.get("mean_time_per_target_s", 0.0),
            **props}


def main():
    for d in (CFG_DIR, OUT_DIR, EVAL_DIR, os.path.join(HERE, "logs")):
        os.makedirs(d, exist_ok=True)
    open(STATUS, "w").close()
    say(f"START reward experiment (N={N} {ROUTE_SET}, {len(REWARD_CONFIGS)} configs)")

    # shared analysis helpers (yield model + SA + descriptors), all cached
    from rt5mcts.yield_model import ReactionT5Yield
    from rt5mcts.reward import _sa_ease, RewardContext, _desc
    ym = ReactionT5Yield(cache_path=YIELD_CACHE, device="cuda", seed=SEED)
    _ctx = RewardContext()
    sa_fn = lambda s: _sa_ease(s)
    desc_fn = lambda s: _desc(s, _ctx)

    rows = []
    for label, group, overrides in REWARD_CONFIGS:
        rows.append(run_one(label, group, overrides, ym, sa_fn, desc_fn))
        say(f"DONE {label}: solve={rows[-1]['solve_rate']:.3f} "
            f"top1={rows[-1]['top1']:.3f} "
            f"yield={rows[-1].get('mean_top_yield')} "
            f"cd={rows[-1].get('mean_top_cdscore')}")

    json.dump({"meta": {"n_targets": N, "route_set": ROUTE_SET, "base": BASE},
               "rows": rows}, open(SUMMARY, "w"), indent=2)
    say(f"WROTE {SUMMARY}")
    say("REPORT building")
    subprocess.call([RT5PY, "make_reward_report.py"],
                    stdout=open(os.path.join(HERE, "logs", "reward_report.log"), "w"),
                    stderr=subprocess.STDOUT)
    say("DONE")


if __name__ == "__main__":
    main()
