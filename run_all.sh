#!/usr/bin/env bash
# End-to-end PaRoutes benchmark pipeline:
#   1. ReactionT5-MCTS hyperparameter sweep (n1 subset)
#   2. official PaRoutes evaluation of each sweep config
#   3. pick best config, run it on n1 + n5 subsets
#   4. run AiZynthFinder on the same n1 + n5 subsets
#   5. official evaluation of best-config + AiZynthFinder runs
#   6. build the Markdown + HTML report
#
# Deterministic (fixed seed). Fresh one-step predictions are memoised in a
# shared SQLite cache (WAL), so the run is resumable and re-runs are fast.
# ReactionT5-MCTS runs on DEVICE with RT5_WORKERS parallel target workers;
# AiZynthFinder is CPU-bound and parallelised over AZF_WORKERS.
set -uo pipefail
cd "$(dirname "$0")"

RT5PY=/home/kojima/miniconda3/envs/reactiont5/bin/python
AZPY=/home/kojima/miniconda3/envs/aizynth/bin/python
CACHE=cache/rt5_b5.sqlite
MAXBEAMS=5
SEED=42
SWEEP_N=3
MAIN_N=10000
SWEEP_BUDGET=15
MAIN_BUDGET=30
DEVICE=cuda
RT5_WORKERS=32
AZF_WORKERS=32
EVAL_WORKERS=32
REFS=paroutes/data
STATUS=results/STATUS.txt
mkdir -p results/rt5 results/aizynth results/eval logs
: > "$STATUS"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

SWEEP_CONFIGS="base ew3 it200 c0p5 c3p0 rprob md8"

say "START pipeline (sweep_n=$SWEEP_N main_n=$MAIN_N budget sweep=$SWEEP_BUDGET main=$MAIN_BUDGET beams=$MAXBEAMS)"

# ------------------------------------------------------------------ 1. sweep
for cfg in $SWEEP_CONFIGS; do
  say "SWEEP run: $cfg"
  $RT5PY -u run_reactiont5.py --route-set n1 --n-targets $SWEEP_N \
      --config configs/$cfg.json --tag sweep-$cfg --out-dir results/rt5 \
      --cache $CACHE --max-beams $MAXBEAMS --model-budget $SWEEP_BUDGET --seed $SEED \
      --device $DEVICE --workers $RT5_WORKERS \
      >> logs/sweep_$cfg.log 2>&1 || say "  sweep $cfg FAILED"
done

# ------------------------------------------------------------- 2. eval sweep
for cfg in $SWEEP_CONFIGS; do
  say "SWEEP eval: $cfg"
  $AZPY -u evaluate_routes.py --routes results/rt5/routes_sweep-${cfg}_n1.json \
      --references $REFS/n1-routes.json --n $SWEEP_N \
      --method ReactionT5-MCTS --tag sweep-$cfg --route-set n1 \
      --workers $EVAL_WORKERS \
      --out results/eval/rt5_sweep-${cfg}_n1.json \
      >> logs/eval_sweep_$cfg.log 2>&1 || say "  eval $cfg FAILED"
done

# --------------------------------------------------------- 3. best on n1/n5
BEST=$($RT5PY select_best.py)
say "BEST config selected: $BEST"
echo "$BEST" > results/best_config.txt
for rs in n1 n5; do
  say "BEST run: $BEST on $rs ($MAIN_N targets)"
  $RT5PY -u run_reactiont5.py --route-set $rs --n-targets $MAIN_N \
      --config configs/$BEST.json --tag best --out-dir results/rt5 \
      --cache $CACHE --max-beams $MAXBEAMS --model-budget $MAIN_BUDGET --seed $SEED \
      --device $DEVICE --workers $RT5_WORKERS \
      >> logs/best_$rs.log 2>&1 || say "  best $rs FAILED"
  say "BEST eval: $rs"
  $AZPY -u evaluate_routes.py --routes results/rt5/routes_best_${rs}.json \
      --references $REFS/${rs}-routes.json --n $MAIN_N \
      --method ReactionT5-MCTS --tag best --route-set $rs \
      --workers $EVAL_WORKERS \
      --out results/eval/rt5_best_${rs}.json >> logs/eval_best_$rs.log 2>&1 \
      || say "  best eval $rs FAILED"
done

# ------------------------------------------------------- 4/5. AiZynthFinder
for rs in n1 n5; do
  say "AZF run: $rs ($MAIN_N targets)"
  $AZPY -u run_aizynthfinder.py --route-set $rs --n-targets $MAIN_N \
      --out-dir results/aizynth --iteration-limit 100 --time-limit 120 --seed $SEED \
      --workers $AZF_WORKERS \
      >> logs/azf_$rs.log 2>&1 || say "  azf $rs FAILED"
  say "AZF eval: $rs"
  $AZPY -u evaluate_routes.py --routes results/aizynth/routes_azf_${rs}.json \
      --references $REFS/${rs}-routes.json --n $MAIN_N \
      --method AiZynthFinder --tag azf --route-set $rs \
      --workers $EVAL_WORKERS \
      --out results/eval/azf_${rs}.json >> logs/eval_azf_$rs.log 2>&1 \
      || say "  azf eval $rs FAILED"
done

# ---------------------------------------------------------------- 6. report
say "REPORT building"
$RT5PY make_report.py >> logs/report.log 2>&1 || say "  report FAILED"
say "DONE"
