#!/usr/bin/env bash
# Retro* reproduction: AiZynthFinder with the retrostar search algorithm and the
# PaRoutes retrostar_value_model.pickle molecule-cost model. Same public USPTO
# expansion policy, PaRoutes stock, and search budget (iteration_limit=100,
# time_limit=120) as the AiZynthFinder-MCTS runs, so the two are like-for-like.
# Runs n1 then n5 on the full 10,000-target sets, evaluating each with the
# official PaRoutes metrics, then rebuilds the report.
set -uo pipefail
cd "$(dirname "$0")"

AZPY=/home/kojima/miniconda3/envs/aizynth/bin/python
RT5PY=/home/kojima/miniconda3/envs/reactiont5/bin/python
MAIN_N=10000
SEED=42
WORKERS=32
EVAL_WORKERS=24
VALUE_MODEL=aizynth_data/retrostar_value_model.pickle
REFS=paroutes/data
METHOD="AiZynthFinder-Retro*"
STATUS=results/STATUS_retrostar.txt
mkdir -p results/aizynth results/eval logs
: > "$STATUS"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

say "START Retro* reproduction (n=$MAIN_N budget iter=100 time=120 workers=$WORKERS)"

for rs in n1 n5; do
  say "RETROSTAR run: $rs ($MAIN_N targets)"
  $AZPY -u run_aizynthfinder.py --route-set $rs --n-targets $MAIN_N \
      --out-dir results/aizynth --iteration-limit 100 --time-limit 120 --seed $SEED \
      --workers $WORKERS --algorithm retrostar --value-model $VALUE_MODEL \
      --tag retrostar --method "$METHOD" \
      >> logs/retrostar_$rs.log 2>&1 || say "  retrostar $rs FAILED"
  say "RETROSTAR eval: $rs"
  $AZPY -u evaluate_routes.py --routes results/aizynth/routes_retrostar_${rs}.json \
      --references $REFS/${rs}-routes.json --n $MAIN_N \
      --method "$METHOD" --tag retrostar --route-set $rs \
      --workers $EVAL_WORKERS \
      --out results/eval/retrostar_${rs}.json >> logs/eval_retrostar_$rs.log 2>&1 \
      || say "  retrostar eval $rs FAILED"
done

say "REPORT building"
$RT5PY make_report.py >> logs/report.log 2>&1 || say "  report FAILED"
say "DONE"
