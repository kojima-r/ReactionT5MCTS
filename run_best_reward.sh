#!/usr/bin/env bash
# Full-benchmark run of ReactionT5-MCTS with the BEST reward (retrek: the six
# ReTReK-style knowledge scores), on n1 + n5 (10000 targets each), then official
# evaluation and report rebuild.  MCTS hyper-parameters match the existing
# "best config" row (base: expansion_width=5, iterations=100, c_puct=1.4,
# max_depth=12, model_budget=30) so the ONLY difference is the reward — making
# the new row a like-for-like "reward upgrade" of the ReactionT5-MCTS baseline.
set -uo pipefail
cd "$(dirname "$0")"

RT5PY=/home/kojima/miniconda3/envs/reactiont5/bin/python
AZPY=/home/kojima/miniconda3/envs/aizynth/bin/python
CACHE=cache/rt5_b5.sqlite
MAIN_N=10000
BUDGET=30
MAXBEAMS=5
SEED=42
WORKERS=32
EVAL_WORKERS=24
REFS=paroutes/data
STATUS=results/STATUS_bestrwd.txt
mkdir -p results/rt5 results/eval logs
: > "$STATUS"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

say "START best-reward run (retrek, n=$MAIN_N budget=$BUDGET)"
for rs in n1 n5; do
  say "BESTRWD run: $rs ($MAIN_N targets, reward=retrek)"
  $RT5PY -u run_reactiont5.py --route-set $rs --n-targets $MAIN_N \
      --config configs/base_retrek.json --tag bestrwd --out-dir results/rt5 \
      --cache $CACHE --max-beams $MAXBEAMS --model-budget $BUDGET --seed $SEED \
      --device cuda --workers $WORKERS --reward-policy retrek \
      >> logs/bestrwd_$rs.log 2>&1 || say "  bestrwd $rs FAILED"
  say "BESTRWD eval: $rs"
  $AZPY -u evaluate_routes.py --routes results/rt5/routes_bestrwd_${rs}.json \
      --references $REFS/${rs}-routes.json --n $MAIN_N \
      --method ReactionT5-MCTS --tag bestrwd --route-set $rs --workers $EVAL_WORKERS \
      --out results/eval/rt5_bestrwd_${rs}.json >> logs/eval_bestrwd_$rs.log 2>&1 \
      || say "  bestrwd eval $rs FAILED"
done

say "REPORT rebuild"
$RT5PY make_report.py >> logs/report.log 2>&1 || say "  report FAILED"
say "DONE"
