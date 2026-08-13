#!/usr/bin/env bash
# Faithful literature reproduction with the PaRoutes route-set-specific one-step
# model (uspto_rxn_n{1,5}, Keras->ONNX) and the literature search budget
# (iteration_limit=500, time_limit=3600). Runs MCTS first (fast, ~15h for
# n1+n5), then Retro* (slow, ~5 days for n1+n5), evaluating each and rebuilding
# the report after every eval so results land incrementally.
#   MCTS   : literature model + PaRoutes stock
#   Retro* : + retrostar_value_model.pickle molecule cost
set -uo pipefail
cd "$(dirname "$0")"

AZPY=/home/kojima/miniconda3/envs/aizynth/bin/python
RT5PY=/home/kojima/miniconda3/envs/reactiont5/bin/python
MAIN_N=10000
SEED=42
WORKERS=32
EVAL_WORKERS=24
ITER=500
TIME=3600
DATA=aizynth_data
REFS=paroutes/data
VALUE_MODEL=$DATA/retrostar_value_model.pickle
STATUS=results/STATUS_literature.txt
mkdir -p results/aizynth results/eval logs
: > "$STATUS"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

model_of(){ echo "$DATA/uspto_rxn_${1}_model.onnx"; }
tmpl_of(){ echo "$REFS/uspto_rxn_${1}_unique_templates.hdf5"; }

run_one(){  # algo tag method extra_args
  local algo=$1 tag=$2 method=$3; shift 3
  for rs in n1 n5; do
    say "$tag run: $rs ($MAIN_N targets, iter=$ITER time=$TIME)"
    $AZPY -u run_aizynthfinder.py --route-set $rs --n-targets $MAIN_N \
        --out-dir results/aizynth --iteration-limit $ITER --time-limit $TIME --seed $SEED \
        --workers $WORKERS --algorithm $algo --tag $tag --method "$method" \
        --expansion-model "$(model_of $rs)" --expansion-templates "$(tmpl_of $rs)" "$@" \
        >> logs/${tag}_$rs.log 2>&1 || say "  $tag $rs FAILED"
    say "$tag eval: $rs"
    $AZPY -u evaluate_routes.py --routes results/aizynth/routes_${tag}_${rs}.json \
        --references $REFS/${rs}-routes.json --n $MAIN_N \
        --method "$method" --tag $tag --route-set $rs --workers $EVAL_WORKERS \
        --out results/eval/${tag}_${rs}.json >> logs/eval_${tag}_$rs.log 2>&1 \
        || say "  $tag eval $rs FAILED"
    say "REPORT rebuild (after $tag $rs)"
    $RT5PY make_report.py >> logs/report.log 2>&1 || say "  report FAILED"
  done
}

say "START literature reproduction (PaRoutes one-step model, iter=$ITER time=$TIME)"
# fast first, slow second
run_one mcts      litmcts  "AiZynthFinder-MCTS-PaRoutes"
run_one retrostar litretro "AiZynthFinder-Retro*-PaRoutes" --value-model "$VALUE_MODEL"
say "DONE"
