"""Print the best ReactionT5-MCTS sweep config tag (for use by run_all.sh)."""
import json
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
best, best_key = None, None
for path in glob.glob(os.path.join(HERE, "results", "eval", "rt5_sweep-*_n1.json")):
    with open(path) as fh:
        d = json.load(fh)
    key = (d["solve_rate"], d.get("top10", 0), d.get("top5", 0), d.get("top1", 0))
    tag = d["tag"].replace("sweep-", "")
    if best_key is None or key > best_key:
        best_key, best = key, tag
print(best if best else "base")
