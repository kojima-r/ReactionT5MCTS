"""Diagnostic: can ReactionT5 greedy descent approach a solved route, and does
the MCTS descend deep? Runs a pure greedy descent + one MCTS search on a target."""
import sys, time
from rt5mcts.model import ReactionT5
from rt5mcts.stock import Stock
from rt5mcts.mcts import (RetroMCTS, MCTSConfig, RouteMol, RouteReaction,
                          open_leaves, leaves, is_solved, choose_expandable, count_reactions)

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 40

from rt5mcts.reagents import reagent_smiles
tgts = [l.strip() for l in open('paroutes/data/n1-targets.txt')]
stock_lines = [l for l in open('paroutes/data/n1-stock.txt') if l.strip()] + reagent_smiles()
stock = Stock(stock_lines)
model = ReactionT5('cache/rt5_b5.sqlite', max_beams=5, num_threads=18, seed=42)
smi = tgts[idx]
from rdkit import Chem
canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
print(f"target {idx}: {canon} (in_stock={canon in stock})")

# --- pure greedy descent (top-1) ---
model.reset_budget(budget)
root = RouteMol(canon, canon in stock, 0)
t0 = time.time()
for step in range(20):
    if is_solved(root):
        break
    mol = choose_expandable(root, 12)
    if mol is None:
        break
    preds = model.predict(mol.smiles, top_k=5)
    if not preds:
        print(f"  greedy stop at depth {mol.depth}: no preds for {mol.smiles[:50]}")
        break
    def sc(p):
        ns = sum(1 for x in p.reactants if x in stock)
        return (ns - (len(p.reactants) - ns), p.prob)
    r = max(preds, key=sc)
    children = [RouteMol(x, x in stock, mol.depth + 1) for x in r.reactants]
    mol.reaction = RouteReaction(children)
    n_in = sum(1 for c in children if c.in_stock)
    print(f"  step depth{mol.depth}->{mol.depth+1}: {mol.smiles[:40]} -> {len(children)} reactants ({n_in} in stock)")
lvs = leaves(root)
print(f"greedy: reactions={count_reactions(root)} leaves={len(lvs)} "
      f"in_stock={sum(1 for m in lvs if m.in_stock)}/{len(lvs)} solved={is_solved(root)} "
      f"time={time.time()-t0:.0f}s fresh_calls={model.stats()['model_calls']}")

# --- MCTS ---
model.reset_budget(budget)
before = model.stats()['model_calls']
cfg = MCTSConfig(expansion_width=5, iterations=200, c_puct=1.4, max_depth=12,
                 rollout_width=1, model_budget=budget, seed=42)
mcts = RetroMCTS(model, stock, cfg)
t0 = time.time()
mcts.search(smi)
routes = mcts.ranked_routes(10)
print(f"MCTS: solved_routes={len(mcts.solutions)} time={time.time()-t0:.0f}s "
      f"fresh_calls={model.stats()['model_calls']-before}")
