"""Lean check: stock-greedy descent per target, report in-stock leaf fraction."""
import sys, time
from rt5mcts.model import ReactionT5
from rt5mcts.stock import Stock
from rt5mcts.reagents import reagent_smiles
from rt5mcts.mcts import (RouteMol, RouteReaction, leaves, is_solved,
                          choose_expandable, count_reactions)
from rdkit import Chem

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 25
idxs = [int(x) for x in sys.argv[2:]] or [1, 3, 5, 7]
tgts = [l.strip() for l in open('paroutes/data/n1-targets.txt')]
stock = Stock([l for l in open('paroutes/data/n1-stock.txt') if l.strip()] + reagent_smiles())
model = ReactionT5('cache/rt5_b5.sqlite', max_beams=5, num_threads=18, seed=42)

for idx in idxs:
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(tgts[idx]))
    model.reset_budget(budget)
    root = RouteMol(canon, canon in stock, 0)
    t0 = time.time()
    for _ in range(30):
        if is_solved(root):
            break
        mol = choose_expandable(root, 12)
        if mol is None:
            break
        preds = model.predict(mol.smiles, top_k=5)
        if not preds:
            break
        def sc(p):
            ns = sum(1 for x in p.reactants if x in stock)
            return (ns - (len(p.reactants) - ns), p.prob)
        r = max(preds, key=sc)
        mol.reaction = RouteReaction([RouteMol(x, x in stock, mol.depth + 1) for x in r.reactants])
    lv = leaves(root)
    ns = sum(1 for m in lv if m.in_stock)
    print(f"target {idx}: rxn={count_reactions(root)} leaves in_stock={ns}/{len(lv)} "
          f"solved={is_solved(root)} time={time.time()-t0:.0f}s calls={model.stats()['model_calls']}",
          flush=True)
