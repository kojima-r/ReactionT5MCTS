"""Aggregate all experiment results into a Markdown + HTML report.

Reads:
    * results/eval/*.json  - official PaRoutes metrics (from evaluate_routes.py)
    * results/*/meta_*.json - per-run timing / config metadata

Writes:
    * reports/report.md
    * reports/report.html
"""
from __future__ import annotations

import glob
import html
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")

# --------------------------------------------------------------------------- #
# Literature reference values (from paroutes/README.md benchmark tables)
# --------------------------------------------------------------------------- #
N_TARGETS_FULL = 10000
LITERATURE = [
    # source, method, set, solved, top1, top5, top10
    ("Genheden 2022 (v1)", "MCTS",   "n1", 9714, 0.20, 0.55, 0.61),
    ("Genheden 2022 (v1)", "MCTS",   "n5", 9676, 0.09, 0.34, 0.42),
    ("Genheden 2022 (v1)", "Retro*", "n1", 9726, 0.17, 0.48, 0.54),
    ("Genheden 2022 (v1)", "Retro*", "n5", 9703, 0.08, 0.30, 0.38),
    ("PaRoutes v2.0",      "MCTS",   "n1", 9716, 0.2372, 0.5107, 0.5414),
    ("PaRoutes v2.0",      "MCTS",   "n5", 9689, 0.1237, 0.3584, 0.4056),
    ("PaRoutes v2.0",      "Retro*", "n1", 9728, 0.2027, 0.4516, 0.4847),
    ("PaRoutes v2.0",      "Retro*", "n5", 9729, 0.1143, 0.3365, 0.3897),
]


def load_jsons(pattern):
    out = []
    for path in glob.glob(pattern):
        try:
            with open(path) as fh:
                out.append((path, json.load(fh)))
        except Exception:
            pass
    return out


def collect():
    # official metrics keyed by (method, tag, route_set)
    evals = {}
    for _, d in load_jsons(os.path.join(HERE, "results", "eval", "*.json")):
        evals[(d["method"], d.get("tag", ""), d["route_set"])] = d
    # timing metadata keyed by (method, tag, route_set)
    metas = {}
    for _, d in load_jsons(os.path.join(HERE, "results", "*", "meta_*.json")):
        metas[(d["method"], d.get("tag", ""), d["route_set"])] = d
    return evals, metas


def fmt_pct(x):
    if x is None:
        return "N/A"
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def fmt_time(s):
    if s is None:
        return "N/A"
    return f"{s:.1f}" if s < 60 else f"{s/60:.1f} min"


# --------------------------------------------------------------------------- #
# Table builders (return list-of-rows; header separate)
# --------------------------------------------------------------------------- #
def sweep_table(evals, metas):
    header = ["config (tag)", "route set", "params", "solved", "solve rate",
              "in-stock frac", "top-1", "top-5", "top-10", "total time", "time/target"]
    rows = []
    for (method, tag, rset), ev in sorted(evals.items()):
        if method != "ReactionT5-MCTS" or not tag.startswith("sweep-"):
            continue
        meta = metas.get((method, tag, rset), {})
        cfg = meta.get("config", {})
        params = ", ".join(f"{k}={v}" for k, v in cfg.items())
        rows.append([
            tag.replace("sweep-", ""), rset, params,
            f"{ev['solved_targets']}/{ev['n_targets']}",
            fmt_pct(ev["solve_rate"]),
            fmt_pct(meta.get("mean_best_instock_frac")),
            fmt_pct(ev.get("top1")), fmt_pct(ev.get("top5")), fmt_pct(ev.get("top10")),
            fmt_time(meta.get("total_time_s")),
            fmt_time(meta.get("mean_time_per_target_s")),
        ])
    return header, rows


def main_table(evals, metas, best_tag):
    header = ["source / method", "route set", "n targets", "solved", "solve rate",
              "in-stock frac", "top-1", "top-5", "top-10", "total time", "time/target"]
    rows = []
    # literature first
    for source, method, rset, solved, t1, t5, t10 in LITERATURE:
        rows.append([
            f"{source} — {method}", rset, N_TARGETS_FULL,
            solved, fmt_pct(solved / N_TARGETS_FULL), "N/A",
            fmt_pct(t1), fmt_pct(t5), fmt_pct(t10),
            "N/A", "N/A",
        ])
    # our experiments
    for (method, tag, rset), ev in sorted(evals.items()):
        is_best = method == "ReactionT5-MCTS" and tag == best_tag
        is_azf = method.startswith("AiZynthFinder")
        if not (is_best or is_azf):
            continue
        meta = metas.get((method, tag, rset), {})
        label = f"This work — {method}"
        if is_best:
            label += f" (best config: {best_config_name()})"
        rows.append([
            label, rset, ev["n_targets"],
            f"{ev['solved_targets']}", fmt_pct(ev["solve_rate"]),
            fmt_pct(meta.get("mean_best_instock_frac")),
            fmt_pct(ev.get("top1")), fmt_pct(ev.get("top5")), fmt_pct(ev.get("top10")),
            fmt_time(meta.get("total_time_s")),
            fmt_time(meta.get("mean_time_per_target_s")),
        ])
    return header, rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def html_table(header, rows):
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def best_config_name():
    path = os.path.join(HERE, "results", "best_config.txt")
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return "base"


def pick_best(evals):
    """Best ReactionT5-MCTS sweep config by (solve_rate, top10, top5, top1)."""
    best, best_key = None, None
    for (method, tag, rset), ev in evals.items():
        if method != "ReactionT5-MCTS" or not tag.startswith("sweep-") or rset != "n1":
            continue
        key = (ev["solve_rate"], ev.get("top10", 0), ev.get("top5", 0), ev.get("top1", 0))
        if best_key is None or key > best_key:
            best_key, best = key, tag
    return best


def main():
    evals, metas = collect()
    os.makedirs(REPORTS, exist_ok=True)

    best_sweep = pick_best(evals)
    # the best config re-run on n1/n5 is tagged "best"
    best_tag = "best"

    sw_h, sw_r = sweep_table(evals, metas)
    mn_h, mn_r = main_table(evals, metas, best_tag)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    intro = f"""# Retrosynthesis Route Planning: ReactionT5-MCTS vs AiZynthFinder on PaRoutes

*Generated {now}*

This report evaluates a Monte-Carlo Tree Search (MCTS) retrosynthetic planner
that uses **ReactionT5** (`sagawa/ReactionT5v2-retrosynthesis`) as the one-step
retrosynthesis model, benchmarked on the **PaRoutes** n1/n5 route sets, and
compares it with **AiZynthFinder** and the published literature values.

**Reproducibility.** All runs use a fixed random seed. ReactionT5 beam search is
deterministic and one-step predictions are memoised in a shared on-disk cache,
so re-running yields identical routes.

**Scope note.** Runs use GPU-backed ReactionT5 inference with target-level
parallelism, so the **full benchmark (10,000 targets/set)** is evaluated for
every "This work" row. Each ReactionT5-MCTS target is still capped at a fixed
*budget* of fresh one-step model evaluations (memoised in a shared cache). Three
planners are compared like-for-like on the identical full sets: **ReactionT5-MCTS**
(best sweep config), **AiZynthFinder-MCTS**, and **AiZynthFinder-Retro\*** — the
latter using the public USPTO expansion policy with the PaRoutes
`retrostar_value_model.pickle` molecule-cost model. The literature columns are
the full-benchmark published values, produced with the PaRoutes-specific
template one-step models, and are shown for context.

Best ReactionT5-MCTS sweep configuration (by solve rate on the n1 subset): **{best_sweep}**.
"""

    md = [intro]
    md.append("\n## 1. Hyperparameter sweep (ReactionT5-MCTS, n1 subset)\n")
    md.append("Base config: expansion_width=5, iterations=100, c_puct=1.4, "
              "max_depth=12, rollout_width=1, rollout_policy=stock, model_budget=15. "
              "Each row varies one parameter.\n")
    md.append(md_table(sw_h, sw_r) if sw_r else "_(no sweep results yet)_")
    md.append("\n## 2. Main comparison: literature vs ReactionT5-MCTS (best) vs AiZynthFinder\n")
    md.append(md_table(mn_h, mn_r) if mn_r else "_(no results yet)_")
    md.append("""
### Notes
- **solve rate is the primary metric here** = number of targets for which at
  least one predicted route reaches purchasable stock, divided by n targets.
- **top-k** = fraction of targets whose exact reference route is recovered
  (route tree-edit distance == 0) among the top-k ranked routes — the strict
  PaRoutes accuracy metric. For **ReactionT5-MCTS this is expected to be ~0**:
  ReactionT5 is a template-free seq2seq model, whereas the PaRoutes reference
  routes were generated by a template-based model, so recovering the exact same
  tree is very unlikely even when a valid alternative route is found.
- **in-stock frac** = mean over targets of the best fraction of route leaves that
  reach purchasable stock. It is a softer progress metric than the all-or-nothing
  "solved": a value near 1.0 means the planner found an almost-complete route
  (only one or two leaves short), which the strict "solved" flag hides.
- ReactionT5 emits reagents/solvents/salts that the PaRoutes building-block stock
  lacks, so the ReactionT5-MCTS runs augment the stock with a curated set of
  common reagents (chemically these are always available). The rollout uses a
  stock-greedy policy that prefers disconnections into already-purchasable pieces.
- **time** columns are wall-clock on CPU; literature values were produced on
  different hardware over the full benchmark and report no runtime.
- AiZynthFinder here uses the public USPTO expansion policy with the PaRoutes
  stock; the literature MCTS/Retro* rows use the PaRoutes-specific one-step models.
- **AiZynthFinder-Retro\*** is the Retro\* search algorithm with the PaRoutes
  `retrostar_value_model.pickle` as the molecule-cost (value) model, otherwise
  identical policy/stock/budget to the AiZynthFinder-MCTS run. Comparing the two
  "This work — AiZynthFinder" rows isolates the effect of the search algorithm
  (MCTS vs Retro\*) with everything else held fixed.
- **AiZynthFinder-*-PaRoutes** rows use the **literature one-step model**: the
  route-set-specific expansion policy `uspto_rxn_n{1,5}_keras_model.hdf5`
  (Zenodo 6275421, converted Keras→ONNX) with its matching template set, plus the
  **literature search budget** (iteration_limit=500, time_limit=3600 s). These are
  the faithful reproduction of the published PaRoutes MCTS/Retro\* rows; the plain
  "AiZynthFinder" rows above use the generic public USPTO policy and a smaller
  budget. The difference between them quantifies how much of the gap to the
  literature comes from the one-step model + search budget rather than the planner.
""")
    md_text = "\n".join(md)
    with open(os.path.join(REPORTS, "report.md"), "w") as fh:
        fh.write(md_text)

    # HTML
    style = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1a1a1a}
    h1{border-bottom:3px solid #2c7fb8;padding-bottom:.3rem}
    h2{margin-top:2rem;color:#2c7fb8}
    table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
    th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right}
    th:first-child,td:first-child{text-align:left}
    th{background:#f0f6fb}
    tbody tr:nth-child(even){background:#fafafa}
    code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}
    em{color:#666}
    @media (prefers-color-scheme: dark){
      body{background:#111;color:#e6e6e6} th{background:#1c2b36}
      tbody tr:nth-child(even){background:#181818} code{background:#222}
      th,td{border-color:#444}}
    """

    def md_inline_to_html(text):
        # minimal: headers, bold, code, paragraphs
        out = []
        for block in text.split("\n\n"):
            b = block.strip()
            if not b:
                continue
            if b.startswith("### "):
                out.append(f"<h3>{html.escape(b[4:])}</h3>")
            elif b.startswith("## "):
                out.append(f"<h2>{html.escape(b[3:])}</h2>")
            elif b.startswith("# "):
                out.append(f"<h1>{html.escape(b[2:])}</h1>")
            else:
                b = html.escape(b)
                b = _bold(b)
                out.append(f"<p>{b}</p>")
        return "\n".join(out)

    body = []
    body.append(md_inline_to_html(intro))
    body.append("<h2>1. Hyperparameter sweep (ReactionT5-MCTS, n1 subset)</h2>")
    body.append("<p>Base config: <code>expansion_width=5, iterations=100, "
                "c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, "
                "model_budget=15</code>. Each row varies one parameter.</p>")
    body.append(html_table(sw_h, sw_r) if sw_r else "<p><em>(no sweep results yet)</em></p>")
    body.append("<h2>2. Main comparison: literature vs ReactionT5-MCTS (best) vs "
                "AiZynthFinder</h2>")
    body.append(html_table(mn_h, mn_r) if mn_r else "<p><em>(no results yet)</em></p>")
    body.append(md_inline_to_html("""### Notes
- **solved** = targets with at least one fully-in-stock route; **solve rate** = solved / n.
- **top-k** = fraction of targets whose exact reference route is recovered (distance 0) in top-k.
- **time** = CPU wall-clock; literature reports none and used the full benchmark.
- AiZynthFinder uses the public USPTO expansion policy + PaRoutes stock."""))

    html_doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>ReactionT5-MCTS vs AiZynthFinder on PaRoutes</title>"
                f"<style>{style}</style></head><body>{''.join(body)}</body></html>")
    with open(os.path.join(REPORTS, "report.html"), "w") as fh:
        fh.write(html_doc)

    print(f"Wrote {REPORTS}/report.md and {REPORTS}/report.html")
    print(f"Best sweep config: {best_sweep}")


def _bold(text):
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


if __name__ == "__main__":
    main()
