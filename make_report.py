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
        is_bestrwd = method == "ReactionT5-MCTS" and tag == "bestrwd"
        is_azf = method.startswith("AiZynthFinder")
        if not (is_best or is_bestrwd or is_azf):
            continue
        meta = metas.get((method, tag, rset), {})
        label = f"This work — {method}"
        if is_best:
            label += f" (best config: {best_config_name()})"
        elif is_bestrwd:
            label += " (best reward: retrek)"
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


# --------------------------------------------------------------------------- #
# Rich HTML extras (ablation chart + route depictions) — HTML only
# --------------------------------------------------------------------------- #
# what each sweep config changes relative to base
SWEEP_CHANGED = {
    "base": "baseline", "c0p5": "C 1.4→0.5", "c3p0": "C 1.4→3.0",
    "ew3": "width 5→3", "it200": "iters→200",
    "md8": "depth 12→8", "rprob": "rollout→prob",
}
# curated route examples: (target index, kind). "exact" = ReactionT5-MCTS
# recovered the exact reference tree (route-distance 0, a top-k hit); "alt" = a
# chemically valid but different synthesis (distance > 0, a top-k miss).
ROUTE_EXAMPLES = [
    (1841, "exact"), (1974, "exact"), (2083, "exact"), (2417, "exact"), (2699, "exact"),
    (394, "alt"), (352, "alt"), (290, "alt"), (161, "alt"), (140, "alt"),
]


def ablation_rows(evals, metas):
    """Rows for the ablation chart, ordered base-first then by config name."""
    order = ["base", "ew3", "it200", "c0p5", "c3p0", "md8", "rprob"]
    rows = []
    for tag in order:
        meta = metas.get(("ReactionT5-MCTS", f"sweep-{tag}", "n1"))
        ev = evals.get(("ReactionT5-MCTS", f"sweep-{tag}", "n1"))
        if not meta or not ev:
            continue
        solved = ev.get("solved_targets", 0)
        n = ev.get("n_targets", 0) or 1
        rows.append({
            "tag": tag, "label": tag, "changed": SWEEP_CHANGED.get(tag, ""),
            "baseline": tag == "base",
            "instock": meta.get("mean_best_instock_frac", 0.0),
            "solved": solved > 0, "solve_txt": f"{solved}/{n}",
            "time": meta.get("mean_time_per_target_s", 0.0),
        })
    return rows


def route_examples_html():
    """Side-by-side ReactionT5-MCTS vs PaRoutes reference route depictions."""
    try:
        import report_viz as rv
    except Exception as exc:  # rdkit / module missing -> skip section gracefully
        return f"<p><em>(route depictions unavailable: {html.escape(str(exc))})</em></p>"
    routes_path = os.path.join(HERE, "results", "rt5", "routes_best_n1.json")
    ref_path = os.path.join(HERE, "paroutes", "data", "n1-routes.json")
    tgt_path = os.path.join(HERE, "paroutes", "data", "n1-targets.txt")
    if not (os.path.exists(routes_path) and os.path.exists(ref_path)):
        return "<p><em>(no route files yet)</em></p>"
    with open(routes_path) as fh:
        routes = json.load(fh)
    with open(ref_path) as fh:
        refs = json.load(fh)
    with open(tgt_path) as fh:
        targets = [l.strip() for l in fh if l.strip()]

    verdict = {
        "exact": ('<span class="verdict exact">exact recovery &middot; '
                  'route-distance&nbsp;0</span>',
                  "ReactionT5-MCTS reconstructed the <em>same</em> tree as the "
                  "reference &mdash; a top-k hit."),
        "alt": ('<span class="verdict alt">alternative route &middot; '
                'distance&nbsp;&gt;&nbsp;0</span>',
                "A chemically valid <em>different</em> synthesis: solved, but it "
                "does not match the exact reference tree (counts as top-k miss)."),
    }
    n_exact = sum(1 for _, k in ROUTE_EXAMPLES if k == "exact")
    n_alt = len(ROUTE_EXAMPLES) - n_exact
    blocks = []
    for pos, (idx, kind) in enumerate(ROUTE_EXAMPLES):
        if idx >= len(routes) or idx >= len(refs):
            continue
        our = routes[idx][0] if isinstance(routes[idx], list) else routes[idx]
        ref = refs[idx]
        badge, caption = verdict[kind]
        tgt = targets[idx] if idx < len(targets) else our.get("smiles", "")
        tgt_short = tgt if len(tgt) <= 44 else tgt[:44] + "…"
        onr, rnr = rv._n_reactions(our), rv._n_reactions(ref)
        openattr = " open" if pos == 0 else ""  # first one expanded, rest collapsed
        summary = (f'<summary class="rex-sum"><span class="rex-sum-tgt">'
                   f'<code>{html.escape(tgt_short)}</code></span> {badge}'
                   f'<span class="rex-sum-n">{onr} rxn vs ref {rnr}</span></summary>')
        body = (f'<p class="rex-cap">{caption}</p>'
                f'<div class="rex-cols">'
                f'<div class="rex-col"><div class="rex-col-h">ReactionT5-MCTS (this work)</div>'
                f'{rv.route_tree_html(our)}</div>'
                f'<div class="rex-col"><div class="rex-col-h">PaRoutes reference route</div>'
                f'{rv.route_tree_html(ref)}</div></div>')
        blocks.append(f'<details class="route-example"{openattr}>{summary}{body}</details>')
    legend = (f'<p class="rex-legend">{len(ROUTE_EXAMPLES)} solved n1 examples '
              f'({n_exact} exact recoveries, {n_alt} alternative routes) &mdash; '
              f'<strong>click a row to expand</strong>. Read right&#8594;left '
              f'(retrosynthetic): each <span class="rex-arrow-key">&#8594;</span> is one '
              f'predicted reaction; green-bordered cards are purchasable building blocks '
              f'(in stock). Molecule structures drawn with RDKit.</p>'
              f'<div class="rex-toggle-all"><button type="button" onclick="'
              f"document.querySelectorAll('details.route-example').forEach(d=>d.open=true)"
              f'">expand all</button> <button type="button" onclick="'
              f"document.querySelectorAll('details.route-example').forEach(d=>d.open=false)"
              f'">collapse all</button></div>')
    return legend + "".join(blocks)


ABLATION_VARS = [
    ("expansion_width", "expansion width (candidates / expansion)"),
    ("iterations", "MCTS iterations"),
    ("c_puct", "c_puct (exploration constant)"),
    ("max_depth", "max depth (reactions on a path)"),
    ("model_budget", "model budget (fresh calls / target)"),
    ("rollout", "rollout policy / width"),
]


def ablation_section_html():
    """Per-variable one-at-a-time ablation plots (quality + cost), HTML only."""
    path = os.path.join(HERE, "results", "ablation", "summary.json")
    if not os.path.exists(path):
        return "<p><em>(detailed ablation not run yet)</em></p>"
    try:
        import report_viz as rv
    except Exception as exc:
        return f"<p><em>(ablation plots unavailable: {html.escape(str(exc))})</em></p>"
    with open(path) as fh:
        data = json.load(fh)
    meta = data.get("meta", {})
    variables = data.get("variables", {})
    cost_series = [("queries_per_target", "queries/target", "#4a3aa7", "#9085e9")]
    figs = []
    for var, xlabel in ABLATION_VARS:
        pts = variables.get(var)
        if not pts:
            continue
        qpoints, cpoints = [], []
        for p in pts:
            tip = (f"{p.get('queries_per_target',0):.0f} queries/target, "
                   f"{p.get('time_per_target_s',0):.2f}s/target")
            qpoints.append({"label": p["label"], "solve_rate": p.get("solve_rate", 0),
                            "top1": p.get("top1", 0), "top5": p.get("top5", 0),
                            "top10": p.get("top10", 0), "tip": tip})
            cpoints.append({"label": p["label"],
                            "queries_per_target": p.get("queries_per_target", 0),
                            "tip": f"{p.get('time_per_target_s',0):.2f}s/target (warm cache), "
                                   f"{p.get('model_calls',0)} fresh calls this run"})
        quality = rv.ablation_line_svg(
            qpoints, rv._QSERIES, xlabel, f"{var}: quality", value_fmt="{:.2f}",
            ylabel="fraction of targets")
        cost = rv.ablation_line_svg(
            cpoints, cost_series, xlabel, f"{var}: search work", value_fmt="{:.0f}",
            ylabel="queries / target")
        figs.append(f'<div class="abl-fig"><div class="abl-pair">{quality}{cost}</div></div>')
    n = meta.get("n_targets", "?")
    intro = (f'<p>Each variable is swept while all others stay at the baseline '
             f'(<code>expansion_width=5, iterations=100, c_puct=1.4, max_depth=12, '
             f'rollout=stock, model_budget=20</code>), on the first '
             f'<strong>{n}</strong> n1 targets. Left panel = solution quality '
             f'(solve rate and exact-match top-k, all fractions of targets); right '
             f'panel = search work as one-step model <em>queries per target</em> '
             f'(cache-hits + misses) — a cache-independent cost signal that tracks '
             f'search intensity; hover a point for the warm-cache wall-clock time. '
             f'Lines are directly labelled at their right end. Quality metrics are '
             f'exact (deterministic given config + seed); wall-clock is not shown as '
             f'the primary cost because all runs share a warm prediction cache.</p>')
    return intro + "".join(figs)


def main():
    evals, metas = collect()
    os.makedirs(REPORTS, exist_ok=True)

    best_sweep = pick_best(evals)
    # the best config re-run on n1/n5 is tagged "best"
    best_tag = "best"

    sw_h, sw_r = sweep_table(evals, metas)
    mn_h, mn_r = main_table(evals, metas, best_tag)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    intro = rf"""# Retrosynthesis Route Planning: ReactionT5-MCTS vs AiZynthFinder on PaRoutes

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
    md.append(r"""
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
         max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.55;
         color:#1a1a1a;background:#ffffff}
    h1{border-bottom:3px solid #2c7fb8;padding-bottom:.3rem}
    h2{margin-top:2rem;color:#2c7fb8}
    table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
    th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right}
    th:first-child,td:first-child{text-align:left}
    th{background:#f0f6fb}
    tbody tr:nth-child(even){background:#fafafa}
    code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}
    em{color:#666}
    h3{margin-top:1.6rem}
    /* ablation chart */
    .ablation-svg{width:100%;height:auto;max-width:760px;display:block;margin:1rem 0}
    .ablation-svg .grid{stroke:#e2e2e2;stroke-width:1}
    .ablation-svg .ytick{fill:#8a8a86;font-size:11px;text-anchor:end}
    .ablation-svg .barval{fill:#52514e;font-size:12px;font-weight:600;text-anchor:middle}
    .ablation-svg .cfgname{fill:#1a1a1a;font-size:12px;font-weight:600;text-anchor:middle}
    .ablation-svg .cfgchg{fill:#8a8a86;font-size:10px;text-anchor:middle}
    .ablation-svg .axtitle{fill:#52514e;font-size:12px}
    .ablation-svg .axlabel,.abl-line .axlabel{fill:#52514e;font-size:11px;text-anchor:middle;font-weight:600}
    .ablation-svg .status{font-size:13px;text-anchor:middle}
    .ablation-svg .status.ok{fill:#008300} .ablation-svg .status.no{fill:#c0392b}
    .chart-cap{font-size:.85rem;color:#666;max-width:760px;margin:.2rem 0 1.5rem}
    /* detailed ablation line charts */
    .abl-fig{margin:.6rem 0 1.2rem}
    .abl-pair{display:flex;flex-wrap:wrap;gap:1rem}
    .abl-line{width:100%;max-width:400px;height:auto;flex:1 1 340px}
    .abl-line .abl-title{fill:#1a1a1a;font-size:12px;font-weight:600}
    .abl-line .grid{stroke:#e6e6e6;stroke-width:1}
    .abl-line .ytick{fill:#8a8a86;font-size:10px;text-anchor:end}
    .abl-line .xtick{fill:#8a8a86;font-size:10px;text-anchor:middle}
    .abl-line .xlabel{fill:#52514e;font-size:10px;text-anchor:middle}
    .abl-line .ln{fill:none;stroke-width:2}
    .abl-line .endlab{font-size:9px;font-weight:700}
    .abl-line .s-solve-rate{stroke:#2a78d6;fill:#2a78d6}
    .abl-line .s-top1{stroke:#eb6834;fill:#eb6834}
    .abl-line .s-top5{stroke:#1baf7a;fill:#1baf7a}
    .abl-line .s-top10{stroke:#eda100;fill:#eda100}
    .abl-line .s-queries-per-target{stroke:#4a3aa7;fill:#4a3aa7}
    .abl-line polyline.s-solve-rate,.abl-line polyline.s-top1,
    .abl-line polyline.s-top5,.abl-line polyline.s-top10,
    .abl-line polyline.s-queries-per-target{fill:none}
    /* route depictions */
    .rex-legend{font-size:.85rem;color:#666}
    .rex-arrow-key{color:#2c7fb8;font-weight:700}
    .route-example{border:1px solid #d8d8d8;border-radius:8px;padding:.4rem .9rem;
        margin:.5rem 0;background:#fbfbfa}
    details.route-example[open]{padding-bottom:.8rem}
    .rex-sum{cursor:pointer;list-style:none;display:flex;align-items:center;
        gap:.5rem;flex-wrap:wrap;padding:.35rem 0;font-size:.9rem}
    .rex-sum::-webkit-details-marker{display:none}
    .rex-sum::before{content:"\\25B8";color:#2c7fb8;font-size:.8rem;
        transition:transform .15s}
    details[open] > .rex-sum::before{transform:rotate(90deg)}
    .rex-sum-tgt code{font-size:.78rem}
    .rex-sum-n{color:#8a8a86;font-size:.78rem;margin-left:auto}
    .rex-toggle-all{margin:.3rem 0 .6rem;display:flex;gap:.5rem}
    .rex-toggle-all button{font-size:.78rem;padding:.2rem .6rem;cursor:pointer;
        border:1px solid #c3d4e2;background:#eef4fa;color:#2c7fb8;border-radius:5px}
    .rex-head{font-size:.95rem;margin-bottom:.2rem}
    .rex-head code{font-size:.8rem}
    .rex-cap{font-size:.85rem;color:#555;margin:.2rem 0 .8rem}
    .verdict{display:inline-block;padding:.05rem .5rem;border-radius:999px;
        font-size:.75rem;font-weight:700;margin-left:.3rem;white-space:nowrap}
    .verdict.exact{background:#e3f4e3;color:#186818;border:1px solid #9ed49e}
    .verdict.alt{background:#fdeede;color:#9a5b16;border:1px solid #f0c48a}
    .rex-cols{display:flex;gap:1rem;flex-wrap:wrap}
    .rex-col{flex:1 1 320px;min-width:300px;overflow-x:auto}
    .rex-col-h{font-size:.8rem;font-weight:700;color:#2c7fb8;margin-bottom:.4rem;
        text-transform:uppercase;letter-spacing:.03em}
    .route-tree{display:flex;align-items:center;gap:2px;padding:.3rem 0}
    .mol-node{display:flex;align-items:center;gap:2px}
    .rxn{display:flex;align-items:center;gap:2px}
    .rxn-arrow{color:#2c7fb8;font-size:1.3rem;font-weight:700;padding:0 .1rem}
    .rxn-reactants{display:flex;flex-direction:column;gap:6px;
        border-left:2px solid #cfe0ee;padding-left:4px}
    .mol-card{background:#fff;border:1px solid #dcdcdc;border-radius:6px;padding:2px;
        display:flex;flex-direction:column;align-items:center;position:relative}
    .mol-card.stock{border-color:#54b054;box-shadow:0 0 0 1px #b9e0b9 inset}
    .mol-card svg{display:block}
    .mol-fallback{font-size:.6rem;max-width:120px;word-break:break-all;padding:4px}
    .leaf-badge{font-size:.6rem;padding:0 .25rem;border-radius:4px;margin-top:1px}
    .leaf-badge.stock{background:#e3f4e3;color:#186818}
    @media (prefers-color-scheme: dark){
      body{background:#111;color:#e6e6e6} th{background:#1c2b36}
      tbody tr:nth-child(even){background:#181818} code{background:#222}
      th,td{border-color:#444}
      .ablation-svg .grid{stroke:#333}
      .ablation-svg .cfgname{fill:#e6e6e6} .ablation-svg .barval{fill:#c3c2b7}
      .ablation-svg .axtitle{fill:#c3c2b7}
      .abl-line .abl-title{fill:#e6e6e6} .abl-line .grid{stroke:#333}
      .abl-line .xlabel{fill:#c3c2b7}
      .ablation-svg .axlabel,.abl-line .axlabel{fill:#c3c2b7}
      .abl-line .s-solve-rate{stroke:#3987e5;fill:#3987e5}
      .abl-line .s-top1{stroke:#d95926;fill:#d95926}
      .abl-line .s-top5{stroke:#199e70;fill:#199e70}
      .abl-line .s-top10{stroke:#c98500;fill:#c98500}
      .abl-line .s-queries-per-target{stroke:#9085e9;fill:#9085e9}
      .route-example{background:#161616;border-color:#3a3a3a}
      .rex-sum-n{color:#8a8a86}
      .rex-toggle-all button{background:#16242f;border-color:#2f4a63;color:#7fb2e0}
      .rex-cap{color:#b0b0b0}
      .mol-card{background:#f4f4f2;border-color:#555}  /* keep molecules on light cards */
      .mol-fallback{color:#111}
      .rxn-reactants{border-left-color:#2f4a63}
      .verdict.exact{background:#173417;color:#8fd48f;border-color:#2e5e2e}
      .verdict.alt{background:#3a2a12;color:#e6b877;border-color:#6a4a1e}}
    """

    def md_inline_to_html(text):
        # mini-markdown: headings (one line), bullet lists, bold/italic/code,
        # escaped asterisks — handles multi-line blocks (heading + list) properly
        out = []
        for block in text.split("\n\n"):
            lines = [ln for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            i = 0
            # leading heading line
            first = lines[0].strip()
            if first.startswith("### "):
                out.append(f"<h3>{_inline(first[4:])}</h3>"); i = 1
            elif first.startswith("## "):
                out.append(f"<h2>{_inline(first[3:])}</h2>"); i = 1
            elif first.startswith("# "):
                out.append(f"<h1>{_inline(first[2:])}</h1>"); i = 1
            # remaining lines: bullet list and/or paragraph text
            rest = lines[i:]
            buf_items, buf_para = [], []
            def flush_para():
                if buf_para:
                    out.append(f"<p>{_inline(' '.join(buf_para))}</p>")
                    buf_para.clear()
            def flush_list():
                if buf_items:
                    lis = "".join(f"<li>{_inline(x)}</li>" for x in buf_items)
                    out.append(f"<ul>{lis}</ul>")
                    buf_items.clear()
            for ln in rest:
                s = ln.strip()
                if s.startswith("- "):
                    flush_para(); buf_items.append(s[2:])
                else:
                    flush_list(); buf_para.append(s)
            flush_list(); flush_para()
        return "\n".join(out)

    body = []
    body.append(md_inline_to_html(intro))
    body.append("<h2>1. Hyperparameter sweep (ReactionT5-MCTS, n1 subset)</h2>")
    body.append("<p>Base config: <code>expansion_width=5, iterations=100, "
                "c_puct=1.4, max_depth=12, rollout_width=1, rollout_policy=stock, "
                "model_budget=15</code>. Each row varies one parameter.</p>")
    body.append(html_table(sw_h, sw_r) if sw_r else "<p><em>(no sweep results yet)</em></p>")
    # ablation chart (HTML-only)
    ab_rows = ablation_rows(evals, metas)
    if ab_rows:
        try:
            import report_viz as rv
            body.append("<h3>1a. Ablation view</h3>")
            body.append(rv.ablation_chart_svg(ab_rows))
            body.append('<p class="chart-cap">Each bar is one config that changes a '
                        'single hyper-parameter from the baseline (dark blue). Height = '
                        'mean best in-stock fraction (how close the planner gets to a '
                        'fully purchasable route); &#10004;/&#10007; under each bar = '
                        'whether any target was fully solved. Shrinking the search depth '
                        '(<code>md8</code>) or switching to a probabilistic rollout '
                        '(<code>rprob</code>) breaks solving; a narrower expansion '
                        '(<code>ew3</code>) lowers in-stock fraction. Sweep is a small '
                        '3-target probe, so treat solve as coarse and in-stock fraction '
                        'as the sensitive signal.</p>')
        except Exception as exc:
            body.append(f"<p><em>(ablation chart unavailable: {html.escape(str(exc))})</em></p>")
    # detailed one-at-a-time ablation (HTML-only)
    body.append("<h3>1b. Detailed one-at-a-time ablation</h3>")
    body.append(ablation_section_html())
    body.append("<h2>2. Main comparison: literature vs ReactionT5-MCTS (best) vs "
                "AiZynthFinder</h2>")
    body.append(html_table(mn_h, mn_r) if mn_r else "<p><em>(no results yet)</em></p>")
    body.append(md_inline_to_html("""### Notes
- **solved** = targets with at least one fully-in-stock route; **solve rate** = solved / n.
- **top-k** = fraction of targets whose exact reference route is recovered (distance 0) in top-k.
- **time** = CPU wall-clock; literature reports none and used the full benchmark.
- AiZynthFinder uses the public USPTO expansion policy + PaRoutes stock."""))

    # route example gallery (HTML-only)
    body.append("<h2>3. Example routes found by ReactionT5-MCTS</h2>")
    body.append("<p>Solved n1 targets, comparing the ReactionT5-MCTS prediction "
                "with the PaRoutes reference route. These illustrate why <strong>solve "
                "rate is high but exact-match top-k is low</strong>: the planner often "
                "finds a valid <em>alternative</em> synthesis rather than the exact "
                "reference tree.</p>")
    body.append(route_examples_html())

    html_doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>ReactionT5-MCTS vs AiZynthFinder on PaRoutes</title>"
                f"<style>{style}</style></head><body>{''.join(body)}</body></html>")
    with open(os.path.join(REPORTS, "report.html"), "w") as fh:
        fh.write(html_doc)

    print(f"Wrote {REPORTS}/report.md and {REPORTS}/report.html")
    print(f"Best sweep config: {best_sweep}")


def _inline(text):
    """Escape HTML then apply inline markdown: `code`, **bold**, *italic*, \\*."""
    import re
    s = html.escape(text)
    # protect escaped asterisks
    s = s.replace(r"\*", "\x00")
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
    return s.replace("\x00", "*")


def _bold(text):  # kept for backwards compatibility
    return _inline(text)


if __name__ == "__main__":
    main()
