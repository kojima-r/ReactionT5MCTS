"""Build reports/reward_report.{md,html} from the reward-experiment summary.

Standalone report comparing reward functions and weight settings for
ReactionT5-MCTS: a full metrics table plus per-metric bar charts and the
yield-weight sweep, with a short discussion.
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "results", "reward_exp")
REPORTS = os.path.join(HERE, "reports")
ROUTE_SETS = ["n1", "n5"]


def summary_path(rset):
    return os.path.join(EXP_DIR, f"summary_{rset}.json")

COLS = [
    ("label", "reward"), ("solve_rate", "solve"), ("top1", "top-1"),
    ("top5", "top-5"), ("top10", "top-10"),
    ("mean_n_reactions", "#rxn"), ("mean_top_yield", "yield"),
    ("mean_top_sa_ease", "SA-ease"), ("mean_top_cdscore", "convergence"),
    ("queries_per_target", "queries/tgt"), ("time_per_target_s", "s/tgt"),
]


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in COLS)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{_fmt(r.get(k))}</td>" for k, _ in COLS)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _md_table(rows):
    out = ["| " + " | ".join(h for _, h in COLS) + " |",
           "|" + "|".join("---" for _ in COLS) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(k)) for k, _ in COLS) + " |")
    return "\n".join(out)


STYLE = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     max-width:1080px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1a1a1a;background:#fff}
h1{border-bottom:3px solid #2c7fb8;padding-bottom:.3rem}
h2{margin-top:2rem;color:#2c7fb8} h3{margin-top:1.4rem}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.86rem;
      font-variant-numeric:tabular-nums}
th,td{border:1px solid #ccc;padding:.35rem .5rem;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f0f6fb} tbody tr:nth-child(even){background:#fafafa}
code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px} em{color:#666}
.charts{display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start}
.rex-bar,.abl-line{width:100%;max-width:460px;height:auto;flex:1 1 360px}
.rex-bar .bc-title,.abl-line .abl-title{fill:#1a1a1a;font-size:12px;font-weight:600}
.rex-bar .bc-grid{stroke:#e6e6e6} .rex-bar .bc-ytick{fill:#8a8a86;font-size:10px;text-anchor:end}
.rex-bar .bc-val{fill:#52514e;font-size:10px;font-weight:600;text-anchor:middle}
.rex-bar .bc-xlab{fill:#52514e;font-size:10px;text-anchor:start}
.rex-bar .bc-bar{fill:#9db8d6} .rex-bar .bc-hot{fill:#2a78d6}
.abl-line .bc-grid,.abl-line .grid{stroke:#e6e6e6}
.abl-line .ytick,.abl-line .xtick{fill:#8a8a86;font-size:10px}
.abl-line .ytick{text-anchor:end} .abl-line .xtick{text-anchor:middle}
.abl-line .xlabel{fill:#52514e;font-size:10px;text-anchor:middle}
.abl-line .ln{fill:none;stroke-width:2} .abl-line .endlab{font-size:9px;font-weight:700}
.abl-line .s-solve-rate{stroke:#2a78d6;fill:#2a78d6}
.abl-line .s-top1{stroke:#eb6834;fill:#eb6834}
.abl-line .s-mean-top-yield{stroke:#1baf7a;fill:#1baf7a}
.cap{font-size:.85rem;color:#666;max-width:820px}
@media (prefers-color-scheme: dark){
  body{background:#111;color:#e6e6e6} th{background:#1c2b36}
  tbody tr:nth-child(even){background:#181818} code{background:#222} th,td{border-color:#444}
  .rex-bar .bc-title,.abl-line .abl-title{fill:#e6e6e6}
  .rex-bar .bc-grid,.abl-line .bc-grid,.abl-line .grid{stroke:#333}
  .rex-bar .bc-val{fill:#c3c2b7} .rex-bar .bc-xlab,.abl-line .xlabel{fill:#c3c2b7}
  .rex-bar .bc-bar{fill:#3a536e} .rex-bar .bc-hot{fill:#3987e5}
  .abl-line .s-solve-rate{stroke:#3987e5;fill:#3987e5}
  .abl-line .s-top1{stroke:#d95926;fill:#d95926}
  .abl-line .s-mean-top-yield{stroke:#199e70;fill:#199e70}}
"""


def _set_blocks(rows, n, rset, rv, sec):
    """Return (html_str, md_str) for one route set's section."""
    def bars(metric, title, fmt="{:.3f}"):
        pts = [{"label": r["label"], "value": r.get(metric),
                "tip": f"solve {r['solve_rate']:.3f}"} for r in rows
               if r.get(metric) is not None]
        return rv.bar_chart_svg(pts, title, value_fmt=fmt, highlight="stock")

    wmap = {"yield_w0.5": 0.5, "yield": 1.0, "yield_w2": 2.0, "yield_w4": 4.0}
    wpts = []
    for lab, w in sorted(wmap.items(), key=lambda kv: kv[1]):
        r = next((x for x in rows if x["label"] == lab), None)
        if r:
            wpts.append({"label": str(w), "solve_rate": r["solve_rate"],
                         "top1": r["top1"], "mean_top_yield": r.get("mean_top_yield") or 0,
                         "tip": f"solve {r['solve_rate']:.3f}, yield {r.get('mean_top_yield')}"})
    wchart = rv.ablation_line_svg(
        wpts, [("solve_rate", "solve", "#2a78d6", "#3987e5"),
               ("top1", "top-1", "#eb6834", "#d95926"),
               ("mean_top_yield", "top-route yield", "#1baf7a", "#199e70")],
        "yield_weight", f"yield_weight sweep ({rset})", value_fmt="{:.2f}") if len(wpts) >= 2 else ""

    html = [f"<h2>{sec}. {rset}（先頭 {n} ターゲット）</h2>",
            f"<h3>{sec}.1 全条件の比較表</h3>", _table(rows),
            "<p class='cap'>solve/top-k は PaRoutes 公式指標。yield=top ルートの各ステップ予測収率の"
            "幾何平均、SA-ease=葉分子の合成容易性、convergence=断片サイズ均衡(CDScore)、#rxn=反応段数。"
            "queries/tgt は探索強度（cache非依存）。<code>stock</code> を基準色で強調。</p>",
            f"<h3>{sec}.2 報酬ごとの top ルート性質</h3>",
            "<div class='charts'>",
            bars("mean_top_yield", f"top-route 平均予測収率 ({rset})", "{:.2f}"),
            bars("mean_top_cdscore", f"top-route 収束性 CDScore ({rset})", "{:.2f}"),
            bars("mean_top_sa_ease", f"top-route SA 容易性 ({rset})", "{:.2f}"),
            bars("mean_n_reactions", f"top-route 反応段数 ({rset})", "{:.1f}"),
            "</div>",
            f"<h3>{sec}.3 yield_weight スイープ</h3>", wchart,
            f"<h3>{sec}.4 考察（{rset}）</h3>",
            "".join(f"<p>{p}</p>" for p in _discussion_html(rows))]
    md = [f"## {sec}. {rset}（先頭 {n} ターゲット）\n",
          f"### {sec}.1 全条件の比較表\n", _md_table(rows),
          f"\n### {sec}.2 考察\n", _discussion(rows)]
    return "".join(html), "\n".join(md)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    import report_viz as rv
    sets = [(rs, json.load(open(summary_path(rs)))) for rs in ROUTE_SETS
            if os.path.exists(summary_path(rs))]
    if not sets:
        print("no reward summaries yet under", EXP_DIR)
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    intro = ("PaRoutes の各セット先頭 N ターゲットを固定し、報酬（評価関数）だけを変えて "
             "ReactionT5-MCTS を実行。公式指標（solve / top-k）に加え、各報酬が選ぶ "
             "top ルートの性質（平均予測収率・SA 容易性・収束性 CDScore・反応段数）と "
             "コストを比較する。探索本体・シード・一段階モデルは全条件で共通。")

    md = [f"# ReactionT5-MCTS 報酬関数・重み設定の比較実験（n1 / n5）\n\n*Generated {now}*\n",
          intro + "\n"]
    body = ["<h1>ReactionT5-MCTS 報酬関数・重み設定の比較実験（n1 / n5）</h1>",
            f"<p><em>Generated {now}</em></p>", f"<p>{intro}</p>"]
    for i, (rset, data) in enumerate(sets, start=1):
        h, m = _set_blocks(data["rows"], data["meta"]["n_targets"], rset, rv, i)
        body.append(h)
        md.append(m)

    with open(os.path.join(REPORTS, "reward_report.md"), "w") as fh:
        fh.write("\n".join(md))
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Reward Comparison</title><style>{STYLE}</style></head>"
           f"<body>{''.join(body)}</body></html>")
    with open(os.path.join(REPORTS, "reward_report.html"), "w") as fh:
        fh.write(doc)
    print("wrote reports/reward_report.md and reward_report.html for sets:",
          [s for s, _ in sets])


def _by(rows):
    return {r["label"]: r for r in rows}


def _discussion_html(rows):
    d = _by(rows)
    out = []
    st = d.get("stock")

    def cmp_line(metric, label):
        if not st:
            return None
        best = max((r for r in rows if r.get(metric) is not None),
                   key=lambda r: r[metric], default=None)
        if best is None:
            return None
        return (f"<strong>{label}</strong>: 最大は <code>{best['label']}</code> "
                f"({best[metric]:.3f})、stock は {st.get(metric):.3f}。")

    for m, lab in [("mean_top_yield", "top ルート平均収率"),
                   ("mean_top_cdscore", "収束性 (CDScore)"),
                   ("mean_top_sa_ease", "SA 容易性")]:
        line = cmp_line(m, lab)
        if line:
            out.append(line)
    # yield weight trend
    ys = [(0.5, d.get("yield_w0.5")), (1.0, d.get("yield")),
          (2.0, d.get("yield_w2")), (4.0, d.get("yield_w4"))]
    ys = [(w, r.get("mean_top_yield")) for w, r in ys if r and r.get("mean_top_yield") is not None]
    if len(ys) >= 2:
        trend = "上昇" if ys[-1][1] >= ys[0][1] else "低下"
        out.append(f"<strong>yield_weight の効果</strong>: 重みを {ys[0][0]}→{ys[-1][0]} と"
                   f"上げると top ルート平均収率は {ys[0][1]:.3f}→{ys[-1][1]:.3f} と{trend}。")
    out.append("solve 率は報酬にほぼ依存しない（在庫貪欲ロールアウトが解到達を駆動）一方、"
               "top ルートの性質と top-k は報酬で明確に変わる。目的（高収率・収束的・短経路・"
               "入手容易）に応じて報酬項と重みを選べる。")
    return out


def _discussion(rows):
    import re
    return "\n".join("- " + re.sub("<[^>]+>", "", p) for p in _discussion_html(rows))


if __name__ == "__main__":
    main()
