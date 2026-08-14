"""Rich HTML fragments for the PaRoutes benchmark report (HTML-only extras).

Provides three things consumed by make_report.py:
  * mol_svg(smiles)              -> inline RDKit SVG for one molecule
  * route_tree_html(route)       -> nested HTML depiction of a retrosynthetic route
  * ablation_chart_svg(rows)     -> inline SVG bar chart of the hyperparameter sweep

Molecule depictions are drawn on white cards (chemistry convention) so they read
in both light and dark page themes.  Colours for the chart come from the
data-viz reference palette (validated).
"""
from __future__ import annotations

import html
from typing import Any, Dict, List

# validated data-viz palette slots actually used here
SERIES = "#2a78d6"        # baseline bar (categorical slot 1, light)
SERIES_MUTED = "#9db8d6"  # variant bars (desaturated slot 1)
GOOD = "#008300"          # status: solved (reserved status hue + icon/label)
MUTED_INK = "#8a8a86"


def mol_svg(smiles: str, width: int = 150, height: int = 110) -> str:
    """Return an inline SVG depiction of a molecule (empty string on failure)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("unparsable")
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = d.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 1
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        # strip the XML declaration so it embeds inline cleanly
        return svg[svg.find("<svg"):]
    except Exception:
        return ""


def _mol_card(smiles: str, in_stock: bool) -> str:
    svg = mol_svg(smiles)
    badge = ('<span class="leaf-badge stock">in&nbsp;stock</span>'
             if in_stock else "")
    body = svg or f'<code class="mol-fallback">{html.escape(smiles[:40])}</code>'
    ring = " leaf-open" if (not in_stock and not svg) else ""
    stock_cls = " stock" if in_stock else ""
    return (f'<div class="mol-card{stock_cls}{ring}" title="{html.escape(smiles)}">'
            f'{body}{badge}</div>')


def route_tree_html(route: Dict[str, Any]) -> str:
    """Render a route dict (PaRoutes/AiZynth mol/reaction tree) as nested HTML.

    Product at the left; each reaction fans out to its reactant children.  A
    molecule with no children that is in stock is a purchasable leaf.
    """
    def render(node: Dict[str, Any]) -> str:
        if node.get("type") == "reaction":
            kids = "".join(render(c) for c in node.get("children", []))
            return (f'<div class="rxn"><div class="rxn-arrow" '
                    f'title="one retrosynthetic step">&#8594;</div>'
                    f'<div class="rxn-reactants">{kids}</div></div>')
        # molecule node
        children = node.get("children", [])
        card = _mol_card(node.get("smiles", ""), bool(node.get("in_stock")))
        if not children:
            return f'<div class="mol-node">{card}</div>'
        inner = "".join(render(c) for c in children)
        return f'<div class="mol-node">{card}{inner}</div>'

    return f'<div class="route-tree">{render(route)}</div>'


def _n_reactions(node: Dict[str, Any]) -> int:
    n, stack = 0, [node]
    while stack:
        cur = stack.pop()
        for ch in cur.get("children", []):
            if ch.get("type") == "reaction":
                n += 1
            stack.extend(ch.get("children", []))
    return n


# validated categorical slots 1-4 (light, dark) for the quality lines
_QSERIES = [
    ("solve_rate", "solve", "#2a78d6", "#3987e5"),
    ("top1",       "top-1", "#eb6834", "#d95926"),
    ("top5",       "top-5", "#1baf7a", "#199e70"),
    ("top10",      "top-10", "#eda100", "#c98500"),
]


def _fmt_num(v):
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def ablation_line_svg(points, series, xlabel, title, *, ymax=None,
                      value_fmt="{:.3f}", cost=False):
    """Generic multi-line chart over an ordered set of ablation points.

    points: list of dicts each with the metric keys + 'label' (x tick) and a
            'tip' string for hover.
    series: list of (key, legend_label, light_hex, dark_hex).
    Draws grid, y-axis, one polyline+markers per series, a legend, and a direct
    label at each series' last point (secondary encoding so identity is never
    colour-alone).
    """
    W, H = 380, 250
    padL, padR, padT, padB = 44, 54, 34, 46
    plotw, ploth = W - padL - padR, H - padT - padB
    n = len(points)
    xs = [padL + (plotw * (i / (n - 1)) if n > 1 else plotw / 2) for i in range(n)]
    vals = [p.get(k, 0) or 0 for k, *_ in series for p in points]
    top = ymax if ymax is not None else (max(vals) if vals else 1) * 1.15 or 1.0
    if top <= 0:
        top = 1.0

    def y(v):
        return padT + ploth * (1 - v / top)

    P = [f'<svg viewBox="0 0 {W} {H}" class="abl-line" role="img" '
         f'aria-label="{html.escape(title)}" preserveAspectRatio="xMidYMid meet">']
    P.append(f'<text x="{padL}" y="16" class="abl-title">{html.escape(title)}</text>')
    # gridlines + y ticks
    for t in range(5):
        gv = top * t / 4
        yy = y(gv)
        P.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{padL+plotw}" y2="{yy:.1f}" class="grid"/>')
        P.append(f'<text x="{padL-6}" y="{yy+3:.1f}" class="ytick">{value_fmt.format(gv)}</text>')
    # x ticks
    for i, p in enumerate(points):
        P.append(f'<text x="{xs[i]:.1f}" y="{padT+ploth+16:.1f}" class="xtick">'
                 f'{html.escape(str(p["label"]))}</text>')
    P.append(f'<text x="{padL+plotw/2:.1f}" y="{H-6:.1f}" class="xlabel">{html.escape(xlabel)}</text>')
    # one polyline per series (colour via CSS class -> theme-aware)
    for key, lab, cl, cd in series:
        cls = "s-" + str(key).replace("_", "-")
        pts_xy = [(xs[i], y(points[i].get(key, 0) or 0)) for i in range(n)]
        d = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts_xy)
        P.append(f'<polyline points="{d}" class="ln {cls}"/>')
        for i, (x, yy) in enumerate(pts_xy):
            tip = points[i].get("tip", "")
            P.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="3.2" class="mk {cls}">'
                     f'<title>{html.escape(lab)} @ {html.escape(str(points[i]["label"]))}: '
                     f'{_fmt_num(points[i].get(key,0))}'
                     f'{("  " + html.escape(tip)) if tip else ""}</title></circle>')
        ex, ey = pts_xy[-1]
        P.append(f'<text x="{ex+5:.1f}" y="{ey+3:.1f}" class="endlab {cls}">'
                 f'{html.escape(lab)}</text>')
    P.append("</svg>")
    return "".join(P)


def ablation_chart_svg(rows: List[Dict[str, Any]]) -> str:
    """Inline SVG bar chart of in-stock fraction per sweep config.

    rows: list of dicts {tag, changed, instock, solved(bool), time}.
    Single measure (in-stock frac) -> single-hue bars; the baseline is drawn in
    full slot-1 blue, one-parameter variants in a desaturated blue, and each bar
    carries a solved / not-solved status marker (icon + label, never colour
    alone).
    """
    W, H = 720, 300
    padL, padR, padT, padB = 48, 16, 24, 84
    plotw = W - padL - padR
    ploth = H - padT - padB
    n = len(rows)
    gap = 18
    bw = (plotw - gap * (n - 1)) / n
    ymax = 1.0

    def y(v: float) -> float:
        return padT + ploth * (1 - v / ymax)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Ablation: mean in-stock fraction per hyperparameter config" '
             f'class="ablation-svg" preserveAspectRatio="xMidYMid meet">']
    # gridlines + y labels
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y(t)
        parts.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-padR}" y2="{yy:.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{padL-6}" y="{yy+3:.1f}" class="ytick">{t:.2f}</text>')
    # bars
    for i, r in enumerate(rows):
        x = padL + i * (bw + gap)
        h = ploth * (r["instock"] / ymax)
        yy = padT + ploth - h
        fill = SERIES if r["baseline"] else SERIES_MUTED
        parts.append(
            f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h:.1f}" '
            f'rx="4" fill="{fill}"><title>{html.escape(r["label"])}: in-stock '
            f'{r["instock"]:.3f}, {"solved" if r["solved"] else "not solved"} '
            f'(3/3? {r["solve_txt"]}), {r["time"]:.1f}s/target</title></rect>')
        # value label
        parts.append(f'<text x="{x+bw/2:.1f}" y="{yy-5:.1f}" class="barval">'
                     f'{r["instock"]:.2f}</text>')
        # status marker under axis
        icon = "✔" if r["solved"] else "✗"
        icls = "ok" if r["solved"] else "no"
        parts.append(f'<text x="{x+bw/2:.1f}" y="{padT+ploth+16:.1f}" '
                     f'class="status {icls}">{icon}</text>')
        # config label (two lines: tag, changed param)
        parts.append(f'<text x="{x+bw/2:.1f}" y="{padT+ploth+34:.1f}" '
                     f'class="cfgname">{html.escape(r["tag"])}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{padT+ploth+48:.1f}" '
                     f'class="cfgchg">{html.escape(r["changed"])}</text>')
    # axis title
    parts.append(f'<text x="{padL}" y="{padT-10:.1f}" class="axtitle">'
                 f'mean best in-stock fraction (n1 sweep, 3 targets)</text>')
    parts.append("</svg>")
    return "".join(parts)
