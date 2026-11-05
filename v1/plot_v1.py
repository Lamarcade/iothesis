"""
Charts for the v1 scenario projection pipeline and its two key findings.

Reads a solved projection (`v1/project.py`) and the small fixtures used by
`v1/test_v1.py`, and renders interactive HTML into `Figs/v1/`.

Conventions are those of `shortterm/plot_production.py` and `scope3/plot_scope3.py`
-- same `THEME` dict, same `_style`/`_hbar` chrome, same light/dark handling, same
HTML output -- so all three sets of figures read as one system. `SCENARIO_COLOR`,
`THEME`, `FONT`, `SCENARIO_SYMBOL`, `DIVERGING` are imported, not redefined: a
scenario keeps its hue across every figure in the repo (the rule stated in
`plot_scope3.py`'s own docstring).

Colour, reused rather than invented
------------------------------------
No new palette was validated for this module -- every hex below already clears
`shortterm/palette_check.py` as a member of an existing, in-repo dictionary. What
is new here is only the *pairing*: which two or three already-validated colours
sit next to each other in a given chart.

    RESULT_COLOR   base = SCENARIO_COLOR["Baseline"] gray, achieved = the blue
                   used identically as SCOPE_COLOR["scope3"] and DAPS
    STATUS_COLOR   infeasible = DIVERGING's red pole, feasible = SCOPE_COLOR's
                   scope2 aqua
    STAGE_COLOR    the funnel for fig_nonnegative_fix: gray -> red -> red ->
                   the same aqua as STATUS_COLOR["feasible"]
    FLOW_SEQUENTIAL  the 5-step sequential blue ramp from `plot_scope3.py`,
                   extended with each theme's own `plane` colour as the zero anchor

Five figures ground the pipeline itself (what the scenario does to output, what
KRAS achieves against the emission target, who gains and loses intermediate
sales, and how the flow matrix restructures); the sixth and the row-margin-trap
figure are the two most load-bearing modelling choices made in `v1/constraints.py`
and `v1/project.py`, turned into pictures rather than left as printed test output.

Usage
-----
    python v1/plot_v1.py --all
    python v1/plot_v1.py --all --theme dark
    python v1/plot_v1.py --figure row_margin_trap
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shortterm.plot_production import (  # noqa: E402
    DIVERGING,
    FONT,
    SCENARIO_COLOR,
    THEME,
    exio_sector_to_theme,
)
from scope3.aggregate import sector_themes  # noqa: E402
from v1.constraints import unchanged_technology_margins  # noqa: E402
from v1.project import DEFAULT_MAX_CAP_ITERS, load_projection, project, save_projection  # noqa: E402
from v1.resolution import RESOLUTIONS, split_label  # noqa: E402
from v1.scenario import scenario_x  # noqa: E402
from v1.system import DEFAULT_YEAR, build_io_system  # noqa: E402
from v1.test_v1 import _synthetic_system  # noqa: E402

FIG_DIR = REPO_ROOT / "Figs" / "v1"

DEFAULT_SCENARIO = "SWUC"
DEFAULT_HORIZON = "2030"
DEFAULT_BETA = 0.7

# ---------------------------------------------------------------------------
# Palette (see module docstring for provenance of every hex)
# ---------------------------------------------------------------------------

RESULT_COLOR = {
    "base": ("#898781", "#898781"),        # de-emphasis gray, as Baseline everywhere
    "achieved": ("#2a78d6", "#3987e5"),    # the repo's standard blue
}
TARGET_MARK = ("#0b0b0b", "#ffffff")       # tick marking the target, max contrast

STATUS_COLOR = {
    "infeasible": ("#e34948", "#e34948"),  # DIVERGING's red pole
    "feasible": ("#1baf7a", "#199e70"),    # SCOPE_COLOR's scope2 aqua
}

STAGE_LABELS = ["2021 (observed)", "y0 (a priori)", "y' (uncapped)", "y' (capped)"]
#: gray -> red -> red -> green: 3 hues across 4 bars ("a priori" and "uncapped"
#: are both "the problem is present," so they share the hue; height carries the
#: difference). Validated via shortterm/palette_check.py --pairs adjacent: the
#: only FAIL is the deliberate de-emphasis gray reading as low-chroma, the same
#: false positive SCENARIO_COLOR's own Baseline gray produces -- not a defect.
#: The aqua's light-mode contrast WARN is the same one already accepted for
#: SCOPE_COLOR["scope2"]; the relief rule (every bar prints its value) applies.
STAGE_COLOR = ["#898781", "#e34948", "#e34948", "#1baf7a"]
STAGE_COLOR_DARK = ["#898781", "#e34948", "#e34948", "#199e70"]

#: Same ramp as scope3/plot_scope3.py:103-106, extended with the plane colour
#: as the zero anchor so an empty cell recedes into the background.
FLOW_SEQUENTIAL = {
    "light": ["#f9f9f7", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"],
    "dark": ["#0d0d0d", "#0d366b", "#184f95", "#2a78d6", "#3987e5", "#86b6ef"],
}


def _k(theme: str) -> int:
    return 0 if theme == "light" else 1


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class _CachedProjection:
    """
    A `Projection`-shaped accessor for a cache hit.

    `load_projection` reads back only `table`/`Z_new`/`A_new` -- `save_projection`
    never persists `.checks` (row/col residuals, cap statistics). Every figure
    below that only reads `.table` (plus `Z_new`/`x_target` for the heatmap) can
    use this and skip a re-solve; `fig_nonnegative_fix`, which specifically needs
    the real cap statistics, does not use this path -- see its own docstring.
    """

    def __init__(self, table: pd.DataFrame, Z_new: np.ndarray, A_new: np.ndarray,
                x_target: np.ndarray, n: int):
        self.table, self.Z_new, self.A_new = table, Z_new, A_new
        self.x_target, self.n = x_target, n


def _primary_run(scenario: str, year: str, beta: float, resolution: str,
                 table_type: str, base_year: int = DEFAULT_YEAR,
                 rebuild: bool = False):
    """
    The run the CLI parameters describe -- cache-first, same pattern as
    `scope3/plot_scope3.py:131-148 kras_frames`. Only for figures that read
    `.table`/`.Z_new`/`.x_target`, never `.checks`.
    """
    system = build_io_system(base_year, resolution, table_type,
                             use_cache=not rebuild, verbose=False)
    if not rebuild:
        cached = load_projection(scenario, year, resolution, table_type, beta,
                                 base_year)
        if cached is not None:
            tbl, Z_new, A_new = cached
            x_target = scenario_x(scenario, year, system, base_year, verbose=False)
            return system, _CachedProjection(tbl, Z_new, A_new, x_target, system.n)
    x_target = scenario_x(scenario, year, system, base_year, verbose=False)
    proj, _ = project(system, x_target, beta=beta, scenario=scenario, year=year,
                      verbose=False)
    save_projection(proj, beta)
    return system, proj


# ---------------------------------------------------------------------------
# Chart helpers (identical chrome to scope3/plot_scope3.py:181-238)
# ---------------------------------------------------------------------------

def _sector_themes(labels: list[str], table_type: str) -> list[str]:
    """
    Theme per label, whatever the resolution.

    `world` labels are raw EXIOBASE sector names, which `exio_sector_to_theme`
    keys on. Every other resolution labels columns `"Bloc | NGFS sector"`, where
    that lookup misses every key and would silently fill the hovers with "?".
    So the sector half is split off first and tried against both mappings.
    """
    exio = exio_sector_to_theme(table_type)
    ngfs = sector_themes("ngfs", table_type)
    out = []
    for lab in labels:
        sector = split_label(lab)[1]
        out.append(exio.get(sector) or ngfs.get(sector) or "?")
    return out


def _wrap(text: str, width: int = 108) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def _style(fig: go.Figure, theme: str, title: str, subtitle: str = "",
           height: int = 900, left: int = 200) -> go.Figure:
    t = THEME[theme]
    lines = 1 + subtitle.count("<br>") if subtitle else 0
    top = 96 + 18 * lines
    fig.update_layout(
        template=None,
        title=dict(
            text=f"<b>{title}</b>" + (f"<br><span style='font-size:13px;"
                                      f"color:{t['ink2']}'>{subtitle}</span>"
                                      if subtitle else ""),
            font=dict(size=19, color=t["ink"], family=FONT), x=0.01, xanchor="left",
            y=1, yanchor="top", pad=dict(t=18),
        ),
        paper_bgcolor=t["plane"], plot_bgcolor=t["surface"],
        font=dict(family=FONT, color=t["ink2"], size=12),
        height=height, margin=dict(l=left, r=40, t=top, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left",
                    x=0.01, font=dict(color=t["ink2"], size=12),
                    bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        hovermode="closest",
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=t["grid"], gridwidth=1, zeroline=False,
                     linecolor=t["axis"], linewidth=1,
                     tickfont=dict(color=t["muted"], size=11),
                     title_font=dict(color=t["ink2"], size=12))
    fig.update_yaxes(showgrid=False, linecolor=t["axis"], linewidth=1,
                     ticks="", automargin=True,
                     tickfont=dict(color=t["ink2"], size=10),
                     title_font=dict(color=t["ink2"], size=12))
    return fig


def _hbar(y: list, x: np.ndarray, name: str, color, theme: str, unit: str,
          showlegend: bool = True, hover_extra: list[str] | None = None) -> go.Bar:
    t = THEME[theme]
    if hover_extra is None:
        template = f"%{{y}}<br><b>{name}</b> %{{x:,.1f}} {unit}<extra></extra>"
        custom = None
    else:
        template = (f"%{{y}}<br><b>{name}</b> %{{x:,.1f}} {unit}"
                    f"<br>%{{customdata}}<extra></extra>")
        custom = hover_extra
    return go.Bar(
        y=y, x=x, name=name, orientation="h", showlegend=showlegend,
        marker=(dict(color=color, line=dict(color=t["surface"], width=1))
               if not isinstance(color, np.ndarray) else
               dict(color=color, colorscale=DIVERGING, cmid=0,
                    line=dict(color=t["surface"], width=1))),
        customdata=custom, hovertemplate=template,
    )


# ---------------------------------------------------------------------------
# Figure 1 -- what the scenario does to output, before any rebalancing
# ---------------------------------------------------------------------------

def fig_scenario_growth(theme: str = "light", scenario: str = DEFAULT_SCENARIO,
                        year: str = DEFAULT_HORIZON, beta: float = DEFAULT_BETA,
                        resolution: str = "world", table_type: str = "pxp",
                        rebuild: bool = False) -> go.Figure:
    system, proj = _primary_run(scenario, year, beta, resolution, table_type,
                                rebuild=rebuild)
    t = THEME[theme]
    tbl = proj.table.copy()
    tbl["theme"] = _sector_themes(list(tbl.index), table_type)

    ranked = tbl.reindex(tbl["growth_pct"].abs().sort_values().index).tail(24)

    fig = go.Figure()
    fig.add_trace(_hbar(
        list(ranked.index), ranked["growth_pct"].to_numpy(dtype=float),
        "output growth", ranked["growth_pct"].to_numpy(dtype=float), theme, "%",
        showlegend=False,
        hover_extra=[f"{th} · {sc:,.0f} → {tg:,.0f} M.EUR" for th, sc, tg in
                    zip(ranked["theme"], ranked["x_base"], ranked["x_target"])],
    ))
    fig.update_layout(bargap=0.2)
    fig = _style(
        fig, theme,
        f"What the scenario does to output before any rebalancing — "
        f"{scenario} {year}",
        _wrap(f"x^{{T,k}} projected by shortterm/project_production.py, a "
              f"mechanical rescaling of {system.year} EXIOBASE output "
              f"({resolution}, n={system.n}). The 24 sectors with the largest "
              "absolute change; hover for the sector's theme and levels."),
        height=60 + 26 * len(ranked),
    )
    fig.update_xaxes(title_text="Output growth, base year → horizon (%)")
    fig.add_vline(x=0, line=dict(color=t["axis"], width=1))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- the emission target, base vs achieved, against the tick of target
# ---------------------------------------------------------------------------

EMIS_TOP_N = 20


def fig_emission_targets(theme: str = "light", scenario: str = DEFAULT_SCENARIO,
                         year: str = DEFAULT_HORIZON, beta: float = DEFAULT_BETA,
                         resolution: str = "world", table_type: str = "pxp",
                         rebuild: bool = False) -> go.Figure:
    system, proj = _primary_run(scenario, year, beta, resolution, table_type,
                                rebuild=rebuild)
    k, t = _k(theme), THEME[theme]
    tgt = proj.table[proj.table["has_emis_target"]]
    order = tgt["b_base"].sort_values().index[-EMIS_TOP_N:]
    sub = tgt.loc[order] / 1e9  # kg -> Mt for b_base/b_target/b_achieved

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        column_widths=[0.6, 0.4],
    )
    fig.add_trace(_hbar(list(order), sub["b_base"].to_numpy(dtype=float),
                        "base (2021)", RESULT_COLOR["base"][k], theme, "Mt"),
                 row=1, col=1)
    fig.add_trace(_hbar(list(order), sub["b_achieved"].to_numpy(dtype=float),
                        "achieved", RESULT_COLOR["achieved"][k], theme, "Mt"),
                 row=1, col=1)
    fig.add_trace(
        go.Scatter(
            y=list(order), x=sub["b_target"].to_numpy(dtype=float), mode="markers",
            name=f"target (β={beta:g})", showlegend=True,
            marker=dict(symbol="line-ns", size=13, line=dict(
                color=TARGET_MARK[k], width=2)),
            hovertemplate=f"%{{y}}<br>target %{{x:,.1f}} Mt<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(_hbar(list(order), tgt.loc[order, "procurement_effort"].to_numpy(dtype=float),
                        "procurement effort", RESULT_COLOR["achieved"][k], theme, "%",
                        showlegend=False),
                 row=1, col=2)

    n_unreach = int(tgt["target_unreachable"].sum())
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.1)
    fig = _style(
        fig, theme,
        f"Cutting embodied emissions by re-sourcing — {scenario} {year}, β={beta:g}",
        _wrap(f"Column margins are held (each sector keeps its input budget), so "
              f"KRAS only changes who a sector buys from. The achieved bar lands "
              f"on the target tick to numerical precision — {n_unreach} of "
              f"{len(tgt)} targeted sectors sit outside their sourcing band "
              f"(unreachable by arithmetic). Top {EMIS_TOP_N} by base emissions."),
        height=60 + 30 * len(order),
    )
    fig.update_xaxes(title_text="Emissions embodied in intermediate purchases (Mt CO2)",
                     row=1, col=1)
    fig.update_xaxes(title_text="Share of the input budget that changes supplier (%)",
                     row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- who loses and gains intermediate sales
# ---------------------------------------------------------------------------

def fig_sales_shift(theme: str = "light", scenario: str = DEFAULT_SCENARIO,
                    year: str = DEFAULT_HORIZON, beta: float = DEFAULT_BETA,
                    resolution: str = "world", table_type: str = "pxp",
                    rebuild: bool = False) -> go.Figure:
    _, proj = _primary_run(scenario, year, beta, resolution, table_type,
                           rebuild=rebuild)
    t = THEME[theme]
    s = proj.table["sales_change_pct"].sort_values()
    picked = pd.concat([s.head(8), s.tail(8)]).sort_values()

    fig = go.Figure()
    fig.add_trace(_hbar(list(picked.index), picked.to_numpy(dtype=float),
                        "sales change", picked.to_numpy(dtype=float), theme, "%",
                        showlegend=False))
    fig = _style(
        fig, theme,
        f"Rows are free: who loses and gains intermediate sales — {scenario} {year}",
        _wrap("With column margins held and rows free (see V1_HYPOTHESES.md §5), a "
              "sector's total sales to other sectors are an outcome of the "
              "rebalancing, not a constraint. Carbon-intensive suppliers are the "
              "ones asked to sell less."),
        height=60 + 28 * len(picked),
    )
    fig.update_xaxes(title_text="Change in total intermediate sales vs the "
                                "unchanged-technology counterfactual (%)")
    fig.add_vline(x=0, line=dict(color=t["axis"], width=1))
    return fig


# ---------------------------------------------------------------------------
# Figure 4 -- the row-margin trap (test 7, made visible)
# ---------------------------------------------------------------------------

def fig_row_margin_trap(theme: str = "light", **_ignored) -> go.Figure:
    """
    Same fixture as `v1/test_v1.py::test_row_margins_pin_the_total`: fixing the
    row margins pins `sum_j b'_j = sum_i s'_i u0_i` to a constant, so a uniform
    beta < 1 cannot be met. This is the chart that explains *why* v1/constraints.py
    leaves the rows free by default.
    """
    system = _synthetic_system()
    beta = 0.9
    k, t = _k(theme), THEME[theme]

    results = {}
    for label, include_row in (("rows fixed", True), ("rows free", False)):
        proj, _ = project(system, system.x.copy(), beta=beta,
                          include_row=include_row, verbose=False)
        tgt = proj.table[proj.table["has_emis_target"]]
        results[label] = float(np.max(
            np.abs(tgt["b_achieved"] - tgt["b_target"]) / np.abs(tgt["b_target"])))

    labels = list(results)
    values = np.array(list(results.values()))
    colors = [STATUS_COLOR["infeasible"][k], STATUS_COLOR["feasible"][k]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, marker=dict(color=colors),
        text=[f"{v:.1e}" if v < 1e-6 else f"{100 * v:.1f} %" for v in values],
        textposition="outside",
        textfont=dict(color=t["ink2"], size=12, family=FONT),
        hovertemplate="<b>%{x}</b> worst relative miss %{y:.3e}<extra></extra>",
    ))
    fig.update_yaxes(type="log", title_text="Worst relative miss on β "
                                            "|b' − β·b| / (β·b) — log scale")
    fig = _style(
        fig, theme, f"Why the row margins are free — the row-margin trap (β={beta:g})",
        _wrap("With the row margins fixed, sum_j b'_j = sum_i s'_i u0_i is a "
              "constant: the world total of embodied emissions cannot move, so "
              "KRAS can only shuffle it between sectors and every uniform target "
              "is missed. Freeing the rows lets each column reach β exactly. "
              "Small synthetic fixture, n=5 — the same one v1/test_v1.py asserts on."),
        height=560, left=90,
    )
    fig.update_xaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 -- the y' >= 0 fix (test 8, made visible)
# ---------------------------------------------------------------------------

def fig_nonnegative_fix(theme: str = "light", scenario: str = DEFAULT_SCENARIO,
                        year: str = DEFAULT_HORIZON, beta: float = DEFAULT_BETA,
                        resolution: str = "world", table_type: str = "pxp",
                        rebuild: bool = False) -> go.Figure:
    """
    Solves fresh in both directions rather than going through `_primary_run`'s
    cache: `cap_rows_used`/`cap_iterations` are exactly what this figure is
    about, and `save_projection` never persists them, so a cache hit would
    silently show 0/0 regardless of what actually happened. Two small extra
    KRAS solves (world resolution: a few seconds) buys honesty.
    """
    system = build_io_system(DEFAULT_YEAR, resolution, table_type,
                             use_cache=not rebuild, verbose=False)
    x_target = scenario_x(scenario, year, system, verbose=False)
    capped, _ = project(system, x_target, beta=beta, scenario=scenario,
                        year=year, verbose=False)
    uncapped, _ = project(system, x_target, beta=beta, scenario=scenario,
                          year=year, enforce_nonnegative_y=False, verbose=False)
    t = THEME[theme]

    counts = [capped.checks["negative_y_base"], capped.checks["negative_y_apriori"],
             int((uncapped.y_expost < 0).sum()), capped.checks["negative_y_expost"]]
    colors = STAGE_COLOR if theme == "light" else STAGE_COLOR_DARK
    n_capped = capped.checks.get("cap_rows_used", 0)
    n_passes = capped.checks.get("cap_iterations", 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=STAGE_LABELS, y=counts, marker=dict(color=colors),
        text=[str(c) for c in counts], textposition="outside",
        textfont=dict(color=t["ink2"], size=13, family=FONT),
        hovertemplate="<b>%{x}</b> %{y} sector(s)<extra></extra>",
    ))
    fig = _style(
        fig, theme,
        f"Guaranteeing y' ≥ 0 — {scenario} {year}, {system.n} sectors ({resolution})",
        _wrap(f"x^{{T,k}} does not preserve x = Ax + y, so even the a-priori y0 = "
              f"(I-A)x' can be negative. Freeing the rows (needed to hit β at all, "
              "see the row-margin-trap figure) lets the free rebalancing push a "
              f"different set negative. The iterative cap in v1/project.py fixed "
              f"it here by pinning {n_capped} row(s) over {n_passes} pass(es)."),
        height=560, left=90,
    )
    fig.update_yaxes(title_text="Sectors with negative final demand")
    fig.update_xaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Figure 6 -- how the flow matrix restructures
# ---------------------------------------------------------------------------

HEATMAP_N_BUYERS = 10
HEATMAP_N_SUPPLIERS_PER_BUYER = 2


def fig_structural_heatmap(theme: str = "light", scenario: str = DEFAULT_SCENARIO,
                           year: str = DEFAULT_HORIZON, beta: float = DEFAULT_BETA,
                           resolution: str = "world", table_type: str = "pxp",
                           rebuild: bool = False) -> go.Figure:
    system, proj = _primary_run(scenario, year, beta, resolution, table_type,
                                rebuild=rebuild)
    t = THEME[theme]
    labels = np.array(system.labels)
    _, _, Z0 = unchanged_technology_margins(system.A, proj.x_target)
    Z_new = proj.Z_new

    buyers = (proj.table["procurement_effort"].sort_values(ascending=False)
             .head(HEATMAP_N_BUYERS).index)
    buyer_idx = [system.labels.index(s) for s in buyers]
    suppliers = set()
    for j in buyer_idx:
        top = np.argsort(-Z0[:, j])[:HEATMAP_N_SUPPLIERS_PER_BUYER]
        suppliers.update(int(i) for i in top if Z0[i, j] > 0)
    supplier_idx = sorted(suppliers)

    Z0_sub = Z0[np.ix_(supplier_idx, buyer_idx)]
    Z1_sub = Z_new[np.ix_(supplier_idx, buyer_idx)]
    y_labels = list(labels[supplier_idx])
    x_labels = list(labels[buyer_idx])

    floor = min(Z0_sub[Z0_sub > 0].min() if (Z0_sub > 0).any() else 1.0,
               Z1_sub[Z1_sub > 0].min() if (Z1_sub > 0).any() else 1.0)
    vmax = float(np.log10(max(Z0_sub.max(), Z1_sub.max())))
    vmin = float(np.log10(floor))

    colorscale = [[i / (len(FLOW_SEQUENTIAL[theme]) - 1), c]
                  for i, c in enumerate(FLOW_SEQUENTIAL[theme])]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
        subplot_titles=(f"Unchanged technology at x^{{{year}}}", "Rebalanced (KRAS)"),
    )
    for i, (M, showscale) in enumerate([(Z0_sub, False), (Z1_sub, True)]):
        disp = np.where(M > 0, np.log10(np.where(M > 0, M, 1.0)), np.nan)
        trace = go.Heatmap(
            z=disp, x=x_labels, y=y_labels, colorscale=colorscale,
            zmin=vmin, zmax=vmax, xgap=1, ygap=1, showscale=showscale,
            hovertemplate="%{y} → %{x}<br>%{customdata:,.0f} M.EUR<extra></extra>",
            customdata=M,
        )
        if showscale:
            trace.colorbar = dict(title="M.EUR<br>(log10)", thickness=12, len=0.8,
                                  x=1.02, tickfont=dict(color=t["muted"], size=10),
                                  title_font=dict(color=t["ink2"], size=11))
        fig.add_trace(trace, row=1, col=i + 1)

    fig = _style(
        fig, theme,
        f"How the flow matrix restructures — {scenario} {year}, β={beta:g}",
        _wrap(f"The {HEATMAP_N_BUYERS} sectors with the largest procurement shift "
              "and their principal base-year suppliers. Left: intermediate flows "
              "at unchanged 2021 technology, output rescaled to the scenario. "
              "Right: after KRAS rebalancing under the emission target."),
        height=110 + 26 * len(y_labels), left=190,
    )
    fig.update_xaxes(side="top", tickangle=45, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=12, color=t["ink"], family=FONT)
    fig.update_layout(margin=dict(t=140))
    return fig


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

FIGURES = {
    "scenario_growth": fig_scenario_growth,
    "emission_targets": fig_emission_targets,
    "sales_shift": fig_sales_shift,
    "row_margin_trap": fig_row_margin_trap,
    "nonnegative_fix": fig_nonnegative_fix,
    "structural_heatmap": fig_structural_heatmap,
}

#: fig_row_margin_trap needs no scenario/run parameters -- it is entirely
#: self-contained on the synthetic fixture.
_SCENARIO_INDEPENDENT = {"row_margin_trap"}


def run_all(output_dir: Path | str = FIG_DIR, theme: str = "light",
           scenario: str = DEFAULT_SCENARIO, year: str = DEFAULT_HORIZON,
           beta: float = DEFAULT_BETA, resolution: str = "world",
           table_type: str = "pxp", rebuild: bool = False,
           verbose: bool = True) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, builder in FIGURES.items():
        kwargs = {"theme": theme}
        if name not in _SCENARIO_INDEPENDENT:
            kwargs.update(scenario=scenario, year=year, beta=beta,
                         resolution=resolution, table_type=table_type,
                         rebuild=rebuild)
        fig = builder(**kwargs)
        suffix = "" if theme == "light" else "_dark"
        path = out / f"{name}{suffix}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        written.append(path)
        if verbose:
            print(f"  {path}")
    return written


if __name__ == "__main__":
    import argparse

    from shortterm.project_production import available_scenarios

    ap = argparse.ArgumentParser(
        description="Charts for the v1 scenario projection pipeline.")
    ap.add_argument("--all", action="store_true", help="Render every figure.")
    ap.add_argument("--figure", choices=sorted(FIGURES), default=None)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO, choices=available_scenarios())
    ap.add_argument("--year", default=DEFAULT_HORIZON)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--resolution", choices=RESOLUTIONS, default="world")
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--output-dir", default=str(FIG_DIR))
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore v1's system/projection caches.")
    args = ap.parse_args()

    common = dict(theme=args.theme, scenario=args.scenario, year=args.year,
                 beta=args.beta, resolution=args.resolution,
                 table_type=args.table_type, rebuild=args.rebuild)

    if args.all:
        print(f"Rendering {len(FIGURES)} figures ({args.scenario} {args.year} "
              f"β={args.beta:g}, {args.theme}):")
        run_all(args.output_dir, **common)
    elif args.figure:
        kwargs = {"theme": args.theme}
        if args.figure not in _SCENARIO_INDEPENDENT:
            kwargs.update({k: v for k, v in common.items() if k != "theme"})
        fig = FIGURES[args.figure](**kwargs)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.theme == "light" else "_dark"
        path = out / f"{args.figure}{suffix}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"Saved: {path}")
    else:
        ap.print_help()
