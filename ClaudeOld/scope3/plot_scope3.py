"""
Charts for the sectoral scope 3 analysis.

Reads the world-aggregated system (`scope3/aggregate.py`), the scope
decomposition (`scope3/scopes.py`), the KRAS decarbonisation targets
(`scope3/kras_scope3.py`) and the carbon cost model (`scope3/carbon_cost.py`),
and renders interactive HTML into `Figs/scope3/`.

Conventions are those of shortterm/plot_production.py -- same THEME dict, same
`_style` helper, same light/dark handling, same HTML output -- so the two sets
of figures read as one system.

Colour, computed rather than chosen
-----------------------------------
Three palettes, each doing one job.

**Scopes (categorical, 3 slots).** The reference palette's first three slots,
the only subset documented to clear the *all-pairs* gates in both modes. Checked
here with `shortterm/palette_check.py` (a port of the skill's validator; this
machine has no node):

    light  worst all-pairs CVD dE 9.2 (target 8), normal-vision 24.0 (floor 15)
    dark   worst all-pairs CVD dE 9.4,             normal-vision 20.9
    one WARN: aqua sits at 2.74:1 on the light surface, so the relief rule
    applies -- every share is printed, nothing is encoded by colour alone

Hues are assigned by role, not by slot order: **scope 3 takes the blue** (slot 1,
the strongest contrast) because it is the subject of these charts; scope 1 takes
orange, scope 2 aqua. blue/orange/green was tried first and rejected -- green
against orange measures dE 3.2 in protanopia.

**Targets (ordinal, 3 steps).** Base / -30 % / -60 % is an ordered sequence, so
it gets one hue stepped by ambition, with the de-emphasis gray for the reference
(the convention plot_production.py uses for Baseline). Measured: worst pair
normal-vision dE 15.3 light / 17.0 dark, worst CVD 13.0 / 11.5, every step above
the 2:1 ordinal contrast floor.

**Scenarios.** `SCENARIO_COLOR` is imported from plot_production.py unchanged --
a scenario keeps its hue across every figure in the repo.

Usage
-----
    python scope3/plot_scope3.py --all
    python scope3/plot_scope3.py --all --granularity gics --theme dark
    python scope3/plot_scope3.py --figure carbon_cost
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

from scope3.aggregate import build_world_system  # noqa: E402
from scope3.carbon_cost import carbon_bill, carbon_price, carbon_price_table  # noqa: E402
from scope3.kras_scope3 import DEFAULT_TARGETS, decarbonise, load_result, save_result  # noqa: E402
from scope3.scopes import KG_PER_MT, scope_table  # noqa: E402
from shortterm.plot_production import (  # noqa: E402
    FONT, SCENARIO_COLOR, SCENARIO_SYMBOL, THEME,
)

FIG_DIR = REPO_ROOT / "Figs" / "scope3"

YEAR, TABLE_TYPE = 2019, "pxp"
#: The scenario and year whose carbon price scales the cost figures. SWUC 2030
#: is the highest of the five, so it shows the exposure at its widest; every
#: other scenario is the same picture rescaled (the cost is linear in tau).
COST_SCENARIO, COST_YEAR = "SWUC", "2030"

# ---------------------------------------------------------------------------
# Palettes (see the module docstring for how each was validated)
# ---------------------------------------------------------------------------

SCOPE_COLOR = {
    #           light      dark
    "scope1": ("#eb6834", "#d95926"),   # orange -- the sector's own stack
    "scope2": ("#1baf7a", "#199e70"),   # aqua   -- purchased electricity
    "scope3": ("#2a78d6", "#3987e5"),   # blue   -- the subject of these charts
}
SCOPE_LABEL = {
    "scope1": "Scope 1 — direct",
    "scope2": "Scope 2 — purchased electricity",
    "scope3": "Scope 3 — upstream",
}
SCOPE_ORDER = ["scope1", "scope2", "scope3"]

#: Ordered by ambition, one hue, reference in the de-emphasis gray.
TARGET_COLOR = {
    "base": ("#898781", "#898781"),
    "-30 %": ("#5598e7", "#3987e5"),
    "-60 %": ("#1c5cab", "#9ec5f4"),
}

#: Sequential blue, near-zero receding toward the surface in each mode.
SEQUENTIAL = {
    "light": ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"],
    "dark": ["#0d366b", "#184f95", "#2a78d6", "#3987e5", "#86b6ef"],
}


def _k(theme: str) -> int:
    return 0 if theme == "light" else 1


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _system(granularity: str):
    return build_world_system(YEAR, granularity, TABLE_TYPE, verbose=False)


def scopes_frame(granularity: str) -> pd.DataFrame:
    """Scope table in Mt CO2, sorted by scope 3 share."""
    tbl = scope_table(_system(granularity))
    out = tbl[["scope1", "scope2", "scope3_up", "cradle"]] / KG_PER_MT
    out.columns = ["scope1", "scope2", "scope3", "cradle"]
    out["scope3_share"] = 100 * tbl["scope3_share"]
    out["x"] = tbl["x"]
    return out.sort_values("scope3_share")


def kras_frames(granularity: str, targets=DEFAULT_TARGETS,
                verbose: bool = True) -> dict[float, pd.DataFrame]:
    """
    KRAS result per target, solved on first use and cached to disk afterwards.
    """
    sysm = _system(granularity)
    out = {}
    for r in targets:
        cached = load_result(sysm, r)
        if cached is None:
            if verbose:
                print(f"  solving KRAS {granularity} -{r:.0%} (first run)...")
            res = decarbonise(sysm, r, verbose=False)
            save_result(res, sysm)
            out[r] = res.table
        else:
            out[r] = cached[0]
    return out


def cost_frame(granularity: str, scenario: str = COST_SCENARIO,
               year: str = COST_YEAR) -> tuple[pd.DataFrame, float]:
    sysm = _system(granularity)
    tau = carbon_price(scenario, year)
    bill = carbon_bill(sysm, tau)
    bill["share_scope1"] *= 100
    bill["share_full"] *= 100
    bill["exposure"] *= 100
    bill["scope3_share"] *= 100
    return bill.sort_values("share_full"), tau


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 108) -> str:
    """Break a subtitle onto several lines -- a long one overflows the plane."""
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
    """
    Same chrome as shortterm/plot_production.py.

    Panels are titled by their x-axis rather than by `subplot_titles`: a
    horizontal legend and a subplot title both want the strip just above the
    plot area, and they collided.
    """
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


def _hbar(frame: pd.DataFrame, column: str, name: str, color: str,
          theme: str, unit: str, showlegend: bool = True,
          text: np.ndarray | None = None) -> go.Bar:
    """
    One horizontal bar series. The 1px surface line is the mandated spacer
    between adjacent fills -- it is what keeps stacked segments legible.
    """
    t = THEME[theme]
    return go.Bar(
        y=list(frame.index), x=frame[column].to_numpy(dtype=float),
        name=name, orientation="h", showlegend=showlegend,
        marker=dict(color=color, line=dict(color=t["surface"], width=1)),
        text=text, textposition="outside" if text is not None else "none",
        textfont=dict(color=t["ink2"], size=9, family=FONT),
        cliponaxis=False,
        hovertemplate=f"%{{y}}<br><b>{name}</b> %{{x:,.1f}} {unit}<extra></extra>",
    )


# ---------------------------------------------------------------------------
# Figure 1 -- where scope 3 lives
# ---------------------------------------------------------------------------

def fig_scope_decomposition(theme: str = "light",
                            granularity: str = "ngfs") -> go.Figure:
    f = scopes_frame(granularity)
    k, t = _k(theme), THEME[theme]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        column_widths=[0.55, 0.45],
    )

    share = f[["scope1", "scope2", "scope3"]].div(f["cradle"], axis=0) * 100
    for scope in SCOPE_ORDER:
        color = SCOPE_COLOR[scope][k]
        fig.add_trace(_hbar(f, scope, SCOPE_LABEL[scope], color, theme, "Mt"),
                      row=1, col=1)
        fig.add_trace(_hbar(share, scope, SCOPE_LABEL[scope], color, theme, "%",
                            showlegend=False),
                      row=1, col=2)

    # Relief rule: aqua is under 3:1 on the light surface, so the scope 3 share
    # is printed rather than left to colour.
    fig.add_trace(
        go.Scatter(
            y=list(f.index), x=[102] * len(f), mode="text",
            text=[f"{v:.0f}%" for v in f["scope3_share"]],
            textposition="middle right", showlegend=False,
            textfont=dict(color=t["ink2"], size=9, family=FONT),
            hoverinfo="skip", cliponaxis=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(barmode="stack", bargap=0.25)
    fig = _style(
        fig, theme, f"Where sectoral scope 3 lives — world, EXIOBASE {YEAR}",
        _wrap("Upstream scope 3 only (GHG Protocol categories 1-8): an IO table "
              "has no use phase. Sorted by scope 3 share; the printed figure is "
              "that share."),
        height=1000 if granularity == "ngfs" else 520,
    )
    fig.update_xaxes(title_text="Cradle-to-gate emissions (Mt CO2)", row=1, col=1)
    fig.update_xaxes(title_text="Composition (% of cradle-to-gate)",
                     range=[0, 118], row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- the KRAS targets
# ---------------------------------------------------------------------------

#: Sectors shown in the KRAS figure. Scope 3 spans four orders of magnitude
#: across the 36 NGFS sectors -- Geothermal carries 1.4 Mt against Market
#: Services' 6,039 -- so a bar chart of all of them leaves a third of the panel
#: visibly empty. Bars must start at zero, which rules out a log axis, so the
#: figure is cut to the sectors that carry the mass and the coverage is stated.
#: Figure 1 still shows every sector.
KRAS_TOP_N = {"ngfs": 20, "gics": 11}


def fig_kras_targets(theme: str = "light", granularity: str = "ngfs") -> go.Figure:
    frames = kras_frames(granularity)
    targets = sorted(frames)
    k, t = _k(theme), THEME[theme]

    ranked = frames[targets[0]]["scope3_base"].sort_values()
    n_show = min(KRAS_TOP_N.get(granularity, 20), len(ranked))
    order = ranked.index[-n_show:]
    coverage = 100 * ranked.loc[order].sum() / ranked.sum()
    base = frames[targets[0]].loc[order]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        column_widths=[0.55, 0.45],
    )

    series = [("base", base["scope3_base"] / KG_PER_MT, None)]
    for r in targets:
        lab = f"-{r:.0%}".replace("%", " %")
        series.append((lab, frames[r].loc[order, "targeted_exante"] / KG_PER_MT, r))

    for name, values, r in series:
        color = TARGET_COLOR[name][k]
        fig.add_trace(
            go.Bar(y=list(order), x=values.to_numpy(dtype=float), name=name,
                   orientation="h", legendgroup=name,
                   marker=dict(color=color, line=dict(color=t["surface"], width=1)),
                   hovertemplate=f"%{{y}}<br><b>{name}</b> %{{x:,.1f}} Mt<extra></extra>"),
            row=1, col=1,
        )
        # The base has no procurement shift, but the slot has to exist or the
        # right panel centres two bars where the left centres three and the
        # rows stop lining up across the shared axis.
        eff = (np.zeros(len(order)) if r is None
               else frames[r].loc[order, "procurement_effort"].to_numpy(dtype=float))
        fig.add_trace(
            go.Bar(y=list(order), x=eff, name=name, orientation="h",
                   legendgroup=name, showlegend=False,
                   marker=dict(color=color, line=dict(color=t["surface"], width=1)),
                   hoverinfo="skip" if r is None else None,
                   hovertemplate=None if r is None else
                   (f"%{{y}}<br><b>{name}</b> %{{x:,.1f}} % of the "
                    "input budget moves<extra></extra>")),
            row=1, col=2,
        )

    # Sectors that could not reach the target get a printed mark: the shortfall
    # is the finding, and it must not be left to a missing bar length.
    notes = []
    for r in targets:
        short = frames[r][frames[r]["gap_vs_target"] < -0.5]
        for sec, row in short.iterrows():
            why = "below its sourcing floor" if row["target_below_floor"] else "conflict"
            notes.append(f"{sec} stops at -{row['reduction_exante']:.0f} % "
                         f"against -{r:.0%} ({why})")
    subtitle = (
        "Column margins are held, so each sector spends the same on inputs and only "
        "changes who it buys from. Row margins are free — a carbon-intensive supplier "
        "is allowed to sell less. Targets are measured at base emission factors, the "
        f"way a commitment is reported. Top {n_show} sectors by scope 3, "
        f"{coverage:.0f} % of the world total."
    )
    subtitle = _wrap(subtitle)
    if notes:
        subtitle += "<br>Shortfalls: " + "; ".join(notes[:3])

    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig = _style(
        fig, theme, "Cutting sectoral scope 3 by re-sourcing — KRAS",
        subtitle, height=(60 + 42 * n_show) if granularity == "ngfs" else 560,
    )
    fig.update_xaxes(title_text="Scope 3 at base emission factors (Mt CO2)",
                     row=1, col=1)
    fig.update_xaxes(title_text="Share of the input budget that changes supplier (%)",
                     row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- the carbon cost of the value chain
# ---------------------------------------------------------------------------

def fig_carbon_cost(theme: str = "light", granularity: str = "ngfs") -> go.Figure:
    bill, tau = cost_frame(granularity)
    k, t = _k(theme), THEME[theme]
    prices = carbon_price_table("World")

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10, column_widths=[0.62, 0.38],
    )

    # Dumbbell: the segment between the two dots IS the scope 2+3 exposure.
    xs, ys = [], []
    for sec in bill.index:
        xs += [bill.loc[sec, "share_scope1"], bill.loc[sec, "share_full"], None]
        ys += [sec, sec, None]
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                   line=dict(color=t["axis"], width=2)),
        row=1, col=1,
    )
    for col, name, color in (
        ("share_scope1", "Scope 1 only (an ETS on its own stack)",
         SCOPE_COLOR["scope1"][k]),
        ("share_full", "Full value chain (every tier prices carbon)",
         SCOPE_COLOR["scope3"][k]),
    ):
        fig.add_trace(
            go.Scatter(
                x=bill[col].to_numpy(dtype=float), y=list(bill.index),
                mode="markers", name=name,
                marker=dict(color=color, size=9,
                            line=dict(color=t["surface"], width=2)),
                hovertemplate=f"%{{y}}<br><b>{name}</b> %{{x:,.1f}} %<extra></extra>",
            ),
            row=1, col=1,
        )

    # The five scenarios are direct-labelled at their last point rather than
    # legended: the legend belongs to the left panel's two regimes, and mixing
    # seven entries from two different panels into one strip read as one set.
    years = [int(c) for c in prices.columns]
    finals = {}
    for scenario in prices.index:
        color = SCENARIO_COLOR[scenario][k]
        values = prices.loc[scenario].to_numpy(dtype=float)
        finals[scenario] = float(values[-1])
        fig.add_trace(
            go.Scatter(
                x=years, y=values, name=scenario, mode="lines+markers",
                showlegend=False,
                line=dict(color=color, width=2,
                          dash="dot" if scenario == "Baseline" else "solid"),
                marker=dict(symbol=SCENARIO_SYMBOL[scenario], size=6, color=color,
                            line=dict(width=1.5, color=t["surface"])),
                hovertemplate=f"<b>{scenario}</b> %{{y:,.1f}} US$/t<extra></extra>",
            ),
            row=1, col=2,
        )

    # Baseline and DAPS end 0.3 US$/t apart, so their end labels landed on top
    # of each other: push apart any pair closer than a label height.
    span = max(finals.values()) - min(finals.values())
    ranked = sorted(finals.items(), key=lambda kv: kv[1])
    shift, last = {}, -np.inf
    for scenario, value in ranked:
        pos = max(value, last + 0.055 * span)
        shift[scenario] = pos - value
        last = pos
    for scenario, value in finals.items():
        fig.add_annotation(
            x=years[-1], y=value + shift[scenario], text=f" {scenario}",
            xanchor="left", showarrow=False, xref="x2", yref="y2",
            font=dict(color=SCENARIO_COLOR[scenario][k], size=11, family=FONT))

    fig = _style(
        fig, theme, "What carbon costs a sector, and how much of it is upstream",
        _wrap("The gap between the two dots is the scope 2 + 3 exposure. The cost "
              "scale is logarithmic: it spans three orders of magnitude, and the "
              f"three fossil generators (up to {bill['share_full'].max():,.0f} % of "
              "output value) would otherwise flatten everything else. Cost is linear "
              "in the carbon price, so every scenario is this same picture rescaled "
              "— the right panel is what rescales it."),
        height=1000 if granularity == "ngfs" else 560,
    )
    # Plotly's default log minor ticks print a bare "2" for 20, which reads as 2.
    # tickvals take linear data values here, unlike shapes and annotations on the
    # same axis, which take log10.
    decades = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
    fig.update_xaxes(type="log", row=1, col=1, tickmode="array",
                     tickvals=decades, ticktext=[f"{v:g}" for v in decades],
                     title_text=f"Carbon cost at {tau:,.0f} US$/t "
                                f"({COST_SCENARIO} {COST_YEAR}), % of the value of "
                                "output — log scale")
    fig.update_xaxes(gridcolor=t["grid"], title_text="", row=1, col=2)
    fig.update_yaxes(gridcolor=t["grid"], showgrid=True, row=1, col=2,
                     title_text="GEM-E3 carbon price, world (US$2017/t CO2)")
    fig.update_layout(margin=dict(r=110))
    return fig


# ---------------------------------------------------------------------------
# Figure 4 -- the synthesis
# ---------------------------------------------------------------------------

def fig_cost_vs_target(theme: str = "light", granularity: str = "ngfs") -> go.Figure:
    bill, tau = cost_frame(granularity)
    frames = kras_frames(granularity)
    hardest = max(frames)
    k, t = _k(theme), THEME[theme]

    kr = frames[hardest]
    common = [s for s in bill.index if s in kr.index]
    x = bill.loc[common, "share_full"].to_numpy(dtype=float)
    y = kr.loc[common, "procurement_effort"].to_numpy(dtype=float)
    size = bill.loc[common, "x"].to_numpy(dtype=float)
    share3 = bill.loc[common, "scope3_share"].to_numpy(dtype=float)

    colorscale = [[i / (len(SEQUENTIAL[theme]) - 1), c]
                  for i, c in enumerate(SEQUENTIAL[theme])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="sector", showlegend=False,
        marker=dict(
            size=size, sizemode="area", sizeref=2.0 * max(size) / (46.0 ** 2),
            sizemin=6, color=share3, colorscale=colorscale, cmin=0, cmax=100,
            line=dict(color=t["surface"], width=2),
            colorbar=dict(title="Scope 3 share<br>of cradle-to-gate (%)",
                          thickness=12, len=0.55, x=1.02, y=0.5,
                          tickfont=dict(color=t["muted"], size=10),
                          title_font=dict(color=t["ink2"], size=11)),
        ),
        customdata=np.stack([share3, size / 1e6], axis=-1),
        text=common,
        hovertemplate=("<b>%{text}</b><br>carbon cost %{x:,.1f} % of output"
                       "<br>procurement shift %{y:,.1f} %"
                       "<br>scope 3 share %{customdata[0]:,.0f} %"
                       "<br>output %{customdata[1]:,.1f} tn EUR<extra></extra>"),
    ))

    # Direct-label only the extremes -- a label on every point is the thing to
    # avoid -- and alternate the offset so neighbours in y do not collide.
    logx = np.log10(x)
    score = (logx - logx.min()) / np.ptp(logx) + (y - y.min()) / np.ptp(y)
    picked = sorted(np.argsort(-score)[:8], key=lambda i: y[i])
    offsets = [-30, 18, -18, 30]          # cycle four slots, not two: the dense
    for rank, idx in enumerate(picked):   # cluster around y=70 needs the spread
        fig.add_annotation(x=logx[idx], y=y[idx], text=common[idx], showarrow=False,
                           yshift=offsets[rank % len(offsets)],
                           font=dict(color=t["ink2"], size=10, family=FONT))

    # The quadrant lines carry meaning, so they must not read as gridlines:
    # ink2 and a dash, plus a label. Three different conventions on one log axis,
    # all verified against the rendered image: `add_vline` takes the linear
    # value, `add_annotation` takes log10, and `tickvals` takes linear.
    med_x, med_y = float(np.median(x)), float(np.median(y))
    fig.add_vline(x=med_x, line=dict(color=t["ink2"], width=1, dash="dash"))
    fig.add_hline(y=med_y, line=dict(color=t["ink2"], width=1, dash="dash"))
    fig.add_annotation(x=np.log10(med_x), y=1.0, yref="y domain", text="median cost",
                       showarrow=False, xanchor="left", yanchor="bottom", xshift=4,
                       font=dict(color=t["muted"], size=10, family=FONT))
    fig.add_annotation(x=1.0, xref="x domain", y=med_y, text="median shift",
                       showarrow=False, xanchor="right", yanchor="bottom", yshift=4,
                       font=dict(color=t["muted"], size=10, family=FONT))

    fig = _style(
        fig, theme, "Exposed to carbon pricing, and hard to re-source",
        _wrap(f"Carbon cost at {tau:,.0f} US$/t ({COST_SCENARIO} {COST_YEAR}) against "
              f"the procurement shift a -{hardest:.0%} scope 3 target demands. Bubble "
              "area is output, dashed lines are the medians, the cost axis is "
              "logarithmic. Top right is the uncomfortable quadrant: expensive under "
              "a carbon price and slow to fix by switching supplier."),
        height=780, left=80,
    )
    fig.update_layout(margin=dict(r=170))
    decades = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300]
    fig.update_xaxes(type="log", tickmode="array",
                     tickvals=decades, ticktext=[f"{v:g}" for v in decades],
                     title_text="Full-chain carbon cost, % of the value of output "
                                "(log scale)")
    fig.update_yaxes(title_text=f"Input budget that changes supplier at "
                                f"-{hardest:.0%} (%)",
                     gridcolor=t["grid"], showgrid=True)
    return fig


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

FIGURES = {
    "scope_decomposition": fig_scope_decomposition,
    "kras_targets": fig_kras_targets,
    "carbon_cost": fig_carbon_cost,
    "cost_vs_target": fig_cost_vs_target,
}


def run_all(output_dir: Path | str = FIG_DIR, theme: str = "light",
            granularity: str = "ngfs", verbose: bool = True) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, builder in FIGURES.items():
        fig = builder(theme=theme, granularity=granularity)
        suffix = "" if theme == "light" else "_dark"
        path = out / f"{name}_{granularity}{suffix}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        written.append(path)
        if verbose:
            print(f"  {path}")
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Charts for the scope 3 analysis.")
    ap.add_argument("--all", action="store_true", help="Render every figure.")
    ap.add_argument("--figure", choices=sorted(FIGURES), default=None)
    ap.add_argument("--granularity", choices=["gics", "ngfs"], default="ngfs")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--output-dir", default=str(FIG_DIR))
    args = ap.parse_args()

    if args.all:
        print(f"Rendering {len(FIGURES)} figures "
              f"({args.granularity}, {args.theme}):")
        run_all(args.output_dir, args.theme, args.granularity)
    elif args.figure:
        fig = FIGURES[args.figure](theme=args.theme, granularity=args.granularity)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.theme == "light" else "_dark"
        path = out / f"{args.figure}_{args.granularity}{suffix}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"Saved: {path}")
    else:
        ap.print_help()
