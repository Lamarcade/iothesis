"""
Charts for the NGFS short-term production projections.

Reads the scenario trajectories written by `project_production.py` into
`Data/production_evol/` and renders interactive HTML into `Figs/shortterm/`.

Everything is indexed to the base year = 100
----------------------------------------------
Production (EXIOBASE, M.EUR current), GDP (GEM-E3, bn US$2017 constant) and GHG
emissions (Mt CO2e/yr) are three different units on three different bases. Two
of them on one plot with two y-scales would invent a correlation that is not in
the data -- and the levels are not comparable anyway (see
`Data/docs/MAPPING_ST_HYPOTHESES.md` section 11.2: only rates transfer, not
levels). So every series is divided by its base-year value: one axis, three
comparable shapes.

Aggregation axes
----------------
* 6 macro-regions -- reused from the DAPS regional runs, so the grouping that
  drives the scenario is the grouping the charts show.
* 7 sector themes -- the `SectorMap.theme` field, covering all 200 pxp products.

Colour
------
Baseline is the reference, so it takes the de-emphasis gray; the 4 NGFS
scenarios take 4 categorical hues in a fixed order. A scenario keeps its hue in
every panel of every figure, and hue never depends on rank or on which series
are visible.

The order was computed, not chosen: of the 8 documented slots, the 4-hue subsets
were enumerated and scored with `palette_check.py` (a port of the dataviz
skill's validator). No 4-hue subset clears the *all-pairs* gate in both modes --
0 of 70 -- so the palette is validated on the *adjacent* pairlist that a
multi-line chart actually needs, and the mitigation the skill mandates at 4
series is applied: every scenario also carries a distinct marker symbol, so
identity never rests on hue alone.

    order        blue -> green -> violet -> orange
    worst adjacent CVD dE 26.0 (target 8.0), normal-vision 27.0 (floor 15.0)
    separation from the Baseline gray: dE 9.8
    all 4 hues >= 3:1 against both surfaces, in both modes

Usage
-----
    python shortterm/plot_production.py --all
    python shortterm/plot_production.py --all --theme dark
    python shortterm/plot_production.py --figure regions
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

from shortterm.project_production import (  # noqa: E402
    DAPS_REGION_RUN,
    OUTPUT_DIR,
    SHORT_TERM_SCENARIOS,
    load_geme3_variable,
)
from shortterm.st_exiobase_mapping import MAPPING, exio_region_map  # noqa: E402

FIG_DIR = REPO_ROOT / "Figs" / "shortterm"

SCENARIOS = [s.name for s in SHORT_TERM_SCENARIOS]
SCENARIO_LABEL = {s.name: s.label for s in SHORT_TERM_SCENARIOS}

# ---------------------------------------------------------------------------
# Palette (see module docstring for how it was validated)
# ---------------------------------------------------------------------------

SCENARIO_COLOR = {
    #            light      dark
    "Baseline": ("#898781", "#898781"),   # de-emphasis ink: the reference
    "DAPS":     ("#2a78d6", "#3987e5"),   # blue
    "DIRE":     ("#008300", "#008300"),   # green
    "HWTP":     ("#4a3aa7", "#9085e9"),   # violet
    "SWUC":     ("#eb6834", "#d95926"),   # orange
}
# Secondary encoding: identity never rests on hue alone.
SCENARIO_SYMBOL = {
    "Baseline": "circle-open", "DAPS": "circle", "DIRE": "square",
    "HWTP": "diamond", "SWUC": "triangle-up",
}

THEME = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f9f9f7", "ink": "#0b0b0b",
        "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "mid": "#f0efec",
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#0d0d0d", "ink": "#ffffff",
        "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "mid": "#383835",
    },
}

# Diverging ramp: the documented blue <-> red pair (warm/cool, so the poles read
# as opposite) with a neutral gray midpoint. The poles stop at the categorical
# steps rather than going darker, so the value printed in each cell stays
# legible on top of them.
DIVERGING = [
    (0.00, "#2a78d6"), (0.25, "#9ec5f4"), (0.45, "#dfeafa"),
    (0.50, "#f0efec"),
    (0.55, "#fae0df"), (0.75, "#f3b0af"), (1.00, "#e34948"),
]
#: Colour is capped here; cells beyond it keep their true printed value.
#: 135 of the 144 sector cells sit under 40 % (median 2.2 %), so an uncapped
#: symmetric scale would spend 80 % of the negative arm on nothing.
DIVERGENCE_CAP = 60.0

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# ---------------------------------------------------------------------------
# Aggregation axes
# ---------------------------------------------------------------------------

MACRO_LABEL = {
    "DAPS_EUR": "Europe", "DAPS_ASIA": "Asia", "DAPS_NAM": "North America",
    "DAPS_SAM": "South America", "DAPS_AFR_R": "Africa & Middle East",
    "DAPS_OCE": "Oceania",
}
MACRO_ORDER = ["Europe", "Asia", "North America", "South America",
               "Africa & Middle East", "Oceania"]

# GEM-E3 native aggregate to read GDP / emissions from, per macro-region.
# CAVEAT: the EXIOBASE "Europe" group covers EU27 + GB, CH, NO, RU and RoW
# Europe, while GEM-E3 "EU27" is EU27 strict. Indexing to 100 makes the two
# comparable in shape, not in perimeter.
MACRO_GEME3 = {
    "Europe": "EU27", "Asia": "Asia", "North America": "North America",
    "South America": "South America", "Africa & Middle East": "Africa",
    "Oceania": "Oceania",
}

THEME_ORDER = ["Energy supply", "Power generation", "Heavy industry",
               "Manufacturing", "Transport", "Primary", "Services"]


def exio_region_to_macro() -> dict[str, str]:
    """EXIOBASE region code -> macro-region label."""
    return {exio: MACRO_LABEL[DAPS_REGION_RUN[ngfs]]
            for exio, ngfs in exio_region_map().items()}


def exio_sector_to_theme(table_type: str = "pxp") -> dict[str, str]:
    """EXIOBASE sector name -> GEM-E3 theme."""
    return {s: m.theme for m in MAPPING for s in m.exio(table_type)}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_projected_x(scenario: str, table_type: str = "pxp") -> pd.DataFrame:
    """Projected x for one scenario, indexed (region, sector), columns = years."""
    folder = OUTPUT_DIR / scenario
    matches = sorted(folder.glob(f"x_{table_type}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No projection for {scenario!r} in {folder}. "
            f"Run: python shortterm/project_production.py --all"
        )
    return pd.read_csv(matches[-1], index_col=[0, 1])


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    """Divide every row by its first column, x100."""
    return frame.div(frame.iloc[:, 0], axis=0) * 100.0


def x_world(table_type: str = "pxp") -> pd.DataFrame:
    """Total projected x per scenario (rows) and year (columns)."""
    return pd.DataFrame(
        {s: load_projected_x(s, table_type).sum() for s in SCENARIOS}
    ).T


def x_by_macro_region(scenario: str, table_type: str = "pxp") -> pd.DataFrame:
    """Projected x aggregated to the 6 macro-regions."""
    df = load_projected_x(scenario, table_type)
    macro = df.index.get_level_values(0).map(exio_region_to_macro())
    return df.groupby(macro).sum().reindex(MACRO_ORDER)


def x_by_theme(scenario: str, table_type: str = "pxp") -> pd.DataFrame:
    """Projected world x aggregated to the 7 sector themes."""
    df = load_projected_x(scenario, table_type)
    theme = df.index.get_level_values(1).map(exio_sector_to_theme(table_type))
    return df.groupby(theme).sum().reindex(THEME_ORDER)


def x_by_ngfs_sector(scenario: str, table_type: str = "pxp") -> pd.DataFrame:
    """Projected world x aggregated to GEM-E3 sectors (those with a target)."""
    from shortterm.st_exiobase_mapping import exio_to_ngfs
    df = load_projected_x(scenario, table_type)
    sec = df.index.get_level_values(1).map(exio_to_ngfs(table_type))
    return df.groupby(sec).sum()


def geme3_by_macro(variable: str, scenario: str,
                   sub_variable: str | None = None) -> pd.DataFrame:
    """A GEM-E3 variable on the 6 macro-regions, rows relabelled to macro names."""
    regions = [MACRO_GEME3[m] for m in MACRO_ORDER]
    out = load_geme3_variable(variable, scenario, sub_variable, regions=regions)
    out.index = MACRO_ORDER
    return out


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _style(fig: go.Figure, theme: str, title: str, subtitle: str = "",
           height: int = 700) -> go.Figure:
    t = THEME[theme]
    fig.update_layout(
        template=None,
        title=dict(
            text=f"<b>{title}</b>" + (f"<br><span style='font-size:13px;"
                                      f"color:{t['ink2']}'>{subtitle}</span>"
                                      if subtitle else ""),
            font=dict(size=19, color=t["ink"], family=FONT), x=0.01, xanchor="left",
        ),
        paper_bgcolor=t["plane"], plot_bgcolor=t["surface"],
        font=dict(family=FONT, color=t["ink2"], size=12),
        height=height, margin=dict(l=60, r=30, t=110, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left",
                    x=0.01, font=dict(color=t["ink2"], size=12),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], linewidth=1,
                     ticks="outside", tickcolor=t["axis"],
                     tickfont=dict(color=t["muted"], size=11))
    fig.update_yaxes(gridcolor=t["grid"], gridwidth=1, zeroline=False,
                     linecolor=t["axis"], linewidth=1,
                     tickfont=dict(color=t["muted"], size=11))
    return fig


def _add_lines(fig: go.Figure, frame: pd.DataFrame, theme: str,
               row: int | None = None, col: int | None = None,
               showlegend: bool = False, unit: str = "index") -> None:
    """One line per scenario. frame: rows = scenarios, cols = years."""
    k = 0 if theme == "light" else 1
    years = [int(c) for c in frame.columns]
    for scenario in SCENARIOS:
        if scenario not in frame.index:
            continue
        color = SCENARIO_COLOR[scenario][k]
        is_ref = scenario == "Baseline"
        fig.add_trace(
            go.Scatter(
                x=years, y=frame.loc[scenario].to_numpy(dtype=float),
                name=scenario, legendgroup=scenario, showlegend=showlegend,
                mode="lines+markers",
                line=dict(color=color, width=2,
                          dash="dot" if is_ref else "solid"),
                marker=dict(symbol=SCENARIO_SYMBOL[scenario], size=6,
                            color=color,
                            line=dict(width=1.5, color=THEME[theme]["surface"])),
                hovertemplate=f"<b>{scenario}</b> %{{y:.1f}} {unit}<extra></extra>",
            ),
            row=row, col=col,
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_overview(theme: str = "light", table_type: str = "pxp") -> go.Figure:
    """World level: production, GDP, GHG and emission intensity, all indexed."""
    prod = _indexed(x_world(table_type))
    gdp = _indexed(pd.DataFrame(
        {s: load_geme3_variable("GDP", s, regions=["World"]).loc["World"]
         for s in SCENARIOS}).T)
    ghg = _indexed(pd.DataFrame(
        {s: load_geme3_variable("Gross Emissions", s, "GHG",
                                regions=["World"]).loc["World"]
         for s in SCENARIOS}).T)
    common = [c for c in prod.columns if c in ghg.columns]
    intensity = _indexed(ghg[common] / gdp[common])

    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.09,
        subplot_titles=("Production (EXIOBASE x)", "GDP (GEM-E3)",
                        "GHG emissions (GEM-E3)", "GHG per unit of GDP"),
    )
    for i, frame in enumerate([prod, gdp, ghg, intensity]):
        r, c = divmod(i, 2)
        _add_lines(fig, frame, theme, row=r + 1, col=c + 1, showlegend=(i == 0))

    fig = _style(
        fig, theme, "NGFS short-term scenarios — world",
        f"Everything indexed to {prod.columns[0]} = 100. "
        "Levels are not comparable across the three measures; shapes are.",
        height=760,
    )
    for ann in fig.layout.annotations[:4]:
        ann.font = dict(size=13, color=THEME[theme]["ink"], family=FONT)
    fig.add_hline(y=100, line=dict(color=THEME[theme]["axis"], width=1))
    return fig


def _grid_by_macro(measure: dict[str, pd.DataFrame], theme: str,
                   title: str, subtitle: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3, vertical_spacing=0.15, horizontal_spacing=0.07,
        subplot_titles=MACRO_ORDER,
    )
    for i, macro in enumerate(MACRO_ORDER):
        r, c = divmod(i, 3)
        frame = pd.DataFrame({s: measure[s].loc[macro] for s in SCENARIOS}).T
        _add_lines(fig, _indexed(frame), theme, row=r + 1, col=c + 1,
                   showlegend=(i == 0))
    fig = _style(fig, theme, title, subtitle, height=680)
    for ann in fig.layout.annotations[:6]:
        ann.font = dict(size=13, color=THEME[theme]["ink"], family=FONT)
    fig.add_hline(y=100, line=dict(color=THEME[theme]["axis"], width=1))
    return fig


def fig_regions_production(theme: str = "light", table_type: str = "pxp") -> go.Figure:
    data = {s: x_by_macro_region(s, table_type) for s in SCENARIOS}
    return _grid_by_macro(
        data, theme, "Production by macro-region — EXIOBASE x",
        "Indexed to 2021 = 100. Macro-regions follow the DAPS regional runs.",
    )


def fig_regions_gdp(theme: str = "light") -> go.Figure:
    data = {s: geme3_by_macro("GDP", s) for s in SCENARIOS}
    return _grid_by_macro(
        data, theme, "GDP by macro-region — GEM-E3",
        "Indexed to 2021 = 100. Europe is read from GEM-E3 'EU27', "
        "a narrower perimeter than the EXIOBASE Europe group.",
    )


def fig_regions_emissions(theme: str = "light") -> go.Figure:
    data = {s: geme3_by_macro("Gross Emissions", s, "GHG") for s in SCENARIOS}
    return _grid_by_macro(
        data, theme, "GHG emissions by macro-region — GEM-E3",
        "Indexed to 2021 = 100. Only the GHG total is published per region; "
        "the split by gas exists at world level only.",
    )


def fig_sectors(theme: str = "light", table_type: str = "pxp") -> go.Figure:
    """World production by sector theme, indexed."""
    data = {s: x_by_theme(s, table_type) for s in SCENARIOS}
    fig = make_subplots(
        rows=3, cols=3, vertical_spacing=0.11, horizontal_spacing=0.07,
        subplot_titles=THEME_ORDER + ["", ""],
    )
    for i, th in enumerate(THEME_ORDER):
        r, c = divmod(i, 3)
        frame = pd.DataFrame({s: data[s].loc[th] for s in SCENARIOS}).T
        _add_lines(fig, _indexed(frame), theme, row=r + 1, col=c + 1,
                   showlegend=(i == 0))
    fig = _style(
        fig, theme, "World production by sector theme — EXIOBASE x",
        "Indexed to 2021 = 100. Every EXIOBASE sector in a theme carries the "
        "growth rate of its GEM-E3 block.", height=860,
    )
    for ann in fig.layout.annotations[:7]:
        ann.font = dict(size=13, color=THEME[theme]["ink"], family=FONT)
    fig.add_hline(y=100, line=dict(color=THEME[theme]["axis"], width=1))
    return fig


def _divergence(table_type: str = "pxp") -> tuple[pd.DataFrame, pd.DataFrame]:
    """% deviation of x in the final year vs Baseline, by sector and by region."""
    scen = [s for s in SCENARIOS if s != "Baseline"]
    base_sec = x_by_ngfs_sector("Baseline", table_type)
    base_reg = x_by_macro_region("Baseline", table_type)
    last = base_sec.columns[-1]
    by_sector = pd.DataFrame({
        s: 100 * (x_by_ngfs_sector(s, table_type)[last] / base_sec[last] - 1)
        for s in scen})
    by_region = pd.DataFrame({
        s: 100 * (x_by_macro_region(s, table_type)[last] / base_reg[last] - 1)
        for s in scen})
    by_sector = by_sector.reindex(
        by_sector.abs().max(axis=1).sort_values().index)
    return by_sector, by_region.reindex(MACRO_ORDER[::-1])


def fig_divergence(theme: str = "light", table_type: str = "pxp") -> go.Figure:
    by_sector, by_region = _divergence(table_type)
    t = THEME[theme]
    beyond = int((np.abs(by_sector.to_numpy()) > DIVERGENCE_CAP).sum())

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.18,
        subplot_titles=("By GEM-E3 sector", "By macro-region"),
    )
    for i, frame in enumerate([by_sector, by_region]):
        trace = go.Heatmap(
            z=frame.to_numpy(), x=list(frame.columns), y=list(frame.index),
            colorscale=DIVERGING, zmid=0,
            zmin=-DIVERGENCE_CAP, zmax=DIVERGENCE_CAP,
            xgap=2, ygap=2, showscale=(i == 0),
            # Every value is printed, so nothing is encoded by colour alone
            # and the capped cells stay exactly readable.
            text=frame.to_numpy(), texttemplate="%{text:+.1f}",
            textfont=dict(size=9 if i == 0 else 11, color=t["ink"], family=FONT),
            hovertemplate="%{y} · %{x}<br>%{z:+.2f} % vs Baseline<extra></extra>",
        )
        if i == 0:
            trace.colorbar = dict(
                title="% vs Baseline", thickness=12, len=0.8, x=1.02,
                tickfont=dict(color=t["muted"], size=10),
                title_font=dict(color=t["ink2"], size=11),
            )
        fig.add_trace(trace, row=1, col=i + 1)

    fig = _style(
        fig, theme, "Divergence from Baseline in the final year",
        "Baseline is the reference: 0 everywhere, so it is not shown. Sectors "
        f"sorted by amplitude. Colour capped at &#177;{DIVERGENCE_CAP:.0f} % "
        f"({beyond} cells beyond it keep their true printed value).",
        height=1000,
    )
    fig.update_layout(margin=dict(l=230, r=90, t=120, b=60), hovermode="closest")
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=13, color=THEME[theme]["ink"], family=FONT)
    fig.update_yaxes(tickfont=dict(size=10, color=t["ink2"]), showgrid=False,
                     ticks="", automargin=True)
    fig.update_xaxes(showgrid=False, side="top",
                     tickfont=dict(size=12, color=t["ink2"]))
    return fig


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

FIGURES = {
    "overview":  fig_overview,
    "regions_production": fig_regions_production,
    "regions_gdp": fig_regions_gdp,
    "regions_emissions": fig_regions_emissions,
    "sectors":   fig_sectors,
    "divergence": fig_divergence,
}


def run_all(output_dir: Path | str = FIG_DIR, theme: str = "light",
            table_type: str = "pxp", verbose: bool = True) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, builder in FIGURES.items():
        kwargs = {"theme": theme}
        if name not in ("regions_gdp", "regions_emissions"):
            kwargs["table_type"] = table_type
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

    ap = argparse.ArgumentParser(
        description="Charts for the NGFS short-term production projections.")
    ap.add_argument("--all", action="store_true", help="Render every figure.")
    ap.add_argument("--figure", choices=sorted(FIGURES), default=None)
    ap.add_argument("--theme", choices=["light", "dark"], default="light",
                    help="Dark uses the validated dark steps, not a flip.")
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--output-dir", default=str(FIG_DIR))
    args = ap.parse_args()

    if args.all:
        print(f"Rendering {len(FIGURES)} figures ({args.theme}):")
        run_all(args.output_dir, args.theme, args.table_type)
    elif args.figure:
        kwargs = {"theme": args.theme}
        if args.figure not in ("regions_gdp", "regions_emissions"):
            kwargs["table_type"] = args.table_type
        fig = FIGURES[args.figure](**kwargs)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.theme == "light" else "_dark"
        path = out / f"{args.figure}{suffix}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"Saved: {path}")
    else:
        ap.print_help()
