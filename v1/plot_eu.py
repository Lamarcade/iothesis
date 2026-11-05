"""
The European sectoral view: internal flows, one import channel, one export channel.

Runs the `eu_macro` projection once per NGFS short-term scenario, each with its
own decarbonisation target on the European columns, and renders what happens
between European sectors as the ambition rises.

The 37 x 37 view
----------------
`eu_macro` keeps 5 separate non-European blocs, so a European sector has five
import rows and five export columns. Condensing them to one each gives a
legible square view -- rows are suppliers, columns are buyers:

    M[0:36, 0:36] = Z_EE                     intra-European flows
    M[0:36, 36]   = Z_ER . 1   "Exports"     each EU sector's sales to all RoW
    M[36, 0:36]   = 1' . Z_RE  "Imports"     each EU sector's purchases from RoW
    M[36, 36]     = NaN                      RoW->RoW, outside this view

This is a **view, not a system**. The projection still solves on the full closed
216 x 216 matrix, so nothing is sliced off and the accounting stays intact --
slicing Europe out of Z instead would break `x = Z.1 + y` by 78 % (see
`Data/docs/V1_HYPOTHESES.md` section 5.2). The condensation happens at display
time only, and `eu_view` asserts that its two aggregates reproduce the border
flows measured independently at `region_ngfs`.

The ambition ladder
-------------------
One beta per scenario, weaker on the more pessimistic ones. The ordering is not
a judgement call: it follows the GEM-E3 2030 carbon price, verified against
`scope3.carbon_cost.carbon_price_table("World")` --

    SWUC 163.1 > HWTP 151.6 > DIRE 115.3 > DAPS 54.0 > Baseline 53.7  US$2017/t

and agrees with the emissions the scenarios themselves carry (SWUC E' = 29 874
Mt against Baseline's 34 000 at `eu_macro`). Note DAPS and Baseline are only
0.3 US$/t apart, so the gap between their betas is wider than the price
difference warrants -- the ladder is arbitrary in level, only its order is
grounded.

**Sigma is identical across scenarios** (the defaults, 0.05 and 5.0). Only beta
varies, so every difference between panels is attributable to ambition alone.

Each run targets Europe through `target_mask`, never through `beta = 1`
elsewhere: beta is measured against the base year, so beta = 1 pins an
untargeted column to its 2021 emissions despite 2030 output -- a hard constraint
that does not converge. See `v1/constraints.py::region_mask`.

Colour and chrome
-----------------
`_style`, `_wrap`, `_hbar`, `_k` and the flow ramp are imported from
`v1/plot_v1.py` rather than copied. `scope3/plot_scope3.py` redefines its own
because it lives in another package; these two are siblings inside `v1/`, so a
third divergent copy would be the worse outcome.

Usage
-----
    python v1/plot_eu.py --all
    python v1/plot_eu.py --all --theme dark
    python v1/plot_eu.py --figure eu_flows --scenario HWTP
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

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
    SCENARIOS,
    THEME,
)
from v1.constraints import region_mask  # noqa: E402
from v1.plot_v1 import (  # noqa: E402
    FLOW_SEQUENTIAL,
    RESULT_COLOR,
    _hbar,
    _k,
    _style,
    _wrap,
)
from v1.project import load_projection, project, save_projection  # noqa: E402
from v1.resolution import split_label  # noqa: E402
from v1.scenario import scenario_x  # noqa: E402
from v1.system import DEFAULT_YEAR, build_io_system  # noqa: E402

FIG_DIR = REPO_ROOT / "Figs" / "v1" / "eu"

#: The bloc kept in detail, and the tag its runs are cached under.
BLOC = "Europe"
TAG = "Europe"
RESOLUTION = "eu_macro"
DEFAULT_HORIZON = "2030"

#: Ambition per scenario: weaker where the scenario is more pessimistic. Order
#: follows the GEM-E3 2030 carbon price (see module docstring); the levels are
#: arbitrary, as everything parametric in v1 still is.
SCENARIO_BETA = {
    "Baseline": 0.95,   # current policies
    "DAPS":     0.90,   # disasters and policy stagnation
    "DIRE":     0.80,   # diverging realities
    "HWTP":     0.70,   # orderly Paris-aligned
    "SWUC":     0.65,   # sudden wake-up call, highest 2030 carbon price
}

#: Base-year border flows, M.EUR, measured independently at `region_ngfs`.
#: `eu_view` reproduces them or raises -- a transposed or mis-masked view would
#: otherwise render perfectly plausibly.
BORDER_CHECK = {"imports": 1_422_109.0, "exports": 2_191_740.0}

#: Colour cap on the relative-change panel, in %. Same convention as
#: `plot_production.py:117-120`: cells beyond it keep their true hover value.
CHANGE_CAP = 50.0

IMPORT_LABEL = "Imports (RoW)"
EXPORT_LABEL = "Exports (RoW)"

MT = 1e9   # kg -> Mt


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

def eu_view(labels: list[str], M: np.ndarray,
            bloc: str = BLOC) -> tuple[np.ndarray, list[str]]:
    """
    Condense a (n, n) flow matrix into the (37, 37) European view.

    Returns `(view, sectors)` where `sectors` are the 36 European sector names
    with the `"Europe | "` prefix stripped. Row 36 is the aggregate import
    channel, column 36 the aggregate export channel, and cell (36, 36) NaN.
    """
    regions = np.array([split_label(lab)[0] for lab in labels])
    eu = np.flatnonzero(regions == bloc)
    row = np.flatnonzero(regions != bloc)
    if eu.size == 0:
        raise ValueError(f"no {bloc!r} labels; is the resolution region-aware?")

    k = eu.size
    view = np.full((k + 1, k + 1), np.nan)
    view[:k, :k] = M[np.ix_(eu, eu)]
    view[:k, k] = M[np.ix_(eu, row)].sum(axis=1)     # exports: a buyer column
    view[k, :k] = M[np.ix_(row, eu)].sum(axis=0)     # imports: a supplier row

    sectors = [split_label(labels[i])[1] for i in eu]
    return view, sectors


def check_border_flows(system) -> tuple[float, float]:
    """
    Base-year import/export aggregates, checked against the known values.

    Guards the orientation: a transposed view would swap these two and still
    look like a perfectly reasonable heatmap.
    """
    view, sectors = eu_view(system.labels, system.Z)
    k = len(sectors)
    imports = float(np.nansum(view[k, :k]))
    exports = float(np.nansum(view[:k, k]))
    for name, got in (("imports", imports), ("exports", exports)):
        want = BORDER_CHECK[name]
        if abs(got - want) / want > 1e-6:
            raise AssertionError(
                f"base-year {name} = {got:,.0f} M.EUR, expected {want:,.0f}. "
                "The bloc mask or the row/column orientation is wrong."
            )
    return imports, exports


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class EuRun(NamedTuple):
    scenario: str
    beta: float
    system: object
    table: pd.DataFrame
    Z_new: np.ndarray
    Z0: np.ndarray
    x_target: np.ndarray


def eu_run(scenario: str, year: str = DEFAULT_HORIZON,
           table_type: str = "pxp", base_year: int = DEFAULT_YEAR,
           rebuild: bool = False, verbose: bool = True) -> EuRun:
    """
    One scenario solved at `eu_macro` with its own beta on Europe. Cache-first.
    """
    beta = SCENARIO_BETA[scenario]
    system = build_io_system(base_year, RESOLUTION, table_type,
                             use_cache=not rebuild, verbose=False)
    x_target = scenario_x(scenario, year, system, base_year, verbose=False)
    Z0 = system.A * x_target[None, :]

    if not rebuild:
        cached = load_projection(scenario, year, RESOLUTION, table_type, beta,
                                 base_year, TAG)
        if cached is not None:
            table, Z_new, _ = cached
            return EuRun(scenario, beta, system, table, Z_new, Z0, x_target)

    if verbose:
        print(f"  solving {scenario} beta={beta:g} (first run)...")
    proj, _ = project(system, x_target, beta=beta, scenario=scenario, year=year,
                      target_mask=region_mask(system.labels, BLOC),
                      verbose=False)
    save_projection(proj, beta, TAG)
    return EuRun(scenario, beta, system, proj.table, proj.Z_new, Z0, x_target)


def all_runs(year: str = DEFAULT_HORIZON, table_type: str = "pxp",
             base_year: int = DEFAULT_YEAR, rebuild: bool = False,
             verbose: bool = True) -> dict[str, EuRun]:
    return {s: eu_run(s, year, table_type, base_year, rebuild, verbose)
            for s in SCENARIOS}


def _eu_rows(run: EuRun) -> pd.DataFrame:
    """The 36 European rows of a run's table, indexed by bare sector name."""
    regions = np.array([split_label(lab)[0] for lab in run.system.labels])
    sub = run.table.iloc[regions == BLOC].copy()
    sub.index = [split_label(lab)[1] for lab in np.array(run.system.labels)[
        regions == BLOC]]
    return sub


def _ladder_note() -> str:
    steps = ", ".join(f"{s} {SCENARIO_BETA[s]:g}" for s in SCENARIOS)
    return (f"beta by scenario ({steps}), ordered by the GEM-E3 2030 carbon "
            "price. Sigma is identical across scenarios, so what differs "
            "between panels is ambition alone.")


def _log_disp(M: np.ndarray) -> np.ndarray:
    """log10 of the positive cells, NaN elsewhere (a zero stays a zero)."""
    return np.where(M > 0, np.log10(np.where(M > 0, M, 1.0)), np.nan)


# ---------------------------------------------------------------------------
# Figure 1 -- the flow matrix, one file per scenario
# ---------------------------------------------------------------------------

def fig_eu_flows(theme: str = "light", scenario: str = "SWUC",
                 year: str = DEFAULT_HORIZON, table_type: str = "pxp",
                 rebuild: bool = False) -> go.Figure:
    run = eu_run(scenario, year, table_type, rebuild=rebuild)
    t = THEME[theme]

    new, sectors = eu_view(run.system.labels, run.Z_new)
    base, _ = eu_view(run.system.labels, run.Z0)
    y_labels = sectors + [IMPORT_LABEL]
    x_labels = sectors + [EXPORT_LABEL]

    disp = _log_disp(new)
    finite = new[np.isfinite(new) & (new > 0)]
    vmin, vmax = float(np.log10(finite.min())), float(np.log10(finite.max()))

    change = np.full_like(new, np.nan)
    ok = np.isfinite(base) & (base > 0)
    change[ok] = 100.0 * (new[ok] / base[ok] - 1.0)
    beyond = int(np.nansum(np.abs(change) > CHANGE_CAP))

    colorscale = [[i / (len(FLOW_SEQUENTIAL[theme]) - 1), c]
                  for i, c in enumerate(FLOW_SEQUENTIAL[theme])]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        horizontal_spacing=0.05,
                        subplot_titles=("Flows after rebalancing",
                                        "Change vs unchanged technology"))
    fig.add_trace(
        go.Heatmap(z=disp, x=x_labels, y=y_labels, colorscale=colorscale,
                   zmin=vmin, zmax=vmax, xgap=1, ygap=1,
                   customdata=new,
                   hovertemplate="%{y} → %{x}<br>%{customdata:,.0f} M.EUR"
                                 "<extra></extra>",
                   colorbar=dict(title="M.EUR<br>(log10)", thickness=12,
                                 len=0.42, x=0.455, y=0.79,
                                 tickfont=dict(color=t["muted"], size=10),
                                 title_font=dict(color=t["ink2"], size=11))),
        row=1, col=1)
    fig.add_trace(
        go.Heatmap(z=change, x=x_labels, y=y_labels, colorscale=DIVERGING,
                   zmid=0, zmin=-CHANGE_CAP, zmax=CHANGE_CAP, xgap=1, ygap=1,
                   hovertemplate="%{y} → %{x}<br>%{z:+,.1f} % vs Z0"
                                 "<extra></extra>",
                   colorbar=dict(title="% vs Z0", thickness=12, len=0.42,
                                 x=1.02, y=0.79,
                                 tickfont=dict(color=t["muted"], size=10),
                                 title_font=dict(color=t["ink2"], size=11))),
        row=1, col=2)

    fig = _style(
        fig, theme,
        f"European sectoral flows — {scenario} {year}, β = {run.beta:g}",
        _wrap(f"Rows are suppliers, columns buyers. The rest of the world is "
              f"condensed into one inbound row and one outbound column; the "
              f"projection itself still solves on the closed 216×216 system. "
              f"Z0 is the same output level at unchanged 2021 technology, so "
              f"the right panel isolates what the decarbonisation target moved. "
              f"Colour capped at ±{CHANGE_CAP:.0f} % ({beyond} cells beyond it "
              f"keep their true hover value); blank cells were zero in the base "
              f"year and GRAS cannot open them."),
        height=140 + 26 * len(y_labels), left=210)
    fig.update_xaxes(side="top", tickangle=45, tickfont=dict(size=9),
                     showgrid=False)
    fig.update_yaxes(tickfont=dict(size=9), autorange="reversed")
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=12, color=t["ink"], family=FONT)
    fig.update_layout(margin=dict(t=190, r=110))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- the ambition ladder, all five scenarios
# ---------------------------------------------------------------------------

def fig_eu_scenario_ladder(theme: str = "light", year: str = DEFAULT_HORIZON,
                           table_type: str = "pxp",
                           rebuild: bool = False) -> go.Figure:
    runs = all_runs(year, table_type, rebuild=rebuild)
    k, t = _k(theme), THEME[theme]

    betas, dom, imp, cf_tot, channels = [], [], [], [], {}
    for s in SCENARIOS:
        run = runs[s]
        eu = _eu_rows(run)
        betas.append(run.beta)
        dom.append(eu["b_achieved_domestic"].sum() / MT)
        imp.append(eu["b_achieved_imported"].sum() / MT)
        # The unchanged-technology counterfactual at the same output, which is
        # what the target is attributable against -- not the base year.
        cf_tot.append((eu["b_cf_domestic"].sum()
                       + eu["b_cf_imported"].sum()) / MT)

        new, sectors = eu_view(run.system.labels, run.Z_new)
        old, _ = eu_view(run.system.labels, run.Z0)
        n = len(sectors)
        channels[s] = [
            100.0 * (np.nansum(new[:n, :n]) / np.nansum(old[:n, :n]) - 1.0),
            100.0 * (np.nansum(new[n, :n]) / np.nansum(old[n, :n]) - 1.0),
            100.0 * (np.nansum(new[:n, n]) / np.nansum(old[:n, n]) - 1.0),
        ]

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.09,
        subplot_titles=("Ambition applied", "Carbon embodied in EU purchases",
                        "Flow channels vs unchanged technology"))

    colors = [SCENARIO_COLOR[s][k] for s in SCENARIOS]
    fig.add_trace(
        go.Bar(x=SCENARIOS, y=[100 * (1 - b) for b in betas],
               marker=dict(color=colors), showlegend=False,
               text=[f"−{100 * (1 - b):.0f} %" for b in betas],
               textposition="outside",
               textfont=dict(color=t["ink2"], size=11, family=FONT),
               hovertemplate="<b>%{x}</b> cut of %{y:.0f} %<extra></extra>"),
        row=1, col=1)

    for name, values, color in (
        ("domestic suppliers", dom, RESULT_COLOR["achieved"][k]),
        ("imported", imp, "#e34948"),
    ):
        fig.add_trace(
            go.Bar(x=SCENARIOS, y=values, name=name,
                   marker=dict(color=color,
                               line=dict(color=t["surface"], width=1)),
                   hovertemplate=f"<b>%{{x}}</b> {name} %{{y:,.0f}} Mt"
                                 "<extra></extra>"),
            row=1, col=2)
    fig.add_trace(
        go.Scatter(x=SCENARIOS, y=cf_tot, mode="markers",
                   name="unchanged technology",
                   marker=dict(symbol="line-ew", size=22,
                               line=dict(color=t["ink"], width=2)),
                   hovertemplate="<b>%{x}</b> unchanged technology "
                                 "%{y:,.0f} Mt<extra></extra>"),
        row=1, col=2)

    for i, (label, color) in enumerate((
        ("intra-EU", RESULT_COLOR["achieved"][k]),
        ("imports", "#e34948"),
        ("exports", RESULT_COLOR["base"][k]),
    )):
        fig.add_trace(
            go.Bar(x=SCENARIOS, y=[channels[s][i] for s in SCENARIOS],
                   name=label, legendgroup=label,
                   marker=dict(color=color,
                               line=dict(color=t["surface"], width=1)),
                   hovertemplate=f"<b>%{{x}}</b> {label} %{{y:+,.1f}} %"
                                 "<extra></extra>"),
            row=1, col=3)

    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_traces(row=1, col=3, selector=dict(type="bar"))
    fig = _style(
        fig, theme, f"What ambition buys, and what it displaces — Europe {year}",
        _wrap(_ladder_note() + " The middle panel counts only the 36 European "
              "columns; the tick is the unchanged-technology counterfactual at "
              "the same output, which is what the target is attributable "
              "against — measuring against the base year instead would fold in "
              "the scenario's own output shift."),
        height=620, left=80)
    fig.update_yaxes(title_text="Cut in embodied emissions (%)", row=1, col=1)
    fig.update_yaxes(title_text="Mt CO2", row=1, col=2)
    fig.update_yaxes(title_text="% vs Z0", row=1, col=3, zeroline=True,
                     zerolinecolor=t["axis"])
    fig.update_xaxes(showgrid=False)
    for ann in fig.layout.annotations[:3]:
        ann.font = dict(size=12, color=t["ink"], family=FONT)
    fig.update_layout(margin=dict(t=170))
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- per sector, across the ladder
# ---------------------------------------------------------------------------

def fig_eu_sector_scenarios(theme: str = "light", year: str = DEFAULT_HORIZON,
                            table_type: str = "pxp",
                            rebuild: bool = False) -> go.Figure:
    runs = all_runs(year, table_type, rebuild=rebuild)
    t = THEME[theme]

    effort = pd.DataFrame({s: _eu_rows(runs[s])["procurement_effort"]
                           for s in SCENARIOS})
    leak = pd.DataFrame({s: _eu_rows(runs[s])["leakage_pp"]
                         for s in SCENARIOS})
    order = effort.mean(axis=1).sort_values().index
    effort, leak = effort.loc[order], leak.loc[order]

    seq = [[i / (len(FLOW_SEQUENTIAL[theme]) - 1), c]
           for i, c in enumerate(FLOW_SEQUENTIAL[theme])]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
        subplot_titles=("Input budget that changes supplier (%)",
                        "Embodied carbon shifted abroad (pp)"))
    fig.add_trace(
        go.Heatmap(z=effort.to_numpy(), x=SCENARIOS, y=list(effort.index),
                   colorscale=seq, xgap=2, ygap=2,
                   hovertemplate="%{y} · %{x}<br>%{z:,.1f} % of the input "
                                 "budget moves<extra></extra>",
                   colorbar=dict(title="%", thickness=12, len=0.8, x=0.455,
                                 tickfont=dict(color=t["muted"], size=10),
                                 title_font=dict(color=t["ink2"], size=11))),
        row=1, col=1)
    cap = float(np.nanmax(np.abs(leak.to_numpy()))) or 1.0
    fig.add_trace(
        go.Heatmap(z=leak.to_numpy(), x=SCENARIOS, y=list(leak.index),
                   colorscale=DIVERGING, zmid=0, zmin=-cap, zmax=cap,
                   xgap=2, ygap=2,
                   hovertemplate="%{y} · %{x}<br>%{z:+,.1f} pp onto foreign "
                                 "suppliers<extra></extra>",
                   colorbar=dict(title="pp", thickness=12, len=0.8, x=1.02,
                                 tickfont=dict(color=t["muted"], size=10),
                                 title_font=dict(color=t["ink2"], size=11))),
        row=1, col=2)

    fig = _style(
        fig, theme, f"Which European sectors bear the adjustment — {year}",
        _wrap(_ladder_note() + " Sectors sorted by mean procurement effort. The "
              "right panel is the leakage: percentage points of a sector's "
              "embodied carbon that moved from a European to a foreign "
              "supplier."),
        height=140 + 24 * len(order), left=210)
    fig.update_xaxes(side="top", showgrid=False,
                     tickfont=dict(size=11, color=t["ink2"]))
    fig.update_yaxes(tickfont=dict(size=9))
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=12, color=t["ink"], family=FONT)
    fig.update_layout(margin=dict(t=190, r=110))
    return fig


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

FIGURES = {
    "eu_flows": fig_eu_flows,                       # per scenario
    "eu_scenario_ladder": fig_eu_scenario_ladder,
    "eu_sector_scenarios": fig_eu_sector_scenarios,
}
_PER_SCENARIO = {"eu_flows"}


def run_all(output_dir: Path | str = FIG_DIR, theme: str = "light",
            year: str = DEFAULT_HORIZON, table_type: str = "pxp",
            rebuild: bool = False, verbose: bool = True) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme == "light" else "_dark"
    written = []
    for name, builder in FIGURES.items():
        targets = SCENARIOS if name in _PER_SCENARIO else [None]
        for scenario in targets:
            kwargs = dict(theme=theme, year=year, table_type=table_type,
                          rebuild=rebuild)
            stem = name
            if scenario is not None:
                kwargs["scenario"] = scenario
                stem = f"{name}_{scenario}"
            fig = builder(**kwargs)
            path = out / f"{stem}{suffix}.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
            if verbose:
                print(f"  {path}")
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="European sectoral flows across the NGFS scenarios.")
    ap.add_argument("--all", action="store_true", help="Render every figure.")
    ap.add_argument("--figure", choices=sorted(FIGURES), default=None)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--scenario", choices=SCENARIOS, default="SWUC",
                    help="Only used by eu_flows.")
    ap.add_argument("--year", default=DEFAULT_HORIZON)
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--output-dir", default=str(FIG_DIR))
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore the system and projection caches.")
    args = ap.parse_args()

    if args.all:
        sysm = build_io_system(DEFAULT_YEAR, RESOLUTION, args.table_type,
                               verbose=False)
        imp, exp = check_border_flows(sysm)
        print(f"  border check: base-year imports {imp:,.0f} / exports "
              f"{exp:,.0f} M.EUR -- as measured at region_ngfs")
        print(f"Rendering {len(FIGURES) + len(SCENARIOS) - 1} figures "
              f"({args.theme}):")
        run_all(args.output_dir, args.theme, args.year, args.table_type,
                args.rebuild)
    elif args.figure:
        kwargs = dict(theme=args.theme, year=args.year,
                      table_type=args.table_type, rebuild=args.rebuild)
        stem = args.figure
        if args.figure in _PER_SCENARIO:
            kwargs["scenario"] = args.scenario
            stem = f"{args.figure}_{args.scenario}"
        fig = FIGURES[args.figure](**kwargs)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.theme == "light" else "_dark"
        path = out / f"{stem}{suffix}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"Saved: {path}")
    else:
        ap.print_help()
