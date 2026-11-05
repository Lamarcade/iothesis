"""
Project the EXIOBASE output vector x with NGFS short-term (GEM-E3) growth rates.

Takes EXIOBASE x at a base year (2021 by default), applies the GEM-E3 YoY
growth rates sector by sector and region by region through the mapping in
`st_exiobase_mapping.py`, and writes the resulting 2021-2030 trajectories to
`Data/production_evol/{scenario}/`.

Chain
-----
    x_EXIOBASE(base_year)                     9800 = 49 regions x 200 products
      x  GEM-E3 growth rates                  19 scenarios x 60 regions x 50 sectors
      -> aligned via exio_region_map() + exio_to_ngfs()
      -> compounded:  x_t = x_base * prod_{s<=t} (1 + g_s / 100)

Scenarios
---------
The file ships 19 raw run names, which reduce to the 4 NGFS short-term
scenarios plus a reference (see `SHORT_TERM_SCENARIOS`):

    Baseline   reference (current policies)
    DAPS       Disasters and Policy Stagnation   <- composite of 6 regional runs
    DIRE       Diverging Realities
    HWTP       Highway to Paris
    SWUC       Sudden Wake-up Call

`_run1` twins are ignored, and DAPS is rebuilt from its 6 regional runs -- see
the comment above `DAPS_RUNS` for the evidence behind both decisions.

Region resolution
-----------------
40 of the 49 EXIOBASE regions have a GEM-E3 country counterpart. The other 9
(AU, CH, NO, TW and the 5 RoW zones) fall back to a GEM-E3 aggregate, see
`EXIO_REGION_FALLBACK`. Coverage is therefore complete.

Rate clipping -- why it is on by default
----------------------------------------
GEM-E3 growth rates are explosive on a small number of cells: 2,194 values
exceed 100 %/yr and 557 exceed 1000 %/yr, up to +7.2e9 % on `Biofuels`. These
come from near-zero GEM-E3 bases in small regions, where the corresponding
EXIOBASE sector has a perfectly ordinary base -- so the rate is meaningless
once transferred. Compounded over 9 years they make x diverge. World-level
rates are sane (cumulative 2022-2030 factors between 0.22 and 4.8): the problem
is strictly regional. Rates are therefore clipped to `DEFAULT_CLIP` and every
clipped cell is written to `diagnostics_clipped.csv`.

Scope of the output
-------------------
This is a **mechanical** projection: it scales x without restoring accounting
balance. After the shock x != A.x + y. The output is a **target output
vector**, meant to feed a KRAS-type rebalancing (`optim/kras.py`) or a Leontief
solve on Y -- not a balanced IO table.

Two further caveats: all EXIOBASE sectors inside one GEM-E3 block receive the
same rate (the 28 products of `Non Market Services` move identically), and the
14 'overlap' sectors (pharma, batteries, EVs, H2, CCS) have no EXIOBASE target,
so their dynamics stay buried in the parent sector's.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_pipeline import load_x  # noqa: E402
from shortterm.st_exiobase_mapping import (  # noqa: E402
    GEME3_SHEET,
    PROCESSED_DIR,
    exio_region_map,
    exio_to_ngfs,
    sector_map,
)

OUTPUT_DIR = REPO_ROOT / "Data" / "production_evol"

#: Per-year growth band in %, applied before compounding. See module docstring.
DEFAULT_CLIP = (-99.0, 100.0)

DEFAULT_BASE_YEAR = 2021


# ---------------------------------------------------------------------------
# GEM-E3 data
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_geme3() -> pd.DataFrame:
    """
    GEM-E3 growth sheet, read once and cached.

    Without the cache this file would be re-read for every (scenario, year)
    pair -- 171 Excel reads for a full run.
    """
    return pd.read_excel(PROCESSED_DIR / "production_growth.xlsx",
                         sheet_name=GEME3_SHEET)


def _year_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).isdigit()]


def raw_scenario_names() -> list[str]:
    """The 19 raw GEM-E3 scenario names present in the file, sorted."""
    return sorted(_load_geme3()["Scenario"].unique())


# ---------------------------------------------------------------------------
# Scenario selection: 19 raw runs -> 4 NGFS short-term scenarios + Baseline
# ---------------------------------------------------------------------------
# The file ships 19 raw run names, which are NOT 19 scenarios:
#
#   * DAPS is split into 6 runs, each applying its acute physical shock to one
#     world region. The runs are near-orthogonal -- a run's GDP impact on
#     non-target regions is under 0.03 %, against 0.2 % to 3.1 % on its target
#     -- so the full DAPS scenario is the composite that gives every region its
#     own shock. That is what `DAPS_REGION_RUN` encodes.
#   * every scenario also has a `_run1` twin, systematically slightly milder
#     (World GDP 2030: +0.475 % for DIRE_run1, +0.173 % SWUC, +0.105 % HWTP,
#     ~+0.005 % for DAPS). Nothing in the file distinguishes them -- same
#     `Model`, same row count. The unsuffixed names carry the official NGFS
#     scenario names, so those are the ones used; `_run1` is ignored.
#
# Region -> DAPS run, derived from GDP 2030 relative to Baseline: each region is
# assigned the run that hurts it most. All 60 GEM-E3 regions are targeted by
# exactly one run, none is left over. `verify_daps_assignment()` recomputes the
# same thing from the production data and reports any divergence.

DAPS_RUNS = ["DAPS_AFR_R", "DAPS_ASIA", "DAPS_EUR",
             "DAPS_NAM", "DAPS_OCE", "DAPS_SAM"]

_DAPS_TARGETS: dict[str, list[str]] = {
    "DAPS_AFR_R": [
        "Africa", "South Africa - ZAF", "Middle East & Africa (R5)",
        "Rest of Energy Producing Countries", "Rest of World", "Other (R5)",
    ],
    "DAPS_ASIA": [
        "Asia", "Asia (R5)", "China - CHN", "India - IND", "Indonesia - IDN",
        "Japan - JPN", "Saudi Arabia - SAU", "South Korea - KOR",
        "Turkey - TUR", "World",
    ],
    "DAPS_EUR": [
        "Austria - AUT", "Belgium - BEL", "Bulgaria - BGR", "Croatia - HRV",
        "Cyprus - CYP", "Czechia - CZE", "Denmark - DNK", "Estonia - EST",
        "Finland - FIN", "France - FRA", "Germany - DEU", "Greece - GRC",
        "Hungary - HUN", "Ireland - IRL", "Italy - ITA", "Latvia - LVA",
        "Lithuania - LTU", "Luxembourg - LUX", "Malta - MLT",
        "Netherlands - NLD", "Poland - POL", "Portugal - PRT",
        "Romania - ROU", "Russia - RUS", "Slovakia - SVK", "Slovenia - SVN",
        "Spain - ESP", "Sweden - SWE", "United Kingdom - GBR",
        "EU27", "EU28", "Euro Area", "Rest of Europe",
        "Reforming Economies (R5)",
    ],
    "DAPS_NAM": [
        "Canada - CAN", "Mexico - MEX", "USA - USA", "North America",
        "OECD & EU (R5)",
    ],
    "DAPS_OCE": ["Oceania"],
    "DAPS_SAM": [
        "Argentina - ARG", "Brazil - BRA", "South America",
        "Latin America (R5)",
    ],
}

DAPS_REGION_RUN: dict[str, str] = {
    region: run for run, regions in _DAPS_TARGETS.items() for region in regions
}


@dataclass(frozen=True)
class Scenario:
    """
    One output scenario.

    runs : the raw GEM-E3 scenario name, or a {region: raw name} mapping for a
           composite scenario (DAPS).
    """
    name:  str
    label: str
    runs:  str | dict[str, str]

    @property
    def is_composite(self) -> bool:
        return isinstance(self.runs, dict)

    @property
    def source_runs(self) -> list[str]:
        return [self.runs] if isinstance(self.runs, str) else sorted(set(self.runs.values()))


#: The 4 NGFS short-term scenarios, plus Baseline as the reference they read against.
SHORT_TERM_SCENARIOS: list[Scenario] = [
    Scenario("Baseline", "Reference (current policies)", "Baseline"),
    Scenario("DAPS", "Disasters and Policy Stagnation", DAPS_REGION_RUN),
    Scenario("DIRE", "Diverging Realities", "DIRE"),
    Scenario("HWTP", "Highway to Paris", "HWTP"),
    Scenario("SWUC", "Sudden Wake-up Call", "SWUC"),
]


def get_scenario(name: str) -> Scenario:
    """
    Look up a curated scenario by name.

    Falls back to a pass-through Scenario for any raw GEM-E3 run name, so that
    a single run (e.g. `DAPS_EUR` or `DIRE_run1`) can still be projected.
    """
    for s in SHORT_TERM_SCENARIOS:
        if s.name == name:
            return s
    raw = raw_scenario_names()
    if name in raw:
        return Scenario(name, f"raw GEM-E3 run {name}", name)
    raise ValueError(
        f"Unknown scenario {name!r}. Curated: "
        f"{[s.name for s in SHORT_TERM_SCENARIOS]} ; raw runs: {raw}"
    )


def available_scenarios() -> list[str]:
    """Names of the curated output scenarios."""
    return [s.name for s in SHORT_TERM_SCENARIOS]


@lru_cache(maxsize=8)
def _load_raw_variable(variable: str) -> pd.DataFrame:
    """Any GEM-E3 variable from the raw NGFS_ST workbook, read once and cached."""
    from shortterm.load_st import load_st
    return load_st(sources=GEME3_SHEET, variable=variable)[GEME3_SHEET]


def load_geme3_variable(
    variable: str,
    scenario: str,
    sub_variable: str | None = None,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """
    A GEM-E3 variable for one curated scenario, with the DAPS composite applied.

    Same region-by-region composition as the production rates, so GDP and
    emissions stay consistent with the projected x under DAPS.

    Parameters
    ----------
    variable     : root variable ('GDP', 'Gross Emissions', 'Employment'...)
    scenario     : curated name ('Baseline', 'DAPS', ...) or a raw run
    sub_variable : part after the '|' ('GHG' for 'Gross Emissions|GHG');
                   None keeps the rows that have no sub-variable (e.g. GDP)
    regions      : restrict and order the output rows

    Returns
    -------
    DataFrame indexed by GEM-E3 region, columns = year strings.
    """
    spec = get_scenario(scenario)
    df = _load_raw_variable(variable)
    df = (df[df["Secteur"] == sub_variable] if sub_variable is not None
          else df[df["Secteur"].isna()])
    if df.empty:
        raise ValueError(
            f"No rows for variable={variable!r}, sub_variable={sub_variable!r}."
        )

    years = [c for c in df.columns if str(c).isdigit()]

    if not spec.is_composite:
        out = df[df["Scenario"] == spec.runs].set_index("Region")[years]
    else:
        frames = []
        for run in spec.source_runs:
            wanted = [r for r, v in spec.runs.items() if v == run]
            sub = df[(df["Scenario"] == run) & (df["Region"].isin(wanted))]
            frames.append(sub.set_index("Region")[years])
        out = pd.concat(frames).sort_index()

    if out.index.duplicated().any():
        raise ValueError(f"{scenario}/{variable}: duplicated regions after selection")
    if regions is not None:
        missing = [r for r in regions if r not in out.index]
        if missing:
            raise ValueError(f"{variable} missing for regions {missing}")
        out = out.loc[regions]
    return out.apply(pd.to_numeric, errors="coerce")


def verify_daps_assignment() -> pd.DataFrame:
    """
    Recompute the region -> DAPS run assignment from the production data.

    `DAPS_REGION_RUN` was derived from GDP; this recomputes it from the
    cumulative production deviation against Baseline and compares. Returns one
    row per region with both answers and an `agree` flag.

    Two aggregate regions are expected to disagree -- `Latin America (R5)` and
    `OECD & EU (R5)` -- because their GDP impact is tiny (-0.28 % and -0.77 %)
    and spread across runs. Neither is used by the EXIOBASE region mapping, so
    the projection is unaffected.
    """
    df = _load_geme3()
    years = [y for y in _year_columns(df) if y != "2021"]
    piv = {s: g.set_index(["Region", "Secteur"])[years].astype(float)
           for s, g in df.groupby("Scenario")}

    def cumulative(frame: pd.DataFrame) -> pd.Series:
        return (1 + frame / 100).prod(axis=1)

    base = cumulative(piv["Baseline"])
    deviation = pd.DataFrame({
        run: (cumulative(piv[run]) - base).groupby(level=0).sum()
        for run in DAPS_RUNS
    })
    out = pd.DataFrame({
        "declared": pd.Series(DAPS_REGION_RUN),
        "from_production": deviation.idxmin(axis=1),
    })
    out["agree"] = out["declared"] == out["from_production"]
    return out.sort_values(["agree", "declared"])


# ---------------------------------------------------------------------------
# Growth rates on the EXIOBASE index
# ---------------------------------------------------------------------------

def _raw_rates(spec: Scenario, years: list[str],
               df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    GEM-E3 growth rates for one scenario, indexed (Region, Secteur).

    For a composite scenario, each region is taken from the run that applies
    its own shock, so the result still holds exactly one row per
    (region, sector).
    """
    df = _load_geme3() if df is None else df

    if not spec.is_composite:
        sub = df[df["Scenario"] == spec.runs]
        return sub.set_index(["Region", "Secteur"])[years].astype(float)

    frames = []
    for run in spec.source_runs:
        regions = [r for r, v in spec.runs.items() if v == run]
        sub = df[(df["Scenario"] == run) & (df["Region"].isin(regions))]
        frames.append(sub.set_index(["Region", "Secteur"])[years].astype(float))
    out = pd.concat(frames).sort_index()

    expected = set(spec.runs)
    got = set(out.index.get_level_values(0))
    if missing := expected - got:
        raise ValueError(f"{spec.name}: no rows for regions {sorted(missing)}")
    if out.index.duplicated().any():
        raise ValueError(f"{spec.name}: a region was taken from two runs")
    return out


def growth_matrix(
    scenario: str,
    table_type: str = "pxp",
    base_year: int = DEFAULT_BASE_YEAR,
    clip: tuple[float, float] | None = DEFAULT_CLIP,
    exio_index: pd.MultiIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    GEM-E3 growth rates broadcast onto the EXIOBASE (region, sector) index.

    Parameters
    ----------
    scenario   : GEM-E3 scenario name (see `available_scenarios()`)
    base_year  : only years strictly after it are kept -- the YoY rate of year
                 Y takes x from Y-1 to Y
    clip       : (low, high) band in %/yr, or None to disable clipping
    exio_index : EXIOBASE MultiIndex to align on; loaded from `base_year` if None

    Returns
    -------
    rates   : DataFrame indexed like `exio_index`, columns = years, values in %
              (after clipping)
    clipped : DataFrame of clipped cells at GEM-E3 granularity, columns
              [scenario, ngfs_region, ngfs_sector, year, raw_growth,
               clipped_growth]. Empty when nothing was clipped.
    """
    spec = get_scenario(scenario)
    df = _load_geme3()

    # 2021 is entirely NaN (no base year in the source), hence the notna guard.
    probe = df[df["Scenario"] == spec.source_runs[0]]
    years = [y for y in _year_columns(df)
             if int(y) > base_year and probe[y].notna().any()]
    if not years:
        raise ValueError(
            f"No growth year after base_year={base_year}. "
            f"Available: {[y for y in _year_columns(df) if probe[y].notna().any()]}"
        )

    raw = _raw_rates(spec, years, df)

    if clip is not None:
        low, high = clip
        mask = (raw < low) | (raw > high)
        rates_ngfs = raw.clip(lower=low, upper=high)
        # Explicit .dropna(): whether .stack() drops NaN is pandas-version
        # dependent, and relying on it silently listed every cell as clipped.
        clipped = (raw.where(mask).stack().dropna()
                      .rename("raw_growth").reset_index())
        clipped.columns = ["ngfs_region", "ngfs_sector", "year", "raw_growth"]
        clipped["clipped_growth"] = clipped["raw_growth"].clip(low, high)
        # The 14 'overlap' sectors have no EXIOBASE target, so clipping them
        # changes nothing in x. Flag it so the file is not misread.
        with_target = {s for s, e in sector_map(table_type).items() if e}
        clipped["affects_exiobase"] = clipped["ngfs_sector"].isin(with_target)
        clipped.insert(0, "scenario", scenario)
    else:
        rates_ngfs = raw
        clipped = pd.DataFrame(columns=["scenario", "ngfs_region", "ngfs_sector",
                                        "year", "raw_growth", "clipped_growth",
                                        "affects_exiobase"])

    if exio_index is None:
        exio_index = load_x(base_year, table_type).index

    # Vectorised alignment: build the GEM-E3 key for each EXIOBASE row, then
    # reindex in one shot (no per-row loop over 9800 entries).
    regions = exio_index.get_level_values(0)
    sectors = exio_index.get_level_values(1)
    reg_map = exio_region_map(list(dict.fromkeys(regions)))
    sec_map = exio_to_ngfs(table_type)

    keys = pd.MultiIndex.from_arrays([
        regions.map(reg_map),
        sectors.map(sec_map),
    ])
    rates = rates_ngfs.reindex(keys)
    rates.index = exio_index
    return rates, clipped


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_x(
    scenario: str,
    base_year: int = DEFAULT_BASE_YEAR,
    table_type: str = "pxp",
    clip: tuple[float, float] | None = DEFAULT_CLIP,
    x_base: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project EXIOBASE x forward with one scenario's growth rates.

    Returns
    -------
    projected : DataFrame (region, sector) x [base_year, ...] in M.EUR.
                The `base_year` column is x_base untouched.
    clipped   : the clipping diagnostics from `growth_matrix`
    """
    if x_base is None:
        x_base = load_x(base_year, table_type)

    rates, clipped = growth_matrix(
        scenario, table_type=table_type, base_year=base_year,
        clip=clip, exio_index=x_base.index,
    )

    factors = (1.0 + rates / 100.0).cumprod(axis=1)
    projected = factors.mul(x_base, axis=0)
    projected.insert(0, str(base_year), x_base.values)
    projected.index.names = ["region", "sector"]
    return projected, clipped


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_scenario(
    scenario: str,
    base_year: int = DEFAULT_BASE_YEAR,
    table_type: str = "pxp",
    clip: tuple[float, float] | None = DEFAULT_CLIP,
    out_dir: Path | str = OUTPUT_DIR,
    x_base: pd.Series | None = None,
    write_growth: bool = True,
) -> dict:
    """
    Write one scenario's projection to `{out_dir}/{scenario}/`.

    Files
    -----
    x_{table_type}_{base}-{end}.csv       projected output, (region, sector) x years
    growth_{table_type}_{first}-{end}.csv rates actually applied (post-clipping)
    diagnostics_clipped.csv               clipped cells, only if any

    Returns a stats dict, one row of the run summary.
    """
    projected, clipped = project_x(scenario, base_year, table_type, clip, x_base)
    years = list(projected.columns)

    scen_dir = Path(out_dir) / scenario
    scen_dir.mkdir(parents=True, exist_ok=True)

    x_path = scen_dir / f"x_{table_type}_{years[0]}-{years[-1]}.csv"
    projected.to_csv(x_path, encoding="utf-8")

    if write_growth:
        rates, _ = growth_matrix(scenario, table_type=table_type,
                                 base_year=base_year, clip=clip,
                                 exio_index=projected.index)
        rates.index.names = ["region", "sector"]
        rates.to_csv(
            scen_dir / f"growth_{table_type}_{years[1]}-{years[-1]}.csv",
            encoding="utf-8",
        )

    diag_path = scen_dir / "diagnostics_clipped.csv"
    if not clipped.empty:
        clipped.to_csv(diag_path, index=False, encoding="utf-8")
    elif diag_path.exists():
        diag_path.unlink()   # stale file from a previous, wider clip band

    spec = get_scenario(scenario)
    ratio = projected[years[-1]] / projected[years[0]].replace(0.0, pd.NA)
    stats = {
        "scenario":          scenario,
        "label":             spec.label,
        "source_runs":       " + ".join(spec.source_runs),
        "table_type":        table_type,
        "base_year":         years[0],
        "end_year":          years[-1],
        "n_clipped_cells":   len(clipped),
        "n_clipped_effective": (int(clipped["affects_exiobase"].sum())
                               if not clipped.empty else 0),
        "n_clipped_sectors": clipped["ngfs_sector"].nunique() if not clipped.empty else 0,
        "max_ratio_end_base": float(ratio.max(skipna=True)),
        "min_ratio_end_base": float(ratio.min(skipna=True)),
        "n_nan":             int(projected.isna().sum().sum()),
        "n_negative":        int((projected < 0).sum().sum()),
    }
    stats.update({f"x_total_{y}": float(projected[y].sum()) for y in years})
    return stats


def run_all(
    scenarios: list[str] | None = None,
    base_year: int = DEFAULT_BASE_YEAR,
    table_type: str = "pxp",
    clip: tuple[float, float] | None = DEFAULT_CLIP,
    out_dir: Path | str = OUTPUT_DIR,
    write_growth: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Project every scenario and write `{out_dir}/_summary.csv`.

    x for the base year is loaded once and reused across scenarios.
    """
    scenarios = scenarios or available_scenarios()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading EXIOBASE x {base_year} {table_type}...")
    x_base = load_x(base_year, table_type)
    if verbose:
        print(f"  {len(x_base)} sectors, total {x_base.sum():.4e} M.EUR\n")

    rows = []
    for i, scenario in enumerate(scenarios, start=1):
        stats = export_scenario(scenario, base_year, table_type, clip,
                                out_dir, x_base, write_growth)
        rows.append(stats)
        if verbose:
            total_end = stats[f"x_total_{stats['end_year']}"]
            total_base = stats[f"x_total_{stats['base_year']}"]
            print(f"[{i:2d}/{len(scenarios)}] {scenario:20s} "
                  f"clipped={stats['n_clipped_cells']:5d}  "
                  f"x_{stats['end_year']}/x_{stats['base_year']}="
                  f"{total_end / total_base:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "_summary.csv", index=False, encoding="utf-8")
    if verbose:
        print(f"\nSaved: {out_dir / '_summary.csv'}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Project EXIOBASE x with NGFS short-term (GEM-E3) growth rates."
    )
    parser.add_argument("--scenario", default=None,
                        help="One scenario: a curated name (Baseline, DAPS, "
                             "DIRE, HWTP, SWUC) or any raw GEM-E3 run.")
    parser.add_argument("--all", action="store_true",
                        help="Project the 4 NGFS short-term scenarios + Baseline.")
    parser.add_argument("--list", action="store_true",
                        help="List curated scenarios and raw runs, then exit.")
    parser.add_argument("--verify-daps", action="store_true",
                        help="Recheck the region -> DAPS run assignment.")
    parser.add_argument("--base-year", type=int, default=DEFAULT_BASE_YEAR)
    parser.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    parser.add_argument("--clip", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        default=list(DEFAULT_CLIP),
                        help="Growth band in %%/yr (default -99 100).")
    parser.add_argument("--no-clip", action="store_true",
                        help="Disable clipping (x will diverge on some cells).")
    parser.add_argument("--no-growth-csv", action="store_true",
                        help="Skip the applied-rates CSV.")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    clip = None if args.no_clip else tuple(args.clip)

    if args.list:
        print("Curated output scenarios (the 4 NGFS short-term + reference):")
        for s in SHORT_TERM_SCENARIOS:
            src = (f"composite of {len(s.source_runs)} runs"
                   if s.is_composite else s.runs)
            print(f"  {s.name:10s} {s.label:34s} <- {src}")
        print(f"\nRaw GEM-E3 runs in the file ({len(raw_scenario_names())}), "
              f"usable via --scenario:")
        for s in raw_scenario_names():
            print(f"  - {s}")

    elif args.verify_daps:
        chk = verify_daps_assignment()
        n_bad = int((~chk["agree"]).sum())
        print(f"Regions: {len(chk)} | agree: {int(chk['agree'].sum())} | "
              f"differ: {n_bad}")
        if n_bad:
            print("\nDivergences (expected on R5 aggregates, unused by EXIOBASE):")
            print(chk[~chk["agree"]].to_string())

    elif args.all:
        run_all(base_year=args.base_year, table_type=args.table_type,
                clip=clip, out_dir=args.out_dir,
                write_growth=not args.no_growth_csv)

    elif args.scenario:
        summary = run_all([args.scenario], base_year=args.base_year,
                          table_type=args.table_type, clip=clip,
                          out_dir=args.out_dir,
                          write_growth=not args.no_growth_csv)
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(summary.T.to_string())

    else:
        parser.print_help()
