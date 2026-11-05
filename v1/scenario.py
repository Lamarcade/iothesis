"""
The scenario side: x^{T,k}, the projected output vector, at one resolution.

This is the only place the climate scenario enters the projection. Everything
else -- margins, emission targets, the closure -- is derived from it or from the
base-year system.

Where x^{T,k} comes from
------------------------
`shortterm/project_production.py` compounds the NGFS short-term (GEM-E3) growth
rates onto EXIOBASE x at 2021, sector by sector and region by region, and writes
`Data/production_evol/{scenario}/x_{table_type}_2021-2030.csv` -- the full
(region, sector) grid in M.EUR, one column per year. That module is the
authority; this one only reads its output and aggregates it.

If the CSV is absent (the directory is gitignored, ~76 MB) the projection is
recomputed in-process via `project_x`. Same numbers, a few minutes slower.

Five scenarios: Baseline, DAPS, DIRE, HWTP, SWUC (`SHORT_TERM_SCENARIOS`).

What this vector is, and is not
-------------------------------
It is a **mechanical** rescaling of x: after the shock, x != A.x + y. Restoring
that balance is exactly the job of the KRAS step, so an unbalanced x is the
expected input here, not a defect. See the "Scope of the output" section of
`shortterm/project_production.py`.

Alignment
---------
The labels are recomputed from the CSV's own index and checked against the
system's, rather than assumed. A silent misalignment here would rescale the
wrong sectors and produce a plausible, wrong answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shortterm.project_production import (  # noqa: E402
    DEFAULT_BASE_YEAR,
    OUTPUT_DIR,
    available_scenarios,
    project_x,
)
from v1.resolution import aggregate_vector, is_identity, labels_and_order  # noqa: E402
from v1.system import IOSystem  # noqa: E402


def _projection_frame(
    scenario: str,
    table_type: str = "pxp",
    base_year: int = DEFAULT_BASE_YEAR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    The (region, sector) x years projection for one scenario, from disk or fresh.

    The reader mirrors `shortterm/plot_production.py:165`; it is reimplemented
    here so that v1 does not import a plotly module to read a CSV.
    """
    folder = Path(OUTPUT_DIR) / scenario
    matches = sorted(folder.glob(f"x_{table_type}_*.csv")) if folder.is_dir() else []

    if matches:
        if verbose:
            print(f"  reading {matches[-1].relative_to(REPO_ROOT)}")
        frame = pd.read_csv(matches[-1], index_col=[0, 1])
        frame.index.names = ["region", "sector"]
        if str(base_year) not in frame.columns:
            raise ValueError(
                f"{matches[-1].name} has no column {base_year!r} "
                f"(columns {list(frame.columns)[:3]}...). It was written for a "
                f"different base year; regenerate with "
                f"`python shortterm/project_production.py --all`."
            )
        return frame

    if verbose:
        print(f"  no cached projection for {scenario!r}; recomputing "
              f"(this parses the GEM-E3 workbook)...")
    projected, _ = project_x(scenario, base_year=base_year, table_type=table_type)
    return projected


def scenario_x(
    scenario: str,
    year: int | str,
    system: IOSystem,
    base_year: int = DEFAULT_BASE_YEAR,
    verbose: bool = True,
) -> np.ndarray:
    """
    x^{T,k} for `scenario` at horizon `year`, aligned on `system.labels`.

    Returns a plain (n,) ndarray in M.EUR, same order as `system.labels`.
    """
    if system.year != base_year:
        raise ValueError(
            f"the system is built at {system.year} but the projection compounds "
            f"from {base_year}; build the system with year={base_year}."
        )

    frame = _projection_frame(scenario, system.table_type, base_year, verbose)

    col = str(year)
    if col not in frame.columns:
        raise ValueError(f"year {col!r} not in projection "
                         f"({frame.columns[0]}..{frame.columns[-1]})")

    labels, order = labels_and_order(frame.index, system.resolution,
                                     system.table_type)
    if order != system.labels:
        raise ValueError(
            "the projection index does not aggregate to the system's labels "
            f"({len(order)} vs {system.n}). The system cache and the projection "
            "were built from different EXIOBASE tables."
        )

    series = frame[col].astype(float)
    if is_identity(labels, order):
        return series.to_numpy(dtype=float)
    return aggregate_vector(series, labels, order).to_numpy(dtype=float)


def growth_factor(x_target: np.ndarray, system: IOSystem) -> np.ndarray:
    """x^{T,k} / x, with 0 where the base year had no output."""
    base = system.x
    return np.divide(x_target, base, out=np.zeros_like(x_target), where=base > 0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from v1.system import DEFAULT_YEAR, build_io_system
    from v1.resolution import RESOLUTIONS

    ap = argparse.ArgumentParser(description="Inspect x^{T,k} at one resolution.")
    ap.add_argument("--scenario", default="SWUC", choices=available_scenarios())
    ap.add_argument("--year", default="2030")
    ap.add_argument("--resolution", choices=RESOLUTIONS, default="world")
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--base-year", type=int, default=DEFAULT_YEAR)
    args = ap.parse_args()

    sysm = build_io_system(args.base_year, args.resolution, args.table_type)
    xt = scenario_x(args.scenario, args.year, sysm, args.base_year)
    g = growth_factor(xt, sysm)

    print(f"\n  {args.scenario} {args.base_year} -> {args.year}, n = {sysm.n}")
    print(f"  world output {sysm.x.sum():,.0f} -> {xt.sum():,.0f} M.EUR "
          f"({100 * (xt.sum() / sysm.x.sum() - 1):+.1f} %)")

    tbl = pd.DataFrame({"x_base": sysm.x, "x_target": xt, "factor": g},
                       index=pd.Index(sysm.labels, name="sector"))
    live = tbl[tbl["x_base"] > 0].sort_values("factor")
    print("\n  most contracting sectors:")
    print(live.head(8).to_string(float_format=lambda v: f"{v:,.3g}"))
    print("\n  most expanding sectors:")
    print(live.tail(8).to_string(float_format=lambda v: f"{v:,.3g}"))
