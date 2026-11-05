"""
The base-year IO system the projection starts from.

One square system at a chosen `resolution`, carrying exactly the four objects
the projection needs:

    Z  (n, n)  intermediate flows                M.EUR
    x  (n,)    total output                      M.EUR
    y  (n,)    final demand, all categories       M.EUR
    F  (n,)    direct CO2 emissions              kg

Base year
---------
`DEFAULT_YEAR = 2021`, the paper's t0. This is deliberately *not* the 2019 used
across `optim/` and `scope3/`: the NGFS short-term growth rates are compounded
from 2021 (`shortterm/project_production.py:87`), so any other base year would
apply a 2021-anchored trajectory to the wrong starting point.

Final demand
------------
`y = Y.sum(axis=1)` sums **all 343 columns** of Y -- 49 regions x 7 categories
(households, NPISH, government, gross fixed capital formation, changes in
inventories, changes in valuables, exports fob). Not a subset: the row balance
x = Z.1 + y only closes if every final-demand column is counted, and
`row_balance_residual()` is the check that it does. Note that `y` can be
negative for a few sectors -- changes in inventories are a final-demand category
and go both ways.

Emissions
---------
The three fossil CO2 stressors of `scope3/aggregate.py:56-60`, reused verbatim.
CO2_bio is excluded on purpose: biogenic carbon is conventionally reported
outside the scopes.

Cache
-----
Z.txt is ~700 MB of text and takes minutes to parse. The aggregated system is
written to `Data/processed/v1_cache/{resolution}_{year}_{table_type}.npz` and
reused. Delete the file or pass `use_cache=False` to force a rebuild.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_pipeline import _safe_inv, load_extension, load_io_system  # noqa: E402
from scope3.aggregate import CO2_STRESSORS  # noqa: E402
from v1.resolution import (  # noqa: E402
    RESOLUTIONS,
    aggregate_matrix,
    aggregate_vector,
    is_identity,
    labels_and_order,
)

CACHE_DIR = REPO_ROOT / "Data" / "processed" / "v1_cache"

#: The paper's t0, and the base year the NGFS short-term rates compound from.
DEFAULT_YEAR = 2021


@dataclass
class IOSystem:
    """A square IO system with a CO2 extension, at one resolution."""

    labels: list[str]
    Z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    F: np.ndarray
    year: int
    table_type: str
    resolution: str

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def A(self) -> np.ndarray:
        """Technical coefficients Z x^-1; zero-output columns come out zero."""
        return self.Z * _safe_inv(self.x)[None, :]

    @property
    def s(self) -> np.ndarray:
        """Direct CO2 intensity, kg per M.EUR of output."""
        return self.F * _safe_inv(self.x)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"x": self.x, "y": self.y, "F": self.F, "s": self.s},
            index=pd.Index(self.labels, name="sector"),
        )

    def row_balance_residual(self) -> float:
        """
        Max relative gap in x = Z.1 + y.

        EXIOBASE balances by construction and aggregation preserves sums, so
        this is ~1e-15 at every resolution. It is the control that `y` summed
        the right columns and that the aggregator lost no flows.
        """
        gap = np.abs(self.Z.sum(axis=1) + self.y - self.x)
        keep = self.x > 0
        return float(np.max(gap[keep] / self.x[keep])) if keep.any() else 0.0

    def leontief_residual(self) -> float:
        """Max relative gap between (I-A)^-1 y and the observed x."""
        x_hat = np.linalg.solve(np.eye(self.n) - self.A, self.y)
        keep = self.x > 0
        return float(np.max(np.abs(x_hat[keep] - self.x[keep]) / self.x[keep]))


def _cache_path(year: int, resolution: str, table_type: str) -> Path:
    return CACHE_DIR / f"{resolution}_{year}_{table_type}.npz"


def build_io_system(
    year: int = DEFAULT_YEAR,
    resolution: str = "world",
    table_type: str = "pxp",
    use_cache: bool = True,
    verbose: bool = True,
) -> IOSystem:
    """Load EXIOBASE, aggregate to `resolution`, return the base-year system."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")

    path = _cache_path(year, resolution, table_type)
    if use_cache and path.exists():
        blob = np.load(path, allow_pickle=False)
        if verbose:
            print(f"  cache hit: {path.name}")
        return IOSystem(
            labels=[str(s) for s in blob["labels"]],
            Z=blob["Z"], x=blob["x"], y=blob["y"], F=blob["F"],
            year=year, table_type=table_type, resolution=resolution,
        )

    if verbose:
        print(f"  building {resolution} {year} {table_type} from EXIOBASE "
              f"(parsing Z.txt, this takes a few minutes)...")

    io_sys = load_io_system(year, table_type)
    Z, x_raw, Y = io_sys["Z"], io_sys["x"], io_sys["Y"]
    F_raw = load_extension(year, "air_emissions", table_type)["F"]

    labels, order = labels_and_order(Z.index, resolution, table_type)

    # CO2 first, while the columns are still the native EXIOBASE index.
    co2 = F_raw.reindex(index=CO2_STRESSORS).fillna(0.0).sum(axis=0)
    co2 = co2.reindex(Z.columns).fillna(0.0)

    if is_identity(labels, order):
        # "full": grouping would only copy a 9800 x 9800 block to no effect.
        Z_agg = Z.to_numpy(dtype=float)
        x_agg = x_raw.to_numpy(dtype=float)
        y_agg = Y.sum(axis=1).to_numpy(dtype=float)
        F_agg = co2.to_numpy(dtype=float)
    else:
        Z_agg = aggregate_matrix(Z, labels, order).to_numpy(dtype=float)
        x_agg = aggregate_vector(x_raw, labels, order).to_numpy(dtype=float)
        y_agg = aggregate_vector(Y, labels, order).to_numpy(dtype=float)
        F_agg = aggregate_vector(co2, labels, order).to_numpy(dtype=float)

    system = IOSystem(
        labels=order, Z=Z_agg, x=x_agg, y=y_agg, F=F_agg,
        year=year, table_type=table_type, resolution=resolution,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, labels=np.array(order, dtype=object).astype(str),
        Z=system.Z, x=system.x, y=system.y, F=system.F,
    )
    if verbose:
        print(f"  cached: {path}")
    return system


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the v1 base-year IO system.")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--resolution", choices=RESOLUTIONS, default="world")
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--rebuild", action="store_true", help="Ignore the npz cache.")
    args = ap.parse_args()

    sysm = build_io_system(args.year, args.resolution, args.table_type,
                           use_cache=not args.rebuild)

    print(f"\n  n           = {sysm.n}")
    print(f"  x total     = {sysm.x.sum():,.0f} M.EUR")
    print(f"  y total     = {sysm.y.sum():,.0f} M.EUR")
    print(f"  CO2 total   = {sysm.F.sum() / 1e9:,.1f} Mt")
    print(f"  zeros in Z  = {int((sysm.Z == 0).sum()):,} / {sysm.n ** 2:,} "
          f"({100 * (sysm.Z == 0).mean():.1f} %)")
    print(f"  row balance residual  x = Z.1 + y : {sysm.row_balance_residual():.2e}")
    print(f"  leontief residual     x = L y     : {sysm.leontief_residual():.2e}")
    print("\n  largest sectors by output:")
    print(sysm.frame().sort_values("x", ascending=False).head(10)
          .to_string(float_format=lambda v: f"{v:,.3g}"))
