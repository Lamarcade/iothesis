"""
Scope 1 / 2 / 3 decomposition of a world-aggregated IO system.

The identity everything rests on
-------------------------------
With s the direct intensity, A the technical coefficients and m = s L the
cradle-to-gate multiplier, m = s + m A, hence for every sector j

    m_j x_j   =   F_j   +   sum_i m_i Z_ij
    ---------     -----     ---------------
    cradle        scope 1   everything upstream (scopes 2 and 3)

This is exact, not an approximation, so the three scopes always sum back to the
cradle-to-gate total. `check_identity()` asserts it; if it ever fails, the
aggregation lost or duplicated flows.

Scope 2 in an IO table
----------------------
The GHG Protocol counts the emissions of *generating* the electricity a sector
buys. In EXIOBASE the chain is  Coal fired -> Power Supply (grid/trade) -> buyer,
so almost nobody buys from a generator directly: taking only the direct
emissions of the row you purchase from would put nearly all generation into
scope 3 and leave scope 2 at ~0.

So scope 2 is measured with the *power-restricted* multiplier

    m^P = s^P L        where  s^P_i = s_i if i is a power sector, else 0
    scope2_j = sum_{p in power} m^P_p Z_pj

which is "emissions arising inside the power block, reached through j's direct
purchases of electricity". Electricity embodied in the steel that j buys does
not come through a power row, so it correctly stays in scope 3.

What this is not
----------------
* **Upstream only** (GHG Protocol categories 1-8). EXIOBASE has no use phase,
  so downstream scope 3 -- category 11, use of sold products, which dominates
  for oil and gas -- is not computable here. A Ghosh (supply-driven) model would
  give an allocation proxy; that is a different object and is out of scope.
* Scope 2 is location-based, allocated by monetary purchases of electricity.
  There is no market-based / contractual variant in an IO table.
* With regions summed away, a sector's purchases from itself (the diagonal
  Z_jj) land in its own scope 3: an aggregated sector partly counts itself.
  The coarser the grouping the stronger the effect, which is one reason both
  the 11-sector and the 36-sector views are produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scope3.aggregate import WorldSystem, build_world_system, power_sectors  # noqa: E402

#: kg -> Mt
KG_PER_MT = 1e9


def power_mask(system: WorldSystem) -> np.ndarray:
    """Boolean mask of the sectors whose emissions are somebody's scope 2."""
    names = set(power_sectors(system.granularity, system.table_type))
    return np.array([s in names for s in system.sectors], dtype=bool)


def power_multiplier(system: WorldSystem, m: np.ndarray | None = None,
                     A: np.ndarray | None = None) -> np.ndarray:
    """
    m^P = s^P L -- emissions arising *inside the power block* per unit output.

    Same Leontief inverse as `m`, but seeded only with the power sectors'
    direct intensities.
    """
    A = system.A if A is None else A
    s_p = np.where(power_mask(system), system.s, 0.0)
    return np.linalg.solve(np.eye(system.n) - A.T, s_p)


def scope_table(system: WorldSystem, A: np.ndarray | None = None,
                x: np.ndarray | None = None) -> pd.DataFrame:
    """
    Scope 1 / 2 / 3-upstream per sector, in kg CO2.

    Parameters
    ----------
    A, x : optional overrides, so the same decomposition can be applied to a
           KRAS-rebalanced structure and to a reallocated output vector.
           Direct intensities `s` always come from the base system: KRAS
           changes who buys from whom, not how dirty a sector's own furnace is.

    Returns
    -------
    DataFrame indexed by sector with columns
        x, s, m, scope1, scope2, scope3_up, cradle, scope3_share
    """
    A = system.A if A is None else A
    x = system.x if x is None else x
    n = system.n

    Z = A * x[None, :]
    m = np.linalg.solve(np.eye(n) - A.T, system.s)
    m_p = power_multiplier(system, A=A)

    scope1 = system.s * x
    scope2 = (m_p[:, None] * Z)[power_mask(system), :].sum(axis=0)
    cradle = m * x
    scope3 = cradle - scope1 - scope2

    return pd.DataFrame(
        {
            "x": x, "s": system.s, "m": m,
            "scope1": scope1, "scope2": scope2, "scope3_up": scope3,
            "cradle": cradle,
            "scope3_share": np.where(cradle > 0, scope3 / np.where(cradle > 0, cradle, 1.0), 0.0),
        },
        index=pd.Index(system.sectors, name="sector"),
    )


def scope3_by_supplier(system: WorldSystem, A: np.ndarray | None = None,
                       x: np.ndarray | None = None) -> pd.DataFrame:
    """
    Where each sector's scope 3 comes from: (supplier i) x (buyer j), kg CO2.

    Column j sums to `scope3_up[j]`. This is the matrix KRAS acts on -- a
    scope 3 target can only move mass *within* a column, because the column
    margin (what j spends on inputs in total) is held fixed.
    """
    A = system.A if A is None else A
    x = system.x if x is None else x
    n = system.n

    Z = A * x[None, :]
    m = np.linalg.solve(np.eye(n) - A.T, system.s)
    m_p = power_multiplier(system, A=A)

    contrib = m[:, None] * Z                      # full upstream, by supplier
    contrib[power_mask(system), :] -= (m_p[:, None] * Z)[power_mask(system), :]
    return pd.DataFrame(contrib, index=pd.Index(system.sectors, name="supplier"),
                        columns=pd.Index(system.sectors, name="buyer"))


def check_identity(system: WorldSystem, tol: float = 1e-9) -> float:
    """
    Assert  m_j x_j == F_j + sum_i m_i Z_ij  and return the max relative gap.
    """
    m, x, Z = system.m, system.x, system.Z
    lhs = m * x
    rhs = system.F + m @ Z
    keep = lhs > 0
    gap = float(np.max(np.abs(lhs[keep] - rhs[keep]) / lhs[keep]))
    if gap > tol:
        raise AssertionError(f"scope identity broken: max relative gap {gap:.3e} > {tol:.0e}")
    return gap


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Scope 1/2/3 decomposition.")
    ap.add_argument("--granularity", choices=["gics", "ngfs"], default="ngfs")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    args = ap.parse_args()

    sysm = build_world_system(args.year, args.granularity, args.table_type)
    print(f"  scope identity max relative gap: {check_identity(sysm):.2e}")

    tbl = scope_table(sysm)
    neg = tbl[tbl["scope3_up"] < 0]
    if not neg.empty:
        print(f"\n[warning] negative scope 3 in {len(neg)} sector(s): {list(neg.index)}")

    show = tbl.copy()
    for col in ("scope1", "scope2", "scope3_up", "cradle"):
        show[col] = show[col] / KG_PER_MT
    show["scope3_share"] *= 100
    show = show[["x", "scope1", "scope2", "scope3_up", "cradle", "scope3_share"]]
    show = show.sort_values("cradle", ascending=False)

    print(f"\n{args.granularity} {args.year} -- Mt CO2, scope3_share in %")
    print(show.to_string(float_format=lambda v: f"{v:,.1f}"))

    print(f"\nworld scope 1 total : {tbl['scope1'].sum() / KG_PER_MT:,.0f} Mt "
          f"(= direct emissions, {sysm.F.sum() / KG_PER_MT:,.0f} Mt)")
    print(f"world cradle total  : {tbl['cradle'].sum() / KG_PER_MT:,.0f} Mt "
          "(counts every tier again at each stage -- not a world inventory)")
