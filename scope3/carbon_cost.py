"""
Carbon cost along the value chain -- the Leontief price model.

The repo had no price-side model at all: `io_pipeline` solves the quantity
model x = (I-A)^-1 y, and the `factor_inputs` extension (where value added and
taxes live) is loaded by nothing. This module adds the dual.

The model
---------
Unit prices satisfy  p = v + A^T p, with v the value added per unit of output,
so a shock to unit costs propagates as

    dp = (I - A^T)^-1 dv

A carbon price tau applied to each sector's *direct* emissions is the cost shock
dv_j = tau * s_j. Substituting:

    dp_j = tau * (s (I-A)^-1)_j = tau * m_j

**The full-pass-through carbon cost of a sector is exactly tau times its
cradle-to-gate footprint.** So the scope 1 / 2 / 3 split of the emissions *is*
the split of the carbon bill -- one decomposition, two units. `price_shock()`
computes dp by solving the transposed system and `test_scopes.py` checks it
against tau * m: two independent code paths for the same quantity.

Two regimes, and the gap between them is the point
--------------------------------------------------
    scope1_only   tau * s_j    what a sector pays under an ETS that prices its
                               own stack, with no supplier passing anything on
    full_chain    tau * m_j    what it pays once every tier prices carbon and
                               passes it forward (a CBAM-like end state)

The difference is the sector's scope 2 + 3 exposure, in money.

Units
-----
EXIOBASE is in current M.EUR (2019 here), GEM-E3 carbon prices are in
US$2017/t CO2. Converting needs a rate, and there is no honest way to avoid
picking one: `EUR_PER_USD` is stated here rather than buried. It scales every
cost-as-%-of-output figure linearly, so a reader who prefers another rate can
rescale by inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scope3.aggregate import WorldSystem, build_world_system  # noqa: E402
from scope3.scopes import KG_PER_MT, scope_table  # noqa: E402

#: EUR per US$, 2019 annual average (ECB). EXIOBASE is M.EUR current, GEM-E3 is
#: US$2017. The 2017->2019 deflator is not applied: it is well under the spread
#: between scenarios and would give the number a false air of precision.
EUR_PER_USD = 0.893

#: kg per tonne.
KG_PER_T = 1000.0

#: GEM-E3 publishes country-level `Price|Carbon`, but the aggregates
#: (World, EU27, Asia, ...) only exist under this variant.
CARBON_PRICE_VARIABLE = "Price"
CARBON_PRICE_SUB = "Carbon Weighted Average"


# ---------------------------------------------------------------------------
# The price model
# ---------------------------------------------------------------------------

def price_shock(A: np.ndarray, s: np.ndarray, tau: float) -> np.ndarray:
    """
    dp = (I - A^T)^-1 (tau s) -- unit cost increase from a carbon price tau.

    Units: tau in US$/t CO2, s in kg CO2 per M.EUR of output, so dp comes out
    in US$ per M.EUR of output. `cost_share()` turns that into a fraction.
    """
    n = A.shape[0]
    return np.linalg.solve(np.eye(n) - A.T, tau * s / KG_PER_T)


def cost_share(dp: np.ndarray) -> np.ndarray:
    """US$ per M.EUR of output -> fraction of the value of that output."""
    return dp * EUR_PER_USD / 1e6


def carbon_bill(system: WorldSystem, tau: float,
                A: np.ndarray | None = None,
                x: np.ndarray | None = None) -> pd.DataFrame:
    """
    Carbon cost per sector at price `tau`, split by scope and by regime.

    Returns a DataFrame indexed by sector, in US$ and in share of output value:

        bill_scope1/2/3   the carbon bill attributable to each scope, US$
        bill_full         their sum -- the full-chain bill
        share_scope1      tau * s, as a fraction of the value of output
        share_full        tau * m, ditto -- what full pass-through costs
        exposure          share_full - share_scope1, the scope 2+3 exposure
    """
    tbl = scope_table(system, A=A, x=x)
    per_t = tau / KG_PER_T                       # US$ per kg

    out = pd.DataFrame(index=tbl.index)
    for scope in ("scope1", "scope2", "scope3_up"):
        out[f"bill_{scope}"] = tbl[scope] * per_t
    out["bill_full"] = tbl["cradle"] * per_t

    out["share_scope1"] = cost_share(tau * tbl["s"].to_numpy() / KG_PER_T)
    out["share_full"] = cost_share(tau * tbl["m"].to_numpy() / KG_PER_T)
    out["exposure"] = out["share_full"] - out["share_scope1"]
    out["scope3_share"] = tbl["scope3_share"]
    out["x"] = tbl["x"]
    return out


# ---------------------------------------------------------------------------
# GEM-E3 carbon prices
# ---------------------------------------------------------------------------

def carbon_price(scenario: str, year: int | str, region: str = "World") -> float:
    """
    GEM-E3 carbon price, US$2017/t CO2, for one NGFS short-term scenario.

    Regions available under the weighted-average variant: World, EU27, EU28,
    Euro Area, Africa, Asia, North America, South America.
    """
    from shortterm.project_production import load_geme3_variable

    frame = load_geme3_variable(CARBON_PRICE_VARIABLE, scenario,
                                CARBON_PRICE_SUB, regions=[region])
    col = str(year)
    if col not in frame.columns:
        raise ValueError(f"year {col} not in GEM-E3 carbon price "
                         f"({list(frame.columns)[0]}..{list(frame.columns)[-1]})")
    return float(frame.loc[region, col])


def carbon_price_table(region: str = "World") -> pd.DataFrame:
    """Every scenario x year carbon price for one region, US$2017/t CO2."""
    from shortterm.project_production import SHORT_TERM_SCENARIOS, load_geme3_variable

    rows = {}
    for spec in SHORT_TERM_SCENARIOS:
        frame = load_geme3_variable(CARBON_PRICE_VARIABLE, spec.name,
                                    CARBON_PRICE_SUB, regions=[region])
        rows[spec.name] = frame.loc[region]
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Carbon cost along the value chain.")
    ap.add_argument("--granularity", choices=["gics", "ngfs"], default="ngfs")
    ap.add_argument("--year", type=int, default=2019, help="EXIOBASE base year.")
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--scenario", default="SWUC")
    ap.add_argument("--price-year", default="2030")
    ap.add_argument("--region", default="World")
    args = ap.parse_args()

    sysm = build_world_system(args.year, args.granularity, args.table_type)

    print(f"GEM-E3 carbon price, {args.region} (US$2017/t CO2):")
    print(carbon_price_table(args.region).to_string(float_format=lambda v: f"{v:,.1f}"))

    tau = carbon_price(args.scenario, args.price_year, args.region)
    print(f"\nUsing {args.scenario} {args.price_year} {args.region}: "
          f"tau = {tau:,.1f} US$/t")

    # Control: the two code paths for dp must agree.
    dp = price_shock(sysm.A, sysm.s, tau)
    gap = np.max(np.abs(dp - tau * sysm.m / KG_PER_T) / np.maximum(dp, 1e-12))
    print(f"price model vs tau*m: max relative gap {gap:.2e}")

    bill = carbon_bill(sysm, tau)
    show = bill[["share_scope1", "share_full", "exposure", "scope3_share"]] * 100
    show["bill_full_bn$"] = bill["bill_full"] / 1e9
    show = show.sort_values("share_full", ascending=False)
    print(f"\nCarbon cost as % of the value of output, {args.scenario} "
          f"{args.price_year}:")
    print(show.to_string(float_format=lambda v: f"{v:,.1f}"))

    print(f"\nworld bill, scope 1 only  : {bill['bill_scope1'].sum() / 1e9:,.0f} bn US$"
          f"  (control: tau x direct emissions = "
          f"{tau * sysm.F.sum() / KG_PER_T / 1e9:,.0f} bn US$)")
    print(f"world bill, full chain    : {bill['bill_full'].sum() / 1e9:,.0f} bn US$"
          "  (tiers counted again at each stage -- a sum over sectors, not a world total)")
