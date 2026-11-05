"""
Numerical controls for the scope 3 pipeline.

Plain asserts and printed evidence, same style as optim/test_kras.py -- no
pytest dependency. Run:  python scope3/test_scopes.py

Needs the aggregated system, so the first run parses EXIOBASE (slow); after
that it reads Data/processed/scope3_cache/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_pipeline import load_extension, load_x  # noqa: E402
from scope3.aggregate import (  # noqa: E402
    CO2_STRESSORS, GRANULARITIES, build_world_system,
)
from scope3.carbon_cost import KG_PER_T, carbon_bill, price_shock  # noqa: E402
from scope3.kras_scope3 import decarbonise  # noqa: E402
from scope3.scopes import KG_PER_MT, check_identity, scope3_by_supplier, scope_table  # noqa: E402

YEAR, TABLE_TYPE = 2019, "pxp"
_BAR = "=" * 68


def _systems():
    return {g: build_world_system(YEAR, g, TABLE_TYPE, verbose=False)
            for g in GRANULARITIES}


def test_scope_identity(systems):
    """m_j x_j == F_j + sum_i m_i Z_ij, exact by construction."""
    print(_BAR)
    print("1) Scope identity  m x = F + m Z")
    for g, sysm in systems.items():
        gap = check_identity(sysm, tol=1e-9)
        print(f"   {g:5s} n={sysm.n:3d}  max relative gap {gap:.2e}")
        assert gap < 1e-9

    # The three scopes must also add back to the cradle total, per sector.
    for g, sysm in systems.items():
        t = scope_table(sysm)
        recon = t["scope1"] + t["scope2"] + t["scope3_up"]
        err = float(np.max(np.abs(recon - t["cradle"]) / t["cradle"]))
        print(f"   {g:5s} scope1+2+3 vs cradle: max relative gap {err:.2e}")
        assert err < 1e-12
        assert (t["scope3_up"] >= 0).all(), f"{g}: negative scope 3"

    # And scope3_by_supplier's columns must sum to scope3_up.
    for g, sysm in systems.items():
        t = scope_table(sysm)
        cols = scope3_by_supplier(sysm).sum(axis=0)
        err = float(np.max(np.abs(cols - t["scope3_up"]) / t["cradle"]))
        print(f"   {g:5s} scope3_by_supplier column sums: {err:.2e}")
        assert err < 1e-12
    print("PASS\n")


def test_aggregation_conserves(systems):
    """Aggregating regions away must not create or destroy anything."""
    print(_BAR)
    print("2) Aggregation conserves totals")
    x_raw = load_x(YEAR, TABLE_TYPE)
    F_raw = load_extension(YEAR, "air_emissions", TABLE_TYPE)["F"]
    co2_raw = float(F_raw.reindex(index=CO2_STRESSORS).fillna(0.0).to_numpy().sum())
    x_total = float(x_raw.sum())

    for g, sysm in systems.items():
        dx = abs(sysm.x.sum() - x_total) / x_total
        dF = abs(sysm.F.sum() - co2_raw) / co2_raw
        print(f"   {g:5s} output {sysm.x.sum():.6e} vs {x_total:.6e}  ({dx:.2e})")
        print(f"   {g:5s} CO2    {sysm.F.sum():.6e} vs {co2_raw:.6e}  ({dF:.2e})")
        assert dx < 1e-12 and dF < 1e-12

    # The two granularities are two partitions of the same thing.
    a, b = (systems[g] for g in GRANULARITIES)
    assert abs(a.Z.sum() - b.Z.sum()) / a.Z.sum() < 1e-12
    print(f"   both granularities agree on total Z: {a.Z.sum():.6e}")
    print("PASS\n")


def test_leontief_control(systems):
    """(I-A)^-1 y must reproduce the observed x on a balanced world table."""
    print(_BAR)
    print("3) Leontief control  (I-A)^-1 y = x")
    for g, sysm in systems.items():
        res = sysm.leontief_residual()
        print(f"   {g:5s} max relative deviation {res:.2e}")
        assert res < 1e-9
    print("PASS\n")


def test_price_model(systems):
    """
    dp = (I-A^T)^-1 (tau s) must equal tau m.

    Two independent code paths -- a linear solve on the transposed system
    versus the multiplier computed on the untransposed one -- for the same
    quantity. This is what validates the price model, not a tolerance on a
    single number.
    """
    print(_BAR)
    print("4) Price model  dp = tau m")
    for g, sysm in systems.items():
        for tau in (25.0, 163.1):
            dp = price_shock(sysm.A, sysm.s, tau)
            ref = tau * sysm.m / KG_PER_T
            err = float(np.max(np.abs(dp - ref) / np.maximum(ref, 1e-30)))
            print(f"   {g:5s} tau={tau:6.1f}  max relative gap {err:.2e}")
            assert err < 1e-10

        # Linearity in tau: doubling the price doubles every cost.
        d1 = price_shock(sysm.A, sysm.s, 50.0)
        d2 = price_shock(sysm.A, sysm.s, 100.0)
        assert np.allclose(d2, 2 * d1, rtol=1e-12)
    print("   linear in tau: yes")
    print("PASS\n")


def test_world_carbon_bill(systems):
    """Under scope-1-only pricing the world bill is tau x direct emissions."""
    print(_BAR)
    print("5) World carbon bill, scope 1 only")
    tau = 163.1
    for g, sysm in systems.items():
        bill = carbon_bill(sysm, tau)
        got = float(bill["bill_scope1"].sum())
        want = tau * float(sysm.F.sum()) / KG_PER_T
        err = abs(got - want) / want
        print(f"   {g:5s} {got / 1e9:,.1f} bn US$ vs {want / 1e9:,.1f} bn US$ "
              f"({err:.2e})")
        assert err < 1e-12

        # The full-chain bill splits exactly into the three scopes.
        parts = bill[["bill_scope1", "bill_scope2", "bill_scope3_up"]].sum(axis=1)
        assert np.allclose(parts, bill["bill_full"], rtol=1e-12)
    print("   full-chain bill = scope 1 + 2 + 3, per sector")
    print("PASS\n")


def test_kras_margins(systems):
    """
    After a scope 3 target, the constraints that were imposed must hold.

    Column margins are the constraint (each sector's total input spend is
    unchanged); row margins are deliberately free, so they are *not* checked --
    see the kras_scope3 docstring for why fixing them would pin the answer.
    """
    print(_BAR)
    print("6) KRAS: column margins held, targets met or explained")
    sysm = systems["gics"]
    v_base = (sysm.A * sysm.x[None, :]).sum(axis=0)

    for target in (0.30, 0.60):
        res = decarbonise(sysm, target, outer=1, verbose=False)
        v_new = (res.A * sysm.x[None, :]).sum(axis=0)
        drift = float(np.max(np.abs(v_new - v_base) / v_base))
        t = res.table
        short = t[t["gap_vs_target"] < -0.5]
        unexplained = short[~short["target_below_floor"]]

        print(f"   -{target:.0%}: converged={res.converged} sweeps={res.sweeps} "
              f"| column margin drift {drift:.2e}")
        print(f"          {len(short)} sector(s) short of target, "
              f"{len(unexplained)} of them not explained by the sourcing floor")

        assert res.converged, "KRAS did not converge"
        assert drift < 5e-2, f"column margins moved by {drift:.1%}"
        assert not res.negative_output, "ex-post output went negative"
        assert unexplained.empty, f"unexplained shortfall: {list(unexplained.index)}"

        # Signs must make sense: the world total falls, nobody's scope 3 rises.
        assert t["targeted_exante"].sum() < t["targeted_base"].sum()
        assert (t["gap_vs_target"] <= 0.5).all(), "target overshot upward"
    print("PASS\n")


def test_row_margins_pin_the_total(systems):
    """
    The structural claim behind leaving the row margins free.

    With them fixed, sum_j scope3_j = sum_i mu_i u_i is a constant, so no
    across-the-board target can move the world total. Demonstrated rather than
    asserted in prose.
    """
    print(_BAR)
    print("7) Fixing the row margins pins the world total")
    sysm = systems["gics"]
    free = decarbonise(sysm, 0.30, include_row=False, outer=1, verbose=False)
    pinned = decarbonise(sysm, 0.30, include_row=True, outer=1, verbose=False)

    base = free.table["targeted_base"].sum()
    for name, res in (("rows free", free), ("rows fixed", pinned)):
        got = res.table["targeted_exante"].sum()
        print(f"   {name:11s} world scope 3 {base / KG_PER_MT:,.0f} -> "
              f"{got / KG_PER_MT:,.0f} Mt ({100 * (got / base - 1):+.1f} %)")

    cut_free = 1 - free.table["targeted_exante"].sum() / base
    cut_pinned = 1 - pinned.table["targeted_exante"].sum() / base
    assert cut_free > 0.25, "rows free: the -30 % target should be about reached"
    assert cut_pinned < cut_free / 2, "fixing the rows should throttle the cut"
    print("PASS\n")


if __name__ == "__main__":
    print("Loading the aggregated systems...")
    systems = _systems()
    print(f"  {', '.join(f'{g}: n={s.n}' for g, s in systems.items())}\n")

    test_scope_identity(systems)
    test_aggregation_conserves(systems)
    test_leontief_control(systems)
    test_price_model(systems)
    test_world_carbon_bill(systems)
    test_kras_margins(systems)
    test_row_margins_pin_the_total(systems)

    print(_BAR)
    print("All controls passed.")
