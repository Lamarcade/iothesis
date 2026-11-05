"""
Checks for the v1 projection pipeline.

Plain asserts and printed evidence, same style as `optim/test_kras.py` and
`scope3/test_scopes.py` -- no pytest dependency.

    python v1/test_v1.py               synthetic checks + the real system
    python v1/test_v1.py --fast        synthetic checks only (no EXIOBASE)

The synthetic checks run in milliseconds and cover the algebra. The EXIOBASE
checks need the base-year system; the first run parses Z.txt and takes a few
minutes, later runs hit the npz cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v1.constraints import (  # noqa: E402
    beta_by_region,
    build_constraints,
    power_mix_groups,
    region_mask,
    unchanged_technology_margins,
)
from v1.project import project  # noqa: E402
from v1.resolution import split_label  # noqa: E402
from v1.system import DEFAULT_YEAR, IOSystem, build_io_system  # noqa: E402

_BAR = "=" * 68


def _synthetic_system(n: int = 5, seed: int = 0) -> IOSystem:
    """A small balanced IO system: x = Z.1 + y holds by construction."""
    rng = np.random.default_rng(seed)
    x = np.array([100.0, 120.0, 90.0, 150.0, 80.0])[:n]
    A = rng.uniform(0.02, 0.18, size=(n, n))
    Z = A * x[None, :]
    y = x - Z.sum(axis=1)
    # Intensities spread over two orders of magnitude, as in the real data.
    F = rng.uniform(0.05, 5.0, size=n) * x
    return IOSystem(
        labels=[f"S{i}" for i in range(n)], Z=Z, x=x, y=y, F=F,
        year=DEFAULT_YEAR, table_type="synthetic", resolution="synthetic",
    )


def _asymmetric_growth_target(system: IOSystem) -> np.ndarray:
    """
    x_target that deterministically drives y0_1 < 0.

    Sector 1 supplies every other sector fairly heavily (row 1 of A is the
    densest). Shrinking sector 1's own output while growing everyone who buys
    from it -- exactly the "buyers outgrow the supplier" mechanism described in
    the module docstrings -- forces `u0_1 = sum_j A_1j x'_j` past `x'_1`.
    """
    x_target = system.x.copy()
    x_target[1] = 20.0
    others = [i for i in range(system.n) if i != 1]
    x_target[others] *= 5.0
    return x_target


# ---------------------------------------------------------------------------

def test_base_year_accounting(system: IOSystem, name: str) -> None:
    print(f"{_BAR}\n1. base-year accounting  x = Z.1 + y   [{name}]\n{_BAR}")
    resid = system.row_balance_residual()
    leon = system.leontief_residual()
    print(f"  n = {system.n}")
    print(f"  max relative gap  x - (Z.1 + y)   : {resid:.3e}")
    print(f"  max relative gap  x - (I-A)^-1 y  : {leon:.3e}")
    assert resid < 1e-9, f"row balance broken: {resid:.3e}"
    assert leon < 1e-8, f"leontief control broken: {leon:.3e}"
    print("PASS\n")


def test_degenerate_identity(system: IOSystem, name: str) -> None:
    """Margins only, a0 = Z0: KRAS must return exactly the input."""
    print(f"{_BAR}\n2. degenerate identity  (margins only -> A' = A)   [{name}]\n{_BAR}")
    x_target = system.x * 1.15          # a real, uniform-ish shock
    for include_row in (False, True):
        proj, _ = project(system, x_target, beta=1.0, include_emissions=False,
                          include_row=include_row, verbose=False)
        err_A = float(np.max(np.abs(proj.A_new - system.A)))
        tag = "rows fixed" if include_row else "rows free "
        print(f"  [{tag}] converged={proj.converged} sweeps={proj.sweeps}  "
              f"max|A'-A| = {err_A:.3e}  col residual = "
              f"{proj.checks['col_residual']:.3e}")
        assert proj.converged, "margins-only run did not converge"
        assert err_A < 1e-10, f"A' drifted from A: {err_A:.3e}"
    print("PASS\n")


def test_neutral_scenario(system: IOSystem, name: str) -> None:
    """x' = x and beta = 1: y' must reproduce the observed final demand."""
    print(f"{_BAR}\n3. neutral scenario  (x' = x, beta = 1 -> y' = y)   [{name}]\n{_BAR}")
    proj, _ = project(system, system.x.copy(), beta=1.0, verbose=False)

    scale = max(float(np.mean(np.abs(system.y))), 1.0)
    err_y = float(np.max(np.abs(proj.y_expost - system.y))) / scale
    err_A = float(np.max(np.abs(proj.A_new - system.A)))
    print(f"  converged={proj.converged}  sweeps={proj.sweeps}")
    print(f"  max |y' - y| / mean|y|            : {err_y:.3e}")
    print(f"  max |A' - A|                      : {err_A:.3e}")
    assert proj.converged, "neutral run did not converge"
    assert err_y < 1e-9, f"final demand not reproduced: {err_y:.3e}"
    print("PASS\n")


def test_closure_and_footprint(system: IOSystem, name: str) -> None:
    """y' = (I-A')x' and E' = m'y' = s'x', whatever KRAS did."""
    print(f"{_BAR}\n4. closure and footprint identity   [{name}]\n{_BAR}")
    rng = np.random.default_rng(3)
    x_target = system.x * rng.uniform(0.8, 1.4, size=system.n)   # heterogeneous
    proj, _ = project(system, x_target, beta=0.7, verbose=False)

    ck = proj.checks
    print(f"  converged={proj.converged}  sweeps={proj.sweeps}")
    print(f"  closure   y' vs (I-A')x'          : {ck['closure_residual']:.3e}")
    print(f"  footprint m'y' vs s'x'            : {ck['footprint_identity']:.3e}")
    print(f"  E' = {ck['footprint_Mt']:,.4g} Mt   <- set by the scenario, not by KRAS")
    assert ck["closure_residual"] < 1e-9, "eq. (8) does not hold"
    assert ck["footprint_identity"] < 1e-9, "E' != s'x': the closure is broken"
    print("PASS\n")


def test_reachable_target(system: IOSystem) -> None:
    """A target inside every column's sourcing band must be met exactly."""
    print(f"{_BAR}\n5. reachable emission target   [synthetic]\n{_BAR}")
    beta = 0.9
    proj, cs = project(system, system.x.copy(), beta=beta, verbose=False)

    t = proj.table[proj.table["has_emis_target"]]
    rel = np.max(np.abs(t["b_achieved"] - t["b_target"]) / np.abs(t["b_target"]))
    print(f"  converged={proj.converged}  sweeps={proj.sweeps}  "
          f"targeted columns={len(t)}/{proj.n}")
    print(f"  targets outside the sourcing band : "
          f"{int(t['target_unreachable'].sum())}")
    print(f"  worst |b' - beta.b| / beta.b      : {rel:.3e}")
    print(f"  column margin residual            : "
          f"{proj.checks['col_residual']:.3e}   <- imposed")
    print(f"  row margin drift                  : "
          f"{proj.checks['row_residual']:.3e}   <- free, an outcome")
    print(f"  procurement effort, mean          : "
          f"{proj.table['procurement_effort'].mean():.2f} % of the input budget")
    assert not t["target_unreachable"].any(), "the test case is not reachable"
    assert proj.converged, "reachable target did not converge"
    assert rel < 1e-6, f"target missed by {rel:.3e}"
    assert proj.checks["col_residual"] < 1e-6, "column margins deformed"
    print("PASS\n")


def test_row_margins_pin_the_total(system: IOSystem) -> None:
    """
    Why the row margins are not imposed: they pin the world embodied total.

        sum_j b'_j = sum_i s'_i (sum_j Z'_ij) = sum_i s'_i u0_i

    is a constant once the rows are fixed, so a uniform beta < 1 is out of reach
    by arithmetic. This check states the identity and then shows KRAS failing to
    reach the target with the rows on, and reaching it with the rows off.
    """
    print(f"{_BAR}\n7. the row-margin trap   [synthetic]\n{_BAR}")
    u0, v0, Z0 = unchanged_technology_margins(system.A, system.x)
    b_total = float((system.s @ system.Z).sum())
    pinned = float(system.s @ u0)
    print(f"  sum_j b_j          = {b_total:.6f}")
    print(f"  sum_i s_i u0_i     = {pinned:.6f}")
    assert abs(b_total - pinned) / abs(pinned) < 1e-12, "the identity does not hold"

    beta = 0.9
    locked, _ = project(system, system.x.copy(), beta=beta,
                        include_row=True, verbose=False)
    free, _ = project(system, system.x.copy(), beta=beta,
                      include_row=False, verbose=False)

    def worst(p):
        t = p.table[p.table["has_emis_target"]]
        return float(np.max(np.abs(t["b_achieved"] - t["b_target"])
                            / np.abs(t["b_target"])))

    w_locked, w_free = worst(locked), worst(free)
    print(f"  worst relative miss, rows FIXED   : {w_locked:.3e}")
    print(f"  worst relative miss, rows FREE    : {w_free:.3e}")
    assert w_locked > 1e-3, "rows fixed should NOT be able to reach beta"
    assert w_free < 1e-6, "rows free should reach beta"
    print("PASS\n")


def test_nonnegative_final_demand(system: IOSystem) -> None:
    """
    y'_i can go negative with the rows free; the iterative cap must remove it.

    Uses `_asymmetric_growth_target`, which forces y0_1 < 0 by construction (a
    supplier shrinks while everyone who buys from it grows). This is the same
    mechanism observed on real EXIOBASE data (SWUC 2030: 39/200 sectors), just
    made deterministic and small enough to assert on.
    """
    print(f"{_BAR}\n8. non-negative final demand   [synthetic]\n{_BAR}")
    x_target = _asymmetric_growth_target(system)

    u0, v0, Z0 = unchanged_technology_margins(system.A, x_target)
    y0 = x_target - u0
    print(f"  y0 (a priori)  = {np.array2string(y0, precision=1)}")
    assert y0[1] < 0, "test setup is wrong: y0_1 should be negative by design"

    uncapped, _ = project(system, x_target, beta=1.0, include_emissions=False,
                          enforce_nonnegative_y=False, verbose=False)
    capped, _ = project(system, x_target, beta=1.0, include_emissions=False,
                        enforce_nonnegative_y=True, verbose=False)

    n_neg_uncapped = int((uncapped.y_expost < 0).sum())
    n_neg_capped = int((capped.y_expost < 0).sum())
    print(f"  y' negatives, --allow-negative-y  : {n_neg_uncapped}")
    print(f"  y' negatives, capped (default)    : {n_neg_capped}")
    print(f"  rows capped                       : "
          f"{capped.checks['cap_rows_used']} over "
          f"{capped.checks['cap_iterations']} pass(es)")
    print(f"  col margin residual, capped        : "
          f"{capped.checks['col_residual']:.3e}   <- still imposed")

    assert n_neg_uncapped > 0, "the uncapped run should reproduce the negative"
    assert capped.converged, "the capped run did not converge"
    assert n_neg_capped == 0, f"{n_neg_capped} sector(s) still negative after capping"
    assert capped.checks["col_residual"] < 1e-6, "capping broke the column margins"
    print("PASS\n")


def test_exponent_range(system: IOSystem) -> None:
    """G's coefficients become exponents in KRAS: they must stay bounded."""
    print(f"{_BAR}\n6. exponent range in G   [synthetic]\n{_BAR}")
    u0, v0, Z0 = unchanged_technology_margins(system.A, system.x)
    cs = build_constraints(system.Z, system.A, system.x, system.s, beta=0.8)

    kind = np.asarray(cs.kind)
    G = cs.G.tocsr()
    margin_rows = np.flatnonzero((kind == "row") | (kind == "col"))
    emis_rows = np.flatnonzero(kind == "emis")

    margin_data = np.concatenate(
        [G.data[G.indptr[i]:G.indptr[i + 1]] for i in margin_rows])
    emis_data = np.concatenate(
        [G.data[G.indptr[i]:G.indptr[i + 1]] for i in emis_rows])

    print(f"  margin coefficients   min={margin_data.min():.6f} "
          f"max={margin_data.max():.6f}")
    print(f"  emission coefficients min={emis_data.min():.3e} "
          f"max={emis_data.max():.6f}   (s' / max s' = {cs.s_scale:.3g})")
    assert np.all(margin_data == 1.0), "margin coefficients are not exactly 1"
    assert emis_data.max() <= 1.0 + 1e-12, "emission exponent above 1"
    assert emis_data.min() >= 0.0, "negative emission exponent"

    # The margins must be consistent with each other by construction.
    gap = abs(u0.sum() - v0.sum()) / max(abs(v0.sum()), 1.0)
    print(f"  |sum(u0) - sum(v0)| / sum(v0)     : {gap:.3e}   "
          f"<- consistent by construction")
    assert gap < 1e-12, "the margin construction is not self-consistent"
    print("PASS\n")


def test_region_targeting() -> None:
    """
    Regional selection, and the trap it exists to avoid.

    `beta = 1` on the untargeted columns is NOT "leave alone": beta is measured
    against the base year, so it pins their embodied emissions to the base-year
    level despite scenario output growth. Only `target_mask` genuinely removes
    the constraint, and the two must produce different constraint counts.
    """
    print(f"{_BAR}\n11. regional targeting   [synthetic]\n{_BAR}")
    labels = ([f"Europe | S{i}" for i in range(36)]
              + [f"Asia | S{i}" for i in range(36)]
              + [f"Oceania | S{i}" for i in range(36)])

    mask = region_mask(labels, "Europe")
    print(f"  labels={len(labels)}  selected={int(mask.sum())}")
    assert mask.shape == (len(labels),), "wrong shape"
    assert int(mask.sum()) == 36, f"expected 36, got {int(mask.sum())}"
    assert mask[:36].all() and not mask[36:].any(), "wrong columns selected"
    assert int(region_mask(labels, ["Europe", "Asia"]).sum()) == 72, "multi wrong"

    beta_vec = beta_by_region(labels, "Europe", 0.7)
    assert np.allclose(beta_vec[:36], 0.7) and np.allclose(beta_vec[36:], 1.0)

    # A typo must fail loudly, not silently select nothing and look converged.
    for bad in (lambda: region_mask(labels, "Erope"),
                lambda: beta_by_region(labels, "Erope", 0.7)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown region should raise")
    print("  unknown region raises in both helpers, as it must")

    # target_mask really removes rows; beta=1 really does not.
    system = _synthetic_system()
    lab5 = ["Europe | A", "Europe | B", "Asia | C", "Asia | D", "Asia | E"]
    m5 = region_mask(lab5, "Europe")
    masked = build_constraints(system.Z, system.A, system.x, system.s,
                               beta=0.7, target_mask=m5)
    beta_one = build_constraints(system.Z, system.A, system.x, system.s,
                                 beta=beta_by_region(lab5, "Europe", 0.7))
    n_masked = int((np.asarray(masked.kind) == "emis").sum())
    n_beta1 = int((np.asarray(beta_one.kind) == "emis").sum())
    print(f"  emission rows: target_mask={n_masked}   beta=1 elsewhere={n_beta1}")
    assert n_masked == 2, f"target_mask should leave 2 emission rows, got {n_masked}"
    assert n_beta1 == 5, f"beta=1 still constrains every column, got {n_beta1}"
    print("PASS\n")


def test_power_mix_prior(system: IOSystem) -> None:
    """
    The power-mix-consistent prior reallocates inside a group, and nothing else.

    Rows 0, 2, 3 play three power technologies: one shrinks, one triples, one
    grows modestly, while every buyer grows by 10 %. Under the plain Z0 each
    technology's sales grow with its buyers; under the prior they follow the
    technology's own growth, at an unchanged column total.
    """
    print(f"{_BAR}\n13. power-mix-consistent prior   [synthetic]\n{_BAR}")
    group = [np.array([0, 2, 3])]
    x_target = system.x * 1.1
    x_target[[0, 2, 3]] = system.x[[0, 2, 3]] * np.array([0.5, 3.0, 1.2])
    g = x_target / system.x

    _, v0, Z0 = unchanged_technology_margins(system.A, x_target)
    _, v0m, Zm = unchanged_technology_margins(system.A, x_target, system.x, group)

    col_gap = float(np.max(np.abs(v0m - v0)) / np.mean(v0))
    others = [1, 4]
    ratio = (Zm[0] / Zm[2]) / ((Z0[0] * g[0]) / (Z0[2] * g[2]))
    print(f"  column totals |v0m - v0| / mean   : {col_gap:.3e}   <- unchanged")
    print(f"  rows outside the group            : "
          f"{'identical' if np.array_equal(Zm[others], Z0[others]) else 'CHANGED'}")
    print(f"  within-column ratio identity      : max dev {np.max(np.abs(ratio - 1)):.3e}")
    print(f"  shrinking row 0, sales / output   : Z0 {Z0[0].sum() / x_target[0]:.2f}"
          f" -> prior {Zm[0].sum() / x_target[0]:.2f}")
    assert col_gap < 1e-12, "the prior changed the column margins"
    assert np.array_equal(Zm[others], Z0[others]), "rows outside the group moved"
    assert np.max(np.abs(ratio - 1)) < 1e-12, "reallocation is not proportional to growth"
    assert Zm[0].sum() < Z0[0].sum(), "the shrinking technology should sell less"

    # Uniform growth inside the group: nothing to reallocate.
    _, _, Zu = unchanged_technology_margins(system.A, system.x * 1.3, system.x, group)
    _, _, Z0u = unchanged_technology_margins(system.A, system.x * 1.3)
    assert np.max(np.abs(Zu - Z0u)) < 1e-12, "uniform growth should leave Z0 alone"

    # Margins only, started from the prior: KRAS has nothing to do.
    proj, cs = project(system, x_target, beta=1.0, include_emissions=False,
                       enforce_nonnegative_y=False, power_mix_prior=True,
                       mix_groups=group, verbose=False)
    err = float(np.max(np.abs(proj.A_new - Zm / x_target[None, :])))
    print(f"  margins-only run from the prior   : sweeps={proj.sweeps}  "
          f"max|A' - Zm/x'| = {err:.3e}   prior rows = {cs.prior_rows}")
    assert proj.converged and err < 1e-10, "KRAS moved away from a consistent prior"
    assert cs.prior_rows == 3 and proj.checks["power_mix_prior"]
    print("PASS\n")


def test_power_mix_prior_exiobase(eu: IOSystem, world: IOSystem) -> None:
    """
    On EXIOBASE: the power rows are found at both label schemes, and the prior
    keeps every buyer's spend by supplier region -- the property that leaves the
    money import share invariant and makes leakage measurable against it.
    """
    print(f"{_BAR}\n14. power-mix prior on EXIOBASE   [exiobase]\n{_BAR}")
    gw = power_mix_groups(world.labels, world.table_type)
    ge = power_mix_groups(eu.labels, eu.table_type)
    print(f"  world   : {len(gw)} group(s) of {[len(r) for r in gw]} rows")
    print(f"  eu_macro: {len(ge)} group(s) of {sorted({len(r) for r in ge})} rows")
    assert len(gw) == 1 and "Electricity by coal" in [world.labels[i] for i in gw[0]]
    assert "Electricity nec" not in [world.labels[i] for i in gw[0]], "grid in group"
    assert len(ge) == 6, f"expected one group per eu_macro bloc, got {len(ge)}"

    rng = np.random.default_rng(5)
    x_target = eu.x * rng.uniform(0.4, 3.0, size=eu.n)
    _, v0, Z0 = unchanged_technology_margins(eu.A, x_target)
    _, v0m, Zm = unchanged_technology_margins(eu.A, x_target, eu.x, ge)

    regions = np.array([split_label(lab)[0] for lab in eu.labels], dtype=object)

    def imported_budget_share(Z):
        total = Z.sum(axis=0)
        domestic = np.zeros_like(total)
        for r in set(regions):
            m = regions == r
            domestic[m] = Z[np.ix_(m, m)].sum(axis=0)
        return np.divide(total - domestic, total, out=np.zeros_like(total),
                         where=total > 0)

    col_gap = float(np.max(np.abs(v0m - v0)) / np.mean(v0))
    share_gap = float(np.max(np.abs(imported_budget_share(Zm)
                                    - imported_budget_share(Z0))))
    moved = float(np.abs(Zm - Z0).sum() / 2 / Z0.sum())
    print(f"  column totals |v0m - v0| / mean   : {col_gap:.3e}")
    print(f"  imported money share, max change  : {share_gap:.3e}   <- spend by origin kept")
    print(f"  intermediate flows reallocated    : {100 * moved:.2f} %")
    assert col_gap < 1e-12, "the prior changed the column margins"
    assert share_gap < 1e-12, "the prior moved spend between supplier regions"
    assert moved > 0, "heterogeneous growth should reallocate something"
    print("PASS\n")


def test_origin_share_invariance(eu: IOSystem) -> None:
    """
    The imported *share* is the same against the base year and against Z0.

    Z0[:, j] = A[:, j] x'_j and Z[:, j] = A[:, j] x_j, so column j of Z0 is
    column j of Z times a scalar: composition cannot change. `project()` relies
    on this to report a single share column instead of two identical ones, so
    if a future change makes Z0 something other than a column rescaling of Z,
    this check is what says the leakage attribution now needs a choice of
    reference.
    """
    print(f"{_BAR}\n12. origin share invariance   [exiobase]\n{_BAR}")
    rng = np.random.default_rng(11)
    x_target = eu.x * rng.uniform(0.7, 1.5, size=eu.n)

    _, _, Z0 = unchanged_technology_margins(eu.A, x_target)
    s = eu.s

    def imported_share(Z):
        regions = np.array([split_label(lab)[0] for lab in eu.labels],
                           dtype=object)
        total = s @ Z
        domestic = np.zeros_like(total)
        for r in set(regions):
            m = regions == r
            domestic[m] = s[m] @ Z[np.ix_(m, m)]
        return 100.0 * np.divide(total - domestic, total,
                                 out=np.full(eu.n, np.nan), where=total > 0)

    gap = np.nanmax(np.abs(imported_share(Z0) - imported_share(eu.Z)))
    print(f"  heterogeneous growth applied, x'/x in [0.7, 1.5]")
    print(f"  max |share(Z0) - share(Z)|        : {gap:.3e}")
    assert gap < 1e-9, f"the share is not invariant: {gap:.3e}"
    print("PASS\n")


def test_eu_macro_conserves(eu: IOSystem, world: IOSystem) -> None:
    """
    `eu_macro` is an aggregation of the same table as `world`, so every total
    must agree. This is what makes the regional system *closed*.
    """
    print(f"{_BAR}\n9. eu_macro balances and conserves   [exiobase]\n{_BAR}")
    resid = eu.row_balance_residual()
    print(f"  n = {eu.n}   (expected 216 at pxp: 6 blocs x 36 NGFS sectors)")
    print(f"  row balance  x = Z.1 + y          : {resid:.3e}")
    for name, a, b in (("x", eu.x.sum(), world.x.sum()),
                       ("y", eu.y.sum(), world.y.sum()),
                       ("F", eu.F.sum(), world.F.sum()),
                       ("Z", eu.Z.sum(), world.Z.sum())):
        rel = abs(a - b) / max(abs(b), 1.0)
        print(f"  total {name}: eu_macro {a:>20,.1f}  vs world {b:>20,.1f}  "
              f"rel {rel:.2e}")
        assert rel < 1e-9, f"total {name} differs between resolutions: {rel:.3e}"
    assert resid < 1e-9, f"eu_macro does not balance: {resid:.3e}"

    blocs = sorted({split_label(lab)[0] for lab in eu.labels})
    print(f"  blocs: {blocs}")
    assert "Europe" in blocs, "no Europe bloc"
    assert len(blocs) == 6, f"expected 6 blocs, got {len(blocs)}"
    print("PASS\n")


def test_slice_breaks_aggregate_does_not(eu: IOSystem) -> None:
    """
    The justification for `eu_macro` existing at all: cutting Europe out of the
    matrix destroys the accounting, aggregating the rest of the world does not.
    """
    print(f"{_BAR}\n10. slicing breaks, aggregating does not   [exiobase]\n{_BAR}")
    regions = np.array([split_label(lab)[0] for lab in eu.labels])
    idx = np.flatnonzero(regions == "Europe")

    # The naive slice: keep only the Europe block of Z, x, y.
    Z_eu = eu.Z[np.ix_(idx, idx)]
    x_eu, y_eu = eu.x[idx], eu.y[idx]
    gap = np.abs(Z_eu.sum(axis=1) + y_eu - x_eu)
    keep = x_eu > 0
    sliced = float(np.max(gap[keep] / x_eu[keep]))

    dropped = eu.Z[np.ix_(idx, np.flatnonzero(regions != "Europe"))].sum()
    closed = eu.row_balance_residual()

    print(f"  Europe sectors                     : {len(idx)}")
    print(f"  naive slice, max relative residual : {sliced:.3e}   <- broken")
    print(f"  closed system, same check          : {closed:.3e}   <- fine")
    print(f"  European intermediate sales to non-European buyers, dropped by "
          f"the slice: {dropped:,.0f} M.EUR")
    assert sliced > 0.1, ("the naive slice was expected to break the balance; "
                          f"got only {sliced:.3e}")
    assert closed < 1e-9, "the closed system must still balance"
    print("PASS\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="v1 pipeline checks.")
    ap.add_argument("--fast", action="store_true",
                    help="Synthetic checks only; skip loading EXIOBASE.")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--resolution", default="world")
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    args = ap.parse_args()

    syn = _synthetic_system()
    test_base_year_accounting(syn, "synthetic")
    test_degenerate_identity(syn, "synthetic")
    test_neutral_scenario(syn, "synthetic")
    test_closure_and_footprint(syn, "synthetic")
    test_reachable_target(syn)
    test_exponent_range(syn)
    test_row_margins_pin_the_total(syn)
    test_nonnegative_final_demand(syn)
    test_region_targeting()
    test_power_mix_prior(syn)

    if args.fast:
        print("all synthetic checks passed (--fast: EXIOBASE checks skipped)")
        raise SystemExit(0)

    print(f"{_BAR}\nloading EXIOBASE {args.year} {args.resolution} "
          f"{args.table_type}\n{_BAR}")
    real = build_io_system(args.year, args.resolution, args.table_type)

    test_base_year_accounting(real, "exiobase")
    test_degenerate_identity(real, "exiobase")
    test_neutral_scenario(real, "exiobase")
    test_closure_and_footprint(real, "exiobase")

    # The regional closure. Needs both caches; build them first with
    #   python v1/system.py --resolution eu_macro
    # or this will re-parse Z.txt (minutes).
    eu = build_io_system(args.year, "eu_macro", args.table_type)
    world = (real if args.resolution == "world"
             else build_io_system(args.year, "world", args.table_type))
    test_eu_macro_conserves(eu, world)
    test_slice_breaks_aggregate_does_not(eu)
    test_origin_share_invariance(eu)
    test_power_mix_prior_exiobase(eu, world)

    print("all checks passed")
