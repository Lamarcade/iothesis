"""
Scope 3 decarbonisation targets imposed with KRAS.

`optim/io_footprint.py` constrains **rows**: the footprint embodied in what a
supplier sells. A scope 3 target is the transpose of that question -- it
constrains a **column**: the footprint embodied in what a buyer buys. Same
b_ij parametrisation, different constraint structure, so it needs its own
builder.

    scope3_j  =  sum_i  mu_i * b_ij * x_j          coefficient on b_ij: mu_i x_j

with mu the *effective* upstream intensity: mu_i = m_i for an ordinary sector,
and m_i - m^P_i for a power sector, because the power block's own emissions are
the buyer's scope 2, not its scope 3 (see scopes.py). Column j of
`scopes.scope3_by_supplier` is exactly mu_i Z_ij, so the constraint sums that
column.

Column margins are kept, row margins are not -- and that is the whole model
--------------------------------------------------------------------------
**Column** margins are held: sector j's total spend on intermediate inputs is
fixed, so KRAS cannot make j buy *less*. It can only change **who j buys from**,
shifting the sourcing mix toward suppliers with a lower mu. That is the right
reading of a scope 3 target: supplier switching at constant input volume.

**Row** margins are deliberately left free. Holding them too would lock the
answer at zero: with the row margins u fixed,

    sum_j scope3_j  =  sum_j sum_i mu_i b_ij x_j  =  sum_i mu_i u_i

is a constant. Every sector's target could then only be met by pushing scope 3
onto some other sector -- the world total cannot move by one gram, and KRAS
spends itself redistributing (some sectors came out three times *worse* when
this was tried). It is the transpose of the remark in optim/footprint_demo.py:
with column margins fixed, per-row footprint targets can only redistribute.

Letting the rows go is not a relaxation of rigour, it is the economics: if
every buyer switches away from carbon-intensive suppliers, those suppliers
**sell less**. Their sales are an outcome, not a constraint. Output levels then
follow from the Leontief rebalance at fixed final demand, reported as
`x_expost`. Pass `include_row=True` to see the locked variant.

A sector whose suppliers are all carbon-intensive still cannot reach -60 %.
KRAS does not fail there -- the sigma relaxation lets the target slip, and the
gap between target and achieved is where the constraint binds. That gap is the
result, not a defect.

The substitution artefact, stated up front
------------------------------------------
Targeting scope 3 *alone* rewards shifting purchases toward power sectors,
whose emissions are reclassified as the buyer's scope 2 rather than removed.
This mirrors a real criticism of scope-3-only targets. It is why the achieved
scope 2 is reported next to the achieved scope 3 in every result table, and why
`--basis upstream` exists: it targets scopes 2+3 together (coefficient m_i x_j),
which closes the loophole.

Normalisation -- the trap
-------------------------
In KRAS the constraint coefficients are **exponents**: the update is
`a[idx] * r ** (sign(a) * g)` (optim/kras.py:223). In raw units g would be
x ~ 1e7 and mu*x ~ 1e12, and r**g overflows. Everything is therefore divided
down to order 1 before being handed to the solver; only relative reductions
matter, so nothing is lost. Same reasoning as optim/exioptim.py:112-115.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optim.kras import kras  # noqa: E402
from scope3.aggregate import WorldSystem, build_world_system  # noqa: E402
from scope3.scopes import KG_PER_MT, power_mask, power_multiplier, scope_table  # noqa: E402

RESULT_DIR = REPO_ROOT / "Data" / "processed" / "scope3_kras"

DEFAULT_TARGETS = (0.30, 0.60)
#: sigma_scope3 / sigma_margin. **Above** 1: the scope 3 target is the more
#: relaxable constraint, so KRAS holds the margins and lets the target slip
#: where it cannot be met. optim/exioptim.py used 0.2, the opposite way round,
#: because it asked the opposite question ("how far do margins bend if the
#: footprint target is non-negotiable?"). Here the margins carry the meaning --
#: they are what makes this "same input volume, different suppliers" -- and the
#: shortfall against the target is the result we are after.
DEFAULT_SIGMA_RATIO = 5.0
#: Margin sigma as a fraction of the mean margin.
MARGIN_SIGMA_FRAC = 0.05


def effective_intensity(system: WorldSystem, A: np.ndarray | None = None,
                        basis: str = "scope3") -> np.ndarray:
    """
    mu -- the intensity that makes  sum_i mu_i Z_ij  the quantity being targeted.

    basis="scope3"    excludes the power block's own emissions (they are the
                      buyer's scope 2)
    basis="upstream"  scopes 2+3 together, mu = m
    """
    A = system.A if A is None else A
    m = np.linalg.solve(np.eye(system.n) - A.T, system.s)
    if basis == "upstream":
        return m
    if basis != "scope3":
        raise ValueError("basis must be 'scope3' or 'upstream'")
    mu = m.copy()
    mask = power_mask(system)
    mu[mask] -= power_multiplier(system, A=A)[mask]
    return mu


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def build_scope3_constraints(x: np.ndarray, u: np.ndarray, v: np.ndarray,
                             mu: np.ndarray, E3: np.ndarray,
                             include_row: bool = False):
    """
    (G, c, kind) for column margins, one scope 3 target per sector, and
    optionally row margins, in the b_ij parametrisation vectorised row-major
    (b_ij sits at i*n + j).

    `include_row=False` by default -- see the module docstring: fixing the row
    margins pins the world scope 3 total at a constant and leaves KRAS nothing
    to do but shuffle it between sectors.

    Everything is passed in **raw** units and normalised internally, because
    the coefficients of G become exponents inside KRAS.
    """
    n = len(x)
    X0 = float(np.mean(x))
    MU0 = float(np.mean(np.abs(mu[mu != 0]))) if np.any(mu != 0) else 1.0

    xs = x / X0                                    # O(1) margin coefficients
    mus = mu / MU0                                 # O(1) intensity coefficients

    rows, cols, data, c_list, kind = [], [], [], [], []
    r = 0

    if include_row:
        for i in range(n):                         # row margin i
            for j in range(n):
                rows.append(r); cols.append(i * n + j); data.append(xs[j])
            c_list.append(u[i] / X0); kind.append("row"); r += 1

    for j in range(n):                             # column margin j
        for i in range(n):
            rows.append(r); cols.append(i * n + j); data.append(xs[j])
        c_list.append(v[j] / X0); kind.append("col"); r += 1

    for j in range(n):                             # scope 3 target for buyer j
        for i in range(n):
            rows.append(r); cols.append(i * n + j); data.append(mus[i] * xs[j])
        c_list.append(E3[j] / (MU0 * X0)); kind.append("scope3"); r += 1

    G = sp.csr_matrix((data, (rows, cols)), shape=(r, n * n))
    return G, np.asarray(c_list, dtype=float), kind, (X0, MU0)


def feasible_floor(mu: np.ndarray, A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Lowest scope 3 each sector could reach by sourcing everything from its
    cleanest available supplier, at unchanged input spend.

    A target below this floor is arithmetically out of reach whatever KRAS
    does, so it is worth knowing before blaming the solver. GRAS cannot move
    an exact zero either, so only suppliers the sector already buys from count.
    """
    v = (A * x[None, :]).sum(axis=0)
    floor = np.empty_like(v)
    for j in range(len(v)):
        used = A[:, j] > 0
        floor[j] = v[j] * (mu[used].min() if used.any() else 0.0)
    return floor


def build_sigma(kind: list[str], v_norm_mean: float,
                sigma_ratio: float = DEFAULT_SIGMA_RATIO) -> np.ndarray:
    """Per-constraint reliability: margins loose, scope 3 targets tighter."""
    base = v_norm_mean * MARGIN_SIGMA_FRAC
    return np.array([base * sigma_ratio if k == "scope3" else base for k in kind])


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass
class KrasResult:
    """One decarbonisation target, solved."""
    target: float
    basis: str
    A: np.ndarray                       # rebalanced technical coefficients
    x_expost: np.ndarray                # (I - A_new)^-1 y, final demand held
    table: pd.DataFrame                 # per-sector base / target / achieved
    converged: bool
    sweeps: int
    outer_used: int
    margin_distortion: float
    negative_output: bool
    include_row: bool = False
    history: list[float] = field(default_factory=list)


def decarbonise(system: WorldSystem, target: float,
                basis: str = "scope3",
                sigma_ratio: float = DEFAULT_SIGMA_RATIO,
                include_row: bool = False,
                outer: int = 1,
                damping: float = 0.5,
                outer_tol: float = 1e-3,
                max_sweeps: int = 20000,
                tol: float = 1e-7,
                verbose: bool = True) -> KrasResult:
    """
    Cut every sector's scope 3 by `target` (0.30 = -30 %) and rebalance with KRAS.

    Ex-ante and ex-post
    -------------------
    The target is imposed at the **base** emission factors mu -- which is what a
    scope 3 commitment actually means: a company measures its inventory against
    a fixed factor database, not against a general equilibrium. `outer=1` (the
    default) does exactly that.

    Re-measuring the result with the mu implied by the *new* structure gives a
    substantially larger cut, because every supplier got cleaner at the same
    time and the multipliers compound. That number is reported too, and the
    difference between the two is a genuine finding rather than an error.

    `outer > 1` iterates towards the self-consistent fixed point where the
    constraint uses the mu of its own solution. Undamped Picard oscillates
    hard on this map (it amplifies), so the iteration is damped by `damping`.
    It is offered for inspection; it answers a different question from the one
    a corporate target asks.
    """
    n = system.n
    A0, x, y = system.A, system.x, system.y
    B0 = A0.flatten()

    base = scope_table(system)
    base_target_qty = (base["scope3_up"] if basis == "scope3"
                       else base["scope2"] + base["scope3_up"]).to_numpy()
    E3 = (1.0 - target) * base_target_qty

    u = (A0 * x[None, :]).sum(axis=1)        # row margins, raw units
    v = (A0 * x[None, :]).sum(axis=0)        # column margins = input spend

    A_new = A0
    mu = effective_intensity(system, A=A0, basis=basis)   # base factors
    achieved_prev = np.inf
    history: list[float] = []
    info: dict = {"converged": False, "sweeps": 0}
    used = 0

    for it in range(1, outer + 1):
        G, c, kind, (X0, MU0) = build_scope3_constraints(
            x, u, v, mu, E3, include_row=include_row)
        sigma = build_sigma(kind, float(np.mean(v / X0)), sigma_ratio)

        a, info = kras(B0, G, c, sigma=sigma, alpha=0.1,
                       tol=tol, max_sweeps=max_sweeps)
        A_new = a.reshape(n, n)
        used = it

        tbl_it = scope_table(system, A=A_new)
        achieved = float((tbl_it["scope3_up"] if basis == "scope3"
                          else tbl_it["scope2"] + tbl_it["scope3_up"]).sum())
        history.append(achieved)
        if verbose:
            print(f"    outer {it}: total {basis} re-measured = "
                  f"{achieved / KG_PER_MT:,.0f} Mt "
                  f"(target at base factors {E3.sum() / KG_PER_MT:,.0f}), "
                  f"converged={info['converged']} sweeps={info['sweeps']}")
        if it == outer:
            break
        if abs(achieved - achieved_prev) <= outer_tol * max(achieved, 1.0):
            break
        achieved_prev = achieved
        # Damped update towards the self-consistent factors: undamped Picard
        # oscillates because the map amplifies.
        mu = (1.0 - damping) * mu + damping * effective_intensity(
            system, A=A_new, basis=basis)

    # Distortion of the constraints that were actually imposed, relative to the
    # mean column margin. Row margins are an outcome here, not a constraint, so
    # they only enter when include_row asked for them.
    col_ach = (A_new * x[None, :]).sum(axis=0)
    worst = np.max(np.abs(col_ach - v))
    if include_row:
        row_ach = (A_new * x[None, :]).sum(axis=1)
        worst = max(worst, np.max(np.abs(row_ach - u)))
    distortion = float(worst / np.mean(v))

    # Ex-post: let output react to the new structure at fixed final demand.
    x_expost = np.linalg.solve(np.eye(n) - A_new, y)
    negative = bool((x_expost < 0).any())

    post = scope_table(system, A=A_new)
    post_expost = scope_table(system, A=A_new, x=x_expost)

    # Two different bookkeepings, kept apart on purpose:
    #   targeted_*  the quantity the constraint acts on (scope 3, or scopes 2+3
    #               under basis="upstream"), measured at BASE emission factors --
    #               the way a commitment is actually reported
    #   scope1/2/3  the real scopes, always meaning the same thing whatever the
    #               basis, so the substitution into scope 2 stays visible
    mu_base = effective_intensity(system, A=A0, basis=basis)
    targeted_exante = mu_base @ (A_new * x[None, :])

    tbl = pd.DataFrame({
        "x_base": base["x"],
        "x_expost": post_expost["x"],
        "targeted_base": base_target_qty,
        "targeted_target": E3,
        "targeted_floor": feasible_floor(mu_base, A0, x),
        "targeted_exante": targeted_exante,
        "scope1_base": base["scope1"],
        "scope2_base": base["scope2"],
        "scope3_base": base["scope3_up"],
        "scope2_achieved": post["scope2"],
        "scope3_achieved": post["scope3_up"],
        "scope2_expost": post_expost["scope2"],
        "scope3_expost": post_expost["scope3_up"],
        "cradle_base": base["cradle"],
        "cradle_expost": post_expost["cradle"],
    })

    def _cut(new, ref):
        ref = ref.where(ref > 0, np.nan) if hasattr(ref, "where") else ref
        return 100 * (1 - new / ref)

    ref_t = tbl["targeted_base"].where(tbl["targeted_base"] > 0, np.nan)
    ref_3 = tbl["scope3_base"].where(tbl["scope3_base"] > 0, np.nan)

    #: On the targeted basis, at base factors -- what the sector would report.
    tbl["reduction_exante"] = 100 * (1 - tbl["targeted_exante"] / ref_t)
    #: On the targeted basis, re-measured with the factors the new structure
    #: implies. Larger, because every supplier got cleaner at the same time.
    tbl["reduction_remeasured"] = 100 * (
        1 - (post["scope3_up"] if basis == "scope3"
             else post["scope2"] + post["scope3_up"]) / ref_t)
    #: Actual scope 3, re-measured, after output has rebalanced.
    tbl["scope3_cut_expost"] = 100 * (1 - tbl["scope3_expost"] / ref_3)
    #: Shortfall against the commitment: negative = could not source its way there.
    tbl["gap_vs_target"] = tbl["reduction_exante"] - 100 * target
    #: The target is arithmetically unreachable where it sits below the floor.
    tbl["target_below_floor"] = tbl["targeted_target"] < tbl["targeted_floor"]

    #: Procurement effort: the share of sector j's input budget that has to move
    #: to a different supplier. Half the L1 distance between the old and new
    #: sourcing shares -- the standard reading of that quantity as "fraction
    #: reallocated". With enough suppliers to choose from a target is almost
    #: always *reachable*; this is what it costs to reach it.
    share0 = np.divide(A0 * x[None, :], v[None, :], out=np.zeros_like(A0),
                       where=v[None, :] > 0)
    share1 = np.divide(A_new * x[None, :], v[None, :], out=np.zeros_like(A_new),
                       where=v[None, :] > 0)
    tbl["procurement_effort"] = 100 * 0.5 * np.abs(share1 - share0).sum(axis=0)

    #: Row margins are free, so sales are an outcome: who gains and who loses
    #: when every buyer switches away from carbon-intensive suppliers.
    u_new = (A_new * x[None, :]).sum(axis=1)
    tbl["sales_change"] = 100 * np.divide(u_new - u, u, out=np.zeros_like(u),
                                          where=u > 0)

    return KrasResult(
        target=target, basis=basis, A=A_new, x_expost=x_expost, table=tbl,
        converged=bool(info["converged"]), sweeps=int(info["sweeps"]),
        outer_used=used, margin_distortion=distortion,
        negative_output=negative, history=history, include_row=include_row,
    )


# ---------------------------------------------------------------------------
# Persistence, so the charts do not re-solve
# ---------------------------------------------------------------------------

def result_path(granularity: str, target: float, basis: str, year: int,
                table_type: str, include_row: bool = False) -> Path:
    suffix = "_rowfixed" if include_row else ""
    return RESULT_DIR / (f"{granularity}_{year}_{table_type}_{basis}_"
                         f"r{int(round(target * 100)):02d}{suffix}")


def save_result(res: KrasResult, system: WorldSystem) -> Path:
    stem = result_path(system.granularity, res.target, res.basis,
                       system.year, system.table_type, res.include_row)
    stem.parent.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(stem.with_suffix(".csv"))
    np.savez_compressed(stem.with_suffix(".npz"), A=res.A, x_expost=res.x_expost,
                        meta=np.array([res.converged, res.sweeps, res.outer_used,
                                       res.margin_distortion, res.negative_output],
                                      dtype=float))
    return stem.with_suffix(".csv")


#: Columns a cached result must carry to be usable. A file written by an older
#: version of `decarbonise` is silently stale otherwise -- the reader would hand
#: back a table missing the column its caller asks for, and fail far from here.
REQUIRED_COLUMNS = frozenset({
    "targeted_base", "targeted_target", "targeted_floor", "targeted_exante",
    "scope1_base", "scope2_base", "scope3_base",
    "scope2_achieved", "scope3_achieved", "scope3_expost",
    "reduction_exante", "gap_vs_target", "target_below_floor",
    "procurement_effort", "sales_change",
})


def load_result(system: WorldSystem, target: float, basis: str = "scope3",
                include_row: bool = False):
    """
    (table, A, x_expost) for a solved target, or None if it must be re-solved.

    None is returned both when nothing is cached and when what is cached
    predates the current column set, so a stale file just costs one re-solve.
    """
    stem = result_path(system.granularity, target, basis,
                       system.year, system.table_type, include_row)
    if not stem.with_suffix(".csv").exists():
        return None
    tbl = pd.read_csv(stem.with_suffix(".csv"), index_col=0)
    if not REQUIRED_COLUMNS.issubset(tbl.columns):
        return None
    blob = np.load(stem.with_suffix(".npz"))
    return tbl, blob["A"], blob["x_expost"]


def run_targets(system: WorldSystem, targets=DEFAULT_TARGETS,
                basis: str = "scope3", sigma_ratio: float = DEFAULT_SIGMA_RATIO,
                include_row: bool = False, outer: int = 1,
                verbose: bool = True) -> dict[float, KrasResult]:
    out = {}
    for r in targets:
        if verbose:
            print(f"\n  target -{r:.0%} ({system.granularity}, basis={basis}):")
        res = decarbonise(system, r, basis=basis, sigma_ratio=sigma_ratio,
                          include_row=include_row, outer=outer, verbose=verbose)
        save_result(res, system)
        out[r] = res
    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="KRAS scope 3 decarbonisation targets.")
    ap.add_argument("--granularity", choices=["gics", "ngfs"], default="ngfs")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--targets", default="0.30,0.60",
                    help="Comma-separated relative reductions.")
    ap.add_argument("--basis", choices=["scope3", "upstream"], default="scope3")
    ap.add_argument("--sigma-ratio", type=float, default=DEFAULT_SIGMA_RATIO,
                    help=">1 keeps the margins and lets the target slip.")
    ap.add_argument("--outer", type=int, default=1,
                    help="1 = target at base factors; >1 iterates to the "
                         "self-consistent fixed point (damped).")
    ap.add_argument("--include-row", action="store_true",
                    help="Also fix the row margins. This pins the world scope 3 "
                         "total, so targets can only be redistributed.")
    args = ap.parse_args()

    targets = [float(t) for t in args.targets.split(",")]
    sysm = build_world_system(args.year, args.granularity, args.table_type)
    results = run_targets(sysm, targets, args.basis, args.sigma_ratio,
                          args.include_row, args.outer)

    for r, res in results.items():
        t = res.table
        print(f"\n=== target -{r:.0%} ===")
        print(f"  KRAS converged={res.converged} sweeps={res.sweeps} "
              f"outer={res.outer_used}")
        print(f"  margin distortion incurred: {res.margin_distortion:.4f} "
              "(relative to the mean column margin)")
        if res.negative_output:
            print("  [warning] negative ex-post output: structure not admissible")
        def line(label, before, after, note=""):
            print(f"  {label:<34s} {before / KG_PER_MT:>9,.0f} -> "
                  f"{after / KG_PER_MT:>9,.0f} Mt  ({100 * (after / before - 1):+6.1f} %)"
                  f"{note}")

        line(f"{res.basis} at base factors", t["targeted_base"].sum(),
             t["targeted_exante"].sum(), "   <- what a commitment reports")
        line("scope 2", t["scope2_base"].sum(), t["scope2_achieved"].sum(),
             "   <- substitution into scope 2?")
        line("scope 3", t["scope3_base"].sum(), t["scope3_achieved"].sum())
        line("cradle-to-gate (ex-post)", t["cradle_base"].sum(),
             t["cradle_expost"].sum())

        n_below = int(t["target_below_floor"].sum())
        print(f"  targets below the sourcing floor (unreachable by arithmetic): "
              f"{n_below} of {len(t)}")

        short = t[t["gap_vs_target"] < -0.5]
        if not short.empty:
            show = short[["reduction_exante", "gap_vs_target",
                          "target_below_floor"]].copy()
            show["scope3_base_Mt"] = t["scope3_base"] / KG_PER_MT
            print("\n  sectors that fell short of the target:")
            print(show.sort_values("gap_vs_target").to_string(
                float_format=lambda v: f"{v:,.1f}"))
        else:
            print("\n  every sector reached the target; what varies is the effort.")

        eff = t[["procurement_effort", "sales_change"]].copy()
        eff["scope3_base_Mt"] = t["scope3_base"] / KG_PER_MT
        print("\n  hardest procurement shifts (% of the input budget that moves):")
        print(eff.sort_values("procurement_effort", ascending=False).head(8)
                 .to_string(float_format=lambda v: f"{v:,.1f}"))
        print("\n  suppliers that lose the most sales (%):")
        print(t["sales_change"].sort_values().head(6).to_string(
            float_format=lambda v: f"{v:,.1f}"))
