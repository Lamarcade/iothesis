"""
The projection: scenario -> constraints -> KRAS -> Z', A', y', E'.

This is the paper's Figure 1, end to end. `optim/kras.py` is used unchanged; it
already accepts a sparse G and pre-extracts the per-row supports from the CSR
buffers (kras.py:174-191).

The closure is production-oriented
----------------------------------
Output is the entry point of the scenario (section 3.3), so x^{T,k} is imposed
and final demand is what adjusts:

    y' = (I - A') x^{T,k}                                          eq. (8)

which, since A' = Z' diag(x')^-1, is just `y' = x' - Z' . 1`: no matrix product,
and exact. This is the opposite direction from `scope3/kras_scope3.py:303`,
which holds y and lets output react. Both are defensible; this one is the
paper's.

Two identities that fall out, and what they mean
------------------------------------------------
**The aggregate footprint is pinned by the scenario, not by KRAS.** Since
L' y' = L' (I - A') x' = x', eq. (4) collapses to

    E' = S' L' y' = S' x'

whatever KRAS does to the flows. That is not a defect, it is what a
production-oriented closure *means*: once output and intensities are given, the
production-based total is determined and rebalancing can only move emissions
between value chains, never remove them. So `E'` is used here as an **accounting
check** (`checks["footprint_identity"]` must be at machine precision), and the
quantities actually reported as results are the ones that do move:

    b'_j = sum_i s'_i Z'_ij     realised vs target -- where the constraint bound
    u'_j = sum_i m'_i Z'_ij     full upstream, nonlinear, ex post only
    y' - y0                     the final demand shift the target forces
    procurement_effort          the share of each input budget that changed hands

**Zeros stay zero.** GRAS cannot move an exact zero, so no supply relationship
absent from the base year can open (section 4.2, Limitations). The count is
reported: at `world` resolution Z is nearly dense and this barely bites; at
`region_ngfs` it bites hard.

y' can go negative, and is capped iteratively
----------------------------------------------
x^{T,k} does not itself preserve x = Ax + y (it is a mechanical rescaling, see
`shortterm/project_production.py`), so a sector whose buyers grow faster than it
does can already show y0 < 0 before any rebalancing. With the rows free (they
have to be, see `v1/constraints.py`), KRAS's emission-driven reallocation can
also push a *previously fine* row negative, since a supplier that gains a lot of
substituted demand can end up selling more than it produces.

There is no native inequality constraint in KRAS (equalities with soft
relaxation only), so `y'_i >= 0` is approximated: any row found negative after a
solve is pinned to the boundary, `sum_j Z'_ij = x_target_i` (i.e. y'_i = 0), and
the system is re-solved. Repeated up to `max_cap_iters` times, because capping
some rows can, in turn, push others across the line -- confirmed necessary on
real data, not a theoretical worry. This is a heuristic that finds *a*
feasible-looking point, not the minimal-distortion point a true
inequality-constrained balancer would find; `enforce_nonnegative_y=False`
reproduces the original, uncapped behaviour for comparison.
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

from io_pipeline import _safe_inv  # noqa: E402
from optim.kras import kras  # noqa: E402
from v1.constraints import (  # noqa: E402
    DEFAULT_SIGMA_CAP_RATIO,
    DEFAULT_SIGMA_FRAC,
    DEFAULT_SIGMA_RATIO,
    ConstraintSet,
    build_constraints,
    power_mix_groups,
)
from v1.resolution import split_label  # noqa: E402
from v1.system import IOSystem  # noqa: E402

RESULT_DIR = REPO_ROOT / "Data" / "processed" / "v1_runs"

KG_PER_MT = 1e9

#: Max re-solves for the y' >= 0 cap loop. Each is a fresh, cheap KRAS solve
#: (world resolution: ~1 s), so 5 bounds the cost at ~5x one run.
DEFAULT_MAX_CAP_ITERS = 5


@dataclass
class Projection:
    """One solved (scenario, horizon, ambition)."""

    scenario: str
    year: str
    labels: list[str]
    resolution: str
    table_type: str
    base_year: int

    Z_new: np.ndarray
    A_new: np.ndarray
    x_target: np.ndarray
    y_expost: np.ndarray
    table: pd.DataFrame

    converged: bool
    sweeps: int
    phase_switch: int | None
    checks: dict
    relaxation: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _multipliers(A_new: np.ndarray, s: np.ndarray) -> np.ndarray:
    """m' = s' (I - A')^-1, via the transposed solve -- no explicit inverse."""
    n = A_new.shape[0]
    return np.linalg.solve(np.eye(n) - A_new.T, s)


def _split_by_origin(labels: list[str], weights: np.ndarray,
                     Z: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Split `sum_i weights_i Z_ij` by whether supplier i sits in buyer j's region.

    Returns `(domestic, imported)`, or None when the labels carry no region at
    all (`world` resolution) -- there is no such thing as an import in a system
    with one region, and a column of zeros would read as "nothing is imported",
    which is a different and false claim.

    Loops over the distinct regions (6 at `eu_macro`, 49 at `region_ngfs`)
    rather than materialising an n x n same-region boolean, which would be
    96 MB at `full`.
    """
    codes = np.array([split_label(lab)[0] for lab in labels], dtype=object)
    if all(c is None for c in codes):
        return None
    total = weights @ Z
    domestic = np.zeros_like(total)
    for region in set(codes):
        rows = codes == region
        cols = codes == region
        domestic[cols] = weights[rows] @ Z[np.ix_(rows, cols)]
    return domestic, total - domestic


def _procurement_effort(Z0: np.ndarray, Z_new: np.ndarray) -> np.ndarray:
    """
    Share of each column's input budget that changed supplier, in %.

    Half the L1 distance between the old and new sourcing shares -- the standard
    reading of that quantity as "fraction reallocated". Each column is
    normalised by its own total, so this measures *recomposition*, not the
    change of scale that the margins already carry.
    """
    v0 = Z0.sum(axis=0)
    v1 = Z_new.sum(axis=0)
    share0 = np.divide(Z0, v0[None, :], out=np.zeros_like(Z0), where=v0[None, :] != 0)
    share1 = np.divide(Z_new, v1[None, :], out=np.zeros_like(Z_new),
                       where=v1[None, :] != 0)
    return 100.0 * 0.5 * np.abs(share1 - share0).sum(axis=0)


def project(
    system: IOSystem,
    x_target: np.ndarray,
    beta: float | np.ndarray = 1.0,
    scenario: str = "?",
    year: str = "?",
    s_target: np.ndarray | None = None,
    include_emissions: bool = True,
    include_row: bool = False,
    target_mask: np.ndarray | None = None,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
    sigma_ratio: float = DEFAULT_SIGMA_RATIO,
    sigma_cap_ratio: float = DEFAULT_SIGMA_CAP_RATIO,
    enforce_nonnegative_y: bool = True,
    max_cap_iters: int = DEFAULT_MAX_CAP_ITERS,
    cap_tol: float = 1e-6,
    power_mix_prior: bool = False,
    mix_groups: list[np.ndarray] | None = None,
    alpha: float = 0.1,
    tol: float = 1e-8,
    max_sweeps: int = 5000,
    verbose: bool = True,
) -> tuple[Projection, ConstraintSet]:
    """
    Balance Z under the scenario constraints and close the accounting.

    `include_row=False` by default: fixing the row margins pins the world total
    of embodied emissions and pins y' to y0, which makes eq. (8) vacuous. See
    the docstring of `v1/constraints.py`.

    `enforce_nonnegative_y=True` by default: any row whose y'_i comes out
    negative (beyond `cap_tol`, a fraction of mean|x_target| -- KRAS's own `tol`
    is an *absolute* residual, so at M.EUR scale a row it considers converged
    can still leave a -1e-9-scale y'_i that a strict `< 0` would wrongly flag)
    is pinned to `sum_j Z'_ij = x_target_i` and the system is re-solved, up to
    `max_cap_iters` times (capping some rows can push others negative in turn).
    A heuristic stand-in for a true inequality constraint -- see the module
    docstring. `False` reproduces the original, uncapped behaviour, for
    comparison.

    `power_mix_prior=True` starts KRAS from the power-mix-consistent prior
    (`v1/constraints.py`): inside each buyer's column, purchases of power
    technologies follow each technology's own growth in x', at an unchanged
    total. `mix_groups` overrides the automatic detection of the power rows.
    Everything measured "against the counterfactual" (y0, b_counterfactual,
    procurement effort, leakage) is then measured against that prior.
    """
    n = system.n
    A, s = system.A, system.s
    x_target = np.asarray(x_target, dtype=float)
    y_tol = cap_tol * float(np.mean(np.abs(x_target)))

    if power_mix_prior and mix_groups is None:
        mix_groups = power_mix_groups(system.labels, system.table_type)
        if not mix_groups:
            raise ValueError("power_mix_prior: no power-generation rows found in "
                             "the labels of this system")
    if not power_mix_prior:
        mix_groups = None

    cap_mask = np.zeros(n, dtype=bool)
    n_iters = max_cap_iters if enforce_nonnegative_y else 1
    for it in range(1, n_iters + 1):
        cs = build_constraints(
            system.Z, A, x_target, s, beta=beta, s_target=s_target,
            include_emissions=include_emissions, include_row=include_row,
            target_mask=target_mask,
            row_cap=cap_mask if enforce_nonnegative_y else None,
            sigma_frac=sigma_frac, sigma_ratio=sigma_ratio,
            sigma_cap_ratio=sigma_cap_ratio,
            x_base=system.x, mix_groups=mix_groups,
        )

        if verbose:
            cap_note = (f" + {cs.cap_rows.size} row_cap" if cs.cap_rows.size
                       else "")
            print(f"  [cap pass {it}/{n_iters}] constraints: {len(cs.kind)} "
                  f"rows ({n if include_row else 0} row + {n} col"
                  f"{cap_note} + {cs.emis_cols.size} emis), "
                  f"{cs.G.nnz:,} nnz over {n * n:,} cells")
            if it == 1 and (cs.dropped_zero_b.size or cs.dropped_no_support.size):
                print(f"  no emission target for {cs.dropped_zero_b.size} "
                      f"column(s) with b_j = 0 and {cs.dropped_no_support.size} "
                      f"with no adjustable carbon-carrying supplier")

        z, info = kras(cs.Z0.ravel(), cs.G, cs.c, sigma=cs.sigma, alpha=alpha,
                       tol=tol, max_sweeps=max_sweeps)
        Z_new = z.reshape(n, n)

        if not enforce_nonnegative_y:
            break
        y_now = x_target - Z_new.sum(axis=1)
        newly_negative = (y_now < -y_tol) & ~cap_mask
        if not newly_negative.any():
            break
        if verbose:
            print(f"  {int(newly_negative.sum())} sector(s) still have y' < "
                  f"-{y_tol:.3g}; capping and re-solving")
        cap_mask |= newly_negative
    cap_iterations = it

    # --- close the accounting, production-oriented ---------------------------
    A_new = Z_new * _safe_inv(x_target)[None, :]
    y_expost = x_target - Z_new.sum(axis=1)          # = (I - A') x'
    y0 = x_target - cs.Z0.sum(axis=1)                # = (I - A) x', a priori

    s_t = cs.s_target
    m_new = _multipliers(A_new, s_t)
    m_base = _multipliers(A, s)

    b_new = s_t @ Z_new
    b_counterfactual = s_t @ cs.Z0
    upstream_expost = m_new @ Z_new

    # --- checks --------------------------------------------------------------
    row_ach, col_ach = Z_new.sum(axis=1), Z_new.sum(axis=0)
    scale = max(float(np.mean(np.abs(cs.v0))), 1.0)
    footprint_my = float(m_new @ y_expost)
    footprint_sx = float(s_t @ x_target)
    checks = {
        # Only a check when the rows were imposed; otherwise it is the size of
        # the sales reallocation, which is a result and can legitimately be big.
        "row_residual": float(np.max(np.abs(row_ach - cs.u0)) / scale),
        "row_imposed": include_row,
        "col_residual": float(np.max(np.abs(col_ach - cs.v0)) / scale),
        "closure_residual": float(
            np.max(np.abs(y_expost - (x_target - A_new @ x_target))) / scale),
        "footprint_identity": abs(footprint_my - footprint_sx)
                              / max(abs(footprint_sx), 1.0),
        "footprint_Mt": footprint_sx / KG_PER_MT,
        # The fair baseline for y' is y0, not the base year: x^{T,k} is a
        # mechanical rescaling and is already IO-inconsistent, so (I-A)x' goes
        # negative wherever the scenario grew a sector's buyers faster than the
        # sector itself. Restoring that balance is what the rebalancing is for,
        # so the number that matters is whether y' has fewer negatives than y0.
        "negative_y_base": int((system.y < 0).sum()),
        "negative_y_apriori": int((y0 < 0).sum()),
        # y_tol, not a strict < 0: KRAS's own `tol` is an absolute residual, so
        # a row it considers converged can still leave a -1e-9-scale y'_i.
        "negative_y_expost": int((y_expost < -y_tol).sum()),
        "frozen_zeros": int((cs.Z0 == 0).sum()),
        "frozen_zeros_pct": 100.0 * float((cs.Z0 == 0).mean()),
        "nonneg_y_enforced": enforce_nonnegative_y,
        "cap_rows_used": int(cs.cap_rows.size),
        "cap_iterations": cap_iterations,
        "power_mix_prior": bool(mix_groups),
        "prior_rows": cs.prior_rows,
    }

    # --- which targets had to give way (paper section 7.4) -------------------
    drift = info["c_final"] - cs.c
    kind_arr = np.asarray(cs.kind)
    rows = []
    for k in ("row", "col", "row_cap", "emis"):
        m = kind_arr == k
        if not m.any():
            continue
        base = float(np.mean(np.abs(cs.c[m])))
        rows.append({
            "kind": k,
            "n": int(m.sum()),
            "n_relaxed": int((np.abs(drift[m]) > 1e-12).sum()),
            "max_abs_drift": float(np.max(np.abs(drift[m]))),
            "max_rel_drift_pct": 100.0 * float(np.max(np.abs(drift[m]))) / max(base, 1e-30),
            "sigma": float(cs.sigma[m][0]) if m.any() else 0.0,
        })
    relaxation = pd.DataFrame(rows).set_index("kind")

    # --- per-sector table ----------------------------------------------------
    idx = pd.Index(system.labels, name="sector")
    b_target_full = np.full(n, np.nan)
    b_target_full[cs.emis_cols] = (cs.beta * cs.b_base)[cs.emis_cols]

    tbl = pd.DataFrame({
        "x_base": system.x,
        "x_target": x_target,
        "growth_pct": 100.0 * np.divide(x_target - system.x, system.x,
                                        out=np.zeros(n), where=system.x > 0),
        "y_base": system.y,
        "y_apriori": y0,
        "y_expost": y_expost,
        "u_target": cs.u0,
        "u_achieved": row_ach,
        "v_target": cs.v0,
        "v_achieved": col_ach,
        "b_base": cs.b_base,
        "b_counterfactual": b_counterfactual,
        "b_target": b_target_full,
        "b_achieved": b_new,
        "b_floor": cs.b_floor,
        "b_ceiling": cs.b_ceiling,
        "upstream_expost": upstream_expost,
        "m_base": m_base,
        "m_expost": m_new,
        "procurement_effort": _procurement_effort(cs.Z0, Z_new),
    }, index=idx)

    ref = tbl["b_base"].where(tbl["b_base"] > 0, np.nan)
    #: Realised cut in embodied emissions against the base year -- the quantity
    #: eq. (7) targets, measured the way the target was written.
    tbl["b_cut_pct"] = 100.0 * (1.0 - tbl["b_achieved"] / ref)
    #: Shortfall against the commitment. Negative = could not source its way there.
    tbl["gap_vs_beta_pct"] = tbl["b_cut_pct"] - 100.0 * (1.0 - cs.beta)
    tbl.loc[np.isnan(b_target_full), "gap_vs_beta_pct"] = np.nan
    #: Final demand the decarbonization displaces, relative to the
    #: unchanged-technology counterfactual at the same output level.
    tbl["y_shift"] = tbl["y_expost"] - tbl["y_apriori"]
    tbl["has_emis_target"] = ~np.isnan(b_target_full)
    #: The target sits outside what the sector's existing supplier set can
    #: deliver at its fixed input budget: unreachable by arithmetic, not by
    #: solver failure.
    tbl["target_unreachable"] = (tbl["has_emis_target"]
                                 & ((tbl["b_target"] < tbl["b_floor"])
                                    | (tbl["b_target"] > tbl["b_ceiling"])))
    #: Carbon and money by supplier origin, "domestic" meaning the buyer's own
    #: region. With a region-targeted beta, KRAS can meet the target partly by
    #: buying from an untargeted region -- leakage, not abatement.
    #:
    #: Levels are given against two references -- `base` is the observed base
    #: year, `cf` the unchanged-technology counterfactual Z0 at the scenario's
    #: output -- because those genuinely differ and the ladder figure needs the
    #: second as its reference tick.
    #:
    #: For the *share*, the reference is the counterfactual. With the default
    #: prior this is no choice at all: Z0[:, j] = A[:, j] x'_j while
    #: Z[:, j] = A[:, j] x_j, so column j of Z0 is column j of Z times a
    #: **scalar**, composition is invariant, and the base-year and counterfactual
    #: shares are identical (verified at ~1e-13). The power-mix prior breaks that
    #: for the carbon share -- the technology mix of each region's electricity
    #: changes, hence its carbon, at unchanged spend by origin -- so
    #: `leakage_pp` is measured against `import_carbon_share_cf_pct`: what KRAS
    #: shifted abroad, net of what the scenario's power mix already moved.
    carbon_base = _split_by_origin(system.labels, s_t, system.Z)
    carbon_cf = _split_by_origin(system.labels, s_t, cs.Z0)
    carbon_new = _split_by_origin(system.labels, s_t, Z_new)
    budget_cf = _split_by_origin(system.labels, np.ones(n), cs.Z0)
    budget_new = _split_by_origin(system.labels, np.ones(n), Z_new)

    if carbon_base is None:          # `world`: one region, imports undefined
        for col in ("b_base_domestic", "b_base_imported",
                    "b_cf_domestic", "b_cf_imported",
                    "b_achieved_domestic", "b_achieved_imported",
                    "import_carbon_share_base_pct", "import_carbon_share_cf_pct",
                    "import_carbon_share_achieved_pct", "leakage_pp",
                    "import_budget_share_cf_pct",
                    "import_budget_share_achieved_pct"):
            tbl[col] = np.nan
    else:
        def _share(split):
            imported, total = split[1], split[0] + split[1]
            return 100.0 * np.divide(imported, total, out=np.full(n, np.nan),
                                     where=total > 0)

        tbl["b_base_domestic"], tbl["b_base_imported"] = carbon_base
        tbl["b_cf_domestic"], tbl["b_cf_imported"] = carbon_cf
        tbl["b_achieved_domestic"], tbl["b_achieved_imported"] = carbon_new
        tbl["import_carbon_share_base_pct"] = _share(carbon_base)
        tbl["import_carbon_share_cf_pct"] = _share(carbon_cf)
        tbl["import_carbon_share_achieved_pct"] = _share(carbon_new)
        #: Percentage points of embodied carbon that moved onto foreign
        #: suppliers. Positive = leakage, negative = the target was met by
        #: sourcing more at home. Measured against the counterfactual share,
        #: which equals the base-year share unless the power-mix prior is on.
        tbl["leakage_pp"] = (tbl["import_carbon_share_achieved_pct"]
                             - tbl["import_carbon_share_cf_pct"])
        tbl["import_budget_share_cf_pct"] = _share(budget_cf)
        tbl["import_budget_share_achieved_pct"] = _share(budget_new)

    #: With the rows free, sales are an outcome: who gains and who loses when
    #: every buyer shifts away from carbon-intensive suppliers.
    tbl["sales_change_pct"] = 100.0 * np.divide(
        row_ach - cs.u0, cs.u0, out=np.zeros(n), where=cs.u0 > 0)

    proj = Projection(
        scenario=scenario, year=str(year), labels=list(system.labels),
        resolution=system.resolution, table_type=system.table_type,
        base_year=system.year,
        Z_new=Z_new, A_new=A_new, x_target=x_target, y_expost=y_expost,
        table=tbl, converged=bool(info["converged"]), sweeps=int(info["sweeps"]),
        phase_switch=info["phase_switch"], checks=checks, relaxation=relaxation,
    )
    return proj, cs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

#: Columns a cached run must carry to be usable. A file written by an older
#: version of `project` is silently stale otherwise. Same guard as
#: scope3/kras_scope3.py:408-414.
REQUIRED_COLUMNS = frozenset({
    "x_base", "x_target", "y_apriori", "y_expost", "u_target", "u_achieved",
    "v_target", "v_achieved", "b_base", "b_target", "b_achieved",
    "b_floor", "b_ceiling", "upstream_expost", "m_expost",
    "procurement_effort", "b_cut_pct", "gap_vs_beta_pct", "y_shift",
    "has_emis_target", "target_unreachable", "sales_change_pct",
    "b_base_domestic", "b_base_imported", "b_cf_domestic", "b_cf_imported",
    "b_achieved_domestic", "b_achieved_imported",
    "import_carbon_share_base_pct", "import_carbon_share_achieved_pct",
    "leakage_pp", "import_budget_share_cf_pct",
    "import_budget_share_achieved_pct",
})


def result_stem(scenario: str, year: str, resolution: str, table_type: str,
                beta: float | np.ndarray, base_year: int,
                tag: str = "") -> Path:
    """
    Path stem for one run.

    `tag` distinguishes runs whose mean beta collides but whose scenario does
    not -- a Europe-only beta=0.7 at eu_macro averages to 0.95 and would
    otherwise silently overwrite a uniform beta=0.95 run.
    """
    b = float(np.mean(beta))
    suffix = f"_{tag}" if tag else ""
    return RESULT_DIR / (f"{scenario}_{base_year}-{year}_{resolution}_"
                         f"{table_type}_b{int(round(b * 100)):03d}{suffix}")


def save_projection(proj: Projection, beta: float | np.ndarray,
                    tag: str = "") -> Path:
    stem = result_stem(proj.scenario, proj.year, proj.resolution,
                       proj.table_type, beta, proj.base_year, tag)
    stem.parent.mkdir(parents=True, exist_ok=True)
    proj.table.to_csv(stem.with_suffix(".csv"), encoding="utf-8")
    np.savez_compressed(
        stem.with_suffix(".npz"),
        Z_new=proj.Z_new, A_new=proj.A_new, x_target=proj.x_target,
        y_expost=proj.y_expost,
        labels=np.array(proj.labels, dtype=object).astype(str),
    )
    return stem.with_suffix(".csv")


def load_projection(scenario: str, year: str, resolution: str, table_type: str,
                    beta: float, base_year: int, tag: str = ""):
    """(table, Z_new, A_new) for a solved run, or None if it must be re-solved."""
    stem = result_stem(scenario, year, resolution, table_type, beta, base_year,
                       tag)
    if not stem.with_suffix(".csv").exists():
        return None
    tbl = pd.read_csv(stem.with_suffix(".csv"), index_col=0)
    if not REQUIRED_COLUMNS.issubset(tbl.columns):
        return None
    blob = np.load(stem.with_suffix(".npz"), allow_pickle=False)
    return tbl, blob["Z_new"], blob["A_new"]


# ---------------------------------------------------------------------------

def report(proj: Projection) -> None:
    """Printed summary of one run."""
    t = proj.table
    bar = "=" * 74
    print(f"\n{bar}\n  {proj.scenario}  {proj.base_year} -> {proj.year}   "
          f"n = {proj.n}  ({proj.resolution}, {proj.table_type})\n{bar}")

    print(f"  KRAS converged={proj.converged}  sweeps={proj.sweeps}  "
          f"relaxation on at sweep {proj.phase_switch}")

    ck = proj.checks
    print("\n  accounting checks (relative):")
    row_note = "" if ck["row_imposed"] else "   <- free except capped rows, see below"
    print(f"    row margins  |Z'.1 - u0|   {ck['row_residual']:.2e}{row_note}")
    print(f"    col margins  |1'.Z' - v0|  {ck['col_residual']:.2e}")
    print(f"    closure      y' = (I-A')x' {ck['closure_residual']:.2e}")
    print(f"    footprint    m'y' = s'x'   {ck['footprint_identity']:.2e}"
          f"   (E' = {ck['footprint_Mt']:,.0f} Mt, set by the scenario)")
    print(f"    frozen zeros in Z0         {ck['frozen_zeros']:,} "
          f"({ck['frozen_zeros_pct']:.1f} %)")
    prior = (f"power-mix-consistent ({ck['prior_rows']} power rows reweighted "
             f"by their own growth)" if ck["power_mix_prior"]
             else "unchanged technology, Z0 = A diag(x')")
    print(f"    prior                      {prior}")

    cap_note = (f"   (enforced: {ck['cap_rows_used']} row(s) capped over "
               f"{ck['cap_iterations']} pass(es))" if ck["nonneg_y_enforced"]
               else "   (--allow-negative-y: not enforced)")
    print(f"    negative final demand      base {ck['negative_y_base']:>3d} "
          f"->  y0 {ck['negative_y_apriori']:>3d}  ->  y' "
          f"{ck['negative_y_expost']:>3d}   (of {proj.n}){cap_note}")
    if ck["nonneg_y_enforced"] and ck["negative_y_expost"] > 0:
        print(f"    [warning] {ck['negative_y_expost']} sector(s) still have "
              f"y' < 0 after {ck['cap_iterations']} capping pass(es) -- the cap "
              f"and the emission target could not both be satisfied at the "
              f"current sigma; raise --max-cap-iters or --sigma-cap-ratio")
    elif not ck["nonneg_y_enforced"] and ck["negative_y_expost"] > ck["negative_y_apriori"]:
        print("    [note] the free rebalancing added negative final demand "
              "rather than absorbing it (expected with --allow-negative-y)")

    print("\n  which targets had to give way:")
    print(proj.relaxation.to_string(float_format=lambda v: f"{v:,.4g}"))

    tgt = t[t["has_emis_target"]]
    if not tgt.empty:
        print(f"\n  embodied emissions of intermediate purchases, "
              f"{len(tgt)} targeted sectors:")
        print(f"    base        {tgt['b_base'].sum() / KG_PER_MT:>10,.0f} Mt")
        print(f"    target      {tgt['b_target'].sum() / KG_PER_MT:>10,.0f} Mt")
        print(f"    achieved    {tgt['b_achieved'].sum() / KG_PER_MT:>10,.0f} Mt")

        n_unreach = int(tgt["target_unreachable"].sum())
        print(f"    outside the sourcing band (unreachable by arithmetic): "
              f"{n_unreach} of {len(tgt)}")

        short = tgt[tgt["gap_vs_beta_pct"] < -0.5]
        if short.empty:
            print("\n  every targeted sector reached beta; what varies is the effort.")
        else:
            show = short[["b_cut_pct", "gap_vs_beta_pct", "procurement_effort",
                          "target_unreachable"]].copy()
            show["b_base_Mt"] = short["b_base"] / KG_PER_MT
            print(f"\n  {len(short)} sector(s) fell short of beta "
                  f"(worst 10):")
            print(show.sort_values("gap_vs_beta_pct").head(10)
                      .to_string(float_format=lambda v: f"{v:,.1f}"))

    eff = t[["procurement_effort", "growth_pct", "b_cut_pct"]].copy()
    print("\n  hardest procurement shifts (% of the input budget that moves):")
    print(eff.sort_values("procurement_effort", ascending=False).head(8)
             .to_string(float_format=lambda v: f"{v:,.1f}"))

    if t["b_cf_imported"].notna().any():
        # Against the unchanged-technology counterfactual, not the base year:
        # output is then held at the scenario level, so what is left is what the
        # target did. Restricted to the targeted columns -- untargeted ones move
        # only as a side effect and would dilute the number.
        tg = t[t["has_emis_target"]]
        if not tg.empty:
            dom_cf, imp_cf = tg["b_cf_domestic"].sum(), tg["b_cf_imported"].sum()
            dom_new = tg["b_achieved_domestic"].sum()
            imp_new = tg["b_achieved_imported"].sum()
            sh_cf = 100 * imp_cf / (imp_cf + dom_cf)
            sh_new = 100 * imp_new / (imp_new + dom_new)
            print(f"\n  embodied carbon of the {len(tg)} targeted sectors, by "
                  "supplier origin (vs the counterfactual at the same output):")
            print(f"    domestic  {dom_cf / KG_PER_MT:>9,.0f} -> "
                  f"{dom_new / KG_PER_MT:>9,.0f} Mt")
            print(f"    imported  {imp_cf / KG_PER_MT:>9,.0f} -> "
                  f"{imp_new / KG_PER_MT:>9,.0f} Mt   "
                  f"({sh_cf:.1f} % -> {sh_new:.1f} % of the total)")
            verdict = ("leakage: the target is met partly by importing the "
                       "emissions" if sh_new > sh_cf else
                       "no leakage: the target is met by sourcing more at home")
            print(f"    -> {verdict}")

            leak = tg["leakage_pp"].dropna()
            if not leak.empty:
                tail = leak.sort_values(ascending=False).head(5)
                if tail.max() > 0.5:
                    print("\n  sectors shifting carbon abroad (leakage, pp):")
                    print(tail.to_string(float_format=lambda v: f"{v:,.1f}"))

    if not proj.checks["row_imposed"]:
        sales = t["sales_change_pct"].sort_values()
        print("\n  sales are an outcome here (row margins free) -- who loses:")
        print(sales.head(6).to_string(float_format=lambda v: f"{v:,.1f}"))
        print("  who gains:")
        print(sales.tail(6).to_string(float_format=lambda v: f"{v:,.1f}"))

    shift = t["y_shift"]
    print(f"\n  final demand shift vs the unchanged-technology counterfactual: "
          f"{shift.abs().sum() / 2 / 1e6:,.2f} T.EUR reallocated")
    print(t[["y_apriori", "y_expost", "y_shift"]]
          .reindex(shift.abs().sort_values(ascending=False).index).head(6)
          .to_string(float_format=lambda v: f"{v:,.0f}"))


if __name__ == "__main__":
    import argparse

    from shortterm.project_production import available_scenarios
    from v1.resolution import RESOLUTIONS
    from v1.scenario import scenario_x
    from v1.system import DEFAULT_YEAR, build_io_system

    ap = argparse.ArgumentParser(
        description="Project the EXIOBASE Z matrix under a climate scenario.")
    ap.add_argument("--scenario", default="SWUC", choices=available_scenarios())
    ap.add_argument("--year", default="2030")
    ap.add_argument("--beta", type=float, default=0.7,
                    help="Ambition in (0,1]: 0.7 = cut embodied emissions to 70%%.")
    ap.add_argument("--beta-region", default=None,
                    help="Target only these regions (comma-separated, e.g. "
                         "'Europe'); every other column gets no emission "
                         "constraint. Needs a region-preserving resolution.")
    ap.add_argument("--resolution", choices=RESOLUTIONS, default="world")
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--base-year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--sigma-frac", type=float, default=DEFAULT_SIGMA_FRAC)
    ap.add_argument("--sigma-ratio", type=float, default=DEFAULT_SIGMA_RATIO,
                    help=">1 holds the margins and lets the target slip.")
    ap.add_argument("--sigma-cap-ratio", type=float, default=DEFAULT_SIGMA_CAP_RATIO,
                    help="Tightness of the y'>=0 row cap relative to the margins.")
    ap.add_argument("--allow-negative-y", action="store_true",
                    help="Do not cap rows to enforce y' >= 0; reproduces the "
                         "original, uncapped behaviour.")
    ap.add_argument("--power-mix-prior", action="store_true",
                    help="Start from the power-mix-consistent prior: buyers' "
                         "purchases of power technologies follow each "
                         "technology's growth in x', at unchanged total.")
    ap.add_argument("--max-cap-iters", type=int, default=DEFAULT_MAX_CAP_ITERS)
    ap.add_argument("--cap-tol", type=float, default=1e-6,
                    help="y' < -cap_tol*mean|x_target| counts as negative; "
                         "filters KRAS's own numerical residual.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--max-sweeps", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--no-emissions", action="store_true",
                    help="Margins only: the degenerate run that returns A' = A.")
    ap.add_argument("--include-row", action="store_true",
                    help="Also fix the row margins. This pins the world total of "
                         "embodied emissions and pins y' to y0, so beta can only "
                         "be redistributed. Diagnostic use only.")
    ap.add_argument("--rebuild", action="store_true", help="Ignore the npz cache.")
    args = ap.parse_args()

    sysm = build_io_system(args.base_year, args.resolution, args.table_type,
                           use_cache=not args.rebuild)
    xt = scenario_x(args.scenario, args.year, sysm, args.base_year)

    if args.beta_region:
        from v1.constraints import region_mask
        regions = [r.strip() for r in args.beta_region.split(",")]
        tmask = region_mask(sysm.labels, regions)
        tag = "-".join(r.replace(" ", "") for r in regions)
        print(f"  beta={args.beta:g} on {int(tmask.sum())} of {sysm.n} columns "
              f"({', '.join(regions)}); the rest get no emission target at all")
    else:
        tmask, tag = None, ""
    if args.power_mix_prior:
        # Part of the cache key: a prior run must not overwrite the default one.
        tag = f"{tag}_mixprior" if tag else "mixprior"

    proj, _ = project(
        sysm, xt, beta=args.beta, scenario=args.scenario, year=args.year,
        include_emissions=not args.no_emissions, include_row=args.include_row,
        target_mask=tmask,
        sigma_frac=args.sigma_frac, sigma_ratio=args.sigma_ratio,
        sigma_cap_ratio=args.sigma_cap_ratio,
        enforce_nonnegative_y=not args.allow_negative_y,
        max_cap_iters=args.max_cap_iters, cap_tol=args.cap_tol,
        power_mix_prior=args.power_mix_prior,
        alpha=args.alpha, tol=args.tol, max_sweeps=args.max_sweeps,
    )
    report(proj)
    out = save_projection(proj, args.beta, tag)
    print(f"\n  saved: {out.relative_to(REPO_ROOT)}")
