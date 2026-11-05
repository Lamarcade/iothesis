"""
Scenario information as linear constraints:  G z = c,  with uncertainties sigma.

This is section 3.3 of the paper, made concrete. Three kinds of constraint, one
row of G each, on z = vec(Z') in row-major order (Z'_ij sits at i*n + j):

    kind      row of G                    target c              coefficient
    ----      -------------------------   -------------------   -----------
    col       sum_i Z'_ij                 v0_j                  1
    emis      sum_i s'_i Z'_ij            beta_j * b_j          s'_i / max(s')
    row       sum_j Z'_ij                 u0_i                  1     (opt-in, diagnostic)
    row_cap   sum_j Z'_ij                 x_target_i            1     (opt-in, per-row)

Balancing Z, not A
------------------
The paper states the problem on Z (section 3.3), and so does this module. That
is not only fidelity, it is numerical safety: KRAS uses the coefficients of G as
**exponents** (`a[idx] * r ** (sign(a) * g)`, optim/kras.py:223). In the b_ij =
A_ij parametrisation used by `optim/io_footprint.py` and `scope3/kras_scope3.py`
every margin coefficient is x_j ~ 1e7 and has to be scaled down by hand before
the solver sees it. Stated on Z, the margin coefficients are exactly 1 and
cannot overflow; only the emission rows carry a non-unit coefficient, and those
are normalised to [0, 1] by dividing by max(s') -- see `s_scale` below.

Where the margins come from: unchanged technology
-------------------------------------------------
Given base-year A and scenario output x^{T,k},

    Z0 = A diag(x')          the no-structural-change counterfactual
    u0 = Z0 . 1  = A x'      row margins    (intermediate sales)
    v0 = 1' . Z0 = x' (1'A)  column margins (intermediate purchases)

`v0_j = x'_j sum_i A_ij` is exactly `x'_j - v'_j` under a constant value-added
share, so the column balance of eq. (1) is honoured **without any VA data** --
the VA share is implied, never an input. On the row side the implied final
demand is `y0 = (I - A) x'`, which is eq. (8) evaluated at base-year technology.

The alternative -- holding each sector's intermediate/final split constant,
`u0_i = x'_i (u_i / x_i)` -- was rejected: it makes the row and column totals
diverge whenever growth is heterogeneous, injecting a conflict into G z = c that
has nothing to do with climate. With the unchanged-technology margins the two
sets are **mutually consistent by construction**, so the decarbonization target
is the *only* source of conflict, and any deformation is attributable to it.
That is the paper's own claim in section 2.6.

Consequence, and it is the first test: with a0 = Z0 and no emission rows, KRAS
converges on the first sweep and returns A' = A exactly.

Optional: a power-mix-consistent prior
--------------------------------------
Z0 rescales each *column* by the buyer's growth, so every buyer keeps its
base-year supplier shares. That ignores what x^{T,k} says about the *rows* of
substitutable suppliers: GEM-E3 projects each power technology separately (SWUC
2030, world: coal power x0.51, wind x3.04, solar PV x7.21), yet under Z0 every
technology's intermediate sales grow by the buyers' average ~x1.17. Coal power
then sells about twice what it produces (y0 = -190 bn EUR) and most new wind and
solar output finds no intermediate buyer.

`mix_consistent_prior` fixes this inside groups of substitutable rows -- the
power technologies of one supplier region (`power_mix_groups`). Within each
column j and group G:

    Z0m_ij = Z0_ij g_i * (sum_{k in G} Z0_kj) / (sum_{k in G} Z0_kj g_k),
    g_i = x'_i / x_i

Each buyer's total power purchase from a region is unchanged, so v0 is exactly
unchanged and so is the money spent by supplier origin; only the technology mix
of the purchase follows the scenario. Rows outside the groups are untouched.
Off by default (`power_mix_prior=False` in `project`).

Why the row margins are NOT imposed by default
-----------------------------------------------
Only the column margins are used. Fixing the rows too is available
(`include_row=True`) but off, for two reasons that are the same reason.

**It makes the target arithmetically impossible.** With the row margins fixed,

    sum_j b'_j = sum_j sum_i s'_i Z'_ij = sum_i s'_i (sum_j Z'_ij) = sum_i s'_i u0_i

is a **constant**. The world total of embodied emissions cannot move by one
gram, so a uniform beta < 1 across all columns cannot be met at all -- KRAS can
only shuffle emissions between value chains, and spends itself doing so. This is
verified numerically: `sum_j b_j` and `sum_i s_i u0_i` agree to the last digit.
It is the transpose of the remark at `scope3/kras_scope3.py:19-40`.

**It makes the paper's own closure vacuous.** Eq. (8) computes y' = (I - A') x'
*after* balancing, i.e. final demand is an outcome. But `sum_j Z'_ij = u0_i` is
exactly `y'_i = y0_i`: fixing the row margins is fixing the answer. Section 3.3
derives the *column* sums from output and value added and never claims the row
sums; letting the rows go is the reading that keeps eq. (8) meaningful.

Economically this is the right way round: if every buyer shifts away from
carbon-intensive suppliers, those suppliers **sell less**. Their sales are a
result, not a constraint, and `sales_change` reports who wins and who loses.

What binds instead is per-column and interpretable: with its budget v0_j fixed
and only its existing suppliers to choose from, sector j's embodied emissions
live in `[v0_j min s'_i, v0_j max s'_i]` over its support -- see
`feasible_band`. A target outside that band is unreachable by arithmetic, not by
solver failure, and is flagged as such.

Free rows have a cost: y' can go negative
------------------------------------------
Since `x^{T,k}` is a mechanical rescaling of x that does not itself preserve
`x = Ax + y` (see `shortterm/project_production.py`'s own docstring), the
a-priori `y0 = (I-A)x'` can already be negative wherever a sector's buyers grow
faster than the sector itself -- observed on real EXIOBASE data (SWUC 2030,
world resolution: 39 of 200 sectors). With the rows free, the emission-driven
reallocation can shift this set: it can incidentally fix some rows and push
others -- including previously fine ones -- negative, because a "clean"
supplier that gains a lot of substituted demand can end up selling more than it
produces. `row_cap` (below) is the targeted, iterative fix; see the "Follow-up"
section of the v1 plan for the full reasoning and its limits.

`row_cap`: an equality standing in for an inequality
-----------------------------------------------------
KRAS solves equalities with soft relaxation; there is no inequality primitive,
and adding one means touching the shared solver `scope3/` and `optim/exioptim.py`
also depend on -- out of scope here. The practical approximation used instead:
for a row found to violate `y'_i >= 0`, pin it to the boundary,
`sum_j Z'_ij = x_target_i` (i.e. `y'_i = 0`), and re-solve. Because the
violating set is only known from the *result* of a solve, this is applied
iteratively in `v1/project.py::project` -- one pass cannot guarantee every row,
since capping some rows can push others over the line in turn. This is a
heuristic that finds *a* feasible-looking point, not the minimal-distortion
point a true inequality-constrained balancer would find.

Which columns get an emission target
------------------------------------
Not all of them. A column with `b_j = 0` (nothing carbon-intensive in its input
mix) would be handed the target 0, which no multiplicative update can reach:
the solver would drive its scaling factor towards zero and crush the column.
Columns with no adjustable support -- every supplier with s'_i > 0 already at an
exact zero flow -- are excluded for the same reason, plus they would block
convergence by leaving a residual no update can touch. Both exclusions are
counted and reported rather than silently applied.

Sparsity
--------
G is filtered to the sparsity pattern of Z0. GRAS cannot move an exact zero, so
a cell that starts at zero stays at zero forever and its column in G is pure
memory. Dropping it changes nothing about the solution or the residuals.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shortterm.st_exiobase_mapping import MAPPING  # noqa: E402
from v1.resolution import split_label  # noqa: E402

#: Uncertainty of a target, as a fraction of the mean target of its own kind.
#: Arbitrary in v1 -- section 4.3 of the paper (calibration) is not implemented.
DEFAULT_SIGMA_FRAC = 0.05
#: sigma_emis / sigma_margin. Above 1 the emission target is the more relaxable
#: constraint, so KRAS holds the margins and lets the target slip where it
#: cannot be met -- the shortfall then *is* the result. Same reading as
#: scope3/kras_scope3.py:86-93.
DEFAULT_SIGMA_RATIO = 5.0
#: sigma_row_cap / sigma_margin. Kept at the same tightness as the margins by
#: default: if "no negative final demand" and "hit beta exactly" genuinely
#: conflict, the (already more relaxable) emission target should give way
#: first, not physical feasibility.
DEFAULT_SIGMA_CAP_RATIO = 1.0


# ---------------------------------------------------------------------------
# Margins
# ---------------------------------------------------------------------------

def power_generation_sectors(table_type: str = "pxp") -> frozenset[str]:
    """
    Names of the power-generation technologies, under both label schemes.

    GEM-E3 names (`"Coal fired"`, for `region_ngfs` / `eu_macro` labels) and the
    EXIOBASE products they cover (`"Electricity by coal"`, for `world` / `full`).
    `Power Supply` (transmission, distribution, Electricity nec) is excluded: it
    is not a technology a buyer substitutes between. So are the CCS sectors,
    which have no EXIOBASE product of their own.
    """
    techs = [m for m in MAPPING
             if m.theme == "Power generation" and m.ngfs_sector != "Power Supply"
             and m.exio(table_type)]
    return frozenset([m.ngfs_sector for m in techs]
                     + [p for m in techs for p in m.exio(table_type)])


def power_mix_groups(labels: list[str], table_type: str = "pxp") -> list[np.ndarray]:
    """
    Row indices of the power technologies, one group per supplier region.

    Grouping by supplier region keeps the reallocation inside one country's (or
    bloc's) generation fleet: a French buyer's purchases of German electricity
    stay German, only their technology mix changes. At `world` resolution there
    is a single group. Groups with fewer than two rows have nothing to
    reallocate and are dropped.
    """
    names = power_generation_sectors(table_type)
    by_region: dict[str | None, list[int]] = {}
    for k, lab in enumerate(labels):
        region, sector = split_label(lab)
        if sector in names:
            by_region.setdefault(region, []).append(k)
    return [np.array(rows) for rows in by_region.values() if len(rows) > 1]


def mix_consistent_prior(
    Z0: np.ndarray,
    x_base: np.ndarray,
    x_target: np.ndarray,
    groups: list[np.ndarray],
) -> np.ndarray:
    """
    Z0 with each column's purchases inside every group reweighted by row growth.

    For column j and group G, the cells Z0_ij (i in G) are multiplied by
    g_i = x'_i / x_i and renormalised to their original column total, so
    `1' Z0m = 1' Z0` exactly and the money spent on each group is unchanged.
    A column whose group cells all have zero growth keeps its Z0 cells rather
    than losing the purchase (that would break v0). Rows with x_i = 0 carry no
    growth information and get g_i = 1.
    """
    x_base = np.asarray(x_base, dtype=float)
    x_target = np.asarray(x_target, dtype=float)
    growth = np.divide(x_target, x_base, out=np.ones_like(x_target),
                       where=x_base > 0)
    Zm = Z0.copy()
    for rows in groups:
        block = Z0[rows, :]
        weighted = block * growth[rows, None]
        total, total_w = block.sum(axis=0), weighted.sum(axis=0)
        ok = total_w > 0
        scale = np.divide(total, total_w, out=np.zeros_like(total), where=ok)
        Zm[rows, :] = np.where(ok[None, :], weighted * scale[None, :], block)
    return Zm


def unchanged_technology_margins(
    A: np.ndarray,
    x_target: np.ndarray,
    x_base: np.ndarray | None = None,
    mix_groups: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (u0, v0, Z0) -- the margins the scenario implies at base-year technology.

    Z0 doubles as the natural starting point a0 for KRAS: the entropy is then
    measured against the constant-technology projection rather than against a
    base-year matrix that no longer matches the scenario's output levels.

    With `mix_groups` (and `x_base`), Z0 is the power-mix-consistent prior of
    `mix_consistent_prior`: v0 is identical, u0 follows the scenario's mix.
    """
    Z0 = A * np.asarray(x_target, dtype=float)[None, :]
    if mix_groups:
        if x_base is None:
            raise ValueError("a mix-consistent prior needs x_base for the row growth")
        Z0 = mix_consistent_prior(Z0, x_base, x_target, mix_groups)
    return Z0.sum(axis=1), Z0.sum(axis=0), Z0


def embodied_emissions(s: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    b_j = sum_i s_i Z_ij -- emissions embodied in what sector j buys.

    The paper's eq. (7) quantity: linear in z, reads column j of Z, and is what
    the emission constraints act on. Not additive across sectors (a tonne from a
    power producer is counted again in every buyer's column), so it is only ever
    imposed and reported sector by sector.
    """
    return s @ Z


def region_mask(labels: list[str], regions: str | list[str]) -> np.ndarray:
    """
    Boolean mask selecting the columns that sit in `regions`.

    This is what "Europe decarbonises, the world does not" needs -- pass it as
    `build_constraints(..., target_mask=...)` so the untargeted columns get **no
    emission row at all**.

    Do not reach for `beta = 1` on the others instead: beta is measured against
    the *base year*, so beta=1 means "hold embodied emissions at their 2021
    level despite 2030 output", which is a hard and often tighter constraint,
    not an absence of one. Measured at eu_macro on SWUC 2030: 157 of the 180
    non-European columns would have to move by more than 10 %, one by a factor
    20, to satisfy a beta of 1.

    Raises
    ------
    ValueError if `regions` matches nothing -- silently returning an all-False
    mask would produce a converged run with no target at all.
    """
    wanted = {regions} if isinstance(regions, str) else set(regions)
    hit = np.array([split_label(lab)[0] in wanted for lab in labels])
    if not hit.any():
        present = sorted({split_label(lab)[0] for lab in labels} - {None})
        raise ValueError(
            f"no label is in region(s) {sorted(wanted)}; present: {present}"
        )
    return hit


def beta_by_region(
    labels: list[str],
    regions: str | list[str],
    beta: float,
    beta_other: float = 1.0,
) -> np.ndarray:
    """
    An ambition *vector* giving `regions` one ambition and the rest another.

    For "target these regions and leave the rest alone" you want
    `region_mask` + `target_mask`, not this with `beta_other=1.0` -- see the
    warning in `region_mask`. This function is for the case where the other
    regions genuinely get their own, different, non-trivial target.

    Takes `labels` rather than an `IOSystem`, keeping this module decoupled from
    the system object as the rest of it is.

    Note what any regional target allows: `build_constraints` sums eq. (7) over
    **all** suppliers, imports included, so a targeted sector can meet its
    target partly by buying from an untargeted region instead. That is carbon
    leakage rather than abatement, and it is deliberately left possible -- the
    alternative, dropping imports from the matrix, would hide it. `project()`
    reports it as `leakage_pp`.
    """
    out = np.full(len(labels), float(beta_other))
    out[region_mask(labels, regions)] = float(beta)
    return out


def feasible_band(
    s_target: np.ndarray, Z0: np.ndarray, v0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    (floor, ceiling) -- the embodied emissions column j can actually reach.

    With its input budget v0_j fixed and only its existing suppliers available
    (GRAS cannot open a new relationship), sector j's embodied emissions are a
    weighted average of supplier intensities times the budget:

        floor_j = v0_j * min{ s'_i : Z0_ij != 0 }
        ceil_j  = v0_j * max{ s'_i : Z0_ij != 0 }

    A target outside that band is out of reach whatever KRAS does, so it is
    worth knowing before blaming the solver -- the same role as
    `scope3/kras_scope3.py:168 feasible_floor`. Both bounds are open, not
    attained: reaching the floor exactly means sourcing 100 % from the single
    cleanest supplier.
    """
    n = Z0.shape[0]
    floor = np.zeros(n)
    ceiling = np.zeros(n)
    for j in range(n):
        used = Z0[:, j] != 0
        if used.any():
            floor[j] = v0[j] * float(s_target[used].min())
            ceiling[j] = v0[j] * float(s_target[used].max())
    return floor, ceiling


# ---------------------------------------------------------------------------
# The constraint set
# ---------------------------------------------------------------------------

@dataclass
class ConstraintSet:
    """Everything KRAS needs, plus what the report needs to interpret it."""

    G: sp.csr_matrix
    c: np.ndarray                    #: targets, emission rows in scaled units
    kind: list[str]                  #: one of {"row", "col", "row_cap", "emis"} per row
    sigma: np.ndarray

    Z0: np.ndarray                   #: unchanged-technology matrix, the a0
    u0: np.ndarray                   #: row margins
    v0: np.ndarray                   #: column margins
    b_base: np.ndarray               #: base-year embodied emissions, kg
    beta: np.ndarray                 #: ambition per column
    s_target: np.ndarray             #: s^{T,k} used in the constraint
    s_scale: float                   #: divisor that puts the exponents in [0,1]
    emis_cols: np.ndarray            #: columns that actually got a target
    b_floor: np.ndarray              #: lowest embodied emissions column j can reach
    b_ceiling: np.ndarray            #: highest, sourcing from its dirtiest supplier
    cap_rows: np.ndarray             #: rows actually capped (subset of the request)
    include_row: bool = False
    prior_rows: int = 0              #: rows reweighted by the power-mix prior (0 = off)
    dropped_zero_b: np.ndarray = field(default_factory=lambda: np.array([], int))
    dropped_no_support: np.ndarray = field(default_factory=lambda: np.array([], int))

    @property
    def n(self) -> int:
        return self.Z0.shape[0]

    def mask(self, kind: str) -> np.ndarray:
        return np.asarray(self.kind) == kind

    def emission_targets_kg(self) -> np.ndarray:
        """The emission targets back in kg, for reporting."""
        return self.c[self.mask("emis")] * self.s_scale


def build_sigma(
    kind: list[str],
    c: np.ndarray,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
    sigma_ratio: float = DEFAULT_SIGMA_RATIO,
    sigma_cap_ratio: float = DEFAULT_SIGMA_CAP_RATIO,
) -> np.ndarray:
    """
    Per-constraint uncertainty, as a fraction of the mean target of its kind.

    Margin targets are in M.EUR and emission targets in scaled kg, so a single
    absolute sigma would be meaningless across kinds. Expressing each as a
    fraction of its own scale makes `sigma_ratio` unitless and readable: it is
    the only knob that says which family of information is allowed to give way.
    """
    kind_arr = np.asarray(kind)
    sigma = np.zeros(len(c), dtype=float)
    for k, mult in (("row", 1.0), ("col", 1.0), ("row_cap", sigma_cap_ratio),
                    ("emis", sigma_ratio)):
        m = kind_arr == k
        if m.any():
            scale = float(np.mean(np.abs(c[m])))
            sigma[m] = sigma_frac * mult * scale
    return sigma


def build_constraints(
    Z: np.ndarray,
    A: np.ndarray,
    x_target: np.ndarray,
    s_base: np.ndarray,
    beta: float | np.ndarray = 1.0,
    s_target: np.ndarray | None = None,
    include_emissions: bool = True,
    include_row: bool = False,
    target_mask: np.ndarray | None = None,
    row_cap: np.ndarray | None = None,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
    sigma_ratio: float = DEFAULT_SIGMA_RATIO,
    sigma_cap_ratio: float = DEFAULT_SIGMA_CAP_RATIO,
    x_base: np.ndarray | None = None,
    mix_groups: list[np.ndarray] | None = None,
) -> ConstraintSet:
    """
    Assemble (G, c, kind, sigma) for one scenario / horizon / ambition.

    Parameters
    ----------
    Z, A, s_base
        The base-year system: flows, technical coefficients, direct intensities.
    x_target
        x^{T,k}, the scenario output vector.
    beta
        Ambition in (0, 1], scalar or per-sector. `beta=0.7` means "cut the
        emissions embodied in this sector's purchases to 70 % of their base-year
        level". `beta=1` holds them at the base-year level despite output growth
        -- still a real constraint, not a no-op.
    s_target
        s^{T,k}. Defaults to `s_base`: v1 does not project intensities, so the
        left-hand side of eq. (7) uses base-year factors. The hook exists so an
        intensity pathway can be dropped in without touching this builder.
    include_emissions
        False builds the margins alone -- the degenerate case that must return
        A' = A.
    include_row
        Also fix the row margins. **Off by default**: doing so pins the world
        total of embodied emissions and makes any uniform beta < 1 arithmetically
        impossible. See the module docstring.
    target_mask
        Boolean mask, one per column. Only masked columns get an emission row;
        the others are genuinely unconstrained. This is how a regional scenario
        ("Europe decarbonises, the world does not") is expressed -- see
        `region_mask`, and do not use `beta = 1` for that purpose.
    row_cap
        Boolean mask, one per row. Rows flagged True get an extra equality
        `sum_j Z'_ij = x_target_i`, the heuristic stand-in for `y'_i >= 0`
        described in the module docstring. Ignored (with a check) if
        `include_row=True`, since those rows are already fully pinned.
    x_base, mix_groups
        Optional power-mix-consistent prior (module docstring): row groups,
        typically from `power_mix_groups`, whose purchases are reweighted by
        each row's growth x'/x_base inside every column. v0 is unchanged.
    """
    n = Z.shape[0]
    x_target = np.asarray(x_target, dtype=float)
    s_base = np.asarray(s_base, dtype=float)
    s_target = s_base if s_target is None else np.asarray(s_target, dtype=float)
    # np.ndim, not np.isscalar: np.isscalar(np.float64(0.7)) is False.
    beta = (np.full(n, float(beta)) if np.ndim(beta) == 0
            else np.asarray(beta, dtype=float))
    if beta.shape != (n,):
        raise ValueError(f"beta must be scalar or ({n},), got {beta.shape}")

    u0, v0, Z0 = unchanged_technology_margins(A, x_target, x_base, mix_groups)
    b_base = embodied_emissions(s_base, Z)

    # Exponents in [0, 1]. CO2 intensities are non-negative, so max is the right
    # normaliser; it keeps r ** g bounded by r for every cell.
    s_scale = float(np.max(np.abs(s_target)))
    if s_scale == 0.0:
        raise ValueError("s_target is identically zero: no emission information")
    s_scaled = s_target / s_scale

    # Only cells that GRAS can actually move.
    nz = Z0.ravel() != 0
    cells = np.flatnonzero(nz)
    cell_i, cell_j = cells // n, cells % n

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []
    kind: list[str] = []
    c_list: list[np.ndarray] = []
    offset = 0

    if include_row:
        rows_list.append(offset + cell_i)
        cols_list.append(cells)
        data_list.append(np.ones(cells.size))
        kind += ["row"] * n
        c_list.append(u0)
        offset += n

    rows_list.append(offset + cell_j)
    cols_list.append(cells)
    data_list.append(np.ones(cells.size))
    kind += ["col"] * n
    c_list.append(v0)
    offset += n

    cap_rows = np.array([], dtype=int)
    if row_cap is not None and not include_row:
        requested = np.flatnonzero(np.asarray(row_cap, dtype=bool))
        # A row with no adjustable cell (Z0 row entirely zero) cannot be capped
        # at all -- GRAS cannot move a zero cell -- so it would sit as a dead
        # residual forever. Excluded here; in practice this cannot be the row
        # that triggered the cap, since triggering requires a positive,
        # therefore adjustable, row sum in the first place.
        has_support = np.zeros(n, dtype=bool)
        has_support[cell_i] = True
        cap_rows = requested[has_support[requested]]
        if cap_rows.size:
            row_of_cap = np.full(n, -1, dtype=int)
            row_of_cap[cap_rows] = np.arange(cap_rows.size)
            keep = row_of_cap[cell_i] >= 0
            rows_list.append(offset + row_of_cap[cell_i[keep]])
            cols_list.append(cells[keep])
            data_list.append(np.ones(int(keep.sum())))
            kind += ["row_cap"] * cap_rows.size
            c_list.append(x_target[cap_rows])
            offset += cap_rows.size

    emis_cols = np.array([], dtype=int)
    dropped_zero_b = np.array([], dtype=int)
    dropped_no_support = np.array([], dtype=int)

    if include_emissions:
        # A column is targetable only if it has emissions to cut and at least
        # one adjustable carbon-carrying supply relationship. Counted off the
        # sparse cell arrays rather than a `(s != 0) @ (Z0 != 0)` matmul, which
        # would materialise two n^2 float blocks for a boolean question.
        carbon_cells = cell_j[s_scaled[cell_i] != 0]
        support = np.bincount(carbon_cells, minlength=n) > 0
        positive_b = b_base > 0
        wanted = (np.ones(n, dtype=bool) if target_mask is None
                  else np.asarray(target_mask, dtype=bool))
        if wanted.shape != (n,):
            raise ValueError(f"target_mask must be ({n},), got {wanted.shape}")
        emis_cols = np.flatnonzero(positive_b & support & wanted)
        dropped_zero_b = np.flatnonzero(wanted & ~positive_b)
        dropped_no_support = np.flatnonzero(wanted & positive_b & ~support)

        if emis_cols.size:
            # row_of_col[j] = index of j inside emis_cols, or -1
            row_of_col = np.full(n, -1, dtype=int)
            row_of_col[emis_cols] = np.arange(emis_cols.size)

            keep = (s_scaled[cell_i] != 0) & (row_of_col[cell_j] >= 0)
            rows_list.append(offset + row_of_col[cell_j[keep]])
            cols_list.append(cells[keep])
            data_list.append(s_scaled[cell_i[keep]])

            kind += ["emis"] * emis_cols.size
            c_list.append(beta[emis_cols] * b_base[emis_cols] / s_scale)
            offset += emis_cols.size

    G = sp.csr_matrix(
        (np.concatenate(data_list),
         (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(len(kind), n * n),
    )
    c = np.concatenate(c_list)
    sigma = build_sigma(kind, c, sigma_frac, sigma_ratio, sigma_cap_ratio)
    b_floor, b_ceiling = feasible_band(s_target, Z0, v0)

    return ConstraintSet(
        G=G, c=c, kind=kind, sigma=sigma,
        Z0=Z0, u0=u0, v0=v0, b_base=b_base, beta=beta,
        s_target=s_target, s_scale=s_scale, emis_cols=emis_cols,
        b_floor=b_floor, b_ceiling=b_ceiling, cap_rows=cap_rows,
        include_row=include_row,
        prior_rows=int(sum(len(g) for g in mix_groups)) if mix_groups else 0,
        dropped_zero_b=dropped_zero_b, dropped_no_support=dropped_no_support,
    )
