# v1 — projecting Z under a climate scenario: assumptions

What `v1/` does, what it assumes, and what it deliberately leaves out. Companion to
`MAPPING_ST_HYPOTHESES.md` (which documents where `x^{T,k}` comes from) and to the paper's
sections 3 and 4.

---

## 1. The chain

```
  EXIOBASE 2021                NGFS short-term (GEM-E3)
       Z, x, y, F                     growth rates
          |                                |
          |     shortterm/project_production.py
          |                                |
          |                          x^{T,k}  (Data/production_evol/)
          |                                |
          +--------------+-----------------+
                         |
                v1/constraints.py     ->  G z = c,  sigma
                         |
                    optim/kras.py     ->  Z'
                         |
                  v1/project.py       ->  A' = Z' x'^-1
                                          y' = (I - A') x'      eq. (8)
                                          E' = S' L' y'
```

Entry point:

```bash
python v1/project.py --scenario SWUC --year 2030 --beta 0.7 --resolution world
python v1/project.py --scenario SWUC --year 2030 --beta 0.7 --power-mix-prior   # H1b
python v1/test_v1.py                     # all checks, incl. EXIOBASE
python v1/test_v1.py --fast              # synthetic checks only, ~1 s
python v1/plot_v1.py --all               # 6 figures, Figs/v1/ — see §10

# Europe alone decarbonises, the rest of the world does not (§5.2, §5.3)
python v1/system.py --resolution eu_macro          # build the cache once
python v1/project.py --scenario SWUC --year 2030 --beta 0.7 \
                     --resolution eu_macro --beta-region Europe
```

---

## 2. Base year, table, resolution

| choice | value | why |
|---|---|---|
| base year `t0` | **2021** | The paper's t0, and the year the NGFS short-term rates compound from (`project_production.py:87`). Not the 2019 used in `optim/` and `scope3/`. |
| table | **pxp** (49 x 200, n = 9800) | What is already projected on disk. The paper's §3.1 states `N = 163, n = 7987`, which is **ixi** — an inconsistency to settle later, on the paper side or by regenerating the projections with `--table-type ixi`. |
| resolution | **`world`** by default (regions summed, n = 200) | KRAS balances n² variables and `optim/kras.py` cycles over constraints in Python. See §8. |

Four resolutions:

| name | shape | n (pxp) |
|---|---|---|
| `world` | regions summed, EXIOBASE sectors kept | 200 |
| `eu_macro` | Europe as one bloc + 5 macro regions, × 36 GEM-E3 sectors | 216 |
| `region_ngfs` | 49 regions × 36 GEM-E3 sectors | 1 764 |
| `full` | no aggregation | 9 800 |

`eu_macro` is for studying one region — see §5.2.

Summing regions keeps international supply chains *inside* the flows — French cars bought from
German steel stay an intermediate flow — but stops distinguishing where they happen.
`--resolution region_ngfs` (49 x 36, n = 1764) preserves the multi-regional setting of §3.1.

---

## 3. The constraint set

Stated on **Z**, not on `A`. The paper does the same (§3.3), and it is the numerically safe
choice: KRAS uses the coefficients of `G` as **exponents** (`a * r ** (sign(a) * g)`,
`optim/kras.py:223`). On Z the margin coefficients are exactly `1`; in the `b_ij = A_ij`
parametrisation of `optim/io_footprint.py` and `scope3/kras_scope3.py` every margin coefficient
is `x_j ~ 1e7` and has to be hand-normalised to avoid overflow.

| kind | row of G | target | coefficient | on by default |
|---|---|---|---|---|
| `col` | `sum_i Z'_ij` | `v0_j` | 1 | yes |
| `emis` | `sum_i s'_i Z'_ij` | `beta_j · b_j` | `s'_i / max(s')` | yes |
| `row` | `sum_j Z'_ij` | `u0_i` | 1 | **no**, see §5 |

### H1 — margins come from the unchanged-technology projection

```
Z0 = A · diag(x')          the no-structural-change counterfactual
u0 = Z0 · 1  = A x'
v0 = 1' · Z0 = x' · (1'A)
```

`v0_j = x'_j · sum_i A_ij` is exactly `x'_j - v'_j` under a **constant value-added share**, so
the column balance of eq. (1) is honoured **without loading any VA data**. The VA share is
*implied*, never an input — which is what "we don't deal with VA in v1" means in practice. The
`factor_inputs` extension is still loaded by nothing.

Rejected alternative: holding each sector's intermediate/final split constant,
`u0_i = x'_i (u_i/x_i)`. It makes the row and column totals diverge whenever growth is
heterogeneous, injecting a conflict into `Gz = c` that has nothing to do with climate. With the
unchanged-technology margins the two sets are **mutually consistent by construction**
(`|sum(u0) - sum(v0)| / sum(v0) ~ 1e-16`, checked), so the decarbonization target is the only
source of conflict and every deformation is attributable to it — the paper's claim in §2.6.

**Consequence and first test**: with `a0 = Z0` and no emission rows, KRAS converges on sweep 1
and returns `A' = A` to 1e-17.

### H1b — optional: a power-mix-consistent prior (`--power-mix-prior`)

`Z0` rescales each **column** by the buyer's growth, so every buyer keeps its 2021 supplier
shares. GEM-E3 projects each power technology separately, so `x'` carries a mix shift on the
**rows** that `Z0` ignores. SWUC 2030, world: coal power output ×0.51, wind ×3.04, solar PV ×7.21,
yet under `Z0` every technology's intermediate sales grow by the buyers' average ×1.17. Coal power
then sells 389 bn€ against 199 bn€ of output (`y0 = −190`), and most new wind and PV output finds no
intermediate buyer. Embodied emissions of the counterfactual come out **+16.6 %** above 2021 while
direct emissions fall.

The prior reweights, inside each column, the purchases of the power technologies of one supplier
region by each technology's own growth `x'_i / x_i`, at an unchanged total
(`constraints.mix_consistent_prior`, groups from `power_mix_groups`). Invariants, checked by tests
13 and 14: `v0` is unchanged to ~1e-15, rows outside the groups are untouched, and spend by
supplier region is kept, so the money import share is unchanged. The carbon import share does
change, so `leakage_pp` is measured against `import_carbon_share_cf_pct`. Without the prior that
share equals the base-year one (test 12), so default results are identical.

SWUC 2030, world, `beta = 0.7`:

| | `Z0` (default) | prior |
|---|---|---|
| embodied, counterfactual vs 2021 | +16.6 % | **−7.2 %** |
| cut KRAS has to deliver | 46.6 pp | **22.8 pp** |
| embodied, achieved | 16 768 Mt | 16 768 Mt (all 184 targets hit in both) |
| procurement effort, mean | 5.79 % | 5.51 % |
| `y0 < 0` a priori / rows capped | 39 / 36 | 37 / 35 |
| wind / PV sold to final demand, `y'` | 135 / 234 bn€ | 40 / 71 bn€ |
| final demand shift | 4.22 T€ | 3.96 T€ |

The end state of the targeted columns is the same, since they hit `beta · b` either way. What
changes is **attribution**: without the prior, half of the "restructuring" KRAS reports is the
scenario's own power mix. Procurement effort barely moves because power is a small share of input
budgets in money. The prior moves 5.7 Gt of counterfactual carbon but little money. Two limits:
the mix is weighted in euros, not MWh; and it does not stop carbon-intensive output from going to
final demand (gas power `y'` = 229 bn€ against 69 bn€ in 2021, see §5.1). Off by default so the
figures and numbers in this document stay reproducible.

### H2 — final demand: two distinct objects

The paper's closure needs `y` twice over, in two different roles. They must not be confused:

- `y0 = (I - A) x'` — **a priori**, base-year technology. It only *defines* the row margins
  (`u0 = x' - y0`, algebraically identical to `A x'`) and the counterfactual against which the
  final-demand shift is measured. It is a hypothesis, not data.
- `y' = (I - A') x'` — **ex post**, eq. (8), the deliverable. Since `A' = Z' diag(x')^-1`, this
  is just `y' = x' - Z' · 1`: no matrix product, exact.

`y_shift = y' - y0` is a first-class result: the final demand the decarbonization displaces at
constant output.

### H3 — intensities are held at the base year

`s^{T,k} = s^{t0}`. v1 does **not** project emission intensities; nothing in the repo does. The
hook exists (`project(..., s_target=...)`, `build_constraints(..., s_target=...)`) so an
intensity pathway drops in without touching the constraint builder. The right-hand side of
eq. (7) always uses base-year `s`, so `beta` is measured against the observed base-year
quantity as the paper specifies.

### H4 — beta and sigma are arbitrary

No calibration. §4.3 of the paper is not implemented.

- `beta_j ∈ (0,1]`, a scalar broadcast to all sectors (`--beta 0.7`) or a per-sector vector.
  `beta = 1` is **not** a no-op under growth: it means "hold embodied emissions at the
  base-year level despite producing more".
- `sigma` is expressed as a **fraction of the mean target of its own kind**
  (`DEFAULT_SIGMA_FRAC = 0.05`). Margin targets are in M.EUR and emission targets in scaled kg,
  so a single absolute sigma would be meaningless across kinds; this makes `sigma_ratio`
  unitless. `DEFAULT_SIGMA_RATIO = 5.0` > 1 means the emission target is the more relaxable
  constraint: KRAS holds the margins and lets the target slip where it cannot be met, so the
  shortfall *is* the result. Same reading as `scope3/kras_scope3.py:86-93`.

### H5 — normalisation of the emission coefficients

`s' / max(s')`, so every exponent lands in `[0, 1]` and `r ** g` stays bounded by `r`.
Checked by test 6.

### H6 — which columns get an emission target

Not all. Excluded, counted, and reported:

- `b_j = 0` — nothing carbon-intensive in the input mix. The target would be 0, which no
  multiplicative update can reach; the solver would drive the scaling factor towards zero and
  crush the column.
- no adjustable support — every supplier with `s'_i > 0` already at an exact zero flow. The
  residual would be untouchable and would block convergence.

---

## 4. GDP is deliberately absent

World GDP is **not** exactly a sum of Z cells: net exports outside the 49 EXIOBASE regions,
taxes and subsidies on products, and valuation differences all break the identity
`GDP = sum(Y)`. Writing `sum_ij Z'_ij = sum_i x'_i - GDP'` as a row of `G` would therefore
impose an accounting identity that does not hold, and KRAS would deform the flows to satisfy an
artefact. Production is the only scenario driver in v1.

Value added and bilateral trade are out of scope for the same reason plus the missing data —
see §7.

---

## 5. Why the row margins are free

Only the column margins are imposed. `--include-row` exists and is a diagnostic, not the model.
Two reasons, which are the same reason.

**It makes the target arithmetically impossible.** With the row margins fixed,

```
sum_j b'_j = sum_j sum_i s'_i Z'_ij = sum_i s'_i (sum_j Z'_ij) = sum_i s'_i u0_i
```

is a **constant**. The world total of embodied emissions cannot move by one gram, so a uniform
`beta < 1` cannot be met at all — KRAS can only shuffle emissions between value chains. Checked
numerically on the synthetic system: `sum_j b_j` and `sum_i s_i u0_i` agree to the last digit,
and with the rows fixed the worst column misses its target by 9.7 % while with the rows free
it hits it to 2.5e-15. This is the transpose of the remark at `scope3/kras_scope3.py:19-40`.

**It makes eq. (8) vacuous.** `sum_j Z'_ij = u0_i` is exactly `y'_i = y0_i`. Fixing the row
margins is fixing the answer the closure is supposed to produce. §3.3 of the paper derives the
*column* sums from output and value added and never claims the row sums; letting the rows go is
the reading that keeps the paper's own closure meaningful.

Economically it is also the right way round: if every buyer shifts away from carbon-intensive
suppliers, those suppliers **sell less**. Their sales are a result, not a constraint —
`sales_change_pct` reports who wins and who loses.

**What binds instead** is per-column and interpretable. With its budget `v0_j` fixed and only
its existing suppliers available (GRAS cannot open a new relationship), sector j's embodied
emissions live in

```
[ v0_j · min{s'_i : Z0_ij != 0} ,  v0_j · max{s'_i : Z0_ij != 0} ]
```

A target outside that band is unreachable **by arithmetic**, not by solver failure, and is
flagged `target_unreachable`. Same role as `scope3/kras_scope3.py:168 feasible_floor`.

### 5.1 Free rows have a cost: `y'` can go negative, and how it is handled

`x^{T,k}` does not itself preserve `x = Ax + y` — it is a mechanical rescaling (see
`shortterm/project_production.py`'s own docstring) — so even the a-priori `y0 = (I-A)x'` can be
negative wherever a sector's buyers grow faster than the sector itself. Observed on real
EXIOBASE data, SWUC 2030, `world` resolution: **39 of 200 sectors** have `y0 < 0` (e.g. Basic
iron and steel, `y0 = -107,601` M.EUR). With the rows free, KRAS's emission-driven reallocation
can shift this set in either direction — it can incidentally fix some rows, but it can also push
a *previously fine* row negative, since a supplier that gains a lot of substituted demand can end
up selling more than it produces. Left alone, the SWUC 2030 run came out with 31 of 200 sectors
still negative after balancing — an improvement over 39, but not a fix, and not guaranteed.

**Why this can't be a native inequality constraint.** `optim/kras.py` solves equalities with soft
relaxation (`Gz = c`, targets drift by at most `alpha · sigma` per touch, eq. 10) — there is no
inequality primitive, and adding one would mean modifying the shared solver `scope3/` and
`optim/exioptim.py` also depend on. Out of scope for v1.

**The approximation: iterative row capping.** Any row found negative after a solve is pinned to
the boundary, `sum_j Z'_ij = x_target_i` (i.e. `y'_i = 0`), a new `"row_cap"` kind in
`v1/constraints.py`, and the system is re-solved — because capping some rows can push others
across the line in turn, one pass is not enough. `v1/project.py::project` loops this up to
`max_cap_iters` (default 5); on SWUC 2030 it took 3 passes, capping 36 rows, to bring the count
from 39 to **0**. `sigma_cap_ratio` (default `1.0`, same tightness as the column margins) sets
where this sits in the priority order: tighter than the emission target (`sigma_ratio = 5`), so
if "no negative final demand" and "hit beta exactly" genuinely compete, the climate ambition
gives way first — and empirically, on SWUC 2030 it never had to: all 184 targeted sectors still
hit `beta` exactly with capping on.

`enforce_nonnegative_y=True` is the default (`--allow-negative-y` reproduces the original,
uncapped behaviour for comparison). **This is a heuristic, not a principled inequality solve**:
it finds *a* feasible-looking point by iteratively adding the tightest equality that removes the
current violation, not the minimal-distortion point a true inequality-constrained balancer would
find — the same standard of honesty as the `E' = s'.x'` pinning and the zero-preservation limit
below. A residual near zero (`~1e-9`-scale, KRAS's own absolute `tol`) does not count as negative;
`cap_tol` (default `1e-6`, a fraction of mean `|x_target|`) filters that noise from the real thing.

### 5.2 Studying one region: aggregate the rest, never slice it off

`eu_macro` answers "what if Europe alone decarbonises". The obvious shortcut — cut the European
rows and columns out of Z and work on that block — **breaks the accounting**. Measured on
EXIOBASE 2021 at `region_ngfs`: `x = Z·1 + y` then fails with a **78 % max relative residual**,
228 of 1152 sectors off by more than 10 %, because **2 191 740 M.EUR** of European intermediate
*sales to non-European buyers* (`Z_ER·1`) have nowhere left to go. The dropped mass matches
`Z_ER·1` to 1e-6, and the same figure comes out identically at `eu_macro` — asserted by check 10.

The classic single-region fix (exports → final demand, imports → value added) is arithmetically
exact but **destroys this model's mechanism**, so it was rejected:

- imports leave the matrix, so KRAS can no longer switch between a domestic and an imported
  supplier — 8.2 % of European intermediate purchases in aggregate, but **50 sectors source
  >30 % abroad**;
- **305 Mt CO₂, 11 % of the carbon embodied in what Europe buys, disappears from `b_j`**,
  modelling Europe as if its imports were carbon-free. That is the carbon-leakage blind spot
  turned into an assumption.

**What v1 does instead: aggregate the rest of the world.** Because this is an *aggregation*,
balance holds by construction — aggregation preserves sums, the property `world` and
`region_ngfs` already rely on. Verified: `eu_macro` totals for x, y, F and Z equal `world`'s to
1e-16, and it balances at 3.7e-12 (check 9). `Z_ER` and `Z_RE` survive as real cells, so
re-sourcing toward or away from imports stays possible and imported carbon stays inside `b_j`.

**Perimeter**: the bloc is named `"Europe"`, not `"EU"`, and that matters — it is the repo's
existing `DAPS_EUR` grouping, EU27 + GB + CH + NO + **RU** + RoW-Europe, 32 EXIOBASE regions.
Russian gas and oil are therefore *inside* the bloc and read as domestic flows, not imports. An
EU27-strict variant was considered and deliberately not taken.

### 5.3 Targeting one region — and why `beta = 1` is not "leave alone"

```python
from v1.constraints import region_mask
project(system, x_target, beta=0.7,
        target_mask=region_mask(system.labels, "Europe"))
```
or `python v1/project.py --resolution eu_macro --beta-region Europe --beta 0.7`.

`target_mask` gives the untargeted columns **no emission row at all**. Setting `beta = 1` on them
instead is a trap: `beta` is measured against the **base year**, so `beta = 1` means "hold
embodied emissions at their 2021 level despite 2030 output" — a hard, and for a growing region a
*tighter*, constraint than `beta < 1`. Measured at `eu_macro` on SWUC 2030: 157 of the 180
non-European columns would have to move more than 10 %, one by a factor 20. The first attempt at
this run did exactly that and ground for over ten minutes without converging; with `target_mask`
it converges in 62 sweeps. Check 11 asserts the two produce different constraint counts.

`beta_by_region` remains available for the different case where the other regions genuinely get
their own, non-trivial target.

**Leakage is left possible, and reported.** Eq. (7) sums over *all* suppliers, imports included,
so a targeted sector can meet its target partly by buying from an untargeted region — carbon
leakage, not abatement. Hiding it would mean dropping imports from the matrix, i.e. Option A
above. Instead `project()` reports it as `leakage_pp`, and the measured result is not the one to
expect — see §5.4.

### 5.4 Leakage falls as ambition rises

Running the five scenarios each with its own β on Europe (see §10) gives, for the 36 targeted
European columns, mean leakage in percentage points of embodied carbon shifted to foreign
suppliers:

| | Baseline β=0.95 | DAPS 0.90 | DIRE 0.80 | HWTP 0.70 | SWUC 0.65 |
|---|---|---|---|---|---|
| procurement effort (%) | 4.52 | 4.69 | 6.32 | 8.06 | 9.13 |
| leakage (pp) | **6.35** | 2.84 | 1.21 | 1.00 | **0.91** |
| imports vs unchanged tech (%) | +1.06 | −0.49 | −1.73 | −3.68 | −4.49 |

Effort rises monotonically with ambition, as expected. **Leakage does the opposite.** Across the
scenarios both `x^{T,k}` and β change, so that ladder is not a controlled experiment; holding
output fixed at SWUC's path and varying β alone reproduces the same ranking (8.75 → 5.18 → 1.00 →
1.04 → 0.91 pp), so the effect belongs to ambition, not to the output path.

The mechanism is consistent with the base-year data: European imports carry **11 % of the carbon
on 8.2 % of the spend**, so foreign supply is *more* carbon-intensive per euro than domestic. A
weak target can be shaved by nudging purchases abroad at the margin; a −35 % target cannot, and
has to restructure toward genuinely clean suppliers, most of which are domestic. Above β ≈ 0.8 the
adjustment reshores: intra-EU flows rise and imports fall by up to 4.5 % against the
unchanged-technology counterfactual.

**A note on the reference, because it is not obvious.** The imported *share* is identical whether
measured against the base year or against the counterfactual Z0: since `Z0[:, j] = A[:, j] x'_j`
and `Z[:, j] = A[:, j] x_j`, column *j* of Z0 is column *j* of Z times a **scalar**, so
composition cannot change. Verified at ~1e-13 and asserted by check 12. `leakage_pp` is therefore
well defined without choosing a reference — and a second share column would look like information
while carrying none.

---

## 6. Two properties of the production-oriented closure

### The aggregate footprint is pinned by the scenario, not by KRAS

Since `L' y' = L' (I - A') x' = x'`, eq. (4) collapses to

```
E' = S' L' y' = S' x'
```

whatever KRAS does to the flows. This is not a defect: it is what a production-oriented closure
*means*. Once output and intensities are given, the production-based total is determined, and
rebalancing can only move emissions between value chains, never remove them.

So `E'` is used as an **accounting check** (`checks["footprint_identity"]`, at machine
precision), and the quantities reported as results are the ones that actually move:

| column | meaning |
|---|---|
| `b_achieved` vs `b_target` | eq. (7) realised vs asked — where the constraint bound |
| `upstream_expost` | `sum_i m'_i Z'_ij`, full upstream; nonlinear, ex post only |
| `y_shift` | final demand displaced vs the unchanged-technology counterfactual |
| `procurement_effort` | share of each input budget that changed supplier (½ L1 of the sourcing shares) |
| `sales_change_pct` | who loses sales when buyers switch away |
| `relaxation` table | which targets had to give way, by kind — the paper's §7.4 material |
| `b_*_domestic` / `b_*_imported` | embodied carbon by supplier origin (domestic = the buyer's own region), for the base year, the counterfactual `cf` (= Z0) and the result |
| `import_carbon_share_*_pct`, `leakage_pp` | how much embodied carbon moved onto foreign suppliers — §5.3, §5.4 |
| `import_budget_share_*_pct` | the same split in money; it diverges from the carbon split (8.2 % of spend carries 11 % of the carbon) |

The origin columns are `NaN` at `world` resolution: with one region there is no such thing as an
import, and zeros would read as "nothing is imported", a different and false claim.

**This matters for the paper.** Reporting "the footprint fell by X %" under this closure would
be reporting `s'·x'`, i.e. the scenario's own assumption, not a result.

### Zeros stay zero

GRAS cannot move an exact zero, so no supply relationship absent from the base year can open
(§4.2, Limitations). `checks["frozen_zeros"]` counts them. At `world` resolution Z is nearly
dense and this barely bites; at `region_ngfs` it bites hard, and a sector whose only clean
suppliers are in regions it does not currently buy from cannot reach its target.

---

## 7. Not implemented in v1

Named, so each is a decision and not an omission.

- **Value added and trade constraints.** §3.3 announces rows of `G` for VA aggregates and
  bilateral submatrix totals. The `factor_inputs` extension is still loaded by nothing.
- **GDP.** §4 above.
- **`sigma` calibration** — §4.3 of the paper, empty there too.
- **`beta` from NGFS.** Hand-set. There is no derivation from scenario emission pathways yet.
- **Projected intensities `S^{T,k}`.** H3.
- **`ixi`.** Requires regenerating `Data/production_evol/` with `--table-type ixi`.
- **Full `n = 7987 / 9800`.** See §8.

---

## 8. The scale limit, and how it would be lifted

`optim/kras.py` cycles over constraints **one at a time in Python**. At n = 9800 that is
~19 600 constraint solves per sweep over 9 800-element supports, and `G` is ~2.3 GB in CSR.
`--resolution full` is wired and correct but will not run in acceptable time.

The fix is structural, not incremental, and the structure is favourable:

- row-constraint supports are **contiguous and disjoint** (`[i·n, (i+1)·n)`)
- column and emission supports are **strided and disjoint** (`i·n + j`, stride n)

so all `n` constraints of a block can be solved in **one vectorised Newton iteration** on an
`(n, n)` array instead of `n` scalar ones. A sweep becomes three vectorised solves rather than
`3n` scalar ones. That is a rewrite of `_solve_scaling` and the sweep loop, kept out of v1 on
purpose: it would have to come before any data.

---

## 9. Reuse and what was written fresh

| reused as-is | from |
|---|---|
| the KRAS solver | `optim/kras.py` (sparse `G` accepted unchanged) |
| EXIOBASE loading, `A`, `S`, `_safe_inv` | `io_pipeline.py` |
| `aggregate_matrix`, `aggregate_vector`, `CO2_STRESSORS` | `scope3/aggregate.py` — the two aggregators are already generic in `labels`/`order`; only `build_world_system` around them hardcodes region-summing, so passing `"region \| sector"` labels makes them region-preserving with no change |
| `x^{T,k}`, scenario names, `project_x`, `DAPS_REGION_RUN` | `shortterm/project_production.py` |
| `exio_to_ngfs`, `exio_region_map` | `shortterm/st_exiobase_mapping.py` |
| `THEME`, `FONT`, `SCENARIO_COLOR`, `SCENARIO_SYMBOL`, `DIVERGING`, `exio_sector_to_theme` | `shortterm/plot_production.py` |

Written fresh, with the reason:

- **`v1/constraints.py`** — `optim/io_footprint.py` builds a dense `np.zeros(n_var)` per row
  (impossible past a few hundred sectors) and parametrises on `b_ij` rather than `Z`.
  `scope3/kras_scope3.py` builds its triplets with O(n²) Python list appends. v1 assembles the
  triplets with `np.repeat`/`np.tile` and filters `G` to the sparsity pattern of `Z0` — a cell
  that starts at zero stays at zero, so its entry in `G` is pure memory.
- **`v1/resolution.py`** — `scope3/aggregate.py` discards the region level outright
  (`aggregate.py:135`); v1 needs it optional. Its 6-entry `MACRO_LABEL` dict duplicates
  `plot_production.py:128-132` rather than importing `exio_region_to_macro()`, because that module
  pulls in **plotly** and this one sits on v1's core path — the same refusal `v1/scenario.py`
  makes for the CSV reader.
- **`v1/system.py`, `v1/scenario.py`** — thin; `scenario.py` reimplements the six-line CSV
  reader of `plot_production.py:165` so that v1 does not import a plotly module to read a CSV.
- **`v1/plot_v1.py`** — chrome (`_style`, `_hbar`, light/dark handling, CLI shape) copied from
  `scope3/plot_scope3.py`, which states the same rule for itself: import the cross-cutting
  identity constants, redefine everything module-specific locally. No shared style module exists
  in the repo to import instead.

---

## 10. Figures

```bash
python v1/plot_v1.py --all                                    # 6 light HTML files
python v1/plot_v1.py --all --theme dark                        # 6 dark variants
python v1/plot_v1.py --figure row_margin_trap                  # single figure
```

Output: `Figs/v1/{name}.html` / `{name}_dark.html`. Five figures read a solved run (default
`--scenario SWUC --year 2030 --beta 0.7 --resolution world`, cache-first via
`load_projection`/`save_projection`); `row_margin_trap` is self-contained on the small synthetic
fixture `v1/test_v1.py::_synthetic_system` and takes no scenario parameters.

| figure | what it shows |
|---|---|
| `scenario_growth` | `x` → `x^{T,k}` by sector, before any rebalancing — grounds what the scenario alone does |
| `emission_targets` | base / achieved embodied emissions against the `beta` target (achieved lands on target to numerical precision) + procurement effort |
| `sales_shift` | who gains and loses intermediate sales — the direct picture of §5 ("rows are free, sales are an outcome") |
| `row_margin_trap` | test 7, visualised: worst relative miss on `beta`, rows fixed vs rows free (9.7 % vs 2.5e-15) — the figure that explains why the rows are free |
| `nonnegative_fix` | test 8 / §5.1, visualised: negative-final-demand sector count at each stage, `0 → 39 → 31 → 0` on SWUC 2030 |
| `structural_heatmap` | `Z0` vs `Z'` for the sectors with the largest procurement shift and their principal base-year suppliers — the paper's Figure 4, done on the Z-parametrised, column-margin formulation |

### The European figure set — `v1/plot_eu.py`

```bash
python v1/plot_eu.py --all                     # 7 light files in Figs/v1/eu/
python v1/plot_eu.py --all --theme dark
python v1/plot_eu.py --figure eu_flows --scenario HWTP
```

Solves `eu_macro` once per scenario, each with its own β on the European columns (via
`target_mask`, never β=1 elsewhere — §5.3), **σ identical across scenarios** so the only thing
varying is ambition. The ladder is ordered by the GEM-E3 2030 carbon price, verified against
`carbon_price_table("World")`: SWUC 163.1 > HWTP 151.6 > DIRE 115.3 > DAPS 54.0 > Baseline 53.7
US$2017/t. Note DAPS and Baseline sit 0.3 US$/t apart, so the gap between their β is wider than
the price warrants — the ladder is arbitrary in level, only its order is grounded.

| scenario | Baseline | DAPS | DIRE | HWTP | SWUC |
|---|---|---|---|---|---|
| β | 0.95 | 0.90 | 0.80 | 0.70 | 0.65 |

| figure | what it shows |
|---|---|
| `eu_flows_{scenario}` | the 37×37 European view — 36 sectors plus one aggregate import row and one aggregate export column — as flows after rebalancing, and as change against unchanged technology |
| `eu_scenario_ladder` | the five scenarios side by side: ambition applied, embodied carbon domestic vs imported, and the three flow channels against Z0 |
| `eu_sector_scenarios` | 36 sectors × 5 scenarios: procurement effort and leakage, so you can see which sectors bear the adjustment as ambition rises |

The 37×37 is a **view, not a system**: the projection still solves on the closed 216×216, and
`eu_view` asserts its two aggregates reproduce the border flows measured independently at
`region_ngfs` (1 422 109 imports / 2 191 740 exports M.EUR). That assertion is the guard against a
transposed or mis-masked view, which would otherwise render perfectly plausibly.

`nonnegative_fix` always solves fresh in both directions rather than using the cache: the cap
statistics it exists to show (`cap_rows_used`, `cap_iterations`) are never persisted by
`save_projection`, so a cache hit would silently read as 0/0. Every colour used is an
already-validated hex reused from an existing repo palette (`SCENARIO_COLOR`, `DIVERGING`,
`SCOPE_COLOR`'s aqua and blue) in a new pairing; `shortterm/palette_check.py` was re-run on the
two genuinely new adjacencies (`STATUS_COLOR`, `STAGE_COLOR`) before finalising them — see
`v1/plot_v1.py`'s module docstring.
