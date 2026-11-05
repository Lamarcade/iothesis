# NGFS short-term (GEM-E3) ↔ EXIOBASE sector mapping — assumptions

Correspondence between the sectors of `Data/processed/production_growth.xlsx`
and the EXIOBASE 3.10.1 nomenclatures (200 `pxp` products / 163 `ixi`
industries), **anchored on the official GEM-E3 → NACE Rev. 2 concordance**
(Table 21 of the GEM-E3 documentation).

Code: `shortterm/st_exiobase_mapping.py`
Exports: `Data/mappings/st_geme3_exiobase_{pxp,ixi}.{csv,xlsx}`

Short-term counterpart of `Data/docs/MAPPING_HYPOTHESES.md` (NGFS Phase V long
term, which maps **variables**; this one maps **sectors**).

---

## 1. Correspondence chain

```
GEM-E3  --Table 21-->  NACE Rev.2 (class)  --exiobase_concordance.csv-->  EXIOBASE
  50 sectors            ~350 codes                division (C20)          200 / 163
```

The two links are not at the same granularity:

| Link | Granularity | Nature |
|---|---|---|
| Table 21 (GEM-E3 → NACE 2) | **class** (`20.14`) | many-to-many: 26 divisions claimed by ≥2 sectors |
| `exiobase_concordance.csv` | **division** (`C20`) | one division per EXIOBASE product |

The full chain therefore does not resolve mechanically: it yields **candidates**,
and the final assignment is made at class level. `--crosscheck` materialises
that gap (§6).

### Two GEM-E3 nomenclatures

Table 21 describes the **22-sector** version of the model. The NGFS short-term
version has **50**: it splits `Electricity supply` into generation technologies
and adds low-carbon activities. Hence a `source` field on every sector:

| `source` | n | Meaning |
|---|---:|---|
| `table21` | **21** | anchored on Table 21 (official NACE 2 codes) |
| `extension` | **29** | specific to the NGFS / GEM-E3-Power version; NACE 2 codes **proposed here**, not official |

Label reconciliation Table 21 → `production_growth.xlsx`:
`Electricity supply`→`Power Supply`, `Non ferrous metals`→`Non-ferrous metals`,
`Paper Products`→`Paper products, publishing`, `Non metallic minerals`→`Non-metallic minerals`,
`Transport equipment`→`Transport equipment (excluding EV)`,
`Transport (Air|Land|Water)`→`Air|Land|Water transport`.
`Electric Goods` has no counterpart: the NGFS version splits it into
`Computer, electronic and optical products` (C26) and `Other Equipment Goods` (C27/C28).

---

## 2. Cleaning Table 21

Two corrections applied to the PDF transcription (`GEME3_NACE2` in the module):

1. **Page numbers.** The bare integers `162`, `163`, `164` are page numbers
   injected into the text flow — one of them **in the middle of the
   `Transport equipment` row**, between `30.3` and `30.91`. No NACE 2 division
   exceeds 99, so the identification is unambiguous.
2. **Lost label.** The block `25.29 25.3 25.5 25.61 25.62 25.71 25.72 25.73
   25.91 25.92 25.93 25.99 33.2 38.21` appears orphaned between
   `Non metallic minerals` and `Electric Goods`. It is
   **`Fabricated Metal products`**: the 25.x codes are fabricated metal
   products, and it is the only GEM-E3 sector otherwise missing.

---

## 3. Scope of the NGFS source

`production_growth.xlsx` has 6 sheets but **only one is populated**:

| Sheet | Rows | Reason |
|---|---:|---|
| `GEME3_IIASA_2025_07_02` | 57,000 | the only source with monetary production |
| `CLIMACRED_…`, `direct_impacts_daps_…`, `direct_impacts_dire_…`, `EIRIN_…` | 0 | no `Production\|*` variable in a monetary unit |

Responsible filter: `_is_monetary_unit` + `_has_growth_label` in
[load_st.py](../../shortterm/load_st.py). **The mapping therefore covers GEM-E3 only.**

| Dimension | Value |
|---|---|
| Sectors | **50** |
| Unit | `billion US$2017` → growth in **% YoY** |
| Years | 2021–2030 (**2021 = NaN**, no base year) |
| Scenarios | 19: `Baseline`, `DIRE`, `SWUC`, `HWTP`, `DAPS_{AFR_R,ASIA,EUR,NAM,OCE,SAM}` (+ `_run1`) |
| Regions | 60 |

> ⚠️ In `production.xlsx` (levels), two spurious rows appear as sectors:
> `Coal YoY Growth` and `Coal fired YoY Growth`.
> `production_growth.xlsx` correctly excludes them (50 sectors, not 52).

---

## 4. Design principle: a partition on the EXIOBASE side

On the EXIOBASE side the mapping is a **strict partition**: every sector is
assigned to exactly one GEM-E3 sector. That is what allows a growth shock to be
applied to every column of `Z`/`A` with no double counting and no orphan.
Validated:

```
=== Validation pxp 2019 ===              === Validation ixi 2019 ===
EXIOBASE sectors        : 200            EXIOBASE sectors        : 163
GEM-E3 sectors mapped   : 50 (21 anchored on Table 21)            same
Output coverage (x)     : 100.00%        Output coverage (x)     : 100.00%
Unassigned EXIOBASE     : 0              Unassigned EXIOBASE     : 0
Assigned twice          : 0              Assigned twice          : 0
```

Table 21 itself is **not** a partition: 26 NACE 2 divisions are claimed by
several GEM-E3 sectors (`C20` by 5, `C28` by 4, `C10` by 3…). Every conflict is
arbitrated explicitly (§5.6).

### Quality flags

| Quality | n | Meaning |
|---|---:|---|
| `direct` | 18 | one-to-one or nearly so |
| `aggregate` | 17 | n EXIOBASE sectors → 1 GEM-E3 sector |
| `proxy` | 1 | `Biomass Solid` ↔ `Charcoal` |
| `overlap` | 14 | the GEM-E3 sector **does not exist** in EXIOBASE 3 |

---

## 5. What Table 21 changes

Nine corrections relative to a common-sense reading of the nomenclature. Seven
are **surprises** — Table 21 says the opposite of intuition.

### 5.1 All mining → `Non-metallic minerals` ⚠

The `Non metallic minerals` row contains **`7.1` (iron ore), `7.21`
(uranium/thorium), `7.29` (other non-ferrous ores)**, on top of
`8.11 8.12 8.91 8.93 8.99` (quarrying, chemical minerals, salt) and `6.1` `9.9`.

The `Ferrous metals` (24.x) and `Non ferrous metals` (24.4x, 25.1x) rows contain
**no extraction code at all**. The sector therefore reads as "processed
non-metallic minerals **plus non-energy mining**".

→ **9 EXIOBASE products moved** to `Non-metallic minerals`:
`Iron ores`, `Uranium and thorium ores (12)`, `Copper ores`, `Nickel ores`,
`Aluminium ores`, `Precious metal ores`, `Lead, zinc and tin ores`,
`Other non-ferrous metal ores`, `Chemical and fertilizer minerals, salt…`.

This is the heaviest correction: `Non-metallic minerals` goes from 9 to 18
products, `Ferrous metals` from 6 to 3, `Non-ferrous metals` from 16 to 10.

### 5.2 `Nuclear fuel` → `Oil`

`24.46` (processing of nuclear fuel) is explicitly in the `Oil` row. Combined
with §5.1 (`7.21` → Non-metallic minerals), the `Nuclear` sector reduces to
generation alone (`Electricity by nuclear`) — and becomes `direct` instead of
`aggregate`.

### 5.3 Steam and manufactured gases → `Gas`

The `Gas` row contains `35.21 35.22 35.23` (gas manufacture, distribution,
trade) and **`35.3` (steam and air conditioning)**, whereas `Electricity supply`
stops at `35.14`.

→ Moved to `Gas`: `Steam and hot water supply services` (35.3),
`Coke oven gas`, `Blast Furnace Gas`, `Oxygen Steel Furnace Gas`, `Biogas` (35.21).

Steelmaking gases therefore leave `Ferrous metals` despite being process
by-products: the repo concordance files them under `D35`, and 35.21 belongs to
`Gas`.

### 5.4 Waste and sewerage → `Non Market Services`

The `Non Market Services` row contains `37` (sewerage),
`38.11 38.12 38.22` (waste collection and disposal) and `39` (remediation).

→ **22 products moved** from `Market Services` to `Non Market Services`: the 20
waste treatments + `Secondary raw materials` + `Bottles for treatment`.
`Market Services` goes from 38 to 16 products, `Non Market Services` from 6 to 28.

Water (`36`) correctly stays in `Market Services`, as Table 21 indicates.

### 5.5 `Additives/Blending Components` → `Chemical Products`

The concordance files this product under `C20`, a division the `Oil` row does not
claim (it has `19.1 19.2 24.46 21.2 22.x 26.8 27.33 32.5 32.99`).

### 5.6 Arbitration of multi-claimed divisions

| Division | Claimed by (Table 21) | Assigned to | Rationale |
|---|---|---|---|
| `C10` `C12` `C16` | Agriculture, Non metallic minerals, Consumer Goods | **Consumer Goods** | Agriculture keeps primary production (1.x 2.x 3.x); all processing goes to Consumer Goods |
| `C19` | Coal, Oil, Chemical Products | **Coal** (coke) / **Oil** (refined) | see anomaly §7.1 |
| `C20` | Chemical, Paper, Non metallic minerals, Other Equipment, Consumer Goods | **Chemical Products** | classes 20.11-20.51; 20.13 (Non metallic minerals) and 20.14 (Paper/Consumer) have no distinct EXIOBASE product |
| `C24` | Oil, Ferrous, Non ferrous | by class | 24.46→Oil, 24.1-24.34+24.51/52→Ferrous, 24.41-45+24.53/54→Non-ferrous |
| `C25` | Non ferrous, Fabricated Metal, Other Equipment | **Fabricated Metal** | 25.11 25.12 25.21 (Non ferrous) have no separate EXIOBASE product |
| `C28` | Electric Goods, Transport eq., Other Equipment, Consumer Goods | **Other Equipment Goods** | `Machinery and equipment n.e.c. (29)` is the core of C28 |
| `D35` | Gas, Electricity supply | by class | 35.11→technologies, 35.12-14→Power Supply, 35.2/35.3→Gas |
| `E38` | Fabricated Metal, Non Market Services | **Non Market Services** | see anomaly §7.3 |
| `F43` | Construction, Market Services | **Construction** | see anomaly §7.2 |

---

## 6. The 35 cross-check mismatches — all accounted for

`python shortterm/st_exiobase_mapping.py --crosscheck` compares the retained
assignment against Table 21's candidates **at division level**. Result:
165 coherent, 35 mismatches, **0 unexplained**.

| Reason | n | Nature |
|---|---:|---|
| `NGFS sector absent from Table 21` | 21 | power technologies, Biofuels, Biomass Solid, Rubber/plastic, Computer/electronic, Warehousing, R&D — absent from the 22-sector version |
| `NACE 2 division absent from Table 21` | 12 | **gaps in the source** (§7.4) |
| `arbitrated at class level` | 2 | `Construction work (45)` and `Secondary construction material` (§7.2, §7.3) |

Without this triage, a naive division-level comparison reported **72
mismatches** — the overwhelming majority being granularity artefacts.

---

## 7. Table 21 anomalies — not followed

Four points where Table 21 is internally inconsistent or incomplete. Decisions
are recorded in each sector's `notes` field (and in the `justification` column
of the `sectors` sheet).

### 7.1 `Coal` gets `19.2` instead of `19.1`

`19.1` = coke oven products, `19.2` = refined petroleum. The `Coal` row lists
`19.2`, which makes no sense for a coal sector; `19.1` is assigned to `Oil` and
`Chemical Products`.

→ Read as a slip for `19.1`. `Coke Oven Coke`, `Gas Coke` and `Coal Tar` stay in
`Coal` (coherent solid fossil chain), the 18 refined products in `Oil`. A literal
reading would send them all to `Oil`.

### 7.2 Building construction (`41.1 41.2`) → `Market Services`

The `Market Services` row contains `41.1 41.2` (construction of buildings),
while `Construction` only has `42.x` (civil engineering) and `43.x` (specialised
works). Since EXIOBASE has a **single** construction product covering
F41+F42+F43, `Construction work (45)` is kept in `Construction`.

### 7.3 `38.21` inside `Fabricated Metal products`

`38.21` (treatment and disposal of non-hazardous waste) appears in the
fabricated-metal row — unrelated, and inconsistent with `38.11 38.12 38.22`
sitting under `Non Market Services`. Not followed: all waste goes to
`Non Market Services` (§5.4).

Corollary: `Secondary construction material for treatment` (aggregate recovery,
`38.32`) is listed **nowhere** in Table 21 → kept in `Construction`.

### 7.4 NACE 2 divisions absent from Table 21

Twelve divisions appear in no row. Assigned by judgement:

| Division | Content | → assigned to |
|---|---|---|
| `G47` | retail trade | Market Services |
| `I56` | accommodation and food service | Market Services |
| `M69` | legal and accounting | Market Services |
| `N77` | rental and leasing | Market Services |
| `R90` | arts and entertainment | Market Services |
| `S96` | personal services | Market Services |
| `P85` | education | Non Market Services |
| `Q86` | human health | Non Market Services |
| `S94` | membership organisations | Non Market Services |
| `T97` | households as employers | Non Market Services |
| `U99` | extraterritorial bodies | Non Market Services |

(The `Market Services` row lists only `46.11`–`46.18`, i.e. wholesale **on a fee
or contract basis** — neither `46.2`–`46.9` nor `47.x`. The source is visibly
truncated on trade.)

### 7.5 Other oddities noted, without effect

- `21.2` (pharmaceutical preparations) sits in the `Oil` row; `21.1` appears
  nowhere. No effect: EXIOBASE has no pharma product (§8).
- The `Oil` row contains `22.11`–`22.29` (rubber and plastics): a remnant of an
  earlier GEM-E3 aggregation. The NGFS 50-sector version has a dedicated
  `Rubber and plastic products` sector — that is the one followed.
- `Chemical Products` claims `1.64` (seed processing), with no corresponding
  EXIOBASE product.

---

## 8. The 14 `overlap` sectors — the limit of the mapping

GEM-E3 (NGFS version) breaks out activities that EXIOBASE 3 does not
distinguish. They get an **empty EXIOBASE list** plus an `overlaps_with`
pointer, so the partition stays intact.

| GEM-E3 sector | Subsumed in (EXIOBASE) | World 2025 output, bn$ |
|---|---|---:|
| `Basic pharmaceutical products` | `Chemicals nec` | **1,961** |
| `EV Transport Equipment` | `Motor vehicles … (34)` | 310 |
| `Equipment for PV panels` | `Electrical machinery … (31)` | 180 |
| `Equipment for wind power technology` | `Machinery and equipment n.e.c. (29)` | 74 |
| `Batteries` | `Electrical machinery … (31)` | 70 |
| `Hydrogen` | `Chemicals nec` | 0.79 |
| `Advanced Heating and Cooking Appliances` | `Machinery and equipment n.e.c. (29)` | 0.23 |
| `Advanced Electric Appliances` | `Electrical machinery … (31)` | 0.19 |
| `CSS Bio` (BECCS) | — | 0.009 |
| `CSS Gas` | — | 0.005 |
| `CCS coal` | — | 0.004 |
| `Clean Gas` | `Biogas` (→ `Gas`) | 8e-9 |
| `Equipment for CCS power technology` | `Machinery and equipment n.e.c. (29)` | 7e-5 |
| `CO2 Capture` | — | 2e-9 |

**Total ≈ 1.4 % of GEM-E3 world output 2025**, of which pharmaceuticals alone
account for 1.05 %. The other 13 add up to 0.34 % — but these are precisely the
sectors that **grow fastest under transition scenarios**.

Consequences:

1. **A shock on these sectors cannot be applied as such**: their dynamics stay
   buried in the parent sector's.
2. For `EV Transport Equipment` / `Transport equipment (excluding EV)`, the
   correct approach is to **recombine the two GEM-E3 trajectories weighted by
   their levels**, then apply the composite rate to `Motor vehicles … (34)`.
   Same for `Batteries` + `Equipment for PV/wind` → `(31)`/`(29)`.
3. Pharmaceuticals are the biggest sacrifice: applying `Chemical Products`
   growth to `Chemicals nec` contaminates ~2,000 bn$ of pharma whose dynamics
   are very different. Table 21 confirms there is no way out: it has no pharma
   sector either.

---

## 9. Regions

`ngfs_region_to_exiobase()` converts `"France - FRA"` into EXIOBASE ISO2, plus
aggregates: `World`→`'*'`, `EU27`→27 ISO2, `EU28`→EU27+`GB`.

**No EXIOBASE counterpart** (→ `None`): `Argentina - ARG` (inside `WL`),
`Saudi Arabia - SAU` (inside `WM`), and every non-EU aggregate (`Africa`, `Asia`,
`Asia (R5)`, `Euro Area`, `North America`, `Oceania`, `OECD & EU (R5)`,
`Reforming Economies (R5)`, `Latin America (R5)`, `Middle East & Africa (R5)`,
`Other (R5)`, `Rest of Europe`, `Rest of Energy Producing Countries`,
`Rest of World`, `South America`).

EXIOBASE only has 44 countries + 5 RoW zones: disaggregating R5 → EXIOBASE would
require a split key (output or GDP), not implemented.

---

## 10. `pxp` vs `ixi`

| | `pxp` (200) | `ixi` (163) |
|---|---|---|
| Nature | products | industries |
| Conceptual fit with GEM-E3 (activities) | moderate | **good** |
| Fit with the repo's NACE 2 concordance | **native** (it is pxp-based) | reconstructed by hand |
| Energy detail | **excellent** (8 coals, 18 refined, biofuels) | poor (1 coal, 1 refining, 0 biofuels) |
| `Biofuels`, `Biomass Solid` | mappable | **not mappable** (0 products) |
| Continuity with the long-term pipeline | ✅ | ❌ |

**Recommendation: `pxp`** — it is the nomenclature of `exiobase_concordance.csv`,
hence the only one where the GEM-E3 → NACE 2 → EXIOBASE chain is verifiable end
to end (`--crosscheck` runs on pxp only).

---

## 11. Limitations to keep in mind

1. **One uniform growth rate per block.** `growth_to_exiobase` applies the same
   % YoY to every EXIOBASE sector of a block. The 28 products of
   `Non Market Services` therefore move identically — a strong assumption for
   the wide blocks (`Non Market Services`, `Oil`, `Agriculture`,
   `Consumer Goods`, `Non-metallic minerals`).
2. **Growth ≠ level.** GEM-E3 is in constant `US$2017`, EXIOBASE in current M€,
   with different sector boundaries: only **rates** transfer.
3. **Output ≠ final demand.** A shock on `x` is not a shock on `Y`. To stay
   inside the Leontief framework: shock `Y` and let `x = L·y` respond, or go
   through a KRAS-type rebalancing ([optim/kras.py](../../optim/kras.py)).
4. **14 sectors not individually shockable** (§8).
5. **The repo concordance is at division level**: it cannot validate the
   class-level arbitrations (§5.6), which rest on a direct reading of Table 21.
6. **2021 is NaN**: the first usable year is **2022**.

---

## 12. Usage

### Command line

```powershell
# Summary: one row per GEM-E3 sector (provenance, NACE 2 count, EXIOBASE count)
python shortterm/st_exiobase_mapping.py --list --table-type pxp

# Full detail: NACE 2 codes, divisions, both pxp and ixi lists, justification
python shortterm/st_exiobase_mapping.py --detail "Non-metallic minerals"

# Partition check: names, duplicates, orphans, coverage
python shortterm/st_exiobase_mapping.py --validate --table-type ixi

# Table 21 confrontation, with a reason for every mismatch
python shortterm/st_exiobase_mapping.py --crosscheck

# Project a growth shock onto the 200 EXIOBASE sectors
python shortterm/st_exiobase_mapping.py --growth DIRE 2026 --region EU27

# Re-export
python shortterm/st_exiobase_mapping.py --export Data/mappings/st_geme3_exiobase_pxp.csv --table-type pxp
python shortterm/st_exiobase_mapping.py --export-workbook Data/mappings/st_geme3_exiobase_pxp.xlsx --table-type pxp
```

### Excel workbook layout

`st_geme3_exiobase_{pxp,ixi}.xlsx`, 5 sheets. The justification lives on
`sectors` (once per sector), not repeated on every join row:

| Sheet | Rows (pxp) | Content |
|---|---:|---|
| `sectors` | 50 | one row per GEM-E3 sector + **`justification`** column |
| `mapping` | 214 | one row per (GEM-E3, EXIOBASE) pair — the join table |
| `nace2` | 446 | one row per (GEM-E3, NACE 2 class code) + division |
| `table21` | 416 | Table 21 as transcribed, one row per (label, code) |
| `crosscheck` | 200 | Table 21 confrontation + `reason` for each mismatch |

### Python

Everything is exposed as native `list` / `dict`, no string parsing needed:

```python
from shortterm.st_exiobase_mapping import (
    GEME3_NACE2, get_map, nace2_map, nace2_table,
    sector_map, exio_to_ngfs, growth_to_exiobase, crosscheck_table21,
)

GEME3_NACE2["Coal"]                   # ['5.1', '5.2', '8.92', '9.9', '19.2']
nace2_map()["Gas"]                    # ['6.2', '9.1', '35.21', '35.22', '35.23', '35.3']

m = get_map("Non-metallic minerals")
m.nace2                               # list[str] of NACE 2 class codes
m.nace2_divisions()                   # ['B06', 'B07', 'B08', 'B09', 'C10', 'C13', 'C20', 'C23']
m.exio("pxp")                         # list[str], 18 EXIOBASE products
m.exio("ixi")                         # list[str], 18 EXIOBASE industries
m.notes                               # the justification string

sector_map("pxp")                     # dict[str, list[str]]  GEM-E3 -> EXIOBASE
exio_to_ngfs("pxp")                   # dict[str, str]        EXIOBASE -> GEM-E3 (partition)
nace2_table()                         # tidy DataFrame (446, 5), one row per code

# Apply a shock
g = growth_to_exiobase("DIRE", 2026, region="EU27")   # Series (200,) in % YoY
A_shocked = A.mul(1 + g.reindex(A.columns.get_level_values("sector")).values / 100)
```

Reading the CSV back, the `nace2` column is space-separated and splits cleanly:

```python
df = pd.read_csv("Data/mappings/st_geme3_exiobase_pxp.csv")
df["nace2_list"] = df["nace2"].fillna("").str.split()   # -> list[str] per row
```

---

## 13. Downstream use: projecting x

`shortterm/project_production.py` consumes this mapping to project the EXIOBASE
output vector `x` (base year 2021) with the GEM-E3 growth rates, and writes
`Data/production_evol/{scenario}/`.

```powershell
python shortterm/project_production.py --list
python shortterm/project_production.py --all           # the 5 curated scenarios
python shortterm/project_production.py --verify-daps
python shortterm/project_production.py --scenario DAPS_EUR   # any raw run too
```

### Scenario selection: 19 raw runs → 4 scenarios + reference

The file ships **19 raw run names**, which are not 19 scenarios.

| Output | NGFS scenario | Built from |
|---|---|---|
| `Baseline` | Reference (current policies) | `Baseline` |
| `DAPS` | Disasters and Policy Stagnation | **composite of 6 regional runs** |
| `DIRE` | Diverging Realities | `DIRE` |
| `HWTP` | Highway to Paris | `HWTP` |
| `SWUC` | Sudden Wake-up Call | `SWUC` |

**DAPS is split into 6 runs**, each applying its acute physical shock to one
world region. The runs are near-orthogonal — GDP 2030 vs Baseline:

| run \ region | EU27 | Asia | N.Am | S.Am | Africa | Oceania |
|---|---:|---:|---:|---:|---:|---:|
| `DAPS_EUR` | **−0.77** | −0.00 | −0.00 | 0.00 | −0.00 | 0.00 |
| `DAPS_ASIA` | 0.00 | **−1.65** | −0.00 | −0.00 | −0.03 | −0.02 |
| `DAPS_NAM` | −0.00 | −0.00 | **−0.66** | 0.00 | 0.00 | 0.00 |
| `DAPS_SAM` | −0.00 | −0.00 | −0.00 | **−1.09** | −0.00 | 0.00 |
| `DAPS_AFR_R` | −0.02 | −0.02 | −0.02 | −0.02 | **−0.81** | −0.01 |
| `DAPS_OCE` | −0.00 | −0.00 | −0.00 | 0.00 | 0.00 | **−0.76** |

Off-diagonal spillovers stay under 0.03 %, so the runs compose cleanly: the
full DAPS scenario gives **every region its own shock**. `DAPS_REGION_RUN`
encodes that assignment, derived by giving each of the 60 GEM-E3 regions the
run that hurts it most; all 60 are targeted by exactly one run.

`verify_daps_assignment()` recomputes the same assignment from the production
data instead of GDP: 58/60 agree. The two that differ, `Latin America (R5)` and
`OECD & EU (R5)`, are aggregates that the EXIOBASE region mapping never uses —
on the 46 GEM-E3 regions actually mobilised, agreement is total.

**`_run1` twins are ignored.** Every scenario has one, systematically slightly
milder (World GDP 2030: +0.475 % for `DIRE_run1`, +0.173 % `SWUC`, +0.105 %
`HWTP`, ~+0.005 % for DAPS). Nothing in the file distinguishes them — same
`Model`, same row count — and the unsuffixed names carry the official NGFS
scenario names. They remain reachable via `--scenario DIRE_run1`.

Resulting world trajectories (cumulative factor 2021→2030):

| Scenario | Coal | Coal fired | Oil Fired | Wind | PV |
|---|---:|---:|---:|---:|---:|
| `Baseline` | 0.823 | 0.592 | 0.227 | 2.057 | 4.808 |
| `DAPS` | 0.810 | 0.584 | 0.225 | 2.041 | 4.768 |
| `DIRE` | 0.762 | 0.524 | 0.224 | 2.656 | 5.473 |
| `HWTP` | 0.538 | 0.500 | 0.250 | 3.100 | 7.231 |
| `SWUC` | 0.532 | 0.513 | 0.248 | 3.035 | 7.202 |

The ordering is as expected: the two ambitious-transition scenarios halve coal
and multiply PV by ~7, DIRE sits in between, and DAPS tracks Baseline with a
small drag — it is Baseline plus regional physical damage, not a policy shift.
Regional impact of the DAPS composite on x (2030 vs Baseline): EU27 −0.67 %,
Asia −1.71 %, North America −0.64 %, South America −1.04 %, Africa −0.79 %,
Oceania −0.68 %.

### Region resolution

40 of the 49 EXIOBASE regions have a GEM-E3 country counterpart. The other 9
fall back to a GEM-E3 aggregate (`EXIO_REGION_FALLBACK`), so coverage is
complete:

| EXIOBASE | → GEM-E3 |
|---|---|
| `AU` | Oceania |
| `CH`, `NO`, `WE` | Rest of Europe |
| `TW`, `WA` | Asia |
| `WF` | Africa |
| `WL` | South America |
| `WM` | Rest of Energy Producing Countries |

`validate()` checks that all 49 resolve and that each target actually exists in
the data (`unmapped_regions`, `unknown_ngfs_regions`).

### Rate clipping — necessary, and mostly harmless

GEM-E3 rates are explosive on a few cells: **2,194 values above 100 %/yr and
557 above 1000 %/yr**, up to **+7.2e9 %** on `Biofuels`. They come from
near-zero GEM-E3 bases in small regions, where the matching EXIOBASE sector has
an ordinary base — so the rate is meaningless once transferred, and compounding
it over 9 years makes `x` diverge. World-level rates are sane, so the problem is
strictly regional.

Rates are clipped to **[-99 %, +100 %] per year** by default (`--clip`,
`--no-clip`) and every clipped cell is written to `diagnostics_clipped.csv`.

Across the 5 exported scenarios, 2,434 cells are clipped but **only 683 (28 %)
affect EXIOBASE** — the rest hit `overlap` sectors (`Clean Gas`, `CO2 Capture`,
`Batteries`, `Hydrogen`, …) that have no EXIOBASE target. The
`affects_exiobase` column marks the distinction.

### Validation obtained

World aggregate cumulative factors 2022-2030, EXIOBASE projection vs the GEM-E3
World reference:

| Block | EXIOBASE | GEM-E3 World |
|---|---:|---:|
| `Oil Fired` | 0.227 | 0.22 |
| `Coal fired` | 0.592 | 0.61 |
| `Coal` | 0.823 | 0.80 |
| `Wind` | 2.057 | 2.07 |
| `PV` | 4.808 | 4.79 |

The two do not have to coincide (different regional weights), and they agree
closely. Scenario contrast behaves as expected: `DIRE` vs `Baseline` gives
`Coal fired` −0.068, `Coal` −0.061, `Wind` +0.599, `PV` +0.665.

### What this output is not

The projection is **mechanical**: it scales `x` without restoring accounting
balance, so `x ≠ A·x + y` after the shock. The result is a **target output
vector**, to be fed into a KRAS-type rebalancing
([optim/kras.py](../../optim/kras.py)) or a Leontief solve on `Y` — not a
balanced IO table.

`A` is left unchanged, so `Z_t = A · diag(x_t)` is derivable on demand. A full
projected `Z` is not written: at 9800×9800 dense it would be ~768 MB per year,
~130 GB over all scenarios and years.

Reminder of the two resolution losses (§4, §8): every EXIOBASE sector inside one
GEM-E3 block receives the same rate, and the 14 `overlap` sectors have no
EXIOBASE target at all.

---

## 14. Charts

`shortterm/plot_production.py` renders the projections to `Figs/shortterm/`
(interactive HTML; `*.html` is gitignored).

```powershell
python shortterm/plot_production.py --all
python shortterm/plot_production.py --all --theme dark
python shortterm/plot_production.py --figure divergence
```

| File | Content |
|---|---|
| `overview.html` | World: production, GDP, GHG, GHG per unit of GDP |
| `regions_production.html` | 6 macro-regions × 5 scenarios — EXIOBASE x |
| `regions_gdp.html` | same layout — GEM-E3 GDP |
| `regions_emissions.html` | same layout — GEM-E3 GHG |
| `sectors.html` | 7 sector themes × 5 scenarios, world |
| `divergence.html` | 36 sectors × 4 scenarios and 6 regions × 4 scenarios, % vs Baseline |

### Everything is indexed to the base year = 100

Production is in current M€, GDP in constant bn US$2017, emissions in Mt CO2e —
three units on three bases. Plotting two of them against two y-scales would
invent a correlation that is not in the data, and §11.2 already establishes the
levels are not comparable. Indexing puts all three on one axis, where only the
shapes are compared. This is why GDP and production appear as two grids with an
identical layout rather than as one dual-axis chart.

### Aggregation

- **6 macro-regions**, reused from the DAPS regional runs, so the grouping that
  drives the scenario is the grouping the charts show.
- **7 sector themes** (`SectorMap.theme`), covering all 200 pxp products.

Caveat: GDP and emissions are read from GEM-E3's own aggregates, so "Europe" is
GEM-E3 `EU27` (strict) while the EXIOBASE Europe group also holds GB, CH, NO, RU
and RoW Europe. Indexing makes the two comparable in shape, not in perimeter.

### Colour

Baseline takes the de-emphasis gray (it is the reference); the 4 scenarios take
4 categorical hues in a fixed order, identical in every panel of every figure.

The order was computed with `shortterm/palette_check.py`, a Python port of the
dataviz skill's validator (this machine has no node). Enumerating the 4-hue
subsets of the 8 documented slots shows **none clears the all-pairs gate in
either mode** — 0 of 70 — so the palette is validated on the *adjacent*
pairlist a multi-line chart actually needs, and the mitigation the skill
mandates at 4 series is applied: each scenario also carries a distinct marker
symbol, so identity never rests on hue alone.

| | value | gate |
|---|---:|---|
| order | blue → green → violet → orange | |
| worst adjacent CVD ΔE | 26.5 light / 26.0 dark | ≥ 8 |
| worst adjacent normal-vision ΔE | 27.0 | ≥ 15 |
| separation from the Baseline gray | 9.8 light / 11.3 dark | |
| min contrast vs surface | 3.12:1 light / 3.52:1 dark | ≥ 3 |

The weak all-pairs pair is DAPS↔HWTP (ΔE 1.9 under protanopia, dark). That is
the acceptable one: those two scenarios sit far apart on every chart, whereas
HWTP and SWUC — whose lines nearly coincide (Coal 0.538 vs 0.532) — are
separated by ΔE ≥ 26.

The divergence heatmap uses the documented blue↔red diverging pair with a
neutral gray midpoint, capped at ±60 %: 141 of the 144 sector cells sit under
that (median 2.2 %), so an uncapped symmetric scale would spend most of the
negative arm on nothing. Every cell prints its own value, so the 3 capped cells
stay exactly readable and nothing is encoded by colour alone.

### Verified

Every indexed series starts at exactly 100; the world production index matches
`_summary.csv` to 1e-6; macro-regions and themes sum back to the world total at
machine precision; no NaN in any trace, in either theme.

**Not verified visually** — this environment has no browser and kaleido is not
installed, so the HTML was checked structurally, not eyeballed. Worth a look on
first open: label collisions on the 36-row heatmap, and the two empty cells in
the 3×3 sector grid.

---

*Generated 2026-07-30. Anchored on Table 21 "Mapping of GEME3 sectors to NACE 2
sectors" and validated against EXIOBASE 3.10.1 year 2019, both pxp and ixi:
complete partition, 100 % output coverage, 0 unexplained cross-check mismatch.*
