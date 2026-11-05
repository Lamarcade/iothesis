"""
Sector mapping between NGFS short-term (GEM-E3) and EXIOBASE, anchored on the
official GEM-E3 -> NACE Rev. 2 concordance (Table 21 of the GEM-E3
documentation).

Correspondence chain
--------------------
    GEM-E3  --Table 21-->  NACE Rev.2 (class)  --repo concordance-->  EXIOBASE
                                                (Data/mappings/exiobase_concordance.csv)

NGFS source: `Data/processed/production_growth.xlsx`, sheet
`GEME3_IIASA_2025_07_02` -- the only one of the 5 sources that publishes
monetary production. 50 sectors, unit `billion US$2017`.

EXIOBASE 3.10.1 target: 200 products (`pxp`) or 163 industries (`ixi`).

Two GEM-E3 nomenclatures
------------------------
Table 21 describes the **22-sector** version of the model. The NGFS short-term
version has **50**: it splits `Electricity supply` into generation technologies
and adds low-carbon activities (batteries, H2, CCS...). Hence a `source` field
on every sector:

    source='table21'   : anchored on Table 21 (21 sectors, official NACE 2 codes)
    source='extension' : specific to the NGFS / GEM-E3-Power version (29
                         sectors); NACE 2 codes proposed here, NOT official

Design principle
----------------
On the EXIOBASE side the mapping is a **partition**: every EXIOBASE sector is
assigned to exactly one GEM-E3 sector. This is what allows an NGFS growth rate
to be projected onto every column of Z / A with no double counting and no gap
(see `growth_to_exiobase`). Table 21 itself is many-to-many: 26 NACE 2
divisions are claimed by more than one GEM-E3 sector, and each conflict is
arbitrated explicitly in the `notes` field.

Quality flags
-------------
'direct'    : one-to-one or nearly so (Electricity by wind <-> Wind)
'aggregate' : n EXIOBASE sectors -> 1 GEM-E3 sector
'proxy'     : imperfect semantic match (Biomass Solid <-> Charcoal)
'overlap'   : the GEM-E3 sector **does not exist** in EXIOBASE 3 and is
              subsumed in a parent sector already assigned elsewhere
              (Batteries is inside Electrical machinery). Empty EXIOBASE list
              plus an `overlaps_with` pointer, so the partition stays intact.

Python-friendly access
----------------------
    GEME3_NACE2["Non metallic minerals"]   -> list[str] of NACE 2 class codes
    get_map("Non-metallic minerals").nace2 -> list[str]
    get_map("Non-metallic minerals").exio_pxp -> list[str]
    exio_to_ngfs("pxp")                    -> dict[str, str]  (partition)
    sector_map("pxp")                      -> dict[str, list[str]]
    nace2_table()                          -> tidy DataFrame, one row per code
    export_workbook(path)                  -> multi-sheet .xlsx

See `Data/docs/MAPPING_ST_HYPOTHESES.md` for the full write-up: assumptions,
corrections brought by Table 21, and Table 21's own internal anomalies.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_pipeline import load_x  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "Data" / "processed"
MAPPINGS_DIR = REPO_ROOT / "Data" / "mappings"
CONCORDANCE_CSV = MAPPINGS_DIR / "exiobase_concordance.csv"
GEME3_SHEET = "GEME3_IIASA_2025_07_02"


# ---------------------------------------------------------------------------
# Table 21: GEM-E3 (22 sectors) -> NACE Rev. 2
# ---------------------------------------------------------------------------
# Transcription of Table 21 "Mapping of GEME3 sectors to NACE 2 sectors".
#
#   the label "Fabricated Metal products" was lost in the page break: its
#      block of codes (25.29 ... 38.21) appears orphaned between
#      `Non metallic minerals` and `Electric Goods`. Restored here from the
#      codes themselves (25.x = fabricated metal products) and by elimination
#      (it is the only GEM-E3 sector otherwise missing).
#
# Values are plain `list[str]` of NACE Rev. 2 **class** codes.

GEME3_NACE2: dict[str, list[str]] = {
    "Agriculture": """
        1.11 1.12 1.13 1.14 1.15 1.16 1.19 1.21 1.22 1.23 1.24 1.25 1.26 1.27
        1.28 1.29 1.3 1.41 1.42 1.43 1.44 1.45 1.46 1.47 1.49 1.61 1.62 1.63
        1.64 1.7 2.1 2.2 2.3 2.4 3.1 3.11 3.12 3.21 3.22
        10.11 10.12 10.39 10.61 10.84 12 16.1""",
    "Coal": "5.1 5.2 8.92 9.9 19.2",
    "Crude Oil": "6.1 9.1",
    "Oil": """19.1 19.2 24.46 21.2 22.11 22.19 22.21 22.22 22.23 22.29
              26.8 27.33 32.5 32.99""",
    "Gas": "6.2 9.1 35.21 35.22 35.23 35.3",
    "Electricity supply": "35.11 35.12 35.13 35.14",
    "Ferrous metals": "24.1 24.2 24.31 24.32 24.33 24.34 24.51 24.52",
    "Non ferrous metals": "24.41 24.42 24.43 24.44 24.45 24.53 24.54 25.11 25.12 25.21",
    "Chemical Products": """1.64 19.1 20.11 20.12 20.13 20.14 20.15 20.16 20.17
                            20.2 20.3 20.41 20.42 20.51""",
    "Paper Products": """17.11 17.12 17.21 17.22 17.23 17.24 17.29
                         18.11 18.12 18.13 18.14 18.2 20.14 22.23""",
    "Non metallic minerals": """
        6.1 7.1 7.21 7.29 8.11 8.12 8.91 8.93 8.99 9.9 10.84 13.2 20.13
        23.11 23.12 23.13 23.14 23.19 23.2 23.31 23.32 23.41 23.42 23.43
        23.44 23.49""",
    "Fabricated Metal products": """25.29 25.3 25.5 25.61 25.62 25.71 25.72
                                    25.73 25.91 25.92 25.93 25.99 33.2 38.21""",
    "Electric Goods": """26.11 26.12 26.2 26.3 26.4 26.7 27.9
                         28.11 28.12 28.13 28.23 33.12 33.2""",
    "Transport equipment": """28.11 28.22 28.99 29.1 29.2 29.32 30.11 30.12
                              30.2 30.3 30.91 30.92 30.99
                              33.12 33.15 33.16 33.17 33.2""",
    "Other Equipment Goods": """
        13.99 15.12 16.29 17.29 20.51 22.19 22.23 22.29 25.4 25.73 25.99
        26.11 26.3 26.4 26.51 26.52 26.6 26.7 27.11 27.12 27.2 27.31 27.32
        28.23 28.99 29.32 30.3""",
    "Consumer Goods Industries": """
        10.11 10.12 10.13 10.2 10.31 10.32 10.39 10.41 10.42 10.51 10.52
        10.61 10.62 10.71 10.72 10.73 10.81 10.82 10.83 10.84 10.85 10.89
        11.01 11.02 11.03 11.04 11.05 11.06 11.07 12
        13.1 13.2 13.3 13.91 13.92 13.93 13.94 13.95 13.96 13.99
        14.11 14.12 14.13 14.14 14.19 14.2 14.31 14.39
        15.11 15.12 15.2 16.1 16.21 16.22 16.23 16.24 16.29
        17.22 20.14 20.6 22.19 22.29 28.23 31.01 31.09 32.3""",
    "Construction": """42.11 42.12 42.13 42.21 42.22 42.91 42.99
                       43.11 43.12 43.13 43.21 43.22 43.29 43.31""",
    "Transport (Air)": "51.1 51.21 51.22",
    "Transport (Land)": """49.1 49.2 49.31 49.32 49.39 49.41 49.42 49.5
                           52.1 52.21 52.22 52.23 52.24 52.29""",
    "Transport (Water)": "50.1 50.2 50.3 50.4",
    "Market Services": """
        36 41.1 41.2 43.91 43.99 45.11 45.19 45.2 45.31 45.32 45.4
        46.11 46.12 46.13 46.14 46.15 46.16 46.17 46.18 53.1 53.2
        58.21 58.29 61.1 61.2 61.3 61.9 62.01 62.02 62.03 62.09
        63.11 63.12 63.99 64.11 64.19 64.2 64.3 64.91 64.92 64.99
        65.11 65.12 65.2 65.3 66.11 66.12 66.19 66.21 66.22 66.29 66.3
        68.1 68.2 68.31 68.32 70.22 71.11 71.12 71.2
        72.11 72.19 72.2 73.11 73.12 73.2""",
    "Non Market Services": """
        37 38.11 38.12 38.21 38.22 39 59.11 59.12 59.13 59.14 59.2
        60.1 60.2 63.91 75 78.1 79.9 80.2 81.29 81.3 82.19
        84.11 84.12 84.13 84.21 84.22 84.23 84.24""",
}
GEME3_NACE2 = {k: v.split() for k, v in GEME3_NACE2.items()}

# Table 21 label -> label used in production_growth.xlsx
TABLE21_TO_NGFS: dict[str, str | None] = {
    "Electricity supply":    "Power Supply",
    "Non ferrous metals":    "Non-ferrous metals",
    "Paper Products":        "Paper products, publishing",
    "Non metallic minerals": "Non-metallic minerals",
    "Transport equipment":   "Transport equipment (excluding EV)",
    "Transport (Air)":       "Air transport",
    "Transport (Land)":      "Land transport",
    "Transport (Water)":     "Water transport",
    # Split into two sectors in the NGFS 50-sector version:
    "Electric Goods":        None,
}

# NACE 2: division ranges -> section letter
_DIVISION_SECTIONS = [
    (1, 3, "A"), (5, 9, "B"), (10, 33, "C"), (35, 35, "D"), (36, 39, "E"),
    (41, 43, "F"), (45, 47, "G"), (49, 53, "H"), (55, 56, "I"), (58, 63, "J"),
    (64, 66, "K"), (68, 68, "L"), (69, 75, "M"), (77, 82, "N"), (84, 84, "O"),
    (85, 85, "P"), (86, 88, "Q"), (90, 93, "R"), (94, 96, "S"), (97, 98, "T"),
    (99, 99, "U"),
]


def nace2_division(code: str) -> str:
    """'20.14' -> 'C20' (section+division code used by the repo concordance)."""
    d = int(str(code).split(".")[0])
    for lo, hi, letter in _DIVISION_SECTIONS:
        if lo <= d <= hi:
            return f"{letter}{d:02d}"
    raise ValueError(f"NACE 2 division out of range: {code!r}")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SectorMap:
    """
    One GEM-E3 (NGFS short-term) sector and its EXIOBASE counterpart.

    Attributes
    ----------
    ngfs_sector   : exact value of the `Secteur` column in production_growth.xlsx
    theme         : readable grouping (Energy supply, Power generation, ...)
    exio_pxp      : list[str] of EXIOBASE pxp product names (exact)
    exio_ixi      : list[str] of EXIOBASE ixi industry names (exact)
    quality       : 'direct' | 'aggregate' | 'proxy' | 'overlap'
    nace2         : list[str] of NACE Rev. 2 class codes -- from Table 21 when
                    source='table21', proposed here when source='extension'
    source        : 'table21' | 'extension'
    table21_name  : Table 21 label when it differs from the NGFS label
    overlaps_with : for quality='overlap', the GEM-E3 sector that subsumes this
                    one on the EXIOBASE side
    notes         : justification of the EXIOBASE assignment and caveats
    """
    ngfs_sector:   str
    theme:         str
    exio_pxp:      list[str]
    exio_ixi:      list[str]
    quality:       str
    nace2:         list[str] = field(default_factory=list)
    source:        str = "extension"
    table21_name:  str | None = None
    overlaps_with: str | None = None
    notes:         str = ""

    def nace2_divisions(self) -> list[str]:
        """Sorted unique NACE 2 divisions covered, e.g. ['B07', 'C23']."""
        return sorted({nace2_division(c) for c in self.nace2})

    def exio(self, table_type: str = "pxp") -> list[str]:
        """EXIOBASE sector names for the requested table type."""
        if table_type not in ("pxp", "ixi"):
            raise ValueError(f"table_type must be 'pxp' or 'ixi', got {table_type!r}")
        return self.exio_pxp if table_type == "pxp" else self.exio_ixi


def _t21(name: str) -> list[str]:
    """Table 21 NACE 2 codes for a Table 21 label."""
    return GEME3_NACE2[name]


# ---------------------------------------------------------------------------
# Reused EXIOBASE blocks
# ---------------------------------------------------------------------------

_PXP_CROPS_LIVESTOCK = [
    "Paddy rice", "Wheat", "Cereal grains nec", "Vegetables, fruit, nuts",
    "Oil seeds", "Sugar cane, sugar beet", "Plant-based fibers", "Crops nec",
    "Cattle", "Pigs", "Poultry", "Meat animals nec", "Animal products nec",
    "Raw milk", "Wool, silk-worm cocoons",
    "Manure (conventional treatment)", "Manure (biogas treatment)",
    "Products of forestry, logging and related services (02)",
    "Fish and other fishing products; services incidental of fishing (05)",
]
_IXI_CROPS_LIVESTOCK = [
    "Cultivation of paddy rice", "Cultivation of wheat",
    "Cultivation of cereal grains nec", "Cultivation of vegetables, fruit, nuts",
    "Cultivation of oil seeds", "Cultivation of sugar cane, sugar beet",
    "Cultivation of plant-based fibers", "Cultivation of crops nec",
    "Cattle farming", "Pigs farming", "Poultry farming", "Meat animals nec",
    "Animal products nec", "Raw milk", "Wool, silk-worm cocoons",
    "Manure treatment (conventional), storage and land application",
    "Manure treatment (biogas), storage and land application",
    "Forestry, logging and related service activities (02)",
    "Fishing, operating of fish hatcheries and fish farms; service activities incidental to fishing (05)",
]

_PXP_REFINED_OIL = [
    "Motor Gasoline", "Aviation Gasoline", "Gasoline Type Jet Fuel",
    "Kerosene Type Jet Fuel", "Kerosene", "Gas/Diesel Oil", "Heavy Fuel Oil",
    "Refinery Gas", "Liquefied Petroleum Gases (LPG)", "Refinery Feedstocks",
    "Ethane", "Naphtha", "White Spirit & SBP", "Lubricants", "Bitumen",
    "Paraffin Waxes", "Petroleum Coke", "Non-specified Petroleum Products",
]

_PXP_WASTE = [
    "Food waste for treatment: incineration",
    "Paper waste for treatment: incineration",
    "Plastic waste for treatment: incineration",
    "Intert/metal waste for treatment: incineration",
    "Textiles waste for treatment: incineration",
    "Wood waste for treatment: incineration",
    "Oil/hazardous waste for treatment: incineration",
    "Food waste for treatment: biogasification and land application",
    "Paper waste for treatment: biogasification and land application",
    "Sewage sludge for treatment: biogasification and land application",
    "Food waste for treatment: composting and land application",
    "Paper and wood waste for treatment: composting and land application",
    "Food waste for treatment: waste water treatment",
    "Other waste for treatment: waste water treatment",
    "Food waste for treatment: landfill",
    "Paper for treatment: landfill",
    "Plastic waste for treatment: landfill",
    "Inert/metal/hazardous waste for treatment: landfill",
    "Textiles waste for treatment: landfill",
    "Wood waste for treatment: landfill",
]
_IXI_WASTE = [
    "Incineration of waste: Food", "Incineration of waste: Paper",
    "Incineration of waste: Plastic",
    "Incineration of waste: Metals and Inert materials",
    "Incineration of waste: Textiles", "Incineration of waste: Wood",
    "Incineration of waste: Oil/Hazardous waste",
    "Biogasification of food waste, incl. land application",
    "Biogasification of paper, incl. land application",
    "Biogasification of sewage slugde, incl. land application",
    "Composting of food waste, incl. land application",
    "Composting of paper and wood, incl. land application",
    "Waste water treatment, food", "Waste water treatment, other",
    "Landfill of waste: Food", "Landfill of waste: Paper",
    "Landfill of waste: Plastic", "Landfill of waste: Inert/metal/hazardous",
    "Landfill of waste: Textiles", "Landfill of waste: Wood",
]


# ---------------------------------------------------------------------------
# MAPPING
# ---------------------------------------------------------------------------

MAPPING: list[SectorMap] = [

    # =====================================================================
    # 1) Energy supply
    # =====================================================================
    SectorMap(
        "Coal", "Energy supply",
        ["Anthracite", "Coking Coal", "Other Bituminous Coal",
         "Sub-Bituminous Coal", "Patent Fuel", "Lignite/Brown Coal",
         "BKB/Peat Briquettes", "Peat",
         "Coke Oven Coke", "Gas Coke", "Coal Tar"],
        ["Mining of coal and lignite; extraction of peat (10)",
         "Manufacture of coke oven products"],
        "aggregate",
        nace2=_t21("Coal"), source="table21",
        notes="Table 21: 5.1 5.2 (hard coal, lignite) + 8.92 (peat) + 9.9 + "
              "19.2. ANOMALY: 19.2 is refined petroleum, whereas coke is 19.1 "
              "(which Table 21 assigns to Oil and Chemical Products). Read as "
              "a slip for 19.1, so coke-oven products stay here and the solid "
              "fossil chain remains coherent. A literal reading would move "
              "them to Oil. Coke oven GAS goes to 'Gas' instead (35.21)."),

    SectorMap(
        "Crude Oil", "Energy supply",
        ["Crude petroleum and services related to crude oil extraction, excluding surveying"],
        ["Extraction of crude petroleum and services related to crude oil extraction, excluding surveying"],
        "direct",
        nace2=_t21("Crude Oil"), source="table21",
        notes="Table 21: 6.1 extraction + 9.1 support services. Refining sits "
              "in 'Oil'."),

    SectorMap(
        "Oil", "Energy supply",
        [*_PXP_REFINED_OIL, "Nuclear fuel"],
        ["Petroleum Refinery", "Processing of nuclear fuel"],
        "aggregate",
        nace2=_t21("Oil"), source="table21",
        notes="Refined products (19.2). Table 21 also puts 24.46 (processing "
              "of nuclear fuel) in this row, so 'Nuclear fuel' belongs here "
              "and NOT in 'Nuclear'. The 22.x (rubber and plastics) and "
              "26.8 / 27.33 / 32.5 / 32.99 codes in the Oil row come from an "
              "older, coarser GEM-E3 aggregation: the NGFS 50-sector version "
              "has a dedicated 'Rubber and plastic products' sector."),

    SectorMap(
        "Gas", "Energy supply",
        ["Natural gas and services related to natural gas extraction, excluding surveying",
         "Natural Gas Liquids", "Other Hydrocarbons",
         "Distribution services of gaseous fuels through mains",
         "Gas Works Gas", "Coke oven gas", "Blast Furnace Gas",
         "Oxygen Steel Furnace Gas", "Biogas",
         "Steam and hot water supply services"],
        ["Extraction of natural gas and services related to natural gas extraction, excluding surveying",
         "Extraction, liquefaction, and regasification of other petroleum and gaseous materials",
         "Manufacture of gas; distribution of gaseous fuels through mains",
         "Steam and hot water supply"],
        "aggregate",
        nace2=_t21("Gas"), source="table21",
        notes="Table 21: 6.2 + 9.1 (extraction) + 35.21 35.22 35.23 (gas "
              "manufacture, distribution, trade) + 35.3 (steam and air "
              "conditioning). Two consequences: every manufactured gas "
              "(coke oven gas, blast furnace gas, oxygen steel furnace gas, "
              "biogas) falls under 35.21 and lands here; and district "
              "steam/heat (35.3) belongs here rather than in 'Power Supply', "
              "whose Table 21 row stops at 35.14."),

    SectorMap(
        "Biofuels", "Energy supply",
        ["Biogasoline", "Biodiesels", "Other Liquid Biofuels"],
        [],
        "direct",
        nace2=["20.14", "20.59"], source="extension",
        notes="Not in Table 21 (sector specific to the NGFS version). "
              "pxp only: ixi has no separate biofuel industry, it is merged "
              "into 'Petroleum Refinery'."),

    SectorMap(
        "Biomass Solid", "Energy supply",
        ["Charcoal"],
        [],
        "proxy",
        nace2=["20.14"], source="extension",
        notes="Not in Table 21. GEM-E3 'Biomass Solid' means fuelwood, "
              "pellets and residues; EXIOBASE only offers 'Charcoal' (which "
              "the repo concordance files under C20). Forestry stays in "
              "'Agriculture' (Table 21 codes 2.1-2.4) to avoid double "
              "counting. Very narrow proxy: use for relative growth only, "
              "never for levels."),

    SectorMap(
        "Clean Gas", "Energy supply", [], [], "overlap",
        overlaps_with="Gas",
        nace2=["35.21"], source="extension",
        notes="Synthetic gas / e-methane. Absent from both Table 21 and "
              "EXIOBASE 3. 'Biogas' falls under 35.21 and is therefore "
              "assigned to 'Gas'. GEM-E3 output is ~1e-9 bn$ world 2025, so "
              "the loss of resolution is immaterial."),

    SectorMap(
        "Hydrogen", "Energy supply", [], [], "overlap",
        overlaps_with="Chemical Products",
        nace2=["20.11"], source="extension",
        notes="20.11 (industrial gases) is inside EXIOBASE 'Chemicals nec'. "
              "0.79 bn$ world 2025, not separable."),

    # =====================================================================
    # 2) Power generation -- NGFS split of NACE 35.11
    # =====================================================================
    SectorMap(
        "Coal fired", "Power generation",
        ["Electricity by coal"], ["Production of electricity by coal"],
        "direct", nace2=["35.11"], source="extension"),

    SectorMap(
        "Gas Fired", "Power generation",
        ["Electricity by gas"], ["Production of electricity by gas"],
        "direct", nace2=["35.11"], source="extension"),

    SectorMap(
        "Oil Fired", "Power generation",
        ["Electricity by petroleum and other oil derivatives"],
        ["Production of electricity by petroleum and other oil derivatives"],
        "direct", nace2=["35.11"], source="extension"),

    SectorMap(
        "Nuclear", "Power generation",
        ["Electricity by nuclear"], ["Production of electricity by nuclear"],
        "direct", nace2=["35.11"], source="extension",
        notes="Generation only. Table 21 puts the rest of the fuel cycle "
              "elsewhere: uranium ore (7.21) -> 'Non-metallic minerals', "
              "fuel fabrication (24.46) -> 'Oil'."),

    SectorMap(
        "Hydro electric", "Power generation",
        ["Electricity by hydro", "Electricity by tide, wave, ocean"],
        ["Production of electricity by hydro",
         "Production of electricity by tide, wave, ocean"],
        "direct", nace2=["35.11"], source="extension",
        notes="Tidal/wave attached to hydro since GEM-E3 has no ocean "
              "technology. Negligible weight."),

    SectorMap(
        "Wind", "Power generation",
        ["Electricity by wind"], ["Production of electricity by wind"],
        "direct", nace2=["35.11"], source="extension"),

    SectorMap(
        "PV", "Power generation",
        ["Electricity by solar photovoltaic", "Electricity by solar thermal"],
        ["Production of electricity by solar photovoltaic",
         "Production of electricity by solar thermal"],
        "direct", nace2=["35.11"], source="extension",
        notes="EXIOBASE separates PV from concentrated solar thermal; GEM-E3 "
              "has a single solar technology here, so both are summed."),

    SectorMap(
        "Biomass", "Power generation",
        ["Electricity by biomass and waste"],
        ["Production of electricity by biomass and waste"],
        "direct", nace2=["35.11"], source="extension",
        notes="EXIOBASE lumps biomass and waste-to-energy together."),

    SectorMap(
        "Geothermal", "Power generation",
        ["Electricity by Geothermal"],
        ["Production of electricity by Geothermal"],
        "direct", nace2=["35.11"], source="extension"),

    SectorMap(
        "Power Supply", "Power generation",
        ["Transmission services of electricity",
         "Distribution and trade services of electricity",
         "Electricity nec"],
        ["Transmission of electricity", "Distribution and trade of electricity",
         "Production of electricity nec"],
        "aggregate",
        nace2=["35.12", "35.13", "35.14"], source="table21",
        table21_name="Electricity supply",
        notes="Table 21 'Electricity supply' spans 35.11-35.14. The NGFS "
              "version splits 35.11 (generation) into technologies, so only "
              "35.12 transmission, 35.13 distribution and 35.14 trade remain "
              "here. CAUTION: 'Power Supply' is NOT the sum of the generation "
              "technologies -- 3216 vs 2056 bn$ world 2025 baseline. It is "
              "the downstream supply sector that buys generation and adds "
              "network and retail margins. 'Electricity nec' (residual "
              "generation) is attached here for lack of an 'other' GEM-E3 "
              "technology. Steam (35.3) sits in 'Gas'."),

    SectorMap(
        "CCS coal", "Power generation", [], [], "overlap",
        overlaps_with="Coal fired", nace2=["35.11"], source="extension",
        notes="No CCS plant exists in EXIOBASE 3.10.1."),
    SectorMap(
        "CSS Gas", "Power generation", [], [], "overlap",
        overlaps_with="Gas Fired", nace2=["35.11"], source="extension",
        notes="NGFS spelling 'CSS' (means CCS gas). Absent from EXIOBASE."),
    SectorMap(
        "CSS Bio", "Power generation", [], [], "overlap",
        overlaps_with="Biomass", nace2=["35.11"], source="extension",
        notes="BECCS. Absent from EXIOBASE."),
    SectorMap(
        "CO2 Capture", "Power generation", [], [], "overlap",
        overlaps_with="Chemical Products", source="extension",
        notes="CO2 capture service: no dedicated NACE 2 code, absent from "
              "EXIOBASE 3."),

    # =====================================================================
    # 3) Heavy industry
    # =====================================================================
    SectorMap(
        "Ferrous metals", "Heavy industry",
        ["Basic iron and steel and of ferro-alloys and first products thereof",
         "Secondary steel for treatment, Re-processing of secondary steel into new steel",
         "Foundry work services"],
        ["Manufacture of basic iron and steel and of ferro-alloys and first products thereof",
         "Re-processing of secondary steel into new steel",
         "Casting of metals"],
        "aggregate",
        nace2=_t21("Ferrous metals"), source="table21",
        notes="Table 21: 24.1-24.34 (steelmaking) + 24.51 24.52 (ferrous "
              "casting). Iron ore mining (7.1) is NOT here -- Table 21 "
              "assigns it to 'Non-metallic minerals'. Steelmaking gases "
              "(35.21) go to 'Gas'."),

    SectorMap(
        "Non-ferrous metals", "Heavy industry",
        ["Precious metals",
         "Secondary preciuos metals for treatment, Re-processing of secondary preciuos metals into new preciuos metals",
         "Aluminium and aluminium products",
         "Secondary aluminium for treatment, Re-processing of secondary aluminium into new aluminium",
         "Lead, zinc and tin and products thereof",
         "Secondary lead for treatment, Re-processing of secondary lead into new lead",
         "Copper products",
         "Secondary copper for treatment, Re-processing of secondary copper into new copper",
         "Other non-ferrous metal products",
         "Secondary other non-ferrous metals for treatment, Re-processing of secondary other non-ferrous metals into new other non-ferrous metals"],
        ["Precious metals production",
         "Re-processing of secondary preciuos metals into new preciuos metals",
         "Aluminium production",
         "Re-processing of secondary aluminium into new aluminium",
         "Lead, zinc and tin production",
         "Re-processing of secondary lead into new lead, zinc and tin",
         "Copper production", "Re-processing of secondary copper into new copper",
         "Other non-ferrous metal production",
         "Re-processing of secondary other non-ferrous metals into new other non-ferrous metals"],
        "aggregate",
        nace2=_t21("Non ferrous metals"), source="table21",
        table21_name="Non ferrous metals",
        notes="Table 21: 24.41-24.45 (NF metallurgy) + 24.53 24.54 (NF "
              "casting) + 25.11 25.12 25.21 (metal structures, tanks). "
              "Non-ferrous ore mining (7.29) goes to 'Non-metallic minerals' "
              "per Table 21. The 25.x codes have no distinct EXIOBASE "
              "product, so they resolve to 'Fabricated Metal products'."),

    SectorMap(
        "Chemical Products", "Heavy industry",
        ["Plastics, basic",
         "Secondary plastic for treatment, Re-processing of secondary plastic into new plastic",
         "N-fertiliser", "P- and other fertiliser", "Chemicals nec",
         "Additives/Blending Components"],
        ["Plastics, basic", "Re-processing of secondary plastic into new plastic",
         "N-fertiliser", "P- and other fertiliser", "Chemicals nec"],
        "aggregate",
        nace2=_t21("Chemical Products"), source="table21",
        notes="Table 21: 20.11-20.51. 'Additives/Blending Components' is "
              "filed under C20 by the repo concordance, and the Oil row does "
              "not claim C20, so it lands here rather than in 'Oil'. Chemical "
              "and fertiliser minerals and salt (8.91 8.93) are assigned by "
              "Table 21 to 'Non-metallic minerals'. CAUTION: EXIOBASE "
              "'Chemicals nec' includes pharmaceuticals, which the NGFS "
              "version breaks out (see 'Basic pharmaceutical products')."),

    SectorMap(
        "Non-metallic minerals", "Heavy industry",
        ["Stone", "Sand and clay",
         "Chemical and fertilizer minerals, salt and other mining and quarrying products n.e.c.",
         "Iron ores", "Uranium and thorium ores (12)",
         "Copper ores and concentrates", "Nickel ores and concentrates",
         "Aluminium ores and concentrates", "Precious metal ores and concentrates",
         "Lead, zinc and tin ores and concentrates",
         "Other non-ferrous metal ores and concentrates",
         "Glass and glass products",
         "Secondary glass for treatment, Re-processing of secondary glass into new glass",
         "Ceramic goods",
         "Bricks, tiles and construction products, in baked clay",
         "Cement, lime and plaster",
         "Ash for treatment, Re-processing of ash into clinker",
         "Other non-metallic mineral products"],
        ["Quarrying of stone", "Quarrying of sand and clay",
         "Mining of chemical and fertilizer minerals, production of salt, other mining and quarrying n.e.c.",
         "Mining of iron ores", "Mining of uranium and thorium ores (12)",
         "Mining of copper ores and concentrates",
         "Mining of nickel ores and concentrates",
         "Mining of aluminium ores and concentrates",
         "Mining of precious metal ores and concentrates",
         "Mining of lead, zinc and tin ores and concentrates",
         "Mining of other non-ferrous metal ores and concentrates",
         "Manufacture of glass and glass products",
         "Re-processing of secondary glass into new glass",
         "Manufacture of ceramic goods",
         "Manufacture of bricks, tiles and construction products, in baked clay",
         "Manufacture of cement, lime and plaster",
         "Re-processing of ash into clinker",
         "Manufacture of other non-metallic mineral products n.e.c."],
        "aggregate",
        nace2=_t21("Non metallic minerals"), source="table21",
        table21_name="Non metallic minerals",
        notes="COUNTER-INTUITIVE BUT EXPLICIT: Table 21 assigns ALL non-energy "
              "mining to this sector, metal ores included -- 7.1 (iron), 7.21 "
              "(uranium/thorium), 7.29 (other non-ferrous) -- on top of 8.11 "
              "8.12 8.91 8.93 8.99 (quarrying, chemical minerals, salt). The "
              "'Ferrous metals' and 'Non ferrous metals' rows contain no "
              "extraction code at all. The sector therefore reads as "
              "'processed non-metallic minerals PLUS non-energy mining', and "
              "the 8 EXIOBASE ore products sit here rather than in Ferrous / "
              "Non-ferrous / Nuclear. This is the single heaviest correction "
              "the official table imposes."),

    SectorMap(
        "Rubber and plastic products", "Heavy industry",
        ["Rubber and plastic products (25)"],
        ["Manufacture of rubber and plastic products (25)"],
        "direct",
        nace2=["22.1", "22.2"], source="extension",
        notes="Not in Table 21, which files 22.11-22.29 under 'Oil' (an "
              "earlier GEM-E3 aggregation). The NGFS 50-sector version makes "
              "it a sector of its own, so the NGFS version is followed here. "
              "Basic plastics (20.16) stay in 'Chemical Products'."),

    SectorMap(
        "Paper products, publishing", "Heavy industry",
        ["Pulp",
         "Secondary paper for treatment, Re-processing of secondary paper into new pulp",
         "Paper and paper products",
         "Printed matter and recorded media (22)"],
        ["Pulp", "Re-processing of secondary paper into new pulp", "Paper",
         "Publishing, printing and reproduction of recorded media (22)"],
        "aggregate",
        nace2=_t21("Paper Products"), source="table21",
        table21_name="Paper Products",
        notes="Table 21: 17.11-17.29 (pulp and paper) + 18.11-18.2 (printing "
              "and reproduction of recorded media)."),

    SectorMap(
        "Basic pharmaceutical products", "Heavy industry",
        [], [], "overlap", overlaps_with="Chemical Products",
        nace2=["21.1", "21.2"], source="extension",
        notes="Table 21 has no pharma sector: it files 21.2 under 'Oil' "
              "(almost certainly an error) and 21.1 nowhere at all. The NGFS "
              "50-sector version breaks it out. EXIOBASE 3 does not separate "
              "pharmaceuticals either -- they sit inside 'Chemicals nec', "
              "already assigned to 'Chemical Products'. Not separately "
              "shockable, and at 1961 bn$ world 2025 this is the largest loss "
              "of resolution in the whole mapping."),

    # =====================================================================
    # 4) Capital goods and manufacturing
    # =====================================================================
    SectorMap(
        "Fabricated Metal products", "Manufacturing",
        ["Fabricated metal products, except machinery and equipment (28)"],
        ["Manufacture of fabricated metal products, except machinery and equipment (28)"],
        "direct",
        nace2=_t21("Fabricated Metal products"), source="table21",
        notes="Label restored: this block of codes appears orphaned in "
              "Table 21 after a page break (see GEME3_NACE2). ANOMALY: the "
              "row contains 38.21 (waste treatment and disposal), unrelated "
              "to fabricated metal products and inconsistent with 38.11 38.12 "
              "38.22 sitting under 'Non Market Services'. Not followed -- all "
              "waste goes to 'Non Market Services'."),

    SectorMap(
        "Other Equipment Goods", "Manufacturing",
        ["Machinery and equipment n.e.c. (29)",
         "Electrical machinery and apparatus n.e.c. (31)"],
        ["Manufacture of machinery and equipment n.e.c. (29)",
         "Manufacture of electrical machinery and apparatus n.e.c. (31)"],
        "aggregate",
        nace2=_t21("Other Equipment Goods"), source="table21",
        notes="Table 21: 27.11-27.32 (electrical equipment) + 28.23 28.99 "
              "(machinery) among others. Also acts as the parent for the "
              "NGFS-only 'new' sectors (batteries, PV/wind/CCS equipment, "
              "advanced appliances) -- see the 'overlap' entries."),

    SectorMap(
        "Computer, electronic and optical products", "Manufacturing",
        ["Office machinery and computers (30)",
         "Radio, television and communication equipment and apparatus (32)",
         "Medical, precision and optical instruments, watches and clocks (33)"],
        ["Manufacture of office machinery and computers (30)",
         "Manufacture of radio, television and communication equipment and apparatus (32)",
         "Manufacture of medical, precision and optical instruments, watches and clocks (33)"],
        "aggregate",
        nace2=["26.11", "26.12", "26.2", "26.3", "26.4", "26.51", "26.52",
               "26.6", "26.7", "26.8"], source="extension",
        notes="Table 21 has no such sector: it has 'Electric Goods' (26.x + "
              "27.9 + 28.1x), which the NGFS 50-sector version splits into "
              "'Computer, electronic and optical products' (C26) and "
              "'Other Equipment Goods' (C27/C28)."),

    SectorMap(
        "Transport equipment (excluding EV)", "Manufacturing",
        ["Motor vehicles, trailers and semi-trailers (34)",
         "Other transport equipment (35)"],
        ["Manufacture of motor vehicles, trailers and semi-trailers (34)",
         "Manufacture of other transport equipment (35)"],
        "aggregate",
        nace2=_t21("Transport equipment"), source="table21",
        table21_name="Transport equipment",
        notes="Table 21: 29.1 29.2 29.32 (automotive) + 30.x (other transport "
              "equipment). NOTE: page number 164 was removed from this row, "
              "between 30.3 and 30.91. EXIOBASE does not separate ICE from "
              "electric vehicles, so this block de facto contains EVs too "
              "(see 'EV Transport Equipment'). For a realistic shock, "
              "recombine the two GEM-E3 trajectories weighted by their levels."),

    SectorMap(
        "Consumer Goods Industries", "Manufacturing",
        ["Products of meat cattle", "Products of meat pigs",
         "Products of meat poultry", "Meat products nec",
         "products of Vegetable oils and fats", "Dairy products",
         "Processed rice", "Sugar", "Food products nec", "Beverages",
         "Fish products", "Tobacco products (16)",
         "Textiles (17)", "Wearing apparel; furs (18)",
         "Leather and leather products (19)",
         "Wood and products of wood and cork (except furniture); articles of straw and plaiting materials (20)",
         "Wood material for treatment, Re-processing of secondary wood material into new wood material",
         "Furniture; other manufactured goods n.e.c. (36)"],
        ["Processing of meat cattle", "Processing of meat pigs",
         "Processing of meat poultry", "Production of meat products nec",
         "Processing vegetable oils and fats", "Processing of dairy products",
         "Processed rice", "Sugar refining", "Processing of Food products nec",
         "Manufacture of beverages", "Manufacture of fish products",
         "Manufacture of tobacco products (16)",
         "Manufacture of textiles (17)",
         "Manufacture of wearing apparel; dressing and dyeing of fur (18)",
         "Tanning and dressing of leather; manufacture of luggage, handbags, saddlery, harness and footwear (19)",
         "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials (20)",
         "Re-processing of secondary wood material into new wood material",
         "Manufacture of furniture; manufacturing n.e.c. (36)"],
        "aggregate",
        nace2=_t21("Consumer Goods Industries"), source="table21",
        notes="Table 21: 10.x 11.x 12 (food, beverages, tobacco) + 13.x 14.x "
              "15.x (textiles, apparel, leather) + 16.x (wood) + 31.01 31.09 "
              "(furniture) + 32.3 (sporting goods). GEM-E3 has no separate "
              "agri-food sector. Overlaps 'Agriculture' on 10.11 10.12 10.39 "
              "10.61 10.84, 12 and 16.1: arbitrated in favour of this sector, "
              "which keeps all processing while Agriculture keeps primary "
              "production."),

    SectorMap(
        "Batteries", "Manufacturing", [], [], "overlap",
        overlaps_with="Other Equipment Goods",
        nace2=["27.2"], source="extension",
        notes="Inside 'Electrical machinery and apparatus n.e.c. (31)'."),
    SectorMap(
        "EV Transport Equipment", "Manufacturing", [], [], "overlap",
        overlaps_with="Transport equipment (excluding EV)",
        nace2=["29.1"], source="extension",
        notes="Inside 'Motor vehicles, trailers and semi-trailers (34)'."),
    SectorMap(
        "Equipment for PV panels", "Manufacturing", [], [], "overlap",
        overlaps_with="Other Equipment Goods",
        nace2=["26.11", "27.11"], source="extension",
        notes="Inside 'Electrical machinery and apparatus n.e.c. (31)'."),
    SectorMap(
        "Equipment for wind power technology", "Manufacturing", [], [], "overlap",
        overlaps_with="Other Equipment Goods",
        nace2=["27.11", "28.11"], source="extension",
        notes="Inside 'Machinery and equipment n.e.c. (29)'."),
    SectorMap(
        "Equipment for CCS power technology", "Manufacturing", [], [], "overlap",
        overlaps_with="Other Equipment Goods",
        nace2=["28.29"], source="extension",
        notes="Inside 'Machinery and equipment n.e.c. (29)'. Output ~0."),
    SectorMap(
        "Advanced Electric Appliances", "Manufacturing", [], [], "overlap",
        overlaps_with="Other Equipment Goods",
        nace2=["27.51"], source="extension",
        notes="Inside 'Electrical machinery and apparatus n.e.c. (31)'."),
    SectorMap(
        "Advanced Heating and Cooking Appliances", "Manufacturing", [], [], "overlap",
        overlaps_with="Other Equipment Goods",
        nace2=["27.51", "28.21"], source="extension",
        notes="Heat pumps etc. Inside 'Machinery and equipment n.e.c. (29)'."),

    # =====================================================================
    # 5) Primary and construction
    # =====================================================================
    SectorMap(
        "Agriculture", "Primary",
        _PXP_CROPS_LIVESTOCK, _IXI_CROPS_LIVESTOCK,
        "aggregate",
        nace2=_t21("Agriculture"), source="table21",
        notes="Table 21: 1.x (crops, livestock) + 2.x (forestry) + 3.x "
              "(fishing). The food-processing codes in the row (10.11 10.12 "
              "10.39 10.61 10.84, 12, 16.1) are also claimed by 'Consumer "
              "Goods Industries': all processing is left to the latter and "
              "only primary production kept here, otherwise the EXIOBASE "
              "partition breaks."),

    SectorMap(
        "Construction", "Primary",
        ["Construction work (45)",
         "Secondary construction material for treatment, Re-processing of secondary construction material into aggregates"],
        ["Construction (45)",
         "Re-processing of secondary construction material into aggregates"],
        "direct",
        nace2=_t21("Construction"), source="table21",
        notes="Table 21: 42.x (civil engineering) + 43.1x 43.2x 43.31 "
              "(specialised works). ANOMALY: 41.1 41.2 (construction of "
              "buildings) are filed under 'Market Services' by Table 21. "
              "EXIOBASE has a single construction product covering "
              "F41+F42+F43, so it is kept here. 'Secondary construction "
              "material' (aggregate recovery, 38.32) is listed nowhere in "
              "Table 21 and is also kept here."),

    # =====================================================================
    # 6) Transport
    # =====================================================================
    SectorMap(
        "Land transport", "Transport",
        ["Railway transportation services", "Other land transportation services",
         "Transportation services via pipelines"],
        ["Transport via railways", "Other land transport", "Transport via pipelines"],
        "aggregate",
        nace2=["49.1", "49.2", "49.31", "49.32", "49.39", "49.41", "49.42", "49.5"],
        source="table21", table21_name="Transport (Land)",
        notes="Table 21: 49.x, including 49.5 pipelines. The 52.x codes in "
              "the same row (warehousing and support activities) are a "
              "separate sector in the NGFS 50-sector version -> 'Warehousing'."),

    SectorMap(
        "Air transport", "Transport",
        ["Air transport services (62)"], ["Air transport (62)"],
        "direct",
        nace2=_t21("Transport (Air)"), source="table21",
        table21_name="Transport (Air)"),

    SectorMap(
        "Water transport", "Transport",
        ["Sea and coastal water transportation services",
         "Inland water transportation services"],
        ["Sea and coastal water transport", "Inland water transport"],
        "direct",
        nace2=_t21("Transport (Water)"), source="table21",
        table21_name="Transport (Water)"),

    SectorMap(
        "Warehousing", "Transport",
        ["Supporting and auxiliary transport services; travel agency services (63)"],
        ["Supporting and auxiliary transport activities; activities of travel agencies (63)"],
        "direct",
        nace2=["52.1", "52.21", "52.22", "52.23", "52.24", "52.29"],
        source="extension",
        notes="The 52.x block, which Table 21 files under 'Transport (Land)' "
              "and which the NGFS 50-sector version breaks out. EXIOBASE "
              "lumps warehousing/cargo handling together with travel agencies."),

    # =====================================================================
    # 7) Services
    # =====================================================================
    SectorMap(
        "Market Services", "Services",
        ["Collected and purified water, distribution services of water (41)",
         "Sale, maintenance, repair of motor vehicles, motor vehicles parts, motorcycles, motor cycles parts and accessoiries",
         "Retail trade services of motor fuel",
         "Wholesale trade and commission trade services, except of motor vehicles and motorcycles (51)",
         "Retail  trade services, except of motor vehicles and motorcycles; repair services of personal and household goods (52)",
         "Hotel and restaurant services (55)",
         "Post and telecommunication services (64)",
         "Financial intermediation services, except insurance and pension funding services (65)",
         "Insurance and pension funding services, except compulsory social security services (66)",
         "Services auxiliary to financial intermediation (67)",
         "Real estate services (70)",
         "Renting services of machinery and equipment without operator and of personal and household goods (71)",
         "Computer and related services (72)",
         "Other business services (74)",
         "Recreational, cultural and sporting services (92)",
         "Other services (93)"],
        ["Collection, purification and distribution of water (41)",
         "Sale, maintenance, repair of motor vehicles, motor vehicles parts, motorcycles, motor cycles parts and accessoiries",
         "Retail sale of automotive fuel",
         "Wholesale trade and commission trade, except of motor vehicles and motorcycles (51)",
         "Retail trade, except of motor vehicles and motorcycles; repair of personal and household goods (52)",
         "Hotels and restaurants (55)",
         "Post and telecommunications (64)",
         "Financial intermediation, except insurance and pension funding (65)",
         "Insurance and pension funding, except compulsory social security (66)",
         "Activities auxiliary to financial intermediation (67)",
         "Real estate activities (70)",
         "Renting of machinery and equipment without operator and of personal and household goods (71)",
         "Computer and related activities (72)",
         "Other business activities (74)",
         "Recreational, cultural and sporting activities (92)",
         "Other service activities (93)"],
        "aggregate",
        nace2=_t21("Market Services"), source="table21",
        notes="Table 21: 36 (water) + 45.x 46.1x (trade) + 53.x (postal) + "
              "61.x 62.x (telecoms, IT) + 64-66 (finance) + 68.x (real "
              "estate) + 70.22 71.x 73.x (business services). NOT IN TABLE 21 "
              "but attached here by judgement: 47.x retail trade, 55/56 "
              "accommodation and food service, 77 rental and leasing, 69 "
              "legal and accounting, 90-93 arts and recreation, 96 personal "
              "services -- Table 21 lists none of them anywhere (its trade "
              "coverage stops at 46.11-46.18, i.e. wholesale on a fee basis "
              "only, so the source is visibly truncated). R&D (72.x) is a "
              "separate sector in the NGFS 50-sector version."),

    SectorMap(
        "Non Market Services", "Services",
        ["Public administration and defence services; compulsory social security services (75)",
         "Education services (80)", "Health and social work services (85)",
         "Membership organisation services n.e.c. (91)",
         "Private households with employed persons (95)",
         "Extra-territorial organizations and bodies",
         "Secondary raw materials",
         "Bottles for treatment, Recycling of bottles by direct reuse",
         *_PXP_WASTE],
        ["Public administration and defence; compulsory social security (75)",
         "Education (80)", "Health and social work (85)",
         "Activities of membership organisation n.e.c. (91)",
         "Private households with employed persons (95)",
         "Extra-territorial organizations and bodies",
         "Recycling of waste and scrap", "Recycling of bottles by direct reuse",
         *_IXI_WASTE],
        "aggregate",
        nace2=_t21("Non Market Services"), source="table21",
        notes="Table 21: 84.x (public administration) + 75 (veterinary) + "
              "59.x 60.x (media) + 78.1 79.9 80.2 81.x 82.19 (support "
              "services) AND CRUCIALLY 37 (sewerage), 38.11 38.12 38.22 "
              "(waste collection and disposal) and 39 (remediation). That is "
              "why the 20 waste-treatment products and the recycling products "
              "sit here and NOT in 'Market Services'. NOT IN TABLE 21 but "
              "attached here by judgement: 85 education, 86-88 human health, "
              "94 membership organisations, 97 households as employers, "
              "99 extraterritorial bodies."),

    SectorMap(
        "R&D", "Services",
        ["Research and development services (73)"],
        ["Research and development (73)"],
        "direct",
        nace2=["72.11", "72.19", "72.2"], source="extension",
        notes="The 72.x block, which Table 21 files under 'Market Services' "
              "and which the NGFS 50-sector version breaks out."),
]


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
# Single source of truth: the exact GEM-E3 country label as it appears in the
# `Region` column of production_growth.xlsx -> the EXIOBASE ISO2 code, or None
# when EXIOBASE has no matching region. Both directions derive from this, so
# the two maps cannot drift apart.

NGFS_COUNTRY_TO_EXIO: dict[str, str | None] = {
    "Austria - AUT": "AT", "Belgium - BEL": "BE", "Bulgaria - BGR": "BG",
    "Croatia - HRV": "HR", "Cyprus - CYP": "CY", "Czechia - CZE": "CZ",
    "Denmark - DNK": "DK", "Estonia - EST": "EE", "Finland - FIN": "FI",
    "France - FRA": "FR", "Germany - DEU": "DE", "Greece - GRC": "GR",
    "Hungary - HUN": "HU", "Ireland - IRL": "IE", "Italy - ITA": "IT",
    "Latvia - LVA": "LV", "Lithuania - LTU": "LT", "Luxembourg - LUX": "LU",
    "Malta - MLT": "MT", "Netherlands - NLD": "NL", "Poland - POL": "PL",
    "Portugal - PRT": "PT", "Romania - ROU": "RO", "Slovakia - SVK": "SK",
    "Slovenia - SVN": "SI", "Spain - ESP": "ES", "Sweden - SWE": "SE",
    "United Kingdom - GBR": "GB",
    "USA - USA": "US", "Japan - JPN": "JP", "China - CHN": "CN",
    "Canada - CAN": "CA", "South Korea - KOR": "KR", "Brazil - BRA": "BR",
    "India - IND": "IN", "Mexico - MEX": "MX", "Russia - RUS": "RU",
    "Turkey - TUR": "TR", "South Africa - ZAF": "ZA", "Indonesia - IDN": "ID",
    "Argentina - ARG": None,     # inside WL (RoW America)
    "Saudi Arabia - SAU": None,  # inside WM (RoW Middle East)
}

EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
        "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
        "NL", "PL", "PT", "RO", "SE", "SI", "SK"]

# The 49 EXIOBASE regions: 44 countries + 5 Rest-of-World zones.
EXIOBASE_REGIONS = [
    "AT", "AU", "BE", "BG", "BR", "CA", "CH", "CN", "CY", "CZ", "DE", "DK",
    "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU", "ID", "IE", "IN", "IT",
    "JP", "KR", "LT", "LU", "LV", "MT", "MX", "NL", "NO", "PL", "PT", "RO",
    "RU", "SE", "SI", "SK", "TR", "TW", "US", "ZA",
    "WA", "WE", "WF", "WL", "WM",
]

_REGION_AGGREGATES = {"World": "*", "EU27": EU27, "EU28": EU27 + ["GB"]}

# 40 of the 49 EXIOBASE regions have a GEM-E3 country counterpart. These 9 do
# not, and fall back to the closest GEM-E3 aggregate so that regional coverage
# is complete when projecting growth rates onto x.
EXIO_REGION_FALLBACK: dict[str, str] = {
    "AU": "Oceania",
    "CH": "Rest of Europe",
    "NO": "Rest of Europe",
    "TW": "Asia",
    "WA": "Asia",                                  # RoW Asia and Pacific
    "WE": "Rest of Europe",                        # RoW Europe
    "WF": "Africa",                                # RoW Africa
    "WL": "South America",                         # RoW America
    "WM": "Rest of Energy Producing Countries",    # RoW Middle East
}

_EXIO_TO_NGFS_COUNTRY = {v: k for k, v in NGFS_COUNTRY_TO_EXIO.items() if v}


def ngfs_region_to_exiobase(region: str) -> list[str] | str | None:
    """
    Convert a GEM-E3 region label into EXIOBASE region code(s).

    Returns '*' for World, a list of ISO2 codes for aggregates and countries,
    or None when the region has no EXIOBASE counterpart (R5 aggregates,
    'Rest of World', 'Euro Area', ARG, SAU...).
    """
    if region in _REGION_AGGREGATES:
        return _REGION_AGGREGATES[region]
    iso2 = NGFS_COUNTRY_TO_EXIO.get(region)
    return [iso2] if iso2 else None


def exiobase_region_to_ngfs(exio_region: str) -> str:
    """
    Convert an EXIOBASE region code into the GEM-E3 region to read rates from.

    Inverse of `ngfs_region_to_exiobase` for the 40 countries that match
    one-to-one, falling back to `EXIO_REGION_FALLBACK` for the 9 EXIOBASE
    regions with no GEM-E3 country (AU, CH, NO, TW and the 5 RoW zones).

    Raises
    ------
    KeyError if the code is not a known EXIOBASE region.
    """
    if exio_region in _EXIO_TO_NGFS_COUNTRY:
        return _EXIO_TO_NGFS_COUNTRY[exio_region]
    if exio_region in EXIO_REGION_FALLBACK:
        return EXIO_REGION_FALLBACK[exio_region]
    raise KeyError(f"Unknown EXIOBASE region: {exio_region!r}")


def exio_region_map(exio_regions: list[str] | None = None) -> dict[str, str]:
    """
    EXIOBASE region -> GEM-E3 region label, for every region at once.

    Vectorisation helper for `shortterm/project_production.py`. Pass the actual
    region list from an EXIOBASE table; defaults to `EXIOBASE_REGIONS`.
    """
    if exio_regions is None:
        exio_regions = EXIOBASE_REGIONS
    return {r: exiobase_region_to_ngfs(r) for r in exio_regions}


# ---------------------------------------------------------------------------
# Lookups and tables
# ---------------------------------------------------------------------------

def get_map(ngfs_sector: str) -> SectorMap:
    """Return the SectorMap for a GEM-E3 sector name."""
    for m in MAPPING:
        if m.ngfs_sector == ngfs_sector:
            return m
    raise KeyError(f"Unknown sector {ngfs_sector!r}. "
                   f"Choices: {[m.ngfs_sector for m in MAPPING]}")


def sector_map(table_type: str = "pxp") -> dict[str, list[str]]:
    """GEM-E3 sector -> list of EXIOBASE sectors (empty for 'overlap')."""
    return {m.ngfs_sector: m.exio(table_type) for m in MAPPING}


def exio_to_ngfs(table_type: str = "pxp") -> dict[str, str]:
    """Inverse mapping EXIOBASE -> GEM-E3 (a partition: 1 key -> 1 value)."""
    return {s: m.ngfs_sector for m in MAPPING for s in m.exio(table_type)}


def nace2_map() -> dict[str, list[str]]:
    """GEM-E3 sector (NGFS naming) -> list of NACE Rev. 2 class codes."""
    return {m.ngfs_sector: list(m.nace2) for m in MAPPING}


def nace2_table() -> pd.DataFrame:
    """
    Tidy NACE 2 table: one row per (GEM-E3 sector, NACE 2 class code).

    Columns: ngfs_sector, table21_name, source, nace2_code, nace2_division.
    """
    rows = [
        {"ngfs_sector": m.ngfs_sector,
         "table21_name": m.table21_name or (m.ngfs_sector if m.source == "table21" else ""),
         "source": m.source,
         "nace2_code": code,
         "nace2_division": nace2_division(code)}
        for m in MAPPING for code in m.nace2
    ]
    return pd.DataFrame(rows)


def mapping_table(table_type: str = "pxp", with_notes: bool = False) -> pd.DataFrame:
    """
    Long mapping table: one row per (GEM-E3 sector, EXIOBASE sector) pair.

    'overlap' sectors produce a single row with `exio_sector` = NA.

    Parameters
    ----------
    with_notes : repeat the full justification on every row. Off by default --
                 the justification belongs to the sector, not to the pair, so
                 use `sector_table()` for it.
    """
    rows = []
    for m in MAPPING:
        base = {
            "ngfs_sector": m.ngfs_sector, "theme": m.theme,
            "quality": m.quality, "source": m.source,
            "table21_name": m.table21_name or "",
            "nace2": " ".join(m.nace2),
            "overlaps_with": m.overlaps_with or "",
        }
        if with_notes:
            base["notes"] = m.notes
        exio = m.exio(table_type)
        if not exio:
            rows.append({**base, "exio_sector": pd.NA})
        else:
            rows.extend({**base, "exio_sector": s} for s in exio)

    cols = ["ngfs_sector", "theme", "quality", "source", "table21_name",
            "nace2", "exio_sector", "overlaps_with"]
    if with_notes:
        cols.append("notes")
    return pd.DataFrame(rows)[cols]


def sector_table(table_type: str = "pxp") -> pd.DataFrame:
    """
    One row per GEM-E3 sector, with the justification. This is where the
    `notes` text lives -- it is a property of the sector, so repeating it on
    every EXIOBASE row (as `mapping_table(with_notes=True)` does) is noise.
    """
    rows = []
    for m in MAPPING:
        exio = m.exio(table_type)
        rows.append({
            "ngfs_sector":     m.ngfs_sector,
            "theme":           m.theme,
            "quality":         m.quality,
            "source":          m.source,
            "table21_name":    m.table21_name or "",
            "n_nace2":         len(m.nace2),
            "nace2_divisions": " ".join(m.nace2_divisions()),
            "n_exio":          len(exio),
            "overlaps_with":   m.overlaps_with or "",
            "exio_sectors":    "; ".join(exio),
            "justification":   m.notes,
        })
    return pd.DataFrame(rows)


def summary_table(table_type: str = "pxp") -> pd.DataFrame:
    """Compact one-row-per-sector view for terminal display."""
    df = sector_table(table_type)
    df = df[["ngfs_sector", "theme", "quality", "source", "n_nace2",
             "n_exio", "overlaps_with", "exio_sectors"]].copy()
    df["exio_head"] = df["exio_sectors"].str.slice(0, 45)
    return df.drop(columns=["exio_sectors"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def load_ngfs_sectors() -> list[str]:
    """Sector names present in production_growth.xlsx (GEM-E3 sheet)."""
    df = pd.read_excel(PROCESSED_DIR / "production_growth.xlsx",
                       sheet_name=GEME3_SHEET)
    return sorted(df["Secteur"].dropna().astype(str).unique())


def load_ngfs_regions() -> list[str]:
    """Region labels present in production_growth.xlsx (GEM-E3 sheet)."""
    df = pd.read_excel(PROCESSED_DIR / "production_growth.xlsx",
                       sheet_name=GEME3_SHEET)
    return sorted(df["Region"].dropna().astype(str).unique())


def validate(year: int = 2019, table_type: str = "pxp") -> dict:
    """
    Check the mapping against the real nomenclatures.

    Returns a dict with 'unknown_exio', 'unmapped_exio', 'duplicated_exio',
    'missing_ngfs', 'extra_ngfs', 'unknown_table21', 'unmapped_regions',
    'unknown_ngfs_regions' and 'coverage_x'.
    All list entries should be empty for a valid partition.
    """
    x = load_x(year, table_type)
    exio_sectors = list(dict.fromkeys(x.index.get_level_values(1)))
    exio_regions = list(dict.fromkeys(x.index.get_level_values(0)))

    seen: dict[str, list[str]] = {}
    for m in MAPPING:
        for s in m.exio(table_type):
            seen.setdefault(s, []).append(m.ngfs_sector)

    ngfs_data = set(load_ngfs_sectors())
    ngfs_map = {m.ngfs_sector for m in MAPPING}

    declared_t21 = {m.table21_name or m.ngfs_sector
                    for m in MAPPING if m.source == "table21"}

    # Region side: every EXIOBASE region must resolve to a GEM-E3 region that
    # actually exists in the data.
    unmapped_regions, resolved_regions = [], {}
    for r in exio_regions:
        try:
            resolved_regions[r] = exiobase_region_to_ngfs(r)
        except KeyError:
            unmapped_regions.append(r)
    ngfs_regions_data = set(load_ngfs_regions())
    unknown_ngfs_regions = sorted(
        {v for v in resolved_regions.values()} - ngfs_regions_data
    )

    x_by_sector = x.groupby(level=1).sum()
    covered = x_by_sector[[s for s in exio_sectors if s in seen]].sum()

    return {
        "table_type":           table_type,
        "year":                 year,
        "n_exio":               len(exio_sectors),
        "n_exio_regions":       len(exio_regions),
        "n_ngfs_mapping":       len(MAPPING),
        "n_table21":            sum(1 for m in MAPPING if m.source == "table21"),
        "unknown_exio":         sorted(s for s in seen if s not in exio_sectors),
        "unmapped_exio":        [s for s in exio_sectors if s not in seen],
        "duplicated_exio":      {s: v for s, v in seen.items() if len(v) > 1},
        "missing_ngfs":         sorted(ngfs_data - ngfs_map),
        "extra_ngfs":           sorted(ngfs_map - ngfs_data),
        "unknown_table21":      sorted(declared_t21 - set(GEME3_NACE2)),
        "unmapped_regions":     unmapped_regions,
        "unknown_ngfs_regions": unknown_ngfs_regions,
        "coverage_x":           float(covered / x_by_sector.sum()),
    }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace(",", "")).strip().lower()


def crosscheck_table21() -> pd.DataFrame:
    """
    Confront the EXIOBASE -> GEM-E3 assignment with Table 21's NACE 2 divisions.

    For every EXIOBASE pxp product: its NACE 2 division (from
    `exiobase_concordance.csv`), the GEM-E3 sectors Table 21 makes candidates
    for that division, and whether the retained assignment is among them.

    CAUTION: the repo concordance is at **division** level (C20) while Table 21
    is at **class** level (20.14). A `coherent=False` row is therefore not
    necessarily an error -- it may come from that loss of precision, or from a
    sector specific to the NGFS 50-sector version (absent from Table 21). Use
    the `reason` column to triage.
    """
    conc = pd.read_csv(CONCORDANCE_CSV)
    exio_div = dict(zip(conc["exiobase_sector"].map(_norm), conc["nace_rev2"]))

    div_owners: dict[str, list[str]] = {}
    for sec, codes in GEME3_NACE2.items():
        for d in {nace2_division(c) for c in codes}:
            div_owners.setdefault(d, []).append(sec)

    by_sector = {m.ngfs_sector: m for m in MAPPING}
    rows = []
    for exio, ngfs in exio_to_ngfs("pxp").items():
        d = exio_div.get(_norm(exio))
        owners_t21 = div_owners.get(d, [])
        owners_ngfs = {TABLE21_TO_NGFS.get(o, o) for o in owners_t21}
        coherent = ngfs in owners_ngfs
        if coherent:
            reason = ""
        elif by_sector[ngfs].source == "extension":
            reason = "NGFS sector absent from Table 21"
        elif not owners_t21:
            reason = "NACE 2 division absent from Table 21"
        else:
            reason = "arbitrated at class level"
        rows.append({
            "exio_sector": exio, "nace2_division": d, "ngfs_sector": ngfs,
            "table21_candidates": " | ".join(owners_t21),
            "coherent": coherent, "reason": reason,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Application: project NGFS growth rates onto EXIOBASE
# ---------------------------------------------------------------------------

def growth_to_exiobase(
    scenario: str,
    year: int | str,
    region: str = "World",
    table_type: str = "pxp",
    fill_unmapped: float | None = 0.0,
) -> pd.Series:
    """
    Project GEM-E3 YoY growth rates onto the EXIOBASE nomenclature.

    Every EXIOBASE sector inherits the growth rate (in %) of the GEM-E3 sector
    it is assigned to. 'overlap' sectors (Batteries, Hydrogen, CCS, pharma,
    EVs, Clean Gas) have no EXIOBASE target and are ignored: their dynamics
    stay buried in the parent sector's.

    Parameters
    ----------
    scenario      : GEM-E3 scenario ('Baseline', 'DIRE', 'SWUC', 'HWTP', 'DAPS_*')
    year          : 2022..2030 (2021 is NaN by construction, no base year)
    region        : GEM-E3 region ('World', 'EU27', 'France - FRA', ...)
    fill_unmapped : value for EXIOBASE sectors with no mapping
                    (None leaves them as NaN)

    Returns
    -------
    pd.Series indexed by EXIOBASE sector name, values in % YoY.
    """
    df = pd.read_excel(PROCESSED_DIR / "production_growth.xlsx",
                       sheet_name=GEME3_SHEET)
    sub = df[(df["Scenario"] == scenario) & (df["Region"] == region)]
    if sub.empty:
        raise ValueError(
            f"No rows for scenario={scenario!r}, region={region!r}. "
            f"Scenarios: {sorted(df['Scenario'].unique())}"
        )
    col = str(year)
    if col not in sub.columns:
        raise ValueError(f"Year {col} not found. Columns: "
                         f"{[c for c in sub.columns if c.isdigit()]}")

    g = pd.to_numeric(sub.set_index("Secteur")[col], errors="coerce")

    inverse = exio_to_ngfs(table_type)
    x = load_x(2019, table_type)
    exio_sectors = list(dict.fromkeys(x.index.get_level_values(1)))

    values = {s: g.get(inverse[s], float("nan")) if s in inverse else float("nan")
              for s in exio_sectors}
    out = pd.Series(values, name=f"{scenario}|{region}|{col} (% YoY)")
    if fill_unmapped is not None:
        out = out.fillna(fill_unmapped)
    return out


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def export_mapping_csv(path: str | Path, table_type: str = "pxp") -> Path:
    """Write the long (GEM-E3, EXIOBASE) mapping to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping_table(table_type).to_csv(path, index=False, encoding="utf-8")
    return path


def export_workbook(path: str | Path, table_type: str = "pxp") -> Path:
    """
    Write a multi-sheet .xlsx: the readable form of the whole mapping.

    Sheets
    ------
    sectors    : one row per GEM-E3 sector, with the justification
    mapping    : one row per (GEM-E3, EXIOBASE) pair -- the join table
    nace2      : one row per (GEM-E3, NACE 2 class code)
    table21    : Table 21 as transcribed, one row per (label, code)
    crosscheck : Table 21 confrontation, with a reason for every mismatch
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t21 = pd.DataFrame(
        [{"table21_sector": k, "nace2_code": c,
          "nace2_division": nace2_division(c),
          "ngfs_sector": TABLE21_TO_NGFS.get(k, k) or "(split in NGFS version)"}
         for k, codes in GEME3_NACE2.items() for c in codes]
    )

    sheets = {
        "sectors":    sector_table(table_type),
        "mapping":    mapping_table(table_type),
        "nace2":      nace2_table(),
        "table21":    t21,
        "crosscheck": crosscheck_table21(),
    }

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for i, col in enumerate(frame.columns, start=1):
                width = 60 if col in ("justification", "exio_sectors", "nace2") else \
                    min(max(len(str(col)) + 2,
                            int(frame[col].astype(str).str.len().max() or 0) + 2), 45)
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
            ws.freeze_panes = "A2"
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NGFS short-term (GEM-E3) <-> EXIOBASE sector mapping, "
                    "anchored on Table 21 (GEM-E3 -> NACE 2)."
    )
    parser.add_argument("--list", action="store_true",
                        help="Summary: one row per GEM-E3 sector.")
    parser.add_argument("--detail", default=None,
                        help="Full detail for one GEM-E3 sector.")
    parser.add_argument("--validate", action="store_true",
                        help="Check names, partition and coverage.")
    parser.add_argument("--crosscheck", action="store_true",
                        help="Confront the mapping with Table 21 divisions.")
    parser.add_argument("--growth", nargs=2, metavar=("SCENARIO", "YEAR"),
                        default=None,
                        help="Project YoY growth rates onto EXIOBASE.")
    parser.add_argument("--region", default="World")
    parser.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    parser.add_argument("--year", type=int, default=2019,
                        help="EXIOBASE year for --validate.")
    parser.add_argument("--export", default=None,
                        help="Output path for the long mapping CSV.")
    parser.add_argument("--export-workbook", default=None,
                        help="Output path for the multi-sheet .xlsx.")
    args = parser.parse_args()

    if args.list:
        df = summary_table(args.table_type)
        with pd.option_context("display.max_rows", 100,
                               "display.max_colwidth", 45,
                               "display.width", 240):
            print(df.to_string(index=False))
        print(f"\n{len(df)} GEM-E3 sectors | "
              f"quality: {df['quality'].value_counts().to_dict()} | "
              f"provenance: {df['source'].value_counts().to_dict()} | "
              f"EXIOBASE {args.table_type} covered: {df['n_exio'].sum()}")

    elif args.detail:
        m = get_map(args.detail)
        print(f"\n=== {m.ngfs_sector} ({m.theme}, {m.quality}) ===")
        print(f"Provenance : {m.source}"
              + (f" (Table 21: '{m.table21_name}')" if m.table21_name else ""))
        print(f"NACE 2 ({len(m.nace2)}) : {' '.join(m.nace2) or '-'}")
        print(f"Divisions  : {' '.join(m.nace2_divisions()) or '-'}")
        if m.overlaps_with:
            print(f"Subsumed in: {m.overlaps_with}")
        if m.notes:
            print(f"\nJustification:\n  {m.notes}")
        for tt in ("pxp", "ixi"):
            lst = m.exio(tt)
            print(f"\nEXIOBASE {tt} ({len(lst)}):")
            for s in lst:
                print(f"  - {s}")

    elif args.validate:
        r = validate(args.year, args.table_type)
        print(f"\n=== Validation {r['table_type']} {r['year']} ===")
        print(f"EXIOBASE sectors        : {r['n_exio']}")
        print(f"EXIOBASE regions        : {r['n_exio_regions']}")
        print(f"GEM-E3 sectors mapped   : {r['n_ngfs_mapping']} "
              f"({r['n_table21']} anchored on Table 21)")
        print(f"Output coverage (x)     : {r['coverage_x']:.2%}")
        for label, key in (("Unknown EXIOBASE names", "unknown_exio"),
                           ("Unassigned EXIOBASE sectors", "unmapped_exio"),
                           ("GEM-E3 sectors missing from mapping", "missing_ngfs"),
                           ("Mapping sectors missing from GEM-E3", "extra_ngfs"),
                           ("Unknown Table 21 labels", "unknown_table21"),
                           ("Unresolved EXIOBASE regions", "unmapped_regions"),
                           ("Resolved to a non-existent GEM-E3 region",
                            "unknown_ngfs_regions")):
            v = r[key]
            print(f"\n{label} ({len(v)}):")
            for s in v:
                print(f"  ! {s}")
        print(f"\nEXIOBASE sectors assigned twice ({len(r['duplicated_exio'])}):")
        for s, owners in r["duplicated_exio"].items():
            print(f"  ! {s} -> {owners}")

    elif args.crosscheck:
        df = crosscheck_table21()
        n_bad = int((~df["coherent"]).sum())
        print(f"\nEXIOBASE pxp products: {len(df)} | "
              f"coherent with Table 21: {df['coherent'].sum()} | "
              f"mismatches: {n_bad}")
        print("\nMismatch reasons:\n"
              f"{df.loc[~df['coherent'], 'reason'].value_counts().to_string()}")
        with pd.option_context("display.max_rows", 300, "display.width", 240,
                               "display.max_colwidth", 50):
            print("\n" + df[~df["coherent"]].to_string(index=False))

    elif args.growth:
        scenario, year = args.growth
        s = growth_to_exiobase(scenario, year, region=args.region,
                               table_type=args.table_type)
        print(f"\n{s.name} - {len(s)} {args.table_type} sectors")
        print(s.sort_values().to_string())

    elif args.export:
        print(f"Saved: {export_mapping_csv(args.export, args.table_type)}")

    elif args.export_workbook:
        print(f"Saved: {export_workbook(args.export_workbook, args.table_type)}")

    else:
        parser.print_help()
