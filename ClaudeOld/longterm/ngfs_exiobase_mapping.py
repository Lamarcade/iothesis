"""
Mapping entre les extensions EXIOBASE et les variables NGFS Phase V.

Permet de comparer les valeurs historiques EXIOBASE (1995-2024) aux
trajectoires scénariques NGFS (2020-2100) sur les variables qui ont une
correspondance directe ou raisonnable.

Trois grands types de correspondances :
    1. **Directe** : même grandeur physique, juste un facteur d'échelle
       (ex : Emissions|CO2 [Mt CO2/yr] ↔ Σ stresseurs CO2 EXIOBASE [kg] × 1e-9).
    2. **Proxie** : approximation sémantique
       (ex : Water Withdrawal|Irrigation ≈ Water Consumption Blue Agriculture).
    3. **Non commensurable** : EXIOBASE en M€, NGFS en tonnes physiques
       — comparable seulement en croissance relative
       (ex : Production|Cement vs x du secteur Cement).

Scope :
    'world'  → EXIOBASE = somme toutes régions ; NGFS = région 'World'
    'europe' → EXIOBASE = somme EU27 ; NGFS = 'EU27' (downscaled) ou
                                              somme des agrégats Europe (IAM)
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_pipeline   import load_extension, load_io_system
from longterm.ngfs_data import load_ngfs

EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
        "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
        "NL", "PL", "PT", "RO", "SE", "SI", "SK"]


# ---------------------------------------------------------------------------
# Dataclasse de mapping
# ---------------------------------------------------------------------------

@dataclass
class Map:
    """
    Une correspondance entre une variable NGFS et un agrégat EXIOBASE.

    Attributes
    ----------
    key, theme, ngfs_variable, ngfs_unit : voir documentation NGFS
    exiobase_extension : extension EXIOBASE ('air_emissions', 'energy',
                         'land', 'water', 'material', 'labour',
                         'factor_inputs', 'nutrients') ou 'system' (= x, Y, Z)
    exiobase_stressors : liste de stresseurs (ou pattern 'X - *').
                         Vide si 'system'.
    exiobase_sectors   : list[str] | None — filtre par secteur acheteur dans F.
                         None = tous les secteurs (agrégat global).
                         Utilisé pour mapper aux secteurs NGFS (Industry|Cement, ...).
    exiobase_demand_categories : list[str] | None — filtre par catégorie F_Y.
                         None = toutes (ménages + gov + GFCF + ...).
                         Ex: ["Final consumption expenditure by households"]
                         pour Residential.
    include_F          : sommer les colonnes de F (production)
    include_F_Y        : sommer F_Y (demande finale directe)
    exiobase_unit      : unité native EXIOBASE
    to_ngfs_factor     : facteur exiobase_value → ngfs_unit
    quality            : 'direct' | 'proxy' | 'incommensurable'
    notes              : caveats / explication
    """
    key:                       str
    theme:                     str
    ngfs_variable:             str
    ngfs_unit:                 str
    exiobase_extension:        str
    exiobase_stressors:        list
    exiobase_unit:             str
    to_ngfs_factor:            float
    quality:                   str
    notes:                     str         = ""        # 10ᵉ arg positionnel
    exiobase_sectors:          list | None = None      # keyword-only en pratique
    exiobase_demand_categories: list | None = None
    include_F:                 bool        = True
    include_F_Y:               bool        = True


# ---------------------------------------------------------------------------
# MAPPING : table de correspondance
# ---------------------------------------------------------------------------

# Stresseurs précomptés (vérifiés sur EXIOBASE 2019 pxp)
_CO2_FOSSIL = [
    "CO2 - combustion - air",
    "CO2 - non combustion - Cement production - air",
    "CO2 - non combustion - Lime production - air",
    "CO2 - waste - fossil - air",
]
_CO2_ALL = _CO2_FOSSIL + [
    "CO2 - agriculture - peat decay - air",
    "CO2 - waste - biogenic - air",   # comptabilisé séparément en général
]
_CH4_ALL = [
    "CH4 - combustion - air",
    "CH4 - non combustion - Extraction/production of (natural) gas - air",
    "CH4 - non combustion - Extraction/production of crude oil - air",
    "CH4 - non combustion - Mining of antracite - air",
    "CH4 - non combustion - Mining of bituminous coal - air",
    "CH4 - non combustion - Mining of coking coal - air",
    "CH4 - non combustion - Mining of lignite (brown coal) - air",
    "CH4 - non combustion - Mining of sub-bituminous coal - air",
    "CH4 - non combustion - Oil refinery - air",
    "CH4 - agriculture - air",
    "CH4 - waste - air",
]
_CH4_ENERGY = [s for s in _CH4_ALL if "agriculture" not in s and "waste" not in s]
_CH4_AFOLU  = ["CH4 - agriculture - air"]

_N2O_ALL    = ["N2O - combustion - air", "N2O - agriculture - air"]
_N2O_AFOLU  = ["N2O - agriculture - air"]
_N2O_ENERGY = ["N2O - combustion - air"]

_NOX_ALL    = None   # détecté dynamiquement (23 stresseurs)
_SOX_ALL    = None   # détecté dynamiquement (21 SOx + SO2)
_LAND_CROPLAND = None  # tous "Cropland - *"
_LAND_PASTURE  = None  # tous "Permanent pastures - *"
_WATER_IRRIG   = None  # tous "Water Consumption Blue - Agriculture - *"
_WATER_BLUE_TOTAL = None
_MAT_FOSSILS   = None  # tous "Domestic Extraction Used - Fossil Fuels - *"


MAPPING: list[Map] = [
    # =====================================================================
    # 1) EMISSIONS (kg → Mt = 1e-9)
    # =====================================================================
    Map("CO2_fossil_IPCC", "Emissions",
        "Emissions|CO2|Energy and Industrial Processes", "Mt CO2/yr",
        "air_emissions", _CO2_FOSSIL,
        "kg", 1e-9, "direct",
        "Match standard FFI (Fossil Fuel + Industry). Exclut biogénique et peat decay."),

    Map("CO2_total", "Emissions",
        "Emissions|CO2", "Mt CO2/yr",
        "air_emissions", _CO2_ALL,
        "kg", 1e-9, "direct",
        "EXIOBASE = somme CO2 fossile + biogénique + peat. NGFS = total CO2 anthropique."),

    Map("CH4_total", "Emissions",
        "Emissions|CH4", "Mt CH4/yr",
        "air_emissions", _CH4_ALL,
        "kg", 1e-9, "direct",
        "Inclut combustion, mining/extraction fossile, agriculture, waste."),

    Map("CH4_energy", "Emissions",
        "Emissions|CH4|Energy", "Mt CH4/yr",
        "air_emissions", _CH4_ENERGY,
        "kg", 1e-9, "direct",
        "Fuites de CH4 du système énergétique (combustion + mining + extraction)."),

    Map("CH4_AFOLU", "Emissions",
        "Emissions|CH4|AFOLU", "Mt CH4/yr",
        "air_emissions", _CH4_AFOLU,
        "kg", 1e-9, "direct",
        "Méthane agricole (élevage, riz, etc.)."),

    Map("N2O_total", "Emissions",
        "Emissions|N2O", "kt N2O/yr",
        "air_emissions", _N2O_ALL,
        "kg", 1e-6, "direct",
        "Attention : NGFS en kt N2O/yr (pas Mt) → facteur 1e-6."),

    Map("N2O_AFOLU", "Emissions",
        "Emissions|N2O|AFOLU", "kt N2O/yr",
        "air_emissions", _N2O_AFOLU,
        "kg", 1e-6, "direct",
        "N2O agricole (fertilisants, sols)."),

    Map("N2O_energy", "Emissions",
        "Emissions|N2O|Energy", "kt N2O/yr",
        "air_emissions", _N2O_ENERGY,
        "kg", 1e-6, "direct",
        "N2O issu de la combustion."),

    # NOx / SOx — listes détectées dynamiquement (cf. _build_dynamic_lists)
    Map("NOx_total", "Emissions",
        "Emissions|NOx", "Mt NOx/yr",
        "air_emissions", "NOx - *",      # pattern : tous les stresseurs NOx
        "kg", 1e-9, "direct",
        "Somme tous NOx (combustion + procédés industriels)."),

    Map("Sulfur_total", "Emissions",
        "Emissions|Sulfur", "Mt SO2/yr",
        "air_emissions", ["SOx - *"],   # SOx = SO2 + SO3 + ... (inclut SO2)
        "kg", 1e-9, "direct",
        "Utilise SOx EXIOBASE (qui agrège SO2 + SO3 etc.). "
        "Ne pas ajouter SO2 - combustion qui est déjà dans SOx."),

    # NB : PM10 / PM2.5 / BC / OC retirés du mapping (NGFS Phase V ne les contient pas).
    # Pour ces données EXIOBASE seules, utiliser directement load_extension('air_emissions').

    Map("CO_total", "Emissions",
        "Emissions|CO", "Mt CO/yr",
        "air_emissions", ["CO - *"],
        "kg", 1e-9, "direct"),

    Map("NH3_total", "Emissions",
        "Emissions|NH3", "Mt NH3/yr",
        "air_emissions", ["NH3 - *"],
        "kg", 1e-9, "direct",
        "Ammonia (NH3) : combustion + agriculture."),

    Map("VOC_total", "Emissions",
        "Emissions|VOC", "Mt VOC/yr",
        "air_emissions", ["NMVOC - *"],
        "kg", 1e-9, "direct",
        "NMVOC EXIOBASE ↔ VOC NGFS (NMVOC = Non Methane VOC)."),

    # =====================================================================
    # 2) ÉNERGIE (TJ → EJ = 1e-6)
    # =====================================================================
    Map("FinalEnergy", "Energy",
        "Final Energy", "EJ/yr",
        "energy", ["Energy use - Final"],
        "TJ", 1e-6, "proxy",
        "ATTENTION : extension 'energy' EXIOBASE 3.10.1 pxp a F=0 (énergie "
        "uniquement dans F_Y, soit ~part ménages). Sous-estime largement "
        "vs NGFS ~400 EJ. Pour le total IEA-comparable, utiliser ixi ou "
        "agréger directement les outputs sectoriels Electricity/Gas/Oil."),

    Map("PrimaryEnergy", "Energy",
        "Primary Energy", "EJ/yr",
        "energy", ["Energy use - Gross"],
        "TJ", 1e-6, "proxy",
        "Idem caveat FinalEnergy : ~part ménages seulement en pxp."),

    # =====================================================================
    # 3) LAND COVER (km² → million ha = 1e-4)
    # =====================================================================
    Map("Land_Cropland", "Land",
        "Land Cover|Cropland", "million ha",
        "land", "Cropland - *",         # pattern
        "km2", 1e-4, "direct",
        "Inclut cropped + fallowed (toutes cultures)."),

    Map("Land_Forest", "Land",
        "Land Cover|Forest", "million ha",
        "land", ["Forest"],
        "km2", 1e-4, "direct"),

    Map("Land_Pasture", "Land",
        "Land Cover|Pasture", "million ha",
        "land", "Permanent pastures - *",
        "km2", 1e-4, "direct",
        "Permanent pastures (Grazing-Cattle, Meat animals, Raw milk)."),

    Map("Land_BuiltUp", "Land",
        "Land Cover|Built-up Area", "million ha",
        "land", ["Artificial Surfaces"],
        "km2", 1e-4, "direct"),

    # =====================================================================
    # 4) WATER (Mm³ → km³ = 1e-3)
    # =====================================================================
    Map("Water_Irrigation", "Water",
        "Water Withdrawal|Irrigation", "km3/yr",
        "water", "Water Consumption Blue - Agriculture - *",
        "Mm3", 1e-3, "proxy",
        "NGFS = Withdrawal (prélèvement) ; EXIOBASE = Consumption "
        "(prélèvement − retour). Pour irrigation, les deux sont proches."),

    # =====================================================================
    # 5) PRODUCTION (M€ EXIOBASE vs Mt NGFS — non commensurable)
    # =====================================================================
    Map("Production_Cement", "Production",
        "Production|Cement", "Mt/yr",
        "system", ["Cement, lime and plaster"],
        "M.EUR", float("nan"), "incommensurable",
        "EXIOBASE = x sectoriel en M€, NGFS = volume physique en Mt. "
        "Comparer en croissance relative ou via prix moyen ~100 €/t."),

    Map("Production_Steel", "Production",
        "Production|Steel", "Mt/yr",
        "system",
        ["Basic iron and steel and of ferro-alloys and first products thereof"],
        "M.EUR", float("nan"), "incommensurable",
        "Idem (prix moyen ~600 €/t)."),

    Map("Production_Chemicals", "Production",
        "Production|Chemicals", "Mt/yr",
        "system", ["Chemicals nec"],
        "M.EUR", float("nan"), "incommensurable",
        "EXIOBASE secteur 'Chemicals nec' agrège énormément."),

    # =====================================================================
    # 6) MACRO (GDP, Population) — non commensurable strictement
    # =====================================================================
    Map("GDP", "Macro",
        "GDP|MER|Counterfactual without damage", "billion US$2010/yr",
        "system", ["VA_total"],     # marqueur : sum(x) - sum(Z col)
        "M.EUR", float("nan"), "incommensurable",
        "EXIOBASE VA = Σ x - Σ Z par colonne (M€ courant). "
        "NGFS = USD2010. Conversion EUR→USD + déflation requise."),

    # Population retirée : EXIOBASE n'a que de l'emploi (∼ pop active employée),
    # ce n'est pas la population totale. Utiliser une source démographique
    # externe (UN WPP, World Bank) si besoin.

    # =====================================================================
    # 7) RESSOURCES (matières premières)
    # =====================================================================
    Map("Material_FossilFuels", "Resources",
        "Primary Energy|Fossil", "EJ/yr",
        "material", "Domestic Extraction Used - Fossil Fuels - *",
        "kt", float("nan"), "incommensurable",
        notes="EXIOBASE en tonnage extrait ; NGFS en contenu énergétique. "
              "Conversion possible via PCI moyens (charbon ~28 GJ/t, "
              "pétrole ~42 GJ/t, gaz ~38 MJ/m³)."),

    # =====================================================================
    # 8) SECTORAL — Emissions CO2 par secteur industriel (NGFS Industry|*)
    # =====================================================================
    # Ces mappings utilisent exiobase_sectors pour filtrer les colonnes de F.
    # On utilise seulement F (production) — pas F_Y (qui n'a pas de secteur).
    Map("Sec_CO2_Industry_Cement", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Industry|Cement", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "direct",
        exiobase_sectors=["Cement, lime and plaster"],
        include_F_Y=False,
        notes="NGFS Industry|Cement = combustion énergétique seulement (~40% du CO2 cement). "
              "EXIOBASE 'CO2 - non combustion - Cement production' = process (calcination, "
              "~60%) qui n'est pas dans NGFS Energy|Demand."),

    Map("Sec_CO2_Industry_Steel", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Industry|Steel", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "direct",
        exiobase_sectors=["Basic iron and steel and of ferro-alloys and first products thereof"],
        include_F_Y=False,
        notes="NGFS = combustion (Energy|Demand). Process steel (réduction directe avec coke) "
              "n'est pas inclus côté NGFS Industry|Steel."),

    Map("Sec_CO2_Industry_Chemicals", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Industry|Chemicals", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "direct",
        exiobase_sectors=["Chemicals nec"],
        include_F_Y=False,
        notes="EXIOBASE 'Chemicals nec' agrège tous chimiques. Combustion seulement."),

    Map("Sec_CO2_Industry_NonFerrous", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Industry|Non-ferrous metals", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "direct",
        exiobase_sectors=[
            "Aluminium and aluminium products",
            "Copper products",
            "Lead, zinc and tin and products thereof",
            "Other non-ferrous metal products",
        ],
        include_F_Y=False),

    Map("Sec_CO2_Industry_Total", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Industry", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "proxy",
        exiobase_sectors=None,    # surchargé via INDUSTRY_SECTORS au runtime
        include_F_Y=False,
        notes="Total Industry NGFS vs secteurs industriels EXIOBASE (liste). "
              "Combustion seulement."),

    # --- Transportation (combustion seulement) ---
    Map("Sec_CO2_Transp_Total", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Transportation", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "proxy",
        exiobase_sectors=[
            "Railway transportation services",
            "Other land transportation services",
            "Transportation services via pipelines",
            "Sea and coastal water transportation services",
            "Inland water transportation services",
            "Air transport services (62)",
            "Supporting and auxiliary transport services; travel agency services (63)",
        ],
        include_F_Y=False,
        notes="EXIOBASE transport entreprises (combustion). NGFS Transport total inclut "
              "aussi VP particuliers — à ajouter via F_Y si voulu."),

    Map("Sec_CO2_Transp_Aviation", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Transportation|Aviation", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "direct",
        exiobase_sectors=["Air transport services (62)"],
        include_F_Y=False),

    Map("Sec_CO2_Transp_Maritime", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Transportation|Maritime|Freight", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "proxy",
        exiobase_sectors=[
            "Sea and coastal water transportation services",
            "Inland water transportation services",
        ],
        include_F_Y=False),

    Map("Sec_CO2_Transp_Rail", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Transportation|Rail", "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "direct",
        exiobase_sectors=["Railway transportation services"],
        include_F_Y=False),

    # --- Industrial Processes (non-combustion) ---
    # NGFS sépare Energy|Demand (combustion) et Industrial Processes (chimie/réduction).
    # EXIOBASE CO2 - non combustion - Cement production = calcination (~60% des CO2 ciment).
    Map("Sec_CO2_IP_Cement", "Sectoral Emissions",
        "Emissions|CO2|Industrial Processes|Cement", "Mt CO2/yr",
        "air_emissions", ["CO2 - non combustion - Cement production - air"],
        "kg", 1e-9, "direct",
        exiobase_sectors=["Cement, lime and plaster"],
        include_F_Y=False,
        notes="Process calcination CaCO3 -> CaO + CO2 (distinct de la combustion). "
              "Complete Sec_CO2_Industry_Cement (combustion)."),

    # --- Residential and Commercial ---
    # NGFS R&C = émissions directes consommation finale (chauffage, cuisson, ECS, VP)
    # EXIOBASE équivalent = F_Y "Final consumption expenditure by households"
    Map("Sec_CO2_ResComm_Residential", "Sectoral Emissions",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Residential",
        "Mt CO2/yr",
        "air_emissions", ["CO2 - combustion - air"],
        "kg", 1e-9, "proxy",
        include_F=False,
        include_F_Y=True,
        exiobase_demand_categories=[
            "Final consumption expenditure by households",
        ],
        notes="EXIOBASE = CO2 direct des ménages (chauffage, voitures, etc.). "
              "Inclut transport individuel (NGFS sépare ces deux)."),

    # =====================================================================
    # 9) SECTORAL — Final Energy par secteur
    # =====================================================================
    # NB : les mappings Sec_FE_Industry_* et Sec_FE_Transp_* ont été retirés
    #      car en pxp l'extension `energy` a F=0 (tout est en F_Y, càd la
    #      consommation finale des ménages — pas la production sectorielle).
    #      Pour ces séries, il faudrait passer à ixi (mais les noms de
    #      secteurs sectoriels seraient différents — refonte du mapping).
    #      Sec_FE_ResComm reste valide car il utilise F_Y (ménages) seul.

    Map("Sec_FE_ResComm", "Sectoral Energy",
        "Final Energy|Residential and Commercial", "EJ/yr",
        "energy", ["Energy use - Final"],
        "TJ", 1e-6, "proxy",
        include_F=False,
        include_F_Y=True,
        exiobase_demand_categories=[
            "Final consumption expenditure by households",
        ]),

    # =====================================================================
    # 10) SECTORAL — Électricité par source
    #     EXIOBASE x[secteur] en M€ vs NGFS en EJ/yr ou GW → incommensurable
    #     Sauf si on multiplie par un prix moyen ou par les facteurs heures*GW
    # =====================================================================
    Map("Sec_SE_Elec_Coal", "Electricity Mix",
        "Secondary Energy|Electricity|Coal", "EJ/yr",
        "system", ["Electricity by coal"],
        "M.EUR", float("nan"), "incommensurable",
        notes="EXIOBASE x[Elec by coal] (M€) vs NGFS génération électrique (EJ/yr). "
              "Comparable en croissance relative."),

    Map("Sec_SE_Elec_Gas", "Electricity Mix",
        "Secondary Energy|Electricity|Gas", "EJ/yr",
        "system", ["Electricity by gas"],
        "M.EUR", float("nan"), "incommensurable"),

    Map("Sec_SE_Elec_Oil", "Electricity Mix",
        "Secondary Energy|Electricity|Oil", "EJ/yr",
        "system", ["Electricity by petroleum and other oil derivatives"],
        "M.EUR", float("nan"), "incommensurable"),

    Map("Sec_SE_Elec_Nuclear", "Electricity Mix",
        "Secondary Energy|Electricity|Nuclear", "EJ/yr",
        "system", ["Electricity by nuclear"],
        "M.EUR", float("nan"), "incommensurable"),

    Map("Sec_SE_Elec_Hydro", "Electricity Mix",
        "Secondary Energy|Electricity|Hydro", "EJ/yr",
        "system", ["Electricity by hydro"],
        "M.EUR", float("nan"), "incommensurable"),

    Map("Sec_SE_Elec_Wind", "Electricity Mix",
        "Secondary Energy|Electricity|Wind", "EJ/yr",
        "system", ["Electricity by wind"],
        "M.EUR", float("nan"), "incommensurable"),

    Map("Sec_SE_Elec_Solar", "Electricity Mix",
        "Secondary Energy|Electricity|Solar", "EJ/yr",
        "system", [
            "Electricity by solar photovoltaic",
            "Electricity by solar thermal",
        ],
        "M.EUR", float("nan"), "incommensurable",
        notes="EXIOBASE sépare PV / thermal solaire (somme = total Solar)."),

    Map("Sec_SE_Elec_Biomass", "Electricity Mix",
        "Secondary Energy|Electricity|Biomass", "EJ/yr",
        "system", ["Electricity by biomass and waste"],
        "M.EUR", float("nan"), "incommensurable"),

    Map("Sec_SE_Elec_Geothermal", "Electricity Mix",
        "Secondary Energy|Electricity|Geothermal", "EJ/yr",
        "system", ["Electricity by Geothermal"],
        "M.EUR", float("nan"), "incommensurable"),
]


# Liste explicite des secteurs Industry EXIOBASE pour Sec_CO2_Industry_Total
INDUSTRY_SECTORS_EXIOBASE = [
    "Cement, lime and plaster",
    "Basic iron and steel and of ferro-alloys and first products thereof",
    "Chemicals nec",
    "Aluminium and aluminium products",
    "Copper products",
    "Lead, zinc and tin and products thereof",
    "Other non-ferrous metal products",
    "Pulp",
    "Paper and paper products",
    "Glass and glass products",
    "Ceramic goods",
    "Bricks, tiles and construction products, in baked clay",
    "Fabricated metal products, except machinery and equipment (28)",
    "Machinery and equipment n.e.c. (29)",
    "Plastics, basic",
    "N-fertiliser",
    "P- and other fertiliser",
]
# Patch : Sec_CO2_Industry_Total reçoit cette liste explicite
for _m in MAPPING:
    if _m.key == "Sec_CO2_Industry_Total":
        _m.exiobase_sectors = INDUSTRY_SECTORS_EXIOBASE
        break


# ---------------------------------------------------------------------------
# Résolution des patterns "PREFIX - *"
# ---------------------------------------------------------------------------

def _resolve_stressors(stressors, extension_index: pd.Index) -> list[str]:
    """
    Résout `stressors` (str | list[str]) en liste de noms exacts présents
    dans extension_index. Supporte les patterns 'PREFIX - *'.
    """
    if isinstance(stressors, str):
        stressors = [stressors]
    out = []
    for item in stressors:
        if isinstance(item, str) and item.endswith(" - *"):
            prefix = item[:-2]    # garde 'PREFIX - '
            matches = [s for s in extension_index if str(s).startswith(prefix)]
            out.extend(matches)
        elif item in extension_index:
            out.append(item)
        # sinon : silently skipped (peut être absent en ixi par ex.)
    return list(dict.fromkeys(out))  # dédupliqué en gardant l'ordre


# ---------------------------------------------------------------------------
# Aggregation EXIOBASE
# ---------------------------------------------------------------------------

def aggregate_exiobase(
    m: Map,
    year: int,
    table_type: str = "pxp",
    scope: str = "world",
) -> tuple[float, dict]:
    """
    Calcule la valeur EXIOBASE agrégée pour un mapping donné, dans l'unité NGFS.

    Returns
    -------
    value : float (dans l'unité NGFS, sauf si quality='incommensurable' →
            renvoie la valeur native sans conversion)
    info  : dict avec stressors résolus, n_regions, etc.
    """
    # --- Cas "system" : utilise Z, x, Y plutôt qu'une extension ---
    if m.exiobase_extension == "system":
        sys = load_io_system(year, table_type)
        x   = sys["x"]
        Z   = sys["Z"]

        # Sélection régions
        region_mask = (x.index.get_level_values("region").isin(EU27)
                       if scope == "europe" else slice(None))

        if m.exiobase_stressors == ["VA_total"]:
            # Valeur ajoutée = sum(x) - sum(Z par colonne)
            x_sel  = x.loc[region_mask] if scope == "europe" else x
            Z_cols = Z.sum(axis=0)        # somme par colonne (intermédiaires achetés)
            if scope == "europe":
                Z_cols = Z_cols.loc[region_mask]
            value = float(x_sel.sum() - Z_cols.sum())
        else:
            # Production sectorielle x[secteur]
            sector_mask = x.index.get_level_values("sector").isin(m.exiobase_stressors)
            sel = x[sector_mask]
            if scope == "europe":
                sel = sel.loc[sel.index.get_level_values("region").isin(EU27)]
            value = float(sel.sum())

        return value, {
            "stressors_resolved": m.exiobase_stressors,
            "scope":             scope,
            "year":              year,
            "unit_native":       m.exiobase_unit,
        }

    # --- Cas extension ---
    ext = load_extension(year, m.exiobase_extension, table_type)
    F   = ext["F"]
    F_Y = ext["F_Y"]
    resolved = _resolve_stressors(m.exiobase_stressors, F.index)
    if not resolved:
        return float("nan"), {"warning": "Aucun stresseur résolu",
                                "stressors_requested": m.exiobase_stressors}

    # ----- F (production) -----
    value_F = 0.0
    if m.include_F:
        sub_F = F.loc[resolved]
        # Filtre par secteur acheteur (NGFS sectoral mapping)
        if m.exiobase_sectors is not None:
            sec_mask = sub_F.columns.get_level_values("sector").isin(m.exiobase_sectors)
            sub_F = sub_F.loc[:, sec_mask]
        # Filtre par région
        if scope == "europe":
            sub_F = sub_F.loc[:, sub_F.columns.get_level_values("region").isin(EU27)]
        value_F = float(sub_F.values.sum())

    # ----- F_Y (demande finale directe) -----
    # F_Y n'a pas de dimension "secteur" ; il a (region × category)
    # On l'inclut seulement si include_F_Y=True et qu'on n'a pas filtré par sector
    # (sauf si c'est explicitement Resi&Com → demand_categories spécifique).
    value_F_Y = 0.0
    if m.include_F_Y and m.exiobase_sectors is None:
        sub_F_Y = F_Y.loc[F_Y.index.intersection(resolved)]
        if m.exiobase_demand_categories is not None:
            cat_mask = sub_F_Y.columns.get_level_values(1).isin(m.exiobase_demand_categories)
            sub_F_Y = sub_F_Y.loc[:, cat_mask]
        if scope == "europe":
            sub_F_Y = sub_F_Y.loc[:, sub_F_Y.columns.get_level_values(0).isin(EU27)]
        value_F_Y = float(sub_F_Y.values.sum()) if not sub_F_Y.empty else 0.0

    value_native = value_F + value_F_Y
    value_ngfs   = (value_native * m.to_ngfs_factor
                    if not np.isnan(m.to_ngfs_factor) else value_native)

    return value_ngfs, {
        "stressors_resolved": resolved,
        "n_stressors":        len(resolved),
        "n_sectors_filter":   len(m.exiobase_sectors) if m.exiobase_sectors else "all",
        "scope":              scope,
        "year":               year,
        "value_F":            value_F,
        "value_F_Y":          value_F_Y,
        "value_native":       value_native,
        "unit_native":        m.exiobase_unit,
    }


# ---------------------------------------------------------------------------
# Aggregation NGFS
# ---------------------------------------------------------------------------

def aggregate_ngfs(
    m: Map,
    year: int,
    scope: str = "world",
    scenario: str | None = None,
    model: str | None = None,
    source: str | None = None,
    sum_regions: bool = True,
) -> pd.DataFrame:
    """
    Récupère la valeur NGFS pour une variable, à une année donnée.

    Parameters
    ----------
    sum_regions : bool, default True
        Si plusieurs régions matchent le scope pour un même (source, Model,
        Scenario) — ex. GCAM EU-12 + EU-15 = EU27 — somme les valeurs.
        Si False, renvoie une ligne par région.

    Returns
    -------
    pd.DataFrame : colonnes [source, Model, Scenario, Region, Year, value, Unit]
                   (Region = liste agrégée si sum_regions=True)
    """
    if source is None:
        sources = list(load_ngfs(scope=scope).keys())
    elif isinstance(source, str):
        sources = [source]
    else:
        sources = list(source)

    rows = []
    for src in sources:
        data = load_ngfs(sources=src, scope=scope)
        df = data[src]
        sub = df[df["Variable"] == m.ngfs_variable]
        if scenario:
            sub = sub[sub["Scenario"] == scenario]
        if model:
            sub = sub[sub["Model"] == model]
        if sub.empty:
            continue
        year_col = str(year)
        if year_col not in sub.columns:
            continue
        for _, r in sub.iterrows():
            v = r[year_col]
            if pd.isna(v):
                continue
            rows.append({
                "source":   src,
                "Model":    r["Model"],
                "Scenario": r["Scenario"],
                "Region":   r["Region"],
                "Year":     year,
                "value":    float(v),
                "Unit":     r["Unit"],
            })

    df_long = pd.DataFrame(rows)
    if df_long.empty or not sum_regions:
        return df_long

    # Aggrege les regions au sein d'un meme (source, Model, Scenario)
    # (utile pour GCAM EU-12 + EU-15 = EU27)
    agg = df_long.groupby(
        ["source", "Model", "Scenario", "Year", "Unit"], as_index=False
    ).agg(
        value=("value",  "sum"),
        Region=("Region", lambda x: " + ".join(sorted(set(x)))),
    )
    return agg[["source", "Model", "Scenario", "Region", "Year", "value", "Unit"]]


# ---------------------------------------------------------------------------
# Helpers de haut niveau
# ---------------------------------------------------------------------------

def get_mapping(key: str) -> Map:
    """Retourne le Map avec la clé donnée."""
    for m in MAPPING:
        if m.key == key:
            return m
    raise KeyError(f"Mapping introuvable : {key!r}. "
                    f"Choices : {[m.key for m in MAPPING]}")


def list_mapping(theme: str | None = None) -> pd.DataFrame:
    """Tableau récapitulatif de tous les mappings (filtrable par thème)."""
    rows = []
    for m in MAPPING:
        if theme and m.theme != theme:
            continue
        rows.append({
            "key":         m.key,
            "theme":       m.theme,
            "quality":     m.quality,
            "ngfs":        m.ngfs_variable,
            "ngfs_unit":   m.ngfs_unit,
            "exiobase":    m.exiobase_extension,
            "exio_unit":   m.exiobase_unit,
            "factor":      m.to_ngfs_factor,
            "n_stress":    (len(m.exiobase_stressors)
                            if isinstance(m.exiobase_stressors, list)
                            else "pattern"),
        })
    return pd.DataFrame(rows)


def compare(
    key: str,
    year: int = 2020,
    scope: str = "world",
    scenario: str | None = None,
    table_type: str = "pxp",
) -> dict:
    """
    Comparaison complète EXIOBASE ↔ NGFS pour un mapping donné.

    Returns
    -------
    dict : {
        'mapping', 'year', 'scope', 'scenario',
        'exiobase_value', 'exiobase_info',
        'ngfs_values' (DataFrame),
        'ratio_exio_over_ngfs' (None si non commensurable)
    }
    """
    m = get_mapping(key)
    exio_val, exio_info = aggregate_exiobase(m, year, table_type, scope)
    ngfs_df = aggregate_ngfs(m, year, scope, scenario=scenario)

    ratio = None
    if m.quality != "incommensurable" and not ngfs_df.empty:
        ngfs_median = float(ngfs_df["value"].median())
        if ngfs_median != 0:
            ratio = exio_val / ngfs_median

    return {
        "mapping":               m,
        "year":                  year,
        "scope":                 scope,
        "scenario":              scenario,
        "exiobase_value":        exio_val,
        "exiobase_info":         exio_info,
        "ngfs_values":           ngfs_df,
        "ratio_exio_over_ngfs":  ratio,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mapping NGFS ↔ EXIOBASE : liste ou comparaison."
    )
    parser.add_argument("--list", action="store_true",
                         help="Liste tous les mappings disponibles.")
    parser.add_argument("--theme", default=None,
                         help="Filtre par thème (Emissions, Energy, Land, ...).")
    parser.add_argument("--compare", default=None,
                         help="Clé du mapping à comparer (ex: 'CO2_fossil_IPCC').")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--scope", choices=["world", "europe"], default="world")
    parser.add_argument("--scenario", default=None,
                         help="Filtre scénario NGFS (ex: 'Net Zero 2050').")
    parser.add_argument("--all", action="store_true",
                         help="Comparaison de tous les mappings 'direct'.")

    # Trajectoires
    parser.add_argument("--trajectory", default=None,
                         help="Clé du mapping pour générer une trajectoire HTML.")
    parser.add_argument("--all-trajectories", action="store_true",
                         help="Génère 1 HTML par mapping (direct+proxy) + dashboard.")
    parser.add_argument("--exio-years", nargs=2, type=int, metavar=("START", "END"),
                         default=[2010, 2022],
                         help="Plage années EXIOBASE (default 2010 2022 ; "
                              "2023-2024 sont des nowcasts partiels pour les extensions).")
    parser.add_argument("--ngfs-years", nargs=2, type=int, metavar=("START", "END"),
                         default=[2020, 2030],
                         help="Plage années NGFS (default 2020 2030).")
    parser.add_argument("--scenarios", nargs="+", default=None,
                         help="Filtre scénarios NGFS (sinon les 7).")
    parser.add_argument("--themes", nargs="+", default=None,
                         help="Filtre thèmes pour --all-trajectories.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "Figs" / "trajectories"),
                         help="Dossier de sortie HTML (default Figs/trajectories).")
    parser.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    parser.add_argument("--force-reload", action="store_true",
                         help="Bypass le cache CSV des trajectoires EXIOBASE.")
    parser.add_argument("--mode", choices=["auto", "absolute", "normalized"],
                         default="auto",
                         help="Mode plot (auto=normalisé pour incommensurables).")
    parser.add_argument("--normalize-year", type=int, default=2020,
                         help="Année de référence pour mode normalisé (default 2020).")
    parser.add_argument("--export-mapping", default=None,
                         help="Génère uniquement le CSV des mappings à ce chemin "
                              "(ex. mapping_table.csv) et termine.")

    args = parser.parse_args()

    if args.list:
        df = list_mapping(theme=args.theme)
        with pd.option_context("display.max_rows", 100,
                                "display.max_colwidth", 60,
                                "display.width", 200):
            print(df.to_string(index=False))
        print(f"\n{len(df)} mappings (qualité : "
              f"{df['quality'].value_counts().to_dict()})")

    elif args.compare:
        result = compare(args.compare, year=args.year, scope=args.scope,
                         scenario=args.scenario)
        m = result["mapping"]
        print(f"\n=== {m.key} ({m.theme}, {m.quality}) ===")
        print(f"NGFS variable : {m.ngfs_variable} [{m.ngfs_unit}]")
        print(f"EXIOBASE      : {m.exiobase_extension} / "
              f"{result['exiobase_info'].get('n_stressors', 'system')} stresseurs")
        print(f"Notes         : {m.notes}")
        print()
        print(f"Année {args.year} | scope {args.scope}"
              + (f" | scenario={args.scenario}" if args.scenario else ""))
        print(f"  EXIOBASE  : {result['exiobase_value']:.4e} {m.ngfs_unit}")
        if not result["ngfs_values"].empty:
            print(f"  NGFS (n={len(result['ngfs_values'])}) :")
            with pd.option_context("display.max_rows", 30, "display.width", 200):
                print(result["ngfs_values"].to_string(index=False))
            if result["ratio_exio_over_ngfs"] is not None:
                print(f"  Ratio EXIO / NGFS (median) : "
                       f"{result['ratio_exio_over_ngfs']:.3f}")
        else:
            print("  NGFS : aucun match")

    elif args.all:
        rows = []
        for m in MAPPING:
            if m.quality != "direct":
                continue
            try:
                result = compare(m.key, year=args.year, scope=args.scope,
                                  scenario=args.scenario)
            except Exception as e:
                rows.append({"key": m.key, "error": str(e)[:80]})
                continue
            ng = result["ngfs_values"]
            ngfs_median = float(ng["value"].median()) if not ng.empty else None
            rows.append({
                "key":         m.key,
                "theme":       m.theme,
                "exio":        result["exiobase_value"],
                "ngfs_median": ngfs_median,
                "n_ngfs":      len(ng),
                "ratio":       result["ratio_exio_over_ngfs"],
                "unit":        m.ngfs_unit,
            })
        df = pd.DataFrame(rows)
        with pd.option_context("display.float_format", "{:.3e}".format,
                                "display.width", 200):
            print(df.to_string(index=False))

    elif args.export_mapping:
        from longterm.ngfs_exiobase_trajectories import export_mapping_table
        export_mapping_table(args.export_mapping)

    elif args.trajectory:
        from longterm.ngfs_exiobase_trajectories import plot_trajectory
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{args.trajectory}_{args.scope}.html"
        plot_trajectory(
            args.trajectory, scope=args.scope,
            exio_years=tuple(args.exio_years),
            ngfs_years=tuple(args.ngfs_years),
            scenarios=args.scenarios,
            table_type=args.table_type,
            save_path=str(save_path),
            force_reload=args.force_reload,
            mode=args.mode,
            normalize_at_year=args.normalize_year,
        )

    elif args.all_trajectories:
        from longterm.ngfs_exiobase_trajectories import run_all_trajectories
        run_all_trajectories(
            scope=args.scope,
            output_dir=args.output_dir,
            exio_years=tuple(args.exio_years),
            ngfs_years=tuple(args.ngfs_years),
            themes=args.themes,
            table_type=args.table_type,
            force_reload=args.force_reload,
            mode=args.mode,
            normalize_at_year=args.normalize_year,
        )

    else:
        parser.print_help()
