# Hypothèses du mapping NGFS ↔ EXIOBASE

Document de référence détaillant les choix de modélisation, conventions et limites
de la correspondance entre les variables NGFS Phase V et les extensions
environnementales d'EXIOBASE 3.10.1.

À lire en complément de `Data/mappings/mapping_table.csv` (export auto via
`python longterm/ngfs_exiobase_mapping.py --export-mapping Data/mappings/mapping_table.csv`).

---

## 1. Cadre général

| Choix | Valeur | Justification |
|---|---|---|
| Type de table EXIOBASE | **pxp** (product × product) | Sectoral names (Cement, Steel, Electricity by coal) sont des produits → présents en pxp uniquement |
| Année de référence (validation) | **2019 pxp** | Dernière année avec extensions complètes (2020 OK aussi) |
| Horizon historique EXIOBASE | **2010-2022** par défaut | 2023-2024 sont des nowcasts partiels (air_emissions F ÷10, F_Y=0) |
| Horizon projection NGFS | **2020-2030** par défaut | Court terme — fenêtre transition la plus utile |
| Scopes | `world` ou `europe` | Voir §2.3 |

---

## 2. Hypothèses techniques

### 2.1 Sommation F + F_Y (extension EXIOBASE)

- **F** : matrice émissions production-based par secteur producteur (région × secteur)
- **F_Y** : matrice émissions directes de la demande finale (région × catégorie : ménages, gov, GFCF…)
- **Par défaut** : on somme `F + F_Y` (couvre tout)
- **Si `exiobase_sectors` filtré** : on ignore F_Y (F_Y n'a pas de dimension secteur)
- **Si `exiobase_demand_categories` filtré** : on filtre F_Y par catégorie (ex. `Final consumption expenditure by households`)
- **Flags par mapping** : `include_F` (default True), `include_F_Y` (default True)

### 2.2 Conversion d'unités EXIOBASE → NGFS

| Grandeur | EXIOBASE | NGFS | Facteur |
|---|---|---|---|
| Émissions GHG/polluants | kg | Mt | ×1e-9 |
| Émissions GHG (N2O) | kg | kt | ×1e-6 |
| Énergie | TJ | EJ | ×1e-6 |
| Surface | km² | million ha | ×1e-4 |
| Eau | Mm³ | km³ | ×1e-3 |
| Production sectorielle | M€ courant | Mt | **incommensurable** (mode normalisé) |
| GDP | M€ courant | B USD2010 réel | **incommensurable** (mode normalisé) |

### 2.3 Scope géographique

| Scope | EXIOBASE | NGFS |
|---|---|---|
| `world` | Σ toutes régions (49) | "World" |
| `europe` | Σ EU27 (27 ISO3) | GCAM `EU-12 + EU-15` = EU27 ; REMIND `EU 28` (≈ EU27 + UK) ; Downscaled `EU27` strict ; NiGEM `Europe` (proche EU27) ; MESSAGE omis (pas d'agrégat EU27 propre) |

NGFS `sum_regions=True` somme automatiquement les régions au sein d'un même (source, Model, Scenario) — GCAM EU-12 + EU-15 → GCAM EU27.

### 2.4 Agrégation inter-modèles NGFS

- Pour chaque scénario, on calcule la **médiane** des 3 IAMs (GCAM, MESSAGE, REMIND)
- La **bande min-max** sur le plot représente l'incertitude inter-modèles
- 7 scénarios couvert : Below 2°C, Current Policies, Delayed transition, Fragmented World, Low demand, NDCs, Net Zero 2050

### 2.5 Mode normalisé (variables incommensurables)

- Activé automatiquement pour `quality="incommensurable"` (GDP, Production, Electricity Mix)
- Chaque série (EXIOBASE + chaque NGFS scenario médiane) divisée par sa valeur en année de référence
- **Base = 2020** par défaut (année de chevauchement)
- Y-axis : "indice (base 2020 = 1)"
- Limitation : ne compare que les **taux de croissance**, pas les niveaux absolus

---

## 3. Hypothèses par catégorie

### 3.1 Émissions agrégées (GHG + polluants)

| Mapping | Hypothèse |
|---|---|
| `CO2_fossil_IPCC` | EXIOBASE = combustion + cement process + lime process + waste fossile. Match standard IPCC "FFI" |
| `CO2_total` | + bio + peat decay (≠ NGFS qui inclut land use side) |
| `CH4_total` | combustion + extraction/refining fossile + agri + waste |
| `CH4_energy` | combustion + tous les "non combustion" fossiles |
| `CH4_AFOLU` | seul `CH4 - agriculture - air` (élevage, riz, fumier) |
| `N2O_total` | combustion + agriculture (kt N2O/yr, ×1e-6) |
| `N2O_AFOLU` | seul `N2O - agriculture - air` |
| `N2O_energy` | seul `N2O - combustion - air` |
| `NOx_total` | tous "NOx - *" (combustion + 22 procédés industriels) |
| `Sulfur_total` | tous "SOx - *" SEUL (SOx inclut SO2 + SO3 + …). Ne PAS ajouter "SO2 - *" → double comptage |
| `NH3_total` | tous "NH3 - *" (combustion + agriculture). Ratio NGFS ~1.6× car EXIOBASE plus large côté agri |
| `VOC_total` | NMVOC EXIOBASE = Non Methane VOC, équivalent NGFS VOC |
| `CO_total` | tous "CO - *". Ratio ~0.6 car EXIOBASE manque CO non-combustion |

### 3.2 Émissions sectorielles (Sectoral Emissions)

**Hypothèse fondamentale** : NGFS `Industry|X` est sous `Energy|Demand`, donc représente la **combustion énergétique** dans le secteur X (pas le process).

Pour matcher correctement :
- **Combustion seule** : stresseur = `["CO2 - combustion - air"]` (pas `_CO2_FOSSIL` qui inclut cement/lime process)
- **Filtre secteur acheteur** via `exiobase_sectors`
- `include_F_Y=False` (F_Y n'a pas de dimension secteur)

| Mapping | NGFS variable | EXIOBASE secteur(s) | Ratio 2020 World |
|---|---|---|---|
| `Sec_CO2_Industry_Cement` | `Industry\|Cement` (combustion only) | "Cement, lime and plaster" | **0.59** |
| `Sec_CO2_IP_Cement` (**nouveau**) | `Industrial Processes\|Cement` (calcination) | idem, stressor=non combustion Cement production | à vérifier |
| `Sec_CO2_Industry_Steel` | `Industry\|Steel` | "Basic iron and steel..." | 0.89 ✓ |
| `Sec_CO2_Industry_Chemicals` | `Industry\|Chemicals` | "Chemicals nec" | 0.16 ⚠ |
| `Sec_CO2_Industry_NonFerrous` | `Industry\|Non-ferrous metals` | Aluminium + Copper + Lead/zinc/tin + autres | 3.03 ⚠ |
| `Sec_CO2_Industry_Total` | `Industry` (all) | 17 secteurs explicites (cf. `INDUSTRY_SECTORS_EXIOBASE`) | 0.62 |
| `Sec_CO2_Transp_Total` | `Transportation` | 7 secteurs transport | 0.28 ⚠ |
| `Sec_CO2_Transp_Aviation` | `Transportation\|Aviation` | "Air transport services (62)" | — (NGFS rare) |
| `Sec_CO2_Transp_Maritime` | `Transportation\|Maritime\|Freight` | Sea + Inland water | 6.28 ⚠ |
| `Sec_CO2_Transp_Rail` | `Transportation\|Rail` | "Railway transportation services" | 1.25 ✓ |
| `Sec_CO2_ResComm_Residential` | `R&C\|Residential` | F_Y catégorie "Households" | 2.40 ⚠ |

**Caveats expliquant les ratios** :

- **Cement (0.59)** : EXIOBASE attribue les émissions de combustion par **secteur producteur** (territorial) ; NGFS Industry|Cement peut inclure l'énergie secondaire (électricité, vapeur) consommée par les cimenteries que EXIOBASE attribue aux secteurs `Electricity by X` / `Steam supply` au lieu du secteur Cement.
- **Chemicals (0.16)** : "Chemicals nec" EXIOBASE est très étroit. NGFS Industry|Chemicals couvre aussi pétrochimie, ammoniac (`N-fertiliser`), polymères (`Plastics, basic`), fibres synthétiques.
- **NonFerrous (3.03)** : EXIOBASE inclut probablement émissions process aluminium (anodes carbone consommées en électrolyse) que NGFS classe ailleurs.
- **Transport Total (0.28)** : EXIOBASE n'inclut que les transporteurs commerciaux (F). NGFS inclut aussi les VP particuliers (= F_Y EXIOBASE).
- **Maritime (6.28)** : EXIOBASE attribue tout le bunker fuel international au pays vendeur (US, KR, JP, SG), ce qui concentre fortement. NGFS répartit autrement.
- **R&C Residential (2.40)** : EXIOBASE F_Y "households" = chauffage + ECS **+ voitures particulières** ; NGFS sépare Residential du Transportation.
- **Aviation** : variable NGFS rarement rapportée (MESSAGE = 0). Comparaison non fiable.

### 3.3 Énergie (FinalEnergy / PrimaryEnergy)

- `Energy use - Final` / `Energy use - Gross` en TJ → EJ ×1e-6
- **Limite majeure** : extension `energy` EXIOBASE **pxp** a F=0 (toute la valeur est en F_Y, qui = consommation finale des ménages uniquement)
- → EXIOBASE sous-estime ~75% : 109 EJ vs NGFS ~430 EJ pour Final Energy 2020 World
- Quality = `proxy` avec note explicite

Pour valeurs complètes : passer à **ixi** où F est populé (3.35e+8 TJ). Mais les sectoral mappings ne fonctionneront plus (industries ≠ produits).

### 3.4 Land

- `Cropland` : somme tous "Cropland - *" (cropped + fallowed × cultures et animaux)
- `Forest` : "Forest" seul
- `Pasture` : somme "Permanent pastures - *" (Grazing-Cattle, Meat animals, Raw milk)
- `Built-up` : "Artificial Surfaces" (uniquement dans F_Y)
- Conversion km² → million ha : ×1e-4
- Ratios excellents (0.93 - 1.03) ✓

### 3.5 Water

- `Water_Irrigation` : somme "Water Consumption Blue - Agriculture - *" (Mm³) × 1e-3 → km³
- **Proxy** : NGFS = Water Withdrawal (prélèvement), EXIOBASE = Water Consumption (prélèvement − retour)
- Pour irrigation, les deux concepts sont proches (peu de retour)
- Catégories EXIOBASE non utilisées : Blue Domestic, Blue Electricity, Blue Manufacturing, Blue Livestock, Green Agriculture, Withdrawal Blue *

### 3.6 Production (Cement, Steel, Chemicals)

- **Incommensurable** : EXIOBASE x[secteur] en M€ courant, NGFS en Mt/yr volumétriques
- Plotté en **mode normalisé** (taux de croissance relatif, base 2020)
- Approximation absolue possible via prix moyens : Cement ~100 €/t, Steel ~600 €/t, Chemicals très variable

### 3.7 Macro

- `GDP` : EXIOBASE = `Σ x − Σ Z par colonne` = VA totale en **M€ courants**. NGFS = **B USD2010 constants**
- Doublement incommensurable : devise (EUR vs USD) + déflation (courant vs 2010)
- → mode normalisé recommandé. EXIOBASE croît plus vite (inflation ~2%/an incluse)
- **Variable retirée** : `Population` (EXIOBASE = emploi, pas population totale → proxy faux)

### 3.8 Electricity Mix

- 9 mappings `Sec_SE_Elec_*` (Coal, Gas, Oil, Nuclear, Hydro, Wind, Solar, Biomass, Geothermal)
- EXIOBASE = x du secteur "Electricity by X" en **M€ courants**
- NGFS `Secondary Energy|Electricity|X` en **EJ/yr** (énergie électrique générée)
- **Incommensurable** → mode normalisé pour voir la transition relative
- Solar EXIOBASE = somme `Electricity by solar photovoltaic` + `Electricity by solar thermal`

---

## 4. Limites connues

### 4.1 Variables retirées du mapping

| Variable | Raison du retrait |
|---|---|
| `Population` | EXIOBASE n'a que l'emploi (∼ pop active employée), pas la pop totale. Utiliser UN WPP / World Bank. |
| `PM10_total`, `PM2_5_total` | NGFS Phase V ne contient pas PM10/PM2.5 (BC/OC à la place, qui ne sont pas dans EXIOBASE). |
| `Sec_FE_Industry_Cement/Steel/Chemicals` | extension energy EXIOBASE pxp a F=0 → toujours 0. Nécessite ixi (mais sectoral names cassent). |
| `Sec_FE_Transp_Aviation/Maritime` | idem |

### 4.2 EXIOBASE 2023-2024 nowcasts

- Extensions environnementales partielles : air_emissions F.sum ÷10, F_Y=0
- Trajectoires limitées à **2010-2022** par défaut
- Les cache CSV contiennent quand même 2023-2024 (fausses) — réutilisables si EXIOBASE met à jour

### 4.3 Périmètres NGFS vs EXIOBASE — différences conceptuelles

- **NGFS** : émissions territoriales par secteur **consommateur** d'énergie (allocation IPCC)
- **EXIOBASE F** : émissions production-based par secteur **producteur**
- Conséquence : pour les secteurs intensifs en électricité (Cement, Steel, Aluminium), EXIOBASE peut sous-estimer (les émissions de l'élec achetée sont chez Utilities)

### 4.4 Bunker fuels & aviation internationale

- EXIOBASE attribue le bunker maritime au pays vendeur de carburant
- NGFS Aviation Phase V partiellement rapporté (souvent =0 chez MESSAGE/GCAM)

---

## 5. Recommandations d'amélioration

### Mappings à ajouter
- ✓ **`Sec_CO2_IP_Cement`** : ajouté (process calcination, complète Industry|Cement)
- À ajouter : `Sec_CO2_IP_Steel` (réduction directe avec coke), `Sec_CO2_IP_Chemicals` (Haber-Bosch, etc.) si stresseurs EXIOBASE existent

### Mappings à raffiner
- `Sec_CO2_Industry_Chemicals` : étendre à "N-fertiliser", "Plastics, basic", "Synthetic fibres", "Petrochemicals" pour matcher la portée NGFS
- `Sec_CO2_Transp_Maritime` : exclure bunker fuel international (méthode complexe)
- `GDP` : ajouter conversion EUR courant → USD2010 (déflateur EU + taux de change 2010)

### Données complémentaires souhaitables
- Population : import UN WPP / World Bank
- Final Energy sectoriel : passer en ixi avec remapping des secteurs

---

## 6. Récapitulatif chiffré 2020 World

Validation des mappings (ratio = EXIOBASE / NGFS médiane).
✓ = ratio entre 0.7 et 1.3 (cohérent) ; ⚠ = écart > 50% qui mérite explication.

| Mapping | EXIOBASE | NGFS médiane | Ratio | Status |
|---|---:|---:|---:|---|
| CO2_fossil_IPCC | 33.5 Gt | 38.5 Gt | 0.87 | ✓ |
| CO2_total | 33.6 Gt | 44.3 Gt | 0.76 | ✓ (NGFS inclut land use) |
| CH4_total | 326 Mt | 352 Mt | 0.93 | ✓ |
| CH4_energy | 127 Mt | 115 Mt | 1.11 | ✓ |
| CH4_AFOLU | 141 Mt | 172 Mt | 0.82 | ✓ |
| N2O_total | 8.66 Mt | 10.3 Mt | 0.84 | ✓ |
| N2O_AFOLU | 7.88 Mt | 7.79 Mt | 1.01 | ✓ excellent |
| N2O_energy | 0.77 Mt | 0.84 Mt | 0.92 | ✓ |
| NOx_total | 113 Mt | 121 Mt | 0.93 | ✓ |
| Sulfur_total | 134 Mt | 72 Mt | 1.86 | ⚠ EXIOBASE compte plus de procédés industriels |
| NH3_total | 87 Mt | 54 Mt | 1.63 | ⚠ EXIOBASE plus large côté agri |
| VOC_total | 90 Mt | 142 Mt | 0.63 | ⚠ EXIOBASE limité à NMVOC combustion |
| CO_total | 460 Mt | 781 Mt | 0.59 | ⚠ EXIOBASE manque CO non-combustion |
| Land_Cropland | 1549 Mha | 1593 Mha | 0.97 | ✓ excellent |
| Land_Forest | 4042 Mha | 3918 Mha | 1.03 | ✓ excellent |
| Land_Pasture | 3113 Mha | 3347 Mha | 0.93 | ✓ |
| Land_BuiltUp | 60 Mha | 65 Mha | 0.93 | ✓ |
| Water_Irrigation | 1068 km³ | (à vérifier) | — | proxy concept |
| FinalEnergy | 109 EJ | 427 EJ | 0.25 | ⚠ pxp F=0 (cf §3.3) |
| Sec_CO2_Industry_Steel | 2401 Mt | 2701 Mt | 0.89 | ✓ |
| Sec_CO2_Industry_Cement | 631 Mt | 1066 Mt | 0.59 | ⚠ cf §3.2 |
| Sec_CO2_Industry_Chemicals | 221 Mt | 1353 Mt | 0.16 | ⚠ secteur EXIOBASE trop étroit |
| Sec_CO2_Industry_NonFerrous | 201 Mt | 66 Mt | 3.03 | ⚠ inclut process Al |
| Sec_CO2_Transp_Total | 2333 Mt | 8197 Mt | 0.28 | ⚠ manque VP particuliers |
| Sec_CO2_Transp_Rail | 185 Mt | 148 Mt | 1.25 | ✓ |

**Médiane des ratios "directs" sans caveats = 0.93** (cohérence globale du mapping).

---

## 7. Comment utiliser

```powershell
# Exporter la table de mapping complète (à jour avec les modifs)
python longterm/ngfs_exiobase_mapping.py --export-mapping Data/mappings/mapping_table.csv

# Liste filtrée par thème
python longterm/ngfs_exiobase_mapping.py --list --theme "Sectoral Emissions"

# Comparaison single-year d'un mapping
python longterm/ngfs_exiobase_mapping.py --compare Sec_CO2_Industry_Cement --year 2020 --scope world

# Trajectoire d'un mapping
python longterm/ngfs_exiobase_mapping.py --trajectory Sec_CO2_IP_Cement --scope world

# Tournée complète (45+ mappings)
python longterm/ngfs_exiobase_mapping.py --all-trajectories --scope world

# Mode forcé absolute pour incommensurable (donnera ratios bruts)
python longterm/ngfs_exiobase_mapping.py --trajectory GDP --scope world --mode absolute
```

---

*Document généré le 2026-05-26. Mappings : voir `Data/mappings/mapping_table.csv` (auto-exporté
par `run_all_trajectories`).*
