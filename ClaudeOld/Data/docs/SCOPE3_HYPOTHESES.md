# Scope 3 sectoriel : hypothèses, contrôles et lecture des figures

Documentation méthodologique du paquet `scope3/`, dans la convention de
`MAPPING_HYPOTHESES.md` et `MAPPING_ST_HYPOTHESES.md` : ce qui est posé, ce qui
est mesuré, et ce qui ne peut pas l'être.

Objet : la décomposition **scope 1 / 2 / 3** par secteur, les **cibles de
décarbonation** (−30 %, −60 %) imposées par KRAS, et le **coût carbone le long
de la chaîne de valeur**.

---

## 1. Le résultat qui structure tout

Avec `s` l'intensité directe, `A` les coefficients techniques et `m = s·L` le
multiplicateur berceau-porte :

```
m = s + m·A        ⟹        m_j·x_j  =  F_j  +  Σ_i m_i·Z_ij
                            └ berceau ┘   └sc.1┘  └── amont (scopes 2+3) ──┘
```

Exact, pas approché : les trois scopes se recomposent toujours en berceau-porte
(contrôle 1, écart 1,8e-16).

Côté prix, un prix carbone τ appliqué aux émissions directes donne un choc de
coût unitaire `Δv_j = τ·s_j`, propagé par le modèle de prix de Leontief
`Δp = (I − Aᵀ)⁻¹·Δv`, soit :

```
Δp_j = τ · m_j
```

**Le surcoût carbone en plein report est exactement τ × l'empreinte
berceau-porte.** La ventilation scope 1/2/3 des émissions *est* donc la
ventilation de la facture carbone : une seule décomposition, deux unités.
C'est ce qui autorise la figure de synthèse. Ce n'est pas une approximation, et
c'est vérifié par deux chemins de calcul indépendants (contrôle 4).

---

## 2. Périmètre : monde agrégé

Les 49 régions EXIOBASE sont sommées. KRAS équilibre n² variables : à 9800
secteurs-régions cela ferait 9,6·10⁷ inconnues, hors d'atteinte
(`optim/benchmark_kras.py` s'arrête à n = 320).

Conséquence à assumer : **les chaînes internationales sont incluses dans les
flux mais ne sont plus distinguées.** Un secteur agrégé achète en partie à
lui-même (la diagonale `Z_jj`), et ces achats tombent dans son propre scope 3.
L'effet est d'autant plus fort que le regroupement est grossier — une des
raisons de produire les deux granularités.

| Granularité | n | Source du mapping |
| --- | --- | --- |
| `gics` | 11 | `Data/mappings/exiobase_concordance.csv`, via `_map_sector` de `convert_sectors.py` |
| `ngfs` | 36 | `exio_to_ngfs()` de `shortterm/st_exiobase_mapping.py` — partition validée à 100 % de couverture. 50 secteurs GEM-E3 existent, mais 14 sont des secteurs `overlap` sans produit EXIOBASE propre |

`A` est **agrégé puis divisé** (`A_agg = Z_agg / x_agg`), jamais moyenné : la
moyenne des coefficients techniques d'un groupe n'est le coefficient technique
de rien. Même convention que `optim/exioptim.py:122`.

**Année de base 2019, classification `pxp`** (recommandation de
`MAPPING_ST_HYPOTHESES.md` §10).

### Stresseurs retenus

Les trois CO2 fossiles de `io_pipeline.py` :
`CO2 - combustion - air`, `CO2 - non combustion - Cement production - air`,
`CO2 - non combustion - Lime production - air`. Total monde 2019 : **31 003 Mt**.

`CO2_bio - combustion - air` est **exclu** : carbone biogénique, conventionnellement
reporté hors des scopes et hors tarification carbone. `io_pipeline.py`'s
`__main__` l'inclut dans sa liste ; ce n'est pas la même question.

---

## 3. Définition de chaque scope

```
scope1_j    = F_j
scope2_j    = Σ_{p ∈ électricité}  m^P_p · Z_pj
berceau_j   = m_j · x_j
scope3_up_j = berceau_j − scope1_j − scope2_j
```

### 3.1 Le scope 2 ne peut pas être « l'intensité directe de la ligne achetée »

Le Protocole GHG compte les émissions de **production** de l'électricité achetée.
Or dans EXIOBASE la chaîne est `Coal fired → Power Supply (réseau/négoce) →
acheteur` : presque personne n'achète directement à un producteur. Prendre les
seules émissions directes de la ligne achetée mettrait la quasi-totalité de la
production électrique dans le scope 3 et laisserait le scope 2 à ~0.

Le scope 2 est donc mesuré avec le multiplicateur **restreint au bloc
électrique** :

```
m^P = s^P · L        avec  s^P_i = s_i si i est un secteur électrique, 0 sinon
```

soit « les émissions survenant à l'intérieur du bloc électrique, atteintes par
les achats directs d'électricité de j ». L'électricité incorporée dans l'acier
que j achète ne passe pas par une ligne électrique et reste donc, correctement,
en scope 3.

Vérification : `Power Supply` ressort avec scope 1 = 22 Mt (pertes réseau) et
scope 2 = 2 748 Mt — la production électrique est bien attribuée au distributeur.

**Secteurs électriques** : thème `Power generation` de `MAPPING` (10 secteurs
présents en pxp) pour `ngfs` ; `Utilities` pour `gics`. Liste explicite, pas une
expression régulière sur « power ».

### 3.2 Ce qui n'est pas calculable ici

* **Scope 3 amont uniquement** (catégories 1-8 du Protocole GHG). Une table
  entrées-sorties n'a pas de phase d'usage : le **scope 3 aval** — catégorie 11,
  usage des produits vendus, qui domine pour le pétrole et le gaz — n'est pas
  calculable. Un modèle de Ghosh en donnerait un proxy d'allocation ; c'est un
  autre objet, hors périmètre. **Les chiffres pétrole/gaz de ces figures sont
  donc structurellement en dessous de leur empreinte réelle.**
* Le scope 2 est en **base localisation**, alloué par les achats monétaires
  d'électricité. Aucune variante contractuelle / market-based n'existe dans une
  table entrées-sorties.
* Le total « berceau-porte » sommé sur les secteurs (81 732 Mt à 36 secteurs)
  **n'est pas un inventaire mondial** : chaque étage est recompté à chaque
  stade. Le seul total additif est le scope 1 (31 003 Mt).

### 3.3 Résultats de référence (36 secteurs, 2019)

| Secteur | Scope 1 | Scope 2 | Scope 3 | Part scope 3 |
| --- | ---: | ---: | ---: | ---: |
| PV | 0,5 | 0,1 | 10,3 | 95 % |
| Transport equipment (hors VE) | 108 | 184 | 2 447 | 89 % |
| Construction | 755 | 198 | 5 702 | 86 % |
| Fabricated Metal products | 152 | 191 | 1 655 | 83 % |
| Market Services | 1 314 | 1 996 | 6 039 | 65 % |
| Ferrous metals | 2 703 | 509 | 2 344 | 42 % |
| Power Supply | 22 | 2 748 | 226 | 8 % |
| Coal fired | 8 881 | 185 | 297 | 3 % |

(Mt CO2. Tableau complet : `python scope3/scopes.py --granularity ngfs`.)

---

## 4. Coût carbone

### 4.1 Deux régimes

| Régime | Formule | Lecture |
| --- | --- | --- |
| `scope1_only` | `τ · s_j` | ce que le secteur paie sous un ETS ne portant que sur sa propre cheminée, sans aucun report des fournisseurs |
| `full_chain` | `τ · m_j` | ce qu'il paie une fois que chaque étage tarifie et reporte le carbone (état final de type MACF) |

**L'écart entre les deux est l'exposition scope 2 + 3, en euros.** C'est la
longueur de l'haltère de la figure `carbon_cost`.

### 4.2 Conversion monétaire — hypothèse à assumer

EXIOBASE est en **M.EUR courants** (2019), GEM-E3 en **US$2017/t CO2**. Il faut
un taux ; il est nommé plutôt qu'enfoui :

```python
EUR_PER_USD = 0.893     # moyenne annuelle 2019, BCE
```

Le déflateur 2017→2019 n'est **pas** appliqué : il est très inférieur à l'écart
entre scénarios et donnerait au chiffre une fausse allure de précision. Le coût
en % de la valeur de production est **linéaire** dans ce taux : un lecteur qui
en préfère un autre remet à l'échelle par simple proportion.

### 4.3 Prix carbone

`Price|Carbon Weighted Average` de `Data/NGFS_ST/GEME3_IIASA_2025_07_02.xlsx`,
lu par `load_geme3_variable` (déjà présent dans le dépôt, jusqu'ici inutilisé).
Les agrégats (World, EU27, Asia…) n'existent que sous cette variante ; le
`Price|Carbon` simple est au niveau pays.

Monde, US$2017/t CO2 :

| Scénario | 2022 | 2025 | 2030 |
| --- | ---: | ---: | ---: |
| Baseline | 19,5 | 23,0 | 53,7 |
| DAPS | 19,5 | 23,0 | 54,0 |
| DIRE | 19,5 | 29,4 | 115,3 |
| HWTP | 19,5 | 32,9 | 151,6 |
| SWUC | 19,5 | 23,0 | 163,1 |

**Le coût est linéaire en τ : tous les scénarios donnent la même figure, à
l'échelle près.** C'est pourquoi les cinq scénarios ne sont *pas* déclinés en
petits multiples — ce serait cinq fois la même image. Les figures utilisent
SWUC 2030 (le plus élevé) et affichent la trajectoire des prix à côté.

### 4.4 Résultats saillants (SWUC 2030, τ = 163 US$/t)

| Secteur | Scope 1 seul | Chaîne complète | Exposition |
| --- | ---: | ---: | ---: |
| Coal fired | 227,5 % | 239,8 % | 12,3 pts |
| Power Supply | **0,3 %** | **41,1 %** | **40,8 pts** |
| Ferrous metals | 17,5 % | 36,0 % | 18,5 pts |
| Fabricated Metal products | 1,0 % | 12,5 % | 11,6 pts |
| Transport equipment (hors VE) | 0,3 % | 7,6 % | 7,3 pts |
| Market Services | 0,4 % | 2,5 % | 2,2 pts |

En % de la valeur de production. `Power Supply` est le cas d'école : quasi rien
sous un ETS scope 1, 41 % de sa valeur de production si toute la chaîne est
tarifée. Les valeurs > 100 % (les trois centrales fossiles) sont réelles et
signifient que l'activité devient non rentable à ce prix.

---

## 5. Les cibles KRAS

### 5.1 Une contrainte colonne, pas ligne

`optim/io_footprint.py` contraint les **lignes** : l'empreinte incorporée dans
ce qu'un fournisseur vend. Le scope 3 est la question transposée — une
contrainte **colonne**, sur ce qu'un acheteur achète :

```
scope3_j = Σ_i mu_i · b_ij · x_j        ⟹  coefficient sur b_ij : mu_i · x_j
```

avec `mu` l'intensité amont **effective** : `mu_i = m_i` pour un secteur
ordinaire, `m_i − m^P_i` pour un secteur électrique (ses émissions propres sont
le scope 2 de l'acheteur, pas son scope 3). D'où un constructeur dédié,
`build_scope3_constraints` ; `io_footprint.py` n'est pas touché.

### 5.2 Marges colonnes tenues, marges lignes libres — c'est tout le modèle

Les **marges colonnes** sont tenues : la dépense totale d'intrants de j est
inchangée. KRAS ne peut donc pas faire acheter *moins* — seulement **changer de
fournisseur**. C'est la bonne lecture d'une cible scope 3 : substitution à
volume constant.

Les **marges lignes sont délibérément libres**. Les tenir aussi verrouillerait
la réponse à zéro :

```
Σ_j scope3_j = Σ_j Σ_i mu_i·b_ij·x_j = Σ_i mu_i·u_i
```

est une **constante** si `u` est fixé. Chaque secteur ne pourrait alors atteindre
sa cible qu'en poussant du scope 3 sur un autre : le total mondial ne bouge pas
d'un gramme. C'est la transposée de la remarque de `optim/footprint_demo.py`.
Mesuré (contrôle 7) :

| | scope 3 monde, cible −30 % |
| --- | ---: |
| marges lignes **libres** | 36 345 → 25 441 Mt (**−30,0 %**) |
| marges lignes **fixées** | 36 345 → 34 567 Mt (−4,9 %) |

Une première tentative avec les deux marges fixées produisait des secteurs
**trois fois pires** qu'au départ, KRAS s'épuisant à redistribuer.

Libérer les lignes n'est pas un relâchement, c'est l'économie du problème : si
tous les acheteurs se détournent des fournisseurs carbonés, ces fournisseurs
**vendent moins**. Leurs ventes sont un résultat, pas une contrainte. Les
niveaux de production suivent ensuite par rebouclage de Leontief à demande
finale fixe (`x_expost`). `--include-row` reproduit la variante verrouillée.

### 5.3 Ex-ante et ex-post

La cible est imposée aux **facteurs d'émission de base** — ce qu'une
« engagement scope 3 » signifie réellement : une entreprise mesure son
inventaire contre une base de facteurs figée, pas contre un équilibre général.
C'est le défaut, `outer=1`.

Re-mesurer le même résultat avec les `mu` qu'implique la **nouvelle** structure
donne une baisse nettement plus forte, parce que tous les fournisseurs se sont
assainis en même temps et que les multiplicateurs se composent :

| Cible (36 secteurs) | Aux facteurs de base | Re-mesuré |
| --- | ---: | ---: |
| −30 % | −30,0 % | −49,9 % |
| −60 % | −60,0 % | −87,8 % |

L'écart est un résultat, pas une erreur. `outer > 1` itère vers le point fixe
auto-cohérent ; l'itération de Picard non amortie **oscille** sur cette
application (18 679 → 37 596 → 18 645 Mt observés), d'où l'amortissement à 0,5.
Elle répond à une autre question que celle d'un engagement d'entreprise.

### 5.4 Réglage de sigma — inversé par rapport à `exioptim.py`

`DEFAULT_SIGMA_RATIO = 5.0` : la cible scope 3 est la contrainte **la plus
relâchable**, donc KRAS tient les marges et laisse la cible glisser là où elle
est inatteignable. `optim/exioptim.py` utilise 0.2, l'inverse, parce qu'il pose
la question inverse (« de combien plient les marges si la cible d'empreinte est
non négociable ? »). Ici les marges portent le sens du modèle, et le déficit
contre la cible est précisément le résultat cherché.

Avec 5.0 : distorsion des marges colonnes **1,9e-07** à −30 % et 2,7e-02 à −60 %.
Avec 0.2 : 45 % — inexploitable.

### 5.5 Plancher de sourcing

`feasible_floor` calcule le scope 3 le plus bas atteignable en s'approvisionnant
entièrement chez le fournisseur le plus propre **déjà utilisé** (GRAS ne peut pas
déplacer un zéro exact). Une cible sous ce plancher est hors d'atteinte par
arithmétique, quoi que fasse le solveur. À 11 secteurs GICS, `Financials`
plafonne à −50 % contre une cible de −60 %, et le plancher l'explique. À 36
secteurs, aucune cible n'est sous son plancher : il y a assez de fournisseurs
pour toujours en trouver un plus propre.

### 5.6 L'artefact de substitution, annoncé

Cibler le **scope 3 seul** récompense le basculement des achats vers les
secteurs électriques, dont les émissions sont **reclassées** en scope 2 de
l'acheteur plutôt que supprimées. C'est le reflet exact d'une critique réelle
des cibles scope-3-seul. Mesuré :

| Base ciblée | Cible | Scope 2 monde | Scope 3 monde | Berceau ex-post |
| --- | --- | ---: | ---: | ---: |
| `scope3` (11 sect.) | −30 % | **+79,5 %** | −40,3 % | −20,6 % |
| `scope3` (11 sect.) | −60 % | **+187,9 %** | −79,3 % | −36,7 % |
| `upstream` (11 sect.) | −30 % | −81,0 % | −48,8 % | −54,1 % |
| `upstream` (11 sect.) | −60 % | −96,0 % | −83,9 % | −79,3 % |

D'où : le scope 2 atteint est **toujours affiché à côté** du scope 3 atteint, et
`--basis upstream` (cible sur scopes 2+3 ensemble, coefficient `m_i·x_j`) referme
la faille. Le défaut reste `scope3`, qui est la question posée.

### 5.7 Effort de réallocation

À 36 secteurs, toutes les cibles sont atteintes : ce qui varie est le **coût**
de les atteindre. `procurement_effort` mesure la part du budget d'achats qui
change de fournisseur (demi-distance L1 entre les parts de sourcing initiales et
finales).

| Cible | Effort le plus lourd | Fournisseurs les plus déclassés |
| --- | --- | --- |
| −30 % | Oil Fired 30 %, Coal fired 29 %, Oil 27 % | Air transport −78 %, Water transport −67 %, Ferrous metals −65 % |
| −60 % | **Market Services 88 %**, Oil 72 %, Consumer Goods 71 % | Air transport −97 %, Non-metallic minerals −97 %, Ferrous metals −94 % |

**Lecture : une cible généralisée à −60 % revient à faire disparaître la
demande intermédiaire adressée à l'industrie lourde et au transport.** Le modèle
n'a aucune contrainte de capacité, de délai ni de substituabilité technique —
il dit ce que la comptabilité autorise, pas ce que l'ingénierie permet.

### 5.8 Normalisation — le piège

Dans KRAS les coefficients de contrainte sont des **exposants** :
`a[idx] * r ** (sign(a) * g)` (`optim/kras.py:223`). En unités brutes
(`x ~ 1e7 M€`, `mu·x ~ 1e12 kg`), `r**g` déborde. Tout est ramené à l'ordre 1
avant le solveur : `X0 = mean(x)`, `MU0 = mean(|mu|)`, coefficient de marge
`x_j/X0`, coefficient scope 3 `(mu_i/MU0)·(x_j/X0)`, cible `E3_j/(MU0·X0)`.
Seules les réductions relatives comptent, rien n'est perdu. Même avertissement
que `optim/exioptim.py:112-115`.

---

## 6. Contrôles numériques

`python scope3/test_scopes.py` — 7 contrôles, tous passants.

| # | Contrôle | Mesure |
| --- | --- | --- |
| 1 | Identité `m·x = F + m·Z` ; scope1+2+3 = berceau ; colonnes de `scope3_by_supplier` | ≤ 4,8e-16 |
| 2 | Conservation à l'agrégation (x, CO2, Z), aux deux granularités | 0 à 1,3e-16 |
| 3 | Contrôle Leontief `(I−A)⁻¹·y = x` | 9,5e-13 / 2,7e-12 |
| 4 | Modèle de prix : `(I−Aᵀ)⁻¹(τ·s)` contre `τ·m`, deux chemins indépendants ; linéarité en τ | ≤ 7,9e-16 |
| 5 | Facture mondiale scope 1 = τ × émissions directes ; facture = scope 1+2+3 | exact |
| 6 | Post-KRAS : convergence, marges colonnes tenues, production ex-post positive, tout déficit expliqué par le plancher | dérive 1,9e-07 / 2,7e-02 |
| 7 | Marges lignes fixées ⟹ total verrouillé (§5.2) | −30,0 % contre −4,9 % |

---

## 7. Figures

`python scope3/plot_scope3.py --all [--granularity gics|ngfs] [--theme light|dark]`
→ `Figs/scope3/`.

| Figure | Contenu |
| --- | --- |
| `scope_decomposition` | Où vit le scope 3 : barres empilées en Mt + composition en %, les 36 secteurs, triés par part de scope 3 |
| `kras_targets` | Scope 3 base / −30 % / −60 % et l'effort de réallocation associé. Top 20 secteurs (94 % du total) |
| `carbon_cost` | Haltère « scope 1 seul → chaîne complète » en % de la valeur de production, axe log ; trajectoires de prix carbone à côté |
| `cost_vs_target` | Synthèse : exposition au coût carbone × effort de réallocation à −60 %, aire = production, couleur = part de scope 3 |

### 7.1 Palette — calculée, pas choisie

Aucun `node` sur cette machine : le validateur utilisé est
`shortterm/palette_check.py`, le port documenté du validateur de la skill.

**Scopes (catégoriel, 3 slots)** — les trois premiers slots de la palette de
référence, seul sous-ensemble documenté comme franchissant les portes
*all-pairs* dans les deux modes :

```
clair  pire paire CVD ΔE 9,2 (cible 8)   vision normale 24,0 (plancher 15)
sombre pire paire CVD ΔE 9,4              vision normale 20,9
1 WARN : l'aqua est à 2,74:1 sur la surface claire → règle de relief appliquée,
         chaque part est imprimée, rien n'est encodé par la couleur seule
```

Attribution **par rôle et non par ordre de slot** : le **scope 3 prend le bleu**
(slot 1, meilleur contraste) car c'est le sujet ; scope 1 l'orange, scope 2
l'aqua. `bleu / orange / vert` a été essayé d'abord et **rejeté** : vert contre
orange mesure ΔE 3,2 en protanopie.

**Cibles (ordinal, 3 pas)** — base / −30 % / −60 % est une séquence ordonnée :
une seule teinte échelonnée par ambition, gris de mise en retrait pour la
référence (convention `plot_production.py` pour Baseline). Mesuré : pire paire
vision normale ΔE 15,3 clair / 17,0 sombre, pire CVD 13,0 / 11,5, tous les pas
au-dessus du plancher ordinal de 2:1.

**Scénarios** — `SCENARIO_COLOR` importé tel quel de `plot_production.py` : un
scénario garde sa teinte dans toutes les figures du dépôt.

### 7.2 Choix de forme, et ce qu'ils corrigent

Contrairement aux figures court terme (cf. `MAPPING_ST_HYPOTHESES.md` §14, où
aucune figure n'avait été relue faute de kaleido), **celles-ci ont été rendues
en image et inspectées**. Défauts trouvés et corrigés :

* La légende horizontale et les `subplot_titles` se disputaient la bande
  au-dessus de la zone de tracé → les panneaux sont titrés par leur **axe x**.
* Plotly inverse l'ordre de légende sur les barres empilées → `traceorder="normal"`.
* Le panneau droit avait 2 barres par groupe contre 3 à gauche, désalignant les
  lignes sur l'axe partagé → une barre `base` nulle est ajoutée à droite.
* Le coût carbone couvre trois ordres de grandeur (0,2 % à 240 %) → **axe log**,
  sur lequel la longueur de l'haltère se lit comme le *rapport* entre les deux
  régimes. Les barres, elles, restent en échelle linéaire (une barre doit partir
  de zéro), d'où la coupe aux 20 premiers secteurs pour `kras_targets`.
* Trois conventions de coordonnées coexistent sur un même axe log Plotly,
  vérifiées sur l'image rendue : `add_vline` prend la valeur **linéaire**,
  `add_annotation` le **log₁₀**, `tickvals` le **linéaire**.
* Les graduations mineures logarithmiques de Plotly impriment « 2 » pour 20 →
  graduations explicites.
* Baseline et DAPS finissent à 0,3 US$/t l'un de l'autre → écartement
  automatique des étiquettes de fin de ligne.

---

## 8. Cache et fichiers produits

| Chemin | Suivi par git ? |
| --- | --- |
| `Data/processed/scope3_cache/{granularité}_{année}_{type}.npz` | **oui**, 14 Ko. Le reconstruire exige de reparser `Z.txt` (~700 Mo) depuis `EXIOBASE_data.zip`, lui-même ignoré par la règle `*.zip` : un clone frais ne pourrait pas le refaire |
| `Data/processed/scope3_kras/*.{csv,npz}` | non — quelques secondes à recalculer depuis le cache ci-dessus |
| `Figs/scope3/*.html` | non (règle `*.html` existante) |

`load_result` vérifie le jeu de colonnes (`REQUIRED_COLUMNS`) et renvoie `None`
si le fichier date d'une version antérieure : un cache périmé ne coûte qu'un
recalcul, jamais une erreur lointaine.

---

## 9. Limites, en résumé

1. **Pas de scope 3 aval** : pas de phase d'usage dans une table entrées-sorties.
   Pétrole et gaz sont structurellement sous-estimés.
2. **Régions agrégées** : chaînes internationales incluses mais indistinctes ;
   auto-consommation de secteur comptée en scope 3.
3. **Scope 2 en base localisation**, alloué par les achats monétaires.
4. **Plein report** du coût carbone supposé pour `full_chain` — borne haute ;
   `scope1_only` est la borne basse. Aucune élasticité, aucune destruction de
   demande.
5. **Aucune contrainte physique** dans KRAS : ni capacité, ni délai, ni
   substituabilité technique. Les −60 % de §5.7 sont ce que la comptabilité
   autorise.
6. **Coefficients techniques figés** sauf là où KRAS les déplace : pas de
   progrès technique, pas de changement d'intensité directe `s` (KRAS change
   qui achète à qui, pas la propreté d'un four).
7. **Un taux de change posé** (§4.2), à l'échelle près duquel tous les coûts en
   % sont linéaires.

---

*Rédigé le 2026-08-09. EXIOBASE 3.10.1, année 2019, `pxp`. 7 contrôles numériques
passants, figures inspectées en image.*
