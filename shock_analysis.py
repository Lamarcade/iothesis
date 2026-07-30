"""
Analyse de chocs sur la matrice Z d'EXIOBASE.

Logique du choc :
    On redirige shock_pct % du flux Z[origin → decrease]
    vers Z[origin → increase].

    ΔZ[origin, decrease] = -shock_pct * Z[origin, decrease]
    ΔZ[origin, increase] = +shock_pct * Z[origin, decrease]

    A' = Z' · x̂⁻¹   (x inchangé : choc purement technologique)
    footprint' = S · (I−A')⁻¹ · Y + F_Y

Visualisation :
    Empreinte agrégée par secteur GICS (11 groupes) — avant / après choc.
    Barchart delta + grouped bar avant/après.
"""

import warnings
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from convert_sectors import load_concordance, _build_lookup, _map_sector
from io_pipeline import _safe_inv, _build_leontief_sparse

GICS_ORDER = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Communication Services",
    "Utilities", "Real Estate",
]

GICS_COLORS = {
    "Energy":                  "#c0392b",
    "Materials":               "#8e44ad",
    "Industrials":             "#2980b9",
    "Consumer Discretionary":  "#f39c12",
    "Consumer Staples":        "#27ae60",
    "Health Care":             "#16a085",
    "Financials":              "#2c3e50",
    "Information Technology":  "#3498db",
    "Communication Services":  "#e67e22",
    "Utilities":               "#1abc9c",
    "Real Estate":             "#d35400",
    "UNKNOWN":                 "#bdc3c7",
}

# ---------------------------------------------------------------------------
# 1. Application du choc sur Z
# ---------------------------------------------------------------------------

def apply_shock(
    Z: pd.DataFrame,
    origin: str,
    decrease: str,
    increase: str,
    shock_pct: float,
    scope: str = "intra",
) -> tuple[pd.DataFrame, dict]:
    """
    Redirige shock_pct du flux Z[origin → decrease] vers Z[origin → increase].

    Parameters
    ----------
    Z : pd.DataFrame
        Matrice Z avec MultiIndex (region, sector).
    origin : str
        Secteur vendeur (ligne de Z) — ex: 'Electricity by gas'.
    decrease : str
        Secteur acheteur qui PERD du flux — ex: 'Air transport services (62)'.
    increase : str
        Secteur acheteur qui GAGNE du flux — ex: 'Construction work (45)'.
    shock_pct : float
        Pourcentage redirigé (0.10 = 10 %).
    scope : str
        'intra' → choc intra-régional seulement (r_vendeur == r_acheteur)
        'all'   → choc sur toutes les paires de régions

    Returns
    -------
    Z_shocked : pd.DataFrame — Z modifiée
    info      : dict — statistiques sur le choc appliqué
    """
    Z_shocked = Z.copy()

    row_mask = Z.index.get_level_values("sector") == origin
    col_dec   = Z.columns.get_level_values("sector") == decrease
    col_inc   = Z.columns.get_level_values("sector") == increase

    if not row_mask.any():
        raise ValueError(f"Secteur origin '{origin}' introuvable dans Z.")
    if not col_dec.any():
        raise ValueError(f"Secteur decrease '{decrease}' introuvable dans Z.")
    if not col_inc.any():
        raise ValueError(f"Secteur increase '{increase}' introuvable dans Z.")

    total_redirected = 0.0
    n_pairs = 0

    regions_row = Z.index[row_mask].get_level_values("region").unique()
    regions_col_dec = Z.columns[col_dec].get_level_values("region").unique()
    regions_col_inc = Z.columns[col_inc].get_level_values("region").unique()

    for r_sell in regions_row:
        buy_regions = [r_sell] if scope == "intra" else regions_col_dec
        for r_buy in buy_regions:
            if r_buy not in regions_col_dec or r_buy not in regions_col_inc:
                continue
            idx_row = (r_sell, origin)
            idx_dec = (r_buy, decrease)
            idx_inc = (r_buy, increase)
            if idx_row not in Z.index or idx_dec not in Z.columns or idx_inc not in Z.columns:
                continue

            flow = Z.loc[idx_row, idx_dec]
            delta = shock_pct * flow
            if delta == 0:
                continue

            Z_shocked.loc[idx_row, idx_dec] -= delta
            Z_shocked.loc[idx_row, idx_inc] += delta
            total_redirected += delta
            n_pairs += 1

    info = {
        "origin": origin,
        "decrease": decrease,
        "increase": increase,
        "shock_pct": shock_pct,
        "scope": scope,
        "total_redirected_M€": total_redirected,
        "n_region_pairs": n_pairs,
        "total_flow_before": Z.values[np.ix_(row_mask, col_dec)].sum(),
    }
    return Z_shocked, info


def get_sectors_for_gics(gics_name: str, concordance: pd.DataFrame) -> list[str]:
    """Retourne les noms de secteurs EXIOBASE appartenant à un groupe GICS."""
    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    # Inversion : GICS → liste de secteurs EXIOBASE via la concordance
    return [
        row["exiobase_sector"]
        for _, row in concordance.iterrows()
        if row["gics_name"] == gics_name
    ]


def apply_gics_shock(
    Z: pd.DataFrame,
    origin_gics: str,
    decrease_gics: str,
    increase_gics: str,
    shock_pct: float,
    scope: str = "intra",
    distribute: str = "proportional",
    concordance: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Choc au niveau GICS : tous les secteurs EXIOBASE du groupe origin_gics
    redirigent shock_pct de leurs achats aux secteurs decrease_gics
    vers les secteurs increase_gics.

    Pour chaque secteur origine i, la redistribution vers increase suit :
        - 'proportional' : proportionnelle aux flux existants Z[i → increase]
        - 'equal'        : parts égales entre tous les secteurs increase

    Parameters
    ----------
    Z : pd.DataFrame
        Matrice Z avec MultiIndex (region, sector).
    origin_gics : str
        Groupe GICS du secteur acheteur (ex: 'Industrials').
    decrease_gics : str
        Groupe GICS qui perd les flux (ex: 'Materials').
    increase_gics : str
        Groupe GICS qui gagne les flux (ex: 'Energy').
    shock_pct : float
        Fraction redirigée (0.01 = 1 %).
    scope : str
        'intra' → choc intra-régional | 'all' → toutes les paires de régions.
    distribute : str
        'proportional' | 'equal' — méthode d'allocation vers increase_gics.
    concordance : pd.DataFrame, optional

    Returns
    -------
    Z_shocked : pd.DataFrame
    info      : dict
    """
    if concordance is None:
        concordance = load_concordance()

    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    all_sectors = Z.index.get_level_values("sector").unique().tolist()

    origin_secs  = set(s for s in all_sectors if _map_sector(s, lookup, norm_lookup) == origin_gics)
    dec_secs     = set(s for s in all_sectors if _map_sector(s, lookup, norm_lookup) == decrease_gics)
    inc_secs     = set(s for s in all_sectors if _map_sector(s, lookup, norm_lookup) == increase_gics)

    for label, sset in [("origin_gics", origin_secs), ("decrease_gics", dec_secs), ("increase_gics", inc_secs)]:
        if not sset:
            raise ValueError(f"Aucun secteur EXIOBASE pour GICS '{label}'.")

    # Vérification que decrease ≠ increase pour éviter les chocs nuls
    if dec_secs == inc_secs:
        raise ValueError("decrease_gics et increase_gics sont identiques.")

    # Structure de Z : blocs régionaux consécutifs [r0×200, r1×200, …]
    region_order  = Z.index.get_level_values("region").unique().tolist()
    sector_order  = Z.index.get_level_values("sector").unique().tolist()
    n_sec = len(sector_order)
    sec_idx = {s: i for i, s in enumerate(sector_order)}

    # Positions locales (dans un bloc de 200) pour chaque groupe
    origin_pos = np.array([sec_idx[s] for s in origin_secs if s in sec_idx])
    dec_pos    = np.array([sec_idx[s] for s in dec_secs    if s in sec_idx])
    inc_pos    = np.array([sec_idx[s] for s in inc_secs    if s in sec_idx])

    Z_arr = Z.values.astype(float).copy()
    total_redirected = 0.0

    if scope == "intra":
        buy_regions_map = {r: [r] for r in region_order}
    else:
        buy_regions_map = {r: region_order for r in region_order}

    for r_sell_idx, r_sell in enumerate(region_order):
        sell_offset = r_sell_idx * n_sec
        row_pos = sell_offset + origin_pos            # positions absolues des lignes

        for r_buy in buy_regions_map[r_sell]:
            r_buy_idx = region_order.index(r_buy)
            buy_offset = r_buy_idx * n_sec

            dec_abs = buy_offset + dec_pos            # positions absolues des colonnes decrease
            inc_abs = buy_offset + inc_pos            # positions absolues des colonnes increase

            # Sous-matrice origin × decrease pour cette paire de régions
            sub_dec = Z_arr[np.ix_(row_pos, dec_abs)]   # (n_origin, n_dec)
            delta_per_row = shock_pct * sub_dec.sum(axis=1)  # (n_origin,)

            if delta_per_row.sum() == 0:
                continue

            # Appliquer la baisse sur decrease
            Z_arr[np.ix_(row_pos, dec_abs)] -= shock_pct * sub_dec

            # Redistribuer vers increase
            sub_inc = Z_arr[np.ix_(row_pos, inc_abs)]   # (n_origin, n_inc)
            inc_totals = sub_inc.sum(axis=1)             # (n_origin,)

            for i, (delta, inc_total) in enumerate(zip(delta_per_row, inc_totals)):
                if delta == 0:
                    continue
                if distribute == "proportional" and inc_total > 0:
                    Z_arr[row_pos[i], inc_abs] += delta * sub_inc[i] / inc_total
                else:
                    Z_arr[row_pos[i], inc_abs] += delta / len(inc_abs)
                total_redirected += delta

    Z_shocked = pd.DataFrame(Z_arr, index=Z.index, columns=Z.columns)

    info = {
        "mode": "gics",
        "origin_gics": origin_gics,
        "decrease_gics": decrease_gics,
        "increase_gics": increase_gics,
        "shock_pct": shock_pct,
        "scope": scope,
        "distribute": distribute,
        "n_origin_sectors": len(origin_secs),
        "n_decrease_sectors": len(dec_secs),
        "n_increase_sectors": len(inc_secs),
        "total_redirected_M€": total_redirected,
        # Champ commun avec apply_shock pour la compatibilité avec plot_shock_impact
        "origin":   f"[GICS] {origin_gics}",
        "decrease": f"[GICS] {decrease_gics}",
        "increase": f"[GICS] {increase_gics}",
    }
    return Z_shocked, info


def run_gics_shock_comparison(
    Z: pd.DataFrame,
    x: pd.Series,
    Y: pd.DataFrame,
    S: pd.DataFrame,
    F_Y: pd.DataFrame,
    origin_gics: str,
    decrease_gics: str,
    increase_gics: str,
    shock_pct: float,
    scope: str = "intra",
    distribute: str = "proportional",
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> dict:
    """
    Pipeline complet pour un choc au niveau GICS.
    Même interface de sortie que run_shock_comparison().

    Parameters
    ----------
    origin_gics, decrease_gics, increase_gics : str
        Noms GICS (ex: 'Industrials', 'Materials', 'Energy').
    distribute : str
        'proportional' | 'equal' — répartition du flux vers increase_gics.
    Autres paramètres : identiques à run_shock_comparison().

    Returns
    -------
    dict avec 'fp_baseline', 'fp_shocked', 'delta', 'delta_pct', 'info'
    """
    if concordance is None:
        concordance = load_concordance()

    print(f"Choc GICS : {shock_pct*100:.1f}% [{origin_gics}] -> [{decrease_gics}] "
          f"redirige vers [{increase_gics}]")

    print("  [1/3] Empreinte baseline...")
    A_base = pd.DataFrame(
        Z.values * _safe_inv(x.values)[np.newaxis, :],
        index=Z.index, columns=Z.columns,
    )
    fp_base = compute_gics_footprint(A_base, S, Y, F_Y, x, stressor_agg, concordance)

    print("  [2/3] Application du choc GICS...")
    Z_shocked, info = apply_gics_shock(
        Z, origin_gics, decrease_gics, increase_gics,
        shock_pct, scope, distribute, concordance,
    )
    n_o, n_d, n_i = info["n_origin_sectors"], info["n_decrease_sectors"], info["n_increase_sectors"]
    print(f"         {n_o} secteurs origine x {n_d} decrease x {n_i} increase, "
          f"{info['total_redirected_M€']:.1f} M€ redirigés")

    print("  [3/3] Empreinte post-choc...")
    A_shocked = pd.DataFrame(
        Z_shocked.values * _safe_inv(x.values)[np.newaxis, :],
        index=Z_shocked.index, columns=Z_shocked.columns,
    )
    fp_shocked = compute_gics_footprint(A_shocked, S, Y, F_Y, x, stressor_agg, concordance)

    delta     = fp_shocked - fp_base
    delta_pct = (delta / fp_base.replace(0, np.nan)) * 100

    return {
        "fp_baseline": fp_base,
        "fp_shocked":  fp_shocked,
        "delta":       delta,
        "delta_pct":   delta_pct,
        "info":        info,
    }


# ---------------------------------------------------------------------------
# 1b. Choc de production sur x
# ---------------------------------------------------------------------------

def apply_production_shock(
    x: pd.Series,
    decrease: str,
    increase: str,
    shock_pct: float,
    scope: str = "intra",
) -> tuple[pd.Series, dict]:
    """
    Choc sur le vecteur de production x :
        x'[decrease] -= shock_pct · x[decrease]      (la production diminue)
        x'[increase] += shock_pct · x[decrease]      (conservation du total)

    Modélise une substitution de source de production :
    ex. remplacer 50 % de l'électricité au gaz par de l'électricité éolienne.

    Parameters
    ----------
    x : pd.Series
        Production totale par secteur, MultiIndex (region, sector).
    decrease : str
        Secteur dont la production diminue (ex: 'Electricity by gas').
    increase : str
        Secteur dont la production augmente (ex: 'Electricity by wind').
    shock_pct : float
        Fraction de la production de decrease transférée à increase (0.5 = 50 %).
    scope : str
        'intra' → choc intra-régional (chaque région redirige sa propre production)
        'all'   → global (le total decrease mondial va vers increase mondial,
                          proportionnellement aux capacités existantes)

    Returns
    -------
    x_shocked : pd.Series
    info      : dict
    """
    x_shocked = x.copy().astype(float)
    region_order = x.index.get_level_values("region").unique().tolist()
    total_redirected = 0.0
    n_regions = 0

    if scope == "intra":
        for r in region_order:
            idx_dec = (r, decrease)
            idx_inc = (r, increase)
            if idx_dec not in x.index or idx_inc not in x.index:
                continue
            flow = x.loc[idx_dec]
            if flow <= 0:
                continue
            delta = shock_pct * flow
            x_shocked.loc[idx_dec] -= delta
            x_shocked.loc[idx_inc] += delta
            total_redirected += delta
            n_regions += 1
    else:  # "all" : redistribution globale
        dec_mask = x.index.get_level_values("sector") == decrease
        inc_mask = x.index.get_level_values("sector") == increase
        total_dec = x[dec_mask].sum()
        delta = shock_pct * total_dec
        x_shocked[dec_mask] *= (1 - shock_pct)
        inc_total = x[inc_mask].sum()
        if inc_total > 0:
            x_shocked[inc_mask] += delta * x[inc_mask] / inc_total
        else:
            x_shocked[inc_mask] += delta / int(inc_mask.sum())
        total_redirected = delta
        n_regions = int(dec_mask.sum())

    info = {
        "mode":                 "production",
        "decrease":             decrease,
        "increase":             increase,
        "shock_pct":            shock_pct,
        "scope":                scope,
        "total_redirected_M€":  total_redirected,
        "n_regions":            n_regions,
        "x_decrease_before_M€": float(x[x.index.get_level_values("sector") == decrease].sum()),
        "x_increase_before_M€": float(x[x.index.get_level_values("sector") == increase].sum()),
        # Compatibilité avec plot_shock_impact
        "origin":   decrease,
    }
    return x_shocked, info


def compute_production_gics_footprint(
    S: pd.DataFrame,
    x: pd.Series,
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Empreinte environnementale par GICS, base PRODUCTION :

        fp_g = Σ_{i ∈ g} Σ_k  S[k, i] · x[i]

    Attribue les émissions aux secteurs qui PRODUISENT (empreinte territoriale /
    production-based account), par opposition à compute_gics_footprint() qui
    attribue aux secteurs qui consomment via le modèle de Leontief
    (empreinte consommation / footprint).

    Avantage : pas d'inversion de matrice, calcul instantané.
    Limite : ignore les effets indirects de chaîne d'approvisionnement.

    Parameters
    ----------
    S : pd.DataFrame (e×n)  — intensités stresseurs
    x : pd.Series    (n,)   — production totale
    stressor_agg : list[str], optional — sous-ensemble de stresseurs à sommer
    concordance  : pd.DataFrame, optional

    Returns
    -------
    pd.Series indexée par GICS group (11 valeurs)
    """
    if concordance is None:
        concordance = load_concordance()

    gics_labels = _sector_to_gics(x.index, concordance)
    gics_series = pd.Series(gics_labels, index=x.index)

    S_use = S.loc[S.index.intersection(stressor_agg)] if stressor_agg else S
    S_aligned = S_use.reindex(columns=x.index, fill_value=0.0)   # (e, n)

    # Émissions directes par secteur : somme des stresseurs → (n,)
    direct_by_sector = (S_aligned.values * x.values[np.newaxis, :]).sum(axis=0)

    unique_gics = [g for g in GICS_ORDER if g in set(gics_labels)] + \
                  [g for g in set(gics_labels) if g not in GICS_ORDER]

    result = {
        g: float(direct_by_sector[np.array(gics_series == g)].sum())
        for g in unique_gics
    }
    return pd.Series(result)


def run_production_shock_comparison(
    x: pd.Series,
    S: pd.DataFrame,
    decrease: str,
    increase: str,
    shock_pct: float,
    scope: str = "intra",
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> dict:
    """
    Pipeline complet pour un choc de production (substitution de source).

    Modèle :
        Δx[decrease] = -shock_pct · x[decrease]          (production perdue)
        Δx[increase] = +shock_pct · x[decrease]           (production gagnée)
        Δfp_g        = Σ_{i∈g} S[:,i] · Δx[i]            (pas d'inversion Leontief)

    Interprétation :
        Empreinte PRODUCTION — quelles émissions directes chaque GICS génère-t-il ?
        Ex : remplacer 50 % de la production d'élec. au gaz par de l'éolien
        → le GICS Utilities perd l'empreinte carbone du gaz et gagne celle du vent.

    Différence avec run_shock_comparison (choc sur Z) :
        - Z-choc : change la TECHNOLOGIE (A = Z·x̂⁻¹), recalcule le Leontief,
                   donne l'empreinte CONSOMMATION.
        - x-choc : change les VOLUMES produits, calcul direct S·Δx,
                   donne l'empreinte PRODUCTION (plus rapide, sans LU).

    Parameters
    ----------
    x          : pd.Series — production totale (from load_io_system)
    S          : pd.DataFrame — intensités stresseurs (from compute_S)
    decrease   : str — secteur dont la production diminue
    increase   : str — secteur dont la production augmente
    shock_pct  : float — fraction transférée (0.5 = 50 %)
    scope      : 'intra' | 'all'
    stressor_agg : list[str], optional
    concordance  : pd.DataFrame, optional

    Returns
    -------
    dict avec 'fp_baseline', 'fp_shocked', 'delta', 'delta_pct', 'info'
        (même interface que run_shock_comparison pour compatibilité avec plot_shock_impact)
    """
    if concordance is None:
        concordance = load_concordance()

    print(f"Choc production : {shock_pct*100:.0f}% de [{decrease}] -> [{increase}]")

    print("  [1/3] Empreinte baseline (base production)...")
    fp_base = compute_production_gics_footprint(S, x, stressor_agg, concordance)

    print("  [2/3] Application du choc sur x...")
    x_shocked, info = apply_production_shock(x, decrease, increase, shock_pct, scope)
    print(f"         {info['n_regions']} regions touchees, "
          f"{info['total_redirected_M€']:.1f} M€ de production redirigee")
    print(f"         x[{decrease}] avant : {info['x_decrease_before_M€']:.1f} M€  "
          f"-> apres : {info['x_decrease_before_M€'] - info['total_redirected_M€']:.1f} M€")
    print(f"         x[{increase}] avant : {info['x_increase_before_M€']:.1f} M€  "
          f"-> apres : {info['x_increase_before_M€'] + info['total_redirected_M€']:.1f} M€")

    print("  [3/3] Empreinte post-choc (base production)...")
    fp_shocked = compute_production_gics_footprint(S, x_shocked, stressor_agg, concordance)

    delta     = fp_shocked - fp_base
    delta_pct = (delta / fp_base.replace(0, np.nan)) * 100

    return {
        "fp_baseline": fp_base,
        "fp_shocked":  fp_shocked,
        "delta":       delta,
        "delta_pct":   delta_pct,
        "info":        info,
    }


# ---------------------------------------------------------------------------
# 1c. Choc de demande finale sur Y (mono-secteur et multi-secteurs)
# ---------------------------------------------------------------------------

def apply_Y_shock(
    Y: pd.DataFrame,
    decrease: str,
    increase: str,
    shock_pct: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Choc sur la matrice de demande finale Y.

    Pour chaque région productrice r :
        ΔY[(r, decrease), :] = -shock_pct · Y[(r, decrease), :]
        ΔY[(r, increase), :] = +shock_pct · Y[(r, decrease), :]

    Interprétation : les consommateurs (toutes catégories : ménages,
    gouvernements, investissements) substituent le produit `decrease`
    par le produit `increase`, toutes régions consommatrices confondues.

    Différence avec apply_shock (Z-choc) :
        - Z-choc : même flux total, technologie modifiée (A change)
        - Y-choc : même technologie (A inchangé), demande modifiée

    Parameters
    ----------
    Y : pd.DataFrame (n×f) — MultiIndex (region, sector) × (region, category)
    decrease : str — secteur dont la demande finale diminue
    increase : str — secteur dont la demande finale augmente
    shock_pct : float — fraction redirigée (0.5 = 50 %)

    Returns
    -------
    Y_shocked : pd.DataFrame
    info      : dict
    """
    sector_order = Y.index.get_level_values("sector").unique().tolist()
    region_order = Y.index.get_level_values("region").unique().tolist()
    n_sec = len(sector_order)
    sec_idx = {s: i for i, s in enumerate(sector_order)}

    for label, s in [("decrease", decrease), ("increase", increase)]:
        if s not in sec_idx:
            raise ValueError(f"Secteur '{s}' ({label}) introuvable dans Y.")

    dec_local = sec_idx[decrease]
    inc_local = sec_idx[increase]

    Y_arr = Y.values.copy().astype(float)
    total_redirected = 0.0
    n_regions = 0

    for r_idx in range(len(region_order)):
        row_dec = r_idx * n_sec + dec_local
        row_inc = r_idx * n_sec + inc_local
        delta = shock_pct * Y_arr[row_dec, :]
        if delta.sum() == 0:
            continue
        Y_arr[row_dec, :] -= delta
        Y_arr[row_inc, :] += delta
        total_redirected += float(delta.sum())
        n_regions += 1

    Y_shocked = pd.DataFrame(Y_arr, index=Y.index, columns=Y.columns)

    dec_mask = Y.index.get_level_values("sector") == decrease
    inc_mask = Y.index.get_level_values("sector") == increase
    info = {
        "mode":                 "demand",
        "decrease":             decrease,
        "increase":             increase,
        "shock_pct":            shock_pct,
        "total_redirected_M€":  total_redirected,
        "n_regions":            n_regions,
        "Y_decrease_before_M€": float(Y.values[dec_mask].sum()),
        "Y_increase_before_M€": float(Y.values[inc_mask].sum()),
        # Compatibilité avec plot_shock_impact
        "origin":   decrease,
    }
    return Y_shocked, info


def apply_Y_shock_multi(
    Y: pd.DataFrame,
    decrease_sectors: list[str],
    increase_sectors: list[str],
    shock_pct: float,
    demand_categories: list[str] | None = None,
    distribute: str = "proportional",
) -> tuple[pd.DataFrame, dict]:
    """
    Choc multi-secteurs sur Y : redirige shock_pct de la demande finale
    depuis plusieurs secteurs (decrease) vers plusieurs secteurs (increase).

    Parameters
    ----------
    Y : pd.DataFrame (n×f)
    decrease_sectors : secteurs dont la demande diminue
    increase_sectors : secteurs dont la demande augmente
    shock_pct : fraction redirigée (0.10 = 10 %)
    demand_categories : list de catégories Y à cibler (None = toutes)
        ex. ["Final consumption expenditure by government",
              "Gross fixed capital formation"]
    distribute : 'proportional' (aux flux existants) | 'equal'

    Returns
    -------
    Y_shocked : pd.DataFrame
    info      : dict
    """
    sector_order = Y.index.get_level_values("sector").unique().tolist()
    region_order = Y.index.get_level_values("region").unique().tolist()
    n_sec = len(sector_order)
    sec_idx = {s: i for i, s in enumerate(sector_order)}

    dec_local = [sec_idx[s] for s in decrease_sectors if s in sec_idx]
    inc_local = [sec_idx[s] for s in increase_sectors if s in sec_idx]

    missing_dec = [s for s in decrease_sectors if s not in sec_idx]
    missing_inc = [s for s in increase_sectors if s not in sec_idx]
    if missing_dec:
        print(f"  [Warning] decrease non trouvés dans Y : {missing_dec}")
    if missing_inc:
        print(f"  [Warning] increase non trouvés dans Y : {missing_inc}")

    if demand_categories:
        col_cats = Y.columns.get_level_values(1)
        col_mask = np.isin(col_cats, demand_categories)
    else:
        col_mask = np.ones(Y.shape[1], dtype=bool)

    Y_arr = Y.values.copy().astype(float)
    total_redirected = 0.0

    for r_idx in range(len(region_order)):
        offset = r_idx * n_sec

        # --- Accumule le flux à rediriger (proportionnel à chaque secteur decrease) ---
        delta = np.zeros(Y.shape[1])
        for li in dec_local:
            row = offset + li
            d = shock_pct * np.where(col_mask, Y_arr[row, :], 0.0)
            Y_arr[row, :] -= d
            delta += d

        if delta.sum() == 0:
            continue

        # --- Redistribue vers les secteurs increase ---
        if distribute == "proportional":
            inc_flows = np.zeros(Y.shape[1])
            for li in inc_local:
                inc_flows += np.maximum(Y_arr[offset + li, :], 0.0)
            inc_total = inc_flows.sum()

            for li in inc_local:
                row = offset + li
                if inc_total > 0:
                    # Part de ce secteur dans le total increase, colonne par colonne
                    col_inc = np.maximum(Y_arr[row, :], 0.0)
                    col_total = np.where(inc_flows > 0, inc_flows, 1.0)
                    share = col_inc / col_total
                    Y_arr[row, :] += delta * share
                else:
                    Y_arr[row, :] += delta / len(inc_local)
        else:  # equal
            for li in inc_local:
                Y_arr[offset + li, :] += delta / len(inc_local)

        total_redirected += float(delta.sum())

    Y_shocked = pd.DataFrame(Y_arr, index=Y.index, columns=Y.columns)

    info = {
        "mode":                 "demand_multi",
        "decrease_sectors":     decrease_sectors,
        "increase_sectors":     increase_sectors,
        "n_decrease_matched":   len(dec_local),
        "n_increase_matched":   len(inc_local),
        "shock_pct":            shock_pct,
        "demand_categories":    demand_categories,
        "distribute":           distribute,
        "total_redirected_M€":  total_redirected,
        # Compatibilité plot_shock_impact
        "decrease": f"[{len(dec_local)} secteurs fossiles]",
        "increase": f"[{len(inc_local)} secteurs renouvelables]",
        "origin":   "GOV + GFCF",
    }
    return Y_shocked, info


def run_green_deal_scenario(
    Z: pd.DataFrame,
    x: pd.Series,
    Y: pd.DataFrame,
    S: pd.DataFrame,
    F_Y: pd.DataFrame,
    shock_pct: float = 0.10,
    demand_categories: list[str] | None = None,
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> dict:
    """
    Scénario Green New Deal :
    Redirige shock_pct des dépenses publiques (GOV) et d'investissement (GFCF)
    depuis les secteurs d'électricité fossile vers les secteurs d'électricité
    renouvelable et la construction d'infrastructures vertes.

    Secteurs fossiles (decrease) :
        Electricity by coal, gas, petroleum

    Secteurs renouvelables (increase) :
        Electricity by wind, solar PV, solar thermal, hydro, geothermal,
        biomass & waste, tide/wave + Construction work (45)

    Catégories de demande ciblées (par défaut) :
        - Final consumption expenditure by government
        - Gross fixed capital formation

    Parameters
    ----------
    shock_pct : float
        Fraction de la demande fossile redirigée (0.10 = 10 %).
    demand_categories : list[str], optional
        Catégories Y à cibler. None = GOV + GFCF.
    stressor_agg : list[str], optional
        Sous-ensemble de stresseurs CO2 (ou autre extension).
    concordance : pd.DataFrame, optional

    Returns
    -------
    dict avec 'fp_baseline', 'fp_shocked', 'delta', 'delta_pct', 'info'
    """
    if concordance is None:
        concordance = load_concordance()

    FOSSIL_DECREASE = [
        "Electricity by coal",
        "Electricity by gas",
        "Electricity by petroleum and other oil derivatives",
    ]
    RENEWABLE_INCREASE = [
        "Electricity by wind",
        "Electricity by solar photovoltaic",
        "Electricity by solar thermal",
        "Electricity by hydro",
        "Electricity by Geothermal",
        "Electricity by biomass and waste",
        "Electricity by tide, wave, ocean",
        "Construction work (45)",          # infrastructures renouvelables
    ]

    if demand_categories is None:
        demand_categories = [
            "Final consumption expenditure by government",
            "Gross fixed capital formation",
        ]

    print(f"Scenario Green New Deal : {shock_pct*100:.0f}% des depenses fossiles (GOV+GFCF)")
    print(f"  Decrease : {FOSSIL_DECREASE}")
    print(f"  Increase : {RENEWABLE_INCREASE}")
    print(f"  Categories ciblees : {demand_categories}")

    A = pd.DataFrame(
        Z.values * _safe_inv(x.values)[np.newaxis, :],
        index=Z.index, columns=Z.columns,
    )

    print("  [1/4] Empreinte baseline (base production)...")
    fp_base = compute_production_gics_footprint(S, x, stressor_agg, concordance)

    print("  [2/4] Application du choc Green Deal sur Y...")
    Y_shocked, info = apply_Y_shock_multi(
        Y, FOSSIL_DECREASE, RENEWABLE_INCREASE,
        shock_pct, demand_categories, distribute="proportional",
    )
    print(f"         {info['total_redirected_M€']:.1f} M€ redirigés "
          f"({info['n_decrease_matched']} secteurs fossiles -> "
          f"{info['n_increase_matched']} secteurs verts)")

    print("  [3/4] Propagation Leontief : Delta_x = L . Delta_y...")
    lu, _ = _build_leontief_sparse(A)
    delta_y = (
        Y_shocked.reindex(index=A.index, fill_value=0.0).sum(axis=1).values
        - Y.reindex(index=A.index, fill_value=0.0).sum(axis=1).values
    )
    delta_x = lu.solve(delta_y)
    x_shocked = pd.Series(x.values + delta_x, index=x.index)

    dx = pd.Series(delta_x, index=A.index)
    top5 = dx.abs().sort_values(ascending=False).head(5)
    print("         Top 5 secteurs par |Delta_x| (M€) :")
    for idx, val in top5.items():
        print(f"           {idx[0]} | {idx[1]:<45s} {val:+.1f}")

    print("  [4/4] Empreinte post-choc (base production)...")
    fp_shocked = compute_production_gics_footprint(S, x_shocked, stressor_agg, concordance)

    delta     = fp_shocked - fp_base
    delta_pct = (delta / fp_base.replace(0, np.nan)) * 100

    info["delta_x_total_M€"] = float(delta_x.sum())

    return {
        "fp_baseline": fp_base,
        "fp_shocked":  fp_shocked,
        "delta":       delta,
        "delta_pct":   delta_pct,
        "info":        info,
    }


def run_combined_shock(
    Z: pd.DataFrame,
    x: pd.Series,
    Y: pd.DataFrame,
    S: pd.DataFrame,
    Z_shocked: pd.DataFrame,
    Y_shocked: pd.DataFrame,
    z_info: dict,
    y_info: dict,
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> dict:
    """
    Choc combiné Z + Y : x' déduit directement de l'identité comptable.

    L'identité de bilan (toujours vraie dans les tables IO) :
        x  = Z  · 1  +  Y  · 1
        x' = Z' · 1  +  Y' · 1      ← aucune inversion, exact par construction

    Aucune approximation : on choque directement les matrices Z et Y, et x'
    suit par simple sommation des lignes. On peut ensuite recalculer si besoin :
        A' = Z' · x̂'⁻¹    (nouveaux coefficients techniques)

    Utilisation typique
    -------------------
    >>> Z_s, z_info = apply_gics_shock(Z, "Industrials", "Materials", "Energy", 0.05)
    >>> Y_s, y_info = apply_Y_shock_multi(Y, fossil, renewable, 0.10, [GOV, GFCF])
    >>> results = run_combined_shock(Z, x, Y, S, Z_s, Y_s, z_info, y_info,
    ...                              stressor_agg=CO2)

    Parameters
    ----------
    Z, x, Y, S  : matrices baseline
    Z_shocked    : Z après Z-choc (sortie de apply_shock / apply_gics_shock)
    Y_shocked    : Y après Y-choc (sortie de apply_Y_shock / apply_Y_shock_multi)
    z_info       : dict info du Z-choc
    y_info       : dict info du Y-choc
    stressor_agg : list de stresseurs à agréger
    concordance  : pd.DataFrame, optional

    Returns
    -------
    dict avec 'fp_baseline', 'fp_shocked', 'delta', 'delta_pct', 'x_shocked', 'info'
    """
    if concordance is None:
        concordance = load_concordance()

    print("Choc combine Z + Y -> x' = Z'.1 + Y'.1 (identite comptable)")

    # ------------------------------------------------------------------ baseline
    print("  [1/3] Empreinte baseline...")
    fp_base = compute_production_gics_footprint(S, x, stressor_agg, concordance)

    # ------------------------------------------------------------------ x' = Z'·1 + Y'·1
    print("  [2/3] x' = Z'.1 + Y'.1 ...")
    x_shocked = (
        Z_shocked.sum(axis=1)
        + Y_shocked.reindex(index=Z_shocked.index, fill_value=0.0).sum(axis=1)
    )
    x_shocked.index = x.index

    dx = x_shocked - x
    print(f"         Delta_x total : {dx.sum():+.1f} M€")
    top5 = dx.abs().sort_values(ascending=False).head(5)
    print("         Top 5 secteurs |Delta_x| :")
    for idx, val in top5.items():
        print(f"           {idx[0]} | {idx[1]:<45s} {val:+.1f}")

    # ------------------------------------------------------------------ empreinte post-choc
    print("  [3/3] Empreinte post-choc...")
    fp_shocked = compute_production_gics_footprint(S, x_shocked, stressor_agg, concordance)

    delta     = fp_shocked - fp_base
    delta_pct = (delta / fp_base.replace(0, np.nan)) * 100

    info = {
        "mode":      "combined",
        "z_shock":   z_info,
        "y_shock":   y_info,
        "shock_pct": z_info.get("shock_pct", float("nan")),
        "total_redirected_M€": (
            z_info.get("total_redirected_M€", 0)
            + y_info.get("total_redirected_M€", 0)
        ),
        # Compat plot_shock_impact
        "decrease": (
            f"Z:[{z_info.get('decrease','?')}] + "
            f"Y:[{y_info.get('decrease','?')}]"
        ),
        "increase": (
            f"Z:[{z_info.get('increase','?')}] + "
            f"Y:[{y_info.get('increase','?')}]"
        ),
        "origin": "Z+Y combined",
    }

    return {
        "fp_baseline": fp_base,
        "fp_shocked":  fp_shocked,
        "delta":       delta,
        "delta_pct":   delta_pct,
        "x_shocked":   x_shocked,
        "info":        info,
    }


# ---------------------------------------------------------------------------
# 1d. Choc de redirection pollution : top fournisseurs sales -> propres
# ---------------------------------------------------------------------------

def apply_pollution_redirect_shock(
    Z: pd.DataFrame,
    S: pd.DataFrame,
    gics_buyer: str,
    stressor: str | list[str],
    shock_pct: float,
    n_top: int = 10,
    n_alternatives: int | None = None,
    distribute: str = "proportional",
    min_supply_M: float = 100.0,
    clean_criterion: str = "intensity",
    restrict_alternatives_to_gics: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Choc Z ciblé : pour le GICS `gics_buyer`, identifie les n_top secteurs
    fournisseurs les plus polluants (en termes d'émissions induites), puis
    redirige shock_pct de leur flux vers les n_alternatives fournisseurs
    les moins polluants.

    Etape 1 — Pollution induite par fournisseur :
        pollution[i] = S[stressor, i] · Σ_{j ∈ gics_buyer} Z[i, j]
        Agrégée par secteur (somme sur les régions).

    Etape 2 — Sélection :
        top_dirty   : n_top secteurs par pollution induite descendante
                      (plus grand levier de réduction).
        bottom_clean : n_alternatives parmi les secteurs ayant supply
                      ≥ min_supply_M, classés par :
            'intensity' (défaut) → effective_intensity = pollution/supply
                                   ascendante (technologie la plus propre
                                   par € — évite de prendre des secteurs
                                   à très haute intensité mais petit flux).
            'emission'           → pollution absolue ascendante.

    Etape 3 — Application, pour chaque colonne acheteur j ∈ gics_buyer :
        delta_j = shock_pct · Σ_{i ∈ dirty} Z[i, j]
        Z[i, j] *= (1 − shock_pct)  pour i ∈ dirty
        Distribution de delta_j sur les "propres" :
            - proportional : au prorata des flux existants
            - equal        : parts égales

    Parameters
    ----------
    Z : pd.DataFrame
    S : pd.DataFrame                — intensités stresseurs (e×n)
    gics_buyer : str                — GICS acheteur (ex: "Industrials")
    stressor : str | list[str]      — nom(s) de stresseur(s) à sommer
    shock_pct : float               — fraction redirigée (0.5 = 50 %)
    n_top : int                     — nb de fournisseurs sales à dégrader
    n_alternatives : int, optional  — nb d'alternatives propres
                                       (par défaut = n_top)
    distribute : 'proportional' | 'equal'
    min_supply_M : float            — seuil M€ pour qu'un secteur soit
                                       éligible comme alternative propre
                                       (défaut 100 M€).
    clean_criterion : 'intensity' | 'emission'
        Métrique de classement des alternatives propres. 'intensity'
        (défaut) évite les pièges des secteurs à petit flux mais
        haute intensité (Patent Fuel, Peat, ...).
    restrict_alternatives_to_gics : list[str], optional
        Si fourni, restreint les alternatives aux GICS de cette liste.
    concordance : pd.DataFrame, optional

    Returns
    -------
    Z_shocked : pd.DataFrame
    info      : dict (sectors choisis, pollution avant/apres, M€ redirigés)
    """
    if concordance is None:
        concordance = load_concordance()
    if n_alternatives is None:
        n_alternatives = n_top

    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    all_sectors = Z.index.get_level_values("sector").unique()
    gics_of_sector = {s: _map_sector(s, lookup, norm_lookup) for s in all_sectors}

    # --- Colonnes acheteuses : secteurs ∈ gics_buyer ---
    col_sectors = Z.columns.get_level_values("sector")
    buyer_mask  = np.array([gics_of_sector.get(s) == gics_buyer for s in col_sectors])
    if not buyer_mask.any():
        raise ValueError(f"Aucun secteur EXIOBASE pour GICS '{gics_buyer}'.")

    # --- Intensité par (région, secteur) ---
    stressors = [stressor] if isinstance(stressor, str) else list(stressor)
    S_sub     = S.loc[S.index.intersection(stressors)]
    if S_sub.empty:
        raise ValueError(f"Aucun stresseur trouve : {stressors}.")
    intensity = S_sub.sum(axis=0).reindex(Z.index, fill_value=0.0).values  # (n,)

    Z_arr           = Z.values
    supply_to_gics  = Z_arr[:, buyer_mask].sum(axis=1)   # (n,) flux i -> gics_buyer
    pollution_cell  = supply_to_gics * intensity          # (n,) émissions induites

    pollution_by_sector = (
        pd.Series(pollution_cell, index=Z.index).groupby(level="sector").sum()
    )
    supply_by_sector = (
        pd.Series(supply_to_gics, index=Z.index).groupby(level="sector").sum()
    )

    # --- Top dirty ---
    top_dirty = pollution_by_sector.sort_values(ascending=False).head(n_top).index.tolist()

    # --- Alternatives propres ---
    eligible = supply_by_sector[supply_by_sector >= min_supply_M].index
    eligible = [s for s in eligible if s not in top_dirty]
    if restrict_alternatives_to_gics is not None:
        eligible = [s for s in eligible
                     if gics_of_sector.get(s) in restrict_alternatives_to_gics]
    if not eligible:
        raise ValueError(
            "Aucun secteur eligible comme alternative propre "
            f"(supply >= {min_supply_M} M€)."
        )

    if clean_criterion == "intensity":
        # Intensité effective = kg / € de supply au GICS (pollution/supply)
        effective_intensity = (
            pollution_by_sector.loc[eligible] / supply_by_sector.loc[eligible]
        ).sort_values(ascending=True)
        bottom_clean = effective_intensity.head(n_alternatives).index.tolist()
    elif clean_criterion == "emission":
        bottom_clean = (
            pollution_by_sector.loc[eligible].sort_values(ascending=True)
            .head(n_alternatives).index.tolist()
        )
    else:
        raise ValueError(
            f"clean_criterion doit être 'intensity' ou 'emission', "
            f"reçu : {clean_criterion!r}"
        )

    # --- Application du choc ---
    row_sectors     = Z.index.get_level_values("sector")
    dirty_rows_mask = np.isin(row_sectors, top_dirty)
    clean_rows_mask = np.isin(row_sectors, bottom_clean)
    n_clean         = int(clean_rows_mask.sum())

    Z_new            = Z_arr.copy()
    total_redirected = 0.0
    buyer_cols       = np.where(buyer_mask)[0]

    for j in buyer_cols:
        delta = shock_pct * Z_new[dirty_rows_mask, j].sum()
        if delta <= 0:
            continue
        Z_new[dirty_rows_mask, j] *= (1 - shock_pct)
        clean_flows = Z_new[clean_rows_mask, j]
        clean_total = float(clean_flows.sum())
        if distribute == "proportional" and clean_total > 0:
            share = clean_flows / clean_total
            Z_new[clean_rows_mask, j] += delta * share
        else:
            Z_new[clean_rows_mask, j] += delta / max(n_clean, 1)
        total_redirected += float(delta)

    Z_shocked = pd.DataFrame(Z_new, index=Z.index, columns=Z.columns)

    # Intensités effectives (kg/€) pour reporting
    eff_int_all = pollution_by_sector / supply_by_sector.replace(0, np.nan)

    info = {
        "mode":                 "pollution_redirect",
        "gics_buyer":           gics_buyer,
        "stressor":             stressors[0] if len(stressors) == 1
                                 else f"{len(stressors)} stresseurs",
        "shock_pct":            shock_pct,
        "n_top":                n_top,
        "n_alternatives":       len(bottom_clean),
        "dirty_sectors":        top_dirty,
        "clean_sectors":        bottom_clean,
        "dirty_pollution":      pollution_by_sector.loc[top_dirty].to_dict(),
        "clean_pollution":      pollution_by_sector.loc[bottom_clean].to_dict(),
        "dirty_intensity":      eff_int_all.loc[top_dirty].to_dict(),
        "clean_intensity":      eff_int_all.loc[bottom_clean].to_dict(),
        "total_redirected_M€":  total_redirected,
        "distribute":           distribute,
        "clean_criterion":      clean_criterion,
        "min_supply_M":         min_supply_M,
        # Champs de compatibilité pour plot_shock_impact
        "origin":   gics_buyer,
        "decrease": f"top {n_top} polluants",
        "increase": f"top {len(bottom_clean)} propres",
    }
    return Z_shocked, info


def run_pollution_redirect_comparison(
    Z: pd.DataFrame,
    x: pd.Series,
    Y: pd.DataFrame,
    S: pd.DataFrame,
    F_Y: pd.DataFrame,
    gics_buyer: str,
    stressor: str | list[str],
    shock_pct: float = 0.5,
    n_top: int = 10,
    n_alternatives: int | None = None,
    distribute: str = "proportional",
    min_supply_M: float = 100.0,
    clean_criterion: str = "intensity",
    restrict_alternatives_to_gics: list[str] | None = None,
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> dict:
    """
    Pipeline complet du choc "redirection pollution" :
    repère les top-N fournisseurs sales d'un GICS, redirige shock_pct vers
    les N alternatives propres, puis recalcule l'empreinte via l'identité
    comptable x' = Z'·1 + Y·1 (exact, sans Leontief).

    Equivalent : sociologiquement, "le GICS achète moins aux fournisseurs
    les plus carbonés et plus aux moins carbonés à efficacité comparable".

    Parameters
    ----------
    Z, x, Y, S, F_Y : matrices du système IO
    gics_buyer : str
    stressor   : str | list[str]   — stresseur(s) pour identifier les sales
    shock_pct  : float             — fraction redirigée (0.5 par défaut)
    n_top, n_alternatives, distribute, min_supply_M,
    restrict_alternatives_to_gics : voir apply_pollution_redirect_shock
    stressor_agg : list[str], optional
        Stresseur(s) à agréger pour le calcul d'empreinte finale.
        Si None, on utilise `stressor`.
    concordance : pd.DataFrame, optional

    Returns
    -------
    dict avec 'fp_baseline', 'fp_shocked', 'delta', 'delta_pct',
              'x_shocked', 'info'
    """
    if concordance is None:
        concordance = load_concordance()

    if stressor_agg is None:
        stressor_agg = [stressor] if isinstance(stressor, str) else list(stressor)

    print(f"Pollution redirect : {shock_pct*100:.0f}% des achats sales de [{gics_buyer}]")
    print(f"  Top {n_top} polluants -> top {n_alternatives or n_top} propres")

    print("  [1/4] Empreinte baseline (production-based)...")
    fp_base = compute_production_gics_footprint(S, x, stressor_agg, concordance)

    print("  [2/4] Identification top dirty / clean + application du choc...")
    Z_shocked, info = apply_pollution_redirect_shock(
        Z, S, gics_buyer, stressor, shock_pct,
        n_top, n_alternatives, distribute, min_supply_M,
        clean_criterion, restrict_alternatives_to_gics, concordance,
    )
    print(f"         Flux redirige : {info['total_redirected_M€']:,.1f} M€")
    print(f"         Critere clean : {info['clean_criterion']} "
          f"(min_supply >= {info['min_supply_M']} M€)")
    print("         Top 5 fournisseurs sales (intensite kg/€) :")
    for s in info["dirty_sectors"][:5]:
        i = info["dirty_intensity"][s]
        p = info["dirty_pollution"][s]
        print(f"            {s:55s} intensite={i:.2f}  pollution_tot={p:.2e}")
    print("         Top 5 alternatives propres (intensite kg/€) :")
    for s in info["clean_sectors"][:5]:
        i = info["clean_intensity"][s]
        p = info["clean_pollution"][s]
        print(f"            {s:55s} intensite={i:.2f}  pollution_tot={p:.2e}")

    print("  [3/4] x' = Z'.1 + Y.1 (identite comptable, sans Leontief)...")
    x_shocked = (
        Z_shocked.sum(axis=1)
        + Y.reindex(index=Z_shocked.index, fill_value=0.0).sum(axis=1)
    )
    x_shocked.index = x.index
    dx = x_shocked - x
    print(f"         Delta_x total : {dx.sum():+.1f} M€")

    print("  [4/4] Empreinte post-choc (production-based)...")
    fp_shocked = compute_production_gics_footprint(S, x_shocked, stressor_agg, concordance)

    delta     = fp_shocked - fp_base
    delta_pct = (delta / fp_base.replace(0, np.nan)) * 100

    return {
        "fp_baseline": fp_base,
        "fp_shocked":  fp_shocked,
        "delta":       delta,
        "delta_pct":   delta_pct,
        "x_shocked":   x_shocked,
        "info":        info,
    }


def run_Y_shock_comparison(
    Z: pd.DataFrame,
    x: pd.Series,
    Y: pd.DataFrame,
    S: pd.DataFrame,
    F_Y: pd.DataFrame,
    decrease: str,
    increase: str,
    shock_pct: float,
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> dict:
    """
    Pipeline complet pour un choc de demande finale sur Y.

    Modèle :
        ΔY[(r, decrease), :] = -shock_pct · Y[(r, decrease), :]   (demande perdue)
        ΔY[(r, increase), :] = +shock_pct · Y[(r, decrease), :]   (demande gagnée)
        A inchangé → L = (I−A)⁻¹ inchangé
        x'         = L · y'                (y' = Y'·1)
        footprint' = S · L · Y' + F_Y

    Propagation Leontief complète : tous les effets indirects de chaîne
    (ex. moins de gaz extrait, moins de transport gazier...) sont capturés.

    Comparaison des trois types de chocs :
        Z-choc (run_shock_comparison)           : change A (technologie)
                                                  → L change → empreinte consommation
        x-choc (run_production_shock_comparison): change x (volumes)
                                                  → calcul direct S·Δx, pas de LU
        Y-choc (run_Y_shock_comparison)         : change Y (demande)
                                                  → L inchangé, propagation via L·Δy

    Parameters
    ----------
    Z, x, Y, S, F_Y  : matrices du système IO (from load_io_system + compute_S)
    decrease, increase: secteurs dans Y
    shock_pct         : fraction redirigée (0.5 = 50 %)
    stressor_agg      : list[str], optional
    concordance       : pd.DataFrame, optional

    Returns
    -------
    dict avec 'fp_baseline', 'fp_shocked', 'delta', 'delta_pct', 'info'
    """
    if concordance is None:
        concordance = load_concordance()

    print(f"Choc demande finale : {shock_pct*100:.0f}% de [{decrease}] -> [{increase}]")

    # A inchangé : même technologie de production
    A = pd.DataFrame(
        Z.values * _safe_inv(x.values)[np.newaxis, :],
        index=Z.index, columns=Z.columns,
    )

    # --- Empreinte baseline (base production : attribue les émissions aux secteurs producteurs) ---
    print("  [1/4] Empreinte baseline (base production)...")
    fp_base = compute_production_gics_footprint(S, x, stressor_agg, concordance)

    # --- Choc sur Y ---
    print("  [2/4] Application du choc sur Y...")
    Y_shocked, info = apply_Y_shock(Y, decrease, increase, shock_pct)
    print(f"         {info['n_regions']} regions touchees, "
          f"{info['total_redirected_M€']:.1f} M€ de demande redirigee")
    print(f"         Y[{decrease}] avant : {info['Y_decrease_before_M€']:.1f} M€  "
          f"-> apres : {info['Y_decrease_before_M€'] - info['total_redirected_M€']:.1f} M€")
    print(f"         Y[{increase}] avant : {info['Y_increase_before_M€']:.1f} M€  "
          f"-> apres : {info['Y_increase_before_M€'] + info['total_redirected_M€']:.1f} M€")

    # --- Propagation Leontief : Delta_x = L · Delta_y  (un seul solve LU) ---
    # La demande finale change → la production de TOUS les secteurs s'ajuste.
    # C'est ici que la chaîne d'approvisionnement se propage :
    #   moins de demande d'élec. gaz → moins de production de gaz (Energy),
    #   moins de métaux pour l'infrastructure gaz (Materials), etc.
    print("  [3/4] Propagation Leontief : Delta_x = L . Delta_y...")
    lu, _ = _build_leontief_sparse(A)

    delta_y = (
        Y_shocked.reindex(index=A.index, fill_value=0.0).sum(axis=1).values
        - Y.reindex(index=A.index, fill_value=0.0).sum(axis=1).values
    )                                           # (n,)  variation de demande finale totale
    delta_x = lu.solve(delta_y)                # (n,)  L · Delta_y = variation de production

    x_shocked = pd.Series(x.values + delta_x, index=x.index)

    # Secteurs dont la production change le plus
    dx_series = pd.Series(delta_x, index=A.index)
    top_dx = dx_series.abs().sort_values(ascending=False).head(5)
    print("         Top 5 secteurs par |Delta_x| (M€) :")
    for idx, val in top_dx.items():
        print(f"           {idx[0]} | {idx[1]:<45s} {val:+.1f}")

    # --- Empreinte post-choc (production, avec x') ---
    print("  [4/4] Empreinte post-choc (base production, x apres Leontief)...")
    fp_shocked = compute_production_gics_footprint(S, x_shocked, stressor_agg, concordance)

    delta     = fp_shocked - fp_base
    delta_pct = (delta / fp_base.replace(0, np.nan)) * 100

    # Stocke Delta_x par GICS pour diagnostic
    info["delta_x_M€_total"] = float(delta_x.sum())
    info["x_shocked"] = x_shocked

    return {
        "fp_baseline": fp_base,
        "fp_shocked":  fp_shocked,
        "delta":       delta,
        "delta_pct":   delta_pct,
        "info":        info,
    }


# ---------------------------------------------------------------------------
# 2. Empreinte agrégée par GICS
# ---------------------------------------------------------------------------

def _sector_to_gics(index: pd.MultiIndex, concordance: pd.DataFrame) -> list[str]:
    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    return [_map_sector(s, lookup, norm_lookup) for s in index.get_level_values("sector")]


def compute_gics_footprint(
    A: pd.DataFrame,
    S: pd.DataFrame,
    Y: pd.DataFrame,
    F_Y: pd.DataFrame,
    x: pd.Series,
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Calcule l'empreinte totale agrégée par GICS du secteur PRODUCTEUR.

    Pour chaque groupe GICS g, on calcule :
        fp_g = Σ_k S_k · (I−A)⁻¹ · y_g  +  Σ_k F_Y_k · 1_g

    où y_g[i] = y[i] si le secteur i ∈ GICS g, sinon 0
    et y = Y · 1 (demande finale totale par secteur).

    Cela donne l'émission totale causée par la production des secteurs GICS g
    pour répondre à l'ensemble de la demande finale mondiale.

    Parameters
    ----------
    stressor_agg : list[str], optional
        Sous-ensemble de stresseurs à sommer. Si None, somme tous les stresseurs.

    Returns
    -------
    pd.Series indexée par GICS group (11 valeurs)
    """
    if concordance is None:
        concordance = load_concordance()

    gics_labels = _sector_to_gics(A.index, concordance)
    gics_series = pd.Series(gics_labels, index=A.index)

    y = Y.reindex(index=A.index, fill_value=0.0).sum(axis=1)   # (n,) demande finale totale

    lu, _ = _build_leontief_sparse(A)

    # Filtrage stresseurs
    if stressor_agg:
        S_use   = S.loc[S.index.intersection(stressor_agg)]
        F_Y_use = F_Y.loc[F_Y.index.intersection(stressor_agg)]
    else:
        S_use, F_Y_use = S, F_Y

    S_arr   = S_use.reindex(columns=A.index, fill_value=0.0).values   # (e, n)
    F_Y_arr = F_Y_use.reindex(columns=Y.columns, fill_value=0.0).values  # (e, f)

    result = {}
    unique_gics = [g for g in GICS_ORDER if g in set(gics_labels)] + \
                  [g for g in set(gics_labels) if g not in GICS_ORDER]

    for g in unique_gics:
        mask = np.array(gics_series == g)
        y_g = np.where(mask, y.values, 0.0)   # (n,) — demande finale filtrée sur GICS g

        q_g = lu.solve(y_g)                    # (I−A)⁻¹ · y_g
        indirect = (S_arr @ q_g).sum()          # Σ stresseurs des émissions indirectes

        # Émissions directes de la demande finale allouées à GICS g
        # F_Y regroupe toutes catégories — on alloue proportionnellement à la part de y_g dans y
        y_total = y.values.sum()
        share_g = y_g.sum() / y_total if y_total > 0 else 0.0
        direct = F_Y_arr.sum() * share_g

        result[g] = indirect + direct

    return pd.Series(result)


# ---------------------------------------------------------------------------
# 3. Pipeline complet choc → comparaison d'empreintes
# ---------------------------------------------------------------------------

def run_shock_comparison(
    Z: pd.DataFrame,
    x: pd.Series,
    Y: pd.DataFrame,
    S: pd.DataFrame,
    F_Y: pd.DataFrame,
    origin: str,
    decrease: str,
    increase: str,
    shock_pct: float,
    scope: str = "intra",
    stressor_agg: list[str] | None = None,
    concordance: pd.DataFrame | None = None,
) -> dict:
    """
    Pipeline complet :
        1. Applique le choc sur Z
        2. Recalcule A' = Z' · x̂⁻¹
        3. Calcule empreinte GICS baseline et shocked
        4. Retourne baseline, shocked, delta et infos choc

    Returns
    -------
    dict avec :
        'fp_baseline' : pd.Series GICS
        'fp_shocked'  : pd.Series GICS
        'delta'       : pd.Series GICS (shocked − baseline)
        'delta_pct'   : pd.Series GICS
        'info'        : dict infos du choc
    """
    if concordance is None:
        concordance = load_concordance()

    print(f"Choc : {shock_pct*100:.0f}% de [{origin}] -> [{decrease}] redirige vers [{increase}]")

    # Baseline
    print("  [1/3] Empreinte baseline...")
    A_base = pd.DataFrame(
        Z.values * _safe_inv(x.values)[np.newaxis, :],
        index=Z.index, columns=Z.columns,
    )
    fp_base = compute_gics_footprint(A_base, S, Y, F_Y, x, stressor_agg, concordance)

    # Choc
    print("  [2/3] Application du choc...")
    Z_shocked, info = apply_shock(Z, origin, decrease, increase, shock_pct, scope)
    print(f"         {info['n_region_pairs']} paires de régions touchées, "
          f"{info['total_redirected_M€']:.2f} M€ redirigés "
          f"sur {info['total_flow_before']:.2f} M€ de flux total")

    # Footprint choquée
    print("  [3/3] Empreinte post-choc...")
    A_shocked = pd.DataFrame(
        Z_shocked.values * _safe_inv(x.values)[np.newaxis, :],
        index=Z_shocked.index, columns=Z_shocked.columns,
    )
    fp_shocked = compute_gics_footprint(A_shocked, S, Y, F_Y, x, stressor_agg, concordance)

    delta = fp_shocked - fp_base
    delta_pct = (delta / fp_base.replace(0, np.nan)) * 100

    return {
        "fp_baseline": fp_base,
        "fp_shocked":  fp_shocked,
        "delta":       delta,
        "delta_pct":   delta_pct,
        "info":        info,
    }


# ---------------------------------------------------------------------------
# 4. Visualisation
# ---------------------------------------------------------------------------

def _auto_scale(values: list[float], base_unit: str) -> tuple[float, str]:
    """Choisit l'échelle et l'unité en fonction de la magnitude des valeurs."""
    max_abs = max(abs(v) for v in values if v != 0) if any(v != 0 for v in values) else 1
    if max_abs >= 1e12:
        return 1e12, f"Gt {base_unit}"
    elif max_abs >= 1e9:
        return 1e9,  f"Mt {base_unit}"
    elif max_abs >= 1e6:
        return 1e6,  f"kt {base_unit}"
    elif max_abs >= 1e3:
        return 1e3,  f"t {base_unit}"
    else:
        return 1.0,  f"kg {base_unit}"


def _fmt_pct(v: float) -> str:
    """Formate un pourcentage avec suffisamment de décimales selon la magnitude."""
    if v == 0:
        return "0%"
    abs_v = abs(v)
    if abs_v >= 0.1:
        return f"{v:+.2f}%"
    elif abs_v >= 0.01:
        return f"{v:+.3f}%"
    elif abs_v >= 1e-3:
        return f"{v:+.4f}%"
    else:
        return f"{v:+.2e}%"


def plot_shock_impact(
    results: dict,
    stressor_label: str = "CO2",
    unit: str = "kg",
    save_path: str | None = None,
) -> go.Figure:
    """
    Deux sous-graphiques :
        (gauche)  Delta absolu par GICS — à l'échelle du choc (pas du footprint total)
        (droite)  Delta relatif par GICS — pourcentage adaptatif ou points de base

    Le graphe gauche est mis à l'échelle du DELTA (pas du footprint total) pour
    rendre les variations visibles même quand elles sont infimes relativement.

    Parameters
    ----------
    results : dict
        Sortie de run_shock_comparison().
    """
    info  = results["info"]
    fp_b  = results["fp_baseline"]
    delta = results["delta"]
    dpct  = results["delta_pct"]

    gics_list = [g for g in GICS_ORDER if g in fp_b.index]
    colors    = [GICS_COLORS.get(g, "#bdc3c7") for g in gics_list]

    # --- Échelle du footprint baseline (panneau info-bulle uniquement) ---
    scale_b, unit_b = _auto_scale([fp_b.get(g, 0) for g in gics_list], unit)

    # --- Échelle du delta : calculée sur les valeurs du delta lui-même ---
    d_raw  = [delta.get(g, 0) for g in gics_list]
    p_vals = [dpct.get(g, 0) for g in gics_list]
    scale_d, unit_d = _auto_scale(d_raw, unit)
    d_vals = [v / scale_d for v in d_raw]

    delta_colors = ["#e74c3c" if v > 0 else "#27ae60" for v in d_vals]
    pct_labels   = [_fmt_pct(v) for v in p_vals]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=[
            f"Delta absolu par GICS ({unit_d})",
            "Delta relatif par GICS (% ou points de base)",
        ],
        horizontal_spacing=0.14,
    )

    # --- Panneau gauche : delta absolu à son propre échelle ---
    hover_left = [
        f"<b>{g}</b><br>"
        f"Δ absolu : {d_raw[i]:+.3g} {unit}<br>"
        f"Baseline : {fp_b.get(g,0)/scale_b:.3g} {unit_b}<br>"
        f"Δ relatif : {pct_labels[i]}"
        for i, g in enumerate(gics_list)
    ]

    fig.add_trace(go.Bar(
        name="Δ absolu",
        x=gics_list,
        y=d_vals,
        marker_color=delta_colors,
        text=[f"{v:+.3g}" for v in d_vals],
        textposition="outside",
        textfont=dict(size=8),
        hovertext=hover_left,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(y=0, line_color="black", line_width=0.8, row=1, col=1)

    # --- Panneau droit : delta relatif (%) ---
    hover_right = [
        f"<b>{g}</b><br>"
        f"Δ relatif : {pct_labels[i]}<br>"
        f"Δ absolu : {d_raw[i]:+.3g} {unit}<br>"
        f"Baseline : {fp_b.get(g,0)/scale_b:.3g} {unit_b}"
        for i, g in enumerate(gics_list)
    ]

    fig.add_trace(go.Bar(
        name="Δ relatif",
        x=gics_list,
        y=p_vals,
        marker_color=delta_colors,
        text=pct_labels,
        textposition="outside",
        textfont=dict(size=8),
        hovertext=hover_right,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=0, line_color="black", line_width=0.8, row=1, col=2)

    # Titre global — adapté selon le mode du choc
    mode = info.get("mode", "flow")
    if mode == "production":
        shock_desc = (
            f"PRODUCTION {info['shock_pct']*100:.0f}%  "
            f"[{info['decrease']}] -> [{info['increase']}]   |   "
            f"{info['total_redirected_M€']:.1f} M€ redirigés   |   "
            f"Stresseur : {stressor_label}"
        )
    elif mode in ("demand", "demand_multi"):
        shock_desc = (
            f"DEMANDE FINALE {info['shock_pct']*100:.0f}%  "
            f"[{info['decrease']}] -> [{info['increase']}]   |   "
            f"{info['total_redirected_M€']:.1f} M€ redirigés   |   "
            f"Stresseur : {stressor_label}"
        )
    elif mode == "pollution_redirect":
        shock_desc = (
            f"REDIRECTION POLLUTION {info['shock_pct']*100:.0f}%  "
            f"[GICS {info['gics_buyer']}] : "
            f"top {info['n_top']} polluants -> top {info['n_alternatives']} propres   |   "
            f"{info['total_redirected_M€']:,.1f} M€ redirigés   |   "
            f"Stresseur : {stressor_label}"
        )
    else:
        shock_desc = (
            f"{info['shock_pct']*100:.0f}% de [{info.get('origin', '?')}] -> [{info['decrease']}] "
            f"redirigé vers [{info['increase']}]   |   "
            f"Flux : {info['total_redirected_M€']:.1f} M€   |   "
            f"Stresseur : {stressor_label}"
        )

    fig.update_layout(
        title=dict(text=f"Impact du choc {shock_desc}", font=dict(size=12), x=0.5),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=580,
        width=1250,
        margin=dict(l=60, r=60, t=100, b=130),
    )

    for col in [1, 2]:
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=9), row=1, col=col)

    # Annotation qui contextualise la petitesse relative des chocs
    max_abs_pct = max(abs(v) for v in p_vals) if p_vals else 0
    note = (
        f"Note : variations de l'ordre de {max_abs_pct:.1e}%<br>"
        f"(choc de {info['total_redirected_M€']:.1f} M€ / footprint total ~{fp_b.sum()/scale_b:.2g} {unit_b})"
    )
    fig.add_annotation(
        text=note, xref="paper", yref="paper",
        x=1.0, y=-0.22, showarrow=False,
        font=dict(size=8, color="#7f8c8d"),
        xanchor="right", align="right",
    )

    fig.update_yaxes(title_text=unit_d, row=1, col=1,
                     gridcolor="#ecf0f1", zeroline=True, zerolinecolor="black")
    fig.update_yaxes(title_text="% variation", row=1, col=2,
                     gridcolor="#ecf0f1", zeroline=True, zerolinecolor="black",
                     exponentformat="e", tickformat=".1e")

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from io_pipeline import load_io_system, load_extension, compute_S

    CO2 = [
        "CO2 - combustion - air",
        "CO2 - non combustion - Cement production - air",
        "CO2 - non combustion - Lime production - air",
        "CO2_bio - combustion - air",
    ]

    print("Chargement des données...")
    io  = load_io_system(2019, "pxp")
    ext = load_extension(2019, "air_emissions", "pxp")
    Z, x, Y = io["Z"], io["x"], io["Y"]
    F, F_Y  = ext["F"], ext["F_Y"]
    S = compute_S(F.loc[F.index.intersection(CO2)], x)
    F_Y_filt = F_Y.loc[F_Y.index.intersection(CO2)]

    # --- Choc secteur (exemple existant) ---
    results = run_shock_comparison(
        Z=Z, x=x, Y=Y, S=S, F_Y=F_Y_filt,
        origin   = "Electricity by gas",
        decrease = "Air transport services (62)",
        increase = "Construction work (45)",
        shock_pct = 0.30,
        scope     = "intra",
        stressor_agg = CO2,
    )

    print("\n--- Resultats (choc sectoriel) ---")
    print("Delta empreinte CO2 par GICS (kg) :")
    print(results["delta"].sort_values().to_string())
    print(f"\nDelta total : {results['delta'].sum():.3e} kg CO2")

    fig = plot_shock_impact(
        results,
        stressor_label="CO2 total",
        unit="kg CO2",
        save_path="Figs/shock_impact.html",
    )
    print("Graphique sauvegarde : Figs/shock_impact.html")

    # --- Choc GICS : Industrials reduit flux vers Materials, augmente vers Energy ---
    print("\n" + "="*60)
    print("Choc GICS : Industrials -> Materials (-1%) -> Energy (+1%)")
    print("="*60)

    results_gics = run_gics_shock_comparison(
        Z=Z, x=x, Y=Y, S=S, F_Y=F_Y_filt,
        origin_gics   = "Industrials",
        decrease_gics = "Materials",
        increase_gics = "Energy",
        shock_pct     = 0.01,
        scope         = "intra",
        distribute    = "proportional",
        stressor_agg  = CO2,
    )

    print("\n--- Resultats (choc GICS) ---")
    print("Delta empreinte CO2 par GICS (kg) :")
    print(results_gics["delta"].sort_values().to_string())
    print(f"\nDelta total : {results_gics['delta'].sum():.3e} kg CO2")
    print(f"Flux redirige : {results_gics['info']['total_redirected_M€']:.1f} M€")

    fig_gics = plot_shock_impact(
        results_gics,
        stressor_label="CO2 total",
        unit="kg CO2",
        save_path="Figs/shock_gics_impact.html",
    )
    print("Graphique sauvegarde : Figs/shock_gics_impact.html")

    # --- Choc de PRODUCTION : Electricity by gas -> Electricity by wind ---
    print("\n" + "="*60)
    print("Choc PRODUCTION : 50% Electricity by gas -> Electricity by wind")
    print("="*60)

    results_prod = run_production_shock_comparison(
        x          = x,
        S          = S,
        decrease   = "Electricity by gas",
        increase   = "Electricity by wind",
        shock_pct  = 0.50,
        scope      = "intra",
        stressor_agg = CO2,
    )

    print("\n--- Resultats (choc production) ---")
    print("Delta empreinte CO2 par GICS (production-based, kg) :")
    print(results_prod["delta"].sort_values().to_string())
    print(f"\nDelta total : {results_prod['delta'].sum():.3e} kg CO2")
    info_p = results_prod["info"]
    print(f"x[decrease] avant : {info_p['x_decrease_before_M€']:.0f} M€  "
          f"  x[increase] avant : {info_p['x_increase_before_M€']:.0f} M€")

    fig_prod = plot_shock_impact(
        results_prod,
        stressor_label = "CO2 total",
        unit           = "kg CO2",
        save_path      = "Figs/shock_production_impact.html",
    )
    print("Graphique sauvegarde : Figs/shock_production_impact.html")

    # --- Choc de DEMANDE FINALE sur Y : même exemple, troisième approche ---
    print("\n" + "="*60)
    print("Choc DEMANDE FINALE : 50% Y[Electricity by gas] -> Y[Electricity by wind]")
    print("="*60)

    results_Y = run_Y_shock_comparison(
        Z=Z, x=x, Y=Y, S=S, F_Y=F_Y_filt,
        decrease  = "Electricity by gas",
        increase  = "Electricity by wind",
        shock_pct = 0.50,
        stressor_agg = CO2,
    )

    print("\n--- Resultats (choc demande finale) ---")
    print("Delta empreinte CO2 par GICS (consommation-based, kg) :")
    print(results_Y["delta"].sort_values().to_string())
    print(f"\nDelta total : {results_Y['delta'].sum():.3e} kg CO2")

    fig_Y = plot_shock_impact(
        results_Y,
        stressor_label = "CO2 total",
        unit           = "kg CO2",
        save_path      = "Figs/shock_Y_impact.html",
    )
    print("Graphique sauvegarde : Figs/shock_Y_impact.html")

    # --- Scenario Green New Deal : 10% GOV+GFCF fossile -> renouvelable ---
    print("\n" + "="*60)
    print("Scenario GREEN NEW DEAL : 10% depenses publiques fossiles -> renouvelables")
    print("="*60)

    results_gnd = run_green_deal_scenario(
        Z=Z, x=x, Y=Y, S=S, F_Y=F_Y_filt,
        shock_pct=0.10,
        stressor_agg=CO2,
    )

    print("\n--- Resultats (Green New Deal) ---")
    print("Delta empreinte CO2 par GICS (production-based, kg) :")
    print(results_gnd["delta"].sort_values().to_string())
    print(f"\nDelta total : {results_gnd['delta'].sum():.3e} kg CO2")
    print(f"Flux redirige : {results_gnd['info']['total_redirected_M€']:.1f} M€")

    fig_gnd = plot_shock_impact(
        results_gnd,
        stressor_label="CO2 total",
        unit="kg CO2",
        save_path="Figs/shock_green_deal.html",
    )
    print("Graphique sauvegarde : Figs/shock_green_deal.html")

    # --- Choc COMBINE Z + Y : GICS shock + Green Deal ---
    # Le Z-choc change la technologie (Industrials achete moins de Materials, plus d'Energy)
    # Le Y-choc change la demande finale (Green Deal)
    # x' = (I-A')^-1 . y' est la seule solution coherente avec les deux chocs simultanement
    print("\n" + "="*60)
    print("Choc COMBINE : GICS Z-choc (Industrials) + Green Deal Y-choc")
    print("="*60)

    # Prepare Z_shocked via GICS shock
    concordance = load_concordance()
    Z_shocked_comb, z_info_comb = apply_gics_shock(
        Z,
        origin_gics="Industrials",
        decrease_gics="Materials",
        increase_gics="Energy",
        shock_pct=0.01,
        scope="intra",
        distribute="proportional",
        concordance=concordance,
    )
    print(f"  Z-choc : {z_info_comb['total_redirected_M€']:.1f} M€ redirigés "
          f"(Industrials: Materials -> Energy, 1%)")

    # Prepare Y_shocked via Green Deal
    FOSSIL_DECREASE = [
        "Electricity by coal",
        "Electricity by gas",
        "Electricity by petroleum and other oil derivatives",
    ]
    RENEWABLE_INCREASE = [
        "Electricity by wind",
        "Electricity by solar photovoltaic",
        "Electricity by solar thermal",
        "Electricity by hydro",
        "Electricity by Geothermal",
        "Electricity by biomass and waste",
        "Electricity by tide, wave, ocean",
        "Construction work (45)",
    ]
    Y_shocked_comb, y_info_comb = apply_Y_shock_multi(
        Y, FOSSIL_DECREASE, RENEWABLE_INCREASE,
        shock_pct=0.10,
        demand_categories=[
            "Final consumption expenditure by government",
            "Gross fixed capital formation",
        ],
        distribute="proportional",
    )
    print(f"  Y-choc : {y_info_comb['total_redirected_M€']:.1f} M€ redirigés "
          f"(Green Deal 10% GOV+GFCF fossile -> renouvelable)")

    results_comb = run_combined_shock(
        Z=Z, x=x, Y=Y, S=S,
        Z_shocked=Z_shocked_comb,
        Y_shocked=Y_shocked_comb,
        z_info=z_info_comb,
        y_info=y_info_comb,
        stressor_agg=CO2,
        concordance=concordance,
    )

    print("\n--- Resultats (choc combine) ---")
    print("Delta empreinte CO2 par GICS (production-based, kg) :")
    print(results_comb["delta"].sort_values().to_string())
    print(f"\nDelta total choc combine  : {results_comb['delta'].sum():.3e} kg CO2")
    print(f"Delta total GICS seul     : {results_gics['delta'].sum():.3e} kg CO2")
    print(f"Delta total Green Deal    : {results_gnd['delta'].sum():.3e} kg CO2")
    print(f"Somme des chocs separes   : "
          f"{results_gics['delta'].sum() + results_gnd['delta'].sum():.3e} kg CO2")
    print("(Non-additivite : le choc combine != somme des chocs separes)")

    fig_comb = plot_shock_impact(
        results_comb,
        stressor_label="CO2 total",
        unit="kg CO2",
        save_path="Figs/shock_combined.html",
    )
    print("Graphique sauvegarde : Figs/shock_combined.html")

    # --- Choc REDIRECTION POLLUTION : top 10 sales de Industrials -> 10 propres ---
    print("\n" + "="*60)
    print("Choc REDIRECTION POLLUTION : GICS Industrials, 50% top 10 sales -> 10 propres")
    print("="*60)

    results_red = run_pollution_redirect_comparison(
        Z=Z, x=x, Y=Y, S=S, F_Y=F_Y_filt,
        gics_buyer="Materials",
        stressor="CO2 - combustion - air",
        shock_pct=1.0,
        n_top=10,
        n_alternatives=10,
        min_supply_M= 1.0,
        stressor_agg=CO2,
    )

    print("\n--- Resultats (redirection pollution) ---")
    print("Delta empreinte CO2 par GICS (production-based, Mt) :")
    print((results_red["delta"] / 1e9).round(2).sort_values().to_string())
    print(f"\nDelta total : {results_red['delta'].sum()/1e9:+.2f} Mt CO2")

    fig_red = plot_shock_impact(
        results_red,
        stressor_label="CO2 total",
        unit="kg CO2",
        save_path="Figs/shock_pollution_redirect.html",
    )
    print("Graphique sauvegarde : Figs/shock_pollution_redirect.html")
