"""
Évolution temporelle des matrices techniques A (ou Z) EXIOBASE 1995-2024.

Par défaut, l'analyse porte sur A = Z·x̂⁻¹ (coefficients techniques) plutôt
que Z directement, car A élimine l'effet de volume (croissance économique)
et isole le changement structurel/technologique pur.

Métriques calculées entre M_t et M_{t+1} (M = A ou Z) :
    - L_inf  : max |ΔM_ij|              — pire variation locale
    - L_2    : sqrt(Σ ΔM_ij²)           — norme de Frobenius
    - L_1    : Σ |ΔM_ij|                — variation totale absolue
    - versions relatives (/ ||M_t||)    — sans dimension
    - rel_sum_change                    — pour Z : effet volume pur,
                                          pour A : variation de la part
                                          totale des intermédiaires

Deux scopes :
    - global : toute la matrice (49 régions)
    - EU27   : sous-matrice intra-Europe

pxp ou ixi au choix, A ou Z au choix.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from io_pipeline import _open_inner, _safe_inv

EU27 = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
]


# ---------------------------------------------------------------------------
# 1. Chargement bas niveau (numpy seulement)
# ---------------------------------------------------------------------------

def load_Z_values(year: int, table_type: str = "pxp") -> tuple[np.ndarray, pd.MultiIndex]:
    """
    Charge Z(year, table_type) en numpy.

    Returns
    -------
    Z_arr : np.ndarray (n×n)
    index : pd.MultiIndex (region, sector)  -- même structure pour rows et cols
    """
    inner = _open_inner(year, table_type)
    with inner.open("Z.txt") as fh:
        Z = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=[0, 1])
    return Z.values.astype(np.float64), Z.index


def load_matrix_values(
    year: int,
    table_type: str = "pxp",
    matrix: str = "A",
    min_x: float = 1.0,
) -> tuple[np.ndarray, pd.MultiIndex]:
    """
    Charge la matrice désirée (A ou Z) pour une année.

    Parameters
    ----------
    matrix : 'A' (default) ou 'Z'
        'Z' : flux intermédiaires bruts (M€)
        'A' = Z · x̂⁻¹ : coefficients techniques (sans dimension)
    min_x : float
        Seuil d'output (M€). Pour A : les colonnes avec x_j < min_x sont
        mises à 0 (sinon A_ij = Z_ij / x_j devient instable / artefacts
        ~1.0 quand x_j est proche de zéro). Sans effet sur Z.
        Sur EXIOBASE pxp, ~2700 colonnes ont x=0 (secteurs inexistants
        dans certaines régions).

    Returns
    -------
    M_arr : np.ndarray (n×n)
    index : pd.MultiIndex (region, sector)
    """
    inner = _open_inner(year, table_type)

    with inner.open("Z.txt") as fh:
        Z = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=[0, 1])
    Z_arr = Z.values.astype(np.float64)

    if matrix.upper() == "Z":
        return Z_arr, Z.index

    if matrix.upper() != "A":
        raise ValueError(f"matrix doit être 'A' ou 'Z', reçu : {matrix!r}")

    # A = Z · x̂⁻¹  → divise chaque colonne par x (sauf si x < min_x)
    with inner.open("x.txt") as fh:
        x = pd.read_csv(fh, sep="\t", header=0, index_col=[0, 1]).squeeze("columns")
    x_vals = x.values.astype(np.float64)

    valid = x_vals >= min_x                              # colonnes à conserver
    x_inv = np.zeros_like(x_vals)
    x_inv[valid] = 1.0 / x_vals[valid]

    A_arr = Z_arr * x_inv[np.newaxis, :]                 # broadcast colonnes
    return A_arr, Z.index


def _build_region_mask(index: pd.MultiIndex, region_subset: list[str]) -> np.ndarray:
    """Boolean mask (n,) : True si la région ∈ region_subset."""
    regions = index.get_level_values("region")
    return np.isin(regions, region_subset)


# ---------------------------------------------------------------------------
# 2. Calcul des métriques d'évolution
# ---------------------------------------------------------------------------

def _compute_transition(
    year_from: int,
    year_to: int,
    M_prev: np.ndarray,
    M_curr: np.ndarray,
    index: pd.MultiIndex,
    n_top: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """
    Calcule toutes les normes sur la transition M_prev -> M_curr
    et identifie les n_top cellules avec la plus grande variation |ΔM_ij|.

    Returns
    -------
    metrics : dict avec toutes les normes scalaires
    top_var : pd.DataFrame avec les n_top variations triées par |ΔM| décroissant
              colonnes : year_from, year_to, supplier_region, supplier_sector,
                         buyer_region, buyer_sector, value_from, value_to,
                         delta, abs_delta
    """
    diff     = M_curr - M_prev
    abs_diff = np.abs(diff)

    L_inf = float(abs_diff.max())
    L_2   = float(np.sqrt((diff ** 2).sum()))
    L_1   = float(abs_diff.sum())

    norm_inf_prev = float(np.abs(M_prev).max())
    norm_2_prev   = float(np.sqrt((M_prev ** 2).sum()))
    norm_1_prev   = float(np.abs(M_prev).sum())
    norm_2_curr   = float(np.sqrt((M_curr ** 2).sum()))
    norm_1_curr   = float(np.abs(M_curr).sum())

    # Cosinus de Frobenius : <A, B>_F / (||A||_F · ||B||_F)
    inner     = float((M_prev * M_curr).sum())
    denom     = norm_2_prev * norm_2_curr
    cos_theta = inner / denom if denom > 0 else 0.0

    # L_1 symetrique normalisee dans [0, 1] (= Bray-Curtis matricielle) :
    #   d = ||ΔM||_1 / (||M_t||_1 + ||M_{t+1}||_1)
    # Pour M >= 0, inegalite triangulaire => d in [0, 1] strict.
    L1_denom     = norm_1_prev + norm_1_curr
    rel_L_1_symm = L_1 / L1_denom if L1_denom > 0 else 0.0

    sum_prev = float(M_prev.sum())
    sum_curr = float(M_curr.sum())

    metrics = {
        "year_from":      year_from,
        "year_to":        year_to,
        "L_inf":          L_inf,
        "L_2":            L_2,
        "L_1":            L_1,
        "rel_L_inf":      L_inf / norm_inf_prev if norm_inf_prev > 0 else 0.0,
        "rel_L_2":        L_2   / norm_2_prev   if norm_2_prev   > 0 else 0.0,
        "rel_L_1":        L_1   / norm_1_prev   if norm_1_prev   > 0 else 0.0,
        "rel_L_1_symm":   rel_L_1_symm,
        "cos_theta":      cos_theta,
        "sum_from":       sum_prev,
        "sum_to":         sum_curr,
        "rel_sum_change": (sum_curr - sum_prev) / sum_prev if sum_prev > 0 else 0.0,
        "n_cells":        M_prev.size,
    }

    # --- Top n cells by |ΔM_ij| ---
    if n_top <= 0:
        return metrics, pd.DataFrame()

    flat   = abs_diff.ravel()
    n_eff  = min(n_top, flat.size)
    # argpartition O(N) puis tri du partial
    part   = np.argpartition(flat, -n_eff)[-n_eff:]
    sorted_idx = part[np.argsort(-flat[part])]
    rows, cols = np.unravel_index(sorted_idx, diff.shape)

    records = []
    for r, c in zip(rows, cols):
        sup_reg, sup_sec = index[r]
        buy_reg, buy_sec = index[c]
        records.append({
            "year_from":       year_from,
            "year_to":         year_to,
            "supplier_region": sup_reg,
            "supplier_sector": sup_sec,
            "buyer_region":    buy_reg,
            "buyer_sector":    buy_sec,
            "value_from":      float(M_prev[r, c]),
            "value_to":        float(M_curr[r, c]),
            "delta":           float(diff[r, c]),
            "abs_delta":       float(abs_diff[r, c]),
        })
    return metrics, pd.DataFrame(records)


def compute_evolution_metrics_multi(
    years: list[int],
    table_type: str = "pxp",
    matrix: str = "A",
    scopes: dict[str, list[str] | None] | None = None,
    min_x: float = 1.0,
    n_top_variations: int = 10,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Calcule les normes ||M_{t+1} - M_t|| et les top-N variations cellulaires
    pour plusieurs scopes en un seul passage sur les années.

    Parameters
    ----------
    years : liste d'années consécutives
    table_type : 'pxp' | 'ixi'
    matrix : 'A' (par défaut) ou 'Z'
    scopes : dict {nom_scope: liste_de_régions_ou_None}
    min_x : float — seuil colonne pour A
    n_top_variations : int — nb de top variations cellulaires par transition
                              (0 = désactivé)

    Returns
    -------
    (metrics_per_scope, top_variations_per_scope) :
        deux dicts {nom_scope: DataFrame}.
    """
    if scopes is None:
        scopes = {"global": None, "EU27": EU27}

    results       = {name: [] for name in scopes}
    top_var_list  = {name: [] for name in scopes}
    M_prev        = {name: None for name in scopes}
    masks         = {name: None for name in scopes}
    sub_indices   = {name: None for name in scopes}

    unit_label = "(adim)" if matrix.upper() == "A" else "M€"

    for i, year in enumerate(years):
        print(f"  Loading {matrix}({year}, {table_type})...", end=" ", flush=True)
        M_full, idx = load_matrix_values(year, table_type, matrix=matrix, min_x=min_x)
        print(f"shape={M_full.shape}, sum={M_full.sum():.3e} {unit_label}")

        for name, subset in scopes.items():
            if subset is None:
                M_arr     = M_full
                sub_index = idx
            else:
                if masks[name] is None:
                    masks[name]       = _build_region_mask(idx, subset)
                    sub_indices[name] = idx[masks[name]]
                m         = masks[name]
                M_arr     = M_full[m][:, m]
                sub_index = sub_indices[name]

            if M_prev[name] is not None:
                metrics, top_var = _compute_transition(
                    years[i - 1], year,
                    M_prev[name], M_arr,
                    sub_index,
                    n_top=n_top_variations,
                )
                results[name].append(metrics)
                if not top_var.empty:
                    top_var_list[name].append(top_var)

            # Copie indépendante pour que la sous-matrice survive au del()
            M_prev[name] = M_arr if subset is None else M_arr.copy()

        del M_full

    metrics_dfs  = {name: pd.DataFrame(rows) for name, rows in results.items()}
    top_var_dfs  = {
        name: (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame())
        for name, dfs in top_var_list.items()
    }
    return metrics_dfs, top_var_dfs


def compute_evolution_metrics(
    years: list[int],
    table_type: str = "pxp",
    matrix: str = "A",
    region_subset: list[str] | None = None,
) -> pd.DataFrame:
    """
    Version single-scope (compat). Préférer compute_evolution_metrics_multi
    pour calculer plusieurs scopes en un seul passage.

    Returns
    -------
    pd.DataFrame des métriques scalaires uniquement (pas de top variations).
    """
    scope_name = "subset" if region_subset is not None else "global"
    metrics, _ = compute_evolution_metrics_multi(
        years, table_type, matrix=matrix,
        scopes={scope_name: region_subset},
        n_top_variations=0,
    )
    return metrics[scope_name]


# ---------------------------------------------------------------------------
# 3. Visualisation : 2×2 (global vs EU27, absolu vs relatif)
# ---------------------------------------------------------------------------

def plot_Z_evolution(
    df_global: pd.DataFrame,
    df_eu: pd.DataFrame,
    table_type: str = "pxp",
    matrix: str = "A",
    save_path: str | None = None,
) -> go.Figure:
    """
    Grille 2×2 :
        (1,1) M global - normes absolues (échelle log, M = A ou Z)
        (1,2) M EU27   - normes absolues
        (2,1) M global - normes relatives (%)
        (2,2) M EU27   - normes relatives
    """
    M = matrix.upper()
    unit_abs = "adim" if M == "A" else "M€"
    sum_label = "Δ Σ coef. interméd." if M == "A" else "Δ volume total"

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            f"{M} global (49 régions) - normes absolues",
            f"{M} EU27 (27 régions) - normes absolues",
            f"{M} global - normes relatives",
            f"{M} EU27 - normes relatives",
            f"{M} global - cosinus de Frobenius (similarité)",
            f"{M} EU27 - cosinus de Frobenius (similarité)",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        row_heights=[0.34, 0.34, 0.32],
    )

    colors = {
        "L_inf":         "#c0392b",
        "L_2":           "#2980b9",
        "rel_L_inf":     "#c0392b",
        "rel_L_2":       "#2980b9",
        "rel_sum_change":"#27ae60",
        "cos_theta":     "#8e44ad",
    }

    # --- Ligne 1 : normes absolues, échelle log ---
    for col, df, scope in [(1, df_global, "global"), (2, df_eu, "EU27")]:
        fig.add_trace(go.Scatter(
            x=df["year_to"], y=df["L_inf"],
            mode="lines+markers", name=f"L∞  (max |Δ{M}_ij|)",
            line=dict(color=colors["L_inf"], width=2),
            marker=dict(size=6),
            legendgroup="L_inf",
            showlegend=(col == 1),
            hovertemplate=f"%{{x}}<br>L∞ = %{{y:.2e}} {unit_abs}<extra></extra>",
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=df["year_to"], y=df["L_2"],
            mode="lines+markers", name="L₂  (Frobenius)",
            line=dict(color=colors["L_2"], width=2),
            marker=dict(size=6),
            legendgroup="L_2",
            showlegend=(col == 1),
            hovertemplate=f"%{{x}}<br>L₂ = %{{y:.2e}} {unit_abs}<extra></extra>",
        ), row=1, col=col)

    # --- Ligne 2 : normes relatives (%) ---
    for col, df, scope in [(1, df_global, "global"), (2, df_eu, "EU27")]:
        fig.add_trace(go.Scatter(
            x=df["year_to"], y=df["rel_L_inf"] * 100,
            mode="lines+markers", name="L∞ relatif",
            line=dict(color=colors["rel_L_inf"], width=2, dash="solid"),
            marker=dict(size=6),
            legendgroup="rel_L_inf",
            showlegend=(col == 1),
            hovertemplate=f"%{{x}}<br>L∞/||{M}|| = %{{y:.2f}}%<extra></extra>",
        ), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=df["year_to"], y=df["rel_L_2"] * 100,
            mode="lines+markers", name="L₂ relatif",
            line=dict(color=colors["rel_L_2"], width=2, dash="solid"),
            marker=dict(size=6),
            legendgroup="rel_L_2",
            showlegend=(col == 1),
            hovertemplate=f"%{{x}}<br>L₂/||{M}|| = %{{y:.2f}}%<extra></extra>",
        ), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=df["year_to"], y=df["rel_sum_change"] * 100,
            mode="lines+markers", name=f"{sum_label}",
            line=dict(color=colors["rel_sum_change"], width=2, dash="dot"),
            marker=dict(size=6),
            legendgroup="rel_sum",
            showlegend=(col == 1),
            hovertemplate=f"%{{x}}<br>ΔΣ{M} = %{{y:+.2f}}%<extra></extra>",
        ), row=2, col=col)

        # zéro
        fig.add_hline(y=0, line_color="#95a5a6", line_width=0.8,
                       line_dash="dot", row=2, col=col)

    # --- Ligne 3 : cosinus de Frobenius (similarité dans [-1, 1]) ---
    for col, df, scope in [(1, df_global, "global"), (2, df_eu, "EU27")]:
        fig.add_trace(go.Scatter(
            x=df["year_to"], y=df["cos_theta"],
            mode="lines+markers", name="cos θ  ⟨A,B⟩/(||A||·||B||)",
            line=dict(color=colors["cos_theta"], width=2),
            marker=dict(size=7),
            legendgroup="cos_theta",
            showlegend=(col == 1),
            hovertemplate="%{x}<br>cos θ = %{y:.4f}<extra></extra>",
        ), row=3, col=col)

        # Référence 1.0 (identité parfaite)
        fig.add_hline(y=1.0, line_color="#27ae60", line_width=0.8,
                       line_dash="dash", row=3, col=col)
        fig.add_hline(y=0.0, line_color="#95a5a6", line_width=0.8,
                       line_dash="dot", row=3, col=col)

    # Marque la crise 2008-2009 et le Covid 2020-2021 sur les 6 sous-graphes
    for r in [1, 2, 3]:
        for c in [1, 2]:
            fig.add_vrect(
                x0=2008, x1=2009,
                fillcolor="#bdc3c7", opacity=0.18, line_width=0,
                row=r, col=c,
            )
            fig.add_vrect(
                x0=2020, x1=2021,
                fillcolor="#e74c3c", opacity=0.12, line_width=0,
                row=r, col=c,
            )

    years_range = (int(df_global["year_to"].iloc[0]) - 1,
                    int(df_global["year_to"].iloc[-1]))
    tick0 = int(df_global["year_to"].iloc[0])

    matrix_label = "des coefficients techniques A = Z·x̂⁻¹" if M == "A" else "des flux Z"
    fig.update_layout(
        title=dict(
            text=f"Évolution annuelle {matrix_label} ({table_type}) — "
                 f"{years_range[0]}→{years_range[1]}",
            font=dict(size=14), x=0.5,
        ),
        height=1050, width=1280,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#bdc3c7", borderwidth=0.5,
            font=dict(size=10),
        ),
        margin=dict(l=70, r=200, t=80, b=60),
    )

    # Y-axes : log absolu, % relatif, [-1, 1] cosinus
    for c in [1, 2]:
        fig.update_yaxes(type="log",
                         title_text=f"{unit_abs} (log)",
                         row=1, col=c, gridcolor="#ecf0f1")
        fig.update_yaxes(title_text="%", row=2, col=c, gridcolor="#ecf0f1")
        fig.update_yaxes(title_text="cos θ", row=3, col=c, gridcolor="#ecf0f1",
                         range=[-1.05, 1.05], zeroline=True, zerolinecolor="#95a5a6")
        fig.update_xaxes(title_text="Année", row=3, col=c, gridcolor="#ecf0f1",
                         tickmode="linear", tick0=tick0, dtick=2)
        for r in [1, 2]:
            fig.update_xaxes(row=r, col=c, gridcolor="#ecf0f1",
                              tickmode="linear", tick0=tick0, dtick=2)

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3b. Plot dédié : norme L_1 symétrique normalisée dans [0, 1]
# ---------------------------------------------------------------------------

def plot_L1_symmetric_evolution(
    df_global: pd.DataFrame,
    df_eu: pd.DataFrame,
    table_type: str = "pxp",
    matrix: str = "A",
    save_path: str | None = None,
) -> go.Figure:
    """
    Graphique séparé : norme L_1 symétrique normalisée dans [0, 1].

        d = ||ΔM||_1 / (||M_t||_1 + ||M_{t+1}||_1)

    Propriétés (M ≥ 0) :
        d = 0  : matrices identiques
        d = 1  : matrices "complémentaires" (support disjoint)
        Borne stricte [0, 1] grâce à l'inégalité triangulaire matricielle.

    Comparé à rel_L_2 / cos θ :
        - L_1 pondère linéairement (vs. quadratique pour L_2) — moins biaisée
          par les quelques outliers
        - Bornée dans [0, 1] (vs. rel_L_2 pouvant dépasser 1)
        - Sensible aux changements diffus (beaucoup de petites variations)
    """
    M = matrix.upper()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_global["year_to"], y=df_global["rel_L_1_symm"],
        mode="lines+markers", name="Global (49 régions)",
        line=dict(color="#2980b9", width=2.5),
        marker=dict(size=8),
        hovertemplate="%{x}<br>d = %{y:.4f}<extra>Global</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_eu["year_to"], y=df_eu["rel_L_1_symm"],
        mode="lines+markers", name="EU27",
        line=dict(color="#c0392b", width=2.5, dash="solid"),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="%{x}<br>d = %{y:.4f}<extra>EU27</extra>",
    ))

    # Bandes 2008-2009 (crise financière) et 2020-2021 (Covid + énergie)
    fig.add_vrect(x0=2008, x1=2009, fillcolor="#bdc3c7", opacity=0.18,
                   line_width=0,
                   annotation_text="Crise 2008-09", annotation_position="top left",
                   annotation_font=dict(size=9, color="#7f8c8d"))
    fig.add_vrect(x0=2020, x1=2021, fillcolor="#e74c3c", opacity=0.13,
                   line_width=0,
                   annotation_text="Covid 2020-21", annotation_position="top right",
                   annotation_font=dict(size=9, color="#c0392b"))

    # Lignes de référence
    fig.add_hline(y=0, line_color="#27ae60", line_width=0.8, line_dash="dash",
                   annotation_text="identité parfaite",
                   annotation_position="bottom right",
                   annotation_font=dict(size=9, color="#27ae60"))
    fig.add_hline(y=1, line_color="#95a5a6", line_width=0.8, line_dash="dot",
                   annotation_text="changement total",
                   annotation_position="top right",
                   annotation_font=dict(size=9, color="#7f8c8d"))

    years_range = (int(df_global["year_to"].iloc[0]) - 1,
                    int(df_global["year_to"].iloc[-1]))
    tick0       = int(df_global["year_to"].iloc[0])

    matrix_label = "des coefficients techniques A" if M == "A" else "des flux Z"

    fig.update_layout(
        title=dict(
            text=(
                f"Norme L_1 symétrique normalisée de l'évolution {matrix_label} "
                f"({table_type})<br>"
                f"<sub>d = ||ΔM||_1 / (||M_t||_1 + ||M_{{t+1}}||_1) ∈ [0, 1]"
                f" — {years_range[0]}→{years_range[1]}</sub>"
            ),
            font=dict(size=13), x=0.5,
        ),
        xaxis=dict(
            title="Année", gridcolor="#ecf0f1",
            tickmode="linear", tick0=tick0, dtick=2,
        ),
        yaxis=dict(
            title="d (L_1 symétrique normalisée)",
            range=[0, 1.05], gridcolor="#ecf0f1",
            zeroline=True, zerolinecolor="#95a5a6",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99,
                     bgcolor="rgba(255,255,255,0.85)",
                     bordercolor="#bdc3c7", borderwidth=0.5),
        height=500, width=1100,
        margin=dict(l=70, r=40, t=80, b=55),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4. Pipeline complet
# ---------------------------------------------------------------------------

def run(
    years: list[int] | None = None,
    table_type: str = "pxp",
    matrix: str = "A",
    save_dir: str = "Figs",
    min_x: float = 1.0,
    n_top_variations: int = 10,
    print_top_for_last_year: bool = True,
) -> dict:
    """
    Pipeline complet : charge M(t) pour chaque année (M = A ou Z),
    calcule les normes d'évolution global + EU27, identifie les top-N
    variations cellulaires par transition, et génère le graphique.

    Parameters
    ----------
    years : list[int], optional       — par défaut 1995→2024.
    table_type : 'pxp' | 'ixi'
    matrix     : 'A' (par défaut) ou 'Z'
    save_dir   : dossier pour les CSV et HTML
    min_x      : seuil colonne pour A (M€)
    n_top_variations : nb de cellules à extraire par transition (0 = off)
    print_top_for_last_year : print en stdout du top pour la dernière année

    Returns
    -------
    dict avec 'df_global', 'df_eu', 'top_global', 'top_eu', 'fig'
    """
    if years is None:
        years = list(range(1995, 2025))

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    M = matrix.upper()
    print(f"=== Evolution {M} {table_type} : {years[0]} -> {years[-1]} ===\n")

    print("[1/2] Metriques + top variations (global + EU27)...")
    metrics, top_vars = compute_evolution_metrics_multi(
        years, table_type, matrix=M,
        scopes={"global": None, "EU27": EU27},
        min_x=min_x,
        n_top_variations=n_top_variations,
    )
    df_global = metrics["global"]
    df_eu     = metrics["EU27"]
    top_global = top_vars["global"]
    top_eu     = top_vars["EU27"]
    print()

    stem = f"{M}_evolution"
    csv_global    = save_dir_path / f"{stem}_global_{table_type}.csv"
    csv_eu        = save_dir_path / f"{stem}_EU27_{table_type}.csv"
    df_global.to_csv(csv_global, index=False)
    df_eu.to_csv(csv_eu, index=False)
    print(f"  CSV metrics global : {csv_global}")
    print(f"  CSV metrics EU27   : {csv_eu}")

    if n_top_variations > 0:
        csv_top_global = save_dir_path / f"{stem}_top{n_top_variations}_global_{table_type}.csv"
        csv_top_eu     = save_dir_path / f"{stem}_top{n_top_variations}_EU27_{table_type}.csv"
        top_global.to_csv(csv_top_global, index=False)
        top_eu.to_csv(csv_top_eu, index=False)
        print(f"  CSV top variations global : {csv_top_global}")
        print(f"  CSV top variations EU27   : {csv_top_eu}")

        if print_top_for_last_year and not top_global.empty:
            last = int(top_global["year_to"].max())
            recent = top_global[top_global["year_to"] == last]
            print(f"\n  Top {n_top_variations} variations |delta_{M}_ij| (global, "
                   f"{last-1}->{last}) :")
            for _, row in recent.iterrows():
                sup = f"{row.supplier_region}|{row.supplier_sector[:32]}"
                buy = f"{row.buyer_region}|{row.buyer_sector[:32]}"
                print(f"    {sup:42s} -> {buy:42s}"
                      f"  {row.value_from:>10.5f} -> {row.value_to:>10.5f}"
                      f"  (delta={row.delta:+.5f})")

    print("\n[2/2] Graphiques...")
    html_path = save_dir_path / f"{stem}_{table_type}.html"
    fig = plot_Z_evolution(df_global, df_eu, table_type, matrix=M,
                            save_path=str(html_path))

    html_L1 = save_dir_path / f"{stem}_L1_symm_{table_type}.html"
    fig_L1  = plot_L1_symmetric_evolution(df_global, df_eu, table_type, matrix=M,
                                            save_path=str(html_L1))

    return {
        "df_global":  df_global,
        "df_eu":      df_eu,
        "top_global": top_global,
        "top_eu":     top_eu,
        "fig":        fig,
        "fig_L1":     fig_L1,
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Évolution annuelle des matrices A ou Z EXIOBASE"
    )
    parser.add_argument("--matrix", choices=["A", "Z"], default="A",
                         help="Matrice à analyser : A (coefficients) ou Z (flux).")
    parser.add_argument("--table", choices=["pxp", "ixi"], default="pxp",
                         help="Type de table : pxp (product) ou ixi (industry).")
    parser.add_argument("--start", type=int, default=1995, help="Année de début.")
    parser.add_argument("--end",   type=int, default=2024, help="Année de fin (incluse).")
    parser.add_argument("--save-dir", default="Figs", help="Dossier de sortie.")
    parser.add_argument("--min-x", type=float, default=1.0,
                         help="Seuil d'output minimum (M€) pour A. Colonnes "
                              "avec x_j < min_x mises à 0. Sans effet sur Z.")
    parser.add_argument("--n-top", type=int, default=10,
                         help="Nb de top variations cellulaires par transition "
                              "(0 = desactive).")
    args = parser.parse_args()

    years = list(range(args.start, args.end + 1))
    result = run(years=years, table_type=args.table, matrix=args.matrix,
                 save_dir=args.save_dir, min_x=args.min_x,
                 n_top_variations=args.n_top)

    print("\n=== Synthese global ===")
    print(result["df_global"][["year_to", "rel_L_2", "rel_L_inf", "rel_sum_change"]]
          .round(4).to_string(index=False))

    print("\n=== Synthese EU27 ===")
    print(result["df_eu"][["year_to", "rel_L_2", "rel_L_inf", "rel_sum_change"]]
          .round(4).to_string(index=False))
