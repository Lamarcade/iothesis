"""
Visualisations IO :
    - plot_multipliers  : lollipop chart des multiplicateurs par secteur
    - plot_intensity_output : scatter intensité stresseur × output sectoriel
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

from convert_sectors import load_concordance, _build_lookup, _map_sector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _add_gics_column(df: pd.DataFrame, concordance: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Ajoute une colonne 'gics_name' à un DataFrame dont l'index est
    un MultiIndex (region, sector).
    """
    if concordance is None:
        concordance = load_concordance()
    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    sectors = df.index.get_level_values("sector")
    df = df.copy()
    df["gics_name"] = [_map_sector(s, lookup, norm_lookup) for s in sectors]
    return df


# ---------------------------------------------------------------------------
# Graphique 4 — Lollipop des multiplicateurs
# ---------------------------------------------------------------------------

def plot_multipliers(
    m: pd.Series,
    stressor: str = "",
    top_n: int = 30,
    min_output: float = 10.0,
    max_multiplier: float = 1e9,
    unit: str = "kg / M€",
    save_path: str | None = None,
    concordance: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Lollipop chart des top-N secteurs par multiplicateur de Leontief.

    m[i] = émission totale (directe + indirecte via la chaîne) par unité
    de demande finale adressée au secteur i.

    Parameters
    ----------
    m : pd.Series
        Sortie de compute_multipliers(), indexée (region, sector).
    stressor : str
        Nom du stresseur (pour le titre).
    top_n : int
        Nombre de secteurs à afficher.
    min_output : float
        Filtre les secteurs dont le output x est inférieur à ce seuil (M€).
        Évite les artefacts numériques liés aux secteurs quasi-vides.
    max_multiplier : float
        Filtre les outliers numériques (instabilité quand x ≈ 0).
    unit : str
        Unité affiché sur l'axe X.
    save_path : str, optional
        Chemin pour sauvegarder en HTML interactif.
    concordance : pd.DataFrame, optional

    Returns
    -------
    plotly Figure
    """
    # --- Filtrage ---
    m_df = m.to_frame(name="multiplier")
    m_df = _add_gics_column(m_df, concordance)
    m_df = m_df[m_df["multiplier"].between(0, max_multiplier)]

    top = (
        m_df.sort_values("multiplier", ascending=False)
            .head(top_n)
            .sort_values("multiplier", ascending=True)   # ordre croissant pour l'axe Y
    )

    regions  = top.index.get_level_values("region")
    sectors  = top.index.get_level_values("sector")
    labels   = [f"{r} — {s}" for r, s in zip(regions, sectors)]
    gics     = top["gics_name"].tolist()
    values   = top["multiplier"].tolist()
    colors   = [GICS_COLORS.get(g, "#bdc3c7") for g in gics]

    fig = go.Figure()

    # Tiges (lignes horizontales)
    for i, (label, val, col) in enumerate(zip(labels, values, colors)):
        fig.add_shape(
            type="line",
            x0=0, x1=val, y0=i, y1=i,
            line=dict(color=col, width=1.5),
        )

    # Points
    fig.add_trace(go.Scatter(
        x=values,
        y=list(range(len(labels))),
        mode="markers",
        marker=dict(
            color=colors,
            size=10,
            line=dict(width=0.8, color="white"),
        ),
        text=[f"<b>{l}</b><br>GICS: {g}<br>Multiplicateur: {v:,.0f} {unit}"
              for l, g, v in zip(labels, gics, values)],
        hovertemplate="%{text}<extra></extra>",
    ))

    # Légende GICS manuelle
    shown = set()
    for g, col in GICS_COLORS.items():
        if g in set(gics) and g not in shown:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=col, size=9),
                name=g,
                showlegend=True,
            ))
            shown.add(g)

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} multiplicateurs — {stressor}",
            font=dict(size=15),
        ),
        xaxis=dict(
            title=f"Multiplicateur ({unit})",
            tickformat=".2e",
            gridcolor="#ecf0f1",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            tickfont=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(500, top_n * 22),
        width=950,
        legend=dict(title="GICS", x=1.01, y=1),
        margin=dict(l=320, r=160, t=60, b=50),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Graphique 6 — Scatter intensité × output
# ---------------------------------------------------------------------------

def plot_intensity_output(
    S: pd.DataFrame,
    x: pd.Series,
    stressor: str,
    min_output: float = 10.0,
    min_intensity: float = 1e-6,
    unit_stressor: str = "kg",
    save_path: str | None = None,
    concordance: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Scatter plot : intensité stresseur (S[k,i]) × output sectoriel (x[i]).

    Axes log-log pour gérer les ordres de grandeur.
    Taille des bulles ∝ émissions directes totales = S[k,i] · x[i].
    Couleur = GICS.

    Lecture :
        Haut-droite  → secteurs larges ET intensifs  (priorité politique)
        Bas-droite   → secteurs larges mais propres  (champions d'efficacité)
        Haut-gauche  → secteurs petits mais sales    (niches à surveiller)
        Bas-gauche   → secteurs marginaux

    Parameters
    ----------
    S : pd.DataFrame (e×n)
        Intensités stresseurs, sortie de compute_S().
    x : pd.Series (n,)
        Output total par secteur, issu de load_io_system().
    stressor : str
        Ligne de S à utiliser (ex: 'CO2 - combustion - air').
    min_output : float
        Seuil minimum de x (M€) — filtre secteurs quasi-vides.
    min_intensity : float
        Seuil minimum de S[k,i] — filtre secteurs sans émissions.
    unit_stressor : str
        Unité des émissions (pour les labels).
    save_path : str, optional
        Chemin HTML pour sauvegarder.
    concordance : pd.DataFrame, optional

    Returns
    -------
    plotly Figure
    """
    # --- Construction du DataFrame de travail ---
    s_row = S.loc[stressor].reindex(x.index, fill_value=0.0)

    df = pd.DataFrame({
        "output":    x.values,
        "intensity": s_row.values,
    }, index=x.index)

    df = _add_gics_column(df, concordance)

    # Filtre : x > seuil et S > seuil (évite log(0) et outliers numériques)
    df = df[(df["output"] > min_output) & (df["intensity"] > min_intensity)]

    df["total_emission"] = df["output"] * df["intensity"]
    df["region"] = df.index.get_level_values("region")
    df["sector"] = df.index.get_level_values("sector")
    df["label"]  = df["region"] + " — " + df["sector"]

    # Taille des bulles sur une échelle raisonnable
    size_ref = df["total_emission"].quantile(0.95) / 30
    sizes    = np.clip(df["total_emission"] / size_ref, 3, 60)

    colors = [GICS_COLORS.get(g, "#bdc3c7") for g in df["gics_name"]]

    # Hover text
    hover = [
        f"<b>{row.label}</b><br>"
        f"GICS: {row.gics_name}<br>"
        f"Output: {row.output:,.0f} M€<br>"
        f"Intensité: {row.intensity:.2e} {unit_stressor}/M€<br>"
        f"Émissions directes: {row.total_emission:.2e} {unit_stressor}"
        for row in df.itertuples()
    ]

    fig = go.Figure()

    # Un trace par GICS pour la légende
    for gics_name, col in GICS_COLORS.items():
        mask = df["gics_name"] == gics_name
        if mask.sum() == 0:
            continue
        sub = df[mask]
        fig.add_trace(go.Scatter(
            x=sub["output"],
            y=sub["intensity"],
            mode="markers",
            name=gics_name,
            marker=dict(
                color=col,
                size=sizes[mask].tolist(),
                opacity=0.72,
                line=dict(width=0.4, color="white"),
            ),
            text=[hover[i] for i in np.where(mask)[0]],
            hovertemplate="%{text}<extra></extra>",
        ))

    # Lignes médianes (repères visuels)
    x_med = df["output"].median()
    y_med = df["intensity"].median()

    fig.add_vline(x=x_med,   line_dash="dot", line_color="#95a5a6", line_width=1)
    fig.add_hline(y=y_med,   line_dash="dot", line_color="#95a5a6", line_width=1)

    # Annotations des quadrants
    x_range = [df["output"].min(), df["output"].max()]
    y_range = [df["intensity"].min(), df["intensity"].max()]

    annotations = [
        ("Larges & intenses<br><i>priorité politique</i>",   x_range[1], y_range[1], "right", "top"),
        ("Larges & efficaces",                                x_range[1], y_range[0], "right", "bottom"),
        ("Petits & intenses<br><i>niches à risque</i>",      x_range[0], y_range[1], "left",  "top"),
        ("Petits & propres",                                  x_range[0], y_range[0], "left",  "bottom"),
    ]
    for text, ax, ay, xanchor, yanchor in annotations:
        fig.add_annotation(
            x=np.log10(ax), y=np.log10(ay),
            text=text, showarrow=False,
            font=dict(size=9, color="#7f8c8d"),
            xanchor=xanchor, yanchor=yanchor,
            xref="x", yref="y",
        )

    fig.update_layout(
        title=dict(
            text=f"Intensité × Output — {stressor}",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="Output total (M€)",
            type="log",
            gridcolor="#ecf0f1",
            showgrid=True,
        ),
        yaxis=dict(
            title=f"Intensité ({unit_stressor} / M€)",
            type="log",
            gridcolor="#ecf0f1",
            showgrid=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=620,
        width=1000,
        legend=dict(title="GICS", x=1.01, y=1),
        margin=dict(l=70, r=180, t=60, b=60),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Graphique 7 — Sous-secteurs EXIOBASE composant un GICS donné
# ---------------------------------------------------------------------------

def plot_gics_subsectors(
    values: pd.Series,
    gics_name: str,
    top_n: int = 15,
    aggregate_regions: bool = True,
    metric_label: str = "Output total",
    unit: str = "M€",
    save_path: str | None = None,
    concordance: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Lollipop des sous-secteurs EXIOBASE composant un secteur GICS,
    triés par ordre d'importance (valeur fournie).

    Utilisation typique :
        plot_gics_subsectors(x, "Energy")                       # par output
        plot_gics_subsectors(footprint_sector, "Materials")     # par empreinte

    Parameters
    ----------
    values : pd.Series
        Série indexée par MultiIndex (region, sector).
        Valeur = métrique d'importance (output, émissions, footprint, ...).
    gics_name : str
        Nom du groupe GICS à décomposer (ex: "Energy", "Materials").
    top_n : int
        Nombre de sous-secteurs à afficher.
    aggregate_regions : bool
        True (défaut) : somme par secteur sur toutes les régions
                        (vue mondiale du GICS).
        False         : garde la granularité (region, sector)
                        (top combinaisons region × secteur).
    metric_label : str
        Étiquette de la métrique (axe X, hover).
    unit : str
        Unité de la métrique.
    save_path : str, optional
        Chemin HTML pour sauvegarde.
    concordance : pd.DataFrame, optional

    Returns
    -------
    plotly Figure
    """
    if concordance is None:
        concordance = load_concordance()

    # --- Identifier les secteurs EXIOBASE appartenant au GICS ---
    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    all_sectors = values.index.get_level_values("sector").unique()
    sectors_in_gics = [s for s in all_sectors
                       if _map_sector(s, lookup, norm_lookup) == gics_name]

    if not sectors_in_gics:
        raise ValueError(f"Aucun secteur EXIOBASE trouvé pour le GICS '{gics_name}'.")

    mask = values.index.get_level_values("sector").isin(sectors_in_gics)
    sub  = values[mask]

    # --- Agrégation / sélection top-N ---
    top_regions_per_sector: dict[str, pd.Series] = {}
    if aggregate_regions:
        sub_agg = sub.groupby(level="sector").sum().sort_values(ascending=False)
        top     = sub_agg.head(top_n).sort_values(ascending=True)   # bas->haut sur l'axe Y
        labels  = top.index.tolist()
        for s in labels:
            sub_s = sub[sub.index.get_level_values("sector") == s]
            top_regions_per_sector[s] = (
                sub_s.groupby(level="region").sum()
                     .sort_values(ascending=False).head(3)
            )
    else:
        sub_sorted = sub.sort_values(ascending=False).head(top_n).sort_values(ascending=True)
        labels = [f"{r} — {s}" for r, s in sub_sorted.index]
        top    = sub_sorted

    vals  = top.values
    color = GICS_COLORS.get(gics_name, "#bdc3c7")

    fig = go.Figure()

    # Tiges horizontales
    for i, v in enumerate(vals):
        fig.add_shape(
            type="line",
            x0=0, x1=v, y0=i, y1=i,
            line=dict(color=color, width=1.5),
        )

    # Hover personnalisé
    if aggregate_regions:
        hover = []
        for lab, v in zip(labels, vals):
            tr = top_regions_per_sector[lab]
            tr_str = "<br>".join(f"  {r} : {val:,.2e}" for r, val in tr.items())
            hover.append(
                f"<b>{lab}</b><br>"
                f"GICS : {gics_name}<br>"
                f"{metric_label} : {v:,.3e} {unit}<br>"
                f"Top régions :<br>{tr_str}"
            )
    else:
        hover = [
            f"<b>{lab}</b><br>GICS : {gics_name}<br>{metric_label} : {v:,.3e} {unit}"
            for lab, v in zip(labels, vals)
        ]

    fig.add_trace(go.Scatter(
        x=vals,
        y=list(range(len(vals))),
        mode="markers",
        marker=dict(color=color, size=11, line=dict(width=0.8, color="white")),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    # Annotation totale
    total = float(sub.sum())
    shown = float(top.sum())
    coverage_pct = 100.0 * shown / total if total > 0 else 0.0

    scope = "monde (somme régions)" if aggregate_regions else "top combinaisons region × secteur"
    fig.update_layout(
        title=dict(
            text=(
                f"GICS « {gics_name} » — top {len(labels)} sous-secteurs "
                f"par {metric_label.lower()}<br>"
                f"<sub>{scope} · couvre {coverage_pct:.1f}% du total GICS "
                f"({total:,.2e} {unit})</sub>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(
            title=f"{metric_label} ({unit})",
            tickformat=".2e",
            gridcolor="#ecf0f1",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            tickfont=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(450, len(labels) * 26),
        width=1000,
        margin=dict(l=380, r=80, t=85, b=55),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Graphique 8 — Top fournisseurs externes (hors même GICS) d'un secteur GICS
# ---------------------------------------------------------------------------

def plot_gics_external_inputs(
    Z: pd.DataFrame,
    gics_name: str,
    top_n: int = 15,
    aggregate_regions: bool = True,
    exclude_same_gics: bool = True,
    S: pd.DataFrame | None = None,
    stressor: str | list[str] | None = None,
    stressor_unit: str = "kg",
    unit: str = "M€",
    save_path: str | None = None,
    concordance: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Lollipop des sous-secteurs EXIOBASE qui sont les plus grands fournisseurs
    intermédiaires d'un GICS donné — typiquement filtré pour exclure les
    fournisseurs appartenant au même GICS (vraie chaîne externe).

    Deux modes :
      1. **Flux monétaires** (S=None) — sommes Σ_j Z[i, j] pour j ∈ GICS
      2. **Impact environnemental direct** (S + stressor fournis) —
         pondère chaque flux par l'intensité du fournisseur :
            emissions[i] = S[stressor, i] · Σ_j Z[i, j]
         C'est la quantité d'émissions du fournisseur i directement
         causées par ses ventes intermédiaires au GICS cible.

    Note sur les multiplicateurs : pour pondérer par le multiplicateur de
    Leontief m = S · L (qui ajoute aussi les effets indirects en amont du
    fournisseur), passer la sortie de compute_multipliers reconditionnée :
        S_mult = pd.DataFrame([m.values], index=["multiplier"], columns=Z.index)
        plot_gics_external_inputs(Z, ..., S=S_mult, stressor="multiplier",
                                   stressor_unit="kg / vente directe au GICS")

    Parameters
    ----------
    Z : pd.DataFrame
        Matrice des flux intermédiaires (lignes = fournisseurs i,
        colonnes = acheteurs j), MultiIndex (region, sector).
    gics_name : str
        GICS acheteur dont on étudie les inputs (ex: "Energy").
    top_n : int
        Nombre de fournisseurs à afficher.
    aggregate_regions : bool
        True (défaut) : agrège les fournisseurs par secteur (somme régions).
        False : garde la granularité (region, sector) côté fournisseur.
    exclude_same_gics : bool
        True (défaut) : exclut les fournisseurs appartenant au même GICS.
    S : pd.DataFrame, optional
        Matrice d'intensités stresseurs (rows: stressors, cols: (region, sector)).
        Si fournie avec `stressor`, active le mode "impact environnemental".
    stressor : str | list[str], optional
        Nom du stresseur (ou liste à sommer) dans S — ex: "CO2 - combustion - air".
    stressor_unit : str
        Unité de la métrique environnementale (par défaut "kg").
    unit : str
        Unité monétaire des flux (par défaut "M€"). Utilisée si S/stressor absents.
    save_path : str, optional
    concordance : pd.DataFrame, optional

    Returns
    -------
    plotly Figure
    """
    if concordance is None:
        concordance = load_concordance()

    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    all_sectors = Z.index.get_level_values("sector").unique()
    gics_of_sector = {s: _map_sector(s, lookup, norm_lookup) for s in all_sectors}

    # --- Colonnes acheteuses : secteurs EXIOBASE du GICS cible ---
    col_sectors = Z.columns.get_level_values("sector")
    buyer_mask  = np.array([gics_of_sector.get(s) == gics_name for s in col_sectors])
    if not buyer_mask.any():
        raise ValueError(f"Aucun secteur EXIOBASE trouvé pour le GICS '{gics_name}'.")

    # Inputs totaux reçus par les colonnes du GICS, par ligne (fournisseur)
    inputs = pd.Series(
        Z.values[:, buyer_mask].sum(axis=1),
        index=Z.index,
        name="inputs",
    )

    # --- Pondération environnementale optionnelle ---
    weighted       = False
    display_unit   = unit
    metric_label   = "Flux monétaire"
    stressor_label = ""
    if (S is not None) ^ (stressor is not None):
        raise ValueError("S et stressor doivent être fournis ensemble (ou aucun).")
    if S is not None and stressor is not None:
        stressors = [stressor] if isinstance(stressor, str) else list(stressor)
        S_sub     = S.loc[S.index.intersection(stressors)]
        if S_sub.empty:
            raise ValueError(f"Aucun stresseur trouvé dans S parmi {stressors}.")
        # Intensité agrégée par (region, sector), alignée sur l'index de inputs
        intensity = S_sub.sum(axis=0).reindex(inputs.index, fill_value=0.0)
        inputs    = inputs * intensity.values   # impact direct du fournisseur
        weighted       = True
        display_unit   = stressor_unit
        stressor_label = stressors[0] if len(stressors) == 1 else f"{len(stressors)} stresseurs"
        metric_label   = f"Émissions directes ({stressor_label})"

    # --- Agrégation et filtrage same-GICS ---
    if aggregate_regions:
        agg      = inputs.groupby(level="sector").sum()
        agg_gics = pd.Series(
            [gics_of_sector.get(s, "UNKNOWN") for s in agg.index],
            index=agg.index,
        )
        if exclude_same_gics:
            keep = agg_gics != gics_name
            agg, agg_gics = agg[keep], agg_gics[keep]
        top    = agg.sort_values(ascending=False).head(top_n).sort_values(ascending=True)
        labels = top.index.tolist()
        gics_per_label = [agg_gics.get(s, "UNKNOWN") for s in labels]

        # Top 3 régions fournisseuses par secteur (pour le hover)
        top_regions: dict[str, pd.Series] = {}
        for s in labels:
            sub = inputs[inputs.index.get_level_values("sector") == s]
            top_regions[s] = (
                sub.groupby(level="region").sum()
                   .sort_values(ascending=False).head(3)
            )

        external_total = float(agg.sum())  # total filtré (déjà sans intra si flag)
    else:
        sup_gics = pd.Series(
            [gics_of_sector.get(s, "UNKNOWN") for _, s in inputs.index],
            index=inputs.index,
        )
        if exclude_same_gics:
            keep = sup_gics != gics_name
            inputs, sup_gics = inputs[keep], sup_gics[keep]
        top    = inputs.sort_values(ascending=False).head(top_n).sort_values(ascending=True)
        labels = [f"{r} — {s}" for r, s in top.index]
        gics_per_label = [sup_gics[idx] for idx in top.index]
        external_total = float(inputs.sum())

    vals   = top.values
    colors = [GICS_COLORS.get(g, "#bdc3c7") for g in gics_per_label]

    fig = go.Figure()

    # Tiges horizontales (couleur = GICS du fournisseur)
    for i, (val, col) in enumerate(zip(vals, colors)):
        fig.add_shape(
            type="line",
            x0=0, x1=val, y0=i, y1=i,
            line=dict(color=col, width=1.5),
        )

    # Hover détaillé
    flow_label = f"{metric_label} vers GICS {gics_name}"
    if aggregate_regions:
        hover = []
        for lab, v, g in zip(labels, vals, gics_per_label):
            tr = top_regions[lab]
            tr_str = "<br>".join(f"  {r} : {val:,.2e}" for r, val in tr.items())
            hover.append(
                f"<b>{lab}</b><br>"
                f"GICS fournisseur : {g}<br>"
                f"{flow_label} : {v:,.3e} {display_unit}<br>"
                f"Top régions fournisseuses :<br>{tr_str}"
            )
    else:
        hover = [
            f"<b>{lab}</b><br>GICS fournisseur : {g}<br>"
            f"{flow_label} : {v:,.3e} {display_unit}"
            for lab, v, g in zip(labels, vals, gics_per_label)
        ]

    fig.add_trace(go.Scatter(
        x=vals,
        y=list(range(len(vals))),
        mode="markers",
        marker=dict(color=colors, size=11, line=dict(width=0.8, color="white")),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    # Légende GICS (seulement ceux présents dans le top)
    shown = set()
    for g, col in GICS_COLORS.items():
        if g in set(gics_per_label) and g not in shown:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=col, size=9),
                name=g, showlegend=True,
            ))
            shown.add(g)

    shown_total  = float(top.sum())
    coverage_pct = 100.0 * shown_total / external_total if external_total > 0 else 0.0

    excl_str  = "hors fournisseurs même GICS" if exclude_same_gics else "tous fournisseurs"
    scope     = "agrégé monde" if aggregate_regions else "top combinaisons region × secteur"
    mode_str  = (f" · pondéré par {stressor_label}" if weighted else "")
    title_main = (
        f"Top {len(labels)} fournisseurs du GICS « {gics_name} »"
        f" — impact CO2/stresseur direct" if weighted
        else f"Top {len(labels)} fournisseurs externes du GICS « {gics_name} »"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{title_main} ({excl_str})<br>"
                f"<sub>{scope}{mode_str} · couvre {coverage_pct:.1f}% du total filtré "
                f"({external_total:,.2e} {display_unit})</sub>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(
            title=f"{metric_label} ({display_unit})",
            tickformat=".2e",
            gridcolor="#ecf0f1",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            tickfont=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title="GICS fournisseur", x=1.01, y=1),
        height=max(450, len(labels) * 26),
        width=1050,
        margin=dict(l=380, r=180, t=90, b=55),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Graphique 10 — Origine GICS des émissions intermédiaires (11×11 matrix)
# ---------------------------------------------------------------------------

GICS_ORDER = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Communication Services",
    "Utilities", "Real Estate",
]


def compute_gics_emission_matrix(
    Z: pd.DataFrame,
    S: pd.DataFrame,
    stressor: str | list[str],
    concordance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construit la matrice (GICS × GICS) des émissions directes des fournisseurs
    GICS dues à leurs ventes intermédiaires aux acheteurs GICS.

        emissions[g_buyer, g_supplier] =
            Σ_{i ∈ g_supplier} Σ_{j ∈ g_buyer} S[k, i] · Z[i, j]

    Lecture :
        Ligne g_b = combien le GICS g_b "subit" indirectement en achats à chaque
        GICS fournisseur. La diagonale = émissions intra-GICS, le hors-diagonale
        = émissions causées par les autres GICS.

    Parameters
    ----------
    Z : pd.DataFrame (9800×9800)
    S : pd.DataFrame (e×9800)
    stressor : str | list[str]
    concordance : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame (11×11) : rows = g_buyer, columns = g_supplier
    """
    if concordance is None:
        concordance = load_concordance()

    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    all_sectors    = Z.index.get_level_values("sector").unique()
    gics_of_sector = {s: _map_sector(s, lookup, norm_lookup) for s in all_sectors}

    # Intensité par ligne (region, sector) : somme des stresseurs demandés
    stressors = [stressor] if isinstance(stressor, str) else list(stressor)
    S_sub     = S.loc[S.index.intersection(stressors)]
    if S_sub.empty:
        raise ValueError(f"Aucun stresseur trouvé dans S parmi {stressors}.")
    intensity_vec = (
        S_sub.sum(axis=0).reindex(Z.index, fill_value=0.0).values.astype(np.float64)
    )

    row_sectors = Z.index.get_level_values("sector")
    col_sectors = Z.columns.get_level_values("sector")
    unique_gics = sorted(set(gics_of_sector.values()))

    row_masks = {g: np.array([gics_of_sector.get(s) == g for s in row_sectors])
                 for g in unique_gics}
    col_masks = {g: np.array([gics_of_sector.get(s) == g for s in col_sectors])
                 for g in unique_gics}

    Z_arr = Z.values
    result = pd.DataFrame(0.0, index=unique_gics, columns=unique_gics)

    for g_supplier in unique_gics:
        rm = row_masks[g_supplier]
        if not rm.any():
            continue
        # Pré-pondère les lignes du fournisseur par l'intensité
        Z_rows = Z_arr[rm] * intensity_vec[rm, np.newaxis]   # (n_sup, 9800)
        for g_buyer in unique_gics:
            cm = col_masks[g_buyer]
            if not cm.any():
                continue
            result.loc[g_buyer, g_supplier] = float(Z_rows[:, cm].sum())

    return result


DIRECT_KEY   = "Direct (own prod.)"
DIRECT_COLOR = "#2c3e50"   # sombre, distinct des couleurs GICS


def compute_gics_direct_emissions(
    x: pd.Series,
    S: pd.DataFrame,
    stressor: str | list[str],
    concordance: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Émissions directes (production-based, scope 1) agrégées par GICS :

        direct[g] = Σ_{j ∈ g} S[k, j] · x[j]

    Inclut toutes les émissions des secteurs de g, quelle que soit la
    destination de leur output (intra-GICS, extra-GICS, ou demande finale).

    Parameters
    ----------
    x : pd.Series (n,)       — output total, MultiIndex (region, sector)
    S : pd.DataFrame (e×n)   — intensités stresseurs
    stressor : str | list[str]
    concordance : pd.DataFrame, optional

    Returns
    -------
    pd.Series indexée par GICS
    """
    if concordance is None:
        concordance = load_concordance()

    lookup, norm_lookup = _build_lookup(concordance, "gics_name")
    stressors = [stressor] if isinstance(stressor, str) else list(stressor)
    S_sub     = S.loc[S.index.intersection(stressors)]
    if S_sub.empty:
        raise ValueError(f"Aucun stresseur trouvé dans S parmi {stressors}.")

    intensity        = S_sub.sum(axis=0).reindex(x.index, fill_value=0.0)
    sector_emissions = intensity * x                                  # (n,)
    sectors          = x.index.get_level_values("sector")
    gics_labels      = pd.Series(
        [_map_sector(s, lookup, norm_lookup) for s in sectors],
        index=x.index,
    )
    return sector_emissions.groupby(gics_labels).sum()


def plot_gics_emission_origin(
    Z: pd.DataFrame,
    S: pd.DataFrame,
    stressor: str | list[str],
    x: pd.Series | None = None,
    include_direct: bool = False,
    stressor_unit: str = "kg",
    save_path: str | None = None,
    concordance: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Pour chaque GICS acheteur, montre l'origine de ses émissions associées,
    en deux sous-graphiques :
        (gauche)  Absolu       — barres empilées en {stressor_unit}
        (droite)  Normalisé    — barres empilées à 100 % (proportion)

    Deux modes :
      - include_direct=False (défaut) :
            décomposition par GICS fournisseur des émissions causées par
            les achats intermédiaires de g (existing matrix, diagonale = intra-GICS).
            Annotation "intra : X%" à droite de chaque barre.

      - include_direct=True (nécessite x) :
            la diagonale intra-GICS est remplacée par les émissions directes
            totales de g (scope 1, inclut déjà la part intra-intermédiaire).
            La colonne "Direct (own prod.)" en sombre représente cette part.
            Annotation "direct : X%" à droite (% de Direct dans le total).
            Total par ligne = direct[g] + emissions causees a l'exterieur du GICS.

    Parameters
    ----------
    Z, S, stressor, stressor_unit, concordance : voir compute_gics_emission_matrix
    x : pd.Series, optionnel
        Output total. Obligatoire si include_direct=True.
    include_direct : bool
        Si True, remplace la diagonale intra-GICS par les emissions directes
        totales du GICS.
    save_path : str, optional

    Returns
    -------
    plotly Figure
    """
    if include_direct and x is None:
        raise ValueError("x est requis quand include_direct=True")

    M = compute_gics_emission_matrix(Z, S, stressor, concordance)

    if include_direct:
        direct = compute_gics_direct_emissions(x, S, stressor, concordance)
        # On vide la diagonale (intra-GICS intermédiaire, déjà incluse dans direct)
        for g in M.index:
            if g in M.columns:
                M.loc[g, g] = 0.0
        # Nouvelle colonne dédiée
        M[DIRECT_KEY] = direct.reindex(M.index, fill_value=0.0)

    # Ordre des colonnes : Direct en premier (si présent), puis GICS_ORDER
    if include_direct:
        col_order = [DIRECT_KEY] + [g for g in GICS_ORDER if g in M.columns]
    else:
        col_order = [g for g in GICS_ORDER if g in M.columns] + \
                    [g for g in M.columns if g not in GICS_ORDER]
    M = M[col_order]

    # Tri : GICS acheteurs par émissions totales croissantes (plus gros en haut)
    totals = M.sum(axis=1)
    order  = totals.sort_values(ascending=True).index.tolist()
    M_abs  = M.loc[order]
    M_pct  = M_abs.div(M_abs.sum(axis=1).replace(0, np.nan), axis=0) * 100
    M_pct  = M_pct.fillna(0.0)

    stress_label = stressor if isinstance(stressor, str) else f"{len(stressor)} stresseurs"

    abs_subtitle = f"Émissions absolues ({stressor_unit})"
    if include_direct:
        pct_subtitle = "Part par origine (% — Direct + autres GICS)"
    else:
        pct_subtitle = "Part par GICS fournisseur (%)"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[abs_subtitle, pct_subtitle],
        horizontal_spacing=0.09,
        column_widths=[0.5, 0.5],
        shared_yaxes=True,
    )

    def _color(g: str) -> str:
        return DIRECT_COLOR if g == DIRECT_KEY else GICS_COLORS.get(g, "#bdc3c7")

    for g_supplier in M_abs.columns:
        color = _color(g_supplier)
        fig.add_trace(go.Bar(
            y=M_abs.index,
            x=M_abs[g_supplier],
            name=g_supplier,
            orientation="h",
            marker_color=color,
            legendgroup=g_supplier,
            showlegend=True,
            hovertemplate=(
                f"<b>%{{y}}</b> ← {g_supplier}<br>"
                f"%{{x:,.3e}} {stressor_unit}<extra></extra>"
            ),
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            y=M_pct.index,
            x=M_pct[g_supplier],
            name=g_supplier,
            orientation="h",
            marker_color=color,
            legendgroup=g_supplier,
            showlegend=False,
            hovertemplate=(
                f"<b>%{{y}}</b> ← {g_supplier}<br>"
                f"%{{x:.2f}}%<extra></extra>"
            ),
        ), row=1, col=2)

    # Annotation : part intra-GICS (sans direct) ou part directe (avec direct)
    for g in M_pct.index:
        if include_direct:
            share = float(M_pct.loc[g, DIRECT_KEY]) if DIRECT_KEY in M_pct.columns else 0.0
            text  = f"direct : <b>{share:.1f}%</b>"
        else:
            share = float(M_pct.loc[g, g]) if g in M_pct.columns else 0.0
            text  = f"intra : <b>{share:.1f}%</b>"
        fig.add_annotation(
            x=102, y=g,
            text=text,
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color="#34495e"),
            xref="x2", yref="y2",
        )

    mode_str = (
        "scope 1 (direct) + supply chain extra-GICS"
        if include_direct
        else "supply chain intermédiaire complète (intra + extra GICS)"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Origine des émissions par GICS acheteur — {stress_label}<br>"
                f"<sub>{mode_str}</sub>"
            ),
            font=dict(size=13), x=0.5,
        ),
        barmode="stack",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title="Origine",
            x=1.02, y=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#bdc3c7", borderwidth=0.5,
            font=dict(size=10),
        ),
        height=560,
        width=1300,
        margin=dict(l=180, r=210, t=85, b=55),
    )

    fig.update_xaxes(title_text=stressor_unit, row=1, col=1,
                     tickformat=".1e", gridcolor="#ecf0f1")
    fig.update_xaxes(title_text="%", row=1, col=2,
                     range=[0, 115], gridcolor="#ecf0f1")
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=1)

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from io_pipeline import run_pipeline, compute_multipliers

    STRESSOR = "CO2 - combustion - air"

    print("Running pipeline...")
    result = run_pipeline(
        year=2019, table_type="pxp", extension="air_emissions",
        stressor_filter=[STRESSOR],
    )
    A, S, x = result["A"], result["S"], result["x"]
    unit_label = result["unit"].get(STRESSOR, "kg")

    print("Computing multipliers...")
    m = compute_multipliers(A, S, STRESSOR)

    print("Generating graph 4 — multiplier lollipop...")
    fig4 = plot_multipliers(
        m,
        stressor=STRESSOR,
        top_n=30,
        unit=f"{unit_label} / M€",
        save_path="graph4_multipliers.html",
    )

    print("Generating graph 6 — intensity × output scatter...")
    fig6 = plot_intensity_output(
        S, x,
        stressor=STRESSOR,
        unit_stressor=unit_label,
        save_path="graph6_intensity_output.html",
    )

    print("Done. Ouvrir graph4_multipliers.html et graph6_intensity_output.html dans un navigateur.")

    # --- Graph 7 : décomposition d'un GICS en ses sous-secteurs EXIOBASE ---
    print("Generating graph 7 — GICS Energy decomposition (sous-secteurs)...")
    fig7 = plot_gics_subsectors(
        x, gics_name="Energy", top_n=15,
        save_path="Figs/graph7_subsectors_energy.html",
    )

    # --- Graph 8 : top fournisseurs externes d'un GICS ---
    print("Loading Z for graph 8 (top external inputs)...")
    Z = result["Z"]

    print("Generating graph 8 — top fournisseurs externes du GICS Energy...")
    fig8 = plot_gics_external_inputs(
        Z, gics_name="Energy", top_n=15,
        aggregate_regions=True, exclude_same_gics=True,
        save_path="Figs/graph8_inputs_energy.html",
    )

    print("Generating graph 8b — top fournisseurs externes du GICS Industrials...")
    fig8b = plot_gics_external_inputs(
        Z, gics_name="Industrials", top_n=15,
        aggregate_regions=True, exclude_same_gics=True,
        save_path="Figs/graph8_inputs_industrials.html",
    )

    # --- Graph 9 : meme top mais pondere par intensite CO2 ---
    # Variante : qui sont les plus gros emetteurs DIRECTS de CO2
    # parmi les fournisseurs du GICS (Z * S[CO2]) ?
    print("Generating graph 9 — top fournisseurs CO2 du GICS Energy...")
    fig9 = plot_gics_external_inputs(
        Z, gics_name="Energy", top_n=15,
        aggregate_regions=True, exclude_same_gics=True,
        S=S, stressor=STRESSOR, stressor_unit=unit_label,
        save_path="Figs/graph9_co2_inputs_energy.html",
    )

    print("Generating graph 9b — top fournisseurs CO2 du GICS Industrials...")
    fig9b = plot_gics_external_inputs(
        Z, gics_name="Industrials", top_n=15,
        aggregate_regions=True, exclude_same_gics=True,
        S=S, stressor=STRESSOR, stressor_unit=unit_label,
        save_path="Figs/graph9_co2_inputs_industrials.html",
    )

    # --- Graph 10 : origine GICS des émissions intermédiaires (11 GICS) ---
    print("Generating graph 10 — origine GICS des emissions intermediaires...")
    fig10 = plot_gics_emission_origin(
        Z, S=S, stressor=STRESSOR, stressor_unit=unit_label,
        save_path="Figs/graph10_emission_origin.html",
    )

    # --- Graph 10b : meme avec emissions directes incluses (scope 1) ---
    print("Generating graph 10b — avec emissions directes (scope 1)...")
    fig10b = plot_gics_emission_origin(
        Z, S=S, stressor=STRESSOR, x=x,
        include_direct=True, stressor_unit=unit_label,
        save_path="Figs/graph10_emission_origin_with_direct.html",
    )

    print("Done. Ouvrir Figs/graph{7,8,9,10}_*.html dans un navigateur.")
