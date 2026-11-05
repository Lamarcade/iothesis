"""
Trajectoires temporelles EXIOBASE (1995-2024) vs NGFS (2020-2100).

S'appuie sur le mapping NGFS↔EXIOBASE défini dans `ngfs_exiobase_mapping.py`
et étend chaque comparaison single-year à une trajectoire annuelle complète.

Structure :
    trajectory_exiobase(mapping, years, ...) → pd.Series           (cache CSV)
    trajectory_ngfs(mapping, years, ...)     → pd.DataFrame long
    aggregate_models_median_band(df)         → pd.DataFrame
                                               (scenario × year → median/min/max)
    plot_trajectory(key, ...)                → go.Figure (1 mapping)
    plot_dashboard(keys, ...)                → go.Figure (grille subplots)
    run_all_trajectories(scope, ...)         → dict {key: fig}

Cache : Data/NGFS5/_cache/exio_traj_{key}_{scope}_{table}.csv
"""

from pathlib import Path
from math import ceil
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from longterm.ngfs_exiobase_mapping import (
    Map, MAPPING, get_mapping,
    aggregate_exiobase, aggregate_ngfs,
)
from longterm.ngfs_data import NGFS_DIR

CACHE_DIR = NGFS_DIR / "_cache"

# Couleurs scénarios NGFS (cohérence visuelle entre tous les graphes)
NGFS_SCENARIO_COLORS = {
    "Net Zero 2050":                              "#2ca02c",   # vert
    "Below 2°C":                             "#1f77b4",   # bleu
    "Nationally Determined Contributions (NDCs)": "#ff7f0e",   # orange
    "Delayed transition":                         "#9467bd",   # violet
    "Current Policies":                           "#d62728",   # rouge
    "Fragmented World":                           "#8c564b",   # marron
    "Low demand":                                 "#17becf",   # teal
    "Baseline":                                   "#7f7f7f",   # gris (NiGEM)
}

# Couleur EXIOBASE
EXIO_COLOR = "#000000"


# ---------------------------------------------------------------------------
# 1) Trajectoires EXIOBASE (avec cache CSV)
# ---------------------------------------------------------------------------

def _exio_cache_path(key: str, scope: str, table_type: str) -> Path:
    return CACHE_DIR / f"exio_traj_{key}_{scope}_{table_type}.csv"


def trajectory_exiobase(
    mapping: Map | str,
    years: list[int] | None = None,
    table_type: str = "pxp",
    scope: str = "world",
    cache: bool = True,
    force_reload: bool = False,
) -> pd.Series:
    """
    Calcule l'agrégat EXIOBASE pour un mapping, année par année.

    Parameters
    ----------
    mapping     : Map ou clé (str)
    years       : liste d'années (par défaut 2010-2024 = 15 années)
    table_type  : 'pxp' | 'ixi'
    scope       : 'world' | 'europe'
    cache       : utilise / écrit le cache CSV
    force_reload: bypass le cache

    Returns
    -------
    pd.Series indexée par year, valeurs dans l'unité NGFS du mapping
    """
    if isinstance(mapping, str):
        m = get_mapping(mapping)
    else:
        m = mapping

    if years is None:
        years = list(range(2010, 2025))     # défaut horizon historique récent

    # --- Cache ---
    cache_path = _exio_cache_path(m.key, scope, table_type)
    if cache and not force_reload and cache_path.exists():
        df = pd.read_csv(cache_path)
        cached = df.set_index("year")["value"]
        missing = [y for y in years if y not in cached.index]
        if not missing:
            return cached.loc[years]
        # Sinon on complète et on resauvegarde
        print(f"  [cache partiel] manque {len(missing)} années, complète...")
    else:
        cached = pd.Series(dtype=float, name="value")

    # --- Calcul des années manquantes ---
    new_vals = {}
    to_compute = [y for y in years if y not in cached.index]
    for i, y in enumerate(to_compute, 1):
        print(f"  [{i}/{len(to_compute)}] aggregate_exiobase({m.key}, year={y}, "
              f"{table_type}, {scope})...", flush=True)
        v, _info = aggregate_exiobase(m, y, table_type=table_type, scope=scope)
        new_vals[y] = v

    # Fusion + sauvegarde
    full = pd.concat(
        [cached, pd.Series(new_vals, name="value")]
    ).sort_index()
    full.index.name = "year"
    full = full[~full.index.duplicated(keep="last")]

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        full.reset_index().to_csv(cache_path, index=False)

    return full.loc[full.index.intersection(years)].sort_index()


# ---------------------------------------------------------------------------
# 2) Trajectoires NGFS (pas de cache — déjà parquet)
# ---------------------------------------------------------------------------

def trajectory_ngfs(
    mapping: Map | str,
    years: list[int] | None = None,
    scope: str = "world",
    scenarios: list[str] | None = None,
    sources: list[str] | None = None,
    sum_regions: bool = True,
) -> pd.DataFrame:
    """
    Récupère les trajectoires NGFS pour un mapping, sur la liste d'années.

    Returns
    -------
    pd.DataFrame long : [source, Model, Scenario, Region, Year, value, Unit]
    """
    if isinstance(mapping, str):
        m = get_mapping(mapping)
    else:
        m = mapping

    if years is None:
        years = list(range(2020, 2031))

    frames = []
    for y in years:
        df = aggregate_ngfs(m, y, scope=scope, source=sources,
                             sum_regions=sum_regions)
        if df.empty:
            continue
        if scenarios:
            df = df[df["Scenario"].isin(scenarios)]
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_models_median_band(df_ngfs: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les trajectoires par (Scenario, Year) en calculant median/min/max
    sur les (source, Model) disponibles.

    Returns
    -------
    pd.DataFrame [Scenario, Year, median, min, max, n_models]
    """
    if df_ngfs.empty:
        return pd.DataFrame(columns=["Scenario", "Year", "median", "min", "max", "n_models"])

    grp = df_ngfs.groupby(["Scenario", "Year"]).agg(
        median=("value",  "median"),
        min=("value",     "min"),
        max=("value",     "max"),
        n_models=("Model", "nunique"),
    ).reset_index()
    return grp.sort_values(["Scenario", "Year"])


# ---------------------------------------------------------------------------
# 3) Plot single trajectory
# ---------------------------------------------------------------------------

def plot_trajectory(
    mapping_key: str,
    scope: str = "world",
    exio_years: tuple[int, int] = (2010, 2022),
    ngfs_years: tuple[int, int] = (2020, 2030),
    scenarios: list[str] | None = None,
    sources: list[str] | None = None,
    table_type: str = "pxp",
    save_path: str | None = None,
    cache: bool = True,
    force_reload: bool = False,
    mode: str = "auto",
    normalize_at_year: int = 2020,
) -> go.Figure:
    """
    Génère un graphique combiné EXIOBASE (historique) + NGFS (projections).

    Parameters
    ----------
    mode : 'auto' | 'absolute' | 'normalized'
        'auto'       : 'normalized' si quality='incommensurable', sinon 'absolute'
        'absolute'   : unités natives (NGFS_unit)
        'normalized' : tout divisé par valeur à `normalize_at_year` (indice)
    normalize_at_year : année de référence pour mode normalisé (default 2020)
    """
    m = get_mapping(mapping_key)
    effective_mode = _resolve_mode(mode, m.quality)

    print(f"\n=== Trajectoire {mapping_key} ({scope}, mode={effective_mode}) ===")
    print(f"  EXIOBASE : {exio_years[0]}-{exio_years[1]}  |  "
          f"NGFS : {ngfs_years[0]}-{ngfs_years[1]}")

    # --- 1. EXIOBASE ---
    exio_years_list = list(range(exio_years[0], exio_years[1] + 1))
    exio_series = trajectory_exiobase(m, exio_years_list, table_type=table_type,
                                       scope=scope, cache=cache,
                                       force_reload=force_reload)

    # --- 2. NGFS ---
    ngfs_years_list = list(range(ngfs_years[0], ngfs_years[1] + 1))
    ngfs_df = trajectory_ngfs(m, ngfs_years_list, scope=scope,
                               scenarios=scenarios, sources=sources)
    band = aggregate_models_median_band(ngfs_df)

    # --- 2b. Normalisation éventuelle ---
    if effective_mode == "normalized":
        exio_series = _normalize_series(exio_series, normalize_at_year)
        band = _normalize_band(band, normalize_at_year)
        y_unit = f"indice (base {normalize_at_year} = 1)"
        y_tickfmt = ".3f"
        hover_fmt = ".3f"
    else:
        y_unit = m.ngfs_unit
        y_tickfmt = ".2e"
        hover_fmt = ".3e"

    # --- 3. Plot ---
    fig = go.Figure()

    # Bandes NGFS par scénario (min-max + médiane)
    scenarios_seen = []
    for scen in sorted(band["Scenario"].unique() if not band.empty else []):
        sub = band[band["Scenario"] == scen].sort_values("Year")
        color = NGFS_SCENARIO_COLORS.get(scen, "#888888")
        scenarios_seen.append(scen)

        # Bande min-max (deux traces invisibles pour fill)
        fig.add_trace(go.Scatter(
            x=sub["Year"], y=sub["max"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
            name=f"{scen} max",
        ))
        fig.add_trace(go.Scatter(
            x=sub["Year"], y=sub["min"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=_rgba(color, 0.15),
            showlegend=False, hoverinfo="skip",
            name=f"{scen} min",
        ))

        # Ligne médiane
        fig.add_trace(go.Scatter(
            x=sub["Year"], y=sub["median"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=scen,
            hovertemplate=(
                f"<b>{scen}</b><br>"
                "Year %{x}<br>"
                f"median %{{y:{hover_fmt}}} {y_unit}<br>"
                f"n_models %{{customdata}}<extra></extra>"
            ),
            customdata=sub["n_models"],
        ))

    # Ligne EXIOBASE
    if not exio_series.empty:
        fig.add_trace(go.Scatter(
            x=exio_series.index, y=exio_series.values,
            mode="lines+markers",
            line=dict(color=EXIO_COLOR, width=3),
            marker=dict(size=6, color=EXIO_COLOR),
            name=f"EXIOBASE ({table_type})",
            hovertemplate=(
                "<b>EXIOBASE</b><br>"
                "Year %{x}<br>"
                f"%{{y:{hover_fmt}}} {y_unit}<extra></extra>"
            ),
        ))

    # Zone de chevauchement
    overlap_lo = max(exio_years[0], ngfs_years[0])
    overlap_hi = min(exio_years[1], ngfs_years[1])
    if overlap_lo < overlap_hi:
        fig.add_vrect(
            x0=overlap_lo, x1=overlap_hi,
            fillcolor="#bdc3c7", opacity=0.18, line_width=0,
            annotation_text="overlap", annotation_position="top left",
            annotation_font=dict(size=9, color="#7f8c8d"),
        )

    # Mise en page
    quality_tag = f"[{m.quality}]" if m.quality != "direct" else ""
    norm_tag = f" — normalisé base {normalize_at_year}" if effective_mode == "normalized" else ""
    fig.update_layout(
        title=dict(
            text=(
                f"{m.ngfs_variable} — {scope} {quality_tag}{norm_tag}<br>"
                f"<sub>mapping: {m.key}  |  {m.theme}  |  "
                f"unit: {y_unit}</sub>"
            ),
            font=dict(size=13), x=0.5,
        ),
        xaxis=dict(
            title="Année", gridcolor="#ecf0f1",
            tickmode="linear",
            tick0=exio_years[0], dtick=2,
            range=[exio_years[0] - 0.5, ngfs_years[1] + 0.5],
        ),
        yaxis=dict(
            title=y_unit, gridcolor="#ecf0f1",
            zeroline=True, zerolinecolor="#95a5a6",
            tickformat=y_tickfmt,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title="Source / Scénario",
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#bdc3c7", borderwidth=0.5,
            font=dict(size=10),
        ),
        height=560, width=1200,
        margin=dict(l=70, r=240, t=80, b=55),
        hovermode="x unified",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4) Dashboard multi-mapping
# ---------------------------------------------------------------------------

def plot_dashboard(
    mapping_keys: list[str],
    scope: str = "world",
    exio_years: tuple[int, int] = (2010, 2022),
    ngfs_years: tuple[int, int] = (2020, 2030),
    scenarios: list[str] | None = None,
    table_type: str = "pxp",
    n_cols: int = 3,
    save_path: str | None = None,
    cache: bool = True,
    force_reload: bool = False,
    mode: str = "auto",
    normalize_at_year: int = 2020,
) -> go.Figure:
    """
    Grille de trajectoires : n_cols colonnes × ceil(N/n_cols) lignes.

    Le `mode` ('auto'|'absolute'|'normalized') est appliqué par mapping selon
    sa quality (auto = normalisé si incommensurable).
    """
    n = len(mapping_keys)
    n_rows = ceil(n / n_cols)

    titles = []
    for k in mapping_keys:
        try:
            m = get_mapping(k)
            titles.append(f"{m.key} — {m.theme}")
        except KeyError:
            titles.append(f"{k} (introuvable)")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=(1 / (n_cols +1)),
        vertical_spacing=(1 / (2* n_rows)),
    )

    legend_seen = set()
    exio_years_list = list(range(exio_years[0], exio_years[1] + 1))
    ngfs_years_list = list(range(ngfs_years[0], ngfs_years[1] + 1))

    for idx, key in enumerate(mapping_keys):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        try:
            m = get_mapping(key)
        except KeyError:
            continue

        # Mode effectif par mapping (auto → normalisé si incommensurable)
        eff_mode = _resolve_mode(mode, m.quality)
        tickfmt  = ".3f" if eff_mode == "normalized" else ".1e"
        hov      = ".3f" if eff_mode == "normalized" else ".2e"

        # NGFS
        ngfs_df = trajectory_ngfs(m, ngfs_years_list, scope=scope,
                                   scenarios=scenarios)
        band = aggregate_models_median_band(ngfs_df)

        # EXIOBASE
        exio_series = trajectory_exiobase(m, exio_years_list,
                                           table_type=table_type, scope=scope,
                                           cache=cache, force_reload=force_reload)

        # Normalisation éventuelle (post-fetch, avant trace)
        if eff_mode == "normalized":
            exio_series = _normalize_series(exio_series, normalize_at_year)
            band        = _normalize_band(band, normalize_at_year)

        for scen in sorted(band["Scenario"].unique() if not band.empty else []):
            sub = band[band["Scenario"] == scen].sort_values("Year")
            color = NGFS_SCENARIO_COLORS.get(scen, "#888888")
            show_in_legend = scen not in legend_seen
            legend_seen.add(scen)

            # Bande min-max
            fig.add_trace(go.Scatter(
                x=sub["Year"], y=sub["max"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
                legendgroup=scen,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=sub["Year"], y=sub["min"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=_rgba(color, 0.12),
                showlegend=False, hoverinfo="skip",
                legendgroup=scen,
            ), row=r, col=c)
            # Médiane
            fig.add_trace(go.Scatter(
                x=sub["Year"], y=sub["median"],
                mode="lines",
                line=dict(color=color, width=1.6),
                name=scen, legendgroup=scen,
                showlegend=show_in_legend,
                hovertemplate=(
                    f"<b>{scen}</b><br>"
                    f"%{{x}}: %{{y:{hov}}}<extra></extra>"
                ),
            ), row=r, col=c)

        if not exio_series.empty:
            show_exio = "EXIOBASE" not in legend_seen
            legend_seen.add("EXIOBASE")
            fig.add_trace(go.Scatter(
                x=exio_series.index, y=exio_series.values,
                mode="lines+markers",
                line=dict(color=EXIO_COLOR, width=2.2),
                marker=dict(size=4),
                name=f"EXIOBASE ({table_type})",
                legendgroup="EXIOBASE",
                showlegend=show_exio,
                hovertemplate=f"<b>EXIOBASE</b> %{{x}}: %{{y:{hov}}}<extra></extra>",
            ), row=r, col=c)

        # Overlap
        overlap_lo = max(exio_years[0], ngfs_years[0])
        overlap_hi = min(exio_years[1], ngfs_years[1])
        if overlap_lo < overlap_hi:
            fig.add_vrect(
                x0=overlap_lo, x1=overlap_hi,
                fillcolor="#bdc3c7", opacity=0.12, line_width=0,
                row=r, col=c,
            )

        fig.update_xaxes(
            row=r, col=c, gridcolor="#ecf0f1",
            tickmode="linear", tick0=exio_years[0], dtick=4,
        )
        fig.update_yaxes(row=r, col=c, gridcolor="#ecf0f1",
                          tickformat=tickfmt)

    fig.update_layout(
        title=dict(
            text=(
                f"Trajectoires NGFS vs EXIOBASE — {scope} "
                f"({exio_years[0]}-{ngfs_years[1]})"
            ),
            font=dict(size=14), x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#bdc3c7", borderwidth=0.5,
            font=dict(size=9),
        ),
        height=max(420, 320 * n_rows),
        width=420 * n_cols + 220,
        margin=dict(l=60, r=240, t=80, b=50),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"\nSaved dashboard: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 5) Pipeline complet : un HTML par mapping + dashboard
# ---------------------------------------------------------------------------

def run_all_trajectories(
    scope: str = "world",
    output_dir: str | Path = "Figs/trajectories",
    exio_years: tuple[int, int] = (2010, 2022),
    ngfs_years: tuple[int, int] = (2020, 2030),
    themes: list[str] | None = None,
    quality_filter: tuple[str, ...] = ("direct", "proxy", "incommensurable"),
    table_type: str = "pxp",
    cache: bool = True,
    force_reload: bool = False,
    mode: str = "auto",
    normalize_at_year: int = 2020,
) -> dict[str, go.Figure]:
    """
    Génère :
      - 1 HTML par mapping (filtré par theme / quality)
      - 1 dashboard global agrégeant ceux du même thème
      - 1 fichier mapping_table.csv dans output_dir

    Defaut quality_filter inclut maintenant 'incommensurable' (ex. GDP) —
    ces variables sont plotées en mode normalisé automatiquement (mode='auto').

    Returns
    -------
    dict {mapping_key: go.Figure}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export du tableau de mapping (CSV)
    try:
        export_mapping_table(output_dir / "mapping_table.csv")
    except Exception as e:
        print(f"[WARN] export_mapping_table : {e}")

    # Filtrage
    selected = [
        m for m in MAPPING
        if m.quality in quality_filter
        and (themes is None or m.theme in themes)
    ]
    print(f"Génération de {len(selected)} trajectoires "
          f"(scope={scope}, quality={quality_filter}, "
          f"themes={themes or 'all'})...")

    figs = {}
    for i, m in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {m.key} ({m.theme})...")
        try:
            html_path = output_dir / f"{m.key}_{scope}.html"
            fig = plot_trajectory(
                m.key, scope=scope,
                exio_years=exio_years, ngfs_years=ngfs_years,
                table_type=table_type,
                save_path=str(html_path),
                cache=cache, force_reload=force_reload,
                mode=mode, normalize_at_year=normalize_at_year,
            )
            figs[m.key] = fig
        except Exception as e:
            print(f"  [WARN] erreur sur {m.key} : {e}")

    # Dashboard global
    if figs:
        dashboard_path = output_dir / f"dashboard_{scope}.html"
        plot_dashboard(
            list(figs.keys()), scope=scope,
            exio_years=exio_years, ngfs_years=ngfs_years,
            table_type=table_type,
            save_path=str(dashboard_path),
            cache=cache, force_reload=force_reload,
            mode=mode, normalize_at_year=normalize_at_year,
        )

    return figs


# ---------------------------------------------------------------------------
# 6) Export du tableau de mapping (NGFS ↔ EXIOBASE)
# ---------------------------------------------------------------------------

def export_mapping_table(
    path: str | Path,
    include_resolved: bool = True,
    resolve_year: int = 2019,
    table_type: str = "pxp",
) -> pd.DataFrame:
    """
    Exporte MAPPING en CSV/Excel détaillé pour documentation et audit.

    Colonnes :
        key, theme, quality,
        ngfs_variable, ngfs_unit,
        exiobase_extension, exiobase_unit, to_ngfs_factor,
        n_stressors, exiobase_stressors,
        n_sectors, exiobase_sectors,
        exiobase_demand_categories, include_F, include_F_Y,
        notes

    Parameters
    ----------
    path : chemin .csv ou .xlsx
    include_resolved : si True, résout les patterns 'PREFIX - *' en listant les
        stresseurs réels présents dans `resolve_year` / `table_type`.
    """
    from ngfs_exiobase_mapping import _resolve_stressors
    from io_pipeline import load_extension

    path = Path(path)

    # Pré-charge les extensions une seule fois pour résoudre les patterns
    ext_cache: dict[str, pd.Index] = {}
    if include_resolved:
        needed_exts = sorted({m.exiobase_extension for m in MAPPING
                              if m.exiobase_extension != "system"})
        for ext in needed_exts:
            try:
                ext_cache[ext] = load_extension(resolve_year, ext, table_type)["F"].index
            except Exception as e:
                print(f"[WARN] cache extension {ext} : {e}")

    rows = []
    for m in MAPPING:
        # Résolution des stresseurs (patterns 'PREFIX - *' → listes explicites)
        if include_resolved and m.exiobase_extension in ext_cache:
            resolved = _resolve_stressors(m.exiobase_stressors,
                                           ext_cache[m.exiobase_extension])
        else:
            resolved = (m.exiobase_stressors
                        if isinstance(m.exiobase_stressors, list)
                        else [m.exiobase_stressors])

        sectors = m.exiobase_sectors if m.exiobase_sectors else None
        dcats   = m.exiobase_demand_categories if m.exiobase_demand_categories else None

        rows.append({
            "key":                        m.key,
            "theme":                      m.theme,
            "quality":                    m.quality,
            "ngfs_variable":              m.ngfs_variable,
            "ngfs_unit":                  m.ngfs_unit,
            "exiobase_extension":         m.exiobase_extension,
            "exiobase_unit":              m.exiobase_unit,
            "to_ngfs_factor":             m.to_ngfs_factor,
            "n_stressors":                len(resolved),
            "exiobase_stressors":         " | ".join(map(str, resolved)),
            "n_sectors":                  len(sectors) if sectors else "all",
            "exiobase_sectors":           " | ".join(sectors) if sectors else "",
            "exiobase_demand_categories": " | ".join(dcats) if dcats else "",
            "include_F":                  m.include_F,
            "include_F_Y":                m.include_F_Y,
            "notes":                      m.notes,
        })

    df = pd.DataFrame(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Mapping table exporté : {path}  ({len(df)} mappings)")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgba(hex_color: str, alpha: float) -> str:
    """Convertit '#RRGGBB' en 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _resolve_mode(mode: str, quality: str) -> str:
    """Mode = 'auto' renvoie 'normalized' pour incommensurable, sinon 'absolute'."""
    if mode == "auto":
        return "normalized" if quality == "incommensurable" else "absolute"
    if mode in ("absolute", "normalized"):
        return mode
    raise ValueError(f"mode doit être 'auto'|'absolute'|'normalized', reçu {mode!r}")


def _normalize_series(s: pd.Series, base_year: int) -> pd.Series:
    """Divise par s.loc[base_year]. Si base_year absent, prend l'année dispo la
    plus proche. Retourne s inchangée si tout est NaN ou base=0."""
    if s.empty:
        return s
    if base_year in s.index:
        ref = s.loc[base_year]
    else:
        # plus proche en valeur absolue
        closest = min(s.index, key=lambda y: abs(y - base_year))
        ref = s.loc[closest]
    if pd.isna(ref) or ref == 0:
        return s
    return s / float(ref)


def _normalize_band(band: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """Pour chaque scénario, normalise median/min/max par la valeur médiane à base_year."""
    if band.empty:
        return band
    out = []
    for scen in band["Scenario"].unique():
        sub = band[band["Scenario"] == scen].sort_values("Year").copy()
        if base_year in sub["Year"].values:
            ref = float(sub.loc[sub["Year"] == base_year, "median"].iloc[0])
        else:
            closest = sub.iloc[(sub["Year"] - base_year).abs().argsort().iloc[0]]
            ref = float(closest["median"])
        if pd.isna(ref) or ref == 0:
            out.append(sub)
            continue
        for col in ("median", "min", "max"):
            sub[col] = sub[col] / ref
        out.append(sub)
    return pd.concat(out, ignore_index=True)
