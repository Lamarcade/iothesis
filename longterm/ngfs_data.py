"""
Loader pour les données NGFS Phase V (5 fichiers Excel IAMC).

Structure des fichiers (`sheet="data"`) :
    colonnes = [Model, Scenario, Region, Variable, Unit, year_1, year_2, ...]
    long form, années en colonnes.

Fichiers :
    - IAM_data.xlsx                          : IAM (R5/R12 regions + World)
    - Downscaled_GCAM 6.0 NGFS_data.xlsx     : pays (ISO3) + World/EU27/DEU
    - Downscaled_MESSAGEix-GLOBIOM ...       : idem
    - Downscaled_REMIND-MAgPIE ...           : idem
    - NiGEM_data.xlsx                        : régions économiques NiGEM préfixées

Scopes :
    'world'  → régions World/Global (toutes sources)
    'europe' → agrégats EU pour IAM, ISO3 'EU27' pour Downscaled,
               'Europe' pour NiGEM

Le premier chargement convertit Excel → parquet dans `Data/NGFS5/_cache/` ;
les appels suivants sont quasi instantanés.
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NGFS_DIR  = REPO_ROOT / "Data" / "NGFS5"
CACHE_DIR = NGFS_DIR / "_cache"

NGFS_FILES = {
    "IAM":        "IAM_data.xlsx",
    "GCAM_ds":    "Downscaled_GCAM 6.0 NGFS_data.xlsx",
    "MESSAGE_ds": "Downscaled_MESSAGEix-GLOBIOM 2.0-M-R12-NGFS_data.xlsx",
    "REMIND_ds":  "Downscaled_REMIND-MAgPIE 3.3-4.8_data.xlsx",
    "NiGEM":      "NiGEM_data.xlsx",
}

# Régions exactes pour chaque scope, par source
WORLD_REGIONS = {
    "IAM":        ["World"],
    "GCAM_ds":    ["World"],
    "MESSAGE_ds": ["World"],
    "REMIND_ds":  ["World"],
    "NiGEM":      ["NiGEM NGFS v1.24.2|World"],
}

EUROPE_REGIONS = {
    # IAM : agregat EU27 (ou plus proche) par modèle, sans double comptage :
    #   - GCAM 6.0 : EU-12 + EU-15 = EU27 (sommer)
    #   - REMIND   : EU 28 (≈ EU27 + UK)
    #   - MESSAGE  : pas de match propre EU27 → omis
    "IAM": [
        "GCAM 6.0 NGFS|EU-12",
        "GCAM 6.0 NGFS|EU-15",
        "REMIND-MAgPIE 3.3-4.8|EU 28",
    ],
    # Downscaled : EU27 strict, le plus propre pour comparer EXIOBASE EU27
    "GCAM_ds":    ["EU27"],
    "MESSAGE_ds": ["EU27"],
    "REMIND_ds":  ["EU27"],
    # NiGEM : "Europe" ≈ EU27 (pas exact, mais agrégat le plus proche)
    "NiGEM":      ["NiGEM NGFS v1.24.2|Europe"],
}


# ---------------------------------------------------------------------------
# Loading (avec cache parquet)
# ---------------------------------------------------------------------------

def _cache_path(file_key: str) -> Path:
    return CACHE_DIR / f"{file_key}.parquet"


def _load_one(file_key: str, force_reload: bool = False) -> pd.DataFrame:
    """Charge un fichier NGFS, avec cache parquet local."""
    if file_key not in NGFS_FILES:
        raise ValueError(f"Source inconnue : '{file_key}'. Choix : {list(NGFS_FILES)}")

    cache = _cache_path(file_key)
    if cache.exists() and not force_reload:
        return pd.read_parquet(cache)

    xlsx = NGFS_DIR / NGFS_FILES[file_key]
    print(f"  Reading {xlsx.name} (Excel, premier chargement)...", flush=True)
    df = pd.read_excel(xlsx, sheet_name="data")
    # Normalise les noms de colonnes "Year" en str pour homogeneite
    df.columns = [str(c) if isinstance(c, int) else c for c in df.columns]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"  -> cache parquet : {cache.relative_to(NGFS_DIR.parent)}", flush=True)
    return df


def load_ngfs(
    sources: list[str] | str | None = None,
    scope: str | None = None,
    force_reload: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Charge un ou plusieurs fichiers NGFS, optionnellement filtrés par région.

    Parameters
    ----------
    sources : str | list[str] | None
        Source(s) à charger. None = toutes (5).
        Options : 'IAM', 'GCAM_ds', 'MESSAGE_ds', 'REMIND_ds', 'NiGEM'
    scope : 'world' | 'europe' | None
        Filtre les régions. None = pas de filtre.
    force_reload : bool
        Bypass le cache parquet.

    Returns
    -------
    dict {source_key: DataFrame}
    """
    if sources is None:
        sources = list(NGFS_FILES)
    elif isinstance(sources, str):
        sources = [sources]

    if scope is not None and scope not in ("world", "europe"):
        raise ValueError(f"scope doit être 'world' ou 'europe', reçu : {scope!r}")

    out = {}
    for key in sources:
        df = _load_one(key, force_reload=force_reload)
        if scope is not None:
            regs = WORLD_REGIONS[key] if scope == "world" else EUROPE_REGIONS[key]
            df   = df[df["Region"].isin(regs)].copy()
        out[key] = df
    return out


# ---------------------------------------------------------------------------
# Listing variables / metadata
# ---------------------------------------------------------------------------

def list_variables(
    scope: str = "world",
    sources: list[str] | str | None = None,
    only_top_level: bool = False,
) -> pd.DataFrame:
    """
    Catalogue des variables disponibles pour un scope donné, agrégé sur les sources.

    Parameters
    ----------
    scope : 'world' | 'europe'
    sources : voir load_ngfs
    only_top_level : si True, ne garde que les variables racine (sans '|').

    Returns
    -------
    pd.DataFrame  colonnes : source, Variable, Unit, n_models, n_scenarios, n_regions
    """
    data = load_ngfs(sources=sources, scope=scope)
    rows = []
    for key, df in data.items():
        if df.empty:
            continue
        grp = df.groupby(["Variable", "Unit"], dropna=False).agg(
            n_models=("Model", "nunique"),
            n_scenarios=("Scenario", "nunique"),
            n_regions=("Region", "nunique"),
        ).reset_index()
        grp.insert(0, "source", key)
        rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["source", "Variable", "Unit",
                                      "n_models", "n_scenarios", "n_regions"])
    cat = pd.concat(rows, ignore_index=True)
    if only_top_level:
        cat = cat[~cat["Variable"].astype(str).str.contains(r"\|", regex=True)]
    return cat.sort_values(["Variable", "source"]).reset_index(drop=True)


def variable_hierarchy(
    scope: str = "world",
    sources: list[str] | str | None = None,
) -> pd.DataFrame:
    """
    Décompose chaque variable selon le séparateur IAMC '|' :
        level_1 / level_2 / ... / full_variable / unit / source
    Pratique pour explorer les branches (Emissions, Energy, GDP, ...).
    """
    cat   = list_variables(scope=scope, sources=sources)
    parts = cat["Variable"].astype(str).str.split("|", expand=True)
    parts.columns = [f"level_{i+1}" for i in range(parts.shape[1])]
    full  = pd.concat([parts, cat[["Variable", "Unit", "source"]]
                                  .rename(columns={"Variable": "full_variable",
                                                    "Unit":     "unit"})],
                      axis=1)
    return full


def list_top_categories(
    scope: str = "world",
    sources: list[str] | str | None = None,
) -> pd.DataFrame:
    """
    Compte le nombre de variables par catégorie de premier niveau, par source.
    Utile pour avoir une vue d'ensemble des thèmes (Emissions, Energy, GDP, ...).
    """
    cat = list_variables(scope=scope, sources=sources)
    cat["top_level"] = cat["Variable"].astype(str).str.split("|", n=1).str[0]
    pivot = (cat.groupby(["top_level", "source"]).size()
                .unstack(fill_value=0)
                .assign(total=lambda d: d.sum(axis=1))
                .sort_values("total", ascending=False))
    return pivot


def list_models(sources: list[str] | str | None = None) -> dict[str, list[str]]:
    data = load_ngfs(sources=sources)
    return {key: sorted(df["Model"].dropna().unique().tolist())
            for key, df in data.items()}


def list_scenarios(sources: list[str] | str | None = None) -> dict[str, list[str]]:
    data = load_ngfs(sources=sources)
    return {key: sorted(df["Scenario"].dropna().unique().tolist())
            for key, df in data.items()}


def list_regions(sources: list[str] | str | None = None) -> dict[str, list[str]]:
    data = load_ngfs(sources=sources)
    return {key: sorted(df["Region"].dropna().unique().tolist())
            for key, df in data.items()}


# ---------------------------------------------------------------------------
# Helper: prepare data for one variable
# ---------------------------------------------------------------------------

def get_series(
    variable: str,
    scope: str = "world",
    source: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """
    Récupère une variable précise, formatée long (Year, value).

    Parameters
    ----------
    variable : nom exact NGFS (ex: "Emissions|CO2", "Final Energy|Electricity|Solar")
    scope    : 'world' | 'europe'
    source   : None = toutes les sources où la variable existe
    scenario : filtre scénario (ex: "Net Zero 2050", "Delayed transition")
    model    : filtre modèle

    Returns
    -------
    pd.DataFrame (long) : Model, Scenario, Region, Variable, Unit, source, Year, value
    """
    data = load_ngfs(sources=source, scope=scope)
    out = []
    for src, df in data.items():
        sub = df[df["Variable"] == variable]
        if scenario:
            sub = sub[sub["Scenario"] == scenario]
        if model:
            sub = sub[sub["Model"] == model]
        if sub.empty:
            continue
        # Colonnes années = celles convertibles en int
        year_cols = [c for c in sub.columns if str(c).isdigit()]
        long = sub.melt(
            id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
            value_vars=year_cols,
            var_name="Year", value_name="value",
        )
        long["Year"] = long["Year"].astype(int)
        long["source"] = src
        out.append(long)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore NGFS Phase V data")
    parser.add_argument("--scope", choices=["world", "europe"], default="world")
    parser.add_argument("--sources", nargs="+", default=None,
                         help=f"Sources subset: {list(NGFS_FILES)}")
    parser.add_argument("--top-level", action="store_true",
                         help="Only show top-level variables (no '|').")
    parser.add_argument("--summary", action="store_true",
                         help="Show categories summary instead of full catalog.")
    parser.add_argument("--save-csv", default=None,
                         help="Save catalog to this CSV path.")
    parser.add_argument("--force-reload", action="store_true",
                         help="Bypass parquet cache.")
    args = parser.parse_args()

    print(f"Scope = {args.scope}, sources = {args.sources or 'all 5'}")

    if args.summary:
        print("\n=== Top-level categories (variables count per source) ===")
        pivot = list_top_categories(scope=args.scope, sources=args.sources)
        print(pivot.to_string())
    else:
        cat = list_variables(scope=args.scope, sources=args.sources,
                              only_top_level=args.top_level)
        print(f"\n=== Catalog ({len(cat)} (variable, source) pairs) ===")
        with pd.option_context("display.max_rows", 80,
                                "display.max_colwidth", 80,
                                "display.width", 200):
            print(cat.to_string(index=False))

        # Resume par source
        print("\n--- Resume par source ---")
        for src, n in cat.groupby("source").size().items():
            print(f"  {src:12s} : {n:>5} variables")

    if args.save_csv:
        cat_save = list_variables(scope=args.scope, sources=args.sources,
                                   only_top_level=args.top_level)
        cat_save.to_csv(args.save_csv, index=False)
        print(f"\nSaved: {args.save_csv}")
