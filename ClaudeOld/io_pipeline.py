"""
EXIOBASE IO pipeline : Z, x, Y, F, F_Y → A, S, L (optionnel), empreinte.

Formules clés :
    A = Z · x̂⁻¹                           coefficients techniques
    S = F · x̂⁻¹                           intensités stresseurs
    x = (I − A)⁻¹ · y  =  L · y           modèle de Leontief
    footprint = S · L · Y + F_Y            empreinte par catégorie de demande

Note : L (9800×9800) n'est jamais calculée explicitement.
On résout (I − A) · Q = Y via décomposition LU sparse (scipy),
ce qui est équivalent mais évite l'inversion et l'allocation mémoire.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ZIP_PATH = Path(__file__).parent / "EXIOBASE_data.zip"

# ---------------------------------------------------------------------------
# 1. Chargement des matrices depuis les zips imbriqués
# ---------------------------------------------------------------------------

def _open_inner(year: int, table_type: str, zip_path: Path = ZIP_PATH):
    """Context manager helper — yields the open inner ZipFile."""
    inner_name = f"IOT_{year}_{table_type}.zip"
    outer = zipfile.ZipFile(zip_path)
    raw = outer.open(inner_name).read()
    return zipfile.ZipFile(io.BytesIO(raw))


def load_io_system(
    year: int,
    table_type: str = "pxp",
    zip_path: Path = ZIP_PATH,
) -> dict[str, pd.DataFrame]:
    """
    Load Z, x, Y from one EXIOBASE IOT zip.

    Returns
    -------
    dict with keys: 'Z', 'x', 'Y'
        Z : (n×n) DataFrame, MultiIndex (region, sector)
        x : (n,) Series,    MultiIndex (region, sector)
        Y : (n×f) DataFrame, MultiIndex rows / MultiIndex columns (region, category)
    """
    inner = _open_inner(year, table_type, zip_path)

    with inner.open("Z.txt") as fh:
        Z = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=[0, 1])

    with inner.open("x.txt") as fh:
        x = pd.read_csv(fh, sep="\t", header=0, index_col=[0, 1]).squeeze("columns")

    with inner.open("Y.txt") as fh:
        Y = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=[0, 1])

    return {"Z": Z, "x": x, "Y": Y}


def load_x(
    year: int,
    table_type: str = "pxp",
    zip_path: Path = ZIP_PATH,
) -> pd.Series:
    """
    Charge uniquement x.txt (production totale par secteur).

    Raccourci de `load_io_system` quand seul x est nécessaire : évite la
    lecture de Z.txt (9800×9800, ~700 Mo de texte) et de Y.txt.

    Returns
    -------
    pd.Series (n,) — MultiIndex (region, sector), en M.EUR
    """
    inner = _open_inner(year, table_type, zip_path)
    with inner.open("x.txt") as fh:
        return pd.read_csv(fh, sep="\t", header=0, index_col=[0, 1]).squeeze("columns")


def load_extension(
    year: int,
    extension: str,
    table_type: str = "pxp",
    zip_path: Path = ZIP_PATH,
) -> dict[str, pd.DataFrame]:
    """
    Load F, F_Y (and unit) for a given extension satellite account.

    Parameters
    ----------
    extension : str
        One of: 'air_emissions', 'energy', 'water', 'material',
                'nutrients', 'land', 'labour', 'factor_inputs'

    Returns
    -------
    dict with keys: 'F', 'F_Y', 'unit'
        F   : (e×n) DataFrame — stresseurs liés à la production
        F_Y : (e×f) DataFrame — stresseurs liés à la demande finale
        unit: (e,) Series
    """
    inner = _open_inner(year, table_type, zip_path)
    prefix = f"{extension}/"

    with inner.open(prefix + "F.txt") as fh:
        F = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=0).fillna(0.0)

    with inner.open(prefix + "F_Y.txt") as fh:
        F_Y = pd.read_csv(fh, sep="\t", header=[0, 1], index_col=0).fillna(0.0)

    with inner.open(prefix + "unit.txt") as fh:
        unit = pd.read_csv(fh, sep="\t", header=0, index_col=0).squeeze("columns")

    return {"F": F, "F_Y": F_Y, "unit": unit}


# ---------------------------------------------------------------------------
# 2. Calculs matriciels IO
# ---------------------------------------------------------------------------

def _safe_inv(x_values: np.ndarray) -> np.ndarray:
    """1/x avec protection contre les zéros (évite le RuntimeWarning numpy)."""
    denom = np.where(x_values > 0, x_values, 1.0)  # remplace 0 par 1 avant la division
    return np.where(x_values > 0, 1.0 / denom, 0.0)


def compute_A(Z: pd.DataFrame, x: pd.Series) -> pd.DataFrame:
    """
    Matrice des coefficients techniques : A = Z · x̂⁻¹

    Les secteurs à production nulle reçoivent une colonne zéro.
    """
    x_inv = _safe_inv(x.values)
    A_values = Z.values * x_inv[np.newaxis, :]   # broadcast : divise chaque colonne
    return pd.DataFrame(A_values, index=Z.index, columns=Z.columns)


def compute_S(F: pd.DataFrame, x: pd.Series) -> pd.DataFrame:
    """
    Intensités stresseurs : S = F · x̂⁻¹

    S[k, i] = émission k par unité monétaire produite par i.
    """
    x_inv = _safe_inv(x.values)
    F_aligned = F.reindex(columns=x.index, fill_value=0.0)
    S_values = F_aligned.values * x_inv[np.newaxis, :]
    return pd.DataFrame(S_values, index=F.index, columns=x.index)


def _build_leontief_sparse(A: pd.DataFrame):
    """
    Construit (I − A) en format sparse CSC et retourne l'objet LU.

    La décomposition LU est faite une seule fois.
    lu.solve(b)         → (I−A)⁻¹ · b   (système direct)
    lu.solve(b, trans='T') → (I−A)⁻ᵀ · b  (système transposé, pour les multiplicateurs)

    Returns
    -------
    lu    : SuperLU object (scipy)
    IminA : csc_matrix
    """
    n = A.shape[0]
    IminA = sp.eye(n, format="csc") - sp.csc_matrix(A.values)
    lu = spla.splu(IminA)
    return lu, IminA


def compute_footprint(
    A: pd.DataFrame,
    S: pd.DataFrame,
    Y: pd.DataFrame,
    F_Y: pd.DataFrame,
) -> pd.DataFrame:
    """
    Empreinte par stresseur et par catégorie de demande finale.

        footprint = S · (I−A)⁻¹ · Y + F_Y

    La résolution (I−A)⁻¹ · Y est faite via LU sparse (scipy) :
    pas d'inversion explicite de L, une seule factorisation pour toutes
    les colonnes de Y.

    Parameters
    ----------
    A   : (n×n) coefficients techniques
    S   : (e×n) intensités stresseurs
    Y   : (n×f) demande finale
    F_Y : (e×f) stresseurs directs de la demande finale

    Returns
    -------
    pd.DataFrame (e×f) — même index que S (stresseurs),
                          mêmes colonnes que Y (région × catégorie)
    """
    lu, _ = _build_leontief_sparse(A)

    # (I−A)⁻¹ · Y  : résoudre pour chaque colonne de Y
    Y_arr = Y.reindex(index=A.index, fill_value=0.0).values      # (n×f)
    Q = lu.solve(Y_arr)                                            # (n×f)

    # S · Q  : (e×n) · (n×f) = (e×f)
    indirect = S.values @ Q

    # F_Y alignée sur les colonnes de Y
    F_Y_aligned = F_Y.reindex(columns=Y.columns, fill_value=0.0).values

    fp_values = indirect + F_Y_aligned
    return pd.DataFrame(fp_values, index=S.index, columns=Y.columns)


def compute_multipliers(
    A: pd.DataFrame,
    S: pd.DataFrame,
    stressor: str,
) -> pd.Series:
    """
    Multiplicateurs de Leontief pour un stresseur donné.

        m = S[stressor] · (I−A)⁻¹

    m[j] = émission totale (directe + indirecte) générée par 1€
           de demande finale adressée au secteur j.

    Equivalent à une ligne de M = S · L, sans calculer L entier.

    Parameters
    ----------
    stressor : str  — nom exact de la ligne dans S (ex: 'CO2 - combustion - air')

    Returns
    -------
    pd.Series (n,) indexée comme les colonnes de A
    """
    s_vec = S.loc[stressor].reindex(A.index, fill_value=0.0).values   # (n,)

    lu, _ = _build_leontief_sparse(A)
    # m = s · L  ↔  mᵀ = Lᵀ · sᵀ  ↔  (I−A)ᵀ · mᵀ = sᵀ
    # scipy splu supporte trans='T' : une seule factorisation suffit
    m = lu.solve(s_vec, trans="T")   # (n,)

    return pd.Series(m, index=A.index, name=stressor)


def compute_L(A: pd.DataFrame) -> pd.DataFrame:
    """
    Inverse de Leontief explicite : L = (I−A)⁻¹

    ATTENTION : matrice dense n×n (~768 MB pour n=9800).
    Utiliser compute_footprint ou compute_multipliers si possible.
    Pratique seulement sur des matrices agrégées (ex: 11×11 GICS).

    Returns
    -------
    pd.DataFrame (n×n)
    """
    n = A.shape[0]
    if n > 500:
        print(f"[Warning] L sera une matrice dense {n}×{n} (~{n**2*8/1e9:.1f} GB). "
              "Préférer compute_footprint() pour les grandes matrices.")
    IminA = np.eye(n) - A.values
    L_values = np.linalg.inv(IminA)
    return pd.DataFrame(L_values, index=A.index, columns=A.columns)


# ---------------------------------------------------------------------------
# 3. Pipeline complet en une fonction
# ---------------------------------------------------------------------------

def run_pipeline(
    year: int = 2019,
    table_type: str = "pxp",
    extension: str = "air_emissions",
    stressor_filter: list[str] | None = None,
    zip_path: Path = ZIP_PATH,
) -> dict:
    """
    Pipeline complet Z, F → empreinte.

    Parameters
    ----------
    stressor_filter : list[str], optional
        Sous-ensemble de stresseurs à garder (ex: ['CO2 - combustion - air']).
        Si None, conserve tous les stresseurs.

    Returns
    -------
    dict avec les clés :
        'A'           : coefficients techniques  (n×n)
        'S'           : intensités stresseurs    (e×n)
        'footprint'   : empreinte par catégorie  (e×f)
        'x', 'Y', 'Z', 'F', 'F_Y', 'unit'
    """
    print(f"[1/4] Chargement IO system {year} {table_type}...")
    io_sys = load_io_system(year, table_type, zip_path)
    Z, x, Y = io_sys["Z"], io_sys["x"], io_sys["Y"]

    print(f"[2/4] Chargement extension '{extension}'...")
    ext = load_extension(year, extension, table_type, zip_path)
    F, F_Y = ext["F"], ext["F_Y"]

    if stressor_filter:
        F   = F.loc[F.index.intersection(stressor_filter)]
        F_Y = F_Y.loc[F_Y.index.intersection(stressor_filter)]

    print("[3/4] Calcul A et S...")
    A = compute_A(Z, x)
    S = compute_S(F, x)

    print("[4/4] Calcul empreinte (LU sparse)...")
    footprint = compute_footprint(A, S, Y, F_Y)

    print(f"  Done. Empreinte shape: {footprint.shape}")
    return {
        "A": A, "S": S, "footprint": footprint,
        "Z": Z, "x": x, "Y": Y,
        "F": F, "F_Y": F_Y, "unit": ext["unit"],
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CO2_STRESSORS = [
        "CO2 - combustion - air",
        "CO2 - non combustion - Cement production - air",
        "CO2 - non combustion - Lime production - air",
        "CO2_bio - combustion - air",
    ]

    result = run_pipeline(
        year=2019,
        table_type="pxp",
        extension="air_emissions",
        stressor_filter=CO2_STRESSORS,
    )

    fp = result["footprint"]
    unit = result["unit"]

    # Empreinte CO2 totale par pays (somme stresseurs x somme catégories de demande)
    fp_by_country = (
        fp.sum(axis=0)                          # somme des stresseurs -> (f,) Series
          .groupby(level=0).sum()               # somme des catégories par pays -> (49,)
    )
    fp_by_country.name = f"CO2 footprint ({unit.iloc[0]})"
    print("\nEmpreinte CO2 totale par pays (top 10) :")
    print(fp_by_country.sort_values(ascending=False).head(10).to_string())

    # Multiplicateur CO2 combustion — top 10 secteurs
    print("\nCalcul des multiplicateurs CO2 combustion...")
    A = result["A"]
    S = result["S"]
    stressor = "CO2 - combustion - air"
    m = compute_multipliers(A, S, stressor)
    top = m.sort_values(ascending=False).head(10)
    print(f"\nTop 10 secteurs par multiplicateur '{stressor}' :")
    print(top.to_string())
