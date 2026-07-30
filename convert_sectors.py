import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive: no GUI window, safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONCORDANCE_PATH = Path(__file__).parent / "Data" / "mappings" / "exiobase_concordance.csv"


def load_concordance() -> pd.DataFrame:
    return pd.read_csv(CONCORDANCE_PATH, dtype=str)


def _normalise(s: str) -> str:
    return s.replace(",", "").replace(";", "").replace("  ", " ").lower().strip()


def _build_lookup(concordance: pd.DataFrame, target_col: str) -> tuple[dict, dict]:
    """Return (exact_lookup, normalised_lookup)."""
    lookup = dict(zip(concordance["exiobase_sector"], concordance[target_col]))
    norm_lookup = {_normalise(k): v for k, v in lookup.items()}
    return lookup, norm_lookup


def _map_sector(sector: str, lookup: dict, norm_lookup: dict) -> str:
    if sector in lookup:
        return lookup[sector]
    n = _normalise(sector)
    if n in norm_lookup:
        return norm_lookup[n]
    for key in sorted(norm_lookup, key=len, reverse=True):
        if key in n or n in key:
            return norm_lookup[key]
    return "UNKNOWN"


def remap_Z(
    Z: pd.DataFrame,
    target: str = "nace_rev2",
    agg: str = "sum",
    concordance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Remap the EXIOBASE Z matrix from product sectors to NACE Rev 2 or GICS,
    keeping the region level (result MultiIndex: region × target).

    Parameters
    ----------
    Z : pd.DataFrame
        Z matrix with MultiIndex (region, sector) on rows and columns.
    target : str
        'nace_rev2' | 'nace_name' | 'gics_code' | 'gics_name'
    agg : str
        'sum' (default) or 'mean'.
    concordance : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame with MultiIndex (region, target) on both axes.
    """
    valid = {"nace_rev2", "nace_name", "gics_code", "gics_name"}
    if target not in valid:
        raise ValueError(f"target must be one of {valid}")

    if concordance is None:
        concordance = load_concordance()

    lookup, norm_lookup = _build_lookup(concordance, target)

    def remap_index(idx: pd.MultiIndex) -> pd.MultiIndex:
        regions = idx.get_level_values("region")
        new_sectors = [_map_sector(s, lookup, norm_lookup) for s in idx.get_level_values("sector")]
        return pd.MultiIndex.from_arrays([regions, new_sectors], names=["region", target])

    Z_out = Z.copy()
    Z_out.index = remap_index(Z.index)
    Z_out.columns = remap_index(Z.columns)

    Z_out = getattr(Z_out.groupby(level=[0, 1]), agg)()
    Z_out = getattr(Z_out.T.groupby(level=[0, 1]), agg)().T
    return Z_out


def aggregate_to_sector_matrix(
    Z: pd.DataFrame,
    target: str = "gics_name",
    concordance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Directly aggregate Z to a pure sector×sector matrix (no region level).
    Much faster than remap_Z + aggregate_regions: no intermediate copy of the
    full 9800×9800 matrix, single flat-index groupby on each axis.

    Parameters
    ----------
    Z : pd.DataFrame
        Z matrix with MultiIndex (region, sector) on rows and columns.
    target : str
        'nace_rev2' | 'nace_name' | 'gics_code' | 'gics_name'
    concordance : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame  — square, indexed by target labels (e.g. 11×11 for GICS name).
    """
    if concordance is None:
        concordance = load_concordance()

    lookup, norm_lookup = _build_lookup(concordance, target)

    row_labels = [_map_sector(s, lookup, norm_lookup) for s in Z.index.get_level_values("sector")]
    col_labels = [_map_sector(s, lookup, norm_lookup) for s in Z.columns.get_level_values("sector")]

    # Use .values view (no data copy) with flat string labels
    Z_flat = pd.DataFrame(Z.values, index=row_labels, columns=col_labels)
    Z_flat = Z_flat.groupby(level=0).sum()
    Z_flat = Z_flat.T.groupby(level=0).sum().T
    return Z_flat


def aggregate_to_sector_vector(
    data,
    target: str = "gics_name",
    concordance: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Aggregate a (region, sector)-indexed vector to a pure sector vector (GICS),
    the 1-D analogue of aggregate_to_sector_matrix. Used for x (total output) and
    Y (final demand).

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Object with MultiIndex (region, sector) on rows. If a DataFrame, all
        columns are summed first (e.g. Y's region×category columns -> total
        final demand per row; x's single 'indout' column -> itself).
    target : str
        'nace_rev2' | 'nace_name' | 'gics_code' | 'gics_name'
    concordance : pd.DataFrame, optional

    Returns
    -------
    pd.Series indexed by the target sector labels (e.g. 11 GICS names).
    """
    if concordance is None:
        concordance = load_concordance()

    values = data.sum(axis=1) if isinstance(data, pd.DataFrame) else data
    lookup, norm_lookup = _build_lookup(concordance, target)
    labels = [_map_sector(s, lookup, norm_lookup) for s in values.index.get_level_values("sector")]
    return pd.Series(np.asarray(values.values).ravel(), index=labels).groupby(level=0).sum()


def aggregate_regions(Z_remapped: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the region level of a remap_Z output to get a pure sector×sector matrix.

    Parameters
    ----------
    Z_remapped : pd.DataFrame
        Output of remap_Z — MultiIndex (region, target) on both axes.

    Returns
    -------
    pd.DataFrame — square, indexed by target labels only.
    """
    sector_level = Z_remapped.index.names[1]
    Z_out = Z_remapped.groupby(level=sector_level).sum()
    Z_out = Z_out.T.groupby(level=sector_level).sum().T
    return Z_out


def plot_heatmap(
    M: pd.DataFrame,
    title: str = "Inter-sector flows (M€)",
    log_scale: bool = True,
    figsize: tuple = (11, 9),
    cmap: str = "YlOrRd",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a sector×sector matrix as a heatmap.

    Parameters
    ----------
    M : pd.DataFrame
        Square sector×sector matrix.
    title : str
        Figure title.
    log_scale : bool
        Apply log10 transform (recommended — Z values span several orders of magnitude).
    figsize : tuple
        Figure size in inches.
    cmap : str
        Matplotlib colormap.
    save_path : str, optional
        Save the figure to this path if provided.

    Returns
    -------
    matplotlib Figure
    """
    data = M.values.astype(float)
    labels = list(M.index)
    n = len(labels)

    if log_scale:
        display = np.log10(np.where(data > 0, data, np.nan))
        cbar_label = "log₁₀(M€)"
    else:
        display = data
        cbar_label = "M€"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display, cmap=cmap, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    if log_scale:
        vmin, vmax = np.nanmin(display), np.nanmax(display)
        tick_powers = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
        cbar.set_ticks(tick_powers)
        cbar.set_ticklabels([f"$10^{{{int(p)}}}$" for p in tick_powers])

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("Destination sector", fontsize=10)
    ax.set_ylabel("Origin sector", fontsize=10)

    # Annotate cells with the actual value
    if n <= 15:
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                if v > 0:
                    txt = f"{v/1e6:.1f}T" if v >= 1e6 else f"{v/1e3:.0f}B" if v >= 1e3 else f"{v:.0f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color="white" if display[i, j] > (np.nanmax(display) * 0.6) else "black")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def check_coverage(Z: pd.DataFrame, concordance: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a DataFrame showing which EXIOBASE sectors matched or were missed."""
    if concordance is None:
        concordance = load_concordance()
    lookup, norm_lookup = _build_lookup(concordance, "nace_rev2")
    sectors = Z.index.get_level_values("sector").unique()
    rows = [
        {"exiobase_sector": s, "matched": _map_sector(s, lookup, norm_lookup) != "UNKNOWN",
         "nace_rev2": _map_sector(s, lookup, norm_lookup)}
        for s in sectors
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from read_Z import read_Z

    print("Reading Z (2019 pxp)...")
    Z = read_Z(year=2019, table_type="pxp")
    print(f"  Raw shape: {Z.shape}")

    print("Aggregating to GICS 11×11...")
    Z_11 = aggregate_to_sector_matrix(Z, target="gics_name")
    print(f"  Done. Shape: {Z_11.shape}")
    print(Z_11.round(0).to_string())

    print("\nPlotting heatmap...")
    plot_heatmap(
        Z_11,
        title="Global inter-sector intermediate flows — GICS 2019 (M€, log scale)",
        log_scale=True,
        save_path=str(Path(__file__).parent / "Figs" / "plots" / "Z_gics_heatmap.png"),
    )
    print("Done.")
