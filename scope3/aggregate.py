"""
World-aggregated EXIOBASE system, at a choice of sector granularity.

Everything downstream (scope 1/2/3, the Leontief price model, the KRAS
decarbonisation scenarios) needs one square sector x sector matrix, not the
9800 x 9800 multi-region table: KRAS balances n^2 variables, so the region
axis has to go. Regions are summed away, which keeps international supply
chains *inside* the flows but stops distinguishing them.

Two granularities, both driven by mappings that already exist in the repo:

    "gics"  11 sectors  -- Data/mappings/exiobase_concordance.csv, via the
                           fuzzy `_map_sector` of convert_sectors.py
    "ngfs"  36 sectors  -- `exio_to_ngfs()` of shortterm/st_exiobase_mapping.py,
                           the partition validated at 100 % output coverage.
                           50 GEM-E3 sectors exist but 14 are `overlap` sectors
                           with no EXIOBASE product of their own, so 36 appear.

A is aggregated *then* divided
------------------------------
    A_agg = Z_agg / x_agg          NOT  mean(A) over the members of a group

The second is not a technical coefficient of anything. Same convention as
optim/exioptim.py:122.

Cache
-----
Z.txt is ~700 MB of text and takes minutes to parse, while the figures want the
aggregated system many times over. The aggregate is written to
Data/processed/scope3_cache/{granularity}_{year}_{table_type}.npz and reused.
Delete the file to force a rebuild, or pass `use_cache=False`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convert_sectors import _build_lookup, _map_sector, load_concordance  # noqa: E402
from io_pipeline import load_extension, load_io_system  # noqa: E402
from shortterm.st_exiobase_mapping import MAPPING, exio_to_ngfs  # noqa: E402

CACHE_DIR = REPO_ROOT / "Data" / "processed" / "scope3_cache"

#: The CO2 stressors of io_pipeline.py's __main__, minus the biogenic one.
#: CO2_bio is excluded on purpose: it is biogenic carbon, conventionally
#: reported outside the scopes and outside carbon pricing.
CO2_STRESSORS = [
    "CO2 - combustion - air",
    "CO2 - non combustion - Cement production - air",
    "CO2 - non combustion - Lime production - air",
]

GRANULARITIES = ("gics", "ngfs")


# ---------------------------------------------------------------------------
# Sector labels
# ---------------------------------------------------------------------------

def map_sectors(sectors, granularity: str, table_type: str = "pxp") -> list[str]:
    """EXIOBASE sector names -> aggregated labels, one per input sector."""
    if granularity == "ngfs":
        lookup = exio_to_ngfs(table_type)
        missing = sorted({s for s in sectors if s not in lookup})
        if missing:
            raise KeyError(f"{len(missing)} EXIOBASE sectors absent from the NGFS "
                           f"partition, e.g. {missing[:3]}")
        return [lookup[s] for s in sectors]

    if granularity == "gics":
        exact, norm = _build_lookup(load_concordance(), "gics_name")
        labels = [_map_sector(s, exact, norm) for s in sectors]
        if "UNKNOWN" in labels:
            n = labels.count("UNKNOWN")
            raise KeyError(f"{n} EXIOBASE sectors did not match the GICS concordance")
        return labels

    raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")


def sector_order(granularity: str, table_type: str = "pxp") -> list[str]:
    """
    Stable display order for the aggregated sectors.

    NGFS follows the order of `MAPPING`, which groups by theme (Energy supply,
    Power generation, Heavy industry, ...) -- so a bar chart reads as blocks
    rather than as an alphabet. GICS has no theme, so alphabetical.
    """
    if granularity == "ngfs":
        present = set(exio_to_ngfs(table_type).values())
        return [m.ngfs_sector for m in MAPPING if m.ngfs_sector in present]
    return sorted(set(load_concordance()["gics_name"]))


def sector_themes(granularity: str, table_type: str = "pxp") -> dict[str, str]:
    """Aggregated sector -> theme. GICS sectors are their own theme."""
    if granularity == "ngfs":
        return {m.ngfs_sector: m.theme for m in MAPPING}
    return {s: s for s in sector_order("gics", table_type)}


#: Sectors whose *direct* emissions are the scope 2 of everyone who buys from
#: them. Electricity generation and the grid; heat is bundled into Power Supply
#: in the NGFS partition. Listed explicitly rather than pattern-matched -- a
#: regex on "power" would also catch "Power Supply equipment"-style labels.
def power_sectors(granularity: str, table_type: str = "pxp") -> list[str]:
    if granularity == "ngfs":
        present = set(exio_to_ngfs(table_type).values())
        return [m.ngfs_sector for m in MAPPING
                if m.theme == "Power generation" and m.ngfs_sector in present]
    return ["Utilities"]


# ---------------------------------------------------------------------------
# Generic aggregation
# ---------------------------------------------------------------------------

def aggregate_matrix(M: pd.DataFrame, labels: list[str],
                     order: list[str]) -> pd.DataFrame:
    """
    Sum a (region, sector) x (region, sector) matrix into sector x sector.

    Uses the flat-index groupby of convert_sectors.aggregate_to_sector_matrix:
    a .values view with string labels, so the 9800^2 block is never copied.
    """
    flat = pd.DataFrame(M.values, index=labels, columns=labels)
    flat = flat.groupby(level=0).sum()
    flat = flat.T.groupby(level=0).sum().T
    return flat.reindex(index=order, columns=order).fillna(0.0)


def aggregate_vector(v, labels: list[str], order: list[str]) -> pd.Series:
    """Sum a (region, sector)-indexed vector (or row-summed frame) by label."""
    values = v.sum(axis=1) if isinstance(v, pd.DataFrame) else v
    out = pd.Series(np.asarray(values.values, dtype=float).ravel(), index=labels)
    return out.groupby(level=0).sum().reindex(order).fillna(0.0)


# ---------------------------------------------------------------------------
# The system
# ---------------------------------------------------------------------------

@dataclass
class WorldSystem:
    """
    One world sector x sector IO system with a CO2 extension.

    All arrays are plain numpy in the order of `sectors`.

        Z  (n,n)  intermediate flows, M.EUR
        x  (n,)   total output, M.EUR
        y  (n,)   final demand, M.EUR
        F  (n,)   direct CO2 emissions, kg
    """
    sectors: list[str]
    Z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    F: np.ndarray
    year: int
    granularity: str
    table_type: str

    @property
    def n(self) -> int:
        return len(self.sectors)

    @property
    def A(self) -> np.ndarray:
        """Technical coefficients, Z / x, with zero-output columns zeroed."""
        inv = np.where(self.x > 0, 1.0 / np.where(self.x > 0, self.x, 1.0), 0.0)
        return self.Z * inv[None, :]

    @property
    def s(self) -> np.ndarray:
        """Direct CO2 intensity, kg per M.EUR of output."""
        inv = np.where(self.x > 0, 1.0 / np.where(self.x > 0, self.x, 1.0), 0.0)
        return self.F * inv

    @property
    def L(self) -> np.ndarray:
        """Leontief inverse. Dense is fine here: n is 11 or 36."""
        return np.linalg.inv(np.eye(self.n) - self.A)

    @property
    def m(self) -> np.ndarray:
        """Cradle-to-gate CO2 multiplier, m = s L, kg per M.EUR of final demand."""
        return self.s @ self.L

    def frame(self) -> pd.DataFrame:
        """The vectors as a labelled table, for printing and for the charts."""
        return pd.DataFrame(
            {"x": self.x, "y": self.y, "F": self.F, "s": self.s, "m": self.m},
            index=pd.Index(self.sectors, name="sector"),
        )

    def leontief_residual(self) -> float:
        """
        Max relative gap between (I-A)^-1 y and the observed x.

        A world table balances by construction, so this is ~1e-15. It is the
        control that the aggregation did not lose or duplicate flows -- the
        same check as optim/exioptim.py:244.
        """
        x_hat = np.linalg.solve(np.eye(self.n) - self.A, self.y)
        keep = self.x > 0
        return float(np.max(np.abs(x_hat[keep] - self.x[keep]) / self.x[keep]))


def _cache_path(year: int, granularity: str, table_type: str) -> Path:
    return CACHE_DIR / f"{granularity}_{year}_{table_type}.npz"


def build_world_system(
    year: int = 2019,
    granularity: str = "ngfs",
    table_type: str = "pxp",
    use_cache: bool = True,
    verbose: bool = True,
) -> WorldSystem:
    """
    Load EXIOBASE, aggregate regions away, return the world sector system.

    First call parses Z.txt (slow, minutes); later calls hit the npz cache.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")

    path = _cache_path(year, granularity, table_type)
    if use_cache and path.exists():
        blob = np.load(path, allow_pickle=False)
        if verbose:
            print(f"  cache hit: {path.name}")
        return WorldSystem(
            sectors=[str(s) for s in blob["sectors"]],
            Z=blob["Z"], x=blob["x"], y=blob["y"], F=blob["F"],
            year=year, granularity=granularity, table_type=table_type,
        )

    if verbose:
        print(f"  building {granularity} {year} {table_type} from EXIOBASE "
              f"(parsing Z.txt, this takes a few minutes)...")
    io_sys = load_io_system(year, table_type)
    Z, x_raw, Y = io_sys["Z"], io_sys["x"], io_sys["Y"]
    ext = load_extension(year, "air_emissions", table_type)
    F_raw = ext["F"]

    sectors_in = list(Z.index.get_level_values("sector"))
    labels = map_sectors(sectors_in, granularity, table_type)
    order = sector_order(granularity, table_type)

    present = set(labels)
    order = [s for s in order if s in present]

    Z_agg = aggregate_matrix(Z, labels, order)
    x_agg = aggregate_vector(x_raw, labels, order)
    y_agg = aggregate_vector(Y, labels, order)

    co2 = F_raw.reindex(index=CO2_STRESSORS).fillna(0.0).sum(axis=0)
    co2 = co2.reindex(Z.columns).fillna(0.0)
    F_agg = aggregate_vector(co2, labels, order)

    system = WorldSystem(
        sectors=order, Z=Z_agg.to_numpy(dtype=float),
        x=x_agg.to_numpy(dtype=float), y=y_agg.to_numpy(dtype=float),
        F=F_agg.to_numpy(dtype=float),
        year=year, granularity=granularity, table_type=table_type,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sectors=np.array(order, dtype=object).astype(str),
                        Z=system.Z, x=system.x, y=system.y, F=system.F)
    if verbose:
        print(f"  cached: {path}")
    return system


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the world-aggregated system.")
    ap.add_argument("--granularity", choices=GRANULARITIES, default="ngfs")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--table-type", choices=["pxp", "ixi"], default="pxp")
    ap.add_argument("--rebuild", action="store_true", help="Ignore the cache.")
    args = ap.parse_args()

    sysm = build_world_system(args.year, args.granularity, args.table_type,
                              use_cache=not args.rebuild)
    print(f"\n{sysm.granularity} {sysm.year} {sysm.table_type}: n = {sysm.n} sectors")
    print(f"  total output      {sysm.x.sum():.4e} M.EUR")
    print(f"  total final dem.  {sysm.y.sum():.4e} M.EUR")
    print(f"  total direct CO2  {sysm.F.sum() / 1e9:.1f} Mt")
    print(f"  Leontief residual {sysm.leontief_residual():.2e}")
    print()
    print(sysm.frame().sort_values("m", ascending=False).head(12).to_string(
        float_format=lambda v: f"{v:.4g}"))
