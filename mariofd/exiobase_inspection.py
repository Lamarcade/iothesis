r"""Import EXIOBASE with MARIO and run basic inspections + calculations.

Usage (PowerShell):
    python exiobase_inspection.py                     # auto: local -> download -> load_test
    python exiobase_inspection.py --path D:\exio\IOT_2019_ixi
    python exiobase_inspection.py --download --year 2019 --system ixi

Docs:
    https://mario-suite.readthedocs.io/en/latest/notebooks/parsers/exiobase/monetary.html
    https://mario-suite.readthedocs.io/en/latest/user_guide/inspection/basic_inspections.html
"""
from __future__ import annotations

import argparse
import csv
import difflib
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mario
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CACHE = REPO_ROOT / "EXIOBASE_data"
DEFAULT_ZIP = REPO_ROOT / "EXIOBASE_data.zip"
SHOCK_DIR = HERE / "shock"
SECTOR_CLUSTERS_FILE = REPO_ROOT / "Data" / "docs" / "EXIOBASE20i_7sectors.txt"


def parse_sector_clusters(path: Path) -> Dict[str, List[str]]:
    """Parse the EXIOBASE20i_7sectors.txt correspondence into
    {cluster_name: [EXIOBASE sector labels]}.

    The file is tab-separated; sector names containing commas are quoted with
    double-quotes, which csv.reader handles natively.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        cluster_names = [c.strip() for c in header[1:] if c.strip()]
        clusters: Dict[str, List[str]] = {name: [] for name in cluster_names}
        for row in reader:
            if not row or not row[0].strip():
                continue
            label = row[0].strip()
            flags = row[1:]
            ones = [i for i, v in enumerate(flags) if v.strip() == "1"]
            if len(ones) != 1:
                raise ValueError(
                    f"Row {label!r} has {len(ones)} '1' flags (expected exactly 1)"
                )
            clusters[cluster_names[ones[0]]].append(label)
    return clusters


def build_mario_clusters(by_cluster: Dict[str, List[str]],
                         dimension: str = "Sector") -> Dict[str, Dict[str, List[str]]]:
    """Wrap {cluster: [labels]} in the MARIO clusters structure expected by
    `db.set_clusters` and `db.get_shock_excel(clusters=...)`."""
    return {dimension: dict(by_cluster)}


def report_cluster_coverage(by_cluster: Dict[str, List[str]], db) -> None:
    """Print which cluster labels do/don't match the loaded db's Sector index."""
    try:
        sectors = set(db.get_index("Sector"))
    except Exception as e:
        print(f"[clusters] could not read sector index: {e}")
        return
    total = sum(len(v) for v in by_cluster.values())
    matched = 0
    print(f"[clusters] DB has {len(sectors)} sectors; correspondence has {total} labels")
    for name, labels in by_cluster.items():
        ok = [s for s in labels if s in sectors]
        ko = [s for s in labels if s not in sectors]
        matched += len(ok)
        print(f"  {name:<28s} matched={len(ok):>3d}/{len(labels):<3d}  missing={len(ko)}")
        if ko and len(ko) <= 5:
            print(f"    e.g. missing: {ko}")
    print(f"[clusters] total matched: {matched}/{total}")


def reconcile_e2_to_e3(by_cluster: Dict[str, List[str]], db,
                       auto_cutoff: float = 0.65,
                       suggest_cutoff: float = 0.45
                       ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Try to recover EXIOBASE 2 sector labels missing from EXIOBASE 3 by
    fuzzy-matching them against the DB's actual sector index.

    Returns (renamed_clusters, applied_renames):
      - renamed_clusters: a copy of `by_cluster` with confidently-matched
        labels (similarity >= auto_cutoff) replaced by their E3 equivalent.
      - applied_renames: {e2_label: e3_label} for what was auto-rewritten.

    Anything below `auto_cutoff` is left as-is (and will be dropped by
    filter_clusters_to_db) but suggestions are printed for manual review.
    """
    sectors = set(db.get_index("Sector"))
    used = {s for labels in by_cluster.values() for s in labels if s in sectors}
    e3_pool = sorted(sectors - used)  # candidates not yet claimed by E2 names

    applied: Dict[str, str] = {}
    renamed: Dict[str, List[str]] = {}
    print("\n[clusters] E2 -> E3 fuzzy reconciliation")
    for cluster_name, labels in by_cluster.items():
        new_labels: List[str] = []
        for label in labels:
            if label in sectors:
                new_labels.append(label)
                continue
            matches = difflib.get_close_matches(
                label, e3_pool, n=3, cutoff=suggest_cutoff,
            )
            best_score = (
                difflib.SequenceMatcher(None, label, matches[0]).ratio()
                if matches else 0.0
            )
            if matches and best_score >= auto_cutoff:
                chosen = matches[0]
                applied[label] = chosen
                e3_pool.remove(chosen)  # don't reuse for another E2 label
                new_labels.append(chosen)
                print(f"  [{cluster_name}] auto: {label!r}")
                print(f"     -> {chosen!r} (score={best_score:.2f})")
            else:
                # Keep the E2 label so it gets dropped (and surfaced) downstream.
                new_labels.append(label)
                print(f"  [{cluster_name}] unresolved: {label!r}")
                for m in matches:
                    score = difflib.SequenceMatcher(None, label, m).ratio()
                    print(f"     candidate: {m!r} (score={score:.2f})")
                if not matches:
                    print("     (no candidate above suggest_cutoff)")
        renamed[cluster_name] = new_labels

    if applied:
        print(f"[clusters] auto-applied {len(applied)} renames")
    return renamed, applied


def filter_clusters_to_db(by_cluster: Dict[str, List[str]], db
                          ) -> Dict[str, List[str]]:
    """Drop labels not present in db.get_index('Sector') so MARIO accepts the
    cluster dict. Returns a new dict; original is untouched. Logs the dropped
    labels (they likely come from EXIOBASE 2 -> 3 renamings)."""
    sectors = set(db.get_index("Sector"))
    filtered: Dict[str, List[str]] = {}
    dropped: List[str] = []
    for name, labels in by_cluster.items():
        kept = [s for s in labels if s in sectors]
        filtered[name] = kept
        dropped.extend(s for s in labels if s not in sectors)
    if dropped:
        print(f"[clusters] dropped {len(dropped)} labels not found in the DB; "
              f"they will not belong to any cluster.")
        # Surface all of them once so the correspondence file can be patched.
        for s in dropped:
            print(f"    - {s}")
    return filtered


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract `zip_path` into `extract_dir` once (idempotent via marker)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"[zip] reusing already-extracted contents in {extract_dir}")
        return
    print(f"[zip] extracting {zip_path.name} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.touch()


def _ensure_dataset(outer_zip: Path, cache: Path, table: str, year: int,
                    system: str) -> Path:
    """Make sure the EXIOBASE dataset folder for (table, year, system) is on
    disk, extracting the outer and inner zips as needed. Returns the path to
    the dataset folder ready for `mario.parse_exiobase`.

    Layout assumed inside `outer_zip`:
        IOT_{year}_{ixi|pxp}.zip   (or SUT_{year}.zip)
        hashes.csv
    """
    _extract_zip(outer_zip, cache)

    if table == "IOT":
        inner_name = f"IOT_{year}_{system}"
    else:
        inner_name = f"SUT_{year}"
    inner_zip = cache / f"{inner_name}.zip"
    dataset_dir = cache / inner_name

    if not dataset_dir.exists():
        if not inner_zip.exists():
            raise FileNotFoundError(
                f"Expected {inner_zip.name} inside {outer_zip.name} but it's missing. "
                f"Available zips: {[p.name for p in cache.glob('*.zip')][:5]}..."
            )
        _extract_zip(inner_zip, dataset_dir)

    # If the inner zip wrapped its contents in a single top-level folder,
    # descend into it so MARIO sees the EXIOBASE files directly.
    entries = [p for p in dataset_dir.iterdir() if p.name != ".extracted"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dataset_dir


def load_database(args: argparse.Namespace):
    """Return a MARIO Database: from local path, download, or test fallback."""
    if args.path:
        path = Path(args.path)
        print(f"[load] parsing EXIOBASE from local path: {path}")
        return mario.parse_exiobase(
            path=str(path),
            table=args.table,
            unit="Monetary",
        )

    if args.download:
        DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[load] downloading EXIOBASE v{args.version} {args.table}/{args.system} "
              f"year={args.year} -> {DEFAULT_CACHE}")
        kwargs = dict(
            path=str(DEFAULT_CACHE),
            years=[args.year],
            table=args.table,
            version=args.version,
        )
        if args.table == "IOT":
            kwargs["system"] = args.system
        mario.download_exiobase3(**kwargs)

        # After download, locate the extracted folder and parse it.
        candidates = sorted(DEFAULT_CACHE.glob(f"*{args.year}*"))
        if not candidates:
            raise FileNotFoundError(
                f"No EXIOBASE folder found in {DEFAULT_CACHE} after download."
            )
        target = candidates[0]
        print(f"[load] parsing downloaded folder: {target}")
        return mario.parse_exiobase(
            path=str(target),
            table=args.table,
            unit="Monetary",
        )

    if DEFAULT_ZIP.exists():
        target = _ensure_dataset(
            DEFAULT_ZIP, DEFAULT_CACHE, args.table, args.year, args.system,
        )
        print(f"[load] parsing EXIOBASE from default zip: {DEFAULT_ZIP.name} -> {target}")
        return mario.parse_exiobase(
            path=str(target),
            table=args.table,
            unit="Monetary",
        )

    print(f"[load] no --path/--download given and {DEFAULT_ZIP.name} not found; "
          f"falling back to mario.load_test({args.table!r})")
    return mario.load_test(args.table)


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def basic_inspections(db) -> None:
    """Reproduce the basic inspection workflow from the MARIO docs."""
    section("Database summary")
    print(db)

    section("db.sets: available dimension names")
    print(db.sets)

    section("db.scenarios: available scenarios")
    print(db.scenarios)

    section("db.get_index('all'): keys (per-dimension index)")
    idx_all = db.get_index("all")
    if isinstance(idx_all, dict):
        for key, val in idx_all.items():
            n = len(val) if hasattr(val, "__len__") else "?"
            print(f"  {key:<30s} -> {n} items")
    else:
        print(idx_all)

    for name in ("Region", "Sector", "Satellite account", "Factor of production",
                 "Consumption category"):
        try:
            items = db.get_index(name)
            section(f"db.get_index({name!r}): first 10 of {len(items)}")
            print(list(items)[:10])
        except Exception as e:
            print(f"[skip] get_index({name!r}): {e}")

    section("db.search('electricity'): labels matching the keyword")
    try:
        print(db.search("electricity"))
    except TypeError:
        # newer API takes (set_name, keyword)
        print(db.search("Sector", "electricity"))


def basic_calculations(db) -> None:
    """List available matrices and trigger a couple of computations."""
    section("db.list_matrices(scenario='baseline')")
    try:
        print(db.list_matrices(scenario="baseline"))
    except Exception as e:
        print(f"[skip] list_matrices: {e}")

    section("Direct access: shapes of Z, Y, V, E (if present)")
    for m in ("Z", "Y", "V", "E", "EY", "U", "S"):
        try:
            mat = getattr(db, m)
            shape = getattr(mat, "shape", None)
            print(f"  {m}: shape={shape}")
        except Exception as e:
            print(f"  {m}: not available ({e})")

    section("db.calc_all(): materialize all derived matrices")
    try:
        db.calc_all()
        print("calc_all() OK")
    except Exception as e:
        print(f"[skip] calc_all: {e}")

    section("db.query(matrices=['e'], scenarios='baseline'): intensity matrix")
    try:
        out = db.query(matrices=["e"], scenarios="baseline")
        print(type(out))
        if hasattr(out, "shape"):
            print("shape:", out.shape)
        elif isinstance(out, dict):
            for k, v in out.items():
                print(f"  {k}: shape={getattr(v, 'shape', None)}")
    except Exception as e:
        print(f"[skip] query: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--path", default=None,
                   help="Local path to an EXIOBASE dataset folder (e.g. IOT_2019_ixi).")
    p.add_argument("--download", action="store_true",
                   help="Download EXIOBASE via mario.download_exiobase3 into ./exiobase_data.")
    p.add_argument("--table", choices=["IOT", "SUT"], default="IOT")
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--system", choices=["ixi", "pxp"], default="ixi",
                   help="Only used when --table IOT.")
    p.add_argument("--version", default="3.10.2",
                   help="EXIOBASE version (IOT only). SUT downloads use 3.8.2.")
    if any(arg.startswith("--f=") for arg in sys.argv[1:]):
        args, _ = p.parse_known_args([])
        return args
    return p.parse_args()


def sector_to_cluster_map(by_cluster: Dict[str, List[str]]) -> Dict[str, str]:
    """Reverse {cluster: [sectors]} into {sector: cluster}. Raises on overlaps."""
    out: Dict[str, str] = {}
    for cluster, sectors in by_cluster.items():
        for s in sectors:
            if s in out and out[s] != cluster:
                raise ValueError(f"Sector {s!r} is in both {out[s]!r} and {cluster!r}")
            out[s] = cluster
    return out


# MARIO names every single-level matrix axis "Item" (see INDEX_NAME_LAYOUTS in
# mario/model/conventions.py). To recover the semantic name (factor, satellite
# etc.) we look it up here keyed by (matrix, side). Only the 1-level axes need
# this table; 3-level axes get their semantic name from the Level column.
_SINGLE_LEVEL_DIMS = {
    "V":  {"from": "Factor of production"},
    "VY": {"from": "Factor of production"},
    "e":  {"from": "Satellite account"},
    "E":  {"from": "Satellite account"},
    "EY": {"from": "Satellite account"},
}


def _cluster_axis(mat: pd.DataFrame, axis: int,
                  s2c: Dict[str, str]) -> pd.DataFrame:
    """Replace the 'Item' level on `axis` (0=index, 1=columns) by its cluster
    name, drop entries whose sector has no cluster, and sum rows/cols that
    collapse into the same cluster. No-op when the axis has no 'Item' level
    OR when none of its items map to a cluster (e.g. Y's column side which
    carries consumption-category items, not sector items)."""
    target = mat.index if axis == 0 else mat.columns
    if not target.names or "Item" not in target.names:
        return mat
    new_items = target.get_level_values("Item").map(s2c)
    if new_items.notna().sum() == 0:
        return mat
    arrays = [
        new_items if name == "Item" else target.get_level_values(name)
        for name in target.names
    ]
    new_idx = pd.MultiIndex.from_arrays(arrays, names=list(target.names))
    if axis == 0:
        mat = mat.copy()
        mat.index = new_idx
        mat = mat.loc[mat.index.get_level_values("Item").notna()]
        return mat.groupby(level=list(mat.index.names), sort=False).sum()
    mat = mat.copy()
    mat.columns = new_idx
    mat = mat.loc[:, mat.columns.get_level_values("Item").notna()]
    return mat.T.groupby(level=list(mat.columns.names), sort=False).sum().T


def _describe_for_plot(df: pd.DataFrame, label: str,
                       filter_cols: Iterable[str] = ()) -> None:
    """Print shape, columns and unique values of filter columns of a frame
    about to be plotted. Helps diagnose 'empty dataframe' plot errors."""
    print(f"\n[plot-debug] {label}: shape={df.shape}, columns={df.columns.tolist()}")
    for col in filter_cols:
        if col not in df.columns:
            print(f"  [warn] expected filter column {col!r} missing")
            continue
        uniques = df[col].dropna().unique().tolist()
        preview = uniques[:10] + (["..."] if len(uniques) > 10 else [])
        print(f"  {col}: {len(uniques)} unique -> {preview}")


def clustered_matrix_frame(db, matrix: str, scenarios: Iterable[str],
                           by_cluster: Dict[str, List[str]]) -> pd.DataFrame:
    """Build a long-form dataframe of `matrix` aggregated at cluster level,
    ready to feed `db.plot(data=..., x='Sector_from', color='Sector_to', ...)`.

    The 'Item' level is collapsed to its cluster at the MATRIX level (before
    stacking) so we never materialize the full cartesian product of sectors --
    critical for memory on EXIOBASE-sized Z.
    """
    scenarios = list(scenarios) if not isinstance(scenarios, str) else [scenarios]
    s2c = sector_to_cluster_map(by_cluster)

    frames: List[pd.DataFrame] = []
    for scenario in scenarios:
        mat = db.query(matrix, scenarios=scenario)
        mat = _cluster_axis(mat, axis=0, s2c=s2c)
        mat = _cluster_axis(mat, axis=1, s2c=s2c)
        # Suffix index level names so reset_index doesn't collide when both
        # axes share names (Region/Level/Item for Z).
        mat = mat.copy()
        mat.index = mat.index.set_names([f"{n}_from" for n in mat.index.names])
        mat.columns = mat.columns.set_names([f"{n}_to" for n in mat.columns.names])
        stacked = mat.stack(list(range(mat.columns.nlevels)), future_stack=True)
        stacked.name = "Value"
        flat = stacked.reset_index()
        # Two layouts to handle:
        # * 3-level axes (Region, Level, Item): emit one alias per Level value
        #   (Sector_<side>, "Consumption category_<side>"...), like MARIO's
        #   _attach_level_aliases.
        # * 1-level axes (Item only, e.g. V/e/E rows): look the semantic
        #   dimension up in _SINGLE_LEVEL_DIMS and rename Item_<side> to it.
        for side in ("from", "to"):
            level_col = f"Level_{side}"
            item_col = f"Item_{side}"
            if item_col not in flat.columns:
                continue
            if level_col in flat.columns:
                for level_name in flat[level_col].dropna().astype(str).unique():
                    alias = f"{level_name}_{side}"
                    if alias not in flat.columns:
                        mask = flat[level_col].astype(str) == level_name
                        flat[alias] = flat[item_col].where(mask)
                flat = flat.drop(columns=[item_col, level_col])
            else:
                dim = _SINGLE_LEVEL_DIMS.get(matrix, {}).get(side)
                if dim:
                    flat = flat.rename(columns={item_col: f"{dim}_{side}"})
        flat["Scenario"] = scenario
        frames.append(flat)

    df = pd.concat(frames, ignore_index=True)
    group_cols = [c for c in df.columns if c != "Value"]
    return df.groupby(group_cols, dropna=False, as_index=False)["Value"].sum()


def try_shock(db):

    by_cluster = parse_sector_clusters(SECTOR_CLUSTERS_FILE)
    report_cluster_coverage(by_cluster, db)
    # EXIOBASE 3 renamed/merged ~20 sectors from EXIOBASE 2 (waste/biogas/etc.).
    # First try to recover them by fuzzy-matching against the actual E3 index;
    # whatever remains unmatched is dropped before building the MARIO dict.
    by_cluster, _applied = reconcile_e2_to_e3(by_cluster, db)
    by_cluster = filter_clusters_to_db(by_cluster, db)
    mario_clusters = build_mario_clusters(by_cluster)
    try:
        db.set_clusters(mario_clusters)
        print(f"[clusters] registered {sum(len(v) for v in by_cluster.values())} "
              f"sector labels across {len(by_cluster)} clusters")
    except Exception as e:
        print(f"[clusters] set_clusters failed: {e}")

    SHOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Each Excel template targets a different matrix; apply them as independent
    # scenarios so we can compare baseline vs. each shock without mixing
    # effects. calc_all() is needed after Z/Y/V shocks so downstream matrices
    # (X, E, multipliers) are recomputed; an e shock writes the intensity
    # matrix directly, so MARIO already keeps E in sync.
    shock_runs = [
        ("shock_IOT_exio.xlsx",  {"z": True}, "Z shock",  True),
        ("shock_IOT_exioY.xlsx", {"Y": True}, "Y shock",  True),
        ("shock_IOT_exioV.xlsx", {"v": True}, "V shock",  True),
        ("shock_IOT_exioE.xlsx", {"e": True}, "e shock",  False),
    ]
    for fname, kwargs, scenario, needs_recalc in shock_runs:
        io_path = SHOCK_DIR / fname
        if not io_path.exists():
            print(f"[shock] skipping {scenario}: {io_path.name} not found")
            continue
        print(f"[shock] applying {scenario} from {io_path.name}")
        db.shock_calc(io=str(io_path), scenario=scenario, **kwargs)
        if needs_recalc:
            db.calc_all()

    # Z shock 
    clustered_Z = clustered_matrix_frame(
        db, "Z", ["baseline", "Z shock"], by_cluster,
    )
    db.plot(
        data=clustered_Z,
        preset=None,
        kind="bar",
        x="Sector_from",
        color="Sector_to",
        facet_col="Scenario",
        filters={"Region_from": ["US"], "Region_to": ["FR"]},
        auto_open=True,
        path=str(SHOCK_DIR / "plot_Z.html"),
    )
    db.plot(
        matrix="Z",
        scenarios=["Z shock"],
        base_scenario="baseline",
        difference="relative",
        preset="heatmap",
        filters={"Region_from": ["US"], "Region_to": ["FR"]},
        auto_open=True,
        path=str(SHOCK_DIR / "plot_Z_heatmap.html"),
    )

    # Y shock - final demand, cluster x consumption category
    clustered_Y = clustered_matrix_frame(
        db, "Y", ["baseline", "Y shock"], by_cluster,
    )
    _describe_for_plot(clustered_Y, "Y", ["Region_from", "Sector_from",
                                          "Consumption category_to"])
    db.plot(
        data=clustered_Y,
        kind="bar",
        x="Sector_from",
        color="Consumption category_to",
        facet_col="Scenario",
        filters={"Region_from": ["US"]},
        auto_open=True,
        path=str(SHOCK_DIR / "plot_Y_cluster.html"),
    )

    # V shock - value added, factor of production x cluster
    clustered_V = clustered_matrix_frame(
        db, "V", ["baseline", "V shock"], by_cluster,
    )
    _describe_for_plot(clustered_V, "V", ["Region_to", "Sector_to",
                                          "Factor of production_from"])
    db.plot(
        data=clustered_V,
        kind="bar",
        x="Sector_to",
        color="Factor of production_from",
        facet_col="Scenario",
        filters={"Region_to": ["FR"]},
        auto_open=True,
        path=str(SHOCK_DIR / "plot_V_cluster.html"),
    )

    # e shock - satellite intensity by cluster, top-5 baseline satellite accounts
    clustered_e = clustered_matrix_frame(
        db, "e", ["baseline", "e shock"], by_cluster,
    )
    _describe_for_plot(clustered_e, "e", ["Region_to", "Sector_to",
                                          "Satellite account_from"])
    top_satellites = (
        clustered_e.query("Scenario == 'baseline'")
        .groupby("Satellite account_from")["Value"]
        .sum().abs().nlargest(5).index.tolist()
    )
    db.plot(
        data=clustered_e,
        kind="bar",
        x="Sector_to",
        color="Satellite account_from",
        facet_col="Scenario",
        filters={"Satellite account_from": top_satellites,
                 "Region_to": ["US"]},
        auto_open=True,
        path=str(SHOCK_DIR / "plot_e_cluster.html"),
    )

    # E footprint - world-wide absolute emissions per cluster for each shock
    clustered_E = clustered_matrix_frame(
        db, "E", ["baseline", "Z shock", "Y shock", "V shock", "e shock"],
        by_cluster,
    )
    _describe_for_plot(clustered_E, "E", ["Region_to", "Sector_to",
                                          "Satellite account_from"])
    db.plot(
        data=clustered_E,
        kind="bar",
        x="Sector_to",
        color="Satellite account_from",
        facet_col="Scenario",
        filters={"Satellite account_from": top_satellites},
        auto_open=True,
        path=str(SHOCK_DIR / "plot_E_footprint_cluster.html"),
    )

    # Manual flow shock
    db.clone_scenario("baseline", "manual flow shock")
    db.reset_to_flows("manual flow shock")

    Z = db.query("Z", scenarios="manual flow shock").copy()
    Z.loc[:, :] *= 1.10

    db.update_scenarios("manual flow shock", Z=Z)

    X = db.query("X", scenarios="manual flow shock")

    # Manual coefficient shock
    db.clone_scenario("baseline", "manual coefficient shock")
    db.reset_to_coefficients("manual coefficient shock")

    z = db.query("z", scenarios="manual coefficient shock").copy()
    z.loc[:, :] *= 0.95

    db.update_scenarios("manual coefficient shock", z=z)

    X = db.query("X", scenarios="manual coefficient shock")
    V = db.query("V", scenarios="manual coefficient shock")
    GDP = db.GDP(scenario="manual coefficient shock")
    GDPbase = db.GDP(scenario="baseline")
    
    # Extract scalar values from GDP (which returns a DataFrame)
    try:
        gdp_val = float(GDP.sum().sum()) if hasattr(GDP, 'sum') else float(GDP)
        gdp_base_val = float(GDPbase.sum().sum()) if hasattr(GDPbase, 'sum') else float(GDPbase)
        print(f"GDP shock: {gdp_val:.2f} vs baseline {gdp_base_val:.2f} ({(gdp_val/gdp_base_val-1)*100:.2f}%)")
    except Exception as e:
        print(f"[shock] warning: could not compute GDP impact: {e}")

def main() -> int:
    args = parse_args()
    try:
        db = load_database(args)
    except Exception as e:
        print(f"[fatal] could not load EXIOBASE: {e}")
        return 1

    print(f"\nLoaded MARIO database of type: {type(db).__name__}")
    basic_inspections(db)
    basic_calculations(db)
    try_shock(db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
