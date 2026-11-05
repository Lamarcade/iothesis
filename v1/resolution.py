"""
Resolution -- at what granularity the projection problem is stated.

KRAS balances n^2 variables, so the dimension of the system is not a detail: it
is the parameter that decides whether a run takes seconds or is impossible.
EXIOBASE ships 49 regions x 200 products (pxp), n = 9800, hence 96 million
variables and a constraint matrix of ~2.3 GB in CSR. `optim/kras.py` cycles over
constraints one at a time in Python, so that size is out of reach today (see
`Data/docs/V1_HYPOTHESES.md`). Everything downstream is therefore written
against a *label vector*, and the resolution is what produces it.

Four resolutions, from cheap to faithful:

    "world"        regions summed, EXIOBASE sectors kept       n = 200 (pxp)
    "eu_macro"     Europe as one bloc + 5 macro regions,
                   each x 36 GEM-E3 sectors                    n = 216
    "region_ngfs"  49 regions x 36 GEM-E3 sectors              n = 1764
    "full"         no aggregation at all                       n = 9800

`"world"` is the default. Summing regions keeps international supply chains
*inside* the flows -- a French car bought from German steel stays an
intermediate flow -- but stops distinguishing where they happen. `"region_ngfs"`
is the one that actually matches the multi-regional setting of the paper's
section 3.1; it is wired and correct but slow.

Studying one region: aggregate the rest, never slice it off
------------------------------------------------------------
`"eu_macro"` exists to answer "what if Europe alone decarbonises". The tempting
shortcut -- cut the European rows and columns out of Z and work on that block --
**breaks the accounting**, measured on EXIOBASE 2021: `x = Z.1 + y` then fails
with a 78 % max relative residual, 228 of 1152 sectors off by more than 10 %,
because 2 191 740 M.EUR of European intermediate *sales to non-European buyers*
have nowhere left to go.

Aggregating the rest of the world instead keeps the system closed, because
aggregation preserves sums -- the same property `"world"` and `"region_ngfs"`
already rely on. Europe's exports and imports survive as real cells of Z, so
KRAS can still re-source between a domestic and an imported supplier, and the
carbon embodied in imports (305 Mt, 11 % of what Europe buys) stays inside
`b_j` instead of vanishing. Pair it with `constraints.beta_by_region` to put a
target on the European columns only. See `Data/docs/V1_HYPOTHESES.md`.

Aggregate Z and x, then divide
------------------------------
    A_agg = Z_agg / x_agg          NOT  mean(A) over the members of a group

The second is not a technical coefficient of anything. Same convention as
`scope3/aggregate.py:19-24` and `optim/exioptim.py:122`. That is why this module
only ever produces *labels*, and the division happens once, in `IOSystem.A`.

The two aggregation primitives are imported from `scope3/aggregate.py` rather
than copied: `aggregate_matrix` and `aggregate_vector` there are already fully
generic in their `labels`/`order` arguments -- it is only `build_world_system`
around them that hardcodes region-summing. Passing composite "region | sector"
labels makes them region-preserving with no change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scope3.aggregate import aggregate_matrix, aggregate_vector  # noqa: E402,F401
from shortterm.project_production import DAPS_REGION_RUN  # noqa: E402
from shortterm.st_exiobase_mapping import exio_region_map, exio_to_ngfs  # noqa: E402

RESOLUTIONS = ("world", "eu_macro", "region_ngfs", "full")

#: Separator inside composite "region | sector" labels. EXIOBASE sector names
#: contain commas, parentheses and slashes but never a pipe, so this round-trips.
SEP = " | "

#: DAPS run -> macro-region label. Mirrors `shortterm/plot_production.py:128-132`
#: rather than importing it: that module pulls in plotly, and this one sits on
#: v1's core path. Same refusal as `v1/scenario.py`, which reimplements a
#: six-line CSV reader for the same reason.
MACRO_LABEL = {
    "DAPS_EUR": "Europe", "DAPS_ASIA": "Asia", "DAPS_NAM": "North America",
    "DAPS_SAM": "South America", "DAPS_AFR_R": "Africa & Middle East",
    "DAPS_OCE": "Oceania",
}

#: The bloc `"eu_macro"` keeps whole. Named "Europe", **not** "EU": the DAPS_EUR
#: grouping is EU27 + GB + CH + NO + **RU** + RoW-Europe, 32 EXIOBASE regions.
#: Calling it EU would misrepresent the perimeter -- Russia's gas and oil are
#: inside this bloc, so they read as domestic flows, not as imports.
EUROPE_BLOC = "Europe"


def macro_region_map() -> dict[str, str]:
    """
    EXIOBASE region code -> macro-region label, for all 49 regions.

    Composed from the two non-plotting sources so v1 stays plotly-free:

        EXIOBASE region --exio_region_map--> GEM-E3 region
                        --DAPS_REGION_RUN--> DAPS run --> MACRO_LABEL

    Coverage is complete by construction: every one of the 60 GEM-E3 regions is
    targeted by exactly one DAPS run (see `project_production.py:131-134`), and
    `exio_region_map` covers all 49 EXIOBASE regions.
    """
    return {exio: MACRO_LABEL[DAPS_REGION_RUN[ngfs]]
            for exio, ngfs in exio_region_map().items()}


def labels_and_order(
    index: pd.MultiIndex,
    resolution: str = "world",
    table_type: str = "pxp",
) -> tuple[list[str], list[str]]:
    """
    Map an EXIOBASE (region, sector) index onto aggregation labels.

    Returns
    -------
    labels : list[str]
        One label per row of `index`, in the order of `index`. This is what
        `aggregate_matrix` / `aggregate_vector` group on.
    order : list[str]
        The unique labels, in first-appearance order. Because the EXIOBASE
        index is region-major and sector-ordered inside each region, this comes
        out region-major for the composite resolutions and in native EXIOBASE
        sector order for `"world"` -- deterministic without an explicit sort.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

    regions = index.get_level_values(0)
    sectors = index.get_level_values(1)

    if resolution == "world":
        labels = list(sectors)
    elif resolution == "full":
        labels = [f"{r}{SEP}{s}" for r, s in zip(regions, sectors)]
    else:  # region_ngfs, eu_macro -- both on the GEM-E3 sector partition
        mapping = exio_to_ngfs(table_type)
        missing = sorted({s for s in set(sectors) if s not in mapping})
        if missing:
            # The mapping is validated at 100 % coverage, so this should stay
            # empty; if EXIOBASE ever adds a product, fail loudly rather than
            # silently dropping its flows into a bucket of its own.
            raise KeyError(
                f"{len(missing)} EXIOBASE sector(s) absent from exio_to_ngfs"
                f"({table_type!r}): {missing[:5]}"
            )
        if resolution == "eu_macro":
            macro = macro_region_map()
            blocs = [macro[r] for r in regions]
        else:
            blocs = list(regions)
        labels = [f"{b}{SEP}{mapping[s]}" for b, s in zip(blocs, sectors)]

    order = list(dict.fromkeys(labels))
    return labels, order


def is_identity(labels: list[str], order: list[str]) -> bool:
    """
    True when aggregation is a no-op, i.e. `"full"` on a native EXIOBASE index.

    Worth checking: `aggregate_matrix` would otherwise materialise a second
    9800 x 9800 float64 block (768 MB) to compute a groupby that changes
    nothing.
    """
    return len(labels) == len(order) and labels == order


def split_label(label: str) -> tuple[str | None, str]:
    """`"FR | Chemicals"` -> `("FR", "Chemicals")`; a bare sector -> `(None, s)`."""
    if SEP in label:
        region, sector = label.split(SEP, 1)
        return region, sector
    return None, label
